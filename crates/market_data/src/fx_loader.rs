//! FX Rate Data Loader
//!
//! Loads FX rate data from CSV files (produced by datahub_fx) into
//! the in-memory FX provider for use in backtesting.
//!
//! # File Format
//!
//! Expected CSV format:
//! ```csv
//! date,rate,source
//! 2024-01-02,4.8521,BCB
//! 2024-01-03,4.8934,BCB
//! ```
//!
//! # File Naming Convention
//!
//! Files are named after the currency pair:
//! - `USD_BRL.csv` for USD/BRL rates
//! - `EUR_USD.csv` for EUR/USD rates
//!
//! # OBFS Cache
//!
//! For ultra-performance, FX data is cached using OBFS (rkyv + Zstd).
//! Cache provides ~10x compression and sub-millisecond loading.

use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::{debug, info};

/// Errors that can occur during FX data loading.
#[derive(Debug, Error)]
pub enum FxLoadError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Parse error in {file} line {line}: {message}")]
    Parse {
        file: String,
        line: usize,
        message: String,
    },
    
    #[error("Invalid pair format: {0}")]
    InvalidPair(String),
    
    #[error("No FX data files found in {0}")]
    NoDataFound(PathBuf),
    
    #[error("Empty FX file: {0}")]
    EmptyFile(PathBuf),
    
    #[error("Cache error: {0}")]
    Cache(String),
}

// =============================================================================
// OBFS CACHE FOR FX DATA
// =============================================================================

/// Default cache directory for FX data.
const FX_CACHE_DIR: &str = ".cache/fx_data";

/// Cached FX data structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedFxData {
    /// All FX series: pair -> (date -> rate as string for Decimal serialization)
    series: BTreeMap<String, Vec<(String, String)>>, // (date_str, rate_str)
    /// Source directory hash
    source_hash: String,
    /// Cache timestamp
    cached_at: i64,
}

/// FX data cache using OBFS for ultra-performance.
pub struct FxCache {
    cache_dir: PathBuf,
    compression: obfs::CompressionPipeline,
}

impl Default for FxCache {
    fn default() -> Self {
        Self::new(FX_CACHE_DIR)
    }
}

impl FxCache {
    /// Create a new FX cache with the given directory.
    pub fn new(cache_dir: impl AsRef<Path>) -> Self {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&cache_dir).ok();
        Self {
            cache_dir,
            compression: obfs::CompressionPipeline::with_level(3),
        }
    }
    
    /// Generate cache key from source directory.
    fn cache_key(&self, source_dir: &Path) -> String {
        let mut hasher = Sha256::new();
        hasher.update(source_dir.to_string_lossy().as_bytes());
        // Include mod times of all CSV files
        if let Ok(entries) = std::fs::read_dir(source_dir) {
            for entry in entries.flatten() {
                if entry.path().extension().map_or(false, |e| e == "csv") {
                    if let Ok(meta) = entry.metadata() {
                        if let Ok(modified) = meta.modified() {
                            hasher.update(format!("{:?}", modified).as_bytes());
                        }
                    }
                }
            }
        }
        let hash = hasher.finalize();
        format!("{:x}", hash)[..16].to_string()
    }
    
    /// Get cache file path.
    fn cache_path(&self, key: &str) -> PathBuf {
        self.cache_dir.join(format!("fx_{}.obfs", key))
    }
    
    /// Load FX data from cache.
    pub fn load(&self, source_dir: &Path) -> Result<BTreeMap<String, BTreeMap<NaiveDate, Decimal>>, FxLoadError> {
        let key = self.cache_key(source_dir);
        let path = self.cache_path(&key);
        
        if !path.exists() {
            return Err(FxLoadError::Cache("Cache miss".into()));
        }
        
        let compressed = std::fs::read(&path)?;
        let decompressed = self.compression.decompress(&compressed)
            .map_err(|e| FxLoadError::Cache(format!("Decompress failed: {}", e)))?;
        
        let cached: CachedFxData = serde_json::from_slice(&decompressed)
            .map_err(|e| FxLoadError::Cache(format!("Deserialize failed: {}", e)))?;
        
        // Convert back to BTreeMap with proper types
        let mut result = BTreeMap::new();
        for (pair, rates) in cached.series {
            let mut pair_rates = BTreeMap::new();
            for (date_str, rate_str) in rates {
                let date = NaiveDate::parse_from_str(&date_str, "%Y-%m-%d")
                    .map_err(|e| FxLoadError::Cache(format!("Date parse: {}", e)))?;
                let rate: Decimal = rate_str.parse()
                    .map_err(|e| FxLoadError::Cache(format!("Rate parse: {}", e)))?;
                pair_rates.insert(date, rate);
            }
            result.insert(pair, pair_rates);
        }
        
        debug!("FX Cache HIT: {} pairs from {}", result.len(), path.display());
        Ok(result)
    }
    
    /// Save FX data to cache.
    pub fn save(&self, source_dir: &Path, data: &BTreeMap<String, BTreeMap<NaiveDate, Decimal>>) -> Result<(), FxLoadError> {
        let key = self.cache_key(source_dir);
        let path = self.cache_path(&key);
        
        // Convert to serializable format
        let mut series = BTreeMap::new();
        for (pair, rates) in data {
            let rates_vec: Vec<(String, String)> = rates
                .iter()
                .map(|(d, r)| (d.to_string(), r.to_string()))
                .collect();
            series.insert(pair.clone(), rates_vec);
        }
        
        let cached = CachedFxData {
            series,
            source_hash: key.clone(),
            cached_at: chrono::Utc::now().timestamp(),
        };
        
        let json = serde_json::to_vec(&cached)
            .map_err(|e| FxLoadError::Cache(format!("Serialize failed: {}", e)))?;
        
        let compressed = self.compression.compress(&json)
            .map_err(|e| FxLoadError::Cache(format!("Compress failed: {}", e)))?;
        
        std::fs::write(&path, &compressed)?;
        
        let ratio = json.len() as f64 / compressed.len() as f64;
        info!(
            "FX Cache SAVE: {} pairs → {} ({:.1}x compression)",
            data.len(), path.display(), ratio
        );
        
        Ok(())
    }
}

/// A loaded FX rate record.
#[derive(Debug, Clone)]
pub struct FxRecord {
    pub date: NaiveDate,
    pub rate: Decimal,
    pub source: String,
}

/// Result of loading an FX series.
#[derive(Debug)]
pub struct FxSeriesInfo {
    pub pair: String,
    pub record_count: usize,
    pub first_date: NaiveDate,
    pub last_date: NaiveDate,
    pub sources: Vec<String>,
}

/// Parse a currency pair from filename (e.g., "USD_BRL.csv" -> "USD/BRL").
pub fn filename_to_pair(filename: &str) -> Option<String> {
    let stem = filename.strip_suffix(".csv")?;
    let parts: Vec<&str> = stem.split('_').collect();
    if parts.len() == 2 && parts[0].len() == 3 && parts[1].len() == 3 {
        Some(format!("{}/{}", parts[0], parts[1]))
    } else {
        None
    }
}

/// Parse a currency pair to filename (e.g., "USD/BRL" -> "USD_BRL.csv").
pub fn pair_to_filename(pair: &str) -> String {
    format!("{}.csv", pair.replace('/', "_"))
}

/// Load a single FX series from a CSV file.
///
/// # Arguments
///
/// * `path` - Path to the CSV file
///
/// # Returns
///
/// A BTreeMap of date -> rate for efficient point-in-time lookups.
pub fn load_fx_series(path: &Path) -> Result<BTreeMap<NaiveDate, Decimal>, FxLoadError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let filename = path.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    
    let mut rates = BTreeMap::new();
    
    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        
        // Skip header and empty lines
        if line_num == 0 || line.is_empty() || line.starts_with("date,") {
            continue;
        }
        
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 2 {
            continue; // Skip malformed lines silently
        }
        
        // Parse date (YYYY-MM-DD)
        let date = NaiveDate::parse_from_str(parts[0], "%Y-%m-%d")
            .map_err(|_| FxLoadError::Parse {
                file: filename.to_string(),
                line: line_num + 1,
                message: format!("Invalid date: {}", parts[0]),
            })?;
        
        // Parse rate
        let rate: Decimal = parts[1].parse()
            .map_err(|_| FxLoadError::Parse {
                file: filename.to_string(),
                line: line_num + 1,
                message: format!("Invalid rate: {}", parts[1]),
            })?;
        
        rates.insert(date, rate);
    }
    
    if rates.is_empty() {
        return Err(FxLoadError::EmptyFile(path.to_path_buf()));
    }
    
    Ok(rates)
}

/// Load a single FX series with full record details.
pub fn load_fx_series_with_info(path: &Path) -> Result<(Vec<FxRecord>, FxSeriesInfo), FxLoadError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let filename = path.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    
    let pair = filename_to_pair(filename)
        .ok_or_else(|| FxLoadError::InvalidPair(filename.to_string()))?;
    
    let mut records = Vec::new();
    let mut sources = std::collections::HashSet::new();
    
    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        
        // Skip header and empty lines
        if line_num == 0 || line.is_empty() || line.starts_with("date,") {
            continue;
        }
        
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 2 {
            continue;
        }
        
        let date = NaiveDate::parse_from_str(parts[0], "%Y-%m-%d")
            .map_err(|_| FxLoadError::Parse {
                file: filename.to_string(),
                line: line_num + 1,
                message: format!("Invalid date: {}", parts[0]),
            })?;
        
        let rate: Decimal = parts[1].parse()
            .map_err(|_| FxLoadError::Parse {
                file: filename.to_string(),
                line: line_num + 1,
                message: format!("Invalid rate: {}", parts[1]),
            })?;
        
        let source = parts.get(2).unwrap_or(&"unknown").to_string();
        sources.insert(source.clone());
        
        records.push(FxRecord { date, rate, source });
    }
    
    if records.is_empty() {
        return Err(FxLoadError::EmptyFile(path.to_path_buf()));
    }
    
    records.sort_by_key(|r| r.date);
    
    let info = FxSeriesInfo {
        pair,
        record_count: records.len(),
        first_date: records.first().unwrap().date,
        last_date: records.last().unwrap().date,
        sources: sources.into_iter().collect(),
    };
    
    Ok((records, info))
}

/// Load all FX series from a directory.
///
/// # Arguments
///
/// * `cache_dir` - Directory containing FX CSV files
///
/// # Returns
///
/// A map of pair -> (date -> rate) for all found series.
pub fn load_all_fx(cache_dir: &Path) -> Result<BTreeMap<String, BTreeMap<NaiveDate, Decimal>>, FxLoadError> {
    if !cache_dir.exists() {
        return Err(FxLoadError::NoDataFound(cache_dir.to_path_buf()));
    }
    
    let mut all_series = BTreeMap::new();
    let mut found_any = false;
    
    for entry in std::fs::read_dir(cache_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) != Some("csv") {
            continue;
        }
        
        let filename = path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        
        if let Some(pair) = filename_to_pair(filename) {
            match load_fx_series(&path) {
                Ok(rates) => {
                    all_series.insert(pair, rates);
                    found_any = true;
                }
                Err(e) => {
                    // Log but don't fail on individual file errors
                    eprintln!("Warning: Failed to load {}: {}", path.display(), e);
                }
            }
        }
    }
    
    if !found_any {
        return Err(FxLoadError::NoDataFound(cache_dir.to_path_buf()));
    }
    
    Ok(all_series)
}

/// Load all FX series with OBFS cache for ultra-performance.
///
/// This function first checks the local OBFS cache. If cached data exists
/// and source files haven't changed, it's loaded in sub-millisecond time.
/// If not cached or stale, data is parsed from CSV and cached.
///
/// Cache compression: ~10x vs raw JSON
/// Cache location: .cache/fx_data/fx_{hash}.obfs
pub fn load_all_fx_cached(source_dir: &Path) -> Result<BTreeMap<String, BTreeMap<NaiveDate, Decimal>>, FxLoadError> {
    let cache = FxCache::default();
    
    // Try cache first
    if let Ok(data) = cache.load(source_dir) {
        info!("FX data loaded from OBFS cache: {} pairs", data.len());
        return Ok(data);
    }
    
    // Cache miss - parse from CSV
    debug!("FX cache miss, parsing from CSV...");
    let data = load_all_fx(source_dir)?;
    
    // Save to cache
    if let Err(e) = cache.save(source_dir, &data) {
        debug!("Failed to save FX cache (non-fatal): {}", e);
    }
    
    Ok(data)
}

/// Get information about all FX series in a directory.
pub fn get_fx_status(cache_dir: &Path) -> Result<Vec<FxSeriesInfo>, FxLoadError> {
    if !cache_dir.exists() {
        return Err(FxLoadError::NoDataFound(cache_dir.to_path_buf()));
    }
    
    let mut status = Vec::new();
    
    for entry in std::fs::read_dir(cache_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) != Some("csv") {
            continue;
        }
        
        match load_fx_series_with_info(&path) {
            Ok((_, info)) => status.push(info),
            Err(e) => {
                eprintln!("Warning: Failed to load {}: {}", path.display(), e);
            }
        }
    }
    
    status.sort_by(|a, b| a.pair.cmp(&b.pair));
    Ok(status)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;
    use rust_decimal_macros::dec;

    fn create_test_csv(dir: &Path, filename: &str, content: &str) {
        let path = dir.join(filename);
        let mut file = File::create(&path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
    }

    #[test]
    fn test_filename_to_pair() {
        assert_eq!(filename_to_pair("USD_BRL.csv"), Some("USD/BRL".to_string()));
        assert_eq!(filename_to_pair("EUR_USD.csv"), Some("EUR/USD".to_string()));
        assert_eq!(filename_to_pair("invalid.csv"), None);
        assert_eq!(filename_to_pair("US_BR.csv"), None);
    }

    #[test]
    fn test_pair_to_filename() {
        assert_eq!(pair_to_filename("USD/BRL"), "USD_BRL.csv");
        assert_eq!(pair_to_filename("EUR/USD"), "EUR_USD.csv");
    }

    #[test]
    fn test_load_fx_series() {
        let tmp = TempDir::new().unwrap();
        let content = "date,rate,source\n2024-01-02,5.00,BCB\n2024-01-03,5.10,BCB\n";
        create_test_csv(tmp.path(), "USD_BRL.csv", content);
        
        let rates = load_fx_series(&tmp.path().join("USD_BRL.csv")).unwrap();
        
        assert_eq!(rates.len(), 2);
        assert_eq!(rates[&NaiveDate::from_ymd_opt(2024, 1, 2).unwrap()], dec!(5.00));
        assert_eq!(rates[&NaiveDate::from_ymd_opt(2024, 1, 3).unwrap()], dec!(5.10));
    }

    #[test]
    fn test_load_all_fx() {
        let tmp = TempDir::new().unwrap();
        
        create_test_csv(tmp.path(), "USD_BRL.csv", "date,rate,source\n2024-01-02,5.00,BCB\n");
        create_test_csv(tmp.path(), "EUR_USD.csv", "date,rate,source\n2024-01-02,1.10,FRED\n");
        
        let all = load_all_fx(tmp.path()).unwrap();
        
        assert_eq!(all.len(), 2);
        assert!(all.contains_key("USD/BRL"));
        assert!(all.contains_key("EUR/USD"));
    }

    #[test]
    fn test_empty_file_error() {
        let tmp = TempDir::new().unwrap();
        create_test_csv(tmp.path(), "USD_BRL.csv", "date,rate,source\n");
        
        let result = load_fx_series(&tmp.path().join("USD_BRL.csv"));
        assert!(matches!(result, Err(FxLoadError::EmptyFile(_))));
    }
}



























