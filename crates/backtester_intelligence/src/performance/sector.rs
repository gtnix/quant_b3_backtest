//! Sector Provider - Asset sector classification for exposure analysis.
//!
//! Provides sector information for portfolio exposure breakdown.
//! Implements a trait-based design for flexibility (InMemory, CSV, DB backends).
//!
//! # Design Decisions
//!
//! - `get_sector()` never fails - returns `Sector::unknown()` with tracing::warn for missing symbols
//! - HashMap-based O(1) lookup for performance
//! - Thread-safe (Send + Sync)
//!
//! # Example
//!
//! ```ignore
//! let mut provider = InMemorySectorProvider::new();
//! provider.add("PETR4", "Energy");
//! provider.add("ITUB4", "Financials");
//!
//! assert_eq!(provider.get_sector("PETR4").as_str(), "Energy");
//! assert_eq!(provider.get_sector("UNKNOWN").as_str(), "Unknown");
//! ```

use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::warn;

// =============================================================================
// SECTOR TYPE
// =============================================================================

/// Sector classification for an asset.
///
/// Wraps a String to provide type safety and standardized handling.
/// Uses GICS-like sector names but accepts any string.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Sector(pub String);

impl Sector {
    /// Constant for unknown/unclassified sectors.
    pub const UNKNOWN: &'static str = "Unknown";

    /// Create a new sector.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Create an unknown sector.
    #[must_use]
    pub fn unknown() -> Self {
        Self(Self::UNKNOWN.to_string())
    }

    /// Get sector name as string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Check if this is an unknown sector.
    #[must_use]
    pub fn is_unknown(&self) -> bool {
        self.0 == Self::UNKNOWN
    }

    /// Standard GICS sectors for reference.
    pub const ENERGY: &'static str = "Energy";
    pub const MATERIALS: &'static str = "Materials";
    pub const INDUSTRIALS: &'static str = "Industrials";
    pub const CONSUMER_DISCRETIONARY: &'static str = "Consumer Discretionary";
    pub const CONSUMER_STAPLES: &'static str = "Consumer Staples";
    pub const HEALTH_CARE: &'static str = "Health Care";
    pub const FINANCIALS: &'static str = "Financials";
    pub const INFORMATION_TECHNOLOGY: &'static str = "Information Technology";
    pub const COMMUNICATION_SERVICES: &'static str = "Communication Services";
    pub const UTILITIES: &'static str = "Utilities";
    pub const REAL_ESTATE: &'static str = "Real Estate";
}

impl fmt::Display for Sector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for Sector {
    fn default() -> Self {
        Self::unknown()
    }
}

impl From<&str> for Sector {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for Sector {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<Option<String>> for Sector {
    fn from(s: Option<String>) -> Self {
        s.map(Sector::new).unwrap_or_else(Sector::unknown)
    }
}

// =============================================================================
// SECTOR PROVIDER TRAIT
// =============================================================================

/// Trait for providing sector information.
///
/// Implementations must be thread-safe (Send + Sync) for use in parallel processing.
pub trait SectorProvider: Send + Sync {
    /// Get sector for a symbol. Returns `Sector::unknown()` if not found.
    ///
    /// Implementations should log a warning for missing symbols but never panic.
    fn get_sector(&self, symbol: &str) -> Sector;

    /// Batch lookup for performance.
    ///
    /// Default implementation calls `get_sector` for each symbol.
    /// Implementations may override for batch-optimized lookups (e.g., DB).
    fn get_sectors(&self, symbols: &[&str]) -> Vec<Sector> {
        symbols.iter().map(|s| self.get_sector(s)).collect()
    }

    /// Get all known symbols.
    fn symbols(&self) -> Vec<String>;

    /// Get number of known symbols.
    fn len(&self) -> usize {
        self.symbols().len()
    }

    /// Check if provider has no symbols.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// =============================================================================
// IN-MEMORY PROVIDER
// =============================================================================

/// In-memory sector provider with O(1) HashMap lookup.
///
/// Best for small to medium universes (< 10,000 symbols).
/// Thread-safe.
#[derive(Debug, Default)]
pub struct InMemorySectorProvider {
    sectors: HashMap<String, Sector>,
    /// Track symbols that had missing sector (for diagnostics)
    missing_count: std::sync::atomic::AtomicUsize,
}

impl InMemorySectorProvider {
    /// Create a new empty provider.
    pub fn new() -> Self {
        Self {
            sectors: HashMap::new(),
            missing_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            sectors: HashMap::with_capacity(capacity),
            missing_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Add a symbol-sector mapping.
    pub fn add(&mut self, symbol: impl Into<String>, sector: impl Into<Sector>) {
        self.sectors.insert(symbol.into(), sector.into());
    }

    /// Add multiple mappings from iterator.
    pub fn add_many<I, S1, S2>(&mut self, mappings: I)
    where
        I: IntoIterator<Item = (S1, S2)>,
        S1: Into<String>,
        S2: Into<Sector>,
    {
        for (symbol, sector) in mappings {
            self.add(symbol, sector);
        }
    }

    /// Create from a HashMap.
    pub fn from_map(map: HashMap<String, String>) -> Self {
        let sectors = map
            .into_iter()
            .map(|(k, v)| (k, Sector::new(v)))
            .collect();
        Self {
            sectors,
            missing_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Get count of missing symbol lookups since creation.
    pub fn missing_count(&self) -> usize {
        self.missing_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl SectorProvider for InMemorySectorProvider {
    fn get_sector(&self, symbol: &str) -> Sector {
        match self.sectors.get(symbol) {
            Some(sector) => sector.clone(),
            None => {
                self.missing_count
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                warn!(symbol = %symbol, "Sector not found, using Unknown");
                Sector::unknown()
            }
        }
    }

    fn symbols(&self) -> Vec<String> {
        self.sectors.keys().cloned().collect()
    }

    fn len(&self) -> usize {
        self.sectors.len()
    }
}

// =============================================================================
// CSV PROVIDER
// =============================================================================

/// CSV-backed sector provider.
///
/// Loads sector mappings from a CSV file at initialization.
/// File format: `symbol,sector` (with or without header).
///
/// # Example CSV
///
/// ```csv
/// symbol,sector
/// PETR4,Energy
/// VALE3,Materials
/// ITUB4,Financials
/// ```
#[derive(Debug)]
pub struct CsvSectorProvider {
    inner: InMemorySectorProvider,
    source_path: String,
}

/// Error loading CSV sector data.
#[derive(Debug)]
pub enum CsvSectorError {
    IoError(std::io::Error),
    ParseError(String),
}

impl fmt::Display for CsvSectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IoError(e) => write!(f, "IO error: {}", e),
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for CsvSectorError {}

impl From<std::io::Error> for CsvSectorError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

impl CsvSectorProvider {
    /// Load from CSV file.
    ///
    /// Expects format: `symbol,sector` (first row may be header).
    /// Skips rows with fewer than 2 columns.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, CsvSectorError> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)?;
        Self::from_content(&content, path.to_string_lossy().to_string())
    }

    /// Load from CSV content string.
    pub fn from_content(content: &str, source_path: String) -> Result<Self, CsvSectorError> {
        let mut inner = InMemorySectorProvider::new();
        let mut line_count = 0;
        let mut skipped = 0;
        let mut is_first_data_line = true;

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Skip header if detected (first non-empty line with header-like content)
            if is_first_data_line {
                is_first_data_line = false;
                let lower = line.to_lowercase();
                if lower.contains("symbol") || lower.contains("sector") || lower.contains("ticker") {
                    continue;
                }
            }

            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() >= 2 {
                let symbol = parts[0].to_string();
                let sector = parts[1].to_string();
                if !symbol.is_empty() && !sector.is_empty() {
                    inner.add(symbol, sector);
                    line_count += 1;
                } else {
                    skipped += 1;
                }
            } else {
                skipped += 1;
            }
        }

        if line_count == 0 {
            return Err(CsvSectorError::ParseError(
                "No valid sector mappings found in CSV".to_string(),
            ));
        }

        tracing::info!(
            source = %source_path,
            loaded = line_count,
            skipped = skipped,
            "Loaded sector mappings from CSV"
        );

        Ok(Self { inner, source_path })
    }

    /// Get the source file path.
    pub fn source_path(&self) -> &str {
        &self.source_path
    }
}

impl SectorProvider for CsvSectorProvider {
    fn get_sector(&self, symbol: &str) -> Sector {
        self.inner.get_sector(symbol)
    }

    fn get_sectors(&self, symbols: &[&str]) -> Vec<Sector> {
        self.inner.get_sectors(symbols)
    }

    fn symbols(&self) -> Vec<String> {
        self.inner.symbols()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

// =============================================================================
// NULL PROVIDER (for when sector data is unavailable)
// =============================================================================

/// Null provider that returns Unknown for all symbols.
///
/// Use when sector data is not available but the pipeline needs to continue.
#[derive(Debug, Clone, Default)]
pub struct NullSectorProvider;

impl SectorProvider for NullSectorProvider {
    fn get_sector(&self, _symbol: &str) -> Sector {
        Sector::unknown()
    }

    fn symbols(&self) -> Vec<String> {
        Vec::new()
    }

    fn len(&self) -> usize {
        0
    }
}

// =============================================================================
// ARC WRAPPER FOR TRAIT OBJECTS
// =============================================================================

impl<T: SectorProvider + ?Sized> SectorProvider for Arc<T> {
    fn get_sector(&self, symbol: &str) -> Sector {
        (**self).get_sector(symbol)
    }

    fn get_sectors(&self, symbols: &[&str]) -> Vec<Sector> {
        (**self).get_sectors(symbols)
    }

    fn symbols(&self) -> Vec<String> {
        (**self).symbols()
    }

    fn len(&self) -> usize {
        (**self).len()
    }
}

impl<T: SectorProvider + ?Sized> SectorProvider for Box<T> {
    fn get_sector(&self, symbol: &str) -> Sector {
        (**self).get_sector(symbol)
    }

    fn get_sectors(&self, symbols: &[&str]) -> Vec<Sector> {
        (**self).get_sectors(symbols)
    }

    fn symbols(&self) -> Vec<String> {
        (**self).symbols()
    }

    fn len(&self) -> usize {
        (**self).len()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sector_creation() {
        let s = Sector::new("Energy");
        assert_eq!(s.as_str(), "Energy");
        assert!(!s.is_unknown());

        let unknown = Sector::unknown();
        assert_eq!(unknown.as_str(), "Unknown");
        assert!(unknown.is_unknown());
    }

    #[test]
    fn test_sector_from_option() {
        let s: Sector = Some("Financials".to_string()).into();
        assert_eq!(s.as_str(), "Financials");

        let s: Sector = None.into();
        assert!(s.is_unknown());
    }

    #[test]
    fn test_inmemory_provider_basic() {
        let mut provider = InMemorySectorProvider::new();
        provider.add("PETR4", "Energy");
        provider.add("ITUB4", "Financials");
        provider.add("VALE3", "Materials");

        assert_eq!(provider.get_sector("PETR4").as_str(), "Energy");
        assert_eq!(provider.get_sector("ITUB4").as_str(), "Financials");
        assert_eq!(provider.get_sector("VALE3").as_str(), "Materials");
        assert_eq!(provider.len(), 3);
    }

    #[test]
    fn test_inmemory_provider_unknown() {
        let provider = InMemorySectorProvider::new();
        let sector = provider.get_sector("NONEXISTENT");
        assert!(sector.is_unknown());
        assert_eq!(provider.missing_count(), 1);
    }

    #[test]
    fn test_inmemory_provider_batch() {
        let mut provider = InMemorySectorProvider::new();
        provider.add("A", "S1");
        provider.add("B", "S2");
        provider.add("C", "S3");

        let sectors = provider.get_sectors(&["A", "B", "C", "D"]);
        assert_eq!(sectors.len(), 4);
        assert_eq!(sectors[0].as_str(), "S1");
        assert_eq!(sectors[1].as_str(), "S2");
        assert_eq!(sectors[2].as_str(), "S3");
        assert!(sectors[3].is_unknown());
    }

    #[test]
    fn test_inmemory_from_map() {
        let mut map = HashMap::new();
        map.insert("PETR4".to_string(), "Energy".to_string());
        map.insert("ITUB4".to_string(), "Financials".to_string());

        let provider = InMemorySectorProvider::from_map(map);
        assert_eq!(provider.get_sector("PETR4").as_str(), "Energy");
        assert_eq!(provider.len(), 2);
    }

    #[test]
    fn test_csv_provider_from_content() {
        let csv = r#"
symbol,sector
PETR4,Energy
VALE3,Materials
ITUB4,Financials
BBDC4,Financials
"#;
        let provider = CsvSectorProvider::from_content(csv, "test.csv".to_string()).unwrap();

        assert_eq!(provider.get_sector("PETR4").as_str(), "Energy");
        assert_eq!(provider.get_sector("VALE3").as_str(), "Materials");
        assert_eq!(provider.get_sector("ITUB4").as_str(), "Financials");
        assert_eq!(provider.get_sector("BBDC4").as_str(), "Financials");
        assert_eq!(provider.len(), 4);
    }

    #[test]
    fn test_csv_provider_no_header() {
        let csv = "PETR4,Energy\nVALE3,Materials";
        let provider = CsvSectorProvider::from_content(csv, "test.csv".to_string()).unwrap();

        assert_eq!(provider.get_sector("PETR4").as_str(), "Energy");
        assert_eq!(provider.get_sector("VALE3").as_str(), "Materials");
        assert_eq!(provider.len(), 2);
    }

    #[test]
    fn test_csv_provider_empty() {
        let csv = "";
        let result = CsvSectorProvider::from_content(csv, "test.csv".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_csv_provider_malformed_lines() {
        let csv = r#"
symbol,sector
PETR4,Energy
invalid_line
VALE3,Materials
,
"#;
        let provider = CsvSectorProvider::from_content(csv, "test.csv".to_string()).unwrap();
        assert_eq!(provider.len(), 2); // Only 2 valid lines
    }

    #[test]
    fn test_null_provider() {
        let provider = NullSectorProvider;
        assert!(provider.get_sector("ANYTHING").is_unknown());
        assert!(provider.is_empty());
    }

    #[test]
    fn test_arc_provider() {
        let mut inner = InMemorySectorProvider::new();
        inner.add("PETR4", "Energy");

        let provider: Arc<dyn SectorProvider> = Arc::new(inner);
        assert_eq!(provider.get_sector("PETR4").as_str(), "Energy");
    }

    #[test]
    fn test_sector_display() {
        let s = Sector::new("Energy");
        assert_eq!(format!("{}", s), "Energy");
    }

    #[test]
    fn test_sector_equality() {
        let s1 = Sector::new("Energy");
        let s2 = Sector::new("Energy");
        let s3 = Sector::new("Financials");

        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
    }

    #[test]
    fn test_sector_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(Sector::new("Energy"));
        set.insert(Sector::new("Financials"));
        set.insert(Sector::new("Energy")); // Duplicate

        assert_eq!(set.len(), 2);
    }
}

