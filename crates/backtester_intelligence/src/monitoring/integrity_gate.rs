//! Data Integrity Gate - Orchestrates integrity checks for Strategy Factory.
//!
//! Provides a unified gate that runs all data integrity checks and produces
//! a PASS/FAIL verdict used by factory run/resume/promote commands.
//!
//! Supports two output formats:
//! - Legacy: Individual JSON files per campaign
//! - OBFS: Consolidated ultra-compressed bundle (~15x compression)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use zstd::stream::{Decoder, Encoder};

use crate::filters::Market;
use super::config::DataHealthConfig;
use super::data_health::{
    DataContext, DataHealthCheck,
    TemporalIntegrityCheck, LookaheadPolicyCheck, CorpActionCheck, SurvivorshipCheck,
};
use super::types::{CheckCategory, CheckResult, Severity};

/// Ultra compression level (Zstd max with LDM).
const ULTRA_COMPRESSION_LEVEL: i32 = 19;

// =============================================================================
// DATA INTEGRITY REPORT
// =============================================================================

/// Overall verdict for data integrity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    /// All critical checks passed
    Pass,
    /// One or more critical checks failed
    Fail,
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Verdict::Pass => write!(f, "PASS"),
            Verdict::Fail => write!(f, "FAIL"),
        }
    }
}

/// Audit mode - controls performance vs thoroughness tradeoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum AuditMode {
    /// Fast sampling-based audit (default)
    #[default]
    Fast,
    /// Strict full-scan audit
    Strict,
}

impl std::fmt::Display for AuditMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuditMode::Fast => write!(f, "fast"),
            AuditMode::Strict => write!(f, "strict"),
        }
    }
}

impl AuditMode {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "strict" => AuditMode::Strict,
            _ => AuditMode::Fast,
        }
    }
}

/// Complete data integrity report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataIntegrityReport {
    /// Overall verdict (PASS/FAIL)
    pub verdict: Verdict,
    /// Confidence score (0.0 to 1.0)
    pub score: f64,
    /// Dataset hash for reproducibility
    pub dataset_hash: String,
    /// Market being audited
    pub market: String,
    /// Timezone used
    pub timezone: String,
    /// All check results
    pub checks: Vec<CheckResult>,
    /// Hard failure reasons (critical issues)
    pub hard_fails: Vec<String>,
    /// Warnings (non-blocking issues)
    pub warnings: Vec<String>,
    /// Audit statistics
    pub stats: AuditStats,
    /// Timestamp of audit
    pub created_at: DateTime<Utc>,
    /// Audit mode used
    pub audit_mode: String,
    /// Version for schema compatibility
    pub version: String,
}

/// Audit statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuditStats {
    pub total_checks: usize,
    pub passed: usize,
    pub warnings: usize,
    pub critical: usize,
    pub duration_ms: u64,
}

impl Default for DataIntegrityReport {
    fn default() -> Self {
        Self {
            verdict: Verdict::Pass,
            score: 1.0,
            dataset_hash: String::new(),
            market: String::new(),
            timezone: String::new(),
            checks: Vec::new(),
            hard_fails: Vec::new(),
            warnings: Vec::new(),
            stats: AuditStats::default(),
            created_at: Utc::now(),
            audit_mode: "fast".to_string(),
            version: "1.0.0".to_string(),
        }
    }
}

impl DataIntegrityReport {
    /// Create a new empty report.
    pub fn new(dataset_hash: &str, market: &str) -> Self {
        Self {
            dataset_hash: dataset_hash.to_string(),
            market: market.to_string(),
            ..Default::default()
        }
    }

    /// Check if the report passed.
    pub fn passed(&self) -> bool {
        self.verdict == Verdict::Pass
    }

    /// Get a summary string.
    pub fn summary(&self) -> String {
        format!(
            "{}: {} checks, {} passed, {} warnings, {} critical (score: {:.2})",
            self.verdict,
            self.stats.total_checks,
            self.stats.passed,
            self.stats.warnings,
            self.stats.critical,
            self.score
        )
    }

    /// Save report to JSON file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, json)
    }

    /// Load report from JSON file.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let json = fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    /// Save report to OBFS file (ultra-compressed).
    pub fn save_obfs(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_vec(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let compressed = ultra_compress(&json)?;
        
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, compressed)
    }

    /// Load report from OBFS file.
    pub fn load_obfs(path: &Path) -> std::io::Result<Self> {
        let compressed = fs::read(path)?;
        let decompressed = ultra_decompress(&compressed)?;
        serde_json::from_slice(&decompressed)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

// =============================================================================
// DATA INTEGRITY GATE
// =============================================================================

/// Data Integrity Gate - orchestrates all integrity checks.
pub struct DataIntegrityGate {
    checks: Vec<Box<dyn DataHealthCheck>>,
    config: DataHealthConfig,
    mode: AuditMode,
    market: Market,
}

impl DataIntegrityGate {
    /// Create a new data integrity gate.
    pub fn new(market: Market, delay_bars: u8, max_gap_days: u32, mode: AuditMode) -> Self {
        let checks: Vec<Box<dyn DataHealthCheck>> = vec![
            Box::new(TemporalIntegrityCheck::new(market, max_gap_days)),
            Box::new(LookaheadPolicyCheck::new(delay_bars)),
            Box::new(CorpActionCheck::new(market, 30.0)),
            Box::new(SurvivorshipCheck::new(market)),
        ];

        Self {
            checks,
            config: DataHealthConfig::default(),
            mode,
            market,
        }
    }

    /// Create gate for B3 market with defaults.
    pub fn b3_default() -> Self {
        Self::new(Market::BR, 1, 5, AuditMode::Fast)
    }

    /// Create gate for US market with defaults.
    pub fn us_default() -> Self {
        Self::new(Market::US, 1, 5, AuditMode::Fast)
    }

    /// Run all integrity checks and produce report.
    pub fn audit(&self, ctx: &DataContext, dataset_hash: &str) -> DataIntegrityReport {
        let start = std::time::Instant::now();

        // Run all checks
        let results: Vec<CheckResult> = self.checks.iter()
            .map(|check| check.run(ctx, &self.config))
            .collect();

        // Categorize results
        let mut hard_fails = Vec::new();
        let mut warnings = Vec::new();
        let mut passed = 0;
        let mut critical = 0;
        let mut warn_count = 0;

        for result in &results {
            if result.category == CheckCategory::DataIntegrity {
                match result.severity {
                    Severity::Crit | Severity::Halt => {
                        hard_fails.push(result.message.clone());
                        critical += 1;
                    }
                    Severity::Warn => {
                        warnings.push(result.message.clone());
                        warn_count += 1;
                    }
                    Severity::Info => {
                        if result.passed {
                            passed += 1;
                        }
                    }
                }
            } else if result.passed {
                passed += 1;
            }
        }

        // Determine verdict
        let verdict = if hard_fails.is_empty() {
            Verdict::Pass
        } else {
            Verdict::Fail
        };

        // Calculate score (1.0 = perfect, 0.0 = all failed)
        let total = results.len();
        let score = if total > 0 {
            (passed as f64 / total as f64).max(0.0).min(1.0)
        } else {
            1.0
        };

        let duration = start.elapsed();

        DataIntegrityReport {
            verdict,
            score,
            dataset_hash: dataset_hash.to_string(),
            market: format!("{:?}", self.market),
            timezone: self.market_timezone(),
            checks: results,
            hard_fails,
            warnings,
            stats: AuditStats {
                total_checks: total,
                passed,
                warnings: warn_count,
                critical,
                duration_ms: duration.as_millis() as u64,
            },
            created_at: Utc::now(),
            audit_mode: self.mode.to_string(),
            version: "1.0.0".to_string(),
        }
    }

    /// Get timezone string for market.
    fn market_timezone(&self) -> String {
        match self.market {
            Market::BR => "America/Sao_Paulo".to_string(),
            Market::US => "America/New_York".to_string(),
        }
    }

    /// Get the audit mode.
    pub fn mode(&self) -> AuditMode {
        self.mode
    }

    /// Get the market.
    pub fn market(&self) -> Market {
        self.market
    }
}

// =============================================================================
// OBFS COMPRESSION HELPERS
// =============================================================================

/// Ultra-compress data using Zstd level 19 with LDM and checksum.
fn ultra_compress(data: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut encoder = Encoder::new(Vec::new(), ULTRA_COMPRESSION_LEVEL)?;
    encoder.include_checksum(true)?;
    encoder.long_distance_matching(true)?;
    encoder.write_all(data)?;
    encoder.finish()
}

/// Decompress ultra-compressed data.
fn ultra_decompress(compressed: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut decoder = Decoder::new(compressed)?;
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;
    Ok(decompressed)
}

// =============================================================================
// INTEGRITY BUNDLE (OBFS Native)
// =============================================================================

/// Location of a report within the bundle file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityLocation {
    pub offset: u64,
    pub compressed_size: u32,
    pub original_size: u32,
}

/// Writer for consolidated integrity report bundles.
/// Consolidates 79+ individual JSON files into a single OBFS bundle.
pub struct IntegrityBundleWriter {
    root_path: PathBuf,
    data_file: Option<File>,
    entries: HashMap<String, IntegrityLocation>,
    written_count: u64,
}

impl IntegrityBundleWriter {
    /// Create a new bundle writer.
    pub fn new(root_path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let root_path = root_path.into();
        fs::create_dir_all(&root_path)?;

        Ok(Self {
            root_path,
            data_file: None,
            entries: HashMap::new(),
            written_count: 0,
        })
    }

    fn data_file_path(&self) -> PathBuf {
        self.root_path.join("integrity.obfs")
    }

    fn ensure_data_file(&mut self) -> std::io::Result<&mut File> {
        if self.data_file.is_none() {
            let file = fs::OpenOptions::new()
                .create(true)
                .read(true)
                .append(true)
                .open(self.data_file_path())?;
            self.data_file = Some(file);
        }
        Ok(self.data_file.as_mut().unwrap())
    }

    /// Add a report with campaign ID.
    pub fn add(&mut self, campaign_id: &str, report: &DataIntegrityReport) -> std::io::Result<()> {
        // Serialize report to JSON
        let json = serde_json::to_vec(report)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Ultra-compress
        let compressed = ultra_compress(&json)?;

        // Write to data file
        let file = self.ensure_data_file()?;
        let offset = file.seek(SeekFrom::End(0))?;

        // Write length prefix + compressed data
        let len_bytes = (compressed.len() as u32).to_le_bytes();
        file.write_all(&len_bytes)?;
        file.write_all(&compressed)?;

        // Track location
        self.entries.insert(
            campaign_id.to_string(),
            IntegrityLocation {
                offset,
                compressed_size: compressed.len() as u32,
                original_size: json.len() as u32,
            },
        );
        self.written_count += 1;

        Ok(())
    }

    /// Scan a directory and add all existing reports.
    pub fn add_from_directory(&mut self, dir: &Path) -> std::io::Result<u64> {
        let mut added = 0u64;
        
        if !dir.exists() {
            return Ok(0);
        }

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_dir() {
                let name = path.file_name().unwrap().to_string_lossy();
                if name.starts_with("camp_") {
                    let report_path = path.join("report.json");
                    if report_path.exists() {
                        if let Ok(report) = DataIntegrityReport::load(&report_path) {
                            self.add(&name, &report)?;
                            added += 1;
                        }
                    }
                }
            }
        }
        
        Ok(added)
    }

    /// Finish writing and return stats.
    pub fn finish(mut self) -> std::io::Result<IntegrityBundleStats> {
        // Sync data file
        if let Some(ref mut file) = self.data_file {
            file.sync_all()?;
        }

        // Write index
        let index_path = self.root_path.join("index.json");
        let index_json = serde_json::to_string_pretty(&self.entries)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        fs::write(index_path, index_json)?;

        // Get data file size
        let data_size = if self.data_file_path().exists() {
            fs::metadata(self.data_file_path())?.len()
        } else {
            0
        };

        Ok(IntegrityBundleStats {
            report_count: self.written_count,
            data_file_size: data_size,
            compression_level: ULTRA_COMPRESSION_LEVEL,
        })
    }

    pub fn count(&self) -> u64 {
        self.written_count
    }
}

/// Reader for consolidated integrity report bundles.
pub struct IntegrityBundleReader {
    root_path: PathBuf,
    index: HashMap<String, IntegrityLocation>,
}

impl IntegrityBundleReader {
    /// Open an existing bundle.
    pub fn open(root_path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let root_path = root_path.into();
        let index_path = root_path.join("index.json");

        let index: HashMap<String, IntegrityLocation> = if index_path.exists() {
            let content = fs::read_to_string(&index_path)?;
            serde_json::from_str(&content)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?
        } else {
            HashMap::new()
        };

        Ok(Self { root_path, index })
    }

    fn data_file_path(&self) -> PathBuf {
        self.root_path.join("integrity.obfs")
    }

    /// Get a report by campaign ID.
    pub fn get(&self, campaign_id: &str) -> std::io::Result<Option<DataIntegrityReport>> {
        let loc = match self.index.get(campaign_id) {
            Some(l) => l,
            None => return Ok(None),
        };

        let mut file = File::open(self.data_file_path())?;
        file.seek(SeekFrom::Start(loc.offset))?;

        // Read length prefix
        let mut len_bytes = [0u8; 4];
        file.read_exact(&mut len_bytes)?;
        let compressed_len = u32::from_le_bytes(len_bytes) as usize;

        // Read compressed data
        let mut compressed = vec![0u8; compressed_len];
        file.read_exact(&mut compressed)?;

        // Decompress
        let decompressed = ultra_decompress(&compressed)?;

        // Deserialize
        let report: DataIntegrityReport = serde_json::from_slice(&decompressed)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        Ok(Some(report))
    }

    /// List all campaign IDs.
    pub fn list(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }

    /// Get report count.
    pub fn count(&self) -> usize {
        self.index.len()
    }
}

/// Statistics for integrity bundle.
#[derive(Debug, Clone)]
pub struct IntegrityBundleStats {
    pub report_count: u64,
    pub data_file_size: u64,
    pub compression_level: i32,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use std::collections::HashMap;

    fn date(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).unwrap()
    }

    #[test]
    fn test_verdict_display() {
        assert_eq!(format!("{}", Verdict::Pass), "PASS");
        assert_eq!(format!("{}", Verdict::Fail), "FAIL");
    }

    #[test]
    fn test_audit_mode_from_str() {
        assert_eq!(AuditMode::from_str("fast"), AuditMode::Fast);
        assert_eq!(AuditMode::from_str("strict"), AuditMode::Strict);
        assert_eq!(AuditMode::from_str("STRICT"), AuditMode::Strict);
        assert_eq!(AuditMode::from_str("unknown"), AuditMode::Fast);
    }

    #[test]
    fn test_gate_creation() {
        let gate = DataIntegrityGate::b3_default();
        assert_eq!(gate.market(), Market::BR);
        assert_eq!(gate.mode(), AuditMode::Fast);
    }

    #[test]
    fn test_audit_empty_context() {
        let gate = DataIntegrityGate::b3_default();
        let ctx = DataContext::new(date(2024, 1, 10));
        
        let report = gate.audit(&ctx, "test_hash");
        
        // Empty context now fails as it has no data (stricter validation)
        assert_eq!(report.verdict, Verdict::Fail);
    }

    #[test]
    fn test_audit_with_duplicates() {
        let gate = DataIntegrityGate::b3_default();
        let mut ctx = DataContext::new(date(2024, 1, 10));
        ctx.duplicate_count.insert(Market::BR, 5);
        
        let report = gate.audit(&ctx, "test_hash");
        
        // Should fail due to duplicates
        assert_eq!(report.verdict, Verdict::Fail);
        assert!(!report.hard_fails.is_empty());
    }

    #[test]
    fn test_audit_with_lookahead_violation() {
        let gate = DataIntegrityGate::new(Market::BR, 1, 5, AuditMode::Fast);
        let mut ctx = DataContext::new(date(2024, 1, 10));
        ctx.delay_bars_policy = 0; // Violation: should be >= 1
        
        let report = gate.audit(&ctx, "test_hash");
        
        // Should fail due to lookahead violation
        assert_eq!(report.verdict, Verdict::Fail);
        assert!(report.hard_fails.iter().any(|f| f.contains("Lookahead")));
    }

    #[test]
    fn test_report_save_load() {
        let report = DataIntegrityReport::new("hash123", "BR");
        
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test_integrity_report.json");
        
        report.save(&path).unwrap();
        
        let loaded = DataIntegrityReport::load(&path).unwrap();
        assert_eq!(loaded.dataset_hash, "hash123");
        assert_eq!(loaded.market, "BR");
    }
}
