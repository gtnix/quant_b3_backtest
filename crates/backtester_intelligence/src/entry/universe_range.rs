//! Universe Range Provider - Time-dependent asset eligibility.
//!
//! Provides point-in-time universe validation to prevent survivorship bias.
//! Uses `cache/universe.csv` as source of truth for asset existence periods.
//!
//! # V1 Pragmatic Approach
//!
//! - Uses `min_date` as proxy for "asset started trading"
//! - Uses `max_date` as proxy for "asset last traded"
//! - An asset is eligible only if `min_date <= rebalance_date <= max_date`
//!
//! # Limitations
//!
//! - `min_date` is first data point, not necessarily IPO date
//! - `max_date` is last data point, not necessarily delisting date
//! - Does not reconstruct historical index membership

use chrono::NaiveDate;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;
use tracing::{debug, warn};

use super::eligibility::{
    EligibilityDetails, EligibilityProvider, EligibilitySource, EligibilityStats,
    EligibilityStatsSnapshot,
};

/// Errors that can occur during universe range loading.
#[derive(Debug, Error)]
pub enum UniverseLoadError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("CSV parse error at line {line}: {message}")]
    ParseError { line: usize, message: String },

    #[error("Missing required column: {0}")]
    MissingColumn(String),

    #[error("Empty universe file: {0}")]
    EmptyFile(String),

    #[error("File not found: {0}")]
    FileNotFound(String),
}

/// Date range for an asset's existence in the universe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DateRange {
    /// First date with data (proxy for listing/IPO)
    pub min_date: NaiveDate,
    /// Last date with data (proxy for delisting or current)
    pub max_date: NaiveDate,
}

impl DateRange {
    /// Create a new date range.
    pub fn new(min_date: NaiveDate, max_date: NaiveDate) -> Self {
        Self { min_date, max_date }
    }

    /// Check if a date falls within this range (inclusive).
    #[inline]
    pub fn contains(&self, date: NaiveDate) -> bool {
        date >= self.min_date && date <= self.max_date
    }
}

/// Result of checking asset eligibility at a specific date.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EligibilityResult {
    /// Asset is eligible at this date
    Eligible,
    /// Asset exists but date is outside its range
    OutsideDateRange {
        min_date: NaiveDate,
        max_date: NaiveDate,
    },
    /// Asset not found in universe data
    SymbolNotInUniverse,
}

impl EligibilityResult {
    /// Returns true if the asset is eligible.
    #[inline]
    pub fn is_eligible(&self) -> bool {
        matches!(self, EligibilityResult::Eligible)
    }
}

/// Provider for universe date ranges.
///
/// Loads `cache/universe.csv` once and provides O(1) lookups for eligibility checks.
/// Implements `EligibilityProvider` trait for use as V1 fallback in V2.
#[derive(Debug)]
pub struct UniverseRangeProvider {
    /// Symbol -> DateRange mapping
    ranges: HashMap<String, DateRange>,
    /// Statistics for observability
    load_stats: LoadStats,
    /// Eligibility check statistics (for trait compliance)
    eligibility_stats: EligibilityStats,
}

impl Clone for UniverseRangeProvider {
    fn clone(&self) -> Self {
        Self {
            ranges: self.ranges.clone(),
            load_stats: self.load_stats.clone(),
            // Reset eligibility stats on clone (they're per-instance counters)
            eligibility_stats: EligibilityStats::new(),
        }
    }
}

/// Statistics from loading the universe file.
#[derive(Debug, Clone, Default)]
pub struct LoadStats {
    /// Total rows parsed
    pub total_rows: usize,
    /// Rows successfully loaded
    pub loaded_rows: usize,
    /// Rows skipped due to parse errors
    pub skipped_rows: usize,
    /// Duplicate symbols encountered (last wins)
    pub duplicate_symbols: usize,
}

impl UniverseRangeProvider {
    /// Load universe ranges from a CSV file.
    ///
    /// Expected format:
    /// ```csv
    /// symbol,avg_volume,bar_count,min_date,max_date
    /// PETR4,57968488,2723,2015-01-02,2025-12-23
    /// ```
    ///
    /// # Behavior
    /// - Skips header row
    /// - Skips rows with invalid dates (with warning)
    /// - Duplicate symbols: last row wins (with warning)
    /// - Returns error if file is empty or missing required columns
    pub fn from_csv<P: AsRef<Path>>(path: P) -> Result<Self, UniverseLoadError> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(UniverseLoadError::FileNotFound(
                path.display().to_string(),
            ));
        }

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Parse header
        let header = lines
            .next()
            .ok_or_else(|| UniverseLoadError::EmptyFile(path.display().to_string()))??;

        let columns: Vec<&str> = header.split(',').collect();
        let symbol_idx = Self::find_column(&columns, "symbol")?;
        let min_date_idx = Self::find_column(&columns, "min_date")?;
        let max_date_idx = Self::find_column(&columns, "max_date")?;

        let mut ranges = HashMap::new();
        let mut stats = LoadStats::default();

        for (line_num, line_result) in lines.enumerate() {
            let line_num = line_num + 2; // 1-indexed, skip header
            stats.total_rows += 1;

            let line = match line_result {
                Ok(l) => l,
                Err(e) => {
                    warn!("Universe CSV line {}: IO error: {}", line_num, e);
                    stats.skipped_rows += 1;
                    continue;
                }
            };

            if line.trim().is_empty() {
                continue;
            }

            let fields: Vec<&str> = line.split(',').collect();

            // Validate field count
            let max_idx = symbol_idx.max(min_date_idx).max(max_date_idx);
            if fields.len() <= max_idx {
                warn!(
                    "Universe CSV line {}: insufficient fields (got {}, need {})",
                    line_num,
                    fields.len(),
                    max_idx + 1
                );
                stats.skipped_rows += 1;
                continue;
            }

            let symbol = fields[symbol_idx].trim().to_uppercase();
            if symbol.is_empty() {
                warn!("Universe CSV line {}: empty symbol", line_num);
                stats.skipped_rows += 1;
                continue;
            }

            // Parse dates
            let min_date = match NaiveDate::parse_from_str(fields[min_date_idx].trim(), "%Y-%m-%d") {
                Ok(d) => d,
                Err(e) => {
                    warn!(
                        "Universe CSV line {}: invalid min_date '{}': {}",
                        line_num,
                        fields[min_date_idx],
                        e
                    );
                    stats.skipped_rows += 1;
                    continue;
                }
            };

            let max_date = match NaiveDate::parse_from_str(fields[max_date_idx].trim(), "%Y-%m-%d") {
                Ok(d) => d,
                Err(e) => {
                    warn!(
                        "Universe CSV line {}: invalid max_date '{}': {}",
                        line_num,
                        fields[max_date_idx],
                        e
                    );
                    stats.skipped_rows += 1;
                    continue;
                }
            };

            // Validate date range
            if min_date > max_date {
                warn!(
                    "Universe CSV line {}: min_date {} > max_date {} for {}",
                    line_num, min_date, max_date, symbol
                );
                stats.skipped_rows += 1;
                continue;
            }

            // Check for duplicates
            if ranges.contains_key(&symbol) {
                warn!(
                    "Universe CSV line {}: duplicate symbol {}, using latest",
                    line_num, symbol
                );
                stats.duplicate_symbols += 1;
            }

            ranges.insert(symbol, DateRange::new(min_date, max_date));
            stats.loaded_rows += 1;
        }

        if ranges.is_empty() {
            return Err(UniverseLoadError::EmptyFile(path.display().to_string()));
        }

        debug!(
            "Loaded universe ranges: {} symbols ({} skipped, {} duplicates)",
            stats.loaded_rows, stats.skipped_rows, stats.duplicate_symbols
        );

        Ok(Self {
            ranges,
            load_stats: stats,
            eligibility_stats: EligibilityStats::new(),
        })
    }

    /// Find column index by name (case-insensitive).
    fn find_column(columns: &[&str], name: &str) -> Result<usize, UniverseLoadError> {
        columns
            .iter()
            .position(|c| c.trim().eq_ignore_ascii_case(name))
            .ok_or_else(|| UniverseLoadError::MissingColumn(name.to_string()))
    }

    /// Check if a symbol is eligible at a specific date.
    ///
    /// Returns `Eligible` if `min_date <= date <= max_date`.
    #[inline]
    pub fn is_eligible(&self, symbol: &str, date: NaiveDate) -> EligibilityResult {
        match self.ranges.get(symbol) {
            Some(range) => {
                if range.contains(date) {
                    EligibilityResult::Eligible
                } else {
                    EligibilityResult::OutsideDateRange {
                        min_date: range.min_date,
                        max_date: range.max_date,
                    }
                }
            }
            None => EligibilityResult::SymbolNotInUniverse,
        }
    }

    /// Get the date range for a symbol, if it exists.
    #[inline]
    pub fn get_range(&self, symbol: &str) -> Option<&DateRange> {
        self.ranges.get(symbol)
    }

    /// Get the number of symbols in the universe.
    #[inline]
    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    /// Check if the universe is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Get loading statistics.
    pub fn stats(&self) -> &LoadStats {
        &self.load_stats
    }

    /// Get loading statistics (alias for stats()).
    pub fn load_stats(&self) -> &LoadStats {
        &self.load_stats
    }

    /// Get all symbols in the universe.
    pub fn symbols(&self) -> impl Iterator<Item = &str> {
        self.ranges.keys().map(|s| s.as_str())
    }

    /// Create an Arc-wrapped provider for sharing across threads.
    pub fn into_arc(self) -> Arc<Self> {
        Arc::new(self)
    }

    /// Create an empty provider (for testing or when universe validation is disabled).
    pub fn empty() -> Self {
        Self {
            ranges: HashMap::new(),
            load_stats: LoadStats::default(),
            eligibility_stats: EligibilityStats::new(),
        }
    }

    /// Create a provider from a pre-built map.
    ///
    /// Useful for testing or when universe data comes from a non-CSV source.
    pub fn from_map(ranges: HashMap<String, DateRange>) -> Self {
        let loaded_rows = ranges.len();
        Self {
            ranges,
            load_stats: LoadStats {
                total_rows: loaded_rows,
                loaded_rows,
                skipped_rows: 0,
                duplicate_symbols: 0,
            },
            eligibility_stats: EligibilityStats::new(),
        }
    }
}

// ============================================================================
// EligibilityProvider Trait Implementation (V1)
// ============================================================================

impl EligibilityProvider for UniverseRangeProvider {
    fn is_eligible(&self, symbol: &str, date: NaiveDate) -> EligibilityResult {
        self.eligibility_stats.record_v1_fallback(); // Always V1 for this provider

        match self.ranges.get(symbol) {
            Some(range) => {
                if range.contains(date) {
                    EligibilityResult::Eligible
                } else {
                    // Track pre-listing vs post-delisting
                    if date < range.min_date {
                        self.eligibility_stats.record_pre_listing();
                    } else {
                        self.eligibility_stats.record_post_delisting();
                    }
                    EligibilityResult::OutsideDateRange {
                        min_date: range.min_date,
                        max_date: range.max_date,
                    }
                }
            }
            None => {
                self.eligibility_stats.record_not_found();
                EligibilityResult::SymbolNotInUniverse
            }
        }
    }

    fn get_details(&self, symbol: &str) -> Option<EligibilityDetails> {
        self.ranges.get(symbol).map(|range| {
            EligibilityDetails::from_v1(range.min_date, range.max_date)
        })
    }

    fn stats(&self) -> EligibilityStatsSnapshot {
        self.eligibility_stats.snapshot()
    }

    fn get_source(&self, symbol: &str) -> EligibilitySource {
        if self.ranges.contains_key(symbol) {
            EligibilitySource::V1Range
        } else {
            EligibilitySource::Unknown
        }
    }
}

// Thread-safety: HashMap is read-only after construction, atomics are thread-safe
unsafe impl Send for UniverseRangeProvider {}
unsafe impl Sync for UniverseRangeProvider {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    fn create_temp_csv(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "{}", content).unwrap();
        file
    }

    // ========================================================================
    // CSV Parsing Tests
    // ========================================================================

    #[test]
    fn test_load_valid_csv() {
        let csv = "symbol,avg_volume,bar_count,min_date,max_date\n\
                   PETR4,57968488,2723,2015-01-02,2025-12-23\n\
                   VALE3,20476129,2723,2015-01-02,2025-12-23\n\
                   RAIZ4,17555716,1097,2021-08-05,2025-12-23";
        let file = create_temp_csv(csv);

        let provider = UniverseRangeProvider::from_csv(file.path()).unwrap();

        assert_eq!(provider.len(), 3);
        assert_eq!(provider.stats().loaded_rows, 3);
        assert_eq!(provider.stats().skipped_rows, 0);

        let petr4 = provider.get_range("PETR4").unwrap();
        assert_eq!(petr4.min_date, date(2015, 1, 2));
        assert_eq!(petr4.max_date, date(2025, 12, 23));

        let raiz4 = provider.get_range("RAIZ4").unwrap();
        assert_eq!(raiz4.min_date, date(2021, 8, 5));
    }

    #[test]
    fn test_missing_required_column() {
        let csv = "symbol,avg_volume,bar_count,min_date\n\
                   PETR4,57968488,2723,2015-01-02";
        let file = create_temp_csv(csv);

        let result = UniverseRangeProvider::from_csv(file.path());
        assert!(matches!(result, Err(UniverseLoadError::MissingColumn(_))));
    }

    #[test]
    fn test_invalid_date_skipped() {
        let csv = "symbol,avg_volume,bar_count,min_date,max_date\n\
                   PETR4,57968488,2723,2015-01-02,2025-12-23\n\
                   BAD1,100,10,invalid-date,2025-12-23\n\
                   VALE3,20476129,2723,2015-01-02,2025-12-23";
        let file = create_temp_csv(csv);

        let provider = UniverseRangeProvider::from_csv(file.path()).unwrap();

        assert_eq!(provider.len(), 2);
        assert_eq!(provider.stats().skipped_rows, 1);
        assert!(provider.get_range("BAD1").is_none());
    }

    #[test]
    fn test_duplicate_symbols_last_wins() {
        let csv = "symbol,avg_volume,bar_count,min_date,max_date\n\
                   PETR4,100,10,2015-01-01,2020-12-31\n\
                   PETR4,200,20,2018-01-01,2025-12-31";
        let file = create_temp_csv(csv);

        let provider = UniverseRangeProvider::from_csv(file.path()).unwrap();

        assert_eq!(provider.len(), 1);
        assert_eq!(provider.stats().duplicate_symbols, 1);

        // Last row wins
        let petr4 = provider.get_range("PETR4").unwrap();
        assert_eq!(petr4.min_date, date(2018, 1, 1));
        assert_eq!(petr4.max_date, date(2025, 12, 31));
    }

    #[test]
    fn test_empty_file_error() {
        let csv = "";
        let file = create_temp_csv(csv);

        let result = UniverseRangeProvider::from_csv(file.path());
        assert!(matches!(result, Err(UniverseLoadError::EmptyFile(_))));
    }

    #[test]
    fn test_header_only_error() {
        let csv = "symbol,avg_volume,bar_count,min_date,max_date\n";
        let file = create_temp_csv(csv);

        let result = UniverseRangeProvider::from_csv(file.path());
        assert!(matches!(result, Err(UniverseLoadError::EmptyFile(_))));
    }

    #[test]
    fn test_file_not_found() {
        let result = UniverseRangeProvider::from_csv("/nonexistent/path/universe.csv");
        assert!(matches!(result, Err(UniverseLoadError::FileNotFound(_))));
    }

    #[test]
    fn test_min_date_greater_than_max_date_skipped() {
        let csv = "symbol,avg_volume,bar_count,min_date,max_date\n\
                   PETR4,100,10,2025-01-01,2020-12-31\n\
                   VALE3,200,20,2015-01-01,2025-12-31";
        let file = create_temp_csv(csv);

        let provider = UniverseRangeProvider::from_csv(file.path()).unwrap();

        assert_eq!(provider.len(), 1);
        assert_eq!(provider.stats().skipped_rows, 1);
        assert!(provider.get_range("PETR4").is_none());
        assert!(provider.get_range("VALE3").is_some());
    }

    #[test]
    fn test_case_insensitive_columns() {
        let csv = "Symbol,avg_volume,bar_count,Min_Date,MAX_DATE\n\
                   PETR4,100,10,2015-01-01,2025-12-31";
        let file = create_temp_csv(csv);

        let provider = UniverseRangeProvider::from_csv(file.path()).unwrap();
        assert_eq!(provider.len(), 1);
    }

    #[test]
    fn test_symbol_uppercase_normalization() {
        let csv = "symbol,avg_volume,bar_count,min_date,max_date\n\
                   petr4,100,10,2015-01-01,2025-12-31";
        let file = create_temp_csv(csv);

        let provider = UniverseRangeProvider::from_csv(file.path()).unwrap();

        // Should be stored as uppercase
        assert!(provider.get_range("PETR4").is_some());
        assert!(provider.get_range("petr4").is_none());
    }

    // ========================================================================
    // Eligibility Check Tests
    // ========================================================================

    #[test]
    fn test_eligible_at_min_date() {
        let mut ranges = HashMap::new();
        ranges.insert(
            "PETR4".to_string(),
            DateRange::new(date(2015, 1, 2), date(2025, 12, 23)),
        );
        let provider = UniverseRangeProvider::from_map(ranges);

        let result = provider.is_eligible("PETR4", date(2015, 1, 2));
        assert_eq!(result, EligibilityResult::Eligible);
    }

    #[test]
    fn test_eligible_at_max_date() {
        let mut ranges = HashMap::new();
        ranges.insert(
            "PETR4".to_string(),
            DateRange::new(date(2015, 1, 2), date(2025, 12, 23)),
        );
        let provider = UniverseRangeProvider::from_map(ranges);

        let result = provider.is_eligible("PETR4", date(2025, 12, 23));
        assert_eq!(result, EligibilityResult::Eligible);
    }

    #[test]
    fn test_eligible_in_middle() {
        let mut ranges = HashMap::new();
        ranges.insert(
            "PETR4".to_string(),
            DateRange::new(date(2015, 1, 2), date(2025, 12, 23)),
        );
        let provider = UniverseRangeProvider::from_map(ranges);

        let result = provider.is_eligible("PETR4", date(2020, 6, 15));
        assert_eq!(result, EligibilityResult::Eligible);
    }

    #[test]
    fn test_before_min_date() {
        let mut ranges = HashMap::new();
        ranges.insert(
            "PETR4".to_string(),
            DateRange::new(date(2015, 1, 2), date(2025, 12, 23)),
        );
        let provider = UniverseRangeProvider::from_map(ranges);

        let result = provider.is_eligible("PETR4", date(2015, 1, 1));
        assert_eq!(
            result,
            EligibilityResult::OutsideDateRange {
                min_date: date(2015, 1, 2),
                max_date: date(2025, 12, 23),
            }
        );
    }

    #[test]
    fn test_after_max_date() {
        let mut ranges = HashMap::new();
        ranges.insert(
            "PETR4".to_string(),
            DateRange::new(date(2015, 1, 2), date(2025, 12, 23)),
        );
        let provider = UniverseRangeProvider::from_map(ranges);

        let result = provider.is_eligible("PETR4", date(2025, 12, 24));
        assert_eq!(
            result,
            EligibilityResult::OutsideDateRange {
                min_date: date(2015, 1, 2),
                max_date: date(2025, 12, 23),
            }
        );
    }

    #[test]
    fn test_symbol_not_in_universe() {
        let provider = UniverseRangeProvider::from_map(HashMap::new());

        let result = provider.is_eligible("UNKNOWN", date(2020, 1, 1));
        assert_eq!(result, EligibilityResult::SymbolNotInUniverse);
    }

    #[test]
    fn test_is_eligible_helper() {
        let mut ranges = HashMap::new();
        ranges.insert(
            "PETR4".to_string(),
            DateRange::new(date(2015, 1, 2), date(2025, 12, 23)),
        );
        let provider = UniverseRangeProvider::from_map(ranges);

        assert!(provider.is_eligible("PETR4", date(2020, 1, 1)).is_eligible());
        assert!(!provider.is_eligible("PETR4", date(2010, 1, 1)).is_eligible());
        assert!(!provider.is_eligible("UNKNOWN", date(2020, 1, 1)).is_eligible());
    }

    // ========================================================================
    // DateRange Tests
    // ========================================================================

    #[test]
    fn test_date_range_contains() {
        let range = DateRange::new(date(2015, 1, 1), date(2020, 12, 31));

        assert!(range.contains(date(2015, 1, 1)));
        assert!(range.contains(date(2020, 12, 31)));
        assert!(range.contains(date(2017, 6, 15)));
        assert!(!range.contains(date(2014, 12, 31)));
        assert!(!range.contains(date(2021, 1, 1)));
    }

    // ========================================================================
    // Utility Tests
    // ========================================================================

    #[test]
    fn test_empty_provider() {
        let provider = UniverseRangeProvider::empty();

        assert!(provider.is_empty());
        assert_eq!(provider.len(), 0);
        assert_eq!(
            provider.is_eligible("ANY", date(2020, 1, 1)),
            EligibilityResult::SymbolNotInUniverse
        );
    }

    #[test]
    fn test_symbols_iterator() {
        let mut ranges = HashMap::new();
        ranges.insert(
            "PETR4".to_string(),
            DateRange::new(date(2015, 1, 1), date(2025, 12, 31)),
        );
        ranges.insert(
            "VALE3".to_string(),
            DateRange::new(date(2015, 1, 1), date(2025, 12, 31)),
        );
        let provider = UniverseRangeProvider::from_map(ranges);

        let symbols: Vec<&str> = provider.symbols().collect();
        assert_eq!(symbols.len(), 2);
        assert!(symbols.contains(&"PETR4"));
        assert!(symbols.contains(&"VALE3"));
    }

    #[test]
    fn test_into_arc() {
        let mut ranges = HashMap::new();
        ranges.insert(
            "PETR4".to_string(),
            DateRange::new(date(2015, 1, 1), date(2025, 12, 31)),
        );
        let provider = UniverseRangeProvider::from_map(ranges);

        let arc_provider = provider.into_arc();
        assert_eq!(arc_provider.len(), 1);
    }
}

