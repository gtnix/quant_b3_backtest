//! Dividend loader from database and cache.

use std::collections::HashMap;
use std::path::Path;

use chrono::NaiveDate;
use rust_decimal::Decimal;
use thiserror::Error;
use tokio_postgres::Client;

use super::types::{DividendEntry, DividendIndex};

// =============================================================================
// ERROR TYPES
// =============================================================================

/// Errors that can occur when loading dividends.
#[derive(Debug, Error)]
pub enum DividendLoadError {
    #[error("Database error: {0}")]
    Database(String),

    #[error("Parse error for {symbol}: {message}")]
    Parse { symbol: String, message: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("No dividends found for symbol: {0}")]
    NotFound(String),
}

// =============================================================================
// DIVIDEND LOADER
// =============================================================================

/// Loads dividend data from database or cache.
///
/// Supports loading from:
/// - PostgreSQL database (dividends_history table)
/// - CSV cache files (for offline/deterministic tests)
#[derive(Debug)]
pub struct DividendLoader {
    /// Optional CSV cache directory
    cache_dir: Option<String>,
}

impl DividendLoader {
    /// Create a new loader (database-only).
    pub fn new() -> Self {
        Self { cache_dir: None }
    }

    /// Create loader with CSV cache fallback.
    pub fn with_cache(cache_dir: impl Into<String>) -> Self {
        Self {
            cache_dir: Some(cache_dir.into()),
        }
    }

    /// Load dividends from database for a symbol within a date range.
    pub async fn load_from_db(
        &self,
        client: &Client,
        symbol: &str,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<Vec<DividendEntry>, DividendLoadError> {
        let query = r#"
            SELECT symbol, ex_date, payment_date, rate, dividend_type, related_to
            FROM dividends_history
            WHERE symbol = $1 AND ex_date >= $2 AND ex_date <= $3
            ORDER BY ex_date
        "#;

        let rows = client
            .query(query, &[&symbol, &start, &end])
            .await
            .map_err(|e| DividendLoadError::Database(e.to_string()))?;

        let entries: Vec<DividendEntry> = rows
            .into_iter()
            .map(|row| {
                let rate_f64: f64 = row.get("rate");
                DividendEntry {
                    symbol: row.get("symbol"),
                    ex_date: row.get("ex_date"),
                    payment_date: row.get("payment_date"),
                    rate: Decimal::try_from(rate_f64).unwrap_or_default(),
                    dividend_type: row.get("dividend_type"),
                    related_to: row.get("related_to"),
                }
            })
            .collect();

        Ok(entries)
    }

    /// Load dividends for multiple symbols.
    pub async fn load_batch_from_db(
        &self,
        client: &Client,
        symbols: &[String],
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<Vec<DividendEntry>, DividendLoadError> {
        if symbols.is_empty() {
            return Ok(Vec::new());
        }

        // Build parameterized query for multiple symbols
        let placeholders: Vec<String> = (1..=symbols.len())
            .map(|i| format!("${}", i))
            .collect();
        let symbols_list = placeholders.join(", ");

        let query = format!(
            r#"
            SELECT symbol, ex_date, payment_date, rate, dividend_type, related_to
            FROM dividends_history
            WHERE symbol IN ({}) AND ex_date >= ${} AND ex_date <= ${}
            ORDER BY ex_date, symbol
            "#,
            symbols_list,
            symbols.len() + 1,
            symbols.len() + 2
        );

        // Build params vector
        let mut params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = Vec::new();
        for sym in symbols {
            params.push(sym);
        }
        params.push(&start);
        params.push(&end);

        let rows = client
            .query(&query, &params)
            .await
            .map_err(|e| DividendLoadError::Database(e.to_string()))?;

        let entries: Vec<DividendEntry> = rows
            .into_iter()
            .map(|row| {
                let rate_f64: f64 = row.get("rate");
                DividendEntry {
                    symbol: row.get("symbol"),
                    ex_date: row.get("ex_date"),
                    payment_date: row.get("payment_date"),
                    rate: Decimal::try_from(rate_f64).unwrap_or_default(),
                    dividend_type: row.get("dividend_type"),
                    related_to: row.get("related_to"),
                }
            })
            .collect();

        Ok(entries)
    }

    /// Load dividends and build an index.
    pub async fn load_index(
        &self,
        client: &Client,
        symbols: &[String],
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<DividendIndex, DividendLoadError> {
        let entries = self.load_batch_from_db(client, symbols, start, end).await?;
        Ok(DividendIndex::from_entries(entries))
    }

    /// Load dividends from CSV cache file.
    ///
    /// CSV format: symbol,ex_date,payment_date,rate,dividend_type
    pub fn load_from_csv(&self, path: &Path) -> Result<Vec<DividendEntry>, DividendLoadError> {
        let content = std::fs::read_to_string(path)?;
        let mut entries = Vec::new();

        for (i, line) in content.lines().enumerate() {
            // Skip header
            if i == 0 && line.contains("symbol") {
                continue;
            }

            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 4 {
                continue;
            }

            let ex_date = NaiveDate::parse_from_str(parts[1].trim(), "%Y-%m-%d")
                .map_err(|_| DividendLoadError::Parse {
                    symbol: parts[0].to_string(),
                    message: format!("Invalid ex_date: {}", parts[1]),
                })?;

            let payment_date = if parts.len() > 2 && !parts[2].trim().is_empty() {
                NaiveDate::parse_from_str(parts[2].trim(), "%Y-%m-%d").ok()
            } else {
                None
            };

            let rate: Decimal = parts[3]
                .trim()
                .parse()
                .map_err(|_| DividendLoadError::Parse {
                    symbol: parts[0].to_string(),
                    message: format!("Invalid rate: {}", parts[3]),
                })?;

            let dividend_type = if parts.len() > 4 {
                parts[4].trim().to_string()
            } else {
                "CASH".to_string()
            };

            entries.push(DividendEntry {
                symbol: parts[0].trim().to_string(),
                ex_date,
                payment_date,
                rate,
                dividend_type,
                related_to: None,
            });
        }

        Ok(entries)
    }

    /// Load from CSV and build index.
    pub fn load_index_from_csv(&self, path: &Path) -> Result<DividendIndex, DividendLoadError> {
        let entries = self.load_from_csv(path)?;
        Ok(DividendIndex::from_entries(entries))
    }
}

impl Default for DividendLoader {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// FIXTURE GENERATOR
// =============================================================================

/// Generate synthetic dividend data for testing.
pub struct DividendFixtureGenerator;

impl DividendFixtureGenerator {
    /// Generate quarterly dividends for a symbol.
    pub fn quarterly(
        symbol: &str,
        start_year: i32,
        end_year: i32,
        rate: Decimal,
    ) -> Vec<DividendEntry> {
        let mut entries = Vec::new();
        for year in start_year..=end_year {
            for month in [3, 6, 9, 12] {
                if let Some(date) = NaiveDate::from_ymd_opt(year, month, 15) {
                    entries.push(DividendEntry::cash(symbol, date, rate));
                }
            }
        }
        entries
    }

    /// Generate monthly dividends for a symbol (e.g., REITs).
    pub fn monthly(
        symbol: &str,
        start_year: i32,
        end_year: i32,
        rate: Decimal,
    ) -> Vec<DividendEntry> {
        let mut entries = Vec::new();
        for year in start_year..=end_year {
            for month in 1..=12 {
                if let Some(date) = NaiveDate::from_ymd_opt(year, month, 15) {
                    entries.push(DividendEntry::cash(symbol, date, rate));
                }
            }
        }
        entries
    }

    /// Generate a single dividend event.
    pub fn single(symbol: &str, ex_date: NaiveDate, rate: Decimal) -> DividendEntry {
        DividendEntry::cash(symbol, ex_date, rate)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_quarterly_fixture() {
        let divs = DividendFixtureGenerator::quarterly("TAEE11", 2024, 2024, dec!(0.50));
        assert_eq!(divs.len(), 4);
        assert_eq!(divs[0].ex_date.month(), 3);
        assert_eq!(divs[1].ex_date.month(), 6);
    }

    #[test]
    fn test_monthly_fixture() {
        let divs = DividendFixtureGenerator::monthly("O", 2024, 2024, dec!(0.25));
        assert_eq!(divs.len(), 12);
    }

    #[test]
    fn test_csv_parsing() {
        use tempfile::NamedTempFile;
        use std::io::Write;

        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "symbol,ex_date,payment_date,rate,dividend_type").unwrap();
        writeln!(file, "TAEE11,2025-03-15,2025-03-25,0.45,CASH").unwrap();
        writeln!(file, "TAEE11,2025-06-15,,0.50,CASH").unwrap();

        let loader = DividendLoader::new();
        let entries = loader.load_from_csv(file.path()).unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].rate, dec!(0.45));
        assert!(entries[0].payment_date.is_some());
        assert!(entries[1].payment_date.is_none());
    }
}


