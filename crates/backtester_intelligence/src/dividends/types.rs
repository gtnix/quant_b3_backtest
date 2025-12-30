//! Dividend types for backtest engine.

use std::collections::HashMap;
use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

// =============================================================================
// PRICE TYPE (Anti-Double-Count Policy)
// =============================================================================

/// Price type for anti-double-count policy.
///
/// # Policy
///
/// - **Signals**: Use adjusted prices (dividend adjustments baked in)
/// - **Valuation**: Use raw prices (dividends enter via cashflow)
///
/// This ensures dividends are counted exactly once in the PnL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PriceType {
    /// Adjusted close prices for signals/indicators.
    /// Dividend adjustments are already reflected in the price series.
    Signals,
    /// Raw close prices for valuation/mark-to-market.
    /// Dividends must be added separately as cashflow.
    Valuation,
}

impl Default for PriceType {
    fn default() -> Self {
        Self::Signals
    }
}

// =============================================================================
// DIVIDEND ENTRY
// =============================================================================

/// A dividend event from the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DividendEntry {
    /// Asset symbol
    pub symbol: String,
    /// Ex-dividend date (when holder must own shares to receive dividend)
    pub ex_date: NaiveDate,
    /// Payment date (when dividend is actually paid)
    pub payment_date: Option<NaiveDate>,
    /// Dividend rate per share
    pub rate: Decimal,
    /// Type of dividend (CASH, STOCK, etc.)
    pub dividend_type: String,
    /// Related event (e.g., stock split that triggered this)
    pub related_to: Option<String>,
}

impl DividendEntry {
    /// Create a new cash dividend entry.
    pub fn cash(symbol: impl Into<String>, ex_date: NaiveDate, rate: Decimal) -> Self {
        Self {
            symbol: symbol.into(),
            ex_date,
            payment_date: None,
            rate,
            dividend_type: "CASH".to_string(),
            related_to: None,
        }
    }

    /// Set payment date.
    pub fn with_payment_date(mut self, date: NaiveDate) -> Self {
        self.payment_date = Some(date);
        self
    }

    /// Set dividend type.
    pub fn with_type(mut self, dtype: impl Into<String>) -> Self {
        self.dividend_type = dtype.into();
        self
    }
}

// =============================================================================
// DIVIDEND INDEX
// =============================================================================

/// Efficient index for dividend lookup by (symbol, ex_date).
///
/// Provides O(1) lookup per day for dividend events.
///
/// # Performance
///
/// - Index is built once at simulation start
/// - Lookup per day is O(1) via HashMap
/// - Memory usage is O(#dividends)
#[derive(Debug, Clone, Default)]
pub struct DividendIndex {
    /// Primary index: date -> (symbol -> dividend)
    by_date: HashMap<NaiveDate, HashMap<String, DividendEntry>>,
    /// Secondary index: symbol -> list of (ex_date, rate) for quick queries
    by_symbol: HashMap<String, Vec<(NaiveDate, Decimal)>>,
    /// Total dividends indexed
    count: usize,
}

impl DividendIndex {
    /// Create a new empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build index from a list of dividend entries.
    pub fn from_entries(entries: Vec<DividendEntry>) -> Self {
        let mut index = Self::new();
        for entry in entries {
            index.add(entry);
        }
        index
    }

    /// Add a dividend entry to the index.
    pub fn add(&mut self, entry: DividendEntry) {
        // Add to by_symbol index
        self.by_symbol
            .entry(entry.symbol.clone())
            .or_default()
            .push((entry.ex_date, entry.rate));

        // Add to by_date index
        self.by_date
            .entry(entry.ex_date)
            .or_default()
            .insert(entry.symbol.clone(), entry);

        self.count += 1;
    }

    /// Get all dividends for a specific date. O(1).
    pub fn get_by_date(&self, date: NaiveDate) -> impl Iterator<Item = &DividendEntry> {
        self.by_date
            .get(&date)
            .into_iter()
            .flat_map(|m: &HashMap<String, DividendEntry>| m.values())
    }

    /// Get dividend for a specific symbol on a date. O(1).
    pub fn get(&self, date: NaiveDate, symbol: &str) -> Option<&DividendEntry> {
        self.by_date.get(&date)?.get(symbol)
    }

    /// Check if there are any dividends on a date.
    pub fn has_dividends(&self, date: NaiveDate) -> bool {
        self.by_date
            .get(&date)
            .is_some_and(|m: &HashMap<String, DividendEntry>| !m.is_empty())
    }

    /// Get all dividends for a symbol within a date range.
    pub fn get_for_symbol(&self, symbol: &str, start: NaiveDate, end: NaiveDate) -> Vec<&DividendEntry> {
        self.by_symbol
            .get(symbol)
            .into_iter()
            .flat_map(|dates| dates.iter())
            .filter(|(date, _)| *date >= start && *date <= end)
            .filter_map(|(date, _)| self.get(*date, symbol))
            .collect()
    }

    /// Get all unique dates with dividends.
    pub fn dividend_dates(&self) -> impl Iterator<Item = &NaiveDate> {
        self.by_date.keys()
    }

    /// Get all symbols with dividends.
    pub fn symbols(&self) -> impl Iterator<Item = &String> {
        self.by_symbol.keys()
    }

    /// Total number of dividend entries.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if index is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get total dividend value for a symbol in a date range.
    pub fn total_dividends(&self, symbol: &str, start: NaiveDate, end: NaiveDate) -> Decimal {
        self.by_symbol
            .get(symbol)
            .into_iter()
            .flat_map(|dates| dates.iter())
            .filter(|(date, _)| *date >= start && *date <= end)
            .map(|(_, rate)| *rate)
            .sum()
    }
}

// =============================================================================
// DIVIDEND APPLICATION
// =============================================================================

/// Result of applying a dividend to a position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DividendApplication {
    /// Symbol that received dividend
    pub symbol: String,
    /// Date dividend was applied (ex_date)
    pub date: NaiveDate,
    /// Dividend rate per share
    pub rate: Decimal,
    /// Number of shares held
    pub shares: i64,
    /// Total cashflow credited (rate * shares)
    pub cashflow: Decimal,
}

impl DividendApplication {
    /// Create a new dividend application record.
    pub fn new(symbol: impl Into<String>, date: NaiveDate, rate: Decimal, shares: i64) -> Self {
        let cashflow = rate * Decimal::from(shares);
        Self {
            symbol: symbol.into(),
            date,
            rate,
            shares,
            cashflow,
        }
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
    fn test_dividend_index_basic() {
        let mut index = DividendIndex::new();
        let date = NaiveDate::from_ymd_opt(2025, 3, 15).unwrap();

        index.add(DividendEntry::cash("TAEE11", date, dec!(0.45)));
        index.add(DividendEntry::cash("BBSE3", date, dec!(0.30)));

        assert!(index.has_dividends(date));
        assert_eq!(index.len(), 2);

        let div = index.get(date, "TAEE11").unwrap();
        assert_eq!(div.rate, dec!(0.45));
    }

    #[test]
    fn test_dividend_index_by_symbol() {
        let mut index = DividendIndex::new();
        let d1 = NaiveDate::from_ymd_opt(2025, 1, 15).unwrap();
        let d2 = NaiveDate::from_ymd_opt(2025, 4, 15).unwrap();
        let d3 = NaiveDate::from_ymd_opt(2025, 7, 15).unwrap();

        index.add(DividendEntry::cash("TAEE11", d1, dec!(0.40)));
        index.add(DividendEntry::cash("TAEE11", d2, dec!(0.45)));
        index.add(DividendEntry::cash("TAEE11", d3, dec!(0.50)));

        let total = index.total_dividends("TAEE11", d1, d3);
        assert_eq!(total, dec!(1.35));

        let divs = index.get_for_symbol("TAEE11", d1, d2);
        assert_eq!(divs.len(), 2);
    }

    #[test]
    fn test_no_dividends_on_date() {
        let index = DividendIndex::new();
        let date = NaiveDate::from_ymd_opt(2025, 3, 15).unwrap();

        assert!(!index.has_dividends(date));
        assert_eq!(index.get_by_date(date).count(), 0);
    }

    #[test]
    fn test_dividend_application() {
        let app = DividendApplication::new("TAEE11", 
            NaiveDate::from_ymd_opt(2025, 3, 15).unwrap(),
            dec!(0.50),
            1000
        );

        assert_eq!(app.cashflow, dec!(500));
    }

    #[test]
    fn test_price_type_default() {
        assert_eq!(PriceType::default(), PriceType::Signals);
    }
}








