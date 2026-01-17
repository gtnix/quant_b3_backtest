//! Market Data Provider for Experiment Runner.
//!
//! Provides market data for backtesting simulations.
//! Supports loading from CSV files, in-memory data, or Neon database.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// Re-export market_data types for database loading
pub use market_data::backtest_loader::{
    BacktestMarketData, 
    OhlcvBar as DbOhlcvBar,
    load_ohlcv_for_backtest_with_market,
    list_symbols_for_market,
};

/// Error types for market data loading.
#[derive(Debug, Error)]
pub enum MarketDataError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("CSV parse error: {0}")]
    Csv(#[from] csv::Error),
    
    #[error("No data found for symbol: {0}")]
    NoDataForSymbol(String),
    
    #[error("Invalid date range: start {0} > end {1}")]
    InvalidDateRange(NaiveDate, NaiveDate),
}

/// OHLCV bar for backtesting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OhlcvBar {
    pub symbol: String,
    pub date: NaiveDate,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub adj_close: Option<Decimal>,
    pub volume: i64,
}

impl OhlcvBar {
    /// Get the signal price (adjusted close if available, else close).
    pub fn signal_price(&self) -> Decimal {
        self.adj_close.unwrap_or(self.close)
    }
    
    /// Calculate daily return.
    pub fn daily_return(&self, prev_close: Decimal) -> f64 {
        if prev_close.is_zero() {
            return 0.0;
        }
        ((self.close - prev_close) / prev_close)
            .to_f64()
            .unwrap_or(0.0)
    }
}

/// Market data provider for backtesting.
#[derive(Debug, Clone, Default)]
pub struct MarketDataProvider {
    /// Bars by symbol, sorted by date.
    bars_by_symbol: HashMap<String, Vec<OhlcvBar>>,
    /// All unique trading dates.
    trading_dates: Vec<NaiveDate>,
    /// Bars indexed by date -> symbol -> bar.
    bars_by_date: BTreeMap<NaiveDate, HashMap<String, OhlcvBar>>,
}

impl MarketDataProvider {
    /// Create an empty provider.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Load from a CSV file.
    ///
    /// Expected columns: symbol, date, open, high, low, close, adj_close, volume
    pub fn from_csv(path: &Path) -> Result<Self, MarketDataError> {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)?;
        
        let mut bars_by_symbol: HashMap<String, Vec<OhlcvBar>> = HashMap::new();
        let mut bars_by_date: BTreeMap<NaiveDate, HashMap<String, OhlcvBar>> = BTreeMap::new();
        
        for result in rdr.deserialize() {
            let bar: OhlcvBar = result?;
            
            bars_by_date
                .entry(bar.date)
                .or_default()
                .insert(bar.symbol.clone(), bar.clone());
            
            bars_by_symbol
                .entry(bar.symbol.clone())
                .or_default()
                .push(bar);
        }
        
        // Sort bars by date within each symbol
        for bars in bars_by_symbol.values_mut() {
            bars.sort_by_key(|b| b.date);
        }
        
        let trading_dates: Vec<NaiveDate> = bars_by_date.keys().copied().collect();
        
        Ok(Self {
            bars_by_symbol,
            trading_dates,
            bars_by_date,
        })
    }
    
    /// Create from in-memory bars.
    pub fn from_bars(bars: Vec<OhlcvBar>) -> Self {
        let mut bars_by_symbol: HashMap<String, Vec<OhlcvBar>> = HashMap::new();
        let mut bars_by_date: BTreeMap<NaiveDate, HashMap<String, OhlcvBar>> = BTreeMap::new();
        
        for bar in bars {
            bars_by_date
                .entry(bar.date)
                .or_default()
                .insert(bar.symbol.clone(), bar.clone());
            
            bars_by_symbol
                .entry(bar.symbol.clone())
                .or_default()
                .push(bar);
        }
        
        // Sort bars by date
        for bars in bars_by_symbol.values_mut() {
            bars.sort_by_key(|b| b.date);
        }
        
        let trading_dates: Vec<NaiveDate> = bars_by_date.keys().copied().collect();
        
        Self {
            bars_by_symbol,
            trading_dates,
            bars_by_date,
        }
    }
    
    /// Create from database-loaded BacktestMarketData.
    ///
    /// Converts from the market_data crate's BacktestMarketData type
    /// which is loaded from Neon database (ohlcv_daily or ohlcv_daily_us).
    pub fn from_backtest_data(data: BacktestMarketData) -> Self {
        let mut bars_by_symbol: HashMap<String, Vec<OhlcvBar>> = HashMap::new();
        let mut bars_by_date: BTreeMap<NaiveDate, HashMap<String, OhlcvBar>> = BTreeMap::new();
        
        // Convert DbOhlcvBar to local OhlcvBar and build indexes
        for (symbol, db_bars) in data.bars_by_symbol {
            let local_bars: Vec<OhlcvBar> = db_bars
                .into_iter()
                .map(|db_bar| {
                    let bar = OhlcvBar {
                        symbol: db_bar.symbol,
                        date: db_bar.date,
                        open: db_bar.open,
                        high: db_bar.high,
                        low: db_bar.low,
                        close: db_bar.close,
                        adj_close: db_bar.adj_close,
                        volume: db_bar.volume,
                    };
                    
                    // Also add to date index
                    bars_by_date
                        .entry(bar.date)
                        .or_default()
                        .insert(bar.symbol.clone(), bar.clone());
                    
                    bar
                })
                .collect();
            
            bars_by_symbol.insert(symbol, local_bars);
        }
        
        // Use the trading dates from the database data (already sorted)
        let trading_dates = data.trading_dates;
        
        Self {
            bars_by_symbol,
            trading_dates,
            bars_by_date,
        }
    }
    
    /// Check if data is available.
    pub fn is_empty(&self) -> bool {
        self.bars_by_symbol.is_empty()
    }
    
    /// Get number of symbols.
    pub fn num_symbols(&self) -> usize {
        self.bars_by_symbol.len()
    }
    
    /// Get number of trading days.
    pub fn num_trading_days(&self) -> usize {
        self.trading_dates.len()
    }
    
    /// Get all trading dates.
    pub fn trading_dates(&self) -> &[NaiveDate] {
        &self.trading_dates
    }
    
    /// Get bars for a specific date.
    pub fn bars_for_date(&self, date: NaiveDate) -> Option<&HashMap<String, OhlcvBar>> {
        self.bars_by_date.get(&date)
    }
    
    /// Get all bars for a symbol.
    pub fn bars_for_symbol(&self, symbol: &str) -> Option<&Vec<OhlcvBar>> {
        self.bars_by_symbol.get(symbol)
    }
    
    /// Get bar for a specific symbol and date.
    pub fn get_bar(&self, symbol: &str, date: NaiveDate) -> Option<&OhlcvBar> {
        self.bars_by_date.get(&date)?.get(symbol)
    }
    
    /// Get all symbols.
    pub fn symbols(&self) -> Vec<&str> {
        self.bars_by_symbol.keys().map(|s| s.as_str()).collect()
    }
    
    /// Get date range.
    pub fn date_range(&self) -> Option<(NaiveDate, NaiveDate)> {
        let start = self.trading_dates.first()?;
        let end = self.trading_dates.last()?;
        Some((*start, *end))
    }
    
    /// Filter to a specific date range.
    pub fn filter_date_range(&self, start: NaiveDate, end: NaiveDate) -> Self {
        let bars: Vec<OhlcvBar> = self.bars_by_symbol
            .values()
            .flat_map(|bars| {
                bars.iter()
                    .filter(|b| b.date >= start && b.date <= end)
                    .cloned()
            })
            .collect();
        
        Self::from_bars(bars)
    }
    
    /// Get total bar count.
    pub fn total_bars(&self) -> usize {
        self.bars_by_symbol.values().map(|b| b.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    
    fn sample_bars() -> Vec<OhlcvBar> {
        vec![
            OhlcvBar {
                symbol: "PETR4".to_string(),
                date: NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
                open: dec!(40.00),
                high: dec!(41.00),
                low: dec!(39.50),
                close: dec!(40.50),
                adj_close: Some(dec!(40.25)),
                volume: 1000000,
            },
            OhlcvBar {
                symbol: "PETR4".to_string(),
                date: NaiveDate::from_ymd_opt(2024, 1, 3).unwrap(),
                open: dec!(40.50),
                high: dec!(42.00),
                low: dec!(40.00),
                close: dec!(41.80),
                adj_close: Some(dec!(41.55)),
                volume: 1200000,
            },
            OhlcvBar {
                symbol: "VALE3".to_string(),
                date: NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
                open: dec!(70.00),
                high: dec!(71.50),
                low: dec!(69.00),
                close: dec!(71.00),
                adj_close: Some(dec!(70.80)),
                volume: 800000,
            },
        ]
    }
    
    #[test]
    fn test_from_bars() {
        let provider = MarketDataProvider::from_bars(sample_bars());
        
        assert_eq!(provider.num_symbols(), 2);
        assert_eq!(provider.num_trading_days(), 2);
        assert_eq!(provider.total_bars(), 3);
    }
    
    #[test]
    fn test_bars_for_date() {
        let provider = MarketDataProvider::from_bars(sample_bars());
        let date = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();
        
        let bars = provider.bars_for_date(date).unwrap();
        assert_eq!(bars.len(), 2);
        assert!(bars.contains_key("PETR4"));
        assert!(bars.contains_key("VALE3"));
    }
    
    #[test]
    fn test_get_bar() {
        let provider = MarketDataProvider::from_bars(sample_bars());
        let date = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();
        
        let bar = provider.get_bar("PETR4", date).unwrap();
        assert_eq!(bar.close, dec!(40.50));
    }
    
    #[test]
    fn test_date_range() {
        let provider = MarketDataProvider::from_bars(sample_bars());
        let (start, end) = provider.date_range().unwrap();
        
        assert_eq!(start, NaiveDate::from_ymd_opt(2024, 1, 2).unwrap());
        assert_eq!(end, NaiveDate::from_ymd_opt(2024, 1, 3).unwrap());
    }
    
    #[test]
    fn test_daily_return() {
        let bar = OhlcvBar {
            symbol: "TEST".to_string(),
            date: NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
            open: dec!(100.00),
            high: dec!(110.00),
            low: dec!(95.00),
            close: dec!(105.00),
            adj_close: None,
            volume: 1000,
        };
        
        let ret = bar.daily_return(dec!(100.00));
        assert!((ret - 0.05).abs() < 0.0001); // 5% return
    }
}

