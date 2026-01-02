//! Backtest Data Loader - Loads OHLCV data from Neon for backtesting.
//!
//! This module provides efficient data loading for the backtester,
//! fetching historical price data from the Neon Postgres database.

use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use tokio_postgres::Client;
use tracing::info;

/// Error types for backtest data loading.
#[derive(Debug, Error)]
pub enum BacktestDataError {
    #[error("Database error: {0}")]
    Database(#[from] tokio_postgres::Error),
    
    #[error("No data found for symbols: {0:?}")]
    NoDataFound(Vec<String>),
    
    #[error("Date range error: start {0} > end {1}")]
    InvalidDateRange(NaiveDate, NaiveDate),
    
    #[error("Connection error: {0}")]
    Connection(String),
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
    /// Get the price to use for signals (adjusted close if available).
    pub fn signal_price(&self) -> Decimal {
        self.adj_close.unwrap_or(self.close)
    }
    
    /// Get the raw close price for valuation.
    pub fn valuation_price(&self) -> Decimal {
        self.close
    }
    
    /// Calculate daily return from previous bar.
    pub fn daily_return(&self, prev_close: Decimal) -> Option<f64> {
        if prev_close.is_zero() {
            return None;
        }
        let ret = (self.close - prev_close) / prev_close;
        ret.to_string().parse().ok()
    }
}

/// Market data for backtesting - organized by symbol.
#[derive(Debug, Clone, Default)]
pub struct BacktestMarketData {
    /// OHLCV bars by symbol.
    pub bars_by_symbol: HashMap<String, Vec<OhlcvBar>>,
    /// All unique trading dates in order.
    pub trading_dates: Vec<NaiveDate>,
    /// Total bar count.
    pub total_bars: usize,
}

impl BacktestMarketData {
    /// Get bars for a specific date across all symbols.
    pub fn bars_for_date(&self, date: NaiveDate) -> Vec<&OhlcvBar> {
        self.bars_by_symbol
            .values()
            .filter_map(|bars| bars.iter().find(|b| b.date == date))
            .collect()
    }
    
    /// Get bars for a symbol in date range.
    pub fn bars_for_symbol(&self, symbol: &str) -> Option<&Vec<OhlcvBar>> {
        self.bars_by_symbol.get(symbol)
    }
    
    /// Get number of symbols.
    pub fn num_symbols(&self) -> usize {
        self.bars_by_symbol.len()
    }
    
    /// Check if data is empty.
    pub fn is_empty(&self) -> bool {
        self.total_bars == 0
    }
}

/// Load OHLCV data from Neon for backtesting.
///
/// # Arguments
/// * `client` - Postgres client connection
/// * `symbols` - List of symbols to load (empty = all symbols)
/// * `start_date` - Start of date range (inclusive)
/// * `end_date` - End of date range (inclusive)
///
/// # Returns
/// `BacktestMarketData` with all bars organized by symbol.
pub async fn load_ohlcv_for_backtest(
    client: &Client,
    symbols: &[String],
    start_date: NaiveDate,
    end_date: NaiveDate,
) -> Result<BacktestMarketData, BacktestDataError> {
    if start_date > end_date {
        return Err(BacktestDataError::InvalidDateRange(start_date, end_date));
    }
    
    info!(
        "Loading OHLCV data: {} symbols, {} to {}",
        if symbols.is_empty() { "all".to_string() } else { symbols.len().to_string() },
        start_date,
        end_date
    );
    
    let query = if symbols.is_empty() {
        // Load all symbols
        "SELECT symbol, trading_date, open, high, low, close, adj_close, volume
         FROM ohlcv_daily
         WHERE trading_date >= $1 AND trading_date <= $2
         ORDER BY symbol, trading_date".to_string()
    } else {
        // Load specific symbols
        format!(
            "SELECT symbol, trading_date, open, high, low, close, adj_close, volume
             FROM ohlcv_daily
             WHERE trading_date >= $1 AND trading_date <= $2
               AND symbol = ANY($3)
             ORDER BY symbol, trading_date"
        )
    };
    
    let rows = if symbols.is_empty() {
        client.query(&query, &[&start_date, &end_date]).await?
    } else {
        client.query(&query, &[&start_date, &end_date, &symbols]).await?
    };
    
    if rows.is_empty() {
        return Err(BacktestDataError::NoDataFound(symbols.to_vec()));
    }
    
    let mut bars_by_symbol: HashMap<String, Vec<OhlcvBar>> = HashMap::new();
    let mut all_dates: std::collections::BTreeSet<NaiveDate> = std::collections::BTreeSet::new();
    
    for row in &rows {
        let symbol: String = row.get(0);
        let date: NaiveDate = row.get(1);
        let open: Decimal = row.get(2);
        let high: Decimal = row.get(3);
        let low: Decimal = row.get(4);
        let close: Decimal = row.get(5);
        let adj_close: Option<Decimal> = row.get(6);
        let volume: i64 = row.get(7);
        
        let bar = OhlcvBar {
            symbol: symbol.clone(),
            date,
            open,
            high,
            low,
            close,
            adj_close,
            volume,
        };
        
        all_dates.insert(date);
        bars_by_symbol.entry(symbol).or_default().push(bar);
    }
    
    let total_bars = rows.len();
    let trading_dates: Vec<NaiveDate> = all_dates.into_iter().collect();
    
    info!(
        "Loaded {} bars for {} symbols across {} trading days",
        total_bars,
        bars_by_symbol.len(),
        trading_dates.len()
    );
    
    Ok(BacktestMarketData {
        bars_by_symbol,
        trading_dates,
        total_bars,
    })
}

/// Load symbols from the active universe.
pub async fn load_universe_symbols(
    client: &Client,
    universe_name: Option<&str>,
) -> Result<Vec<String>, BacktestDataError> {
    let query = if let Some(name) = universe_name {
        format!(
            "SELECT DISTINCT i.symbol
             FROM instruments i
             JOIN universe_membership um ON i.symbol = um.symbol
             WHERE i.active = true AND um.universe_name = '{}'
             ORDER BY i.symbol",
            name
        )
    } else {
        "SELECT symbol FROM instruments WHERE active = true ORDER BY symbol".to_string()
    };
    
    let rows = client.query(&query, &[]).await?;
    let symbols: Vec<String> = rows.iter().map(|r| r.get(0)).collect();
    
    info!("Loaded {} symbols from universe {:?}", symbols.len(), universe_name);
    Ok(symbols)
}

/// Get available date range for a symbol.
pub async fn get_symbol_date_range(
    client: &Client,
    symbol: &str,
) -> Result<Option<(NaiveDate, NaiveDate)>, BacktestDataError> {
    let row = client
        .query_opt(
            "SELECT MIN(trading_date), MAX(trading_date) 
             FROM ohlcv_daily 
             WHERE symbol = $1",
            &[&symbol],
        )
        .await?;
    
    match row {
        Some(r) => {
            let min_date: Option<NaiveDate> = r.get(0);
            let max_date: Option<NaiveDate> = r.get(1);
            Ok(min_date.zip(max_date))
        }
        None => Ok(None),
    }
}

/// Get summary statistics for loaded data.
#[derive(Debug, Clone, Serialize)]
pub struct DataSummary {
    pub symbols: usize,
    pub total_bars: usize,
    pub trading_days: usize,
    pub start_date: Option<NaiveDate>,
    pub end_date: Option<NaiveDate>,
    pub avg_bars_per_symbol: f64,
}

impl BacktestMarketData {
    /// Generate summary statistics.
    pub fn summary(&self) -> DataSummary {
        DataSummary {
            symbols: self.bars_by_symbol.len(),
            total_bars: self.total_bars,
            trading_days: self.trading_dates.len(),
            start_date: self.trading_dates.first().copied(),
            end_date: self.trading_dates.last().copied(),
            avg_bars_per_symbol: if self.bars_by_symbol.is_empty() {
                0.0
            } else {
                self.total_bars as f64 / self.bars_by_symbol.len() as f64
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ohlcv_bar_signal_price() {
        let bar = OhlcvBar {
            symbol: "PETR4".to_string(),
            date: NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
            open: Decimal::new(100, 1),
            high: Decimal::new(105, 1),
            low: Decimal::new(95, 1),
            close: Decimal::new(102, 1),
            adj_close: Some(Decimal::new(101, 1)),
            volume: 1000000,
        };
        
        // Should use adj_close when available
        assert_eq!(bar.signal_price(), Decimal::new(101, 1));
        assert_eq!(bar.valuation_price(), Decimal::new(102, 1));
    }
    
    #[test]
    fn test_ohlcv_bar_no_adj_close() {
        let bar = OhlcvBar {
            symbol: "PETR4".to_string(),
            date: NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
            open: Decimal::new(100, 1),
            high: Decimal::new(105, 1),
            low: Decimal::new(95, 1),
            close: Decimal::new(102, 1),
            adj_close: None,
            volume: 1000000,
        };
        
        // Should fall back to close
        assert_eq!(bar.signal_price(), Decimal::new(102, 1));
    }
    
    #[test]
    fn test_daily_return() {
        let bar = OhlcvBar {
            symbol: "PETR4".to_string(),
            date: NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
            open: Decimal::new(100, 1),
            high: Decimal::new(105, 1),
            low: Decimal::new(95, 1),
            close: Decimal::new(110, 1), // 11.0
            adj_close: None,
            volume: 1000000,
        };
        
        let prev_close = Decimal::new(100, 1); // 10.0
        let ret = bar.daily_return(prev_close).unwrap();
        assert!((ret - 0.1).abs() < 0.001); // 10% return
    }
    
    #[test]
    fn test_market_data_empty() {
        let data = BacktestMarketData::default();
        assert!(data.is_empty());
        assert_eq!(data.num_symbols(), 0);
    }
}

