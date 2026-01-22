//! In-memory market data cache for ultra-fast backtest evaluation.
//!
//! This module provides pre-loaded market data that can be reused across
//! many backtest evaluations, eliminating CSV parsing overhead.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use backtester_engine::{DualPriceBar, SymbolRegistry};
use backtester_core::fixed::Price;
use chrono::NaiveDate;
use thiserror::Error;

/// Errors during data loading.
#[derive(Debug, Error)]
pub enum DataCacheError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Parse error on line {line}: {message}")]
    Parse { line: usize, message: String },
    
    #[error("Empty data file")]
    EmptyData,
}

/// In-memory market data organized by date for sequential iteration.
/// 
/// This is the core data structure for ultra-fast backtesting.
/// Data is pre-loaded once and reused for all evaluations.
#[derive(Debug, Clone)]
pub struct InMemoryMarketData {
    /// Bars grouped by date, sorted chronologically
    days: Vec<(NaiveDate, Vec<DualPriceBar>)>,
    /// Symbol registry for name <-> id mapping
    symbol_registry: Arc<SymbolRegistry>,
    /// Total number of bars
    total_bars: usize,
    /// Date range
    start_date: Option<NaiveDate>,
    end_date: Option<NaiveDate>,
}

impl InMemoryMarketData {
    /// Load market data from a CSV file.
    /// 
    /// Expected format: symbol,date,open,high,low,close,adj_close,volume
    pub fn from_csv<P: AsRef<Path>>(path: P) -> Result<Self, DataCacheError> {
        let file = File::open(path.as_ref())?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        
        // Skip header
        let _header = lines.next().ok_or(DataCacheError::EmptyData)??;
        
        // Build symbol registry
        let mut symbol_registry = SymbolRegistry::new();
        let mut temp_data: HashMap<NaiveDate, Vec<DualPriceBar>> = HashMap::new();
        let mut total_bars = 0;
        
        for (line_num, line_result) in lines.enumerate() {
            let line = line_result?;
            if line.is_empty() {
                continue;
            }
            
            let bar = Self::parse_line(&line, line_num + 2, &mut symbol_registry)?;
            temp_data.entry(bar.date).or_default().push(bar);
            total_bars += 1;
        }
        
        if temp_data.is_empty() {
            return Err(DataCacheError::EmptyData);
        }
        
        // Sort by date
        let mut days: Vec<_> = temp_data.into_iter().collect();
        days.sort_by_key(|(date, _)| *date);
        
        // Sort bars within each day by symbol_id for determinism
        for (_, bars) in days.iter_mut() {
            bars.sort_by_key(|b| b.symbol_id);
        }
        
        let start_date = days.first().map(|(d, _)| *d);
        let end_date = days.last().map(|(d, _)| *d);
        
        Ok(Self {
            days,
            symbol_registry: Arc::new(symbol_registry),
            total_bars,
            start_date,
            end_date,
        })
    }
    
    /// Parse a single CSV line into a DualPriceBar.
    fn parse_line(
        line: &str,
        line_num: usize,
        registry: &mut SymbolRegistry,
    ) -> Result<DualPriceBar, DataCacheError> {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 8 {
            return Err(DataCacheError::Parse {
                line: line_num,
                message: format!("Expected 8 columns, got {}", parts.len()),
            });
        }
        
        let symbol = parts[0];
        let symbol_id = registry.register(symbol);
        
        let date = NaiveDate::parse_from_str(parts[1], "%Y-%m-%d")
            .map_err(|e| DataCacheError::Parse {
                line: line_num,
                message: format!("Invalid date '{}': {}", parts[1], e),
            })?;
        
        let parse_price = |s: &str, field: &str| -> Result<Price, DataCacheError> {
            s.parse::<f64>()
                .map(Price::from_f64)
                .map_err(|e| DataCacheError::Parse {
                    line: line_num,
                    message: format!("Invalid {} '{}': {}", field, s, e),
                })
        };
        
        let open = parse_price(parts[2], "open")?;
        let high = parse_price(parts[3], "high")?;
        let low = parse_price(parts[4], "low")?;
        let close = parse_price(parts[5], "close")?;
        let adj_close = parse_price(parts[6], "adj_close")?;
        
        let volume: i64 = parts[7].parse().map_err(|e| DataCacheError::Parse {
            line: line_num,
            message: format!("Invalid volume '{}': {}", parts[7], e),
        })?;
        
        Ok(DualPriceBar::new(
            symbol_id,
            date,
            adj_close,  // adjusted for signals
            close,      // raw for valuation
            open,
            high,
            low,
            volume,
        ))
    }
    
    /// Iterate over days in chronological order.
    /// 
    /// Returns (date, &[DualPriceBar]) for each trading day.
    #[inline]
    pub fn iter_days(&self) -> impl Iterator<Item = (NaiveDate, &[DualPriceBar])> {
        self.days.iter().map(|(date, bars)| (*date, bars.as_slice()))
    }
    
    /// Get number of trading days.
    #[inline]
    pub fn num_days(&self) -> usize {
        self.days.len()
    }
    
    /// Get total number of bars.
    #[inline]
    pub fn total_bars(&self) -> usize {
        self.total_bars
    }
    
    /// Get symbol registry for name resolution.
    pub fn symbol_registry(&self) -> &Arc<SymbolRegistry> {
        &self.symbol_registry
    }
    
    /// Get number of unique symbols.
    pub fn num_symbols(&self) -> usize {
        self.symbol_registry.len()
    }
    
    /// Get date range.
    pub fn date_range(&self) -> (Option<NaiveDate>, Option<NaiveDate>) {
        (self.start_date, self.end_date)
    }
    
    /// Get bars for a specific date.
    pub fn get_day(&self, date: NaiveDate) -> Option<&[DualPriceBar]> {
        self.days
            .binary_search_by_key(&date, |(d, _)| *d)
            .ok()
            .map(|idx| self.days[idx].1.as_slice())
    }
    
    /// Get all unique symbol names.
    pub fn symbol_names(&self) -> Vec<String> {
        self.symbol_registry.symbols().to_vec()
    }
    
    /// Get price history for a symbol up to (and including) a given date.
    /// Returns Vec of (date, close_price) tuples in chronological order.
    pub fn price_history(&self, symbol: &str, up_to_date: NaiveDate) -> Vec<(NaiveDate, f64)> {
        let symbol_id = match self.symbol_registry.get(symbol) {
            Some(id) => id,
            None => return Vec::new(),
        };
        
        self.days
            .iter()
            .filter(|(date, _)| *date <= up_to_date)
            .filter_map(|(date, bars)| {
                bars.iter()
                    .find(|b| b.symbol_id == symbol_id)
                    .map(|b| (*date, b.raw_close.to_f64()))
            })
            .collect()
    }
    
    /// Get close prices for a symbol up to a given date (just the prices, no dates).
    pub fn close_prices(&self, symbol: &str, up_to_date: NaiveDate) -> Vec<f64> {
        self.price_history(symbol, up_to_date)
            .into_iter()
            .map(|(_, price)| price)
            .collect()
    }
    
    /// Calculate simple returns from price history.
    pub fn returns(&self, symbol: &str, up_to_date: NaiveDate) -> Vec<f64> {
        let prices = self.close_prices(symbol, up_to_date);
        if prices.len() < 2 {
            return Vec::new();
        }
        prices
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }
    
    /// Calculate annualized volatility from returns.
    pub fn volatility(&self, symbol: &str, up_to_date: NaiveDate) -> Option<f64> {
        let returns = self.returns(symbol, up_to_date);
        if returns.len() < 20 {
            return None;
        }
        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
        Some(variance.sqrt() * (252.0_f64).sqrt())
    }
    
    /// Calculate momentum return over a lookback period.
    pub fn momentum_return(&self, symbol: &str, up_to_date: NaiveDate, lookback_days: usize) -> Option<f64> {
        let prices = self.close_prices(symbol, up_to_date);
        if prices.len() < lookback_days + 1 {
            return None;
        }
        let start_idx = prices.len() - lookback_days - 1;
        let start_price = prices[start_idx];
        let end_price = prices[prices.len() - 1];
        if start_price > 0.0 {
            Some((end_price - start_price) / start_price)
        } else {
            None
        }
    }
}

/// Shared reference-counted market data.
pub type SharedMarketData = Arc<InMemoryMarketData>;

/// Load and share market data.
pub fn load_shared<P: AsRef<Path>>(path: P) -> Result<SharedMarketData, DataCacheError> {
    Ok(Arc::new(InMemoryMarketData::from_csv(path)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    fn create_test_csv() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "symbol,date,open,high,low,close,adj_close,volume").unwrap();
        writeln!(file, "PETR4,2024-01-02,35.0,36.0,34.5,35.5,35.5,1000000").unwrap();
        writeln!(file, "VALE3,2024-01-02,70.0,71.0,69.0,70.5,70.5,500000").unwrap();
        writeln!(file, "PETR4,2024-01-03,35.5,37.0,35.0,36.5,36.5,1200000").unwrap();
        writeln!(file, "VALE3,2024-01-03,70.5,72.0,70.0,71.5,71.5,600000").unwrap();
        file
    }
    
    #[test]
    fn test_load_csv() {
        let file = create_test_csv();
        let data = InMemoryMarketData::from_csv(file.path()).unwrap();
        
        assert_eq!(data.num_days(), 2);
        assert_eq!(data.num_symbols(), 2);
        assert_eq!(data.total_bars(), 4);
    }
    
    #[test]
    fn test_iter_days() {
        let file = create_test_csv();
        let data = InMemoryMarketData::from_csv(file.path()).unwrap();
        
        let days: Vec<_> = data.iter_days().collect();
        assert_eq!(days.len(), 2);
        
        // First day should have 2 bars
        assert_eq!(days[0].1.len(), 2);
        // Second day should have 2 bars
        assert_eq!(days[1].1.len(), 2);
    }
    
    #[test]
    fn test_date_order() {
        let file = create_test_csv();
        let data = InMemoryMarketData::from_csv(file.path()).unwrap();
        
        let dates: Vec<_> = data.iter_days().map(|(d, _)| d).collect();
        assert!(dates.windows(2).all(|w| w[0] <= w[1]));
    }
}
