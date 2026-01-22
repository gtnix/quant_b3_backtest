//! Symbol Dictionary - Centralized symbol management for BR and US markets.
//!
//! This module provides a single source of truth for all trading symbols,
//! eliminating hardcoded symbols scattered across the codebase.

use crate::calendar::Market;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use thiserror::Error;

/// Error types for symbol dictionary operations.
#[derive(Debug, Error)]
pub enum SymbolDictError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("CSV parse error: {0}")]
    CsvParse(String),
    
    #[error("No symbols found in file")]
    NoSymbols,
}

// =============================================================================
// TEST SYMBOLS - Static symbols for unit tests (no file dependency)
// =============================================================================

/// Brazilian market test symbols (B3 - IBOV constituents).
pub const TEST_SYMBOLS_BR: &[&str] = &[
    "PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3",
    "B3SA3", "WEGE3", "RENT3", "MGLU3", "GGBR4",
];

/// US market test symbols (S&P 500 constituents).
pub const TEST_SYMBOLS_US: &[&str] = &[
    "AAPL", "MSFT", "GOOGL", "AMZN", "SPY",
    "META", "NVDA", "TSLA", "BRK-B", "JPM",
];

/// Get static test symbols for a market.
/// These are hardcoded for unit tests that don't need real data.
#[must_use]
pub fn test_symbols(market: Market) -> &'static [&'static str] {
    match market {
        Market::BR => TEST_SYMBOLS_BR,
        Market::US => TEST_SYMBOLS_US,
    }
}

/// Get the first test symbol for a market (convenience for single-symbol tests).
#[must_use]
pub fn first_test_symbol(market: Market) -> &'static str {
    test_symbols(market)[0]
}

// =============================================================================
// SYMBOL DICTIONARY - Dynamic symbol loading from market data CSVs
// =============================================================================

/// Symbol dictionary with symbols loaded from market data files.
#[derive(Debug, Clone, Default)]
pub struct SymbolDictionary {
    br_symbols: Vec<String>,
    us_symbols: Vec<String>,
}

impl SymbolDictionary {
    /// Create an empty dictionary.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Load symbols from a market data CSV file.
    /// Expects first column to be "symbol" or similar.
    pub fn load_from_csv(&mut self, path: &Path, market: Market) -> Result<usize, SymbolDictError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut symbols = HashSet::new();
        let mut header_skipped = false;
        
        for line in reader.lines() {
            let line = line?;
            if !header_skipped {
                header_skipped = true;
                continue;
            }
            
            // Extract first column (symbol)
            if let Some(symbol) = line.split(',').next() {
                let symbol = symbol.trim().to_uppercase();
                if !symbol.is_empty() && symbol != "SYMBOL" {
                    symbols.insert(symbol);
                }
            }
        }
        
        if symbols.is_empty() {
            return Err(SymbolDictError::NoSymbols);
        }
        
        let mut sorted: Vec<String> = symbols.into_iter().collect();
        sorted.sort();
        let count = sorted.len();
        
        match market {
            Market::BR => self.br_symbols = sorted,
            Market::US => self.us_symbols = sorted,
        }
        
        Ok(count)
    }
    
    /// Get symbols for a market.
    #[must_use]
    pub fn symbols(&self, market: Market) -> &[String] {
        match market {
            Market::BR => &self.br_symbols,
            Market::US => &self.us_symbols,
        }
    }
    
    /// Check if a symbol exists in the dictionary.
    #[must_use]
    pub fn contains(&self, symbol: &str, market: Market) -> bool {
        let symbols = self.symbols(market);
        symbols.iter().any(|s| s.eq_ignore_ascii_case(symbol))
    }
    
    /// Get the number of symbols for a market.
    #[must_use]
    pub fn len(&self, market: Market) -> usize {
        self.symbols(market).len()
    }
    
    /// Check if the dictionary is empty for a market.
    #[must_use]
    pub fn is_empty(&self, market: Market) -> bool {
        self.symbols(market).is_empty()
    }
    
    /// Load default market data files from standard paths.
    pub fn load_defaults(data_dir: &Path) -> Result<Self, SymbolDictError> {
        let mut dict = Self::new();
        
        let br_path = data_dir.join("market_data_ibov.csv");
        if br_path.exists() {
            dict.load_from_csv(&br_path, Market::BR)?;
        }
        
        let us_path = data_dir.join("market_data_us.csv");
        if us_path.exists() {
            dict.load_from_csv(&us_path, Market::US)?;
        }
        
        Ok(dict)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_static_symbols_br() {
        let symbols = test_symbols(Market::BR);
        assert!(symbols.len() >= 5);
        assert!(symbols.contains(&"PETR4"));
        assert!(symbols.contains(&"VALE3"));
    }
    
    #[test]
    fn test_static_symbols_us() {
        let symbols = test_symbols(Market::US);
        assert!(symbols.len() >= 5);
        assert!(symbols.contains(&"AAPL"));
        assert!(symbols.contains(&"MSFT"));
    }
    
    #[test]
    fn test_first_symbol() {
        assert_eq!(first_test_symbol(Market::BR), "PETR4");
        assert_eq!(first_test_symbol(Market::US), "AAPL");
    }
    
    #[test]
    fn test_empty_dictionary() {
        let dict = SymbolDictionary::new();
        assert!(dict.is_empty(Market::BR));
        assert!(dict.is_empty(Market::US));
    }
}
