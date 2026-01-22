//! Market Data Library
//!
//! Provides market data ingestion, calendar management, and data quality modules.

#![allow(dead_code)]

pub mod backtest_loader;
pub mod brapi;
pub mod calendar;
pub mod db;
pub mod fx_loader;
pub mod interest_rates;
pub mod symbol_dictionary;

pub use backtest_loader::{
    load_ohlcv_for_backtest, load_ohlcv_for_backtest_with_market, load_ohlcv_cached, 
    load_universe_symbols, get_symbol_date_range, list_symbols_for_market, get_market_date_range,
    BacktestDataError, BacktestMarketData, OhlcvBar, DataSummary, MarketDataCache,
};
pub use fx_loader::{
    load_fx_series, load_all_fx, load_all_fx_cached, get_fx_status,
    FxLoadError, FxRecord, FxSeriesInfo, FxCache,
    filename_to_pair, pair_to_filename,
};
pub use symbol_dictionary::{
    test_symbols, first_test_symbol, SymbolDictionary, SymbolDictError,
    TEST_SYMBOLS_BR, TEST_SYMBOLS_US,
};

