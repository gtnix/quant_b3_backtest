//! Market Data Library
//!
//! Provides market data ingestion, calendar management, and data quality modules.

#![allow(dead_code)]

pub mod backtest_loader;
pub mod calendar;
pub mod fx_loader;

pub use backtest_loader::{
    load_ohlcv_for_backtest, load_universe_symbols, get_symbol_date_range,
    BacktestDataError, BacktestMarketData, OhlcvBar, DataSummary,
};
pub use fx_loader::{
    load_fx_series, load_all_fx, get_fx_status,
    FxLoadError, FxRecord, FxSeriesInfo,
    filename_to_pair, pair_to_filename,
};

