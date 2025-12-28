//! Combiner Runner - Backtest execution for the Generative Combiner (SCG)
//!
//! This crate provides:
//!
//! - BacktestExecutor trait
//! - LibraryExecutor (via ExperimentRunner)
//! - CliExecutor (fallback via CLI)
//! - Genome result caching
//! - Parallel execution
//!
//! # Example
//!
//! ```ignore
//! use combiner_runner::{BacktestExecutor, LibraryExecutor};
//!
//! let executor = LibraryExecutor::new();
//! let config = genome.to_strategy_config()?;
//! let result = executor.execute(&config)?;
//! ```

pub mod executor;
pub mod cache;
pub mod metrics;

pub use executor::{BacktestExecutor, LibraryExecutor, CliExecutor, BacktestOutput, ExecutionError};
pub use cache::GenomeCache;
pub use metrics::MetricsParser;

