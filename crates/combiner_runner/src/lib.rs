//! Combiner Runner - Backtest execution for the Generative Combiner (SCG)
//!
//! This crate provides:
//!
//! - BacktestExecutor trait
//! - LibraryExecutor (via ExperimentRunner)
//! - CliExecutor (fallback via CLI)
//! - Ultra-fast genome result caching (lock-free)
//! - Split-level validation caching
//! - Parallel execution
//!
//! # Example
//!
//! ```ignore
//! use combiner_runner::{BacktestExecutor, LibraryExecutor, ValidationCache};
//!
//! let executor = LibraryExecutor::new();
//! let cache = ValidationCache::new();
//!
//! // Check cache first
//! if let Some(fitness) = cache.get_fitness(genome.hash()) {
//!     // Use cached result
//! } else {
//!     let config = genome.to_strategy_config()?;
//!     let result = executor.execute(&config)?;
//!     cache.insert_fitness(genome.hash(), fitness, generation);
//! }
//! ```

pub mod executor;
pub mod cache;
pub mod metrics;
pub mod data_loader;
pub mod data_cache;
pub mod in_process;

pub use executor::{BacktestExecutor, LibraryExecutor, CliExecutor, BacktestOutput, ExecutionError};
pub use cache::{
    GenomeCache, SplitCache, ValidationCache,
    SplitMetrics, ValidationCacheEntry,
    CacheStats, CacheStatsSnapshot, CombinedCacheStats,
    make_split_key, genome_hash_from_key, split_index_from_key,
};
pub use metrics::MetricsParser;
pub use data_loader::{MmapOhlcv, SharedMmapOhlcv, load_shared};
pub use data_cache::{InMemoryMarketData, SharedMarketData, DataCacheError};
pub use in_process::InProcessExecutor;

