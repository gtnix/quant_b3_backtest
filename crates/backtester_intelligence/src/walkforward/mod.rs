//! Walk-Forward Validation Module
//!
//! Provides rolling window validation for robust strategy testing.
//!
//! # Features
//!
//! - Rolling splits with configurable train/test/step periods
//! - Purge and embargo to prevent data leakage
//! - Grid search for parameter optimization
//! - Aggregate metrics and robustness scoring
//!
//! # Example
//!
//! ```ignore
//! use backtester_intelligence::walkforward::*;
//!
//! let config = WalkForwardConfig {
//!     train_months: 6,
//!     test_months: 3,
//!     step_months: 3,
//!     purge_days: 5,
//!     embargo_days: 5,
//!     ..Default::default()
//! };
//!
//! let splitter = RollingSplitter::new(&config);
//! let splits = splitter.generate_splits(start_date, end_date);
//! ```

pub mod types;
pub mod splitter;
pub mod metrics;
pub mod runner;
pub mod reporter;

pub use types::{
    WindowType, WindowSpec, WindowSplit, WalkForwardConfig,
    ParamSet, ParamRange, GridConfig, WindowMetrics, WindowResult,
    AggregateMetrics, AggregateReport,
};

pub use splitter::{TimeSplitter, RollingSplitter};
pub use metrics::{MetricsCalculator, RobustnessScorer};
pub use runner::WalkForwardRunner;
pub use reporter::WalkForwardReporter;

