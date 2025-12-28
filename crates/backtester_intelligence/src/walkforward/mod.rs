//! Walk-Forward Validation Module
//!
//! Provides rolling window validation for robust strategy testing.
//!
//! # Features
//!
//! - Rolling splits with configurable train/test/step periods
//! - Nested 3-segment windows (Train/Val/Test) for research-grade validation
//! - Purge and embargo to prevent data leakage
//! - PSR (Probabilistic Sharpe Ratio) and DSR (Deflated Sharpe Ratio)
//! - Grid search for parameter optimization with deterministic tie-breakers
//! - Aggregate metrics and robustness scoring
//!
//! # Example
//!
//! ```ignore
//! use backtester_intelligence::walkforward::*;
//!
//! let config = NestedWalkForwardConfig {
//!     train_months: 4,
//!     val_months: 1,
//!     test_months: 1,
//!     step_months: 3,
//!     purge_days: 5,
//!     embargo_days: 5,
//!     selection_criteria: SelectionCriteria::PSR,
//!     psr_threshold: dec!(0.5),
//!     ..Default::default()
//! };
//!
//! let splitter = NestedSplitter::new(&config);
//! let splits = splitter.generate_splits(start_date, end_date);
//! ```

pub mod types;
pub mod splitter;
pub mod metrics;
pub mod runner;
pub mod reporter;
pub mod statistics;
pub mod calendar_aware;

// Legacy 2-segment types
pub use types::{
    WindowType, WindowSpec, WindowSplit, WalkForwardConfig,
    ParamSet, ParamRange, GridConfig, WindowMetrics, WindowResult,
    AggregateMetrics, AggregateReport,
};

// New 3-segment nested types
pub use types::{
    SelectionCriteria, SelectionReason, PenaltyConfig,
    NestedWindowSplit, NestedWindowResult, NestedWalkForwardConfig,
    NestedAggregateReport, SelectionCandidate,
};

pub use splitter::{TimeSplitter, RollingSplitter, NestedSplitter};
pub use metrics::{MetricsCalculator, RobustnessScorer};
pub use runner::WalkForwardRunner;
pub use reporter::WalkForwardReporter;
pub use statistics::{calculate_psr, calculate_dsr, calculate_skewness, calculate_kurtosis};
pub use calendar_aware::{
    TradingDayCalendar, CalendarAwareRollingSplitter, CalendarAwareNestedSplitter, WindowSpecExt,
};

/// Prelude for convenient imports.
pub mod prelude {
    pub use super::{
        WindowType, WindowSpec, WindowSplit, WalkForwardConfig,
        ParamSet, ParamRange, GridConfig, WindowMetrics, WindowResult,
        AggregateMetrics, AggregateReport,
        SelectionCriteria, SelectionReason, PenaltyConfig,
        NestedWindowSplit, NestedWindowResult, NestedWalkForwardConfig,
        NestedAggregateReport, SelectionCandidate,
        TimeSplitter, RollingSplitter, NestedSplitter,
        TradingDayCalendar, CalendarAwareRollingSplitter, CalendarAwareNestedSplitter, WindowSpecExt,
        MetricsCalculator, RobustnessScorer,
        WalkForwardRunner, WalkForwardReporter,
        calculate_psr, calculate_dsr, calculate_skewness, calculate_kurtosis,
    };
}

