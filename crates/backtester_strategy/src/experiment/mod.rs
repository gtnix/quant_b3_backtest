//! Experiment Orchestrator - execute strategies and generate artifacts.
//!
//! This module provides:
//! - `ExperimentRunner`: Execute single or batch strategy configs
//! - `MetricsCalculator`: Compute performance metrics (CAGR, Sharpe, etc.)
//! - `ArtifactWriter`: Generate standardized output files
//! - `Comparator`: Compare runs and detect regressions
//! - `BlockCatalog`: Generate documentation of available blocks
//! - `StressTestReport`: Stress testing with increased friction

pub mod types;
pub mod runner;
pub mod metrics;
pub mod artifacts;
pub mod comparator;
pub mod catalog;
pub mod stress;
pub mod stability;
pub mod market_data;
pub mod sensitivity;

pub use types::*;
pub use runner::{
    DividendTraceEntry, ExperimentRunner, RunnerConfig, RunnerError, SimulationOutput,
};
pub use metrics::{
    MetricsCalculator,
    VolatilityType,
    TRADING_DAYS_PER_YEAR,
    DEFAULT_RISK_FREE_RATE,
    WEIGHT_SUM_TOLERANCE,
    MIN_VOLATILITY_THRESHOLD,
    MAX_RATIO_VALUE,
};
pub use artifacts::{ArtifactFormat, ArtifactWriter};
pub use comparator::{Comparator, RegressionThresholds, RegressionThresholdsBuilder, ComparatorError};
pub use catalog::BlockCatalog;
pub use stress::{
    StressScenario, StressTestResult, StressTestReport, StressSummary, StressThresholds, ReportMetadata,
};
pub use stability::{
    StabilityAnalyzer, StabilityConfig, StabilityReport, StabilitySummary, BlockResult, BlockMetrics, StabilityMetadata,
};
pub use market_data::{MarketDataProvider, OhlcvBar, MarketDataError};
pub use sensitivity::{
    SensitivityAnalyzer, SensitivityConfig, SensitivityReport, PerturbationGenerator,
    Perturbation, PerturbationResult, MetricsSnapshot as SensitivityMetrics,
};

