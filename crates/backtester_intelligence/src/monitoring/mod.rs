//! Monitoring & Alerting Module
//!
//! Production-grade monitoring for quantitative trading systems:
//! - Data health checks (freshness, coverage, outliers, watermarks)
//! - Strategy drift detection (score distributions, selection stability)
//! - Performance regression checks (drawdown, turnover, costs)
//! - Circuit breaker for NO-TRADE conditions
//! - Structured reporting (JSON, Markdown, GitHub Actions)

mod config;
mod types;
mod statistics;
mod data_health;
mod drift;
mod regressions;
mod circuit_breaker;
mod engine;
mod reporter;
mod context_builder;
mod baseline;
mod integrity_gate;

pub use config::{
    MonitoringConfig, DataHealthConfig, DriftConfig, RegressionConfig,
    CircuitBreakerConfig, KnownLimitationsConfig, DynamicThreshold,
    DrawdownThreshold, ThresholdEvaluator,
};

pub use types::{
    Severity, CheckCategory, CheckResult, Evidence, BaselineStats, CurrentStats,
    CircuitAction, MonitoringSummary, CircuitBreakerState, MonitoringReport,
};

pub use statistics::{
    calculate_percentile, calculate_mean, calculate_std, calculate_baseline,
    hoeffding_bound, ks_two_sample, jaccard_similarity, sigma_deviation,
};

pub use data_health::{
    DataHealthCheck, DataHealthEngine, DataContext,
    FreshnessCheck, CoverageCheck, WatermarkCheck, NullCheck, OutlierCheck,
    DividendsCheck, InterestRatesCheck, SchemaCheck,
    // Data Integrity checks
    PriceJumpEvent, UniverseType,
    TemporalIntegrityCheck, LookaheadPolicyCheck, CorpActionCheck, SurvivorshipCheck,
};

pub use drift::{
    DriftCheck, DriftEngine, DriftContext,
    ScoreDistributionDrift, MeanScoreDrift, SelectionStabilityCheck,
    ExclusionReasonsDrift, TurnoverDrift, CostDrift,
};

pub use regressions::{
    RegressionCheck, RegressionEngine, RegressionContext,
    DrawdownGuardrail, TurnoverBudget, CostBudget, SharpeRegression,
    VolatilityRegression, LatencyCheck,
};

pub use circuit_breaker::{
    CircuitBreaker, CircuitState,
};

pub use engine::{
    MonitoringEngine, MonitoringContext,
};

pub use reporter::{
    MonitoringReporter, ReportFormat,
};

pub use context_builder::{
    ContextBuilder, BuilderError, BuilderResult,
    OhlcvAudit, WatermarkData, InterestRateStats,
};

pub use baseline::{
    BaselineAggregator, BaselineReport, DailyResult,
    CheckFrequency, CategoryBreakdown, CircuitBreakerStats,
    ThresholdRecommendation,
};


pub use integrity_gate::{
    DataIntegrityGate, DataIntegrityReport, Verdict, AuditMode, AuditStats,
};
