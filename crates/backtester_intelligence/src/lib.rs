//! Asset Selection Intelligence Modules
//!
//! This crate provides modular asset filtering and scoring capabilities
//! for the backtester. Filters can be combined to select assets based on:
//!
//! - **Momentum**: Price performance over lookback periods
//! - **Value**: P/E, P/B ratios
//! - **Quality**: ROE, debt levels, profit margins
//! - **Low Volatility**: Historical volatility filters
//! - **Dividend Yield**: Dividend-based selection
//! - **Size**: Market capitalization filters
//! - **Carry**: Dividend yield vs risk-free rate (Technique 7)
//!
//! # Entry Module
//!
//! The `entry` module handles the full portfolio construction flow:
//! - **Gating**: Filter ineligible assets (liquidity, data requirements)
//! - **Selection**: Pick top-N per market based on scores
//! - **Weighting**: Apply risk-parity (1/volatility)
//! - **Orders**: Generate buy/sell with costs and lot sizes
//! - **Audit**: Log all decisions for traceability
//!
//! # Exit Module
//!
//! The `exit` module handles position exits and risk management:
//! - **Stop-Loss**: Exit on excessive loss
//! - **Take-Profit**: Exit on target gain
//! - **Time-Based Exit**: Exit after max holding period
//! - **Trailing Stop**: Exit on drawdown from high-water mark
//! - **Risk Guards**: Exposure limits, turnover limits, drawdown guard

pub mod accounting;
pub mod config;
pub mod entry;
pub mod exit;
pub mod filters;
pub mod monitoring;
pub mod orchestrator;
pub mod performance;
pub mod risk_free;
pub mod scorer;
pub mod walkforward;

pub use config::{
    AssetFilterConfig, CarryConfig, FilterMode, FundamentalsConfig, IntelligenceConfig,
    RiskFreeConfig,
};
pub use filters::{infer_market_from_symbol, AssetData, AssetFilter, FilterResult, Market};
pub use risk_free::{DbRiskFreeRateProvider, FallbackRiskFreeProvider, RiskFreeRateProvider};
pub use scorer::{AssetScorer, ScoredAsset};

// Entry module exports
pub use entry::{
    AssetCandidate, EntryEngine, EntryEngineConfig, EntryContext, EntryResult, EntryTarget, 
    EntryExclusion, EntryDiagnostics, GatingFilter, GatingConfig, Selector, SelectionConfig, 
    Weighter, WeightingConfig, WeightingMethod, OrderGenerator, OrderGeneratorConfig, Order, 
    OrderSide, AuditLogger, RebalanceAuditLog, ExclusionReason, ExclusionStage, SelectionReason,
};

// Exit module exports
pub use exit::{
    ExitEngine, ExitEngineConfig, ExitContext, ExitResult, ExitTarget, ExitReason,
    ExitDiagnostics, Position, RiskViolation, DrawdownAction,
    ExitPolicy, ExitPolicyConfig, StopLossConfig, StopLossPolicy,
    TakeProfitConfig, TakeProfitPolicy, TimeExitConfig, TimeExitPolicy,
    TrailingStopConfig, TrailingStopPolicy, RiskConfig, RiskGuard,
    ExitAuditLog, ExitedPosition,
};

// Orchestrator exports
pub use orchestrator::{
    RebalanceOrchestrator, OrchestratorConfig, RebalanceStepResult, RebalanceStepAudit,
};

// Accounting exports
pub use accounting::{PortfolioState, AccountingError};

// Performance exports
pub use performance::{
    PerformanceSnapshot, PnLBreakdown, CostBreakdown, ExposureBreakdown,
    DrawdownMetrics, TurnoverMetrics, PositionLot, TradeRecord, TradeSide,
    VolatilityMetrics, VaRMetrics, VaRMethod, TechniqueAttribution,
    AttributionBreakdown, CIOView, TradeLedger, AttributionEngine,
    RiskCalculator, PerformanceEngine, PerformanceReporter,
};

// Walk-forward exports
pub use walkforward::{
    WindowType, WindowSpec, WindowSplit, WalkForwardConfig,
    ParamSet, ParamRange, GridConfig, WindowMetrics, WindowResult,
    AggregateMetrics, AggregateReport, TimeSplitter, RollingSplitter,
    MetricsCalculator, RobustnessScorer, WalkForwardRunner, WalkForwardReporter,
};

// Monitoring exports
pub use monitoring::{
    Severity, CheckCategory, CheckResult, Evidence, BaselineStats, CurrentStats,
    CircuitAction, MonitoringSummary, CircuitBreakerState, MonitoringReport,
    MonitoringConfig, DataHealthConfig, DriftConfig, RegressionConfig,
    CircuitBreakerConfig, KnownLimitationsConfig, DynamicThreshold,
    DrawdownThreshold, ThresholdEvaluator,
    DataHealthCheck, DataHealthEngine, DataContext,
    DriftCheck, DriftEngine, DriftContext,
    RegressionCheck, RegressionEngine, RegressionContext,
    CircuitBreaker, CircuitState,
    MonitoringEngine, MonitoringContext,
    MonitoringReporter, ReportFormat,
    // Baseline runner exports
    ContextBuilder, BuilderError, BaselineAggregator, BaselineReport, DailyResult,
    CheckFrequency, CategoryBreakdown, ThresholdRecommendation,
};

/// Prelude for common imports.
pub mod prelude {
    pub use crate::config::{
        AssetFilterConfig, CarryConfig, FilterMode, FundamentalsConfig, IntelligenceConfig,
        RiskFreeConfig,
    };
    pub use crate::filters::{infer_market_from_symbol, AssetData, AssetFilter, FilterResult, Market};
    pub use crate::risk_free::{
        DbRiskFreeRateProvider, FallbackRiskFreeProvider, RiskFreeRateProvider,
    };
    pub use crate::scorer::{AssetScorer, ScoredAsset};
    
    // Entry module
    pub use crate::entry::{
        EntryEngine, EntryContext, EntryResult, EntryTarget, EntryExclusion,
        GatingFilter, GatingConfig, Selector, SelectionConfig,
        Weighter, WeightingConfig, WeightingMethod,
        OrderGenerator, OrderGeneratorConfig, Order, OrderSide,
        AuditLogger, RebalanceAuditLog,
    };
    
    // Exit module
    pub use crate::exit::{
        ExitEngine, ExitEngineConfig, ExitContext, ExitResult, ExitTarget, ExitReason,
        Position, RiskViolation, DrawdownAction, RiskConfig, RiskGuard,
        ExitAuditLog,
    };
    
    // Orchestrator
    pub use crate::orchestrator::{
        RebalanceOrchestrator, OrchestratorConfig, RebalanceStepResult, RebalanceStepAudit,
    };
    
    // Performance
    pub use crate::performance::{
        PerformanceSnapshot, PnLBreakdown, TradeLedger, PerformanceEngine,
        AttributionBreakdown, CIOView, PerformanceReporter,
    };
    
    // Walk-forward
    pub use crate::walkforward::{
        WalkForwardConfig, WalkForwardRunner, WalkForwardReporter,
        AggregateReport, WindowResult, RollingSplitter, TimeSplitter,
    };
    
    // Monitoring
    pub use crate::monitoring::{
        MonitoringEngine, MonitoringContext, MonitoringConfig,
        MonitoringReport, MonitoringReporter, Severity, CircuitAction,
    };
}
