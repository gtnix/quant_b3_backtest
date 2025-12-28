//! Performance, Attribution and Reporting Module
//!
//! Provides Wall Street-grade performance tracking:
//! - WAP (Weighted Average Price) cost basis
//! - Multi-currency (BR/US separate portfolios)
//! - FX attribution with 3-term decomposition (asset, fx, interaction)
//! - Proportional technique attribution
//! - Historical VaR
//! - AI-readable JSON exports

pub mod types;
pub mod ledger;
pub mod attribution;
pub mod fx_attribution;
pub mod risk;
pub mod engine;
pub mod reporter;

pub use types::{
    PerformanceSnapshot, PnLBreakdown, CostBreakdown, ExposureBreakdown,
    DrawdownMetrics, TurnoverMetrics, PositionLot, TradeRecord, TradeSide,
    VolatilityMetrics, VaRMetrics, VaRMethod, TechniqueAttribution,
    AttributionBreakdown, CIOView, FxRateInfo, FxResolutionMethod,
};
pub use ledger::TradeLedger;
pub use attribution::AttributionEngine;
pub use fx_attribution::{
    FxAttributionEngine, FxAttributionBreakdown, CurrencyAttribution,
    calculate_fx_attribution,
};
pub use risk::RiskCalculator;
pub use engine::{PerformanceEngine, PerformanceConfig};
pub use reporter::{
    PerformanceReporter, PerformanceReport, PERFORMANCE_REPORT_SCHEMA_VERSION,
    // JSON types for schema stability tests
    PnLJson, CostsJson, AttributionJson, RiskJson, ExposureJson, MarketExposure,
    TurnoverJson, FxAttributionJson, CurrencyAttributionJson, CurrencyExposureJson,
    FxRateUsedJson,
};







