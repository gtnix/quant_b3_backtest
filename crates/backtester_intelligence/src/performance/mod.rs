//! Performance, Attribution and Reporting Module
//!
//! Provides Wall Street-grade performance tracking:
//! - WAP (Weighted Average Price) cost basis
//! - Multi-currency (BR/US separate portfolios)
//! - FX attribution with 3-term decomposition (asset, fx, interaction)
//! - Proportional technique attribution
//! - Historical VaR
//! - Sector exposure analysis
//! - Portfolio constraints and compliance
//! - AI-readable JSON exports

pub mod types;
pub mod ledger;
pub mod attribution;
pub mod fx_attribution;
pub mod risk;
pub mod engine;
pub mod reporter;
pub mod sector;
pub mod concentration;
pub mod regime;
pub mod compliance;
pub mod constraints;

pub use types::{
    PerformanceSnapshot, PnLBreakdown, CostBreakdown, ExposureBreakdown,
    DrawdownMetrics, TurnoverMetrics, PositionLot, TradeRecord, TradeSide,
    VolatilityMetrics, VaRMetrics, VaRMethod, TechniqueAttribution,
    AttributionBreakdown, CIOView, FxRateInfo, FxResolutionMethod,
    SectorExposure,
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
    // Research-grade JSON types (v1.2)
    SectorExposureJson, ConcentrationJson, RegimeSummaryJson, RegimePerformanceJson,
    RegimeConfigJson,
    // Compliance JSON types (v1.3)
    ComplianceJson, ComplianceSummaryJson, BreachEventJson, BreachEvidenceJson,
    ActionRecordJson, ConstraintPolicyJson,
};
pub use sector::{Sector, SectorProvider, InMemorySectorProvider, CsvSectorProvider, NullSectorProvider};
pub use concentration::{ConcentrationMetrics, ConcentrationCalculator};
pub use regime::{
    TrendState, VolQuantile, RegimeTag, RegimePerformance, RegimeConfig, RegimeSummary, RegimeEngine,
};
pub use compliance::{
    BreachEvent, BreachEvidence, BreachLog, ComplianceSummary, ComplianceReport,
    ActionRecord, ConfigSnapshotEntry,
};
pub use constraints::{
    ConstraintId, ConstraintScope, ConstraintAction, ConstraintPolicy,
    ConstraintsConfig, ConstraintsEngine,
};







