//! Performance, Attribution and Reporting Module
//!
//! Provides Wall Street-grade performance tracking:
//! - WAP (Weighted Average Price) cost basis
//! - Multi-currency (BR/US separate portfolios)
//! - Proportional technique attribution
//! - Historical VaR
//! - AI-readable JSON exports

pub mod types;
pub mod ledger;
pub mod attribution;
pub mod risk;
pub mod engine;
pub mod reporter;

pub use types::{
    PerformanceSnapshot, PnLBreakdown, CostBreakdown, ExposureBreakdown,
    DrawdownMetrics, TurnoverMetrics, PositionLot, TradeRecord, TradeSide,
    VolatilityMetrics, VaRMetrics, VaRMethod, TechniqueAttribution,
    AttributionBreakdown, CIOView,
};
pub use ledger::TradeLedger;
pub use attribution::AttributionEngine;
pub use risk::RiskCalculator;
pub use engine::PerformanceEngine;
pub use reporter::PerformanceReporter;

