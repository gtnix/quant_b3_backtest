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

pub mod config;
pub mod entry;
pub mod filters;
pub mod risk_free;
pub mod scorer;

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
}
