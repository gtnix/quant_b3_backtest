//! Entry Module - Transforms AO7 scores into portfolio positions and orders.
//!
//! # Architecture
//!
//! ```text
//! Universe → GATING → SCORING → SELECTION → WEIGHTING → ORDERS → AUDIT
//! ```
//!
//! # Components
//!
//! - **Gating**: Filters out ineligible assets (liquidity, price, data requirements)
//! - **Selection**: Picks top-N assets per market based on scores
//! - **Weighting**: Applies risk-parity (1/volatility) to determine weights
//! - **Orders**: Generates buy/sell orders with costs and lot sizes
//! - **Audit**: Logs all decisions for traceability

mod types;
mod gating;
mod selection;
mod weighting;
mod orders;
mod engine;
mod audit;

pub use types::{
    EntryContext, EntryTarget, EntryExclusion, EntryResult, EntryDiagnostics,
    ExclusionReason, ExclusionStage, SelectionReason, Order, OrderSide,
};
pub use gating::{GatingFilter, GatingConfig, GatingCandidate};
pub use selection::{SelectionConfig, Selector, ScoredCandidate};
pub use weighting::{WeightingConfig, WeightingMethod, Weighter, WeightingCandidate};
pub use orders::{OrderGeneratorConfig, OrderGenerator};
pub use engine::{AssetCandidate, EntryEngine, EntryEngineConfig};
pub use audit::{RebalanceAuditLog, AuditLogger};

