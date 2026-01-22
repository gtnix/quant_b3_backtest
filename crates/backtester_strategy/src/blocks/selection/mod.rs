//! Selection blocks - asset filtering and ranking (Techniques 1-7).
//!
//! This module contains selection blocks for filtering and ranking assets
//! before generating trading signals.

mod adapter;
mod sector_rotation;
mod multi_factor;
mod factor_variants;
mod portfolio_opt;
mod advanced;

// Re-export adapter functions (includes create_selection_block)
pub use adapter::*;

// Re-export new selection blocks
pub use sector_rotation::SectorRotationBlock;
pub use multi_factor::MultiFactorBlock;
pub use factor_variants::{ValuePBBlock, ValuePEBlock, QualityROEBlock, QualityMultiBlock, DividendGrowthBlock, BusinessCycleDefBlock};
pub use portfolio_opt::{MaxSharpeBlock, MinVarianceBlock};
pub use advanced::{LiquidityFilterBlock, RegimeFilterBlock};
