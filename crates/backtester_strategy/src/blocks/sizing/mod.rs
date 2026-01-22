//! Sizing blocks - position weighting.

mod equal_weight;
mod risk_parity;
mod vol_targeting;
mod advanced;

pub use equal_weight::EqualWeightBlock;
pub use risk_parity::RiskParityBlock;
pub use vol_targeting::VolTargetingBlock;
pub use advanced::*;

use crate::blocks::StrategyBlock;

/// Create sizing block from block_id.
pub fn create_sizing_block(block_id: &str) -> Option<Box<dyn StrategyBlock>> {
    match block_id {
        // Basic sizing
        "equal_weight" => Some(Box::new(EqualWeightBlock::new())),
        "risk_parity" => Some(Box::new(RiskParityBlock::new())),
        "vol_targeting" => Some(Box::new(VolTargetingBlock::new())),
        // Advanced sizing (prop-trading level)
        "kelly_fractional" => Some(Box::new(KellyFractionalBlock::new())),
        "vol_target_buffer" => Some(Box::new(VolTargetBufferBlock::new())),
        "exposure_cap" => Some(Box::new(ExposureCapBlock::new())),
        _ => None,
    }
}

