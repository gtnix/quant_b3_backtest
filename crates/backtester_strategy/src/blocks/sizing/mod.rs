//! Sizing blocks - position weighting.

mod equal_weight;
mod risk_parity;
mod vol_targeting;

pub use equal_weight::EqualWeightBlock;
pub use risk_parity::RiskParityBlock;
pub use vol_targeting::VolTargetingBlock;

use crate::blocks::StrategyBlock;

/// Create sizing block from block_id.
pub fn create_sizing_block(block_id: &str) -> Option<Box<dyn StrategyBlock>> {
    match block_id {
        "equal_weight" => Some(Box::new(EqualWeightBlock::new())),
        "risk_parity" => Some(Box::new(RiskParityBlock::new())),
        "vol_targeting" => Some(Box::new(VolTargetingBlock::new())),
        _ => None,
    }
}

