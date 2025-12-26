//! Entry blocks - signal generation (Techniques 8-12).

mod ma_crossover;
mod bollinger;
mod rsi;
mod macd;
mod zscore;

pub use ma_crossover::MaCrossoverBlock;
pub use bollinger::BollingerBlock;
pub use rsi::RsiBlock;
pub use macd::MacdBlock;
pub use zscore::ZScoreBlock;

use crate::blocks::StrategyBlock;

/// Create entry block from block_id.
pub fn create_entry_block(block_id: &str) -> Option<Box<dyn StrategyBlock>> {
    match block_id {
        "ma_crossover" => Some(Box::new(MaCrossoverBlock::new())),
        "bollinger" => Some(Box::new(BollingerBlock::new())),
        "rsi" => Some(Box::new(RsiBlock::new())),
        "macd" => Some(Box::new(MacdBlock::new())),
        "zscore" => Some(Box::new(ZScoreBlock::new())),
        _ => None,
    }
}

