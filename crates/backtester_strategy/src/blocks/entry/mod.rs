//! Entry blocks - signal generation (Techniques 8-27).
//!
//! This module contains all entry/exit signal generation blocks derived from
//! the 116 strategies in the Strategy Catalog. Each block is documented with
//! its mathematical foundation and academic references.

// Core technical blocks (Techniques 8-12)
mod ma_crossover;
mod bollinger;
mod rsi;
mod macd;
mod zscore;

// Breakout blocks (Techniques 13-18)
mod donchian;
mod orb_breakout;
mod channel_breakout;
mod atr_breakout;
mod vol_expansion;
mod volume_breakout;

// Mean Reversion blocks (Techniques 19-22)
mod vwap;
mod gap;
mod vix_reversion;

// Momentum blocks (Techniques 23-25)
mod adx;
mod adaptive_momentum;

// Pairs/Statistical Arbitrage blocks
mod pairs_arb;

// Seasonal/Calendar blocks
mod seasonal;

// Event-driven blocks
mod event_driven;

// Trend/Momentum variations
mod trend_momentum;

// Filtered blocks
mod filtered;

// Re-exports - Core blocks
pub use ma_crossover::MaCrossoverBlock;
pub use bollinger::BollingerBlock;
pub use rsi::RsiBlock;
pub use macd::MacdBlock;
pub use zscore::ZScoreBlock;

// Re-exports - Breakout blocks
pub use donchian::DonchianChannelBlock;
pub use orb_breakout::ORBBreakoutBlock;
pub use channel_breakout::ChannelBreakoutBlock;
pub use atr_breakout::ATRBreakoutBlock;
pub use vol_expansion::VolatilityExpansionBlock;
pub use volume_breakout::{VolumeBreakoutBlock, VolumeProfileBlock};

// Re-exports - Mean Reversion blocks
pub use vwap::VWAPBlock;
pub use gap::{GapFillBlock, GapContinuationBlock};
pub use vix_reversion::VIXReversionBlock;

// Re-exports - Momentum blocks
pub use adx::ADXMomentumBlock;
pub use adaptive_momentum::AdaptiveMomentumBlock;

// Re-exports - Pairs/Arb blocks
pub use pairs_arb::{CointegrationBlock, DistanceBlock, MultiPairBlock, MAArbitrageBlock};

// Re-exports - Seasonal blocks
pub use seasonal::{JanuaryEffectBlock, SellInMayBlock, GrainsSeasonalBlock, NatgasSeasonalBlock};

// Re-exports - Event-driven blocks
pub use event_driven::{PreEarningsBlock, PostEarningsBlock, NewsVolatilityBlock};

// Re-exports - Trend/Momentum variations
pub use trend_momentum::{DualMABlock, TrendMABlock, TimeSeriesMomentumBlock, CrossSectionalBlock, BuyHoldBlock};

// Re-exports - Filtered blocks
pub use filtered::{RSIFilteredBlock, BollingerFilteredBlock};

use crate::blocks::StrategyBlock;

/// Create entry block from block_id.
/// Supports all blocks from the Strategy Catalog (116 strategies).
pub fn create_entry_block(block_id: &str) -> Option<Box<dyn StrategyBlock>> {
    match block_id {
        // Core technical
        "ma_crossover" => Some(Box::new(MaCrossoverBlock::new())),
        "bollinger" | "bb_reversion" | "bb_swing" => Some(Box::new(BollingerBlock::new())),
        "rsi" | "rsi_reversion" | "rsi_swing" => Some(Box::new(RsiBlock::new())),
        "rsi_filtered" => Some(Box::new(RSIFilteredBlock::new())),
        "macd" | "macd_momentum" | "macd_swing" => Some(Box::new(MacdBlock::new())),
        "zscore" => Some(Box::new(ZScoreBlock::new())),
        "bb_filtered" => Some(Box::new(BollingerFilteredBlock::new())),
        
        // Breakout
        "donchian" | "donchian_dual" => Some(Box::new(DonchianChannelBlock::new())),
        "orb_breakout" => Some(Box::new(ORBBreakoutBlock::new())),
        "channel_breakout" => Some(Box::new(ChannelBreakoutBlock::new())),
        "atr_breakout" => Some(Box::new(ATRBreakoutBlock::new())),
        "vol_expansion" | "bb_squeeze" => Some(Box::new(VolatilityExpansionBlock::new())),
        "volume_breakout" => Some(Box::new(VolumeBreakoutBlock::new())),
        "volume_poc" | "volume_profile" => Some(Box::new(VolumeProfileBlock::new())),
        
        // Mean Reversion
        "vwap" | "vwap_reversion" | "vwap_trend" => Some(Box::new(VWAPBlock::new())),
        "gap_fill" => Some(Box::new(GapFillBlock::new())),
        "gap_continuation" => Some(Box::new(GapContinuationBlock::new())),
        "vix_reversion" => Some(Box::new(VIXReversionBlock::new())),
        
        // Momentum
        "adx_momentum" | "trend_adx" => Some(Box::new(ADXMomentumBlock::new())),
        "adaptive_momentum" => Some(Box::new(AdaptiveMomentumBlock::new())),
        
        // Pairs/Statistical Arbitrage
        "cointegration" => Some(Box::new(CointegrationBlock::new())),
        "cointegration_fast" => Some(Box::new(CointegrationBlock::fast())),
        "distance" => Some(Box::new(DistanceBlock::new())),
        "distance_fast" => Some(Box::new(DistanceBlock::fast())),
        "multi_pair" => Some(Box::new(MultiPairBlock::new())),
        "multi_pair_div" => Some(Box::new(MultiPairBlock::with_dividends())),
        "ma_arb" => Some(Box::new(MAArbitrageBlock::new())),
        
        // Seasonal/Calendar
        "january_effect" => Some(Box::new(JanuaryEffectBlock::new())),
        "sell_in_may" => Some(Box::new(SellInMayBlock::new())),
        "grains_seasonal" => Some(Box::new(GrainsSeasonalBlock::new())),
        "natgas_seasonal" => Some(Box::new(NatgasSeasonalBlock::new())),
        
        // Event-driven
        "pre_earnings" => Some(Box::new(PreEarningsBlock::new())),
        "post_earnings" => Some(Box::new(PostEarningsBlock::new())),
        "news_volatility" => Some(Box::new(NewsVolatilityBlock::new())),
        
        // Trend/Momentum variations
        "dual_ma" => Some(Box::new(DualMABlock::new())),
        "trend_ma" => Some(Box::new(TrendMABlock::new())),
        "time_series" => Some(Box::new(TimeSeriesMomentumBlock::new())),
        "cross_sectional" => Some(Box::new(CrossSectionalBlock::new())),
        "cross_sectional_multi" => Some(Box::new(CrossSectionalBlock::multi())),
        "buy_hold" => Some(Box::new(BuyHoldBlock::new())),
        
        _ => None,
    }
}

