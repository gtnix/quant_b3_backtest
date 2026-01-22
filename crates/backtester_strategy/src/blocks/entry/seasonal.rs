//! Seasonal and Calendar Effect Blocks
//!
//! References: Thaler (1987), Bouman & Jacobsen (2002)

use crate::blocks::{BlockParams, BlockResult, BlockType, Signal, SignalDirection, StrategyBlock, ValidationError};
use crate::context::StrategyContext;
use crate::blocks::get_f64;

/// January Effect block.
pub struct JanuaryEffectBlock;
impl JanuaryEffectBlock { pub fn new() -> Self { Self } }

impl StrategyBlock for JanuaryEffectBlock {
    fn block_id(&self) -> &'static str { "january_effect" }
    fn block_type(&self) -> BlockType { BlockType::Entry }

    fn execute(&self, ctx: &mut StrategyContext, _params: &BlockParams) -> BlockResult {
        let mut signals = Vec::new();
        
        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            if prices.len() < 21 { continue; }
            
            let current = *prices.last().unwrap();
            let prev = prices[prices.len() - 21];
            let momentum = (current - prev) / prev;
            
            let (direction, strength) = if momentum < -0.05 {
                (SignalDirection::Long, 0.7)
            } else if momentum > 0.1 {
                (SignalDirection::Short, 0.5)
            } else {
                (SignalDirection::Flat, 0.0)
            };
            
            if direction != SignalDirection::Flat {
                signals.push(Signal::new(&candidate.symbol, direction, strength)
                    .with_source(self.block_id()));
            }
        }
        
        BlockResult::success(format!("january_effect: {} signals", signals.len()))
            .with_signals(signals)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "January effect calendar anomaly" }
}

/// Sell in May block.
pub struct SellInMayBlock;
impl SellInMayBlock { pub fn new() -> Self { Self } }

impl StrategyBlock for SellInMayBlock {
    fn block_id(&self) -> &'static str { "sell_in_may" }
    fn block_type(&self) -> BlockType { BlockType::Entry }

    fn execute(&self, ctx: &mut StrategyContext, _params: &BlockParams) -> BlockResult {
        let mut signals = Vec::new();
        
        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            if prices.len() < 126 { continue; }
            
            let current = *prices.last().unwrap();
            let prev = prices[prices.len() - 126];
            let momentum = (current - prev) / prev;
            
            let (direction, strength) = if momentum < 0.0 {
                (SignalDirection::Short, 0.5)
            } else {
                (SignalDirection::Long, 0.5)
            };
            
            signals.push(Signal::new(&candidate.symbol, direction, strength)
                .with_source(self.block_id()));
        }
        
        BlockResult::success(format!("sell_in_may: {} signals", signals.len()))
            .with_signals(signals)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "Sell in May seasonal effect" }
}

/// Grains Seasonal block.
pub struct GrainsSeasonalBlock;
impl GrainsSeasonalBlock { pub fn new() -> Self { Self } }

impl StrategyBlock for GrainsSeasonalBlock {
    fn block_id(&self) -> &'static str { "grains_seasonal" }
    fn block_type(&self) -> BlockType { BlockType::Entry }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let lookback = get_f64(params, "lookback", 63.0) as usize;
        let mut signals = Vec::new();
        
        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            if prices.len() < lookback { continue; }
            
            let current = *prices.last().unwrap();
            let prev = prices[prices.len() - lookback];
            let momentum = (current - prev) / prev;
            
            let (direction, strength) = if momentum > 0.05 {
                (SignalDirection::Long, 0.7)
            } else if momentum < -0.05 {
                (SignalDirection::Short, 0.7)
            } else {
                (SignalDirection::Flat, 0.0)
            };
            
            if direction != SignalDirection::Flat {
                signals.push(Signal::new(&candidate.symbol, direction, strength)
                    .with_source(self.block_id()));
            }
        }
        
        BlockResult::success(format!("grains_seasonal: {} signals", signals.len()))
            .with_signals(signals)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "Grains seasonal patterns" }
}

/// Natural Gas Seasonal block.
pub struct NatgasSeasonalBlock;
impl NatgasSeasonalBlock { pub fn new() -> Self { Self } }

impl StrategyBlock for NatgasSeasonalBlock {
    fn block_id(&self) -> &'static str { "natgas_seasonal" }
    fn block_type(&self) -> BlockType { BlockType::Entry }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let lookback = get_f64(params, "lookback", 42.0) as usize;
        let mut signals = Vec::new();
        
        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            if prices.len() < lookback { continue; }
            
            let current = *prices.last().unwrap();
            let prev = prices[prices.len() - lookback];
            let momentum = (current - prev) / prev;
            
            let (direction, strength) = if momentum > 0.1 {
                (SignalDirection::Long, 0.7)
            } else if momentum < -0.1 {
                (SignalDirection::Short, 0.7)
            } else {
                (SignalDirection::Flat, 0.0)
            };
            
            if direction != SignalDirection::Flat {
                signals.push(Signal::new(&candidate.symbol, direction, strength)
                    .with_source(self.block_id()));
            }
        }
        
        BlockResult::success(format!("natgas_seasonal: {} signals", signals.len()))
            .with_signals(signals)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "Natural gas seasonal patterns" }
}
