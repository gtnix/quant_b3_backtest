//! Trend and Momentum Variation Blocks
//!
//! References: Moskowitz et al. (2012), Jegadeesh & Titman (1993)

use crate::blocks::{BlockParams, BlockResult, BlockType, Signal, SignalDirection, StrategyBlock, ValidationError};
use crate::context::StrategyContext;
use crate::blocks::get_f64;

/// Dual MA block.
pub struct DualMABlock;
impl DualMABlock { pub fn new() -> Self { Self } }

impl StrategyBlock for DualMABlock {
    fn block_id(&self) -> &'static str { "dual_ma" }
    fn block_type(&self) -> BlockType { BlockType::Entry }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let fast_period = get_f64(params, "fast_period", 10.0) as usize;
        let slow_period = get_f64(params, "slow_period", 30.0) as usize;
        let mut signals = Vec::new();
        
        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            if prices.len() < slow_period { continue; }
            
            let fast_ma: f64 = prices.iter().rev().take(fast_period).sum::<f64>() / fast_period as f64;
            let slow_ma: f64 = prices.iter().rev().take(slow_period).sum::<f64>() / slow_period as f64;
            
            let (direction, strength) = if fast_ma > slow_ma * 1.001 {
                (SignalDirection::Long, 0.7)
            } else if fast_ma < slow_ma * 0.999 {
                (SignalDirection::Short, 0.7)
            } else {
                (SignalDirection::Flat, 0.0)
            };
            
            if direction != SignalDirection::Flat {
                signals.push(Signal::new(&candidate.symbol, direction, strength).with_source(self.block_id()));
            }
        }
        
        BlockResult::success(format!("dual_ma: {} signals", signals.len())).with_signals(signals)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "Dual moving average crossover" }
}

/// Trend MA block.
pub struct TrendMABlock;
impl TrendMABlock { pub fn new() -> Self { Self } }

impl StrategyBlock for TrendMABlock {
    fn block_id(&self) -> &'static str { "trend_ma" }
    fn block_type(&self) -> BlockType { BlockType::Entry }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let period = get_f64(params, "period", 50.0) as usize;
        let mut signals = Vec::new();
        
        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            if prices.len() < period { continue; }
            
            let current = *prices.last().unwrap();
            let ma: f64 = prices.iter().rev().take(period).sum::<f64>() / period as f64;
            
            let (direction, strength) = if current > ma * 1.02 {
                (SignalDirection::Long, 0.7)
            } else if current < ma * 0.98 {
                (SignalDirection::Short, 0.7)
            } else {
                (SignalDirection::Flat, 0.0)
            };
            
            if direction != SignalDirection::Flat {
                signals.push(Signal::new(&candidate.symbol, direction, strength).with_source(self.block_id()));
            }
        }
        
        BlockResult::success(format!("trend_ma: {} signals", signals.len())).with_signals(signals)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "Trend following with MA" }
}

/// Time Series Momentum block (Moskowitz et al. 2012).
pub struct TimeSeriesMomentumBlock;
impl TimeSeriesMomentumBlock { pub fn new() -> Self { Self } }

impl StrategyBlock for TimeSeriesMomentumBlock {
    fn block_id(&self) -> &'static str { "time_series" }
    fn block_type(&self) -> BlockType { BlockType::Entry }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let lookback = get_f64(params, "lookback", 252.0) as usize;
        let mut signals = Vec::new();
        
        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            if prices.len() < lookback { continue; }
            
            let current = *prices.last().unwrap();
            let prev = prices[prices.len() - lookback];
            let momentum = (current - prev) / prev;
            
            let direction = if momentum > 0.0 { SignalDirection::Long } else { SignalDirection::Short };
            signals.push(Signal::new(&candidate.symbol, direction, 0.7)
                .with_source(self.block_id())
                .with_metadata("momentum", momentum));
        }
        
        BlockResult::success(format!("time_series: {} signals", signals.len())).with_signals(signals)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "Time-series momentum (Moskowitz et al.)" }
}

/// Cross-Sectional Momentum block (Jegadeesh & Titman 1993).
pub struct CrossSectionalBlock { multi: bool }

impl CrossSectionalBlock {
    pub fn new() -> Self { Self { multi: false } }
    pub fn multi() -> Self { Self { multi: true } }
}

impl StrategyBlock for CrossSectionalBlock {
    fn block_id(&self) -> &'static str {
        if self.multi { "cross_sectional_multi" } else { "cross_sectional" }
    }
    fn block_type(&self) -> BlockType { BlockType::Entry }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let lookback = get_f64(params, "lookback", if self.multi { 63.0 } else { 126.0 }) as usize;
        let mut signals = Vec::new();
        
        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            if prices.len() < lookback { continue; }
            
            let current = *prices.last().unwrap();
            let prev = prices[prices.len() - lookback];
            let momentum = (current - prev) / prev;
            
            let (direction, strength) = if momentum > 0.1 {
                (SignalDirection::Long, 0.8)
            } else if momentum < -0.1 {
                (SignalDirection::Short, 0.8)
            } else {
                (SignalDirection::Flat, 0.0)
            };
            
            if direction != SignalDirection::Flat {
                signals.push(Signal::new(&candidate.symbol, direction, strength).with_source(self.block_id()));
            }
        }
        
        BlockResult::success(format!("{}: {} signals", self.block_id(), signals.len())).with_signals(signals)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "Cross-sectional momentum" }
}

/// Buy and Hold block - baseline.
pub struct BuyHoldBlock;
impl BuyHoldBlock { pub fn new() -> Self { Self } }

impl StrategyBlock for BuyHoldBlock {
    fn block_id(&self) -> &'static str { "buy_hold" }
    fn block_type(&self) -> BlockType { BlockType::Entry }

    fn execute(&self, ctx: &mut StrategyContext, _params: &BlockParams) -> BlockResult {
        let signals: Vec<Signal> = ctx.candidates.iter()
            .map(|c| Signal::long(&c.symbol, 1.0).with_source("buy_hold"))
            .collect();
        
        BlockResult::success(format!("buy_hold: {} signals", signals.len())).with_signals(signals)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "Buy and hold benchmark" }
}
