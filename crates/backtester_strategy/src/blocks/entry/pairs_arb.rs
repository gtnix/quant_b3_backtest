//! Pairs Trading and Statistical Arbitrage Blocks
//!
//! Implements cointegration-based and distance-based pairs trading.
//! Reference: Gatev et al. (2006) "Pairs Trading: Performance of a Relative-Value Arbitrage Rule"

use crate::blocks::{BlockParams, BlockResult, BlockType, Signal, SignalDirection, StrategyBlock, ValidationError};
use crate::context::StrategyContext;
use crate::blocks::get_f64;

/// Cointegration-based pairs trading block.
pub struct CointegrationBlock {
    fast_mode: bool,
}

impl CointegrationBlock {
    pub fn new() -> Self { Self { fast_mode: false } }
    pub fn fast() -> Self { Self { fast_mode: true } }
}

impl StrategyBlock for CointegrationBlock {
    fn block_id(&self) -> &'static str {
        if self.fast_mode { "cointegration_fast" } else { "cointegration" }
    }
    fn block_type(&self) -> BlockType { BlockType::Entry }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let lookback = get_f64(params, "lookback", if self.fast_mode { 20.0 } else { 60.0 }) as usize;
        let entry_threshold = get_f64(params, "entry_threshold", 2.0);
        
        let mut signals = Vec::new();
        
        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            if prices.len() < lookback { continue; }
            
            let current = *prices.last().unwrap();
            let mean: f64 = prices.iter().rev().take(lookback).sum::<f64>() / lookback as f64;
            let variance: f64 = prices.iter().rev().take(lookback)
                .map(|p| (p - mean).powi(2)).sum::<f64>() / lookback as f64;
            let std = variance.sqrt().max(0.0001);
            let zscore = (current - mean) / std;
            
            let (direction, strength) = if zscore < -entry_threshold {
                (SignalDirection::Long, 0.8)
            } else if zscore > entry_threshold {
                (SignalDirection::Short, 0.8)
            } else {
                (SignalDirection::Flat, 0.0)
            };
            
            if direction != SignalDirection::Flat {
                signals.push(Signal::new(&candidate.symbol, direction, strength)
                    .with_source(self.block_id())
                    .with_metadata("zscore", zscore));
            }
        }
        
        BlockResult::success(format!("{}: {} signals", self.block_id(), signals.len()))
            .with_signals(signals)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "Cointegration-based pairs trading" }
}

/// Distance-based pairs trading block.
pub struct DistanceBlock { fast_mode: bool }

impl DistanceBlock {
    pub fn new() -> Self { Self { fast_mode: false } }
    pub fn fast() -> Self { Self { fast_mode: true } }
}

impl StrategyBlock for DistanceBlock {
    fn block_id(&self) -> &'static str {
        if self.fast_mode { "distance_fast" } else { "distance" }
    }
    fn block_type(&self) -> BlockType { BlockType::Entry }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let lookback = get_f64(params, "lookback", if self.fast_mode { 10.0 } else { 20.0 }) as usize;
        let threshold = get_f64(params, "threshold", 2.0);
        
        let mut signals = Vec::new();
        
        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            if prices.len() < lookback { continue; }
            
            let current = *prices.last().unwrap();
            let mean: f64 = prices.iter().rev().take(lookback).sum::<f64>() / lookback as f64;
            let distance = (current - mean) / mean * 100.0;
            
            let (direction, strength) = if distance < -threshold {
                (SignalDirection::Long, 0.7)
            } else if distance > threshold {
                (SignalDirection::Short, 0.7)
            } else {
                (SignalDirection::Flat, 0.0)
            };
            
            if direction != SignalDirection::Flat {
                signals.push(Signal::new(&candidate.symbol, direction, strength)
                    .with_source(self.block_id()));
            }
        }
        
        BlockResult::success(format!("{}: {} signals", self.block_id(), signals.len()))
            .with_signals(signals)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "Distance-based pairs trading" }
}

/// Multi-pair trading block.
pub struct MultiPairBlock { with_dividends: bool }

impl MultiPairBlock {
    pub fn new() -> Self { Self { with_dividends: false } }
    pub fn with_dividends() -> Self { Self { with_dividends: true } }
}

impl StrategyBlock for MultiPairBlock {
    fn block_id(&self) -> &'static str {
        if self.with_dividends { "multi_pair_div" } else { "multi_pair" }
    }
    fn block_type(&self) -> BlockType { BlockType::Entry }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let lookback = get_f64(params, "lookback", 20.0) as usize;
        let threshold = get_f64(params, "threshold", 1.5);
        
        let mut signals = Vec::new();
        
        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            if prices.len() < lookback { continue; }
            
            let current = *prices.last().unwrap();
            let mean: f64 = prices.iter().rev().take(lookback).sum::<f64>() / lookback as f64;
            let zscore = (current - mean) / mean.abs().max(0.01);
            
            let (direction, strength) = if zscore < -threshold {
                (SignalDirection::Long, 0.7)
            } else if zscore > threshold {
                (SignalDirection::Short, 0.7)
            } else {
                (SignalDirection::Flat, 0.0)
            };
            
            if direction != SignalDirection::Flat {
                signals.push(Signal::new(&candidate.symbol, direction, strength)
                    .with_source(self.block_id()));
            }
        }
        
        BlockResult::success(format!("{}: {} signals", self.block_id(), signals.len()))
            .with_signals(signals)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "Multi-pair trading strategy" }
}

/// MA Arbitrage block.
pub struct MAArbitrageBlock;
impl MAArbitrageBlock { pub fn new() -> Self { Self } }

impl StrategyBlock for MAArbitrageBlock {
    fn block_id(&self) -> &'static str { "ma_arb" }
    fn block_type(&self) -> BlockType { BlockType::Entry }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let fast_period = get_f64(params, "fast_period", 5.0) as usize;
        let slow_period = get_f64(params, "slow_period", 20.0) as usize;
        
        let mut signals = Vec::new();
        
        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            if prices.len() < slow_period { continue; }
            
            let fast_ma: f64 = prices.iter().rev().take(fast_period).sum::<f64>() / fast_period as f64;
            let slow_ma: f64 = prices.iter().rev().take(slow_period).sum::<f64>() / slow_period as f64;
            let spread = (fast_ma - slow_ma) / slow_ma * 100.0;
            
            let (direction, strength) = if spread > 1.0 {
                (SignalDirection::Short, 0.6)
            } else if spread < -1.0 {
                (SignalDirection::Long, 0.6)
            } else {
                (SignalDirection::Flat, 0.0)
            };
            
            if direction != SignalDirection::Flat {
                signals.push(Signal::new(&candidate.symbol, direction, strength)
                    .with_source(self.block_id()));
            }
        }
        
        BlockResult::success(format!("ma_arb: {} signals", signals.len()))
            .with_signals(signals)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "MA-based arbitrage" }
}
