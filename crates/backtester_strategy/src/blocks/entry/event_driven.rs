//! Event-Driven Trading Blocks
//!
//! Reference: Bernard & Thomas (1990) Post-Earnings Announcement Drift

use crate::blocks::{BlockParams, BlockResult, BlockType, Signal, SignalDirection, StrategyBlock, ValidationError};
use crate::context::StrategyContext;
use crate::blocks::get_f64;

/// Pre-Earnings block.
pub struct PreEarningsBlock;
impl PreEarningsBlock { pub fn new() -> Self { Self } }

impl StrategyBlock for PreEarningsBlock {
    fn block_id(&self) -> &'static str { "pre_earnings" }
    fn block_type(&self) -> BlockType { BlockType::Entry }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let lookback = get_f64(params, "lookback", 20.0) as usize;
        let mut signals = Vec::new();
        
        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            if prices.len() < lookback + 5 { continue; }
            
            // Use volatility compression as proxy
            let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();
            if returns.len() < lookback { continue; }
            
            let recent_vol: f64 = returns.iter().rev().take(5).map(|r| r.powi(2)).sum::<f64>().sqrt();
            let avg_vol: f64 = returns.iter().rev().take(lookback).map(|r| r.powi(2)).sum::<f64>().sqrt() / (lookback as f64 / 5.0);
            
            if recent_vol < avg_vol * 0.6 {
                let current = *prices.last().unwrap();
                let ma: f64 = prices.iter().rev().take(lookback).sum::<f64>() / lookback as f64;
                if current > ma {
                    signals.push(Signal::long(&candidate.symbol, 0.7).with_source(self.block_id()));
                }
            }
        }
        
        BlockResult::success(format!("pre_earnings: {} signals", signals.len()))
            .with_signals(signals)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "Pre-earnings positioning" }
}

/// Post-Earnings block (PEAD).
pub struct PostEarningsBlock;
impl PostEarningsBlock { pub fn new() -> Self { Self } }

impl StrategyBlock for PostEarningsBlock {
    fn block_id(&self) -> &'static str { "post_earnings" }
    fn block_type(&self) -> BlockType { BlockType::Entry }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let threshold = get_f64(params, "threshold", 0.03);
        let mut signals = Vec::new();
        
        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            if prices.len() < 2 { continue; }
            
            let current = *prices.last().unwrap();
            let prev = prices[prices.len() - 2];
            let daily_return = (current - prev) / prev;
            
            let (direction, strength) = if daily_return > threshold {
                (SignalDirection::Long, 0.8)
            } else if daily_return < -threshold {
                (SignalDirection::Short, 0.8)
            } else {
                (SignalDirection::Flat, 0.0)
            };
            
            if direction != SignalDirection::Flat {
                signals.push(Signal::new(&candidate.symbol, direction, strength)
                    .with_source(self.block_id()));
            }
        }
        
        BlockResult::success(format!("post_earnings: {} signals", signals.len()))
            .with_signals(signals)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "Post-earnings announcement drift" }
}

/// News Volatility block.
pub struct NewsVolatilityBlock;
impl NewsVolatilityBlock { pub fn new() -> Self { Self } }

impl StrategyBlock for NewsVolatilityBlock {
    fn block_id(&self) -> &'static str { "news_volatility" }
    fn block_type(&self) -> BlockType { BlockType::Entry }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let lookback = get_f64(params, "lookback", 20.0) as usize;
        let vol_threshold = get_f64(params, "vol_threshold", 2.0);
        let mut signals = Vec::new();
        
        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            if prices.len() < lookback + 5 { continue; }
            
            let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();
            if returns.len() < lookback { continue; }
            
            let recent_vol: f64 = returns.iter().rev().take(5).map(|r| r.powi(2)).sum::<f64>() / 5.0;
            let avg_vol: f64 = returns.iter().rev().take(lookback).map(|r| r.powi(2)).sum::<f64>() / lookback as f64;
            
            if recent_vol > avg_vol * vol_threshold {
                let current = *prices.last().unwrap();
                let prev = prices[prices.len() - 6];
                let direction = if current > prev { SignalDirection::Long } else { SignalDirection::Short };
                signals.push(Signal::new(&candidate.symbol, direction, 0.7).with_source(self.block_id()));
            }
        }
        
        BlockResult::success(format!("news_volatility: {} signals", signals.len()))
            .with_signals(signals)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "News-driven volatility trading" }
}
