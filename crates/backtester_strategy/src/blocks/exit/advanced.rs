//! Advanced Exit Blocks - Prop Trading Level
//!
//! High-quality exit strategies based on:
//! - ATR-based trailing (Chandelier)
//! - Volatility regime detection
//! - Drawdown limits
//! - Trend reversal detection
//! - Time-based hybrid exits

use crate::blocks::{
    get_f64, get_i64, BlockParams, BlockResult, BlockType, Signal, StrategyBlock, ValidationError,
};
use crate::context::StrategyContext;
use rust_decimal::prelude::ToPrimitive;
use std::collections::HashMap;

// =============================================================================
// CHANDELIER ATR EXIT
// Reference: Chuck LeBeau's Chandelier Exit
// =============================================================================

/// ATR-based trailing stop that trails from the highest high.
/// Formula: Stop = Highest_High(n) - multiplier * ATR(n)
pub struct ChandelierAtrBlock;

impl ChandelierAtrBlock {
    pub fn new() -> Self { Self }
}

impl Default for ChandelierAtrBlock {
    fn default() -> Self { Self::new() }
}

impl StrategyBlock for ChandelierAtrBlock {
    fn block_id(&self) -> &'static str { "chandelier_atr" }
    fn block_type(&self) -> BlockType { BlockType::Exit }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let period = get_i64(params, "period", 22) as usize;
        let multiplier = get_f64(params, "multiplier", 3.0);

        let mut signals = Vec::new();

        for position in &ctx.positions {
            if let Some(candidate) = ctx.candidates.iter().find(|c| c.symbol == position.symbol) {
                let prices = &candidate.prices;
                
                if prices.len() < period {
                    continue;
                }

                // Calculate ATR proxy using close-to-close (simplified)
                let returns: Vec<f64> = prices.windows(2)
                    .map(|w| (w[1] - w[0]).abs())
                    .collect();
                let atr = if returns.len() >= period {
                    returns.iter().rev().take(period).sum::<f64>() / period as f64
                } else {
                    continue;
                };

                // Highest close in period (proxy for highest high)
                let highest = prices.iter().rev().take(period)
                    .copied().fold(f64::NEG_INFINITY, f64::max);

                // Chandelier stop level
                let stop_level = highest - (multiplier * atr);
                let current_price = *prices.last().unwrap();

                if current_price < stop_level {
                    signals.push(Signal::exit(&position.symbol, 1.0)
                        .with_source("chandelier_atr")
                        .with_metadata("stop_level", stop_level)
                        .with_metadata("atr", atr));
                }
            }
        }

        BlockResult::success(format!("Chandelier ATR: {} exits", signals.len()))
            .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let period = get_i64(params, "period", 22);
        let mult = get_f64(params, "multiplier", 3.0);
        if period < 5 || period > 100 { 
            return Err(ValidationError::OutOfRange("period".into(), "5-100".into())); 
        }
        if mult < 1.0 || mult > 6.0 { 
            return Err(ValidationError::OutOfRange("multiplier".into(), "1-6".into())); 
        }
        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut p = HashMap::new();
        p.insert("period".into(), toml::Value::Integer(22));
        p.insert("multiplier".into(), toml::Value::Float(3.0));
        p
    }

    fn description(&self) -> &'static str {
        "Chandelier ATR: Trailing stop based on ATR from highest high"
    }
}

// =============================================================================
// VOLATILITY SPIKE EXIT
// Exit when volatility exceeds historical threshold
// =============================================================================

/// Exit when current volatility spikes above historical norm.
/// Useful for risk-off during regime changes.
pub struct VolatilitySpikeBlock;

impl VolatilitySpikeBlock {
    pub fn new() -> Self { Self }
}

impl Default for VolatilitySpikeBlock {
    fn default() -> Self { Self::new() }
}

impl StrategyBlock for VolatilitySpikeBlock {
    fn block_id(&self) -> &'static str { "volatility_spike" }
    fn block_type(&self) -> BlockType { BlockType::Exit }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let lookback = get_i64(params, "lookback", 20) as usize;
        let spike_threshold = get_f64(params, "spike_threshold", 2.0); // std devs
        let exit_pct = get_f64(params, "exit_pct", 1.0); // 1.0 = full exit

        let mut signals = Vec::new();

        for position in &ctx.positions {
            if let Some(candidate) = ctx.candidates.iter().find(|c| c.symbol == position.symbol) {
                let prices = &candidate.prices;
                if prices.len() < lookback + 20 {
                    continue;
                }

                // Calculate returns
                let returns: Vec<f64> = prices.windows(2)
                    .map(|w| (w[1] / w[0]) - 1.0)
                    .collect();
                
                if returns.len() < lookback + 20 {
                    continue;
                }

                // Current volatility (last 5 days)
                let recent_returns = &returns[returns.len()-5..];
                let current_vol = (recent_returns.iter().map(|r| r.powi(2)).sum::<f64>() / 5.0).sqrt();

                // Historical volatility and std
                let hist_returns = &returns[returns.len()-lookback-20..returns.len()-5];
                let hist_vol = (hist_returns.iter().map(|r| r.powi(2)).sum::<f64>() / hist_returns.len() as f64).sqrt();
                
                let vol_mean = hist_vol;
                let vol_std = (hist_returns.iter()
                    .map(|r| (r.powi(2).sqrt() - vol_mean).powi(2))
                    .sum::<f64>() / hist_returns.len() as f64).sqrt().max(0.001);

                let z_score = (current_vol - vol_mean) / vol_std;

                if z_score > spike_threshold {
                    signals.push(Signal::exit(&position.symbol, exit_pct)
                        .with_source("volatility_spike")
                        .with_metadata("z_score", z_score)
                        .with_metadata("current_vol", current_vol));
                }
            }
        }

        BlockResult::success(format!("Vol Spike: {} exits", signals.len()))
            .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let lookback = get_i64(params, "lookback", 20);
        let threshold = get_f64(params, "spike_threshold", 2.0);
        if lookback < 10 || lookback > 100 { 
            return Err(ValidationError::OutOfRange("lookback".into(), "10-100".into())); 
        }
        if threshold < 1.0 || threshold > 5.0 { 
            return Err(ValidationError::OutOfRange("spike_threshold".into(), "1-5".into())); 
        }
        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut p = HashMap::new();
        p.insert("lookback".into(), toml::Value::Integer(20));
        p.insert("spike_threshold".into(), toml::Value::Float(2.0));
        p.insert("exit_pct".into(), toml::Value::Float(1.0));
        p
    }

    fn description(&self) -> &'static str {
        "Volatility Spike: Exit when vol exceeds historical norm"
    }
}

// =============================================================================
// TIME-ATR HYBRID EXIT
// Time stop with dynamic ATR-based trailing
// =============================================================================

/// Hybrid exit: time-based + ATR trailing.
/// Good for weekly radar strategies.
pub struct TimeAtrHybridBlock;

impl TimeAtrHybridBlock {
    pub fn new() -> Self { Self }
}

impl Default for TimeAtrHybridBlock {
    fn default() -> Self { Self::new() }
}

impl StrategyBlock for TimeAtrHybridBlock {
    fn block_id(&self) -> &'static str { "time_atr_hybrid" }
    fn block_type(&self) -> BlockType { BlockType::Exit }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let max_days = get_i64(params, "max_days", 5) as i64; // 5 days = weekly
        let atr_mult = get_f64(params, "atr_multiplier", 2.0);
        let atr_period = get_i64(params, "atr_period", 14) as usize;

        let mut signals = Vec::new();

        for position in &ctx.positions {
            let days_held = (ctx.date - position.entry_date).num_days();
            
            // Time-based exit
            if days_held >= max_days {
                signals.push(Signal::exit(&position.symbol, 1.0)
                    .with_source("time_atr_hybrid")
                    .with_metadata("days_held", days_held as f64));
                continue;
            }

            // ATR-based trailing
            if let Some(candidate) = ctx.candidates.iter().find(|c| c.symbol == position.symbol) {
                let prices = &candidate.prices;
                
                if prices.len() >= atr_period + 1 {
                    // ATR proxy using close-to-close
                    let returns: Vec<f64> = prices.windows(2)
                        .map(|w| (w[1] - w[0]).abs())
                        .collect();
                    let atr = returns.iter().rev().take(atr_period).sum::<f64>() / atr_period as f64;

                    let entry_price: f64 = position.cost_basis.to_f64();
                    let stop_level = entry_price - (atr_mult * atr);
                    let current = *prices.last().unwrap();

                    if current < stop_level {
                        signals.push(Signal::exit(&position.symbol, 1.0)
                            .with_source("time_atr_hybrid")
                            .with_metadata("atr", atr));
                    }
                }
            }
        }

        BlockResult::success(format!("Time-ATR Hybrid: {} exits", signals.len()))
            .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let max_days = get_i64(params, "max_days", 5);
        if max_days < 1 || max_days > 60 { 
            return Err(ValidationError::OutOfRange("max_days".into(), "1-60".into())); 
        }
        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut p = HashMap::new();
        p.insert("max_days".into(), toml::Value::Integer(5));
        p.insert("atr_multiplier".into(), toml::Value::Float(2.0));
        p.insert("atr_period".into(), toml::Value::Integer(14));
        p
    }

    fn description(&self) -> &'static str {
        "Time-ATR Hybrid: Time-based exit with ATR trailing stop"
    }
}

// =============================================================================
// DRAWDOWN LIMIT EXIT
// Exit when position or portfolio drawdown exceeds threshold
// =============================================================================

/// Exit when drawdown from peak exceeds limit.
pub struct DrawdownLimitBlock;

impl DrawdownLimitBlock {
    pub fn new() -> Self { Self }
}

impl Default for DrawdownLimitBlock {
    fn default() -> Self { Self::new() }
}

impl StrategyBlock for DrawdownLimitBlock {
    fn block_id(&self) -> &'static str { "drawdown_limit" }
    fn block_type(&self) -> BlockType { BlockType::Exit }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let max_dd_pct = get_f64(params, "max_drawdown_pct", 0.15);
        let use_portfolio = get_f64(params, "use_portfolio", 0.0) > 0.5; // 0=position, 1=portfolio

        let mut signals = Vec::new();

        if use_portfolio {
            // Portfolio-level drawdown
            let peak = ctx.peak_equity.to_f64().unwrap_or(0.0);
            let current = ctx.equity.to_f64().unwrap_or(0.0);
            
            if peak > 0.0 {
                let dd = (peak - current) / peak;
                if dd >= max_dd_pct {
                    // Exit ALL positions
                    for position in &ctx.positions {
                        signals.push(Signal::exit(&position.symbol, 1.0)
                            .with_source("drawdown_limit")
                            .with_metadata("portfolio_dd", dd));
                    }
                }
            }
        } else {
            // Position-level drawdown
            for position in &ctx.positions {
                if let Some(candidate) = ctx.candidates.iter().find(|c| c.symbol == position.symbol) {
                    let prices = &candidate.prices;
                    if prices.len() < 5 { continue; }
                    
                    // Find peak since entry
                    let entry_idx = prices.len().saturating_sub(
                        (ctx.date - position.entry_date).num_days().max(1) as usize
                    );
                    let peak = prices[entry_idx..].iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let current = *prices.last().unwrap();
                    
                    if peak > 0.0 {
                        let dd = (peak - current) / peak;
                        if dd >= max_dd_pct {
                            signals.push(Signal::exit(&position.symbol, 1.0)
                                .with_source("drawdown_limit")
                                .with_metadata("position_dd", dd));
                        }
                    }
                }
            }
        }

        BlockResult::success(format!("Drawdown Limit: {} exits", signals.len()))
            .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let dd = get_f64(params, "max_drawdown_pct", 0.15);
        if dd < 0.02 || dd > 0.50 { 
            return Err(ValidationError::OutOfRange("max_drawdown_pct".into(), "0.02-0.50".into())); 
        }
        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut p = HashMap::new();
        p.insert("max_drawdown_pct".into(), toml::Value::Float(0.15));
        p.insert("use_portfolio".into(), toml::Value::Float(0.0));
        p
    }

    fn description(&self) -> &'static str {
        "Drawdown Limit: Exit when drawdown exceeds threshold"
    }
}

// =============================================================================
// TREND REVERSAL EXIT
// Exit when trend reverses (MA slope or price breaks)
// =============================================================================

/// Exit when trend reverses based on MA slope or price breakdowns.
pub struct TrendReversalBlock;

impl TrendReversalBlock {
    pub fn new() -> Self { Self }
}

impl Default for TrendReversalBlock {
    fn default() -> Self { Self::new() }
}

impl StrategyBlock for TrendReversalBlock {
    fn block_id(&self) -> &'static str { "trend_reversal" }
    fn block_type(&self) -> BlockType { BlockType::Exit }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let ma_period = get_i64(params, "ma_period", 20) as usize;
        let slope_threshold = get_f64(params, "slope_threshold", 0.0); // negative = bearish

        let mut signals = Vec::new();

        for position in &ctx.positions {
            if let Some(candidate) = ctx.candidates.iter().find(|c| c.symbol == position.symbol) {
                let prices = &candidate.prices;
                if prices.len() < ma_period + 5 { continue; }

                // Current MA
                let ma_current: f64 = prices.iter().rev().take(ma_period).sum::<f64>() / ma_period as f64;
                
                // MA 5 days ago
                let ma_prev: f64 = prices.iter().rev().skip(5).take(ma_period).sum::<f64>() / ma_period as f64;
                
                // Slope (normalized)
                let slope = (ma_current - ma_prev) / ma_prev;

                // Exit if slope turns negative (for longs)
                let is_long = position.shares > 0;
                
                if is_long && slope < slope_threshold {
                    signals.push(Signal::exit(&position.symbol, 1.0)
                        .with_source("trend_reversal")
                        .with_metadata("ma_slope", slope));
                } else if !is_long && slope > -slope_threshold {
                    signals.push(Signal::exit(&position.symbol, 1.0)
                        .with_source("trend_reversal")
                        .with_metadata("ma_slope", slope));
                }
            }
        }

        BlockResult::success(format!("Trend Reversal: {} exits", signals.len()))
            .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let period = get_i64(params, "ma_period", 20);
        if period < 5 || period > 100 { 
            return Err(ValidationError::OutOfRange("ma_period".into(), "5-100".into())); 
        }
        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut p = HashMap::new();
        p.insert("ma_period".into(), toml::Value::Integer(20));
        p.insert("slope_threshold".into(), toml::Value::Float(0.0));
        p
    }

    fn description(&self) -> &'static str {
        "Trend Reversal: Exit when MA slope turns against position"
    }
}
