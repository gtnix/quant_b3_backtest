//! Price Channel Breakout entry block.
//!
//! # Mathematical Foundation
//!
//! **Price Channel (similar to Donchian but uses Close):**
//! - Upper Channel: max(Close, period)
//! - Lower Channel: min(Close, period)
//!
//! **Breakout Confirmation:**
//! - Long: Close > Upper Channel (new high)
//! - Short: Close < Lower Channel (new low)
//!
//! **Percent From Channel:**
//! - Distance from channel edges as percentage for strength calculation
//!
//! # References
//! - Keltner, C. (1960). "How to Make Money in Commodities"
//! - Covel, M. (2004). "Trend Following"

use crate::blocks::{
    get_f64, get_usize, BlockParams, BlockResult, BlockType, Signal, SignalDirection,
    StrategyBlock, ValidationError,
};
use crate::context::StrategyContext;
use std::collections::HashMap;

/// Price Channel Breakout entry block.
pub struct ChannelBreakoutBlock;

impl ChannelBreakoutBlock {
    pub fn new() -> Self {
        Self
    }

    /// Calculate price channel (highest/lowest close).
    pub fn calculate_channel(prices: &[f64], period: usize) -> Option<(f64, f64, f64)> {
        if prices.len() < period + 1 {
            return None;
        }

        let n = prices.len();
        let window = &prices[(n - period - 1)..(n - 1)]; // Exclude current

        let upper = window.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lower = window.iter().copied().fold(f64::INFINITY, f64::min);

        if upper.is_finite() && lower.is_finite() {
            let middle = (upper + lower) / 2.0;
            Some((upper, lower, middle))
        } else {
            None
        }
    }
}

impl Default for ChannelBreakoutBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for ChannelBreakoutBlock {
    fn block_id(&self) -> &'static str {
        "channel_breakout"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Entry
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let period = get_usize(params, "period", 20);
        let buffer_pct = get_f64(params, "buffer_pct", 0.001);

        let mut signals = Vec::new();
        let mut long_breakouts = 0;
        let mut short_breakouts = 0;

        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            
            if prices.len() < period + 2 {
                continue;
            }

            let current_price = *prices.last().unwrap();

            if let Some((upper, lower, middle)) = Self::calculate_channel(prices, period) {
                let upper_with_buffer = upper * (1.0 + buffer_pct);
                let lower_with_buffer = lower * (1.0 - buffer_pct);
                let channel_width = upper - lower;

                let (direction, strength) = if current_price > upper_with_buffer {
                    // Breakout above channel
                    long_breakouts += 1;
                    let pct_above = (current_price - upper) / channel_width;
                    (SignalDirection::Long, (0.6 + pct_above.min(0.4)).min(1.0))
                } else if current_price < lower_with_buffer {
                    // Breakout below channel
                    short_breakouts += 1;
                    let pct_below = (lower - current_price) / channel_width;
                    (SignalDirection::Short, (0.6 + pct_below.min(0.4)).min(1.0))
                } else {
                    (SignalDirection::Flat, 0.0)
                };

                let signal = Signal::new(&candidate.symbol, direction, strength)
                    .with_source("channel_breakout")
                    .with_metadata("price", current_price)
                    .with_metadata("upper", upper)
                    .with_metadata("lower", lower)
                    .with_metadata("middle", middle);

                signals.push(signal);
            }
        }

        ctx.trace_step(
            self.block_id(),
            &format!(
                "ChannelBreakout({}): {} long, {} short from {} candidates",
                period, long_breakouts, short_breakouts, ctx.candidates.len()
            ),
        );

        BlockResult::success(format!(
            "Channel Breakout: {} long, {} short",
            long_breakouts, short_breakouts
        ))
        .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let period = get_usize(params, "period", 20);

        if period < 5 || period > 252 {
            return Err(ValidationError::OutOfRange(
                "period".into(),
                "must be between 5 and 252".into(),
            ));
        }

        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("period".into(), toml::Value::Integer(20));
        params.insert("buffer_pct".into(), toml::Value::Float(0.001));
        params
    }

    fn description(&self) -> &'static str {
        "Channel Breakout: Trade new n-period highs/lows"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_calculation() {
        let prices = vec![100.0, 102.0, 98.0, 101.0, 99.0, 103.0, 100.0];
        
        let result = ChannelBreakoutBlock::calculate_channel(&prices, 5);
        assert!(result.is_some());
        
        let (upper, lower, middle) = result.unwrap();
        assert!((upper - 103.0).abs() < 0.01);
        assert!((lower - 98.0).abs() < 0.01);
        assert!((middle - 100.5).abs() < 0.01);
    }
}
