//! Donchian Channel entry block - Turtle Trading System.
//!
//! # Mathematical Foundation
//! 
//! **Donchian Channel (Richard Donchian, 1960):**
//! - Upper Band: U_t = max(H_{t-n}, ..., H_{t-1}) where H = High price
//! - Lower Band: L_t = min(L_{t-n}, ..., L_{t-1}) where L = Low price
//! - Middle Band: M_t = (U_t + L_t) / 2
//!
//! **Turtle Trading Rules (1983):**
//! - System 1: Entry on 20-day breakout, exit on 10-day breakout
//! - System 2: Entry on 55-day breakout, exit on 20-day breakout
//!
//! # References
//! - Donchian, R. (1960). "Trend-Following Methods in Commodity Price Analysis"
//! - Faith, C. (2003). "Way of the Turtle: The Secret Methods that Turned Ordinary People into Legendary Traders"

use crate::blocks::{
    get_usize, BlockParams, BlockResult, BlockType, Signal, SignalDirection,
    StrategyBlock, ValidationError,
};
use crate::context::StrategyContext;
use std::collections::HashMap;

/// Donchian Channel breakout block - Turtle Trading System.
/// 
/// Generates long signals when price breaks above the n-period high,
/// and short signals when price breaks below the n-period low.
pub struct DonchianChannelBlock;

impl DonchianChannelBlock {
    pub fn new() -> Self {
        Self
    }

    /// Calculate Donchian Channel bands.
    /// 
    /// # Arguments
    /// * `highs` - High prices (oldest to newest)
    /// * `lows` - Low prices (oldest to newest)
    /// * `period` - Lookback period in bars
    /// 
    /// # Returns
    /// * (upper, lower, middle) bands or None if insufficient data
    /// 
    /// # Formula
    /// - Upper = max(high[t-period:t-1])
    /// - Lower = min(low[t-period:t-1])
    /// - Middle = (Upper + Lower) / 2
    pub fn calculate_channel(
        highs: &[f64],
        lows: &[f64],
        period: usize,
    ) -> Option<(f64, f64, f64)> {
        if highs.len() < period || lows.len() < period {
            return None;
        }

        // Use period bars before the current bar (excluding current)
        let n = highs.len();
        let start = n.saturating_sub(period + 1);
        let end = n - 1; // Exclude current bar
        
        if start >= end {
            return None;
        }

        let upper = highs[start..end]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let lower = lows[start..end]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let middle = (upper + lower) / 2.0;

        if upper.is_finite() && lower.is_finite() {
            Some((upper, lower, middle))
        } else {
            None
        }
    }
}

impl Default for DonchianChannelBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for DonchianChannelBlock {
    fn block_id(&self) -> &'static str {
        "donchian"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Entry
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let entry_period = get_usize(params, "period", 20);
        let exit_period = get_usize(params, "exit_period", 10);

        let mut signals = Vec::new();
        let mut long_breakouts = 0;
        let mut short_breakouts = 0;

        for candidate in &ctx.candidates {
            // Need highs and lows - use prices as proxy if not available
            // In production, StrategyCandidate should have highs/lows
            let prices = &candidate.prices;
            
            if prices.len() < entry_period + 1 {
                continue;
            }

            // Use close prices as proxy for high/low (simplified)
            // TODO: Add highs/lows to StrategyCandidate
            let highs = prices;
            let lows = prices;

            let current_price = *prices.last().unwrap();

            if let Some((upper, lower, middle)) = 
                Self::calculate_channel(highs, lows, entry_period) 
            {
                let (direction, strength) = if current_price > upper {
                    // Breakout above - Turtle long entry
                    let breakout_pct = (current_price - upper) / upper;
                    long_breakouts += 1;
                    (SignalDirection::Long, (0.7 + breakout_pct.min(0.3)).min(1.0))
                } else if current_price < lower {
                    // Breakout below - Turtle short entry
                    let breakout_pct = (lower - current_price) / lower;
                    short_breakouts += 1;
                    (SignalDirection::Short, (0.7 + breakout_pct.min(0.3)).min(1.0))
                } else {
                    // Inside channel - check exit signals
                    if let Some((exit_upper, exit_lower, _)) = 
                        Self::calculate_channel(highs, lows, exit_period) 
                    {
                        if current_price < exit_lower || current_price > exit_upper {
                            (SignalDirection::Exit, 0.5)
                        } else {
                            (SignalDirection::Flat, 0.0)
                        }
                    } else {
                        (SignalDirection::Flat, 0.0)
                    }
                };

                let signal = Signal::new(&candidate.symbol, direction, strength)
                    .with_source("donchian")
                    .with_metadata("price", current_price)
                    .with_metadata("upper", upper)
                    .with_metadata("lower", lower)
                    .with_metadata("middle", middle)
                    .with_metadata("period", entry_period as f64);

                signals.push(signal);
            }
        }

        ctx.trace_step(
            self.block_id(),
            &format!(
                "Donchian({}/{}): {} long, {} short breakouts from {} candidates",
                entry_period, exit_period, long_breakouts, short_breakouts, ctx.candidates.len()
            ),
        );

        BlockResult::success(format!(
            "Donchian: {} long, {} short breakouts",
            long_breakouts, short_breakouts
        ))
        .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let period = get_usize(params, "period", 20);
        let exit_period = get_usize(params, "exit_period", 10);

        if period < 5 {
            return Err(ValidationError::OutOfRange(
                "period".into(),
                "must be at least 5".into(),
            ));
        }

        if exit_period >= period {
            return Err(ValidationError::OutOfRange(
                "exit_period".into(),
                "must be less than entry period".into(),
            ));
        }

        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("period".into(), toml::Value::Integer(20));
        params.insert("exit_period".into(), toml::Value::Integer(10));
        params
    }

    fn description(&self) -> &'static str {
        "Donchian Channel: Turtle Trading breakout system (n-day high/low)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_donchian_calculation() {
        // Create a simple channel test
        let highs: Vec<f64> = vec![10.0, 11.0, 12.0, 11.5, 10.5, 13.0, 12.0];
        let lows: Vec<f64> = vec![9.0, 10.0, 11.0, 10.5, 9.5, 11.0, 10.0];
        
        let result = DonchianChannelBlock::calculate_channel(&highs, &lows, 5);
        assert!(result.is_some());
        
        let (upper, lower, middle) = result.unwrap();
        // Upper = max of bars 1-5 = 13.0
        // Lower = min of bars 1-5 = 9.5
        assert!((upper - 13.0).abs() < 0.01);
        assert!((lower - 9.5).abs() < 0.01);
        assert!((middle - 11.25).abs() < 0.01);
    }

    #[test]
    fn test_donchian_insufficient_data() {
        let highs: Vec<f64> = vec![10.0, 11.0];
        let lows: Vec<f64> = vec![9.0, 10.0];
        
        let result = DonchianChannelBlock::calculate_channel(&highs, &lows, 5);
        assert!(result.is_none());
    }
}
