//! MA Crossover entry block (Technique 8).
//! 
//! Generates long signal when fast MA crosses above slow MA.

use crate::blocks::{
    get_usize, BlockParams, BlockResult, BlockType, Signal, SignalDirection, StrategyBlock,
    ValidationError,
};
use crate::context::StrategyContext;
use std::collections::HashMap;

/// MA Crossover block - Technique 8.
pub struct MaCrossoverBlock;

impl MaCrossoverBlock {
    pub fn new() -> Self {
        Self
    }

    fn calculate_sma(prices: &[f64], period: usize) -> Option<f64> {
        if prices.len() < period {
            return None;
        }
        let sum: f64 = prices.iter().rev().take(period).sum();
        Some(sum / period as f64)
    }
}

impl Default for MaCrossoverBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for MaCrossoverBlock {
    fn block_id(&self) -> &'static str {
        "ma_crossover"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Entry
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let fast_period = get_usize(params, "fast_period", 50);
        let slow_period = get_usize(params, "slow_period", 200);

        let mut signals = Vec::new();
        let mut long_count = 0;
        let mut skipped_insufficient_data = 0;

        for candidate in &ctx.candidates {
            // Adaptive fallback: use available data if periods are too long
            // Minimum 20 days for slow MA to be meaningful
            let available = candidate.prices.len();
            let effective_slow = if available >= slow_period + 1 {
                slow_period
            } else if available >= 20 {
                // Fallback to available data minus 1 (for prev calculation)
                available.saturating_sub(1)
            } else {
                skipped_insufficient_data += 1;
                continue; // Skip if less than 20 days (unreliable)
            };
            
            // Maintain fast/slow ratio for fallback periods
            let effective_fast = if effective_slow < slow_period {
                // Scale fast period proportionally
                (((fast_period as f64 / slow_period as f64) * effective_slow as f64)
                    .round() as usize)
                    .max(5) // Minimum 5 days for fast MA
            } else {
                fast_period
            };

            // Calculate current and previous MAs
            let fast_ma = Self::calculate_sma(&candidate.prices, effective_fast);
            let slow_ma = Self::calculate_sma(&candidate.prices, effective_slow);

            // Calculate previous MAs (exclude last price)
            let prev_prices: Vec<f64> = candidate.prices[..candidate.prices.len() - 1].to_vec();
            let prev_fast_ma = Self::calculate_sma(&prev_prices, effective_fast);
            let prev_slow_ma = Self::calculate_sma(&prev_prices, effective_slow);

            if let (Some(fast), Some(slow), Some(prev_fast), Some(prev_slow)) =
                (fast_ma, slow_ma, prev_fast_ma, prev_slow_ma)
            {
                // Bullish crossover: fast crosses above slow
                let is_bullish_cross = prev_fast <= prev_slow && fast > slow;
                // Bearish crossover: fast crosses below slow
                let is_bearish_cross = prev_fast >= prev_slow && fast < slow;
                // Currently bullish (fast above slow)
                let is_bullish = fast > slow;

                let (direction, strength) = if is_bullish_cross {
                    (SignalDirection::Long, 1.0)
                } else if is_bearish_cross {
                    (SignalDirection::Exit, 0.8)
                } else if is_bullish {
                    // Already in trend, moderate signal
                    let strength = ((fast - slow) / slow).abs().min(0.1) * 10.0;
                    (SignalDirection::Long, 0.5 + strength * 0.3)
                } else {
                    (SignalDirection::Flat, 0.0)
                };

                if direction == SignalDirection::Long {
                    long_count += 1;
                }

                let signal = Signal::new(&candidate.symbol, direction, strength)
                    .with_source("ma_crossover")
                    .with_metadata("fast_ma", fast)
                    .with_metadata("slow_ma", slow);

                signals.push(signal);
            }
        }

        let msg = if skipped_insufficient_data > 0 {
            format!(
                "{} long signals from {} candidates ({} skipped: insufficient data)",
                long_count, ctx.candidates.len(), skipped_insufficient_data
            )
        } else {
            format!("{} long signals from {} candidates", long_count, ctx.candidates.len())
        };
        
        ctx.trace_step(self.block_id(), &msg);

        BlockResult::success(format!(
            "MA Crossover: {} long signals, {} candidates",
            long_count,
            ctx.candidates.len()
        ))
        .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let fast = get_usize(params, "fast_period", 50);
        let slow = get_usize(params, "slow_period", 200);

        if fast >= slow {
            return Err(ValidationError::OutOfRange(
                "fast_period".into(),
                "must be less than slow_period".into(),
            ));
        }

        if fast < 5 {
            return Err(ValidationError::OutOfRange(
                "fast_period".into(),
                "must be at least 5".into(),
            ));
        }

        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("fast_period".into(), toml::Value::Integer(50));
        params.insert("slow_period".into(), toml::Value::Integer(200));
        params
    }

    fn description(&self) -> &'static str {
        "MA Crossover: Long when fast MA crosses above slow MA"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma_calculation() {
        let prices = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let sma = MaCrossoverBlock::calculate_sma(&prices, 3);
        assert!((sma.unwrap() - 13.0).abs() < 0.01); // (12+13+14)/3 = 13
    }

    #[test]
    fn test_sma_insufficient_data() {
        let prices = vec![10.0, 11.0];
        let sma = MaCrossoverBlock::calculate_sma(&prices, 5);
        assert!(sma.is_none());
    }
}

