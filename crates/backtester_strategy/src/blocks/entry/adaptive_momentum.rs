//! Adaptive Momentum entry block.
//!
//! # Mathematical Foundation
//!
//! **Adaptive Momentum (Kaufman, 1995):**
//! The lookback period adapts based on market volatility/efficiency.
//!
//! **Efficiency Ratio (ER):**
//! ER = |Price_t - Price_{t-n}| / Σ|Price_i - Price_{i-1}|
//! ER ranges from 0 (choppy) to 1 (trending)
//!
//! **Adaptive Period:**
//! adaptive_period = min_period + (1 - ER) × (max_period - min_period)
//! - High ER (trending): shorter period for faster response
//! - Low ER (choppy): longer period for noise filtering
//!
//! **Momentum Signal:**
//! momentum = (Price_t / Price_{t-adaptive_period}) - 1
//!
//! # References
//! - Kaufman, P. (1995). "Smarter Trading"
//! - Kaufman, P. (2013). "Trading Systems and Methods" (5th Edition)

use crate::blocks::{
    get_f64, get_usize, BlockParams, BlockResult, BlockType, Signal, SignalDirection,
    StrategyBlock, ValidationError,
};
use crate::context::StrategyContext;
use std::collections::HashMap;

/// Adaptive Momentum entry block.
pub struct AdaptiveMomentumBlock;

impl AdaptiveMomentumBlock {
    pub fn new() -> Self {
        Self
    }

    /// Calculate Efficiency Ratio (Kaufman).
    /// ER = Direction / Volatility = |net change| / sum|daily changes|
    pub fn calculate_efficiency_ratio(prices: &[f64], period: usize) -> Option<f64> {
        if prices.len() < period + 1 {
            return None;
        }

        let n = prices.len();
        let direction = (prices[n - 1] - prices[n - period - 1]).abs();

        let mut volatility = 0.0;
        for i in (n - period)..n {
            volatility += (prices[i] - prices[i - 1]).abs();
        }

        if volatility > 0.0 {
            Some((direction / volatility).clamp(0.0, 1.0))
        } else {
            Some(0.0)
        }
    }

    /// Calculate adaptive period based on efficiency ratio.
    pub fn calculate_adaptive_period(er: f64, min_period: usize, max_period: usize) -> usize {
        let adaptive = min_period as f64 + (1.0 - er) * (max_period - min_period) as f64;
        (adaptive as usize).clamp(min_period, max_period)
    }

    /// Calculate momentum using adaptive period.
    pub fn calculate_adaptive_momentum(
        prices: &[f64],
        min_period: usize,
        max_period: usize,
    ) -> Option<(f64, f64, usize)> {
        let er_period = max_period.min(prices.len().saturating_sub(1));
        
        let er = Self::calculate_efficiency_ratio(prices, er_period)?;
        let adaptive_period = Self::calculate_adaptive_period(er, min_period, max_period);

        if prices.len() < adaptive_period + 1 {
            return None;
        }

        let n = prices.len();
        let momentum = (prices[n - 1] / prices[n - adaptive_period - 1]) - 1.0;

        Some((momentum, er, adaptive_period))
    }
}

impl Default for AdaptiveMomentumBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for AdaptiveMomentumBlock {
    fn block_id(&self) -> &'static str {
        "adaptive_momentum"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Entry
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let min_period = get_usize(params, "min_period", 10);
        let max_period = get_usize(params, "max_period", 50);
        let momentum_threshold = get_f64(params, "momentum_threshold", 0.05);

        let mut signals = Vec::new();
        let mut long_signals = 0;
        let mut short_signals = 0;

        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            
            if prices.len() < max_period + 5 {
                continue;
            }

            let current_price = *prices.last().unwrap();

            if let Some((momentum, er, adaptive_period)) = 
                Self::calculate_adaptive_momentum(prices, min_period, max_period) 
            {
                let (direction, strength) = if momentum > momentum_threshold {
                    // Positive momentum
                    long_signals += 1;
                    let strength = (0.5 + (momentum - momentum_threshold).min(0.5)).min(1.0);
                    // Boost strength for high efficiency (trending market)
                    let adjusted_strength = strength * (0.5 + er * 0.5);
                    (SignalDirection::Long, adjusted_strength)
                } else if momentum < -momentum_threshold {
                    // Negative momentum
                    short_signals += 1;
                    let strength = (0.5 + (-momentum - momentum_threshold).min(0.5)).min(1.0);
                    let adjusted_strength = strength * (0.5 + er * 0.5);
                    (SignalDirection::Short, adjusted_strength)
                } else {
                    (SignalDirection::Flat, 0.0)
                };

                let signal = Signal::new(&candidate.symbol, direction, strength)
                    .with_source("adaptive_momentum")
                    .with_metadata("price", current_price)
                    .with_metadata("momentum", momentum)
                    .with_metadata("efficiency_ratio", er)
                    .with_metadata("adaptive_period", adaptive_period as f64);

                signals.push(signal);
            }
        }

        ctx.trace_step(
            self.block_id(),
            &format!(
                "AdaptiveMomentum({}-{}): {} long, {} short from {} candidates",
                min_period, max_period, long_signals, short_signals, ctx.candidates.len()
            ),
        );

        BlockResult::success(format!(
            "Adaptive Momentum: {} long, {} short",
            long_signals, short_signals
        ))
        .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let min_period = get_usize(params, "min_period", 10);
        let max_period = get_usize(params, "max_period", 50);
        let threshold = get_f64(params, "momentum_threshold", 0.05);

        if min_period < 5 {
            return Err(ValidationError::OutOfRange(
                "min_period".into(),
                "must be at least 5".into(),
            ));
        }

        if max_period <= min_period {
            return Err(ValidationError::OutOfRange(
                "max_period".into(),
                "must be greater than min_period".into(),
            ));
        }

        if threshold <= 0.0 || threshold > 0.5 {
            return Err(ValidationError::OutOfRange(
                "momentum_threshold".into(),
                "must be between 0 and 0.5".into(),
            ));
        }

        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("min_period".into(), toml::Value::Integer(10));
        params.insert("max_period".into(), toml::Value::Integer(50));
        params.insert("momentum_threshold".into(), toml::Value::Float(0.05));
        params
    }

    fn description(&self) -> &'static str {
        "Adaptive Momentum: Variable-period momentum based on market efficiency (Kaufman)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_efficiency_ratio_trending() {
        // Strong uptrend = high ER
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        let er = AdaptiveMomentumBlock::calculate_efficiency_ratio(&prices, 20);
        assert!(er.is_some());
        assert!(er.unwrap() > 0.9); // Should be close to 1 for perfect trend
    }

    #[test]
    fn test_efficiency_ratio_choppy() {
        // Choppy market = low ER
        let mut prices = Vec::new();
        for i in 0..30 {
            prices.push(100.0 + if i % 2 == 0 { 1.0 } else { -1.0 });
        }
        let er = AdaptiveMomentumBlock::calculate_efficiency_ratio(&prices, 20);
        assert!(er.is_some());
        assert!(er.unwrap() < 0.2); // Should be low for choppy market
    }

    #[test]
    fn test_adaptive_period() {
        // High ER -> short period
        assert_eq!(AdaptiveMomentumBlock::calculate_adaptive_period(0.9, 10, 50), 14);
        // Low ER -> long period
        assert_eq!(AdaptiveMomentumBlock::calculate_adaptive_period(0.1, 10, 50), 46);
    }
}
