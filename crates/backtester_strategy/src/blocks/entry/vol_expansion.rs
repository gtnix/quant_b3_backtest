//! Volatility Expansion (Squeeze Breakout) entry block.
//!
//! # Mathematical Foundation
//!
//! **Bollinger Bandwidth:**
//! BBW = (Upper - Lower) / Middle = 2 × std_dev × num_std / SMA
//!
//! **Squeeze Detection:**
//! Squeeze = BBW < threshold (volatility contraction)
//!
//! **Expansion Signal:**
//! - Long: Price breaks above upper band after squeeze
//! - Short: Price breaks below lower band after squeeze
//!
//! The squeeze-breakout pattern exploits volatility clustering:
//! periods of low volatility tend to precede high volatility moves.
//!
//! # References
//! - Bollinger, J. (2002). "Bollinger on Bollinger Bands"
//! - Mandelbrot, B. (1963). "The Variation of Certain Speculative Prices"

use crate::blocks::{
    get_f64, get_usize, BlockParams, BlockResult, BlockType, Signal, SignalDirection,
    StrategyBlock, ValidationError,
};
use crate::context::StrategyContext;
use std::collections::HashMap;

/// Volatility Expansion (Squeeze Breakout) entry block.
pub struct VolatilityExpansionBlock;

impl VolatilityExpansionBlock {
    pub fn new() -> Self {
        Self
    }

    /// Calculate Bollinger Bandwidth (BBW).
    /// BBW = (Upper - Lower) / Middle
    pub fn calculate_bandwidth(prices: &[f64], period: usize, num_std: f64) -> Option<(f64, f64, f64, f64)> {
        if prices.len() < period {
            return None;
        }

        let n = prices.len();
        let window = &prices[(n - period)..n];

        let mean: f64 = window.iter().sum::<f64>() / period as f64;
        let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        let std = variance.sqrt();

        let upper = mean + num_std * std;
        let lower = mean - num_std * std;
        let bandwidth = if mean > 0.0 {
            (upper - lower) / mean
        } else {
            0.0
        };

        Some((bandwidth, mean, upper, lower))
    }

    /// Calculate percentile of current bandwidth vs historical.
    pub fn bandwidth_percentile(prices: &[f64], period: usize, num_std: f64, lookback: usize) -> Option<f64> {
        if prices.len() < period + lookback {
            return None;
        }

        let n = prices.len();
        let mut bandwidths = Vec::with_capacity(lookback);

        for i in 0..lookback {
            let end = n - i;
            let start = end.saturating_sub(period);
            if start >= end {
                continue;
            }
            
            let window = &prices[start..end];
            let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
            let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window.len() as f64;
            let std = variance.sqrt();
            
            if mean > 0.0 {
                let bw = 2.0 * num_std * std / mean;
                bandwidths.push(bw);
            }
        }

        if bandwidths.is_empty() {
            return None;
        }

        let current_bw = bandwidths[0];
        let count_below = bandwidths.iter().filter(|&&b| b < current_bw).count();
        Some(count_below as f64 / bandwidths.len() as f64)
    }
}

impl Default for VolatilityExpansionBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for VolatilityExpansionBlock {
    fn block_id(&self) -> &'static str {
        "vol_expansion"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Entry
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let period = get_usize(params, "period", 20);
        let num_std = get_f64(params, "num_std", 2.0);
        let squeeze_percentile = get_f64(params, "squeeze_percentile", 20.0);
        let lookback = get_usize(params, "lookback", 126);

        let mut signals = Vec::new();
        let mut long_signals = 0;
        let mut short_signals = 0;
        let mut squeeze_count = 0;

        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            
            if prices.len() < period + lookback {
                continue;
            }

            let current_price = *prices.last().unwrap();

            // Calculate current bandwidth and percentile
            if let Some((bandwidth, _middle, upper, lower)) = 
                Self::calculate_bandwidth(prices, period, num_std) 
            {
                let percentile = Self::bandwidth_percentile(prices, period, num_std, lookback)
                    .unwrap_or(50.0);

                let in_squeeze = percentile * 100.0 < squeeze_percentile;
                
                if in_squeeze {
                    squeeze_count += 1;
                }

                // Look for expansion breakout from squeeze
                let (direction, strength) = if current_price > upper && in_squeeze {
                    // Breakout above after squeeze
                    long_signals += 1;
                    let strength = (0.6 + (1.0 - percentile).min(0.4)).min(1.0);
                    (SignalDirection::Long, strength)
                } else if current_price < lower && in_squeeze {
                    // Breakout below after squeeze
                    short_signals += 1;
                    let strength = (0.6 + (1.0 - percentile).min(0.4)).min(1.0);
                    (SignalDirection::Short, strength)
                } else {
                    (SignalDirection::Flat, 0.0)
                };

                let signal = Signal::new(&candidate.symbol, direction, strength)
                    .with_source("vol_expansion")
                    .with_metadata("price", current_price)
                    .with_metadata("bandwidth", bandwidth)
                    .with_metadata("percentile", percentile * 100.0)
                    .with_metadata("upper", upper)
                    .with_metadata("lower", lower);

                signals.push(signal);
            }
        }

        ctx.trace_step(
            self.block_id(),
            &format!(
                "VolExpansion: {} in squeeze, {} long, {} short from {} candidates",
                squeeze_count, long_signals, short_signals, ctx.candidates.len()
            ),
        );

        BlockResult::success(format!(
            "Vol Expansion: {} squeeze, {} long, {} short",
            squeeze_count, long_signals, short_signals
        ))
        .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let period = get_usize(params, "period", 20);
        let num_std = get_f64(params, "num_std", 2.0);
        let squeeze_pct = get_f64(params, "squeeze_percentile", 20.0);

        if period < 5 {
            return Err(ValidationError::OutOfRange(
                "period".into(),
                "must be at least 5".into(),
            ));
        }

        if num_std <= 0.0 || num_std > 5.0 {
            return Err(ValidationError::OutOfRange(
                "num_std".into(),
                "must be between 0 and 5".into(),
            ));
        }

        if squeeze_pct <= 0.0 || squeeze_pct > 50.0 {
            return Err(ValidationError::OutOfRange(
                "squeeze_percentile".into(),
                "must be between 0 and 50".into(),
            ));
        }

        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("period".into(), toml::Value::Integer(20));
        params.insert("num_std".into(), toml::Value::Float(2.0));
        params.insert("squeeze_percentile".into(), toml::Value::Float(20.0));
        params.insert("lookback".into(), toml::Value::Integer(126));
        params
    }

    fn description(&self) -> &'static str {
        "Volatility Expansion: Breakout signals after squeeze (low volatility)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bandwidth_calculation() {
        let prices: Vec<f64> = (1..=20).map(|x| 100.0 + x as f64).collect();
        
        let result = VolatilityExpansionBlock::calculate_bandwidth(&prices, 20, 2.0);
        assert!(result.is_some());
        
        let (bandwidth, middle, upper, lower) = result.unwrap();
        assert!(bandwidth > 0.0);
        assert!(upper > middle);
        assert!(lower < middle);
    }

    #[test]
    fn test_bandwidth_percentile() {
        // Create trending prices with varying volatility
        let mut prices = Vec::new();
        for i in 0..200 {
            // Low vol period followed by high vol
            let noise = if i < 100 { 0.5 } else { 2.0 };
            prices.push(100.0 + (i as f64 * 0.1) + noise * (i as f64).sin());
        }
        
        let percentile = VolatilityExpansionBlock::bandwidth_percentile(&prices, 20, 2.0, 50);
        assert!(percentile.is_some());
    }
}
