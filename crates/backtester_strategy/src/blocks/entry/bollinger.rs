//! Bollinger Bands breakout entry block (Technique 9).
//!
//! Generates signals on breakouts above/below Bollinger Bands.

use crate::blocks::{
    get_f64, get_usize, BlockParams, BlockResult, BlockType, Signal, SignalDirection,
    StrategyBlock, ValidationError,
};
use crate::context::StrategyContext;
use std::collections::HashMap;

/// Bollinger Bands breakout block - Technique 9.
pub struct BollingerBlock;

impl BollingerBlock {
    pub fn new() -> Self {
        Self
    }

    fn calculate_bollinger(prices: &[f64], period: usize, std_dev: f64) -> Option<(f64, f64, f64)> {
        if prices.len() < period {
            return None;
        }

        let recent: Vec<f64> = prices.iter().rev().take(period).copied().collect();
        let mean: f64 = recent.iter().sum::<f64>() / period as f64;
        
        let variance: f64 = recent.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        let std = variance.sqrt();

        let upper = mean + std_dev * std;
        let lower = mean - std_dev * std;

        Some((mean, upper, lower))
    }
}

impl Default for BollingerBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for BollingerBlock {
    fn block_id(&self) -> &'static str {
        "bollinger"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Entry
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let period = get_usize(params, "period", 20);
        let std_dev = get_f64(params, "std_dev", 2.0);

        let mut signals = Vec::new();
        let mut long_count = 0;
        let mut short_count = 0;

        for candidate in &ctx.candidates {
            if candidate.prices.len() < period + 1 {
                continue;
            }

            let current_price = *candidate.prices.last().unwrap();

            if let Some((middle, upper, lower)) =
                Self::calculate_bollinger(&candidate.prices, period, std_dev)
            {
                let (direction, strength) = if current_price > upper {
                    // Breakout above upper band - bullish
                    let strength = ((current_price - upper) / (upper - middle)).min(1.0);
                    long_count += 1;
                    (SignalDirection::Long, 0.7 + strength * 0.3)
                } else if current_price < lower {
                    // Breakout below lower band - mean reversion buy
                    let strength = ((lower - current_price) / (middle - lower)).min(1.0);
                    short_count += 1;
                    // For mean reversion strategy, this could be a long signal
                    (SignalDirection::Long, 0.6 + strength * 0.3)
                } else {
                    (SignalDirection::Flat, 0.0)
                };

                let signal = Signal::new(&candidate.symbol, direction, strength)
                    .with_source("bollinger")
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
                "{} upper breakouts, {} lower breakouts from {} candidates",
                long_count,
                short_count,
                ctx.candidates.len()
            ),
        );

        BlockResult::success(format!(
            "Bollinger: {} upper, {} lower breakouts",
            long_count, short_count
        ))
        .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let period = get_usize(params, "period", 20);
        let std_dev = get_f64(params, "std_dev", 2.0);

        if period < 5 {
            return Err(ValidationError::OutOfRange(
                "period".into(),
                "must be at least 5".into(),
            ));
        }

        if std_dev <= 0.0 || std_dev > 5.0 {
            return Err(ValidationError::OutOfRange(
                "std_dev".into(),
                "must be between 0 and 5".into(),
            ));
        }

        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("period".into(), toml::Value::Integer(20));
        params.insert("std_dev".into(), toml::Value::Float(2.0));
        params
    }

    fn description(&self) -> &'static str {
        "Bollinger Bands: Signal on breakouts above/below bands"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bollinger_calculation() {
        let prices: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let result = BollingerBlock::calculate_bollinger(&prices, 20, 2.0);
        
        assert!(result.is_some());
        let (middle, upper, lower) = result.unwrap();
        assert!((middle - 10.5).abs() < 0.1); // Mean of 1..20
        assert!(upper > middle);
        assert!(lower < middle);
    }
}

