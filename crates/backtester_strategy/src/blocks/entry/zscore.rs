//! Z-Score mean reversion entry block (Technique 12).
//!
//! Generates signals when price deviates significantly from mean.

use crate::blocks::{
    get_f64, get_usize, BlockParams, BlockResult, BlockType, Signal, SignalDirection,
    StrategyBlock, ValidationError,
};
use crate::context::StrategyContext;
use std::collections::HashMap;

/// Z-Score block - Technique 12.
pub struct ZScoreBlock;

impl ZScoreBlock {
    pub fn new() -> Self {
        Self
    }

    fn calculate_zscore(prices: &[f64], period: usize) -> Option<f64> {
        if prices.len() < period {
            return None;
        }

        let recent: Vec<f64> = prices.iter().rev().take(period).copied().collect();
        let current_price = *prices.last()?;
        
        let mean: f64 = recent.iter().sum::<f64>() / period as f64;
        let variance: f64 = recent.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        let std_dev = variance.sqrt();

        if std_dev < 0.0001 {
            return Some(0.0);
        }

        Some((current_price - mean) / std_dev)
    }
}

impl Default for ZScoreBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for ZScoreBlock {
    fn block_id(&self) -> &'static str {
        "zscore"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Entry
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let period = get_usize(params, "period", 20);
        let threshold = get_f64(params, "threshold", 2.0);

        let mut signals = Vec::new();
        let mut oversold_count = 0;
        let mut overbought_count = 0;

        for candidate in &ctx.candidates {
            if let Some(zscore) = Self::calculate_zscore(&candidate.prices, period) {
                let (direction, strength) = if zscore < -threshold {
                    // Oversold - mean reversion buy
                    let strength = ((-zscore - threshold) / threshold).min(1.0);
                    oversold_count += 1;
                    (SignalDirection::Long, 0.6 + strength * 0.4)
                } else if zscore > threshold {
                    // Overbought - mean reversion sell/exit
                    let strength = ((zscore - threshold) / threshold).min(1.0);
                    overbought_count += 1;
                    (SignalDirection::Exit, 0.5 + strength * 0.5)
                } else {
                    // Near mean - flat
                    (SignalDirection::Flat, 0.0)
                };

                let signal = Signal::new(&candidate.symbol, direction, strength)
                    .with_source("zscore")
                    .with_metadata("zscore", zscore);

                signals.push(signal);
            }
        }

        ctx.trace_step(
            self.block_id(),
            &format!(
                "{} oversold (z<-{}), {} overbought (z>{}) from {} candidates",
                oversold_count,
                threshold,
                overbought_count,
                threshold,
                ctx.candidates.len()
            ),
        );

        BlockResult::success(format!(
            "Z-Score: {} oversold, {} overbought signals",
            oversold_count, overbought_count
        ))
        .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let period = get_usize(params, "period", 20);
        let threshold = get_f64(params, "threshold", 2.0);

        if period < 5 {
            return Err(ValidationError::OutOfRange(
                "period".into(),
                "must be at least 5".into(),
            ));
        }

        if threshold <= 0.0 || threshold > 5.0 {
            return Err(ValidationError::OutOfRange(
                "threshold".into(),
                "must be between 0 and 5".into(),
            ));
        }

        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("period".into(), toml::Value::Integer(20));
        params.insert("threshold".into(), toml::Value::Float(2.0));
        params
    }

    fn description(&self) -> &'static str {
        "Z-Score: Long on z < -2 (oversold), exit on z > 2 (overbought)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zscore_calculation() {
        // Normal distribution-like prices centered at 100
        let mut prices: Vec<f64> = (1..=20).map(|_| 100.0).collect();
        // Current price is 2 std devs above
        prices.push(110.0);
        
        let zscore = ZScoreBlock::calculate_zscore(&prices, 20);
        assert!(zscore.is_some());
        // Z-score should be very high since all other prices are 100
    }

    #[test]
    fn test_zscore_zero_variance() {
        let prices: Vec<f64> = vec![100.0; 20];
        let zscore = ZScoreBlock::calculate_zscore(&prices, 20);
        
        assert!(zscore.is_some());
        assert!((zscore.unwrap()).abs() < 0.01);
    }
}

