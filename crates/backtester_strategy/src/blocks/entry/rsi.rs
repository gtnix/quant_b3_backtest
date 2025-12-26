//! RSI entry block (Technique 10).
//!
//! Generates signals based on RSI oversold/overbought levels.

use crate::blocks::{
    get_f64, get_usize, BlockParams, BlockResult, BlockType, Signal, SignalDirection,
    StrategyBlock, ValidationError,
};
use crate::context::StrategyContext;
use std::collections::HashMap;

/// RSI block - Technique 10.
pub struct RsiBlock;

impl RsiBlock {
    pub fn new() -> Self {
        Self
    }

    fn calculate_rsi(prices: &[f64], period: usize) -> Option<f64> {
        if prices.len() < period + 1 {
            return None;
        }

        let mut gains = 0.0;
        let mut losses = 0.0;

        // Calculate changes for last `period` days
        let start = prices.len().saturating_sub(period + 1);
        for i in (start + 1)..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains += change;
            } else {
                losses -= change; // Make positive
            }
        }

        let avg_gain = gains / period as f64;
        let avg_loss = losses / period as f64;

        if avg_loss == 0.0 {
            return Some(100.0);
        }

        let rs = avg_gain / avg_loss;
        let rsi = 100.0 - (100.0 / (1.0 + rs));

        Some(rsi)
    }
}

impl Default for RsiBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for RsiBlock {
    fn block_id(&self) -> &'static str {
        "rsi"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Entry
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let period = get_usize(params, "period", 14);
        let oversold = get_f64(params, "oversold", 30.0);
        let overbought = get_f64(params, "overbought", 70.0);

        let mut signals = Vec::new();
        let mut oversold_count = 0;
        let mut overbought_count = 0;

        for candidate in &ctx.candidates {
            if let Some(rsi) = Self::calculate_rsi(&candidate.prices, period) {
                let (direction, strength) = if rsi < oversold {
                    // Oversold - buy signal
                    let strength = (oversold - rsi) / oversold;
                    oversold_count += 1;
                    (SignalDirection::Long, 0.6 + strength.min(1.0) * 0.4)
                } else if rsi > overbought {
                    // Overbought - exit signal
                    let strength = (rsi - overbought) / (100.0 - overbought);
                    overbought_count += 1;
                    (SignalDirection::Exit, 0.5 + strength.min(1.0) * 0.5)
                } else {
                    (SignalDirection::Flat, 0.0)
                };

                let signal = Signal::new(&candidate.symbol, direction, strength)
                    .with_source("rsi")
                    .with_metadata("rsi", rsi);

                signals.push(signal);
            }
        }

        ctx.trace_step(
            self.block_id(),
            &format!(
                "{} oversold, {} overbought from {} candidates",
                oversold_count,
                overbought_count,
                ctx.candidates.len()
            ),
        );

        BlockResult::success(format!(
            "RSI: {} oversold, {} overbought signals",
            oversold_count, overbought_count
        ))
        .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let period = get_usize(params, "period", 14);
        let oversold = get_f64(params, "oversold", 30.0);
        let overbought = get_f64(params, "overbought", 70.0);

        if period < 2 {
            return Err(ValidationError::OutOfRange(
                "period".into(),
                "must be at least 2".into(),
            ));
        }

        if oversold >= overbought {
            return Err(ValidationError::OutOfRange(
                "oversold".into(),
                "must be less than overbought".into(),
            ));
        }

        if oversold < 0.0 || overbought > 100.0 {
            return Err(ValidationError::OutOfRange(
                "thresholds".into(),
                "must be between 0 and 100".into(),
            ));
        }

        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("period".into(), toml::Value::Integer(14));
        params.insert("oversold".into(), toml::Value::Float(30.0));
        params.insert("overbought".into(), toml::Value::Float(70.0));
        params
    }

    fn description(&self) -> &'static str {
        "RSI: Long on oversold (<30), exit on overbought (>70)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsi_calculation() {
        // Create upward trending prices
        let prices: Vec<f64> = (1..=20).map(|x| 100.0 + x as f64).collect();
        let rsi = RsiBlock::calculate_rsi(&prices, 14);
        
        assert!(rsi.is_some());
        assert!(rsi.unwrap() > 50.0); // Uptrend should have RSI > 50
    }

    #[test]
    fn test_rsi_all_gains() {
        // Pure uptrend should give RSI close to 100
        let prices: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let rsi = RsiBlock::calculate_rsi(&prices, 14);
        
        assert!(rsi.is_some());
        assert!(rsi.unwrap() > 90.0);
    }
}

