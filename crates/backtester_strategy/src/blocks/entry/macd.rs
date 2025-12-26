//! MACD entry block (Technique 11).
//!
//! Generates signals based on MACD line crossing signal line.

use crate::blocks::{
    get_usize, BlockParams, BlockResult, BlockType, Signal, SignalDirection, StrategyBlock,
    ValidationError,
};
use crate::context::StrategyContext;
use std::collections::HashMap;

/// MACD block - Technique 11.
pub struct MacdBlock;

impl MacdBlock {
    pub fn new() -> Self {
        Self
    }

    fn calculate_ema(prices: &[f64], period: usize) -> Option<f64> {
        if prices.len() < period {
            return None;
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        
        // Start with SMA for first value
        let sma: f64 = prices.iter().take(period).sum::<f64>() / period as f64;
        
        // Calculate EMA from SMA
        let mut ema = sma;
        for price in prices.iter().skip(period) {
            ema = (price - ema) * multiplier + ema;
        }

        Some(ema)
    }

    fn calculate_macd(
        prices: &[f64],
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    ) -> Option<(f64, f64, f64)> {
        if prices.len() < slow_period + signal_period {
            return None;
        }

        let fast_ema = Self::calculate_ema(prices, fast_period)?;
        let slow_ema = Self::calculate_ema(prices, slow_period)?;
        let macd_line = fast_ema - slow_ema;

        // Calculate MACD line history for signal line
        let mut macd_history = Vec::new();
        for i in (slow_period + signal_period)..=prices.len() {
            let slice = &prices[..i];
            if let (Some(f), Some(s)) = (
                Self::calculate_ema(slice, fast_period),
                Self::calculate_ema(slice, slow_period),
            ) {
                macd_history.push(f - s);
            }
        }

        if macd_history.len() < signal_period {
            return None;
        }

        let signal_line = Self::calculate_ema(&macd_history, signal_period)?;
        let histogram = macd_line - signal_line;

        Some((macd_line, signal_line, histogram))
    }
}

impl Default for MacdBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for MacdBlock {
    fn block_id(&self) -> &'static str {
        "macd"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Entry
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let fast_ema = get_usize(params, "fast_ema", 12);
        let slow_ema = get_usize(params, "slow_ema", 26);
        let signal = get_usize(params, "signal", 9);

        let mut signals = Vec::new();
        let mut bullish_count = 0;
        let mut bearish_count = 0;

        for candidate in &ctx.candidates {
            if let Some((macd_line, signal_line, histogram)) =
                Self::calculate_macd(&candidate.prices, fast_ema, slow_ema, signal)
            {
                // Calculate previous values for crossover detection
                let prev_prices = &candidate.prices[..candidate.prices.len().saturating_sub(1)];
                let prev_macd = Self::calculate_macd(prev_prices, fast_ema, slow_ema, signal);

                let (direction, strength) = if let Some((_, _prev_signal, prev_hist)) = prev_macd {
                    // Bullish crossover: MACD crosses above signal
                    if prev_hist <= 0.0 && histogram > 0.0 {
                        bullish_count += 1;
                        (SignalDirection::Long, 0.8 + histogram.abs().min(0.2))
                    }
                    // Bearish crossover: MACD crosses below signal
                    else if prev_hist >= 0.0 && histogram < 0.0 {
                        bearish_count += 1;
                        (SignalDirection::Exit, 0.7 + histogram.abs().min(0.3))
                    }
                    // Already bullish (histogram positive)
                    else if histogram > 0.0 {
                        (SignalDirection::Long, 0.5 + histogram.abs().min(0.3))
                    } else {
                        (SignalDirection::Flat, 0.0)
                    }
                } else {
                    (SignalDirection::Flat, 0.0)
                };

                let signal_out = Signal::new(&candidate.symbol, direction, strength)
                    .with_source("macd")
                    .with_metadata("macd_line", macd_line)
                    .with_metadata("signal_line", signal_line)
                    .with_metadata("histogram", histogram);

                signals.push(signal_out);
            }
        }

        ctx.trace_step(
            self.block_id(),
            &format!(
                "{} bullish, {} bearish crossovers from {} candidates",
                bullish_count,
                bearish_count,
                ctx.candidates.len()
            ),
        );

        BlockResult::success(format!(
            "MACD: {} bullish, {} bearish crossovers",
            bullish_count, bearish_count
        ))
        .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let fast = get_usize(params, "fast_ema", 12);
        let slow = get_usize(params, "slow_ema", 26);
        let signal = get_usize(params, "signal", 9);

        if fast >= slow {
            return Err(ValidationError::OutOfRange(
                "fast_ema".into(),
                "must be less than slow_ema".into(),
            ));
        }

        if signal < 2 {
            return Err(ValidationError::OutOfRange(
                "signal".into(),
                "must be at least 2".into(),
            ));
        }

        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("fast_ema".into(), toml::Value::Integer(12));
        params.insert("slow_ema".into(), toml::Value::Integer(26));
        params.insert("signal".into(), toml::Value::Integer(9));
        params
    }

    fn description(&self) -> &'static str {
        "MACD: Long on bullish crossover, exit on bearish crossover"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_calculation() {
        let prices: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let ema = MacdBlock::calculate_ema(&prices, 12);
        
        assert!(ema.is_some());
        // EMA should be close to recent prices due to weighting
        assert!(ema.unwrap() > 10.0);
    }

    #[test]
    fn test_macd_calculation() {
        let prices: Vec<f64> = (1..=50).map(|x| 100.0 + x as f64 * 0.5).collect();
        let result = MacdBlock::calculate_macd(&prices, 12, 26, 9);
        
        assert!(result.is_some());
        let (macd, signal, histogram) = result.unwrap();
        // In uptrend, fast EMA should be above slow EMA, so MACD > 0
        assert!(macd > 0.0);
    }
}

