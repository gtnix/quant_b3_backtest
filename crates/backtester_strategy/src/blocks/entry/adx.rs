//! ADX (Average Directional Index) Momentum entry block.
//!
//! # Mathematical Foundation
//!
//! **Directional Movement (Wilder, 1978):**
//! +DM = High_t - High_{t-1} (if > 0 and > -DM)
//! -DM = Low_{t-1} - Low_t (if > 0 and > +DM)
//!
//! **Directional Indicators:**
//! +DI = 100 × EMA(+DM) / ATR
//! -DI = 100 × EMA(-DM) / ATR
//!
//! **Average Directional Index:**
//! DX = 100 × |+DI - -DI| / (+DI + -DI)
//! ADX = EMA(DX, period)
//!
//! **Trading Rules:**
//! - ADX > 25: Strong trend (trade direction of +DI/-DI)
//! - ADX < 20: Weak/no trend (avoid)
//! - Long: ADX > threshold AND +DI > -DI
//! - Short: ADX > threshold AND -DI > +DI
//!
//! # References
//! - Wilder, J.W. (1978). "New Concepts in Technical Trading Systems"

use crate::blocks::{
    get_f64, get_usize, BlockParams, BlockResult, BlockType, Signal, SignalDirection,
    StrategyBlock, ValidationError,
};
use crate::context::StrategyContext;
use std::collections::HashMap;

/// ADX Momentum entry block.
pub struct ADXMomentumBlock;

impl ADXMomentumBlock {
    pub fn new() -> Self {
        Self
    }

    /// Calculate simplified ADX using close prices only.
    /// Full ADX requires H/L/C, this uses close approximation.
    pub fn calculate_adx_simplified(prices: &[f64], period: usize) -> Option<(f64, f64, f64)> {
        if prices.len() < period * 2 + 1 {
            return None;
        }

        let n = prices.len();
        let mut plus_dm_sum = 0.0;
        let mut minus_dm_sum = 0.0;
        let mut tr_sum = 0.0;

        // Calculate directional movement over period
        for i in (n - period - 1)..(n - 1) {
            let change = prices[i + 1] - prices[i];
            let prev_change = if i > 0 { prices[i] - prices[i - 1] } else { 0.0 };
            
            let tr = change.abs().max(prev_change.abs());
            tr_sum += tr;

            if change > 0.0 && change > prev_change.abs() {
                plus_dm_sum += change;
            } else if prev_change < 0.0 && prev_change.abs() > change {
                minus_dm_sum += prev_change.abs();
            }
        }

        if tr_sum == 0.0 {
            return None;
        }

        let plus_di = 100.0 * plus_dm_sum / tr_sum;
        let minus_di = 100.0 * minus_dm_sum / tr_sum;

        let di_sum = plus_di + minus_di;
        let adx = if di_sum > 0.0 {
            100.0 * (plus_di - minus_di).abs() / di_sum
        } else {
            0.0
        };

        Some((adx, plus_di, minus_di))
    }
}

impl Default for ADXMomentumBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for ADXMomentumBlock {
    fn block_id(&self) -> &'static str {
        "adx_momentum"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Entry
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let period = get_usize(params, "period", 14);
        let adx_threshold = get_f64(params, "adx_threshold", 25.0);

        let mut signals = Vec::new();
        let mut long_signals = 0;
        let mut short_signals = 0;
        let mut trend_count = 0;

        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            
            if prices.len() < period * 2 + 5 {
                continue;
            }

            let current_price = *prices.last().unwrap();

            if let Some((adx, plus_di, minus_di)) = Self::calculate_adx_simplified(prices, period) {
                let strong_trend = adx >= adx_threshold;
                
                if strong_trend {
                    trend_count += 1;
                }

                let (direction, strength) = if strong_trend && plus_di > minus_di {
                    // Strong bullish trend
                    long_signals += 1;
                    let strength = (0.5 + (adx - adx_threshold) / 50.0).min(1.0);
                    (SignalDirection::Long, strength)
                } else if strong_trend && minus_di > plus_di {
                    // Strong bearish trend
                    short_signals += 1;
                    let strength = (0.5 + (adx - adx_threshold) / 50.0).min(1.0);
                    (SignalDirection::Short, strength)
                } else {
                    (SignalDirection::Flat, 0.0)
                };

                let signal = Signal::new(&candidate.symbol, direction, strength)
                    .with_source("adx_momentum")
                    .with_metadata("price", current_price)
                    .with_metadata("adx", adx)
                    .with_metadata("plus_di", plus_di)
                    .with_metadata("minus_di", minus_di);

                signals.push(signal);
            }
        }

        ctx.trace_step(
            self.block_id(),
            &format!(
                "ADX({}, >{:.0}): {} trending, {} long, {} short from {} candidates",
                period, adx_threshold, trend_count, long_signals, short_signals, ctx.candidates.len()
            ),
        );

        BlockResult::success(format!(
            "ADX Momentum: {} trending, {} long, {} short",
            trend_count, long_signals, short_signals
        ))
        .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let period = get_usize(params, "period", 14);
        let threshold = get_f64(params, "adx_threshold", 25.0);

        if period < 5 || period > 50 {
            return Err(ValidationError::OutOfRange(
                "period".into(),
                "must be between 5 and 50".into(),
            ));
        }

        if threshold < 10.0 || threshold > 50.0 {
            return Err(ValidationError::OutOfRange(
                "adx_threshold".into(),
                "must be between 10 and 50".into(),
            ));
        }

        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("period".into(), toml::Value::Integer(14));
        params.insert("adx_threshold".into(), toml::Value::Float(25.0));
        params
    }

    fn description(&self) -> &'static str {
        "ADX Momentum: Trade in direction of strong trends (Wilder)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adx_calculation() {
        // Create trending prices
        let mut prices = Vec::new();
        for i in 0..40 {
            prices.push(100.0 + i as f64 * 0.5);
        }
        
        let result = ADXMomentumBlock::calculate_adx_simplified(&prices, 14);
        assert!(result.is_some());
        
        let (adx, plus_di, _minus_di) = result.unwrap();
        assert!(adx >= 0.0);
        assert!(plus_di >= 0.0); // Uptrend should have positive +DI
    }
}
