//! VWAP (Volume-Weighted Average Price) entry blocks.
//!
//! # Mathematical Foundation
//!
//! **VWAP Calculation:**
//! VWAP = Σ(Price_i × Volume_i) / Σ(Volume_i)
//!
//! **Mean Reversion Strategy:**
//! - Long when price << VWAP (Z-score < -threshold)
//! - Short when price >> VWAP (Z-score > +threshold)
//!
//! **Trend Following Strategy:**
//! - Long when price > VWAP and trending up
//! - Short when price < VWAP and trending down
//!
//! # References
//! - Berkowitz, S. et al. (1988). "The Total Cost of Transactions on the NYSE"
//! - Almgren, R. & Chriss, N. (2000). "Optimal Execution of Portfolio Transactions"

use crate::blocks::{
    get_f64, get_usize, get_bool, BlockParams, BlockResult, BlockType, Signal, SignalDirection,
    StrategyBlock, ValidationError,
};
use crate::context::StrategyContext;
use std::collections::HashMap;

/// VWAP-based entry block supporting both reversion and trend modes.
pub struct VWAPBlock;

impl VWAPBlock {
    pub fn new() -> Self {
        Self
    }

    /// Calculate VWAP from typical prices and volumes.
    ///
    /// # Formula
    /// VWAP = Σ(TP × V) / Σ(V)
    /// where TP = (H + L + C) / 3
    ///
    /// Using close prices as proxy when HLC not available.
    pub fn calculate_vwap(prices: &[f64], volumes: &[f64], period: usize) -> Option<f64> {
        if prices.len() < period || volumes.len() < period {
            return None;
        }

        let n = prices.len();
        let start = n.saturating_sub(period);

        let mut sum_pv = 0.0;
        let mut sum_v = 0.0;

        for i in start..n {
            let price = prices[i];
            let volume = volumes.get(i).copied().unwrap_or(1.0);
            sum_pv += price * volume;
            sum_v += volume;
        }

        if sum_v > 0.0 {
            Some(sum_pv / sum_v)
        } else {
            None
        }
    }

    /// Calculate VWAP standard deviation for Z-score.
    pub fn calculate_vwap_std(
        prices: &[f64],
        volumes: &[f64],
        vwap: f64,
        period: usize,
    ) -> Option<f64> {
        if prices.len() < period {
            return None;
        }

        let n = prices.len();
        let start = n.saturating_sub(period);

        let mut sum_sq_diff = 0.0;
        let mut sum_v = 0.0;

        for i in start..n {
            let price = prices[i];
            let volume = volumes.get(i).copied().unwrap_or(1.0);
            sum_sq_diff += volume * (price - vwap).powi(2);
            sum_v += volume;
        }

        if sum_v > 0.0 {
            Some((sum_sq_diff / sum_v).sqrt())
        } else {
            None
        }
    }
}

impl Default for VWAPBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for VWAPBlock {
    fn block_id(&self) -> &'static str {
        "vwap"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Entry
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let period = get_usize(params, "period", 20);
        let threshold = get_f64(params, "threshold", 1.5);
        let mode = if get_bool(params, "trend_mode", false) {
            "trend"
        } else {
            "reversion"
        };

        let mut signals = Vec::new();
        let mut long_signals = 0;
        let mut short_signals = 0;

        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            
            if prices.len() < period + 1 {
                continue;
            }

            // Use constant volume if not available
            let volumes: Vec<f64> = vec![1.0; prices.len()];
            
            let current_price = *prices.last().unwrap();

            if let Some(vwap) = Self::calculate_vwap(prices, &volumes, period) {
                let std = Self::calculate_vwap_std(prices, &volumes, vwap, period)
                    .unwrap_or(vwap * 0.02);
                
                let z_score = if std > 0.0 {
                    (current_price - vwap) / std
                } else {
                    0.0
                };

                let (direction, strength) = match mode {
                    "reversion" => {
                        // Mean reversion: buy low, sell high
                        if z_score < -threshold {
                            long_signals += 1;
                            (SignalDirection::Long, (0.5 + (-z_score - threshold).min(1.5) / 3.0).min(1.0))
                        } else if z_score > threshold {
                            short_signals += 1;
                            (SignalDirection::Short, (0.5 + (z_score - threshold).min(1.5) / 3.0).min(1.0))
                        } else {
                            (SignalDirection::Flat, 0.0)
                        }
                    }
                    "trend" => {
                        // Trend following: buy above VWAP, sell below
                        if z_score > threshold {
                            long_signals += 1;
                            (SignalDirection::Long, (0.5 + (z_score - threshold).min(1.5) / 3.0).min(1.0))
                        } else if z_score < -threshold {
                            short_signals += 1;
                            (SignalDirection::Short, (0.5 + (-z_score - threshold).min(1.5) / 3.0).min(1.0))
                        } else {
                            (SignalDirection::Flat, 0.0)
                        }
                    }
                    _ => (SignalDirection::Flat, 0.0),
                };

                let signal = Signal::new(&candidate.symbol, direction, strength)
                    .with_source("vwap")
                    .with_metadata("price", current_price)
                    .with_metadata("vwap", vwap)
                    .with_metadata("z_score", z_score)
                    .with_metadata("std", std);

                signals.push(signal);
            }
        }

        ctx.trace_step(
            self.block_id(),
            &format!(
                "VWAP({}, {}): {} long, {} short from {} candidates",
                mode, threshold, long_signals, short_signals, ctx.candidates.len()
            ),
        );

        BlockResult::success(format!(
            "VWAP {}: {} long, {} short",
            mode, long_signals, short_signals
        ))
        .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let period = get_usize(params, "period", 20);
        let threshold = get_f64(params, "threshold", 1.5);

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
        params.insert("threshold".into(), toml::Value::Float(1.5));
        params.insert("trend_mode".into(), toml::Value::Boolean(false));
        params
    }

    fn description(&self) -> &'static str {
        "VWAP: Volume-Weighted Average Price reversion or trend signals"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vwap_calculation() {
        let prices = vec![100.0, 102.0, 101.0, 103.0, 105.0];
        let volumes = vec![1000.0, 1200.0, 800.0, 1500.0, 1100.0];
        
        let vwap = VWAPBlock::calculate_vwap(&prices, &volumes, 5);
        assert!(vwap.is_some());
        
        // Manual: (100*1000 + 102*1200 + 101*800 + 103*1500 + 105*1100) / 5600
        let expected = (100000.0 + 122400.0 + 80800.0 + 154500.0 + 115500.0) / 5600.0;
        assert!((vwap.unwrap() - expected).abs() < 0.01);
    }

    #[test]
    fn test_vwap_std() {
        let prices = vec![100.0, 102.0, 101.0, 103.0, 105.0];
        let volumes = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let vwap = 102.2; // Approximate mean
        
        let std = VWAPBlock::calculate_vwap_std(&prices, &volumes, vwap, 5);
        assert!(std.is_some());
        assert!(std.unwrap() > 0.0);
    }
}
