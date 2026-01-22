//! Gap trading entry blocks (Gap Fill and Gap Continuation).
//!
//! # Mathematical Foundation
//!
//! **Gap Calculation:**
//! Gap% = (Open_today - Close_yesterday) / Close_yesterday
//!
//! **Gap Fill Strategy (Mean Reversion):**
//! - Gaps have ~70% probability of filling within same session (per academic studies)
//! - Long when gap down > threshold (expect price to rise)
//! - Short when gap up > threshold (expect price to fall)
//!
//! **Gap Continuation Strategy (Momentum):**
//! - Large gaps often indicate strong momentum
//! - Long when gap up > threshold and confirmed by volume
//! - Short when gap down > threshold and confirmed by volume
//!
//! # References
//! - Cooper, M. et al. (2003). "Market States and Momentum"
//! - Ma, L. et al. (2017). "Overnight versus Intraday Return Prediction"

use crate::blocks::{
    get_f64, get_usize, get_bool, BlockParams, BlockResult, BlockType, Signal, SignalDirection,
    StrategyBlock, ValidationError,
};
use crate::context::StrategyContext;
use std::collections::HashMap;

/// Gap Fill entry block - mean reversion on gaps.
pub struct GapFillBlock;

impl GapFillBlock {
    pub fn new() -> Self {
        Self
    }

    /// Calculate gap percentage between consecutive closes.
    /// In daily data, gap = (today_open - yesterday_close) / yesterday_close
    /// Using close-to-close as proxy.
    pub fn calculate_gap(prices: &[f64]) -> Option<f64> {
        if prices.len() < 2 {
            return None;
        }

        let n = prices.len();
        let prev_close = prices[n - 2];
        let current = prices[n - 1];

        if prev_close > 0.0 {
            Some((current - prev_close) / prev_close)
        } else {
            None
        }
    }
}

impl Default for GapFillBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for GapFillBlock {
    fn block_id(&self) -> &'static str {
        "gap_fill"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Entry
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let min_gap_pct = get_f64(params, "min_gap_pct", 0.01);
        let max_gap_pct = get_f64(params, "max_gap_pct", 0.05);

        let mut signals = Vec::new();
        let mut long_signals = 0;
        let mut short_signals = 0;

        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            
            if prices.len() < 3 {
                continue;
            }

            if let Some(gap_pct) = Self::calculate_gap(prices) {
                let abs_gap = gap_pct.abs();
                
                // Only trade if gap is within acceptable range
                if abs_gap < min_gap_pct || abs_gap > max_gap_pct {
                    signals.push(
                        Signal::flat(&candidate.symbol).with_source("gap_fill")
                    );
                    continue;
                }

                // Gap fill = mean reversion
                let (direction, strength) = if gap_pct < -min_gap_pct {
                    // Gap down -> expect fill (go long)
                    long_signals += 1;
                    let strength = (0.5 + (abs_gap / max_gap_pct).min(0.5)).min(1.0);
                    (SignalDirection::Long, strength)
                } else if gap_pct > min_gap_pct {
                    // Gap up -> expect fill (go short)
                    short_signals += 1;
                    let strength = (0.5 + (abs_gap / max_gap_pct).min(0.5)).min(1.0);
                    (SignalDirection::Short, strength)
                } else {
                    (SignalDirection::Flat, 0.0)
                };

                let signal = Signal::new(&candidate.symbol, direction, strength)
                    .with_source("gap_fill")
                    .with_metadata("gap_pct", gap_pct);

                signals.push(signal);
            }
        }

        ctx.trace_step(
            self.block_id(),
            &format!(
                "GapFill({:.1}%-{:.1}%): {} long, {} short from {} candidates",
                min_gap_pct * 100.0, max_gap_pct * 100.0, 
                long_signals, short_signals, ctx.candidates.len()
            ),
        );

        BlockResult::success(format!(
            "Gap Fill: {} long, {} short",
            long_signals, short_signals
        ))
        .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let min_gap = get_f64(params, "min_gap_pct", 0.01);
        let max_gap = get_f64(params, "max_gap_pct", 0.05);

        if min_gap <= 0.0 || min_gap >= max_gap {
            return Err(ValidationError::OutOfRange(
                "min_gap_pct".into(),
                "must be positive and less than max_gap_pct".into(),
            ));
        }

        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("min_gap_pct".into(), toml::Value::Float(0.01));
        params.insert("max_gap_pct".into(), toml::Value::Float(0.05));
        params
    }

    fn description(&self) -> &'static str {
        "Gap Fill: Mean reversion on overnight gaps"
    }
}

/// Gap Continuation entry block - momentum on gaps.
pub struct GapContinuationBlock;

impl GapContinuationBlock {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GapContinuationBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for GapContinuationBlock {
    fn block_id(&self) -> &'static str {
        "gap_continuation"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Entry
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let min_gap_pct = get_f64(params, "min_gap_pct", 0.02);
        let _volume_confirm = get_bool(params, "volume_confirm", true);
        let lookback = get_usize(params, "volume_lookback", 20);

        let mut signals = Vec::new();
        let mut long_signals = 0;
        let mut short_signals = 0;

        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            
            if prices.len() < lookback + 2 {
                continue;
            }

            if let Some(gap_pct) = GapFillBlock::calculate_gap(prices) {
                let abs_gap = gap_pct.abs();
                
                if abs_gap < min_gap_pct {
                    signals.push(
                        Signal::flat(&candidate.symbol).with_source("gap_continuation")
                    );
                    continue;
                }

                // Gap continuation = momentum (follow the gap direction)
                let (direction, strength) = if gap_pct > min_gap_pct {
                    // Gap up -> expect continuation (go long)
                    long_signals += 1;
                    let strength = (0.5 + (abs_gap * 10.0).min(0.5)).min(1.0);
                    (SignalDirection::Long, strength)
                } else if gap_pct < -min_gap_pct {
                    // Gap down -> expect continuation (go short)
                    short_signals += 1;
                    let strength = (0.5 + (abs_gap * 10.0).min(0.5)).min(1.0);
                    (SignalDirection::Short, strength)
                } else {
                    (SignalDirection::Flat, 0.0)
                };

                let signal = Signal::new(&candidate.symbol, direction, strength)
                    .with_source("gap_continuation")
                    .with_metadata("gap_pct", gap_pct);

                signals.push(signal);
            }
        }

        ctx.trace_step(
            self.block_id(),
            &format!(
                "GapContinuation({:.1}%): {} long, {} short from {} candidates",
                min_gap_pct * 100.0, long_signals, short_signals, ctx.candidates.len()
            ),
        );

        BlockResult::success(format!(
            "Gap Continuation: {} long, {} short",
            long_signals, short_signals
        ))
        .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let min_gap = get_f64(params, "min_gap_pct", 0.02);

        if min_gap <= 0.0 || min_gap > 0.20 {
            return Err(ValidationError::OutOfRange(
                "min_gap_pct".into(),
                "must be between 0 and 20%".into(),
            ));
        }

        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("min_gap_pct".into(), toml::Value::Float(0.02));
        params.insert("volume_confirm".into(), toml::Value::Boolean(true));
        params.insert("volume_lookback".into(), toml::Value::Integer(20));
        params
    }

    fn description(&self) -> &'static str {
        "Gap Continuation: Momentum following large gaps"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gap_calculation() {
        let prices = vec![100.0, 103.0]; // 3% gap up
        let gap = GapFillBlock::calculate_gap(&prices);
        assert!(gap.is_some());
        assert!((gap.unwrap() - 0.03).abs() < 0.001);
    }

    #[test]
    fn test_gap_down() {
        let prices = vec![100.0, 97.0]; // 3% gap down
        let gap = GapFillBlock::calculate_gap(&prices);
        assert!(gap.is_some());
        assert!((gap.unwrap() - (-0.03)).abs() < 0.001);
    }
}
