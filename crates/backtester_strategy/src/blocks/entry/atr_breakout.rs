//! ATR (Average True Range) Breakout entry block.
//!
//! # Mathematical Foundation
//!
//! **True Range (Wilder, 1978):**
//! TR = max(High - Low, |High - Close_prev|, |Low - Close_prev|)
//!
//! **Average True Range:**
//! ATR = EMA(TR, period) or SMA(TR, period)
//!
//! **Breakout Signal:**
//! - Long: Close > Close_prev + (multiplier × ATR)
//! - Short: Close < Close_prev - (multiplier × ATR)
//!
//! The ATR breakout filters noise by requiring price movement
//! to exceed a volatility-adjusted threshold.
//!
//! # References
//! - Wilder, J.W. (1978). "New Concepts in Technical Trading Systems"
//! - Kestner, L. (2003). "Quantitative Trading Strategies"

use crate::blocks::{
    get_f64, get_usize, BlockParams, BlockResult, BlockType, Signal, SignalDirection,
    StrategyBlock, ValidationError,
};
use crate::context::StrategyContext;
use std::collections::HashMap;

/// ATR Breakout entry block.
pub struct ATRBreakoutBlock;

impl ATRBreakoutBlock {
    pub fn new() -> Self {
        Self
    }

    /// Calculate True Range for a bar.
    /// TR = max(H-L, |H-C_prev|, |L-C_prev|)
    pub fn true_range(high: f64, low: f64, prev_close: f64) -> f64 {
        let hl = high - low;
        let hc = (high - prev_close).abs();
        let lc = (low - prev_close).abs();
        hl.max(hc).max(lc)
    }

    /// Calculate ATR using simple moving average of True Range.
    /// Using close prices as proxy for H/L when not available.
    pub fn calculate_atr(prices: &[f64], period: usize) -> Option<f64> {
        if prices.len() < period + 1 {
            return None;
        }

        let n = prices.len();
        let mut tr_sum = 0.0;

        // Calculate TR for each bar (using close-to-close as proxy)
        for i in (n - period)..n {
            let current = prices[i];
            let prev = prices[i - 1];
            // Simplified TR when only close is available
            let tr = (current - prev).abs();
            tr_sum += tr;
        }

        Some(tr_sum / period as f64)
    }

    /// Calculate ATR using full OHLC data.
    pub fn calculate_atr_ohlc(
        highs: &[f64],
        lows: &[f64],
        closes: &[f64],
        period: usize,
    ) -> Option<f64> {
        let n = closes.len();
        if n < period + 1 || highs.len() != n || lows.len() != n {
            return None;
        }

        let mut tr_sum = 0.0;

        for i in (n - period)..n {
            let tr = Self::true_range(highs[i], lows[i], closes[i - 1]);
            tr_sum += tr;
        }

        Some(tr_sum / period as f64)
    }
}

impl Default for ATRBreakoutBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for ATRBreakoutBlock {
    fn block_id(&self) -> &'static str {
        "atr_breakout"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Entry
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let period = get_usize(params, "period", 14);
        let multiplier = get_f64(params, "multiplier", 2.0);

        let mut signals = Vec::new();
        let mut long_breakouts = 0;
        let mut short_breakouts = 0;

        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            
            if prices.len() < period + 2 {
                continue;
            }

            let n = prices.len();
            let current_price = prices[n - 1];
            let prev_close = prices[n - 2];

            if let Some(atr) = Self::calculate_atr(prices, period) {
                let upper_band = prev_close + multiplier * atr;
                let lower_band = prev_close - multiplier * atr;

                let (direction, strength) = if current_price > upper_band {
                    // Breakout above ATR band
                    let breakout_atr = (current_price - upper_band) / atr;
                    long_breakouts += 1;
                    (SignalDirection::Long, (0.6 + breakout_atr.min(0.4)).min(1.0))
                } else if current_price < lower_band {
                    // Breakout below ATR band
                    let breakout_atr = (lower_band - current_price) / atr;
                    short_breakouts += 1;
                    (SignalDirection::Short, (0.6 + breakout_atr.min(0.4)).min(1.0))
                } else {
                    (SignalDirection::Flat, 0.0)
                };

                let signal = Signal::new(&candidate.symbol, direction, strength)
                    .with_source("atr_breakout")
                    .with_metadata("price", current_price)
                    .with_metadata("atr", atr)
                    .with_metadata("upper_band", upper_band)
                    .with_metadata("lower_band", lower_band)
                    .with_metadata("multiplier", multiplier);

                signals.push(signal);
            }
        }

        ctx.trace_step(
            self.block_id(),
            &format!(
                "ATR({}, {}x): {} long, {} short from {} candidates",
                period, multiplier, long_breakouts, short_breakouts, ctx.candidates.len()
            ),
        );

        BlockResult::success(format!(
            "ATR Breakout: {} long, {} short",
            long_breakouts, short_breakouts
        ))
        .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let period = get_usize(params, "period", 14);
        let multiplier = get_f64(params, "multiplier", 2.0);

        if period < 5 || period > 100 {
            return Err(ValidationError::OutOfRange(
                "period".into(),
                "must be between 5 and 100".into(),
            ));
        }

        if multiplier <= 0.0 || multiplier > 10.0 {
            return Err(ValidationError::OutOfRange(
                "multiplier".into(),
                "must be between 0 and 10".into(),
            ));
        }

        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("period".into(), toml::Value::Integer(14));
        params.insert("multiplier".into(), toml::Value::Float(2.0));
        params
    }

    fn description(&self) -> &'static str {
        "ATR Breakout: Volatility-adjusted breakout signals (Wilder)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_true_range() {
        // H=105, L=95, C_prev=100
        let tr = ATRBreakoutBlock::true_range(105.0, 95.0, 100.0);
        assert!((tr - 10.0).abs() < 0.01); // H-L = 10
    }

    #[test]
    fn test_true_range_gap_up() {
        // Gap up: H=110, L=105, C_prev=100
        let tr = ATRBreakoutBlock::true_range(110.0, 105.0, 100.0);
        assert!((tr - 10.0).abs() < 0.01); // |H-C_prev| = 10
    }

    #[test]
    fn test_atr_calculation() {
        let prices = vec![100.0, 101.0, 99.0, 102.0, 100.0, 103.0, 101.0];
        let atr = ATRBreakoutBlock::calculate_atr(&prices, 5);
        assert!(atr.is_some());
        assert!(atr.unwrap() > 0.0);
    }
}
