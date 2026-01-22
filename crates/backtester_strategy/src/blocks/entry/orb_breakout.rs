//! Opening Range Breakout (ORB) entry block.
//!
//! # Mathematical Foundation
//!
//! **Opening Range Breakout (Toby Crabel, 1990):**
//! - Opening Range: The price range during the first n minutes of trading
//! - OR_high = max(High) during opening period
//! - OR_low = min(Low) during opening period
//! - OR_range = OR_high - OR_low
//!
//! **Entry Conditions:**
//! - Long: Price > OR_high + (stretch_factor * OR_range)
//! - Short: Price < OR_low - (stretch_factor * OR_range)
//!
//! **Academic Validation:**
//! Opening range breakouts exploit the tendency for volatility clustering
//! and momentum following the opening session.
//!
//! # References
//! - Crabel, T. (1990). "Day Trading with Short Term Price Patterns and Opening Range Breakout"
//! - Taylor, N. (2005). "The Trading Methods of WD Gann"

use crate::blocks::{
    get_f64, get_usize, BlockParams, BlockResult, BlockType, Signal, SignalDirection,
    StrategyBlock, ValidationError,
};
use crate::context::StrategyContext;
use std::collections::HashMap;

/// Opening Range Breakout block.
///
/// Generates signals when price breaks above/below the opening range.
pub struct ORBBreakoutBlock;

impl ORBBreakoutBlock {
    pub fn new() -> Self {
        Self
    }

    /// Calculate opening range and breakout levels.
    ///
    /// # Arguments
    /// * `prices` - Price series (oldest to newest)
    /// * `or_bars` - Number of bars in opening range
    /// * `stretch_factor` - Multiplier for breakout threshold
    ///
    /// # Returns
    /// (or_high, or_low, long_entry, short_entry) or None
    pub fn calculate_or_levels(
        prices: &[f64],
        or_bars: usize,
        stretch_factor: f64,
    ) -> Option<(f64, f64, f64, f64)> {
        if prices.len() < or_bars + 1 {
            return None;
        }

        // Opening range from first n bars of the day
        // In daily data, we use first n historical bars as proxy
        let or_prices = &prices[..or_bars];
        
        let or_high = or_prices.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let or_low = or_prices.iter().copied().fold(f64::INFINITY, f64::min);
        
        if !or_high.is_finite() || !or_low.is_finite() {
            return None;
        }

        let or_range = or_high - or_low;
        let stretch = stretch_factor * or_range;

        let long_entry = or_high + stretch;
        let short_entry = or_low - stretch;

        Some((or_high, or_low, long_entry, short_entry))
    }
}

impl Default for ORBBreakoutBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for ORBBreakoutBlock {
    fn block_id(&self) -> &'static str {
        "orb_breakout"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Entry
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let or_bars = get_usize(params, "or_bars", 2);
        let stretch_factor = get_f64(params, "stretch_factor", 0.0);
        let min_range_pct = get_f64(params, "min_range_pct", 0.005);

        let mut signals = Vec::new();
        let mut long_breakouts = 0;
        let mut short_breakouts = 0;

        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            
            if prices.len() < or_bars + 5 {
                continue;
            }

            let current_price = *prices.last().unwrap();

            if let Some((or_high, or_low, long_entry, short_entry)) =
                Self::calculate_or_levels(prices, or_bars, stretch_factor)
            {
                // Filter by minimum range (avoid thin ranges)
                let range_pct = (or_high - or_low) / or_low;
                if range_pct < min_range_pct {
                    signals.push(
                        Signal::flat(&candidate.symbol).with_source("orb_breakout")
                    );
                    continue;
                }

                let (direction, strength) = if current_price > long_entry {
                    let breakout_strength = (current_price - long_entry) / (or_high - or_low);
                    long_breakouts += 1;
                    (SignalDirection::Long, (0.6 + breakout_strength.min(0.4)).min(1.0))
                } else if current_price < short_entry {
                    let breakout_strength = (short_entry - current_price) / (or_high - or_low);
                    short_breakouts += 1;
                    (SignalDirection::Short, (0.6 + breakout_strength.min(0.4)).min(1.0))
                } else {
                    (SignalDirection::Flat, 0.0)
                };

                let signal = Signal::new(&candidate.symbol, direction, strength)
                    .with_source("orb_breakout")
                    .with_metadata("price", current_price)
                    .with_metadata("or_high", or_high)
                    .with_metadata("or_low", or_low)
                    .with_metadata("long_entry", long_entry)
                    .with_metadata("short_entry", short_entry);

                signals.push(signal);
            }
        }

        ctx.trace_step(
            self.block_id(),
            &format!(
                "ORB({} bars, stretch={}): {} long, {} short from {} candidates",
                or_bars, stretch_factor, long_breakouts, short_breakouts, ctx.candidates.len()
            ),
        );

        BlockResult::success(format!(
            "ORB Breakout: {} long, {} short",
            long_breakouts, short_breakouts
        ))
        .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let or_bars = get_usize(params, "or_bars", 2);
        let stretch_factor = get_f64(params, "stretch_factor", 0.0);

        if or_bars < 1 || or_bars > 10 {
            return Err(ValidationError::OutOfRange(
                "or_bars".into(),
                "must be between 1 and 10".into(),
            ));
        }

        if stretch_factor < 0.0 || stretch_factor > 2.0 {
            return Err(ValidationError::OutOfRange(
                "stretch_factor".into(),
                "must be between 0 and 2".into(),
            ));
        }

        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("or_bars".into(), toml::Value::Integer(2));
        params.insert("stretch_factor".into(), toml::Value::Float(0.0));
        params.insert("min_range_pct".into(), toml::Value::Float(0.005));
        params
    }

    fn description(&self) -> &'static str {
        "Opening Range Breakout: Trade breakouts from the opening price range"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_or_calculation() {
        let prices = vec![100.0, 102.0, 101.0, 103.0, 105.0, 104.0];
        
        let result = ORBBreakoutBlock::calculate_or_levels(&prices, 3, 0.0);
        assert!(result.is_some());
        
        let (or_high, or_low, long_entry, short_entry) = result.unwrap();
        assert!((or_high - 102.0).abs() < 0.01);
        assert!((or_low - 100.0).abs() < 0.01);
        assert!((long_entry - 102.0).abs() < 0.01); // No stretch
        assert!((short_entry - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_or_with_stretch() {
        let prices = vec![100.0, 102.0, 101.0, 103.0, 105.0];
        
        let result = ORBBreakoutBlock::calculate_or_levels(&prices, 3, 0.5);
        assert!(result.is_some());
        
        let (or_high, or_low, long_entry, short_entry) = result.unwrap();
        // Range = 102 - 100 = 2, stretch = 1
        assert!((long_entry - 103.0).abs() < 0.01);
        assert!((short_entry - 99.0).abs() < 0.01);
    }
}
