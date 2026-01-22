//! VIX Mean Reversion entry block.
//!
//! # Mathematical Foundation
//!
//! **VIX Mean Reversion:**
//! The VIX (CBOE Volatility Index) shows strong mean-reverting properties.
//! Extreme VIX spikes tend to revert, creating trading opportunities.
//!
//! **Z-Score Signal:**
//! - Long (buy stocks): VIX Z-score > +threshold (fear extreme = buy opportunity)
//! - Short (sell stocks): VIX Z-score < -threshold (complacency = sell opportunity)
//!
//! **Regime Filter:**
//! - VIX < 15: Low volatility regime (risk-on)
//! - VIX 15-25: Normal regime
//! - VIX 25-35: Elevated fear
//! - VIX > 35: Crisis/extreme fear
//!
//! # References
//! - Whaley, R.E. (2000). "The Investor Fear Gauge" - VIX Original Paper
//! - Simon, D.P. & Wiggins, R.A. (2001). "S&P Futures Returns and VIX"

use crate::blocks::{
    get_f64, get_usize, BlockParams, BlockResult, BlockType, Signal, SignalDirection,
    StrategyBlock, ValidationError,
};
use crate::context::StrategyContext;
use std::collections::HashMap;

/// VIX Mean Reversion entry block.
/// 
/// Uses volatility as contrarian indicator: extreme fear = buy, complacency = sell.
pub struct VIXReversionBlock;

impl VIXReversionBlock {
    pub fn new() -> Self {
        Self
    }

    /// Calculate realized volatility as VIX proxy.
    /// RV = sqrt(252) × std(daily_returns)
    pub fn calculate_realized_vol(prices: &[f64], period: usize) -> Option<f64> {
        if prices.len() < period + 1 {
            return None;
        }

        let n = prices.len();
        let mut returns = Vec::with_capacity(period);

        for i in (n - period)..n {
            let ret = (prices[i] / prices[i - 1]).ln();
            returns.push(ret);
        }

        let mean: f64 = returns.iter().sum::<f64>() / period as f64;
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (period - 1) as f64;
        let daily_std = variance.sqrt();

        // Annualize (252 trading days)
        let annual_vol = daily_std * (252.0_f64).sqrt();

        Some(annual_vol * 100.0) // Return as percentage (like VIX)
    }

    /// Calculate Z-score of current volatility vs historical.
    pub fn calculate_vol_zscore(prices: &[f64], vol_period: usize, lookback: usize) -> Option<f64> {
        if prices.len() < vol_period + lookback + 1 {
            return None;
        }

        let n = prices.len();
        let mut vol_history = Vec::with_capacity(lookback);

        for i in 0..lookback {
            let end_idx = n - i;
            if end_idx <= vol_period + 1 {
                break;
            }
            
            let slice = &prices[..(end_idx)];
            if let Some(vol) = Self::calculate_realized_vol(slice, vol_period) {
                vol_history.push(vol);
            }
        }

        if vol_history.len() < 10 {
            return None;
        }

        let current_vol = vol_history[0];
        let mean_vol: f64 = vol_history.iter().sum::<f64>() / vol_history.len() as f64;
        let var_vol: f64 = vol_history.iter().map(|v| (v - mean_vol).powi(2)).sum::<f64>() 
            / (vol_history.len() - 1) as f64;
        let std_vol = var_vol.sqrt();

        if std_vol > 0.0 {
            Some((current_vol - mean_vol) / std_vol)
        } else {
            None
        }
    }

    /// Determine volatility regime.
    pub fn vol_regime(vol: f64) -> &'static str {
        match vol {
            v if v < 15.0 => "low",
            v if v < 25.0 => "normal",
            v if v < 35.0 => "elevated",
            _ => "crisis",
        }
    }
}

impl Default for VIXReversionBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for VIXReversionBlock {
    fn block_id(&self) -> &'static str {
        "vix_reversion"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Entry
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let vol_period = get_usize(params, "vol_period", 20);
        let lookback = get_usize(params, "lookback", 126);
        let threshold = get_f64(params, "threshold", 1.5);

        let mut signals = Vec::new();
        let mut long_signals = 0;
        let mut short_signals = 0;

        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            
            if prices.len() < vol_period + lookback + 10 {
                continue;
            }

            let current_price = *prices.last().unwrap();

            if let Some(vol_zscore) = Self::calculate_vol_zscore(prices, vol_period, lookback) {
                let current_vol = Self::calculate_realized_vol(prices, vol_period).unwrap_or(20.0);
                let _regime = Self::vol_regime(current_vol);

                // Contrarian: high fear = buy, low fear = sell
                let (direction, strength) = if vol_zscore > threshold {
                    // High volatility (fear) = contrarian long (stocks cheap)
                    long_signals += 1;
                    let strength = (0.5 + (vol_zscore - threshold).min(1.5) / 3.0).min(1.0);
                    (SignalDirection::Long, strength)
                } else if vol_zscore < -threshold {
                    // Low volatility (complacency) = contrarian short (stocks expensive)
                    short_signals += 1;
                    let strength = (0.5 + (-vol_zscore - threshold).min(1.5) / 3.0).min(1.0);
                    (SignalDirection::Short, strength)
                } else {
                    (SignalDirection::Flat, 0.0)
                };

                let signal = Signal::new(&candidate.symbol, direction, strength)
                    .with_source("vix_reversion")
                    .with_metadata("price", current_price)
                    .with_metadata("vol", current_vol)
                    .with_metadata("vol_zscore", vol_zscore);

                signals.push(signal);
            }
        }

        ctx.trace_step(
            self.block_id(),
            &format!(
                "VIXReversion(z>{}): {} long, {} short from {} candidates",
                threshold, long_signals, short_signals, ctx.candidates.len()
            ),
        );

        BlockResult::success(format!(
            "VIX Reversion: {} long, {} short",
            long_signals, short_signals
        ))
        .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let vol_period = get_usize(params, "vol_period", 20);
        let threshold = get_f64(params, "threshold", 1.5);

        if vol_period < 5 || vol_period > 60 {
            return Err(ValidationError::OutOfRange(
                "vol_period".into(),
                "must be between 5 and 60".into(),
            ));
        }

        if threshold <= 0.0 || threshold > 4.0 {
            return Err(ValidationError::OutOfRange(
                "threshold".into(),
                "must be between 0 and 4".into(),
            ));
        }

        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("vol_period".into(), toml::Value::Integer(20));
        params.insert("lookback".into(), toml::Value::Integer(126));
        params.insert("threshold".into(), toml::Value::Float(1.5));
        params
    }

    fn description(&self) -> &'static str {
        "VIX Reversion: Contrarian signals on extreme volatility (fear/greed)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_realized_vol() {
        // Create trending prices with some volatility
        let mut prices = Vec::new();
        for i in 0..30 {
            prices.push(100.0 + (i as f64 * 0.5) + (i as f64 * 0.1).sin() * 2.0);
        }
        
        let vol = VIXReversionBlock::calculate_realized_vol(&prices, 20);
        assert!(vol.is_some());
        assert!(vol.unwrap() > 0.0);
    }

    #[test]
    fn test_vol_regime() {
        assert_eq!(VIXReversionBlock::vol_regime(12.0), "low");
        assert_eq!(VIXReversionBlock::vol_regime(20.0), "normal");
        assert_eq!(VIXReversionBlock::vol_regime(30.0), "elevated");
        assert_eq!(VIXReversionBlock::vol_regime(50.0), "crisis");
    }

    #[test]
    fn test_vol_zscore() {
        // Need longer price history for z-score
        let mut prices = Vec::new();
        for i in 0..200 {
            // Varying volatility over time
            let vol_mult = 1.0 + 0.5 * ((i as f64 / 50.0).sin());
            prices.push(100.0 + (i as f64 * 0.1) + vol_mult * (i as f64).sin());
        }
        
        let zscore = VIXReversionBlock::calculate_vol_zscore(&prices, 20, 100);
        assert!(zscore.is_some());
    }
}
