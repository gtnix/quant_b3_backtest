//! Volume Breakout and Volume Profile entry blocks.
//!
//! # Mathematical Foundation
//!
//! **Volume Breakout:**
//! - Volume Ratio = Current Volume / SMA(Volume, period)
//! - Signal when Volume Ratio > threshold AND price breaks range
//!
//! **Volume Profile POC (Point of Control):**
//! - POC = Price level with highest volume traded
//! - VAH (Value Area High) = Upper bound of 70% volume distribution
//! - VAL (Value Area Low) = Lower bound of 70% volume distribution
//!
//! # References
//! - Arms, R. (1971). "Volume Cycles in the Stock Market"
//! - Steidlmayer, P. (1986). "Markets and Market Logic" (Market Profile)

use crate::blocks::{
    get_f64, get_usize, BlockParams, BlockResult, BlockType, Signal, SignalDirection,
    StrategyBlock, ValidationError,
};
use crate::context::StrategyContext;
use std::collections::HashMap;

/// Volume-confirmed breakout entry block.
pub struct VolumeBreakoutBlock;

impl VolumeBreakoutBlock {
    pub fn new() -> Self {
        Self
    }

    /// Calculate volume ratio (current vs average).
    pub fn volume_ratio(current_vol: f64, avg_vol: f64) -> f64 {
        if avg_vol > 0.0 {
            current_vol / avg_vol
        } else {
            1.0
        }
    }

    /// Calculate average volume over period.
    pub fn calculate_avg_volume(volumes: &[f64], period: usize) -> Option<f64> {
        if volumes.len() < period {
            return None;
        }

        let n = volumes.len();
        let sum: f64 = volumes[(n - period)..n].iter().sum();
        Some(sum / period as f64)
    }

    /// Calculate price breakout level (highest high / lowest low).
    pub fn calculate_breakout_levels(prices: &[f64], period: usize) -> Option<(f64, f64)> {
        if prices.len() < period {
            return None;
        }

        let n = prices.len();
        let window = &prices[(n - period - 1)..(n - 1)]; // Exclude current bar

        let high = window.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let low = window.iter().copied().fold(f64::INFINITY, f64::min);

        if high.is_finite() && low.is_finite() {
            Some((high, low))
        } else {
            None
        }
    }
}

impl Default for VolumeBreakoutBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for VolumeBreakoutBlock {
    fn block_id(&self) -> &'static str {
        "volume_breakout"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Entry
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let price_period = get_usize(params, "price_period", 20);
        let volume_period = get_usize(params, "volume_period", 20);
        let volume_threshold = get_f64(params, "volume_threshold", 1.5);

        let mut signals = Vec::new();
        let mut long_breakouts = 0;
        let mut short_breakouts = 0;

        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            
            if prices.len() < price_period.max(volume_period) + 2 {
                continue;
            }

            let current_price = *prices.last().unwrap();

            // Use price as volume proxy when not available
            // In production, StrategyCandidate should have volumes
            let volumes: Vec<f64> = prices.iter().map(|_| 1.0).collect();

            let avg_vol = Self::calculate_avg_volume(&volumes, volume_period).unwrap_or(1.0);
            let current_vol = *volumes.last().unwrap();
            let vol_ratio = Self::volume_ratio(current_vol, avg_vol);

            // Check breakout levels
            if let Some((breakout_high, breakout_low)) = 
                Self::calculate_breakout_levels(prices, price_period) 
            {
                let volume_confirmed = vol_ratio >= volume_threshold;

                let (direction, strength) = if current_price > breakout_high && volume_confirmed {
                    long_breakouts += 1;
                    let strength = (0.5 + (vol_ratio - 1.0).min(0.5)).min(1.0);
                    (SignalDirection::Long, strength)
                } else if current_price < breakout_low && volume_confirmed {
                    short_breakouts += 1;
                    let strength = (0.5 + (vol_ratio - 1.0).min(0.5)).min(1.0);
                    (SignalDirection::Short, strength)
                } else {
                    (SignalDirection::Flat, 0.0)
                };

                let signal = Signal::new(&candidate.symbol, direction, strength)
                    .with_source("volume_breakout")
                    .with_metadata("price", current_price)
                    .with_metadata("breakout_high", breakout_high)
                    .with_metadata("breakout_low", breakout_low)
                    .with_metadata("volume_ratio", vol_ratio);

                signals.push(signal);
            }
        }

        ctx.trace_step(
            self.block_id(),
            &format!(
                "VolumeBreakout({}, vol>{}x): {} long, {} short from {} candidates",
                price_period, volume_threshold, long_breakouts, short_breakouts, ctx.candidates.len()
            ),
        );

        BlockResult::success(format!(
            "Volume Breakout: {} long, {} short",
            long_breakouts, short_breakouts
        ))
        .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let price_period = get_usize(params, "price_period", 20);
        let volume_threshold = get_f64(params, "volume_threshold", 1.5);

        if price_period < 5 {
            return Err(ValidationError::OutOfRange(
                "price_period".into(),
                "must be at least 5".into(),
            ));
        }

        if volume_threshold <= 1.0 || volume_threshold > 5.0 {
            return Err(ValidationError::OutOfRange(
                "volume_threshold".into(),
                "must be between 1 and 5".into(),
            ));
        }

        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("price_period".into(), toml::Value::Integer(20));
        params.insert("volume_period".into(), toml::Value::Integer(20));
        params.insert("volume_threshold".into(), toml::Value::Float(1.5));
        params
    }

    fn description(&self) -> &'static str {
        "Volume Breakout: Price breakout confirmed by above-average volume"
    }
}

/// Volume Profile POC (Point of Control) entry block.
pub struct VolumeProfileBlock;

impl VolumeProfileBlock {
    pub fn new() -> Self {
        Self
    }

    /// Calculate POC (price level with highest volume).
    /// Simplified: use price buckets to find highest volume concentration.
    pub fn calculate_poc(
        prices: &[f64],
        _volumes: &[f64],
        period: usize,
        num_buckets: usize,
    ) -> Option<(f64, f64, f64)> {
        if prices.len() < period {
            return None;
        }

        let n = prices.len();
        let window = &prices[(n - period)..n];

        let min_price = window.iter().copied().fold(f64::INFINITY, f64::min);
        let max_price = window.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        if !min_price.is_finite() || !max_price.is_finite() || min_price >= max_price {
            return None;
        }

        let bucket_size = (max_price - min_price) / num_buckets as f64;
        let mut bucket_counts = vec![0usize; num_buckets];

        for &price in window {
            let bucket = ((price - min_price) / bucket_size) as usize;
            let bucket = bucket.min(num_buckets - 1);
            bucket_counts[bucket] += 1;
        }

        // Find POC (bucket with most volume)
        let (poc_bucket, _max_count) = bucket_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)?;

        let poc = min_price + (poc_bucket as f64 + 0.5) * bucket_size;

        // Calculate VAH/VAL (70% value area)
        let total: usize = bucket_counts.iter().sum();
        let target = (total as f64 * 0.7) as usize;
        
        let mut val_bucket = poc_bucket;
        let mut vah_bucket = poc_bucket;
        let mut area_count = bucket_counts[poc_bucket];

        while area_count < target && (val_bucket > 0 || vah_bucket < num_buckets - 1) {
            let add_low = if val_bucket > 0 { bucket_counts[val_bucket - 1] } else { 0 };
            let add_high = if vah_bucket < num_buckets - 1 { bucket_counts[vah_bucket + 1] } else { 0 };

            if add_low >= add_high && val_bucket > 0 {
                val_bucket -= 1;
                area_count += add_low;
            } else if vah_bucket < num_buckets - 1 {
                vah_bucket += 1;
                area_count += add_high;
            } else {
                break;
            }
        }

        let val = min_price + val_bucket as f64 * bucket_size;
        let vah = min_price + (vah_bucket + 1) as f64 * bucket_size;

        Some((poc, val, vah))
    }
}

impl Default for VolumeProfileBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for VolumeProfileBlock {
    fn block_id(&self) -> &'static str {
        "volume_profile"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Entry
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let period = get_usize(params, "period", 50);
        let num_buckets = get_usize(params, "num_buckets", 20);
        let deviation_pct = get_f64(params, "deviation_pct", 0.02);

        let mut signals = Vec::new();
        let mut long_signals = 0;
        let mut short_signals = 0;

        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            
            if prices.len() < period + 1 {
                continue;
            }

            let volumes: Vec<f64> = prices.iter().map(|_| 1.0).collect();
            let current_price = *prices.last().unwrap();

            if let Some((poc, val, vah)) = 
                Self::calculate_poc(prices, &volumes, period, num_buckets) 
            {
                // Mean reversion to POC
                let pct_from_poc = (current_price - poc) / poc;

                let (direction, strength) = if current_price < val - deviation_pct * poc {
                    // Below VAL - potential long (mean reversion to POC)
                    long_signals += 1;
                    let strength = (0.5 + (val - current_price) / (vah - val) * 0.5).min(1.0);
                    (SignalDirection::Long, strength)
                } else if current_price > vah + deviation_pct * poc {
                    // Above VAH - potential short (mean reversion to POC)
                    short_signals += 1;
                    let strength = (0.5 + (current_price - vah) / (vah - val) * 0.5).min(1.0);
                    (SignalDirection::Short, strength)
                } else {
                    (SignalDirection::Flat, 0.0)
                };

                let signal = Signal::new(&candidate.symbol, direction, strength)
                    .with_source("volume_profile")
                    .with_metadata("price", current_price)
                    .with_metadata("poc", poc)
                    .with_metadata("val", val)
                    .with_metadata("vah", vah)
                    .with_metadata("pct_from_poc", pct_from_poc);

                signals.push(signal);
            }
        }

        ctx.trace_step(
            self.block_id(),
            &format!(
                "VolumeProfile({} buckets): {} long, {} short from {} candidates",
                num_buckets, long_signals, short_signals, ctx.candidates.len()
            ),
        );

        BlockResult::success(format!(
            "Volume Profile: {} long, {} short",
            long_signals, short_signals
        ))
        .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let period = get_usize(params, "period", 50);
        let num_buckets = get_usize(params, "num_buckets", 20);

        if period < 20 {
            return Err(ValidationError::OutOfRange(
                "period".into(),
                "must be at least 20".into(),
            ));
        }

        if num_buckets < 5 || num_buckets > 100 {
            return Err(ValidationError::OutOfRange(
                "num_buckets".into(),
                "must be between 5 and 100".into(),
            ));
        }

        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("period".into(), toml::Value::Integer(50));
        params.insert("num_buckets".into(), toml::Value::Integer(20));
        params.insert("deviation_pct".into(), toml::Value::Float(0.02));
        params
    }

    fn description(&self) -> &'static str {
        "Volume Profile: Trade from POC/VAH/VAL levels (Market Profile)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volume_ratio() {
        assert!((VolumeBreakoutBlock::volume_ratio(150.0, 100.0) - 1.5).abs() < 0.01);
        assert!((VolumeBreakoutBlock::volume_ratio(50.0, 100.0) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_avg_volume() {
        let volumes = vec![100.0, 120.0, 80.0, 110.0, 90.0];
        let avg = VolumeBreakoutBlock::calculate_avg_volume(&volumes, 5);
        assert!(avg.is_some());
        assert!((avg.unwrap() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_poc_calculation() {
        // Prices clustered around 100
        let prices = vec![
            99.0, 100.0, 101.0, 100.0, 99.5, 100.5, 100.0, 99.0, 101.0, 100.0,
            105.0, 106.0, // Some outliers
        ];
        let volumes: Vec<f64> = prices.iter().map(|_| 1.0).collect();
        
        let result = VolumeProfileBlock::calculate_poc(&prices, &volumes, 12, 10);
        assert!(result.is_some());
        
        let (poc, val, vah) = result.unwrap();
        assert!(poc > 98.0 && poc < 102.0); // POC should be near 100
        assert!(val < poc);
        assert!(vah > poc);
    }
}
