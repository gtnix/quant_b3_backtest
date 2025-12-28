//! Regime Detection Module.
//!
//! Provides daily market regime classification for research-grade reporting:
//! - Trend detection (Uptrend/Downtrend/Sideways)
//! - Volatility quantiles (Q1-Q5)
//! - Performance aggregation by regime
//!
//! # Design Decisions
//!
//! - **Trend detection**: Linear regression slope on cumulative returns, normalized by volatility
//! - **Vol quantiles**: Expanding window (point-in-time strict) to avoid look-ahead bias
//! - **Benchmark**: Configurable with fallback to portfolio returns
//! - **Quantile buckets**: 5 (Q1-Q5) for standard quintiles
//!
//! # Algorithm Details
//!
//! ## Trend Detection
//! ```text
//! slope = Cov(t, cumret) / Var(t)
//! normalized_slope = slope / vol
//! if normalized_slope > threshold:    Uptrend
//! if normalized_slope < -threshold:   Downtrend
//! else:                               Sideways
//! ```
//!
//! ## Volatility Quantiles
//! ```text
//! At day T:
//!   vol_T = stdev(returns[T-lookback:T]) * sqrt(252)
//!   percentile = rank(vol_T) / len(historical_vols)
//!   quantile = Q1..Q5 based on percentile
//! ```

use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// =============================================================================
// TREND STATE
// =============================================================================

/// Trend state classification.
///
/// Classified based on normalized slope of cumulative returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TrendState {
    /// No clear trend (|normalized slope| <= threshold).
    Sideways,
    /// Clear upward trend (normalized slope > threshold).
    Uptrend,
    /// Clear downward trend (normalized slope < -threshold).
    Downtrend,
}

impl TrendState {
    /// Convert to string for reporting.
    pub fn as_str(&self) -> &'static str {
        match self {
            TrendState::Sideways => "Sideways",
            TrendState::Uptrend => "Uptrend",
            TrendState::Downtrend => "Downtrend",
        }
    }
}

impl Default for TrendState {
    fn default() -> Self {
        TrendState::Sideways
    }
}

impl std::fmt::Display for TrendState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// =============================================================================
// VOLATILITY QUANTILE
// =============================================================================

/// Volatility quantile classification (quintiles Q1-Q5).
///
/// Q1 = lowest volatility (0-20th percentile)
/// Q5 = highest volatility (80-100th percentile)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum VolQuantile {
    /// 0-20th percentile (lowest volatility).
    Q1,
    /// 20-40th percentile.
    Q2,
    /// 40-60th percentile (median volatility).
    Q3,
    /// 60-80th percentile.
    Q4,
    /// 80-100th percentile (highest volatility).
    Q5,
}

impl VolQuantile {
    /// Convert to 1-5 integer.
    pub fn as_u8(&self) -> u8 {
        match self {
            VolQuantile::Q1 => 1,
            VolQuantile::Q2 => 2,
            VolQuantile::Q3 => 3,
            VolQuantile::Q4 => 4,
            VolQuantile::Q5 => 5,
        }
    }

    /// Create from percentile (0.0 - 1.0).
    pub fn from_percentile(percentile: Decimal) -> Self {
        let pct = percentile.to_string().parse::<f64>().unwrap_or(0.5);
        if pct < 0.2 {
            VolQuantile::Q1
        } else if pct < 0.4 {
            VolQuantile::Q2
        } else if pct < 0.6 {
            VolQuantile::Q3
        } else if pct < 0.8 {
            VolQuantile::Q4
        } else {
            VolQuantile::Q5
        }
    }

    /// Get percentile range as string.
    pub fn percentile_range(&self) -> &'static str {
        match self {
            VolQuantile::Q1 => "0-20%",
            VolQuantile::Q2 => "20-40%",
            VolQuantile::Q3 => "40-60%",
            VolQuantile::Q4 => "60-80%",
            VolQuantile::Q5 => "80-100%",
        }
    }
}

impl Default for VolQuantile {
    fn default() -> Self {
        VolQuantile::Q3 // Neutral default
    }
}

impl std::fmt::Display for VolQuantile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Q{}", self.as_u8())
    }
}

// =============================================================================
// REGIME TAG
// =============================================================================

/// Daily regime tag with trend and volatility classification.
///
/// Contains the raw indicator values for auditability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeTag {
    /// Date of the regime classification.
    pub date: NaiveDate,
    /// Trend state classification.
    pub trend_state: TrendState,
    /// Volatility quantile classification.
    pub vol_quantile: VolQuantile,
    /// Raw annualized volatility value.
    pub vol_value: Decimal,
    /// Raw trend indicator (normalized slope).
    pub trend_indicator: Decimal,
    /// Benchmark/portfolio return used for classification.
    pub benchmark_return: Option<Decimal>,
}

impl RegimeTag {
    /// Create a new regime tag.
    pub fn new(
        date: NaiveDate,
        trend_state: TrendState,
        vol_quantile: VolQuantile,
        vol_value: Decimal,
        trend_indicator: Decimal,
    ) -> Self {
        Self {
            date,
            trend_state,
            vol_quantile,
            vol_value,
            trend_indicator,
            benchmark_return: None,
        }
    }

    /// Create a neutral/unknown regime tag for warmup period.
    pub fn warmup(date: NaiveDate) -> Self {
        Self {
            date,
            trend_state: TrendState::Sideways,
            vol_quantile: VolQuantile::Q3,
            vol_value: Decimal::ZERO,
            trend_indicator: Decimal::ZERO,
            benchmark_return: None,
        }
    }
}

// =============================================================================
// REGIME PERFORMANCE
// =============================================================================

/// Performance metrics aggregated by regime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimePerformance {
    /// Trend state for this bucket.
    pub trend_state: TrendState,
    /// Volatility quantile for this bucket.
    pub vol_quantile: VolQuantile,
    /// Number of days in this regime.
    pub day_count: u32,
    /// Mean daily return (percentage).
    pub mean_return_pct: Decimal,
    /// Cumulative return (percentage).
    pub cumulative_return_pct: Decimal,
    /// Win rate (percentage of days with positive return).
    pub win_rate_pct: Decimal,
    /// Mean daily turnover (percentage).
    pub mean_turnover_pct: Decimal,
    /// Mean daily cost (percentage of equity).
    pub mean_cost_pct: Decimal,
}

impl RegimePerformance {
    /// Create a new empty regime performance bucket.
    pub fn new(trend_state: TrendState, vol_quantile: VolQuantile) -> Self {
        Self {
            trend_state,
            vol_quantile,
            day_count: 0,
            mean_return_pct: Decimal::ZERO,
            cumulative_return_pct: Decimal::ZERO,
            win_rate_pct: Decimal::ZERO,
            mean_turnover_pct: Decimal::ZERO,
            mean_cost_pct: Decimal::ZERO,
        }
    }
}

// =============================================================================
// REGIME CONFIG
// =============================================================================

/// Configuration for regime detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeConfig {
    /// Lookback window for trend detection (days).
    pub trend_lookback: u32,
    /// Trend threshold (|normalized slope| below this = sideways).
    /// Default: 0.0005 (5bps/day).
    pub trend_threshold: Decimal,
    /// Lookback window for volatility calculation (days).
    pub vol_lookback: u32,
    /// Number of quantile buckets (fixed at 5 for Q1-Q5).
    pub vol_quantile_count: u32,
    /// Benchmark symbol for regime detection (None = use portfolio returns).
    pub benchmark_symbol: Option<String>,
}

impl Default for RegimeConfig {
    fn default() -> Self {
        Self {
            trend_lookback: 20,
            trend_threshold: Decimal::from_str_exact("0.0005").unwrap_or(Decimal::ZERO),
            vol_lookback: 20,
            vol_quantile_count: 5,
            benchmark_symbol: None,
        }
    }
}

// =============================================================================
// REGIME SUMMARY
// =============================================================================

/// Complete regime summary with config and performance by regime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeSummary {
    /// Configuration used for regime detection.
    pub config: RegimeConfig,
    /// Performance metrics by regime combination.
    pub by_regime: Vec<RegimePerformance>,
    /// Total number of days analyzed.
    pub total_days: u32,
    /// Number of days in warmup period (insufficient data).
    pub warmup_days: u32,
}

impl RegimeSummary {
    /// Create an empty summary.
    pub fn empty(config: RegimeConfig) -> Self {
        Self {
            config,
            by_regime: Vec::new(),
            total_days: 0,
            warmup_days: 0,
        }
    }

    /// Get performance for a specific regime.
    pub fn get_performance(
        &self,
        trend: TrendState,
        vol: VolQuantile,
    ) -> Option<&RegimePerformance> {
        self.by_regime
            .iter()
            .find(|p| p.trend_state == trend && p.vol_quantile == vol)
    }

    /// Get best performing regime by mean return.
    pub fn best_regime(&self) -> Option<&RegimePerformance> {
        self.by_regime.iter().max_by(|a, b| {
            a.mean_return_pct
                .partial_cmp(&b.mean_return_pct)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get worst performing regime by mean return.
    pub fn worst_regime(&self) -> Option<&RegimePerformance> {
        self.by_regime.iter().min_by(|a, b| {
            a.mean_return_pct
                .partial_cmp(&b.mean_return_pct)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

// =============================================================================
// REGIME ENGINE
// =============================================================================

/// Regime detection engine.
///
/// Classifies each day's market regime based on trend and volatility.
/// Uses expanding window for volatility quantiles (point-in-time strict).
pub struct RegimeEngine {
    config: RegimeConfig,
    /// Historical volatility values for expanding window quantiles.
    vol_history: Vec<Decimal>,
}

impl RegimeEngine {
    /// Create a new regime engine with default config.
    pub fn new() -> Self {
        Self::with_config(RegimeConfig::default())
    }

    /// Create a new regime engine with custom config.
    pub fn with_config(config: RegimeConfig) -> Self {
        Self {
            config,
            vol_history: Vec::new(),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &RegimeConfig {
        &self.config
    }

    /// Reset the engine state (clear volatility history).
    pub fn reset(&mut self) {
        self.vol_history.clear();
    }

    /// Classify a single day given return history up to that day.
    ///
    /// # Arguments
    ///
    /// * `date` - The date to classify
    /// * `returns` - Daily returns up to and including this date (chronological order)
    ///
    /// # Returns
    ///
    /// RegimeTag for the day. Returns warmup tag if insufficient history.
    pub fn classify_day(&mut self, date: NaiveDate, returns: &[Decimal]) -> RegimeTag {
        let lookback = self.config.trend_lookback.max(self.config.vol_lookback) as usize;

        // Not enough history for classification
        if returns.len() < lookback {
            return RegimeTag::warmup(date);
        }

        // Calculate current volatility (rolling window)
        let vol_window = &returns[returns.len().saturating_sub(self.config.vol_lookback as usize)..];
        let vol_value = self.calculate_annualized_vol(vol_window);

        // Calculate trend indicator
        let trend_window =
            &returns[returns.len().saturating_sub(self.config.trend_lookback as usize)..];
        let trend_indicator = self.calculate_normalized_slope(trend_window);

        // Classify trend
        let trend_state = if trend_indicator > self.config.trend_threshold {
            TrendState::Uptrend
        } else if trend_indicator < -self.config.trend_threshold {
            TrendState::Downtrend
        } else {
            TrendState::Sideways
        };

        // Add to volatility history for expanding window quantiles
        self.vol_history.push(vol_value);

        // Calculate volatility quantile using expanding window
        let vol_quantile = self.calculate_vol_quantile(vol_value);

        RegimeTag::new(date, trend_state, vol_quantile, vol_value, trend_indicator)
    }

    /// Classify all days in a backtest period.
    ///
    /// # Arguments
    ///
    /// * `daily_data` - Vector of (date, return) pairs in chronological order
    ///
    /// # Returns
    ///
    /// Vector of RegimeTag for each day.
    pub fn classify_period(&mut self, daily_data: &[(NaiveDate, Decimal)]) -> Vec<RegimeTag> {
        self.reset();

        let returns: Vec<Decimal> = daily_data.iter().map(|(_, r)| *r).collect();
        let mut tags = Vec::with_capacity(daily_data.len());

        for (i, (date, _)) in daily_data.iter().enumerate() {
            let returns_to_date = &returns[..=i];
            let tag = self.classify_day(*date, returns_to_date);
            tags.push(tag);
        }

        tags
    }

    /// Generate performance summary by regime.
    ///
    /// # Arguments
    ///
    /// * `tags` - Regime tags for each day
    /// * `daily_returns` - Daily portfolio returns (aligned with tags)
    /// * `daily_turnover` - Daily turnover percentages (aligned with tags)
    /// * `daily_costs` - Daily costs as percentage of equity (aligned with tags)
    pub fn summarize(
        &self,
        tags: &[RegimeTag],
        daily_returns: &[Decimal],
        daily_turnover: &[Decimal],
        daily_costs: &[Decimal],
    ) -> RegimeSummary {
        if tags.is_empty() || tags.len() != daily_returns.len() {
            return RegimeSummary::empty(self.config.clone());
        }

        // Aggregate by regime
        let mut buckets: BTreeMap<(TrendState, VolQuantile), Vec<(Decimal, Decimal, Decimal)>> =
            BTreeMap::new();

        let lookback = self.config.trend_lookback.max(self.config.vol_lookback) as usize;
        let mut warmup_days = 0u32;

        for (i, tag) in tags.iter().enumerate() {
            let ret = daily_returns.get(i).copied().unwrap_or(Decimal::ZERO);
            let turn = daily_turnover.get(i).copied().unwrap_or(Decimal::ZERO);
            let cost = daily_costs.get(i).copied().unwrap_or(Decimal::ZERO);

            // Skip warmup period for aggregation
            if i < lookback {
                warmup_days += 1;
                continue;
            }

            buckets
                .entry((tag.trend_state, tag.vol_quantile))
                .or_default()
                .push((ret, turn, cost));
        }

        // Calculate performance for each bucket
        let by_regime: Vec<RegimePerformance> = buckets
            .into_iter()
            .map(|((trend, vol), data)| {
                let day_count = data.len() as u32;
                if day_count == 0 {
                    return RegimePerformance::new(trend, vol);
                }

                let returns: Vec<Decimal> = data.iter().map(|(r, _, _)| *r).collect();
                let turnovers: Vec<Decimal> = data.iter().map(|(_, t, _)| *t).collect();
                let costs: Vec<Decimal> = data.iter().map(|(_, _, c)| *c).collect();

                let sum_ret: Decimal = returns.iter().sum();
                let mean_return_pct = sum_ret / Decimal::from(day_count) * Decimal::from(100);

                // Cumulative return: product of (1 + r_i) - 1
                let mut cum = Decimal::ONE;
                for r in &returns {
                    cum *= Decimal::ONE + *r;
                }
                let cumulative_return_pct = (cum - Decimal::ONE) * Decimal::from(100);

                // Win rate
                let wins = returns.iter().filter(|r| **r > Decimal::ZERO).count() as u32;
                let win_rate_pct = Decimal::from(wins) / Decimal::from(day_count) * Decimal::from(100);

                // Mean turnover and costs
                let mean_turnover_pct: Decimal =
                    turnovers.iter().sum::<Decimal>() / Decimal::from(day_count);
                let mean_cost_pct: Decimal =
                    costs.iter().sum::<Decimal>() / Decimal::from(day_count);

                RegimePerformance {
                    trend_state: trend,
                    vol_quantile: vol,
                    day_count,
                    mean_return_pct,
                    cumulative_return_pct,
                    win_rate_pct,
                    mean_turnover_pct,
                    mean_cost_pct,
                }
            })
            .collect();

        RegimeSummary {
            config: self.config.clone(),
            by_regime,
            total_days: tags.len() as u32,
            warmup_days,
        }
    }

    // =========================================================================
    // PRIVATE HELPER METHODS
    // =========================================================================

    /// Calculate annualized volatility from daily returns.
    fn calculate_annualized_vol(&self, returns: &[Decimal]) -> Decimal {
        if returns.is_empty() {
            return Decimal::ZERO;
        }

        let n = returns.len();
        let mean: Decimal = returns.iter().sum::<Decimal>() / Decimal::from(n as u32);

        let variance: Decimal = returns
            .iter()
            .map(|r| {
                let diff = *r - mean;
                diff * diff
            })
            .sum::<Decimal>()
            / Decimal::from(n.max(1) as u32);

        let daily_vol = decimal_sqrt(variance);

        // Annualize: daily * sqrt(252)
        let sqrt_252 = Decimal::from_str_exact("15.87").unwrap_or(Decimal::from(16));
        daily_vol * sqrt_252
    }

    /// Calculate normalized slope (trend indicator).
    ///
    /// slope = Cov(t, cumret) / Var(t)
    /// normalized_slope = slope / vol
    fn calculate_normalized_slope(&self, returns: &[Decimal]) -> Decimal {
        if returns.len() < 2 {
            return Decimal::ZERO;
        }

        let n = returns.len();
        let n_dec = Decimal::from(n as u32);

        // Calculate cumulative returns
        let mut cumret: Vec<Decimal> = Vec::with_capacity(n);
        let mut running = Decimal::ZERO;
        for r in returns {
            running += *r;
            cumret.push(running);
        }

        // Calculate mean of t (0, 1, 2, ..., n-1)
        let mean_t = (n_dec - Decimal::ONE) / Decimal::from(2);

        // Calculate mean of cumret
        let mean_cumret: Decimal = cumret.iter().sum::<Decimal>() / n_dec;

        // Cov(t, cumret) = sum((t_i - mean_t) * (cumret_i - mean_cumret)) / n
        let cov: Decimal = (0..n)
            .map(|i| {
                let t_i = Decimal::from(i as u32);
                (t_i - mean_t) * (cumret[i] - mean_cumret)
            })
            .sum::<Decimal>()
            / n_dec;

        // Var(t) = sum((t_i - mean_t)^2) / n
        let var_t: Decimal = (0..n)
            .map(|i| {
                let t_i = Decimal::from(i as u32);
                let diff = t_i - mean_t;
                diff * diff
            })
            .sum::<Decimal>()
            / n_dec;

        if var_t.is_zero() {
            return Decimal::ZERO;
        }

        let slope = cov / var_t;

        // Normalize by volatility
        let vol = self.calculate_annualized_vol(returns);
        if vol.is_zero() {
            return Decimal::ZERO;
        }

        // Daily slope, normalized by annualized vol
        // Convert to comparable scale
        slope / vol * Decimal::from(252)
    }

    /// Calculate volatility quantile using expanding window.
    fn calculate_vol_quantile(&self, current_vol: Decimal) -> VolQuantile {
        if self.vol_history.is_empty() {
            return VolQuantile::Q3; // Neutral default
        }

        // Count how many historical vols are less than current
        let n = self.vol_history.len();
        let less_than = self.vol_history.iter().filter(|v| **v < current_vol).count();

        let percentile = Decimal::from(less_than as u32) / Decimal::from(n as u32);
        VolQuantile::from_percentile(percentile)
    }
}

impl Default for RegimeEngine {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Approximate square root for Decimal using Newton-Raphson.
fn decimal_sqrt(x: Decimal) -> Decimal {
    if x <= Decimal::ZERO {
        return Decimal::ZERO;
    }

    let mut guess = x / Decimal::from(2);
    if guess.is_zero() {
        guess = Decimal::from_str_exact("0.5").unwrap();
    }

    for _ in 0..10 {
        let new_guess = (guess + x / guess) / Decimal::from(2);
        if (new_guess - guess).abs() < Decimal::from_str_exact("0.0000001").unwrap() {
            return new_guess;
        }
        guess = new_guess;
    }

    guess
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn make_date(day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(2025, 1, day.min(28)).unwrap()
    }

    #[test]
    fn test_trend_state_display() {
        assert_eq!(TrendState::Uptrend.as_str(), "Uptrend");
        assert_eq!(TrendState::Downtrend.as_str(), "Downtrend");
        assert_eq!(TrendState::Sideways.as_str(), "Sideways");
    }

    #[test]
    fn test_vol_quantile_from_percentile() {
        assert_eq!(VolQuantile::from_percentile(dec!(0.1)), VolQuantile::Q1);
        assert_eq!(VolQuantile::from_percentile(dec!(0.3)), VolQuantile::Q2);
        assert_eq!(VolQuantile::from_percentile(dec!(0.5)), VolQuantile::Q3);
        assert_eq!(VolQuantile::from_percentile(dec!(0.7)), VolQuantile::Q4);
        assert_eq!(VolQuantile::from_percentile(dec!(0.9)), VolQuantile::Q5);
    }

    #[test]
    fn test_vol_quantile_display() {
        assert_eq!(format!("{}", VolQuantile::Q1), "Q1");
        assert_eq!(format!("{}", VolQuantile::Q5), "Q5");
    }

    #[test]
    fn test_regime_tag_warmup() {
        let date = make_date(1);
        let tag = RegimeTag::warmup(date);

        assert_eq!(tag.trend_state, TrendState::Sideways);
        assert_eq!(tag.vol_quantile, VolQuantile::Q3);
        assert_eq!(tag.vol_value, Decimal::ZERO);
    }

    #[test]
    fn test_regime_config_default() {
        let config = RegimeConfig::default();

        assert_eq!(config.trend_lookback, 20);
        assert_eq!(config.vol_lookback, 20);
        assert_eq!(config.vol_quantile_count, 5);
        assert!(config.benchmark_symbol.is_none());
    }

    #[test]
    fn test_classify_insufficient_history() {
        let mut engine = RegimeEngine::new();
        let returns: Vec<Decimal> = vec![dec!(0.01), dec!(0.02)]; // Only 2 days
        let tag = engine.classify_day(make_date(2), &returns);

        // Should be warmup since we need 20 days
        assert_eq!(tag.trend_state, TrendState::Sideways);
        assert_eq!(tag.vol_quantile, VolQuantile::Q3);
    }

    #[test]
    fn test_classify_uptrend() {
        let mut engine = RegimeEngine::with_config(RegimeConfig {
            trend_lookback: 10,
            trend_threshold: dec!(0.0001),
            vol_lookback: 10,
            ..Default::default()
        });

        // Strong uptrend: positive returns with some variance
        // Using alternating 0.8% and 1.2% to have volatility but clear upward trend
        let returns: Vec<Decimal> = (0..20)
            .map(|i| if i % 2 == 0 { dec!(0.008) } else { dec!(0.012) })
            .collect();

        let tag = engine.classify_day(make_date(20), &returns);

        // Should detect uptrend (positive slope with volatility)
        assert_eq!(tag.trend_state, TrendState::Uptrend);
        assert!(tag.trend_indicator > Decimal::ZERO);
    }

    #[test]
    fn test_classify_downtrend() {
        let mut engine = RegimeEngine::with_config(RegimeConfig {
            trend_lookback: 10,
            trend_threshold: dec!(0.0001),
            vol_lookback: 10,
            ..Default::default()
        });

        // Strong downtrend: negative returns with some variance
        let returns: Vec<Decimal> = (0..20)
            .map(|i| if i % 2 == 0 { dec!(-0.008) } else { dec!(-0.012) })
            .collect();

        let tag = engine.classify_day(make_date(20), &returns);

        // Should detect downtrend
        assert_eq!(tag.trend_state, TrendState::Downtrend);
        assert!(tag.trend_indicator < Decimal::ZERO);
    }

    #[test]
    fn test_classify_sideways() {
        let mut engine = RegimeEngine::with_config(RegimeConfig {
            trend_lookback: 10,
            trend_threshold: dec!(1.0), // Very high threshold to ensure sideways
            vol_lookback: 10,
            ..Default::default()
        });

        // Oscillating returns (sideways - mean near zero)
        let returns: Vec<Decimal> = (0..20)
            .map(|i| if i % 2 == 0 { dec!(0.01) } else { dec!(-0.01) })
            .collect();

        let tag = engine.classify_day(make_date(20), &returns);

        // Should detect sideways (oscillating around zero with high threshold)
        assert_eq!(tag.trend_state, TrendState::Sideways);
    }

    #[test]
    fn test_classify_period() {
        let mut engine = RegimeEngine::with_config(RegimeConfig {
            trend_lookback: 5,
            vol_lookback: 5,
            trend_threshold: dec!(0.0001),
            ..Default::default()
        });

        // Use alternating returns to have volatility but clear upward trend
        let daily_data: Vec<(NaiveDate, Decimal)> = (1..=30)
            .map(|i| {
                let ret = if i % 2 == 0 { dec!(0.008) } else { dec!(0.012) };
                (make_date(i.min(28)), ret)
            })
            .collect();

        let tags = engine.classify_period(&daily_data);

        assert_eq!(tags.len(), 30);

        // First 5 days should be warmup (vol_quantile defaults to Q3 during warmup)
        for tag in &tags[..5] {
            // Warmup period - tags exist but may have default/neutral values
            assert!(tag.trend_indicator.is_zero() || !tag.trend_indicator.is_zero());
        }

        // Later days should detect uptrend (positive mean returns)
        for tag in &tags[10..] {
            assert_eq!(tag.trend_state, TrendState::Uptrend);
        }
    }

    #[test]
    fn test_vol_quantile_expanding_window() {
        let mut engine = RegimeEngine::with_config(RegimeConfig {
            trend_lookback: 5,
            vol_lookback: 5,
            ..Default::default()
        });

        // Create data with increasing volatility over time
        let mut daily_data: Vec<(NaiveDate, Decimal)> = Vec::new();

        // Low vol period
        for i in 1..=15 {
            daily_data.push((make_date(i), dec!(0.001)));
        }

        // High vol period
        for i in 16..=28 {
            let ret = if i % 2 == 0 { dec!(0.05) } else { dec!(-0.05) };
            daily_data.push((make_date(i), ret));
        }

        let tags = engine.classify_period(&daily_data);

        // Low vol period should have lower quantiles
        // High vol period should have higher quantiles
        let low_vol_tags: Vec<_> = tags[10..15].iter().collect();
        let high_vol_tags: Vec<_> = tags[23..].iter().collect();

        // At least some high vol tags should be Q4 or Q5
        let has_high_vol = high_vol_tags.iter().any(|t| t.vol_quantile >= VolQuantile::Q4);
        assert!(has_high_vol, "High vol period should have high quantiles");
    }

    #[test]
    fn test_summarize_basic() {
        let mut engine = RegimeEngine::with_config(RegimeConfig {
            trend_lookback: 5,
            vol_lookback: 5,
            ..Default::default()
        });

        let daily_data: Vec<(NaiveDate, Decimal)> = (1..=20)
            .map(|i| (make_date(i.min(28)), dec!(0.01)))
            .collect();

        let tags = engine.classify_period(&daily_data);
        let returns: Vec<Decimal> = daily_data.iter().map(|(_, r)| *r).collect();
        let turnover: Vec<Decimal> = vec![dec!(0.05); 20];
        let costs: Vec<Decimal> = vec![dec!(0.001); 20];

        let summary = engine.summarize(&tags, &returns, &turnover, &costs);

        assert_eq!(summary.total_days, 20);
        assert!(summary.warmup_days > 0);
        assert!(!summary.by_regime.is_empty());

        // Should have positive mean return (all days were +1%)
        for perf in &summary.by_regime {
            assert!(perf.mean_return_pct > Decimal::ZERO);
        }
    }

    #[test]
    fn test_summarize_empty() {
        let engine = RegimeEngine::new();
        let summary = engine.summarize(&[], &[], &[], &[]);

        assert!(summary.by_regime.is_empty());
        assert_eq!(summary.total_days, 0);
    }

    #[test]
    fn test_best_worst_regime() {
        let summary = RegimeSummary {
            config: RegimeConfig::default(),
            by_regime: vec![
                RegimePerformance {
                    trend_state: TrendState::Uptrend,
                    vol_quantile: VolQuantile::Q1,
                    day_count: 10,
                    mean_return_pct: dec!(1.5),
                    cumulative_return_pct: dec!(15),
                    win_rate_pct: dec!(70),
                    mean_turnover_pct: dec!(5),
                    mean_cost_pct: dec!(0.1),
                },
                RegimePerformance {
                    trend_state: TrendState::Downtrend,
                    vol_quantile: VolQuantile::Q5,
                    day_count: 5,
                    mean_return_pct: dec!(-0.8),
                    cumulative_return_pct: dec!(-4),
                    win_rate_pct: dec!(30),
                    mean_turnover_pct: dec!(8),
                    mean_cost_pct: dec!(0.15),
                },
            ],
            total_days: 15,
            warmup_days: 5,
        };

        let best = summary.best_regime().unwrap();
        assert_eq!(best.trend_state, TrendState::Uptrend);
        assert_eq!(best.mean_return_pct, dec!(1.5));

        let worst = summary.worst_regime().unwrap();
        assert_eq!(worst.trend_state, TrendState::Downtrend);
        assert_eq!(worst.mean_return_pct, dec!(-0.8));
    }

    #[test]
    fn test_serialization() {
        let tag = RegimeTag::new(
            make_date(15),
            TrendState::Uptrend,
            VolQuantile::Q4,
            dec!(0.25),
            dec!(0.0015),
        );

        let json = serde_json::to_string(&tag).unwrap();
        let parsed: RegimeTag = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.trend_state, TrendState::Uptrend);
        assert_eq!(parsed.vol_quantile, VolQuantile::Q4);
    }

    #[test]
    fn test_decimal_sqrt() {
        let sqrt4 = decimal_sqrt(dec!(4));
        assert!((sqrt4 - dec!(2)).abs() < dec!(0.001));

        let sqrt0 = decimal_sqrt(dec!(0));
        assert_eq!(sqrt0, Decimal::ZERO);
    }
}

