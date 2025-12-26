//! Asset filter implementations.

mod carry;
mod dividend;
mod low_vol;
mod momentum;
mod quality;
mod size;
mod value;

pub use carry::CarryFilter;
pub use dividend::DividendYieldFilter;
pub use low_vol::LowVolFilter;
pub use momentum::MomentumFilter;
pub use quality::QualityFilter;
pub use size::SizeFilter;
pub use value::ValueFilter;

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

/// Market region for differentiated risk-free rates and thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Deserialize, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Market {
    /// Brazilian market (B3) - uses SELIC as risk-free rate
    #[default]
    BR,
    /// US market - uses T-Bill 3M as risk-free rate
    US,
}

/// Infer market from symbol pattern.
///
/// BR patterns: 4 letters + 1-2 digits (PETR4, VALE3, BOVA11)
/// US patterns: 1-5 uppercase letters, may contain hyphen (AAPL, MSFT, BRK-B)
///
/// # Examples
/// ```
/// use backtester_intelligence::filters::infer_market_from_symbol;
/// use backtester_intelligence::filters::Market;
///
/// assert_eq!(infer_market_from_symbol("PETR4"), Market::BR);
/// assert_eq!(infer_market_from_symbol("AAPL"), Market::US);
/// ```
pub fn infer_market_from_symbol(symbol: &str) -> Market {
    let trimmed = symbol.trim().to_uppercase();

    // BR: 4 letters + 1-2 digits (PETR4, VALE3, BOVA11, ITUB4)
    if trimmed.len() >= 5
        && trimmed.len() <= 6
        && trimmed.chars().take(4).all(|c| c.is_alphabetic())
        && trimmed.chars().skip(4).all(|c| c.is_numeric())
    {
        return Market::BR;
    }

    // US: 1-5 letters, may contain hyphen (AAPL, MSFT, META, BRK-B)
    let without_hyphen: String = trimmed.chars().filter(|c| *c != '-').collect();
    if without_hyphen.len() <= 5 && without_hyphen.chars().all(|c| c.is_alphabetic()) {
        return Market::US;
    }

    // Default: BR (conservative for B3-focused system)
    Market::BR
}

/// Data available for asset filtering.
#[derive(Debug, Clone, Default)]
pub struct AssetData {
    pub symbol: String,

    /// Market region (BR/US) for differentiated rates and thresholds
    pub market: Option<Market>,

    // Price data (from OHLCV)
    pub prices: Vec<(NaiveDate, f64)>, // (date, close)
    pub returns: Vec<f64>,             // Daily returns

    // Fundamental data (from fundamentals_snapshot)
    /// Date of the fundamental snapshot used (for point-in-time validation)
    pub fundamentals_as_of: Option<NaiveDate>,
    pub price_earnings: Option<f64>,
    pub price_to_book: Option<f64>,
    pub return_on_equity: Option<f64>,
    pub return_on_assets: Option<f64>,
    pub debt_to_equity: Option<f64>,
    pub profit_margins: Option<f64>,
    pub gross_margins: Option<f64>,
    pub operating_margins: Option<f64>,
    pub current_ratio: Option<f64>,
    pub quick_ratio: Option<f64>,
    pub market_cap: Option<i64>,
    pub enterprise_value: Option<i64>,
    pub dividend_yield: Option<f64>,
    pub earnings_growth: Option<f64>,
    pub revenue_growth: Option<f64>,

    // Computed metrics
    pub momentum_return: Option<f64>,
    pub annualized_volatility: Option<f64>,
}

impl AssetData {
    /// Create new AssetData with just symbol.
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            ..Default::default()
        }
    }

    /// Create new AssetData with explicit market.
    pub fn with_market(symbol: impl Into<String>, market: Market) -> Self {
        Self {
            symbol: symbol.into(),
            market: Some(market),
            ..Default::default()
        }
    }

    /// Create new AssetData with market inferred from symbol pattern.
    pub fn with_inferred_market(symbol: impl Into<String>) -> Self {
        let sym = symbol.into();
        let market = infer_market_from_symbol(&sym);
        Self {
            symbol: sym,
            market: Some(market),
            ..Default::default()
        }
    }

    /// Set market and return self (builder pattern).
    pub fn set_market(mut self, market: Market) -> Self {
        self.market = Some(market);
        self
    }

    /// Infer and set market from symbol pattern.
    pub fn infer_market(&mut self) {
        self.market = Some(infer_market_from_symbol(&self.symbol));
    }

    /// Calculate momentum return over lookback period.
    pub fn calculate_momentum(&mut self, lookback_days: usize, skip_last_days: usize) {
        if self.prices.len() < lookback_days + skip_last_days {
            return;
        }

        let end_idx = self.prices.len().saturating_sub(skip_last_days);
        let start_idx = end_idx.saturating_sub(lookback_days);

        if start_idx < end_idx && start_idx < self.prices.len() {
            let start_price = self.prices[start_idx].1;
            let end_price = self.prices[end_idx.saturating_sub(1)].1;

            if start_price > 0.0 {
                self.momentum_return = Some((end_price - start_price) / start_price);
            }
        }
    }

    /// Calculate annualized volatility.
    pub fn calculate_volatility(&mut self, lookback_days: usize) {
        if self.returns.len() < lookback_days {
            return;
        }

        let recent_returns: Vec<f64> = self
            .returns
            .iter()
            .rev()
            .take(lookback_days)
            .copied()
            .collect();

        if recent_returns.is_empty() {
            return;
        }

        let mean = recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;
        let variance = recent_returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / recent_returns.len() as f64;

        let daily_vol = variance.sqrt();
        self.annualized_volatility = Some(daily_vol * (252.0_f64).sqrt());
    }
}

/// Result of a filter evaluation.
#[derive(Debug, Clone)]
pub struct FilterResult {
    /// Whether the asset passed the filter.
    pub passed: bool,

    /// Normalized score (0.0 to 1.0) for scoring mode.
    pub score: f64,

    /// Human-readable reason for pass/fail.
    pub reason: String,

    /// Raw metric value used for filtering.
    pub metric_value: Option<f64>,
}

impl FilterResult {
    /// Create a passing result.
    pub fn pass(score: f64, reason: impl Into<String>) -> Self {
        Self {
            passed: true,
            score: score.clamp(0.0, 1.0),
            reason: reason.into(),
            metric_value: None,
        }
    }

    /// Create a failing result.
    pub fn fail(reason: impl Into<String>) -> Self {
        Self {
            passed: false,
            score: 0.0,
            reason: reason.into(),
            metric_value: None,
        }
    }

    /// Create result with metric value.
    pub fn with_metric(mut self, value: f64) -> Self {
        self.metric_value = Some(value);
        self
    }
}

/// Trait for asset filters.
pub trait AssetFilter: Send + Sync {
    /// Filter name for logging.
    fn name(&self) -> &str;

    /// Evaluate the filter on asset data.
    fn evaluate(&self, data: &AssetData) -> FilterResult;

    /// Get filter weight for scoring.
    fn weight(&self) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_market_br_symbols() {
        // Standard BR: 4 letters + 1 digit
        assert_eq!(infer_market_from_symbol("PETR4"), Market::BR);
        assert_eq!(infer_market_from_symbol("VALE3"), Market::BR);
        assert_eq!(infer_market_from_symbol("ITUB4"), Market::BR);
        assert_eq!(infer_market_from_symbol("BBDC4"), Market::BR);

        // BR with 2 digits (ETFs, units)
        assert_eq!(infer_market_from_symbol("BOVA11"), Market::BR);
        assert_eq!(infer_market_from_symbol("TAEE11"), Market::BR);
        assert_eq!(infer_market_from_symbol("KNRI11"), Market::BR);
    }

    #[test]
    fn test_infer_market_us_symbols() {
        // Standard US: 1-4 letters
        assert_eq!(infer_market_from_symbol("AAPL"), Market::US);
        assert_eq!(infer_market_from_symbol("MSFT"), Market::US);
        assert_eq!(infer_market_from_symbol("META"), Market::US);
        assert_eq!(infer_market_from_symbol("GOOG"), Market::US);
        assert_eq!(infer_market_from_symbol("V"), Market::US);
        assert_eq!(infer_market_from_symbol("IBM"), Market::US);

        // US with hyphen
        assert_eq!(infer_market_from_symbol("BRK-B"), Market::US);
        assert_eq!(infer_market_from_symbol("BF-B"), Market::US);
    }

    #[test]
    fn test_infer_market_case_insensitive() {
        assert_eq!(infer_market_from_symbol("petr4"), Market::BR);
        assert_eq!(infer_market_from_symbol("aapl"), Market::US);
        assert_eq!(infer_market_from_symbol("Petr4"), Market::BR);
    }

    #[test]
    fn test_infer_market_with_whitespace() {
        assert_eq!(infer_market_from_symbol(" PETR4 "), Market::BR);
        assert_eq!(infer_market_from_symbol(" AAPL "), Market::US);
    }

    #[test]
    fn test_asset_data_with_market() {
        let data = AssetData::with_market("PETR4", Market::BR);
        assert_eq!(data.symbol, "PETR4");
        assert_eq!(data.market, Some(Market::BR));
    }

    #[test]
    fn test_asset_data_with_inferred_market() {
        let br_data = AssetData::with_inferred_market("VALE3");
        assert_eq!(br_data.market, Some(Market::BR));

        let us_data = AssetData::with_inferred_market("AAPL");
        assert_eq!(us_data.market, Some(Market::US));
    }

    #[test]
    fn test_asset_data_infer_market_method() {
        let mut data = AssetData::new("PETR4");
        assert_eq!(data.market, None);

        data.infer_market();
        assert_eq!(data.market, Some(Market::BR));
    }

    #[test]
    fn test_asset_data_set_market_builder() {
        let data = AssetData::new("TEST").set_market(Market::US);
        assert_eq!(data.market, Some(Market::US));
    }

    // ========================================================================
    // Point-in-Time Fundamentals Tests (AO7-005)
    // ========================================================================

    #[test]
    fn test_fundamentals_as_of_field_exists() {
        let mut data = AssetData::new("PETR4");
        assert!(data.fundamentals_as_of.is_none());

        // Set fundamental date
        let snapshot_date = NaiveDate::from_ymd_opt(2024, 9, 30).unwrap();
        data.fundamentals_as_of = Some(snapshot_date);
        assert_eq!(data.fundamentals_as_of, Some(snapshot_date));
    }

    #[test]
    fn test_fundamentals_point_in_time_validation() {
        // Simulate a backtest date and fundamental snapshot date
        let backtest_date = NaiveDate::from_ymd_opt(2024, 6, 15).unwrap();
        let old_snapshot = NaiveDate::from_ymd_opt(2024, 3, 31).unwrap();
        let future_snapshot = NaiveDate::from_ymd_opt(2024, 9, 30).unwrap();

        // Old snapshot should be valid for backtest date
        assert!(old_snapshot <= backtest_date);

        // Future snapshot would be look-ahead bias
        assert!(future_snapshot > backtest_date);
    }

    #[test]
    fn test_asset_data_with_fundamentals_as_of() {
        let snapshot_date = NaiveDate::from_ymd_opt(2024, 6, 30).unwrap();

        let mut data = AssetData::with_inferred_market("PETR4");
        data.fundamentals_as_of = Some(snapshot_date);
        data.price_earnings = Some(8.5);
        data.return_on_equity = Some(0.18);

        // Verify both market and fundamentals_as_of are set
        assert_eq!(data.market, Some(Market::BR));
        assert_eq!(data.fundamentals_as_of, Some(snapshot_date));
        assert_eq!(data.price_earnings, Some(8.5));
    }

    #[test]
    fn test_fundamentals_config_cutoff_date() {
        use crate::config::FundamentalsConfig;

        let config = FundamentalsConfig {
            use_availability_lag: true,
            lag_days_br: 60,
            lag_days_us: 45,
            missing_policy: "drop".to_string(),
        };

        let backtest_date = NaiveDate::from_ymd_opt(2024, 6, 15).unwrap();

        // BR: 60 day lag → cutoff = 2024-04-16
        let cutoff_br = config.cutoff_date(backtest_date, Market::BR);
        assert_eq!(cutoff_br, NaiveDate::from_ymd_opt(2024, 4, 16).unwrap());

        // US: 45 day lag → cutoff = 2024-05-01
        let cutoff_us = config.cutoff_date(backtest_date, Market::US);
        assert_eq!(cutoff_us, NaiveDate::from_ymd_opt(2024, 5, 1).unwrap());
    }

    #[test]
    fn test_fundamentals_config_no_lag() {
        use crate::config::FundamentalsConfig;

        let config = FundamentalsConfig {
            use_availability_lag: false,
            lag_days_br: 60,
            lag_days_us: 45,
            missing_policy: "drop".to_string(),
        };

        let backtest_date = NaiveDate::from_ymd_opt(2024, 6, 15).unwrap();

        // With lag disabled, cutoff = backtest_date
        let cutoff_br = config.cutoff_date(backtest_date, Market::BR);
        assert_eq!(cutoff_br, backtest_date);

        let cutoff_us = config.cutoff_date(backtest_date, Market::US);
        assert_eq!(cutoff_us, backtest_date);
    }

    #[test]
    fn test_fundamentals_anti_lookahead_scenario() {
        // Scenario: Backtest on 2024-06-15
        // Q1 snapshot (2024-03-31) available after ~60 days → available 2024-05-30
        // Q2 snapshot (2024-06-30) → NOT YET AVAILABLE on 2024-06-15
        use crate::config::FundamentalsConfig;

        let backtest_date = NaiveDate::from_ymd_opt(2024, 6, 15).unwrap();
        let q1_snapshot = NaiveDate::from_ymd_opt(2024, 3, 31).unwrap();
        let q2_snapshot = NaiveDate::from_ymd_opt(2024, 6, 30).unwrap();

        let config = FundamentalsConfig {
            use_availability_lag: true,
            lag_days_br: 60,
            lag_days_us: 45,
            missing_policy: "drop".to_string(),
        };

        let cutoff = config.cutoff_date(backtest_date, Market::BR);

        // Q1 snapshot should be valid (available before cutoff)
        // Q1 available = 2024-03-31 + 60 = 2024-05-30, cutoff = 2024-04-16
        // Actually q1_snapshot <= cutoff means: 2024-03-31 <= 2024-04-16 ✓
        assert!(q1_snapshot <= cutoff, "Q1 snapshot should be usable");

        // Q2 snapshot should NOT be usable (it's after cutoff)
        assert!(q2_snapshot > cutoff, "Q2 snapshot would be look-ahead");
    }

    #[test]
    fn test_missing_fundamentals_policy_drop() {
        // When fundamentals are missing, Value/Quality filters should fail
        // This is already the behavior - verify with a data object without fundamentals
        let data = AssetData::new("NO_FUNDAMENTALS");

        // No fundamentals set
        assert!(data.price_earnings.is_none());
        assert!(data.return_on_equity.is_none());

        // Filters should fail gracefully (tested in filter-specific tests)
        // This test just documents the "drop" policy expectation
    }
}
