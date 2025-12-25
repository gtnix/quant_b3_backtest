//! Carry filter - selects assets with positive carry over risk-free rate.
//!
//! Technique 7 from TRADING_TECHNIQUES_GUIDE.md
//!
//! Carry = dividend_yield - risk_free_rate(date, market)
//!
//! Assets with higher carry offer return even without price appreciation.

use super::{AssetData, AssetFilter, FilterResult, Market};
use crate::config::CarryConfig;
use crate::risk_free::RiskFreeRateProvider;
use chrono::NaiveDate;
use std::sync::Arc;

/// Carry filter based on dividend yield vs risk-free rate.
pub struct CarryFilter {
    config: CarryConfig,
    rate_provider: Option<Arc<dyn RiskFreeRateProvider>>,
}

impl CarryFilter {
    /// Create filter using config fallback values only.
    pub fn new(config: CarryConfig) -> Self {
        Self {
            config,
            rate_provider: None,
        }
    }

    /// Create filter with a risk-free rate provider for point-in-time lookups.
    pub fn with_provider(config: CarryConfig, provider: Arc<dyn RiskFreeRateProvider>) -> Self {
        Self {
            config,
            rate_provider: Some(provider),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(CarryConfig::default())
    }

    /// Get risk-free rate based on market and date.
    /// Uses provider if available, otherwise falls back to config values.
    fn get_risk_free_rate(&self, date: Option<NaiveDate>, market: Option<Market>) -> f64 {
        let market = market.unwrap_or(Market::BR);

        if let Some(ref provider) = self.rate_provider {
            let query_date = date.unwrap_or_else(|| chrono::Local::now().date_naive());
            return provider.get_rate(query_date, market);
        }

        // Fallback to config values
        match market {
            Market::BR => self.config.fallback_selic_br,
            Market::US => self.config.fallback_tbill_us,
        }
    }
}

impl AssetFilter for CarryFilter {
    fn name(&self) -> &str {
        "carry"
    }

    fn weight(&self) -> f64 {
        self.config.base.weight
    }

    fn evaluate(&self, data: &AssetData) -> FilterResult {
        let dy = match data.dividend_yield {
            Some(v) => v,
            None => {
                return FilterResult::fail("No dividend yield data available");
            }
        };

        // Get evaluation date from price data or use today
        let eval_date = data.prices.last().map(|(d, _)| *d);
        let rf = self.get_risk_free_rate(eval_date, data.market);
        let carry = dy - rf;

        if carry < self.config.min_carry {
            return FilterResult::fail(format!(
                "Carry {:.2}% (DY {:.2}% - RF {:.2}%) below minimum {:.2}%",
                carry * 100.0,
                dy * 100.0,
                rf * 100.0,
                self.config.min_carry * 100.0
            ))
            .with_metric(carry);
        }

        // Score: normalize carry to 0-1 range
        // Assuming reasonable carry range: -5% to +10%
        let score = ((carry + 0.05) / 0.15).clamp(0.0, 1.0);

        FilterResult::pass(
            score,
            format!(
                "Carry {:.2}% (DY {:.2}% - RF {:.2}%)",
                carry * 100.0,
                dy * 100.0,
                rf * 100.0
            ),
        )
        .with_metric(carry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_carry_pass_br() {
        // DY = 12%, SELIC = 10.75% → carry = 1.25% → PASS
        let config = CarryConfig {
            fallback_selic_br: 0.1075,
            fallback_tbill_us: 0.0435,
            min_carry: 0.0,
            ..Default::default()
        };
        let filter = CarryFilter::new(config);

        let mut data = AssetData::new("TAEE11");
        data.dividend_yield = Some(0.12);
        data.market = Some(Market::BR);

        let result = filter.evaluate(&data);
        assert!(result.passed);
        assert!(result.metric_value.unwrap() > 0.0);
    }

    #[test]
    fn test_carry_fail_br() {
        // DY = 8%, SELIC = 10.75% → carry = -2.75% → FAIL
        let config = CarryConfig {
            fallback_selic_br: 0.1075,
            fallback_tbill_us: 0.0435,
            min_carry: 0.0,
            ..Default::default()
        };
        let filter = CarryFilter::new(config);

        let mut data = AssetData::new("ITUB4");
        data.dividend_yield = Some(0.08);
        data.market = Some(Market::BR);

        let result = filter.evaluate(&data);
        assert!(!result.passed);
        assert!(result.metric_value.unwrap() < 0.0);
    }

    #[test]
    fn test_carry_pass_us() {
        // DY = 6%, T-Bill = 4.35% → carry = 1.65% → PASS
        let config = CarryConfig {
            fallback_selic_br: 0.1075,
            fallback_tbill_us: 0.0435,
            min_carry: 0.0,
            ..Default::default()
        };
        let filter = CarryFilter::new(config);

        let mut data = AssetData::new("VZ");
        data.dividend_yield = Some(0.06);
        data.market = Some(Market::US);

        let result = filter.evaluate(&data);
        assert!(result.passed);
        assert!(result.metric_value.unwrap() > 0.01);
    }

    #[test]
    fn test_carry_fail_us() {
        // DY = 2%, T-Bill = 4.35% → carry = -2.35% → FAIL
        let config = CarryConfig {
            fallback_selic_br: 0.1075,
            fallback_tbill_us: 0.0435,
            min_carry: 0.0,
            ..Default::default()
        };
        let filter = CarryFilter::new(config);

        let mut data = AssetData::new("AAPL");
        data.dividend_yield = Some(0.02);
        data.market = Some(Market::US);

        let result = filter.evaluate(&data);
        assert!(!result.passed);
    }

    #[test]
    fn test_carry_no_market_defaults_br() {
        // No market specified → defaults to BR (SELIC)
        let config = CarryConfig {
            fallback_selic_br: 0.1075,
            fallback_tbill_us: 0.0435,
            min_carry: 0.0,
            ..Default::default()
        };
        let filter = CarryFilter::new(config);

        let mut data = AssetData::new("TEST");
        data.dividend_yield = Some(0.12);
        // market = None (default)

        let result = filter.evaluate(&data);
        // Uses SELIC: 12% - 10.75% = 1.25% → PASS
        assert!(result.passed);
    }

    #[test]
    fn test_carry_min_threshold() {
        // Carry must be >= min_carry
        let config = CarryConfig {
            fallback_selic_br: 0.10,
            fallback_tbill_us: 0.05,
            min_carry: 0.02, // Require at least 2% carry
            ..Default::default()
        };
        let filter = CarryFilter::new(config);

        // DY = 11%, RF = 10% → carry = 1% → FAIL (< 2%)
        let mut data = AssetData::new("TEST1");
        data.dividend_yield = Some(0.11);
        data.market = Some(Market::BR);
        assert!(!filter.evaluate(&data).passed);

        // DY = 13%, RF = 10% → carry = 3% → PASS (> 2%)
        let mut data = AssetData::new("TEST2");
        data.dividend_yield = Some(0.13);
        data.market = Some(Market::BR);
        assert!(filter.evaluate(&data).passed);
    }

    #[test]
    fn test_carry_no_dividend_yield() {
        let filter = CarryFilter::with_defaults();
        let data = AssetData::new("NO_DY");

        let result = filter.evaluate(&data);
        assert!(!result.passed);
        assert!(result.reason.contains("No dividend yield"));
    }
}
