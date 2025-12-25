//! Risk-free rate provider abstraction for point-in-time rates.
//!
//! Supports multiple markets (BR/US) with fallback values when
//! historical data is not available.
//!
//! # Data Sources
//! - BR: BCB SGS Serie 432 (SELIC Meta) - api.bcb.gov.br
//! - US: FRED TB3MS (3-Month Treasury Bill) - api.stlouisfed.org
//!
//! # Providers
//! - `FallbackRiskFreeProvider`: Static rates (config-based)
//! - `DbRiskFreeRateProvider`: Point-in-time from database

use crate::filters::Market;
use chrono::NaiveDate;
use std::collections::BTreeMap;

/// Trait for providing risk-free rates by date and market.
pub trait RiskFreeRateProvider: Send + Sync {
    /// Get the risk-free rate for a given date and market.
    /// Returns annualized rate (e.g., 0.1075 = 10.75%).
    fn get_rate(&self, date: NaiveDate, market: Market) -> f64;

    /// Get rate with optional market, defaulting to BR if None.
    fn get_rate_or_default(&self, date: NaiveDate, market: Option<Market>) -> f64 {
        self.get_rate(date, market.unwrap_or(Market::BR))
    }
}

/// Fallback provider that returns static configured rates.
/// Used when historical rate data is not available.
#[derive(Debug, Clone)]
pub struct FallbackRiskFreeProvider {
    /// SELIC Meta rate for BR market (annualized).
    pub selic_br: f64,
    /// T-Bill 3M rate for US market (annualized).
    pub tbill_us: f64,
}

impl FallbackRiskFreeProvider {
    /// Create with default rates (Dec 2024 values).
    pub fn new() -> Self {
        Self {
            selic_br: 0.1075, // 10.75% SELIC Meta
            tbill_us: 0.0435, // ~4.35% T-Bill 3M
        }
    }

    /// Create with custom rates.
    pub fn with_rates(selic_br: f64, tbill_us: f64) -> Self {
        Self { selic_br, tbill_us }
    }
}

impl Default for FallbackRiskFreeProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl RiskFreeRateProvider for FallbackRiskFreeProvider {
    fn get_rate(&self, _date: NaiveDate, market: Market) -> f64 {
        match market {
            Market::BR => self.selic_br,
            Market::US => self.tbill_us,
        }
    }
}

// ============================================================================
// Database Provider (Point-in-Time)
// ============================================================================

/// Provider that uses pre-loaded rates from database with point-in-time semantics.
/// Uses BTreeMap for efficient O(log n) lookups of "latest rate <= date".
#[derive(Debug, Clone)]
pub struct DbRiskFreeRateProvider {
    /// BR rates indexed by date
    rates_br: BTreeMap<NaiveDate, f64>,
    /// US rates indexed by date
    rates_us: BTreeMap<NaiveDate, f64>,
    /// Whether to allow fallback when no data available
    fallback_enabled: bool,
    /// Fallback rate for BR
    fallback_br: f64,
    /// Fallback rate for US
    fallback_us: f64,
}

impl DbRiskFreeRateProvider {
    /// Create a new provider with pre-loaded data.
    pub fn new(
        rates_br: BTreeMap<NaiveDate, f64>,
        rates_us: BTreeMap<NaiveDate, f64>,
        fallback_enabled: bool,
        fallback_br: f64,
        fallback_us: f64,
    ) -> Self {
        Self {
            rates_br,
            rates_us,
            fallback_enabled,
            fallback_br,
            fallback_us,
        }
    }

    /// Create from vectors of (date, rate) tuples.
    pub fn from_data(
        br_data: Vec<(NaiveDate, f64)>,
        us_data: Vec<(NaiveDate, f64)>,
        fallback_enabled: bool,
        fallback_br: f64,
        fallback_us: f64,
    ) -> Self {
        let rates_br: BTreeMap<_, _> = br_data.into_iter().collect();
        let rates_us: BTreeMap<_, _> = us_data.into_iter().collect();
        Self::new(
            rates_br,
            rates_us,
            fallback_enabled,
            fallback_br,
            fallback_us,
        )
    }

    /// Get rate at date using point-in-time semantics (last available <= date).
    fn get_rate_pit(&self, date: NaiveDate, market: Market) -> Option<f64> {
        let rates = match market {
            Market::BR => &self.rates_br,
            Market::US => &self.rates_us,
        };

        // Use BTreeMap range to find last entry <= date
        rates.range(..=date).next_back().map(|(_, &rate)| rate)
    }

    /// Get fallback rate for a market.
    fn get_fallback(&self, market: Market) -> f64 {
        match market {
            Market::BR => self.fallback_br,
            Market::US => self.fallback_us,
        }
    }

    /// Check if data is available for a market.
    pub fn has_data(&self, market: Market) -> bool {
        match market {
            Market::BR => !self.rates_br.is_empty(),
            Market::US => !self.rates_us.is_empty(),
        }
    }

    /// Get date range for a market.
    pub fn date_range(&self, market: Market) -> Option<(NaiveDate, NaiveDate)> {
        let rates = match market {
            Market::BR => &self.rates_br,
            Market::US => &self.rates_us,
        };

        if rates.is_empty() {
            None
        } else {
            let min = *rates.keys().next()?;
            let max = *rates.keys().next_back()?;
            Some((min, max))
        }
    }
}

/// Error returned when rate is not available and fallback is disabled.
#[derive(Debug, Clone)]
pub struct RateNotAvailableError {
    pub date: NaiveDate,
    pub market: Market,
}

impl std::fmt::Display for RateNotAvailableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "No interest rate available for {:?} on {} and fallback disabled",
            self.market, self.date
        )
    }
}

impl std::error::Error for RateNotAvailableError {}

impl RiskFreeRateProvider for DbRiskFreeRateProvider {
    fn get_rate(&self, date: NaiveDate, market: Market) -> f64 {
        match self.get_rate_pit(date, market) {
            Some(rate) => rate,
            None if self.fallback_enabled => self.get_fallback(market),
            None => {
                // Log warning but don't panic - return 0 which will make carry calculation obvious
                tracing::warn!(
                    "No interest rate for {:?} on {} and fallback disabled - returning 0",
                    market,
                    date
                );
                0.0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_provider_br() {
        let provider = FallbackRiskFreeProvider::new();
        let date = NaiveDate::from_ymd_opt(2024, 12, 25).unwrap();

        let rate = provider.get_rate(date, Market::BR);
        assert!((rate - 0.1075).abs() < 0.0001);
    }

    #[test]
    fn test_fallback_provider_us() {
        let provider = FallbackRiskFreeProvider::new();
        let date = NaiveDate::from_ymd_opt(2024, 12, 25).unwrap();

        let rate = provider.get_rate(date, Market::US);
        assert!((rate - 0.0435).abs() < 0.0001);
    }

    #[test]
    fn test_fallback_provider_custom_rates() {
        let provider = FallbackRiskFreeProvider::with_rates(0.12, 0.05);
        let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();

        assert!((provider.get_rate(date, Market::BR) - 0.12).abs() < 0.0001);
        assert!((provider.get_rate(date, Market::US) - 0.05).abs() < 0.0001);
    }

    #[test]
    fn test_get_rate_or_default() {
        let provider = FallbackRiskFreeProvider::new();
        let date = NaiveDate::from_ymd_opt(2024, 6, 15).unwrap();

        // None defaults to BR
        let rate = provider.get_rate_or_default(date, None);
        assert!((rate - 0.1075).abs() < 0.0001);

        // Explicit US
        let rate = provider.get_rate_or_default(date, Some(Market::US));
        assert!((rate - 0.0435).abs() < 0.0001);
    }

    // ========================================================================
    // DbRiskFreeRateProvider Tests
    // ========================================================================

    fn make_test_db_provider() -> DbRiskFreeRateProvider {
        let br_data = vec![
            (NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(), 0.1075),
            (NaiveDate::from_ymd_opt(2024, 6, 1).unwrap(), 0.1025),
            (NaiveDate::from_ymd_opt(2024, 9, 1).unwrap(), 0.1175),
        ];
        let us_data = vec![
            (NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(), 0.0520),
            (NaiveDate::from_ymd_opt(2024, 6, 1).unwrap(), 0.0435),
        ];
        DbRiskFreeRateProvider::from_data(br_data, us_data, true, 0.10, 0.04)
    }

    #[test]
    fn test_db_provider_pit_exact_date() {
        let provider = make_test_db_provider();

        // Exact date match
        let rate = provider.get_rate(NaiveDate::from_ymd_opt(2024, 6, 1).unwrap(), Market::BR);
        assert!((rate - 0.1025).abs() < 0.0001);
    }

    #[test]
    fn test_db_provider_pit_interpolation() {
        let provider = make_test_db_provider();

        // Date between entries - should use last available
        let rate = provider.get_rate(NaiveDate::from_ymd_opt(2024, 7, 15).unwrap(), Market::BR);
        assert!(
            (rate - 0.1025).abs() < 0.0001,
            "Expected 0.1025, got {}",
            rate
        );

        // Date after last entry
        let rate = provider.get_rate(NaiveDate::from_ymd_opt(2024, 12, 25).unwrap(), Market::BR);
        assert!(
            (rate - 0.1175).abs() < 0.0001,
            "Expected 0.1175, got {}",
            rate
        );
    }

    #[test]
    fn test_db_provider_fallback_enabled() {
        let provider = make_test_db_provider();

        // Date before any data - should use fallback
        let rate = provider.get_rate(NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), Market::BR);
        assert!(
            (rate - 0.10).abs() < 0.0001,
            "Expected fallback 0.10, got {}",
            rate
        );
    }

    #[test]
    fn test_db_provider_fallback_disabled() {
        let provider = DbRiskFreeRateProvider::from_data(
            vec![(NaiveDate::from_ymd_opt(2024, 6, 1).unwrap(), 0.10)],
            vec![],
            false, // fallback disabled
            0.10,
            0.04,
        );

        // Date before any data with fallback disabled - should return 0
        let rate = provider.get_rate(NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), Market::BR);
        assert!((rate - 0.0).abs() < 0.0001);
    }

    #[test]
    fn test_db_provider_us_market() {
        let provider = make_test_db_provider();

        let rate = provider.get_rate(NaiveDate::from_ymd_opt(2024, 8, 1).unwrap(), Market::US);
        assert!((rate - 0.0435).abs() < 0.0001);
    }

    #[test]
    fn test_db_provider_has_data() {
        let provider = make_test_db_provider();
        assert!(provider.has_data(Market::BR));
        assert!(provider.has_data(Market::US));

        let empty = DbRiskFreeRateProvider::from_data(vec![], vec![], true, 0.0, 0.0);
        assert!(!empty.has_data(Market::BR));
        assert!(!empty.has_data(Market::US));
    }

    #[test]
    fn test_db_provider_date_range() {
        let provider = make_test_db_provider();

        let (min, max) = provider.date_range(Market::BR).unwrap();
        assert_eq!(min, NaiveDate::from_ymd_opt(2024, 1, 1).unwrap());
        assert_eq!(max, NaiveDate::from_ymd_opt(2024, 9, 1).unwrap());
    }

    #[test]
    fn test_carry_with_variable_rates() {
        let provider = make_test_db_provider();
        let dividend_yield = 0.08; // 8%

        // Jan 2024: SELIC = 10.75%, carry = 8% - 10.75% = -2.75%
        let rate_jan = provider.get_rate(NaiveDate::from_ymd_opt(2024, 1, 15).unwrap(), Market::BR);
        let carry_jan = dividend_yield - rate_jan;
        assert!((carry_jan - (-0.0275)).abs() < 0.0001);

        // Jun 2024: SELIC = 10.25%, carry = 8% - 10.25% = -2.25%
        let rate_jun = provider.get_rate(NaiveDate::from_ymd_opt(2024, 6, 15).unwrap(), Market::BR);
        let carry_jun = dividend_yield - rate_jun;
        assert!((carry_jun - (-0.0225)).abs() < 0.0001);

        // Carry changes when rate changes
        assert!((carry_jun - carry_jan).abs() > 0.001);
    }
}
