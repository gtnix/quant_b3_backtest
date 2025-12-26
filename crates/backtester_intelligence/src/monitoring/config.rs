//! Configuration for Monitoring & Alerting module.
//!
//! Wall Street grade thresholds with:
//! - Dynamic thresholds (percentile-based)
//! - Hard caps (guardrails)
//! - Market-specific overrides
//! - Known limitation flags

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::filters::Market;
use super::types::Severity;

/// Main monitoring configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Data health thresholds
    pub data_health: DataHealthConfig,
    /// Drift detection thresholds
    pub drift: DriftConfig,
    /// Performance regression thresholds
    pub regression: RegressionConfig,
    /// Circuit breaker settings
    pub circuit_breaker: CircuitBreakerConfig,
    /// Known limitations (to avoid false positives)
    pub known_limitations: KnownLimitationsConfig,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            data_health: DataHealthConfig::default(),
            drift: DriftConfig::default(),
            regression: RegressionConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            known_limitations: KnownLimitationsConfig::default(),
        }
    }
}

/// Data health check thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataHealthConfig {
    /// Max days since last OHLCV data (by market)
    pub freshness_max_days: HashMap<Market, u32>,
    /// WARN threshold for freshness
    pub freshness_warn_days: HashMap<Market, u32>,
    /// Min coverage percentage (symbols with sufficient data)
    pub coverage_min_pct: Decimal,
    /// WARN threshold for coverage
    pub coverage_warn_pct: Decimal,
    /// Outlier detection: standard deviations threshold
    pub outlier_std_threshold: Decimal,
    /// Min dividends expected in last 30 days
    pub dividends_min_30d: u32,
    /// Max days since last interest rate update
    pub interest_rates_max_days: u32,
    /// Enable watermark regression check
    pub check_watermark_regression: bool,
    /// Enable null checks for critical fields
    pub check_nulls: bool,
    /// Enable schema validation
    pub check_schema: bool,
}

impl Default for DataHealthConfig {
    fn default() -> Self {
        let mut freshness_max = HashMap::new();
        freshness_max.insert(Market::BR, 5);  // 5 business days CRIT
        freshness_max.insert(Market::US, 5);

        let mut freshness_warn = HashMap::new();
        freshness_warn.insert(Market::BR, 2);  // 2 days WARN
        freshness_warn.insert(Market::US, 3);

        Self {
            freshness_max_days: freshness_max,
            freshness_warn_days: freshness_warn,
            coverage_min_pct: dec!(50),    // < 50% = CRIT
            coverage_warn_pct: dec!(80),   // < 80% = WARN
            outlier_std_threshold: dec!(5), // 5 sigma
            dividends_min_30d: 10,          // < 10 = WARN, 0 = CRIT
            interest_rates_max_days: 7,     // > 7 days = WARN
            check_watermark_regression: true,
            check_nulls: true,
            check_schema: true,
        }
    }
}

impl DataHealthConfig {
    /// Get freshness CRIT threshold for market.
    pub fn freshness_crit(&self, market: Market) -> u32 {
        *self.freshness_max_days.get(&market).unwrap_or(&5)
    }

    /// Get freshness WARN threshold for market.
    pub fn freshness_warn(&self, market: Market) -> u32 {
        *self.freshness_warn_days.get(&market).unwrap_or(&2)
    }
}

/// Drift detection thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftConfig {
    /// Historical baseline window in days
    pub baseline_days: u32,
    /// Sigma threshold for WARN
    pub sigma_warn: Decimal,
    /// Sigma threshold for CRIT
    pub sigma_crit: Decimal,
    /// Min Jaccard overlap for selection stability (WARN)
    pub selection_overlap_warn: Decimal,
    /// Min Jaccard overlap for selection stability (CRIT)
    pub selection_overlap_crit: Decimal,
    /// KS test p-value for WARN
    pub ks_pvalue_warn: Decimal,
    /// KS test p-value for CRIT
    pub ks_pvalue_crit: Decimal,
    /// Multiplier for exclusion reason anomaly (WARN)
    pub exclusion_multiplier_warn: Decimal,
    /// Multiplier for exclusion reason anomaly (CRIT)
    pub exclusion_multiplier_crit: Decimal,
    /// Min samples for statistical tests
    pub min_samples: usize,
}

impl Default for DriftConfig {
    fn default() -> Self {
        Self {
            baseline_days: 60,              // 60 days baseline
            sigma_warn: dec!(2.0),          // 2 sigma WARN
            sigma_crit: dec!(3.0),          // 3 sigma CRIT
            selection_overlap_warn: dec!(60), // < 60% overlap WARN
            selection_overlap_crit: dec!(40), // < 40% overlap CRIT
            ks_pvalue_warn: dec!(0.05),     // p < 0.05 WARN
            ks_pvalue_crit: dec!(0.01),     // p < 0.01 CRIT
            exclusion_multiplier_warn: dec!(2), // 2x baseline WARN
            exclusion_multiplier_crit: dec!(3), // 3x baseline CRIT
            min_samples: 30,                 // Min N for KS test
        }
    }
}

/// Performance regression thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionConfig {
    /// Drawdown thresholds
    pub drawdown: DrawdownThreshold,
    /// Turnover thresholds
    pub turnover: DynamicThreshold,
    /// Cost thresholds
    pub cost: DynamicThreshold,
    /// Sharpe regression threshold
    pub sharpe_min: Decimal,
    /// Volatility multiplier threshold
    pub volatility_multiplier_warn: Decimal,
    /// Volatility multiplier threshold (CRIT)
    pub volatility_multiplier_crit: Decimal,
    /// Execution latency WARN (seconds)
    pub latency_warn_seconds: u32,
    /// Execution latency CRIT (seconds)
    pub latency_crit_seconds: u32,
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            drawdown: DrawdownThreshold::default(),
            turnover: DynamicThreshold::for_turnover(),
            cost: DynamicThreshold::for_cost(),
            sharpe_min: dec!(0),             // Sharpe < 0 = WARN
            volatility_multiplier_warn: dec!(1.5),
            volatility_multiplier_crit: dec!(2.0),
            latency_warn_seconds: 5,
            latency_crit_seconds: 10,
        }
    }
}

/// Drawdown thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownThreshold {
    /// WARN threshold percentage
    pub warn_pct: Decimal,
    /// CRIT threshold percentage
    pub crit_pct: Decimal,
    /// HALT threshold percentage (circuit breaker)
    pub halt_pct: Decimal,
}

impl Default for DrawdownThreshold {
    fn default() -> Self {
        Self {
            warn_pct: dec!(15),   // 15% DD = WARN
            crit_pct: dec!(20),   // 20% DD = CRIT
            halt_pct: dec!(25),   // 25% DD = HALT
        }
    }
}

/// Dynamic threshold with hard cap and percentile-based soft cap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicThreshold {
    /// Hard cap - never exceed (guardrail)
    pub hard_cap: Decimal,
    /// Percentile for soft cap calculation
    pub soft_percentile: Decimal,
    /// Baseline window in days
    pub baseline_days: u32,
    /// Update frequency (days)
    pub update_frequency: u32,
    /// Use percentile as soft cap
    pub use_percentile: bool,
}

impl DynamicThreshold {
    /// Threshold for turnover.
    pub fn for_turnover() -> Self {
        Self {
            hard_cap: dec!(50),      // 50% hard cap = CRIT
            soft_percentile: dec!(95), // p95 = WARN
            baseline_days: 60,
            update_frequency: 1,
            use_percentile: true,
        }
    }

    /// Threshold for cost.
    pub fn for_cost() -> Self {
        Self {
            hard_cap: dec!(0.5),     // 0.5% AUM hard cap = CRIT
            soft_percentile: dec!(95), // p95 = WARN
            baseline_days: 60,
            update_frequency: 1,
            use_percentile: true,
        }
    }
}

/// Circuit breaker configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Enable circuit breaker
    pub enabled: bool,
    /// Number of CRITs to trigger HALT
    pub halt_on_crit_count: usize,
    /// Cooldown period in minutes
    pub cooldown_minutes: u32,
    /// Halt on data freshness critical
    pub halt_on_data_crit: bool,
    /// Halt on drawdown critical
    pub halt_on_drawdown_crit: bool,
    /// Halt on cost critical
    pub halt_on_cost_crit: bool,
    /// Auto-recover after cooldown
    pub auto_recover: bool,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            halt_on_crit_count: 3,    // 3 CRITs = HALT
            cooldown_minutes: 60,      // 1 hour cooldown
            halt_on_data_crit: true,   // Immediate halt on data CRIT
            halt_on_drawdown_crit: true,
            halt_on_cost_crit: false,  // Cost CRIT doesn't auto-halt
            auto_recover: true,
        }
    }
}

/// Known limitations to avoid false positives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnownLimitationsConfig {
    /// US fundamentals missing - what severity?
    pub us_fundamentals_missing: Severity,
    /// US dividends partial coverage
    pub us_dividends_partial: bool,
    /// Min samples required for drift tests
    pub drift_min_samples: usize,
    /// Ignore drift on first N days
    pub drift_warmup_days: u32,
}

impl Default for KnownLimitationsConfig {
    fn default() -> Self {
        Self {
            us_fundamentals_missing: Severity::Warn,  // WARN, not CRIT
            us_dividends_partial: true,               // Known partial coverage
            drift_min_samples: 30,                    // Fallback to mean comparison
            drift_warmup_days: 60,                    // Skip drift on first 60 days
        }
    }
}

/// Threshold helper for determining severity.
#[derive(Debug, Clone)]
pub struct ThresholdEvaluator;

impl ThresholdEvaluator {
    /// Evaluate freshness and return severity.
    pub fn freshness_severity(days: u32, warn: u32, crit: u32) -> Severity {
        if days > crit {
            Severity::Crit
        } else if days > warn {
            Severity::Warn
        } else {
            Severity::Info
        }
    }

    /// Evaluate coverage and return severity.
    pub fn coverage_severity(pct: Decimal, warn: Decimal, crit: Decimal) -> Severity {
        if pct < crit {
            Severity::Crit
        } else if pct < warn {
            Severity::Warn
        } else {
            Severity::Info
        }
    }

    /// Evaluate sigma deviation and return severity.
    pub fn sigma_severity(sigma: Decimal, warn: Decimal, crit: Decimal) -> Severity {
        let abs_sigma = sigma.abs();
        if abs_sigma > crit {
            Severity::Crit
        } else if abs_sigma > warn {
            Severity::Warn
        } else {
            Severity::Info
        }
    }

    /// Evaluate value against dynamic threshold.
    pub fn dynamic_threshold_severity(
        value: Decimal, 
        soft_cap: Decimal, 
        hard_cap: Decimal
    ) -> Severity {
        if value > hard_cap {
            Severity::Crit
        } else if value > soft_cap {
            Severity::Warn
        } else {
            Severity::Info
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MonitoringConfig::default();
        
        assert_eq!(config.data_health.freshness_crit(Market::BR), 5);
        assert_eq!(config.data_health.freshness_warn(Market::BR), 2);
        assert_eq!(config.drift.baseline_days, 60);
        assert_eq!(config.circuit_breaker.halt_on_crit_count, 3);
    }

    #[test]
    fn test_freshness_severity() {
        assert_eq!(
            ThresholdEvaluator::freshness_severity(1, 2, 5),
            Severity::Info
        );
        assert_eq!(
            ThresholdEvaluator::freshness_severity(3, 2, 5),
            Severity::Warn
        );
        assert_eq!(
            ThresholdEvaluator::freshness_severity(6, 2, 5),
            Severity::Crit
        );
    }

    #[test]
    fn test_coverage_severity() {
        assert_eq!(
            ThresholdEvaluator::coverage_severity(dec!(90), dec!(80), dec!(50)),
            Severity::Info
        );
        assert_eq!(
            ThresholdEvaluator::coverage_severity(dec!(70), dec!(80), dec!(50)),
            Severity::Warn
        );
        assert_eq!(
            ThresholdEvaluator::coverage_severity(dec!(40), dec!(80), dec!(50)),
            Severity::Crit
        );
    }

    #[test]
    fn test_sigma_severity() {
        assert_eq!(
            ThresholdEvaluator::sigma_severity(dec!(1.5), dec!(2), dec!(3)),
            Severity::Info
        );
        assert_eq!(
            ThresholdEvaluator::sigma_severity(dec!(2.5), dec!(2), dec!(3)),
            Severity::Warn
        );
        assert_eq!(
            ThresholdEvaluator::sigma_severity(dec!(3.5), dec!(2), dec!(3)),
            Severity::Crit
        );
        // Negative sigma should also work
        assert_eq!(
            ThresholdEvaluator::sigma_severity(dec!(-3.5), dec!(2), dec!(3)),
            Severity::Crit
        );
    }

    #[test]
    fn test_dynamic_threshold_severity() {
        assert_eq!(
            ThresholdEvaluator::dynamic_threshold_severity(dec!(30), dec!(40), dec!(50)),
            Severity::Info
        );
        assert_eq!(
            ThresholdEvaluator::dynamic_threshold_severity(dec!(45), dec!(40), dec!(50)),
            Severity::Warn
        );
        assert_eq!(
            ThresholdEvaluator::dynamic_threshold_severity(dec!(55), dec!(40), dec!(50)),
            Severity::Crit
        );
    }

    #[test]
    fn test_serialization() {
        let config = MonitoringConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: MonitoringConfig = serde_json::from_str(&json).unwrap();
        
        assert_eq!(
            deserialized.data_health.freshness_crit(Market::BR),
            config.data_health.freshness_crit(Market::BR)
        );
    }

    #[test]
    fn test_known_limitations_default() {
        let config = KnownLimitationsConfig::default();
        assert_eq!(config.us_fundamentals_missing, Severity::Warn);
        assert!(config.us_dividends_partial);
    }
}

