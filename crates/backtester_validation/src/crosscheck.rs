//! Cross-check validation: recompute metrics from nav_history and compare.
//!
//! Detects calculation bugs by independently computing metrics and comparing
//! against reported values.

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::{ValidationError, ValidationWarning, Verdict};

/// Configuration for cross-check validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrosscheckConfig {
    /// Relative tolerance for metric comparison (e.g., 0.001 = 0.1%).
    pub tolerance: f64,
    /// Risk-free rate for Sharpe calculation.
    pub risk_free_rate: f64,
    /// Trading days per year.
    pub trading_days_per_year: f64,
}

impl Default for CrosscheckConfig {
    fn default() -> Self {
        Self {
            tolerance: 0.001, // 0.1%
            risk_free_rate: 0.05,
            trading_days_per_year: 252.0,
        }
    }
}

/// Result of cross-check validation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CrosscheckResult {
    /// Whether cross-check passed.
    pub passed: bool,
    /// Verdict.
    pub verdict: Verdict,
    /// Comparison results for each metric.
    pub comparisons: Vec<MetricComparison>,
    /// Warnings generated.
    pub warnings: Vec<ValidationWarning>,
    /// Recomputed metrics.
    pub recomputed: RecomputedMetrics,
}

/// Comparison of a single metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComparison {
    /// Metric name.
    pub name: String,
    /// Reported value.
    pub reported: f64,
    /// Recomputed value.
    pub recomputed: f64,
    /// Absolute difference.
    pub difference: f64,
    /// Relative difference (as fraction).
    pub relative_diff: f64,
    /// Whether comparison passed tolerance.
    pub passed: bool,
}

/// Metrics recomputed from nav_history.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecomputedMetrics {
    /// Daily returns computed from NAV.
    pub daily_returns_count: usize,
    /// Recomputed CAGR.
    pub cagr: f64,
    /// Recomputed annual volatility.
    pub volatility: f64,
    /// Recomputed Sharpe ratio.
    pub sharpe_ratio: f64,
    /// Recomputed max drawdown.
    pub max_drawdown: f64,
}

/// NAV history row.
#[derive(Debug, Clone)]
pub struct NavRow {
    /// Date or timestamp.
    pub date: String,
    /// NAV value.
    pub nav: f64,
}

/// Cross-checker for validating metrics against nav_history.
pub struct CrossChecker {
    config: CrosscheckConfig,
}

impl Default for CrossChecker {
    fn default() -> Self {
        Self::new(CrosscheckConfig::default())
    }
}

impl CrossChecker {
    /// Create a new cross-checker.
    pub fn new(config: CrosscheckConfig) -> Self {
        Self { config }
    }

    /// Load NAV history from CSV.
    pub fn load_nav_history(&self, path: &Path) -> Result<Vec<NavRow>, ValidationError> {
        let mut reader = csv::Reader::from_path(path)?;
        let mut rows = Vec::new();

        for result in reader.records() {
            let record = result?;
            
            // Try to parse NAV from common column names
            let date = record.get(0).unwrap_or("").to_string();
            let nav = self.parse_nav_from_record(&record);
            
            if let Some(nav) = nav {
                rows.push(NavRow { date, nav });
            }
        }

        Ok(rows)
    }

    /// Try to parse NAV from a CSV record.
    fn parse_nav_from_record(&self, record: &csv::StringRecord) -> Option<f64> {
        // Try common column positions: nav (1), equity (1), value (1)
        for i in 1..record.len() {
            if let Some(val) = record.get(i) {
                if let Ok(nav) = val.parse::<f64>() {
                    if nav > 0.0 {
                        return Some(nav);
                    }
                }
            }
        }
        None
    }

    /// Compute metrics from NAV series.
    pub fn compute_metrics(&self, nav_series: &[f64]) -> RecomputedMetrics {
        if nav_series.len() < 2 {
            return RecomputedMetrics::default();
        }

        // Compute daily returns
        let returns: Vec<f64> = nav_series
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        // CAGR
        let initial = nav_series[0];
        let final_nav = *nav_series.last().unwrap();
        let years = nav_series.len() as f64 / self.config.trading_days_per_year;
        let cagr = if years > 0.0 && initial > 0.0 {
            (final_nav / initial).powf(1.0 / years) - 1.0
        } else {
            0.0
        };

        // Volatility
        let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>()
            / returns.len() as f64;
        let daily_vol = variance.sqrt();
        let annual_vol = daily_vol * self.config.trading_days_per_year.sqrt();

        // Sharpe
        let annual_return = mean_return * self.config.trading_days_per_year;
        let sharpe = if annual_vol > 0.0 {
            (annual_return - self.config.risk_free_rate) / annual_vol
        } else {
            0.0
        };

        // Max drawdown
        let mut peak = nav_series[0];
        let mut max_dd = 0.0;
        for &nav in nav_series {
            if nav > peak {
                peak = nav;
            }
            let dd = (nav - peak) / peak;
            if dd < max_dd {
                max_dd = dd;
            }
        }

        RecomputedMetrics {
            daily_returns_count: returns.len(),
            cagr,
            volatility: annual_vol,
            sharpe_ratio: sharpe,
            max_drawdown: max_dd,
        }
    }

    /// Compare reported vs recomputed metric.
    fn compare_metric(&self, name: &str, reported: f64, recomputed: f64) -> MetricComparison {
        let difference = (reported - recomputed).abs();
        let relative_diff = if reported.abs() > 1e-10 {
            difference / reported.abs()
        } else if recomputed.abs() > 1e-10 {
            difference / recomputed.abs()
        } else {
            0.0
        };

        MetricComparison {
            name: name.to_string(),
            reported,
            recomputed,
            difference,
            relative_diff,
            passed: relative_diff <= self.config.tolerance,
        }
    }

    /// Run cross-check against reported metrics.
    pub fn crosscheck(
        &self,
        reported: &ReportedMetrics,
        nav_series: &[f64],
    ) -> CrosscheckResult {
        let recomputed = self.compute_metrics(nav_series);
        let mut comparisons = Vec::new();
        let mut warnings = Vec::new();

        // Compare each metric
        let cagr_cmp = self.compare_metric("cagr", reported.cagr, recomputed.cagr);
        if !cagr_cmp.passed {
            warnings.push(ValidationWarning::with_field(
                "CAGR_MISMATCH",
                format!(
                    "CAGR mismatch: reported {:.4}, recomputed {:.4} (diff {:.2}%)",
                    reported.cagr, recomputed.cagr, cagr_cmp.relative_diff * 100.0
                ),
                "cagr",
            ));
        }
        comparisons.push(cagr_cmp);

        let vol_cmp = self.compare_metric("volatility", reported.volatility, recomputed.volatility);
        if !vol_cmp.passed {
            warnings.push(ValidationWarning::with_field(
                "VOL_MISMATCH",
                format!(
                    "Volatility mismatch: reported {:.4}, recomputed {:.4}",
                    reported.volatility, recomputed.volatility
                ),
                "volatility",
            ));
        }
        comparisons.push(vol_cmp);

        let sharpe_cmp = self.compare_metric("sharpe_ratio", reported.sharpe_ratio, recomputed.sharpe_ratio);
        if !sharpe_cmp.passed {
            warnings.push(ValidationWarning::with_field(
                "SHARPE_MISMATCH",
                format!(
                    "Sharpe mismatch: reported {:.4}, recomputed {:.4}",
                    reported.sharpe_ratio, recomputed.sharpe_ratio
                ),
                "sharpe_ratio",
            ));
        }
        comparisons.push(sharpe_cmp);

        let dd_cmp = self.compare_metric("max_drawdown", reported.max_drawdown, recomputed.max_drawdown);
        if !dd_cmp.passed {
            warnings.push(ValidationWarning::with_field(
                "DD_MISMATCH",
                format!(
                    "Max drawdown mismatch: reported {:.4}, recomputed {:.4}",
                    reported.max_drawdown, recomputed.max_drawdown
                ),
                "max_drawdown",
            ));
        }
        comparisons.push(dd_cmp);

        let passed = comparisons.iter().all(|c| c.passed);
        let verdict = if passed { Verdict::Pass } else { Verdict::Fail };

        CrosscheckResult {
            passed,
            verdict,
            comparisons,
            warnings,
            recomputed,
        }
    }
}

/// Reported metrics from metrics.json.
#[derive(Debug, Clone, Default)]
pub struct ReportedMetrics {
    /// CAGR.
    pub cagr: f64,
    /// Annual volatility.
    pub volatility: f64,
    /// Sharpe ratio.
    pub sharpe_ratio: f64,
    /// Max drawdown.
    pub max_drawdown: f64,
}

impl ReportedMetrics {
    /// Parse from JSON value.
    pub fn from_json(json: &serde_json::Value) -> Self {
        Self {
            cagr: json.get("cagr").and_then(|v| v.as_f64()).unwrap_or(0.0),
            volatility: json.get("volatility").and_then(|v| v.as_f64()).unwrap_or(0.0),
            sharpe_ratio: json.get("sharpe_ratio").and_then(|v| v.as_f64()).unwrap_or(0.0),
            max_drawdown: json.get("max_drawdown").and_then(|v| v.as_f64()).unwrap_or(0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_metrics_basic() {
        let checker = CrossChecker::default();
        
        // Simple NAV series: 100 -> 110 (10% return)
        let nav = vec![100.0, 105.0, 110.0, 108.0, 115.0];
        let metrics = checker.compute_metrics(&nav);

        assert!(metrics.cagr > 0.0);
        assert!(metrics.volatility > 0.0);
        assert!(metrics.max_drawdown <= 0.0);
    }

    #[test]
    fn test_crosscheck_matching() {
        let checker = CrossChecker::new(CrosscheckConfig {
            tolerance: 0.05, // 5% tolerance for test
            ..Default::default()
        });

        let nav = vec![100.0, 102.0, 105.0, 103.0, 108.0, 110.0];
        let recomputed = checker.compute_metrics(&nav);

        let reported = ReportedMetrics {
            cagr: recomputed.cagr,
            volatility: recomputed.volatility,
            sharpe_ratio: recomputed.sharpe_ratio,
            max_drawdown: recomputed.max_drawdown,
        };

        let result = checker.crosscheck(&reported, &nav);
        assert!(result.passed);
    }

    #[test]
    fn test_crosscheck_mismatch() {
        let checker = CrossChecker::default();

        let nav = vec![100.0, 102.0, 105.0, 103.0, 108.0, 110.0];
        
        // Deliberately wrong values
        let reported = ReportedMetrics {
            cagr: 0.5, // Way off
            volatility: 0.3,
            sharpe_ratio: 2.0,
            max_drawdown: -0.01,
        };

        let result = checker.crosscheck(&reported, &nav);
        assert!(!result.passed);
        assert!(!result.warnings.is_empty());
    }
}


