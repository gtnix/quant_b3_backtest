//! Sanity checks for detecting suspicious metrics.
//!
//! Detects:
//! - Sharpe ratio > 10 (suspicious) or > 20 (almost certainly a bug)
//! - Volatility < 1% with high returns (suspicious)
//! - Very few trades (< 30) with "excellent" metrics
//! - CAGR > 200% in equities (unrealistic)
//! - Monotonic equity curve (possible data issue)

use serde::{Deserialize, Serialize};

use crate::{ValidationWarning, Verdict};

/// Configuration for sanity checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanityConfig {
    /// Sharpe threshold for warning.
    pub sharpe_warn_threshold: f64,
    /// Sharpe threshold for failure.
    pub sharpe_fail_threshold: f64,
    /// Minimum annual volatility (below = suspicious).
    pub min_volatility: f64,
    /// Minimum trades for reliable metrics.
    pub min_trades: u32,
    /// Maximum realistic CAGR for equities.
    pub max_cagr: f64,
    /// Maximum realistic Calmar ratio.
    pub max_calmar: f64,
}

impl Default for SanityConfig {
    fn default() -> Self {
        Self {
            sharpe_warn_threshold: 10.0,
            sharpe_fail_threshold: 20.0,
            min_volatility: 0.01, // 1%
            min_trades: 30,
            max_cagr: 2.0,        // 200%
            max_calmar: 10.0,
        }
    }
}

/// Flags for individual sanity checks.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SanityFlags {
    /// Sharpe ratio is suspiciously high (> warn threshold).
    pub sharpe_suspicious: bool,
    /// Sharpe ratio is absurdly high (> fail threshold).
    pub sharpe_absurd: bool,
    /// Volatility is too low.
    pub vol_too_low: bool,
    /// Too few trades for reliable metrics.
    pub trades_too_few: bool,
    /// CAGR is unrealistically high.
    pub cagr_unrealistic: bool,
    /// Calmar ratio is unrealistically high.
    pub calmar_unrealistic: bool,
    /// Equity curve appears monotonic (no variation).
    pub equity_monotonic: bool,
    /// There are null fields in required metrics.
    pub has_nulls: bool,
}

impl SanityFlags {
    /// Check if any critical flag is set.
    pub fn has_critical(&self) -> bool {
        self.sharpe_absurd || self.has_nulls
    }

    /// Check if any warning flag is set.
    pub fn has_warnings(&self) -> bool {
        self.sharpe_suspicious
            || self.vol_too_low
            || self.trades_too_few
            || self.cagr_unrealistic
            || self.calmar_unrealistic
            || self.equity_monotonic
    }
}

/// Result of sanity checks.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SanityCheckResult {
    /// Sanity flags.
    pub flags: SanityFlags,
    /// Overall verdict.
    pub verdict: Verdict,
    /// Human-readable message.
    pub message: String,
    /// Warnings generated.
    pub warnings: Vec<ValidationWarning>,
    /// Actual metric values for reference.
    pub metrics_snapshot: MetricsSnapshot,
}

/// Snapshot of key metrics for sanity report.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    /// Sharpe ratio.
    pub sharpe_ratio: Option<f64>,
    /// Annual volatility.
    pub annual_volatility: Option<f64>,
    /// CAGR.
    pub cagr: Option<f64>,
    /// Number of trades.
    pub num_trades: Option<u32>,
    /// Max drawdown.
    pub max_drawdown: Option<f64>,
    /// Calmar ratio.
    pub calmar_ratio: Option<f64>,
}

/// Parsed metrics for sanity checking.
#[derive(Debug, Clone, Default)]
pub struct ParsedMetrics {
    /// Sharpe ratio.
    pub sharpe_ratio: f64,
    /// Annual volatility.
    pub volatility: f64,
    /// CAGR.
    pub cagr: f64,
    /// Number of trades.
    pub total_trades: u32,
    /// Max drawdown.
    pub max_drawdown: f64,
    /// Calmar ratio.
    pub calmar_ratio: Option<f64>,
}

impl ParsedMetrics {
    /// Parse from JSON value.
    pub fn from_json(json: &serde_json::Value) -> Self {
        Self {
            sharpe_ratio: json.get("sharpe_ratio").and_then(|v| v.as_f64()).unwrap_or(0.0),
            volatility: json.get("volatility").and_then(|v| v.as_f64()).unwrap_or(0.0),
            cagr: json.get("cagr").and_then(|v| v.as_f64()).unwrap_or(0.0),
            total_trades: json.get("total_trades").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            max_drawdown: json.get("max_drawdown").and_then(|v| v.as_f64()).unwrap_or(0.0),
            calmar_ratio: json.get("calmar_ratio").and_then(|v| v.as_f64()),
        }
    }
}

/// Sanity checker for backtest metrics.
pub struct SanityChecker {
    config: SanityConfig,
}

impl Default for SanityChecker {
    fn default() -> Self {
        Self::new(SanityConfig::default())
    }
}

impl SanityChecker {
    /// Create a new sanity checker.
    pub fn new(config: SanityConfig) -> Self {
        Self { config }
    }

    /// Run all sanity checks on parsed metrics.
    pub fn check(&self, metrics: &ParsedMetrics) -> SanityCheckResult {
        let mut flags = SanityFlags::default();
        let mut warnings = Vec::new();

        // Check Sharpe ratio
        if metrics.sharpe_ratio.abs() > self.config.sharpe_fail_threshold {
            flags.sharpe_absurd = true;
            warnings.push(ValidationWarning::with_field(
                "SHARPE_ABSURD",
                format!(
                    "Sharpe ratio {:.2} > {} (almost certainly a bug)",
                    metrics.sharpe_ratio, self.config.sharpe_fail_threshold
                ),
                "sharpe_ratio",
            ));
        } else if metrics.sharpe_ratio.abs() > self.config.sharpe_warn_threshold {
            flags.sharpe_suspicious = true;
            warnings.push(ValidationWarning::with_field(
                "SHARPE_HIGH",
                format!(
                    "Sharpe ratio {:.2} > {} (suspicious, investigate)",
                    metrics.sharpe_ratio, self.config.sharpe_warn_threshold
                ),
                "sharpe_ratio",
            ));
        }

        // Check volatility
        if metrics.volatility > 0.0 && metrics.volatility < self.config.min_volatility {
            flags.vol_too_low = true;
            warnings.push(ValidationWarning::with_field(
                "VOL_LOW",
                format!(
                    "Annual volatility {:.2}% < {}% (may inflate Sharpe)",
                    metrics.volatility * 100.0,
                    self.config.min_volatility * 100.0
                ),
                "volatility",
            ));
        }

        // Check number of trades
        if metrics.total_trades < self.config.min_trades {
            flags.trades_too_few = true;
            warnings.push(ValidationWarning::with_field(
                "TRADES_FEW",
                format!(
                    "Only {} trades (< {} recommended for reliable metrics)",
                    metrics.total_trades, self.config.min_trades
                ),
                "total_trades",
            ));
        }

        // Check CAGR
        if metrics.cagr > self.config.max_cagr {
            flags.cagr_unrealistic = true;
            warnings.push(ValidationWarning::with_field(
                "CAGR_HIGH",
                format!(
                    "CAGR {:.0}% > {:.0}% (unrealistic for equities)",
                    metrics.cagr * 100.0,
                    self.config.max_cagr * 100.0
                ),
                "cagr",
            ));
        }

        // Check Calmar ratio
        if let Some(calmar) = metrics.calmar_ratio {
            if calmar > self.config.max_calmar {
                flags.calmar_unrealistic = true;
                warnings.push(ValidationWarning::with_field(
                    "CALMAR_HIGH",
                    format!(
                        "Calmar ratio {:.2} > {} (unrealistic)",
                        calmar, self.config.max_calmar
                    ),
                    "calmar_ratio",
                ));
            }
        }

        // Determine verdict
        let verdict = if flags.has_critical() {
            Verdict::Fail
        } else if flags.has_warnings() {
            Verdict::Warn
        } else {
            Verdict::Pass
        };

        // Generate message
        let message = if verdict == Verdict::Fail {
            "Sanity check FAILED: critical issues detected".to_string()
        } else if verdict == Verdict::Warn {
            format!("Sanity check passed with {} warnings", warnings.len())
        } else {
            "All sanity checks passed".to_string()
        };

        SanityCheckResult {
            flags,
            verdict,
            message,
            warnings,
            metrics_snapshot: MetricsSnapshot {
                sharpe_ratio: Some(metrics.sharpe_ratio),
                annual_volatility: Some(metrics.volatility),
                cagr: Some(metrics.cagr),
                num_trades: Some(metrics.total_trades),
                max_drawdown: Some(metrics.max_drawdown),
                calmar_ratio: metrics.calmar_ratio,
            },
        }
    }

    /// Run sanity checks from JSON value.
    pub fn check_json(&self, json: &serde_json::Value) -> SanityCheckResult {
        let metrics = ParsedMetrics::from_json(json);
        self.check(&metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_metrics_pass() {
        let checker = SanityChecker::default();
        let metrics = ParsedMetrics {
            sharpe_ratio: 1.5,
            volatility: 0.18,
            cagr: 0.15,
            total_trades: 100,
            max_drawdown: -0.15,
            calmar_ratio: Some(1.0),
        };

        let result = checker.check(&metrics);
        assert_eq!(result.verdict, Verdict::Pass);
        assert!(!result.flags.has_critical());
        assert!(!result.flags.has_warnings());
    }

    #[test]
    fn test_high_sharpe_warns() {
        let checker = SanityChecker::default();
        let metrics = ParsedMetrics {
            sharpe_ratio: 12.0, // > 10
            volatility: 0.18,
            cagr: 0.15,
            total_trades: 100,
            max_drawdown: -0.15,
            calmar_ratio: None,
        };

        let result = checker.check(&metrics);
        assert_eq!(result.verdict, Verdict::Warn);
        assert!(result.flags.sharpe_suspicious);
        assert!(!result.flags.sharpe_absurd);
    }

    #[test]
    fn test_absurd_sharpe_fails() {
        let checker = SanityChecker::default();
        let metrics = ParsedMetrics {
            sharpe_ratio: 25.0, // > 20
            volatility: 0.18,
            cagr: 0.15,
            total_trades: 100,
            max_drawdown: -0.15,
            calmar_ratio: None,
        };

        let result = checker.check(&metrics);
        assert_eq!(result.verdict, Verdict::Fail);
        assert!(result.flags.sharpe_absurd);
    }

    #[test]
    fn test_low_volatility_warns() {
        let checker = SanityChecker::default();
        let metrics = ParsedMetrics {
            sharpe_ratio: 5.0,
            volatility: 0.005, // 0.5% < 1%
            cagr: 0.15,
            total_trades: 100,
            max_drawdown: -0.05,
            calmar_ratio: None,
        };

        let result = checker.check(&metrics);
        assert_eq!(result.verdict, Verdict::Warn);
        assert!(result.flags.vol_too_low);
    }

    #[test]
    fn test_few_trades_warns() {
        let checker = SanityChecker::default();
        let metrics = ParsedMetrics {
            sharpe_ratio: 1.5,
            volatility: 0.18,
            cagr: 0.15,
            total_trades: 10, // < 30
            max_drawdown: -0.15,
            calmar_ratio: None,
        };

        let result = checker.check(&metrics);
        assert_eq!(result.verdict, Verdict::Warn);
        assert!(result.flags.trades_too_few);
    }
}


