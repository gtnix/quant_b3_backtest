//! Institutional Gates for Candidate Validation
//!
//! Hard constraints that candidates must pass before entering the Pareto frontier.
//! Failing a gate results in rejection, not just a penalty.

use serde::{Deserialize, Serialize};

use crate::config::InstitutionalGatesConfig;
use crate::cost_report::CostReport;

// =============================================================================
// GATE RESULT
// =============================================================================

/// Result of applying institutional gates to a candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    /// Whether the candidate passed all hard gates.
    pub passed: bool,
    /// List of gate checks performed.
    pub checks: Vec<GateCheck>,
    /// Reasons for rejection (if any).
    pub rejection_reasons: Vec<String>,
    /// Warnings (soft gates that were triggered).
    pub warnings: Vec<String>,
}

impl GateResult {
    /// Create a passing result with no issues.
    #[must_use]
    pub fn pass() -> Self {
        Self {
            passed: true,
            checks: Vec::new(),
            rejection_reasons: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Add a gate check.
    pub fn add_check(&mut self, check: GateCheck) {
        if check.status == GateStatus::Failed {
            self.passed = false;
            self.rejection_reasons.push(check.message.clone());
        } else if check.status == GateStatus::Warning {
            self.warnings.push(check.message.clone());
        }
        self.checks.push(check);
    }

    /// Check if there are any warnings.
    #[must_use]
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

impl Default for GateResult {
    fn default() -> Self {
        Self::pass()
    }
}

/// Individual gate check result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateCheck {
    /// Gate identifier.
    pub gate_id: String,
    /// Gate name for display.
    pub name: String,
    /// Actual value observed.
    pub value: f64,
    /// Threshold value.
    pub threshold: f64,
    /// Status of the check.
    pub status: GateStatus,
    /// Human-readable message.
    pub message: String,
}

/// Status of a gate check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateStatus {
    /// Gate passed.
    Passed,
    /// Gate triggered a warning (soft constraint).
    Warning,
    /// Gate failed (hard constraint).
    Failed,
}

// =============================================================================
// GATE CHECKER
// =============================================================================

/// Institutional gate checker.
#[derive(Debug, Clone)]
pub struct GateChecker {
    config: InstitutionalGatesConfig,
}

impl GateChecker {
    /// Create a new gate checker with the given configuration.
    #[must_use]
    pub fn new(config: InstitutionalGatesConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(InstitutionalGatesConfig::default())
    }

    /// Apply all gates to a candidate's metrics.
    #[must_use]
    pub fn check(
        &self,
        turnover_annual: f64,
        gross_pnl: f64,
        total_costs: f64,
        avg_slippage_bps: f64,
        capacity_usd: f64,
    ) -> GateResult {
        let mut result = GateResult::pass();

        // Gate 1: Max turnover (hard constraint)
        result.add_check(self.check_turnover(turnover_annual));

        // Gate 2: Max slippage as % of PnL (hard constraint)
        if gross_pnl.abs() > 0.0 {
            let slippage_pct_of_pnl = (total_costs / gross_pnl.abs()) * 100.0;
            result.add_check(self.check_slippage_pct(slippage_pct_of_pnl));
        }

        // Gate 3: Min capacity (warning)
        result.add_check(self.check_capacity(capacity_usd));

        // Gate 4: Max avg slippage (warning)
        result.add_check(self.check_avg_slippage(avg_slippage_bps));

        result
    }

    /// Apply gates using a CostReport.
    #[must_use]
    pub fn check_from_report(&self, report: &CostReport, gross_pnl: f64) -> GateResult {
        self.check(
            report.turnover_annual,
            gross_pnl,
            report.total_costs,
            report.avg_slippage_bps,
            report.capacity_proxy_usd,
        )
    }

    /// Check turnover gate.
    fn check_turnover(&self, turnover_annual: f64) -> GateCheck {
        let threshold = self.config.max_turnover_annual;
        let passed = turnover_annual <= threshold;

        GateCheck {
            gate_id: "G1".into(),
            name: "Max Annual Turnover".into(),
            value: turnover_annual,
            threshold,
            status: if passed {
                GateStatus::Passed
            } else {
                GateStatus::Failed
            },
            message: if passed {
                format!("Turnover {turnover_annual:.1}x <= {threshold:.1}x")
            } else {
                format!(
                    "Turnover {turnover_annual:.1}x exceeds institutional limit of {threshold:.1}x"
                )
            },
        }
    }

    /// Check slippage as percentage of PnL gate.
    fn check_slippage_pct(&self, slippage_pct: f64) -> GateCheck {
        let threshold = self.config.max_slippage_pct_of_pnl;
        let passed = slippage_pct <= threshold;

        GateCheck {
            gate_id: "G2".into(),
            name: "Max Slippage % of PnL".into(),
            value: slippage_pct,
            threshold,
            status: if passed {
                GateStatus::Passed
            } else {
                GateStatus::Failed
            },
            message: if passed {
                format!("Costs {slippage_pct:.1}% of PnL <= {threshold:.1}%")
            } else {
                format!(
                    "Costs consume {slippage_pct:.1}% of gross PnL, exceeds limit of {threshold:.1}%"
                )
            },
        }
    }

    /// Check minimum capacity gate (warning only).
    fn check_capacity(&self, capacity_usd: f64) -> GateCheck {
        let threshold = self.config.min_capacity_usd;
        let passed = capacity_usd >= threshold;

        GateCheck {
            gate_id: "G3".into(),
            name: "Min Capacity USD".into(),
            value: capacity_usd,
            threshold,
            status: if passed {
                GateStatus::Passed
            } else {
                GateStatus::Warning
            },
            message: if passed {
                format!(
                    "Capacity ${:.1}M >= ${:.1}M",
                    capacity_usd / 1_000_000.0,
                    threshold / 1_000_000.0
                )
            } else {
                format!(
                    "Capacity ${:.1}M below institutional minimum of ${:.1}M",
                    capacity_usd / 1_000_000.0,
                    threshold / 1_000_000.0
                )
            },
        }
    }

    /// Check average slippage gate (warning only).
    fn check_avg_slippage(&self, avg_slippage_bps: f64) -> GateCheck {
        let threshold = self.config.max_avg_slippage_bps;
        let passed = avg_slippage_bps <= threshold;

        GateCheck {
            gate_id: "G4".into(),
            name: "Max Avg Slippage BPS".into(),
            value: avg_slippage_bps,
            threshold,
            status: if passed {
                GateStatus::Passed
            } else {
                GateStatus::Warning
            },
            message: if passed {
                format!("Avg slippage {avg_slippage_bps:.1}bps <= {threshold:.1}bps")
            } else {
                format!(
                    "Avg slippage {avg_slippage_bps:.1}bps exceeds recommended {threshold:.1}bps"
                )
            },
        }
    }
}

impl Default for GateChecker {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_checker_all_pass() {
        let checker = GateChecker::with_defaults();
        let result = checker.check(
            8.0,         // turnover < 12
            100_000.0,   // gross_pnl
            10_000.0,    // total_costs (10% of pnl)
            10.0,        // avg_slippage_bps < 25
            10_000_000.0, // capacity > 5M
        );

        assert!(result.passed);
        assert!(result.rejection_reasons.is_empty());
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_gate_checker_turnover_fail() {
        let checker = GateChecker::with_defaults();
        let result = checker.check(
            20.0,        // turnover > 12 - FAIL
            100_000.0,
            10_000.0,
            10.0,
            10_000_000.0,
        );

        assert!(!result.passed);
        assert_eq!(result.rejection_reasons.len(), 1);
        assert!(result.rejection_reasons[0].contains("Turnover"));
    }

    #[test]
    fn test_gate_checker_slippage_pct_fail() {
        let checker = GateChecker::with_defaults();
        let result = checker.check(
            8.0,
            100_000.0,
            50_000.0, // 50% of pnl > 30% - FAIL
            10.0,
            10_000_000.0,
        );

        assert!(!result.passed);
        assert!(result.rejection_reasons.iter().any(|r| r.contains("Costs consume")));
    }

    #[test]
    fn test_gate_checker_capacity_warning() {
        let checker = GateChecker::with_defaults();
        let result = checker.check(
            8.0,
            100_000.0,
            10_000.0,
            10.0,
            3_000_000.0, // capacity < 5M - WARNING
        );

        assert!(result.passed); // Should still pass
        assert!(result.warnings.iter().any(|w| w.contains("Capacity")));
    }

    #[test]
    fn test_gate_checker_avg_slippage_warning() {
        let checker = GateChecker::with_defaults();
        let result = checker.check(
            8.0,
            100_000.0,
            10_000.0,
            30.0, // avg_slippage > 25 - WARNING
            10_000_000.0,
        );

        assert!(result.passed); // Should still pass
        assert!(result.warnings.iter().any(|w| w.contains("slippage")));
    }

    #[test]
    fn test_gate_result_default() {
        let result = GateResult::default();
        assert!(result.passed);
        assert!(result.checks.is_empty());
    }

    #[test]
    fn test_check_from_report() {
        let mut report = CostReport::new();
        report.turnover_annual = 8.0;
        report.total_costs = 10_000.0;
        report.avg_slippage_bps = 10.0;
        report.capacity_proxy_usd = 10_000_000.0;

        let checker = GateChecker::with_defaults();
        let result = checker.check_from_report(&report, 100_000.0);

        assert!(result.passed);
    }
}

