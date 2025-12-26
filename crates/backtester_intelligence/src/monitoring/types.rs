//! Core types for Monitoring & Alerting module.
//!
//! Wall Street grade monitoring with:
//! - Severity levels: INFO -> WARN -> CRIT -> HALT
//! - Evidence-based alerts with audit trail
//! - Circuit breaker actions

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::filters::Market;

/// Alert severity levels (Wall Street standard).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Severity {
    /// Log only, no action required
    Info,
    /// Alert, continue operation with caution
    Warn,
    /// Critical alert, flag for review
    Crit,
    /// Circuit breaker triggered, NO-TRADE mode
    Halt,
}

impl Severity {
    pub fn is_critical(&self) -> bool {
        matches!(self, Severity::Crit | Severity::Halt)
    }

    pub fn is_actionable(&self) -> bool {
        matches!(self, Severity::Warn | Severity::Crit | Severity::Halt)
    }
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Info => write!(f, "INFO"),
            Severity::Warn => write!(f, "WARN"),
            Severity::Crit => write!(f, "CRIT"),
            Severity::Halt => write!(f, "HALT"),
        }
    }
}

impl Default for Severity {
    fn default() -> Self {
        Severity::Info
    }
}

/// Category of monitoring check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CheckCategory {
    /// Data quality and freshness checks
    DataHealth,
    /// Distribution and selection drift checks
    Drift,
    /// Performance and cost regression checks
    Regression,
    /// Circuit breaker state checks
    CircuitBreaker,
}

impl fmt::Display for CheckCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CheckCategory::DataHealth => write!(f, "DataHealth"),
            CheckCategory::Drift => write!(f, "Drift"),
            CheckCategory::Regression => write!(f, "Regression"),
            CheckCategory::CircuitBreaker => write!(f, "CircuitBreaker"),
        }
    }
}

/// Statistical baseline for comparison.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BaselineStats {
    pub mean: Decimal,
    pub std: Decimal,
    pub min: Decimal,
    pub max: Decimal,
    pub p50: Decimal,
    pub p95: Decimal,
    pub p99: Decimal,
    pub n: usize,
    pub window_days: u32,
    pub computed_at: Option<DateTime<Utc>>,
}

/// Current statistics for comparison.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CurrentStats {
    pub value: Decimal,
    pub mean: Decimal,
    pub std: Decimal,
    pub n: usize,
    pub as_of: Option<DateTime<Utc>>,
}

/// Evidence for audit trail.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Evidence {
    /// Query or data source used
    pub query_or_source: String,
    /// Sample data points for verification
    pub sample_data: Vec<String>,
    /// Historical baseline (if applicable)
    pub baseline: Option<BaselineStats>,
    /// Current measurement
    pub current: CurrentStats,
    /// Additional context
    pub context: Option<String>,
}

impl Evidence {
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            query_or_source: source.into(),
            ..Default::default()
        }
    }

    pub fn with_current(mut self, value: Decimal) -> Self {
        self.current.value = value;
        self
    }

    pub fn with_baseline(mut self, baseline: BaselineStats) -> Self {
        self.baseline = Some(baseline);
        self
    }

    pub fn with_sample(mut self, sample: Vec<String>) -> Self {
        self.sample_data = sample;
        self
    }

    pub fn with_context(mut self, ctx: impl Into<String>) -> Self {
        self.context = Some(ctx.into());
        self
    }
}

/// Result of a single monitoring check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    /// Name of the check
    pub check_name: String,
    /// Category of the check
    pub category: CheckCategory,
    /// Severity level
    pub severity: Severity,
    /// Whether the check passed
    pub passed: bool,
    /// Measured value
    pub value: Decimal,
    /// Threshold value
    pub threshold: Decimal,
    /// Human-readable message
    pub message: String,
    /// Audit evidence
    pub evidence: Evidence,
    /// Timestamp of check
    pub timestamp: DateTime<Utc>,
    /// Market (if applicable)
    pub market: Option<Market>,
}

impl CheckResult {
    pub fn pass(name: impl Into<String>, category: CheckCategory) -> Self {
        Self {
            check_name: name.into(),
            category,
            severity: Severity::Info,
            passed: true,
            value: Decimal::ZERO,
            threshold: Decimal::ZERO,
            message: "Check passed".to_string(),
            evidence: Evidence::default(),
            timestamp: Utc::now(),
            market: None,
        }
    }

    pub fn warn(name: impl Into<String>, category: CheckCategory, msg: impl Into<String>) -> Self {
        Self {
            check_name: name.into(),
            category,
            severity: Severity::Warn,
            passed: false,
            value: Decimal::ZERO,
            threshold: Decimal::ZERO,
            message: msg.into(),
            evidence: Evidence::default(),
            timestamp: Utc::now(),
            market: None,
        }
    }

    pub fn crit(name: impl Into<String>, category: CheckCategory, msg: impl Into<String>) -> Self {
        Self {
            check_name: name.into(),
            category,
            severity: Severity::Crit,
            passed: false,
            value: Decimal::ZERO,
            threshold: Decimal::ZERO,
            message: msg.into(),
            evidence: Evidence::default(),
            timestamp: Utc::now(),
            market: None,
        }
    }

    pub fn halt(name: impl Into<String>, category: CheckCategory, msg: impl Into<String>) -> Self {
        Self {
            check_name: name.into(),
            category,
            severity: Severity::Halt,
            passed: false,
            value: Decimal::ZERO,
            threshold: Decimal::ZERO,
            message: msg.into(),
            evidence: Evidence::default(),
            timestamp: Utc::now(),
            market: None,
        }
    }

    pub fn with_value(mut self, value: Decimal) -> Self {
        self.value = value;
        self
    }

    pub fn with_threshold(mut self, threshold: Decimal) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn with_evidence(mut self, evidence: Evidence) -> Self {
        self.evidence = evidence;
        self
    }

    pub fn with_market(mut self, market: Market) -> Self {
        self.market = Some(market);
        self
    }
}

/// Circuit breaker action to take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CircuitAction {
    /// Normal operation, continue
    Continue,
    /// Log warning, continue with caution
    WarnAndContinue,
    /// Set NO-TRADE flag, continue pipeline
    FlagNoTrade,
    /// Halt pipeline with error (exit code != 0)
    HaltWithError,
}

impl fmt::Display for CircuitAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CircuitAction::Continue => write!(f, "Continue"),
            CircuitAction::WarnAndContinue => write!(f, "WarnAndContinue"),
            CircuitAction::FlagNoTrade => write!(f, "FlagNoTrade"),
            CircuitAction::HaltWithError => write!(f, "HaltWithError"),
        }
    }
}

impl Default for CircuitAction {
    fn default() -> Self {
        CircuitAction::Continue
    }
}

/// Summary of monitoring results.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MonitoringSummary {
    pub total_checks: usize,
    pub passed: usize,
    pub info: usize,
    pub warnings: usize,
    pub criticals: usize,
    pub halts: usize,
}

impl MonitoringSummary {
    pub fn from_results(results: &[CheckResult]) -> Self {
        let mut summary = Self::default();
        summary.total_checks = results.len();
        
        for r in results {
            if r.passed {
                summary.passed += 1;
            }
            match r.severity {
                Severity::Info => summary.info += 1,
                Severity::Warn => summary.warnings += 1,
                Severity::Crit => summary.criticals += 1,
                Severity::Halt => summary.halts += 1,
            }
        }
        summary
    }

    pub fn overall_status(&self) -> Severity {
        if self.halts > 0 {
            Severity::Halt
        } else if self.criticals > 0 {
            Severity::Crit
        } else if self.warnings > 0 {
            Severity::Warn
        } else {
            Severity::Info
        }
    }
}

/// Circuit breaker state in report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerState {
    pub state: String,
    pub crit_count: usize,
    pub halt_threshold: usize,
    pub action: CircuitAction,
    pub last_trip: Option<DateTime<Utc>>,
    pub cooldown_remaining_minutes: Option<u32>,
}

impl Default for CircuitBreakerState {
    fn default() -> Self {
        Self {
            state: "Closed".to_string(),
            crit_count: 0,
            halt_threshold: 3,
            action: CircuitAction::Continue,
            last_trip: None,
            cooldown_remaining_minutes: None,
        }
    }
}

/// Complete monitoring report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringReport {
    /// Report timestamp
    pub timestamp: DateTime<Utc>,
    /// All check results
    pub results: Vec<CheckResult>,
    /// Summary statistics
    pub summary: MonitoringSummary,
    /// Circuit breaker state
    pub circuit_breaker: CircuitBreakerState,
    /// Final action to take
    pub action: CircuitAction,
    /// NO-TRADE flag
    pub no_trade: bool,
    /// Report version for schema compatibility
    pub version: String,
}

impl Default for MonitoringReport {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            results: Vec::new(),
            summary: MonitoringSummary::default(),
            circuit_breaker: CircuitBreakerState::default(),
            action: CircuitAction::Continue,
            no_trade: false,
            version: "1.0.0".to_string(),
        }
    }
}

impl MonitoringReport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_results(mut self, results: Vec<CheckResult>) -> Self {
        self.summary = MonitoringSummary::from_results(&results);
        self.results = results;
        self
    }

    pub fn with_action(mut self, action: CircuitAction) -> Self {
        self.action = action;
        self.no_trade = matches!(action, CircuitAction::FlagNoTrade | CircuitAction::HaltWithError);
        self
    }

    pub fn with_circuit_breaker(mut self, cb: CircuitBreakerState) -> Self {
        self.circuit_breaker = cb;
        self
    }

    /// Get results by category.
    pub fn by_category(&self, category: CheckCategory) -> Vec<&CheckResult> {
        self.results.iter().filter(|r| r.category == category).collect()
    }

    /// Get results by severity.
    pub fn by_severity(&self, severity: Severity) -> Vec<&CheckResult> {
        self.results.iter().filter(|r| r.severity == severity).collect()
    }

    /// Get failed results only.
    pub fn failed(&self) -> Vec<&CheckResult> {
        self.results.iter().filter(|r| !r.passed).collect()
    }

    /// Get critical and halt results.
    pub fn critical_issues(&self) -> Vec<&CheckResult> {
        self.results.iter().filter(|r| r.severity.is_critical()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warn);
        assert!(Severity::Warn < Severity::Crit);
        assert!(Severity::Crit < Severity::Halt);
    }

    #[test]
    fn test_severity_is_critical() {
        assert!(!Severity::Info.is_critical());
        assert!(!Severity::Warn.is_critical());
        assert!(Severity::Crit.is_critical());
        assert!(Severity::Halt.is_critical());
    }

    #[test]
    fn test_check_result_builders() {
        let pass = CheckResult::pass("test", CheckCategory::DataHealth);
        assert!(pass.passed);
        assert_eq!(pass.severity, Severity::Info);

        let warn = CheckResult::warn("test", CheckCategory::Drift, "warning msg");
        assert!(!warn.passed);
        assert_eq!(warn.severity, Severity::Warn);

        let crit = CheckResult::crit("test", CheckCategory::Regression, "critical msg");
        assert!(!crit.passed);
        assert_eq!(crit.severity, Severity::Crit);

        let halt = CheckResult::halt("test", CheckCategory::CircuitBreaker, "halt msg");
        assert!(!halt.passed);
        assert_eq!(halt.severity, Severity::Halt);
    }

    #[test]
    fn test_check_result_with_value() {
        let r = CheckResult::pass("test", CheckCategory::DataHealth)
            .with_value(dec!(42.5))
            .with_threshold(dec!(100));

        assert_eq!(r.value, dec!(42.5));
        assert_eq!(r.threshold, dec!(100));
    }

    #[test]
    fn test_monitoring_summary() {
        let results = vec![
            CheckResult::pass("a", CheckCategory::DataHealth),
            CheckResult::warn("b", CheckCategory::Drift, "warn"),
            CheckResult::crit("c", CheckCategory::Regression, "crit"),
            CheckResult::pass("d", CheckCategory::DataHealth),
        ];

        let summary = MonitoringSummary::from_results(&results);
        assert_eq!(summary.total_checks, 4);
        assert_eq!(summary.passed, 2);
        assert_eq!(summary.warnings, 1);
        assert_eq!(summary.criticals, 1);
        assert_eq!(summary.overall_status(), Severity::Crit);
    }

    #[test]
    fn test_report_by_category() {
        let results = vec![
            CheckResult::pass("a", CheckCategory::DataHealth),
            CheckResult::pass("b", CheckCategory::Drift),
            CheckResult::pass("c", CheckCategory::DataHealth),
        ];

        let report = MonitoringReport::new().with_results(results);
        let data_health = report.by_category(CheckCategory::DataHealth);
        assert_eq!(data_health.len(), 2);
    }

    #[test]
    fn test_circuit_action_display() {
        assert_eq!(format!("{}", CircuitAction::Continue), "Continue");
        assert_eq!(format!("{}", CircuitAction::FlagNoTrade), "FlagNoTrade");
    }

    #[test]
    fn test_evidence_builder() {
        let evidence = Evidence::new("SELECT * FROM ohlcv")
            .with_current(dec!(95.5))
            .with_context("Checking coverage for BR market");

        assert_eq!(evidence.query_or_source, "SELECT * FROM ohlcv");
        assert_eq!(evidence.current.value, dec!(95.5));
        assert_eq!(evidence.context, Some("Checking coverage for BR market".to_string()));
    }
}

