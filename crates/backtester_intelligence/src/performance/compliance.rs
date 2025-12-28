//! Compliance Module - Breach Logging and Summary.
//!
//! Provides auditable breach tracking for portfolio constraints:
//! - BreachEvent: Individual violation record
//! - BreachLog: Timeline of all breaches
//! - ComplianceSummary: Aggregated statistics

use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::monitoring::Severity;

use super::constraints::{ConstraintAction, ConstraintId, ConstraintScope};

// =============================================================================
// BREACH EVIDENCE
// =============================================================================

/// Evidence supporting a breach detection.
///
/// Provides audit trail for understanding why a breach occurred.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BreachEvidence {
    /// Top contributors to the breach (symbol/sector, weight/value)
    pub top_contributors: Vec<(String, Decimal)>,
    /// Human-readable context
    pub context: String,
}

impl BreachEvidence {
    /// Create evidence with context only.
    pub fn with_context(context: impl Into<String>) -> Self {
        Self {
            top_contributors: Vec::new(),
            context: context.into(),
        }
    }

    /// Add a contributor to the evidence.
    pub fn add_contributor(&mut self, name: String, value: Decimal) {
        self.top_contributors.push((name, value));
    }
}

// =============================================================================
// BREACH EVENT
// =============================================================================

/// A single constraint breach event.
///
/// Audit record with full details for reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreachEvent {
    /// Date of the breach
    pub date: NaiveDate,
    /// Which constraint was violated
    pub constraint_id: ConstraintId,
    /// Scope of the violation
    pub scope: ConstraintScope,
    /// Measured value that exceeded the limit
    pub measured_value: Decimal,
    /// Limit value that was exceeded
    pub limit_value: Decimal,
    /// Absolute magnitude of the breach (measured - limit)
    pub magnitude: Decimal,
    /// Magnitude as percentage of the limit
    pub magnitude_pct: Decimal,
    /// Severity level
    pub severity: Severity,
    /// Action taken in response
    pub action_taken: ConstraintAction,
    /// Evidence for the breach
    pub evidence: BreachEvidence,
    /// Whether this was ex-ante (pre-order) or ex-post (EOD)
    pub is_ex_ante: bool,
}

impl BreachEvent {
    /// Create a new breach event.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        date: NaiveDate,
        constraint_id: ConstraintId,
        scope: ConstraintScope,
        measured_value: Decimal,
        limit_value: Decimal,
        severity: Severity,
        action_taken: ConstraintAction,
        is_ex_ante: bool,
    ) -> Self {
        let magnitude = (measured_value - limit_value).max(Decimal::ZERO);
        let magnitude_pct = if limit_value.is_zero() {
            Decimal::ZERO
        } else {
            magnitude / limit_value * dec!(100)
        };

        Self {
            date,
            constraint_id,
            scope,
            measured_value,
            limit_value,
            magnitude,
            magnitude_pct,
            severity,
            action_taken,
            evidence: BreachEvidence::default(),
            is_ex_ante,
        }
    }

    /// Add evidence to the breach.
    pub fn with_evidence(mut self, evidence: BreachEvidence) -> Self {
        self.evidence = evidence;
        self
    }
}

// =============================================================================
// BREACH LOG
// =============================================================================

/// Log of all constraint breaches during a backtest.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BreachLog {
    /// All breach events in chronological order
    breaches: Vec<BreachEvent>,
}

impl BreachLog {
    /// Create an empty breach log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a breach to the log.
    pub fn push(&mut self, breach: BreachEvent) {
        self.breaches.push(breach);
    }

    /// Extend the log with multiple breaches.
    pub fn extend(&mut self, breaches: impl IntoIterator<Item = BreachEvent>) {
        self.breaches.extend(breaches);
    }

    /// Get all breaches.
    pub fn breaches(&self) -> &[BreachEvent] {
        &self.breaches
    }

    /// Get breaches by date.
    pub fn by_date(&self, date: NaiveDate) -> Vec<&BreachEvent> {
        self.breaches.iter().filter(|b| b.date == date).collect()
    }

    /// Get breaches by severity.
    pub fn by_severity(&self, severity: Severity) -> Vec<&BreachEvent> {
        self.breaches
            .iter()
            .filter(|b| b.severity == severity)
            .collect()
    }

    /// Get breaches by constraint ID.
    pub fn by_constraint(&self, id: &ConstraintId) -> Vec<&BreachEvent> {
        self.breaches
            .iter()
            .filter(|b| &b.constraint_id == id)
            .collect()
    }

    /// Check if any HALT-level breaches occurred.
    pub fn has_halt(&self) -> bool {
        self.breaches.iter().any(|b| b.severity == Severity::Halt)
    }

    /// Check if any CRIT-level or higher breaches occurred.
    pub fn has_critical(&self) -> bool {
        self.breaches
            .iter()
            .any(|b| b.severity >= Severity::Crit)
    }

    /// Get the worst severity in the log.
    pub fn worst_severity(&self) -> Severity {
        self.breaches
            .iter()
            .map(|b| b.severity)
            .max()
            .unwrap_or(Severity::Info)
    }

    /// Get the total number of breaches.
    pub fn len(&self) -> usize {
        self.breaches.len()
    }

    /// Check if the log is empty.
    pub fn is_empty(&self) -> bool {
        self.breaches.is_empty()
    }

    /// Get unique dates with breaches.
    pub fn breach_dates(&self) -> BTreeSet<NaiveDate> {
        self.breaches.iter().map(|b| b.date).collect()
    }

    /// Generate summary statistics.
    pub fn summarize(&self) -> ComplianceSummary {
        ComplianceSummary::from_breaches(&self.breaches)
    }
}

// =============================================================================
// COMPLIANCE SUMMARY
// =============================================================================

/// Summary of compliance status for reporting.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComplianceSummary {
    /// Total number of breach events
    pub total_breaches: usize,
    /// Breaches by severity level
    pub breaches_by_severity: BTreeMap<String, usize>,
    /// Worst severity encountered
    pub worst_severity: String,
    /// Number of days with at least one breach
    pub days_out_of_limit: usize,
    /// List of constraint IDs that were violated
    pub constraints_violated: Vec<String>,
    /// Number of ex-ante breaches (pre-order)
    pub ex_ante_breaches: usize,
    /// Number of ex-post breaches (EOD)
    pub ex_post_breaches: usize,
}

impl ComplianceSummary {
    /// Create summary from a list of breaches.
    pub fn from_breaches(breaches: &[BreachEvent]) -> Self {
        let mut breaches_by_severity: BTreeMap<String, usize> = BTreeMap::new();
        let mut constraints_violated: BTreeSet<String> = BTreeSet::new();
        let mut breach_dates: BTreeSet<NaiveDate> = BTreeSet::new();
        let mut ex_ante_count = 0;
        let mut ex_post_count = 0;
        let mut worst_severity = Severity::Info;

        for breach in breaches {
            *breaches_by_severity
                .entry(format!("{}", breach.severity))
                .or_default() += 1;

            constraints_violated.insert(format!("{}", breach.constraint_id));
            breach_dates.insert(breach.date);

            if breach.is_ex_ante {
                ex_ante_count += 1;
            } else {
                ex_post_count += 1;
            }

            if breach.severity > worst_severity {
                worst_severity = breach.severity;
            }
        }

        Self {
            total_breaches: breaches.len(),
            breaches_by_severity,
            worst_severity: format!("{}", worst_severity),
            days_out_of_limit: breach_dates.len(),
            constraints_violated: constraints_violated.into_iter().collect(),
            ex_ante_breaches: ex_ante_count,
            ex_post_breaches: ex_post_count,
        }
    }

    /// Check if there were any violations.
    pub fn has_violations(&self) -> bool {
        self.total_breaches > 0
    }

    /// Check if the worst severity is critical or higher.
    pub fn is_critical(&self) -> bool {
        self.worst_severity == "CRIT" || self.worst_severity == "HALT"
    }
}

// =============================================================================
// ACTION RECORD
// =============================================================================

/// Record of an action taken in response to a breach.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRecord {
    /// Date the action was taken
    pub date: NaiveDate,
    /// Action type
    pub action: ConstraintAction,
    /// Reason for the action
    pub reason: String,
    /// Constraint that triggered the action
    pub triggered_by: ConstraintId,
}

impl ActionRecord {
    /// Create a new action record.
    pub fn new(date: NaiveDate, action: ConstraintAction, reason: String, triggered_by: ConstraintId) -> Self {
        Self {
            date,
            action,
            reason,
            triggered_by,
        }
    }
}

// =============================================================================
// COMPLIANCE REPORT
// =============================================================================

/// Full compliance report for inclusion in PerformanceReport.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComplianceReport {
    /// Configuration snapshot (what limits were active)
    pub config_snapshot: BTreeMap<String, ConfigSnapshotEntry>,
    /// Summary statistics
    pub summary: ComplianceSummary,
    /// Top N breaches (most severe)
    pub breaches: Vec<BreachEvent>,
    /// Actions taken during the backtest
    pub actions_taken: Vec<ActionRecord>,
}

/// Entry in the config snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSnapshotEntry {
    /// Soft threshold (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub soft: Option<String>,
    /// Hard threshold
    pub hard: String,
    /// Action on breach
    pub action: String,
}

impl ComplianceReport {
    /// Create a compliance report from a breach log.
    pub fn from_log(log: &BreachLog, max_breaches: usize) -> Self {
        let summary = log.summarize();

        // Get top N breaches by severity (descending) then by magnitude (descending)
        let mut breaches: Vec<BreachEvent> = log.breaches().to_vec();
        breaches.sort_by(|a, b| {
            b.severity
                .cmp(&a.severity)
                .then_with(|| b.magnitude.cmp(&a.magnitude))
        });
        breaches.truncate(max_breaches);

        // Extract actions from breaches
        let actions_taken: Vec<ActionRecord> = log
            .breaches()
            .iter()
            .filter(|b| b.action_taken != ConstraintAction::LogOnly)
            .map(|b| ActionRecord {
                date: b.date,
                action: b.action_taken,
                reason: b.evidence.context.clone(),
                triggered_by: b.constraint_id.clone(),
            })
            .collect();

        Self {
            config_snapshot: BTreeMap::new(),
            summary,
            breaches,
            actions_taken,
        }
    }

    /// Set the config snapshot.
    pub fn with_config_snapshot(mut self, snapshot: BTreeMap<String, ConfigSnapshotEntry>) -> Self {
        self.config_snapshot = snapshot;
        self
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_breach(date: NaiveDate, severity: Severity, is_ex_ante: bool) -> BreachEvent {
        BreachEvent {
            date,
            constraint_id: ConstraintId::MaxGrossExposurePct,
            scope: ConstraintScope::Portfolio,
            measured_value: dec!(110),
            limit_value: dec!(100),
            magnitude: dec!(10),
            magnitude_pct: dec!(10),
            severity,
            action_taken: ConstraintAction::LogOnly,
            evidence: BreachEvidence::default(),
            is_ex_ante,
        }
    }

    #[test]
    fn test_breach_log_empty() {
        let log = BreachLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
        assert!(!log.has_halt());
        assert!(!log.has_critical());
        assert_eq!(log.worst_severity(), Severity::Info);
    }

    #[test]
    fn test_breach_log_push() {
        let mut log = BreachLog::new();
        let date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();

        log.push(make_breach(date, Severity::Warn, false));
        assert_eq!(log.len(), 1);
        assert!(!log.has_critical());

        log.push(make_breach(date, Severity::Crit, false));
        assert_eq!(log.len(), 2);
        assert!(log.has_critical());
    }

    #[test]
    fn test_breach_log_by_date() {
        let mut log = BreachLog::new();
        let date1 = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
        let date2 = NaiveDate::from_ymd_opt(2025, 1, 2).unwrap();

        log.push(make_breach(date1, Severity::Warn, false));
        log.push(make_breach(date1, Severity::Crit, false));
        log.push(make_breach(date2, Severity::Warn, false));

        assert_eq!(log.by_date(date1).len(), 2);
        assert_eq!(log.by_date(date2).len(), 1);
    }

    #[test]
    fn test_breach_log_worst_severity() {
        let mut log = BreachLog::new();
        let date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();

        log.push(make_breach(date, Severity::Warn, false));
        assert_eq!(log.worst_severity(), Severity::Warn);

        log.push(make_breach(date, Severity::Halt, false));
        assert_eq!(log.worst_severity(), Severity::Halt);
    }

    #[test]
    fn test_compliance_summary() {
        let mut log = BreachLog::new();
        let date1 = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
        let date2 = NaiveDate::from_ymd_opt(2025, 1, 2).unwrap();

        log.push(make_breach(date1, Severity::Warn, true));  // ex-ante
        log.push(make_breach(date1, Severity::Crit, false)); // ex-post
        log.push(make_breach(date2, Severity::Warn, false)); // ex-post

        let summary = log.summarize();

        assert_eq!(summary.total_breaches, 3);
        assert_eq!(summary.days_out_of_limit, 2);
        assert_eq!(summary.ex_ante_breaches, 1);
        assert_eq!(summary.ex_post_breaches, 2);
        assert_eq!(summary.worst_severity, "CRIT");
        assert!(summary.is_critical());
    }

    #[test]
    fn test_compliance_report_from_log() {
        let mut log = BreachLog::new();
        let date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();

        // Add breaches with different severities
        log.push(make_breach(date, Severity::Warn, false));
        
        let mut crit_breach = make_breach(date, Severity::Crit, false);
        crit_breach.action_taken = ConstraintAction::BlockNewTrades;
        log.push(crit_breach);

        let report = ComplianceReport::from_log(&log, 10);

        assert_eq!(report.summary.total_breaches, 2);
        assert_eq!(report.breaches.len(), 2);
        // First breach should be CRIT (sorted by severity)
        assert_eq!(report.breaches[0].severity, Severity::Crit);
        // Should have one action (BlockNewTrades)
        assert_eq!(report.actions_taken.len(), 1);
    }

    #[test]
    fn test_breach_evidence() {
        let mut evidence = BreachEvidence::with_context("Test context");
        evidence.add_contributor("PETR4".to_string(), dec!(50));
        evidence.add_contributor("VALE3".to_string(), dec!(30));

        assert_eq!(evidence.context, "Test context");
        assert_eq!(evidence.top_contributors.len(), 2);
    }
}

