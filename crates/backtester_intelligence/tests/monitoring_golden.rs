//! Golden tests for Monitoring module.
//!
//! Tests: stable output formats for JSON and Markdown.

use backtester_intelligence::monitoring::*;
use backtester_intelligence::filters::Market;
use chrono::{NaiveDate, TimeZone, Utc};
use rust_decimal_macros::dec;

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

/// Create a golden report with fixed data for deterministic output.
fn make_golden_report() -> MonitoringReport {
    let mut report = MonitoringReport::default();

    // Fixed timestamp for determinism
    report.timestamp = Utc.with_ymd_and_hms(2024, 1, 15, 10, 30, 0).unwrap();

    // Add representative results
    report.results = vec![
        // DataHealth - Pass
        CheckResult::pass("Freshness_BR", CheckCategory::DataHealth)
            .with_value(dec!(1))
            .with_threshold(dec!(5))
            .with_market(Market::BR)
            .with_evidence(Evidence::new("ohlcv.max_date")
                .with_current(dec!(1))),

        // DataHealth - Warn
        CheckResult::warn("Coverage_BR", CheckCategory::DataHealth, "Coverage at 78%")
            .with_value(dec!(78))
            .with_threshold(dec!(80))
            .with_market(Market::BR)
            .with_evidence(Evidence::new("symbols.coverage")
                .with_current(dec!(78))),

        // Drift - Pass
        CheckResult::pass("SelectionStability", CheckCategory::Drift)
            .with_value(dec!(75))
            .with_threshold(dec!(60))
            .with_evidence(Evidence::new("jaccard_similarity")),

        // Regression - Crit
        CheckResult::crit("DrawdownGuardrail", CheckCategory::Regression, "DD at 22%")
            .with_value(dec!(22))
            .with_threshold(dec!(20))
            .with_evidence(Evidence::new("drawdown_check")
                .with_current(dec!(22))),

        // Regression - Pass
        CheckResult::pass("TurnoverBudget", CheckCategory::Regression)
            .with_value(dec!(30))
            .with_threshold(dec!(50)),
    ];

    report.summary = MonitoringSummary::from_results(&report.results);
    report.action = CircuitAction::FlagNoTrade;
    report.no_trade = true;
    report.circuit_breaker = CircuitBreakerState {
        state: "Closed".to_string(),
        crit_count: 1,
        halt_threshold: 3,
        action: CircuitAction::FlagNoTrade,
        last_trip: None,
        cooldown_remaining_minutes: None,
    };

    report
}

// ============================================================================
// JSON Golden Tests
// ============================================================================

#[test]
fn golden_json_structure() {
    let report = make_golden_report();
    let reporter = MonitoringReporter::default();
    let json = reporter.to_json(&report).unwrap();

    // Required top-level fields
    assert!(json.contains("\"timestamp\""));
    assert!(json.contains("\"results\""));
    assert!(json.contains("\"summary\""));
    assert!(json.contains("\"circuit_breaker\""));
    assert!(json.contains("\"action\""));
    assert!(json.contains("\"no_trade\""));
    assert!(json.contains("\"version\""));
}

#[test]
fn golden_json_summary_fields() {
    let report = make_golden_report();
    let reporter = MonitoringReporter::default();
    let json = reporter.to_json(&report).unwrap();

    // Summary should have all count fields
    assert!(json.contains("\"total_checks\""));
    assert!(json.contains("\"passed\""));
    assert!(json.contains("\"warnings\""));
    assert!(json.contains("\"criticals\""));
    assert!(json.contains("\"halts\""));
}

#[test]
fn golden_json_check_result_fields() {
    let report = make_golden_report();
    let reporter = MonitoringReporter::default();
    let json = reporter.to_json(&report).unwrap();

    // Each check result should have required fields
    assert!(json.contains("\"check_name\""));
    assert!(json.contains("\"category\""));
    assert!(json.contains("\"severity\""));
    assert!(json.contains("\"passed\""));
    assert!(json.contains("\"value\""));
    assert!(json.contains("\"threshold\""));
    assert!(json.contains("\"message\""));
    assert!(json.contains("\"evidence\""));
}

#[test]
fn golden_json_circuit_breaker_fields() {
    let report = make_golden_report();
    let reporter = MonitoringReporter::default();
    let json = reporter.to_json(&report).unwrap();

    // Circuit breaker should have state fields
    assert!(json.contains("\"state\""));
    assert!(json.contains("\"crit_count\""));
    assert!(json.contains("\"halt_threshold\""));
}

#[test]
fn golden_json_values() {
    let report = make_golden_report();
    let reporter = MonitoringReporter::default();
    let json = reporter.to_json(&report).unwrap();

    // Specific values should appear
    assert!(json.contains("\"Freshness_BR\""));
    assert!(json.contains("\"Coverage_BR\""));
    assert!(json.contains("\"DrawdownGuardrail\""));
    assert!(json.contains("\"FlagNoTrade\""));
    assert!(json.contains("\"no_trade\": true"));
}

// ============================================================================
// Markdown Golden Tests
// ============================================================================

#[test]
fn golden_markdown_header() {
    let report = make_golden_report();
    let reporter = MonitoringReporter::default();
    let md = reporter.to_markdown(&report);

    // Header with date
    assert!(md.contains("## Monitoring Report - 2024-01-15"));
}

#[test]
fn golden_markdown_status_badge() {
    let report = make_golden_report();
    let reporter = MonitoringReporter::default();
    let md = reporter.to_markdown(&report);

    // Status badge with CRITICAL (has 1 crit)
    assert!(md.contains("### Status:"));
    assert!(md.contains("CRITICAL") || md.contains("🔴"));
}

#[test]
fn golden_markdown_summary_table() {
    let report = make_golden_report();
    let reporter = MonitoringReporter::default();
    let md = reporter.to_markdown(&report);

    // Summary table
    assert!(md.contains("| Metric | Count |"));
    assert!(md.contains("| Total Checks |"));
    assert!(md.contains("| Passed |"));
    assert!(md.contains("| Warnings |"));
    assert!(md.contains("| Critical |"));
}

#[test]
fn golden_markdown_circuit_breaker_section() {
    let report = make_golden_report();
    let reporter = MonitoringReporter::default();
    let md = reporter.to_markdown(&report);

    // Circuit breaker section
    assert!(md.contains("### Circuit Breaker"));
    assert!(md.contains("**State**:"));
    assert!(md.contains("**Critical Count**:"));
    assert!(md.contains("**Action**:"));
}

#[test]
fn golden_markdown_critical_issues() {
    let report = make_golden_report();
    let reporter = MonitoringReporter::default();
    let md = reporter.to_markdown(&report);

    // Critical issues section
    assert!(md.contains("### 🔴 Critical Issues"));
    assert!(md.contains("DrawdownGuardrail"));
}

#[test]
fn golden_markdown_warnings() {
    let report = make_golden_report();
    let reporter = MonitoringReporter::default();
    let md = reporter.to_markdown(&report);

    // Warnings section
    assert!(md.contains("### 🟡 Warnings"));
    assert!(md.contains("Coverage_BR"));
}

#[test]
fn golden_markdown_results_by_category() {
    let report = make_golden_report();
    let reporter = MonitoringReporter::default();
    let md = reporter.to_markdown(&report);

    // Category sections
    assert!(md.contains("### Results by Category"));
    assert!(md.contains("#### DataHealth"));
    assert!(md.contains("#### Drift"));
    assert!(md.contains("#### Regression"));
}

#[test]
fn golden_markdown_results_table() {
    let report = make_golden_report();
    let reporter = MonitoringReporter::default();
    let md = reporter.to_markdown(&report);

    // Results table header
    assert!(md.contains("| Check | Status | Value | Threshold | Message |"));
}

#[test]
fn golden_markdown_no_trade_flag() {
    let report = make_golden_report();
    let reporter = MonitoringReporter::default();
    let md = reporter.to_markdown(&report);

    // NO-TRADE flag
    assert!(md.contains("NO-TRADE FLAG SET"));
}

#[test]
fn golden_markdown_footer() {
    let report = make_golden_report();
    let reporter = MonitoringReporter::default();
    let md = reporter.to_markdown(&report);

    // Footer with version
    assert!(md.contains("Report Version:"));
    assert!(md.contains("1.0.0"));
}

// ============================================================================
// Summary Golden Tests
// ============================================================================

#[test]
fn golden_summary_format() {
    let report = make_golden_report();
    let reporter = MonitoringReporter::default();
    let summary = reporter.to_summary(&report);

    // Summary format
    assert!(summary.contains("Monitoring:"));
    assert!(summary.contains("checks"));
    assert!(summary.contains("passed"));
    assert!(summary.contains("warn"));
    assert!(summary.contains("crit"));
    assert!(summary.contains("Action:"));
    assert!(summary.contains("NO-TRADE:"));
}

// ============================================================================
// Determinism Golden Tests
// ============================================================================

#[test]
fn golden_json_deterministic() {
    let report1 = make_golden_report();
    let report2 = make_golden_report();

    let reporter = MonitoringReporter::default();
    let json1 = reporter.to_json(&report1).unwrap();
    let json2 = reporter.to_json(&report2).unwrap();

    // Compare structure (ignoring per-result timestamps which use Utc::now())
    // The report-level timestamp is fixed, but result timestamps vary
    assert_eq!(report1.results.len(), report2.results.len());
    for (r1, r2) in report1.results.iter().zip(report2.results.iter()) {
        assert_eq!(r1.check_name, r2.check_name);
        assert_eq!(r1.severity, r2.severity);
        assert_eq!(r1.value, r2.value);
    }
    
    // Verify both produce valid JSON with same structure
    assert!(json1.contains("\"version\": \"1.0.0\""));
    assert!(json2.contains("\"version\": \"1.0.0\""));
}

#[test]
fn golden_markdown_deterministic() {
    let report1 = make_golden_report();
    let report2 = make_golden_report();

    let reporter = MonitoringReporter::default();
    let md1 = reporter.to_markdown(&report1);
    let md2 = reporter.to_markdown(&report2);

    assert_eq!(md1, md2, "Same report should produce identical Markdown");
}

// ============================================================================
// Edge Case Golden Tests
// ============================================================================

#[test]
fn golden_empty_report() {
    let report = MonitoringReport::default();
    let reporter = MonitoringReporter::default();

    let json = reporter.to_json(&report).unwrap();
    let md = reporter.to_markdown(&report);

    // Should not crash on empty report
    assert!(json.contains("\"results\": []"));
    assert!(md.contains("## Monitoring Report"));
    assert!(md.contains("🟢 HEALTHY")); // No issues = healthy
}

#[test]
fn golden_all_pass_report() {
    let mut report = MonitoringReport::default();
    report.results = vec![
        CheckResult::pass("Check1", CheckCategory::DataHealth),
        CheckResult::pass("Check2", CheckCategory::Drift),
        CheckResult::pass("Check3", CheckCategory::Regression),
    ];
    report.summary = MonitoringSummary::from_results(&report.results);

    let reporter = MonitoringReporter::default();
    let md = reporter.to_markdown(&report);

    // Should show HEALTHY status
    assert!(md.contains("🟢 HEALTHY"));
    // Should NOT have critical or warning sections
    assert!(!md.contains("### 🔴 Critical Issues"));
    assert!(!md.contains("### 🟡 Warnings"));
}

#[test]
fn golden_halt_report() {
    let mut report = MonitoringReport::default();
    report.results = vec![
        CheckResult::halt("CircuitBreaker", CheckCategory::CircuitBreaker, "System halted"),
    ];
    report.summary = MonitoringSummary::from_results(&report.results);
    report.action = CircuitAction::HaltWithError;

    let reporter = MonitoringReporter::default();
    let md = reporter.to_markdown(&report);

    // Should show HALT status
    assert!(md.contains("🛑 HALT"));
}

