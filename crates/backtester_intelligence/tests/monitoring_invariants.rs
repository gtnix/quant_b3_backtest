//! Invariant tests for Monitoring module.
//!
//! Tests: determinism, severity correctness, report structure.

use backtester_intelligence::monitoring::*;
use backtester_intelligence::filters::Market;
use chrono::NaiveDate;
use rust_decimal_macros::dec;

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

// ============================================================================
// Determinism Tests
// ============================================================================

#[test]
fn monitoring_engine_deterministic() {
    let config = MonitoringConfig::default();
    let mut engine1 = MonitoringEngine::new(config.clone());
    let mut engine2 = MonitoringEngine::new(config);

    let ctx = MonitoringContext::new(date(2024, 1, 10));

    let report1 = engine1.run_all(&ctx);
    let report2 = engine2.run_all(&ctx);

    assert_eq!(report1.results.len(), report2.results.len());
    for (r1, r2) in report1.results.iter().zip(report2.results.iter()) {
        assert_eq!(r1.check_name, r2.check_name);
        assert_eq!(r1.severity, r2.severity);
        assert_eq!(r1.passed, r2.passed);
        assert_eq!(r1.value, r2.value);
    }
}

#[test]
fn reporter_json_deterministic() {
    let mut engine = MonitoringEngine::default();
    let ctx = MonitoringContext::new(date(2024, 1, 10));
    let report = engine.run_all(&ctx);

    let reporter = MonitoringReporter::default();
    let json1 = reporter.to_json(&report).unwrap();
    let json2 = reporter.to_json(&report).unwrap();

    // Same report should produce identical JSON
    assert_eq!(json1, json2);
}

#[test]
fn reporter_markdown_structure_stable() {
    let mut engine = MonitoringEngine::default();
    let ctx = MonitoringContext::new(date(2024, 1, 10));
    let report = engine.run_all(&ctx);

    let reporter = MonitoringReporter::default();
    let md = reporter.to_markdown(&report);

    // Required sections
    assert!(md.contains("## Monitoring Report"));
    assert!(md.contains("### Status:"));
    assert!(md.contains("### Circuit Breaker"));
    assert!(md.contains("| Metric | Count |"));
}

// ============================================================================
// Severity Correctness Tests
// ============================================================================

#[test]
fn freshness_severity_correct() {
    let config = DataHealthConfig::default();
    
    // Info: within threshold
    assert_eq!(
        ThresholdEvaluator::freshness_severity(1, 2, 5),
        Severity::Info
    );
    
    // Warn: between warn and crit
    assert_eq!(
        ThresholdEvaluator::freshness_severity(3, 2, 5),
        Severity::Warn
    );
    
    // Crit: above crit threshold
    assert_eq!(
        ThresholdEvaluator::freshness_severity(6, 2, 5),
        Severity::Crit
    );
}

#[test]
fn coverage_severity_correct() {
    // Info: above warn threshold
    assert_eq!(
        ThresholdEvaluator::coverage_severity(dec!(85), dec!(80), dec!(50)),
        Severity::Info
    );
    
    // Warn: between crit and warn
    assert_eq!(
        ThresholdEvaluator::coverage_severity(dec!(70), dec!(80), dec!(50)),
        Severity::Warn
    );
    
    // Crit: below crit threshold
    assert_eq!(
        ThresholdEvaluator::coverage_severity(dec!(40), dec!(80), dec!(50)),
        Severity::Crit
    );
}

#[test]
fn sigma_severity_correct() {
    // Info: within 2 sigma
    assert_eq!(
        ThresholdEvaluator::sigma_severity(dec!(1.5), dec!(2), dec!(3)),
        Severity::Info
    );
    
    // Warn: between 2 and 3 sigma
    assert_eq!(
        ThresholdEvaluator::sigma_severity(dec!(2.5), dec!(2), dec!(3)),
        Severity::Warn
    );
    
    // Crit: above 3 sigma
    assert_eq!(
        ThresholdEvaluator::sigma_severity(dec!(3.5), dec!(2), dec!(3)),
        Severity::Crit
    );
    
    // Negative sigma also counts
    assert_eq!(
        ThresholdEvaluator::sigma_severity(dec!(-3.5), dec!(2), dec!(3)),
        Severity::Crit
    );
}

// ============================================================================
// Report Structure Tests
// ============================================================================

#[test]
fn report_summary_matches_results() {
    let mut engine = MonitoringEngine::default();
    let ctx = MonitoringContext::new(date(2024, 1, 10));
    let report = engine.run_all(&ctx);

    // Summary should match actual results
    let passed = report.results.iter().filter(|r| r.passed).count();
    let warnings = report.results.iter().filter(|r| r.severity == Severity::Warn).count();
    let criticals = report.results.iter().filter(|r| r.severity == Severity::Crit).count();

    assert_eq!(report.summary.passed, passed);
    assert_eq!(report.summary.warnings, warnings);
    assert_eq!(report.summary.criticals, criticals);
    assert_eq!(report.summary.total_checks, report.results.len());
}

#[test]
fn report_by_category_complete() {
    let mut engine = MonitoringEngine::default();
    let ctx = MonitoringContext::new(date(2024, 1, 10));
    let report = engine.run_all(&ctx);

    // Check all categories are represented
    let data_health = report.by_category(CheckCategory::DataHealth);
    let drift = report.by_category(CheckCategory::Drift);
    let regression = report.by_category(CheckCategory::Regression);

    assert!(!data_health.is_empty(), "Should have DataHealth checks");
    assert!(!drift.is_empty(), "Should have Drift checks");
    assert!(!regression.is_empty(), "Should have Regression checks");
}

#[test]
fn check_result_has_evidence() {
    let mut engine = MonitoringEngine::default();
    let mut ctx = MonitoringContext::new(date(2024, 1, 10));
    ctx.data.last_ohlcv_date.insert(Market::BR, date(2024, 1, 1)); // Stale

    let report = engine.run_all(&ctx);

    for result in &report.results {
        // All results should have evidence source
        assert!(!result.evidence.query_or_source.is_empty() || result.passed);
    }
}

// ============================================================================
// Circuit Breaker Invariants
// ============================================================================

#[test]
fn circuit_breaker_state_transitions() {
    let config = CircuitBreakerConfig::default();
    let mut cb = CircuitBreaker::new(config);

    // Initial state: Closed
    assert_eq!(cb.state(), CircuitState::Closed);

    // Trip -> Open
    cb.trip();
    assert_eq!(cb.state(), CircuitState::Open);

    // Reset -> Closed
    cb.reset();
    assert_eq!(cb.state(), CircuitState::Closed);
}

#[test]
fn circuit_breaker_crit_accumulation() {
    let config = CircuitBreakerConfig {
        halt_on_crit_count: 3,
        ..Default::default()
    };
    let mut cb = CircuitBreaker::new(config);

    // 2 CRITs should not trip
    let two_crits = vec![
        CheckResult::crit("a", CheckCategory::Regression, "test"),
        CheckResult::crit("b", CheckCategory::Regression, "test"),
    ];
    let action = cb.evaluate(&two_crits);
    assert!(!matches!(action, CircuitAction::HaltWithError));

    // 3 CRITs should trip
    let three_crits = vec![
        CheckResult::crit("a", CheckCategory::Regression, "test"),
        CheckResult::crit("b", CheckCategory::Regression, "test"),
        CheckResult::crit("c", CheckCategory::Regression, "test"),
    ];
    cb.reset();
    let action = cb.evaluate(&three_crits);
    assert!(matches!(action, CircuitAction::HaltWithError));
}

#[test]
fn no_trade_flag_set_on_critical() {
    let mut engine = MonitoringEngine::default();
    let mut ctx = MonitoringContext::new(date(2024, 1, 10));
    
    // Trigger critical (stale data)
    ctx.data.last_ohlcv_date.insert(Market::BR, date(2023, 12, 1)); // Very stale

    let report = engine.run_all(&ctx);

    // Should have critical issues
    assert!(report.summary.criticals > 0 || report.summary.halts > 0);
    // NO-TRADE flag should be set
    assert!(report.no_trade || report.action != CircuitAction::Continue);
}

// ============================================================================
// Configuration Invariants
// ============================================================================

#[test]
fn default_config_valid() {
    let config = MonitoringConfig::default();

    // Data health thresholds make sense
    assert!(config.data_health.coverage_min_pct < config.data_health.coverage_warn_pct);
    assert!(config.data_health.freshness_crit(Market::BR) > config.data_health.freshness_warn(Market::BR));

    // Drift thresholds make sense
    assert!(config.drift.sigma_crit > config.drift.sigma_warn);
    assert!(config.drift.selection_overlap_crit < config.drift.selection_overlap_warn);

    // Regression thresholds make sense
    assert!(config.regression.drawdown.crit_pct > config.regression.drawdown.warn_pct);
    assert!(config.regression.drawdown.halt_pct > config.regression.drawdown.crit_pct);
}

#[test]
fn known_limitations_respected() {
    let config = MonitoringConfig::default();

    // US fundamentals missing should be WARN, not CRIT
    assert_eq!(config.known_limitations.us_fundamentals_missing, Severity::Warn);
    
    // US dividends partial is expected
    assert!(config.known_limitations.us_dividends_partial);
}

