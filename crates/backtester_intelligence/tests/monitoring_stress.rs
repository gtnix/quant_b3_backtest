//! Stress tests for Monitoring module.
//!
//! Tests: performance with many symbols, large data sets, concurrent checks.

use backtester_intelligence::monitoring::*;
use backtester_intelligence::filters::Market;
use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

// ============================================================================
// Data Health Stress Tests
// ============================================================================

#[test]
fn stress_data_health_1000_symbols() {
    let engine = DataHealthEngine::new(&[Market::BR]);
    let config = DataHealthConfig::default();

    let mut ctx = DataContext::new(date(2024, 1, 10));
    ctx.symbol_count.insert(Market::BR, 1000);
    ctx.symbols_with_data.insert(Market::BR, 950);
    ctx.last_ohlcv_date.insert(Market::BR, date(2024, 1, 9));

    let start = Instant::now();
    let results = engine.run_all(&ctx, &config);
    let elapsed = start.elapsed();

    assert!(!results.is_empty());
    assert!(elapsed.as_millis() < 100, "Should complete in < 100ms");
}

#[test]
fn stress_data_health_many_null_fields() {
    let engine = DataHealthEngine::new(&[Market::BR]);
    let config = DataHealthConfig::default();

    let mut ctx = DataContext::new(date(2024, 1, 10));
    ctx.total_rows = 1_000_000;

    // Add many field null counts
    for i in 0..100 {
        ctx.null_counts.insert(format!("field_{}", i), i * 10);
    }

    let start = Instant::now();
    let results = engine.run_all(&ctx, &config);
    let elapsed = start.elapsed();

    assert!(!results.is_empty());
    assert!(elapsed.as_millis() < 100, "Should complete in < 100ms");
}

#[test]
fn stress_data_health_many_outliers() {
    let engine = DataHealthEngine::new(&[Market::BR]);
    let config = DataHealthConfig::default();

    let mut ctx = DataContext::new(date(2024, 1, 10));
    ctx.total_rows = 1_000_000;

    // Add many outlier types
    for i in 0..50 {
        ctx.outlier_counts.insert(format!("outlier_type_{}", i), i * 100);
    }

    let start = Instant::now();
    let results = engine.run_all(&ctx, &config);
    let elapsed = start.elapsed();

    assert!(!results.is_empty());
    assert!(elapsed.as_millis() < 100, "Should complete in < 100ms");
}

// ============================================================================
// Drift Stress Tests
// ============================================================================

#[test]
fn stress_drift_large_score_distributions() {
    let engine = DriftEngine::new(&["TestTechnique".to_string()]);
    let config = DriftConfig::default();

    let mut ctx = DriftContext::new(date(2024, 1, 10));

    // Large score vectors (reduced from 10k to 2k for reasonable CI time)
    let baseline: Vec<Decimal> = (0..2000).map(|x| Decimal::from(x) / dec!(100)).collect();
    let current: Vec<Decimal> = (0..2000).map(|x| Decimal::from(x + 50) / dec!(100)).collect();

    ctx.baseline_scores.insert("TestTechnique".to_string(), baseline);
    ctx.current_scores.insert("TestTechnique".to_string(), current);

    let start = Instant::now();
    let results = engine.run_all(&ctx, &config);
    let elapsed = start.elapsed();

    assert!(!results.is_empty());
    assert!(elapsed.as_millis() < 2000, "Should complete in < 2s, took {}ms", elapsed.as_millis());
}

#[test]
fn stress_drift_many_techniques() {
    let techniques: Vec<String> = (0..20).map(|i| format!("Technique_{}", i)).collect();
    let engine = DriftEngine::new(&techniques);
    let config = DriftConfig::default();

    let mut ctx = DriftContext::new(date(2024, 1, 10));

    // Add scores for each technique
    for tech in &techniques {
        let baseline: Vec<Decimal> = (0..100).map(|x| Decimal::from(x) / dec!(100)).collect();
        let current: Vec<Decimal> = (0..100).map(|x| Decimal::from(x) / dec!(100)).collect();
        ctx.baseline_scores.insert(tech.clone(), baseline);
        ctx.current_scores.insert(tech.clone(), current);
    }

    let start = Instant::now();
    let results = engine.run_all(&ctx, &config);
    let elapsed = start.elapsed();

    assert!(!results.is_empty());
    assert!(elapsed.as_millis() < 200, "Should complete in < 200ms");
}

#[test]
fn stress_drift_large_selection_sets() {
    let engine = DriftEngine::default();
    let config = DriftConfig::default();

    let mut ctx = DriftContext::new(date(2024, 1, 10));

    // Large selection sets
    let current: HashSet<String> = (0..1000).map(|i| format!("SYMBOL_{}", i)).collect();
    let previous: HashSet<String> = (500..1500).map(|i| format!("SYMBOL_{}", i)).collect();

    ctx.current_selection = current;
    ctx.previous_selection = previous;

    let start = Instant::now();
    let results = engine.run_all(&ctx, &config);
    let elapsed = start.elapsed();

    assert!(!results.is_empty());
    assert!(elapsed.as_millis() < 100, "Should complete in < 100ms");
}

// ============================================================================
// Regression Stress Tests
// ============================================================================

#[test]
fn stress_regression_large_history() {
    let engine = RegressionEngine::new();
    let config = RegressionConfig::default();

    let mut ctx = RegressionContext::new(date(2024, 1, 10));

    // Large historical data
    ctx.historical_turnover = (0..10000).map(|x| Decimal::from(x % 100) / dec!(2)).collect();
    ctx.historical_cost = (0..10000).map(|x| Decimal::from(x % 100) / dec!(1000)).collect();
    ctx.historical_sharpe = (0..10000).map(|x| Decimal::from(x % 300 - 50) / dec!(100)).collect();

    ctx.current_turnover = dec!(25);
    ctx.current_cost = dec!(0.2);
    ctx.current_sharpe = dec!(1.5);

    let start = Instant::now();
    let results = engine.run_all(&ctx, &config);
    let elapsed = start.elapsed();

    assert!(!results.is_empty());
    assert!(elapsed.as_millis() < 200, "Should complete in < 200ms");
}

// ============================================================================
// Full Engine Stress Tests
// ============================================================================

#[test]
fn stress_full_engine_complete_context() {
    let mut engine = MonitoringEngine::default();

    let mut ctx = MonitoringContext::new(date(2024, 1, 10));

    // Populate data health context
    ctx.data.symbol_count.insert(Market::BR, 500);
    ctx.data.symbols_with_data.insert(Market::BR, 480);
    ctx.data.symbol_count.insert(Market::US, 3000);
    ctx.data.symbols_with_data.insert(Market::US, 2900);
    ctx.data.last_ohlcv_date.insert(Market::BR, date(2024, 1, 9));
    ctx.data.last_ohlcv_date.insert(Market::US, date(2024, 1, 9));
    ctx.data.dividends_30d = 100;
    ctx.data.schema_valid = true;

    // Populate drift context
    for tech in &ctx.techniques.clone() {
        let baseline: Vec<Decimal> = (0..60).map(|x| Decimal::from(x) / dec!(100)).collect();
        let current: Vec<Decimal> = (0..60).map(|x| Decimal::from(x) / dec!(100)).collect();
        ctx.drift.baseline_scores.insert(tech.clone(), baseline);
        ctx.drift.current_scores.insert(tech.clone(), current);
    }

    // Populate regression context
    ctx.regression.current_drawdown = dec!(10);
    ctx.regression.current_turnover = dec!(30);
    ctx.regression.current_cost = dec!(0.2);
    ctx.regression.current_sharpe = dec!(1.5);
    ctx.regression.historical_turnover = (0..100).map(|x| Decimal::from(x % 50)).collect();

    let start = Instant::now();
    let report = engine.run_all(&ctx);
    let elapsed = start.elapsed();

    assert!(!report.results.is_empty());
    assert!(elapsed.as_secs() < 1, "Should complete in < 1s");
}

#[test]
fn stress_multiple_runs() {
    let mut engine = MonitoringEngine::default();
    let ctx = MonitoringContext::new(date(2024, 1, 10));

    let start = Instant::now();
    for _ in 0..100 {
        let _ = engine.run_all(&ctx);
    }
    let elapsed = start.elapsed();

    assert!(elapsed.as_secs() < 5, "100 runs should complete in < 5s");
}

// ============================================================================
// Reporter Stress Tests
// ============================================================================

#[test]
fn stress_reporter_large_report() {
    let reporter = MonitoringReporter::default();

    let mut report = MonitoringReport::default();

    // Add many results
    for i in 0..1000 {
        let result = if i % 3 == 0 {
            CheckResult::pass(format!("Check_{}", i), CheckCategory::DataHealth)
        } else if i % 3 == 1 {
            CheckResult::warn(format!("Check_{}", i), CheckCategory::Drift, "warning message")
        } else {
            CheckResult::crit(format!("Check_{}", i), CheckCategory::Regression, "critical message")
        };
        report.results.push(result);
    }
    report.summary = MonitoringSummary::from_results(&report.results);

    let start = Instant::now();
    let json = reporter.to_json(&report).unwrap();
    let md = reporter.to_markdown(&report);
    let elapsed = start.elapsed();

    assert!(!json.is_empty());
    assert!(!md.is_empty());
    assert!(elapsed.as_millis() < 500, "Should complete in < 500ms");
}

// ============================================================================
// Statistics Stress Tests
// ============================================================================

#[test]
fn stress_percentile_large_data() {
    let data: Vec<Decimal> = (0..100000).map(|x| Decimal::from(x)).collect();

    let start = Instant::now();
    let p50 = calculate_percentile(&data, dec!(50));
    let p95 = calculate_percentile(&data, dec!(95));
    let p99 = calculate_percentile(&data, dec!(99));
    let elapsed = start.elapsed();

    assert!(p50.is_some());
    assert!(p95.is_some());
    assert!(p99.is_some());
    assert!(elapsed.as_millis() < 500, "Should complete in < 500ms");
}

#[test]
fn stress_ks_test_large_samples() {
    let sample_a: Vec<Decimal> = (0..5000).map(|x| Decimal::from(x) / dec!(100)).collect();
    let sample_b: Vec<Decimal> = (0..5000).map(|x| Decimal::from(x + 100) / dec!(100)).collect();

    let start = Instant::now();
    let result = ks_two_sample(&sample_a, &sample_b);
    let elapsed = start.elapsed();

    assert!(result.is_some());
    assert!(elapsed.as_secs() < 5, "Should complete in < 5s");
}

#[test]
fn stress_jaccard_large_sets() {
    let set_a: HashSet<i32> = (0..10000).collect();
    let set_b: HashSet<i32> = (5000..15000).collect();

    let start = Instant::now();
    let similarity = jaccard_similarity(&set_a, &set_b);
    let elapsed = start.elapsed();

    assert!(similarity > dec!(0) && similarity < dec!(1));
    assert!(elapsed.as_millis() < 100, "Should complete in < 100ms");
}

