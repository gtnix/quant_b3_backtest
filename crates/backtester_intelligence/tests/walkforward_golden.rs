//! Walk-Forward golden snapshot tests.
//!
//! Verifies output format stability.

use chrono::Datelike;
use backtester_intelligence::walkforward::*;
use backtester_intelligence::filters::Market;
use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

fn make_golden_report() -> AggregateReport {
    use backtester_intelligence::walkforward::types::{WindowSplit, WindowSpec, WindowType};

    let config = WalkForwardConfig {
        train_months: 6,
        test_months: 3,
        step_months: 3,
        purge_days: 5,
        embargo_days: 5,
        market: Market::BR,
        grid: None,
        execution_config: None,
    };

    let make_window = |idx: usize, train_end: NaiveDate, test_end: NaiveDate, sharpe: f64, ret: f64, dd: f64| -> WindowResult {
        WindowResult {
            split: WindowSplit {
                train: WindowSpec::new(
                    date(2020, 1, 1),
                    train_end,
                    WindowType::Train,
                    idx,
                ),
                test: WindowSpec::new(
                    train_end + chrono::Duration::days(10),
                    test_end,
                    WindowType::Test,
                    idx,
                ),
                purge_days: 5,
                embargo_days: 5,
                index: idx,
            },
            train_metrics: WindowMetrics {
                sharpe_ratio: Decimal::try_from(sharpe + 0.3).unwrap(),
                total_return_pct: Decimal::try_from(ret + 2.0).unwrap(),
                ..Default::default()
            },
            test_metrics: WindowMetrics {
                sharpe_ratio: Decimal::try_from(sharpe).unwrap(),
                total_return_pct: Decimal::try_from(ret).unwrap(),
                max_drawdown_pct: Decimal::try_from(dd).unwrap(),
                volatility_ann: dec!(15.5),
                cagr_pct: Decimal::try_from(ret * 4.0).unwrap(),
                turnover_avg_pct: dec!(25),
                total_costs: dec!(150),
                dd_duration_days: 15,
                hit_rate: None,
                ..Default::default()
            },
            selected_params: ParamSet {
                top_n: 10,
                stop_loss_pct: dec!(0.15),
                take_profit_pct: dec!(0.30),
                max_weight: dec!(0.20),
                turnover_cap: dec!(0.50),
                min_score: dec!(0.0),
            },
            is_oos: true,
        }
    };

    let windows = vec![
        make_window(0, date(2020, 6, 25), date(2020, 9, 30), 1.20, 5.50, 8.20),
        make_window(1, date(2020, 9, 25), date(2020, 12, 31), 0.85, 3.20, 12.10),
        make_window(2, date(2020, 12, 26), date(2021, 3, 31), 1.45, 7.10, 6.50),
    ];

    let aggregate = AggregateMetrics {
        mean_sharpe: dec!(1.167),
        median_sharpe: dec!(1.200),
        std_sharpe: dec!(0.247),
        mean_return: dec!(5.267),
        median_return: dec!(5.500),
        std_return: dec!(1.602),
        mean_drawdown: dec!(8.933),
        worst_drawdown: dec!(12.100),
        mean_volatility: dec!(15.500),
        stability_score: dec!(0.789),
        robustness_score: dec!(0.732),
        best_window_idx: 2,
        worst_window_idx: 1,
        total_windows: 3,
        total_months_tested: 9,
        ..Default::default()
    };

    AggregateReport {
        config,
        windows,
        aggregate,
        most_selected_params: ParamSet {
            top_n: 10,
            stop_loss_pct: dec!(0.15),
            take_profit_pct: dec!(0.30),
            max_weight: dec!(0.20),
            turnover_cap: dec!(0.50),
            min_score: dec!(0.0),
        },
        generated_at: date(2024, 1, 15),
    }
}

// ============================================================================
// GOLDEN SUMMARY TEST
// ============================================================================

const GOLDEN_SUMMARY_HEADER: &str = "WALK-FORWARD VALIDATION REPORT";
const GOLDEN_SUMMARY_CONFIG: &str = "Train period: 6 months";
const GOLDEN_SUMMARY_WINDOWS: &str = "Total windows: 3";
const GOLDEN_SUMMARY_SHARPE: &str = "Mean: 1.167";
const GOLDEN_SUMMARY_ROBUSTNESS: &str = "Robustness: 0.732";

#[test]
fn golden_summary_contains_required_sections() {
    let report = make_golden_report();
    let reporter = WalkForwardReporter::new();
    let summary = reporter.to_summary(&report);

    // Verify required sections exist
    assert!(summary.contains(GOLDEN_SUMMARY_HEADER), "Missing header");
    assert!(summary.contains(GOLDEN_SUMMARY_CONFIG), "Missing config section");
    assert!(summary.contains(GOLDEN_SUMMARY_WINDOWS), "Missing window count");
    assert!(summary.contains(GOLDEN_SUMMARY_SHARPE), "Missing Sharpe mean");
    assert!(summary.contains(GOLDEN_SUMMARY_ROBUSTNESS), "Missing robustness score");

    // Verify structure
    assert!(summary.contains("Configuration:"));
    assert!(summary.contains("AGGREGATE METRICS"));
    assert!(summary.contains("MOST SELECTED PARAMETERS"));
    assert!(summary.contains("PER-WINDOW RESULTS"));
}

#[test]
fn golden_summary_window_table() {
    let report = make_golden_report();
    let reporter = WalkForwardReporter::new();
    let summary = reporter.to_summary(&report);

    // Verify window table headers
    assert!(summary.contains("Train End"));
    assert!(summary.contains("Test End"));
    assert!(summary.contains("Sharpe"));
    assert!(summary.contains("Return%"));
    assert!(summary.contains("MaxDD%"));

    // Verify window data rows exist
    assert!(summary.contains("2020-06-25"));
    assert!(summary.contains("2020-09-30"));
}

// ============================================================================
// GOLDEN JSON TEST
// ============================================================================

#[test]
fn golden_json_structure() {
    let report = make_golden_report();
    let reporter = WalkForwardReporter::new();
    let json_str = reporter.to_json_string(&report);

    // Parse JSON
    let json: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    // Verify top-level structure
    assert!(json.get("config").is_some(), "Missing config");
    assert!(json.get("windows").is_some(), "Missing windows");
    assert!(json.get("aggregate").is_some(), "Missing aggregate");
    assert!(json.get("params_selected").is_some(), "Missing params_selected");

    // Verify config fields
    let config = json.get("config").unwrap();
    assert_eq!(config.get("train_months").unwrap().as_u64(), Some(6));
    assert_eq!(config.get("test_months").unwrap().as_u64(), Some(3));
    assert_eq!(config.get("market").unwrap().as_str(), Some("BR"));

    // Verify windows array
    let windows = json.get("windows").unwrap().as_array().unwrap();
    assert_eq!(windows.len(), 3);

    // Verify first window
    let w0 = &windows[0];
    assert_eq!(w0.get("index").unwrap().as_u64(), Some(0));
    assert!(w0.get("train_period").is_some());
    assert!(w0.get("test_period").is_some());
    assert!(w0.get("test_sharpe").is_some());
    assert!(w0.get("params").is_some());

    // Verify aggregate fields
    let agg = json.get("aggregate").unwrap();
    assert!(agg.get("mean_sharpe").is_some());
    assert!(agg.get("robustness_score").is_some());
    assert_eq!(agg.get("total_windows").unwrap().as_u64(), Some(3));
}

#[test]
fn golden_json_values() {
    let report = make_golden_report();
    let reporter = WalkForwardReporter::new();
    let json_str = reporter.to_json_string(&report);

    let json: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    // Verify specific values
    let agg = json.get("aggregate").unwrap();
    assert_eq!(agg.get("mean_sharpe").unwrap().as_str(), Some("1.1670"));
    assert_eq!(agg.get("median_sharpe").unwrap().as_str(), Some("1.2000"));
    assert_eq!(agg.get("robustness_score").unwrap().as_str(), Some("0.7320"));
    assert_eq!(agg.get("best_window_idx").unwrap().as_u64(), Some(2));
    assert_eq!(agg.get("worst_window_idx").unwrap().as_u64(), Some(1));

    // Verify params
    let params = json.get("params_selected").unwrap();
    assert_eq!(params.get("top_n").unwrap().as_u64(), Some(10));
    assert_eq!(params.get("stop_loss_pct").unwrap().as_str(), Some("15.00"));
}

#[test]
fn golden_json_windows_sorted() {
    let report = make_golden_report();
    let reporter = WalkForwardReporter::new();
    let json = reporter.to_json(&report);

    // Windows must be sorted by index
    for (i, w) in json.windows.iter().enumerate() {
        assert_eq!(w.index, i, "Window {} should have index {}", i, i);
    }
}

// ============================================================================
// GOLDEN COMPACT TEST
// ============================================================================

#[test]
fn golden_compact_format() {
    let report = make_golden_report();
    let reporter = WalkForwardReporter::new();
    let compact = reporter.to_compact(&report);

    // Verify compact format
    assert!(compact.starts_with("WF [BR]"), "Should start with WF [market]");
    assert!(compact.contains("3 windows"), "Should contain window count");
    assert!(compact.contains("Sharpe:"), "Should contain Sharpe");
    assert!(compact.contains("Return:"), "Should contain Return");
    assert!(compact.contains("MaxDD:"), "Should contain MaxDD");
    assert!(compact.contains("Robustness:"), "Should contain Robustness");
}

// ============================================================================
// DETERMINISM WITHIN GOLDEN
// ============================================================================

#[test]
fn golden_determinism_summary() {
    let report = make_golden_report();
    let reporter = WalkForwardReporter::new();

    let s1 = reporter.to_summary(&report);
    let s2 = reporter.to_summary(&report);
    let s3 = reporter.to_summary(&report);

    assert_eq!(s1, s2);
    assert_eq!(s2, s3);
}

#[test]
fn golden_determinism_json() {
    let report = make_golden_report();
    let reporter = WalkForwardReporter::new();

    let j1 = reporter.to_json_string(&report);
    let j2 = reporter.to_json_string(&report);
    let j3 = reporter.to_json_string(&report);

    assert_eq!(j1, j2);
    assert_eq!(j2, j3);
}

#[test]
fn golden_determinism_compact() {
    let report = make_golden_report();
    let reporter = WalkForwardReporter::new();

    let c1 = reporter.to_compact(&report);
    let c2 = reporter.to_compact(&report);
    let c3 = reporter.to_compact(&report);

    assert_eq!(c1, c2);
    assert_eq!(c2, c3);
}

// ============================================================================
// HELPER: Generate golden output for manual update
// ============================================================================

#[test]
#[ignore]
fn generate_golden_output() {
    let report = make_golden_report();
    let reporter = WalkForwardReporter::new();

    println!("=== GOLDEN SUMMARY ===\n");
    println!("{}", reporter.to_summary(&report));

    println!("\n=== GOLDEN COMPACT ===\n");
    println!("{}", reporter.to_compact(&report));

    println!("\n=== GOLDEN JSON ===\n");
    println!("{}", reporter.to_json_string(&report));
}

// ============================================================================
// BACKWARD COMPATIBILITY
// ============================================================================

#[test]
fn golden_backward_compat_window_result_fields() {
    let report = make_golden_report();

    // Verify WindowResult has all expected fields
    let w = &report.windows[0];
    
    // Split fields
    assert!(w.split.train.start_date.year() > 2000);
    assert!(w.split.train.end_date.year() > 2000);
    assert!(w.split.test.start_date.year() > 2000);
    assert!(w.split.test.end_date.year() > 2000);
    assert!(w.split.purge_days > 0);
    assert!(w.split.embargo_days > 0);

    // Metrics fields
    assert!(w.test_metrics.sharpe_ratio != Decimal::ZERO);
    assert!(w.test_metrics.total_return_pct != Decimal::ZERO);
    assert!(w.test_metrics.max_drawdown_pct != Decimal::ZERO);

    // Params fields
    assert!(w.selected_params.top_n > 0);
    assert!(w.selected_params.stop_loss_pct > Decimal::ZERO);

    // Flags
    assert!(w.is_oos);
}

#[test]
fn golden_backward_compat_aggregate_fields() {
    let report = make_golden_report();
    let agg = &report.aggregate;

    // All required fields present and valid
    assert!(agg.total_windows > 0);
    assert!(agg.mean_sharpe != Decimal::ZERO);
    assert!(agg.median_sharpe != Decimal::ZERO);
    assert!(agg.robustness_score >= Decimal::ZERO);
    assert!(agg.stability_score >= Decimal::ZERO);
    assert!(agg.best_window_idx < agg.total_windows);
    assert!(agg.worst_window_idx < agg.total_windows);
}

#[test]
fn golden_backward_compat_config_fields() {
    let report = make_golden_report();
    let config = &report.config;

    // All required config fields
    assert!(config.train_months > 0);
    assert!(config.test_months > 0);
    assert!(config.step_months > 0);
    assert!(config.purge_days >= 0);
    assert!(config.embargo_days >= 0);
}

// ============================================================================
// NESTED 3-SEGMENT GOLDEN TESTS
// ============================================================================

fn make_golden_nested_report() -> NestedAggregateReport {
    use backtester_intelligence::walkforward::types::{
        NestedWindowSplit, NestedWalkForwardConfig, NestedWindowResult,
        WindowSpec, WindowType, SelectionReason, SelectionCriteria, PenaltyConfig,
    };

    let config = NestedWalkForwardConfig {
        train_months: 4,
        val_months: 1,
        test_months: 1,
        step_months: 3,
        purge_days: 5,
        embargo_days: 5,
        market: Market::BR,
        grid: None,
        execution_config: None,
        selection_criteria: SelectionCriteria::PSR,
        psr_threshold: dec!(0.5),
        penalties: PenaltyConfig::default(),
        gates: None,
    };

    let make_nested_window = |idx: usize, train_sharpe: f64, val_sharpe: f64, test_sharpe: f64, psr: f64, dsr: Option<f64>| -> NestedWindowResult {
        let train_start = date(2020, 1, 1) + chrono::Duration::days((idx * 90) as i64);
        let train_end = train_start + chrono::Duration::days(115);
        let val_start = train_end + chrono::Duration::days(10);
        let val_end = val_start + chrono::Duration::days(25);
        let test_start = val_end + chrono::Duration::days(10);
        let test_end = test_start + chrono::Duration::days(25);

        NestedWindowResult {
            split: NestedWindowSplit {
                train: WindowSpec::new(train_start, train_end, WindowType::Train, idx),
                val: WindowSpec::new(val_start, val_end, WindowType::Validation, idx),
                test: WindowSpec::new(test_start, test_end, WindowType::Test, idx),
                purge_train_val: 5,
                purge_val_test: 5,
                embargo_days: 5,
                index: idx,
            },
            metrics_train: WindowMetrics {
                sharpe_ratio: Decimal::try_from(train_sharpe).unwrap(),
                total_return_pct: dec!(8.5),
                max_drawdown_pct: dec!(6.0),
                skewness: dec!(0.15),
                kurtosis: dec!(0.8),
                n_observations: 85,
                ..Default::default()
            },
            metrics_val: WindowMetrics {
                sharpe_ratio: Decimal::try_from(val_sharpe).unwrap(),
                total_return_pct: dec!(2.1),
                max_drawdown_pct: dec!(4.5),
                skewness: dec!(0.10),
                kurtosis: dec!(0.5),
                n_observations: 22,
                psr: Some(Decimal::try_from(psr).unwrap()),
                dsr: dsr.map(|d| Decimal::try_from(d).unwrap()),
                ..Default::default()
            },
            metrics_test: WindowMetrics {
                sharpe_ratio: Decimal::try_from(test_sharpe).unwrap(),
                total_return_pct: dec!(1.8),
                max_drawdown_pct: dec!(5.2),
                skewness: dec!(0.12),
                kurtosis: dec!(0.6),
                n_observations: 22,
                ..Default::default()
            },
            selected_params: ParamSet {
                top_n: 10,
                stop_loss_pct: dec!(0.15),
                take_profit_pct: dec!(0.30),
                max_weight: dec!(0.20),
                turnover_cap: dec!(0.50),
                min_score: dec!(0.0),
            },
            selection_reason: SelectionReason {
                criteria: SelectionCriteria::PSR,
                primary_score: Decimal::try_from(val_sharpe).unwrap(),
                psr: Decimal::try_from(psr).unwrap(),
                dsr: dsr.map(|d| Decimal::try_from(d).unwrap()),
                turnover_penalty: dec!(0),
                cost_penalty: dec!(0),
                drawdown_penalty: dec!(0),
                slippage_penalty: dec!(0),
                capacity_penalty: dec!(0),
                final_score: Decimal::try_from(psr).unwrap(),
                tiebreaker_used: None,
            },
            psr_val: Decimal::try_from(psr).unwrap(),
            dsr_val: dsr.map(|d| Decimal::try_from(d).unwrap()),
            n_trials: 24,
        }
    };

    let windows = vec![
        make_nested_window(0, 1.50, 1.20, 1.05, 0.72, Some(0.65)),
        make_nested_window(1, 1.35, 0.95, 0.88, 0.58, Some(0.52)),
        make_nested_window(2, 1.65, 1.40, 1.22, 0.81, Some(0.74)),
    ];

    let aggregate = AggregateMetrics {
        mean_sharpe: dec!(1.05),
        median_sharpe: dec!(1.05),
        std_sharpe: dec!(0.14),
        mean_return: dec!(1.80),
        median_return: dec!(1.80),
        std_return: dec!(0.25),
        mean_drawdown: dec!(5.20),
        worst_drawdown: dec!(5.20),
        mean_volatility: dec!(14.0),
        stability_score: dec!(0.82),
        robustness_score: dec!(0.76),
        best_window_idx: 2,
        worst_window_idx: 1,
        total_windows: 3,
        total_months_tested: 3,
        mean_psr: dec!(0.703),
        median_psr: dec!(0.72),
        mean_dsr: Some(dec!(0.637)),
        median_dsr: Some(dec!(0.65)),
        oos_sharpe_mean: dec!(1.05),
        oos_return_mean: dec!(1.80),
        oos_psr_mean: dec!(0.703),
    };

    NestedAggregateReport {
        config,
        windows,
        aggregate,
        most_selected_params: ParamSet {
            top_n: 10,
            stop_loss_pct: dec!(0.15),
            take_profit_pct: dec!(0.30),
            max_weight: dec!(0.20),
            turnover_cap: dec!(0.50),
            min_score: dec!(0.0),
        },
        generated_at: date(2024, 1, 15),
    }
}

// ============================================================================
// NESTED SUMMARY TESTS
// ============================================================================

#[test]
fn golden_nested_summary_contains_required_sections() {
    let report = make_golden_nested_report();
    let reporter = WalkForwardReporter::new();
    let summary = reporter.to_summary_nested(&report);

    // Header
    assert!(summary.contains("NESTED WALK-FORWARD VALIDATION REPORT"), "Missing nested header");
    assert!(summary.contains("Train/Val/Test"), "Missing 3-segment indicator");

    // Config with all 3 periods
    assert!(summary.contains("Train period: 4 months"), "Missing train months");
    assert!(summary.contains("Validation period: 1 months"), "Missing val months");
    assert!(summary.contains("Test period: 1 months"), "Missing test months");
    assert!(summary.contains("PSR Threshold:"), "Missing PSR threshold");
    assert!(summary.contains("Selection:"), "Missing selection criteria");

    // Aggregate sections
    assert!(summary.contains("AGGREGATE METRICS"), "Missing aggregate section");
    assert!(summary.contains("PSR (Validation):"), "Missing PSR section");
    assert!(summary.contains("Sharpe Ratio (OOS):"), "Missing OOS Sharpe section");
}

#[test]
fn golden_nested_summary_window_table() {
    let report = make_golden_nested_report();
    let reporter = WalkForwardReporter::new();
    let summary = reporter.to_summary_nested(&report);

    // Table headers for 3-segment
    assert!(summary.contains("TrainSR"), "Missing TrainSR header");
    assert!(summary.contains("ValSR"), "Missing ValSR header");
    assert!(summary.contains("TestSR"), "Missing TestSR header");
    assert!(summary.contains("PSR"), "Missing PSR header");
    assert!(summary.contains("DSR"), "Missing DSR header");
    assert!(summary.contains("Trials"), "Missing Trials header");
    assert!(summary.contains("Selection"), "Missing Selection header");

    // Window data
    assert!(summary.contains("0.72") || summary.contains("0.720"), "Missing PSR value 0.72");
}

// ============================================================================
// NESTED JSON TESTS
// ============================================================================

#[test]
fn golden_nested_json_structure() {
    let report = make_golden_nested_report();
    let reporter = WalkForwardReporter::new();
    let json_str = reporter.to_json_string_nested(&report);

    let json: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    // Top-level structure
    assert!(json.get("config").is_some(), "Missing config");
    assert!(json.get("windows").is_some(), "Missing windows");
    assert!(json.get("aggregate").is_some(), "Missing aggregate");
    assert!(json.get("params_selected").is_some(), "Missing params_selected");

    // Config has val_months (nested-specific)
    let config = json.get("config").unwrap();
    assert_eq!(config.get("train_months").unwrap().as_u64(), Some(4));
    assert_eq!(config.get("val_months").unwrap().as_u64(), Some(1));
    assert_eq!(config.get("test_months").unwrap().as_u64(), Some(1));
    assert!(config.get("selection_criteria").is_some(), "Missing selection_criteria");
    assert!(config.get("psr_threshold").is_some(), "Missing psr_threshold");

    // Windows have 3-segment data
    let windows = json.get("windows").unwrap().as_array().unwrap();
    assert_eq!(windows.len(), 3);

    let w0 = &windows[0];
    assert!(w0.get("train_period").is_some(), "Missing train_period");
    assert!(w0.get("val_period").is_some(), "Missing val_period");
    assert!(w0.get("test_period").is_some(), "Missing test_period");
    assert!(w0.get("train_metrics").is_some(), "Missing train_metrics");
    assert!(w0.get("val_metrics").is_some(), "Missing val_metrics");
    assert!(w0.get("test_metrics").is_some(), "Missing test_metrics");
    assert!(w0.get("psr_val").is_some(), "Missing psr_val");
    assert!(w0.get("dsr_val").is_some(), "Missing dsr_val");
    assert!(w0.get("n_trials").is_some(), "Missing n_trials");
    assert!(w0.get("selection_reason").is_some(), "Missing selection_reason");

    // Aggregate has OOS stats
    let agg = json.get("aggregate").unwrap();
    assert!(agg.get("oos_sharpe_mean").is_some(), "Missing oos_sharpe_mean");
    assert!(agg.get("oos_psr_mean").is_some(), "Missing oos_psr_mean");
    assert!(agg.get("mean_psr").is_some(), "Missing mean_psr");
    assert!(agg.get("mean_dsr").is_some(), "Missing mean_dsr");
}

#[test]
fn golden_nested_json_values() {
    let report = make_golden_nested_report();
    let reporter = WalkForwardReporter::new();
    let json_str = reporter.to_json_string_nested(&report);

    let json: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    // Aggregate values
    let agg = json.get("aggregate").unwrap();
    assert_eq!(agg.get("oos_sharpe_mean").unwrap().as_str(), Some("1.0500"));
    assert_eq!(agg.get("mean_psr").unwrap().as_str(), Some("0.7030"));
    assert_eq!(agg.get("total_windows").unwrap().as_u64(), Some(3));

    // Window PSR/DSR values
    let windows = json.get("windows").unwrap().as_array().unwrap();
    let w0 = &windows[0];
    assert_eq!(w0.get("psr_val").unwrap().as_str(), Some("0.7200"));
    assert_eq!(w0.get("dsr_val").unwrap().as_str(), Some("0.6500"));
    assert_eq!(w0.get("n_trials").unwrap().as_u64(), Some(24));
}

#[test]
fn golden_nested_json_windows_sorted() {
    let report = make_golden_nested_report();
    let reporter = WalkForwardReporter::new();
    let json = reporter.to_json_nested(&report);

    for (i, w) in json.windows.iter().enumerate() {
        assert_eq!(w.index, i, "Window {} should have index {}", i, i);
    }
}

// ============================================================================
// NESTED DETERMINISM TESTS
// ============================================================================

#[test]
fn golden_nested_determinism_summary() {
    let report = make_golden_nested_report();
    let reporter = WalkForwardReporter::new();

    let s1 = reporter.to_summary_nested(&report);
    let s2 = reporter.to_summary_nested(&report);
    let s3 = reporter.to_summary_nested(&report);

    assert_eq!(s1, s2);
    assert_eq!(s2, s3);
}

#[test]
fn golden_nested_determinism_json() {
    let report = make_golden_nested_report();
    let reporter = WalkForwardReporter::new();

    let j1 = reporter.to_json_string_nested(&report);
    let j2 = reporter.to_json_string_nested(&report);
    let j3 = reporter.to_json_string_nested(&report);

    assert_eq!(j1, j2);
    assert_eq!(j2, j3);
}

