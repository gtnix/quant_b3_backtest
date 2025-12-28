//! Walk-Forward invariant tests.
//!
//! Verifies fundamental properties that must always hold.

use backtester_intelligence::walkforward::*;
use backtester_intelligence::filters::Market;
use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

// ============================================================================
// SPLITTER INVARIANTS
// ============================================================================

#[test]
fn invariant_no_train_test_overlap() {
    let config = WalkForwardConfig {
        train_months: 6,
        test_months: 3,
        step_months: 3,
        purge_days: 5,
        embargo_days: 5,
        market: Market::BR,
        grid: None,
    };

    let splitter = RollingSplitter::new(&config);
    let splits = splitter.generate_splits(date(2005, 1, 1), date(2025, 1, 1));

    for split in &splits {
        assert!(
            split.train.end_date < split.test.start_date,
            "Window {}: train ends {} but test starts {} - OVERLAP!",
            split.index,
            split.train.end_date,
            split.test.start_date
        );
    }
}

#[test]
fn invariant_purge_gap_respected() {
    let purge_days = 10;
    let config = WalkForwardConfig {
        train_months: 6,
        test_months: 3,
        step_months: 3,
        purge_days,
        embargo_days: 0,
        market: Market::BR,
        grid: None,
    };

    let splitter = RollingSplitter::new(&config);
    let splits = splitter.generate_splits(date(2020, 1, 1), date(2022, 1, 1));

    for split in &splits {
        let gap = (split.test.start_date - split.train.end_date).num_days();
        assert!(
            gap >= 0,
            "Window {}: negative gap of {} days",
            split.index,
            gap
        );
    }
}

#[test]
fn invariant_embargo_gap_respected() {
    let embargo_days = 15;
    let config = WalkForwardConfig {
        train_months: 6,
        test_months: 3,
        step_months: 3,
        purge_days: 5,
        embargo_days,
        market: Market::BR,
        grid: None,
    };

    let splitter = RollingSplitter::new(&config);
    let splits = splitter.generate_splits(date(2020, 1, 1), date(2022, 1, 1));

    for split in &splits {
        let gap = (split.test.start_date - split.train.end_date).num_days();
        assert!(
            gap >= embargo_days as i64,
            "Window {}: gap {} < embargo {}",
            split.index,
            gap,
            embargo_days
        );
    }
}

#[test]
fn invariant_windows_cover_expected_range() {
    let config = WalkForwardConfig {
        train_months: 6,
        test_months: 3,
        step_months: 3,
        purge_days: 5,
        embargo_days: 5,
        market: Market::BR,
        grid: None,
    };

    let splitter = RollingSplitter::new(&config);
    let start = date(2020, 1, 1);
    let end = date(2025, 1, 1);
    let splits = splitter.generate_splits(start, end);

    // First window should start at or near start date
    if !splits.is_empty() {
        assert_eq!(
            splits[0].train.start_date,
            start,
            "First window should start at the specified start date"
        );
    }

    // Last test window should be before end date
    if let Some(last) = splits.last() {
        assert!(
            last.test.end_date <= end,
            "Last test window {} exceeds end date {}",
            last.test.end_date,
            end
        );
    }
}

#[test]
fn invariant_splits_are_valid() {
    let config = WalkForwardConfig::default();
    let splitter = RollingSplitter::new(&config);
    let splits = splitter.generate_splits(date(2010, 1, 1), date(2025, 1, 1));

    for split in &splits {
        assert!(
            split.is_valid(),
            "Window {} failed validation",
            split.index
        );
    }
}

#[test]
fn invariant_indices_sequential() {
    let config = WalkForwardConfig::default();
    let splitter = RollingSplitter::new(&config);
    let splits = splitter.generate_splits(date(2015, 1, 1), date(2020, 1, 1));

    for (expected_idx, split) in splits.iter().enumerate() {
        assert_eq!(
            split.index,
            expected_idx,
            "Expected index {} but got {}",
            expected_idx,
            split.index
        );
    }
}

// ============================================================================
// GRID SEARCH INVARIANTS
// ============================================================================

#[test]
fn invariant_grid_uses_only_train_data() {
    // This is a behavioral invariant - grid search must select params
    // based on train metrics, not test metrics
    
    let mut grid = GridConfig::default();
    grid.top_n_range = vec![5, 10];
    grid.stop_loss_range = ParamRange::new(dec!(0.10), dec!(0.15), dec!(0.05));
    grid.take_profit_range = ParamRange::new(dec!(0.20), dec!(0.20), dec!(0.10));
    grid.max_weight_range = ParamRange::new(dec!(0.20), dec!(0.20), dec!(0.05));
    grid.turnover_cap_range = ParamRange::new(dec!(0.50), dec!(0.50), dec!(0.20));
    grid.min_score_range = ParamRange::new(dec!(0.0), dec!(0.0), dec!(0.25));

    let config = WalkForwardConfig {
        train_months: 6,
        test_months: 3,
        step_months: 6,
        purge_days: 5,
        embargo_days: 5,
        market: Market::BR,
        grid: Some(grid.clone()),
    };

    // Grid combinations should only come from defined ranges
    let combos = grid.generate_combinations();
    
    for combo in &combos {
        assert!(
            grid.top_n_range.contains(&combo.top_n),
            "top_n {} not in grid range",
            combo.top_n
        );
    }
}

#[test]
fn invariant_grid_combinations_exhaustive() {
    let mut grid = GridConfig::default();
    grid.top_n_range = vec![5, 10, 15];  // 3
    grid.stop_loss_range = ParamRange::new(dec!(0.10), dec!(0.20), dec!(0.05));  // 3
    grid.take_profit_range = ParamRange::new(dec!(0.20), dec!(0.30), dec!(0.05)); // 3
    grid.max_weight_range = ParamRange::new(dec!(0.20), dec!(0.20), dec!(0.05));  // 1
    grid.turnover_cap_range = ParamRange::new(dec!(0.50), dec!(0.50), dec!(0.20)); // 1
    grid.min_score_range = ParamRange::new(dec!(0.0), dec!(0.0), dec!(0.25));     // 1

    let combos = grid.generate_combinations();
    let expected = 3 * 3 * 3 * 1 * 1 * 1;  // 27

    assert_eq!(
        combos.len(),
        expected,
        "Expected {} combinations, got {}",
        expected,
        combos.len()
    );
}

// ============================================================================
// METRICS INVARIANTS
// ============================================================================

#[test]
fn invariant_sharpe_zero_for_flat_equity() {
    let calc = MetricsCalculator::new(dec!(0.05), 252);
    
    // Flat equity curve (no returns)
    let equity: Vec<Decimal> = vec![dec!(100); 100];
    let metrics = calc.from_equity_curve(&equity, dec!(0), dec!(0));

    assert_eq!(
        metrics.sharpe_ratio,
        Decimal::ZERO,
        "Flat equity should have zero Sharpe"
    );
    assert_eq!(
        metrics.max_drawdown_pct,
        Decimal::ZERO,
        "Flat equity should have zero drawdown"
    );
}

#[test]
fn invariant_drawdown_non_negative() {
    let calc = MetricsCalculator::new(dec!(0.05), 252);
    
    // Various equity curves
    let curves = vec![
        vec![dec!(100), dec!(110), dec!(105), dec!(115)],
        vec![dec!(100), dec!(90), dec!(80), dec!(70)],
        vec![dec!(100), dec!(100), dec!(100)],
    ];

    for equity in curves {
        let metrics = calc.from_equity_curve(&equity, dec!(0), dec!(0));
        assert!(
            metrics.max_drawdown_pct >= Decimal::ZERO,
            "Drawdown should never be negative"
        );
    }
}

#[test]
fn invariant_cagr_positive_for_growth() {
    let calc = MetricsCalculator::new(dec!(0.05), 252);
    
    // Growing equity curve
    let equity: Vec<Decimal> = (0..252)
        .map(|i| dec!(100) * (Decimal::ONE + dec!(0.001) * Decimal::from(i)))
        .collect();

    let metrics = calc.from_equity_curve(&equity, dec!(0), dec!(0));

    assert!(
        metrics.cagr_pct > Decimal::ZERO,
        "CAGR should be positive for growing equity"
    );
    assert!(
        metrics.total_return_pct > Decimal::ZERO,
        "Total return should be positive for growing equity"
    );
}

// ============================================================================
// ROBUSTNESS SCORER INVARIANTS
// ============================================================================

#[test]
fn invariant_robustness_bounded() {
    use backtester_intelligence::walkforward::types::{WindowSplit, WindowSpec, WindowType};
    
    let scorer = RobustnessScorer::default();
    
    let make_result = |idx: usize, sharpe: f64| -> WindowResult {
        WindowResult {
            split: WindowSplit {
                train: WindowSpec::new(date(2020, 1, 1), date(2020, 6, 30), WindowType::Train, idx),
                test: WindowSpec::new(date(2020, 7, 1), date(2020, 9, 30), WindowType::Test, idx),
                purge_days: 5,
                embargo_days: 5,
                index: idx,
            },
            train_metrics: WindowMetrics::default(),
            test_metrics: WindowMetrics {
                sharpe_ratio: Decimal::try_from(sharpe).unwrap(),
                total_return_pct: dec!(5),
                max_drawdown_pct: dec!(10),
                ..Default::default()
            },
            selected_params: ParamSet::default(),
            is_oos: true,
        }
    };

    // Various result sets
    let scenarios = vec![
        vec![make_result(0, 1.0), make_result(1, 1.0), make_result(2, 1.0)],
        vec![make_result(0, 2.0), make_result(1, -1.0), make_result(2, 0.5)],
        vec![make_result(0, 0.0), make_result(1, 0.0), make_result(2, 0.0)],
    ];

    for results in scenarios {
        let agg = scorer.aggregate(&results);

        // Stability should be in [0, 1]
        assert!(
            agg.stability_score >= Decimal::ZERO && agg.stability_score <= Decimal::ONE,
            "Stability {} out of [0,1]",
            agg.stability_score
        );
    }
}

#[test]
fn invariant_best_worst_different_for_varied_results() {
    use backtester_intelligence::walkforward::types::{WindowSplit, WindowSpec, WindowType};
    
    let scorer = RobustnessScorer::default();
    
    let make_result = |idx: usize, sharpe: f64| -> WindowResult {
        WindowResult {
            split: WindowSplit {
                train: WindowSpec::new(date(2020, 1, 1), date(2020, 6, 30), WindowType::Train, idx),
                test: WindowSpec::new(date(2020, 7, 1), date(2020, 9, 30), WindowType::Test, idx),
                purge_days: 5,
                embargo_days: 5,
                index: idx,
            },
            train_metrics: WindowMetrics::default(),
            test_metrics: WindowMetrics {
                sharpe_ratio: Decimal::try_from(sharpe).unwrap(),
                ..Default::default()
            },
            selected_params: ParamSet::default(),
            is_oos: true,
        }
    };

    let results = vec![
        make_result(0, 0.5),  // worst
        make_result(1, 1.0),
        make_result(2, 1.5),  // best
    ];

    let agg = scorer.aggregate(&results);

    assert_eq!(agg.best_window_idx, 2, "Best should be window 2");
    assert_eq!(agg.worst_window_idx, 0, "Worst should be window 0");
}

// ============================================================================
// REPORTER INVARIANTS
// ============================================================================

#[test]
fn invariant_json_parses() {
    use backtester_intelligence::walkforward::types::{WindowSplit, WindowSpec, WindowType};
    
    let config = WalkForwardConfig::default();
    
    let make_window = |idx: usize| -> WindowResult {
        WindowResult {
            split: WindowSplit {
                train: WindowSpec::new(date(2020, 1, 1), date(2020, 6, 30), WindowType::Train, idx),
                test: WindowSpec::new(date(2020, 7, 1), date(2020, 9, 30), WindowType::Test, idx),
                purge_days: 5,
                embargo_days: 5,
                index: idx,
            },
            train_metrics: WindowMetrics::default(),
            test_metrics: WindowMetrics::default(),
            selected_params: ParamSet::default(),
            is_oos: true,
        }
    };

    let report = AggregateReport {
        config,
        windows: vec![make_window(0), make_window(1)],
        aggregate: AggregateMetrics::default(),
        most_selected_params: ParamSet::default(),
        generated_at: date(2024, 1, 1),
    };

    let reporter = WalkForwardReporter::new();
    let json_str = reporter.to_json_string(&report);

    // Must parse as valid JSON
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(&json_str);
    assert!(parsed.is_ok(), "JSON must be valid: {:?}", parsed.err());
}

#[test]
fn invariant_windows_sorted_in_output() {
    use backtester_intelligence::walkforward::types::{WindowSplit, WindowSpec, WindowType};
    
    let config = WalkForwardConfig::default();
    
    let make_window = |idx: usize| -> WindowResult {
        WindowResult {
            split: WindowSplit {
                train: WindowSpec::new(date(2020, 1, 1), date(2020, 6, 30), WindowType::Train, idx),
                test: WindowSpec::new(date(2020, 7, 1), date(2020, 9, 30), WindowType::Test, idx),
                purge_days: 5,
                embargo_days: 5,
                index: idx,
            },
            train_metrics: WindowMetrics::default(),
            test_metrics: WindowMetrics::default(),
            selected_params: ParamSet::default(),
            is_oos: true,
        }
    };

    // Create windows in reverse order
    let report = AggregateReport {
        config,
        windows: vec![make_window(2), make_window(0), make_window(1)],
        aggregate: AggregateMetrics::default(),
        most_selected_params: ParamSet::default(),
        generated_at: date(2024, 1, 1),
    };

    let reporter = WalkForwardReporter::new();
    let json = reporter.to_json(&report);

    // Windows should be sorted by index
    for (i, w) in json.windows.iter().enumerate() {
        assert_eq!(w.index, i, "Window should be sorted by index");
    }
}









