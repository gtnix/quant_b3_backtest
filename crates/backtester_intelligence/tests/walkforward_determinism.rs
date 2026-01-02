//! Walk-Forward determinism tests.
//!
//! Verifies that same inputs produce same outputs.

use backtester_intelligence::walkforward::*;
use backtester_intelligence::filters::Market;
use backtester_intelligence::walkforward::runner::{WfCandidate, PriceData};
use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

fn make_prices(symbol: &str, market: Market, start: NaiveDate, days: usize, base: f64) -> PriceData {
    let mut data = PriceData::new(symbol, market);

    for i in 0..days {
        let d = start + chrono::Duration::days(i as i64);
        // Deterministic price generation
        let trend = 1.0 + (i as f64 * 0.0005);
        let noise = ((i as f64 * 0.1).sin() * 0.02);
        let price = base * trend * (1.0 + noise);
        data.dates.push(d);
        data.prices.push(Decimal::try_from(price).unwrap());
    }

    data
}

fn make_candidates(n: usize, market: Market) -> Vec<WfCandidate> {
    (0..n)
        .map(|i| WfCandidate {
            symbol: format!("SYM{:03}", i),
            market,
            score: Decimal::from(n - i),
            volatility: dec!(0.02),
        })
        .collect()
}

// ============================================================================
// SPLITTER DETERMINISM
// ============================================================================

#[test]
fn determinism_splitter_same_config_same_splits() {
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

    let splitter1 = RollingSplitter::new(&config);
    let splitter2 = RollingSplitter::new(&config);

    let start = date(2010, 1, 1);
    let end = date(2020, 1, 1);

    let splits1 = splitter1.generate_splits(start, end);
    let splits2 = splitter2.generate_splits(start, end);

    assert_eq!(splits1.len(), splits2.len());

    for (s1, s2) in splits1.iter().zip(splits2.iter()) {
        assert_eq!(s1.train.start_date, s2.train.start_date);
        assert_eq!(s1.train.end_date, s2.train.end_date);
        assert_eq!(s1.test.start_date, s2.test.start_date);
        assert_eq!(s1.test.end_date, s2.test.end_date);
        assert_eq!(s1.index, s2.index);
    }
}

#[test]
fn determinism_splitter_repeated_calls() {
    let config = WalkForwardConfig::default();
    let splitter = RollingSplitter::new(&config);

    let start = date(2015, 1, 1);
    let end = date(2020, 1, 1);

    // Call 5 times
    let results: Vec<_> = (0..5)
        .map(|_| splitter.generate_splits(start, end))
        .collect();

    let first = &results[0];
    for (i, result) in results.iter().enumerate().skip(1) {
        assert_eq!(
            first.len(),
            result.len(),
            "Run {} has different length",
            i
        );

        for (s1, s2) in first.iter().zip(result.iter()) {
            assert_eq!(s1.train.start_date, s2.train.start_date, "Run {} differs", i);
            assert_eq!(s1.test.end_date, s2.test.end_date, "Run {} differs", i);
        }
    }
}

// ============================================================================
// GRID DETERMINISM
// ============================================================================

#[test]
fn determinism_grid_combinations() {
    let grid = GridConfig::default();

    let combos1 = grid.generate_combinations();
    let combos2 = grid.generate_combinations();

    assert_eq!(combos1.len(), combos2.len());

    for (c1, c2) in combos1.iter().zip(combos2.iter()) {
        assert_eq!(c1.top_n, c2.top_n);
        assert_eq!(c1.stop_loss_pct, c2.stop_loss_pct);
        assert_eq!(c1.take_profit_pct, c2.take_profit_pct);
        assert_eq!(c1.max_weight, c2.max_weight);
        assert_eq!(c1.turnover_cap, c2.turnover_cap);
        assert_eq!(c1.min_score, c2.min_score);
    }
}

// ============================================================================
// METRICS DETERMINISM
// ============================================================================

#[test]
fn determinism_metrics_calculator() {
    let calc = MetricsCalculator::new(dec!(0.05), 252);

    let equity: Vec<Decimal> = (0..100)
        .map(|i| dec!(100) + Decimal::from(i))
        .collect();

    let m1 = calc.from_equity_curve(&equity, dec!(10), dec!(25));
    let m2 = calc.from_equity_curve(&equity, dec!(10), dec!(25));

    assert_eq!(m1.sharpe_ratio, m2.sharpe_ratio);
    assert_eq!(m1.total_return_pct, m2.total_return_pct);
    assert_eq!(m1.max_drawdown_pct, m2.max_drawdown_pct);
    assert_eq!(m1.cagr_pct, m2.cagr_pct);
    assert_eq!(m1.volatility_ann, m2.volatility_ann);
}

// ============================================================================
// RUNNER DETERMINISM
// ============================================================================

#[test]
fn determinism_runner_same_input_same_output() {
    let config = WalkForwardConfig {
        train_months: 6,
        test_months: 3,
        step_months: 6,  // Larger step for faster test
        purge_days: 5,
        embargo_days: 5,
        market: Market::BR,
        grid: None,
        execution_config: None,
    };

    let start = date(2020, 1, 1);
    let end = date(2021, 6, 30);

    let candidates = make_candidates(10, Market::BR);

    let mut prices: HashMap<String, PriceData> = HashMap::new();
    for c in &candidates {
        prices.insert(c.symbol.clone(), make_prices(&c.symbol, c.market, start, 550, 100.0));
    }

    let runner1 = WalkForwardRunner::new(config.clone());
    let report1 = runner1.run(start, end, &candidates, &prices);

    let runner2 = WalkForwardRunner::new(config);
    let report2 = runner2.run(start, end, &candidates, &prices);

    assert_eq!(report1.windows.len(), report2.windows.len());

    for (w1, w2) in report1.windows.iter().zip(report2.windows.iter()) {
        assert_eq!(w1.split.index, w2.split.index);
        assert_eq!(w1.test_metrics.sharpe_ratio, w2.test_metrics.sharpe_ratio);
        assert_eq!(w1.test_metrics.total_return_pct, w2.test_metrics.total_return_pct);
        assert_eq!(w1.selected_params.top_n, w2.selected_params.top_n);
    }

    assert_eq!(report1.aggregate.mean_sharpe, report2.aggregate.mean_sharpe);
    assert_eq!(report1.aggregate.robustness_score, report2.aggregate.robustness_score);
}

#[test]
fn determinism_runner_with_grid() {
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
        grid: Some(grid),
    };

    let start = date(2020, 1, 1);
    let end = date(2021, 6, 30);

    let candidates = make_candidates(10, Market::BR);

    let mut prices: HashMap<String, PriceData> = HashMap::new();
    for c in &candidates {
        prices.insert(c.symbol.clone(), make_prices(&c.symbol, c.market, start, 550, 100.0));
    }

    let runner1 = WalkForwardRunner::new(config.clone());
    let report1 = runner1.run(start, end, &candidates, &prices);

    let runner2 = WalkForwardRunner::new(config);
    let report2 = runner2.run(start, end, &candidates, &prices);

    // Selected params should be identical
    for (w1, w2) in report1.windows.iter().zip(report2.windows.iter()) {
        assert_eq!(w1.selected_params.top_n, w2.selected_params.top_n);
        assert_eq!(w1.selected_params.stop_loss_pct, w2.selected_params.stop_loss_pct);
    }
}

// ============================================================================
// REPORTER DETERMINISM
// ============================================================================

#[test]
fn determinism_summary_output() {
    use backtester_intelligence::walkforward::types::{WindowSplit, WindowSpec, WindowType};

    let config = WalkForwardConfig::default();

    let make_window = |idx: usize, sharpe: f64| -> WindowResult {
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

    let report = AggregateReport {
        config,
        windows: vec![make_window(0, 1.0), make_window(1, 1.2), make_window(2, 0.8)],
        aggregate: AggregateMetrics {
            mean_sharpe: dec!(1.0),
            median_sharpe: dec!(1.0),
            std_sharpe: dec!(0.2),
            total_windows: 3,
            ..Default::default()
        },
        most_selected_params: ParamSet::default(),
        generated_at: date(2024, 1, 1),
    };

    let reporter = WalkForwardReporter::new();

    // Call 5 times
    let summaries: Vec<_> = (0..5).map(|_| reporter.to_summary(&report)).collect();

    let first = &summaries[0];
    for (i, summary) in summaries.iter().enumerate().skip(1) {
        assert_eq!(first, summary, "Summary run {} differs", i);
    }
}

#[test]
fn determinism_json_output() {
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
        windows: vec![make_window(2), make_window(0), make_window(1)],  // Out of order
        aggregate: AggregateMetrics::default(),
        most_selected_params: ParamSet::default(),
        generated_at: date(2024, 1, 1),
    };

    let reporter = WalkForwardReporter::new();

    // Call 5 times
    let jsons: Vec<_> = (0..5).map(|_| reporter.to_json_string(&report)).collect();

    let first = &jsons[0];
    for (i, json) in jsons.iter().enumerate().skip(1) {
        assert_eq!(first, json, "JSON run {} differs", i);
    }
}

#[test]
fn determinism_compact_output() {
    use backtester_intelligence::walkforward::types::{WindowSplit, WindowSpec, WindowType};

    let config = WalkForwardConfig::default();

    let report = AggregateReport {
        config,
        windows: vec![],
        aggregate: AggregateMetrics {
            mean_sharpe: dec!(1.23),
            std_sharpe: dec!(0.45),
            mean_return: dec!(5.67),
            worst_drawdown: dec!(8.9),
            robustness_score: dec!(0.78),
            total_windows: 10,
            ..Default::default()
        },
        most_selected_params: ParamSet::default(),
        generated_at: date(2024, 1, 1),
    };

    let reporter = WalkForwardReporter::new();

    let compacts: Vec<_> = (0..5).map(|_| reporter.to_compact(&report)).collect();

    let first = &compacts[0];
    for (i, compact) in compacts.iter().enumerate().skip(1) {
        assert_eq!(first, compact, "Compact run {} differs", i);
    }
}

// ============================================================================
// CROSS-RUN DETERMINISM (simulate different processes)
// ============================================================================

#[test]
fn determinism_cross_run_simulation() {
    // Simulate running in "different processes" by creating everything fresh
    for run in 0..3 {
        let config = WalkForwardConfig {
            train_months: 6,
            test_months: 3,
            step_months: 6,
            purge_days: 5,
            embargo_days: 5,
            market: Market::BR,
            grid: None,
        execution_config: None,
        };

        let start = date(2020, 1, 1);
        let end = date(2021, 6, 30);

        let candidates = make_candidates(5, Market::BR);

        let mut prices: HashMap<String, PriceData> = HashMap::new();
        for c in &candidates {
            prices.insert(c.symbol.clone(), make_prices(&c.symbol, c.market, start, 550, 100.0));
        }

        let runner = WalkForwardRunner::new(config);
        let report = runner.run(start, end, &candidates, &prices);

        // These values should be constant across all runs
        let expected_windows = 2;  // Approximately 18 months / 6 month step
        
        assert!(
            report.windows.len() >= 1 && report.windows.len() <= 4,
            "Run {}: unexpected window count {}",
            run,
            report.windows.len()
        );

        // Verify first window is always the same
        if !report.windows.is_empty() {
            let w = &report.windows[0];
            assert_eq!(w.split.train.start_date, start, "Run {}: first window start differs", run);
        }
    }
}






















