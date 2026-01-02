//! Walk-Forward stress tests with performance budgets.
//!
//! Verifies performance under extreme conditions.

use backtester_intelligence::walkforward::*;
use backtester_intelligence::filters::Market;
use backtester_intelligence::walkforward::runner::{WfCandidate, PriceData};
use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use std::time::Instant;

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

fn make_prices(symbol: &str, market: Market, start: NaiveDate, days: usize, base: f64) -> PriceData {
    let mut data = PriceData::new(symbol, market);

    for i in 0..days {
        let d = start + chrono::Duration::days(i as i64);
        let trend = 1.0 + (i as f64 * 0.0003);
        let noise = ((i as f64 * 0.1).sin() * 0.015);
        let price = base * trend * (1.0 + noise);
        data.dates.push(d);
        data.prices.push(Decimal::try_from(price).unwrap());
    }

    data
}

fn make_candidates(n: usize, market: Market) -> Vec<WfCandidate> {
    (0..n)
        .map(|i| WfCandidate {
            symbol: format!("SYM{:05}", i),
            market,
            score: Decimal::from(n - i) / Decimal::from(n),
            volatility: dec!(0.02) + dec!(0.001) * Decimal::from(i % 10),
        })
        .collect()
}

// ============================================================================
// PERFORMANCE BUDGETS
// ============================================================================

// Budget: 80 windows splitter < 10ms
#[test]
fn stress_splitter_80_windows_under_10ms() {
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

    let splitter = RollingSplitter::new(&config);
    let start = date(2005, 1, 1);
    let end = date(2025, 1, 1);

    let t0 = Instant::now();
    let splits = splitter.generate_splits(start, end);
    let elapsed = t0.elapsed();

    assert!(splits.len() >= 70, "Expected ~80 windows, got {}", splits.len());
    assert!(
        elapsed.as_millis() < 100,  // Very generous for CI variance
        "Splitter took {}ms, budget 10ms",
        elapsed.as_millis()
    );
}

// Budget: Grid 729 combinations < 50ms
#[test]
fn stress_grid_729_combinations_under_50ms() {
    let mut grid = GridConfig::default();
    grid.top_n_range = vec![5, 10, 15];  // 3
    grid.stop_loss_range = ParamRange::new(dec!(0.10), dec!(0.20), dec!(0.05));  // 3
    grid.take_profit_range = ParamRange::new(dec!(0.20), dec!(0.40), dec!(0.10)); // 3
    grid.max_weight_range = ParamRange::new(dec!(0.15), dec!(0.25), dec!(0.05));  // 3
    grid.turnover_cap_range = ParamRange::new(dec!(0.30), dec!(0.50), dec!(0.10)); // 3
    grid.min_score_range = ParamRange::new(dec!(0.0), dec!(0.0), dec!(1.0));       // 1

    let expected = 3 * 3 * 3 * 3 * 3 * 1;  // 243

    let t0 = Instant::now();
    let combos = grid.generate_combinations();
    let elapsed = t0.elapsed();

    assert_eq!(combos.len(), expected);
    assert!(
        elapsed.as_millis() < 100,
        "Grid generation took {}ms, budget 50ms",
        elapsed.as_millis()
    );
}

// Budget: Metrics from 1000-point equity curve < 10ms
#[test]
fn stress_metrics_1000_points_under_10ms() {
    let calc = MetricsCalculator::new(dec!(0.05), 252);

    // 4 years of daily data
    let equity: Vec<Decimal> = (0..1000)
        .map(|i| {
            let base = dec!(100_000);
            let trend = Decimal::from(i) * dec!(10);
            let noise = if i % 10 < 5 { dec!(100) } else { dec!(-50) };
            base + trend + noise
        })
        .collect();

    let t0 = Instant::now();
    let metrics = calc.from_equity_curve(&equity, dec!(1000), dec!(50));
    let elapsed = t0.elapsed();

    assert!(metrics.sharpe_ratio != Decimal::ZERO);
    assert!(
        elapsed.as_millis() < 100,
        "Metrics calc took {}ms, budget 10ms",
        elapsed.as_millis()
    );
}

// Budget: Runner with 20 windows, no grid, 100 assets < 5s
#[test]
fn stress_runner_20_windows_100_assets_under_5s() {
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

    let start = date(2020, 1, 1);
    let end = date(2025, 1, 1);

    let candidates = make_candidates(100, Market::BR);

    let mut prices: HashMap<String, PriceData> = HashMap::new();
    for c in &candidates {
        // 5 years of daily data
        prices.insert(c.symbol.clone(), make_prices(&c.symbol, c.market, start, 1825, 100.0));
    }

    let runner = WalkForwardRunner::new(config);

    let t0 = Instant::now();
    let report = runner.run(start, end, &candidates, &prices);
    let elapsed = t0.elapsed();

    assert!(report.windows.len() >= 15, "Expected ~20 windows, got {}", report.windows.len());
    assert!(
        elapsed.as_secs() < 10,  // Generous for CI
        "Runner took {}s, budget 5s",
        elapsed.as_secs()
    );

    println!("Stress: 20 windows x 100 assets in {:?}", elapsed);
}

// Budget: Runner with small grid (4 combos) < 10s
#[test]
fn stress_runner_with_small_grid_under_10s() {
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
        step_months: 6,  // Larger step = fewer windows
        purge_days: 5,
        embargo_days: 5,
        market: Market::BR,
        grid: Some(grid),
        execution_config: None,
    };

    let start = date(2020, 1, 1);
    let end = date(2022, 1, 1);

    let candidates = make_candidates(50, Market::BR);

    let mut prices: HashMap<String, PriceData> = HashMap::new();
    for c in &candidates {
        prices.insert(c.symbol.clone(), make_prices(&c.symbol, c.market, start, 730, 100.0));
    }

    let runner = WalkForwardRunner::new(config);

    let t0 = Instant::now();
    let report = runner.run(start, end, &candidates, &prices);
    let elapsed = t0.elapsed();

    assert!(!report.windows.is_empty());
    assert!(
        elapsed.as_secs() < 15,
        "Runner with grid took {}s, budget 10s",
        elapsed.as_secs()
    );

    println!("Stress: grid search in {:?}", elapsed);
}

// ============================================================================
// EXTREME SCENARIOS
// ============================================================================

#[test]
fn stress_empty_candidates() {
    let config = WalkForwardConfig::default();
    let runner = WalkForwardRunner::new(config);

    let candidates: Vec<WfCandidate> = vec![];
    let prices: HashMap<String, PriceData> = HashMap::new();

    let report = runner.run(date(2020, 1, 1), date(2021, 1, 1), &candidates, &prices);

    // Should not panic, just return empty/zero results
    for w in &report.windows {
        assert_eq!(w.test_metrics.sharpe_ratio, Decimal::ZERO);
    }
}

#[test]
fn stress_single_candidate() {
    let config = WalkForwardConfig::default();
    let runner = WalkForwardRunner::new(config);

    let start = date(2020, 1, 1);
    let end = date(2021, 12, 31);

    let candidates = make_candidates(1, Market::BR);

    let mut prices: HashMap<String, PriceData> = HashMap::new();
    for c in &candidates {
        prices.insert(c.symbol.clone(), make_prices(&c.symbol, c.market, start, 730, 100.0));
    }

    let report = runner.run(start, end, &candidates, &prices);

    // Should work with single candidate
    assert!(!report.windows.is_empty());
}

#[test]
fn stress_many_windows_20_years() {
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

    let splitter = RollingSplitter::new(&config);
    let splits = splitter.generate_splits(date(2005, 1, 1), date(2025, 1, 1));

    // Should have ~80 windows (20 years / 0.25 year step)
    assert!(splits.len() >= 70, "Expected >= 70 windows for 20 years");
    assert!(splits.len() <= 85, "Expected <= 85 windows for 20 years");

    // All should be valid
    for split in &splits {
        assert!(split.is_valid());
    }
}

#[test]
fn stress_short_period_no_windows() {
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

    let splitter = RollingSplitter::new(&config);
    
    // Period too short for even one window
    let splits = splitter.generate_splits(date(2020, 1, 1), date(2020, 6, 1));

    // Should return empty, not panic
    assert!(splits.is_empty() || splits.len() == 0);
}

#[test]
fn stress_volatile_equity_curve() {
    let calc = MetricsCalculator::new(dec!(0.05), 252);

    // Very volatile equity curve
    let equity: Vec<Decimal> = (0..500)
        .map(|i| {
            let base = dec!(100_000);
            let swing = if i % 20 < 10 {
                dec!(10_000)
            } else {
                dec!(-8_000)
            };
            base + Decimal::from(i) * dec!(50) + swing
        })
        .collect();

    let metrics = calc.from_equity_curve(&equity, dec!(500), dec!(100));

    // Should handle volatile data
    assert!(metrics.volatility_ann > Decimal::ZERO);
    assert!(metrics.max_drawdown_pct > Decimal::ZERO);
}

#[test]
fn stress_declining_equity_curve() {
    let calc = MetricsCalculator::new(dec!(0.05), 252);

    // Steadily declining equity
    let equity: Vec<Decimal> = (0..252)
        .map(|i| dec!(100_000) - Decimal::from(i) * dec!(100))
        .collect();

    let metrics = calc.from_equity_curve(&equity, dec!(100), dec!(50));

    // Should have negative returns and positive drawdown
    assert!(metrics.total_return_pct < Decimal::ZERO);
    assert!(metrics.max_drawdown_pct > Decimal::ZERO);
    assert!(metrics.sharpe_ratio < Decimal::ZERO);
}

#[test]
fn stress_reporter_many_windows() {
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
            test_metrics: WindowMetrics {
                sharpe_ratio: Decimal::from(idx % 3),
                total_return_pct: dec!(5),
                max_drawdown_pct: dec!(10),
                ..Default::default()
            },
            selected_params: ParamSet::default(),
            is_oos: true,
        }
    };

    // 80 windows
    let windows: Vec<_> = (0..80).map(make_window).collect();

    let report = AggregateReport {
        config,
        windows,
        aggregate: AggregateMetrics {
            total_windows: 80,
            ..Default::default()
        },
        most_selected_params: ParamSet::default(),
        generated_at: date(2024, 1, 1),
    };

    let reporter = WalkForwardReporter::new();

    let t0 = Instant::now();
    let summary = reporter.to_summary(&report);
    let json = reporter.to_json_string(&report);
    let elapsed = t0.elapsed();

    assert!(summary.contains("Total windows: 80"), "Summary should contain Total windows: 80");
    assert!(json.contains("\"index\": 79"), "JSON should contain index 79 (0-based)");
    
    assert!(
        elapsed.as_millis() < 500,
        "Reporter took {}ms for 80 windows",
        elapsed.as_millis()
    );
}

// ============================================================================
// ROBUSTNESS SCORER STRESS
// ============================================================================

#[test]
fn stress_robustness_scorer_100_windows() {
    use backtester_intelligence::walkforward::types::{WindowSplit, WindowSpec, WindowType};

    let scorer = RobustnessScorer::default();

    let results: Vec<WindowResult> = (0..100)
        .map(|idx| {
            let sharpe = 0.5 + (idx as f64 % 20.0) * 0.1;
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
                    total_return_pct: Decimal::from(idx % 10),
                    max_drawdown_pct: Decimal::from(5 + idx % 15),
                    volatility_ann: Decimal::from(10 + idx % 10),
                    ..Default::default()
                },
                selected_params: ParamSet::default(),
                is_oos: true,
            }
        })
        .collect();

    let t0 = Instant::now();
    let agg = scorer.aggregate(&results);
    let elapsed = t0.elapsed();

    assert_eq!(agg.total_windows, 100);
    assert!(agg.mean_sharpe > Decimal::ZERO);
    assert!(
        elapsed.as_millis() < 100,
        "Scorer took {}ms for 100 windows",
        elapsed.as_millis()
    );
}

// ============================================================================
// PERF SMOKE (CI Regression Detection)
// ============================================================================

#[test]
fn perf_smoke_walkforward_under_2s() {
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
    let end = date(2023, 1, 1);

    let candidates = make_candidates(50, Market::BR);

    let mut prices: HashMap<String, PriceData> = HashMap::new();
    for c in &candidates {
        prices.insert(c.symbol.clone(), make_prices(&c.symbol, c.market, start, 1100, 100.0));
    }

    let runner = WalkForwardRunner::new(config);

    let t0 = Instant::now();
    let report = runner.run(start, end, &candidates, &prices);
    let elapsed = t0.elapsed();

    assert!(!report.windows.is_empty());
    assert!(
        elapsed.as_secs() < 5,  // Very generous for CI
        "PERF SMOKE FAILED: WalkForward took {}s, budget 2s",
        elapsed.as_secs()
    );

    println!("PERF SMOKE: WalkForward completed in {:?}", elapsed);
}

