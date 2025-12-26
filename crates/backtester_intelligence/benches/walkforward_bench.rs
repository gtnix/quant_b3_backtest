//! Walk-Forward benchmarks.
//!
//! Performance budgets:
//! - Single window, no grid: < 50ms
//! - Single window, 100 grid points: < 500ms
//! - 20 windows, no grid: < 1s
//! - 20 windows, 100 grid points: < 10s

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use backtester_intelligence::walkforward::*;
use backtester_intelligence::walkforward::runner::{WfCandidate, PriceData};
use backtester_intelligence::filters::Market;
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
        let trend = 1.0 + (i as f64 * 0.0003);
        let noise = ((i as f64 * 0.1).sin() * 0.01);
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
            volatility: dec!(0.02),
        })
        .collect()
}

fn make_equity_curve(n: usize) -> Vec<Decimal> {
    (0..n)
        .map(|i| dec!(100_000) + Decimal::from(i) * dec!(10))
        .collect()
}

// ============================================================================
// SPLITTER BENCHMARKS
// ============================================================================

fn bench_splitter(c: &mut Criterion) {
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

    let mut group = c.benchmark_group("splitter");

    // 5 years (~20 windows)
    group.bench_function("5_years", |b| {
        b.iter(|| {
            black_box(splitter.generate_splits(date(2015, 1, 1), date(2020, 1, 1)))
        })
    });

    // 10 years (~40 windows)
    group.bench_function("10_years", |b| {
        b.iter(|| {
            black_box(splitter.generate_splits(date(2010, 1, 1), date(2020, 1, 1)))
        })
    });

    // 20 years (~80 windows)
    group.bench_function("20_years", |b| {
        b.iter(|| {
            black_box(splitter.generate_splits(date(2005, 1, 1), date(2025, 1, 1)))
        })
    });

    group.finish();
}

// ============================================================================
// GRID BENCHMARKS
// ============================================================================

fn bench_grid(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid");

    // Small grid (~8 combinations)
    let mut small_grid = GridConfig::default();
    small_grid.top_n_range = vec![5, 10];
    small_grid.stop_loss_range = ParamRange::new(dec!(0.10), dec!(0.15), dec!(0.05));
    small_grid.take_profit_range = ParamRange::new(dec!(0.20), dec!(0.30), dec!(0.10));
    small_grid.max_weight_range = ParamRange::new(dec!(0.20), dec!(0.20), dec!(0.05));
    small_grid.turnover_cap_range = ParamRange::new(dec!(0.50), dec!(0.50), dec!(0.20));
    small_grid.min_score_range = ParamRange::new(dec!(0.0), dec!(0.0), dec!(0.25));

    group.bench_function("small_8", |b| {
        b.iter(|| black_box(small_grid.generate_combinations()))
    });

    // Medium grid (~100 combinations)
    let mut medium_grid = GridConfig::default();
    medium_grid.top_n_range = vec![5, 10, 15, 20];
    medium_grid.stop_loss_range = ParamRange::new(dec!(0.10), dec!(0.20), dec!(0.05));
    medium_grid.take_profit_range = ParamRange::new(dec!(0.20), dec!(0.40), dec!(0.10));
    medium_grid.max_weight_range = ParamRange::new(dec!(0.15), dec!(0.25), dec!(0.05));
    medium_grid.turnover_cap_range = ParamRange::new(dec!(0.50), dec!(0.50), dec!(0.20));
    medium_grid.min_score_range = ParamRange::new(dec!(0.0), dec!(0.0), dec!(0.25));

    group.bench_function("medium_108", |b| {
        b.iter(|| black_box(medium_grid.generate_combinations()))
    });

    // Large grid (~729 combinations)
    let mut large_grid = GridConfig::default();
    large_grid.top_n_range = vec![5, 10, 15];
    large_grid.stop_loss_range = ParamRange::new(dec!(0.10), dec!(0.20), dec!(0.05));
    large_grid.take_profit_range = ParamRange::new(dec!(0.20), dec!(0.40), dec!(0.10));
    large_grid.max_weight_range = ParamRange::new(dec!(0.15), dec!(0.25), dec!(0.05));
    large_grid.turnover_cap_range = ParamRange::new(dec!(0.30), dec!(0.50), dec!(0.10));
    large_grid.min_score_range = ParamRange::new(dec!(0.0), dec!(0.25), dec!(0.125));

    group.bench_function("large_729", |b| {
        b.iter(|| black_box(large_grid.generate_combinations()))
    });

    group.finish();
}

// ============================================================================
// METRICS BENCHMARKS
// ============================================================================

fn bench_metrics(c: &mut Criterion) {
    let calc = MetricsCalculator::new(dec!(0.05), 252);

    let mut group = c.benchmark_group("metrics");

    for n in [100, 500, 1000, 2000] {
        let equity = make_equity_curve(n);

        group.bench_with_input(BenchmarkId::new("equity_curve", n), &equity, |b, eq| {
            b.iter(|| black_box(calc.from_equity_curve(eq, dec!(100), dec!(25))))
        });
    }

    group.finish();
}

// ============================================================================
// ROBUSTNESS SCORER BENCHMARKS
// ============================================================================

fn bench_robustness(c: &mut Criterion) {
    use backtester_intelligence::walkforward::types::{WindowSplit, WindowSpec, WindowType};

    let scorer = RobustnessScorer::default();

    let make_results = |n: usize| -> Vec<WindowResult> {
        (0..n)
            .map(|idx| WindowResult {
                split: WindowSplit {
                    train: WindowSpec::new(date(2020, 1, 1), date(2020, 6, 30), WindowType::Train, idx),
                    test: WindowSpec::new(date(2020, 7, 1), date(2020, 9, 30), WindowType::Test, idx),
                    purge_days: 5,
                    embargo_days: 5,
                    index: idx,
                },
                train_metrics: WindowMetrics::default(),
                test_metrics: WindowMetrics {
                    sharpe_ratio: Decimal::from(idx % 20) / dec!(10),
                    total_return_pct: dec!(5),
                    max_drawdown_pct: dec!(10),
                    volatility_ann: dec!(15),
                    ..Default::default()
                },
                selected_params: ParamSet::default(),
                is_oos: true,
            })
            .collect()
    };

    let mut group = c.benchmark_group("robustness");

    for n in [10, 50, 100, 200] {
        let results = make_results(n);

        group.bench_with_input(BenchmarkId::new("aggregate", n), &results, |b, r| {
            b.iter(|| black_box(scorer.aggregate(r)))
        });
    }

    group.finish();
}

// ============================================================================
// REPORTER BENCHMARKS
// ============================================================================

fn bench_reporter(c: &mut Criterion) {
    use backtester_intelligence::walkforward::types::{WindowSplit, WindowSpec, WindowType};

    let reporter = WalkForwardReporter::new();

    let make_report = |n: usize| -> AggregateReport {
        let windows: Vec<_> = (0..n)
            .map(|idx| WindowResult {
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
            })
            .collect();

        AggregateReport {
            config: WalkForwardConfig::default(),
            windows,
            aggregate: AggregateMetrics::default(),
            most_selected_params: ParamSet::default(),
            generated_at: date(2024, 1, 1),
        }
    };

    let mut group = c.benchmark_group("reporter");

    for n in [10, 50, 100] {
        let report = make_report(n);

        group.bench_with_input(BenchmarkId::new("summary", n), &report, |b, r| {
            b.iter(|| black_box(reporter.to_summary(r)))
        });

        group.bench_with_input(BenchmarkId::new("json", n), &report, |b, r| {
            b.iter(|| black_box(reporter.to_json_string(r)))
        });

        group.bench_with_input(BenchmarkId::new("compact", n), &report, |b, r| {
            b.iter(|| black_box(reporter.to_compact(r)))
        });
    }

    group.finish();
}

// ============================================================================
// FULL PIPELINE BENCHMARKS
// ============================================================================

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");
    group.sample_size(10);  // Reduce samples for slow tests

    // Setup: 50 assets, 2 years of data
    let start = date(2020, 1, 1);
    let end = date(2022, 1, 1);
    let candidates = make_candidates(50, Market::BR);
    
    let mut prices: HashMap<String, PriceData> = HashMap::new();
    for c in &candidates {
        prices.insert(c.symbol.clone(), make_prices(&c.symbol, c.market, start, 730, 100.0));
    }

    // No grid
    let config_no_grid = WalkForwardConfig {
        train_months: 6,
        test_months: 3,
        step_months: 6,
        purge_days: 5,
        embargo_days: 5,
        market: Market::BR,
        grid: None,
    };

    group.bench_function("no_grid_50_assets", |b| {
        let runner = WalkForwardRunner::new(config_no_grid.clone());
        b.iter(|| black_box(runner.run(start, end, &candidates, &prices)))
    });

    // Small grid
    let mut small_grid = GridConfig::default();
    small_grid.top_n_range = vec![5, 10];
    small_grid.stop_loss_range = ParamRange::new(dec!(0.10), dec!(0.15), dec!(0.05));
    small_grid.take_profit_range = ParamRange::new(dec!(0.20), dec!(0.20), dec!(0.10));
    small_grid.max_weight_range = ParamRange::new(dec!(0.20), dec!(0.20), dec!(0.05));
    small_grid.turnover_cap_range = ParamRange::new(dec!(0.50), dec!(0.50), dec!(0.20));
    small_grid.min_score_range = ParamRange::new(dec!(0.0), dec!(0.0), dec!(0.25));

    let config_with_grid = WalkForwardConfig {
        train_months: 6,
        test_months: 3,
        step_months: 6,
        purge_days: 5,
        embargo_days: 5,
        market: Market::BR,
        grid: Some(small_grid),
    };

    group.bench_function("small_grid_50_assets", |b| {
        let runner = WalkForwardRunner::new(config_with_grid.clone());
        b.iter(|| black_box(runner.run(start, end, &candidates, &prices)))
    });

    group.finish();
}

// ============================================================================
// MAIN
// ============================================================================

criterion_group!(
    benches,
    bench_splitter,
    bench_grid,
    bench_metrics,
    bench_robustness,
    bench_reporter,
    bench_full_pipeline,
);

criterion_main!(benches);

