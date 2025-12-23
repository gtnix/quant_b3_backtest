//! Chicago-standard benchmark scenarios.
//!
//! Measures: events/sec, latency p99, peak RSS.

use backtester_core::{Bar, MarketEvent, Strategy, StrategyConfig};
use backtester_engine::SimulationEngine;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use strategy_lib::{DailyTrendStrategy, MeanReversionStrategy, NoOpStrategy};

/// Generate synthetic events for benchmarking.
fn generate_events(num_assets: usize, bars_per_asset: usize) -> Vec<MarketEvent> {
    let mut events = Vec::with_capacity(num_assets * bars_per_asset);
    let base_ts: i64 = 1_700_000_000_000_000_000;
    let interval: i64 = 60_000_000_000; // 1 minute in nanos

    for bar_idx in 0..bars_per_asset {
        for asset_id in 0..num_assets {
            let ts = base_ts + (bar_idx as i64) * interval;
            let base_price = 50.0 + (asset_id as f64) * 10.0;
            // Simulate price movement with deterministic pattern
            let price = base_price + (bar_idx as f64 * 0.1).sin() * 2.0;
            
            events.push(MarketEvent {
                asset_id: asset_id as u32,
                bar: Bar {
                    timestamp: ts,
                    open: price - 0.1,
                    high: price + 0.5,
                    low: price - 0.5,
                    close: price,
                    volume: 100_000.0 + (asset_id as f64) * 10_000.0,
                },
            });
        }
    }
    events
}

/// Scenario 1: Intraday Net Zero
/// - 10 assets, 1000 bars each = 10,000 events
/// - 1-minute bars (intraday)
/// - MeanReversionStrategy (high signal frequency)
fn bench_intraday_net_zero(c: &mut Criterion) {
    let num_assets = 10;
    let bars_per_asset = 1000;
    let events = generate_events(num_assets, bars_per_asset);
    let total_events = events.len() as u64;

    let mut group = c.benchmark_group("intraday_net_zero");
    group.throughput(Throughput::Elements(total_events));
    group.sample_size(20);

    group.bench_function("mean_reversion_10k_events", |b| {
        b.iter(|| {
            let mut strategy = MeanReversionStrategy::new(0.005, 50);
            strategy.on_init(&StrategyConfig::default(), num_assets);
            let mut engine = SimulationEngine::with_defaults(strategy, 1_000_000.0, num_assets);
            for event in &events {
                black_box(engine.process_event(event));
            }
            black_box(engine.get_result())
        })
    });

    group.bench_function("noop_baseline_10k_events", |b| {
        b.iter(|| {
            let strategy = NoOpStrategy;
            let mut engine = SimulationEngine::with_defaults(strategy, 1_000_000.0, num_assets);
            for event in &events {
                black_box(engine.process_event(event));
            }
            black_box(engine.get_result())
        })
    });

    group.finish();
}

/// Scenario 2: Daily Swing Trade
/// - 200 assets, 252 bars each = 50,400 events
/// - Daily bars (1 year)
/// - DailyTrendStrategy (MA crossover)
fn bench_daily_swing(c: &mut Criterion) {
    let num_assets = 200;
    let bars_per_asset = 252; // 1 year of daily bars
    let events = generate_events(num_assets, bars_per_asset);
    let total_events = events.len() as u64;

    let mut group = c.benchmark_group("daily_swing");
    group.throughput(Throughput::Elements(total_events));
    group.sample_size(10);

    group.bench_function("trend_200_assets", |b| {
        b.iter(|| {
            let mut strategy = DailyTrendStrategy::new(20, 50);
            strategy.on_init(&StrategyConfig::default(), num_assets);
            let mut engine = SimulationEngine::with_defaults(strategy, 10_000_000.0, num_assets);
            for event in &events {
                black_box(engine.process_event(event));
            }
            black_box(engine.get_result())
        })
    });

    group.finish();
}

/// Scenario 3: Stress Universe
/// - 1000 assets, 252 bars each = 252,000 events
/// - Tests scalability of portfolio mark-to-market
fn bench_stress_universe(c: &mut Criterion) {
    let num_assets = 1000;
    let bars_per_asset = 252;
    let events = generate_events(num_assets, bars_per_asset);
    let total_events = events.len() as u64;

    let mut group = c.benchmark_group("stress_universe");
    group.throughput(Throughput::Elements(total_events));
    group.sample_size(10);

    group.bench_function("noop_1000_assets", |b| {
        b.iter(|| {
            let strategy = NoOpStrategy;
            let mut engine = SimulationEngine::with_defaults(strategy, 100_000_000.0, num_assets);
            for event in &events {
                black_box(engine.process_event(event));
            }
            black_box(engine.get_result())
        })
    });

    group.finish();
}

/// Scalability test: measure throughput at different universe sizes
fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");
    group.sample_size(10);

    for num_assets in [10, 50, 100, 200, 500].iter() {
        let bars_per_asset = 100;
        let events = generate_events(*num_assets, bars_per_asset);
        let total_events = events.len() as u64;

        group.throughput(Throughput::Elements(total_events));
        group.bench_with_input(
            BenchmarkId::new("events_per_sec", num_assets),
            num_assets,
            |b, &n| {
                b.iter(|| {
                    let strategy = NoOpStrategy;
                    let mut engine = SimulationEngine::with_defaults(strategy, 1_000_000.0, n);
                    for event in &events {
                        black_box(engine.process_event(event));
                    }
                    black_box(engine.get_result())
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_intraday_net_zero,
    bench_daily_swing,
    bench_stress_universe,
    bench_scalability
);
criterion_main!(benches);

