//! Benchmark suite for UnifiedEngine - the production hot path.
//!
//! These benchmarks measure the real-world performance of the backtest engine,
//! focusing on `process_day()` which is the critical loop.
//!
//! # Milestone 3 Changes
//!
//! - DualPriceBar uses fixed-point Price instead of Decimal (5-10x faster)
//! - Uses SymbolId instead of String for O(1) price lookups
//! - Pre-registers symbols before benchmark loops
//!
//! Run with: `cargo bench --bench unified_bench`

use backtester_core::{Money, Price};
use backtester_engine::{DualPriceBar, SymbolId, SymbolRegistry, UnifiedEngine, UnifiedEngineConfig};
use backtester_intelligence::entry::AssetCandidate;
use backtester_intelligence::filters::Market;
use chrono::NaiveDate;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

// =============================================================================
// TEST DATA GENERATORS
// =============================================================================

/// Generate symbol names for benchmarking.
fn generate_symbols(num_assets: usize) -> Vec<String> {
    (0..num_assets)
        .map(|i| format!("ASSET{:04}", i))
        .collect()
}

/// Generate synthetic price bars for benchmarking with SymbolId.
///
/// # Performance (Milestone 3)
///
/// Uses fixed-point Price directly (not Decimal conversion) for maximum benchmark accuracy.
fn generate_bars_with_registry(registry: &SymbolRegistry, num_assets: usize, date: NaiveDate) -> Vec<DualPriceBar> {
    (0..num_assets)
        .map(|i| {
            let symbol = format!("ASSET{:04}", i);
            let symbol_id = registry.get(&symbol).unwrap_or_else(|| SymbolId::new(i as u32));
            let base_price = Price::from_int((50 + (i % 100)) as i64);
            DualPriceBar::new(
                symbol_id,
                date,
                base_price + Price::from_f64(0.50),  // adjusted_close
                base_price,                           // raw_close
                base_price - Price::from_f64(0.20),  // open
                base_price + Price::from_f64(1.00),  // high
                base_price - Price::from_f64(0.50),  // low
                (1_000_000 + i * 10_000) as i64,     // volume
            )
        })
        .collect()
}

/// Generate asset candidates for entry evaluation.
/// 
/// # Performance (Milestone 6)
/// 
/// Uses fixed-point Price and Money for monetary fields.
fn generate_candidates(num_assets: usize, _date: NaiveDate) -> Vec<AssetCandidate> {
    (0..num_assets)
        .map(|i| AssetCandidate {
            symbol: format!("ASSET{:04}", i),
            market: Market::BR,
            price: Some(Price::from_int((50 + (i % 100)) as i64)),
            avg_volume: Some(Money::from_int(1_000_000)),
            price_days: 252,
            has_fundamentals: true,
            has_dividends: false,
            is_tradeable: true,
            volatility: Some(0.02),
            score: Some(0.5 + (i as f64 * 0.01)),
            filter_scores: vec![],
            fundamentals_as_of: None,
        })
        .collect()
}

/// Create a pre-configured engine for benchmarking with symbols pre-registered.
fn create_engine_with_symbols(initial_capital: Decimal, num_assets: usize) -> UnifiedEngine {
    let config = UnifiedEngineConfig {
        initial_capital,
        enable_dividends: false, // Disable for pure engine benchmark
        ..Default::default()
    };
    let mut engine = UnifiedEngine::with_config(config);
    
    // Pre-register all symbols (done once during setup, not in hot path)
    let symbols = generate_symbols(num_assets);
    engine.register_symbols(symbols.iter().map(String::as_str));
    
    engine
}

// =============================================================================
// BENCHMARK: PROCESS_DAY - MINIMAL (1 ASSET)
// =============================================================================

fn bench_process_day_1_asset(c: &mut Criterion) {
    let mut group = c.benchmark_group("unified_engine");
    group.throughput(Throughput::Elements(1)); // 1 day processed

    let date = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();
    let candidates = generate_candidates(1, date);
    
    // Create engine with pre-registered symbols to get registry
    let setup_engine = create_engine_with_symbols(dec!(1_000_000), 1);
    let bars = generate_bars_with_registry(setup_engine.registry(), 1, date);

    group.bench_function("process_day_1_asset", |b| {
        b.iter(|| {
            // Reset engine state between iterations for consistency
            let mut fresh_engine = create_engine_with_symbols(dec!(1_000_000), 1);
            // Milestone 5: No clone needed - process_day takes slice
            black_box(fresh_engine.process_day(date, black_box(&bars), black_box(&candidates)))
        })
    });

    group.finish();
}

// =============================================================================
// BENCHMARK: PROCESS_DAY - REALISTIC (10 ASSETS)
// =============================================================================

fn bench_process_day_10_assets(c: &mut Criterion) {
    let mut group = c.benchmark_group("unified_engine");
    group.throughput(Throughput::Elements(1)); // 1 day processed

    let date = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();
    let candidates = generate_candidates(10, date);
    
    let setup_engine = create_engine_with_symbols(dec!(1_000_000), 10);
    let bars = generate_bars_with_registry(setup_engine.registry(), 10, date);

    group.bench_function("process_day_10_assets", |b| {
        b.iter(|| {
            let mut fresh_engine = create_engine_with_symbols(dec!(1_000_000), 10);
            // Milestone 5: No clone needed - process_day takes slice
            black_box(fresh_engine.process_day(date, black_box(&bars), black_box(&candidates)))
        })
    });

    group.finish();
}

// =============================================================================
// BENCHMARK: FULL BACKTEST - 252 DAYS
// =============================================================================

fn bench_full_backtest_252_days(c: &mut Criterion) {
    let mut group = c.benchmark_group("unified_engine");
    
    const NUM_DAYS: usize = 252;
    const NUM_ASSETS: usize = 10;
    
    group.throughput(Throughput::Elements(NUM_DAYS as u64));

    // Create a setup engine to get the registry
    let setup_engine = create_engine_with_symbols(dec!(1_000_000), NUM_ASSETS);
    let registry = setup_engine.registry();

    // Pre-generate all data with SymbolIds
    let start_date = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();
    let days_data: Vec<_> = (0..NUM_DAYS)
        .map(|d| {
            let date = start_date + chrono::Duration::days(d as i64);
            let bars = generate_bars_with_registry(registry, NUM_ASSETS, date);
            let candidates = generate_candidates(NUM_ASSETS, date);
            (date, bars, candidates)
        })
        .collect();

    group.bench_function("full_backtest_252d_10_assets", |b| {
        b.iter(|| {
            let mut engine = create_engine_with_symbols(dec!(1_000_000), NUM_ASSETS);
            for (date, bars, candidates) in &days_data {
                // Milestone 5: No clone needed - process_day takes slice
                black_box(engine.process_day(*date, bars, candidates));
            }
            black_box(engine.get_result())
        })
    });

    group.finish();
}

// =============================================================================
// BENCHMARK: SCALING ANALYSIS
// =============================================================================

fn bench_scaling_assets(c: &mut Criterion) {
    let mut group = c.benchmark_group("unified_scaling");
    
    for num_assets in [1, 5, 10, 25, 50].iter() {
        let date = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();
        let candidates = generate_candidates(*num_assets, date);
        
        let setup_engine = create_engine_with_symbols(dec!(1_000_000), *num_assets);
        let bars = generate_bars_with_registry(setup_engine.registry(), *num_assets, date);

        group.throughput(Throughput::Elements(*num_assets as u64));
        
        group.bench_with_input(
            BenchmarkId::new("process_day", num_assets),
            num_assets,
            |b, &n| {
                b.iter(|| {
                    let mut engine = create_engine_with_symbols(dec!(1_000_000), n);
                    // Milestone 5: No clone needed - process_day takes slice
                    black_box(engine.process_day(date, black_box(&bars), black_box(&candidates)))
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// BENCHMARK: ENGINE INITIALIZATION
// =============================================================================

fn bench_engine_init(c: &mut Criterion) {
    let mut group = c.benchmark_group("unified_engine");

    group.bench_function("engine_init", |b| {
        b.iter(|| {
            black_box(create_engine_with_symbols(dec!(1_000_000), 10))
        })
    });

    group.finish();
}

// =============================================================================
// BENCHMARK: REBALANCE SCALING (Milestone 6)
// =============================================================================
//
// Measures the rebalance path performance with different portfolio sizes.
// The fixed-point migration (Price/Money/Rate) in the rebalance pipeline
// should show improved performance over the previous Decimal-based version.
//
// Scenarios test the entry/exit orchestration cost at scale:
// - 10 assets: Baseline small portfolio
// - 20 assets: Medium portfolio (production typical)
// - 50 assets: Large portfolio stress test
//
// This benchmark runs a full 252-day backtest for each portfolio size,
// allowing measurement of both daily hot path and rebalance overhead.

fn bench_rebalance_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("rebalance_scaling");
    
    const NUM_DAYS: usize = 252;
    
    for num_assets in [10, 20, 50] {
        group.throughput(Throughput::Elements(NUM_DAYS as u64));
        
        // Pre-generate all data for this asset count
        let setup_engine = create_engine_with_symbols(dec!(1_000_000), num_assets);
        let registry = setup_engine.registry();
        
        let start_date = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();
        let days_data: Vec<_> = (0..NUM_DAYS)
            .map(|d| {
                let date = start_date + chrono::Duration::days(d as i64);
                let bars = generate_bars_with_registry(registry, num_assets, date);
                let candidates = generate_candidates(num_assets, date);
                (date, bars, candidates)
            })
            .collect();
        
        group.bench_with_input(
            BenchmarkId::new("full_backtest_252d", num_assets),
            &days_data,
            |b, days| {
                b.iter(|| {
                    let mut engine = create_engine_with_symbols(dec!(1_000_000), num_assets);
                    for (date, bars, candidates) in days {
                        black_box(engine.process_day(*date, bars, candidates));
                    }
                    black_box(engine.get_result())
                })
            },
        );
    }
    
    group.finish();
}

// =============================================================================
// CRITERION MAIN
// =============================================================================

criterion_group!(
    benches,
    bench_process_day_1_asset,
    bench_process_day_10_assets,
    bench_full_backtest_252_days,
    bench_scaling_assets,
    bench_engine_init,
    bench_rebalance_scaling,
);

criterion_main!(benches);
