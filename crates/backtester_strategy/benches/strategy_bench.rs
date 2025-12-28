//! Strategy Factory Benchmarks
//!
//! Compares:
//! 1. Standard Compositor (dynamic block creation, HashMap params)
//! 2. CompiledStrategy (pre-resolved blocks, typed params)
//!
//! Target: ≥30% improvement on scenario M (1K assets × 5K days)

use backtester_intelligence::filters::Market;
use backtester_strategy::{
    compiled::{CompiledStrategy, SymbolTable},
    compositor::Compositor,
    config::load_strategy_from_str,
    context::{StrategyCandidate, StrategyContext},
    fast_context::{CandidatesSoA, FastContext, PreallocBuffers, fast_momentum_select, fast_equal_weight},
    registry::BlockRegistry,
};
use chrono::NaiveDate;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rust_decimal_macros::dec;

/// Generate test candidates for benchmarking.
fn generate_candidates(num_assets: usize) -> Vec<StrategyCandidate> {
    (0..num_assets)
        .map(|i| {
            let symbol = format!("ASSET{:04}", i);
            let mut c = StrategyCandidate::new(symbol, Market::BR);
            c.price = Some(rust_decimal::Decimal::from(20 + (i % 80) as i64));
            c.volatility = Some(0.15 + (i as f64 % 30.0) * 0.01);
            c.momentum_return = Some(0.05 + (i as f64 % 20.0) * 0.005);
            c.prices = (0..252).map(|j| 20.0 + j as f64 * 0.1).collect();
            c
        })
        .collect()
}

/// Create test context with candidates.
fn create_context(candidates: Vec<StrategyCandidate>) -> StrategyContext {
    let date = NaiveDate::from_ymd_opt(2024, 6, 15).unwrap();
    let universe: Vec<String> = candidates.iter().map(|c| c.symbol.clone()).collect();
    StrategyContext::new(date, Market::BR, dec!(10_000_000))
        .with_candidates(candidates)
        .with_universe(universe)
}

/// Standard strategy config TOML.
const MOMENTUM_STRATEGY: &str = r#"
[strategy]
id = "momentum_benchmark"
version = "1.0.0"

[[pipeline]]
type = "selection"
block_id = "momentum"
params = { top_pct = 20 }

[[pipeline]]
type = "sizing"
block_id = "equal_weight"
params = { max_weight = 0.10, max_positions = 50 }

[constraints]
max_weight_per_asset = 0.10
"#;

/// Multi-factor strategy for more complex benchmarks.
const MULTIFACTOR_STRATEGY: &str = r#"
[strategy]
id = "multifactor_benchmark"
version = "1.0.0"

[[pipeline]]
type = "selection"
block_id = "momentum"
params = { top_pct = 40 }

[[pipeline]]
type = "selection"
block_id = "low_vol"
params = { top_pct = 60 }

[[pipeline]]
type = "sizing"
block_id = "risk_parity"
params = { max_weight = 0.10, max_positions = 30 }

[constraints]
max_weight_per_asset = 0.10
"#;

// =============================================================================
// BENCHMARK: Compositor Execution
// =============================================================================

fn bench_compositor(c: &mut Criterion) {
    let mut group = c.benchmark_group("compositor");
    
    for num_assets in [50, 100, 200, 500, 1000] {
        let candidates = generate_candidates(num_assets);
        let config = load_strategy_from_str(MOMENTUM_STRATEGY).unwrap();
        let compositor = Compositor::with_builtins();
        
        group.throughput(Throughput::Elements(num_assets as u64));
        group.bench_with_input(
            BenchmarkId::new("standard", num_assets),
            &num_assets,
            |b, _| {
                b.iter(|| {
                    let mut ctx = create_context(candidates.clone());
                    black_box(compositor.execute(&config, &mut ctx))
                })
            },
        );
    }
    
    group.finish();
}

// =============================================================================
// BENCHMARK: Compiled Strategy Execution
// =============================================================================

fn bench_compiled_strategy(c: &mut Criterion) {
    let mut group = c.benchmark_group("compiled_strategy");
    
    for num_assets in [50, 100, 200, 500, 1000] {
        let candidates = generate_candidates(num_assets);
        let config = load_strategy_from_str(MOMENTUM_STRATEGY).unwrap();
        let registry = BlockRegistry::with_builtins();
        let universe: Vec<String> = candidates.iter().map(|c| c.symbol.clone()).collect();
        
        // Compile once (not measured)
        let mut compiled = CompiledStrategy::compile(&config, &registry, universe.clone()).unwrap();
        
        group.throughput(Throughput::Elements(num_assets as u64));
        group.bench_with_input(
            BenchmarkId::new("compiled", num_assets),
            &num_assets,
            |b, _| {
                b.iter(|| {
                    let mut ctx = create_context(candidates.clone());
                    black_box(compiled.execute_fast(&mut ctx))
                })
            },
        );
    }
    
    group.finish();
}

// =============================================================================
// BENCHMARK: Symbol Table Operations
// =============================================================================

fn bench_symbol_table(c: &mut Criterion) {
    let mut group = c.benchmark_group("symbol_table");
    
    for num_symbols in [100, 500, 1000, 5000] {
        let symbols: Vec<String> = (0..num_symbols)
            .map(|i| format!("SYM{:05}", i))
            .collect();
        
        // Build table
        group.bench_with_input(
            BenchmarkId::new("build", num_symbols),
            &symbols,
            |b, syms| {
                b.iter(|| {
                    black_box(SymbolTable::from_universe(syms.iter().cloned()))
                })
            },
        );
        
        // Lookup
        let table = SymbolTable::from_universe(symbols.iter().cloned());
        group.bench_with_input(
            BenchmarkId::new("lookup", num_symbols),
            &symbols,
            |b, syms| {
                b.iter(|| {
                    for s in syms {
                        black_box(table.get_id(s));
                    }
                })
            },
        );
    }
    
    group.finish();
}

// =============================================================================
// BENCHMARK: Multi-Factor Strategy (more complex pipeline)
// =============================================================================

fn bench_multifactor(c: &mut Criterion) {
    let mut group = c.benchmark_group("multifactor");
    
    for num_assets in [100, 500, 1000] {
        let candidates = generate_candidates(num_assets);
        let config = load_strategy_from_str(MULTIFACTOR_STRATEGY).unwrap();
        let compositor = Compositor::with_builtins();
        let registry = BlockRegistry::with_builtins();
        let universe: Vec<String> = candidates.iter().map(|c| c.symbol.clone()).collect();
        
        // Standard compositor
        group.bench_with_input(
            BenchmarkId::new("standard", num_assets),
            &num_assets,
            |b, _| {
                b.iter(|| {
                    let mut ctx = create_context(candidates.clone());
                    black_box(compositor.execute(&config, &mut ctx))
                })
            },
        );
        
        // Compiled strategy
        let mut compiled = CompiledStrategy::compile(&config, &registry, universe.clone()).unwrap();
        group.bench_with_input(
            BenchmarkId::new("compiled", num_assets),
            &num_assets,
            |b, _| {
                b.iter(|| {
                    let mut ctx = create_context(candidates.clone());
                    black_box(compiled.execute_fast(&mut ctx))
                })
            },
        );
    }
    
    group.finish();
}

// =============================================================================
// BENCHMARK: Scenario M (1K assets × daily execution, simulating 5K days)
// =============================================================================

fn bench_scenario_m(c: &mut Criterion) {
    let mut group = c.benchmark_group("scenario_m");
    group.sample_size(10); // Fewer samples due to longer runtime
    
    let num_assets = 1000;
    let candidates = generate_candidates(num_assets);
    let config = load_strategy_from_str(MOMENTUM_STRATEGY).unwrap();
    let compositor = Compositor::with_builtins();
    let registry = BlockRegistry::with_builtins();
    let universe: Vec<String> = candidates.iter().map(|c| c.symbol.clone()).collect();
    
    // Simulate 100 rebalance days (scaled down from 5K for practical benchmarking)
    let rebalance_count = 100;
    
    group.throughput(Throughput::Elements((num_assets * rebalance_count) as u64));
    
    // Standard compositor - multiple executions
    group.bench_function("standard_100_rebalances", |b| {
        b.iter(|| {
            for _ in 0..rebalance_count {
                let mut ctx = create_context(candidates.clone());
                black_box(compositor.execute(&config, &mut ctx));
            }
        })
    });
    
    // Compiled strategy - multiple executions (compile once, execute many)
    let mut compiled = CompiledStrategy::compile(&config, &registry, universe).unwrap();
    group.bench_function("compiled_100_rebalances", |b| {
        b.iter(|| {
            for _ in 0..rebalance_count {
                let mut ctx = create_context(candidates.clone());
                black_box(compiled.execute_fast(&mut ctx));
            }
        })
    });
    
    group.finish();
}

// =============================================================================
// BENCHMARK: Fast SoA Selection (zero-alloc hot path)
// =============================================================================

fn bench_fast_soa(c: &mut Criterion) {
    let mut group = c.benchmark_group("fast_soa");
    
    for num_assets in [100, 500, 1000, 5000] {
        // Create SoA candidates
        let mut candidates = CandidatesSoA::with_capacity(num_assets);
        for i in 0..num_assets {
            candidates.set(
                i as u16,
                20.0 + (i % 80) as f64,
                0.15 + (i as f64 % 30.0) * 0.01,
                0.05 + (i as f64 % 20.0) * 0.005,
            );
        }
        
        // Preallocate buffers
        let mut buffers = PreallocBuffers::with_capacity(num_assets);
        let mut weights = vec![0.0; num_assets];
        
        group.throughput(Throughput::Elements(num_assets as u64));
        
        // Benchmark fast momentum selection
        group.bench_with_input(
            BenchmarkId::new("momentum_select", num_assets),
            &num_assets,
            |b, _| {
                b.iter(|| {
                    let selected = fast_momentum_select(&candidates, 0.20, &mut buffers);
                    black_box(selected.len())
                })
            },
        );
        
        // Benchmark fast equal weight
        let selected: Vec<u16> = (0..num_assets.min(50) as u16).collect();
        group.bench_with_input(
            BenchmarkId::new("equal_weight", num_assets),
            &num_assets,
            |b, _| {
                b.iter(|| {
                    black_box(fast_equal_weight(&selected, 0.10, 50, &mut weights))
                })
            },
        );
        
        // Benchmark full pipeline (select + size)
        group.bench_with_input(
            BenchmarkId::new("full_pipeline", num_assets),
            &num_assets,
            |b, _| {
                b.iter(|| {
                    let selected = fast_momentum_select(&candidates, 0.20, &mut buffers);
                    let selected_vec: Vec<u16> = selected.to_vec();
                    black_box(fast_equal_weight(&selected_vec, 0.10, 50, &mut weights))
                })
            },
        );
    }
    
    group.finish();
}

// =============================================================================
// BENCHMARK: Comparison - Standard vs Fast SoA
// =============================================================================

fn bench_standard_vs_fast(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard_vs_fast");
    group.sample_size(10);
    
    for num_assets in [500, 1000, 2000] {
        // Standard approach - full context with StrategyCandidate
        let candidates = generate_candidates(num_assets);
        let config = load_strategy_from_str(MOMENTUM_STRATEGY).unwrap();
        let compositor = Compositor::with_builtins();
        
        group.throughput(Throughput::Elements(num_assets as u64));
        
        // 100 iterations to simulate daily rebalancing
        let iterations = 100;
        
        group.bench_with_input(
            BenchmarkId::new("standard_100x", num_assets),
            &num_assets,
            |b, _| {
                b.iter(|| {
                    for _ in 0..iterations {
                        let mut ctx = create_context(candidates.clone());
                        black_box(compositor.execute(&config, &mut ctx));
                    }
                })
            },
        );
        
        // Fast SoA approach
        let mut soa_candidates = CandidatesSoA::with_capacity(num_assets);
        for i in 0..num_assets {
            soa_candidates.set(
                i as u16,
                20.0 + (i % 80) as f64,
                0.15 + (i as f64 % 30.0) * 0.01,
                0.05 + (i as f64 % 20.0) * 0.005,
            );
        }
        let mut buffers = PreallocBuffers::with_capacity(num_assets);
        let mut weights = vec![0.0; num_assets];
        
        group.bench_with_input(
            BenchmarkId::new("fast_soa_100x", num_assets),
            &num_assets,
            |b, _| {
                b.iter(|| {
                    for _ in 0..iterations {
                        buffers.clear();
                        let selected = fast_momentum_select(&soa_candidates, 0.20, &mut buffers);
                        let selected_vec: Vec<u16> = selected.to_vec();
                        black_box(fast_equal_weight(&selected_vec, 0.10, 50, &mut weights));
                    }
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_compositor,
    bench_compiled_strategy,
    bench_symbol_table,
    bench_multifactor,
    bench_scenario_m,
    bench_fast_soa,
    bench_standard_vs_fast,
);
criterion_main!(benches);

