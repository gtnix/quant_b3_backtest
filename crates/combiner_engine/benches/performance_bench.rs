//! Performance benchmarks for the ultra-performant SCG components.
//!
//! Run with: cargo bench --package combiner_engine --bench performance_bench
//!
//! Benchmarks:
//! - SIMD vs scalar metrics calculation
//! - SoA vs AoS iteration
//! - Cache operations (get/insert)
//! - Pareto ranking

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput, black_box};

use combiner_core::{
    sharpe_simd, sharpe_scalar,
    max_drawdown_simd,
    volatility_simd, volatility_scalar,
    sortino_simd, sortino_scalar,
    calculate_all_metrics,
    PopulationFitnessSoA,
    MultiObjectiveFitness, FitnessConfig,
};

use combiner_runner::{GenomeCache, SplitCache, SplitMetrics};

// ============================================================================
// Helper functions
// ============================================================================

/// Generate deterministic pseudo-random returns
fn generate_returns(n: usize, seed: u64) -> Vec<f64> {
    let mut returns = Vec::with_capacity(n);
    let mut state = seed;
    
    for _ in 0..n {
        // Simple LCG for deterministic pseudo-random
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let x = ((state >> 33) as f64) / (u32::MAX as f64) - 0.5;
        returns.push(0.0005 + x * 0.02); // Mean ~0.05% daily, ~2% daily std
    }
    
    returns
}

/// Create test fitness
fn create_test_fitness(sharpe: f64) -> MultiObjectiveFitness {
    let config = FitnessConfig::default();
    MultiObjectiveFitness::from_metrics(
        0.1, sharpe, -0.1, 1.0, 1.0, 1.5, 100, 0.12, 2.5, &config,
    )
}

// ============================================================================
// SIMD Metrics Benchmarks
// ============================================================================

fn bench_sharpe_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("sharpe_ratio");
    
    for size in [100, 252, 1000, 5000, 10000].iter() {
        let returns = generate_returns(*size, 42);
        
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("simd", size),
            &returns,
            |b, returns| b.iter(|| sharpe_simd(black_box(returns), 0.0)),
        );
        
        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &returns,
            |b, returns| b.iter(|| sharpe_scalar(black_box(returns), 0.0)),
        );
    }
    
    group.finish();
}

fn bench_volatility_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("volatility");
    
    for size in [100, 252, 1000, 5000, 10000].iter() {
        let returns = generate_returns(*size, 42);
        
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("simd", size),
            &returns,
            |b, returns| b.iter(|| volatility_simd(black_box(returns))),
        );
        
        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &returns,
            |b, returns| b.iter(|| volatility_scalar(black_box(returns))),
        );
    }
    
    group.finish();
}

fn bench_sortino_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("sortino_ratio");
    
    for size in [100, 252, 1000, 5000, 10000].iter() {
        let returns = generate_returns(*size, 42);
        
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("simd", size),
            &returns,
            |b, returns| b.iter(|| sortino_simd(black_box(returns), 0.0, 0.0)),
        );
        
        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &returns,
            |b, returns| b.iter(|| sortino_scalar(black_box(returns), 0.0, 0.0)),
        );
    }
    
    group.finish();
}

fn bench_max_drawdown(c: &mut Criterion) {
    let mut group = c.benchmark_group("max_drawdown");
    
    for size in [100, 252, 1000, 5000, 10000].iter() {
        let returns = generate_returns(*size, 42);
        
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("simd", size),
            &returns,
            |b, returns| b.iter(|| max_drawdown_simd(black_box(returns))),
        );
    }
    
    group.finish();
}

fn bench_all_metrics_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("all_metrics_batch");
    
    for size in [252, 1000, 2520].iter() {
        let returns = generate_returns(*size, 42);
        
        group.throughput(Throughput::Elements(*size as u64));
        
        // Batch calculation (single pass)
        group.bench_with_input(
            BenchmarkId::new("batch", size),
            &returns,
            |b, returns| b.iter(|| calculate_all_metrics(black_box(returns), 0.0)),
        );
        
        // Individual calculations (multiple passes)
        group.bench_with_input(
            BenchmarkId::new("individual", size),
            &returns,
            |b, returns| b.iter(|| {
                let sharpe = sharpe_simd(returns, 0.0);
                let vol = volatility_simd(returns);
                let dd = max_drawdown_simd(returns);
                let sortino = sortino_simd(returns, 0.0, 0.0);
                black_box((sharpe, vol, dd, sortino))
            }),
        );
    }
    
    group.finish();
}

// ============================================================================
// SoA vs AoS Benchmarks
// ============================================================================

fn bench_soa_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("soa_iteration");
    
    for size in [100, 500, 1000].iter() {
        // SoA approach
        let mut soa = PopulationFitnessSoA::with_capacity(*size);
        for i in 0..*size {
            let sharpe = (i as f64) * 0.01;
            soa.set_fitness(i, sharpe, 0.1, -0.1, 1.0, 1.0, 1.5, 100, 0.12, 2.5, true);
        }
        
        // AoS approach (vector of fitness)
        let aos: Vec<MultiObjectiveFitness> = (0..*size)
            .map(|i| create_test_fitness((i as f64) * 0.01))
            .collect();
        
        group.throughput(Throughput::Elements(*size as u64));
        
        // Sum sharpe ratios (SoA)
        group.bench_with_input(
            BenchmarkId::new("soa_sum", size),
            &soa,
            |b, soa| b.iter(|| {
                let mut sum = 0.0;
                for i in 0..soa.len() {
                    sum += soa.sharpe_ratios[i];
                }
                black_box(sum)
            }),
        );
        
        // Sum sharpe ratios (AoS)
        group.bench_with_input(
            BenchmarkId::new("aos_sum", size),
            &aos,
            |b, aos| b.iter(|| {
                let mut sum = 0.0;
                for fitness in aos {
                    sum += fitness.sharpe_ratio;
                }
                black_box(sum)
            }),
        );
        
        // Find max sharpe (SoA)
        group.bench_with_input(
            BenchmarkId::new("soa_max", size),
            &soa,
            |b, soa| b.iter(|| {
                let mut max = f64::NEG_INFINITY;
                for i in 0..soa.len() {
                    if soa.sharpe_ratios[i] > max {
                        max = soa.sharpe_ratios[i];
                    }
                }
                black_box(max)
            }),
        );
        
        // Find max sharpe (AoS)
        group.bench_with_input(
            BenchmarkId::new("aos_max", size),
            &aos,
            |b, aos| b.iter(|| {
                let mut max = f64::NEG_INFINITY;
                for fitness in aos {
                    if fitness.sharpe_ratio > max {
                        max = fitness.sharpe_ratio;
                    }
                }
                black_box(max)
            }),
        );
    }
    
    group.finish();
}

fn bench_soa_scalar_fitness(c: &mut Criterion) {
    let mut group = c.benchmark_group("soa_scalar_fitness");
    
    for size in [100, 500, 1000].iter() {
        let mut soa = PopulationFitnessSoA::with_capacity(*size);
        for i in 0..*size {
            let sharpe = (i as f64) * 0.01;
            soa.set_fitness(i, sharpe, 0.1, -0.1, 1.0, 1.0, 1.5, 100, 0.12, 2.5, true);
        }
        
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("compute_all", size),
            &soa,
            |b, soa| b.iter(|| {
                for i in 0..soa.len() {
                    black_box(soa.scalar_fitness(i));
                }
            }),
        );
    }
    
    group.finish();
}

// ============================================================================
// Cache Benchmarks
// ============================================================================

fn bench_cache_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_operations");
    
    // Pre-populate cache
    let cache = GenomeCache::with_capacity(10000);
    for i in 0..5000_u64 {
        cache.insert(i, create_test_fitness(1.0), 0);
    }
    
    // Cache hit (existing key)
    group.bench_function("genome_cache_hit", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let key = i % 5000;
            i += 1;
            black_box(cache.get(key))
        })
    });
    
    // Cache miss (non-existing key)
    group.bench_function("genome_cache_miss", |b| {
        let mut i = 5000u64;
        b.iter(|| {
            i += 1;
            black_box(cache.get(i))
        })
    });
    
    // Cache insert
    let cache2 = GenomeCache::with_capacity(10000);
    group.bench_function("genome_cache_insert", |b| {
        let fitness = create_test_fitness(1.0);
        let mut i = 0u64;
        b.iter(|| {
            i += 1;
            cache2.insert(i, fitness.clone(), 0)
        })
    });
    
    group.finish();
}

fn bench_split_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("split_cache");
    
    // Pre-populate cache
    let cache = SplitCache::with_capacity(50000);
    for genome in 0..1000_u64 {
        for split in 0..6_u16 {
            cache.insert(genome, split, SplitMetrics {
                split_index: split,
                oos_sharpe: 0.8,
                ..Default::default()
            });
        }
    }
    
    // Cache hit
    group.bench_function("split_cache_hit", |b| {
        let mut genome = 0u64;
        let mut split = 0u16;
        b.iter(|| {
            let result = cache.get(genome % 1000, split % 6);
            genome += 1;
            split = ((split as u64 + 1) % 6) as u16;
            black_box(result)
        })
    });
    
    // Get all splits for a genome
    group.bench_function("split_cache_get_all", |b| {
        let mut genome = 0u64;
        b.iter(|| {
            let result = cache.get_all_splits(genome % 1000, 6);
            genome += 1;
            black_box(result)
        })
    });
    
    group.finish();
}

// ============================================================================
// Pareto Ranking Benchmarks (placeholder for SIMD version)
// ============================================================================

fn bench_pareto_dominance(c: &mut Criterion) {
    let mut group = c.benchmark_group("pareto_dominance");
    
    for size in [50, 100, 200].iter() {
        let config = FitnessConfig::default();
        let population: Vec<MultiObjectiveFitness> = (0..*size)
            .map(|i| {
                let x = (i as f64) / (*size as f64);
                MultiObjectiveFitness::from_metrics(
                    x * 0.2, // CAGR
                    x * 2.0 - 0.5, // Sharpe
                    -0.1 - x * 0.2, // Max DD
                    x * 1.5, // Calmar
                    x * 1.5, // Sortino
                    1.0 + x, // Profit Factor
                    100,
                    0.15,
                    2.0,
                    &config,
                )
            })
            .collect();
        
        group.throughput(Throughput::Elements(*size as u64 * *size as u64));
        
        // O(n^2) dominance check
        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &population,
            |b, population| b.iter(|| {
                let mut dominated = vec![false; population.len()];
                for i in 0..population.len() {
                    for j in 0..population.len() {
                        if i != j && population[i].dominates(&population[j]) {
                            dominated[j] = true;
                        }
                    }
                }
                black_box(dominated)
            }),
        );
    }
    
    group.finish();
}

// ============================================================================
// Main
// ============================================================================

criterion_group!(
    simd_metrics,
    bench_sharpe_simd_vs_scalar,
    bench_volatility_simd_vs_scalar,
    bench_sortino_simd_vs_scalar,
    bench_max_drawdown,
    bench_all_metrics_batch,
);

criterion_group!(
    soa_layout,
    bench_soa_iteration,
    bench_soa_scalar_fitness,
);

criterion_group!(
    cache_perf,
    bench_cache_operations,
    bench_split_cache,
);

criterion_group!(
    pareto,
    bench_pareto_dominance,
);

criterion_main!(simd_metrics, soa_layout, cache_perf, pareto);

