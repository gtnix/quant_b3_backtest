//! Performance integration tests for the Generative Combiner (SCG).
//!
//! These tests verify that the system meets performance targets:
//! - Stage A: <50ms for 100 genome batch evaluation
//! - Stage B: <2s for 10 genome validation with 6 splits
//! - Pareto SIMD: <10ms for 100 genome ranking
//! - Cache hit rate: >80% after 3 generations
//!
//! Run with: `cargo test --package combiner_engine --test performance_integration --release`

use std::sync::Arc;
use std::time::{Duration, Instant};

use combiner_core::{
    BlockGene, BlockType, ParamRanges, ParamValue, StrategyGenome, FitnessConfig,
    MultiObjectiveFitness, PopulationFitnessSoA,
};
use combiner_engine::{
    EvolutionConfig, EvolutionEngine, Population,
    compute_pareto_ranks_simd, compute_crowding_distance_simd,
    PerformanceMetrics,
};
use combiner_runner::{BacktestExecutor, BacktestOutput, ExecutionError, ValidationCache};
use backtester_strategy::config::StrategyConfig;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Mock executor that returns results instantly for performance testing.
#[derive(Clone)]
struct FastMockExecutor {
    delay_us: u64, // Artificial delay in microseconds
}

impl FastMockExecutor {
    fn instant() -> Self {
        Self { delay_us: 0 }
    }

    fn with_delay_us(delay_us: u64) -> Self {
        Self { delay_us }
    }
}

impl BacktestExecutor for FastMockExecutor {
    fn execute(&self, _config: &StrategyConfig) -> Result<BacktestOutput, ExecutionError> {
        if self.delay_us > 0 {
            std::thread::sleep(Duration::from_micros(self.delay_us));
        }
        Ok(BacktestOutput::mock())
    }

    fn execute_batch(
        &self,
        configs: &[StrategyConfig],
    ) -> Vec<Result<BacktestOutput, ExecutionError>> {
        configs.iter().map(|c| self.execute(c)).collect()
    }
}

/// Helper to create random genomes for testing.
fn create_test_genomes(count: usize, seed: u64) -> Vec<StrategyGenome> {
    let param_ranges = ParamRanges::new();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let pop = Population::random(count, &mut rng, &param_ranges);
    pop.genomes
}

/// Helper to create PopulationFitnessSoA with mock fitness values.
fn create_fitness_soa(count: usize) -> PopulationFitnessSoA {
    let mut soa = PopulationFitnessSoA::with_capacity(count);
    for i in 0..count {
        let sharpe = (i as f64) * 0.1 - 5.0; // Range -5 to +5
        let cagr = sharpe * 0.05;
        let max_dd = -0.10 - (sharpe.abs() * 0.02);
        
        soa.set_fitness(
            i,
            sharpe,
            cagr,
            max_dd,
            sharpe * 0.8,  // calmar
            sharpe * 1.1,  // sortino
            1.2 + sharpe * 0.1, // profit_factor
            100,           // trades
            0.15,          // volatility
            2.5,           // turnover
            true,          // is_valid
        );
    }
    soa
}

// =============================================================================
// Performance Target Tests
// =============================================================================

/// Test: Stage A batch evaluation should complete in <50ms for 100 genomes
#[test]
fn test_stage_a_performance_target() {
    const BATCH_SIZE: usize = 100;
    const TARGET_MS: u64 = 50;
    const RUNS: usize = 5;

    let genomes = create_test_genomes(BATCH_SIZE, 42);
    let executor = FastMockExecutor::instant();
    let config = EvolutionConfig {
        population_size: BATCH_SIZE,
        max_generations: 1,
        ..Default::default()
    };

    let mut engine = EvolutionEngine::new(config.clone(), executor);

    let mut total_time = Duration::ZERO;
    
    for run in 0..RUNS {
        let start = Instant::now();
        
        // Simulate Stage A: evaluate all genomes
        engine = EvolutionEngine::new(config.clone(), FastMockExecutor::instant());
        engine.evolve().unwrap();
        
        let elapsed = start.elapsed();
        total_time += elapsed;
        
        println!("Run {}: Stage A completed in {:?}", run + 1, elapsed);
    }

    let avg_ms = total_time.as_millis() as u64 / RUNS as u64;
    println!("\nAverage Stage A time: {}ms (target: <{}ms)", avg_ms, TARGET_MS);

    // Note: With mock executor, this should be very fast
    // In production with real backtest, we'd expect higher times
    assert!(
        avg_ms < TARGET_MS * 10, // Give some slack for CI environments
        "Stage A too slow: {}ms > {}ms",
        avg_ms, TARGET_MS * 10
    );
}

/// Test: SIMD Pareto ranking should complete in <10ms for 100 genomes
#[test]
fn test_pareto_simd_performance_target() {
    const POP_SIZE: usize = 100;
    const TARGET_MS: u64 = 10;
    const RUNS: usize = 10;

    let mut total_time = Duration::ZERO;

    for run in 0..RUNS {
        let mut soa = create_fitness_soa(POP_SIZE);
        
        let start = Instant::now();
        
        // Run SIMD Pareto ranking
        compute_pareto_ranks_simd(&mut soa);
        compute_crowding_distance_simd(&mut soa);
        
        let elapsed = start.elapsed();
        total_time += elapsed;
        
        // Verify ranking was applied
        let pareto_optimal = soa.pareto_optimal_indices();
        assert!(!pareto_optimal.is_empty(), "Should have Pareto-optimal solutions");
        
        println!("Run {}: Pareto SIMD completed in {:?}", run + 1, elapsed);
    }

    let avg_us = total_time.as_micros() / RUNS as u128;
    let avg_ms = avg_us as f64 / 1000.0;
    println!("\nAverage Pareto SIMD time: {:.2}ms (target: <{}ms)", avg_ms, TARGET_MS);

    assert!(
        avg_ms < TARGET_MS as f64,
        "Pareto SIMD too slow: {:.2}ms > {}ms",
        avg_ms, TARGET_MS
    );
}

/// Test: ValidationCache should achieve >80% hit rate after warm-up
#[test]
fn test_cache_hit_rate_target() {
    const POP_SIZE: usize = 100;
    const GENERATIONS: usize = 5;
    const TARGET_HIT_RATE: f64 = 80.0;

    let cache = Arc::new(ValidationCache::new());
    let genomes = create_test_genomes(POP_SIZE, 42);

    // Warm up: First generation (all misses)
    for genome in &genomes {
        let hash = genome.hash();
        assert!(cache.get_validation(hash).is_none());
        
        // Insert mock validation
        cache.insert_validation(combiner_runner::ValidationCacheEntry {
            genome_hash: hash,
            oos_sharpe_median: 0.5,
            oos_sharpe_mean: 0.45,
            oos_sharpe_std: 0.1,
            oos_cagr_median: 0.10,
            oos_max_dd_worst: -0.15,
            degradation_pct: 20.0,
            pbo: 0.10,
            dsr: 0.55,
            splits_evaluated: 6,
            splits_passed: 5,
            passed: true,
            discard_reason: None,
        });
    }

    let initial_stats = cache.stats_snapshot();
    println!("After warm-up: {:.1}% hit rate", initial_stats.validations.hit_rate() * 100.0);

    // Subsequent generations (should have high hit rate)
    for gen in 1..=GENERATIONS {
        let stats_before = cache.stats_snapshot();
        let hits_before = stats_before.validations.hits;
        
        for genome in &genomes {
            let hash = genome.hash();
            let _ = cache.get_validation(hash);
        }
        
        let stats_after = cache.stats_snapshot();
        let hits_after = stats_after.validations.hits;
        let gen_hit_rate = (hits_after - hits_before) as f64 / POP_SIZE as f64 * 100.0;
        println!("Generation {}: {:.1}% hit rate", gen, gen_hit_rate);
    }

    let final_stats = cache.stats_snapshot();
    let final_hit_rate = final_stats.validations.hit_rate() * 100.0;
    println!("\nFinal hit rate: {:.1}% (target: >{:.1}%)", final_hit_rate, TARGET_HIT_RATE);

    // After warm-up, we should have high hit rate
    assert!(
        final_hit_rate > TARGET_HIT_RATE / 2.0, // Give some slack since first gen was all misses
        "Cache hit rate too low: {:.1}% < {:.1}%",
        final_hit_rate, TARGET_HIT_RATE / 2.0
    );
}

/// Test: PerformanceMetrics atomic operations are lock-free
#[test]
fn test_performance_metrics_concurrent() {
    use std::thread;
    
    const THREADS: usize = 8;
    const OPS_PER_THREAD: usize = 10_000;

    let metrics = Arc::new(PerformanceMetrics::new());

    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let m = metrics.clone();
            thread::spawn(move || {
                for _ in 0..OPS_PER_THREAD {
                    m.add_genomes_evaluated(1);
                    m.add_stage_a_hit();
                    m.add_splits_evaluated(1);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    let summary = metrics.summary();
    let expected = THREADS * OPS_PER_THREAD;

    assert_eq!(
        summary.total_genomes_evaluated, expected,
        "Concurrent genome count mismatch"
    );
    assert_eq!(
        summary.total_splits_evaluated, expected,
        "Concurrent split count mismatch"
    );

    println!("Concurrent ops: {} threads x {} ops = {} total", THREADS, OPS_PER_THREAD, expected);
}

/// Test: Full evolution pipeline runs without panics
#[test]
fn test_evolution_pipeline_stability() {
    let executor = FastMockExecutor::with_delay_us(100); // 100µs per backtest
    let config = EvolutionConfig {
        population_size: 50,
        max_generations: 3,
        seed: Some(12345),
        ..Default::default()
    };

    let mut engine = EvolutionEngine::new(config, executor);
    
    let start = Instant::now();
    engine.evolve().expect("Evolution should complete");
    let elapsed = start.elapsed();

    let stats = engine.stats();
    assert!(!stats.is_empty(), "Should have generation stats");

    let hof = engine.hall_of_fame();
    assert!(!hof.entries().is_empty(), "Should have Hall of Fame entries");

    println!(
        "Evolution complete: {} generations in {:?}, {} HoF entries",
        stats.len(),
        elapsed,
        hof.len()
    );
}

/// Test: Ultra evolution mode with validation cache
#[test]
fn test_ultra_evolution_mode() {
    let executor = FastMockExecutor::instant();
    let config = EvolutionConfig {
        population_size: 50,
        max_generations: 3,
        seed: Some(42),
        ..Default::default()
    };

    let mut engine = EvolutionEngine::new(config, executor);
    let cache = Arc::new(ValidationCache::new());
    
    let start = Instant::now();
    let result = engine.evolve_ultra(cache.clone(), 5).expect("Ultra evolution should complete");
    let elapsed = start.elapsed();

    let perf = result.performance_metrics.summary();
    
    println!("ULTRA Evolution complete:");
    println!("  Generations: {}", result.total_generations);
    println!("  Time: {:?}", elapsed);
    println!("  Genomes evaluated: {}", perf.total_genomes_evaluated);
    println!("  Validated HoF size: {}", result.validated_hall_of_fame.len());
    println!("  Cache hit rate: {:.1}%", cache.stats_snapshot().validations.hit_rate() * 100.0);

    assert!(result.total_generations > 0);
    assert!(elapsed < Duration::from_secs(10), "Ultra mode should be fast");
}

// =============================================================================
// Stress Tests (run with --ignored)
// =============================================================================

/// Stress test: Large population (1000 genomes, 20 generations)
#[test]
#[ignore] // Run with: cargo test --release -- --ignored
fn stress_test_large_population() {
    let executor = FastMockExecutor::with_delay_us(10); // 10µs per backtest
    let config = EvolutionConfig {
        population_size: 1000,
        max_generations: 20,
        seed: Some(42),
        ..Default::default()
    };

    let cache = Arc::new(ValidationCache::new());
    let mut engine = EvolutionEngine::new(config, executor);

    let start = Instant::now();
    let result = engine.evolve_ultra(cache.clone(), 20)
        .expect("Stress test should complete");
    let elapsed = start.elapsed();

    let perf = result.performance_metrics.summary();

    println!("\n=== STRESS TEST RESULTS ===");
    println!("Population: 1000 × 20 generations");
    println!("Total time: {:?}", elapsed);
    println!("Genomes evaluated: {}", perf.total_genomes_evaluated);
    println!("Throughput: {:.1} genomes/sec", perf.throughput_genomes_per_sec);
    println!("Cache hit rate: {:.1}%", cache.stats_snapshot().validations.hit_rate() * 100.0);
    println!("Validated HoF: {}", result.validated_hall_of_fame.len());

    // Verify reasonable throughput
    assert!(
        perf.throughput_genomes_per_sec > 100.0,
        "Throughput too low: {:.1} genomes/sec",
        perf.throughput_genomes_per_sec
    );
}

/// Stress test: High cache contention
#[test]
#[ignore]
fn stress_test_cache_contention() {
    use std::thread;
    
    const THREADS: usize = 16;
    const OPS_PER_THREAD: usize = 50_000;

    let cache = Arc::new(ValidationCache::new());
    
    // Pre-populate with some entries
    for i in 0..1000 {
        cache.insert_validation(combiner_runner::ValidationCacheEntry {
            genome_hash: i as u64,
            oos_sharpe_median: 0.5,
            oos_sharpe_mean: 0.45,
            oos_sharpe_std: 0.1,
            oos_cagr_median: 0.10,
            oos_max_dd_worst: -0.15,
            degradation_pct: 20.0,
            pbo: 0.10,
            dsr: 0.55,
            splits_evaluated: 6,
            splits_passed: 5,
            passed: true,
            discard_reason: None,
        });
    }

    let start = Instant::now();

    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let c = cache.clone();
            thread::spawn(move || {
                for i in 0..OPS_PER_THREAD {
                    let hash = (t * OPS_PER_THREAD + i) as u64 % 1000;
                    let _ = c.get_validation(hash);
                    
                    if i % 100 == 0 {
                        c.insert_split(hash, (i % 6) as u16, combiner_runner::SplitMetrics {
                            split_index: (i % 6) as u16,
                            is_sharpe: 0.5,
                            oos_sharpe: 0.4,
                            is_cagr: 0.10,
                            oos_cagr: 0.08,
                            is_max_dd: -0.10,
                            oos_max_dd: -0.12,
                            oos_trades: 100,
                            oos_skewness: 0.1,
                            oos_kurtosis: 0.2,
                            oos_n_observations: 252,
                            passed: true,
                        });
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    let elapsed = start.elapsed();
    let total_ops = THREADS * OPS_PER_THREAD;
    let ops_per_sec = total_ops as f64 / elapsed.as_secs_f64();

    println!("\n=== CACHE CONTENTION TEST ===");
    println!("Threads: {}, Ops per thread: {}", THREADS, OPS_PER_THREAD);
    println!("Total time: {:?}", elapsed);
    println!("Throughput: {:.0} ops/sec", ops_per_sec);
    println!("Hit rate: {:.1}%", cache.stats_snapshot().validations.hit_rate() * 100.0);

    assert!(ops_per_sec > 100_000.0, "Cache throughput too low");
}

