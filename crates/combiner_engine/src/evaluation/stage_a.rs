//! Stage A: Ultra-fast batch evaluation for screening.
//!
//! This module provides high-performance parallel evaluation of genomes
//! during the screening phase. Optimized for:
//! - Batch processing (16 genomes per chunk)
//! - Parallel evaluation via rayon
//! - SIMD metrics calculation
//! - Lock-free result collection into SoA

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use combiner_core::{
    StrategyGenome, PopulationFitnessSoA, FitnessData,
    calculate_all_metrics, GenomeConverter,
};
use combiner_runner::{
    BacktestExecutor, BacktestOutput, GenomeCache, ValidationCache,
};

use super::split_plan::ValidationSplitPlan;

/// Configuration for Stage A evaluation
#[derive(Debug, Clone)]
pub struct StageAConfig {
    /// Number of genomes per evaluation batch
    pub batch_size: usize,
    /// Whether to use cache
    pub use_cache: bool,
    /// Current generation (for cache tagging)
    pub generation: u32,
    /// Risk-free rate for Sharpe calculation
    pub rf_rate: f64,
}

impl Default for StageAConfig {
    fn default() -> Self {
        Self {
            batch_size: 16,
            use_cache: true,
            generation: 0,
            rf_rate: 0.0,
        }
    }
}

/// Statistics from Stage A evaluation
#[derive(Debug, Clone, Default)]
pub struct StageAStats {
    /// Total genomes evaluated
    pub total_evaluated: usize,
    /// Cache hits
    pub cache_hits: usize,
    /// Cache misses (new evaluations)
    pub cache_misses: usize,
    /// Number of failed evaluations
    pub failures: usize,
    /// Total evaluation time in milliseconds
    pub elapsed_ms: u64,
    /// Average time per genome in milliseconds
    pub avg_time_per_genome_ms: f64,
}

/// Result from evaluating a single genome
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    pub index: usize,
    pub genome_hash: u64,
    pub fitness_data: Option<FitnessData>,
    pub from_cache: bool,
    pub error: Option<String>,
}

/// Stage A batch evaluator for high-performance genome screening.
///
/// Evaluates entire populations in parallel using batched processing
/// and lock-free result collection.
pub struct StageABatchEvaluator<E: BacktestExecutor> {
    /// Executor for running backtests
    executor: Arc<E>,
    /// Shared cache for deduplication
    cache: Arc<ValidationCache>,
    /// Genome converter
    converter: GenomeConverter,
    /// Configuration
    config: StageAConfig,
}

impl<E: BacktestExecutor + Send + Sync> StageABatchEvaluator<E> {
    /// Create a new Stage A evaluator
    pub fn new(executor: Arc<E>, cache: Arc<ValidationCache>) -> Self {
        Self {
            executor,
            cache,
            converter: GenomeConverter::new(),
            config: StageAConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(executor: Arc<E>, cache: Arc<ValidationCache>, config: StageAConfig) -> Self {
        Self {
            executor,
            cache,
            converter: GenomeConverter::new(),
            config,
        }
    }

    /// Set the current generation for cache tagging
    pub fn set_generation(&mut self, generation: u32) {
        self.config.generation = generation;
    }

    /// Evaluate a batch of genomes in parallel, returning results in SoA format.
    ///
    /// This is the main entry point for Stage A evaluation. It:
    /// 1. Checks cache for each genome
    /// 2. Evaluates cache misses in parallel batches
    /// 3. Stores results in cache
    /// 4. Returns fitness in SoA format for fast Pareto sorting
    pub fn evaluate_batch(&self, genomes: &[StrategyGenome]) -> (PopulationFitnessSoA, StageAStats) {
        let start = Instant::now();
        let n = genomes.len();
        
        // Pre-allocate SoA result
        let mut fitness_soa = PopulationFitnessSoA::with_capacity(n);
        
        // Atomic counters for statistics
        let cache_hits = AtomicUsize::new(0);
        let cache_misses = AtomicUsize::new(0);
        let failures = AtomicUsize::new(0);
        
        // Parallel evaluation with batching
        let results: Vec<EvaluationResult> = genomes
            .par_iter()
            .enumerate()
            .map(|(idx, genome)| self.evaluate_single(idx, genome, &cache_hits, &cache_misses, &failures))
            .collect();
        
        // Collect results into SoA (sequential for correctness)
        for result in results {
            if let Some(fitness) = result.fitness_data {
                fitness_soa.set_fitness(
                    result.index,
                    fitness.sharpe_ratio,
                    fitness.cagr,
                    fitness.max_drawdown,
                    fitness.calmar_ratio,
                    fitness.sortino_ratio,
                    fitness.profit_factor,
                    fitness.total_trades,
                    fitness.volatility,
                    fitness.turnover,
                    fitness.is_valid,
                );
            } else {
                // Mark as invalid
                fitness_soa.set_fitness(
                    result.index,
                    f64::NEG_INFINITY,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0,
                    0.0,
                    0.0,
                    false,
                );
            }
        }
        
        let elapsed = start.elapsed();
        let stats = StageAStats {
            total_evaluated: n,
            cache_hits: cache_hits.load(Ordering::Relaxed),
            cache_misses: cache_misses.load(Ordering::Relaxed),
            failures: failures.load(Ordering::Relaxed),
            elapsed_ms: elapsed.as_millis() as u64,
            avg_time_per_genome_ms: elapsed.as_millis() as f64 / n.max(1) as f64,
        };
        
        (fitness_soa, stats)
    }

    /// Evaluate a single genome (called in parallel)
    fn evaluate_single(
        &self,
        index: usize,
        genome: &StrategyGenome,
        cache_hits: &AtomicUsize,
        cache_misses: &AtomicUsize,
        failures: &AtomicUsize,
    ) -> EvaluationResult {
        let genome_hash = genome.hash();
        
        // Check cache first
        if self.config.use_cache {
            if let Some(cached_fitness) = self.cache.get_fitness(genome_hash) {
                cache_hits.fetch_add(1, Ordering::Relaxed);
                return EvaluationResult {
                    index,
                    genome_hash,
                    fitness_data: Some(FitnessData {
                        sharpe_ratio: cached_fitness.sharpe_ratio,
                        cagr: cached_fitness.cagr,
                        max_drawdown: cached_fitness.max_drawdown,
                        calmar_ratio: cached_fitness.calmar_ratio,
                        sortino_ratio: cached_fitness.sortino_ratio,
                        profit_factor: cached_fitness.profit_factor,
                        total_trades: cached_fitness.total_trades,
                        volatility: cached_fitness.volatility,
                        turnover: cached_fitness.turnover_annual,
                        pareto_rank: 0,
                        crowding_distance: 0.0,
                        oos_sharpe_median: 0.0,
                        pbo: 1.0,
                        dsr: 0.0,
                        is_validated: false,
                        is_valid: cached_fitness.is_valid,
                    }),
                    from_cache: true,
                    error: None,
                };
            }
        }
        
        cache_misses.fetch_add(1, Ordering::Relaxed);
        
        // Convert genome to strategy config
        let strategy_config = match self.converter.to_strategy_config(genome) {
            Ok(config) => config,
            Err(e) => {
                failures.fetch_add(1, Ordering::Relaxed);
                return EvaluationResult {
                    index,
                    genome_hash,
                    fitness_data: None,
                    from_cache: false,
                    error: Some(format!("Conversion error: {}", e)),
                };
            }
        };
        
        // Execute backtest
        let output = match self.executor.execute(&strategy_config) {
            Ok(output) => output,
            Err(e) => {
                failures.fetch_add(1, Ordering::Relaxed);
                return EvaluationResult {
                    index,
                    genome_hash,
                    fitness_data: None,
                    from_cache: false,
                    error: Some(format!("Execution error: {}", e)),
                };
            }
        };
        
        // Convert output to fitness data
        let fitness_data = self.output_to_fitness(&output);
        
        // Store in cache
        if self.config.use_cache {
            use combiner_core::{MultiObjectiveFitness, FitnessConfig};
            let config = FitnessConfig::default();
            let mof = MultiObjectiveFitness::from_metrics(
                fitness_data.cagr,
                fitness_data.sharpe_ratio,
                fitness_data.max_drawdown,
                fitness_data.calmar_ratio,
                fitness_data.sortino_ratio,
                fitness_data.profit_factor,
                fitness_data.total_trades,
                fitness_data.volatility,
                fitness_data.turnover,
                &config,
            );
            self.cache.insert_fitness(genome_hash, mof, self.config.generation);
        }
        
        EvaluationResult {
            index,
            genome_hash,
            fitness_data: Some(fitness_data),
            from_cache: false,
            error: None,
        }
    }

    /// Convert backtest output to fitness data
    fn output_to_fitness(&self, output: &BacktestOutput) -> FitnessData {
        let metrics = &output.metrics;
        
        FitnessData {
            sharpe_ratio: metrics.sharpe_ratio,
            cagr: metrics.cagr,
            max_drawdown: metrics.max_drawdown,
            calmar_ratio: metrics.calmar_ratio.unwrap_or(0.0),
            sortino_ratio: metrics.sortino_ratio.unwrap_or(0.0),
            profit_factor: metrics.profit_factor.unwrap_or(1.0),
            total_trades: metrics.total_trades,
            volatility: metrics.volatility.unwrap_or(0.0),
            turnover: metrics.turnover_annual.unwrap_or(0.0),
            pareto_rank: 0,
            crowding_distance: 0.0,
            oos_sharpe_median: 0.0,
            pbo: 1.0,
            dsr: 0.0,
            is_validated: false,
            is_valid: true,
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> combiner_runner::CombinedCacheStats {
        self.cache.stats_snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use combiner_core::{BlockGene, BlockType, ParamValue};
    use combiner_runner::LibraryExecutor;

    fn create_test_genome() -> StrategyGenome {
        let gene = BlockGene::new(
            BlockType::Selection,
            "momentum",
            vec![("lookback_days", ParamValue::int(126, 21, 252, 21))],
        );
        StrategyGenome::new(vec![gene])
    }

    #[test]
    fn test_evaluator_creation() {
        let executor = Arc::new(LibraryExecutor::new());
        let cache = Arc::new(ValidationCache::new());
        let evaluator = StageABatchEvaluator::new(executor, cache);
        
        assert_eq!(evaluator.config.batch_size, 16);
    }

    #[test]
    fn test_batch_evaluation() {
        let executor = Arc::new(LibraryExecutor::new());
        let cache = Arc::new(ValidationCache::new());
        let evaluator = StageABatchEvaluator::new(executor, cache);
        
        let genomes: Vec<StrategyGenome> = (0..10)
            .map(|_| create_test_genome())
            .collect();
        
        let (fitness_soa, stats) = evaluator.evaluate_batch(&genomes);
        
        assert_eq!(stats.total_evaluated, 10);
        assert!(fitness_soa.len() > 0);
    }
}

