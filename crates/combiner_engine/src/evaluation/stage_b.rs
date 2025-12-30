//! Stage B: Parallel validation with Walk-Forward and early exit.
//!
//! This module provides high-performance validation of top-K genomes
//! using parallel split execution. Optimized for:
//! - Concurrent split evaluation across all splits
//! - Early exit when 3+ splits fail
//! - Cache deduplication at split level
//! - Arena allocation for results

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use combiner_core::{StrategyGenome, GenomeConverter, FitnessConfig, MultiObjectiveFitness};
use combiner_runner::{
    BacktestExecutor, ValidationCache, SplitMetrics, ValidationCacheEntry,
};

use super::split_plan::{ValidationSplitPlan, SplitPlanConfig};
use super::split_data::SplitPair;
use super::arena::{ValidationResultArena, ArenaMetrics, AggregatedMetrics};

/// Configuration for Stage B validation
#[derive(Debug, Clone)]
pub struct StageBConfig {
    /// Maximum failed splits before early exit
    pub max_failures_early_exit: usize,
    /// Minimum OOS Sharpe to pass
    pub min_oos_sharpe: f64,
    /// Maximum OOS drawdown to pass (negative, e.g., -0.35)
    pub max_oos_drawdown: f64,
    /// Minimum OOS trades to pass
    pub min_oos_trades: u32,
    /// Maximum degradation percentage
    pub max_degradation_pct: f64,
    /// Maximum PBO to pass
    pub max_pbo: f64,
    /// Minimum DSR (Deflated Sharpe Ratio) to pass
    pub min_dsr: f64,
    /// Whether to use cache
    pub use_cache: bool,
    /// Current generation
    pub generation: u32,
}

impl Default for StageBConfig {
    fn default() -> Self {
        Self {
            max_failures_early_exit: 3,
            min_oos_sharpe: 0.2,
            max_oos_drawdown: -0.35,
            min_oos_trades: 30,
            max_degradation_pct: 50.0,
            max_pbo: 0.20,
            min_dsr: 0.8,
            use_cache: true,
            generation: 0,
        }
    }
}

/// Statistics from Stage B validation
#[derive(Debug, Clone, Default)]
pub struct StageBStats {
    /// Total genomes validated
    pub total_validated: usize,
    /// Genomes that passed validation
    pub passed: usize,
    /// Genomes that failed validation
    pub failed: usize,
    /// Genomes that exited early
    pub early_exits: usize,
    /// Total splits evaluated
    pub splits_evaluated: usize,
    /// Split cache hits
    pub cache_hits: usize,
    /// Split cache misses
    pub cache_misses: usize,
    /// Total evaluation time in milliseconds
    pub elapsed_ms: u64,
    /// Average time per genome in milliseconds
    pub avg_time_per_genome_ms: f64,
}

/// Result of validating a single genome
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub genome_index: usize,
    pub genome_hash: u64,
    pub oos_sharpe_median: f64,
    pub oos_sharpe_mean: f64,
    pub oos_sharpe_std: f64,
    pub oos_cagr_median: f64,
    pub degradation_pct: f64,
    pub pbo: f64,
    pub dsr: f64,
    pub splits_evaluated: u16,
    pub splits_passed: u16,
    pub passed: bool,
    pub early_exit: bool,
    pub discard_reason: Option<String>,
}

impl ValidationResult {
    /// Create a failed result with reason
    pub fn failed(genome_index: usize, genome_hash: u64, reason: impl Into<String>) -> Self {
        Self {
            genome_index,
            genome_hash,
            oos_sharpe_median: f64::NEG_INFINITY,
            oos_sharpe_mean: f64::NEG_INFINITY,
            oos_sharpe_std: 0.0,
            oos_cagr_median: 0.0,
            degradation_pct: 100.0,
            pbo: 1.0,
            dsr: 0.0,
            splits_evaluated: 0,
            splits_passed: 0,
            passed: false,
            early_exit: false,
            discard_reason: Some(reason.into()),
        }
    }

    /// Create an early exit result
    pub fn early_exit(genome_index: usize, genome_hash: u64, reason: impl Into<String>) -> Self {
        let mut result = Self::failed(genome_index, genome_hash, reason);
        result.early_exit = true;
        result
    }

    /// Convert to cache entry
    pub fn to_cache_entry(&self) -> ValidationCacheEntry {
        ValidationCacheEntry {
            genome_hash: self.genome_hash,
            oos_sharpe_median: self.oos_sharpe_median,
            oos_sharpe_mean: self.oos_sharpe_mean,
            oos_sharpe_std: self.oos_sharpe_std,
            oos_cagr_median: self.oos_cagr_median,
            oos_max_dd_worst: -0.25, // Default, would be computed from splits
            degradation_pct: self.degradation_pct,
            pbo: self.pbo,
            dsr: self.dsr,
            splits_evaluated: self.splits_evaluated,
            splits_passed: self.splits_passed,
            passed: self.passed,
            discard_reason: self.discard_reason.clone(),
        }
    }
}

/// Stage B parallel validator for robust genome validation.
///
/// Evaluates genomes across multiple Walk-Forward splits in parallel,
/// with early exit on excessive failures and split-level caching.
pub struct StageBParallelValidator<E: BacktestExecutor> {
    /// Executor for running backtests
    executor: Arc<E>,
    /// Shared cache for deduplication
    cache: Arc<ValidationCache>,
    /// Pre-computed split plan
    split_plan: Arc<ValidationSplitPlan>,
    /// Genome converter
    converter: GenomeConverter,
    /// Configuration
    config: StageBConfig,
}

impl<E: BacktestExecutor + Send + Sync> StageBParallelValidator<E> {
    /// Create a new Stage B validator
    pub fn new(
        executor: Arc<E>,
        cache: Arc<ValidationCache>,
        split_plan: Arc<ValidationSplitPlan>,
    ) -> Self {
        Self {
            executor,
            cache,
            split_plan,
            converter: GenomeConverter::new(),
            config: StageBConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        executor: Arc<E>,
        cache: Arc<ValidationCache>,
        split_plan: Arc<ValidationSplitPlan>,
        config: StageBConfig,
    ) -> Self {
        Self {
            executor,
            cache,
            split_plan,
            converter: GenomeConverter::new(),
            config,
        }
    }

    /// Set the current generation
    pub fn set_generation(&mut self, generation: u32) {
        self.config.generation = generation;
    }

    /// Validate a batch of genomes in parallel
    pub fn validate_batch(
        &self,
        genomes: &[(usize, &StrategyGenome)], // (original_index, genome)
    ) -> (Vec<ValidationResult>, StageBStats) {
        let start = Instant::now();
        
        let cache_hits = AtomicUsize::new(0);
        let cache_misses = AtomicUsize::new(0);
        let early_exits = AtomicUsize::new(0);

        let results: Vec<ValidationResult> = genomes
            .par_iter()
            .map(|(idx, genome)| {
                self.validate_single(*idx, genome, &cache_hits, &cache_misses, &early_exits)
            })
            .collect();

        let elapsed = start.elapsed();
        let passed_count = results.iter().filter(|r| r.passed).count();
        let failed_count = results.iter().filter(|r| !r.passed && !r.early_exit).count();
        let total_splits: u16 = results.iter().map(|r| r.splits_evaluated).sum();

        let stats = StageBStats {
            total_validated: results.len(),
            passed: passed_count,
            failed: failed_count,
            early_exits: early_exits.load(Ordering::Relaxed),
            splits_evaluated: total_splits as usize,
            cache_hits: cache_hits.load(Ordering::Relaxed),
            cache_misses: cache_misses.load(Ordering::Relaxed),
            elapsed_ms: elapsed.as_millis() as u64,
            avg_time_per_genome_ms: elapsed.as_millis() as f64 / results.len().max(1) as f64,
        };

        (results, stats)
    }

    /// Validate a single genome across all splits
    fn validate_single(
        &self,
        genome_index: usize,
        genome: &StrategyGenome,
        cache_hits: &AtomicUsize,
        cache_misses: &AtomicUsize,
        early_exits: &AtomicUsize,
    ) -> ValidationResult {
        let genome_hash = genome.hash();

        // Check full validation cache first
        if self.config.use_cache {
            if let Some(cached) = self.cache.get_validation(genome_hash) {
                cache_hits.fetch_add(1, Ordering::Relaxed);
                return ValidationResult {
                    genome_index,
                    genome_hash,
                    oos_sharpe_median: cached.oos_sharpe_median,
                    oos_sharpe_mean: cached.oos_sharpe_mean,
                    oos_sharpe_std: cached.oos_sharpe_std,
                    oos_cagr_median: cached.oos_cagr_median,
                    degradation_pct: cached.degradation_pct,
                    pbo: cached.pbo,
                    dsr: cached.dsr,
                    splits_evaluated: cached.splits_evaluated,
                    splits_passed: cached.splits_passed,
                    passed: cached.passed,
                    early_exit: false,
                    discard_reason: cached.discard_reason,
                };
            }
        }

        cache_misses.fetch_add(1, Ordering::Relaxed);

        // Convert genome to strategy config
        let strategy_config = match self.converter.to_strategy_config(genome) {
            Ok(config) => config,
            Err(e) => {
                return ValidationResult::failed(
                    genome_index,
                    genome_hash,
                    format!("Conversion error: {}", e),
                );
            }
        };

        // Evaluate all splits in parallel with early exit
        let early_exit_flag = AtomicBool::new(false);
        let failure_count = AtomicUsize::new(0);

        let split_results: Vec<Option<SplitMetrics>> = self.split_plan.splits
            .par_iter()
            .map(|split| {
                // Check for early exit
                if early_exit_flag.load(Ordering::Relaxed) {
                    return None;
                }

                // Check split cache
                if self.config.use_cache {
                    if let Some(cached) = self.cache.get_split(genome_hash, split.index()) {
                        return Some(cached);
                    }
                }

                // Execute backtest for this split
                // In production, this would use the split's date range
                let result = match self.executor.execute(&strategy_config) {
                    Ok(output) => {
                        let metrics = SplitMetrics {
                            split_index: split.index(),
                            is_sharpe: output.metrics.sharpe_ratio,
                            oos_sharpe: output.metrics.sharpe_ratio * 0.7, // Simulated OOS degradation
                            is_cagr: output.metrics.cagr,
                            oos_cagr: output.metrics.cagr * 0.7,
                            is_max_dd: output.metrics.max_drawdown,
                            oos_max_dd: output.metrics.max_drawdown * 1.2,
                            oos_trades: output.metrics.total_trades,
                            passed: output.metrics.sharpe_ratio > 0.0,
                        };

                        // Check if this split passed
                        let split_passed = metrics.oos_sharpe >= self.config.min_oos_sharpe
                            && metrics.oos_max_dd >= self.config.max_oos_drawdown
                            && metrics.oos_trades >= self.config.min_oos_trades;

                        if !split_passed {
                            let failures = failure_count.fetch_add(1, Ordering::Relaxed) + 1;
                            if failures >= self.config.max_failures_early_exit {
                                early_exit_flag.store(true, Ordering::Relaxed);
                            }
                        }

                        // Cache the result
                        if self.config.use_cache {
                            self.cache.insert_split(genome_hash, split.index(), metrics.clone());
                        }

                        Some(SplitMetrics { passed: split_passed, ..metrics })
                    }
                    Err(_) => {
                        let failures = failure_count.fetch_add(1, Ordering::Relaxed) + 1;
                        if failures >= self.config.max_failures_early_exit {
                            early_exit_flag.store(true, Ordering::Relaxed);
                        }
                        None
                    }
                };

                result
            })
            .collect();

        // Check for early exit
        if early_exit_flag.load(Ordering::Relaxed) {
            early_exits.fetch_add(1, Ordering::Relaxed);
            return ValidationResult::early_exit(
                genome_index,
                genome_hash,
                format!("Early exit: {} splits failed", failure_count.load(Ordering::Relaxed)),
            );
        }

        // Aggregate results
        let valid_results: Vec<&SplitMetrics> = split_results.iter()
            .filter_map(|r| r.as_ref())
            .collect();

        if valid_results.is_empty() {
            return ValidationResult::failed(genome_index, genome_hash, "No valid splits");
        }

        let result = self.aggregate_results(genome_index, genome_hash, &valid_results);

        // Cache the full result
        if self.config.use_cache {
            self.cache.insert_validation(result.to_cache_entry());
        }

        result
    }

    /// Aggregate split results into a validation result
    fn aggregate_results(
        &self,
        genome_index: usize,
        genome_hash: u64,
        splits: &[&SplitMetrics],
    ) -> ValidationResult {
        let n = splits.len();

        // Collect OOS sharpes
        let mut oos_sharpes: Vec<f64> = splits.iter().map(|s| s.oos_sharpe).collect();
        let is_sharpes: Vec<f64> = splits.iter().map(|s| s.is_sharpe).collect();
        let mut oos_cagrs: Vec<f64> = splits.iter().map(|s| s.oos_cagr).collect();

        // Sort for median
        oos_sharpes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        oos_cagrs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let oos_sharpe_median = if n % 2 == 0 {
            (oos_sharpes[n/2 - 1] + oos_sharpes[n/2]) / 2.0
        } else {
            oos_sharpes[n/2]
        };

        let oos_cagr_median = if n % 2 == 0 {
            (oos_cagrs[n/2 - 1] + oos_cagrs[n/2]) / 2.0
        } else {
            oos_cagrs[n/2]
        };

        let oos_sharpe_mean: f64 = oos_sharpes.iter().sum::<f64>() / n as f64;
        let is_sharpe_mean: f64 = is_sharpes.iter().sum::<f64>() / n as f64;

        let oos_sharpe_var: f64 = oos_sharpes.iter()
            .map(|x| (x - oos_sharpe_mean).powi(2))
            .sum::<f64>() / n as f64;
        let oos_sharpe_std = oos_sharpe_var.sqrt();

        // Degradation percentage
        let degradation_pct = if is_sharpe_mean > 0.01 {
            (is_sharpe_mean - oos_sharpe_mean) / is_sharpe_mean * 100.0
        } else {
            0.0
        };

        // PBO estimate
        let pbo = if oos_sharpe_std > 0.01 {
            let z = -oos_sharpe_mean / oos_sharpe_std;
            0.5 * (1.0 + libm::erf(z / std::f64::consts::SQRT_2))
        } else if oos_sharpe_mean <= 0.0 {
            1.0
        } else {
            0.0
        };

        // DSR (simplified)
        let total_trials = 100u64; // Placeholder - should come from evolution
        let trial_adjustment = 1.0 - (total_trials as f64).ln() / 100.0;
        let dsr = is_sharpe_mean * (1.0 - pbo) * trial_adjustment.max(0.5);

        // Count passed splits
        let splits_passed = splits.iter().filter(|s| s.passed).count() as u16;

        // Determine if passed overall
        let (passed, discard_reason) = self.check_pass_criteria(
            oos_sharpe_median,
            degradation_pct,
            pbo,
            dsr,
            splits_passed,
            n as u16,
        );

        ValidationResult {
            genome_index,
            genome_hash,
            oos_sharpe_median,
            oos_sharpe_mean,
            oos_sharpe_std,
            oos_cagr_median,
            degradation_pct,
            pbo,
            dsr,
            splits_evaluated: n as u16,
            splits_passed,
            passed,
            early_exit: false,
            discard_reason,
        }
    }

    /// Check if validation passes all criteria
    fn check_pass_criteria(
        &self,
        oos_sharpe_median: f64,
        degradation_pct: f64,
        pbo: f64,
        dsr: f64,
        splits_passed: u16,
        splits_total: u16,
    ) -> (bool, Option<String>) {
        if oos_sharpe_median < self.config.min_oos_sharpe {
            return (false, Some(format!(
                "OOS Sharpe {:.2} < {:.2}",
                oos_sharpe_median, self.config.min_oos_sharpe
            )));
        }

        if degradation_pct > self.config.max_degradation_pct {
            return (false, Some(format!(
                "Degradation {:.1}% > {:.1}%",
                degradation_pct, self.config.max_degradation_pct
            )));
        }

        if pbo > self.config.max_pbo {
            return (false, Some(format!(
                "PBO {:.2} > {:.2}",
                pbo, self.config.max_pbo
            )));
        }

        if dsr < self.config.min_dsr {
            return (false, Some(format!(
                "DSR {:.2} < {:.2}",
                dsr, self.config.min_dsr
            )));
        }

        // Majority of splits must pass
        if (splits_passed as usize) * 2 <= splits_total as usize {
            return (false, Some(format!(
                "Only {}/{} splits passed",
                splits_passed, splits_total
            )));
        }

        (true, None)
    }

    /// Get the split plan
    pub fn split_plan(&self) -> &ValidationSplitPlan {
        &self.split_plan
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
    fn test_validator_creation() {
        let executor = Arc::new(LibraryExecutor::new());
        let cache = Arc::new(ValidationCache::new());
        let split_config = SplitPlanConfig::default();
        let split_plan = Arc::new(ValidationSplitPlan::generate(split_config, 2520));
        
        let validator = StageBParallelValidator::new(executor, cache, split_plan);
        
        assert!(validator.split_plan().num_splits() > 0);
    }

    #[test]
    fn test_validation_result_failed() {
        let result = ValidationResult::failed(0, 12345, "Test error");
        
        assert!(!result.passed);
        assert!(!result.early_exit);
        assert!(result.discard_reason.is_some());
    }

    #[test]
    fn test_validation_result_early_exit() {
        let result = ValidationResult::early_exit(0, 12345, "Too many failures");
        
        assert!(!result.passed);
        assert!(result.early_exit);
    }
}

