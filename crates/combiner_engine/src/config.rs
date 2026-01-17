//! Evolution configuration.

use backtester_execution::{ExecutionModelConfig, InstitutionalGatesConfig};
use serde::{Deserialize, Serialize};

/// Configuration for the evolution process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    /// Population size per generation.
    #[serde(default = "default_population_size")]
    pub population_size: usize,

    /// Maximum number of generations.
    #[serde(default = "default_max_generations")]
    pub max_generations: u32,

    /// Tournament size for selection.
    #[serde(default = "default_tournament_size")]
    pub tournament_size: usize,

    /// Crossover probability (0.0 - 1.0).
    #[serde(default = "default_crossover_rate")]
    pub crossover_rate: f64,

    /// Mutation probability per gene (0.0 - 1.0).
    #[serde(default = "default_mutation_rate")]
    pub mutation_rate: f64,

    /// Elitism rate (fraction of population preserved).
    #[serde(default = "default_elitism_rate")]
    pub elitism_rate: f64,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,

    /// Convergence threshold (stop if Pareto doesn't change for N generations).
    #[serde(default = "default_convergence_generations")]
    pub convergence_generations: u32,

    /// Maximum runtime in seconds (0 = unlimited).
    #[serde(default)]
    pub max_runtime_seconds: u64,

    /// Number of parallel backtest workers.
    #[serde(default = "default_workers")]
    pub workers: usize,

    /// Hall of Fame size.
    #[serde(default = "default_hall_of_fame_size")]
    pub hall_of_fame_size: usize,

    /// Execution model configuration for Stage B validation.
    /// This controls slippage, fees, delay, and other execution costs.
    #[serde(default)]
    pub execution: ExecutionModelConfig,

    /// Institutional gates configuration.
    /// Hard constraints that candidates must pass before entering the Pareto frontier.
    #[serde(default)]
    pub gates: InstitutionalGatesConfig,

    /// Enable stress testing suite for top candidates.
    #[serde(default)]
    pub stress_testing_enabled: bool,

    /// Minimum stress scenarios that must pass (out of 5).
    #[serde(default = "default_min_stress_pass")]
    pub min_stress_scenarios_passed: usize,
    
    /// Validation tier for Stage B criteria.
    /// Options: "production" (strictest), "research" (default), "lenient" (debugging)
    /// Use "lenient" to diagnose why candidates aren't passing Stage B.
    #[serde(default = "default_validation_tier")]
    pub validation_tier: String,

    /// Incremental cleanup interval (generations between pending file cleanups).
    /// Set to 0 to disable. Default: 25 generations.
    /// This prevents disk explosion during long runs by removing .obfs files
    /// for genomes no longer in Hall of Fame.
    #[serde(default = "default_incremental_cleanup_interval")]
    pub incremental_cleanup_interval: u32,

    /// Parquet compaction interval (generations between compaction runs).
    /// Set to 0 to disable. Default: 500 generations.
    /// Merges small Parquet files into larger ones to reduce file count.
    #[serde(default = "default_compaction_interval")]
    pub compaction_interval: u32,

    /// Minimum number of Parquet files before compaction triggers.
    /// Default: 50 files.
    #[serde(default = "default_compaction_min_files")]
    pub compaction_min_files: usize,

    /// Target size for compacted Parquet files in MB.
    /// Default: 50 MB.
    #[serde(default = "default_compaction_target_size_mb")]
    pub compaction_target_size_mb: f64,
}

fn default_population_size() -> usize {
    100
}
fn default_max_generations() -> u32 {
    50
}
fn default_tournament_size() -> usize {
    3
}
fn default_crossover_rate() -> f64 {
    0.85
}
fn default_mutation_rate() -> f64 {
    0.15 // Increased from 0.1 to improve exploration and avoid stagnation
}
fn default_elitism_rate() -> f64 {
    0.08 // Reduced from 0.1 to allow more diversity
}
fn default_convergence_generations() -> u32 {
    10
}
fn default_workers() -> usize {
    num_cpus::get().min(8)
}
fn default_hall_of_fame_size() -> usize {
    25
}
fn default_min_stress_pass() -> usize {
    4
}
fn default_validation_tier() -> String {
    "research".to_string()
}
fn default_incremental_cleanup_interval() -> u32 {
    25
}
fn default_compaction_interval() -> u32 {
    500
}
fn default_compaction_min_files() -> usize {
    50
}
fn default_compaction_target_size_mb() -> f64 {
    50.0
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: default_population_size(),
            max_generations: default_max_generations(),
            tournament_size: default_tournament_size(),
            crossover_rate: default_crossover_rate(),
            mutation_rate: default_mutation_rate(),
            elitism_rate: default_elitism_rate(),
            seed: None,
            convergence_generations: default_convergence_generations(),
            max_runtime_seconds: 0,
            workers: default_workers(),
            hall_of_fame_size: default_hall_of_fame_size(),
            execution: ExecutionModelConfig::mvp(),
            gates: InstitutionalGatesConfig::default(),
            stress_testing_enabled: false,
            min_stress_scenarios_passed: default_min_stress_pass(),
            validation_tier: default_validation_tier(),
            incremental_cleanup_interval: default_incremental_cleanup_interval(),
            compaction_interval: default_compaction_interval(),
            compaction_min_files: default_compaction_min_files(),
            compaction_target_size_mb: default_compaction_target_size_mb(),
        }
    }
}

impl EvolutionConfig {
    /// Create a configuration optimized for production use.
    /// Uses conservative execution costs and enables stress testing.
    #[must_use]
    pub fn production() -> Self {
        Self {
            execution: ExecutionModelConfig::mvp(),
            gates: InstitutionalGatesConfig::default(),
            stress_testing_enabled: true,
            min_stress_scenarios_passed: 4,
            ..Default::default()
        }
    }

    /// Create a configuration for fast development/testing.
    /// Uses zero costs and disables stress testing.
    #[must_use]
    pub fn development() -> Self {
        Self {
            execution: ExecutionModelConfig::zero_cost(),
            stress_testing_enabled: false,
            ..Default::default()
        }
    }

    /// Create B3 institutional grade configuration.
    #[must_use]
    pub fn b3_institutional() -> Self {
        Self {
            execution: ExecutionModelConfig::b3_institutional(),
            gates: InstitutionalGatesConfig::default(),
            stress_testing_enabled: true,
            min_stress_scenarios_passed: 4,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EvolutionConfig::default();
        assert_eq!(config.population_size, 100);
        assert_eq!(config.max_generations, 50);
        assert!(config.crossover_rate > 0.0);
        assert!(config.execution.has_costs());
    }

    #[test]
    fn test_production_config() {
        let config = EvolutionConfig::production();
        assert!(config.execution.has_costs());
        assert!(config.stress_testing_enabled);
    }

    #[test]
    fn test_development_config() {
        let config = EvolutionConfig::development();
        assert!(!config.execution.has_costs());
        assert!(!config.stress_testing_enabled);
    }

    #[test]
    fn test_serialization() {
        let config = EvolutionConfig::default();
        let toml_str = toml::to_string(&config).expect("Failed to serialize");
        assert!(toml_str.contains("population_size"));
        
        let parsed: EvolutionConfig = toml::from_str(&toml_str).expect("Failed to deserialize");
        assert_eq!(parsed.population_size, config.population_size);
    }
}
