//! Evolution configuration.

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
    0.1
}
fn default_elitism_rate() -> f64 {
    0.1
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
    }
}

