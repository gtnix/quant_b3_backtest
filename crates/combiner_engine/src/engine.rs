//! Evolution engine - Main evolution loop.

use crate::config::EvolutionConfig;
use crate::hall_of_fame::HallOfFame;
use crate::operators::{Crossover, Mutation, Selection};
use crate::pareto::ParetoFrontier;
use crate::population::Population;
use combiner_core::{
    FitnessConfig, GenomeValidator, MultiObjectiveFitness, ParamRanges, StrategyGenome,
};
use combiner_runner::{BacktestExecutor, BacktestOutput};
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use std::time::Instant;
use thiserror::Error;
use tracing::{info, warn};

/// Evolution engine errors.
#[derive(Debug, Error)]
pub enum EngineError {
    #[error("No valid genomes in population")]
    NoValidGenomes,

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("Convergence not achieved after {0} generations")]
    NotConverged(u32),
}

/// Statistics for a generation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GenerationStats {
    pub generation: u32,
    pub population_size: usize,
    pub pareto_size: usize,
    pub best_sharpe: f64,
    pub best_cagr: f64,
    pub mean_sharpe: f64,
    pub evaluated: usize,
    pub cache_hits: usize,
    pub duration_ms: u64,
}

/// Evolution engine.
pub struct EvolutionEngine<E: BacktestExecutor> {
    config: EvolutionConfig,
    param_ranges: ParamRanges,
    fitness_config: FitnessConfig,
    validator: GenomeValidator,
    executor: E,
    rng: ChaCha8Rng,
    population: Population,
    hall_of_fame: HallOfFame,
    generation_stats: Vec<GenerationStats>,
    start_time: Option<Instant>,
    cache_hits: usize,
}

impl<E: BacktestExecutor> EvolutionEngine<E> {
    /// Create a new evolution engine.
    pub fn new(config: EvolutionConfig, executor: E) -> Self {
        let seed = config.seed.unwrap_or(42);
        let rng = ChaCha8Rng::seed_from_u64(seed);

        Self {
            param_ranges: ParamRanges::new(),
            fitness_config: FitnessConfig::default(),
            validator: GenomeValidator::new(),
            hall_of_fame: HallOfFame::new(config.hall_of_fame_size),
            config,
            executor,
            rng,
            population: Population::new(),
            generation_stats: Vec::new(),
            start_time: None,
            cache_hits: 0,
        }
    }

    /// Run the evolution process.
    pub fn evolve(&mut self) -> Result<(), EngineError> {
        self.start_time = Some(Instant::now());
        info!("Starting evolution with {} generations, population {}", 
              self.config.max_generations, self.config.population_size);

        // Generate initial population
        self.population = Population::random(
            self.config.population_size,
            &mut self.rng,
            &self.param_ranges,
        );
        info!("Generated initial population of {} genomes", self.population.len());

        // Main evolution loop
        for gen in 0..self.config.max_generations {
            let gen_start = Instant::now();

            // Evaluate population
            self.evaluate_population()?;

            // Compute Pareto fronts
            ParetoFrontier::compute(&mut self.population.genomes);

            // Update Hall of Fame
            self.hall_of_fame.update(&self.population.genomes, gen);

            // Collect stats
            let stats = self.collect_stats(gen, gen_start.elapsed().as_millis() as u64);
            info!(
                "Gen {}: pareto={}, best_sharpe={:.3}, best_cagr={:.1}%, hof={}",
                gen, stats.pareto_size, stats.best_sharpe, stats.best_cagr * 100.0, 
                self.hall_of_fame.len()
            );
            self.generation_stats.push(stats);

            // Check stopping criteria
            if self.should_stop(gen) {
                info!("Stopping criteria met at generation {}", gen);
                break;
            }

            // Create next generation (except for last iteration)
            if gen < self.config.max_generations - 1 {
                self.create_next_generation(gen + 1);
            }
        }

        info!(
            "Evolution complete. Hall of Fame size: {}, Total generations: {}",
            self.hall_of_fame.len(),
            self.generation_stats.len()
        );

        Ok(())
    }

    /// Evaluate all genomes in the population.
    fn evaluate_population(&mut self) -> Result<(), EngineError> {
        let mut evaluated = 0;

        // Collect indices to evaluate
        let to_evaluate: Vec<usize> = self.population.genomes
            .iter()
            .enumerate()
            .filter(|(_, g)| g.fitness.is_none())
            .map(|(i, _)| i)
            .collect();

        for idx in to_evaluate {
            let genome = &self.population.genomes[idx];

            // Validate genome
            if let Err(e) = self.validator.validate(genome) {
                self.population.genomes[idx].fitness = 
                    Some(MultiObjectiveFitness::invalid(e.to_string()));
                continue;
            }

            // Convert to strategy config and execute
            let config_result = genome.to_strategy_config();
            match config_result {
                Ok(config) => {
                    match self.executor.execute(&config) {
                        Ok(output) => {
                            let fitness = Self::output_to_fitness_static(&output, &self.fitness_config);
                            self.population.genomes[idx].fitness = Some(fitness);
                            evaluated += 1;
                        }
                        Err(e) => {
                            self.population.genomes[idx].fitness = 
                                Some(MultiObjectiveFitness::invalid(e.to_string()));
                        }
                    }
                }
                Err(e) => {
                    self.population.genomes[idx].fitness = 
                        Some(MultiObjectiveFitness::invalid(e.to_string()));
                }
            }
        }

        if evaluated == 0 && self.population.evaluated().is_empty() {
            return Err(EngineError::NoValidGenomes);
        }

        Ok(())
    }

    /// Convert backtest output to fitness (static version to avoid borrow issues).
    fn output_to_fitness_static(output: &BacktestOutput, fitness_config: &FitnessConfig) -> MultiObjectiveFitness {
        MultiObjectiveFitness::from_metrics(
            output.metrics.cagr,
            output.metrics.sharpe_ratio,
            output.metrics.max_drawdown,
            output.metrics.calmar_ratio.unwrap_or(0.0),
            output.metrics.sortino_ratio.unwrap_or(0.0),
            output.metrics.profit_factor.unwrap_or(1.0),
            output.metrics.total_trades,
            output.metrics.volatility.unwrap_or(0.0),
            output.metrics.turnover_annual.unwrap_or(0.0),
            fitness_config,
        )
    }

    /// Create the next generation.
    fn create_next_generation(&mut self, next_gen: u32) {
        let selection = Selection::new(self.config.tournament_size);
        let crossover = Crossover::new(self.config.crossover_rate);
        let mutation = Mutation::new(self.config.mutation_rate, self.param_ranges.clone());

        let mut new_genomes = Vec::with_capacity(self.config.population_size);

        // Elitism: copy top individuals
        let elite_count = (self.config.population_size as f64 * self.config.elitism_rate) as usize;
        let mut elite: Vec<_> = self.population.genomes
            .iter()
            .filter(|g| g.fitness.as_ref().map_or(false, |f| f.is_valid))
            .collect();
        elite.sort_by(|a, b| {
            let fa = a.fitness.as_ref().unwrap();
            let fb = b.fitness.as_ref().unwrap();
            // Sort by Pareto rank, then crowding distance
            match fa.pareto_rank.cmp(&fb.pareto_rank) {
                std::cmp::Ordering::Equal => fb.crowding_distance
                    .partial_cmp(&fa.crowding_distance)
                    .unwrap_or(std::cmp::Ordering::Equal),
                ord => ord,
            }
        });

        for genome in elite.into_iter().take(elite_count) {
            let mut clone = genome.clone_with_new_id();
            clone.fitness = None; // Will be re-evaluated
            new_genomes.push(clone.with_generation(next_gen));
        }

        // Generate rest through selection, crossover, mutation
        while new_genomes.len() < self.config.population_size {
            let parents = selection.select(&self.population.genomes, 2, &mut self.rng);

            if parents.len() < 2 {
                // Not enough parents, generate random
                new_genomes.push(Population::random_genome(
                    &mut self.rng,
                    &self.param_ranges,
                    next_gen,
                ));
                continue;
            }

            let (mut child1, mut child2) =
                crossover.crossover(parents[0], parents[1], &mut self.rng, next_gen);

            mutation.mutate(&mut child1, &mut self.rng);
            mutation.mutate(&mut child2, &mut self.rng);

            new_genomes.push(child1);
            if new_genomes.len() < self.config.population_size {
                new_genomes.push(child2);
            }
        }

        self.population.next_generation(new_genomes);
    }

    /// Collect statistics for the current generation.
    fn collect_stats(&self, generation: u32, duration_ms: u64) -> GenerationStats {
        let evaluated: Vec<_> = self.population.evaluated();
        let pareto_optimal = ParetoFrontier::pareto_optimal(&self.population.genomes);

        let (best_sharpe, best_cagr) = evaluated
            .iter()
            .map(|g| {
                let f = g.fitness.as_ref().unwrap();
                (f.sharpe_ratio, f.cagr)
            })
            .fold((f64::NEG_INFINITY, f64::NEG_INFINITY), |(bs, bc), (s, c)| {
                (bs.max(s), bc.max(c))
            });

        let mean_sharpe = if evaluated.is_empty() {
            0.0
        } else {
            evaluated
                .iter()
                .map(|g| g.fitness.as_ref().unwrap().sharpe_ratio)
                .sum::<f64>()
                / evaluated.len() as f64
        };

        GenerationStats {
            generation,
            population_size: self.population.len(),
            pareto_size: pareto_optimal.len(),
            best_sharpe,
            best_cagr,
            mean_sharpe,
            evaluated: evaluated.len(),
            cache_hits: self.cache_hits,
            duration_ms,
        }
    }

    /// Check if evolution should stop.
    fn should_stop(&self, generation: u32) -> bool {
        // Max generations
        if generation >= self.config.max_generations - 1 {
            return true;
        }

        // Max runtime
        if self.config.max_runtime_seconds > 0 {
            if let Some(start) = self.start_time {
                if start.elapsed().as_secs() >= self.config.max_runtime_seconds {
                    return true;
                }
            }
        }

        // Convergence check (Pareto frontier hasn't changed)
        // TODO: Implement proper convergence detection
        false
    }

    /// Get the Hall of Fame.
    pub fn hall_of_fame(&self) -> &HallOfFame {
        &self.hall_of_fame
    }

    /// Get generation statistics.
    pub fn stats(&self) -> &[GenerationStats] {
        &self.generation_stats
    }

    /// Get the final population.
    pub fn population(&self) -> &Population {
        &self.population
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use combiner_runner::ExecutionError;
    use backtester_strategy::config::StrategyConfig;

    /// Mock executor for testing.
    struct MockExecutor;

    impl BacktestExecutor for MockExecutor {
        fn execute(&self, _config: &StrategyConfig) -> Result<BacktestOutput, ExecutionError> {
            Ok(BacktestOutput::mock())
        }

        fn execute_batch(
            &self,
            configs: &[StrategyConfig],
        ) -> Vec<Result<BacktestOutput, ExecutionError>> {
            configs.iter().map(|c| self.execute(c)).collect()
        }
    }

    #[test]
    fn test_engine_creation() {
        let config = EvolutionConfig {
            population_size: 10,
            max_generations: 2,
            ..Default::default()
        };
        let executor = MockExecutor;
        let engine = EvolutionEngine::new(config, executor);

        assert_eq!(engine.config.population_size, 10);
    }
}

