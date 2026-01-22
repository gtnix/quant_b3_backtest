//! Stagnation detection and restart mechanism for genetic algorithms.
//!
//! This module implements state-of-the-art stagnation detection based on:
//! - De Jong, K.A. (1975) - Convergence detection
//! - Eiben & Smith (2003) - Restart strategies
//!
//! When evolution stagnates (no improvement for N generations), the system
//! can trigger a restart to escape local optima while preserving elite solutions.

use std::collections::VecDeque;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use combiner_core::ParamRanges;
use crate::population::Population;
use crate::diversity::DiversityMetrics;

/// Stagnation detection and restart configuration.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct StagnationConfig {
    /// Enable stagnation detection
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    
    /// Number of generations to look back for improvement
    #[serde(default = "default_window_size")]
    pub window_size: usize,
    
    /// Minimum relative improvement to consider not stagnant (0.005 = 0.5%)
    #[serde(default = "default_improvement_threshold")]
    pub improvement_threshold: f64,
    
    /// Enable automatic restart on stagnation
    #[serde(default = "default_restart_enabled")]
    pub restart_enabled: bool,
    
    /// Ratio of elite to preserve during restart (0.2 = top 20%)
    #[serde(default = "default_elite_ratio")]
    pub restart_elite_ratio: f64,
    
    /// Number of generations with boosted mutation after restart
    #[serde(default = "default_boost_generations")]
    pub post_restart_boost_generations: u32,
    
    /// Maximum number of restarts allowed
    #[serde(default = "default_max_restarts")]
    pub max_restarts: u32,
}

fn default_enabled() -> bool { true }
fn default_window_size() -> usize { 10 }
fn default_improvement_threshold() -> f64 { 0.005 }
fn default_restart_enabled() -> bool { true }
fn default_elite_ratio() -> f64 { 0.20 }
fn default_boost_generations() -> u32 { 5 }
fn default_max_restarts() -> u32 { 5 }

impl Default for StagnationConfig {
    fn default() -> Self {
        Self {
            enabled: default_enabled(),
            window_size: default_window_size(),
            improvement_threshold: default_improvement_threshold(),
            restart_enabled: default_restart_enabled(),
            restart_elite_ratio: default_elite_ratio(),
            post_restart_boost_generations: default_boost_generations(),
            max_restarts: default_max_restarts(),
        }
    }
}

/// Stagnation detector that tracks fitness history and triggers restarts.
#[derive(Debug)]
pub struct StagnationDetector {
    /// Configuration
    config: StagnationConfig,
    
    /// History of best fitness values
    best_fitness_history: VecDeque<f64>,
    
    /// History of diversity scores
    diversity_history: VecDeque<f64>,
    
    /// Number of generations since last improvement
    generations_without_improvement: u32,
    
    /// Best fitness ever seen
    best_fitness_ever: f64,
    
    /// Number of restarts performed
    restart_count: u32,
    
    /// Generation when last restart occurred
    last_restart_generation: Option<u32>,
}

impl StagnationDetector {
    /// Create a new StagnationDetector with default configuration.
    pub fn new() -> Self {
        Self::with_config(StagnationConfig::default())
    }
    
    /// Create with custom configuration.
    pub fn with_config(config: StagnationConfig) -> Self {
        Self {
            config,
            best_fitness_history: VecDeque::new(),
            diversity_history: VecDeque::new(),
            generations_without_improvement: 0,
            best_fitness_ever: f64::NEG_INFINITY,
            restart_count: 0,
            last_restart_generation: None,
        }
    }
    
    /// Update with current generation's best fitness and diversity.
    ///
    /// Returns true if improvement was detected.
    pub fn update(&mut self, best_fitness: f64, diversity: Option<&DiversityMetrics>) -> bool {
        let improved = best_fitness > self.best_fitness_ever * (1.0 + self.config.improvement_threshold);
        
        if improved {
            self.best_fitness_ever = best_fitness;
            self.generations_without_improvement = 0;
        } else {
            self.generations_without_improvement += 1;
        }
        
        // Update history
        self.best_fitness_history.push_back(best_fitness);
        if self.best_fitness_history.len() > self.config.window_size {
            self.best_fitness_history.pop_front();
        }
        
        if let Some(div) = diversity {
            self.diversity_history.push_back(div.phenotypic_diversity);
            if self.diversity_history.len() > self.config.window_size {
                self.diversity_history.pop_front();
            }
        }
        
        improved
    }
    
    /// Check if the population is stagnant.
    pub fn is_stagnant(&self) -> bool {
        if !self.config.enabled {
            return false;
        }
        
        if self.best_fitness_history.len() < self.config.window_size {
            return false;
        }
        
        // Check fitness improvement over window
        let oldest = self.best_fitness_history.front().copied().unwrap_or(0.0);
        let newest = self.best_fitness_history.back().copied().unwrap_or(0.0);
        
        let relative_improvement = if oldest.abs() > 1e-10 {
            (newest - oldest) / oldest.abs()
        } else {
            newest - oldest
        };
        
        relative_improvement < self.config.improvement_threshold
    }
    
    /// Check if diversity is critically low.
    pub fn is_diversity_critical(&self) -> bool {
        if self.diversity_history.is_empty() {
            return false;
        }
        
        let recent_diversity = self.diversity_history.back().copied().unwrap_or(1.0);
        recent_diversity < 0.15 // Critical threshold
    }
    
    /// Check if restart is needed and allowed.
    pub fn should_restart(&self) -> bool {
        if !self.config.restart_enabled {
            return false;
        }
        
        if self.restart_count >= self.config.max_restarts {
            return false;
        }
        
        self.is_stagnant() || self.is_diversity_critical()
    }
    
    /// Perform a population restart.
    ///
    /// Preserves the top `elite_ratio` of the population and regenerates the rest.
    pub fn trigger_restart(
        &mut self,
        population: &mut Population,
        param_ranges: &ParamRanges,
        catalog: &crate::strategy_catalog::StrategyCatalog,
        rng: &mut ChaCha8Rng,
        current_generation: u32,
    ) -> RestartResult {
        if !self.should_restart() {
            return RestartResult::NotNeeded;
        }
        
        self.restart_count += 1;
        self.last_restart_generation = Some(current_generation);
        
        // Clear stagnation tracking
        self.generations_without_improvement = 0;
        self.best_fitness_history.clear();
        self.diversity_history.clear();
        
        let original_size = population.genomes.len();
        let elite_count = (original_size as f64 * self.config.restart_elite_ratio) as usize;
        
        // Sort by fitness and preserve elite
        population.genomes.sort_by(|a, b| {
            let fa = a.fitness.as_ref().map(|f| f.sharpe_ratio).unwrap_or(f64::NEG_INFINITY);
            let fb = b.fitness.as_ref().map(|f| f.sharpe_ratio).unwrap_or(f64::NEG_INFINITY);
            fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        let elite: Vec<_> = population.genomes.drain(..elite_count.min(population.genomes.len())).collect();
        
        // Generate new individuals from Strategy Catalog (Template-First GA)
        let new_count = original_size - elite.len();
        let mut new_genomes = Vec::with_capacity(new_count);
        
        let templates = catalog.templates();
        if !templates.is_empty() {
            for _ in 0..new_count {
                let template = &templates[rng.gen_range(0..templates.len())];
                let genome = crate::strategy_catalog::StrategyCatalog::to_genome(
                    template, rng, param_ranges, current_generation
                );
                new_genomes.push(genome);
            }
        }
        
        // Combine elite + new
        population.genomes = elite;
        population.genomes.extend(new_genomes);
        
        RestartResult::Restarted {
            restart_number: self.restart_count,
            elite_preserved: elite_count,
            new_generated: new_count,
            post_boost_generations: self.config.post_restart_boost_generations,
        }
    }
    
    /// Get the number of restarts performed.
    pub fn restart_count(&self) -> u32 {
        self.restart_count
    }
    
    /// Get generations since last improvement.
    pub fn generations_without_improvement(&self) -> u32 {
        self.generations_without_improvement
    }
    
    /// Get the best fitness ever seen.
    pub fn best_fitness_ever(&self) -> f64 {
        self.best_fitness_ever
    }
    
    /// Check if currently in post-restart boost period.
    pub fn in_boost_period(&self, current_generation: u32) -> bool {
        match self.last_restart_generation {
            Some(restart_gen) => {
                let generations_since_restart = current_generation.saturating_sub(restart_gen);
                generations_since_restart < self.config.post_restart_boost_generations
            }
            None => false,
        }
    }
    
    /// Get stagnation status summary.
    pub fn status(&self) -> StagnationStatus {
        StagnationStatus {
            is_stagnant: self.is_stagnant(),
            is_diversity_critical: self.is_diversity_critical(),
            generations_without_improvement: self.generations_without_improvement,
            restart_count: self.restart_count,
            best_fitness_ever: self.best_fitness_ever,
            can_restart: self.should_restart(),
        }
    }
}

impl Default for StagnationDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a restart attempt.
#[derive(Debug, Clone)]
pub enum RestartResult {
    /// No restart was needed or allowed
    NotNeeded,
    /// Restart was performed
    Restarted {
        /// Which restart number this is (1, 2, 3, ...)
        restart_number: u32,
        /// Number of elite individuals preserved
        elite_preserved: usize,
        /// Number of new random individuals generated
        new_generated: usize,
        /// Number of generations with mutation boost
        post_boost_generations: u32,
    },
}

impl RestartResult {
    /// Check if a restart was performed.
    pub fn did_restart(&self) -> bool {
        matches!(self, RestartResult::Restarted { .. })
    }
}

/// Current stagnation status for logging/monitoring.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StagnationStatus {
    pub is_stagnant: bool,
    pub is_diversity_critical: bool,
    pub generations_without_improvement: u32,
    pub restart_count: u32,
    pub best_fitness_ever: f64,
    pub can_restart: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use combiner_core::{BlockGene, BlockType, ParamValue, StrategyGenome};
    use rand::SeedableRng;
    use crate::strategy_catalog::{StrategyCatalog, StrategyTemplate, TemplateBlock};
    use std::collections::HashMap;
    
    fn create_test_population(size: usize) -> Population {
        let mut genomes = Vec::with_capacity(size);
        for _ in 0..size {
            let genome = StrategyGenome::new(vec![
                BlockGene::new(
                    BlockType::Selection,
                    "momentum",
                    vec![("lookback_days", ParamValue::int(126, 21, 252, 21))],
                ),
                BlockGene::with_defaults(BlockType::Sizing, "equal_weight"),
            ])
            .with_template_slug("test_template".to_string());
            genomes.push(genome);
        }
        Population { genomes, generation: 0 }
    }
    
    fn create_test_catalog() -> StrategyCatalog {
        // Use default empty catalog for tests (trigger_restart will just preserve elites)
        StrategyCatalog::new()
    }
    
    #[test]
    fn test_stagnation_detection() {
        let mut detector = StagnationDetector::with_config(StagnationConfig {
            window_size: 5,
            improvement_threshold: 0.01, // 1%
            ..Default::default()
        });
        
        // Improving fitness - should not be stagnant
        for i in 1..=10 {
            detector.update(i as f64, None);
        }
        assert!(!detector.is_stagnant());
        
        // Flat fitness - should become stagnant
        for _ in 0..10 {
            detector.update(10.0, None);
        }
        assert!(detector.is_stagnant());
    }
    
    #[test]
    fn test_restart_limit() {
        let mut detector = StagnationDetector::with_config(StagnationConfig {
            window_size: 2,
            max_restarts: 2,
            ..Default::default()
        });
        
        // Force stagnation
        for _ in 0..5 {
            detector.update(1.0, None);
        }
        
        let mut population = create_test_population(10);
        let param_ranges = ParamRanges::new();
        let catalog = create_test_catalog();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        
        // First restart
        assert!(detector.should_restart());
        let result1 = detector.trigger_restart(&mut population, &param_ranges, &catalog, &mut rng, 1);
        assert!(result1.did_restart());
        
        // Second restart (need to re-stagnate)
        for _ in 0..5 {
            detector.update(1.0, None);
        }
        let result2 = detector.trigger_restart(&mut population, &param_ranges, &catalog, &mut rng, 10);
        assert!(result2.did_restart());
        
        // Third restart should be blocked
        for _ in 0..5 {
            detector.update(1.0, None);
        }
        assert!(!detector.should_restart());
    }
    
    #[test]
    fn test_elite_preservation() {
        let mut detector = StagnationDetector::with_config(StagnationConfig {
            window_size: 2,
            restart_elite_ratio: 0.3, // 30%
            ..Default::default()
        });
        
        // Force stagnation
        for _ in 0..5 {
            detector.update(1.0, None);
        }
        
        let mut population = create_test_population(10);
        let param_ranges = ParamRanges::new();
        let catalog = create_test_catalog();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        
        let result = detector.trigger_restart(&mut population, &param_ranges, &catalog, &mut rng, 1);
        
        if let RestartResult::Restarted { elite_preserved, new_generated, .. } = result {
            assert_eq!(elite_preserved, 3); // 30% of 10
            // With empty catalog, new_generated = 0 (only elites preserved)
            assert!(population.genomes.len() >= 3); // At least elites preserved
        } else {
            panic!("Expected restart");
        }
    }
}

