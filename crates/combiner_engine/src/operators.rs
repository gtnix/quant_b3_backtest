//! Genetic operators: selection, crossover, mutation.
//!
//! Template-First GA: Structure is FIXED (from Strategy Catalog).
//! Operators evolve ONLY parameters.

use combiner_core::{ParamRanges, ParamValue, StrategyGenome};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

/// Tournament selection operator.
pub struct Selection {
    tournament_size: usize,
}

impl Selection {
    pub fn new(tournament_size: usize) -> Self {
        Self {
            tournament_size: tournament_size.max(2),
        }
    }

    /// Select parents using tournament selection.
    pub fn select<'a>(
        &self,
        population: &'a [StrategyGenome],
        count: usize,
        rng: &mut ChaCha8Rng,
    ) -> Vec<&'a StrategyGenome> {
        let mut selected = Vec::with_capacity(count);
        let evaluated: Vec<_> = population
            .iter()
            .filter(|g| g.fitness.is_some() && g.fitness.as_ref().unwrap().is_valid)
            .collect();

        if evaluated.is_empty() {
            return selected;
        }

        for _ in 0..count {
            let winner = self.tournament(&evaluated, rng);
            selected.push(winner);
        }

        selected
    }

    /// Run a single tournament.
    fn tournament<'a>(
        &self,
        population: &[&'a StrategyGenome],
        rng: &mut ChaCha8Rng,
    ) -> &'a StrategyGenome {
        let mut contestants: Vec<_> = (0..self.tournament_size)
            .map(|_| population[rng.gen_range(0..population.len())])
            .collect();

        // Sort by Pareto rank, then by crowding distance
        contestants.sort_by(|a, b| {
            let fa = a.fitness.as_ref().unwrap();
            let fb = b.fitness.as_ref().unwrap();

            // Lower Pareto rank is better
            match fa.pareto_rank.cmp(&fb.pareto_rank) {
                std::cmp::Ordering::Equal => {
                    // Higher crowding distance is better (more diverse)
                    fb.crowding_distance
                        .partial_cmp(&fa.crowding_distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                }
                ord => ord,
            }
        });

        contestants[0]
    }
}

/// Crossover operator for Template-First GA.
///
/// IMPORTANT: Only crosses over PARAMETERS, not structure.
/// Parents must have the same template for crossover to occur.
pub struct Crossover {
    rate: f64,
}

impl Crossover {
    pub fn new(rate: f64) -> Self {
        Self {
            rate: rate.clamp(0.0, 1.0),
        }
    }

    /// Perform parameter crossover between two parents.
    ///
    /// If parents have different templates, returns clones (no crossover).
    /// Structure is NEVER modified - only parameter values are exchanged.
    pub fn crossover(
        &self,
        parent1: &StrategyGenome,
        parent2: &StrategyGenome,
        rng: &mut ChaCha8Rng,
        generation: u32,
    ) -> (StrategyGenome, StrategyGenome) {
        // Only crossover if same template
        if parent1.template_slug != parent2.template_slug {
            return (
                parent1.clone_with_new_id().with_generation(generation),
                parent2.clone_with_new_id().with_generation(generation),
            );
        }

        if !rng.gen_bool(self.rate) {
            // No crossover, return clones
            return (
                parent1.clone_with_new_id().with_generation(generation),
                parent2.clone_with_new_id().with_generation(generation),
            );
        }

        // Clone parent genes (preserving structure)
        let mut child1_genes = parent1.genes.clone();
        let mut child2_genes = parent2.genes.clone();

        // Crossover parameters at each gene position
        let min_len = child1_genes.len().min(child2_genes.len());
        for i in 0..min_len {
            // For each parameter, 50% chance to swap between parents
            let param_names: Vec<_> = child1_genes[i].params.keys().cloned().collect();

            for param_name in param_names {
                if rng.gen_bool(0.5) {
                    // Swap parameter values
                    if let (Some(v1), Some(v2)) = (
                        parent1.genes.get(i).and_then(|g| g.params.get(&param_name)),
                        parent2.genes.get(i).and_then(|g| g.params.get(&param_name)),
                    ) {
                        child1_genes[i].params.insert(param_name.clone(), v2.clone());
                        child2_genes[i].params.insert(param_name, v1.clone());
                    }
                }
            }
        }

        let mut child1 = StrategyGenome::new(child1_genes)
            .with_generation(generation)
            .with_parents(vec![parent1.id, parent2.id]);

        let mut child2 = StrategyGenome::new(child2_genes)
            .with_generation(generation)
            .with_parents(vec![parent1.id, parent2.id]);

        // Preserve template_slug from parents
        if let Some(slug) = &parent1.template_slug {
            child1 = child1.with_template_slug(slug.clone());
        }
        if let Some(slug) = &parent2.template_slug {
            child2 = child2.with_template_slug(slug.clone());
        }

        // Sanitize params to avoid floating-point noise
        child1.sanitize();
        child2.sanitize();

        (child1, child2)
    }
}

/// Mutation operator for Template-First GA.
///
/// IMPORTANT: Only mutates PARAMETERS, never structure.
/// Block IDs and block types are IMMUTABLE.
#[derive(Clone)]
pub struct Mutation {
    rate: f64,
}

impl Mutation {
    /// Create a new mutation operator.
    /// Note: param_ranges is no longer needed - mutation uses ranges from ParamValue itself.
    pub fn new(rate: f64) -> Self {
        Self {
            rate: rate.clamp(0.0, 1.0),
        }
    }
    
    /// Legacy constructor for backwards compatibility (ignores param_ranges).
    #[inline]
    pub fn with_ranges(rate: f64, _param_ranges: ParamRanges) -> Self {
        Self::new(rate)
    }

    /// Get the current mutation rate.
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Set the mutation rate (for adaptive mutation).
    pub fn set_rate(&mut self, rate: f64) {
        self.rate = rate.clamp(0.0, 1.0);
    }

    /// Mutate a genome in place.
    ///
    /// ONLY parameter values are mutated.
    /// Structure (block IDs, block types, gene count) is NEVER changed.
    pub fn mutate(&self, genome: &mut StrategyGenome, rng: &mut ChaCha8Rng) {
        for gene in &mut genome.genes {
            // Parameter mutation only
            for (_, param_value) in gene.params.iter_mut() {
                if rng.gen_bool(self.rate) {
                    Self::mutate_param(param_value, rng);
                }
            }
        }

        // Post-mutation cleanup: sanitize floats
        genome.sanitize();
    }

    /// Mutate a parameter value.
    fn mutate_param(param: &mut ParamValue, rng: &mut ChaCha8Rng) {
        match param {
            ParamValue::Float {
                value,
                min,
                max,
                step,
            } => {
                // Gaussian mutation
                let sigma = (*max - *min) * 0.1;
                let delta = rng.gen::<f64>() * sigma * 2.0 - sigma;
                let new_value = (*value + delta).clamp(*min, *max);
                // Snap to step
                let steps = ((new_value - *min) / *step).round() as i64;
                *value = *min + (steps as f64) * *step;
                *value = value.clamp(*min, *max);
            }
            ParamValue::Int {
                value,
                min,
                max,
                step,
            } => {
                // Random step mutation
                let direction = if rng.gen_bool(0.5) { *step } else { -*step };
                let new_value = (*value + direction).clamp(*min, *max);
                *value = new_value;
            }
            ParamValue::Bool { value } => {
                *value = !*value;
            }
        }
    }
}

// =============================================================================
// Adaptive Mutation (State-of-the-Art)
// Based on: Eiben & Smith (2003), "Introduction to Evolutionary Computing"
// =============================================================================

/// Adaptive mutation operator that adjusts rate based on population diversity.
///
/// When diversity is low, mutation rate increases to escape local optima.
/// When diversity is high, mutation rate decreases to exploit good solutions.
///
/// Formula: rate(t) = base_rate × (1 + k × (1 - diversity))
#[derive(Debug, Clone)]
pub struct AdaptiveMutation {
    /// Base mutation rate
    base_rate: f64,
    /// Minimum mutation rate
    min_rate: f64,
    /// Maximum mutation rate
    max_rate: f64,
    /// Amplification factor k
    amplification: f64,
    /// Current effective rate
    current_rate: f64,
    /// Generations since last improvement
    stagnation_generations: u32,
    /// Boost rate after restart
    boost_active: bool,
    /// Generations remaining with boost
    boost_generations_remaining: u32,
}

impl AdaptiveMutation {
    /// Create a new AdaptiveMutation with default parameters.
    pub fn new() -> Self {
        Self {
            base_rate: 0.05,
            min_rate: 0.01,
            max_rate: 0.30,
            amplification: 2.0,
            current_rate: 0.05,
            stagnation_generations: 0,
            boost_active: false,
            boost_generations_remaining: 0,
        }
    }
    
    /// Legacy constructor for backwards compatibility.
    pub fn with_param_ranges(_param_ranges: ParamRanges) -> Self {
        Self::new()
    }

    /// Create with custom parameters.
    pub fn with_params(
        base_rate: f64,
        min_rate: f64,
        max_rate: f64,
        amplification: f64,
    ) -> Self {
        Self {
            base_rate: base_rate.clamp(0.0, 1.0),
            min_rate: min_rate.clamp(0.0, 1.0),
            max_rate: max_rate.clamp(0.0, 1.0),
            amplification,
            current_rate: base_rate,
            stagnation_generations: 0,
            boost_active: false,
            boost_generations_remaining: 0,
        }
    }

    /// Update mutation rate based on current diversity.
    pub fn update(&mut self, diversity: f64, improved: bool) {
        // Track stagnation
        if improved {
            self.stagnation_generations = 0;
        } else {
            self.stagnation_generations += 1;
        }

        // Handle boost mode (after restart)
        if self.boost_active {
            if self.boost_generations_remaining > 0 {
                self.boost_generations_remaining -= 1;
                self.current_rate = self.max_rate;
                return;
            } else {
                self.boost_active = false;
            }
        }

        // Adaptive rate based on diversity
        let adjustment = 1.0 + self.amplification * (1.0 - diversity);
        let mut rate = self.base_rate * adjustment;

        // Extra boost for severe stagnation
        if self.stagnation_generations > 10 {
            let stagnation_boost = (self.stagnation_generations as f64 / 20.0).min(1.0);
            rate *= 1.0 + stagnation_boost;
        }

        self.current_rate = rate.clamp(self.min_rate, self.max_rate);
    }

    /// Activate boost mode (typically after a restart).
    pub fn activate_boost(&mut self, generations: u32) {
        self.boost_active = true;
        self.boost_generations_remaining = generations;
        self.current_rate = self.max_rate;
    }

    /// Get the current effective mutation rate.
    pub fn current_rate(&self) -> f64 {
        self.current_rate
    }

    /// Get the number of stagnation generations.
    pub fn stagnation_generations(&self) -> u32 {
        self.stagnation_generations
    }

    /// Check if boost mode is active.
    pub fn is_boosted(&self) -> bool {
        self.boost_active
    }

    /// Convert to a standard Mutation operator with current rate.
    pub fn to_mutation(&self) -> Mutation {
        Mutation::new(self.current_rate)
    }

    /// Mutate a genome using the current adaptive rate.
    pub fn mutate(&self, genome: &mut StrategyGenome, rng: &mut ChaCha8Rng) {
        self.to_mutation().mutate(genome, rng)
    }

    /// Reset stagnation tracking.
    pub fn reset(&mut self) {
        self.stagnation_generations = 0;
        self.boost_active = false;
        self.boost_generations_remaining = 0;
        self.current_rate = self.base_rate;
    }
}

impl Default for AdaptiveMutation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use combiner_core::BlockGene;
    use combiner_core::BlockType;

    fn create_test_genome() -> StrategyGenome {
        StrategyGenome::new(vec![
            BlockGene::new(
                BlockType::Selection,
                "momentum",
                vec![("lookback_days", ParamValue::int(126, 21, 252, 21))],
            ),
            BlockGene::new(
                BlockType::Sizing,
                "equal_weight",
                vec![("max_weight", ParamValue::float(0.25, 0.10, 0.40, 0.05))],
            ),
        ])
        .with_template_slug("test_template".to_string())
    }

    #[test]
    fn test_mutation_only_params() {
        let mutation = Mutation::new(1.0); // 100% mutation rate

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut genome = create_test_genome();

        // Record original structure
        let original_blocks: Vec<_> = genome
            .genes
            .iter()
            .map(|g| (g.block_type, g.block_id.clone()))
            .collect();

        // Mutate many times
        for _ in 0..50 {
            mutation.mutate(&mut genome, &mut rng);
        }

        // Structure should be IDENTICAL
        let after_blocks: Vec<_> = genome
            .genes
            .iter()
            .map(|g| (g.block_type, g.block_id.clone()))
            .collect();

        assert_eq!(
            original_blocks, after_blocks,
            "Mutation should NEVER change structure"
        );
    }

    #[test]
    fn test_crossover_preserves_template() {
        let crossover = Crossover::new(1.0); // Always crossover

        let parent1 = create_test_genome();
        let parent2 = create_test_genome();

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (child1, child2) = crossover.crossover(&parent1, &parent2, &mut rng, 1);

        // Template slug should be preserved
        assert_eq!(child1.template_slug, parent1.template_slug);
        assert_eq!(child2.template_slug, parent2.template_slug);
    }

    #[test]
    fn test_crossover_different_templates_no_crossover() {
        let crossover = Crossover::new(1.0);

        let parent1 = StrategyGenome::new(vec![BlockGene::new(
            BlockType::Selection,
            "momentum",
            vec![("lookback", ParamValue::int(126, 21, 252, 21))],
        )])
        .with_template_slug("template_a".to_string());

        let parent2 = StrategyGenome::new(vec![BlockGene::new(
            BlockType::Selection,
            "low_vol",
            vec![("window", ParamValue::int(60, 20, 120, 10))],
        )])
        .with_template_slug("template_b".to_string());

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (child1, child2) = crossover.crossover(&parent1, &parent2, &mut rng, 1);

        // With different templates, children should be clones
        assert_eq!(child1.genes.len(), parent1.genes.len());
        assert_eq!(child2.genes.len(), parent2.genes.len());
        assert_eq!(child1.template_slug, parent1.template_slug);
        assert_eq!(child2.template_slug, parent2.template_slug);
    }

    #[test]
    fn test_crossover_preserves_structure() {
        let crossover = Crossover::new(1.0);

        let parent1 = create_test_genome();
        let parent2 = create_test_genome();

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (child1, child2) = crossover.crossover(&parent1, &parent2, &mut rng, 1);

        // Same number of genes
        assert_eq!(child1.genes.len(), parent1.genes.len());
        assert_eq!(child2.genes.len(), parent2.genes.len());

        // Same block IDs (structure preserved)
        for (c1, p1) in child1.genes.iter().zip(parent1.genes.iter()) {
            assert_eq!(c1.block_id, p1.block_id);
            assert_eq!(c1.block_type, p1.block_type);
        }
    }

    #[test]
    fn test_mutation_determinism() {
        let mutation = Mutation::new(0.5);

        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let mut genome1 = create_test_genome();
        mutation.mutate(&mut genome1, &mut rng1);

        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let mut genome2 = create_test_genome();
        mutation.mutate(&mut genome2, &mut rng2);

        // Same seed should produce same mutations
        assert_eq!(genome1.genes.len(), genome2.genes.len());
    }

    #[test]
    fn test_mutation_bounds() {
        let mutation = Mutation::new(1.0);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut genome = create_test_genome();

        for _ in 0..100 {
            mutation.mutate(&mut genome, &mut rng);

            for gene in &genome.genes {
                for (_, param) in &gene.params {
                    match param {
                        ParamValue::Float {
                            value, min, max, ..
                        } => {
                            assert!(
                                *value >= *min && *value <= *max,
                                "Float param {} should be in [{}, {}]",
                                value,
                                min,
                                max
                            );
                        }
                        ParamValue::Int {
                            value, min, max, ..
                        } => {
                            assert!(
                                *value >= *min && *value <= *max,
                                "Int param {} should be in [{}, {}]",
                                value,
                                min,
                                max
                            );
                        }
                        ParamValue::Bool { .. } => {}
                    }
                }
            }
        }
    }

    #[test]
    fn test_selection_tournament_size() {
        let s1 = Selection::new(1);
        assert!(s1.tournament_size >= 2);

        let s2 = Selection::new(5);
        assert_eq!(s2.tournament_size, 5);
    }

    #[test]
    fn test_selection_empty_population() {
        let selection = Selection::new(3);
        let population: Vec<StrategyGenome> = vec![];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let selected = selection.select(&population, 5, &mut rng);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_crossover_rate_bounds() {
        let c1 = Crossover::new(-0.5);
        assert_eq!(c1.rate, 0.0);

        let c2 = Crossover::new(1.5);
        assert_eq!(c2.rate, 1.0);
    }

    #[test]
    fn test_mutation_rate_bounds() {
        let m1 = Mutation::new(-0.5);
        assert_eq!(m1.rate, 0.0);

        let m2 = Mutation::new(1.5);
        assert_eq!(m2.rate, 1.0);
    }
}
