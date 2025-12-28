//! Genetic operators: selection, crossover, mutation.

use combiner_core::{
    BlockGene, BlockType, MultiObjectiveFitness, ParamRanges, ParamValue, StrategyGenome,
};
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

/// Crossover operator.
pub struct Crossover {
    rate: f64,
}

impl Crossover {
    pub fn new(rate: f64) -> Self {
        Self {
            rate: rate.clamp(0.0, 1.0),
        }
    }

    /// Perform crossover between two parents.
    pub fn crossover(
        &self,
        parent1: &StrategyGenome,
        parent2: &StrategyGenome,
        rng: &mut ChaCha8Rng,
        generation: u32,
    ) -> (StrategyGenome, StrategyGenome) {
        if !rng.gen_bool(self.rate) {
            // No crossover, return clones
            return (
                parent1.clone_with_new_id().with_generation(generation),
                parent2.clone_with_new_id().with_generation(generation),
            );
        }

        // Uniform crossover at gene level
        let max_len = parent1.genes.len().max(parent2.genes.len());
        let mut child1_genes = Vec::new();
        let mut child2_genes = Vec::new();

        for i in 0..max_len {
            let (g1, g2) = if rng.gen_bool(0.5) {
                (
                    parent1.genes.get(i).cloned(),
                    parent2.genes.get(i).cloned(),
                )
            } else {
                (
                    parent2.genes.get(i).cloned(),
                    parent1.genes.get(i).cloned(),
                )
            };

            if let Some(gene) = g1 {
                child1_genes.push(gene);
            }
            if let Some(gene) = g2 {
                child2_genes.push(gene);
            }
        }

        // Ensure sizing block exists
        if !child1_genes.iter().any(|g| g.block_type == BlockType::Sizing) {
            if let Some(sizing) = parent1
                .genes
                .iter()
                .find(|g| g.block_type == BlockType::Sizing)
            {
                child1_genes.push(sizing.clone());
            }
        }
        if !child2_genes.iter().any(|g| g.block_type == BlockType::Sizing) {
            if let Some(sizing) = parent2
                .genes
                .iter()
                .find(|g| g.block_type == BlockType::Sizing)
            {
                child2_genes.push(sizing.clone());
            }
        }

        let child1 = StrategyGenome::new(child1_genes)
            .with_generation(generation)
            .with_parents(vec![parent1.id, parent2.id]);

        let child2 = StrategyGenome::new(child2_genes)
            .with_generation(generation)
            .with_parents(vec![parent1.id, parent2.id]);

        (child1, child2)
    }
}

/// Mutation operator.
pub struct Mutation {
    rate: f64,
    param_ranges: ParamRanges,
}

impl Mutation {
    pub fn new(rate: f64, param_ranges: ParamRanges) -> Self {
        Self {
            rate: rate.clamp(0.0, 1.0),
            param_ranges,
        }
    }

    /// Mutate a genome in place.
    pub fn mutate(&self, genome: &mut StrategyGenome, rng: &mut ChaCha8Rng) {
        for gene in &mut genome.genes {
            // Parameter mutation
            for (param_name, param_value) in &mut gene.params {
                if rng.gen_bool(self.rate) {
                    self.mutate_param(param_value, rng);
                }
            }

            // Block swap mutation (lower probability)
            if rng.gen_bool(self.rate * 0.3) {
                self.mutate_block(gene, rng);
            }
        }

        // Structural mutation: add/remove gene (very low probability)
        if rng.gen_bool(self.rate * 0.1) {
            self.mutate_structure(genome, rng);
        }
    }

    /// Mutate a parameter value.
    fn mutate_param(&self, param: &mut ParamValue, rng: &mut ChaCha8Rng) {
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

    /// Swap a block for another of the same type.
    fn mutate_block(&self, gene: &mut BlockGene, rng: &mut ChaCha8Rng) {
        let block_ids = self.param_ranges.block_ids_by_type(gene.block_type);
        if block_ids.len() <= 1 {
            return;
        }

        // Choose a different block
        let available: Vec<_> = block_ids
            .iter()
            .filter(|id| **id != gene.block_id)
            .collect();

        if !available.is_empty() {
            let new_id = *available[rng.gen_range(0..available.len())];
            gene.block_id = new_id.to_string();

            // Reset params to defaults for new block
            if let Some(block_spec) = self.param_ranges.get_block(new_id) {
                gene.params = block_spec.default_params();
            }
        }
    }

    /// Add or remove a gene.
    fn mutate_structure(&self, genome: &mut StrategyGenome, rng: &mut ChaCha8Rng) {
        if rng.gen_bool(0.5) && genome.genes.len() > 2 {
            // Remove a non-required gene
            let removable: Vec<usize> = genome
                .genes
                .iter()
                .enumerate()
                .filter(|(_, g)| {
                    g.block_type != BlockType::Sizing
                        && genome.count_block_type(g.block_type) > 1
                })
                .map(|(i, _)| i)
                .collect();

            if !removable.is_empty() {
                let idx = removable[rng.gen_range(0..removable.len())];
                genome.remove_gene(idx);
            }
        } else {
            // Add a random gene
            let block_types = [
                BlockType::Selection,
                BlockType::Entry,
                BlockType::Exit,
            ];
            let block_type = block_types[rng.gen_range(0..block_types.len())];
            let block_ids = self.param_ranges.block_ids_by_type(block_type);

            if !block_ids.is_empty() {
                let block_id = block_ids[rng.gen_range(0..block_ids.len())];
                if let Some(block_spec) = self.param_ranges.get_block(block_id) {
                    let gene = BlockGene::new(block_type, block_id, block_spec.default_params());
                    genome.add_gene(gene);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                vec![("max_weight", ParamValue::float(0.2, 0.05, 0.5, 0.05))],
            ),
        ])
    }

    #[test]
    fn test_mutation_determinism() {
        let param_ranges = ParamRanges::new();
        let mutation = Mutation::new(0.5, param_ranges);

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
    fn test_crossover() {
        let crossover = Crossover::new(1.0); // Always crossover

        let parent1 = create_test_genome();
        let parent2 = StrategyGenome::new(vec![
            BlockGene::with_defaults(BlockType::Selection, "quality"),
            BlockGene::with_defaults(BlockType::Exit, "stop_loss"),
            BlockGene::with_defaults(BlockType::Sizing, "risk_parity"),
        ]);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (child1, child2) = crossover.crossover(&parent1, &parent2, &mut rng, 1);

        // Children should have sizing blocks
        assert!(child1.has_block_type(BlockType::Sizing));
        assert!(child2.has_block_type(BlockType::Sizing));
    }
}

