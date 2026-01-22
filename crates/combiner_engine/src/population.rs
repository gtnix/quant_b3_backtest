//! Population management.

use combiner_core::{
    BlockGene, BlockType, ParamRanges, ParamValue, StrategyGenome,
};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use crate::strategy_catalog::StrategyCatalog;

/// A population of genomes.
#[derive(Debug, Clone)]
pub struct Population {
    /// All genomes in the population.
    pub genomes: Vec<StrategyGenome>,
    /// Current generation number.
    pub generation: u32,
}

impl Population {
    /// Create an empty population.
    pub fn new() -> Self {
        Self {
            genomes: Vec::new(),
            generation: 0,
        }
    }

    /// Create a population with the given genomes.
    pub fn with_genomes(genomes: Vec<StrategyGenome>) -> Self {
        Self {
            genomes,
            generation: 0,
        }
    }

    /// Generate a random initial population.
    pub fn random(size: usize, rng: &mut ChaCha8Rng, param_ranges: &ParamRanges) -> Self {
        let genomes: Vec<_> = (0..size)
            .map(|_| Self::random_genome(rng, param_ranges, 0))
            .collect();

        Self {
            genomes,
            generation: 0,
        }
    }

    /// Generate a population from Strategy Catalog templates.
    /// 
    /// This is the RECOMMENDED way to generate initial population for Template-First GA.
    /// Structure comes from templates, parameters are randomized within ranges.
    /// 
    /// # Arguments
    /// * `catalog` - Strategy catalog with templates to use
    /// * `size` - Population size
    /// * `rng` - Random number generator
    /// * `param_ranges` - Parameter specifications for randomization
    /// * `generation` - Generation number (usually 0 for initial population)
    pub fn from_catalog(
        catalog: &StrategyCatalog,
        size: usize,
        rng: &mut ChaCha8Rng,
        param_ranges: &ParamRanges,
        generation: u32,
    ) -> Self {
        let templates = catalog.templates();
        
        if templates.is_empty() {
            // Fallback to random generation if catalog is empty
            tracing::warn!("Empty catalog, falling back to random generation");
            return Self::random(size, rng, param_ranges);
        }

        let genomes: Vec<_> = (0..size)
            .map(|_| {
                // Pick a random template from the catalog
                let template = &templates[rng.gen_range(0..templates.len())];
                StrategyCatalog::to_genome(template, rng, param_ranges, generation)
            })
            .collect();

        Self {
            genomes,
            generation,
        }
    }

    /// Generate a single random genome.
    pub fn random_genome(rng: &mut ChaCha8Rng, param_ranges: &ParamRanges, generation: u32) -> StrategyGenome {
        let mut genes = Vec::new();

        // Add 1-3 random Selection blocks
        let selection_ids = param_ranges.block_ids_by_type(BlockType::Selection);
        let num_selection = rng.gen_range(1..=3).min(selection_ids.len());
        let mut used_selection: Vec<&str> = Vec::new();
        
        for _ in 0..num_selection {
            let available: Vec<_> = selection_ids
                .iter()
                .filter(|id| !used_selection.contains(id))
                .collect();
            if available.is_empty() {
                break;
            }
            let block_id = *available[rng.gen_range(0..available.len())];
            used_selection.push(block_id);
            genes.push(Self::random_gene(BlockType::Selection, block_id, rng, param_ranges));
        }

        // 50% chance to add an Entry block
        if rng.gen_bool(0.5) {
            let entry_ids = param_ranges.block_ids_by_type(BlockType::Entry);
            if !entry_ids.is_empty() {
                let block_id = entry_ids[rng.gen_range(0..entry_ids.len())];
                genes.push(Self::random_gene(BlockType::Entry, block_id, rng, param_ranges));

                // If Entry, add 1-2 Exit blocks
                let exit_ids = param_ranges.block_ids_by_type(BlockType::Exit);
                let num_exit = rng.gen_range(1..=2).min(exit_ids.len());
                let mut used_exit: Vec<&str> = Vec::new();
                
                for _ in 0..num_exit {
                    let available: Vec<_> = exit_ids
                        .iter()
                        .filter(|id| !used_exit.contains(id))
                        .collect();
                    if available.is_empty() {
                        break;
                    }
                    let exit_id = *available[rng.gen_range(0..available.len())];
                    used_exit.push(exit_id);
                    genes.push(Self::random_gene(BlockType::Exit, exit_id, rng, param_ranges));
                }
            }
        }

        // Always add exactly one Sizing block (required)
        let sizing_ids = param_ranges.block_ids_by_type(BlockType::Sizing);
        if !sizing_ids.is_empty() {
            let block_id = sizing_ids[rng.gen_range(0..sizing_ids.len())];
            genes.push(Self::random_gene(BlockType::Sizing, block_id, rng, param_ranges));
        }

        StrategyGenome::new(genes).with_generation(generation)
    }

    /// Generate a random gene with random parameters.
    fn random_gene(
        block_type: BlockType,
        block_id: &str,
        rng: &mut ChaCha8Rng,
        param_ranges: &ParamRanges,
    ) -> BlockGene {
        let params = if let Some(block_spec) = param_ranges.get_block(block_id) {
            block_spec
                .params
                .iter()
                .map(|spec| {
                    let value = Self::random_param_value(&spec.default, rng);
                    (spec.name.clone(), value)
                })
                .collect()
        } else {
            Vec::new()
        };

        BlockGene::new(block_type, block_id, params)
    }

    /// Generate a random value within the parameter's range.
    fn random_param_value(template: &ParamValue, rng: &mut ChaCha8Rng) -> ParamValue {
        match template {
            ParamValue::Float {
                min, max, step, ..
            } => {
                let steps = ((*max - *min) / *step) as u32;
                let random_steps = rng.gen_range(0..=steps);
                let value = *min + (random_steps as f64) * *step;
                ParamValue::float(value, *min, *max, *step)
            }
            ParamValue::Int {
                min, max, step, ..
            } => {
                let steps = ((*max - *min) / *step) as u32;
                let random_steps = rng.gen_range(0..=steps);
                let value = *min + (random_steps as i64) * *step;
                ParamValue::int(value, *min, *max, *step)
            }
            ParamValue::Bool { .. } => ParamValue::bool(rng.gen_bool(0.5)),
        }
    }

    /// Get the size of the population.
    pub fn len(&self) -> usize {
        self.genomes.len()
    }

    /// Check if the population is empty.
    pub fn is_empty(&self) -> bool {
        self.genomes.is_empty()
    }

    /// Get genomes with valid fitness.
    pub fn evaluated(&self) -> Vec<&StrategyGenome> {
        self.genomes
            .iter()
            .filter(|g| g.fitness.is_some() && g.fitness.as_ref().unwrap().is_valid)
            .collect()
    }

    /// Get the best genome by scalar fitness.
    pub fn best(&self) -> Option<&StrategyGenome> {
        self.genomes
            .iter()
            .filter(|g| g.fitness.is_some() && g.fitness.as_ref().unwrap().is_valid)
            .max_by(|a, b| {
                let fa = a.fitness.as_ref().unwrap().scalar_fitness();
                let fb = b.fitness.as_ref().unwrap().scalar_fitness();
                fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Advance to the next generation with new genomes.
    pub fn next_generation(&mut self, new_genomes: Vec<StrategyGenome>) {
        self.generation += 1;
        self.genomes = new_genomes;
    }

    /// Generate a population from universe-restricted strategies.
    /// This is the HYBRID generation mode: uses allowed strategies as templates
    /// and creates variations within the parameter bounds.
    pub fn from_universe(
        size: usize,
        rng: &mut ChaCha8Rng,
        param_ranges: &ParamRanges,
        allowed_strategies: &[String],
        family_filter: Option<&[String]>,
    ) -> Self {
        if allowed_strategies.is_empty() {
            // Fallback to random generation if no strategies defined
            return Self::random(size, rng, param_ranges);
        }

        // Filter strategies by family if specified
        let filtered_strategies: Vec<&String> = if let Some(families) = family_filter {
            allowed_strategies
                .iter()
                .filter(|s| families.iter().any(|f| s.starts_with(f)))
                .collect()
        } else {
            allowed_strategies.iter().collect()
        };

        if filtered_strategies.is_empty() {
            return Self::random(size, rng, param_ranges);
        }

        let genomes: Vec<_> = (0..size)
            .map(|_| {
                // Pick a random allowed strategy
                let strategy_id = filtered_strategies[rng.gen_range(0..filtered_strategies.len())];
                Self::genome_from_strategy_template(strategy_id, rng, param_ranges, 0)
            })
            .collect();

        Self {
            genomes,
            generation: 0,
        }
    }

    /// Generate a genome based on a strategy template with random parameter variations.
    pub fn genome_from_strategy_template(
        strategy_id: &str,
        rng: &mut ChaCha8Rng,
        param_ranges: &ParamRanges,
        generation: u32,
    ) -> StrategyGenome {
        // Parse family from strategy_id (e.g., "swing_momentum_ma_crossover_conservative" -> "swing")
        let family = strategy_id.split('_').next().unwrap_or("swing");
        
        // Build genes based on family type
        let mut genes = Vec::new();

        // Add selection blocks appropriate for the family
        let selection_ids = param_ranges.block_ids_by_type(BlockType::Selection);
        let family_selection: Vec<_> = selection_ids
            .iter()
            .filter(|id| id.contains(family) || Self::is_generic_block(id))
            .collect();

        if !family_selection.is_empty() {
            let num = rng.gen_range(1..=2).min(family_selection.len());
            for i in 0..num {
                let block_id = family_selection[i % family_selection.len()];
                genes.push(Self::random_gene(BlockType::Selection, block_id, rng, param_ranges));
            }
        } else if !selection_ids.is_empty() {
            // Fallback to any selection block
            let block_id = selection_ids[rng.gen_range(0..selection_ids.len())];
            genes.push(Self::random_gene(BlockType::Selection, block_id, rng, param_ranges));
        }

        // Add entry block if strategy type suggests it
        if Self::strategy_needs_entry(strategy_id) {
            let entry_ids = param_ranges.block_ids_by_type(BlockType::Entry);
            let family_entry: Vec<_> = entry_ids
                .iter()
                .filter(|id| id.contains(family) || Self::is_generic_block(id))
                .collect();

            if !family_entry.is_empty() {
                let block_id = family_entry[rng.gen_range(0..family_entry.len())];
                genes.push(Self::random_gene(BlockType::Entry, block_id, rng, param_ranges));
            } else if !entry_ids.is_empty() {
                let block_id = entry_ids[rng.gen_range(0..entry_ids.len())];
                genes.push(Self::random_gene(BlockType::Entry, block_id, rng, param_ranges));
            }

            // Add exit blocks
            let exit_ids = param_ranges.block_ids_by_type(BlockType::Exit);
            let num_exit = rng.gen_range(1..=2);
            for _ in 0..num_exit.min(exit_ids.len()) {
                let block_id = exit_ids[rng.gen_range(0..exit_ids.len())];
                genes.push(Self::random_gene(BlockType::Exit, block_id, rng, param_ranges));
            }
        }

        // Always add sizing block
        let sizing_ids = param_ranges.block_ids_by_type(BlockType::Sizing);
        if !sizing_ids.is_empty() {
            // Prefer risk-appropriate sizing based on strategy risk level
            let sizing_id = Self::select_sizing_for_strategy(strategy_id, &sizing_ids, rng);
            genes.push(Self::random_gene(BlockType::Sizing, sizing_id, rng, param_ranges));
        }

        StrategyGenome::new(genes).with_generation(generation)
    }

    /// Check if a block is a generic utility block.
    fn is_generic_block(block_id: &str) -> bool {
        matches!(
            block_id,
            "stop_loss" | "take_profit" | "trailing_stop" | "equal_weight" | 
            "volatility_target" | "risk_parity" | "time_exit" | "atr_exit"
        )
    }

    /// Determine if strategy type needs entry/exit blocks.
    fn strategy_needs_entry(strategy_id: &str) -> bool {
        // Portfolio strategies often don't need explicit entry/exit
        !strategy_id.contains("portfolio") 
            && !strategy_id.contains("buy_hold")
            && !strategy_id.contains("equal_weight")
    }

    /// Select appropriate sizing block based on strategy risk level.
    fn select_sizing_for_strategy<'a>(
        strategy_id: &str,
        sizing_ids: &[&'a str],
        rng: &mut ChaCha8Rng,
    ) -> &'a str {
        // Check risk level from strategy name
        let is_conservative = strategy_id.contains("conservative");
        let is_aggressive = strategy_id.contains("aggressive");

        // Prefer matching sizing blocks
        if is_conservative {
            // Prefer equal_weight or low risk sizing
            if let Some(id) = sizing_ids.iter().find(|id| id.contains("equal") || id.contains("fixed")) {
                return id;
            }
        } else if is_aggressive {
            // Prefer volatility-based sizing
            if let Some(id) = sizing_ids.iter().find(|id| id.contains("volatility") || id.contains("kelly")) {
                return id;
            }
        }

        // Fallback to random
        sizing_ids[rng.gen_range(0..sizing_ids.len())]
    }
}

impl Default for Population {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_population() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let param_ranges = ParamRanges::new();
        let pop = Population::random(10, &mut rng, &param_ranges);

        assert_eq!(pop.len(), 10);
        
        // All genomes should have at least one sizing block
        for genome in &pop.genomes {
            assert!(genome.has_block_type(BlockType::Sizing));
        }
    }

    #[test]
    fn test_deterministic_generation() {
        let param_ranges = ParamRanges::new();
        
        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let pop1 = Population::random(5, &mut rng1, &param_ranges);

        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let pop2 = Population::random(5, &mut rng2, &param_ranges);

        // Same seed should produce same population
        assert_eq!(pop1.genomes.len(), pop2.genomes.len());
        for (g1, g2) in pop1.genomes.iter().zip(pop2.genomes.iter()) {
            assert_eq!(g1.genes.len(), g2.genes.len());
            for (gene1, gene2) in g1.genes.iter().zip(g2.genes.iter()) {
                assert_eq!(gene1.block_id, gene2.block_id);
            }
        }
    }

    #[test]
    fn test_from_catalog() {
        let catalog = StrategyCatalog::from_builtin();
        let param_ranges = ParamRanges::new();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        
        let pop = Population::from_catalog(&catalog, 20, &mut rng, &param_ranges, 0);
        
        assert_eq!(pop.len(), 20);
        assert_eq!(pop.generation, 0);
        
        // All genomes should have template_slug set (from catalog)
        for genome in &pop.genomes {
            assert!(genome.template_slug.is_some(), 
                "Genome from catalog should have template_slug");
        }
    }

    #[test]
    fn test_from_catalog_deterministic() {
        let catalog = StrategyCatalog::from_builtin();
        let param_ranges = ParamRanges::new();
        
        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let pop1 = Population::from_catalog(&catalog, 10, &mut rng1, &param_ranges, 0);
        
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let pop2 = Population::from_catalog(&catalog, 10, &mut rng2, &param_ranges, 0);
        
        // Same seed should produce same population
        for (g1, g2) in pop1.genomes.iter().zip(pop2.genomes.iter()) {
            assert_eq!(g1.template_slug, g2.template_slug);
            assert_eq!(g1.genes.len(), g2.genes.len());
        }
    }
}

