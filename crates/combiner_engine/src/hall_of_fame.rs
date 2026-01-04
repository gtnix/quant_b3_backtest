//! Hall of Fame - Best strategies across all generations.

use combiner_core::StrategyGenome;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Hall of Fame entry with ranking info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HofEntry {
    /// The genome.
    pub genome: StrategyGenome,
    /// Generation when this genome was added.
    pub added_generation: u32,
    /// Rank in the Hall of Fame (0 = best).
    pub rank: usize,
}

/// Hall of Fame - maintains the best strategies.
#[derive(Debug, Clone, Default)]
pub struct HallOfFame {
    /// Entries in the Hall of Fame.
    entries: Vec<HofEntry>,
    /// Maximum size.
    max_size: usize,
    /// Hashes of genomes for deduplication.
    seen_hashes: HashSet<u64>,
}

impl HallOfFame {
    /// Create a new Hall of Fame with the given maximum size.
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: Vec::with_capacity(max_size),
            max_size,
            seen_hashes: HashSet::new(),
        }
    }

    /// Update the Hall of Fame with genomes from the current generation.
    pub fn update(&mut self, genomes: &[StrategyGenome], generation: u32) {
        // Collect candidates (valid fitness, Pareto rank 0)
        let candidates: Vec<_> = genomes
            .iter()
            .filter(|g| {
                g.fitness
                    .as_ref()
                    .map_or(false, |f| f.is_valid && f.pareto_rank == 0)
            })
            .collect();

        for genome in candidates {
            let hash = genome.hash();

            // Skip if we've seen this genome before
            if self.seen_hashes.contains(&hash) {
                continue;
            }

            // Check if this genome should be added
            if self.entries.len() < self.max_size {
                // Room available, add directly
                self.add_entry(genome.clone(), generation);
            } else {
                // Check if better than worst
                let worst_idx = self.worst_index();
                if let Some(idx) = worst_idx {
                    let worst_fitness = self.entries[idx]
                        .genome
                        .fitness
                        .as_ref()
                        .map_or(f64::NEG_INFINITY, |f| f.scalar_fitness());
                    let new_fitness = genome
                        .fitness
                        .as_ref()
                        .map_or(f64::NEG_INFINITY, |f| f.scalar_fitness());

                    if new_fitness > worst_fitness {
                        // Remove worst and add new
                        let removed_hash = self.entries[idx].genome.hash();
                        self.seen_hashes.remove(&removed_hash);
                        self.entries.remove(idx);
                        self.add_entry(genome.clone(), generation);
                    }
                }
            }
        }

        // Re-rank entries
        self.rerank();
    }

    /// Add an entry to the Hall of Fame.
    fn add_entry(&mut self, genome: StrategyGenome, generation: u32) {
        let hash = genome.hash();
        self.seen_hashes.insert(hash);
        self.entries.push(HofEntry {
            genome,
            added_generation: generation,
            rank: 0,
        });
    }

    /// Find the index of the worst entry.
    fn worst_index(&self) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let fa = a
                    .genome
                    .fitness
                    .as_ref()
                    .map_or(f64::NEG_INFINITY, |f| f.scalar_fitness());
                let fb = b
                    .genome
                    .fitness
                    .as_ref()
                    .map_or(f64::NEG_INFINITY, |f| f.scalar_fitness());
                fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }

    /// Re-rank entries by scalar fitness.
    fn rerank(&mut self) {
        self.entries.sort_by(|a, b| {
            let fa = a
                .genome
                .fitness
                .as_ref()
                .map_or(f64::NEG_INFINITY, |f| f.scalar_fitness());
            let fb = b
                .genome
                .fitness
                .as_ref()
                .map_or(f64::NEG_INFINITY, |f| f.scalar_fitness());
            fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
        });

        for (i, entry) in self.entries.iter_mut().enumerate() {
            entry.rank = i;
        }
    }

    /// Get all entries.
    pub fn entries(&self) -> &[HofEntry] {
        &self.entries
    }

    /// Get top N entries.
    pub fn top(&self, n: usize) -> Vec<&HofEntry> {
        self.entries.iter().take(n).collect()
    }

    /// Get the best entry.
    pub fn best(&self) -> Option<&HofEntry> {
        self.entries.first()
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get genomes only.
    pub fn genomes(&self) -> Vec<&StrategyGenome> {
        self.entries.iter().map(|e| &e.genome).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use combiner_core::{BlockGene, BlockType, FitnessConfig, MultiObjectiveFitness, ParamValue};

    fn create_genome_with_sharpe(sharpe: f64) -> StrategyGenome {
        let config = FitnessConfig::default();
        // Use sharpe as part of params to make each genome unique
        let mut genome = StrategyGenome::new(vec![BlockGene::new(
            BlockType::Sizing,
            "equal_weight",
            vec![("max_weight", ParamValue::float((sharpe * 0.1).clamp(0.10, 0.40), 0.10, 0.40, 0.05))],
        )]);
        let mut fitness =
            MultiObjectiveFitness::from_metrics(0.1, sharpe, -0.1, 1.0, 1.0, 1.5, 100, 0.12, 2.5, &config);
        fitness.pareto_rank = 0; // Simulate being on Pareto frontier
        genome.fitness = Some(fitness);
        genome
    }

    #[test]
    fn test_hall_of_fame_add() {
        let mut hof = HallOfFame::new(5);

        let genomes = vec![
            create_genome_with_sharpe(1.0),
            create_genome_with_sharpe(1.5),
            create_genome_with_sharpe(0.8),
        ];

        hof.update(&genomes, 0);

        assert_eq!(hof.len(), 3);
        // Best should have sharpe 1.5
        assert!((hof.best().unwrap().genome.fitness.as_ref().unwrap().sharpe_ratio - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_hall_of_fame_max_size() {
        let mut hof = HallOfFame::new(2);

        let genomes = vec![
            create_genome_with_sharpe(1.0),
            create_genome_with_sharpe(1.5),
            create_genome_with_sharpe(0.8),
            create_genome_with_sharpe(2.0),
        ];

        hof.update(&genomes, 0);

        // Should only keep top 2
        assert_eq!(hof.len(), 2);
        // Top should be 2.0 and 1.5
        let sharpes: Vec<f64> = hof
            .entries()
            .iter()
            .map(|e| e.genome.fitness.as_ref().unwrap().sharpe_ratio)
            .collect();
        assert!(sharpes.contains(&2.0));
        assert!(sharpes.contains(&1.5));
    }
}

