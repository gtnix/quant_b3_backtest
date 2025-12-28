//! Pareto frontier calculation using NSGA-II.

use combiner_core::{MultiObjectiveFitness, StrategyGenome};

/// Pareto frontier calculator.
pub struct ParetoFrontier;

impl ParetoFrontier {
    /// Compute Pareto ranks and crowding distances for a population.
    ///
    /// Implements NSGA-II non-dominated sorting.
    pub fn compute(genomes: &mut [StrategyGenome]) {
        if genomes.is_empty() {
            return;
        }

        // Step 1: Non-dominated sorting
        let fronts = Self::non_dominated_sort(genomes);

        // Step 2: Assign Pareto ranks
        for (rank, front) in fronts.iter().enumerate() {
            for &idx in front {
                if let Some(ref mut fitness) = genomes[idx].fitness {
                    fitness.pareto_rank = rank as u32;
                }
            }
        }

        // Step 3: Compute crowding distance within each front
        for front in &fronts {
            Self::compute_crowding_distance(genomes, front);
        }
    }

    /// Non-dominated sorting (NSGA-II).
    fn non_dominated_sort(genomes: &[StrategyGenome]) -> Vec<Vec<usize>> {
        let n = genomes.len();
        let mut dominated_by: Vec<Vec<usize>> = vec![vec![]; n];
        let mut domination_count: Vec<usize> = vec![0; n];
        let mut fronts: Vec<Vec<usize>> = vec![vec![]];

        // Calculate domination relationships
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }

                let fi = match &genomes[i].fitness {
                    Some(f) if f.is_valid => f,
                    _ => continue,
                };
                let fj = match &genomes[j].fitness {
                    Some(f) if f.is_valid => f,
                    _ => continue,
                };

                if fi.dominates(fj) {
                    dominated_by[i].push(j);
                } else if fj.dominates(fi) {
                    domination_count[i] += 1;
                }
            }

            // If i is not dominated by anyone, it's in the first front
            if domination_count[i] == 0 && genomes[i].fitness.as_ref().map_or(false, |f| f.is_valid) {
                fronts[0].push(i);
            }
        }

        // Generate subsequent fronts
        let mut current_front = 0;
        while !fronts[current_front].is_empty() {
            let mut next_front = vec![];

            for &i in &fronts[current_front] {
                for &j in &dominated_by[i] {
                    domination_count[j] = domination_count[j].saturating_sub(1);
                    if domination_count[j] == 0 {
                        next_front.push(j);
                    }
                }
            }

            if next_front.is_empty() {
                break;
            }

            fronts.push(next_front);
            current_front += 1;
        }

        fronts
    }

    /// Compute crowding distance for genomes in a front.
    fn compute_crowding_distance(genomes: &mut [StrategyGenome], front: &[usize]) {
        if front.len() <= 2 {
            // Boundary points get infinite distance
            for &idx in front {
                if let Some(ref mut fitness) = genomes[idx].fitness {
                    fitness.crowding_distance = f64::INFINITY;
                }
            }
            return;
        }

        // Initialize distances to 0
        for &idx in front {
            if let Some(ref mut fitness) = genomes[idx].fitness {
                fitness.crowding_distance = 0.0;
            }
        }

        // Objectives to consider
        let objectives = ["cagr", "sharpe_ratio", "max_drawdown"];

        for objective in &objectives {
            // Sort front by this objective
            let mut sorted: Vec<usize> = front.to_vec();
            sorted.sort_by(|&a, &b| {
                let fa = genomes[a]
                    .fitness
                    .as_ref()
                    .and_then(|f| f.get_objective(objective))
                    .unwrap_or(f64::NEG_INFINITY);
                let fb = genomes[b]
                    .fitness
                    .as_ref()
                    .and_then(|f| f.get_objective(objective))
                    .unwrap_or(f64::NEG_INFINITY);
                fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Get range
            let min_val = genomes[sorted[0]]
                .fitness
                .as_ref()
                .and_then(|f| f.get_objective(objective))
                .unwrap_or(0.0);
            let max_val = genomes[sorted[sorted.len() - 1]]
                .fitness
                .as_ref()
                .and_then(|f| f.get_objective(objective))
                .unwrap_or(0.0);
            let range = (max_val - min_val).abs();

            if range < 1e-10 {
                continue; // Skip if all values are the same
            }

            // Boundary points get infinite distance
            if let Some(ref mut fitness) = genomes[sorted[0]].fitness {
                fitness.crowding_distance = f64::INFINITY;
            }
            if let Some(ref mut fitness) = genomes[sorted[sorted.len() - 1]].fitness {
                fitness.crowding_distance = f64::INFINITY;
            }

            // Interior points
            for i in 1..sorted.len() - 1 {
                let prev_val = genomes[sorted[i - 1]]
                    .fitness
                    .as_ref()
                    .and_then(|f| f.get_objective(objective))
                    .unwrap_or(0.0);
                let next_val = genomes[sorted[i + 1]]
                    .fitness
                    .as_ref()
                    .and_then(|f| f.get_objective(objective))
                    .unwrap_or(0.0);

                if let Some(ref mut fitness) = genomes[sorted[i]].fitness {
                    if fitness.crowding_distance.is_finite() {
                        fitness.crowding_distance += (next_val - prev_val) / range;
                    }
                }
            }
        }
    }

    /// Get indices of the Pareto-optimal genomes (rank 0).
    pub fn pareto_optimal(genomes: &[StrategyGenome]) -> Vec<usize> {
        genomes
            .iter()
            .enumerate()
            .filter(|(_, g)| {
                g.fitness
                    .as_ref()
                    .map_or(false, |f| f.is_valid && f.pareto_rank == 0)
            })
            .map(|(i, _)| i)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use combiner_core::{BlockGene, BlockType, FitnessConfig, ParamValue};

    fn create_genome_with_fitness(cagr: f64, sharpe: f64, dd: f64) -> StrategyGenome {
        let config = FitnessConfig::default();
        let mut genome = StrategyGenome::new(vec![BlockGene::with_defaults(
            BlockType::Sizing,
            "equal_weight",
        )]);
        genome.fitness = Some(MultiObjectiveFitness::from_metrics(
            cagr, sharpe, dd, cagr / dd.abs(), sharpe, 1.5, 100, 0.12, 2.5, &config,
        ));
        genome
    }

    #[test]
    fn test_pareto_sorting() {
        let mut genomes = vec![
            create_genome_with_fitness(0.20, 1.5, -0.10), // Best: high return, high sharpe, low dd
            create_genome_with_fitness(0.15, 1.0, -0.15), // Dominated by first
            create_genome_with_fitness(0.25, 0.8, -0.20), // Trade-off: higher return, lower sharpe
        ];

        ParetoFrontier::compute(&mut genomes);

        // First and third should be on Pareto frontier (rank 0)
        // Second should be dominated (rank 1)
        assert_eq!(genomes[0].fitness.as_ref().unwrap().pareto_rank, 0);
        assert_eq!(genomes[2].fitness.as_ref().unwrap().pareto_rank, 0);
        assert!(genomes[1].fitness.as_ref().unwrap().pareto_rank > 0);
    }

    #[test]
    fn test_pareto_optimal() {
        let mut genomes = vec![
            create_genome_with_fitness(0.20, 1.5, -0.10),
            create_genome_with_fitness(0.10, 0.5, -0.25),
            create_genome_with_fitness(0.25, 0.8, -0.15),
        ];

        ParetoFrontier::compute(&mut genomes);
        let optimal = ParetoFrontier::pareto_optimal(&genomes);

        // At least one genome should be optimal
        assert!(!optimal.is_empty());
    }
}

