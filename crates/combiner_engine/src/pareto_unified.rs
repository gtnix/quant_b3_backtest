//! Unified Pareto Frontier - Adaptive SIMD/Scalar NSGA-II Implementation
//!
//! Scientific implementation combining scalar and SIMD approaches with automatic
//! dispatch based on population size. Uses Structure of Arrays (SoA) layout for
//! maximum cache efficiency with SIMD operations.
//!
//! # Algorithm: NSGA-II (Non-dominated Sorting Genetic Algorithm II)
//!
//! 1. **Non-dominated Sorting**: O(n²) pairwise dominance comparisons
//!    - Dominance: i dominates j iff ∀k: f_i(k) ≥ f_j(k) AND ∃k: f_i(k) > f_j(k)
//!    - SIMD: 4 comparisons/cycle via f64x4
//!
//! 2. **Crowding Distance**: Preserves diversity within Pareto fronts
//!    - CD(i) = Σ_m (f_m(i+1) - f_m(i-1)) / (f_m^max - f_m^min)
//!    - Boundary points: CD = ∞
//!
//! # Performance
//!
//! - n < 8: Scalar implementation (SIMD overhead not worth it)
//! - n ≥ 8: SIMD implementation with f64x4 vectorization

use wide::f64x4;
use combiner_core::{PopulationFitnessSoA, StrategyGenome};

/// Threshold for switching from scalar to SIMD implementation
const SIMD_THRESHOLD: usize = 8;

/// Unified Pareto computer with adaptive dispatch
pub struct ParetoComputer;

impl ParetoComputer {
    /// Compute Pareto ranks and crowding distances for genomes (AoS layout).
    /// Automatically dispatches to scalar or SIMD based on population size.
    pub fn compute(genomes: &mut [StrategyGenome]) {
        if genomes.is_empty() {
            return;
        }

        let n = genomes.len();
        
        if n < SIMD_THRESHOLD {
            Self::compute_scalar(genomes);
        } else {
            Self::compute_simd_aos(genomes);
        }
    }

    /// Compute Pareto ranks directly on SoA layout (maximum performance).
    /// Use this when fitness data is already in SoA format.
    pub fn compute_soa(fitness: &mut PopulationFitnessSoA) {
        let n = fitness.len();
        if n == 0 {
            return;
        }

        if n < SIMD_THRESHOLD {
            compute_pareto_scalar_soa(fitness);
        } else {
            compute_pareto_simd_soa(fitness);
        }
        
        compute_crowding_distance_soa(fitness);
    }

    /// Scalar implementation for small populations (n < 8)
    fn compute_scalar(genomes: &mut [StrategyGenome]) {
        let fronts = Self::non_dominated_sort_scalar(genomes);
        
        // Assign Pareto ranks
        for (rank, front) in fronts.iter().enumerate() {
            for &idx in front {
                if let Some(ref mut fitness) = genomes[idx].fitness {
                    fitness.pareto_rank = rank as u32;
                }
            }
        }

        // Compute crowding distance
        for front in &fronts {
            Self::crowding_distance_scalar(genomes, front);
        }
    }

    /// SIMD implementation via SoA conversion for large populations
    fn compute_simd_aos(genomes: &mut [StrategyGenome]) {
        let n = genomes.len();
        
        // Convert to SoA for SIMD
        let mut soa = PopulationFitnessSoA::with_capacity(n);
        for (i, g) in genomes.iter().enumerate() {
            if let Some(ref f) = g.fitness {
                soa.set_fitness(
                    i, f.sharpe_ratio, f.cagr, f.max_drawdown,
                    f.calmar_ratio, f.sortino_ratio, f.profit_factor,
                    f.total_trades, f.volatility, f.turnover_annual,
                    f.is_valid,
                );
            }
        }

        // Compute in SoA
        compute_pareto_simd_soa(&mut soa);
        compute_crowding_distance_soa(&mut soa);

        // Write back to AoS
        for (i, g) in genomes.iter_mut().enumerate() {
            if let Some(ref mut f) = g.fitness {
                f.pareto_rank = soa.pareto_ranks[i];
                f.crowding_distance = soa.crowding_distances[i];
            }
        }
    }

    /// Non-dominated sorting (NSGA-II) - scalar version
    fn non_dominated_sort_scalar(genomes: &[StrategyGenome]) -> Vec<Vec<usize>> {
        let n = genomes.len();
        let mut dominated_by: Vec<Vec<usize>> = vec![vec![]; n];
        let mut domination_count: Vec<usize> = vec![0; n];
        let mut fronts: Vec<Vec<usize>> = vec![vec![]];

        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }

                let (fi, fj) = match (&genomes[i].fitness, &genomes[j].fitness) {
                    (Some(fi), Some(fj)) if fi.is_valid && fj.is_valid => (fi, fj),
                    _ => continue,
                };

                if fi.dominates(fj) {
                    dominated_by[i].push(j);
                } else if fj.dominates(fi) {
                    domination_count[i] += 1;
                }
            }

            if domination_count[i] == 0 && genomes[i].fitness.as_ref().is_some_and(|f| f.is_valid) {
                fronts[0].push(i);
            }
        }

        // Generate subsequent fronts
        let mut current = 0;
        while !fronts[current].is_empty() {
            let mut next = vec![];
            for &i in &fronts[current] {
                for &j in &dominated_by[i] {
                    domination_count[j] = domination_count[j].saturating_sub(1);
                    if domination_count[j] == 0 {
                        next.push(j);
                    }
                }
            }
            if next.is_empty() { break; }
            fronts.push(next);
            current += 1;
        }

        fronts
    }

    /// Crowding distance - scalar version
    fn crowding_distance_scalar(genomes: &mut [StrategyGenome], front: &[usize]) {
        if front.len() <= 2 {
            for &idx in front {
                if let Some(ref mut f) = genomes[idx].fitness {
                    f.crowding_distance = f64::INFINITY;
                }
            }
            return;
        }

        for &idx in front {
            if let Some(ref mut f) = genomes[idx].fitness {
                f.crowding_distance = 0.0;
            }
        }

        for objective in ["cagr", "sharpe_ratio", "max_drawdown"] {
            let mut sorted: Vec<usize> = front.to_vec();
            sorted.sort_by(|&a, &b| {
                let fa = genomes[a].fitness.as_ref()
                    .and_then(|f| f.get_objective(objective))
                    .unwrap_or(f64::NEG_INFINITY);
                let fb = genomes[b].fitness.as_ref()
                    .and_then(|f| f.get_objective(objective))
                    .unwrap_or(f64::NEG_INFINITY);
                fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
            });

            let min = genomes[sorted[0]].fitness.as_ref()
                .and_then(|f| f.get_objective(objective)).unwrap_or(0.0);
            let max = genomes[*sorted.last().unwrap()].fitness.as_ref()
                .and_then(|f| f.get_objective(objective)).unwrap_or(0.0);
            let range = (max - min).abs();

            if range < 1e-10 { continue; }

            if let Some(ref mut f) = genomes[sorted[0]].fitness {
                f.crowding_distance = f64::INFINITY;
            }
            if let Some(ref mut f) = genomes[*sorted.last().unwrap()].fitness {
                f.crowding_distance = f64::INFINITY;
            }

            for i in 1..sorted.len() - 1 {
                let prev = genomes[sorted[i - 1]].fitness.as_ref()
                    .and_then(|f| f.get_objective(objective)).unwrap_or(0.0);
                let next = genomes[sorted[i + 1]].fitness.as_ref()
                    .and_then(|f| f.get_objective(objective)).unwrap_or(0.0);

                if let Some(ref mut f) = genomes[sorted[i]].fitness {
                    if f.crowding_distance.is_finite() {
                        f.crowding_distance += (next - prev) / range;
                    }
                }
            }
        }
    }

    /// Get indices of Pareto-optimal genomes (rank 0)
    pub fn pareto_optimal(genomes: &[StrategyGenome]) -> Vec<usize> {
        genomes.iter().enumerate()
            .filter(|(_, g)| g.fitness.as_ref().is_some_and(|f| f.is_valid && f.pareto_rank == 0))
            .map(|(i, _)| i)
            .collect()
    }
}

// =============================================================================
// SoA SIMD Implementation
// =============================================================================

/// SIMD Pareto ranking on SoA layout
fn compute_pareto_simd_soa(fitness: &mut PopulationFitnessSoA) {
    let n = fitness.len();
    
    for i in 0..n {
        fitness.pareto_ranks[i] = 0;
    }

    let mut domination_count: Vec<u32> = vec![0; n];
    let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];

    for i in 0..n {
        if !fitness.is_valid[i] {
            domination_count[i] = u32::MAX;
            continue;
        }

        let sharpe_i = fitness.sharpe_ratios[i];
        let cagr_i = fitness.cagrs[i];
        let dd_i = fitness.max_drawdowns[i];

        let sharpe_i_v = f64x4::splat(sharpe_i);
        let cagr_i_v = f64x4::splat(cagr_i);
        let dd_i_v = f64x4::splat(dd_i);

        let mut j = i + 1;
        
        // SIMD batch processing (4 at a time)
        while j + 4 <= n {
            let sharpe_j = f64x4::new([
                fitness.sharpe_ratios[j],
                fitness.sharpe_ratios[j + 1],
                fitness.sharpe_ratios[j + 2],
                fitness.sharpe_ratios[j + 3],
            ]);
            let cagr_j = f64x4::new([
                fitness.cagrs[j],
                fitness.cagrs[j + 1],
                fitness.cagrs[j + 2],
                fitness.cagrs[j + 3],
            ]);
            let dd_j = f64x4::new([
                fitness.max_drawdowns[j],
                fitness.max_drawdowns[j + 1],
                fitness.max_drawdowns[j + 2],
                fitness.max_drawdowns[j + 3],
            ]);

            let valid = [
                fitness.is_valid[j],
                fitness.is_valid[j + 1],
                fitness.is_valid[j + 2],
                fitness.is_valid[j + 3],
            ];

            let ge_sharpe: [f64; 4] = (sharpe_i_v - sharpe_j).into();
            let ge_cagr: [f64; 4] = (cagr_i_v - cagr_j).into();
            let ge_dd: [f64; 4] = (dd_i_v - dd_j).into();

            for k in 0..4 {
                if !valid[k] { continue; }
                let jk = j + k;

                let i_ge_j = ge_sharpe[k] >= 0.0 && ge_cagr[k] >= 0.0 && ge_dd[k] >= 0.0;
                let i_gt_j = ge_sharpe[k] > 0.0 || ge_cagr[k] > 0.0 || ge_dd[k] > 0.0;
                let j_ge_i = ge_sharpe[k] <= 0.0 && ge_cagr[k] <= 0.0 && ge_dd[k] <= 0.0;
                let j_gt_i = ge_sharpe[k] < 0.0 || ge_cagr[k] < 0.0 || ge_dd[k] < 0.0;

                if i_ge_j && i_gt_j {
                    dominated_by[i].push(jk);
                    domination_count[jk] += 1;
                } else if j_ge_i && j_gt_i {
                    dominated_by[jk].push(i);
                    domination_count[i] += 1;
                }
            }
            j += 4;
        }

        // Remainder (scalar)
        while j < n {
            if !fitness.is_valid[j] { j += 1; continue; }

            let i_dom_j = dominates_soa(fitness, i, j);
            let j_dom_i = dominates_soa(fitness, j, i);

            if i_dom_j {
                dominated_by[i].push(j);
                domination_count[j] += 1;
            } else if j_dom_i {
                dominated_by[j].push(i);
                domination_count[i] += 1;
            }
            j += 1;
        }
    }

    // Assign ranks via front propagation
    assign_ranks_soa(&mut fitness.pareto_ranks, &domination_count, &dominated_by);
}

/// Scalar Pareto ranking on SoA layout (for small n)
fn compute_pareto_scalar_soa(fitness: &mut PopulationFitnessSoA) {
    let n = fitness.len();
    
    for i in 0..n {
        fitness.pareto_ranks[i] = 0;
    }

    let mut domination_count: Vec<u32> = vec![0; n];
    let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];

    for i in 0..n {
        if !fitness.is_valid[i] {
            domination_count[i] = u32::MAX;
            continue;
        }

        for j in (i + 1)..n {
            if !fitness.is_valid[j] { continue; }

            let i_dom_j = dominates_soa(fitness, i, j);
            let j_dom_i = dominates_soa(fitness, j, i);

            if i_dom_j {
                dominated_by[i].push(j);
                domination_count[j] += 1;
            } else if j_dom_i {
                dominated_by[j].push(i);
                domination_count[i] += 1;
            }
        }
    }

    assign_ranks_soa(&mut fitness.pareto_ranks, &domination_count, &dominated_by);
}

#[inline]
fn dominates_soa(fitness: &PopulationFitnessSoA, i: usize, j: usize) -> bool {
    let s_i = fitness.sharpe_ratios[i];
    let s_j = fitness.sharpe_ratios[j];
    let c_i = fitness.cagrs[i];
    let c_j = fitness.cagrs[j];
    let d_i = fitness.max_drawdowns[i];
    let d_j = fitness.max_drawdowns[j];

    let ge_all = s_i >= s_j && c_i >= c_j && d_i >= d_j;
    let gt_any = s_i > s_j || c_i > c_j || d_i > d_j;
    ge_all && gt_any
}

fn assign_ranks_soa(ranks: &mut [u32], domination_count: &[u32], dominated_by: &[Vec<usize>]) {
    let n = ranks.len();
    let mut remaining = domination_count.to_vec();
    let mut current: Vec<usize> = (0..n).filter(|&i| remaining[i] == 0).collect();
    let mut rank = 0u32;

    for &i in &current {
        ranks[i] = 0;
    }

    while !current.is_empty() && rank < 1000 {
        let mut next = vec![];
        for &i in &current {
            for &j in &dominated_by[i] {
                if remaining[j] > 0 && remaining[j] != u32::MAX {
                    remaining[j] -= 1;
                    if remaining[j] == 0 {
                        next.push(j);
                        ranks[j] = rank + 1;
                    }
                }
            }
        }
        current = next;
        rank += 1;
    }
}

/// Crowding distance on SoA layout
fn compute_crowding_distance_soa(fitness: &mut PopulationFitnessSoA) {
    let n = fitness.len();
    if n == 0 { return; }

    for i in 0..n {
        fitness.crowding_distances[i] = 0.0;
    }

    let max_rank = fitness.pareto_ranks.iter()
        .filter(|&&r| r < u32::MAX)
        .max().copied().unwrap_or(0);

    for rank in 0..=max_rank {
        let indices: Vec<usize> = (0..n)
            .filter(|&i| fitness.pareto_ranks[i] == rank && fitness.is_valid[i])
            .collect();

        if indices.len() < 3 {
            for &i in &indices {
                fitness.crowding_distances[i] = f64::INFINITY;
            }
            continue;
        }

        crowding_for_objective(&mut fitness.crowding_distances, fitness.sharpe_ratios.as_slice(), &indices);
        crowding_for_objective(&mut fitness.crowding_distances, fitness.cagrs.as_slice(), &indices);
        crowding_for_objective(&mut fitness.crowding_distances, fitness.max_drawdowns.as_slice(), &indices);
    }
}

fn crowding_for_objective(crowding: &mut [f64], objective: &[f64], indices: &[usize]) {
    if indices.len() < 3 { return; }

    let mut sorted = indices.to_vec();
    sorted.sort_by(|&a, &b| objective[a].partial_cmp(&objective[b]).unwrap_or(std::cmp::Ordering::Equal));

    let range = objective[sorted[sorted.len() - 1]] - objective[sorted[0]];
    if range < 1e-10 { return; }

    crowding[sorted[0]] = f64::INFINITY;
    crowding[sorted[sorted.len() - 1]] = f64::INFINITY;

    for i in 1..sorted.len() - 1 {
        let prev = sorted[i - 1];
        let curr = sorted[i];
        let next = sorted[i + 1];
        if crowding[curr].is_finite() {
            crowding[curr] += (objective[next] - objective[prev]) / range;
        }
    }
}

// =============================================================================
// Public re-exports for backward compatibility
// =============================================================================

/// Legacy alias for backward compatibility
pub type ParetoFrontier = ParetoComputer;

/// Compute Pareto ranks using SIMD (SoA layout) - legacy function
pub fn compute_pareto_ranks_simd(fitness: &mut PopulationFitnessSoA) {
    ParetoComputer::compute_soa(fitness);
}

/// Compute crowding distance (SoA layout) - legacy function
pub fn compute_crowding_distance_simd(fitness: &mut PopulationFitnessSoA) {
    compute_crowding_distance_soa(fitness);
}

#[cfg(test)]
mod tests {
    use super::*;
    use combiner_core::{BlockGene, BlockType, FitnessConfig, MultiObjectiveFitness};

    fn genome_with_fitness(cagr: f64, sharpe: f64, dd: f64) -> StrategyGenome {
        let cfg = FitnessConfig::default();
        let mut g = StrategyGenome::new(vec![
            BlockGene::with_defaults(BlockType::Sizing, "equal_weight")
        ]);
        g.fitness = Some(MultiObjectiveFitness::from_metrics(
            cagr, sharpe, dd, cagr / dd.abs(), sharpe, 1.5, 100, 0.12, 2.5, &cfg,
        ));
        g
    }

    #[test]
    fn test_pareto_sorting_scalar() {
        let mut genomes = vec![
            genome_with_fitness(0.20, 1.5, -0.10),
            genome_with_fitness(0.15, 1.0, -0.15),
            genome_with_fitness(0.25, 0.8, -0.20),
        ];

        ParetoComputer::compute(&mut genomes);

        assert_eq!(genomes[0].fitness.as_ref().unwrap().pareto_rank, 0);
        assert_eq!(genomes[2].fitness.as_ref().unwrap().pareto_rank, 0);
        assert!(genomes[1].fitness.as_ref().unwrap().pareto_rank > 0);
    }

    #[test]
    fn test_pareto_sorting_simd() {
        // Create genomes where some clearly dominate others
        let mut genomes = vec![
            genome_with_fitness(0.30, 2.0, -0.05), // Best on all metrics
            genome_with_fitness(0.20, 1.5, -0.10), // Dominated by first
            genome_with_fitness(0.10, 1.0, -0.15), // Dominated by above
            genome_with_fitness(0.05, 0.5, -0.20), // Dominated by all above
        ];

        // Add more for SIMD path
        for i in 5..20 {
            let x = i as f64 / 40.0;
            genomes.push(genome_with_fitness(x, x, -0.2 - x * 0.05));
        }

        ParetoComputer::compute(&mut genomes);

        // First genome should be rank 0 (dominates all others on 3 objectives)
        assert_eq!(genomes[0].fitness.as_ref().unwrap().pareto_rank, 0);
        // At least one genome should have rank > 0
        let has_dominated = genomes.iter()
            .filter_map(|g| g.fitness.as_ref())
            .any(|f| f.pareto_rank > 0);
        assert!(has_dominated, "Should have at least one dominated genome");
    }

    #[test]
    fn test_pareto_optimal() {
        let mut genomes = vec![
            genome_with_fitness(0.20, 1.5, -0.10),
            genome_with_fitness(0.10, 0.5, -0.25),
            genome_with_fitness(0.25, 0.8, -0.15),
        ];

        ParetoComputer::compute(&mut genomes);
        let optimal = ParetoComputer::pareto_optimal(&genomes);

        assert!(!optimal.is_empty());
    }

    #[test]
    fn test_soa_direct() {
        let mut fitness = PopulationFitnessSoA::with_capacity(10);
        for i in 0..10 {
            let x = i as f64 / 10.0;
            fitness.set_fitness(i, 1.0 - x, x, -0.1 - x * 0.1, 1.0, 1.0, 1.5, 100, 0.12, 2.5, true);
        }

        ParetoComputer::compute_soa(&mut fitness);

        let has_ranked = fitness.pareto_ranks.iter().any(|&r| r < u32::MAX);
        assert!(has_ranked);
    }
}

