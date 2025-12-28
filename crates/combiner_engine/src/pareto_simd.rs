//! SIMD-accelerated Pareto ranking and crowding distance.
//!
//! This module provides vectorized implementations of NSGA-II operations
//! using the `wide` crate for SIMD. Processes multiple genomes in parallel
//! for dominance comparisons and crowding distance calculations.

use wide::f64x4;
use combiner_core::PopulationFitnessSoA;

/// SIMD-accelerated Pareto dominance computation.
///
/// Computes Pareto ranks for the entire population using vectorized
/// dominance comparisons. Processes 4 comparisons at a time.
pub fn compute_pareto_ranks_simd(fitness: &mut PopulationFitnessSoA) {
    let n = fitness.len();
    if n == 0 {
        return;
    }

    // Initialize all ranks to 0 (non-dominated by default)
    for i in 0..n {
        fitness.pareto_ranks[i] = 0;
    }

    // Dominance count: how many solutions dominate solution i
    let mut domination_count: Vec<u32> = vec![0; n];
    
    // Solutions dominated by solution i
    let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];

    // Build dominance relationships
    // For small populations, use scalar. For large, use SIMD batching.
    if n < 8 {
        compute_dominance_scalar(fitness, &mut domination_count, &mut dominated_by);
    } else {
        compute_dominance_simd(fitness, &mut domination_count, &mut dominated_by);
    }

    // Assign ranks using non-dominated sorting
    assign_ranks(&mut fitness.pareto_ranks, &domination_count, &dominated_by);
}

/// Scalar dominance computation (for small populations)
fn compute_dominance_scalar(
    fitness: &PopulationFitnessSoA,
    domination_count: &mut [u32],
    dominated_by: &mut [Vec<usize>],
) {
    let n = fitness.len();
    
    for i in 0..n {
        if !fitness.is_valid[i] {
            domination_count[i] = u32::MAX; // Invalid solutions are dominated by everyone
            continue;
        }

        for j in (i + 1)..n {
            if !fitness.is_valid[j] {
                continue;
            }

            let i_dominates_j = dominates(fitness, i, j);
            let j_dominates_i = dominates(fitness, j, i);

            if i_dominates_j {
                dominated_by[i].push(j);
                domination_count[j] += 1;
            } else if j_dominates_i {
                dominated_by[j].push(i);
                domination_count[i] += 1;
            }
        }
    }
}

/// SIMD-accelerated dominance computation
/// Processes comparisons in batches of 4
fn compute_dominance_simd(
    fitness: &PopulationFitnessSoA,
    domination_count: &mut [u32],
    dominated_by: &mut [Vec<usize>],
) {
    let n = fitness.len();

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

        // Process in chunks of 4
        let mut j = i + 1;
        while j + 4 <= n {
            // Load 4 solutions at once
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

            // Check if valid
            let valid_mask = [
                fitness.is_valid[j],
                fitness.is_valid[j + 1],
                fitness.is_valid[j + 2],
                fitness.is_valid[j + 3],
            ];

            // i dominates j if i >= j in all and i > j in at least one
            let ge_sharpe: [f64; 4] = (sharpe_i_v - sharpe_j).into();
            let ge_cagr: [f64; 4] = (cagr_i_v - cagr_j).into();
            let ge_dd: [f64; 4] = (dd_i_v - dd_j).into();

            for k in 0..4 {
                if !valid_mask[k] {
                    continue;
                }

                let jk = j + k;

                // i >= j in all objectives
                let i_ge_j_all = ge_sharpe[k] >= 0.0 && ge_cagr[k] >= 0.0 && ge_dd[k] >= 0.0;
                // i > j in at least one
                let i_gt_j_any = ge_sharpe[k] > 0.0 || ge_cagr[k] > 0.0 || ge_dd[k] > 0.0;
                let i_dominates_jk = i_ge_j_all && i_gt_j_any;

                // j >= i in all objectives
                let j_ge_i_all = ge_sharpe[k] <= 0.0 && ge_cagr[k] <= 0.0 && ge_dd[k] <= 0.0;
                // j > i in at least one
                let j_gt_i_any = ge_sharpe[k] < 0.0 || ge_cagr[k] < 0.0 || ge_dd[k] < 0.0;
                let j_dominates_i = j_ge_i_all && j_gt_i_any;

                if i_dominates_jk {
                    dominated_by[i].push(jk);
                    domination_count[jk] += 1;
                } else if j_dominates_i {
                    dominated_by[jk].push(i);
                    domination_count[i] += 1;
                }
            }

            j += 4;
        }

        // Handle remainder
        while j < n {
            if !fitness.is_valid[j] {
                j += 1;
                continue;
            }

            let i_dominates_j = dominates(fitness, i, j);
            let j_dominates_i = dominates(fitness, j, i);

            if i_dominates_j {
                dominated_by[i].push(j);
                domination_count[j] += 1;
            } else if j_dominates_i {
                dominated_by[j].push(i);
                domination_count[i] += 1;
            }

            j += 1;
        }
    }
}

/// Check if solution i dominates solution j (scalar)
#[inline]
fn dominates(fitness: &PopulationFitnessSoA, i: usize, j: usize) -> bool {
    let sharpe_i = fitness.sharpe_ratios[i];
    let sharpe_j = fitness.sharpe_ratios[j];
    let cagr_i = fitness.cagrs[i];
    let cagr_j = fitness.cagrs[j];
    let dd_i = fitness.max_drawdowns[i];
    let dd_j = fitness.max_drawdowns[j];

    // i >= j in all
    let ge_all = sharpe_i >= sharpe_j && cagr_i >= cagr_j && dd_i >= dd_j;
    // i > j in at least one
    let gt_any = sharpe_i > sharpe_j || cagr_i > cagr_j || dd_i > dd_j;

    ge_all && gt_any
}

/// Assign Pareto ranks based on dominance information
fn assign_ranks(
    pareto_ranks: &mut [u32],
    domination_count: &[u32],
    dominated_by: &[Vec<usize>],
) {
    let n = pareto_ranks.len();
    let mut remaining = domination_count.to_vec();
    let mut current_front: Vec<usize> = Vec::new();
    let mut next_front: Vec<usize> = Vec::new();
    let mut rank = 0u32;

    // Find initial front (non-dominated)
    for i in 0..n {
        if remaining[i] == 0 {
            current_front.push(i);
            pareto_ranks[i] = 0;
        }
    }

    // Process remaining fronts
    while !current_front.is_empty() {
        next_front.clear();

        for &i in &current_front {
            for &j in &dominated_by[i] {
                if remaining[j] > 0 && remaining[j] != u32::MAX {
                    remaining[j] -= 1;
                    if remaining[j] == 0 {
                        next_front.push(j);
                        pareto_ranks[j] = rank + 1;
                    }
                }
            }
        }

        std::mem::swap(&mut current_front, &mut next_front);
        rank += 1;

        // Safety limit
        if rank > 1000 {
            break;
        }
    }
}

/// SIMD-accelerated crowding distance calculation
pub fn compute_crowding_distance_simd(fitness: &mut PopulationFitnessSoA) {
    let n = fitness.len();
    if n == 0 {
        return;
    }

    // Initialize crowding distances
    for i in 0..n {
        fitness.crowding_distances[i] = 0.0;
    }

    // Group by Pareto rank
    let max_rank = fitness.pareto_ranks.iter().filter(|&&r| r < u32::MAX).max().copied().unwrap_or(0);

    for rank in 0..=max_rank {
        let indices: Vec<usize> = (0..n)
            .filter(|&i| fitness.pareto_ranks[i] == rank && fitness.is_valid[i])
            .collect();

        if indices.len() < 3 {
            // Boundary points get infinite distance
            for &i in &indices {
                fitness.crowding_distances[i] = f64::INFINITY;
            }
            continue;
        }

        // Calculate crowding distance for each objective
        compute_crowding_for_objective(&mut fitness.crowding_distances, &fitness.sharpe_ratios.as_slice(), &indices);
        compute_crowding_for_objective(&mut fitness.crowding_distances, &fitness.cagrs.as_slice(), &indices);
        compute_crowding_for_objective(&mut fitness.crowding_distances, &fitness.max_drawdowns.as_slice(), &indices);
    }
}

/// Compute crowding distance contribution from a single objective
fn compute_crowding_for_objective(
    crowding: &mut [f64],
    objective: &[f64],
    indices: &[usize],
) {
    if indices.len() < 3 {
        return;
    }

    // Sort indices by objective value
    let mut sorted: Vec<usize> = indices.to_vec();
    sorted.sort_by(|&a, &b| {
        objective[a].partial_cmp(&objective[b]).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Get range
    let min_val = objective[sorted[0]];
    let max_val = objective[sorted[sorted.len() - 1]];
    let range = max_val - min_val;

    if range < 1e-10 {
        return;
    }

    // Boundary points get infinite distance
    crowding[sorted[0]] = f64::INFINITY;
    crowding[sorted[sorted.len() - 1]] = f64::INFINITY;

    // Interior points
    for i in 1..(sorted.len() - 1) {
        let prev = sorted[i - 1];
        let curr = sorted[i];
        let next = sorted[i + 1];

        if crowding[curr].is_finite() {
            crowding[curr] += (objective[next] - objective[prev]) / range;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_fitness(n: usize) -> PopulationFitnessSoA {
        let mut fitness = PopulationFitnessSoA::with_capacity(n);
        
        for i in 0..n {
            let x = i as f64 / n as f64;
            // Create Pareto-like distribution
            let sharpe = 1.0 - x + 0.1 * (i as f64).sin();
            let cagr = x + 0.1 * (i as f64).cos();
            let dd = -0.1 - x * 0.1;
            
            fitness.set_fitness(i, sharpe, cagr, dd, 1.0, 1.0, 1.5, 100, 0.12, 2.5, true);
        }
        
        fitness
    }

    #[test]
    fn test_pareto_ranks_small() {
        let mut fitness = create_test_fitness(5);
        compute_pareto_ranks_simd(&mut fitness);
        
        // All should have been ranked
        for i in 0..5 {
            assert!(fitness.pareto_ranks[i] < u32::MAX);
        }
    }

    #[test]
    fn test_pareto_ranks_large() {
        let mut fitness = create_test_fitness(100);
        compute_pareto_ranks_simd(&mut fitness);
        
        // Should have multiple fronts
        let max_rank = fitness.pareto_ranks.iter().max().copied().unwrap_or(0);
        assert!(max_rank > 0, "Should have multiple Pareto fronts");
    }

    #[test]
    fn test_crowding_distance() {
        let mut fitness = create_test_fitness(20);
        compute_pareto_ranks_simd(&mut fitness);
        compute_crowding_distance_simd(&mut fitness);
        
        // Boundary points should have infinite distance
        let has_infinite = fitness.crowding_distances.iter().any(|&d| d.is_infinite());
        assert!(has_infinite, "Boundary points should have infinite crowding distance");
    }

    #[test]
    fn test_dominance() {
        let mut fitness = PopulationFitnessSoA::with_capacity(3);
        
        // Solution 0: Best in all objectives
        fitness.set_fitness(0, 2.0, 0.2, -0.05, 2.0, 2.0, 2.0, 100, 0.1, 1.0, true);
        // Solution 1: Worst in all objectives
        fitness.set_fitness(1, 0.5, 0.05, -0.25, 0.5, 0.5, 0.5, 100, 0.2, 2.0, true);
        // Solution 2: Mixed
        fitness.set_fitness(2, 1.0, 0.1, -0.15, 1.0, 1.0, 1.0, 100, 0.15, 1.5, true);

        compute_pareto_ranks_simd(&mut fitness);

        // Solution 0 should be rank 0 (non-dominated)
        assert_eq!(fitness.pareto_ranks[0], 0);
        // Solution 1 should be dominated
        assert!(fitness.pareto_ranks[1] > 0);
    }
}

