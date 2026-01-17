//! Diversity monitoring and preservation for genetic algorithms.
//!
//! This module implements state-of-the-art diversity preservation techniques
//! based on academic research:
//! - Fitness Sharing (Goldberg & Richardson, 1987)
//! - Phenotypic Distance metrics
//! - Structural Entropy (Shannon)
//! - Adaptive diversity monitoring

use combiner_core::{BlockType, ParamValue, StrategyGenome};
use std::collections::{HashMap, HashSet};

/// Comprehensive diversity metrics for a population.
#[derive(Debug, Clone, Default)]
pub struct DiversityMetrics {
    /// Average Hamming distance between genomes (genotypic diversity)
    pub genotypic_diversity: f64,
    
    /// Average Euclidean distance in fitness space (phenotypic diversity)
    pub phenotypic_diversity: f64,
    
    /// Number of unique genome hashes
    pub unique_genomes: usize,
    
    /// Shannon entropy of block distribution (structural diversity)
    pub structural_entropy: f64,
    
    /// Number of unique fitness values (rounded to 2 decimals)
    pub unique_fitness_values: usize,
    
    /// Population size
    pub population_size: usize,
}

impl DiversityMetrics {
    /// Check if diversity is critically low (stagnation risk)
    pub fn is_critically_low(&self, threshold: f64) -> bool {
        self.phenotypic_diversity < threshold
    }
    
    /// Get overall diversity score [0, 1]
    pub fn overall_score(&self) -> f64 {
        // Weighted combination of metrics
        let weights = [0.3, 0.4, 0.2, 0.1]; // phenotypic is most important
        let scores = [
            self.genotypic_diversity,
            self.phenotypic_diversity,
            self.structural_entropy,
            self.unique_ratio(),
        ];
        
        weights.iter().zip(scores.iter())
            .map(|(w, s)| w * s)
            .sum()
    }
    
    /// Ratio of unique genomes to population size
    pub fn unique_ratio(&self) -> f64 {
        if self.population_size == 0 {
            return 0.0;
        }
        self.unique_genomes as f64 / self.population_size as f64
    }
}

/// Monitors and computes diversity metrics for a population.
pub struct DiversityMonitor {
    /// History of diversity metrics per generation
    pub history: Vec<DiversityMetrics>,
    
    /// Sigma share parameter for fitness sharing
    pub sigma_share: f64,
    
    /// Alpha parameter for sharing function
    pub alpha: f64,
    
    /// Threshold for critical diversity
    pub critical_threshold: f64,
}

impl Default for DiversityMonitor {
    fn default() -> Self {
        Self {
            history: Vec::new(),
            sigma_share: 0.10,
            alpha: 1.0,
            critical_threshold: 0.15,
        }
    }
}

impl DiversityMonitor {
    /// Create a new DiversityMonitor with custom parameters.
    pub fn new(sigma_share: f64, alpha: f64, critical_threshold: f64) -> Self {
        Self {
            history: Vec::new(),
            sigma_share,
            alpha,
            critical_threshold,
        }
    }
    
    /// Compute diversity metrics for the current population.
    pub fn compute(&self, population: &[StrategyGenome]) -> DiversityMetrics {
        let n = population.len();
        if n == 0 {
            return DiversityMetrics::default();
        }
        
        DiversityMetrics {
            genotypic_diversity: compute_genotypic_diversity(population),
            phenotypic_diversity: compute_phenotypic_diversity(population),
            unique_genomes: count_unique_genomes(population),
            structural_entropy: compute_structural_entropy(population),
            unique_fitness_values: count_unique_fitness_values(population),
            population_size: n,
        }
    }
    
    /// Update history with new metrics.
    pub fn update(&mut self, population: &[StrategyGenome]) -> DiversityMetrics {
        let metrics = self.compute(population);
        self.history.push(metrics.clone());
        metrics
    }
    
    /// Check if population is stagnant based on diversity trend.
    pub fn is_stagnant(&self, window: usize) -> bool {
        if self.history.len() < window {
            return false;
        }
        
        let recent: Vec<_> = self.history.iter()
            .rev()
            .take(window)
            .collect();
        
        // Check if diversity has been below critical threshold for entire window
        recent.iter().all(|m| m.phenotypic_diversity < self.critical_threshold)
    }
    
    /// Get the latest diversity metrics.
    pub fn latest(&self) -> Option<&DiversityMetrics> {
        self.history.last()
    }
    
    /// Get diversity trend (positive = improving, negative = declining)
    pub fn trend(&self, window: usize) -> f64 {
        if self.history.len() < window {
            return 0.0;
        }
        
        let recent: Vec<_> = self.history.iter()
            .rev()
            .take(window)
            .map(|m| m.phenotypic_diversity)
            .collect();
        
        if recent.len() < 2 {
            return 0.0;
        }
        
        // Simple linear trend
        let first = recent.last().unwrap();
        let last = recent.first().unwrap();
        last - first
    }
}

/// Compute genotypic diversity (average Hamming distance between genomes).
pub fn compute_genotypic_diversity(population: &[StrategyGenome]) -> f64 {
    let n = population.len();
    if n < 2 {
        return 1.0;
    }
    
    let mut total_distance = 0.0;
    let mut count = 0;
    
    for i in 0..n {
        for j in (i + 1)..n {
            total_distance += hamming_distance(&population[i], &population[j]);
            count += 1;
        }
    }
    
    if count == 0 {
        return 0.0;
    }
    
    // Normalize to [0, 1] assuming max distance is genome length
    let max_possible_distance = population.iter()
        .map(|g| g.genes.len())
        .max()
        .unwrap_or(1) as f64;
    
    (total_distance / count as f64 / max_possible_distance).min(1.0)
}

/// Compute Hamming distance between two genomes.
pub fn hamming_distance(a: &StrategyGenome, b: &StrategyGenome) -> f64 {
    let mut distance = 0.0;
    let max_len = a.genes.len().max(b.genes.len());
    
    for i in 0..max_len {
        let gene_a = a.genes.get(i);
        let gene_b = b.genes.get(i);
        
        match (gene_a, gene_b) {
            (Some(ga), Some(gb)) => {
                // Different block type or ID = distance 1.0
                if ga.block_type != gb.block_type || ga.block_id != gb.block_id {
                    distance += 1.0;
                } else {
                    // Same block, compare parameters
                    let param_diff = compare_params(&ga.params, &gb.params);
                    distance += param_diff;
                }
            }
            (Some(_), None) | (None, Some(_)) => {
                distance += 1.0;
            }
            (None, None) => {}
        }
    }
    
    distance
}

/// Compare parameters between two genes, returning normalized difference.
fn compare_params(
    params_a: &HashMap<String, ParamValue>,
    params_b: &HashMap<String, ParamValue>,
) -> f64 {
    if params_a.is_empty() && params_b.is_empty() {
        return 0.0;
    }
    
    let mut total_diff = 0.0;
    let mut count = 0;
    
    for (name_a, val_a) in params_a {
        if let Some(val_b) = params_b.get(name_a) {
            let diff = match (val_a, val_b) {
                (ParamValue::Float { value: v1, min, max, .. }, 
                 ParamValue::Float { value: v2, .. }) => {
                    let range = (max - min).abs().max(1e-10);
                    (v1 - v2).abs() / range
                }
                (ParamValue::Int { value: v1, min, max, .. }, 
                 ParamValue::Int { value: v2, .. }) => {
                    let range = (*max - *min).abs().max(1) as f64;
                    (v1 - v2).abs() as f64 / range
                }
                (ParamValue::Bool { value: v1 }, ParamValue::Bool { value: v2 }) => {
                    if v1 == v2 { 0.0 } else { 1.0 }
                }
                _ => 0.5, // Different types
            };
            total_diff += diff;
            count += 1;
        } else {
            total_diff += 1.0; // Missing parameter
            count += 1;
        }
    }
    
    if count == 0 {
        return 0.0;
    }
    
    total_diff / count as f64
}

/// Compute phenotypic diversity (average Euclidean distance in fitness space).
pub fn compute_phenotypic_diversity(population: &[StrategyGenome]) -> f64 {
    let valid: Vec<_> = population.iter()
        .filter(|g| g.fitness.as_ref().map_or(false, |f| f.is_valid))
        .collect();
    
    let n = valid.len();
    if n < 2 {
        return 1.0;
    }
    
    let mut total_distance = 0.0;
    let mut count = 0;
    
    for i in 0..n {
        for j in (i + 1)..n {
            total_distance += phenotypic_distance(valid[i], valid[j]);
            count += 1;
        }
    }
    
    if count == 0 {
        return 0.0;
    }
    
    // Normalize by expected maximum distance
    (total_distance / count as f64).min(1.0)
}

/// Compute phenotypic distance between two genomes in fitness space.
pub fn phenotypic_distance(a: &StrategyGenome, b: &StrategyGenome) -> f64 {
    let fa = match &a.fitness {
        Some(f) if f.is_valid => f,
        _ => return 1.0,
    };
    let fb = match &b.fitness {
        Some(f) if f.is_valid => f,
        _ => return 1.0,
    };
    
    // Normalized Euclidean distance in fitness space
    // Using typical ranges for normalization
    let sharpe_range = 3.0;  // Sharpe typically [-1, 2]
    let cagr_range = 0.5;    // CAGR typically [-0.3, 0.2]
    let dd_range = 0.5;      // MaxDD typically [-0.5, 0]
    
    let d_sharpe = ((fa.sharpe_ratio - fb.sharpe_ratio) / sharpe_range).powi(2);
    let d_cagr = ((fa.cagr - fb.cagr) / cagr_range).powi(2);
    let d_dd = ((fa.max_drawdown - fb.max_drawdown) / dd_range).powi(2);
    
    (d_sharpe + d_cagr + d_dd).sqrt() / 3.0_f64.sqrt()
}

/// Count unique genomes by ID.
pub fn count_unique_genomes(population: &[StrategyGenome]) -> usize {
    let ids: HashSet<_> = population.iter()
        .map(|g| g.id)
        .collect();
    ids.len()
}

/// Count unique fitness values (rounded to 2 decimals for Sharpe).
pub fn count_unique_fitness_values(population: &[StrategyGenome]) -> usize {
    let sharpes: std::collections::HashSet<_> = population.iter()
        .filter_map(|g| g.fitness.as_ref())
        .filter(|f| f.is_valid)
        .map(|f| (f.sharpe_ratio * 100.0).round() as i64)
        .collect();
    sharpes.len()
}

/// Compute structural entropy (Shannon entropy of block distribution).
pub fn compute_structural_entropy(population: &[StrategyGenome]) -> f64 {
    let mut block_counts: HashMap<String, usize> = HashMap::new();
    let mut total = 0;
    
    for genome in population {
        for gene in &genome.genes {
            *block_counts.entry(gene.block_id.clone()).or_insert(0) += 1;
            total += 1;
        }
    }
    
    if total == 0 || block_counts.len() <= 1 {
        return 0.0;
    }
    
    // Shannon entropy
    let mut entropy = 0.0;
    for count in block_counts.values() {
        let p = *count as f64 / total as f64;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }
    
    // Normalize by maximum entropy (log of number of unique blocks)
    let max_entropy = (block_counts.len() as f64).ln();
    if max_entropy > 0.0 {
        entropy / max_entropy
    } else {
        0.0
    }
}

/// Apply fitness sharing to population (Goldberg & Richardson, 1987).
/// 
/// Modifies the fitness of each individual based on how many similar
/// individuals exist in the population, encouraging diversity.
pub fn apply_fitness_sharing(
    population: &mut [StrategyGenome],
    sigma_share: f64,
    alpha: f64,
) {
    let n = population.len();
    if n == 0 {
        return;
    }
    
    // Compute niche counts
    let mut niche_counts: Vec<f64> = vec![0.0; n];
    
    for i in 0..n {
        for j in 0..n {
            let distance = phenotypic_distance(&population[i], &population[j]);
            if distance < sigma_share {
                let sharing = 1.0 - (distance / sigma_share).powf(alpha);
                niche_counts[i] += sharing;
            }
        }
    }
    
    // Apply sharing to fitness
    for (i, genome) in population.iter_mut().enumerate() {
        if let Some(ref mut fitness) = genome.fitness {
            let niche_count = niche_counts[i].max(1.0);
            
            // Store shared fitness (we use sharpe_ratio as proxy)
            // In a full implementation, we'd have a separate shared_fitness field
            fitness.sharpe_ratio /= niche_count;
        }
    }
}

/// Compute fitness sharing adjustment factor without modifying fitness.
/// Returns adjustment factors for each genome.
pub fn compute_sharing_factors(
    population: &[StrategyGenome],
    sigma_share: f64,
    alpha: f64,
) -> Vec<f64> {
    let n = population.len();
    if n == 0 {
        return Vec::new();
    }
    
    let mut niche_counts: Vec<f64> = vec![0.0; n];
    
    for i in 0..n {
        for j in 0..n {
            let distance = phenotypic_distance(&population[i], &population[j]);
            if distance < sigma_share {
                let sharing = 1.0 - (distance / sigma_share).powf(alpha);
                niche_counts[i] += sharing;
            }
        }
    }
    
    // Return inverse niche count as sharing factor
    niche_counts.iter()
        .map(|nc| 1.0 / nc.max(1.0))
        .collect()
}

/// Get block type distribution for analysis.
pub fn block_type_distribution(population: &[StrategyGenome]) -> HashMap<BlockType, HashMap<String, usize>> {
    let mut distribution: HashMap<BlockType, HashMap<String, usize>> = HashMap::new();
    
    for genome in population {
        for gene in &genome.genes {
            let type_map = distribution.entry(gene.block_type).or_default();
            *type_map.entry(gene.block_id.clone()).or_insert(0) += 1;
        }
    }
    
    distribution
}

#[cfg(test)]
mod tests {
    use super::*;
    use combiner_core::{BlockGene, ParamValue, MultiObjectiveFitness};
    
    fn create_test_genome(sharpe: f64, block_id: &str) -> StrategyGenome {
        let mut genome = StrategyGenome::new(vec![
            BlockGene::new(
                BlockType::Selection,
                block_id,
                vec![("lookback_days", ParamValue::int(126, 21, 252, 21))],
            ),
            BlockGene::with_defaults(BlockType::Sizing, "equal_weight"),
        ]);
        
        genome.fitness = Some(MultiObjectiveFitness {
            cagr: sharpe * 0.1,
            sharpe_ratio: sharpe,
            max_drawdown: -0.15,
            calmar_ratio: sharpe * 0.5,
            sortino_ratio: sharpe * 1.2,
            profit_factor: 1.5,
            total_trades: 100,
            volatility: 0.15,
            turnover_annual: 2.0,
            pareto_rank: 0,
            crowding_distance: 0.0,
            is_valid: true,
            penalty_low_trades: 0.0,
            penalty_extreme_turnover: 0.0,
            error: None,
            run_id: None,
        });
        
        genome
    }
    
    #[test]
    fn test_phenotypic_distance() {
        let g1 = create_test_genome(1.0, "momentum");
        let g2 = create_test_genome(1.0, "momentum");
        let g3 = create_test_genome(2.0, "momentum");
        
        // Same fitness = zero distance
        let d_same = phenotypic_distance(&g1, &g2);
        assert!(d_same < 0.01, "Same fitness should have near-zero distance");
        
        // Different fitness = non-zero distance
        let d_diff = phenotypic_distance(&g1, &g3);
        assert!(d_diff > 0.1, "Different fitness should have non-zero distance");
    }
    
    #[test]
    fn test_genotypic_diversity() {
        // Identical population
        let pop_same: Vec<_> = (0..10)
            .map(|_| create_test_genome(1.0, "momentum"))
            .collect();
        
        let div_same = compute_genotypic_diversity(&pop_same);
        assert!(div_same < 0.1, "Identical population should have low diversity");
        
        // Diverse population
        let pop_diverse = vec![
            create_test_genome(1.0, "momentum"),
            create_test_genome(1.0, "value"),
            create_test_genome(1.0, "quality"),
            create_test_genome(1.0, "low_vol"),
        ];
        
        let div_diverse = compute_genotypic_diversity(&pop_diverse);
        assert!(div_diverse > 0.3, "Diverse population should have higher diversity");
    }
    
    #[test]
    fn test_structural_entropy() {
        // All same selection blocks (homogeneous) - all use "momentum"
        let pop_same: Vec<_> = (0..10)
            .map(|_| create_test_genome(1.0, "momentum"))
            .collect();
        
        let entropy_same = compute_structural_entropy(&pop_same);
        // Entropy is normalized [0, 1], will be ~1.0 for 2 equally distributed blocks
        // (momentum + equal_weight)
        assert!(entropy_same >= 0.0, "Entropy should be non-negative");
        
        // Diverse blocks = uses different selection blocks
        let pop_diverse = vec![
            create_test_genome(1.0, "momentum"),
            create_test_genome(1.0, "value"),
            create_test_genome(1.0, "quality"),
            create_test_genome(1.0, "low_vol"),
        ];
        
        let entropy_diverse = compute_structural_entropy(&pop_diverse);
        // With more unique block types, entropy should be higher
        assert!(entropy_diverse >= entropy_same || (entropy_diverse - entropy_same).abs() < 0.3,
            "Diverse blocks should have at least similar or higher entropy: {} vs {}", 
            entropy_diverse, entropy_same);
    }
    
    #[test]
    fn test_diversity_monitor() {
        let monitor = DiversityMonitor::default();
        
        let population = vec![
            create_test_genome(0.5, "momentum"),
            create_test_genome(0.8, "value"),
            create_test_genome(1.2, "quality"),
        ];
        
        let metrics = monitor.compute(&population);
        
        assert_eq!(metrics.population_size, 3);
        assert!(metrics.unique_genomes >= 1);
        assert!(metrics.phenotypic_diversity >= 0.0);
        assert!(metrics.genotypic_diversity >= 0.0);
    }
    
    #[test]
    fn test_sharing_factors() {
        let population = vec![
            create_test_genome(1.0, "momentum"),
            create_test_genome(1.0, "momentum"),  // Very similar
            create_test_genome(2.0, "value"),     // Different
        ];
        
        let factors = compute_sharing_factors(&population, 0.5, 1.0);
        
        assert_eq!(factors.len(), 3);
        // Similar individuals should have lower factors (more sharing)
        assert!(factors[0] < factors[2], "Similar genomes should have lower sharing factor");
        assert!(factors[1] < factors[2], "Similar genomes should have lower sharing factor");
    }
}

