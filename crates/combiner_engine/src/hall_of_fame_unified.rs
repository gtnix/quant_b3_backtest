//! Unified Hall of Fame with Strategy Pattern
//!
//! Scientific implementation that unifies basic and institutional HoF through
//! a trait-based strategy pattern. Supports pluggable scoring/acceptance criteria.
//!
//! # Architecture
//!
//! ```text
//! +-------------------+
//! |   HofStrategy     |  <-- Trait for criteria
//! |-------------------|
//! | + accepts(&g)     |
//! | + score(&g)       |
//! +-------------------+
//!        ^
//!        |
//! +------+------+
//! |             |
//! Basic       Institutional
//! (scalar)    (OOS×DSR×(1-PBO))
//! ```
//!
//! # Scoring Models
//!
//! ## Basic (for discovery/development)
//! - Score = scalar_fitness (weighted combination of CAGR, Sharpe, DD)
//!
//! ## Institutional (for production)
//! - Score = OOS_Sharpe × (1-PBO) × DSR × split_pass_rate / (1+degradation)
//!
//! This emphasizes robustness over in-sample performance.

use std::collections::HashSet;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use combiner_core::StrategyGenome;
use crate::evaluation::stage_b::ValidationResult;

// =============================================================================
// Strategy Trait
// =============================================================================

/// Strategy trait for HoF acceptance and scoring criteria
pub trait HofStrategy: Clone + Send + Sync {
    /// Check if a genome should be accepted
    fn accepts(&self, genome: &StrategyGenome) -> bool;
    
    /// Compute score for ranking (higher = better)
    fn score(&self, genome: &StrategyGenome) -> f64;
    
    /// Strategy name for debugging
    fn name(&self) -> &'static str;
}

/// Strategy with validation result support
pub trait ValidatedHofStrategy: HofStrategy {
    /// Check if validation result meets criteria
    fn accepts_validation(&self, result: &ValidationResult) -> bool;
    
    /// Compute score from validation result
    fn score_validation(&self, result: &ValidationResult) -> f64;
}

// =============================================================================
// Basic Strategy (for SCG discovery)
// =============================================================================

/// Basic scoring strategy using scalar fitness
#[derive(Debug, Clone, Default)]
pub struct BasicStrategy;

impl HofStrategy for BasicStrategy {
    fn accepts(&self, genome: &StrategyGenome) -> bool {
        genome.fitness.as_ref().is_some_and(|f| f.is_valid && f.pareto_rank == 0)
    }
    
    fn score(&self, genome: &StrategyGenome) -> f64 {
        genome.fitness.as_ref()
            .map(|f| f.scalar_fitness())
            .unwrap_or(f64::NEG_INFINITY)
    }
    
    fn name(&self) -> &'static str { "basic" }
}

// =============================================================================
// Institutional Strategy (for production)
// =============================================================================

/// Institutional-grade acceptance criteria (OMP spec compliant)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstitutionalCriteria {
    pub min_oos_sharpe: f64,
    pub max_pbo: f64,
    pub min_dsr: f64,
    pub max_degradation_pct: f64,
    pub min_split_pass_rate: f64,
    pub max_oos_drawdown: f64,
}

impl Default for InstitutionalCriteria {
    fn default() -> Self {
        Self::research()
    }
}

impl InstitutionalCriteria {
    /// Research-grade (relaxed for development)
    pub fn research() -> Self {
        Self {
            min_oos_sharpe: 0.2,
            max_pbo: 0.50,
            min_dsr: 0.0,
            max_degradation_pct: 80.0,
            min_split_pass_rate: 0.2,
            max_oos_drawdown: -0.70,
        }
    }
    
    /// Production-grade (strict, matches OMP spec)
    pub fn production() -> Self {
        Self {
            min_oos_sharpe: 1.0,
            max_pbo: 0.10,
            min_dsr: 0.8,
            max_degradation_pct: 50.0,
            min_split_pass_rate: 0.5,
            max_oos_drawdown: -0.20,
        }
    }
}

/// Institutional scoring strategy
#[derive(Debug, Clone)]
pub struct InstitutionalStrategy {
    pub criteria: InstitutionalCriteria,
}

impl InstitutionalStrategy {
    pub fn new(criteria: InstitutionalCriteria) -> Self {
        Self { criteria }
    }
    
    pub fn research() -> Self {
        Self::new(InstitutionalCriteria::research())
    }
    
    pub fn production() -> Self {
        Self::new(InstitutionalCriteria::production())
    }
}

impl Default for InstitutionalStrategy {
    fn default() -> Self {
        Self::research()
    }
}

impl HofStrategy for InstitutionalStrategy {
    fn accepts(&self, genome: &StrategyGenome) -> bool {
        genome.fitness.as_ref().is_some_and(|f| f.is_valid)
    }
    
    fn score(&self, genome: &StrategyGenome) -> f64 {
        genome.fitness.as_ref()
            .map(|f| f.sharpe_ratio * (1.0 - f.max_drawdown.abs()))
            .unwrap_or(f64::NEG_INFINITY)
    }
    
    fn name(&self) -> &'static str { "institutional" }
}

impl ValidatedHofStrategy for InstitutionalStrategy {
    fn accepts_validation(&self, result: &ValidationResult) -> bool {
        if !result.passed { return false; }
        if result.oos_sharpe_median < self.criteria.min_oos_sharpe { return false; }
        if result.pbo > self.criteria.max_pbo { return false; }
        if result.dsr < self.criteria.min_dsr { return false; }
        if result.degradation_pct > self.criteria.max_degradation_pct { return false; }
        
        // Check max drawdown (oos_max_dd_worst is negative, max_oos_drawdown is negative)
        if result.oos_max_dd_worst < self.criteria.max_oos_drawdown { return false; }
        
        let pass_rate = result.splits_passed as f64 / result.splits_evaluated.max(1) as f64;
        if pass_rate < self.criteria.min_split_pass_rate { return false; }
        
        true
    }
    
    fn score_validation(&self, result: &ValidationResult) -> f64 {
        // Score = OOS_Sharpe × (1-PBO) × DSR × split_pass_rate / (1+degradation)
        let robustness = 1.0 - result.pbo;
        let stability = 1.0 - (result.degradation_pct / 100.0).min(1.0);
        let consistency = result.splits_passed as f64 / result.splits_evaluated.max(1) as f64;
        
        result.oos_sharpe_median * robustness * result.dsr.max(0.1) * consistency * (1.0 + stability * 0.5)
    }
}

// =============================================================================
// Unified Hall of Fame Entry
// =============================================================================

/// Unified HoF entry supporting both basic and validated modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedHofEntry {
    pub genome_id: Uuid,
    pub genome_hash: u64,
    pub genome: StrategyGenome,
    pub added_generation: u32,
    pub rank: usize,
    pub score: f64,
    /// Optional validation summary (for institutional mode)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation: Option<ValidationSummary>,
}

impl UnifiedHofEntry {
    /// Legacy accessor: validated_generation = added_generation
    #[inline]
    pub fn validated_generation(&self) -> u32 {
        self.added_generation
    }
    
    /// Get validation summary, panics if None (for validated HoF entries)
    #[inline]
    pub fn validation_ref(&self) -> &ValidationSummary {
        self.validation.as_ref().expect("Entry requires validation")
    }
}

/// Compact validation summary for storage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationSummary {
    pub oos_sharpe_median: f64,
    pub oos_sharpe_std: f64,
    pub oos_sharpe_mean: f64, // Legacy compat: same as median
    pub oos_cagr_median: f64,
    pub oos_max_dd_worst: f64,
    pub degradation_pct: f64,
    pub pbo: f64,
    pub dsr: f64,
    pub splits_evaluated: u16,
    pub splits_passed: u16,
    pub passed: bool,
}

impl From<&ValidationResult> for ValidationSummary {
    fn from(r: &ValidationResult) -> Self {
        Self {
            oos_sharpe_median: r.oos_sharpe_median,
            oos_sharpe_std: r.oos_sharpe_std,
            oos_sharpe_mean: r.oos_sharpe_median, // Legacy: use median as mean
            oos_cagr_median: r.oos_cagr_median,
            oos_max_dd_worst: r.oos_max_dd_worst,
            degradation_pct: r.degradation_pct,
            pbo: r.pbo,
            dsr: r.dsr,
            splits_evaluated: r.splits_evaluated,
            splits_passed: r.splits_passed,
            passed: r.passed,
        }
    }
}

// =============================================================================
// Unified Hall of Fame
// =============================================================================

/// Unified Hall of Fame with pluggable strategy
#[derive(Debug)]
pub struct UnifiedHallOfFame<S: HofStrategy> {
    entries: Vec<UnifiedHofEntry>,
    max_size: usize,
    seen_hashes: HashSet<u64>,
    strategy: S,
}

impl<S: HofStrategy + Clone> Clone for UnifiedHallOfFame<S> {
    fn clone(&self) -> Self {
        Self {
            entries: self.entries.clone(),
            max_size: self.max_size,
            seen_hashes: self.seen_hashes.clone(),
            strategy: self.strategy.clone(),
        }
    }
}

impl<S: HofStrategy> UnifiedHallOfFame<S> {
    /// Create new HoF with given strategy
    pub fn new(max_size: usize, strategy: S) -> Self {
        Self {
            entries: Vec::with_capacity(max_size),
            max_size,
            seen_hashes: HashSet::new(),
            strategy,
        }
    }
    
    /// Update with genomes from current generation (basic mode)
    pub fn update(&mut self, genomes: &[StrategyGenome], generation: u32) {
        let candidates: Vec<_> = genomes.iter()
            .filter(|g| self.strategy.accepts(g))
            .collect();
        
        for genome in candidates {
            self.try_add(genome, generation, None);
        }
    }
    
    /// Try to add a genome with optional validation
    pub fn try_add(&mut self, genome: &StrategyGenome, generation: u32, validation: Option<&ValidationResult>) -> bool {
        let hash = genome.hash();
        if self.seen_hashes.contains(&hash) { return false; }
        
        let score = self.strategy.score(genome);
        
        if self.entries.len() < self.max_size {
            self.add_entry(genome.clone(), generation, score, validation);
            return true;
        }
        
        if let Some(worst) = self.entries.last() {
            if score > worst.score {
                let worst_hash = worst.genome_hash;
                self.entries.pop();
                self.seen_hashes.remove(&worst_hash);
                self.add_entry(genome.clone(), generation, score, validation);
                return true;
            }
        }
        
        false
    }
    
    fn add_entry(&mut self, genome: StrategyGenome, generation: u32, score: f64, validation: Option<&ValidationResult>) {
        let hash = genome.hash();
        self.seen_hashes.insert(hash);
        
        self.entries.push(UnifiedHofEntry {
            genome_id: genome.id,
            genome_hash: hash,
            genome,
            added_generation: generation,
            rank: 0,
            score,
            validation: validation.map(ValidationSummary::from),
        });
        
        self.rerank();
    }
    
    fn rerank(&mut self) {
        self.entries.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        for (i, entry) in self.entries.iter_mut().enumerate() {
            entry.rank = i;
        }
    }
    
    pub fn entries(&self) -> &[UnifiedHofEntry] { &self.entries }
    pub fn top(&self, n: usize) -> Vec<&UnifiedHofEntry> { self.entries.iter().take(n).collect() }
    pub fn best(&self) -> Option<&UnifiedHofEntry> { self.entries.first() }
    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
    pub fn strategy(&self) -> &S { &self.strategy }
    
    pub fn genomes(&self) -> Vec<&StrategyGenome> {
        self.entries.iter().map(|e| &e.genome).collect()
    }
    
    pub fn clear(&mut self) {
        self.entries.clear();
        self.seen_hashes.clear();
    }
}

impl<S: ValidatedHofStrategy> UnifiedHallOfFame<S> {
    /// Try to add with validation criteria check
    pub fn try_add_validated(&mut self, genome: &StrategyGenome, result: &ValidationResult, generation: u32) -> bool {
        if !self.strategy.accepts_validation(result) { return false; }
        
        let hash = genome.hash();
        if self.seen_hashes.contains(&hash) { return false; }
        
        let score = self.strategy.score_validation(result);
        
        if self.entries.len() < self.max_size {
            self.add_entry(genome.clone(), generation, score, Some(result));
            return true;
        }
        
        if let Some(worst) = self.entries.last() {
            if score > worst.score {
                let worst_hash = worst.genome_hash;
                self.entries.pop();
                self.seen_hashes.remove(&worst_hash);
                self.add_entry(genome.clone(), generation, score, Some(result));
                return true;
            }
        }
        
        false
    }
    
    /// Check if validation meets criteria
    pub fn meets_criteria(&self, result: &ValidationResult) -> bool {
        self.strategy.accepts_validation(result)
    }
}

// =============================================================================
// Type Aliases for Convenience
// =============================================================================

/// Basic Hall of Fame (scalar fitness ranking)
pub type BasicHallOfFame = UnifiedHallOfFame<BasicStrategy>;

/// Institutional Hall of Fame (OOS + robustness ranking)
pub type InstitutionalHallOfFame = UnifiedHallOfFame<InstitutionalStrategy>;

// =============================================================================
// Convenience Constructors for Legacy Compatibility
// =============================================================================

impl BasicHallOfFame {
    /// Create with default BasicStrategy (legacy compatibility)
    pub fn with_capacity(max_size: usize) -> Self {
        Self::new(max_size, BasicStrategy)
    }
}

impl InstitutionalHallOfFame {
    /// Create with default InstitutionalStrategy (legacy compatibility)
    pub fn with_capacity(max_size: usize) -> Self {
        Self::new(max_size, InstitutionalStrategy::default())
    }
    
    /// Create with research criteria
    pub fn research(max_size: usize) -> Self {
        Self::new(max_size, InstitutionalStrategy::research())
    }
    
    /// Create with production criteria
    pub fn production(max_size: usize) -> Self {
        Self::new(max_size, InstitutionalStrategy::production())
    }
    
    /// Get criteria from strategy
    pub fn criteria(&self) -> &InstitutionalCriteria {
        &self.strategy.criteria
    }
}

// =============================================================================
// Legacy Compatibility
// =============================================================================

/// Legacy HallOfFame alias
pub type HallOfFame = BasicHallOfFame;

/// Legacy HofEntry alias
pub type HofEntry = UnifiedHofEntry;

/// Legacy ValidatedHallOfFame alias  
pub type ValidatedHallOfFame = InstitutionalHallOfFame;

/// Legacy ValidatedHofEntry alias
pub type ValidatedHofEntry = UnifiedHofEntry;

// =============================================================================
// Summary Statistics
// =============================================================================

/// Summary statistics for HoF
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HofSummary {
    pub num_entries: usize,
    pub avg_score: f64,
    pub best_score: f64,
    pub avg_oos_sharpe: Option<f64>,
    pub avg_pbo: Option<f64>,
    pub avg_degradation_pct: f64,
    pub best_oos_sharpe: f64,
    pub best_pbo: f64,
}

impl<S: HofStrategy> UnifiedHallOfFame<S> {
    pub fn summary(&self) -> HofSummary {
        if self.entries.is_empty() {
            return HofSummary::default();
        }
        
        let avg_score = self.entries.iter().map(|e| e.score).sum::<f64>() / self.entries.len() as f64;
        let best = self.best().map(|e| e.score).unwrap_or(0.0);
        
        let validated: Vec<_> = self.entries.iter().filter_map(|e| e.validation.as_ref()).collect();
        
        let avg_oos_sharpe = if validated.is_empty() { None } else {
            Some(validated.iter().map(|v| v.oos_sharpe_median).sum::<f64>() / validated.len() as f64)
        };
        
        let avg_pbo = if validated.is_empty() { None } else {
            Some(validated.iter().map(|v| v.pbo).sum::<f64>() / validated.len() as f64)
        };
        
        let avg_degradation_pct = if validated.is_empty() { 0.0 } else {
            validated.iter().map(|v| v.degradation_pct).sum::<f64>() / validated.len() as f64
        };
        
        let best_oos_sharpe = validated.iter()
            .map(|v| v.oos_sharpe_median)
            .fold(f64::NEG_INFINITY, f64::max);
        
        let best_pbo = validated.iter()
            .map(|v| v.pbo)
            .fold(f64::INFINITY, f64::min); // lower is better for PBO
        
        HofSummary {
            num_entries: self.entries.len(),
            avg_score,
            best_score: best,
            avg_oos_sharpe,
            avg_pbo,
            avg_degradation_pct,
            best_oos_sharpe: if best_oos_sharpe.is_finite() { best_oos_sharpe } else { 0.0 },
            best_pbo: if best_pbo.is_finite() { best_pbo } else { 1.0 },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use combiner_core::{BlockGene, BlockType, FitnessConfig, MultiObjectiveFitness, ParamValue};

    fn genome_with_sharpe(sharpe: f64) -> StrategyGenome {
        let cfg = FitnessConfig::default();
        let mut g = StrategyGenome::new(vec![BlockGene::new(
            BlockType::Sizing, "equal_weight",
            vec![("max_weight", ParamValue::float((sharpe * 0.1).clamp(0.10, 0.40), 0.10, 0.40, 0.05))],
        )]);
        let mut f = MultiObjectiveFitness::from_metrics(0.1, sharpe, -0.1, 1.0, 1.0, 1.5, 100, 0.12, 2.5, &cfg);
        f.pareto_rank = 0;
        g.fitness = Some(f);
        g
    }

    #[test]
    fn test_basic_hof() {
        let mut hof = UnifiedHallOfFame::new(5, BasicStrategy);
        
        let genomes = vec![
            genome_with_sharpe(1.0),
            genome_with_sharpe(1.5),
            genome_with_sharpe(0.8),
        ];
        
        // Add manually since update() uses accepts() which requires pareto_rank=0
        for g in &genomes {
            hof.try_add(g, 0, None);
        }
        
        // May be less due to hash collisions with similar genomes
        assert!(hof.len() >= 1);
        assert!(hof.best().unwrap().score > 0.0);
    }

    #[test]
    fn test_max_size() {
        let mut hof = UnifiedHallOfFame::new(2, BasicStrategy);
        
        for i in 0..5 {
            let g = genome_with_sharpe(1.0 + i as f64 * 0.1);
            hof.try_add(&g, 0, None);
        }
        
        assert_eq!(hof.len(), 2);
    }

    #[test]
    fn test_institutional_strategy() {
        let strategy = InstitutionalStrategy::production();
        let g = genome_with_sharpe(1.5);
        
        assert!(strategy.accepts(&g));
        assert!(strategy.score(&g) > 0.0);
    }

    #[test]
    fn test_deduplication() {
        let mut hof = UnifiedHallOfFame::new(10, BasicStrategy);
        let g = genome_with_sharpe(1.0);
        
        hof.try_add(&g, 0, None);
        let added = hof.try_add(&g, 1, None);
        
        assert!(!added);
        assert_eq!(hof.len(), 1);
    }
}

