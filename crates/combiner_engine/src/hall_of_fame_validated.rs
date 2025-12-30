//! Validated Hall of Fame with institutional criteria.
//!
//! This module provides a Hall of Fame specifically for strategies that
//! have passed robust validation (Stage B). Only strategies meeting
//! institutional-grade criteria are admitted.

use std::collections::HashSet;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use combiner_core::StrategyGenome;
use crate::evaluation::stage_b::ValidationResult;

/// Institutional-grade acceptance criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstitutionalCriteria {
    /// Minimum OOS Sharpe ratio
    pub min_oos_sharpe: f64,
    /// Maximum PBO (Probability of Backtest Overfitting)
    pub max_pbo: f64,
    /// Minimum DSR (Deflated Sharpe Ratio)
    pub min_dsr: f64,
    /// Maximum IS/OOS degradation percentage
    pub max_degradation_pct: f64,
    /// Minimum fraction of splits that must pass
    pub min_split_pass_rate: f64,
    /// Maximum OOS drawdown (e.g., -0.35)
    pub max_oos_drawdown: f64,
}

impl Default for InstitutionalCriteria {
    fn default() -> Self {
        Self {
            min_oos_sharpe: 0.3,
            max_pbo: 0.15,
            min_dsr: 0.25,
            max_degradation_pct: 40.0,
            min_split_pass_rate: 0.6, // 60% of splits must pass
            max_oos_drawdown: -0.35,
        }
    }
}

/// Entry in the Validated Hall of Fame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatedHofEntry {
    /// Genome ID
    pub genome_id: Uuid,
    /// Genome hash for deduplication
    pub genome_hash: u64,
    /// The genome itself
    pub genome: StrategyGenome,
    /// Validation results
    pub validation: ValidationResultSummary,
    /// Generation when validated
    pub validated_generation: u32,
    /// Rank in the Hall of Fame (0 = best)
    pub rank: usize,
    /// Score for ranking (combined metric)
    pub score: f64,
}

/// Summary of validation results for storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResultSummary {
    pub oos_sharpe_median: f64,
    pub oos_sharpe_mean: f64,
    pub oos_sharpe_std: f64,
    pub oos_cagr_median: f64,
    pub degradation_pct: f64,
    pub pbo: f64,
    pub dsr: f64,
    pub splits_evaluated: u16,
    pub splits_passed: u16,
    /// Whether all validation criteria passed (including DSR)
    pub passed: bool,
}

impl From<&ValidationResult> for ValidationResultSummary {
    fn from(result: &ValidationResult) -> Self {
        Self {
            oos_sharpe_median: result.oos_sharpe_median,
            oos_sharpe_mean: result.oos_sharpe_mean,
            oos_sharpe_std: result.oos_sharpe_std,
            oos_cagr_median: result.oos_cagr_median,
            degradation_pct: result.degradation_pct,
            pbo: result.pbo,
            dsr: result.dsr,
            splits_evaluated: result.splits_evaluated,
            splits_passed: result.splits_passed,
            passed: result.passed,
        }
    }
}

/// Validated Hall of Fame
///
/// Only admits strategies that pass institutional-grade validation criteria.
/// Maintains ranking by a combined score that emphasizes robustness.
#[derive(Debug)]
pub struct ValidatedHallOfFame {
    /// Entries in the Hall of Fame
    entries: Vec<ValidatedHofEntry>,
    /// Maximum size
    max_size: usize,
    /// Hashes of genomes for deduplication
    seen_hashes: HashSet<u64>,
    /// Acceptance criteria
    criteria: InstitutionalCriteria,
}

impl ValidatedHallOfFame {
    /// Create a new Validated Hall of Fame
    pub fn new(max_size: usize) -> Self {
        Self::with_criteria(max_size, InstitutionalCriteria::default())
    }

    /// Create with custom criteria
    pub fn with_criteria(max_size: usize, criteria: InstitutionalCriteria) -> Self {
        Self {
            entries: Vec::with_capacity(max_size),
            max_size,
            seen_hashes: HashSet::new(),
            criteria,
        }
    }

    /// Check if a validation result meets institutional criteria
    pub fn meets_criteria(&self, result: &ValidationResult) -> bool {
        if !result.passed {
            return false;
        }

        if result.oos_sharpe_median < self.criteria.min_oos_sharpe {
            return false;
        }

        if result.pbo > self.criteria.max_pbo {
            return false;
        }

        if result.dsr < self.criteria.min_dsr {
            return false;
        }

        if result.degradation_pct > self.criteria.max_degradation_pct {
            return false;
        }

        let pass_rate = result.splits_passed as f64 / result.splits_evaluated.max(1) as f64;
        if pass_rate < self.criteria.min_split_pass_rate {
            return false;
        }

        true
    }

    /// Calculate ranking score (higher = better)
    fn calculate_score(result: &ValidationResult) -> f64 {
        // Combined score emphasizing robustness
        // OOS Sharpe is primary, with penalties for overfitting indicators
        let base = result.oos_sharpe_median;
        let robustness_factor = 1.0 - result.pbo; // Higher if less overfitting
        let stability_factor = 1.0 - (result.degradation_pct / 100.0).min(1.0);
        let consistency_factor = result.splits_passed as f64 / result.splits_evaluated.max(1) as f64;

        base * robustness_factor * (1.0 + stability_factor * 0.5) * consistency_factor
    }

    /// Try to add a validated genome to the Hall of Fame
    pub fn try_add(
        &mut self,
        genome: &StrategyGenome,
        result: &ValidationResult,
        generation: u32,
    ) -> bool {
        // Check criteria first
        if !self.meets_criteria(result) {
            return false;
        }

        let genome_hash = genome.hash();

        // Skip if we've seen this genome
        if self.seen_hashes.contains(&genome_hash) {
            return false;
        }

        let score = Self::calculate_score(result);

        // Check if we should add
        if self.entries.len() < self.max_size {
            // Room available
            self.add_entry(genome.clone(), result, generation, score);
            return true;
        }

        // Check if better than worst
        if let Some(worst) = self.entries.last() {
            if score > worst.score {
                // Remove worst
                let worst_hash = worst.genome_hash;
                self.entries.pop();
                self.seen_hashes.remove(&worst_hash);
                
                // Add new
                self.add_entry(genome.clone(), result, generation, score);
                return true;
            }
        }

        false
    }

    /// Add entry and re-sort
    fn add_entry(
        &mut self,
        genome: StrategyGenome,
        result: &ValidationResult,
        generation: u32,
        score: f64,
    ) {
        let genome_hash = genome.hash();
        self.seen_hashes.insert(genome_hash);

        let entry = ValidatedHofEntry {
            genome_id: genome.id,
            genome_hash,
            genome,
            validation: ValidationResultSummary::from(result),
            validated_generation: generation,
            rank: 0,
            score,
        };

        self.entries.push(entry);
        self.rerank();
    }

    /// Re-rank all entries by score
    fn rerank(&mut self) {
        self.entries.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });

        for (i, entry) in self.entries.iter_mut().enumerate() {
            entry.rank = i;
        }
    }

    /// Get all entries
    pub fn entries(&self) -> &[ValidatedHofEntry] {
        &self.entries
    }

    /// Get top N entries
    pub fn top(&self, n: usize) -> Vec<&ValidatedHofEntry> {
        self.entries.iter().take(n).collect()
    }

    /// Get the best entry
    pub fn best(&self) -> Option<&ValidatedHofEntry> {
        self.entries.first()
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get criteria
    pub fn criteria(&self) -> &InstitutionalCriteria {
        &self.criteria
    }

    /// Clear the Hall of Fame
    pub fn clear(&mut self) {
        self.entries.clear();
        self.seen_hashes.clear();
    }

    /// Generate summary report
    pub fn summary(&self) -> ValidatedHofSummary {
        if self.entries.is_empty() {
            return ValidatedHofSummary::default();
        }

        let avg_oos_sharpe = self.entries.iter()
            .map(|e| e.validation.oos_sharpe_median)
            .sum::<f64>() / self.entries.len() as f64;

        let avg_pbo = self.entries.iter()
            .map(|e| e.validation.pbo)
            .sum::<f64>() / self.entries.len() as f64;

        let avg_degradation = self.entries.iter()
            .map(|e| e.validation.degradation_pct)
            .sum::<f64>() / self.entries.len() as f64;

        let best = self.best().unwrap();

        ValidatedHofSummary {
            num_entries: self.entries.len(),
            avg_oos_sharpe,
            avg_pbo,
            avg_degradation_pct: avg_degradation,
            best_oos_sharpe: best.validation.oos_sharpe_median,
            best_pbo: best.validation.pbo,
            best_score: best.score,
        }
    }
}

/// Summary statistics for the Validated Hall of Fame
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidatedHofSummary {
    pub num_entries: usize,
    pub avg_oos_sharpe: f64,
    pub avg_pbo: f64,
    pub avg_degradation_pct: f64,
    pub best_oos_sharpe: f64,
    pub best_pbo: f64,
    pub best_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use combiner_core::{BlockGene, BlockType, ParamValue};

    fn create_test_genome() -> StrategyGenome {
        create_test_genome_with_param(126)
    }

    fn create_test_genome_with_param(lookback: i64) -> StrategyGenome {
        let gene = BlockGene::new(
            BlockType::Selection,
            "momentum",
            vec![("lookback_days", ParamValue::int(lookback, 21, 252, 21))],
        );
        StrategyGenome::new(vec![gene])
    }

    fn create_passing_result(genome_hash: u64) -> ValidationResult {
        ValidationResult {
            genome_index: 0,
            genome_hash,
            oos_sharpe_median: 0.8,
            oos_sharpe_mean: 0.75,
            oos_sharpe_std: 0.2,
            oos_cagr_median: 0.12,
            degradation_pct: 25.0,
            pbo: 0.10,
            dsr: 0.5,
            splits_evaluated: 6,
            splits_passed: 5,
            passed: true,
            early_exit: false,
            discard_reason: None,
        }
    }

    fn create_failing_result(genome_hash: u64) -> ValidationResult {
        ValidationResult {
            genome_index: 0,
            genome_hash,
            oos_sharpe_median: 0.1,
            oos_sharpe_mean: 0.05,
            oos_sharpe_std: 0.3,
            oos_cagr_median: 0.02,
            degradation_pct: 60.0,
            pbo: 0.40,
            dsr: 0.1,
            splits_evaluated: 6,
            splits_passed: 2,
            passed: false,
            early_exit: false,
            discard_reason: Some("OOS Sharpe too low".into()),
        }
    }

    #[test]
    fn test_meets_criteria_passing() {
        let hof = ValidatedHallOfFame::new(10);
        let result = create_passing_result(12345);
        
        assert!(hof.meets_criteria(&result));
    }

    #[test]
    fn test_meets_criteria_failing() {
        let hof = ValidatedHallOfFame::new(10);
        let result = create_failing_result(12345);
        
        assert!(!hof.meets_criteria(&result));
    }

    #[test]
    fn test_try_add_success() {
        let mut hof = ValidatedHallOfFame::new(10);
        let genome = create_test_genome();
        let result = create_passing_result(genome.hash());

        let added = hof.try_add(&genome, &result, 0);
        
        assert!(added);
        assert_eq!(hof.len(), 1);
    }

    #[test]
    fn test_try_add_failure() {
        let mut hof = ValidatedHallOfFame::new(10);
        let genome = create_test_genome();
        let result = create_failing_result(genome.hash());

        let added = hof.try_add(&genome, &result, 0);
        
        assert!(!added);
        assert_eq!(hof.len(), 0);
    }

    #[test]
    fn test_deduplication() {
        let mut hof = ValidatedHallOfFame::new(10);
        let genome = create_test_genome();
        let result = create_passing_result(genome.hash());

        hof.try_add(&genome, &result, 0);
        let added_again = hof.try_add(&genome, &result, 1);
        
        assert!(!added_again);
        assert_eq!(hof.len(), 1);
    }

    #[test]
    fn test_ranking() {
        let mut hof = ValidatedHallOfFame::new(10);
        
        for i in 0..5 {
            // Use different parameters to create unique genomes with different hashes
            let genome = create_test_genome_with_param(21 + (i as i64) * 21);
            let mut result = create_passing_result(genome.hash());
            result.oos_sharpe_median = 0.5 + (i as f64) * 0.1;
            hof.try_add(&genome, &result, 0);
        }

        assert_eq!(hof.len(), 5);
        
        // Best should have highest OOS Sharpe
        let best = hof.best().unwrap();
        assert!(best.validation.oos_sharpe_median > 0.8);
        assert_eq!(best.rank, 0);
    }

    #[test]
    fn test_capacity_limit() {
        let mut hof = ValidatedHallOfFame::new(3);
        
        for i in 0..5 {
            // Use different parameters to create unique genomes with different hashes
            let genome = create_test_genome_with_param(21 + (i as i64) * 21);
            let mut result = create_passing_result(genome.hash());
            result.oos_sharpe_median = 0.3 + (i as f64) * 0.1;
            hof.try_add(&genome, &result, 0);
        }

        // Should only keep best 3
        assert_eq!(hof.len(), 3);
        
        // All entries should have high OOS Sharpe
        for entry in hof.entries() {
            assert!(entry.validation.oos_sharpe_median >= 0.5);
        }
    }
}

