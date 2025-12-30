//! Genome repair operators for weight constraint normalization.
//!
//! This module provides repair operators that ensure genomes produced by
//! genetic operators (mutation/crossover) satisfy portfolio weight constraints
//! before being evaluated by the backtester.
//!
//! The repair operator is deterministic and introduces minimal bias by:
//! - Clamping values to valid ranges
//! - Adjusting max_positions to be compatible with max_weight
//! - Logging all repairs for observability

use crate::{BlockType, ParamValue, StrategyGenome};
use tracing::{debug, warn};

/// Statistics from a genome repair operation
#[derive(Debug, Clone, Default)]
pub struct GenomeRepairStats {
    /// Number of genomes that needed repair
    pub repaired_count: u32,
    /// Number of max_weight clamps applied
    pub max_weight_clamps: u32,
    /// Number of max_positions adjustments
    pub max_positions_adjustments: u32,
    /// Number of weight sum normalizations
    pub weight_sum_normalizations: u32,
}

impl GenomeRepairStats {
    /// Check if any repair was performed
    pub fn was_repaired(&self) -> bool {
        self.max_weight_clamps > 0 
            || self.max_positions_adjustments > 0 
            || self.weight_sum_normalizations > 0
    }
    
    /// Merge with another stats struct
    pub fn merge(&mut self, other: &GenomeRepairStats) {
        self.repaired_count += other.repaired_count;
        self.max_weight_clamps += other.max_weight_clamps;
        self.max_positions_adjustments += other.max_positions_adjustments;
        self.weight_sum_normalizations += other.weight_sum_normalizations;
    }
}

/// Configuration for genome repair
#[derive(Debug, Clone)]
pub struct RepairConfig {
    /// Maximum weight per asset (e.g., 0.35 = 35%)
    pub max_weight_per_asset: f64,
    /// Minimum weight per asset (e.g., 0.01 = 1%)
    pub min_weight_per_asset: f64,
    /// Maximum number of positions
    pub max_positions: i64,
    /// Minimum number of positions
    pub min_positions: i64,
}

impl Default for RepairConfig {
    fn default() -> Self {
        Self {
            max_weight_per_asset: 0.35,
            min_weight_per_asset: 0.01,
            max_positions: 20,
            min_positions: 3,
        }
    }
}

/// Repair a genome to ensure valid weight constraints.
///
/// This function modifies the genome in place to ensure:
/// 1. max_weight param <= max_weight_per_asset
/// 2. max_positions is compatible with max_weight (1/max_weight <= max_positions)
/// 3. Parameters are within their defined ranges
///
/// Returns statistics about the repairs performed.
pub fn repair_genome(genome: &mut StrategyGenome, config: &RepairConfig) -> GenomeRepairStats {
    let mut stats = GenomeRepairStats::default();
    
    for gene in &mut genome.genes {
        if gene.block_type != BlockType::Sizing {
            continue;
        }
        
        // Repair max_weight param
        if let Some(param) = gene.params.get_mut("max_weight") {
            if let ParamValue::Float { value, min, max, step } = param {
                let original = *value;
                
                // Clamp to config max
                let clamped = original.clamp(*min, config.max_weight_per_asset.min(*max));
                
                if (original - clamped).abs() > 0.001 {
                    debug!(
                        "Genome repair: max_weight {:.3} -> {:.3} (config max: {:.3})",
                        original, clamped, config.max_weight_per_asset
                    );
                    *value = clamped;
                    stats.max_weight_clamps += 1;
                }
                
                // Snap to step
                let steps = ((*value - *min) / *step).round();
                *value = *min + steps * *step;
                *value = value.clamp(*min, *max);
            }
        }
        
        // Get current max_weight for compatibility check
        let current_max_weight = gene.params.get("max_weight")
            .and_then(|p| match p {
                ParamValue::Float { value, .. } => Some(*value),
                _ => None,
            })
            .unwrap_or(1.0);
        
        // Repair max_positions to be compatible with max_weight
        // For equal_weight sizing: weight_per_position = 1/max_positions
        // Constraint: weight_per_position <= max_weight
        // Therefore: max_positions >= 1/max_weight
        if let Some(param) = gene.params.get_mut("max_positions") {
            if let ParamValue::Int { value, min, max, step } = param {
                let min_required = (1.0 / current_max_weight).ceil() as i64;
                
                if *value < min_required {
                    let original = *value;
                    let adjusted = min_required.clamp(*min, *max);
                    
                    // Snap to step
                    let steps = ((adjusted - *min) as f64 / *step as f64).ceil() as i64;
                    let snapped = *min + steps * *step;
                    *value = snapped.clamp(*min, *max);
                    
                    debug!(
                        "Genome repair: max_positions {} -> {} (min required for max_weight {:.3}: {})",
                        original, *value, current_max_weight, min_required
                    );
                    stats.max_positions_adjustments += 1;
                }
            }
        }
    }
    
    if stats.was_repaired() {
        stats.repaired_count = 1;
        warn!(
            "Genome repaired: {} weight clamps, {} position adjustments",
            stats.max_weight_clamps, stats.max_positions_adjustments
        );
    }
    
    stats
}

/// Validate that a genome's weight constraints are valid.
/// Returns Ok(()) if valid, or an error description if invalid.
pub fn validate_genome_weights(genome: &StrategyGenome, config: &RepairConfig) -> Result<(), String> {
    for gene in &genome.genes {
        if gene.block_type != BlockType::Sizing {
            continue;
        }
        
        // Check max_weight
        if let Some(ParamValue::Float { value, .. }) = gene.params.get("max_weight") {
            if *value > config.max_weight_per_asset + 0.001 {
                return Err(format!(
                    "max_weight ({:.3}) exceeds config max ({:.3})",
                    value, config.max_weight_per_asset
                ));
            }
        }
        
        // Check max_positions compatibility
        let max_weight = gene.params.get("max_weight")
            .and_then(|p| match p { ParamValue::Float { value, .. } => Some(*value), _ => None })
            .unwrap_or(1.0);
            
        let max_positions = gene.params.get("max_positions")
            .and_then(|p| match p { ParamValue::Int { value, .. } => Some(*value), _ => None })
            .unwrap_or(20);
        
        let min_required_positions = (1.0 / max_weight).ceil() as i64;
        
        if max_positions < min_required_positions {
            return Err(format!(
                "max_positions ({}) incompatible with max_weight ({:.3}): requires >= {}",
                max_positions, max_weight, min_required_positions
            ));
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BlockGene;
    
    fn make_sizing_gene(max_weight: f64, max_positions: i64) -> BlockGene {
        BlockGene::new(
            BlockType::Sizing,
            "equal_weight",
            vec![
                ("max_weight", ParamValue::float(max_weight, 0.05, 1.0, 0.05)),
                ("max_positions", ParamValue::int(max_positions, 3, 50, 1)),
            ],
        )
    }
    
    #[test]
    fn test_repair_excessive_max_weight() {
        let gene = make_sizing_gene(0.50, 5);
        let mut genome = StrategyGenome::new(vec![gene]);
        
        let config = RepairConfig {
            max_weight_per_asset: 0.35,
            ..Default::default()
        };
        
        let stats = repair_genome(&mut genome, &config);
        
        assert!(stats.was_repaired());
        assert_eq!(stats.max_weight_clamps, 1);
        
        // Check max_weight was clamped
        let sizing = genome.genes_by_type(BlockType::Sizing)[0];
        let max_weight = sizing.params.get("max_weight").unwrap().as_f64();
        assert!(max_weight <= 0.35 + 0.001);
    }
    
    #[test]
    fn test_repair_incompatible_positions() {
        // max_weight=0.20 requires min 5 positions (1/0.20 = 5)
        // But we set max_positions=3, which is incompatible
        let gene = make_sizing_gene(0.20, 3);
        let mut genome = StrategyGenome::new(vec![gene]);
        
        let config = RepairConfig::default();
        let stats = repair_genome(&mut genome, &config);
        
        assert!(stats.was_repaired());
        assert_eq!(stats.max_positions_adjustments, 1);
        
        // Check max_positions was adjusted
        let sizing = genome.genes_by_type(BlockType::Sizing)[0];
        let max_positions = sizing.params.get("max_positions")
            .and_then(|p| match p { ParamValue::Int { value, .. } => Some(*value), _ => None })
            .unwrap();
        assert!(max_positions >= 5);
    }
    
    #[test]
    fn test_valid_genome_no_repair() {
        // Valid genome: max_weight=0.20, max_positions=10 (1/0.20=5, 10>=5)
        let gene = make_sizing_gene(0.20, 10);
        let mut genome = StrategyGenome::new(vec![gene]);
        
        let config = RepairConfig::default();
        let stats = repair_genome(&mut genome, &config);
        
        assert!(!stats.was_repaired());
    }
    
    #[test]
    fn test_validate_genome_weights() {
        let gene = make_sizing_gene(0.50, 5);
        let genome = StrategyGenome::new(vec![gene]);
        
        let strict_config = RepairConfig {
            max_weight_per_asset: 0.35,
            ..Default::default()
        };
        
        // Should fail validation
        let result = validate_genome_weights(&genome, &strict_config);
        assert!(result.is_err());
        
        // Repair it
        let mut genome = genome;
        repair_genome(&mut genome, &strict_config);
        
        // Should pass validation now
        let result = validate_genome_weights(&genome, &strict_config);
        assert!(result.is_ok());
    }
}

