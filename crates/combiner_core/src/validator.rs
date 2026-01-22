//! Genome validation against BlockRegistry and parameter ranges.

use crate::error::ValidationError;
use crate::genome::{BlockType, StrategyGenome};
use crate::param_ranges::ParamRanges;

/// Genome validator.
#[derive(Debug)]
pub struct GenomeValidator {
    param_ranges: ParamRanges,
    strict: bool,
    /// Minimum data days available for validation (0 = skip data validation)
    min_data_days: usize,
}

impl Default for GenomeValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl GenomeValidator {
    /// Create a new validator with default parameter ranges.
    pub fn new() -> Self {
        Self {
            param_ranges: ParamRanges::new(),
            strict: false,
            min_data_days: 0, // Skip data validation by default
        }
    }

    /// Create a validator with custom parameter ranges.
    pub fn with_ranges(param_ranges: ParamRanges) -> Self {
        Self {
            param_ranges,
            strict: false,
            min_data_days: 0,
        }
    }

    /// Enable strict mode (fails on warnings).
    pub fn strict(mut self) -> Self {
        self.strict = true;
        self
    }

    /// Set minimum data days for validation.
    /// When set, genomes requiring more data than available will fail validation.
    pub fn with_data_days(mut self, days: usize) -> Self {
        self.min_data_days = days;
        self
    }

    /// Validate a genome.
    ///
    /// Returns Ok(()) if valid, or Err with validation error.
    pub fn validate(&self, genome: &StrategyGenome) -> Result<(), ValidationError> {
        // Check for empty genome
        if genome.genes.is_empty() {
            return Err(ValidationError::EmptyGenome);
        }

        // Check for required Sizing block
        if !genome.has_block_type(BlockType::Sizing) {
            return Err(ValidationError::MissingSizing);
        }

        // Check Entry requires Exit (warning in non-strict, error in strict)
        if genome.has_block_type(BlockType::Entry) && !genome.has_block_type(BlockType::Exit) {
            if self.strict {
                return Err(ValidationError::EntryWithoutExit);
            }
            // In non-strict mode, just log a warning (caller should handle)
        }

        // Validate weight constraints (P0 fix: detect incompatible max_weight/max_positions)
        Self::validate_weight_constraints(genome)?;

        // Validate data requirements (if min_data_days is set)
        if self.min_data_days > 0 {
            self.validate_data_requirements(genome)?;
        }

        // Validate each gene
        for gene in &genome.genes {
            // Check block_id exists
            let block_spec = self.param_ranges.get_block(&gene.block_id).ok_or_else(|| {
                ValidationError::UnknownBlock(gene.block_id.clone(), gene.block_type.to_string())
            })?;

            // Check block type matches
            if block_spec.block_type != gene.block_type {
                return Err(ValidationError::UnknownBlock(
                    gene.block_id.clone(),
                    gene.block_type.to_string(),
                ));
            }

            // Validate parameters
            for (param_name, param_value) in &gene.params {
                // Find param spec
                let param_spec = block_spec.params.iter().find(|p| p.name == *param_name);

                if param_spec.is_none() && self.strict {
                    // Unknown parameter in strict mode
                    continue; // Allow extra params in non-strict
                }

                // Check value is within range
                if !param_value.is_valid() {
                    return Err(ValidationError::ParamOutOfRange {
                        block: gene.block_id.clone(),
                        param: param_name.clone(),
                        message: "value outside valid range".into(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Validate and return warnings (non-fatal issues).
    pub fn validate_with_warnings(
        &self,
        genome: &StrategyGenome,
    ) -> Result<Vec<String>, ValidationError> {
        let mut warnings = Vec::new();

        // Check for empty genome
        if genome.genes.is_empty() {
            return Err(ValidationError::EmptyGenome);
        }

        // Check for required Sizing block
        if !genome.has_block_type(BlockType::Sizing) {
            return Err(ValidationError::MissingSizing);
        }

        // Check Entry requires Exit (warning)
        if genome.has_block_type(BlockType::Entry) && !genome.has_block_type(BlockType::Exit) {
            warnings.push("Entry blocks present without Exit blocks".into());
        }

        // Validate weight constraints (P0 fix)
        Self::validate_weight_constraints(genome)?;

        // Validate each gene
        for gene in &genome.genes {
            // Check block_id exists
            let block_spec = self.param_ranges.get_block(&gene.block_id).ok_or_else(|| {
                ValidationError::UnknownBlock(gene.block_id.clone(), gene.block_type.to_string())
            })?;

            // Check block type matches
            if block_spec.block_type != gene.block_type {
                return Err(ValidationError::UnknownBlock(
                    gene.block_id.clone(),
                    gene.block_type.to_string(),
                ));
            }

            // Check for missing required parameters
            for param_spec in &block_spec.params {
                if !gene.params.contains_key(&param_spec.name) {
                    warnings.push(format!(
                        "Missing parameter '{}' for block '{}', using default",
                        param_spec.name, gene.block_id
                    ));
                }
            }

            // Validate parameters
            for (param_name, param_value) in &gene.params {
                if !param_value.is_valid() {
                    return Err(ValidationError::ParamOutOfRange {
                        block: gene.block_id.clone(),
                        param: param_name.clone(),
                        message: "value outside valid range".into(),
                    });
                }
            }
        }

        Ok(warnings)
    }

    /// Check if a genome is valid (returns bool).
    pub fn is_valid(&self, genome: &StrategyGenome) -> bool {
        self.validate(genome).is_ok()
    }

    /// Validate that genome data requirements are compatible with available data.
    ///
    /// Checks Entry blocks for lookback period requirements and ensures they
    /// don't exceed the available data days.
    fn validate_data_requirements(&self, genome: &StrategyGenome) -> Result<(), ValidationError> {
        for gene in &genome.genes {
            let required_days = match gene.block_id.as_str() {
                "ma_crossover" => {
                    // MA Crossover requires slow_period + 1 days for calculation
                    let slow = gene.get_param("slow_period")
                        .map(|p| p.as_i64() as usize)
                        .unwrap_or(200);
                    slow + 1
                }
                "rsi" => {
                    // RSI requires period + 1 days
                    let period = gene.get_param("period")
                        .map(|p| p.as_i64() as usize)
                        .unwrap_or(14);
                    period + 1
                }
                "macd" => {
                    // MACD requires slow_ema + signal days for convergence
                    let slow = gene.get_param("slow_ema")
                        .map(|p| p.as_i64() as usize)
                        .unwrap_or(26);
                    let signal = gene.get_param("signal")
                        .map(|p| p.as_i64() as usize)
                        .unwrap_or(9);
                    slow + signal
                }
                "bollinger" => {
                    // Bollinger requires period days
                    let period = gene.get_param("period")
                        .map(|p| p.as_i64() as usize)
                        .unwrap_or(20);
                    period
                }
                "zscore" => {
                    // Z-score requires period days
                    let period = gene.get_param("period")
                        .map(|p| p.as_i64() as usize)
                        .unwrap_or(20);
                    period
                }
                "momentum" => {
                    // Momentum requires lookback_days + skip_last_days
                    let lookback = gene.get_param("lookback_days")
                        .map(|p| p.as_i64() as usize)
                        .unwrap_or(126);
                    let skip = gene.get_param("skip_last_days")
                        .map(|p| p.as_i64() as usize)
                        .unwrap_or(21);
                    lookback + skip
                }
                "low_vol" => {
                    // Low vol requires lookback_days
                    let lookback = gene.get_param("lookback_days")
                        .map(|p| p.as_i64() as usize)
                        .unwrap_or(60);
                    lookback
                }
                _ => 0, // Other blocks don't have data requirements
            };

            if required_days > self.min_data_days {
                return Err(ValidationError::InsufficientData {
                    block: gene.block_id.clone(),
                    required: required_days,
                    available: self.min_data_days,
                });
            }
        }
        Ok(())
    }

    /// Validate weight constraints are compatible.
    ///
    /// Checks that max_weight and max_positions don't create impossible constraints.
    /// For example, with equal_weight sizing and 3 positions, each position gets ~33%.
    /// If max_weight is set to 0.25, this is impossible.
    fn validate_weight_constraints(genome: &StrategyGenome) -> Result<(), ValidationError> {
        let sizing_genes = genome.genes_by_type(BlockType::Sizing);

        // Extract max_weight from sizing blocks
        let max_weight = sizing_genes
            .iter()
            .filter_map(|g| g.get_param("max_weight"))
            .map(|p| p.as_f64())
            .next()
            .unwrap_or(1.0);

        // Extract max_positions from sizing blocks
        let max_positions = sizing_genes
            .iter()
            .filter_map(|g| g.get_param("max_positions"))
            .map(|p| p.as_i64() as usize)
            .next()
            .unwrap_or(20);

        // Skip validation if max_weight is 1.0 (no constraint)
        if max_weight >= 0.99 {
            return Ok(());
        }

        // Calculate minimum weight per position for equal weight distribution
        // This is the theoretical minimum if all max_positions are filled
        let min_weight_per_position = 1.0 / max_positions as f64;

        // If equal distribution would exceed max_weight, it's invalid
        // Allow small epsilon for floating point
        if min_weight_per_position > max_weight + 0.01 {
            return Err(ValidationError::InvalidConstraints(format!(
                "max_weight ({:.2}) incompatible with max_positions ({}): \
                 equal distribution requires {:.2} per position",
                max_weight, max_positions, min_weight_per_position
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::{BlockGene, ParamValue};

    #[test]
    fn test_valid_genome() {
        let validator = GenomeValidator::new();

        let genome = StrategyGenome::new(vec![
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
        ]);

        assert!(validator.validate(&genome).is_ok());
    }

    #[test]
    fn test_missing_sizing() {
        let validator = GenomeValidator::new();

        let genome = StrategyGenome::new(vec![BlockGene::new(
            BlockType::Selection,
            "momentum",
            vec![("lookback_days", ParamValue::int(126, 21, 252, 21))],
        )]);

        let result = validator.validate(&genome);
        assert!(matches!(result, Err(ValidationError::MissingSizing)));
    }

    #[test]
    fn test_unknown_block() {
        let validator = GenomeValidator::new();

        let genome = StrategyGenome::new(vec![
            BlockGene::with_defaults(BlockType::Selection, "nonexistent_block"),
            BlockGene::with_defaults(BlockType::Sizing, "equal_weight"),
        ]);

        let result = validator.validate(&genome);
        assert!(matches!(result, Err(ValidationError::UnknownBlock(_, _))));
    }

    #[test]
    fn test_entry_without_exit_strict() {
        let validator = GenomeValidator::new().strict();

        let genome = StrategyGenome::new(vec![
            BlockGene::with_defaults(BlockType::Entry, "ma_crossover"),
            BlockGene::with_defaults(BlockType::Sizing, "equal_weight"),
        ]);

        let result = validator.validate(&genome);
        assert!(matches!(result, Err(ValidationError::EntryWithoutExit)));
    }

    #[test]
    fn test_entry_without_exit_non_strict() {
        let validator = GenomeValidator::new();

        let genome = StrategyGenome::new(vec![
            BlockGene::with_defaults(BlockType::Entry, "ma_crossover"),
            BlockGene::with_defaults(BlockType::Sizing, "equal_weight"),
        ]);

        // Should pass in non-strict mode (just a warning)
        assert!(validator.validate(&genome).is_ok());
    }

    #[test]
    fn test_param_out_of_range() {
        let validator = GenomeValidator::new();

        let genome = StrategyGenome::new(vec![
            BlockGene::new(
                BlockType::Selection,
                "momentum",
                vec![(
                    "lookback_days",
                    ParamValue::Int {
                        value: 500, // Out of range
                        min: 21,
                        max: 252,
                        step: 21,
                    },
                )],
            ),
            BlockGene::with_defaults(BlockType::Sizing, "equal_weight"),
        ]);

        let result = validator.validate(&genome);
        assert!(matches!(result, Err(ValidationError::ParamOutOfRange { .. })));
    }

    #[test]
    fn test_incompatible_weight_constraints() {
        let validator = GenomeValidator::new();

        // Create a genome with max_weight=0.10 but max_positions=5
        // This is invalid because equal weight would require 0.20 per position
        let genome = StrategyGenome::new(vec![
            BlockGene::new(
                BlockType::Sizing,
                "equal_weight",
                vec![
                    ("max_weight", ParamValue::float(0.10, 0.05, 0.5, 0.05)),
                    ("max_positions", ParamValue::int(5, 5, 50, 5)),
                ],
            ),
        ]);

        let result = validator.validate(&genome);
        assert!(matches!(result, Err(ValidationError::InvalidConstraints(_))));
    }

    #[test]
    fn test_compatible_weight_constraints() {
        let validator = GenomeValidator::new();

        // Create a genome with max_weight=0.25 and max_positions=10
        // This is valid because equal weight would require 0.10 per position
        let genome = StrategyGenome::new(vec![
            BlockGene::new(
                BlockType::Sizing,
                "equal_weight",
                vec![
                    ("max_weight", ParamValue::float(0.25, 0.05, 0.5, 0.05)),
                    ("max_positions", ParamValue::int(10, 5, 50, 5)),
                ],
            ),
        ]);

        let result = validator.validate(&genome);
        assert!(result.is_ok());
    }
}

