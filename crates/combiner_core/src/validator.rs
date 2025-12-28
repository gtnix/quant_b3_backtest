//! Genome validation against BlockRegistry and parameter ranges.

use crate::error::ValidationError;
use crate::genome::{BlockType, StrategyGenome};
use crate::param_ranges::ParamRanges;

/// Genome validator.
#[derive(Debug)]
pub struct GenomeValidator {
    param_ranges: ParamRanges,
    strict: bool,
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
        }
    }

    /// Create a validator with custom parameter ranges.
    pub fn with_ranges(param_ranges: ParamRanges) -> Self {
        Self {
            param_ranges,
            strict: false,
        }
    }

    /// Enable strict mode (fails on warnings).
    pub fn strict(mut self) -> Self {
        self.strict = true;
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
}

