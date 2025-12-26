//! Strategy configuration validation.

use super::{StrategyConfig, PipelineStep};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigValidationError {
    #[error("Empty strategy ID")]
    EmptyStrategyId,
    #[error("Empty pipeline - no steps defined")]
    EmptyPipeline,
    #[error("Unknown block type '{0}' in step {1}")]
    UnknownBlockType(String, usize),
    #[error("Unknown block_id '{0}' for type '{1}' in step {2}")]
    UnknownBlockId(String, String, usize),
    #[error("Invalid constraint: {0}")]
    InvalidConstraint(String),
    #[error("Invalid rebalance frequency: {0}")]
    InvalidRebalanceFrequency(String),
}

const VALID_BLOCK_TYPES: &[&str] = &["selection", "entry", "exit", "sizing", "filter"];

const VALID_SELECTION_BLOCKS: &[&str] = &[
    "momentum", "value", "quality", "low_vol", "dividend", "size", "carry",
];

const VALID_ENTRY_BLOCKS: &[&str] = &[
    "ma_crossover", "bollinger", "rsi", "macd", "zscore",
];

const VALID_EXIT_BLOCKS: &[&str] = &[
    "stop_loss", "take_profit", "trailing_stop", "time_exit", 
    "mean_reversion_exit", "trend_reversal_exit",
];

const VALID_SIZING_BLOCKS: &[&str] = &[
    "equal_weight", "risk_parity", "vol_targeting",
];

const VALID_FREQUENCIES: &[&str] = &["daily", "weekly", "monthly"];

/// Validate strategy configuration.
pub fn validate_config(config: &StrategyConfig) -> Result<(), ConfigValidationError> {
    // Check strategy ID
    if config.strategy.id.trim().is_empty() {
        return Err(ConfigValidationError::EmptyStrategyId);
    }

    // Check pipeline not empty
    if config.pipeline.is_empty() {
        return Err(ConfigValidationError::EmptyPipeline);
    }

    // Validate each step
    for (i, step) in config.pipeline.iter().enumerate() {
        validate_step(step, i)?;
    }

    // Validate rebalance frequency
    if !VALID_FREQUENCIES.contains(&config.rebalance.frequency.as_str()) {
        return Err(ConfigValidationError::InvalidRebalanceFrequency(
            config.rebalance.frequency.clone(),
        ));
    }

    // Validate constraints
    if config.constraints.max_weight_per_asset <= 0.0 
        || config.constraints.max_weight_per_asset > 1.0 
    {
        return Err(ConfigValidationError::InvalidConstraint(
            "max_weight_per_asset must be between 0 and 1".into(),
        ));
    }

    Ok(())
}

fn validate_step(step: &PipelineStep, index: usize) -> Result<(), ConfigValidationError> {
    // Check block type
    if !VALID_BLOCK_TYPES.contains(&step.step_type.as_str()) {
        return Err(ConfigValidationError::UnknownBlockType(
            step.step_type.clone(),
            index,
        ));
    }

    // Check block_id for type
    let valid_blocks = match step.step_type.as_str() {
        "selection" => VALID_SELECTION_BLOCKS,
        "entry" => VALID_ENTRY_BLOCKS,
        "exit" => VALID_EXIT_BLOCKS,
        "sizing" => VALID_SIZING_BLOCKS,
        "filter" => VALID_SELECTION_BLOCKS, // Filters use same blocks
        _ => return Ok(()), // Already checked above
    };

    if !valid_blocks.contains(&step.block_id.as_str()) {
        return Err(ConfigValidationError::UnknownBlockId(
            step.block_id.clone(),
            step.step_type.clone(),
            index,
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::load_strategy_from_str;

    #[test]
    fn test_valid_config() {
        let toml_str = r#"
[strategy]
id = "test"

[[pipeline]]
type = "selection"
block_id = "momentum"

[[pipeline]]
type = "sizing"
block_id = "equal_weight"

[rebalance]
frequency = "weekly"
"#;

        let config = load_strategy_from_str(toml_str).unwrap();
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_empty_id() {
        let toml_str = r#"
[strategy]
id = ""

[[pipeline]]
type = "selection"
block_id = "momentum"
"#;

        let config = load_strategy_from_str(toml_str).unwrap();
        let result = validate_config(&config);
        assert!(matches!(result, Err(ConfigValidationError::EmptyStrategyId)));
    }

    #[test]
    fn test_unknown_block_type() {
        let toml_str = r#"
[strategy]
id = "test"

[[pipeline]]
type = "unknown_type"
block_id = "momentum"
"#;

        let config = load_strategy_from_str(toml_str).unwrap();
        let result = validate_config(&config);
        assert!(matches!(result, Err(ConfigValidationError::UnknownBlockType(_, _))));
    }

    #[test]
    fn test_unknown_block_id() {
        let toml_str = r#"
[strategy]
id = "test"

[[pipeline]]
type = "selection"
block_id = "nonexistent_block"
"#;

        let config = load_strategy_from_str(toml_str).unwrap();
        let result = validate_config(&config);
        assert!(matches!(result, Err(ConfigValidationError::UnknownBlockId(_, _, _))));
    }
}

