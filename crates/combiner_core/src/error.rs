//! Error types for combiner_core.

use thiserror::Error;

/// Main error type for combiner operations.
#[derive(Debug, Error)]
pub enum CombinerError {
    #[error("Conversion error: {0}")]
    Conversion(#[from] ConversionError),

    #[error("Validation error: {0}")]
    Validation(#[from] ValidationError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// Errors during genome to TOML/StrategyConfig conversion.
#[derive(Debug, Error)]
pub enum ConversionError {
    #[error("Missing required block type: {0}")]
    MissingBlockType(String),

    #[error("Invalid block id '{0}' for type {1}")]
    InvalidBlockId(String, String),

    #[error("TOML serialization failed: {0}")]
    TomlSerialize(#[from] toml::ser::Error),

    #[error("JSON serialization failed: {0}")]
    JsonSerialize(#[from] serde_json::Error),

    #[error("Parameter conversion failed for '{0}': {1}")]
    ParamConversion(String, String),
}

/// Errors during genome validation.
#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("Genome must have at least one Sizing block")]
    MissingSizing,

    #[error("Entry blocks require at least one Exit block (has Entry but no Exit)")]
    EntryWithoutExit,

    #[error("Unknown block_id '{0}' for type {1}")]
    UnknownBlock(String, String),

    #[error("Parameter '{param}' out of range for block '{block}': {message}")]
    ParamOutOfRange {
        block: String,
        param: String,
        message: String,
    },

    #[error("Missing required parameter '{param}' for block '{block}'")]
    MissingParam { block: String, param: String },

    #[error("Empty genome (no genes)")]
    EmptyGenome,

    #[error("Duplicate block type not allowed: {0}")]
    DuplicateBlock(String),
}

