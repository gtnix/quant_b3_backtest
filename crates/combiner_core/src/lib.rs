//! Combiner Core - Types for the Generative Combiner (SCG)
//!
//! This crate provides the fundamental types for the evolutionary strategy
//! discovery system:
//!
//! - `StrategyGenome` - Complete genome representing a strategy
//! - `BlockGene` - Individual gene representing a block and its parameters
//! - `ParamValue` - Parameter values with ranges for mutation
//! - `MultiObjectiveFitness` - Multi-objective fitness for Pareto optimization
//!
//! # Example
//!
//! ```ignore
//! use combiner_core::{StrategyGenome, BlockGene, BlockType, ParamValue};
//!
//! let gene = BlockGene::new(
//!     BlockType::Selection,
//!     "momentum",
//!     vec![("lookback_days", ParamValue::int(126, 21, 252, 21))],
//! );
//!
//! let genome = StrategyGenome::new(vec![gene]);
//! let toml = genome.to_toml()?;
//! ```

pub mod genome;
pub mod fitness;
pub mod converter;
pub mod validator;
pub mod param_ranges;
pub mod error;

pub use genome::{StrategyGenome, BlockGene, BlockType, ParamValue};
pub use fitness::{MultiObjectiveFitness, FitnessConfig, ObjectiveSpec, Direction};
pub use converter::GenomeConverter;
pub use validator::GenomeValidator;
pub use param_ranges::ParamRanges;
pub use error::{CombinerError, ConversionError, ValidationError};

