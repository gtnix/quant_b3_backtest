//! Combiner Engine - Evolution engine for the Generative Combiner (SCG)
//!
//! This crate provides the evolutionary algorithm components:
//!
//! - Population management
//! - Genetic operators (selection, crossover, mutation)
//! - Pareto frontier calculation (NSGA-II)
//! - Hall of Fame
//! - Stopping criteria
//!
//! # Example
//!
//! ```ignore
//! use combiner_engine::{EvolutionEngine, EvolutionConfig};
//!
//! let config = EvolutionConfig::default();
//! let mut engine = EvolutionEngine::new(config);
//! engine.evolve()?;
//! let hall_of_fame = engine.hall_of_fame();
//! ```

pub mod config;
pub mod population;
pub mod operators;
pub mod pareto;
pub mod hall_of_fame;
pub mod engine;
pub mod persistence;
pub mod validation;

pub use config::EvolutionConfig;
pub use population::Population;
pub use operators::{Selection, Crossover, Mutation};
pub use pareto::ParetoFrontier;
pub use hall_of_fame::HallOfFame;
pub use engine::{EvolutionEngine, GenerationStats};
pub use persistence::{ExperimentPersistence, ExperimentManifest, ExperimentStatus, generate_experiment_id};
pub use validation::{
    GenomeValidatorAntiOverfit, ValidationConfig, ValidationReport,
    WfaResult, CpcvResult, PboDsrResult,
};

