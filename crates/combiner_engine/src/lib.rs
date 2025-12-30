//! Combiner Engine - Evolution engine for the Generative Combiner (SCG)
//!
//! This crate provides the evolutionary algorithm components:
//!
//! - Population management
//! - Genetic operators (selection, crossover, mutation)
//! - Pareto frontier calculation (NSGA-II)
//! - Hall of Fame
//! - Stopping criteria
//! - Ultra-performant evaluation (Stage A/B)
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
pub mod pareto_simd;
pub mod hall_of_fame;
pub mod hall_of_fame_validated;
pub mod engine;
pub mod persistence;
pub mod validation;
pub mod evaluation;
pub mod performance_metrics;
pub mod report;
pub mod institutional_thresholds;

pub use config::EvolutionConfig;
pub use population::Population;
pub use operators::{Selection, Crossover, Mutation};
pub use pareto::ParetoFrontier;
pub use hall_of_fame::HallOfFame;
pub use engine::{EvolutionEngine, GenerationStats, UltraEvolutionResult};
pub use persistence::{ExperimentPersistence, ExperimentManifest, ExperimentStatus, generate_experiment_id};
pub use validation::{
    GenomeValidatorAntiOverfit, ValidationConfig, ValidationReport,
    WfaResult, CpcvResult, PboDsrResult,
};
pub use evaluation::{
    StageABatchEvaluator, StageBParallelValidator, StageBConfig, StageBStats, ValidationResult,
    SplitDataRef, ValidationSplitPlan, ValidationResultArena,
};
pub use pareto_simd::{compute_pareto_ranks_simd, compute_crowding_distance_simd};
pub use hall_of_fame_validated::{ValidatedHallOfFame, ValidatedHofEntry, InstitutionalCriteria};
pub use performance_metrics::{PerformanceMetrics, PerformanceMetricsSummary, GenerationSnapshot, IntegrityStatus};
pub use report::{FinalReportGenerator, FinalReport};
pub use institutional_thresholds::InstitutionalThresholds;

