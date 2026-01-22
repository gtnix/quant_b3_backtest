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
pub mod pareto_unified;
pub mod hall_of_fame_unified;
pub mod engine;
pub mod strategy_catalog;

// Legacy module aliases (deprecated, use hall_of_fame_unified)
#[doc(hidden)]
pub mod hall_of_fame {
    pub use super::hall_of_fame_unified::{
        BasicHallOfFame as HallOfFame,
        UnifiedHofEntry as HofEntry,
    };
}
#[doc(hidden)]
pub mod hall_of_fame_validated {
    pub use super::hall_of_fame_unified::{
        InstitutionalHallOfFame as ValidatedHallOfFame,
        UnifiedHofEntry as ValidatedHofEntry,
        InstitutionalCriteria,
        ValidationSummary as ValidationResultSummary,
        HofSummary as ValidatedHofSummary,
    };
}

// Legacy module aliases (deprecated, use pareto_unified)
#[doc(hidden)]
pub mod pareto {
    pub use super::pareto_unified::ParetoComputer as ParetoFrontier;
}
#[doc(hidden)]
pub mod pareto_simd {
    pub use super::pareto_unified::{compute_pareto_ranks_simd, compute_crowding_distance_simd};
}
pub mod persistence;
pub mod validation;
pub mod validation_reports;
pub mod evaluation;
pub mod performance_metrics;
pub mod report;
pub mod institutional_thresholds;
pub mod statistics;
pub mod audit_framework;
pub mod audit_checks;
pub mod diversity;
pub mod stagnation;
pub mod wfa;
pub mod cpcv;
pub mod diagnostics;

pub use wfa::{WfaValidator, WfaConfig, WfaResult as WfaValidatorResult, WfaFoldResult};
pub use cpcv::{CpcvValidator, CpcvConfig, CpcvResult};
pub use diagnostics::{StageComparison, MarketDiagnosticReport, GapDiagnosis, Histogram};

pub use config::EvolutionConfig;
pub use population::Population;
pub use operators::{Selection, Crossover, Mutation, AdaptiveMutation};
pub use pareto_unified::{ParetoComputer, ParetoComputer as ParetoFrontier};
pub use hall_of_fame_unified::{
    UnifiedHallOfFame, UnifiedHofEntry, BasicStrategy, InstitutionalStrategy,
    BasicHallOfFame, InstitutionalHallOfFame, HofStrategy, ValidatedHofStrategy,
    HofSummary, ValidationSummary,
    // Legacy aliases
    HallOfFame, HofEntry,
};
pub use engine::{EvolutionEngine, GenerationStats, UltraEvolutionResult, FailedCandidate};
pub use persistence::{ExperimentPersistence, ExperimentManifest, ExperimentStatus, ArtifactFormat, generate_experiment_id};
pub use validation::{
    GenomeValidatorAntiOverfit, ValidationConfig, ValidationReport,
    WfaResult, CpcvResult as ValidationCpcvResult, PboDsrResult,
};
pub use evaluation::{
    StageABatchEvaluator, StageBParallelValidator, StageBConfig, StageBStats, ValidationResult,
    SplitDataRef, ValidationSplitPlan, ValidationResultArena,
};
pub use pareto_unified::{compute_pareto_ranks_simd, compute_crowding_distance_simd};
pub use hall_of_fame_unified::{
    ValidatedHallOfFame, ValidatedHofEntry, InstitutionalCriteria,
};
pub use performance_metrics::{PerformanceMetrics, PerformanceMetricsSummary, GenerationSnapshot, IntegrityStatus};
pub use report::{FinalReportGenerator, FinalReport, ReportReader, ReportFormat, ReportError};
pub use institutional_thresholds::InstitutionalThresholds;
pub use validation_reports::{
    WfaReport, PboDsrReport, StressReport, ValidationBundle,
    ValidationVerdict as ReportVerdict,
};
pub use diversity::{
    DiversityMonitor, DiversityMetrics,
    compute_genotypic_diversity, compute_phenotypic_diversity,
    compute_structural_entropy, phenotypic_distance,
    apply_fitness_sharing, compute_sharing_factors,
};
pub use stagnation::{
    StagnationDetector, StagnationConfig, StagnationStatus,
    RestartResult,
};
pub use strategy_catalog::{StrategyCatalog, StrategyTemplate, TemplateBlock, CatalogError};

