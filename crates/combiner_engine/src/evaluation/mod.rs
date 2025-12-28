//! Evaluation module for ultra-performant strategy evaluation.
//!
//! This module provides:
//! - Stage A batch evaluation (screening, parallel)
//! - Stage B parallel validation (concurrent splits)
//! - Split data references (zero-copy)
//! - Validation split plans (pre-computed)

pub mod stage_a;
pub mod stage_b;
pub mod split_data;
pub mod split_plan;
pub mod arena;

pub use stage_a::StageABatchEvaluator;
pub use stage_b::{StageBParallelValidator, StageBConfig, StageBStats, ValidationResult};
pub use split_data::SplitDataRef;
pub use split_plan::ValidationSplitPlan;
pub use arena::ValidationResultArena;

