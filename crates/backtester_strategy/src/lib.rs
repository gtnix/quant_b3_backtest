//! Strategy Factory - Modular DSL for Declarative Strategy Composition
//!
//! This crate provides a pluggable strategy composition system that allows
//! strategies to be defined via TOML config files without modifying engine code.
//!
//! # Architecture
//!
//! - **Blocks**: Modular units (Selection, Entry, Exit, Sizing) implementing `StrategyBlock`
//! - **Registry**: Maps block_id to implementation for discovery
//! - **Compositor**: Executes pipeline steps in order, maintaining shared context
//! - **Config**: TOML-based strategy definition with validation
//!
//! # Example Strategy Config
//!
//! ```toml
//! [strategy]
//! id = "momentum_pure_v1"
//! version = "1.0.0"
//!
//! [[pipeline]]
//! type = "selection"
//! block_id = "momentum"
//! params = { lookback_days = 126, top_pct = 20 }
//!
//! [[pipeline]]
//! type = "sizing"
//! block_id = "equal_weight"
//! params = { max_weight = 0.20 }
//! ```

pub mod blocks;
pub mod compiled;
pub mod compositor;
pub mod config;
pub mod context;
pub mod experiment;
pub mod fast_context;
pub mod registry;
pub mod universe;

// Re-exports
pub use blocks::{BlockParams, BlockResult, BlockType, Signal, SignalDirection, StrategyBlock};
pub use compiled::{
    CompiledResult, CompiledStrategy, CompileError, IndicatorCache, IndicatorCacheKey,
    ParamsHash, SymbolTable,
};
pub use compositor::Compositor;
pub use config::{PipelineStep, RebalanceConfig, StrategyConfig, StrategyConstraints};
pub use context::{StrategyContext, TraceEntry};
pub use experiment::{
    ArtifactWriter, BlockCatalog, Comparator, ExperimentRunner, MetricsCalculator,
    RunMetadata, RunMetrics, ExperimentResult,
};
pub use fast_context::{
    CandidatesSoA, FastContext, PreallocBuffers, SignalState,
    fast_momentum_select, fast_low_vol_select, fast_equal_weight,
};
pub use registry::BlockRegistry;
pub use universe::{
    ComplexityTier, StrategyFamily, TrainingModel, UniverseConfig, UniverseLoader,
    UniverseRestrictions, UniverseValidator, EffectiveLimits,
};

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::blocks::{
        BlockParams, BlockResult, BlockType, Signal, SignalDirection, StrategyBlock,
    };
    pub use crate::compiled::{
        CompiledResult, CompiledStrategy, IndicatorCache, ParamsHash, SymbolTable,
    };
    pub use crate::compositor::Compositor;
    pub use crate::config::{PipelineStep, RebalanceConfig, StrategyConfig, StrategyConstraints};
    pub use crate::context::{StrategyContext, TraceEntry};
    pub use crate::experiment::{
        ExperimentRunner, MetricsCalculator, ArtifactWriter, Comparator,
        RunMetadata, RunMetrics, ExperimentResult,
    };
    pub use crate::registry::BlockRegistry;
    pub use crate::universe::{
        ComplexityTier, StrategyFamily, TrainingModel, UniverseConfig, UniverseLoader,
        UniverseRestrictions, UniverseValidator,
    };
}

