//! Compositor - pipeline runner for strategy execution.

use crate::blocks::BlockResult;
use crate::config::{validate_config, ConfigValidationError, PipelineStep, StrategyConfig};
use crate::context::{StrategyContext, TraceEntry};
use crate::registry::BlockRegistry;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CompositorError {
    #[error("Config validation failed: {0}")]
    ConfigValidation(#[from] ConfigValidationError),
    #[error("Block not found: {0}")]
    BlockNotFound(String),
    #[error("Block execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Invalid pipeline state: {0}")]
    InvalidState(String),
}

/// Result of compositor execution.
#[derive(Debug, Clone)]
pub struct CompositorResult {
    pub success: bool,
    pub selected: Vec<String>,
    pub weights: HashMap<String, f64>,
    pub trace: Vec<TraceEntry>,
    pub step_results: Vec<(String, BlockResult)>,
    pub message: String,
}

impl CompositorResult {
    fn success(ctx: &StrategyContext, step_results: Vec<(String, BlockResult)>) -> Self {
        Self {
            success: true,
            selected: ctx.selected.clone(),
            weights: ctx.weights.clone(),
            trace: ctx.trace.clone(),
            step_results,
            message: "Pipeline executed successfully".into(),
        }
    }

    fn failure(message: impl Into<String>, ctx: &StrategyContext) -> Self {
        Self {
            success: false,
            selected: Vec::new(),
            weights: HashMap::new(),
            trace: ctx.trace.clone(),
            step_results: Vec::new(),
            message: message.into(),
        }
    }
}

/// Compositor - executes strategy pipelines.
pub struct Compositor {
    registry: BlockRegistry,
}

impl Compositor {
    pub fn new(registry: BlockRegistry) -> Self {
        Self { registry }
    }

    /// Create compositor with built-in blocks.
    pub fn with_builtins() -> Self {
        Self::new(BlockRegistry::with_builtins())
    }

    /// Execute a strategy pipeline.
    pub fn execute(
        &self,
        config: &StrategyConfig,
        ctx: &mut StrategyContext,
    ) -> Result<CompositorResult, CompositorError> {
        // Validate config first
        validate_config(config)?;

        let mut step_results = Vec::new();

        // Execute each enabled step
        for step in config.enabled_steps() {
            let result = self.execute_step(step, ctx)?;
            
            // Apply result to context
            self.apply_result(&result, ctx, &step.step_type);
            
            step_results.push((step.block_id.clone(), result.clone()));

            if !result.success {
                return Ok(CompositorResult::failure(
                    format!("Step '{}' failed: {}", step.block_id, result.message),
                    ctx,
                ));
            }
        }

        // Validate final state
        self.validate_final_state(ctx, config)?;

        Ok(CompositorResult::success(ctx, step_results))
    }

    fn execute_step(
        &self,
        step: &PipelineStep,
        ctx: &mut StrategyContext,
    ) -> Result<BlockResult, CompositorError> {
        // Create block dynamically with params
        let block = self.registry
            .create_block(&step.step_type, &step.block_id, &step.params)
            .ok_or_else(|| CompositorError::BlockNotFound(step.block_id.clone()))?;

        // Validate params
        if let Err(e) = block.validate_params(&step.params) {
            return Ok(BlockResult::failure(format!("Invalid params: {}", e)));
        }

        // Execute block
        let result = block.execute(ctx, &step.params);
        
        Ok(result)
    }

    fn apply_result(&self, result: &BlockResult, ctx: &mut StrategyContext, step_type: &str) {
        match step_type {
            "selection" | "filter" => {
                // Update selected list
                if !result.selected.is_empty() {
                    ctx.selected = result.selected.clone();
                }
            }
            "entry" => {
                // Add signals to context
                for signal in &result.signals {
                    ctx.signals.insert(signal.symbol.clone(), signal.clone());
                }
            }
            "sizing" => {
                // Update weights
                ctx.weights = result.weights.clone();
            }
            "exit" => {
                // Exit signals handled separately
                for signal in &result.signals {
                    ctx.signals.insert(signal.symbol.clone(), signal.clone());
                }
            }
            _ => {}
        }
    }

    fn validate_final_state(
        &self,
        ctx: &StrategyContext,
        config: &StrategyConfig,
    ) -> Result<(), CompositorError> {
        // Check weights sum to ~1.0 if sizing was applied (allow for vol targeting which may sum to <1.0)
        if !ctx.weights.is_empty() {
            let total_weight: f64 = ctx.weights.values().sum();
            // Allow weights less than 1.0 (vol targeting) but not more than 1.0
            if total_weight > 1.05 {
                return Err(CompositorError::InvalidState(format!(
                    "Weights sum to {:.2}, expected <= 1.0",
                    total_weight
                )));
            }
        }

        // Check max weight constraint (with tolerance for single-asset portfolios)
        let max_weight = config.constraints.max_weight_per_asset;
        let single_asset = ctx.weights.len() == 1;
        for (symbol, weight) in &ctx.weights {
            // Single asset portfolios are allowed to have 100% weight
            if !single_asset && *weight > max_weight + 0.01 {
                return Err(CompositorError::InvalidState(format!(
                    "Weight for {} ({:.2}) exceeds max ({:.2})",
                    symbol, weight, max_weight
                )));
            }
        }

        Ok(())
    }

    /// Get registry reference.
    pub fn registry(&self) -> &BlockRegistry {
        &self.registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::load_strategy_from_str;
    use crate::context::StrategyCandidate;
    use backtester_intelligence::filters::Market;
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;

    fn make_test_context() -> StrategyContext {
        let mut ctx = StrategyContext::new(
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            Market::BR,
            dec!(100_000),
        );

        // Add some candidates
        ctx.candidates = vec![
            StrategyCandidate::new("PETR4", Market::BR)
                .with_price(dec!(38))
                .with_volatility(0.25),
            StrategyCandidate::new("VALE3", Market::BR)
                .with_price(dec!(62))
                .with_volatility(0.28),
            StrategyCandidate::new("ITUB4", Market::BR)
                .with_price(dec!(32))
                .with_volatility(0.18),
        ];

        // Set momentum returns for selection
        ctx.candidates[0].momentum_return = Some(0.15);
        ctx.candidates[1].momentum_return = Some(0.10);
        ctx.candidates[2].momentum_return = Some(0.08);

        ctx
    }

    #[test]
    fn test_compositor_simple_pipeline() {
        let compositor = Compositor::with_builtins();
        
        let toml_str = r#"
[strategy]
id = "test"

[[pipeline]]
type = "selection"
block_id = "momentum"
params = { top_pct = 50 }

[[pipeline]]
type = "sizing"
block_id = "equal_weight"
params = { max_weight = 0.50 }

[rebalance]
frequency = "weekly"

[constraints]
max_weight_per_asset = 0.50
"#;

        let config = load_strategy_from_str(toml_str).unwrap();
        let mut ctx = make_test_context();

        let result = compositor.execute(&config, &mut ctx).unwrap();
        
        assert!(result.success);
        assert!(!result.selected.is_empty());
        assert!(!result.weights.is_empty());
    }
}

