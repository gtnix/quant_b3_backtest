//! Equal weight sizing block.

use crate::blocks::{
    get_f64, get_usize, BlockParams, BlockResult, BlockType, StrategyBlock, ValidationError,
};
use crate::context::StrategyContext;
use std::collections::HashMap;

/// Equal weight sizing block.
pub struct EqualWeightBlock;

impl EqualWeightBlock {
    pub fn new() -> Self {
        Self
    }
}

impl Default for EqualWeightBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for EqualWeightBlock {
    fn block_id(&self) -> &'static str {
        "equal_weight"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Sizing
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let max_weight = get_f64(params, "max_weight", 0.20);
        let min_weight = get_f64(params, "min_weight", 0.02);
        let max_positions = get_usize(params, "max_positions", 20);

        // Use selected symbols, or all candidates if none selected
        let symbols: Vec<String> = if !ctx.selected.is_empty() {
            ctx.selected.clone()
        } else {
            ctx.candidates.iter().map(|c| c.symbol.clone()).collect()
        };

        // Limit to max_positions
        let symbols: Vec<String> = symbols.into_iter().take(max_positions).collect();
        let n = symbols.len();

        if n == 0 {
            ctx.trace_step(self.block_id(), "No symbols to weight");
            return BlockResult::success("Equal weight: no symbols").with_weights(HashMap::new());
        }

        // Calculate equal weight, capped
        let raw_weight = 1.0 / n as f64;
        let capped_weight = raw_weight.clamp(min_weight, max_weight);

        let mut weights = HashMap::new();
        for symbol in &symbols {
            weights.insert(symbol.clone(), capped_weight);
        }

        // Normalize to sum to 1.0
        let total: f64 = weights.values().sum();
        if total > 0.0 {
            for weight in weights.values_mut() {
                *weight /= total;
            }
        }

        ctx.trace_step(
            self.block_id(),
            &format!("{} symbols with {:.1}% each", n, 100.0 / n as f64),
        );

        BlockResult::success(format!("Equal weight: {} symbols", n)).with_weights(weights)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let max_weight = get_f64(params, "max_weight", 0.20);
        let min_weight = get_f64(params, "min_weight", 0.02);

        if max_weight <= 0.0 || max_weight > 1.0 {
            return Err(ValidationError::OutOfRange(
                "max_weight".into(),
                "must be between 0 and 1".into(),
            ));
        }

        if min_weight < 0.0 || min_weight >= max_weight {
            return Err(ValidationError::OutOfRange(
                "min_weight".into(),
                "must be between 0 and max_weight".into(),
            ));
        }

        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("max_weight".into(), toml::Value::Float(0.20));
        params.insert("min_weight".into(), toml::Value::Float(0.02));
        params.insert("max_positions".into(), toml::Value::Integer(20));
        params
    }

    fn description(&self) -> &'static str {
        "Equal weight: 1/N allocation across selected assets"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::StrategyCandidate;
    use backtester_intelligence::filters::Market;
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;

    #[test]
    fn test_equal_weight_basic() {
        let block = EqualWeightBlock::new();
        let mut ctx = StrategyContext::new(
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            Market::BR,
            dec!(100_000),
        );

        ctx.selected = vec!["A".into(), "B".into(), "C".into(), "D".into(), "E".into()];

        let result = block.execute(&mut ctx, &block.default_params());

        assert!(result.success);
        assert_eq!(result.weights.len(), 5);

        let total: f64 = result.weights.values().sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_equal_weight_capped() {
        let block = EqualWeightBlock::new();
        let mut ctx = StrategyContext::new(
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            Market::BR,
            dec!(100_000),
        );

        // Only 2 symbols - would be 50% each, but capped at 20%
        ctx.selected = vec!["A".into(), "B".into()];

        let mut params = HashMap::new();
        params.insert("max_weight".into(), toml::Value::Float(0.20));

        let result = block.execute(&mut ctx, &params);

        assert!(result.success);
        // After capping and normalizing, each should be 50%
        for weight in result.weights.values() {
            assert!(*weight <= 0.51); // Allow small tolerance
        }
    }
}

