//! Risk parity sizing block.
//!
//! Weights inversely proportional to volatility.

use crate::blocks::{
    get_f64, get_usize, BlockParams, BlockResult, BlockType, StrategyBlock, ValidationError,
};
use crate::context::StrategyContext;
use std::collections::HashMap;

/// Risk parity sizing block.
pub struct RiskParityBlock;

impl RiskParityBlock {
    pub fn new() -> Self {
        Self
    }
}

impl Default for RiskParityBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for RiskParityBlock {
    fn block_id(&self) -> &'static str {
        "risk_parity"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Sizing
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let max_weight = get_f64(params, "max_weight", 0.20);
        let min_weight = get_f64(params, "min_weight", 0.02);
        let fallback_vol = get_f64(params, "fallback_vol", 0.25);
        let max_positions = get_usize(params, "max_positions", 20);

        // Use selected symbols, or all candidates if none selected
        let symbols: Vec<String> = if !ctx.selected.is_empty() {
            ctx.selected.clone()
        } else {
            ctx.candidates.iter().map(|c| c.symbol.clone()).collect()
        };

        // Limit to max_positions
        let symbols: Vec<String> = symbols.into_iter().take(max_positions).collect();

        if symbols.is_empty() {
            ctx.trace_step(self.block_id(), "No symbols to weight");
            return BlockResult::success("Risk parity: no symbols").with_weights(HashMap::new());
        }

        // Calculate inverse volatility weights
        let mut inverse_vols: Vec<(String, f64)> = Vec::new();
        for symbol in &symbols {
            let vol = ctx
                .candidates
                .iter()
                .find(|c| &c.symbol == symbol)
                .and_then(|c| c.volatility)
                .unwrap_or(fallback_vol)
                .max(0.01); // Minimum 1% vol to avoid division issues

            inverse_vols.push((symbol.clone(), 1.0 / vol));
        }

        let total_inverse: f64 = inverse_vols.iter().map(|(_, inv)| inv).sum();

        if total_inverse <= 0.0 {
            ctx.trace_step(self.block_id(), "Zero total inverse vol, using equal weight");
            // Fallback to equal weight
            let n = symbols.len() as f64;
            let weights: HashMap<String, f64> = symbols.iter().map(|s| (s.clone(), 1.0 / n)).collect();
            return BlockResult::success("Risk parity: fallback to equal").with_weights(weights);
        }

        // Calculate raw weights
        let mut weights: HashMap<String, f64> = inverse_vols
            .iter()
            .map(|(sym, inv)| (sym.clone(), inv / total_inverse))
            .collect();

        // Apply caps
        for weight in weights.values_mut() {
            if *weight > max_weight {
                *weight = max_weight;
            } else if *weight < min_weight {
                *weight = min_weight;
            }
        }

        // Re-normalize
        let total: f64 = weights.values().sum();
        if total > 0.0 && (total - 1.0).abs() > 0.001 {
            for weight in weights.values_mut() {
                *weight /= total;
            }
        }

        let avg_weight = 100.0 / weights.len() as f64;
        ctx.trace_step(
            self.block_id(),
            &format!(
                "{} symbols, avg {:.1}% weight",
                weights.len(),
                avg_weight
            ),
        );

        BlockResult::success(format!("Risk parity: {} symbols", weights.len())).with_weights(weights)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let max_weight = get_f64(params, "max_weight", 0.20);
        let min_weight = get_f64(params, "min_weight", 0.02);
        let fallback_vol = get_f64(params, "fallback_vol", 0.25);

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

        if fallback_vol <= 0.0 {
            return Err(ValidationError::OutOfRange(
                "fallback_vol".into(),
                "must be positive".into(),
            ));
        }

        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("max_weight".into(), toml::Value::Float(0.20));
        params.insert("min_weight".into(), toml::Value::Float(0.02));
        params.insert("fallback_vol".into(), toml::Value::Float(0.25));
        params.insert("max_positions".into(), toml::Value::Integer(20));
        params
    }

    fn description(&self) -> &'static str {
        "Risk parity: Inverse volatility weighting"
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
    fn test_risk_parity_lower_vol_higher_weight() {
        let block = RiskParityBlock::new();
        let mut ctx = StrategyContext::new(
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            Market::BR,
            dec!(100_000),
        );

        ctx.candidates = vec![
            StrategyCandidate::new("LOW_VOL", Market::BR).with_volatility(0.10),
            StrategyCandidate::new("HIGH_VOL", Market::BR).with_volatility(0.30),
        ];
        ctx.selected = vec!["LOW_VOL".into(), "HIGH_VOL".into()];

        let mut params = HashMap::new();
        params.insert("max_weight".into(), toml::Value::Float(0.90));
        params.insert("min_weight".into(), toml::Value::Float(0.01));

        let result = block.execute(&mut ctx, &params);

        assert!(result.success);
        
        let low_vol_weight = result.weights.get("LOW_VOL").unwrap();
        let high_vol_weight = result.weights.get("HIGH_VOL").unwrap();
        
        // Low vol should have ~3x the weight of high vol (1/0.10 : 1/0.30 = 3:1)
        assert!(
            low_vol_weight > high_vol_weight,
            "Low vol ({}) should have higher weight than high vol ({})",
            low_vol_weight,
            high_vol_weight
        );
    }
}

