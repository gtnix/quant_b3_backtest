//! Volatility targeting sizing block (Technique 30/Combo C).
//!
//! Scales position sizes to achieve target portfolio volatility.

use crate::blocks::{
    get_f64, get_usize, BlockParams, BlockResult, BlockType, StrategyBlock, ValidationError,
};
use crate::context::StrategyContext;
use std::collections::HashMap;

/// Volatility targeting sizing block.
pub struct VolTargetingBlock;

impl VolTargetingBlock {
    pub fn new() -> Self {
        Self
    }

    /// Calculate portfolio volatility given weights and individual vols.
    /// Simplified: assumes correlation of 0.5 between all assets.
    fn estimate_portfolio_vol(weights: &HashMap<String, f64>, vols: &HashMap<String, f64>, correlation: f64) -> f64 {
        let mut variance = 0.0;

        // Sum of weighted variances
        for (sym, w) in weights {
            if let Some(vol) = vols.get(sym) {
                variance += w * w * vol * vol;
            }
        }

        // Cross-correlations (simplified)
        let symbols: Vec<&String> = weights.keys().collect();
        for i in 0..symbols.len() {
            for j in (i + 1)..symbols.len() {
                let w_i = weights.get(symbols[i]).unwrap_or(&0.0);
                let w_j = weights.get(symbols[j]).unwrap_or(&0.0);
                let vol_i = vols.get(symbols[i]).unwrap_or(&0.25);
                let vol_j = vols.get(symbols[j]).unwrap_or(&0.25);
                
                variance += 2.0 * w_i * w_j * vol_i * vol_j * correlation;
            }
        }

        variance.sqrt()
    }
}

impl Default for VolTargetingBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for VolTargetingBlock {
    fn block_id(&self) -> &'static str {
        "vol_targeting"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Sizing
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let target_vol = get_f64(params, "target_vol", 0.12); // 12% annualized
        let max_weight = get_f64(params, "max_weight", 0.30);
        let min_weight = get_f64(params, "min_weight", 0.02);
        let max_leverage = get_f64(params, "max_leverage", 1.0); // No leverage by default
        let correlation = get_f64(params, "correlation", 0.5);
        let fallback_vol = get_f64(params, "fallback_vol", 0.25);
        let max_positions = get_usize(params, "max_positions", 20);

        // Use selected symbols, or all candidates if none selected
        let symbols: Vec<String> = if !ctx.selected.is_empty() {
            ctx.selected.clone()
        } else {
            ctx.candidates.iter().map(|c| c.symbol.clone()).collect()
        };

        let symbols: Vec<String> = symbols.into_iter().take(max_positions).collect();

        if symbols.is_empty() {
            ctx.trace_step(self.block_id(), "No symbols to weight");
            return BlockResult::success("Vol targeting: no symbols").with_weights(HashMap::new());
        }

        // Get volatilities
        let mut vols: HashMap<String, f64> = HashMap::new();
        for symbol in &symbols {
            let vol = ctx
                .candidates
                .iter()
                .find(|c| &c.symbol == symbol)
                .and_then(|c| c.volatility)
                .unwrap_or(fallback_vol);
            vols.insert(symbol.clone(), vol);
        }

        // Start with equal weights
        let n = symbols.len() as f64;
        let mut weights: HashMap<String, f64> = symbols
            .iter()
            .map(|s| (s.clone(), 1.0 / n))
            .collect();

        // Calculate current portfolio volatility
        let current_vol = Self::estimate_portfolio_vol(&weights, &vols, correlation);

        if current_vol <= 0.0001 {
            ctx.trace_step(self.block_id(), "Zero volatility, using equal weights");
            return BlockResult::success("Vol targeting: fallback to equal").with_weights(weights);
        }

        // Scale factor to achieve target vol
        let scale = (target_vol / current_vol).min(max_leverage);

        // Apply scale and caps
        for weight in weights.values_mut() {
            *weight *= scale;
            *weight = weight.clamp(min_weight, max_weight);
        }

        // Normalize to sum to at most 1.0 (or leverage limit)
        let total: f64 = weights.values().sum();
        let max_total = max_leverage;
        
        if total > max_total {
            let normalize_factor = max_total / total;
            for weight in weights.values_mut() {
                *weight *= normalize_factor;
            }
        }

        let final_total: f64 = weights.values().sum();
        let estimated_vol = Self::estimate_portfolio_vol(&weights, &vols, correlation);

        ctx.trace_step(
            self.block_id(),
            &format!(
                "{} symbols, target {}%, estimated {}%, total weight {:.1}%",
                weights.len(),
                (target_vol * 100.0) as i32,
                (estimated_vol * 100.0) as i32,
                final_total * 100.0
            ),
        );

        BlockResult::success(format!(
            "Vol targeting: {} symbols, est vol {:.1}%",
            weights.len(),
            estimated_vol * 100.0
        ))
        .with_weights(weights)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let target_vol = get_f64(params, "target_vol", 0.12);
        let max_leverage = get_f64(params, "max_leverage", 1.0);

        if target_vol <= 0.0 || target_vol > 1.0 {
            return Err(ValidationError::OutOfRange(
                "target_vol".into(),
                "must be between 0 and 1".into(),
            ));
        }

        if max_leverage <= 0.0 || max_leverage > 3.0 {
            return Err(ValidationError::OutOfRange(
                "max_leverage".into(),
                "must be between 0 and 3".into(),
            ));
        }

        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("target_vol".into(), toml::Value::Float(0.12));
        params.insert("max_weight".into(), toml::Value::Float(0.30));
        params.insert("min_weight".into(), toml::Value::Float(0.02));
        params.insert("max_leverage".into(), toml::Value::Float(1.0));
        params.insert("correlation".into(), toml::Value::Float(0.5));
        params.insert("fallback_vol".into(), toml::Value::Float(0.25));
        params.insert("max_positions".into(), toml::Value::Integer(20));
        params
    }

    fn description(&self) -> &'static str {
        "Vol targeting: Scale positions to achieve target portfolio volatility"
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
    fn test_vol_targeting_scales_down_high_vol() {
        let block = VolTargetingBlock::new();
        let mut ctx = StrategyContext::new(
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            Market::BR,
            dec!(100_000),
        );

        // High vol portfolio
        ctx.candidates = vec![
            StrategyCandidate::new("A", Market::BR).with_volatility(0.40),
            StrategyCandidate::new("B", Market::BR).with_volatility(0.35),
        ];
        ctx.selected = vec!["A".into(), "B".into()];

        let mut params = HashMap::new();
        params.insert("target_vol".into(), toml::Value::Float(0.12));
        params.insert("max_weight".into(), toml::Value::Float(0.50));
        params.insert("min_weight".into(), toml::Value::Float(0.01));

        let result = block.execute(&mut ctx, &params);

        assert!(result.success);
        
        // Total weight should be less than 1.0 (scaled down)
        let total: f64 = result.weights.values().sum();
        assert!(total <= 1.0, "Total {} should be <= 1.0", total);
    }

    #[test]
    fn test_vol_targeting_respects_max_leverage() {
        let block = VolTargetingBlock::new();
        let mut ctx = StrategyContext::new(
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            Market::BR,
            dec!(100_000),
        );

        // Very low vol portfolio - would want to lever up
        ctx.candidates = vec![
            StrategyCandidate::new("A", Market::BR).with_volatility(0.05),
            StrategyCandidate::new("B", Market::BR).with_volatility(0.05),
        ];
        ctx.selected = vec!["A".into(), "B".into()];

        let mut params = HashMap::new();
        params.insert("target_vol".into(), toml::Value::Float(0.20));
        params.insert("max_leverage".into(), toml::Value::Float(1.0)); // No leverage
        params.insert("max_weight".into(), toml::Value::Float(0.80));

        let result = block.execute(&mut ctx, &params);

        assert!(result.success);
        
        // Total weight should not exceed 1.0
        let total: f64 = result.weights.values().sum();
        assert!(total <= 1.01, "Total {} should be <= 1.0", total);
    }
}

