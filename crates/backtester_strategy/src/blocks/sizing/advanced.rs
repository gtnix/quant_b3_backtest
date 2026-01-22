//! Advanced Sizing Blocks - Prop Trading Level
//!
//! High-quality position sizing based on:
//! - Kelly Criterion (fractional)
//! - Volatility targeting with cash buffer
//! - Exposure caps and controls

use crate::blocks::{
    get_f64, get_i64, BlockParams, BlockResult, BlockType, StrategyBlock, ValidationError,
};
use crate::context::StrategyContext;
use std::collections::HashMap;

// =============================================================================
// KELLY FRACTIONAL SIZING
// Reference: Kelly (1956), Thorp (2006)
// =============================================================================

/// Kelly Criterion sizing with fractional multiplier and caps.
/// Formula: f* = (p*b - q) / b, then apply fraction and caps.
/// Uses historical win rate and avg win/loss ratio.
pub struct KellyFractionalBlock;

impl KellyFractionalBlock {
    pub fn new() -> Self { Self }
}

impl Default for KellyFractionalBlock {
    fn default() -> Self { Self::new() }
}

impl StrategyBlock for KellyFractionalBlock {
    fn block_id(&self) -> &'static str { "kelly_fractional" }
    fn block_type(&self) -> BlockType { BlockType::Sizing }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let kelly_fraction = get_f64(params, "kelly_fraction", 0.25); // 25% Kelly
        let max_weight = get_f64(params, "max_weight", 0.20);
        let min_weight = get_f64(params, "min_weight", 0.02);
        let max_positions = get_i64(params, "max_positions", 10) as usize;
        let assumed_win_rate = get_f64(params, "assumed_win_rate", 0.55);
        let assumed_win_loss_ratio = get_f64(params, "assumed_win_loss_ratio", 1.5);

        // Kelly formula: f* = p - q/b = p - (1-p)/b
        // where p = win probability, q = 1-p, b = win/loss ratio
        let p = assumed_win_rate;
        let q = 1.0 - p;
        let b = assumed_win_loss_ratio;
        
        let full_kelly = (p * b - q) / b;
        let target_weight = (full_kelly * kelly_fraction).clamp(min_weight, max_weight);

        let selected: Vec<String> = ctx.selected.iter().take(max_positions).cloned().collect();
        let n = selected.len();

        if n == 0 {
            return BlockResult::success("Kelly: no positions selected".to_string());
        }

        // Distribute weights (simplified: equal Kelly-adjusted weights)
        let weight_each = (target_weight / n as f64).clamp(min_weight, max_weight);
        let total_exposure = weight_each * n as f64;

        for symbol in &selected {
            ctx.weights.insert(symbol.clone(), weight_each);
        }

        ctx.trace_step(self.block_id(), &format!(
            "Kelly {:.0}%: {} positions @ {:.1}% each, total {:.1}%",
            kelly_fraction * 100.0, n, weight_each * 100.0, total_exposure * 100.0
        ));

        BlockResult::success(format!("Kelly Fractional: {} positions sized", n))
            .with_weights(ctx.weights.clone())
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let frac = get_f64(params, "kelly_fraction", 0.25);
        if frac < 0.1 || frac > 1.0 {
            return Err(ValidationError::OutOfRange("kelly_fraction".into(), "0.1-1.0".into()));
        }
        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut p = HashMap::new();
        p.insert("kelly_fraction".into(), toml::Value::Float(0.25));
        p.insert("max_weight".into(), toml::Value::Float(0.20));
        p.insert("min_weight".into(), toml::Value::Float(0.02));
        p.insert("max_positions".into(), toml::Value::Integer(10));
        p.insert("assumed_win_rate".into(), toml::Value::Float(0.55));
        p.insert("assumed_win_loss_ratio".into(), toml::Value::Float(1.5));
        p
    }

    fn description(&self) -> &'static str {
        "Kelly Fractional: Optimal sizing with safety fraction"
    }
}

// =============================================================================
// VOLATILITY TARGET WITH CASH BUFFER
// Maintains cash reserve and targets portfolio vol
// =============================================================================

/// Vol targeting with mandatory cash buffer.
/// Total exposure = min(target_vol/realized_vol, 1 - cash_buffer).
pub struct VolTargetBufferBlock;

impl VolTargetBufferBlock {
    pub fn new() -> Self { Self }
}

impl Default for VolTargetBufferBlock {
    fn default() -> Self { Self::new() }
}

impl StrategyBlock for VolTargetBufferBlock {
    fn block_id(&self) -> &'static str { "vol_target_buffer" }
    fn block_type(&self) -> BlockType { BlockType::Sizing }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let target_vol = get_f64(params, "target_vol", 0.12);
        let cash_buffer = get_f64(params, "cash_buffer", 0.20); // 20% cash minimum
        let max_weight = get_f64(params, "max_weight", 0.15);
        let min_weight = get_f64(params, "min_weight", 0.02);
        let max_positions = get_i64(params, "max_positions", 15) as usize;
        let fallback_vol = get_f64(params, "fallback_vol", 0.25);

        let max_exposure = 1.0 - cash_buffer;
        
        let selected: Vec<String> = ctx.selected.iter().take(max_positions).cloned().collect();
        let n = selected.len();

        if n == 0 {
            return BlockResult::success("Vol Target Buffer: no positions".to_string());
        }

        // Calculate volatilities
        let mut vols: Vec<(String, f64)> = Vec::new();
        for symbol in &selected {
            let vol = ctx.candidates.iter()
                .find(|c| &c.symbol == symbol)
                .and_then(|c| c.volatility)
                .unwrap_or(fallback_vol);
            vols.push((symbol.clone(), vol));
        }

        // Portfolio vol estimate (assuming some correlation)
        let assumed_corr = 0.3;
        let avg_vol = vols.iter().map(|(_, v)| *v).sum::<f64>() / n as f64;
        let diversification = (1.0 + (n as f64 - 1.0) * assumed_corr).sqrt();
        let portfolio_vol = avg_vol / diversification * (n as f64).sqrt();

        // Scale factor to hit target vol
        let scale = (target_vol / portfolio_vol).min(1.0);
        let total_exposure = (scale * 1.0).min(max_exposure);

        // Inverse volatility weighting
        let inv_vols: Vec<f64> = vols.iter().map(|(_, v)| 1.0 / v.max(0.01)).collect();
        let sum_inv: f64 = inv_vols.iter().sum();

        for (i, (symbol, _)) in vols.iter().enumerate() {
            let raw_weight = (inv_vols[i] / sum_inv) * total_exposure;
            let clamped = raw_weight.clamp(min_weight, max_weight);
            ctx.weights.insert(symbol.clone(), clamped);
        }

        let actual_exposure: f64 = ctx.weights.values().sum();
        
        ctx.trace_step(self.block_id(), &format!(
            "Vol Target {:.0}% + {:.0}% cash buffer: exposure {:.1}%",
            target_vol * 100.0, cash_buffer * 100.0, actual_exposure * 100.0
        ));

        BlockResult::success(format!("Vol Target Buffer: {} positions, {:.1}% exposure", n, actual_exposure * 100.0))
            .with_weights(ctx.weights.clone())
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let buffer = get_f64(params, "cash_buffer", 0.20);
        if buffer < 0.0 || buffer > 0.50 {
            return Err(ValidationError::OutOfRange("cash_buffer".into(), "0-0.5".into()));
        }
        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut p = HashMap::new();
        p.insert("target_vol".into(), toml::Value::Float(0.12));
        p.insert("cash_buffer".into(), toml::Value::Float(0.20));
        p.insert("max_weight".into(), toml::Value::Float(0.15));
        p.insert("min_weight".into(), toml::Value::Float(0.02));
        p.insert("max_positions".into(), toml::Value::Integer(15));
        p.insert("fallback_vol".into(), toml::Value::Float(0.25));
        p
    }

    fn description(&self) -> &'static str {
        "Vol Target Buffer: Volatility targeting with cash reserve"
    }
}

// =============================================================================
// EXPOSURE CAP SIZING
// Hard cap on total exposure (for risk management)
// =============================================================================

/// Simple exposure cap with equal weighting.
/// Limits total exposure to max_exposure parameter.
pub struct ExposureCapBlock;

impl ExposureCapBlock {
    pub fn new() -> Self { Self }
}

impl Default for ExposureCapBlock {
    fn default() -> Self { Self::new() }
}

impl StrategyBlock for ExposureCapBlock {
    fn block_id(&self) -> &'static str { "exposure_cap" }
    fn block_type(&self) -> BlockType { BlockType::Sizing }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let max_exposure = get_f64(params, "max_exposure", 0.80); // 80% max
        let max_weight = get_f64(params, "max_weight", 0.20);
        let min_weight = get_f64(params, "min_weight", 0.02);
        let max_positions = get_i64(params, "max_positions", 15) as usize;

        let selected: Vec<String> = ctx.selected.iter().take(max_positions).cloned().collect();
        let n = selected.len();

        if n == 0 {
            return BlockResult::success("Exposure Cap: no positions".to_string());
        }

        // Equal weight within exposure cap
        let weight_each = (max_exposure / n as f64).clamp(min_weight, max_weight);
        let actual_exposure = weight_each * n as f64;

        for symbol in &selected {
            ctx.weights.insert(symbol.clone(), weight_each);
        }

        ctx.trace_step(self.block_id(), &format!(
            "Exposure Cap {:.0}%: {} positions @ {:.1}%",
            max_exposure * 100.0, n, weight_each * 100.0
        ));

        BlockResult::success(format!("Exposure Cap: {} positions, {:.1}% total", n, actual_exposure * 100.0))
            .with_weights(ctx.weights.clone())
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let exp = get_f64(params, "max_exposure", 0.80);
        if exp < 0.20 || exp > 1.5 {
            return Err(ValidationError::OutOfRange("max_exposure".into(), "0.2-1.5".into()));
        }
        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut p = HashMap::new();
        p.insert("max_exposure".into(), toml::Value::Float(0.80));
        p.insert("max_weight".into(), toml::Value::Float(0.20));
        p.insert("min_weight".into(), toml::Value::Float(0.02));
        p.insert("max_positions".into(), toml::Value::Integer(15));
        p
    }

    fn description(&self) -> &'static str {
        "Exposure Cap: Hard limit on total portfolio exposure"
    }
}
