//! Advanced Selection Blocks - Prop Trading Level
//!
//! High-quality selection filters based on:
//! - Liquidity filtering and ranking
//! - Volatility regime detection

use crate::blocks::{
    get_f64, get_i64, BlockParams, BlockResult, BlockType, StrategyBlock, ValidationError,
};
use crate::context::StrategyContext;
use std::collections::HashMap;

// =============================================================================
// LIQUIDITY FILTER + RANK
// Filter by volume/turnover, then rank by momentum/vol
// =============================================================================

/// Filter assets by liquidity, then rank by secondary factor.
/// Ensures tradability for institutional/prop strategies.
pub struct LiquidityFilterBlock;

impl LiquidityFilterBlock {
    pub fn new() -> Self { Self }
}

impl Default for LiquidityFilterBlock {
    fn default() -> Self { Self::new() }
}

impl StrategyBlock for LiquidityFilterBlock {
    fn block_id(&self) -> &'static str { "liquidity_filter" }
    fn block_type(&self) -> BlockType { BlockType::Selection }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let min_adv_pct = get_f64(params, "min_adv_percentile", 30.0); // Top 70% by volume
        let rank_by = get_f64(params, "rank_by", 0.0); // 0=momentum, 1=low_vol
        let top_pct = get_f64(params, "top_pct", 20.0);
        let max_positions = get_i64(params, "max_positions", 20) as usize;

        // Collect volumes
        let mut with_volume: Vec<(&str, f64, Option<f64>, Option<f64>)> = ctx.candidates.iter()
            .filter_map(|c| {
                let vol = c.avg_volume.and_then(|v| v.to_f64())?;
                Some((c.symbol.as_str(), vol, c.momentum_return, c.volatility))
            })
            .collect();

        if with_volume.is_empty() {
            return BlockResult::success("Liquidity Filter: no data".to_string());
        }

        // Sort by volume descending
        with_volume.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Filter by minimum percentile
        let cutoff_idx = ((100.0 - min_adv_pct) / 100.0 * with_volume.len() as f64).ceil() as usize;
        let liquid: Vec<_> = with_volume.into_iter().take(cutoff_idx).collect();

        // Rank by secondary factor
        let mut ranked: Vec<(&str, f64)> = liquid.iter()
            .filter_map(|(sym, _vol, mom, volatility)| {
                let score = if rank_by < 0.5 {
                    // Momentum ranking (higher is better)
                    mom.unwrap_or(0.0)
                } else {
                    // Low vol ranking (lower is better, so negate)
                    -volatility.unwrap_or(0.5)
                };
                Some((*sym, score))
            })
            .collect();

        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select top N%
        let select_count = ((top_pct / 100.0) * ranked.len() as f64).ceil() as usize;
        let select_count = select_count.min(max_positions).max(1);

        let selected: Vec<String> = ranked.iter()
            .take(select_count)
            .map(|(s, _)| s.to_string())
            .collect();

        for sym in &selected {
            if !ctx.selected.contains(sym) {
                ctx.selected.push(sym.clone());
            }
        }

        ctx.trace_step(self.block_id(), &format!(
            "Liquidity Filter: {} liquid, {} selected",
            liquid.len(), selected.len()
        ));

        BlockResult::success(format!("Liquidity Filter: {} selected", selected.len()))
            .with_selected(selected)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let pct = get_f64(params, "min_adv_percentile", 30.0);
        if pct < 0.0 || pct > 90.0 {
            return Err(ValidationError::OutOfRange("min_adv_percentile".into(), "0-90".into()));
        }
        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut p = HashMap::new();
        p.insert("min_adv_percentile".into(), toml::Value::Float(30.0));
        p.insert("rank_by".into(), toml::Value::Float(0.0)); // momentum
        p.insert("top_pct".into(), toml::Value::Float(20.0));
        p.insert("max_positions".into(), toml::Value::Integer(20));
        p
    }

    fn description(&self) -> &'static str {
        "Liquidity Filter: Filter by volume, rank by momentum/vol"
    }
}

// =============================================================================
// REGIME FILTER
// Activate/deactivate selection based on volatility regime
// =============================================================================

/// Filter that activates/deactivates based on market regime.
/// Uses rolling volatility to detect risk-on/risk-off conditions.
pub struct RegimeFilterBlock;

impl RegimeFilterBlock {
    pub fn new() -> Self { Self }
}

impl Default for RegimeFilterBlock {
    fn default() -> Self { Self::new() }
}

impl StrategyBlock for RegimeFilterBlock {
    fn block_id(&self) -> &'static str { "regime_filter" }
    fn block_type(&self) -> BlockType { BlockType::Selection }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let vol_lookback = get_i64(params, "vol_lookback", 20) as usize;
        let vol_threshold = get_f64(params, "vol_threshold", 1.5); // z-score
        let risk_off_action = get_f64(params, "risk_off_action", 0.0); // 0=pass none, 1=pass all
        let top_pct = get_f64(params, "top_pct", 30.0);

        // Calculate market-wide volatility from candidates
        let mut all_vols: Vec<f64> = Vec::new();
        let mut recent_vols: Vec<f64> = Vec::new();

        for candidate in &ctx.candidates {
            if candidate.prices.len() < vol_lookback + 20 {
                continue;
            }

            // Recent volatility (last 5 days)
            let returns: Vec<f64> = candidate.prices.windows(2)
                .map(|w| (w[1] / w[0]) - 1.0)
                .collect();
            
            let recent = returns.iter().rev().take(5)
                .map(|r| r.powi(2)).sum::<f64>().sqrt();
            recent_vols.push(recent);

            // Historical volatility
            if returns.len() >= vol_lookback {
                let hist = returns.iter().rev().skip(5).take(vol_lookback)
                    .map(|r| r.powi(2)).sum::<f64>().sqrt();
                all_vols.push(hist);
            }
        }

        // Determine regime
        let is_risk_off = if !all_vols.is_empty() && !recent_vols.is_empty() {
            let mean_hist = all_vols.iter().sum::<f64>() / all_vols.len() as f64;
            let mean_recent = recent_vols.iter().sum::<f64>() / recent_vols.len() as f64;
            let std_hist = (all_vols.iter().map(|v| (v - mean_hist).powi(2)).sum::<f64>() 
                / all_vols.len() as f64).sqrt().max(0.001);
            
            let z = (mean_recent - mean_hist) / std_hist;
            z > vol_threshold
        } else {
            false
        };

        // Apply regime logic
        if is_risk_off && risk_off_action < 0.5 {
            // Risk-off: select nothing
            ctx.trace_step(self.block_id(), "Regime Filter: RISK-OFF - no selection");
            return BlockResult::success("Regime Filter: risk-off".to_string())
                .with_selected(Vec::new());
        }

        // Risk-on or pass-through: select by momentum
        let mut scored: Vec<(&str, f64)> = ctx.candidates.iter()
            .filter_map(|c| {
                let score = c.momentum_return.unwrap_or(0.0);
                Some((c.symbol.as_str(), score))
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let select_count = ((top_pct / 100.0) * scored.len() as f64).ceil() as usize;
        let selected: Vec<String> = scored.iter()
            .take(select_count.max(1))
            .map(|(s, _)| s.to_string())
            .collect();

        for sym in &selected {
            if !ctx.selected.contains(sym) {
                ctx.selected.push(sym.clone());
            }
        }

        ctx.trace_step(self.block_id(), &format!(
            "Regime Filter: RISK-ON - {} selected",
            selected.len()
        ));

        BlockResult::success(format!("Regime Filter: {} selected", selected.len()))
            .with_selected(selected)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let threshold = get_f64(params, "vol_threshold", 1.5);
        if threshold < 0.5 || threshold > 3.0 {
            return Err(ValidationError::OutOfRange("vol_threshold".into(), "0.5-3.0".into()));
        }
        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut p = HashMap::new();
        p.insert("vol_lookback".into(), toml::Value::Integer(20));
        p.insert("vol_threshold".into(), toml::Value::Float(1.5));
        p.insert("risk_off_action".into(), toml::Value::Float(0.0));
        p.insert("top_pct".into(), toml::Value::Float(30.0));
        p
    }

    fn description(&self) -> &'static str {
        "Regime Filter: Adapt selection to volatility regime"
    }
}

// Need this for Decimal conversion
use rust_decimal::prelude::ToPrimitive;
