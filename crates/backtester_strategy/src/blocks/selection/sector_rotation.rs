//! Sector Rotation selection blocks.
//!
//! # Mathematical Foundation
//!
//! **Business Cycle Rotation:**
//! Economic cycles favor different sectors at different phases:
//! - Early Expansion: Consumer Discretionary, Financials
//! - Mid Expansion: Technology, Industrials
//! - Late Expansion: Energy, Materials
//! - Contraction: Utilities, Consumer Staples, Healthcare
//!
//! **Relative Strength Rotation:**
//! RS_sector = Return_sector / Return_benchmark
//! Select top N sectors by relative strength
//!
//! # References
//! - Stovall, S. (1996). "Sector Investing"
//! - Faber, M. (2010). "Relative Strength Strategies for Investing"

use crate::blocks::{
    get_f64, get_usize, BlockParams, BlockResult, BlockType,
    StrategyBlock, ValidationError,
};
use crate::context::StrategyContext;
use std::collections::HashMap;

/// Sector Rotation by Relative Strength selection block.
pub struct SectorRotationBlock;

impl SectorRotationBlock {
    pub fn new() -> Self {
        Self
    }

    /// Calculate momentum/relative strength for each candidate.
    pub fn calculate_momentum(prices: &[f64], period: usize) -> Option<f64> {
        if prices.len() < period + 1 {
            return None;
        }
        
        let n = prices.len();
        let current = prices[n - 1];
        let past = prices[n - period - 1];
        
        if past > 0.0 {
            Some((current / past) - 1.0)
        } else {
            None
        }
    }
}

impl Default for SectorRotationBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for SectorRotationBlock {
    fn block_id(&self) -> &'static str {
        "sector_rotation"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Selection
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let lookback = get_usize(params, "lookback", 63);
        let top_n = get_usize(params, "top_n", 3);
        let min_momentum = get_f64(params, "min_momentum", -0.10);

        // Calculate momentum for each candidate
        let mut momentum_scores: Vec<(String, f64)> = Vec::new();

        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            
            if let Some(momentum) = Self::calculate_momentum(prices, lookback) {
                if momentum >= min_momentum {
                    momentum_scores.push((candidate.symbol.clone(), momentum));
                }
            }
        }

        // Sort by momentum (descending) and take top N
        momentum_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let selected: Vec<String> = momentum_scores
            .iter()
            .take(top_n)
            .map(|(sym, _)| sym.clone())
            .collect();

        let excluded: Vec<(String, String)> = momentum_scores
            .iter()
            .skip(top_n)
            .map(|(sym, mom)| (sym.clone(), format!("momentum={:.2}%", mom * 100.0)))
            .collect();

        ctx.trace_step(
            self.block_id(),
            &format!(
                "SectorRotation: selected {} of {} (top {} by {}d momentum)",
                selected.len(), ctx.candidates.len(), top_n, lookback
            ),
        );

        BlockResult::success(format!(
            "Sector Rotation: {} selected by relative strength",
            selected.len()
        ))
        .with_selected(selected)
        .with_excluded(excluded)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let lookback = get_usize(params, "lookback", 63);
        let top_n = get_usize(params, "top_n", 3);

        if lookback < 20 || lookback > 252 {
            return Err(ValidationError::OutOfRange(
                "lookback".into(),
                "must be between 20 and 252".into(),
            ));
        }

        if top_n < 1 || top_n > 20 {
            return Err(ValidationError::OutOfRange(
                "top_n".into(),
                "must be between 1 and 20".into(),
            ));
        }

        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("lookback".into(), toml::Value::Integer(63));
        params.insert("top_n".into(), toml::Value::Integer(3));
        params.insert("min_momentum".into(), toml::Value::Float(-0.10));
        params
    }

    fn description(&self) -> &'static str {
        "Sector Rotation: Select top N assets by relative strength"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_momentum_calculation() {
        let prices = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0];
        let momentum = SectorRotationBlock::calculate_momentum(&prices, 5);
        assert!(momentum.is_some());
        assert!((momentum.unwrap() - 0.10).abs() < 0.01); // 10% gain
    }
}
