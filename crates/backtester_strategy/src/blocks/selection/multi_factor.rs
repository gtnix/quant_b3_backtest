//! Multi-Factor Selection block.
//!
//! # Mathematical Foundation
//!
//! **Factor Model (Fama-French, 1993):**
//! R = α + β_mkt × MKT + β_smb × SMB + β_hml × HML + ε
//!
//! **Common Factors:**
//! - Value: Low P/E, Low P/B
//! - Quality: High ROE, Low Debt
//! - Momentum: 12-1 month returns
//! - Low Volatility: Lower historical volatility
//! - Size: Market capitalization
//!
//! **Factor Scoring:**
//! Combined_Score = Σ(weight_i × Z_score_i)
//!
//! # References
//! - Fama, E. & French, K. (1993). "Common Risk Factors in Stock Returns"
//! - Asness, C. et al. (2013). "Value and Momentum Everywhere"

use crate::blocks::{
    get_f64, get_usize, BlockParams, BlockResult, BlockType,
    StrategyBlock, ValidationError,
};
use crate::context::StrategyContext;
use std::collections::HashMap;

/// Multi-Factor Selection block.
pub struct MultiFactorBlock;

impl MultiFactorBlock {
    pub fn new() -> Self {
        Self
    }

    /// Calculate Z-score for ranking.
    pub fn calculate_zscore(values: &[f64]) -> Vec<f64> {
        if values.is_empty() {
            return vec![];
        }

        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std = variance.sqrt();

        if std == 0.0 {
            return vec![0.0; values.len()];
        }

        values.iter().map(|v| (v - mean) / std).collect()
    }

    /// Calculate momentum factor.
    pub fn momentum_factor(prices: &[f64], period: usize) -> Option<f64> {
        if prices.len() < period + 1 {
            return None;
        }
        let n = prices.len();
        let current = prices[n - 1];
        let past = prices[n - period - 1];
        if past > 0.0 {
            Some(current / past - 1.0)
        } else {
            None
        }
    }

    /// Calculate volatility factor (annualized).
    pub fn volatility_factor(prices: &[f64], period: usize) -> Option<f64> {
        if prices.len() < period + 1 {
            return None;
        }

        let n = prices.len();
        let mut returns = Vec::with_capacity(period);

        for i in (n - period)..n {
            let ret = (prices[i] / prices[i - 1]).ln();
            returns.push(ret);
        }

        let mean: f64 = returns.iter().sum::<f64>() / period as f64;
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (period - 1) as f64;
        
        Some(variance.sqrt() * (252.0_f64).sqrt())
    }
}

impl Default for MultiFactorBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for MultiFactorBlock {
    fn block_id(&self) -> &'static str {
        "multi_factor"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Selection
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let momentum_weight = get_f64(params, "momentum_weight", 0.4);
        let volatility_weight = get_f64(params, "volatility_weight", 0.3);
        let value_weight = get_f64(params, "value_weight", 0.3);
        let momentum_period = get_usize(params, "momentum_period", 126);
        let vol_period = get_usize(params, "vol_period", 60);
        let top_pct = get_f64(params, "top_pct", 20.0);

        // Collect factor values
        let mut factor_data: Vec<(String, Option<f64>, Option<f64>, Option<f64>)> = Vec::new();

        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            let momentum = Self::momentum_factor(prices, momentum_period);
            let volatility = Self::volatility_factor(prices, vol_period);
            // Value uses P/E from candidate (if available)
            let value = candidate.price_earnings;

            factor_data.push((candidate.symbol.clone(), momentum, volatility, value));
        }

        // Calculate Z-scores for each factor
        let momentums: Vec<f64> = factor_data.iter()
            .filter_map(|(_, m, _, _)| *m)
            .collect();
        let momentum_zs = Self::calculate_zscore(&momentums);

        let volatilities: Vec<f64> = factor_data.iter()
            .filter_map(|(_, _, v, _)| *v)
            .collect();
        let vol_zs = Self::calculate_zscore(&volatilities);

        // Calculate combined scores
        let mut scores: Vec<(String, f64)> = Vec::new();
        let mut mom_idx = 0;
        let mut vol_idx = 0;

        for (symbol, momentum, volatility, value) in &factor_data {
            let mut score = 0.0;
            let mut weight_sum = 0.0;

            if momentum.is_some() && mom_idx < momentum_zs.len() {
                score += momentum_weight * momentum_zs[mom_idx];
                weight_sum += momentum_weight;
                mom_idx += 1;
            }

            if volatility.is_some() && vol_idx < vol_zs.len() {
                // Low volatility is good, so negate
                score -= volatility_weight * vol_zs[vol_idx];
                weight_sum += volatility_weight;
                vol_idx += 1;
            }

            if let Some(pe) = value {
                // Low P/E is good for value (simple rank proxy)
                if *pe > 0.0 && *pe < 100.0 {
                    score -= value_weight * (*pe / 20.0); // Normalize
                    weight_sum += value_weight;
                }
            }

            if weight_sum > 0.0 {
                scores.push((symbol.clone(), score / weight_sum));
            }
        }

        // Sort by combined score (descending) and select top %
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let select_count = ((scores.len() as f64 * top_pct / 100.0).ceil() as usize).max(1);
        
        let selected: Vec<String> = scores
            .iter()
            .take(select_count)
            .map(|(sym, _)| sym.clone())
            .collect();

        let excluded: Vec<(String, String)> = scores
            .iter()
            .skip(select_count)
            .map(|(sym, score)| (sym.clone(), format!("score={:.2}", score)))
            .collect();

        ctx.trace_step(
            self.block_id(),
            &format!(
                "MultiFactor: selected {} of {} (top {:.0}% by factor score)",
                selected.len(), ctx.candidates.len(), top_pct
            ),
        );

        BlockResult::success(format!(
            "Multi-Factor: {} selected by combined score",
            selected.len()
        ))
        .with_selected(selected)
        .with_excluded(excluded)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let top_pct = get_f64(params, "top_pct", 20.0);

        if top_pct <= 0.0 || top_pct > 100.0 {
            return Err(ValidationError::OutOfRange(
                "top_pct".into(),
                "must be between 0 and 100".into(),
            ));
        }

        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("momentum_weight".into(), toml::Value::Float(0.4));
        params.insert("volatility_weight".into(), toml::Value::Float(0.3));
        params.insert("value_weight".into(), toml::Value::Float(0.3));
        params.insert("momentum_period".into(), toml::Value::Integer(126));
        params.insert("vol_period".into(), toml::Value::Integer(60));
        params.insert("top_pct".into(), toml::Value::Float(20.0));
        params
    }

    fn description(&self) -> &'static str {
        "Multi-Factor: Combine momentum, value, and low-vol factors"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zscore() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let zscores = MultiFactorBlock::calculate_zscore(&values);
        
        assert_eq!(zscores.len(), 5);
        // Mean should have Z=0
        assert!((zscores[2]).abs() < 0.01);
        // First should be negative, last positive
        assert!(zscores[0] < 0.0);
        assert!(zscores[4] > 0.0);
    }

    #[test]
    fn test_momentum_factor() {
        let prices: Vec<f64> = (0..130).map(|i| 100.0 + i as f64).collect();
        let momentum = MultiFactorBlock::momentum_factor(&prices, 126);
        assert!(momentum.is_some());
        assert!(momentum.unwrap() > 1.0); // 126% gain
    }
}
