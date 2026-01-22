//! Portfolio Optimization Selection Blocks
//!
//! Reference: Markowitz (1952), Merton (1972)

use crate::blocks::{BlockParams, BlockResult, BlockType, StrategyBlock, ValidationError};
use crate::context::StrategyContext;
use crate::blocks::get_f64;

/// Max Sharpe block.
pub struct MaxSharpeBlock;
impl MaxSharpeBlock { pub fn new() -> Self { Self } }

impl StrategyBlock for MaxSharpeBlock {
    fn block_id(&self) -> &'static str { "max_sharpe" }
    fn block_type(&self) -> BlockType { BlockType::Selection }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let lookback = get_f64(params, "lookback", 252.0) as usize;
        let rf_rate = get_f64(params, "rf_rate", 0.02);
        let mut selected = Vec::new();
        
        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            if prices.len() < lookback { continue; }
            
            let returns: Vec<f64> = prices.windows(2)
                .rev().take(lookback)
                .map(|w| (w[1] - w[0]) / w[0])
                .collect();
            
            if returns.is_empty() { continue; }
            
            let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance: f64 = returns.iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>() / returns.len() as f64;
            let std = variance.sqrt().max(0.0001);
            
            let annual_return = mean_return * 252.0;
            let annual_vol = std * 252.0_f64.sqrt();
            let sharpe = (annual_return - rf_rate) / annual_vol;
            
            if sharpe > 0.5 {
                selected.push(candidate.symbol.clone());
            }
        }
        
        BlockResult::success(format!("max_sharpe: {} selected", selected.len())).with_selected(selected)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "Maximum Sharpe ratio selection" }
}

/// Min Variance block.
pub struct MinVarianceBlock;
impl MinVarianceBlock { pub fn new() -> Self { Self } }

impl StrategyBlock for MinVarianceBlock {
    fn block_id(&self) -> &'static str { "min_variance" }
    fn block_type(&self) -> BlockType { BlockType::Selection }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let lookback = get_f64(params, "lookback", 63.0) as usize;
        let max_vol = get_f64(params, "max_vol", 0.20);
        let mut selected = Vec::new();
        
        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            if prices.len() < lookback { continue; }
            
            let returns: Vec<f64> = prices.windows(2)
                .rev().take(lookback)
                .map(|w| (w[1] - w[0]) / w[0])
                .collect();
            
            if returns.is_empty() { continue; }
            
            let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance: f64 = returns.iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>() / returns.len() as f64;
            let annual_vol = variance.sqrt() * 252.0_f64.sqrt();
            
            if annual_vol < max_vol && annual_vol > 0.0 {
                selected.push(candidate.symbol.clone());
            }
        }
        
        BlockResult::success(format!("min_variance: {} selected", selected.len())).with_selected(selected)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "Minimum variance selection" }
}
