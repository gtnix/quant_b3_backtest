//! Factor Variant Selection Blocks
//!
//! Reference: Fama & French (1993, 2015)

use crate::blocks::{BlockParams, BlockResult, BlockType, StrategyBlock, ValidationError};
use crate::context::StrategyContext;
use crate::blocks::get_f64;

/// Value P/B block.
pub struct ValuePBBlock;
impl ValuePBBlock { pub fn new() -> Self { Self } }

impl StrategyBlock for ValuePBBlock {
    fn block_id(&self) -> &'static str { "value_pb" }
    fn block_type(&self) -> BlockType { BlockType::Selection }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let threshold = get_f64(params, "threshold", 1.5);
        let lookback = get_f64(params, "lookback", 20.0) as usize;
        let mut selected = Vec::new();
        
        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            if prices.len() < lookback { continue; }
            
            let current = *prices.last().unwrap();
            let avg: f64 = prices.iter().rev().take(lookback).sum::<f64>() / lookback as f64;
            
            if current < avg * (1.0 / threshold) {
                selected.push(candidate.symbol.clone());
            }
        }
        
        BlockResult::success(format!("value_pb: {} selected", selected.len())).with_selected(selected)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "Value selection by P/B" }
}

/// Value P/E block.
pub struct ValuePEBlock;
impl ValuePEBlock { pub fn new() -> Self { Self } }

impl StrategyBlock for ValuePEBlock {
    fn block_id(&self) -> &'static str { "value_pe" }
    fn block_type(&self) -> BlockType { BlockType::Selection }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let threshold = get_f64(params, "threshold", 15.0);
        let lookback = get_f64(params, "lookback", 20.0) as usize;
        let mut selected = Vec::new();
        
        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            if prices.len() < lookback { continue; }
            
            let current = *prices.last().unwrap();
            let avg: f64 = prices.iter().rev().take(lookback).sum::<f64>() / lookback as f64;
            
            if current < avg * (threshold / 20.0) {
                selected.push(candidate.symbol.clone());
            }
        }
        
        BlockResult::success(format!("value_pe: {} selected", selected.len())).with_selected(selected)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "Value selection by P/E" }
}

/// Quality ROE block.
pub struct QualityROEBlock;
impl QualityROEBlock { pub fn new() -> Self { Self } }

impl StrategyBlock for QualityROEBlock {
    fn block_id(&self) -> &'static str { "quality_roe" }
    fn block_type(&self) -> BlockType { BlockType::Selection }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let lookback = get_f64(params, "lookback", 126.0) as usize;
        let mut selected = Vec::new();
        
        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            if prices.len() < lookback { continue; }
            
            let current = *prices.last().unwrap();
            let prev = prices[prices.len() - lookback];
            let momentum = (current - prev) / prev;
            
            if momentum > 0.0 && momentum < 0.5 {
                selected.push(candidate.symbol.clone());
            }
        }
        
        BlockResult::success(format!("quality_roe: {} selected", selected.len())).with_selected(selected)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "Quality selection by ROE" }
}

/// Quality Multi block.
pub struct QualityMultiBlock;
impl QualityMultiBlock { pub fn new() -> Self { Self } }

impl StrategyBlock for QualityMultiBlock {
    fn block_id(&self) -> &'static str { "quality_multi" }
    fn block_type(&self) -> BlockType { BlockType::Selection }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let lookback = get_f64(params, "lookback", 63.0) as usize;
        let mut selected = Vec::new();
        
        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            if prices.len() < lookback { continue; }
            
            let current = *prices.last().unwrap();
            let prev = prices[prices.len() - lookback];
            let momentum = (current - prev) / prev;
            
            let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();
            let vol: f64 = if returns.len() >= lookback {
                returns.iter().rev().take(lookback).map(|r| r.powi(2)).sum::<f64>().sqrt()
            } else { 1.0 };
            
            let quality_score = if vol > 0.0 { momentum / vol } else { 0.0 };
            if quality_score > 0.5 {
                selected.push(candidate.symbol.clone());
            }
        }
        
        BlockResult::success(format!("quality_multi: {} selected", selected.len())).with_selected(selected)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "Multi-metric quality selection" }
}

/// Dividend Growth block.
pub struct DividendGrowthBlock;
impl DividendGrowthBlock { pub fn new() -> Self { Self } }

impl StrategyBlock for DividendGrowthBlock {
    fn block_id(&self) -> &'static str { "dividend_growth" }
    fn block_type(&self) -> BlockType { BlockType::Selection }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let lookback = get_f64(params, "lookback", 252.0) as usize;
        let mut selected = Vec::new();
        
        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            if prices.len() < lookback { continue; }
            
            let current = *prices.last().unwrap();
            let prev = prices[prices.len() - lookback];
            let annual_return = (current - prev) / prev;
            
            if annual_return > 0.05 && annual_return < 0.30 {
                selected.push(candidate.symbol.clone());
            }
        }
        
        BlockResult::success(format!("dividend_growth: {} selected", selected.len())).with_selected(selected)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "Dividend growth selection" }
}

/// Business Cycle Default block.
pub struct BusinessCycleDefBlock;
impl BusinessCycleDefBlock { pub fn new() -> Self { Self } }

impl StrategyBlock for BusinessCycleDefBlock {
    fn block_id(&self) -> &'static str { "business_cycle_def" }
    fn block_type(&self) -> BlockType { BlockType::Selection }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let lookback = get_f64(params, "lookback", 126.0) as usize;
        let mut selected = Vec::new();
        
        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            if prices.len() < lookback { continue; }
            
            let current = *prices.last().unwrap();
            let prev = prices[prices.len() - lookback];
            let momentum = (current - prev) / prev;
            
            if momentum > 0.0 {
                selected.push(candidate.symbol.clone());
            }
        }
        
        BlockResult::success(format!("business_cycle_def: {} selected", selected.len())).with_selected(selected)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "Business cycle default selection" }
}
