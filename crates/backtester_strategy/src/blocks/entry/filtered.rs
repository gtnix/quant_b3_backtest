//! Filtered Entry Blocks
//!
//! RSI and Bollinger variants with additional filters.

use crate::blocks::{BlockParams, BlockResult, BlockType, Signal, SignalDirection, StrategyBlock, ValidationError};
use crate::context::StrategyContext;
use crate::blocks::get_f64;

/// RSI Filtered block - RSI with trend confirmation.
pub struct RSIFilteredBlock;
impl RSIFilteredBlock { pub fn new() -> Self { Self } }

impl StrategyBlock for RSIFilteredBlock {
    fn block_id(&self) -> &'static str { "rsi_filtered" }
    fn block_type(&self) -> BlockType { BlockType::Entry }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let period = get_f64(params, "period", 14.0) as usize;
        let oversold = get_f64(params, "oversold", 30.0);
        let overbought = get_f64(params, "overbought", 70.0);
        let ma_period = get_f64(params, "ma_period", 50.0) as usize;
        let mut signals = Vec::new();
        
        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            if prices.len() < period.max(ma_period) { continue; }
            
            // Calculate RSI
            let mut gains = 0.0;
            let mut losses = 0.0;
            for i in (prices.len() - period)..prices.len() {
                if i == 0 { continue; }
                let change = prices[i] - prices[i - 1];
                if change > 0.0 { gains += change; } else { losses -= change; }
            }
            
            let avg_gain = gains / period as f64;
            let avg_loss = losses / period as f64;
            let rs = if avg_loss > 0.0 { avg_gain / avg_loss } else { 100.0 };
            let rsi = 100.0 - (100.0 / (1.0 + rs));
            
            // MA trend filter
            let current = *prices.last().unwrap();
            let ma: f64 = prices.iter().rev().take(ma_period).sum::<f64>() / ma_period as f64;
            let above_ma = current > ma;
            
            let (direction, strength) = if rsi < oversold && above_ma {
                (SignalDirection::Long, 0.8)
            } else if rsi > overbought && !above_ma {
                (SignalDirection::Short, 0.8)
            } else {
                (SignalDirection::Flat, 0.0)
            };
            
            if direction != SignalDirection::Flat {
                signals.push(Signal::new(&candidate.symbol, direction, strength)
                    .with_source(self.block_id())
                    .with_metadata("rsi", rsi));
            }
        }
        
        BlockResult::success(format!("rsi_filtered: {} signals", signals.len())).with_signals(signals)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "RSI with trend filter" }
}

/// Bollinger Filtered block - BB with volatility filter.
pub struct BollingerFilteredBlock;
impl BollingerFilteredBlock { pub fn new() -> Self { Self } }

impl StrategyBlock for BollingerFilteredBlock {
    fn block_id(&self) -> &'static str { "bb_filtered" }
    fn block_type(&self) -> BlockType { BlockType::Entry }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let period = get_f64(params, "period", 20.0) as usize;
        let std_dev = get_f64(params, "std_dev", 2.0);
        let mut signals = Vec::new();
        
        for candidate in &ctx.candidates {
            let prices = &candidate.prices;
            if prices.len() < period * 2 { continue; }
            
            let mean: f64 = prices.iter().rev().take(period).sum::<f64>() / period as f64;
            let variance: f64 = prices.iter().rev().take(period)
                .map(|p| (p - mean).powi(2)).sum::<f64>() / period as f64;
            let std = variance.sqrt();
            
            let upper = mean + std_dev * std;
            let lower = mean - std_dev * std;
            let current = *prices.last().unwrap();
            
            // Check bandwidth contraction
            let bandwidth = (upper - lower) / mean;
            let hist_mean: f64 = prices.iter().rev().skip(period).take(period).sum::<f64>() / period as f64;
            let hist_var: f64 = prices.iter().rev().skip(period).take(period)
                .map(|p| (p - hist_mean).powi(2)).sum::<f64>() / period as f64;
            let hist_std = hist_var.sqrt();
            let hist_bw = (2.0 * std_dev * hist_std) / hist_mean.max(0.001);
            let vol_ratio = bandwidth / hist_bw.max(0.001);
            
            // Only signal on low volatility (squeeze)
            if vol_ratio < 0.8 {
                let (direction, strength) = if current < lower {
                    (SignalDirection::Long, 0.7)
                } else if current > upper {
                    (SignalDirection::Short, 0.7)
                } else {
                    (SignalDirection::Flat, 0.0)
                };
                
                if direction != SignalDirection::Flat {
                    signals.push(Signal::new(&candidate.symbol, direction, strength)
                        .with_source(self.block_id()));
                }
            }
        }
        
        BlockResult::success(format!("bb_filtered: {} signals", signals.len())).with_signals(signals)
    }

    fn validate_params(&self, _params: &BlockParams) -> Result<(), ValidationError> { Ok(()) }
    fn default_params(&self) -> BlockParams { BlockParams::new() }
    fn description(&self) -> &'static str { "Bollinger Bands with volatility filter" }
}
