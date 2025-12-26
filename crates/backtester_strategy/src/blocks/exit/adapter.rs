//! Exit block adapters wrapping existing ExitPolicy implementations.

use crate::blocks::{
    get_f64, get_usize, BlockParams, BlockResult, BlockType, Signal,
    StrategyBlock, ValidationError,
};
use crate::context::StrategyContext;
use std::collections::HashMap;

/// Stop-loss exit block (Technique 17).
pub struct StopLossBlock;

impl StopLossBlock {
    pub fn new() -> Self {
        Self
    }
}

impl Default for StopLossBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for StopLossBlock {
    fn block_id(&self) -> &'static str {
        "stop_loss"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Exit
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let threshold_pct = get_f64(params, "threshold_pct", 0.10); // 10% default

        let mut signals = Vec::new();
        let mut exit_count = 0;

        for position in &ctx.positions {
            let cost_basis: f64 = position.cost_basis.try_into().unwrap_or(0.0);
            let current_price: f64 = position.current_price.try_into().unwrap_or(0.0);
            
            if cost_basis <= 0.0 {
                continue;
            }

            let pnl_pct = (current_price - cost_basis) / cost_basis;

            if pnl_pct <= -threshold_pct {
                exit_count += 1;
                let signal = Signal::exit(&position.symbol, 1.0)
                    .with_source("stop_loss")
                    .with_metadata("pnl_pct", pnl_pct)
                    .with_metadata("threshold", -threshold_pct);
                signals.push(signal);
            }
        }

        ctx.trace_step(
            self.block_id(),
            &format!("{} stop-loss exits triggered", exit_count),
        );

        BlockResult::success(format!("Stop-loss: {} exits", exit_count))
            .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let threshold = get_f64(params, "threshold_pct", 0.10);
        if threshold <= 0.0 || threshold > 1.0 {
            return Err(ValidationError::OutOfRange(
                "threshold_pct".into(),
                "must be between 0 and 1".into(),
            ));
        }
        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("threshold_pct".into(), toml::Value::Float(0.10));
        params
    }

    fn description(&self) -> &'static str {
        "Stop-loss: Exit on loss exceeding threshold"
    }
}

/// Take-profit exit block (Technique 16).
pub struct TakeProfitBlock;

impl TakeProfitBlock {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TakeProfitBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for TakeProfitBlock {
    fn block_id(&self) -> &'static str {
        "take_profit"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Exit
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let target_pct = get_f64(params, "target_pct", 0.30); // 30% default

        let mut signals = Vec::new();
        let mut exit_count = 0;

        for position in &ctx.positions {
            let cost_basis: f64 = position.cost_basis.try_into().unwrap_or(0.0);
            let current_price: f64 = position.current_price.try_into().unwrap_or(0.0);
            
            if cost_basis <= 0.0 {
                continue;
            }

            let pnl_pct = (current_price - cost_basis) / cost_basis;

            if pnl_pct >= target_pct {
                exit_count += 1;
                let signal = Signal::exit(&position.symbol, 1.0)
                    .with_source("take_profit")
                    .with_metadata("pnl_pct", pnl_pct)
                    .with_metadata("target", target_pct);
                signals.push(signal);
            }
        }

        ctx.trace_step(
            self.block_id(),
            &format!("{} take-profit exits triggered", exit_count),
        );

        BlockResult::success(format!("Take-profit: {} exits", exit_count))
            .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let target = get_f64(params, "target_pct", 0.30);
        if target <= 0.0 {
            return Err(ValidationError::OutOfRange(
                "target_pct".into(),
                "must be positive".into(),
            ));
        }
        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("target_pct".into(), toml::Value::Float(0.30));
        params
    }

    fn description(&self) -> &'static str {
        "Take-profit: Exit on gain exceeding target"
    }
}

/// Trailing stop exit block (Technique 18).
pub struct TrailingStopBlock;

impl TrailingStopBlock {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TrailingStopBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for TrailingStopBlock {
    fn block_id(&self) -> &'static str {
        "trailing_stop"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Exit
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let trailing_pct = get_f64(params, "trailing_pct", 0.15); // 15% default
        let activation_pct = get_f64(params, "activation_pct", 0.10); // 10% gain to activate

        let mut signals = Vec::new();
        let mut exit_count = 0;

        for position in &ctx.positions {
            let cost_basis: f64 = position.cost_basis.try_into().unwrap_or(0.0);
            let current_price: f64 = position.current_price.try_into().unwrap_or(0.0);
            let high_water_mark: f64 = position.high_water_mark.try_into().unwrap_or(current_price);
            
            if cost_basis <= 0.0 {
                continue;
            }

            let pnl_from_cost = (current_price - cost_basis) / cost_basis;
            let drawdown_from_high = (high_water_mark - current_price) / high_water_mark;

            // Only trigger if we had sufficient gains to activate
            let was_activated = (high_water_mark - cost_basis) / cost_basis >= activation_pct;

            if was_activated && drawdown_from_high >= trailing_pct {
                exit_count += 1;
                let signal = Signal::exit(&position.symbol, 1.0)
                    .with_source("trailing_stop")
                    .with_metadata("drawdown", drawdown_from_high)
                    .with_metadata("pnl_from_cost", pnl_from_cost);
                signals.push(signal);
            }
        }

        ctx.trace_step(
            self.block_id(),
            &format!("{} trailing stop exits triggered", exit_count),
        );

        BlockResult::success(format!("Trailing stop: {} exits", exit_count))
            .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let trailing = get_f64(params, "trailing_pct", 0.15);
        let activation = get_f64(params, "activation_pct", 0.10);

        if trailing <= 0.0 || trailing > 0.5 {
            return Err(ValidationError::OutOfRange(
                "trailing_pct".into(),
                "must be between 0 and 0.5".into(),
            ));
        }

        if activation < 0.0 {
            return Err(ValidationError::OutOfRange(
                "activation_pct".into(),
                "must be non-negative".into(),
            ));
        }

        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("trailing_pct".into(), toml::Value::Float(0.15));
        params.insert("activation_pct".into(), toml::Value::Float(0.10));
        params
    }

    fn description(&self) -> &'static str {
        "Trailing stop: Exit on drawdown from high-water mark"
    }
}

/// Time-based exit block (Technique 19).
pub struct TimeExitBlock;

impl TimeExitBlock {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TimeExitBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyBlock for TimeExitBlock {
    fn block_id(&self) -> &'static str {
        "time_exit"
    }

    fn block_type(&self) -> BlockType {
        BlockType::Exit
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let max_days = get_usize(params, "max_days", 20);

        let mut signals = Vec::new();
        let mut exit_count = 0;

        for position in &ctx.positions {
            let days_held = (ctx.date - position.entry_date).num_days() as usize;

            if days_held >= max_days {
                exit_count += 1;
                let signal = Signal::exit(&position.symbol, 0.8)
                    .with_source("time_exit")
                    .with_metadata("days_held", days_held as f64)
                    .with_metadata("max_days", max_days as f64);
                signals.push(signal);
            }
        }

        ctx.trace_step(
            self.block_id(),
            &format!("{} time-based exits (>{} days)", exit_count, max_days),
        );

        BlockResult::success(format!("Time exit: {} exits", exit_count))
            .with_signals(signals)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        let max_days = get_usize(params, "max_days", 20);
        if max_days == 0 {
            return Err(ValidationError::OutOfRange(
                "max_days".into(),
                "must be positive".into(),
            ));
        }
        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("max_days".into(), toml::Value::Integer(20));
        params
    }

    fn description(&self) -> &'static str {
        "Time exit: Exit after holding for max days"
    }
}

/// Create exit block from block_id.
pub fn create_exit_block(block_id: &str) -> Option<Box<dyn StrategyBlock>> {
    match block_id {
        "stop_loss" => Some(Box::new(StopLossBlock::new())),
        "take_profit" => Some(Box::new(TakeProfitBlock::new())),
        "trailing_stop" => Some(Box::new(TrailingStopBlock::new())),
        "time_exit" => Some(Box::new(TimeExitBlock::new())),
        _ => None,
    }
}

