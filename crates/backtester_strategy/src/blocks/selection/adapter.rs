//! Adapters wrapping existing AssetFilter implementations as StrategyBlock.

use crate::blocks::{get_f64, get_i64, BlockParams, BlockResult, BlockType, StrategyBlock, ValidationError};
use crate::context::StrategyContext;
use backtester_intelligence::filters::{AssetData, AssetFilter, FilterResult};
use backtester_intelligence::config::{
    CarryConfig, DividendYieldConfig, LowVolConfig, MomentumConfig, QualityConfig, SizeConfig,
    ValueConfig, AssetFilterConfig,
};
use backtester_intelligence::filters::{
    CarryFilter, DividendYieldFilter, LowVolFilter, MomentumFilter, QualityFilter, SizeFilter,
    ValueFilter,
};
use std::collections::HashMap;

/// Wrapper that adapts an AssetFilter to StrategyBlock interface.
pub struct SelectionAdapter<F: AssetFilter> {
    filter: F,
    block_id: &'static str,
    description: &'static str,
}

impl<F: AssetFilter + 'static> SelectionAdapter<F> {
    pub fn new(filter: F, block_id: &'static str, description: &'static str) -> Self {
        Self { filter, block_id, description }
    }
}

impl<F: AssetFilter + 'static> StrategyBlock for SelectionAdapter<F> {
    fn block_id(&self) -> &'static str {
        self.block_id
    }

    fn block_type(&self) -> BlockType {
        BlockType::Selection
    }

    fn execute(&self, ctx: &mut StrategyContext, params: &BlockParams) -> BlockResult {
        let top_pct = get_f64(params, "top_pct", 20.0) / 100.0;
        let mut results: Vec<(String, FilterResult)> = Vec::new();

        for candidate in &ctx.candidates {
            let mut asset_data = AssetData::with_inferred_market(&candidate.symbol);
            
            // Populate asset data from candidate
            if let Some(price) = candidate.price {
                let price_f64 = price.try_into().unwrap_or(0.0);
                asset_data.prices.push((ctx.date, price_f64));
            }
            asset_data.momentum_return = candidate.momentum_return;
            asset_data.annualized_volatility = candidate.volatility;
            asset_data.price_earnings = candidate.price_earnings;
            asset_data.price_to_book = candidate.price_to_book;
            asset_data.return_on_equity = candidate.return_on_equity;
            asset_data.debt_to_equity = candidate.debt_to_equity;
            asset_data.profit_margins = candidate.profit_margins;
            asset_data.dividend_yield = candidate.dividend_yield;
            asset_data.market_cap = candidate.market_cap;

            let result = self.filter.evaluate(&asset_data);
            results.push((candidate.symbol.clone(), result));
        }

        // Sort by score descending
        results.sort_by(|a, b| b.1.score.partial_cmp(&a.1.score).unwrap_or(std::cmp::Ordering::Equal));

        // Select top N%
        let select_count = ((results.len() as f64) * top_pct).ceil() as usize;
        let select_count = select_count.max(1).min(results.len());

        let mut selected = Vec::new();
        let mut excluded = Vec::new();

        for (i, (symbol, result)) in results.into_iter().enumerate() {
            if result.passed && i < select_count {
                selected.push(symbol.clone());
                // Update candidate score in context
                if let Some(c) = ctx.candidates.iter_mut().find(|c| c.symbol == symbol) {
                    c.score = Some(result.score);
                }
            } else {
                let reason = if !result.passed {
                    result.reason.clone()
                } else {
                    format!("Outside top {}%", (top_pct * 100.0) as i32)
                };
                excluded.push((symbol, reason));
            }
        }

        ctx.trace_step(self.block_id(), &format!("Selected {} assets", selected.len()));

        BlockResult::success(format!("{}: {} selected, {} excluded", 
            self.block_id(), selected.len(), excluded.len()))
            .with_selected(selected)
            .with_excluded(excluded)
    }

    fn validate_params(&self, params: &BlockParams) -> Result<(), ValidationError> {
        if let Some(top_pct) = params.get("top_pct") {
            if let Some(v) = top_pct.as_float().or_else(|| top_pct.as_integer().map(|i| i as f64)) {
                if !(0.0..=100.0).contains(&v) {
                    return Err(ValidationError::OutOfRange(
                        "top_pct".into(),
                        "must be between 0 and 100".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn default_params(&self) -> BlockParams {
        let mut params = HashMap::new();
        params.insert("top_pct".into(), toml::Value::Float(20.0));
        params
    }

    fn description(&self) -> &'static str {
        self.description
    }
}

// Factory functions for each selection block

/// Momentum selection (Technique 1)
pub fn momentum_block(lookback_days: i32, min_return: f64) -> Box<dyn StrategyBlock> {
    let config = MomentumConfig {
        base: AssetFilterConfig { enabled: true, weight: 1.0 },
        lookback_days,
        min_return,
        skip_last_days: 21,
    };
    Box::new(SelectionAdapter::new(
        MomentumFilter::new(config),
        "momentum",
        "Momentum selection: ranks assets by 6-12 month returns",
    ))
}

/// Value selection (Technique 2)
pub fn value_block(max_pe: f64, max_pb: f64) -> Box<dyn StrategyBlock> {
    let config = ValueConfig {
        base: AssetFilterConfig { enabled: true, weight: 1.0 },
        max_pe,
        max_pb,
        min_pe: 0.0,
    };
    Box::new(SelectionAdapter::new(
        ValueFilter::new(config),
        "value",
        "Value selection: selects low P/E, low P/B stocks",
    ))
}

/// Quality selection (Technique 3)
pub fn quality_block(min_roe: f64, max_debt_equity: f64) -> Box<dyn StrategyBlock> {
    let config = QualityConfig {
        base: AssetFilterConfig { enabled: true, weight: 1.0 },
        min_roe,
        max_debt_equity,
        min_profit_margin: 0.05,
        min_gross_margin: None,
    };
    Box::new(SelectionAdapter::new(
        QualityFilter::new(config),
        "quality",
        "Quality selection: high ROE, low debt companies",
    ))
}

/// Low volatility selection (Technique 4)
pub fn low_vol_block(max_vol: f64, lookback_days: i32) -> Box<dyn StrategyBlock> {
    let config = LowVolConfig {
        base: AssetFilterConfig { enabled: true, weight: 1.0 },
        lookback_days,
        max_annualized_vol: max_vol,
    };
    Box::new(SelectionAdapter::new(
        LowVolFilter::new(config),
        "low_vol",
        "Low volatility selection: selects stable, low-vol assets",
    ))
}

/// Dividend yield selection (Technique 5)
pub fn dividend_block(min_yield: f64, max_yield: Option<f64>) -> Box<dyn StrategyBlock> {
    let config = DividendYieldConfig {
        base: AssetFilterConfig { enabled: true, weight: 1.0 },
        min_yield,
        max_yield,
    };
    Box::new(SelectionAdapter::new(
        DividendYieldFilter::new(config),
        "dividend",
        "Dividend yield selection: high dividend stocks",
    ))
}

/// Size selection (Technique 6)
pub fn size_block(min_market_cap: i64, max_market_cap: Option<i64>) -> Box<dyn StrategyBlock> {
    let config = SizeConfig {
        base: AssetFilterConfig { enabled: true, weight: 1.0 },
        min_market_cap,
        max_market_cap,
    };
    Box::new(SelectionAdapter::new(
        SizeFilter::new(config),
        "size",
        "Size selection: filters by market cap (small/mid/large)",
    ))
}

/// Carry selection (Technique 7)
pub fn carry_block(min_carry: f64) -> Box<dyn StrategyBlock> {
    let config = CarryConfig {
        base: AssetFilterConfig { enabled: true, weight: 1.0 },
        min_carry,
        fallback_selic_br: 0.1075,
        fallback_tbill_us: 0.0435,
    };
    Box::new(SelectionAdapter::new(
        CarryFilter::new(config),
        "carry",
        "Carry selection: dividend yield vs risk-free rate",
    ))
}

/// Create selection block from params dynamically.
pub fn create_selection_block(block_id: &str, params: &BlockParams) -> Option<Box<dyn StrategyBlock>> {
    match block_id {
        "momentum" => Some(momentum_block(
            get_i64(params, "lookback_days", 126) as i32,
            get_f64(params, "min_return", 0.0),
        )),
        "value" => Some(value_block(
            get_f64(params, "max_pe", 15.0),
            get_f64(params, "max_pb", 2.0),
        )),
        "quality" => Some(quality_block(
            get_f64(params, "min_roe", 0.12),
            get_f64(params, "max_debt_equity", 1.0),
        )),
        "low_vol" => Some(low_vol_block(
            get_f64(params, "max_annualized_vol", 0.30),
            get_i64(params, "lookback_days", 60) as i32,
        )),
        "dividend" => Some(dividend_block(
            get_f64(params, "min_yield", 0.03),
            params.get("max_yield").and_then(|v| v.as_float()),
        )),
        "size" => Some(size_block(
            get_i64(params, "min_market_cap", 5_000_000_000),
            params.get("max_market_cap").and_then(|v| v.as_integer()),
        )),
        "carry" => Some(carry_block(
            get_f64(params, "min_carry", 0.0),
        )),
        _ => None,
    }
}

