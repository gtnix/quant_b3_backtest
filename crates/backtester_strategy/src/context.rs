//! Strategy execution context - shared state between pipeline steps.

use backtester_intelligence::filters::Market;
use backtester_intelligence::exit::Position;
use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::blocks::Signal;

/// Candidate asset with all available data for strategy evaluation.
#[derive(Debug, Clone)]
pub struct StrategyCandidate {
    pub symbol: String,
    pub market: Market,
    pub price: Option<Decimal>,
    pub avg_volume: Option<Decimal>,
    pub volatility: Option<f64>,
    pub score: Option<f64>,
    
    // Price history for indicators
    pub prices: Vec<f64>,
    pub returns: Vec<f64>,
    
    // Momentum
    pub momentum_return: Option<f64>,
    
    // Fundamentals
    pub price_earnings: Option<f64>,
    pub price_to_book: Option<f64>,
    pub return_on_equity: Option<f64>,
    pub debt_to_equity: Option<f64>,
    pub profit_margins: Option<f64>,
    pub dividend_yield: Option<f64>,
    pub market_cap: Option<i64>,
}

impl StrategyCandidate {
    pub fn new(symbol: impl Into<String>, market: Market) -> Self {
        Self {
            symbol: symbol.into(),
            market,
            price: None,
            avg_volume: None,
            volatility: None,
            score: None,
            prices: Vec::new(),
            returns: Vec::new(),
            momentum_return: None,
            price_earnings: None,
            price_to_book: None,
            return_on_equity: None,
            debt_to_equity: None,
            profit_margins: None,
            dividend_yield: None,
            market_cap: None,
        }
    }

    pub fn with_price(mut self, price: Decimal) -> Self {
        self.price = Some(price);
        self
    }

    pub fn with_prices(mut self, prices: Vec<f64>) -> Self {
        self.prices = prices;
        self
    }

    pub fn with_volatility(mut self, vol: f64) -> Self {
        self.volatility = Some(vol);
        self
    }
}

/// Trace entry for execution audit log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEntry {
    pub step: usize,
    pub block_id: String,
    pub block_type: String,
    pub message: String,
    pub timestamp_ms: u64,
}

impl TraceEntry {
    pub fn new(step: usize, block_id: &str, block_type: &str, message: &str) -> Self {
        Self {
            step,
            block_id: block_id.into(),
            block_type: block_type.into(),
            message: message.into(),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        }
    }
}

/// Strategy execution context - passed through pipeline steps.
#[derive(Debug, Clone)]
pub struct StrategyContext {
    /// Current evaluation date
    pub date: NaiveDate,
    /// Market being evaluated
    pub market: Market,
    
    // Universe and candidates
    /// Full universe of symbols
    pub universe: Vec<String>,
    /// Candidates with data for evaluation
    pub candidates: Vec<StrategyCandidate>,
    /// Symbols that passed selection
    pub selected: Vec<String>,
    
    // Signals and weights
    /// Entry signals per symbol
    pub signals: HashMap<String, Signal>,
    /// Position weights from sizing
    pub weights: HashMap<String, f64>,
    
    // Portfolio state
    /// Current positions
    pub positions: Vec<Position>,
    /// Available cash
    pub cash: Decimal,
    /// Total equity (cash + positions value)
    pub equity: Decimal,
    /// Peak equity (for drawdown calculations)
    pub peak_equity: Decimal,
    
    // Execution trace
    pub trace: Vec<TraceEntry>,
    step_counter: usize,
}

impl StrategyContext {
    pub fn new(date: NaiveDate, market: Market, cash: Decimal) -> Self {
        Self {
            date,
            market,
            universe: Vec::new(),
            candidates: Vec::new(),
            selected: Vec::new(),
            signals: HashMap::new(),
            weights: HashMap::new(),
            positions: Vec::new(),
            cash,
            equity: cash,
            peak_equity: cash,
            trace: Vec::new(),
            step_counter: 0,
        }
    }

    /// Add a trace entry for the current step.
    pub fn trace_step(&mut self, block_id: &str, message: &str) {
        let block_type = if block_id.contains("selection") || 
            ["momentum", "value", "quality", "low_vol", "dividend", "size", "carry"].contains(&block_id) {
            "selection"
        } else if ["ma_crossover", "bollinger", "rsi", "macd", "zscore"].contains(&block_id) {
            "entry"
        } else if ["stop_loss", "take_profit", "trailing_stop", "time_exit"].contains(&block_id) {
            "exit"
        } else if ["equal_weight", "risk_parity", "vol_targeting"].contains(&block_id) {
            "sizing"
        } else {
            "unknown"
        };

        self.trace.push(TraceEntry::new(
            self.step_counter,
            block_id,
            block_type,
            message,
        ));
        self.step_counter += 1;
    }

    /// Set universe of symbols.
    pub fn with_universe(mut self, symbols: Vec<String>) -> Self {
        self.universe = symbols;
        self
    }

    /// Set candidates for evaluation.
    pub fn with_candidates(mut self, candidates: Vec<StrategyCandidate>) -> Self {
        self.candidates = candidates;
        self
    }

    /// Set current positions.
    pub fn with_positions(mut self, positions: Vec<Position>) -> Self {
        self.positions = positions;
        self
    }

    /// Update equity value.
    pub fn update_equity(&mut self, equity: Decimal) {
        self.equity = equity;
        if equity > self.peak_equity {
            self.peak_equity = equity;
        }
    }

    /// Get symbols that have entry signals.
    pub fn signaled_symbols(&self) -> Vec<String> {
        self.signals
            .iter()
            .filter(|(_, s)| matches!(s.direction, crate::blocks::SignalDirection::Long))
            .map(|(sym, _)| sym.clone())
            .collect()
    }

    /// Get total weight sum (for validation).
    pub fn total_weight(&self) -> f64 {
        self.weights.values().sum()
    }

    /// Get position for symbol if exists.
    pub fn get_position(&self, symbol: &str) -> Option<&Position> {
        self.positions.iter().find(|p| p.symbol == symbol)
    }

    /// Check if symbol is in current positions.
    pub fn has_position(&self, symbol: &str) -> bool {
        self.positions.iter().any(|p| p.symbol == symbol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_context_creation() {
        let ctx = StrategyContext::new(
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            Market::BR,
            dec!(100_000),
        );
        
        assert_eq!(ctx.cash, dec!(100_000));
        assert_eq!(ctx.equity, dec!(100_000));
        assert!(ctx.candidates.is_empty());
    }

    #[test]
    fn test_trace_step() {
        let mut ctx = StrategyContext::new(
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            Market::BR,
            dec!(100_000),
        );
        
        ctx.trace_step("momentum", "Selected 5 assets");
        ctx.trace_step("equal_weight", "Applied equal weights");
        
        assert_eq!(ctx.trace.len(), 2);
        assert_eq!(ctx.trace[0].block_id, "momentum");
        assert_eq!(ctx.trace[1].block_id, "equal_weight");
    }
}

