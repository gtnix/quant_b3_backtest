//! Entry module types and structures.

use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::filters::Market;

/// Context for entry policy evaluation.
#[derive(Debug, Clone)]
pub struct EntryContext {
    /// Current simulation date
    pub date: NaiveDate,
    /// Available capital for this market
    pub capital: Decimal,
    /// Market being evaluated (BR or US)
    pub market: Market,
    /// Maximum weight per asset (e.g., 0.20 for 20%)
    pub max_weight: f64,
    /// Minimum weight per asset (e.g., 0.02 for 2%)
    pub min_weight: f64,
    /// Top N assets to select
    pub top_n: usize,
}

impl EntryContext {
    pub fn new(date: NaiveDate, capital: Decimal, market: Market) -> Self {
        Self {
            date,
            capital,
            market,
            max_weight: 0.20,
            min_weight: 0.02,
            top_n: 10,
        }
    }
}

/// Reason why an asset was selected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionReason {
    /// Composite score (0.0 to 1.0)
    pub score: f64,
    /// Contributing filter scores
    pub filter_scores: Vec<(String, f64)>,
    /// Human-readable summary
    pub summary: String,
}

impl fmt::Display for SelectionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "score={:.3}, {}", self.score, self.summary)
    }
}

/// A target position after entry evaluation.
#[derive(Debug, Clone)]
pub struct EntryTarget {
    /// Asset symbol
    pub symbol: String,
    /// Target weight (0.0 to 1.0)
    pub target_weight: f64,
    /// Target number of shares
    pub target_shares: i64,
    /// Current price used for calculation
    pub price: Decimal,
    /// Reason for selection
    pub reason: SelectionReason,
}

/// Reason for excluding an asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExclusionReason {
    /// Volume below minimum threshold
    InsufficientLiquidity,
    /// Price below minimum threshold
    PriceTooLow,
    /// Missing fundamental data (for techniques requiring it)
    MissingFundamentals,
    /// Missing dividend data (for Carry/Dividend techniques)
    MissingDividends,
    /// Asset not tradeable on this date
    NotTradeable,
    /// Score below minimum threshold
    BelowScoreThreshold,
    /// Not in top-N after ranking
    OutOfTopN,
    /// Volatility data unavailable
    MissingVolatility,
    /// Price data unavailable
    MissingPriceData,
    /// Fundamental data from future date (anti-look-ahead)
    FutureFundamentals,
}

impl fmt::Display for ExclusionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientLiquidity => write!(f, "liquidez insuficiente"),
            Self::PriceTooLow => write!(f, "preço muito baixo"),
            Self::MissingFundamentals => write!(f, "sem dados fundamentalistas"),
            Self::MissingDividends => write!(f, "sem dados de dividendos"),
            Self::NotTradeable => write!(f, "não negociável"),
            Self::BelowScoreThreshold => write!(f, "score abaixo do threshold"),
            Self::OutOfTopN => write!(f, "fora do top-N"),
            Self::MissingVolatility => write!(f, "sem dados de volatilidade"),
            Self::MissingPriceData => write!(f, "sem dados de preço"),
            Self::FutureFundamentals => write!(f, "dados fundamentais do futuro (look-ahead)"),
        }
    }
}

/// Stage where exclusion occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExclusionStage {
    /// Excluded during gating (eligibility check)
    Gating,
    /// Excluded during selection (ranking)
    Selection,
}

impl fmt::Display for ExclusionStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gating => write!(f, "gating"),
            Self::Selection => write!(f, "selection"),
        }
    }
}

/// An asset excluded from entry.
#[derive(Debug, Clone)]
pub struct EntryExclusion {
    /// Asset symbol
    pub symbol: String,
    /// Reason for exclusion
    pub reason: ExclusionReason,
    /// Stage where exclusion occurred
    pub stage: ExclusionStage,
    /// Score if available (for ranking exclusions)
    pub score: Option<f64>,
}

impl fmt::Display for EntryExclusion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(score) = self.score {
            write!(f, "{}: {} ({}, score={:.3})", self.symbol, self.reason, self.stage, score)
        } else {
            write!(f, "{}: {} ({})", self.symbol, self.reason, self.stage)
        }
    }
}

/// Diagnostics from entry evaluation.
#[derive(Debug, Clone, Default)]
pub struct EntryDiagnostics {
    /// Total candidates evaluated
    pub total_candidates: usize,
    /// Excluded during gating
    pub gating_excluded: usize,
    /// Excluded during selection
    pub selection_excluded: usize,
    /// Final selected count
    pub final_selected: usize,
    /// Portfolio turnover (0.0 to 1.0)
    pub turnover: f64,
    /// Estimated transaction costs
    pub estimated_costs: Decimal,
    /// Sum of weights (should be ~1.0)
    pub total_weight: f64,
    /// Cash residual after allocation (capital - sum(shares * price))
    pub cash_residual: Decimal,
}

/// Result of entry policy evaluation.
#[derive(Debug, Clone)]
pub struct EntryResult {
    /// Date of evaluation
    pub date: NaiveDate,
    /// Market evaluated
    pub market: Market,
    /// Assets to enter (targets)
    pub targets: Vec<EntryTarget>,
    /// Assets excluded with reasons
    pub exclusions: Vec<EntryExclusion>,
    /// Diagnostic metrics
    pub diagnostics: EntryDiagnostics,
}

impl EntryResult {
    pub fn new(date: NaiveDate, market: Market) -> Self {
        Self {
            date,
            market,
            targets: Vec::new(),
            exclusions: Vec::new(),
            diagnostics: EntryDiagnostics::default(),
        }
    }
}

/// Order side (buy or sell).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl fmt::Display for OrderSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Buy => write!(f, "BUY"),
            Self::Sell => write!(f, "SELL"),
        }
    }
}

/// A simulated order.
#[derive(Debug, Clone)]
pub struct Order {
    /// Asset symbol
    pub symbol: String,
    /// Buy or Sell
    pub side: OrderSide,
    /// Number of shares
    pub shares: i64,
    /// Execution price
    pub price: Decimal,
    /// Estimated cost (fees + slippage)
    pub estimated_cost: Decimal,
    /// Notional value
    pub notional: Decimal,
}

impl Order {
    pub fn new(symbol: String, side: OrderSide, shares: i64, price: Decimal, cost: Decimal) -> Self {
        let notional = price * Decimal::from(shares);
        Self {
            symbol,
            side,
            shares,
            price,
            estimated_cost: cost,
            notional,
        }
    }
}

impl fmt::Display for Order {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} x {} @ {} (cost: {})",
            self.side, self.symbol, self.shares, self.price, self.estimated_cost
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entry_context_creation() {
        let ctx = EntryContext::new(
            NaiveDate::from_ymd_opt(2025, 1, 3).unwrap(),
            Decimal::from(100_000),
            Market::BR,
        );
        assert_eq!(ctx.top_n, 10);
        assert_eq!(ctx.max_weight, 0.20);
    }

    #[test]
    fn test_exclusion_reason_display() {
        assert_eq!(ExclusionReason::MissingFundamentals.to_string(), "sem dados fundamentalistas");
        assert_eq!(ExclusionReason::OutOfTopN.to_string(), "fora do top-N");
    }

    #[test]
    fn test_order_creation() {
        let order = Order::new(
            "PETR4".to_string(),
            OrderSide::Buy,
            100,
            Decimal::from(38),
            Decimal::new(2380, 2), // R$ 23.80
        );
        assert_eq!(order.notional, Decimal::from(3800));
        assert!(order.to_string().contains("BUY PETR4"));
    }
}

