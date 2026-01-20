//! Core types for the Exit Module.
//!
//! # Performance (Milestone 4)
//!
//! `Position` uses fixed-point `Price` and `Money` internally for all price
//! and monetary calculations, providing 5-10x faster arithmetic than Decimal.
//! Decimal conversions are available at boundaries for compatibility.

use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::fmt;

use backtester_core::{Money, Price};
use crate::filters::Market;

/// Reason for exiting a position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExitReason {
    /// Stop-loss triggered (price dropped below threshold)
    StopLoss,
    /// Take-profit triggered (price rose above threshold)
    TakeProfit,
    /// Time-based exit (max holding period exceeded)
    TimeExit,
    /// Trailing stop triggered (price dropped from high-water mark)
    TrailingStop,
    /// Risk cap exceeded (exposure/concentration limit)
    RiskCap,
    /// Drawdown guard triggered (portfolio drawdown exceeded)
    DrawdownGuard,
    /// Regular rebalance (asset no longer in top-N)
    Rebalance,
    /// Manual/forced exit
    Manual,
}

impl fmt::Display for ExitReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StopLoss => write!(f, "stop-loss"),
            Self::TakeProfit => write!(f, "take-profit"),
            Self::TimeExit => write!(f, "tempo máximo"),
            Self::TrailingStop => write!(f, "trailing-stop"),
            Self::RiskCap => write!(f, "limite de risco"),
            Self::DrawdownGuard => write!(f, "drawdown guard"),
            Self::Rebalance => write!(f, "rebalanceamento"),
            Self::Manual => write!(f, "manual"),
        }
    }
}

/// Risk violation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RiskViolation {
    /// Single asset exposure exceeds max
    ExposureExceeded,
    /// Market exposure exceeds max
    MarketExposureExceeded,
    /// Turnover exceeds max per rebalance
    TurnoverExceeded,
    /// Portfolio drawdown exceeds threshold
    DrawdownExceeded,
    /// CVaR (Conditional Value-at-Risk) exceeds threshold
    CVaRExceeded,
}

impl fmt::Display for RiskViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExposureExceeded => write!(f, "exposição por ativo excedida"),
            Self::MarketExposureExceeded => write!(f, "exposição de mercado excedida"),
            Self::TurnoverExceeded => write!(f, "turnover excedido"),
            Self::DrawdownExceeded => write!(f, "drawdown excedido"),
            Self::CVaRExceeded => write!(f, "CVaR excedido"),
        }
    }
}

/// Action to take when drawdown is exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DrawdownAction {
    /// Reduce risk by selling highest-risk positions
    #[default]
    ReduceRisk,
    /// Exit all positions to cash
    CashOut,
    /// Just alert, don't take action
    Alert,
}

/// A position in the portfolio.
///
/// # Performance (Milestone 4)
///
/// Uses fixed-point `Price` internally for all price fields:
/// - `cost_basis`: Average cost per share
/// - `current_price`: Current market price
/// - `high_water_mark`: Highest price since entry (for trailing stops)
///
/// All arithmetic (market_value, unrealized_pnl) uses i64 operations,
/// which are 10-50x faster than Decimal.
#[derive(Debug, Clone)]
pub struct Position {
    /// Asset symbol
    pub symbol: String,
    /// Market (BR/US)
    pub market: Market,
    /// Number of shares held
    pub shares: i64,
    /// Average cost basis per share (fixed-point)
    pub cost_basis: Price,
    /// Entry date
    pub entry_date: NaiveDate,
    /// Current price (fixed-point)
    pub current_price: Price,
    /// High-water mark price for trailing stop (fixed-point)
    pub high_water_mark: Price,
}

impl Position {
    /// Create a new position with fixed-point prices.
    pub fn new_fast(
        symbol: impl Into<String>,
        market: Market,
        shares: i64,
        cost_basis: Price,
        entry_date: NaiveDate,
        current_price: Price,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            market,
            shares,
            cost_basis,
            entry_date,
            current_price,
            high_water_mark: current_price,
        }
    }

    /// Create a new position from Decimal values (for compatibility).
    /// Use this at API boundaries, NOT in hot path.
    pub fn new(
        symbol: impl Into<String>,
        market: Market,
        shares: i64,
        cost_basis: Decimal,
        entry_date: NaiveDate,
        current_price: Decimal,
    ) -> Self {
        Self::new_fast(
            symbol,
            market,
            shares,
            Price::from(cost_basis),
            entry_date,
            Price::from(current_price),
        )
    }

    // =========================================================================
    // HOT PATH METHODS (zero Decimal)
    // =========================================================================

    /// Calculate unrealized PnL (fixed-point, fast).
    #[inline]
    pub fn unrealized_pnl_fast(&self) -> Money {
        let diff = self.current_price - self.cost_basis;
        diff.mul_shares(self.shares)
    }

    /// Calculate position value at current price (fixed-point, fast).
    #[inline]
    pub fn market_value_fast(&self) -> Money {
        self.current_price.mul_shares(self.shares)
    }

    /// Update high-water mark if current price is higher (fixed-point, fast).
    #[inline]
    pub fn update_high_water_mark(&mut self) {
        if self.current_price > self.high_water_mark {
            self.high_water_mark = self.current_price;
        }
    }

    /// Calculate unrealized return (%) - returns f64 for metrics.
    #[inline]
    pub fn unrealized_return(&self) -> f64 {
        if self.cost_basis.is_zero() {
            return 0.0;
        }
        (self.current_price.to_f64() - self.cost_basis.to_f64()) / self.cost_basis.to_f64()
    }

    /// Calculate drawdown from high-water mark (%) - returns f64 for metrics.
    #[inline]
    pub fn drawdown_from_high(&self) -> f64 {
        if self.high_water_mark.is_zero() {
            return 0.0;
        }
        (self.current_price.to_f64() - self.high_water_mark.to_f64()) / self.high_water_mark.to_f64()
    }

    /// Days held since entry.
    pub fn days_held(&self, as_of: NaiveDate) -> i64 {
        (as_of - self.entry_date).num_days()
    }

    // =========================================================================
    // COMPATIBILITY METHODS (for Decimal API boundaries)
    // =========================================================================

    /// Calculate unrealized PnL as Decimal (for compatibility).
    pub fn unrealized_pnl(&self) -> Decimal {
        self.unrealized_pnl_fast().to_decimal()
    }

    /// Calculate position value as Decimal (for compatibility).
    pub fn market_value(&self) -> Decimal {
        self.market_value_fast().to_decimal()
    }

    /// Get cost basis as Decimal.
    #[inline]
    pub fn cost_basis_decimal(&self) -> Decimal {
        self.cost_basis.to_decimal()
    }

    /// Get current price as Decimal.
    #[inline]
    pub fn current_price_decimal(&self) -> Decimal {
        self.current_price.to_decimal()
    }

    /// Get high water mark as Decimal.
    #[inline]
    pub fn high_water_mark_decimal(&self) -> Decimal {
        self.high_water_mark.to_decimal()
    }
}

// =============================================================================
// PositionFast - Ultra-performance position (Copy, no String)
// =============================================================================

/// Symbol identifier for O(1) array indexing (matches backtester_engine::SymbolId).
pub type SymbolId = u32;

/// Ultra-performance position with Copy semantics.
///
/// Uses `SymbolId` (u32) instead of String for O(1) lookups and Copy trait.
/// ~3x faster iteration and zero allocation on clone.
///
/// # Size
/// 48 bytes (vs ~120+ bytes for Position with String)
#[derive(Debug, Clone, Copy)]
pub struct PositionFast {
    /// Symbol identifier (use registry to resolve to String)
    pub symbol_id: SymbolId,
    /// Market (BR/US)
    pub market: Market,
    /// Number of shares held
    pub shares: i64,
    /// Average cost basis per share (fixed-point)
    pub cost_basis: Price,
    /// Entry date (days since epoch for compactness)
    pub entry_date: NaiveDate,
    /// Current price (fixed-point)
    pub current_price: Price,
    /// High-water mark price for trailing stop (fixed-point)
    pub high_water_mark: Price,
}

impl PositionFast {
    /// Create a new fast position.
    #[inline]
    pub const fn new(
        symbol_id: SymbolId,
        market: Market,
        shares: i64,
        cost_basis: Price,
        entry_date: NaiveDate,
        current_price: Price,
    ) -> Self {
        Self {
            symbol_id,
            market,
            shares,
            cost_basis,
            entry_date,
            current_price,
            high_water_mark: current_price,
        }
    }

    /// Calculate unrealized PnL (fixed-point, fast).
    #[inline]
    pub fn unrealized_pnl_fast(&self) -> Money {
        let diff = self.current_price - self.cost_basis;
        diff.mul_shares(self.shares)
    }

    /// Calculate position value at current price (fixed-point, fast).
    #[inline]
    pub fn market_value_fast(&self) -> Money {
        self.current_price.mul_shares(self.shares)
    }

    /// Update high-water mark if current price is higher.
    #[inline]
    pub fn update_high_water_mark(&mut self) {
        if self.current_price > self.high_water_mark {
            self.high_water_mark = self.current_price;
        }
    }

    /// Calculate unrealized return (%).
    #[inline]
    pub fn unrealized_return(&self) -> f64 {
        if self.cost_basis.is_zero() {
            return 0.0;
        }
        (self.current_price.to_f64() - self.cost_basis.to_f64()) / self.cost_basis.to_f64()
    }

    /// Calculate drawdown from high-water mark (%).
    #[inline]
    pub fn drawdown_from_high(&self) -> f64 {
        if self.high_water_mark.is_zero() {
            return 0.0;
        }
        (self.current_price.to_f64() - self.high_water_mark.to_f64()) / self.high_water_mark.to_f64()
    }

    /// Days held since entry.
    #[inline]
    pub fn days_held(&self, as_of: NaiveDate) -> i64 {
        (as_of - self.entry_date).num_days()
    }
}

/// Target for exiting a position.
#[derive(Debug, Clone)]
pub struct ExitTarget {
    /// Asset symbol
    pub symbol: String,
    /// Market (BR/US)
    pub market: Market,
    /// Shares to sell (negative = all)
    pub shares_to_sell: i64,
    /// Reason for exit
    pub reason: ExitReason,
    /// Price at which to sell (fixed-point)
    pub price: Price,
    /// Unrealized PnL at exit (fixed-point)
    pub unrealized_pnl: Money,
    /// Unrealized return at exit
    pub unrealized_return: f64,
}

impl ExitTarget {
    pub fn from_position(position: &Position, reason: ExitReason, shares: Option<i64>) -> Self {
        Self {
            symbol: position.symbol.clone(),
            market: position.market,
            shares_to_sell: shares.unwrap_or(position.shares),
            reason,
            price: position.current_price,
            unrealized_pnl: position.unrealized_pnl_fast(),
            unrealized_return: position.unrealized_return(),
        }
    }

    /// Notional value of the exit (fixed-point).
    #[inline]
    pub fn notional_fast(&self) -> Money {
        self.price.mul_shares(self.shares_to_sell)
    }

    /// Notional value as Decimal (for compatibility).
    pub fn notional(&self) -> Decimal {
        self.notional_fast().to_decimal()
    }

    /// Price as Decimal (for compatibility).
    pub fn price_decimal(&self) -> Decimal {
        self.price.to_decimal()
    }

    /// Unrealized PnL as Decimal (for compatibility).
    pub fn unrealized_pnl_decimal(&self) -> Decimal {
        self.unrealized_pnl.to_decimal()
    }
}

/// Context for exit evaluation.
#[derive(Debug, Clone)]
pub struct ExitContext {
    /// Current date
    pub date: NaiveDate,
    /// Total portfolio capital (fixed-point)
    pub capital: Money,
    /// Current portfolio equity (NAV) (fixed-point)
    pub equity: Money,
    /// Peak portfolio equity for drawdown (fixed-point)
    pub peak_equity: Money,
    /// Market being evaluated
    pub market: Market,
}

impl ExitContext {
    /// Create new context with fixed-point Money values.
    pub fn new_fast(date: NaiveDate, capital: Money, equity: Money, market: Market) -> Self {
        Self {
            date,
            capital,
            equity,
            peak_equity: equity,
            market,
        }
    }

    /// Create new context from Decimal values (for compatibility).
    pub fn new(date: NaiveDate, capital: Decimal, equity: Decimal, market: Market) -> Self {
        Self::new_fast(date, Money::from(capital), Money::from(equity), market)
    }

    /// Calculate portfolio drawdown from peak (%).
    #[inline]
    pub fn portfolio_drawdown(&self) -> f64 {
        if self.peak_equity.is_zero() {
            return 0.0;
        }
        self.equity.div_money(self.peak_equity) - 1.0
    }

    /// Get capital as Decimal.
    pub fn capital_decimal(&self) -> Decimal {
        self.capital.to_decimal()
    }

    /// Get equity as Decimal.
    pub fn equity_decimal(&self) -> Decimal {
        self.equity.to_decimal()
    }
}

/// Diagnostics from exit evaluation.
#[derive(Debug, Clone, Default)]
pub struct ExitDiagnostics {
    /// Total positions evaluated
    pub positions_evaluated: usize,
    /// Positions exited
    pub positions_exited: usize,
    /// Stop-loss exits
    pub stop_loss_count: usize,
    /// Take-profit exits
    pub take_profit_count: usize,
    /// Time exits
    pub time_exit_count: usize,
    /// Trailing stop exits
    pub trailing_stop_count: usize,
    /// Risk cap exits
    pub risk_cap_count: usize,
    /// Drawdown guard exits
    pub drawdown_guard_count: usize,
    /// Rebalance exits
    pub rebalance_count: usize,
    /// Total unrealized PnL of exits (fixed-point)
    pub total_exit_pnl: Money,
    /// Total turnover from exits (fixed-point)
    pub exit_turnover: Money,
    /// Estimated costs (fixed-point)
    pub estimated_costs: Money,
    /// Risk violations detected
    pub risk_violations: Vec<RiskViolation>,
}

impl ExitDiagnostics {
    /// Increment counter for exit reason.
    pub fn count_exit(&mut self, reason: ExitReason) {
        match reason {
            ExitReason::StopLoss => self.stop_loss_count += 1,
            ExitReason::TakeProfit => self.take_profit_count += 1,
            ExitReason::TimeExit => self.time_exit_count += 1,
            ExitReason::TrailingStop => self.trailing_stop_count += 1,
            ExitReason::RiskCap => self.risk_cap_count += 1,
            ExitReason::DrawdownGuard => self.drawdown_guard_count += 1,
            ExitReason::Rebalance => self.rebalance_count += 1,
            ExitReason::Manual => {}
        }
        self.positions_exited += 1;
    }
}

/// Result of exit policy evaluation.
#[derive(Debug, Clone)]
pub struct ExitResult {
    /// Date of evaluation
    pub date: NaiveDate,
    /// Market evaluated
    pub market: Market,
    /// Exit targets
    pub exits: Vec<ExitTarget>,
    /// Diagnostic metrics
    pub diagnostics: ExitDiagnostics,
}

impl ExitResult {
    pub fn new(date: NaiveDate, market: Market) -> Self {
        Self {
            date,
            market,
            exits: Vec::new(),
            diagnostics: ExitDiagnostics::default(),
        }
    }

    /// Add an exit target.
    pub fn add_exit(&mut self, exit: ExitTarget) {
        self.diagnostics.count_exit(exit.reason);
        self.diagnostics.total_exit_pnl += exit.unrealized_pnl;
        self.diagnostics.exit_turnover += exit.notional_fast();
        self.exits.push(exit);
    }

    /// Check if any exits were triggered.
    pub fn has_exits(&self) -> bool {
        !self.exits.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_position_unrealized_pnl() {
        let pos = Position::new(
            "PETR4",
            Market::BR,
            100,
            dec!(30),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(35),
        );
        assert_eq!(pos.unrealized_pnl(), dec!(500)); // (35-30)*100
    }

    #[test]
    fn test_position_unrealized_return() {
        let pos = Position::new(
            "PETR4",
            Market::BR,
            100,
            dec!(30),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(33),
        );
        let ret = pos.unrealized_return();
        assert!((ret - 0.10).abs() < 0.001); // 10% gain
    }

    #[test]
    fn test_position_days_held() {
        let pos = Position::new(
            "VALE3",
            Market::BR,
            50,
            dec!(60),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(65),
        );
        let days = pos.days_held(NaiveDate::from_ymd_opt(2025, 1, 11).unwrap());
        assert_eq!(days, 10);
    }

    #[test]
    fn test_exit_target_from_position() {
        let pos = Position::new(
            "ITUB4",
            Market::BR,
            200,
            dec!(25),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(22),
        );
        let exit = ExitTarget::from_position(&pos, ExitReason::StopLoss, None);
        assert_eq!(exit.shares_to_sell, 200);
        assert_eq!(exit.reason, ExitReason::StopLoss);
        assert_eq!(exit.unrealized_pnl.to_decimal(), dec!(-600)); // (22-25)*200
    }

    #[test]
    fn test_exit_diagnostics_counting() {
        let mut diag = ExitDiagnostics::default();
        diag.count_exit(ExitReason::StopLoss);
        diag.count_exit(ExitReason::StopLoss);
        diag.count_exit(ExitReason::TakeProfit);
        
        assert_eq!(diag.stop_loss_count, 2);
        assert_eq!(diag.take_profit_count, 1);
        assert_eq!(diag.positions_exited, 3);
    }

    #[test]
    fn test_portfolio_drawdown() {
        use backtester_core::Money;
        let mut ctx = ExitContext::new(
            NaiveDate::from_ymd_opt(2025, 1, 10).unwrap(),
            dec!(1_000_000),
            dec!(850_000),
            Market::BR,
        );
        // Set peak to 1M for 15% drawdown from current 850k
        ctx.peak_equity = Money::from_int(1_000_000);
        let dd = ctx.portfolio_drawdown();
        assert!((dd - (-0.15)).abs() < 0.001); // -15% drawdown
    }
}






























