//! Core types for the Exit Module.

use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::fmt;

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
#[derive(Debug, Clone)]
pub struct Position {
    /// Asset symbol
    pub symbol: String,
    /// Market (BR/US)
    pub market: Market,
    /// Number of shares held
    pub shares: i64,
    /// Average cost basis per share
    pub cost_basis: Decimal,
    /// Entry date
    pub entry_date: NaiveDate,
    /// Current price
    pub current_price: Decimal,
    /// High-water mark price (for trailing stop)
    pub high_water_mark: Decimal,
}

impl Position {
    pub fn new(
        symbol: impl Into<String>,
        market: Market,
        shares: i64,
        cost_basis: Decimal,
        entry_date: NaiveDate,
        current_price: Decimal,
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

    /// Calculate unrealized PnL.
    pub fn unrealized_pnl(&self) -> Decimal {
        (self.current_price - self.cost_basis) * Decimal::from(self.shares)
    }

    /// Calculate unrealized return (%).
    pub fn unrealized_return(&self) -> f64 {
        if self.cost_basis == Decimal::ZERO {
            return 0.0;
        }
        let ret = (self.current_price - self.cost_basis) / self.cost_basis;
        ret.try_into().unwrap_or(0.0)
    }

    /// Calculate position value at current price.
    pub fn market_value(&self) -> Decimal {
        self.current_price * Decimal::from(self.shares)
    }

    /// Days held since entry.
    pub fn days_held(&self, as_of: NaiveDate) -> i64 {
        (as_of - self.entry_date).num_days()
    }

    /// Update high-water mark if current price is higher.
    pub fn update_high_water_mark(&mut self) {
        if self.current_price > self.high_water_mark {
            self.high_water_mark = self.current_price;
        }
    }

    /// Calculate drawdown from high-water mark.
    pub fn drawdown_from_high(&self) -> f64 {
        if self.high_water_mark == Decimal::ZERO {
            return 0.0;
        }
        let dd = (self.current_price - self.high_water_mark) / self.high_water_mark;
        dd.try_into().unwrap_or(0.0)
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
    /// Price at which to sell
    pub price: Decimal,
    /// Unrealized PnL at exit
    pub unrealized_pnl: Decimal,
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
            unrealized_pnl: position.unrealized_pnl(),
            unrealized_return: position.unrealized_return(),
        }
    }

    /// Notional value of the exit.
    pub fn notional(&self) -> Decimal {
        self.price * Decimal::from(self.shares_to_sell)
    }
}

/// Context for exit evaluation.
#[derive(Debug, Clone)]
pub struct ExitContext {
    /// Current date
    pub date: NaiveDate,
    /// Total portfolio capital
    pub capital: Decimal,
    /// Current portfolio equity (NAV)
    pub equity: Decimal,
    /// Peak portfolio equity (for drawdown)
    pub peak_equity: Decimal,
    /// Market being evaluated
    pub market: Market,
}

impl ExitContext {
    pub fn new(date: NaiveDate, capital: Decimal, equity: Decimal, market: Market) -> Self {
        Self {
            date,
            capital,
            equity,
            peak_equity: equity,
            market,
        }
    }

    /// Calculate portfolio drawdown from peak.
    pub fn portfolio_drawdown(&self) -> f64 {
        if self.peak_equity == Decimal::ZERO {
            return 0.0;
        }
        let dd = (self.equity - self.peak_equity) / self.peak_equity;
        dd.try_into().unwrap_or(0.0)
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
    /// Total unrealized PnL of exits
    pub total_exit_pnl: Decimal,
    /// Total turnover from exits
    pub exit_turnover: Decimal,
    /// Estimated costs
    pub estimated_costs: Decimal,
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
        self.diagnostics.exit_turnover += exit.notional();
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
        assert_eq!(exit.unrealized_pnl, dec!(-600)); // (22-25)*200
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
        let ctx = ExitContext {
            date: NaiveDate::from_ymd_opt(2025, 1, 10).unwrap(),
            capital: dec!(1_000_000),
            equity: dec!(850_000),
            peak_equity: dec!(1_000_000),
            market: Market::BR,
        };
        let dd = ctx.portfolio_drawdown();
        assert!((dd - (-0.15)).abs() < 0.001); // -15% drawdown
    }
}






























