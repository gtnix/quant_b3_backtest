//! Portfolio accounting - tracks equity, cash, positions, and drawdown.
//!
//! This module provides a simplified ledger for backtest accounting.

use rust_decimal::Decimal;
use std::collections::HashMap;

use crate::exit::Position;
use crate::filters::Market;

/// Portfolio state at a point in time.
#[derive(Debug, Clone)]
pub struct PortfolioState {
    /// Cash available
    pub cash: Decimal,
    /// Positions by symbol
    pub positions: HashMap<String, Position>,
    /// Current equity (NAV = cash + mark-to-market value)
    pub equity: Decimal,
    /// Peak equity (for drawdown calculation)
    pub peak_equity: Decimal,
    /// Initial capital
    pub initial_capital: Decimal,
}

impl PortfolioState {
    /// Create new portfolio with initial capital.
    pub fn new(initial_capital: Decimal) -> Self {
        Self {
            cash: initial_capital,
            positions: HashMap::new(),
            equity: initial_capital,
            peak_equity: initial_capital,
            initial_capital,
        }
    }

    /// Calculate current equity (cash + positions mark-to-market).
    pub fn calculate_equity(&self) -> Decimal {
        let positions_value: Decimal = self.positions
            .values()
            .map(|p| p.market_value())
            .sum();
        self.cash + positions_value
    }

    /// Update equity and peak tracking.
    pub fn update_equity(&mut self) {
        self.equity = self.calculate_equity();
        if self.equity > self.peak_equity {
            self.peak_equity = self.equity;
        }
    }

    /// Calculate current drawdown from peak.
    pub fn drawdown(&self) -> f64 {
        if self.peak_equity == Decimal::ZERO {
            return 0.0;
        }
        let dd = (self.equity - self.peak_equity) / self.peak_equity;
        dd.try_into().unwrap_or(0.0)
    }

    /// Calculate drawdown from peak (as Decimal).
    pub fn drawdown_decimal(&self) -> Decimal {
        if self.peak_equity == Decimal::ZERO {
            return Decimal::ZERO;
        }
        (self.equity - self.peak_equity) / self.peak_equity
    }

    /// Get total return from initial capital.
    pub fn total_return(&self) -> f64 {
        if self.initial_capital == Decimal::ZERO {
            return 0.0;
        }
        let ret = (self.equity - self.initial_capital) / self.initial_capital;
        ret.try_into().unwrap_or(0.0)
    }

    /// Add cash to portfolio.
    pub fn add_cash(&mut self, amount: Decimal) {
        self.cash += amount;
        self.update_equity();
    }

    /// Withdraw cash from portfolio.
    pub fn withdraw_cash(&mut self, amount: Decimal) -> Result<(), AccountingError> {
        if amount > self.cash {
            return Err(AccountingError::InsufficientCash {
                requested: amount,
                available: self.cash,
            });
        }
        self.cash -= amount;
        self.update_equity();
        Ok(())
    }

    /// Add or update a position.
    pub fn set_position(&mut self, position: Position) {
        self.positions.insert(position.symbol.clone(), position);
        self.update_equity();
    }

    /// Remove a position.
    pub fn remove_position(&mut self, symbol: &str) -> Option<Position> {
        let result = self.positions.remove(symbol);
        if result.is_some() {
            self.update_equity();
        }
        result
    }

    /// Get position by symbol.
    pub fn get_position(&self, symbol: &str) -> Option<&Position> {
        self.positions.get(symbol)
    }

    /// Get all positions for a market.
    pub fn positions_for_market(&self, market: Market) -> Vec<&Position> {
        self.positions
            .values()
            .filter(|p| p.market == market)
            .collect()
    }

    /// Get positions as a simple map of symbol -> shares.
    pub fn position_shares(&self, market: Market) -> HashMap<String, i64> {
        self.positions
            .iter()
            .filter(|(_, p)| p.market == market)
            .map(|(sym, p)| (sym.clone(), p.shares))
            .collect()
    }

    /// Update all position prices (mark-to-market).
    pub fn update_prices(&mut self, prices: &HashMap<String, Decimal>) {
        for (symbol, price) in prices {
            if let Some(pos) = self.positions.get_mut(symbol) {
                pos.current_price = *price;
                pos.update_high_water_mark();
            }
        }
        self.update_equity();
    }

    /// Apply a buy order.
    pub fn apply_buy(
        &mut self,
        symbol: &str,
        shares: i64,
        price: Decimal,
        cost: Decimal,
        market: Market,
        entry_date: chrono::NaiveDate,
    ) -> Result<(), AccountingError> {
        let total_cost = price * Decimal::from(shares) + cost;
        
        if total_cost > self.cash {
            return Err(AccountingError::InsufficientCash {
                requested: total_cost,
                available: self.cash,
            });
        }

        self.cash -= total_cost;

        if let Some(pos) = self.positions.get_mut(symbol) {
            // Average in
            let old_value = pos.cost_basis * Decimal::from(pos.shares);
            let new_value = price * Decimal::from(shares);
            let new_shares = pos.shares + shares;
            pos.cost_basis = (old_value + new_value) / Decimal::from(new_shares);
            pos.shares = new_shares;
            pos.current_price = price;
        } else {
            // New position
            let pos = Position::new(symbol, market, shares, price, entry_date, price);
            self.positions.insert(symbol.to_string(), pos);
        }

        self.update_equity();
        Ok(())
    }

    /// Apply a sell order.
    pub fn apply_sell(
        &mut self,
        symbol: &str,
        shares: i64,
        price: Decimal,
        cost: Decimal,
    ) -> Result<Decimal, AccountingError> {
        let pos = self.positions.get_mut(symbol).ok_or_else(|| {
            AccountingError::PositionNotFound { symbol: symbol.to_string() }
        })?;

        if shares > pos.shares {
            return Err(AccountingError::InsufficientShares {
                symbol: symbol.to_string(),
                requested: shares,
                available: pos.shares,
            });
        }

        // Calculate realized PnL
        let proceeds = price * Decimal::from(shares);
        let cost_basis = pos.cost_basis * Decimal::from(shares);
        let realized_pnl = proceeds - cost_basis - cost;

        self.cash += proceeds - cost;
        pos.shares -= shares;

        if pos.shares == 0 {
            self.positions.remove(symbol);
        }

        self.update_equity();
        Ok(realized_pnl)
    }

    /// Validate portfolio invariants.
    pub fn validate(&self) -> Result<(), AccountingError> {
        // Cash should not be negative
        if self.cash < Decimal::ZERO {
            return Err(AccountingError::NegativeCash { cash: self.cash });
        }

        // Equity should be consistent with positions
        let calculated = self.calculate_equity();
        let diff = (calculated - self.equity).abs();
        if diff > Decimal::new(1, 2) { // 0.01 tolerance
            return Err(AccountingError::EquityMismatch {
                stored: self.equity,
                calculated,
            });
        }

        Ok(())
    }
}

/// Accounting errors.
#[derive(Debug, Clone)]
pub enum AccountingError {
    InsufficientCash {
        requested: Decimal,
        available: Decimal,
    },
    InsufficientShares {
        symbol: String,
        requested: i64,
        available: i64,
    },
    PositionNotFound {
        symbol: String,
    },
    NegativeCash {
        cash: Decimal,
    },
    EquityMismatch {
        stored: Decimal,
        calculated: Decimal,
    },
}

impl std::fmt::Display for AccountingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientCash { requested, available } => {
                write!(f, "Insufficient cash: requested {}, available {}", requested, available)
            }
            Self::InsufficientShares { symbol, requested, available } => {
                write!(f, "Insufficient shares for {}: requested {}, available {}", symbol, requested, available)
            }
            Self::PositionNotFound { symbol } => {
                write!(f, "Position not found: {}", symbol)
            }
            Self::NegativeCash { cash } => {
                write!(f, "Negative cash: {}", cash)
            }
            Self::EquityMismatch { stored, calculated } => {
                write!(f, "Equity mismatch: stored {}, calculated {}", stored, calculated)
            }
        }
    }
}

impl std::error::Error for AccountingError {}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;

    #[test]
    fn test_initial_state() {
        let portfolio = PortfolioState::new(dec!(1_000_000));
        assert_eq!(portfolio.cash, dec!(1_000_000));
        assert_eq!(portfolio.equity, dec!(1_000_000));
        assert_eq!(portfolio.drawdown(), 0.0);
    }

    #[test]
    fn test_buy_updates_equity() {
        let mut portfolio = PortfolioState::new(dec!(1_000_000));
        let date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();

        portfolio.apply_buy("PETR4", 100, dec!(50), dec!(50), Market::BR, date).unwrap();

        assert_eq!(portfolio.cash, dec!(994_950)); // 1M - (100*50 + 50)
        assert_eq!(portfolio.positions.len(), 1);
        assert_eq!(portfolio.equity, dec!(999_950)); // cash + 100*50
    }

    #[test]
    fn test_sell_updates_equity() {
        let mut portfolio = PortfolioState::new(dec!(1_000_000));
        let date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();

        portfolio.apply_buy("VALE3", 200, dec!(60), dec!(120), Market::BR, date).unwrap();
        
        // Price goes up
        let mut prices = HashMap::new();
        prices.insert("VALE3".to_string(), dec!(70));
        portfolio.update_prices(&prices);

        // Sell half
        let pnl = portfolio.apply_sell("VALE3", 100, dec!(70), dec!(70)).unwrap();
        
        assert!(pnl > Decimal::ZERO); // Should have profit
        assert_eq!(portfolio.positions.get("VALE3").unwrap().shares, 100);
    }

    #[test]
    fn test_drawdown_calculation() {
        let mut portfolio = PortfolioState::new(dec!(1_000_000));
        let date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();

        // Buy position
        portfolio.apply_buy("PETR4", 10000, dec!(50), dec!(0), Market::BR, date).unwrap();
        
        // Peak at 1M (initial)
        assert_eq!(portfolio.peak_equity, dec!(1_000_000));

        // Price drops 20%
        let mut prices = HashMap::new();
        prices.insert("PETR4".to_string(), dec!(40));
        portfolio.update_prices(&prices);

        // Equity = 500_000 (cash) + 10000*40 = 900_000
        assert_eq!(portfolio.equity, dec!(900_000));
        
        // Drawdown = (900k - 1M) / 1M = -10%
        let dd = portfolio.drawdown();
        assert!((dd - (-0.10)).abs() < 0.001);
    }

    #[test]
    fn test_validate_catches_negative_cash() {
        let mut portfolio = PortfolioState::new(dec!(1_000_000));
        portfolio.cash = dec!(-100);
        
        let result = portfolio.validate();
        assert!(matches!(result, Err(AccountingError::NegativeCash { .. })));
    }

    #[test]
    fn test_insufficient_cash() {
        let mut portfolio = PortfolioState::new(dec!(1_000));
        let date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();

        let result = portfolio.apply_buy("PETR4", 100, dec!(50), dec!(0), Market::BR, date);
        
        assert!(matches!(result, Err(AccountingError::InsufficientCash { .. })));
    }

    #[test]
    fn test_peak_tracking() {
        let mut portfolio = PortfolioState::new(dec!(100_000));
        let date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();

        portfolio.apply_buy("VALE3", 1000, dec!(50), dec!(0), Market::BR, date).unwrap();
        
        // Price goes up - peak should update
        let mut prices = HashMap::new();
        prices.insert("VALE3".to_string(), dec!(60));
        portfolio.update_prices(&prices);
        assert_eq!(portfolio.peak_equity, dec!(110_000)); // 50k cash + 60k position

        // Price goes down - peak should NOT change
        prices.insert("VALE3".to_string(), dec!(55));
        portfolio.update_prices(&prices);
        assert_eq!(portfolio.peak_equity, dec!(110_000)); // Still at previous peak
        assert_eq!(portfolio.equity, dec!(105_000)); // Current
    }
}

