//! Trade Ledger with WAP (Weighted Average Price) cost basis tracking.
//!
//! The ledger tracks all trades and maintains position cost basis using
//! the weighted average price method. This is the most robust approach
//! as it doesn't require tracking individual lots.

use chrono::NaiveDate;
use rust_decimal::Decimal;
use std::collections::BTreeMap;

use crate::filters::Market;
use super::types::{PositionLot, TradeRecord, TradeSide, PnLBreakdown, CostBreakdown};

/// Trade ledger with WAP cost basis tracking.
#[derive(Debug, Clone)]
pub struct TradeLedger {
    /// All recorded trades (chronological order)
    trades: Vec<TradeRecord>,
    /// Current positions by symbol
    positions: BTreeMap<String, PositionLot>,
    /// Cumulative realized P&L
    realized_pnl: Decimal,
    /// Cumulative costs
    costs: CostBreakdown,
}

impl TradeLedger {
    pub fn new() -> Self {
        Self {
            trades: Vec::new(),
            positions: BTreeMap::new(),
            realized_pnl: Decimal::ZERO,
            costs: CostBreakdown::default(),
        }
    }

    /// Record a buy trade. Updates WAP for existing positions.
    ///
    /// WAP formula: new_wap = (old_shares * old_wap + new_shares * price) / total_shares
    pub fn record_buy(
        &mut self,
        date: NaiveDate,
        symbol: &str,
        shares: i64,
        price: Decimal,
        cost: Decimal,
        market: Market,
    ) {
        if shares <= 0 {
            return;
        }

        // Record trade
        self.trades.push(TradeRecord {
            date,
            symbol: symbol.to_string(),
            side: TradeSide::Buy,
            shares,
            price,
            cost,
            market,
            realized_pnl: None,
        });

        // Update costs
        match market {
            Market::BR => self.costs.fees_br += cost,
            Market::US => self.costs.fees_us += cost,
        }
        self.costs.total += cost;

        // Update position with WAP
        if let Some(pos) = self.positions.get_mut(symbol) {
            let old_shares = pos.shares;
            let old_wap = pos.wap_cost_basis;
            let new_shares = old_shares + shares;
            
            // WAP = (old_shares * old_wap + new_shares * price) / total_shares
            if new_shares > 0 {
                pos.wap_cost_basis = (Decimal::from(old_shares) * old_wap 
                    + Decimal::from(shares) * price) / Decimal::from(new_shares);
            }
            pos.shares = new_shares;
        } else {
            // New position
            self.positions.insert(symbol.to_string(), PositionLot::new(
                symbol.to_string(),
                shares,
                price,
                market,
                date,
            ));
        }
    }

    /// Record a sell trade. Calculates realized P&L based on WAP.
    ///
    /// Realized P&L = (sell_price - wap) * shares_sold
    pub fn record_sell(
        &mut self,
        date: NaiveDate,
        symbol: &str,
        shares: i64,
        price: Decimal,
        cost: Decimal,
        market: Market,
    ) -> Decimal {
        if shares <= 0 {
            return Decimal::ZERO;
        }

        let realized = if let Some(pos) = self.positions.get_mut(symbol) {
            let shares_to_sell = shares.min(pos.shares);
            let realized = (price - pos.wap_cost_basis) * Decimal::from(shares_to_sell);
            
            pos.shares -= shares_to_sell;
            
            // Remove position if fully closed
            if pos.shares <= 0 {
                self.positions.remove(symbol);
            }
            
            realized
        } else {
            Decimal::ZERO
        };

        // Record trade
        self.trades.push(TradeRecord {
            date,
            symbol: symbol.to_string(),
            side: TradeSide::Sell,
            shares,
            price,
            cost,
            market,
            realized_pnl: Some(realized),
        });

        // Update costs
        match market {
            Market::BR => self.costs.fees_br += cost,
            Market::US => self.costs.fees_us += cost,
        }
        self.costs.total += cost;

        self.realized_pnl += realized;
        realized
    }

    /// Get total unrealized P&L across all positions.
    pub fn get_unrealized_pnl(&self, prices: &BTreeMap<String, Decimal>) -> Decimal {
        self.positions.iter()
            .map(|(symbol, pos)| {
                prices.get(symbol)
                    .map(|&p| pos.unrealized_pnl(p))
                    .unwrap_or(Decimal::ZERO)
            })
            .sum()
    }

    /// Get unrealized P&L by symbol.
    pub fn get_unrealized_by_symbol(&self, prices: &BTreeMap<String, Decimal>) -> BTreeMap<String, Decimal> {
        self.positions.iter()
            .map(|(symbol, pos)| {
                let unrealized = prices.get(symbol)
                    .map(|&p| pos.unrealized_pnl(p))
                    .unwrap_or(Decimal::ZERO);
                (symbol.clone(), unrealized)
            })
            .collect()
    }

    /// Get full P&L breakdown.
    pub fn get_pnl_breakdown(&self, prices: &BTreeMap<String, Decimal>) -> PnLBreakdown {
        let unrealized = self.get_unrealized_pnl(prices);
        let by_symbol = self.get_unrealized_by_symbol(prices);
        
        // Aggregate by market
        let mut by_market: BTreeMap<String, Decimal> = BTreeMap::new();
        for (symbol, pos) in &self.positions {
            let pnl = prices.get(symbol)
                .map(|&p| pos.unrealized_pnl(p))
                .unwrap_or(Decimal::ZERO);
            let market_key = match pos.market {
                Market::BR => "BR".to_string(),
                Market::US => "US".to_string(),
            };
            *by_market.entry(market_key).or_default() += pnl;
        }

        PnLBreakdown {
            realized: self.realized_pnl,
            unrealized,
            total: self.realized_pnl + unrealized,
            by_market,
            by_symbol,
        }
    }

    /// Get current positions.
    pub fn positions(&self) -> &BTreeMap<String, PositionLot> {
        &self.positions
    }

    /// Get all trades.
    pub fn trades(&self) -> &[TradeRecord] {
        &self.trades
    }

    /// Get cumulative costs.
    pub fn costs(&self) -> &CostBreakdown {
        &self.costs
    }

    /// Get cumulative realized P&L.
    pub fn realized_pnl(&self) -> Decimal {
        self.realized_pnl
    }

    /// Get total market value of positions.
    pub fn market_value(&self, prices: &BTreeMap<String, Decimal>) -> Decimal {
        self.positions.iter()
            .map(|(symbol, pos)| {
                prices.get(symbol)
                    .map(|&p| pos.market_value(p))
                    .unwrap_or(Decimal::ZERO)
            })
            .sum()
    }

    /// Verify reconciliation: realized + unrealized = total P&L from trades
    pub fn verify_reconciliation(&self, prices: &BTreeMap<String, Decimal>) -> bool {
        let pnl = self.get_pnl_breakdown(prices);
        pnl.total == pnl.realized + pnl.unrealized
    }
}

impl Default for TradeLedger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn make_prices(data: &[(&str, Decimal)]) -> BTreeMap<String, Decimal> {
        data.iter().map(|(s, p)| (s.to_string(), *p)).collect()
    }

    #[test]
    fn test_wap_single_buy() {
        let mut ledger = TradeLedger::new();
        let date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
        
        ledger.record_buy(date, "PETR4", 100, dec!(30), dec!(10), Market::BR);
        
        let pos = ledger.positions().get("PETR4").unwrap();
        assert_eq!(pos.shares, 100);
        assert_eq!(pos.wap_cost_basis, dec!(30));
    }

    #[test]
    fn test_wap_multiple_buys() {
        let mut ledger = TradeLedger::new();
        let date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
        
        // First buy: 100 @ 30
        ledger.record_buy(date, "PETR4", 100, dec!(30), dec!(10), Market::BR);
        // Second buy: 100 @ 40
        ledger.record_buy(date, "PETR4", 100, dec!(40), dec!(10), Market::BR);
        
        let pos = ledger.positions().get("PETR4").unwrap();
        assert_eq!(pos.shares, 200);
        // WAP = (100*30 + 100*40) / 200 = 7000 / 200 = 35
        assert_eq!(pos.wap_cost_basis, dec!(35));
    }

    #[test]
    fn test_realized_pnl_on_sell() {
        let mut ledger = TradeLedger::new();
        let date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
        
        ledger.record_buy(date, "PETR4", 100, dec!(30), dec!(10), Market::BR);
        
        // Sell 50 @ 35 -> realized = (35-30)*50 = 250
        let realized = ledger.record_sell(date, "PETR4", 50, dec!(35), dec!(5), Market::BR);
        
        assert_eq!(realized, dec!(250));
        assert_eq!(ledger.realized_pnl(), dec!(250));
        
        let pos = ledger.positions().get("PETR4").unwrap();
        assert_eq!(pos.shares, 50);
        assert_eq!(pos.wap_cost_basis, dec!(30)); // WAP unchanged
    }

    #[test]
    fn test_full_position_close() {
        let mut ledger = TradeLedger::new();
        let date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
        
        ledger.record_buy(date, "PETR4", 100, dec!(30), dec!(10), Market::BR);
        ledger.record_sell(date, "PETR4", 100, dec!(40), dec!(5), Market::BR);
        
        assert!(ledger.positions().get("PETR4").is_none());
        assert_eq!(ledger.realized_pnl(), dec!(1000)); // (40-30)*100
    }

    #[test]
    fn test_unrealized_pnl() {
        let mut ledger = TradeLedger::new();
        let date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
        
        ledger.record_buy(date, "PETR4", 100, dec!(30), dec!(10), Market::BR);
        
        let prices = make_prices(&[("PETR4", dec!(35))]);
        let unrealized = ledger.get_unrealized_pnl(&prices);
        
        // (35-30)*100 = 500
        assert_eq!(unrealized, dec!(500));
    }

    #[test]
    fn test_reconciliation() {
        let mut ledger = TradeLedger::new();
        let date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
        
        ledger.record_buy(date, "PETR4", 100, dec!(30), dec!(10), Market::BR);
        ledger.record_buy(date, "VALE3", 200, dec!(50), dec!(15), Market::BR);
        ledger.record_sell(date, "PETR4", 50, dec!(35), dec!(5), Market::BR);
        
        let prices = make_prices(&[("PETR4", dec!(36)), ("VALE3", dec!(55))]);
        
        assert!(ledger.verify_reconciliation(&prices));
        
        let pnl = ledger.get_pnl_breakdown(&prices);
        // Realized: (35-30)*50 = 250
        // Unrealized PETR4: (36-30)*50 = 300
        // Unrealized VALE3: (55-50)*200 = 1000
        // Total = 250 + 300 + 1000 = 1550
        assert_eq!(pnl.realized, dec!(250));
        assert_eq!(pnl.unrealized, dec!(1300));
        assert_eq!(pnl.total, dec!(1550));
    }

    #[test]
    fn test_costs_by_market() {
        let mut ledger = TradeLedger::new();
        let date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
        
        ledger.record_buy(date, "PETR4", 100, dec!(30), dec!(10), Market::BR);
        ledger.record_buy(date, "AAPL", 50, dec!(150), dec!(5), Market::US);
        
        let costs = ledger.costs();
        assert_eq!(costs.fees_br, dec!(10));
        assert_eq!(costs.fees_us, dec!(5));
        assert_eq!(costs.total, dec!(15));
    }

    #[test]
    fn test_pnl_by_market() {
        let mut ledger = TradeLedger::new();
        let date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
        
        ledger.record_buy(date, "PETR4", 100, dec!(30), dec!(0), Market::BR);
        ledger.record_buy(date, "AAPL", 50, dec!(150), dec!(0), Market::US);
        
        let prices = make_prices(&[("PETR4", dec!(35)), ("AAPL", dec!(160))]);
        let pnl = ledger.get_pnl_breakdown(&prices);
        
        assert_eq!(pnl.by_market.get("BR"), Some(&dec!(500)));  // (35-30)*100
        assert_eq!(pnl.by_market.get("US"), Some(&dec!(500)));  // (160-150)*50
    }
}















