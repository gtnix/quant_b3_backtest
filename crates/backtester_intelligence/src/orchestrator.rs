//! Rebalance Orchestrator - combines Entry and Exit in same step with netting.
//!
//! # Order of Operations
//! 1. Exit evaluation runs FIRST (frees cash from positions)
//! 2. Entry evaluation runs AFTER (uses freed cash)
//! 3. Orders are netted: if same symbol has BUY and SELL, net to single order
//! 4. Costs charged on NET order only (no double-charging)

use std::collections::HashMap;

use chrono::NaiveDate;
use rust_decimal::Decimal;

use crate::entry::{
    AssetCandidate, EntryContext, EntryEngine, EntryEngineConfig, EntryResult, 
    Order, OrderSide, RebalanceAuditLog,
};
use crate::exit::{
    ExitAuditLog, ExitContext, ExitEngine, ExitEngineConfig, ExitResult, Position,
};
use crate::filters::Market;

/// Combined audit log for a rebalance step.
#[derive(Debug, Clone)]
pub struct RebalanceStepAudit {
    pub date: NaiveDate,
    pub market: Market,
    pub exit_audit: ExitAuditLog,
    pub entry_audit: RebalanceAuditLog,
    pub net_orders: Vec<Order>,
    pub netting_applied: usize,
    pub total_cost: Decimal,
    pub cash_before: Decimal,
    pub cash_after: Decimal,
}

/// Result of a full rebalance step (exit + entry + netting).
#[derive(Debug, Clone)]
pub struct RebalanceStepResult {
    pub date: NaiveDate,
    pub market: Market,
    pub exit_result: ExitResult,
    pub entry_result: EntryResult,
    pub net_orders: Vec<Order>,
    pub netting_count: usize,
}

/// Configuration for the orchestrator.
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    pub entry: EntryEngineConfig,
    pub exit: ExitEngineConfig,
    /// Cost in bps for BR market
    pub br_cost_bps: f64,
    /// Cost in bps for US market  
    pub us_cost_bps: f64,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            entry: EntryEngineConfig::default(),
            exit: ExitEngineConfig::default(),
            br_cost_bps: 10.0,
            us_cost_bps: 5.0,
        }
    }
}

/// Rebalance Orchestrator - coordinates Entry and Exit with netting.
pub struct RebalanceOrchestrator {
    entry_engine: EntryEngine,
    exit_engine: ExitEngine,
    config: OrchestratorConfig,
}

impl RebalanceOrchestrator {
    pub fn new(config: OrchestratorConfig) -> Self {
        Self {
            entry_engine: EntryEngine::new(config.entry.clone()),
            exit_engine: ExitEngine::new(config.exit.clone()),
            config,
        }
    }

    /// Execute a full rebalance step for a single market.
    ///
    /// Order of operations:
    /// 1. Evaluate exits on current positions
    /// 2. Calculate cash freed from exits
    /// 3. Evaluate entries with updated cash
    /// 4. Net orders (combine BUY/SELL on same symbol)
    /// 5. Calculate costs on net orders only
    pub fn execute_rebalance(
        &self,
        date: NaiveDate,
        market: Market,
        positions: &[Position],
        candidates: Vec<AssetCandidate>,
        initial_cash: Decimal,
        equity: Decimal,
        peak_equity: Decimal,
    ) -> (RebalanceStepResult, RebalanceStepAudit) {
        // Build position map for entry engine
        let position_map: HashMap<String, i64> = positions
            .iter()
            .filter(|p| p.market == market)
            .map(|p| (p.symbol.clone(), p.shares))
            .collect();

        // Step 1: Evaluate exits
        let exit_ctx = ExitContext {
            date,
            capital: equity,
            equity,
            peak_equity,
            market,
        };
        let (exit_result, exit_orders, exit_audit) = 
            self.exit_engine.evaluate(positions, &exit_ctx);

        // Step 2: Calculate cash freed from exits
        let cash_from_exits: Decimal = exit_orders
            .iter()
            .map(|o| o.price * Decimal::from(o.shares) - o.estimated_cost)
            .sum();
        
        let available_cash = initial_cash + cash_from_exits;

        // Step 3: Evaluate entries with updated cash
        let entry_ctx = EntryContext::new(date, available_cash, market);
        let (entry_result, entry_orders, entry_audit) = 
            self.entry_engine.evaluate(&entry_ctx, candidates, &position_map);

        // Step 4: Net orders
        let (net_orders, netting_count) = self.net_orders(&exit_orders, &entry_orders, market);

        // Step 5: Calculate final cash
        let orders_impact: Decimal = net_orders
            .iter()
            .map(|o| {
                let notional = o.price * Decimal::from(o.shares);
                match o.side {
                    OrderSide::Buy => -notional - o.estimated_cost,
                    OrderSide::Sell => notional - o.estimated_cost,
                }
            })
            .sum();
        
        let cash_after = initial_cash + orders_impact;
        let total_cost: Decimal = net_orders.iter().map(|o| o.estimated_cost).sum();

        // Build results
        let result = RebalanceStepResult {
            date,
            market,
            exit_result,
            entry_result,
            net_orders: net_orders.clone(),
            netting_count,
        };

        let audit = RebalanceStepAudit {
            date,
            market,
            exit_audit,
            entry_audit,
            net_orders,
            netting_applied: netting_count,
            total_cost,
            cash_before: initial_cash,
            cash_after,
        };

        (result, audit)
    }

    /// Net orders: combine BUY and SELL for same symbol into single order.
    ///
    /// Rules:
    /// - If SELL 500 + BUY 300 = SELL 200 (net)
    /// - If BUY 500 + SELL 300 = BUY 200 (net)  
    /// - If SELL 500 + BUY 500 = no order (cancel out)
    /// - Costs calculated on NET order only
    fn net_orders(
        &self,
        exit_orders: &[Order],
        entry_orders: &[Order],
        market: Market,
    ) -> (Vec<Order>, usize) {
        let mut order_map: HashMap<String, (i64, Decimal)> = HashMap::new();
        let mut netting_count = 0;

        // Add exit orders (SELL = negative shares)
        for order in exit_orders {
            let entry = order_map.entry(order.symbol.clone()).or_insert((0, order.price));
            entry.0 -= order.shares as i64; // SELL reduces position
        }

        // Add entry orders (BUY = positive shares)
        for order in entry_orders {
            let entry = order_map.entry(order.symbol.clone()).or_insert((0, order.price));
            let had_opposite = entry.0 != 0 && (entry.0 > 0) != (order.shares > 0);
            entry.0 += order.shares as i64; // BUY increases position
            if had_opposite {
                netting_count += 1;
            }
        }

        // Convert to net orders
        let cost_bps = match market {
            Market::BR => self.config.br_cost_bps,
            Market::US => self.config.us_cost_bps,
        };

        let net_orders: Vec<Order> = order_map
            .into_iter()
            .filter(|(_, (shares, _))| *shares != 0)
            .map(|(symbol, (net_shares, price))| {
                let (side, shares) = if net_shares > 0 {
                    (OrderSide::Buy, net_shares as i64)
                } else {
                    (OrderSide::Sell, (-net_shares) as i64)
                };

                let notional = price * Decimal::from(shares);
                let cost = notional * Decimal::try_from(cost_bps / 10000.0)
                    .unwrap_or(Decimal::ZERO);

                Order::new(symbol, side, shares, price, cost)
            })
            .collect();

        (net_orders, netting_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn make_position(symbol: &str, shares: i64, cost: Decimal, current: Decimal) -> Position {
        Position::new(
            symbol,
            Market::BR,
            shares,
            cost,
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            current,
        )
    }

    fn make_candidate(symbol: &str, price: Decimal, score: f64) -> AssetCandidate {
        AssetCandidate {
            symbol: symbol.to_string(),
            market: Market::BR,
            price: Some(price),
            avg_volume: Some(dec!(1_000_000)),
            volatility: Some(0.20),
            score: Some(score),
            filter_scores: Vec::new(),
            has_fundamentals: true,
            has_dividends: true,
            is_tradeable: true,
            price_days: 252,
            fundamentals_as_of: None,
        }
    }

    #[test]
    fn test_netting_sell_and_buy_same_symbol() {
        let orchestrator = RebalanceOrchestrator::new(OrchestratorConfig::default());

        let exit_orders = vec![
            Order::new("PETR4".to_string(), OrderSide::Sell, 500, dec!(50), dec!(25)),
        ];
        let entry_orders = vec![
            Order::new("PETR4".to_string(), OrderSide::Buy, 300, dec!(50), dec!(15)),
        ];

        let (net, netting_count) = orchestrator.net_orders(&exit_orders, &entry_orders, Market::BR);

        assert_eq!(netting_count, 1);
        assert_eq!(net.len(), 1);
        assert_eq!(net[0].symbol, "PETR4");
        assert_eq!(net[0].side, OrderSide::Sell);
        assert_eq!(net[0].shares, 200); // 500 - 300 = 200
    }

    #[test]
    fn test_netting_cancel_out() {
        let orchestrator = RebalanceOrchestrator::new(OrchestratorConfig::default());

        let exit_orders = vec![
            Order::new("VALE3".to_string(), OrderSide::Sell, 400, dec!(60), dec!(24)),
        ];
        let entry_orders = vec![
            Order::new("VALE3".to_string(), OrderSide::Buy, 400, dec!(60), dec!(24)),
        ];

        let (net, netting_count) = orchestrator.net_orders(&exit_orders, &entry_orders, Market::BR);

        assert_eq!(netting_count, 1);
        assert!(net.is_empty()); // Cancelled out
    }

    #[test]
    fn test_netting_no_conflict() {
        let orchestrator = RebalanceOrchestrator::new(OrchestratorConfig::default());

        let exit_orders = vec![
            Order::new("PETR4".to_string(), OrderSide::Sell, 500, dec!(50), dec!(25)),
        ];
        let entry_orders = vec![
            Order::new("ITUB4".to_string(), OrderSide::Buy, 300, dec!(30), dec!(9)),
        ];

        let (net, netting_count) = orchestrator.net_orders(&exit_orders, &entry_orders, Market::BR);

        assert_eq!(netting_count, 0);
        assert_eq!(net.len(), 2); // Two separate orders
    }

    #[test]
    fn test_cost_on_net_order_only() {
        let orchestrator = RebalanceOrchestrator::new(OrchestratorConfig {
            br_cost_bps: 10.0, // 0.1%
            ..Default::default()
        });

        let exit_orders = vec![
            Order::new("PETR4".to_string(), OrderSide::Sell, 1000, dec!(100), dec!(100)),
        ];
        let entry_orders = vec![
            Order::new("PETR4".to_string(), OrderSide::Buy, 600, dec!(100), dec!(60)),
        ];

        let (net, _) = orchestrator.net_orders(&exit_orders, &entry_orders, Market::BR);

        // Net: SELL 400 @ 100 = 40000 notional
        // Cost should be 40000 * 0.001 = 40
        assert_eq!(net.len(), 1);
        assert_eq!(net[0].shares, 400);
        assert_eq!(net[0].estimated_cost, dec!(40)); // Cost on net, not original orders
    }

    #[test]
    fn test_buy_wins_over_sell() {
        let orchestrator = RebalanceOrchestrator::new(OrchestratorConfig::default());

        let exit_orders = vec![
            Order::new("WEGE3".to_string(), OrderSide::Sell, 200, dec!(40), dec!(8)),
        ];
        let entry_orders = vec![
            Order::new("WEGE3".to_string(), OrderSide::Buy, 500, dec!(40), dec!(20)),
        ];

        let (net, netting_count) = orchestrator.net_orders(&exit_orders, &entry_orders, Market::BR);

        assert_eq!(netting_count, 1);
        assert_eq!(net.len(), 1);
        assert_eq!(net[0].side, OrderSide::Buy);
        assert_eq!(net[0].shares, 300); // 500 - 200 = 300 BUY
    }
}

