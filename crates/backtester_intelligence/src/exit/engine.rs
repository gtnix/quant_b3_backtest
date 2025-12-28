//! Exit engine orchestrator.

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use super::audit::ExitAuditLog;
use super::policy::{ExitPolicy, ExitPolicyConfig};
use super::risk_guard::{RiskConfig, RiskGuard};
use super::stop_loss::{StopLossConfig, StopLossPolicy};
use super::take_profit::{TakeProfitConfig, TakeProfitPolicy};
use super::time_exit::{TimeExitConfig, TimeExitPolicy};
use super::trailing_stop::{TrailingStopConfig, TrailingStopPolicy};
use super::types::{
    DrawdownAction, ExitContext, ExitReason, ExitResult, ExitTarget, Position,
};
use crate::entry::Order;
use crate::filters::Market;

/// Exit engine configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExitEngineConfig {
    /// Exit policy configuration
    #[serde(default)]
    pub policy: ExitPolicyConfig,

    /// Stop-loss configuration
    #[serde(default)]
    pub stop_loss: StopLossConfig,

    /// Take-profit configuration
    #[serde(default)]
    pub take_profit: TakeProfitConfig,

    /// Time-based exit configuration
    #[serde(default)]
    pub time_exit: TimeExitConfig,

    /// Trailing stop configuration
    #[serde(default)]
    pub trailing_stop: TrailingStopConfig,

    /// Risk guard configuration
    #[serde(default)]
    pub risk: RiskConfig,

    /// BR market cost in bps
    #[serde(default = "default_br_cost")]
    pub br_cost_bps: f64,

    /// US market cost in bps
    #[serde(default = "default_us_cost")]
    pub us_cost_bps: f64,

    /// BR lot size
    #[serde(default = "default_br_lot")]
    pub br_lot_size: i64,

    /// US min shares
    #[serde(default = "default_us_min")]
    pub us_min_shares: i64,
}

fn default_br_cost() -> f64 { 10.0 }
fn default_us_cost() -> f64 { 5.0 }
fn default_br_lot() -> i64 { 100 }
fn default_us_min() -> i64 { 1 }

impl Default for ExitEngineConfig {
    fn default() -> Self {
        Self {
            policy: ExitPolicyConfig::default(),
            stop_loss: StopLossConfig::default(),
            take_profit: TakeProfitConfig::default(),
            time_exit: TimeExitConfig::default(),
            trailing_stop: TrailingStopConfig::default(),
            risk: RiskConfig::default(),
            br_cost_bps: default_br_cost(),
            us_cost_bps: default_us_cost(),
            br_lot_size: default_br_lot(),
            us_min_shares: default_us_min(),
        }
    }
}

/// Exit engine - evaluates positions and generates exit orders.
pub struct ExitEngine {
    config: ExitEngineConfig,
    stop_loss: StopLossPolicy,
    take_profit: TakeProfitPolicy,
    time_exit: TimeExitPolicy,
    trailing_stop: TrailingStopPolicy,
    risk_guard: RiskGuard,
}

impl ExitEngine {
    pub fn new(config: ExitEngineConfig) -> Self {
        let stop_loss = StopLossPolicy::new(StopLossConfig {
            threshold_pct: config.policy.stop_loss_pct,
            use_close_price: config.policy.use_close_price,
            enabled: config.policy.enable_stop_loss,
        });

        let take_profit = TakeProfitPolicy::new(TakeProfitConfig {
            threshold_pct: config.policy.take_profit_pct,
            use_close_price: config.policy.use_close_price,
            enabled: config.policy.enable_take_profit,
        });

        let time_exit = TimeExitPolicy::new(TimeExitConfig {
            max_holding_days: config.policy.max_holding_days,
            enabled: config.policy.enable_time_exit && config.policy.max_holding_days > 0,
        });

        let trailing_stop = TrailingStopPolicy::new(TrailingStopConfig {
            trail_pct: config.policy.trailing_stop_pct,
            activation_gain_pct: config.policy.trailing_activation_pct,
            enabled: config.policy.enable_trailing_stop,
        });

        let risk_guard = RiskGuard::new(config.risk.clone());

        Self {
            config,
            stop_loss,
            take_profit,
            time_exit,
            trailing_stop,
            risk_guard,
        }
    }

    /// Evaluate positions and generate exit targets.
    pub fn evaluate(
        &self,
        positions: &[Position],
        context: &ExitContext,
    ) -> (ExitResult, Vec<Order>, ExitAuditLog) {
        let mut result = ExitResult::new(context.date, context.market);
        result.diagnostics.positions_evaluated = positions.len();

        // Step 1: Evaluate individual exit policies for each position
        let policies: Vec<&dyn ExitPolicy> = vec![
            &self.stop_loss,
            &self.take_profit,
            &self.time_exit,
            &self.trailing_stop,
        ];

        for position in positions {
            // Check each policy in order (first trigger wins)
            for policy in &policies {
                if policy.is_enabled() {
                    if let Some(exit) = policy.evaluate(position, context) {
                        result.add_exit(exit);
                        break; // Only one exit reason per position
                    }
                }
            }
        }

        // Step 2: Check portfolio-level risk
        let turnover = result.diagnostics.exit_turnover;
        let violations = self.risk_guard.run_all_checks(positions, context, turnover);

        // Handle drawdown violation
        if violations.iter().any(|v| matches!(v, super::types::RiskViolation::DrawdownExceeded)) {
            match self.risk_guard.drawdown_action() {
                DrawdownAction::CashOut => {
                    // Exit all remaining positions
                    for position in positions {
                        if !result.exits.iter().any(|e| e.symbol == position.symbol) {
                            result.add_exit(ExitTarget::from_position(
                                position,
                                ExitReason::DrawdownGuard,
                                None,
                            ));
                        }
                    }
                }
                DrawdownAction::ReduceRisk => {
                    // Exit highest-risk positions (highest weight)
                    let mut remaining: Vec<_> = positions
                        .iter()
                        .filter(|p| !result.exits.iter().any(|e| e.symbol == p.symbol))
                        .collect();
                    remaining.sort_by(|a, b| b.market_value().cmp(&a.market_value()));

                    // Exit top 20% by value
                    let exit_count = (remaining.len() / 5).max(1);
                    for pos in remaining.into_iter().take(exit_count) {
                        result.add_exit(ExitTarget::from_position(
                            pos,
                            ExitReason::DrawdownGuard,
                            None,
                        ));
                    }
                }
                DrawdownAction::Alert => {
                    // Just log, no action
                }
            }
        }

        result.diagnostics.risk_violations = violations;

        // Step 3: Generate sell orders
        let orders = self.generate_orders(&result.exits, context.market);

        // Calculate costs
        result.diagnostics.estimated_costs = orders.iter().map(|o| o.estimated_cost).sum();

        // Step 4: Build audit log
        let audit = ExitAuditLog::from_result(&result, &orders);

        (result, orders, audit)
    }

    /// Generate sell orders from exit targets.
    fn generate_orders(&self, exits: &[ExitTarget], market: Market) -> Vec<Order> {
        exits
            .iter()
            .filter(|e| e.shares_to_sell > 0)
            .map(|exit| {
                let shares = self.round_to_lot(exit.shares_to_sell, exit.market);
                if shares == 0 {
                    return None;
                }

                let cost_bps = match exit.market {
                    Market::BR => self.config.br_cost_bps,
                    Market::US => self.config.us_cost_bps,
                };

                let notional = exit.price * Decimal::from(shares);
                let cost = notional * Decimal::try_from(cost_bps / 10000.0).unwrap_or(Decimal::ZERO);

                Some(Order::new(
                    exit.symbol.clone(),
                    crate::entry::OrderSide::Sell,
                    shares,
                    exit.price,
                    cost,
                ))
            })
            .flatten()
            .collect()
    }

    /// Round shares to lot size.
    fn round_to_lot(&self, shares: i64, market: Market) -> i64 {
        match market {
            Market::BR => {
                let lot = self.config.br_lot_size;
                (shares / lot) * lot
            }
            Market::US => {
                shares.max(self.config.us_min_shares)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;

    fn make_context() -> ExitContext {
        ExitContext::new(
            NaiveDate::from_ymd_opt(2025, 1, 10).unwrap(),
            dec!(1_000_000),
            dec!(1_000_000),
            Market::BR,
        )
    }

    #[test]
    fn test_engine_evaluates_stop_loss() {
        let config = ExitEngineConfig {
            policy: ExitPolicyConfig {
                enable_stop_loss: true,
                stop_loss_pct: -0.10,
                ..Default::default()
            },
            ..Default::default()
        };
        let engine = ExitEngine::new(config);
        let ctx = make_context();

        let positions = vec![
            Position::new("PETR4", Market::BR, 500, dec!(50), 
                NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(40)), // -20%
            Position::new("VALE3", Market::BR, 300, dec!(60), 
                NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(65)), // +8%
        ];

        let (result, orders, _) = engine.evaluate(&positions, &ctx);

        assert_eq!(result.exits.len(), 1);
        assert_eq!(result.exits[0].symbol, "PETR4");
        assert_eq!(result.exits[0].reason, ExitReason::StopLoss);
        assert_eq!(orders.len(), 1);
    }

    #[test]
    fn test_engine_evaluates_take_profit() {
        let config = ExitEngineConfig {
            policy: ExitPolicyConfig {
                enable_take_profit: true,
                take_profit_pct: 0.25,
                ..Default::default()
            },
            ..Default::default()
        };
        let engine = ExitEngine::new(config);
        let ctx = make_context();

        let positions = vec![
            Position::new("ITUB4", Market::BR, 400, dec!(30), 
                NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(40)), // +33%
        ];

        let (result, _, _) = engine.evaluate(&positions, &ctx);

        assert_eq!(result.exits.len(), 1);
        assert_eq!(result.exits[0].reason, ExitReason::TakeProfit);
    }

    #[test]
    fn test_engine_no_exits() {
        let engine = ExitEngine::new(ExitEngineConfig::default());
        let ctx = make_context();

        let positions = vec![
            Position::new("WEGE3", Market::BR, 200, dec!(40), 
                NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(42)), // +5%
        ];

        let (result, orders, _) = engine.evaluate(&positions, &ctx);

        assert!(result.exits.is_empty());
        assert!(orders.is_empty());
    }

    #[test]
    fn test_engine_order_generation_br_lot() {
        let engine = ExitEngine::new(ExitEngineConfig::default());
        let ctx = make_context();

        let positions = vec![
            Position::new("PETR4", Market::BR, 550, dec!(50), 
                NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(40)), // 550 shares
        ];

        let (_, orders, _) = engine.evaluate(&positions, &ctx);

        // 550 should round to 500 (BR lot = 100)
        assert_eq!(orders[0].shares, 500);
    }

    #[test]
    fn test_drawdown_guard_cash_out() {
        let config = ExitEngineConfig {
            risk: RiskConfig {
                max_drawdown_pct: -0.10,
                drawdown_action: DrawdownAction::CashOut,
                ..Default::default()
            },
            ..Default::default()
        };
        let engine = ExitEngine::new(config);

        // 20% drawdown
        let ctx = ExitContext {
            date: NaiveDate::from_ymd_opt(2025, 1, 10).unwrap(),
            capital: dec!(1_000_000),
            equity: dec!(800_000),
            peak_equity: dec!(1_000_000),
            market: Market::BR,
        };

        let positions = vec![
            Position::new("PETR4", Market::BR, 500, dec!(50), 
                NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(52)),
            Position::new("VALE3", Market::BR, 300, dec!(60), 
                NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(62)),
        ];

        let (result, _, _) = engine.evaluate(&positions, &ctx);

        // Both positions should be exited due to drawdown guard
        assert_eq!(result.exits.len(), 2);
        assert!(result.exits.iter().all(|e| e.reason == ExitReason::DrawdownGuard));
    }
}








