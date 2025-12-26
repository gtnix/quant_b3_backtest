//! Audit logging for exit decisions.

use chrono::NaiveDate;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::fmt::Write;

use crate::entry::Order;
use crate::filters::Market;

use super::types::{ExitDiagnostics, ExitReason, ExitResult, RiskViolation};

/// Record of an exited position.
#[derive(Debug, Clone)]
pub struct ExitedPosition {
    pub symbol: String,
    pub shares: i64,
    pub reason: ExitReason,
    pub pnl: Decimal,
    pub return_pct: f64,
}

/// Exit audit log with decision trail.
#[derive(Debug, Clone)]
pub struct ExitAuditLog {
    pub date: NaiveDate,
    pub market: Market,
    pub exits: Vec<ExitedPosition>,
    pub orders: Vec<Order>,
    pub diagnostics: ExitDiagnostics,
}

impl ExitAuditLog {
    /// Create audit log from exit result.
    pub fn from_result(result: &ExitResult, orders: &[Order]) -> Self {
        let exits: Vec<ExitedPosition> = result
            .exits
            .iter()
            .map(|e| ExitedPosition {
                symbol: e.symbol.clone(),
                shares: e.shares_to_sell,
                reason: e.reason,
                pnl: e.unrealized_pnl,
                return_pct: e.unrealized_return,
            })
            .collect();

        Self {
            date: result.date,
            market: result.market,
            exits,
            orders: orders.to_vec(),
            diagnostics: result.diagnostics.clone(),
        }
    }

    /// Generate human-readable summary.
    pub fn to_summary(&self) -> String {
        let mut out = String::new();

        // Header
        let market_str = match self.market {
            Market::BR => "BR",
            Market::US => "US",
        };
        writeln!(out, "=== EXIT AUDIT {} ({}) ===", self.date, market_str).unwrap();
        writeln!(out).unwrap();

        // Exits by reason
        if !self.exits.is_empty() {
            writeln!(out, "SAÍDAS ({}):", self.exits.len()).unwrap();
            
            // Group by reason - use sorted order for determinism
            let mut by_reason: Vec<(ExitReason, Vec<&ExitedPosition>)> = Vec::new();
            let reasons = [
                ExitReason::StopLoss,
                ExitReason::TakeProfit,
                ExitReason::TimeExit,
                ExitReason::TrailingStop,
                ExitReason::RiskCap,
                ExitReason::DrawdownGuard,
                ExitReason::Rebalance,
                ExitReason::Manual,
            ];

            for reason in reasons {
                let exits: Vec<&ExitedPosition> = self.exits.iter().filter(|e| e.reason == reason).collect();
                if !exits.is_empty() {
                    by_reason.push((reason, exits));
                }
            }

            for (reason, exits) in by_reason {
                writeln!(out, "  [{}] ({}):", reason, exits.len()).unwrap();
                for exit in exits.iter().take(5) {
                    writeln!(
                        out,
                        "    {}: {} ações, PnL: {}, ret: {:.1}%",
                        exit.symbol, exit.shares, exit.pnl, exit.return_pct * 100.0
                    ).unwrap();
                }
                if exits.len() > 5 {
                    writeln!(out, "    ... e mais {}", exits.len() - 5).unwrap();
                }
            }
            writeln!(out).unwrap();
        }

        // Orders
        if !self.orders.is_empty() {
            writeln!(out, "ORDENS DE VENDA ({}):", self.orders.len()).unwrap();
            for order in &self.orders {
                writeln!(
                    out,
                    "  VENDA {} x {} @ {} (custo: {})",
                    order.symbol, order.shares, order.price, order.estimated_cost
                ).unwrap();
            }
            writeln!(out).unwrap();
        }

        // Risk violations
        if !self.diagnostics.risk_violations.is_empty() {
            writeln!(out, "VIOLAÇÕES DE RISCO:").unwrap();
            for violation in &self.diagnostics.risk_violations {
                writeln!(out, "  ⚠ {}", violation).unwrap();
            }
            writeln!(out).unwrap();
        }

        // Metrics
        writeln!(out, "MÉTRICAS:").unwrap();
        writeln!(out, "  Posições avaliadas: {}", self.diagnostics.positions_evaluated).unwrap();
        writeln!(out, "  Saídas: {}", self.diagnostics.positions_exited).unwrap();
        writeln!(out, "  Stop-loss: {}", self.diagnostics.stop_loss_count).unwrap();
        writeln!(out, "  Take-profit: {}", self.diagnostics.take_profit_count).unwrap();
        writeln!(out, "  Time exit: {}", self.diagnostics.time_exit_count).unwrap();
        writeln!(out, "  Trailing stop: {}", self.diagnostics.trailing_stop_count).unwrap();
        writeln!(out, "  Drawdown guard: {}", self.diagnostics.drawdown_guard_count).unwrap();
        writeln!(out, "  PnL total saídas: {}", self.diagnostics.total_exit_pnl).unwrap();
        writeln!(out, "  Turnover saídas: {}", self.diagnostics.exit_turnover).unwrap();
        writeln!(out, "  Custos estimados: {}", self.diagnostics.estimated_costs).unwrap();

        out
    }

    /// Generate compact one-line summary.
    pub fn to_compact(&self) -> String {
        format!(
            "[{}|{}] exits={} sl={} tp={} pnl={}",
            self.date,
            match self.market {
                Market::BR => "BR",
                Market::US => "US",
            },
            self.diagnostics.positions_exited,
            self.diagnostics.stop_loss_count,
            self.diagnostics.take_profit_count,
            self.diagnostics.total_exit_pnl
        )
    }

    /// Get exit counts by reason (machine-readable).
    pub fn exit_counts_by_reason(&self) -> HashMap<ExitReason, usize> {
        let mut counts = HashMap::new();
        for exit in &self.exits {
            *counts.entry(exit.reason).or_insert(0) += 1;
        }
        counts
    }

    /// Get total PnL from all exits.
    pub fn total_pnl(&self) -> Decimal {
        self.diagnostics.total_exit_pnl
    }

    /// Check if there are any risk violations.
    pub fn has_violations(&self) -> bool {
        !self.diagnostics.risk_violations.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entry::OrderSide;
    use rust_decimal_macros::dec;

    fn make_sample_log() -> ExitAuditLog {
        ExitAuditLog {
            date: NaiveDate::from_ymd_opt(2025, 1, 10).unwrap(),
            market: Market::BR,
            exits: vec![
                ExitedPosition {
                    symbol: "PETR4".to_string(),
                    shares: 500,
                    reason: ExitReason::StopLoss,
                    pnl: dec!(-2500),
                    return_pct: -0.10,
                },
                ExitedPosition {
                    symbol: "VALE3".to_string(),
                    shares: 300,
                    reason: ExitReason::TakeProfit,
                    pnl: dec!(4500),
                    return_pct: 0.25,
                },
            ],
            orders: vec![
                Order::new("PETR4".to_string(), OrderSide::Sell, 500, dec!(45), dec!(225)),
                Order::new("VALE3".to_string(), OrderSide::Sell, 300, dec!(75), dec!(225)),
            ],
            diagnostics: ExitDiagnostics {
                positions_evaluated: 10,
                positions_exited: 2,
                stop_loss_count: 1,
                take_profit_count: 1,
                total_exit_pnl: dec!(2000),
                exit_turnover: dec!(45000),
                estimated_costs: dec!(450),
                ..Default::default()
            },
        }
    }

    #[test]
    fn test_summary_generation() {
        let log = make_sample_log();
        let summary = log.to_summary();

        assert!(summary.contains("EXIT AUDIT"));
        assert!(summary.contains("PETR4"));
        assert!(summary.contains("VALE3"));
        assert!(summary.contains("stop-loss"));
        assert!(summary.contains("take-profit"));
    }

    #[test]
    fn test_compact_generation() {
        let log = make_sample_log();
        let compact = log.to_compact();

        assert!(compact.contains("2025-01-10"));
        assert!(compact.contains("BR"));
        assert!(compact.contains("exits=2"));
    }

    #[test]
    fn test_exit_counts_by_reason() {
        let log = make_sample_log();
        let counts = log.exit_counts_by_reason();

        assert_eq!(counts.get(&ExitReason::StopLoss), Some(&1));
        assert_eq!(counts.get(&ExitReason::TakeProfit), Some(&1));
        assert_eq!(counts.get(&ExitReason::TimeExit), None);
    }
}



