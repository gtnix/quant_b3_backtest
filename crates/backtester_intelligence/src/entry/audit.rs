//! Audit logging for entry decisions.

use chrono::NaiveDate;
use std::collections::HashMap;
use std::fmt::Write;

use crate::filters::Market;
use super::types::{EntryDiagnostics, EntryExclusion, ExclusionReason, Order, OrderSide};

/// Selected asset for audit log.
#[derive(Debug, Clone)]
pub struct SelectedAsset {
    pub symbol: String,
    pub weight: f64,
    pub score: f64,
    pub reason: String,
}

/// Rebalance audit log with full decision trail.
#[derive(Debug, Clone)]
pub struct RebalanceAuditLog {
    pub date: NaiveDate,
    pub market: Market,
    pub selected: Vec<SelectedAsset>,
    pub excluded: Vec<EntryExclusion>,
    pub orders: Vec<Order>,
    pub diagnostics: EntryDiagnostics,
}

impl RebalanceAuditLog {
    /// Generate human-readable summary.
    pub fn to_summary(&self) -> String {
        let mut out = String::new();

        // Header
        let market_str = match self.market {
            Market::BR => "BR",
            Market::US => "US",
        };
        writeln!(out, "=== REBALANCE {} ({}) ===", self.date, market_str).unwrap();
        writeln!(out).unwrap();

        // Selected
        writeln!(out, "SELECIONADOS ({}):", self.selected.len()).unwrap();
        for asset in &self.selected {
            writeln!(
                out,
                "  {}: {:.1}% (score={:.3}, {})",
                asset.symbol,
                asset.weight * 100.0,
                asset.score,
                asset.reason
            ).unwrap();
        }
        writeln!(out).unwrap();

        // Excluded summary by stage
        let gating_excluded: Vec<_> = self.excluded.iter()
            .filter(|e| matches!(e.stage, super::types::ExclusionStage::Gating))
            .collect();
        let selection_excluded: Vec<_> = self.excluded.iter()
            .filter(|e| matches!(e.stage, super::types::ExclusionStage::Selection))
            .collect();

        if !gating_excluded.is_empty() {
            writeln!(out, "EXCLUÍDOS GATING ({}):", gating_excluded.len()).unwrap();
            for exc in gating_excluded.iter().take(5) {
                writeln!(out, "  {}: {}", exc.symbol, exc.reason).unwrap();
            }
            if gating_excluded.len() > 5 {
                writeln!(out, "  ... e mais {}", gating_excluded.len() - 5).unwrap();
            }
            writeln!(out).unwrap();
        }

        if !selection_excluded.is_empty() {
            writeln!(out, "EXCLUÍDOS SELEÇÃO ({}):", selection_excluded.len()).unwrap();
            for exc in selection_excluded.iter().take(5) {
                if let Some(score) = exc.score {
                    writeln!(out, "  {}: {} (score={:.3})", exc.symbol, exc.reason, score).unwrap();
                } else {
                    writeln!(out, "  {}: {}", exc.symbol, exc.reason).unwrap();
                }
            }
            if selection_excluded.len() > 5 {
                writeln!(out, "  ... e mais {}", selection_excluded.len() - 5).unwrap();
            }
            writeln!(out).unwrap();
        }

        // Orders
        if !self.orders.is_empty() {
            writeln!(out, "ORDENS ({}):", self.orders.len()).unwrap();
            for order in &self.orders {
                let side_str = match order.side {
                    OrderSide::Buy => "COMPRA",
                    OrderSide::Sell => "VENDA ",
                };
                writeln!(
                    out,
                    "  {} {} x {} @ {} (custo: {})",
                    side_str, order.symbol, order.shares, order.price, order.estimated_cost
                ).unwrap();
            }
            writeln!(out).unwrap();
        }

        // Metrics
        writeln!(out, "MÉTRICAS:").unwrap();
        writeln!(out, "  Candidatos: {}", self.diagnostics.total_candidates).unwrap();
        writeln!(out, "  Excluídos gating: {}", self.diagnostics.gating_excluded).unwrap();
        writeln!(out, "  Excluídos seleção: {}", self.diagnostics.selection_excluded).unwrap();
        writeln!(out, "  Selecionados: {}", self.diagnostics.final_selected).unwrap();
        writeln!(out, "  Peso total: {:.1}%", self.diagnostics.total_weight * 100.0).unwrap();
        writeln!(out, "  Turnover: {:.1}%", self.diagnostics.turnover * 100.0).unwrap();
        writeln!(out, "  Custos estimados: {}", self.diagnostics.estimated_costs).unwrap();
        writeln!(out, "  Cash residual: {}", self.diagnostics.cash_residual).unwrap();

        out
    }

    /// Generate compact one-line summary.
    pub fn to_compact(&self) -> String {
        format!(
            "[{}|{}] sel={} excl={} orders={} turnover={:.1}%",
            self.date,
            match self.market {
                Market::BR => "BR",
                Market::US => "US",
            },
            self.selected.len(),
            self.excluded.len(),
            self.orders.len(),
            self.diagnostics.turnover * 100.0
        )
    }

    /// Get exclusion counts by reason (machine-readable).
    pub fn exclusion_counts_by_reason(&self) -> HashMap<ExclusionReason, usize> {
        let mut counts: HashMap<ExclusionReason, usize> = HashMap::new();
        for exclusion in &self.excluded {
            *counts.entry(exclusion.reason).or_insert(0) += 1;
        }
        counts
    }

    /// Get total cost of all orders.
    pub fn total_order_cost(&self) -> rust_decimal::Decimal {
        self.orders.iter().map(|o| o.estimated_cost).sum()
    }

    /// Get cash residual from diagnostics.
    pub fn cash_residual(&self) -> rust_decimal::Decimal {
        self.diagnostics.cash_residual
    }
}

/// Audit logger for collecting rebalance logs.
#[derive(Debug, Clone, Default)]
pub struct AuditLogger {
    logs: Vec<RebalanceAuditLog>,
}

impl AuditLogger {
    pub fn new() -> Self {
        Self { logs: Vec::new() }
    }

    /// Add a rebalance log.
    pub fn add(&mut self, log: RebalanceAuditLog) {
        self.logs.push(log);
    }

    /// Get all logs.
    pub fn logs(&self) -> &[RebalanceAuditLog] {
        &self.logs
    }

    /// Get logs for a specific date.
    pub fn logs_for_date(&self, date: NaiveDate) -> Vec<&RebalanceAuditLog> {
        self.logs.iter().filter(|l| l.date == date).collect()
    }

    /// Get logs for a specific market.
    pub fn logs_for_market(&self, market: Market) -> Vec<&RebalanceAuditLog> {
        self.logs.iter().filter(|l| l.market == market).collect()
    }

    /// Generate full report.
    pub fn full_report(&self) -> String {
        let mut out = String::new();
        
        writeln!(out, "╔══════════════════════════════════════════════════════════════╗").unwrap();
        writeln!(out, "║           RELATÓRIO DE AUDITORIA DE REBALANCEAMENTOS          ║").unwrap();
        writeln!(out, "╚══════════════════════════════════════════════════════════════╝").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "Total de rebalanceamentos: {}", self.logs.len()).unwrap();
        writeln!(out).unwrap();

        for log in &self.logs {
            writeln!(out, "{}", log.to_summary()).unwrap();
            writeln!(out, "────────────────────────────────────────────────────────────────").unwrap();
            writeln!(out).unwrap();
        }

        out
    }

    /// Generate compact timeline.
    pub fn timeline(&self) -> String {
        let mut out = String::new();
        
        writeln!(out, "=== TIMELINE DE REBALANCEAMENTOS ===").unwrap();
        for log in &self.logs {
            writeln!(out, "  {}", log.to_compact()).unwrap();
        }

        out
    }

    /// Summary statistics.
    pub fn stats(&self) -> AuditStats {
        let total_rebalances = self.logs.len();
        let total_orders: usize = self.logs.iter().map(|l| l.orders.len()).sum();
        let total_excluded: usize = self.logs.iter().map(|l| l.excluded.len()).sum();
        let avg_turnover = if total_rebalances > 0 {
            self.logs.iter().map(|l| l.diagnostics.turnover).sum::<f64>() / total_rebalances as f64
        } else {
            0.0
        };
        let total_costs: rust_decimal::Decimal = self.logs.iter()
            .map(|l| l.diagnostics.estimated_costs)
            .sum();

        AuditStats {
            total_rebalances,
            total_orders,
            total_excluded,
            avg_turnover,
            total_costs,
        }
    }
}

/// Summary statistics from audit logs.
#[derive(Debug, Clone)]
pub struct AuditStats {
    pub total_rebalances: usize,
    pub total_orders: usize,
    pub total_excluded: usize,
    pub avg_turnover: f64,
    pub total_costs: rust_decimal::Decimal,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    use super::super::types::{ExclusionReason, ExclusionStage};

    fn make_sample_log() -> RebalanceAuditLog {
        RebalanceAuditLog {
            date: NaiveDate::from_ymd_opt(2025, 1, 3).unwrap(),
            market: Market::BR,
            selected: vec![
                SelectedAsset {
                    symbol: "PETR4".to_string(),
                    weight: 0.15,
                    score: 0.85,
                    reason: "momentum + quality".to_string(),
                },
                SelectedAsset {
                    symbol: "VALE3".to_string(),
                    weight: 0.12,
                    score: 0.80,
                    reason: "momentum".to_string(),
                },
            ],
            excluded: vec![
                EntryExclusion {
                    symbol: "OIBR3".to_string(),
                    reason: ExclusionReason::MissingFundamentals,
                    stage: ExclusionStage::Gating,
                    score: None,
                },
                EntryExclusion {
                    symbol: "ABEV3".to_string(),
                    reason: ExclusionReason::OutOfTopN,
                    stage: ExclusionStage::Selection,
                    score: Some(0.45),
                },
            ],
            orders: vec![
                Order::new("PETR4".to_string(), OrderSide::Buy, 300, dec!(38), dec!(24)),
            ],
            diagnostics: EntryDiagnostics {
                total_candidates: 50,
                gating_excluded: 10,
                selection_excluded: 38,
                final_selected: 2,
                turnover: 0.15,
                estimated_costs: dec!(24),
                total_weight: 0.27,
                cash_residual: dec!(730000),
                warnings: vec![],
            },
        }
    }

    #[test]
    fn test_summary_generation() {
        let log = make_sample_log();
        let summary = log.to_summary();

        assert!(summary.contains("REBALANCE 2025-01-03 (BR)"));
        assert!(summary.contains("SELECIONADOS (2)"));
        assert!(summary.contains("PETR4: 15.0%"));
        assert!(summary.contains("EXCLUÍDOS GATING"));
        assert!(summary.contains("OIBR3: sem dados fundamentalistas"));
        assert!(summary.contains("ORDENS"));
        assert!(summary.contains("COMPRA PETR4"));
    }

    #[test]
    fn test_compact_format() {
        let log = make_sample_log();
        let compact = log.to_compact();

        assert!(compact.contains("[2025-01-03|BR]"));
        assert!(compact.contains("sel=2"));
        assert!(compact.contains("orders=1"));
    }

    #[test]
    fn test_audit_logger() {
        let mut logger = AuditLogger::new();
        logger.add(make_sample_log());
        
        let mut log2 = make_sample_log();
        log2.date = NaiveDate::from_ymd_opt(2025, 1, 10).unwrap();
        logger.add(log2);

        assert_eq!(logger.logs().len(), 2);
        
        let stats = logger.stats();
        assert_eq!(stats.total_rebalances, 2);
        assert_eq!(stats.total_orders, 2);
    }

    #[test]
    fn test_filter_by_date() {
        let mut logger = AuditLogger::new();
        logger.add(make_sample_log());
        
        let date = NaiveDate::from_ymd_opt(2025, 1, 3).unwrap();
        let logs = logger.logs_for_date(date);
        
        assert_eq!(logs.len(), 1);
    }

    #[test]
    fn test_filter_by_market() {
        let mut logger = AuditLogger::new();
        logger.add(make_sample_log());
        
        let mut us_log = make_sample_log();
        us_log.market = Market::US;
        logger.add(us_log);

        assert_eq!(logger.logs_for_market(Market::BR).len(), 1);
        assert_eq!(logger.logs_for_market(Market::US).len(), 1);
    }
}

