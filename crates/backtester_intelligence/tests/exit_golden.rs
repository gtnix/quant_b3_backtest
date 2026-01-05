//! Golden snapshot tests for Exit Module audit output.

use backtester_intelligence::exit::{
    ExitAuditLog, ExitContext, ExitDiagnostics, ExitReason, ExitedPosition,
    Position, RiskViolation,
};
use backtester_intelligence::entry::{Order, OrderSide};
use backtester_intelligence::filters::Market;
use chrono::NaiveDate;
use rust_decimal_macros::dec;
use std::collections::HashMap;

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
            ExitedPosition {
                symbol: "ITUB4".to_string(),
                shares: 200,
                reason: ExitReason::StopLoss,
                pnl: dec!(-800),
                return_pct: -0.08,
            },
        ],
        orders: vec![
            Order::new("PETR4".to_string(), OrderSide::Sell, 500, dec!(45), dec!(225)),
            Order::new("VALE3".to_string(), OrderSide::Sell, 300, dec!(75), dec!(225)),
            Order::new("ITUB4".to_string(), OrderSide::Sell, 200, dec!(46), dec!(92)),
        ],
        diagnostics: ExitDiagnostics {
            positions_evaluated: 10,
            positions_exited: 3,
            stop_loss_count: 2,
            take_profit_count: 1,
            time_exit_count: 0,
            trailing_stop_count: 0,
            risk_cap_count: 0,
            drawdown_guard_count: 0,
            rebalance_count: 0,
            total_exit_pnl: dec!(1200),
            exit_turnover: dec!(53700),
            estimated_costs: dec!(542),
            risk_violations: vec![],
        },
    }
}

// =============================================================================
// Golden Test: to_summary() format stability
// =============================================================================

const GOLDEN_EXIT_SUMMARY: &str = r#"=== EXIT AUDIT 2025-01-10 (BR) ===

SAÍDAS (3):
  [stop-loss] (2):
    PETR4: 500 ações, PnL: -2500, ret: -10.0%
    ITUB4: 200 ações, PnL: -800, ret: -8.0%
  [take-profit] (1):
    VALE3: 300 ações, PnL: 4500, ret: 25.0%

ORDENS DE VENDA (3):
  VENDA PETR4 x 500 @ 45 (custo: 225)
  VENDA VALE3 x 300 @ 75 (custo: 225)
  VENDA ITUB4 x 200 @ 46 (custo: 92)

MÉTRICAS:
  Posições avaliadas: 10
  Saídas: 3
  Stop-loss: 2
  Take-profit: 1
  Time exit: 0
  Trailing stop: 0
  Drawdown guard: 0
  PnL total saídas: 1200
  Turnover saídas: 53700
  Custos estimados: 542
"#;

#[test]
fn golden_exit_summary() {
    let log = make_sample_log();
    let summary = log.to_summary();
    
    assert_eq!(summary.trim(), GOLDEN_EXIT_SUMMARY.trim());
}

// =============================================================================
// Golden Test: exit_counts_by_reason()
// =============================================================================

#[test]
fn golden_exit_counts_by_reason() {
    let log = make_sample_log();
    let counts = log.exit_counts_by_reason();

    assert_eq!(counts.get(&ExitReason::StopLoss), Some(&2));
    assert_eq!(counts.get(&ExitReason::TakeProfit), Some(&1));
    assert_eq!(counts.get(&ExitReason::TimeExit), None);
    assert_eq!(counts.get(&ExitReason::DrawdownGuard), None);
}

// =============================================================================
// Golden Test: to_compact() format stability
// =============================================================================

#[test]
fn golden_exit_compact() {
    let log = make_sample_log();
    let compact = log.to_compact();

    assert_eq!(compact, "[2025-01-10|BR] exits=3 sl=2 tp=1 pnl=1200");
}

// =============================================================================
// Golden Test: with risk violations
// =============================================================================

#[test]
fn golden_with_violations() {
    let mut log = make_sample_log();
    log.diagnostics.risk_violations = vec![
        RiskViolation::DrawdownExceeded,
        RiskViolation::ExposureExceeded,
    ];

    let summary = log.to_summary();

    assert!(summary.contains("VIOLAÇÕES DE RISCO:"));
    assert!(summary.contains("drawdown excedido"));
    assert!(summary.contains("exposição por ativo excedida"));
}

// =============================================================================
// Helper: Generate golden output (ignored, run manually to update)
// =============================================================================

#[test]
#[ignore]
fn generate_exit_golden_output() {
    let log = make_sample_log();
    println!("=== GOLDEN EXIT SUMMARY ===");
    println!("{}", log.to_summary());
    println!("\n=== GOLDEN COMPACT ===");
    println!("{}", log.to_compact());
}

// =============================================================================
// Determinism: Multiple runs produce identical output
// =============================================================================

#[test]
fn determinism_exit_summary() {
    let log1 = make_sample_log();
    let log2 = make_sample_log();
    let log3 = make_sample_log();

    assert_eq!(log1.to_summary(), log2.to_summary());
    assert_eq!(log2.to_summary(), log3.to_summary());
}

#[test]
fn determinism_exit_compact() {
    let log1 = make_sample_log();
    let log2 = make_sample_log();

    assert_eq!(log1.to_compact(), log2.to_compact());
}
































