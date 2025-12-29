//! Invariant tests for Exit Module.
//!
//! These tests prove that risk limits are never violated.

use backtester_intelligence::exit::{
    ExitContext, ExitEngine, ExitEngineConfig, ExitPolicyConfig, ExitReason,
    Position, RiskConfig, RiskViolation, DrawdownAction,
};
use backtester_intelligence::filters::Market;
use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

// =============================================================================
// Test Fixtures
// =============================================================================

fn fixed_date() -> NaiveDate {
    NaiveDate::from_ymd_opt(2025, 1, 10).unwrap()
}

fn make_positions(count: usize) -> Vec<Position> {
    (0..count).map(|i| {
        Position::new(
            format!("SYM{:04}", i),
            Market::BR,
            100 + (i as i64 * 100),
            Decimal::from(30 + i as i64),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            Decimal::from(28 + i as i64), // slight loss
        )
    }).collect()
}

fn make_context(equity: Decimal, peak: Decimal) -> ExitContext {
    ExitContext {
        date: fixed_date(),
        capital: dec!(1_000_000),
        equity,
        peak_equity: peak,
        market: Market::BR,
    }
}

// =============================================================================
// Invariant: Stop-loss triggers correctly
// =============================================================================

#[test]
fn invariant_stop_loss_triggers_at_threshold() {
    let config = ExitEngineConfig {
        policy: ExitPolicyConfig {
            enable_stop_loss: true,
            stop_loss_pct: -0.10,
            enable_take_profit: false,
            enable_time_exit: false,
            enable_trailing_stop: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = ExitEngine::new(config);
    let ctx = make_context(dec!(1_000_000), dec!(1_000_000));

    // Create positions with varying returns
    let positions = vec![
        // -15% loss (should trigger)
        Position::new("LOSS15", Market::BR, 100, dec!(100), 
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(85)),
        // -5% loss (should NOT trigger)
        Position::new("LOSS5", Market::BR, 100, dec!(100), 
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(95)),
        // +10% gain (should NOT trigger)
        Position::new("GAIN10", Market::BR, 100, dec!(100), 
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(110)),
    ];

    let (result, _, _) = engine.evaluate(&positions, &ctx);

    // Only LOSS15 should be exited
    assert_eq!(result.exits.len(), 1);
    assert_eq!(result.exits[0].symbol, "LOSS15");
    assert_eq!(result.exits[0].reason, ExitReason::StopLoss);
}

#[test]
fn invariant_stop_loss_exact_threshold() {
    let config = ExitEngineConfig {
        policy: ExitPolicyConfig {
            enable_stop_loss: true,
            stop_loss_pct: -0.10,
            enable_take_profit: false,
            enable_time_exit: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = ExitEngine::new(config);
    let ctx = make_context(dec!(1_000_000), dec!(1_000_000));

    // Exactly at -10%
    let positions = vec![
        Position::new("EXACT", Market::BR, 100, dec!(100), 
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(90)),
    ];

    let (result, _, _) = engine.evaluate(&positions, &ctx);

    // Should trigger at exact threshold
    assert_eq!(result.exits.len(), 1);
    assert_eq!(result.exits[0].reason, ExitReason::StopLoss);
}

// =============================================================================
// Invariant: Take-profit triggers correctly
// =============================================================================

#[test]
fn invariant_take_profit_triggers_at_threshold() {
    let config = ExitEngineConfig {
        policy: ExitPolicyConfig {
            enable_stop_loss: false,
            enable_take_profit: true,
            take_profit_pct: 0.25,
            enable_time_exit: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = ExitEngine::new(config);
    let ctx = make_context(dec!(1_000_000), dec!(1_000_000));

    let positions = vec![
        // +30% gain (should trigger)
        Position::new("GAIN30", Market::BR, 100, dec!(100), 
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(130)),
        // +15% gain (should NOT trigger)
        Position::new("GAIN15", Market::BR, 100, dec!(100), 
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(115)),
    ];

    let (result, _, _) = engine.evaluate(&positions, &ctx);

    assert_eq!(result.exits.len(), 1);
    assert_eq!(result.exits[0].symbol, "GAIN30");
    assert_eq!(result.exits[0].reason, ExitReason::TakeProfit);
}

// =============================================================================
// Invariant: Exposure never exceeds max
// =============================================================================

#[test]
fn invariant_exposure_detected() {
    let config = ExitEngineConfig {
        risk: RiskConfig {
            max_single_exposure: 0.20,
            check_exposure: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = ExitEngine::new(config);
    let ctx = make_context(dec!(1_000_000), dec!(1_000_000));

    // Position with 30% exposure (exceeds 20% limit)
    let positions = vec![
        Position::new("OVER", Market::BR, 6000, dec!(50), 
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(50)),
    ];

    let (result, _, _) = engine.evaluate(&positions, &ctx);

    // Should detect exposure violation
    assert!(result.diagnostics.risk_violations.contains(&RiskViolation::ExposureExceeded));
}

// =============================================================================
// Invariant: Drawdown guard triggers correctly
// =============================================================================

#[test]
fn invariant_drawdown_guard_cash_out() {
    let config = ExitEngineConfig {
        risk: RiskConfig {
            max_drawdown_pct: -0.10,
            drawdown_action: DrawdownAction::CashOut,
            check_drawdown: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = ExitEngine::new(config);

    // 20% portfolio drawdown (exceeds 10% limit)
    let ctx = make_context(dec!(800_000), dec!(1_000_000));

    let positions = vec![
        Position::new("POS1", Market::BR, 100, dec!(50), 
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(52)),
        Position::new("POS2", Market::BR, 100, dec!(60), 
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(62)),
    ];

    let (result, _, _) = engine.evaluate(&positions, &ctx);

    // All positions should be exited with DrawdownGuard
    assert_eq!(result.exits.len(), 2);
    assert!(result.exits.iter().all(|e| e.reason == ExitReason::DrawdownGuard));
    assert!(result.diagnostics.risk_violations.contains(&RiskViolation::DrawdownExceeded));
}

#[test]
fn invariant_drawdown_no_trigger_below_threshold() {
    let config = ExitEngineConfig {
        risk: RiskConfig {
            max_drawdown_pct: -0.15,
            drawdown_action: DrawdownAction::CashOut,
            check_drawdown: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = ExitEngine::new(config);

    // 10% drawdown (below 15% limit)
    let ctx = make_context(dec!(900_000), dec!(1_000_000));

    let positions = vec![
        Position::new("POS1", Market::BR, 100, dec!(50), 
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(52)),
    ];

    let (result, _, _) = engine.evaluate(&positions, &ctx);

    // No exits, no violations
    assert!(result.exits.is_empty());
    assert!(!result.diagnostics.risk_violations.contains(&RiskViolation::DrawdownExceeded));
}

// =============================================================================
// Invariant: Determinism
// =============================================================================

#[test]
fn invariant_determinism() {
    let config = ExitEngineConfig::default();
    let engine = ExitEngine::new(config);
    let ctx = make_context(dec!(900_000), dec!(1_000_000));

    let positions = vec![
        Position::new("PETR4", Market::BR, 500, dec!(50), 
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(42)), // -16%
        Position::new("VALE3", Market::BR, 300, dec!(60), 
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(80)), // +33%
    ];

    // Run 3 times
    let (r1, o1, a1) = engine.evaluate(&positions, &ctx);
    let (r2, o2, a2) = engine.evaluate(&positions, &ctx);
    let (r3, o3, a3) = engine.evaluate(&positions, &ctx);

    // Results must be identical
    assert_eq!(r1.exits.len(), r2.exits.len());
    assert_eq!(r2.exits.len(), r3.exits.len());
    assert_eq!(o1.len(), o2.len());
    assert_eq!(o2.len(), o3.len());
    assert_eq!(a1.to_summary(), a2.to_summary());
    assert_eq!(a2.to_summary(), a3.to_summary());
}

// =============================================================================
// Invariant: All exit orders have positive shares
// =============================================================================

#[test]
fn invariant_no_zero_share_exits() {
    let engine = ExitEngine::new(ExitEngineConfig::default());
    let ctx = make_context(dec!(1_000_000), dec!(1_000_000));

    // Mix of positions with exits
    let positions = vec![
        Position::new("LOSS", Market::BR, 500, dec!(50), 
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(40)), // -20%
        Position::new("GAIN", Market::BR, 300, dec!(60), 
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(85)), // +42%
    ];

    let (_, orders, _) = engine.evaluate(&positions, &ctx);

    for order in &orders {
        assert!(order.shares > 0, "Order {} has zero shares", order.symbol);
    }
}

// =============================================================================
// Invariant: BR lot sizes respected
// =============================================================================

#[test]
fn invariant_br_lot_sizes() {
    let engine = ExitEngine::new(ExitEngineConfig::default());
    let ctx = make_context(dec!(1_000_000), dec!(1_000_000));

    // Position with 550 shares (should round to 500)
    let positions = vec![
        Position::new("PETR4", Market::BR, 550, dec!(50), 
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(40)), // -20%
    ];

    let (_, orders, _) = engine.evaluate(&positions, &ctx);

    assert_eq!(orders.len(), 1);
    assert_eq!(orders[0].shares, 500); // Rounded to lot of 100
    assert_eq!(orders[0].shares % 100, 0);
}

// =============================================================================
// Invariant: Costs are non-negative
// =============================================================================

#[test]
fn invariant_costs_non_negative() {
    let engine = ExitEngine::new(ExitEngineConfig::default());
    let ctx = make_context(dec!(1_000_000), dec!(1_000_000));

    let positions = vec![
        Position::new("PETR4", Market::BR, 500, dec!(50), 
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(40)),
    ];

    let (_, orders, _) = engine.evaluate(&positions, &ctx);

    for order in &orders {
        assert!(order.estimated_cost >= Decimal::ZERO);
    }
}

// =============================================================================
// Invariant: First policy wins (no duplicate exits)
// =============================================================================

#[test]
fn invariant_no_duplicate_exits() {
    let config = ExitEngineConfig {
        policy: ExitPolicyConfig {
            enable_stop_loss: true,
            stop_loss_pct: -0.05,
            enable_take_profit: true,
            take_profit_pct: 0.10,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = ExitEngine::new(config);
    let ctx = make_context(dec!(1_000_000), dec!(1_000_000));

    // Multiple positions that could trigger multiple policies
    let positions = make_positions(10);

    let (result, _, _) = engine.evaluate(&positions, &ctx);

    // No duplicate symbols in exits
    let mut seen: Vec<&str> = Vec::new();
    for exit in &result.exits {
        assert!(!seen.contains(&exit.symbol.as_str()), "Duplicate exit for {}", exit.symbol);
        seen.push(&exit.symbol);
    }
}











