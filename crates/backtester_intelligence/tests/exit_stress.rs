//! Stress tests for Exit Module - extreme scenarios.

use backtester_core::Money;
use backtester_intelligence::exit::{
    ExitContext, ExitEngine, ExitEngineConfig, ExitPolicyConfig, ExitReason,
    Position, RiskConfig, DrawdownAction,
};
use backtester_intelligence::filters::Market;
use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::time::{Duration, Instant};

fn fixed_date() -> NaiveDate {
    NaiveDate::from_ymd_opt(2025, 1, 10).unwrap()
}

// =============================================================================
// Stress Test 1: Large Portfolio (1000 positions)
// =============================================================================

#[test]
fn stress_large_portfolio() {
    let engine = ExitEngine::new(ExitEngineConfig::default());

    // Generate 1000 positions
    let positions: Vec<Position> = (0..1000).map(|i| {
        let cost = Decimal::from(30 + (i % 100));
        let return_pct = (i as f64 % 50.0 - 25.0) / 100.0; // -25% to +25%
        let price = cost * Decimal::try_from(1.0 + return_pct).unwrap();

        Position::new(
            format!("SYM{:05}", i),
            Market::BR,
            (100 + (i % 10) * 100) as i64,
            cost,
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            price,
        )
    }).collect();

    let ctx = ExitContext::new(
        fixed_date(),
        dec!(10_000_000),
        dec!(10_000_000),
        Market::BR,
    );

    let start = Instant::now();
    let (result, orders, audit) = engine.evaluate(&positions, &ctx);
    let elapsed = start.elapsed();

    // Performance: should complete in reasonable time
    assert!(
        elapsed < Duration::from_millis(100),
        "1000 positions took {:?}, should be < 100ms", elapsed
    );

    // Should have some exits (based on default thresholds)
    println!("Large portfolio: {} exits, {:?}", result.exits.len(), elapsed);

    // Audit should work
    let summary = audit.to_summary();
    assert!(summary.contains("EXIT AUDIT"));
}

// =============================================================================
// Stress Test 2: Flash Crash (50% drop)
// =============================================================================

#[test]
fn stress_flash_crash() {
    let config = ExitEngineConfig {
        policy: ExitPolicyConfig {
            enable_stop_loss: true,
            stop_loss_pct: -0.10,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = ExitEngine::new(config);

    // All positions dropped 50%
    let positions: Vec<Position> = (0..100).map(|i| {
        Position::new(
            format!("CRASH{:03}", i),
            Market::BR,
            500,
            dec!(100),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(50), // 50% loss
        )
    }).collect();

    let ctx = ExitContext::new(
        fixed_date(),
        dec!(1_000_000),
        dec!(500_000), // 50% drawdown
        Market::BR,
    );

    let (result, orders, _) = engine.evaluate(&positions, &ctx);

    // All positions should trigger stop-loss
    assert_eq!(result.exits.len(), 100);
    assert!(result.exits.iter().all(|e| e.reason == ExitReason::StopLoss));
    assert_eq!(orders.len(), 100);

    // All PnL should be negative
    assert!(result.diagnostics.total_exit_pnl < Money::ZERO);
}

// =============================================================================
// Stress Test 3: All Stop-Loss Simultaneous
// =============================================================================

#[test]
fn stress_all_stop_loss_simultaneous() {
    let config = ExitEngineConfig {
        policy: ExitPolicyConfig {
            enable_stop_loss: true,
            stop_loss_pct: -0.05,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = ExitEngine::new(config);

    // 500 positions all at exactly -6% (just past threshold)
    let positions: Vec<Position> = (0..500).map(|i| {
        Position::new(
            format!("SL{:04}", i),
            Market::BR,
            200,
            dec!(100),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            dec!(94), // -6%
        )
    }).collect();

    let ctx = ExitContext::new(
        fixed_date(),
        dec!(5_000_000),
        dec!(5_000_000),
        Market::BR,
    );

    let start = Instant::now();
    let (result, orders, _) = engine.evaluate(&positions, &ctx);
    let elapsed = start.elapsed();

    // All should exit
    assert_eq!(result.exits.len(), 500);
    assert_eq!(orders.len(), 500);

    // Performance
    assert!(elapsed < Duration::from_millis(50), "500 exits took {:?}", elapsed);
}

// =============================================================================
// Stress Test 4: Zero/Negative Price Edge Cases
// =============================================================================

#[test]
fn stress_zero_price_positions() {
    let engine = ExitEngine::new(ExitEngineConfig::default());

    let positions = vec![
        // Near-zero price
        Position::new("PENNY1", Market::BR, 1000, dec!(0.01), 
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(0.005)),
        // Very high price
        Position::new("HIGH", Market::BR, 10, dec!(10000), 
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(8000)),
        // Normal price
        Position::new("NORMAL", Market::BR, 500, dec!(50), 
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), dec!(45)),
    ];

    let ctx = ExitContext::new(
        fixed_date(),
        dec!(1_000_000),
        dec!(1_000_000),
        Market::BR,
    );

    // Should not panic
    let (result, _, audit) = engine.evaluate(&positions, &ctx);

    // Should work without errors
    let _ = audit.to_summary();
    let _ = audit.to_compact();
}

// =============================================================================
// Stress Test 5: Mixed Markets (BR + US)
// =============================================================================

#[test]
fn stress_mixed_markets() {
    let engine = ExitEngine::new(ExitEngineConfig::default());

    let mut positions: Vec<Position> = Vec::new();

    // BR positions
    for i in 0..50 {
        positions.push(Position::new(
            format!("BR{:02}", i),
            Market::BR,
            (i as i64 + 1) * 100,
            dec!(50),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            Decimal::from(40 + (i % 20)), // -20% to +20%
        ));
    }

    // US positions
    for i in 0..50 {
        positions.push(Position::new(
            format!("US{:02}", i),
            Market::US,
            i as i64 + 10,
            dec!(150),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            Decimal::from(120 + (i % 60)), // -20% to +20%
        ));
    }

    let ctx = ExitContext::new(
        fixed_date(),
        dec!(5_000_000),
        dec!(5_000_000),
        Market::BR,
    );

    let (result, orders, _) = engine.evaluate(&positions, &ctx);

    // Should process both markets
    assert!(result.diagnostics.positions_evaluated == 100);

    // BR orders should be multiples of 100
    for order in orders.iter().filter(|o| o.symbol.starts_with("BR")) {
        assert_eq!(order.shares % 100, 0, "BR {} has invalid lot", order.symbol);
    }

    // US orders can be any positive integer
    for order in orders.iter().filter(|o| o.symbol.starts_with("US")) {
        assert!(order.shares >= 1, "US {} has invalid shares", order.symbol);
    }
}

// =============================================================================
// Stress Test 6: Drawdown Cascade
// =============================================================================

#[test]
fn stress_drawdown_cascade() {
    let config = ExitEngineConfig {
        risk: RiskConfig {
            max_drawdown_pct: -0.10,
            drawdown_action: DrawdownAction::CashOut,
            check_drawdown: true,
            ..Default::default()
        },
        // Disable individual policies to test only drawdown guard
        policy: ExitPolicyConfig {
            enable_stop_loss: false,
            enable_take_profit: false,
            enable_time_exit: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = ExitEngine::new(config);

    // Positions with mixed returns (no individual triggers)
    let positions: Vec<Position> = (0..50).map(|i| {
        Position::new(
            format!("POS{:02}", i),
            Market::BR,
            500,
            dec!(100),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            // Small losses and gains (within individual thresholds)
            if i % 2 == 0 { dec!(98) } else { dec!(102) },
        )
    }).collect();

    // 15% portfolio drawdown (exceeds 10% limit)
    let mut ctx = ExitContext::new(
        fixed_date(),
        dec!(1_000_000),
        dec!(850_000), // Current equity
        Market::BR,
    );
    ctx.peak_equity = Money::from_int(1_000_000); // Peak was 1M, now 850k = 15% drawdown

    let (result, _, _) = engine.evaluate(&positions, &ctx);

    // All positions should be exited due to drawdown guard (CashOut)
    assert_eq!(result.exits.len(), 50);

    // All should be DrawdownGuard
    let drawdown_exits = result.exits.iter()
        .filter(|e| e.reason == ExitReason::DrawdownGuard)
        .count();
    assert_eq!(drawdown_exits, 50, "All exits should be DrawdownGuard");
}

// =============================================================================
// Stress Test 7: Very Old Positions (Time Exit)
// =============================================================================

#[test]
fn stress_time_exit_old_positions() {
    let config = ExitEngineConfig {
        policy: ExitPolicyConfig {
            enable_time_exit: true,
            max_holding_days: 30,
            enable_stop_loss: false,
            enable_take_profit: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = ExitEngine::new(config);

    // Mix of old and new positions
    let positions: Vec<Position> = (0..20).map(|i| {
        let entry_date = if i < 10 {
            NaiveDate::from_ymd_opt(2024, 11, 1).unwrap() // Very old (70+ days)
        } else {
            NaiveDate::from_ymd_opt(2025, 1, 5).unwrap() // Recent (5 days)
        };

        Position::new(
            format!("TIME{:02}", i),
            Market::BR,
            300,
            dec!(50),
            entry_date,
            dec!(55),
        )
    }).collect();

    let ctx = ExitContext::new(
        fixed_date(),
        dec!(1_000_000),
        dec!(1_000_000),
        Market::BR,
    );

    let (result, _, _) = engine.evaluate(&positions, &ctx);

    // Only old positions should exit
    assert_eq!(result.exits.len(), 10);
    assert!(result.exits.iter().all(|e| e.reason == ExitReason::TimeExit));
}

// =============================================================================
// Performance Smoke Test
// =============================================================================

#[test]
fn perf_smoke_exit_100_under_10ms() {
    let engine = ExitEngine::new(ExitEngineConfig::default());

    let positions: Vec<Position> = (0..100).map(|i| {
        Position::new(
            format!("PERF{:03}", i),
            Market::BR,
            500,
            dec!(50),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            Decimal::from(40 + (i % 20)),
        )
    }).collect();

    let ctx = ExitContext::new(
        fixed_date(),
        dec!(1_000_000),
        dec!(1_000_000),
        Market::BR,
    );

    let start = Instant::now();
    let _ = engine.evaluate(&positions, &ctx);
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_millis(10),
        "100 positions took {:?}, should be < 10ms", elapsed
    );
}

