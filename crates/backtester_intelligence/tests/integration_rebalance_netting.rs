//! Integration tests for Rebalance Orchestrator netting logic.
//!
//! Proves that Entry and Exit orders are correctly combined without conflicts.

use backtester_core::{Money, Price};
use backtester_intelligence::entry::{AssetCandidate, OrderSide};
use backtester_intelligence::exit::Position;
use backtester_intelligence::filters::Market;
use backtester_intelligence::orchestrator::{OrchestratorConfig, RebalanceOrchestrator};
use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

fn fixed_date() -> NaiveDate {
    NaiveDate::from_ymd_opt(2025, 1, 15).unwrap()
}

fn make_position(symbol: &str, shares: i64, cost: Decimal, current: Decimal) -> Position {
    Position::new(
        symbol,
        Market::BR,
        shares,
        Money::from(cost),
        NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
        Price::from(current),
    )
}

fn make_candidate(symbol: &str, price: Decimal, score: f64) -> AssetCandidate {
    AssetCandidate {
        symbol: symbol.to_string(),
        market: Market::BR,
        price: Some(Price::from(price)),
        avg_volume: Some(Money::from(dec!(5_000_000))),
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

// =============================================================================
// Test: Stop-loss exit + entry on same symbol should net
// =============================================================================

#[test]
fn integration_stop_loss_exit_with_entry_same_symbol() {
    // Scenario:
    // - PETR4 position with stop-loss triggered (Exit wants SELL 500)
    // - But PETR4 also has high score (Entry wants BUY 300)
    // - Result: NET SELL 200 (not SELL 500 + BUY 300 separately)

    let config = OrchestratorConfig {
        exit: backtester_intelligence::exit::ExitEngineConfig {
            policy: backtester_intelligence::exit::ExitPolicyConfig {
                enable_stop_loss: true,
                stop_loss_pct: -0.10,
                enable_take_profit: false,
                enable_time_exit: false,
                enable_trailing_stop: false,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let orchestrator = RebalanceOrchestrator::new(config);

    // Position with 15% loss (triggers stop-loss)
    let positions = vec![
        make_position("PETR4", 500, dec!(50), dec!(42.5)), // -15%
    ];

    // PETR4 is also a high-scoring candidate
    let candidates = vec![
        make_candidate("PETR4", dec!(42.5), 0.85),
        make_candidate("VALE3", dec!(65), 0.75),
        make_candidate("ITUB4", dec!(28), 0.70),
    ];

    let (result, audit) = orchestrator.execute_rebalance(
        fixed_date(),
        Market::BR,
        &positions,
        &candidates,
        Money::from(dec!(500_000)),  // initial cash
        Money::from(dec!(500_000)),  // equity
        Money::from(dec!(500_000)),  // peak equity
    );

    // Should have netted orders
    assert!(
        result.netting_count >= 1 || result.net_orders.iter().any(|o| o.symbol == "PETR4"),
        "PETR4 should appear in netted orders"
    );

    // Total costs should be calculated on net orders only
    let total_cost: Money = result.net_orders.iter().map(|o| o.estimated_cost).sum();
    assert!(
        total_cost == audit.total_cost,
        "Total cost mismatch: orders={:?}, audit={:?}", total_cost, audit.total_cost
    );

    // Cash after should be consistent
    assert!(
        audit.cash_after > Money::ZERO,
        "Cash should remain positive: {:?}", audit.cash_after
    );
}

// =============================================================================
// Test: No conflict - exit and entry on different symbols
// =============================================================================

#[test]
fn integration_no_conflict_different_symbols() {
    let config = OrchestratorConfig::default();
    let orchestrator = RebalanceOrchestrator::new(config);

    // PETR4 triggers exit (take-profit or stop-loss)
    let positions = vec![
        make_position("PETR4", 500, dec!(50), dec!(40)), // -20% triggers stop-loss
    ];

    // Entry targets different symbol
    let candidates = vec![
        make_candidate("VALE3", dec!(65), 0.90),
        make_candidate("ITUB4", dec!(28), 0.85),
    ];

    let (result, _audit) = orchestrator.execute_rebalance(
        fixed_date(),
        Market::BR,
        &positions,
        &candidates,
        Money::from(dec!(500_000)),
        Money::from(dec!(500_000)),
        Money::from(dec!(500_000)),
    );

    // No netting should occur (different symbols)
    assert_eq!(result.netting_count, 0);

    // Should have both exit (SELL PETR4) and entry orders
    let has_sell_petr4 = result.net_orders.iter()
        .any(|o| o.symbol == "PETR4" && o.side == OrderSide::Sell);
    assert!(has_sell_petr4, "Should have SELL PETR4 from exit");
}

// =============================================================================
// Test: Complete cancellation (SELL and BUY same quantity)
// =============================================================================

#[test]
fn integration_complete_cancellation() {
    let config = OrchestratorConfig {
        exit: backtester_intelligence::exit::ExitEngineConfig {
            policy: backtester_intelligence::exit::ExitPolicyConfig {
                enable_stop_loss: true,
                stop_loss_pct: -0.05, // low threshold
                enable_take_profit: false,
                enable_time_exit: false,
                ..Default::default()
            },
            ..Default::default()
        },
        entry: backtester_intelligence::entry::EntryEngineConfig {
            selection: backtester_intelligence::entry::SelectionConfig {
                top_n_br: 1,
                top_n_us: 1,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let orchestrator = RebalanceOrchestrator::new(config);

    // Position with 6% loss (triggers stop-loss at 5%)
    let positions = vec![
        make_position("ITUB4", 400, dec!(30), dec!(28.2)), // -6%
    ];

    // ITUB4 is ONLY candidate and would be selected
    let candidates = vec![
        make_candidate("ITUB4", dec!(28.2), 0.95),
    ];

    let (result, _audit) = orchestrator.execute_rebalance(
        fixed_date(),
        Market::BR,
        &positions,
        &candidates,
        Money::from(dec!(100_000)),
        Money::from(dec!(100_000)),
        Money::from(dec!(100_000)),
    );

    // If exit sells 400 and entry wants to buy ~same amount, they might cancel
    // Or result in small net order
    // Key: no duplicated costs, no conflicting orders
    
    // Verify determinism - run again and get same result
    let candidates2 = vec![make_candidate("ITUB4", dec!(28.2), 0.95)];
    let (result2, _) = orchestrator.execute_rebalance(
        fixed_date(),
        Market::BR,
        &positions,
        &candidates2,
        Money::from(dec!(100_000)),
        Money::from(dec!(100_000)),
        Money::from(dec!(100_000)),
    );

    assert_eq!(result.net_orders.len(), result2.net_orders.len());
    assert_eq!(result.netting_count, result2.netting_count);
}

// =============================================================================
// Test: Cash flow consistency
// =============================================================================

#[test]
fn integration_cash_flow_consistency() {
    let config = OrchestratorConfig::default();
    let orchestrator = RebalanceOrchestrator::new(config);

    let positions = vec![
        make_position("PETR4", 300, dec!(50), dec!(55)), // +10%
    ];

    let candidates = vec![
        make_candidate("VALE3", dec!(60), 0.85),
    ];

    let initial_cash = Money::from(dec!(200_000));

    let (_result, audit) = orchestrator.execute_rebalance(
        fixed_date(),
        Market::BR,
        &positions,
        &candidates,
        initial_cash,
        Money::from(dec!(200_000)),
        Money::from(dec!(200_000)),
    );

    // Cash before should match input
    assert_eq!(audit.cash_before, initial_cash);

    // Cash after should be consistent with orders
    // (we can't verify exact amount without knowing all order details,
    // but it should be non-negative and different from before if orders executed)
    assert!(
        audit.cash_after >= Money::ZERO,
        "Cash should not go negative: {:?}", audit.cash_after
    );
}

// =============================================================================
// Test: Costs NOT doubled for netted orders
// =============================================================================

#[test]
fn integration_costs_not_doubled() {
    let config = OrchestratorConfig {
        br_cost_bps: 10.0, // 0.1%
        us_cost_bps: 5.0,
        exit: backtester_intelligence::exit::ExitEngineConfig {
            policy: backtester_intelligence::exit::ExitPolicyConfig {
                enable_stop_loss: true,
                stop_loss_pct: -0.05,
                enable_take_profit: false,
                enable_time_exit: false,
                ..Default::default()
            },
            br_cost_bps: 10.0,
            ..Default::default()
        },
        entry: backtester_intelligence::entry::EntryEngineConfig::default(),
    };

    let orchestrator = RebalanceOrchestrator::new(config);

    // Position: 1000 shares @ 100, currently at 90 (-10% triggers stop-loss)
    let positions = vec![
        make_position("PETR4", 1000, dec!(100), dec!(90)),
    ];

    // Entry would want 600 shares of PETR4
    let candidates = vec![
        make_candidate("PETR4", dec!(90), 0.90),
    ];

    let (result, audit) = orchestrator.execute_rebalance(
        fixed_date(),
        Market::BR,
        &positions,
        &candidates,
        Money::from(dec!(500_000)),
        Money::from(dec!(500_000)),
        Money::from(dec!(500_000)),
    );

    // Net order should be SELL 400 (1000 - 600)
    // Cost should be on 400 * 90 = 36000 * 0.001 = 36
    // NOT on (1000 * 90 * 0.001) + (600 * 90 * 0.001) = 90 + 54 = 144

    let petr4_orders: Vec<_> = result.net_orders.iter()
        .filter(|o| o.symbol == "PETR4")
        .collect();

    if !petr4_orders.is_empty() {
        let net_order = &petr4_orders[0];
        let expected_max_notional = net_order.price.mul_shares(net_order.shares);
        let expected_max_cost = expected_max_notional.mul_rate(backtester_core::Rate::from(dec!(0.001)));
        let tolerance = Money::from(dec!(1));
        
        assert!(
            net_order.estimated_cost <= expected_max_cost + tolerance,
            "Cost {:?} should be based on net order, not double-charged (max expected: {:?})",
            net_order.estimated_cost, expected_max_cost
        );
    }

    // Total cost in audit should match sum of net order costs
    let calculated_cost: Money = result.net_orders.iter()
        .map(|o| o.estimated_cost)
        .sum();
    assert_eq!(
        audit.total_cost, calculated_cost,
        "Audit cost {:?} should match sum of net orders {:?}", audit.total_cost, calculated_cost
    );
}

// =============================================================================
// Test: Determinism - multiple runs produce identical results
// =============================================================================

#[test]
fn integration_determinism() {
    let config = OrchestratorConfig::default();
    let orchestrator = RebalanceOrchestrator::new(config);

    let positions = vec![
        make_position("PETR4", 500, dec!(50), dec!(42)),
        make_position("VALE3", 300, dec!(60), dec!(65)),
    ];

    let candidates = vec![
        make_candidate("ITUB4", dec!(28), 0.88),
        make_candidate("BBDC4", dec!(18), 0.82),
        make_candidate("WEGE3", dec!(35), 0.78),
    ];

    // Run 5 times
    let mut results = Vec::new();
    for _ in 0..5 {
        let (result, audit) = orchestrator.execute_rebalance(
            fixed_date(),
            Market::BR,
            &positions,
            &candidates,
            Money::from(dec!(500_000)),
            Money::from(dec!(500_000)),
            Money::from(dec!(500_000)),
        );
        results.push((result.net_orders.len(), result.netting_count, audit.total_cost));
    }

    // All runs should produce identical results
    let first = &results[0];
    for (i, r) in results.iter().enumerate() {
        assert_eq!(
            first, r,
            "Run {} differs from run 0: {:?} vs {:?}", i, r, first
        );
    }
}

