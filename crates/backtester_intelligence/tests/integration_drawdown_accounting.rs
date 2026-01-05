//! Integration tests for Drawdown Accounting.
//!
//! Proves that equity, drawdown, and RiskGuard actions are consistent.

use std::collections::HashMap;

use backtester_intelligence::accounting::PortfolioState;
use backtester_intelligence::exit::{
    DrawdownAction, ExitContext, ExitEngine, ExitEngineConfig, ExitPolicyConfig,
    RiskConfig,
};
use backtester_intelligence::filters::Market;
use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

fn make_date(day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(2025, 1, day).unwrap()
}

// =============================================================================
// Test: Drawdown trigger during flash crash
// =============================================================================

#[test]
fn integration_drawdown_flash_crash() {
    // Scenario: Market drops 30%, drawdown exceeds 15% limit

    let config = ExitEngineConfig {
        policy: ExitPolicyConfig {
            enable_stop_loss: false, // Disable individual exits
            enable_take_profit: false,
            enable_time_exit: false,
            ..Default::default()
        },
        risk: RiskConfig {
            max_drawdown_pct: -0.15,
            drawdown_action: DrawdownAction::CashOut,
            check_drawdown: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let exit_engine = ExitEngine::new(config);
    let mut portfolio = PortfolioState::new(dec!(1_000_000));

    // Build initial portfolio
    portfolio.apply_buy("PETR4", 5000, dec!(50), dec!(0), Market::BR, make_date(1)).unwrap();
    portfolio.apply_buy("VALE3", 3000, dec!(60), dec!(0), Market::BR, make_date(1)).unwrap();
    portfolio.apply_buy("ITUB4", 4000, dec!(30), dec!(0), Market::BR, make_date(1)).unwrap();

    // Initial equity = 1M - (5000*50 + 3000*60 + 4000*30) + positions value
    // = 1M - 550k + 550k = 1M (just bought at market price)
    assert_eq!(portfolio.equity.to_decimal(), dec!(1_000_000));

    // Simulate flash crash: all positions drop 25%
    let mut crash_prices = HashMap::new();
    crash_prices.insert("PETR4".to_string(), dec!(37.5)); // -25%
    crash_prices.insert("VALE3".to_string(), dec!(45)); // -25%
    crash_prices.insert("ITUB4".to_string(), dec!(22.5)); // -25%
    portfolio.update_prices(&crash_prices);

    // New equity: 450k (cash) + 5000*37.5 + 3000*45 + 4000*22.5
    // = 450k + 187.5k + 135k + 90k = 862.5k
    // Wait, let me recalculate cash after buys:
    // Cash = 1M - (5000*50 + 3000*60 + 4000*30) = 1M - (250k + 180k + 120k) = 1M - 550k = 450k
    // Positions at crash: 5000*37.5 + 3000*45 + 4000*22.5 = 187.5k + 135k + 90k = 412.5k
    // Equity = 450k + 412.5k = 862.5k
    
    // Drawdown = (862.5k - 1M) / 1M = -13.75%
    // Hmm, that's not enough to trigger -15% threshold
    
    // Let's make it a 30% crash
    crash_prices.insert("PETR4".to_string(), dec!(35)); // -30%
    crash_prices.insert("VALE3".to_string(), dec!(42)); // -30%
    crash_prices.insert("ITUB4".to_string(), dec!(21)); // -30%
    portfolio.update_prices(&crash_prices);

    // Positions at crash: 5000*35 + 3000*42 + 4000*21 = 175k + 126k + 84k = 385k
    // Equity = 450k + 385k = 835k
    // Drawdown = (835k - 1M) / 1M = -16.5%

    let dd = portfolio.drawdown();
    assert!(
        dd <= -0.15,
        "Drawdown should exceed threshold: {} <= -0.15", dd
    );

    // Build ExitContext from portfolio
    // Need to set peak_equity manually since it's different from current equity
    use backtester_core::Money;
    let positions: Vec<_> = portfolio.positions.values().cloned().collect();
    let mut exit_ctx = ExitContext::new(
        make_date(5),
        portfolio.equity.to_decimal(),
        portfolio.equity.to_decimal(),
        Market::BR,
    );
    // Set peak_equity to initial capital (1M) for drawdown calculation
    exit_ctx.peak_equity = Money::from_int(1_000_000);

    let (exit_result, _, _) = exit_engine.evaluate(&positions, &exit_ctx);

    // RiskGuard should trigger CashOut
    assert!(
        !exit_result.diagnostics.risk_violations.is_empty(),
        "Should have risk violation"
    );

    // All positions should be marked for exit
    assert_eq!(
        exit_result.exits.len(), 3,
        "All 3 positions should be exited"
    );
}

// =============================================================================
// Test: Drawdown recovery resets peak
// =============================================================================

#[test]
fn integration_drawdown_recovery() {
    let mut portfolio = PortfolioState::new(dec!(1_000_000));

    // Buy position
    portfolio.apply_buy("PETR4", 10000, dec!(50), dec!(0), Market::BR, make_date(1)).unwrap();
    assert_eq!(portfolio.peak_equity.to_decimal(), dec!(1_000_000));

    // Drop 10%
    let mut prices = HashMap::new();
    prices.insert("PETR4".to_string(), dec!(45));
    portfolio.update_prices(&prices);
    
    let dd_after_drop = portfolio.drawdown();
    assert!(dd_after_drop < 0.0, "Should have drawdown");

    // Recover above peak
    prices.insert("PETR4".to_string(), dec!(55));
    portfolio.update_prices(&prices);

    // Peak should update
    assert!(
        portfolio.peak_equity.to_decimal() > dec!(1_000_000),
        "Peak should update on new high: {}", portfolio.peak_equity
    );

    // Drawdown should be zero
    assert!(
        portfolio.drawdown().abs() < 0.001,
        "Drawdown should be ~0 at new peak: {}", portfolio.drawdown()
    );
}

// =============================================================================
// Test: Equity invariant - always equals cash + positions
// =============================================================================

#[test]
fn integration_equity_invariant() {
    let mut portfolio = PortfolioState::new(dec!(500_000));

    // Series of operations
    portfolio.apply_buy("PETR4", 1000, dec!(50), dec!(50), Market::BR, make_date(1)).unwrap();
    assert_eq!(portfolio.equity, portfolio.calculate_equity_fast());

    portfolio.apply_buy("VALE3", 500, dec!(60), dec!(30), Market::BR, make_date(2)).unwrap();
    assert_eq!(portfolio.equity, portfolio.calculate_equity_fast());

    let mut prices = HashMap::new();
    prices.insert("PETR4".to_string(), dec!(55));
    prices.insert("VALE3".to_string(), dec!(58));
    portfolio.update_prices(&prices);
    assert_eq!(portfolio.equity, portfolio.calculate_equity_fast());

    portfolio.apply_sell("PETR4", 500, dec!(55), dec!(28)).unwrap();
    assert_eq!(portfolio.equity, portfolio.calculate_equity_fast());

    // Validate invariants
    portfolio.validate().expect("Invariants should hold");
}

// =============================================================================
// Test: Cash never goes negative without error
// =============================================================================

#[test]
fn integration_cash_never_negative() {
    let mut portfolio = PortfolioState::new(dec!(10_000));

    // Try to buy more than we can afford
    let result = portfolio.apply_buy("PETR4", 1000, dec!(50), dec!(0), Market::BR, make_date(1));
    
    assert!(result.is_err(), "Should reject buy exceeding cash");
    assert!(
        !portfolio.cash.is_negative(),
        "Cash should remain non-negative: {}", portfolio.cash
    );
}

// =============================================================================
// Test: Drawdown with RiskGuard ReduceRisk action
// =============================================================================

#[test]
fn integration_drawdown_reduce_risk() {
    let config = ExitEngineConfig {
        policy: ExitPolicyConfig {
            enable_stop_loss: false,
            enable_take_profit: false,
            enable_time_exit: false,
            ..Default::default()
        },
        risk: RiskConfig {
            max_drawdown_pct: -0.10,
            drawdown_action: DrawdownAction::ReduceRisk, // Only reduce, not cash out
            check_drawdown: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let exit_engine = ExitEngine::new(config);
    let mut portfolio = PortfolioState::new(dec!(1_000_000));

    // Build 5 positions - invest almost all capital
    for (sym, shares, price) in [
        ("PETR4", 4000, dec!(50)),   // 200k
        ("VALE3", 3000, dec!(60)),   // 180k
        ("ITUB4", 6000, dec!(30)),   // 180k
        ("BBDC4", 9000, dec!(20)),   // 180k
        ("WEGE3", 5000, dec!(40)),   // 200k = 940k total
    ] {
        portfolio.apply_buy(sym, shares, price, dec!(0), Market::BR, make_date(1)).unwrap();
    }

    // Drop 15% - should trigger >10% portfolio drawdown since we're 94% invested
    let mut prices = HashMap::new();
    prices.insert("PETR4".to_string(), dec!(42.5));  // -15%
    prices.insert("VALE3".to_string(), dec!(51));    // -15%
    prices.insert("ITUB4".to_string(), dec!(25.5));  // -15%
    prices.insert("BBDC4".to_string(), dec!(17));    // -15%
    prices.insert("WEGE3".to_string(), dec!(34));    // -15%
    portfolio.update_prices(&prices);

    use backtester_core::Money;
    let positions: Vec<_> = portfolio.positions.values().cloned().collect();
    let mut exit_ctx = ExitContext::new(
        make_date(10),
        portfolio.equity.to_decimal(),
        portfolio.equity.to_decimal(),
        Market::BR,
    );
    // Set peak_equity to initial capital (1M) for drawdown calculation
    exit_ctx.peak_equity = Money::from_int(1_000_000);

    let (exit_result, _, _) = exit_engine.evaluate(&positions, &exit_ctx);

    // ReduceRisk should exit top 20% by value (1 out of 5)
    assert!(
        exit_result.exits.len() >= 1 && exit_result.exits.len() < 5,
        "ReduceRisk should exit some but not all: {}", exit_result.exits.len()
    );
}

// =============================================================================
// Test: Determinism - same inputs produce same drawdown calculations
// =============================================================================

#[test]
fn integration_drawdown_determinism() {
    fn run_scenario() -> (Decimal, f64, usize) {
        let config = ExitEngineConfig {
            risk: RiskConfig {
                max_drawdown_pct: -0.10,
                drawdown_action: DrawdownAction::CashOut,
                ..Default::default()
            },
            ..Default::default()
        };

        let exit_engine = ExitEngine::new(config);
        let mut portfolio = PortfolioState::new(dec!(1_000_000));

        portfolio.apply_buy("PETR4", 5000, dec!(50), dec!(0), Market::BR, make_date(1)).unwrap();
        portfolio.apply_buy("VALE3", 3000, dec!(60), dec!(0), Market::BR, make_date(1)).unwrap();

        let mut prices = HashMap::new();
        prices.insert("PETR4".to_string(), dec!(40));
        prices.insert("VALE3".to_string(), dec!(48));
        portfolio.update_prices(&prices);

        use backtester_core::Money;
        let positions: Vec<_> = portfolio.positions.values().cloned().collect();
        let mut exit_ctx = ExitContext::new(
            make_date(5),
            portfolio.equity.to_decimal(),
            portfolio.equity.to_decimal(),
            Market::BR,
        );
        // Set peak_equity to initial capital for drawdown calculation
        exit_ctx.peak_equity = Money::from_int(1_000_000);

        let (exit_result, _, _) = exit_engine.evaluate(&positions, &exit_ctx);

        (portfolio.equity.to_decimal(), portfolio.drawdown(), exit_result.exits.len())
    }

    // Run 5 times
    let results: Vec<_> = (0..5).map(|_| run_scenario()).collect();

    // All should be identical
    let first = &results[0];
    for (i, r) in results.iter().enumerate() {
        assert_eq!(
            first, r,
            "Run {} differs: {:?} vs {:?}", i, first, r
        );
    }
}
