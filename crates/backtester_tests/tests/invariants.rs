//! # Invariants Test Suite
//!
//! Validates portfolio and system invariants cannot be violated.

use backtester_core::{
    AssetId, FillEvent, FillId, OrderDirection, OrderEvent, OrderId, OrderType, TimeInForce,
};
use backtester_portfolio::Portfolio;

fn make_fill(asset_id: u16, quantity: i64, price: f64, cost: f64) -> FillEvent {
    let direction = if quantity >= 0 {
        OrderDirection::Buy
    } else {
        OrderDirection::Sell
    };
    FillEvent::new(
        FillId::new(1),
        OrderId::new(1),
        0,
        AssetId::new(asset_id),
        direction,
        quantity.abs(),
        price,
        cost,
        0.0,
    )
}

#[test]
fn invariant_initial_nav_equals_capital() {
    let portfolio = Portfolio::new(100_000.0, 10);
    assert!(
        (portfolio.nav() - portfolio.initial_capital).abs() < f64::EPSILON,
        "Initial NAV must equal initial capital"
    );
}

#[test]
fn invariant_drawdown_never_negative() {
    let mut portfolio = Portfolio::new(100_000.0, 10);
    portfolio.update_drawdown();
    assert!(portfolio.drawdown() >= 0.0, "Drawdown cannot be negative");
}

#[test]
fn invariant_fill_updates_cash() {
    let mut portfolio = Portfolio::new(100_000.0, 10);
    let initial_cash = portfolio.cash();

    let fill = make_fill(0, 100, 50.0, 10.0);
    portfolio.process_fill(&fill);

    let expected_cash = initial_cash - (100.0 * 50.0) - 10.0;
    assert!(
        (portfolio.cash() - expected_cash).abs() < f64::EPSILON,
        "Cash must be correctly updated after fill"
    );
}

#[test]
fn invariant_b3_round_lot_validation() {
    let valid = OrderEvent::market_buy(OrderId::new(1), 0, AssetId::new(0), 100);
    assert!(valid.is_valid_b3_lot(), "100 shares is valid B3 lot");

    let also_valid = OrderEvent::market_sell(OrderId::new(2), 0, AssetId::new(0), 200);
    assert!(also_valid.is_valid_b3_lot(), "200 shares is valid B3 lot");

    let invalid = OrderEvent {
        order_id: OrderId::new(3),
        timestamp: 0,
        asset_id: AssetId::new(0),
        direction: OrderDirection::Buy,
        quantity: 50,
        order_type: OrderType::Market,
        limit_price: None,
        stop_price: None,
        time_in_force: TimeInForce::Day,
    };
    assert!(!invalid.is_valid_b3_lot(), "50 shares is NOT valid B3 lot");
}

#[test]
fn invariant_position_tracking() {
    let mut portfolio = Portfolio::new(100_000.0, 10);

    // Initial position is zero
    assert_eq!(portfolio.get_position(AssetId::new(0)), 0);

    // After buy, position is positive
    portfolio.process_fill(&make_fill(0, 100, 50.0, 0.0));
    assert_eq!(portfolio.get_position(AssetId::new(0)), 100);

    // After partial sell, position is reduced
    portfolio.process_fill(&make_fill(0, -50, 55.0, 0.0));
    assert_eq!(portfolio.get_position(AssetId::new(0)), 50);
}

#[test]
fn invariant_flat_portfolio_check() {
    let mut portfolio = Portfolio::new(100_000.0, 10);
    assert!(portfolio.is_flat(), "New portfolio should be flat");

    portfolio.process_fill(&make_fill(0, 100, 50.0, 0.0));
    assert!(!portfolio.is_flat(), "Portfolio with position is not flat");

    portfolio.process_fill(&make_fill(0, -100, 55.0, 0.0));
    assert!(portfolio.is_flat(), "Closed position should be flat");
}


// =============================================================================
// EXECUTION COST INVARIANTS (NEW)
// =============================================================================

#[test]
fn invariant_execution_config_zero_cost_has_no_costs() {
    use backtester_execution::ExecutionModelConfig;
    
    let config = ExecutionModelConfig::zero_cost();
    assert!(!config.has_costs(), "Zero cost config should have no costs");
}

#[test]
fn invariant_execution_config_mvp_has_costs() {
    use backtester_execution::ExecutionModelConfig;
    
    let config = ExecutionModelConfig::mvp();
    assert!(config.has_costs(), "MVP config should have costs");
}

#[test]
fn invariant_scale_costs_increases_slippage() {
    use backtester_execution::ExecutionModelConfig;
    
    let config = ExecutionModelConfig::mvp();
    let original_bps = config.slippage.base_bps();
    
    let scaled = config.scale_costs(2.0);
    let scaled_bps = scaled.slippage.base_bps();
    
    assert!(
        (scaled_bps - original_bps * 2.0).abs() < f64::EPSILON,
        "Scaling by 2x should double slippage bps"
    );
}

#[test]
fn invariant_add_delay_increases_delay() {
    use backtester_execution::ExecutionModelConfig;
    
    let config = ExecutionModelConfig::mvp();
    let original_delay = config.delay_bars;
    
    let delayed = config.add_delay(1);
    
    assert_eq!(
        delayed.delay_bars,
        original_delay + 1,
        "Adding 1 delay should increase delay_bars by 1"
    );
}

#[test]
fn invariant_fee_tier_has_positive_costs() {
    use backtester_execution::{FeeModelConfig, FeeTier};
    
    let b3_retail = FeeModelConfig::from_tier(FeeTier::B3Retail);
    assert!(b3_retail.has_any_cost(), "B3Retail should have costs");
    assert!(b3_retail.fixed_per_trade > 0.0, "B3Retail should have fixed costs");
    
    let custom_zero = FeeModelConfig::zero();
    assert!(!custom_zero.has_any_cost(), "Zero fee should have no costs");
}

#[test]
fn invariant_slippage_validation() {
    use backtester_execution::SlippageModelConfig;
    
    let valid = SlippageModelConfig::Constant { bps: 10.0 };
    assert!(valid.validate().is_ok(), "Positive bps should be valid");
    
    let invalid = SlippageModelConfig::Constant { bps: -5.0 };
    assert!(invalid.validate().is_err(), "Negative bps should be invalid");
}

#[test]
fn invariant_gate_checker_passes_good_candidate() {
    use backtester_execution::{GateChecker, InstitutionalGatesConfig};
    
    let checker = GateChecker::new(InstitutionalGatesConfig::default());
    
    let result = checker.check(
        8.0,         // turnover < 12
        100_000.0,   // gross_pnl
        10_000.0,    // total_costs (10% of pnl)
        10.0,        // avg_slippage_bps < 25
        10_000_000.0, // capacity > 5M
    );
    
    assert!(result.passed, "Good candidate should pass all gates");
    assert!(result.rejection_reasons.is_empty(), "Should have no rejections");
}

#[test]
fn invariant_gate_checker_rejects_high_turnover() {
    use backtester_execution::{GateChecker, InstitutionalGatesConfig};
    
    let checker = GateChecker::new(InstitutionalGatesConfig::default());
    
    let result = checker.check(
        20.0,        // turnover > 12 - FAIL
        100_000.0,
        10_000.0,
        10.0,
        10_000_000.0,
    );
    
    assert!(!result.passed, "High turnover should fail gates");
    assert!(!result.rejection_reasons.is_empty(), "Should have rejection reasons");
}

#[test]
fn invariant_stress_suite_has_5_scenarios() {
    use backtester_execution::StressSuite;
    
    let suite = StressSuite::default_institutional();
    assert_eq!(suite.len(), 5, "Default suite should have 5 scenarios");
    assert!(suite.get("S1").is_some(), "Should have S1");
    assert!(suite.get("S2").is_some(), "Should have S2");
    assert!(suite.get("S3").is_some(), "Should have S3");
    assert!(suite.get("S4").is_some(), "Should have S4");
    assert!(suite.get("S5").is_some(), "Should have S5");
}

#[test]
fn invariant_stress_acceptance_criteria() {
    use backtester_execution::AcceptanceCriteria;
    
    let criteria = AcceptanceCriteria {
        min_oos_sharpe: 0.3,
        min_execution_rate: Some(0.8),
        max_oos_mdd: Some(0.25),
    };
    
    // Should pass
    assert!(criteria.check(0.5, Some(0.9), Some(0.20)));
    
    // Should fail - low sharpe
    assert!(!criteria.check(0.2, Some(0.9), Some(0.20)));
}

#[test]
fn invariant_cost_report_builder() {
    use backtester_execution::cost_report::{CostReportBuilder, TradeCostRecord};
    
    let mut builder = CostReportBuilder::new();
    
    builder.add_trade(TradeCostRecord::new(
        "2024-01-01".into(),
        "PETR4".into(),
        "BR".into(),
        "Buy".into(),
        100,
        10_000.0,
        10.0,
        5.0,
    ));
    
    let report = builder.build();
    
    assert_eq!(report.trades_count, 1, "Should have 1 trade");
    assert!(report.total_costs > 0.0, "Should have costs");
}
