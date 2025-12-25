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
