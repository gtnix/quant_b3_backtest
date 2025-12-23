//! # Invariants Test Suite
//!
//! Validates portfolio and system invariants cannot be violated.
//!
//! Examples:
//! - NAV = cash + positions value
//! - Cash cannot go negative without margin (not supported)
//! - Order quantities must be multiples of 100 (B3 round-lot)

use backtester_core::FillEvent;
use backtester_portfolio::Portfolio;

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
    assert!(portfolio.drawdown >= 0.0, "Drawdown cannot be negative");
}

#[test]
fn invariant_fill_updates_cash() {
    let mut portfolio = Portfolio::new(100_000.0, 10);
    let initial_cash = portfolio.cash;

    let fill = FillEvent {
        timestamp: 0,
        asset_id: 0,
        quantity: 100,
        price: 50.0,
        cost: 10.0,
    };
    portfolio.process_fill(&fill);

    let expected_cash = initial_cash - (100.0 * 50.0) - 10.0;
    assert!(
        (portfolio.cash - expected_cash).abs() < f64::EPSILON,
        "Cash must be correctly updated after fill"
    );
}

#[test]
fn invariant_b3_round_lot_validation() {
    use backtester_core::OrderEvent;
    
    let valid = OrderEvent {
        timestamp: 0,
        asset_id: 0,
        quantity: 100,
        limit_price: None,
    };
    assert!(valid.is_valid_b3_lot(), "100 shares is valid B3 lot");
    
    let also_valid = OrderEvent {
        timestamp: 0,
        asset_id: 0,
        quantity: -200,
        limit_price: None,
    };
    assert!(also_valid.is_valid_b3_lot(), "-200 shares is valid B3 lot");
    
    let invalid = OrderEvent {
        timestamp: 0,
        asset_id: 0,
        quantity: 50,
        limit_price: None,
    };
    assert!(!invalid.is_valid_b3_lot(), "50 shares is NOT valid B3 lot");
}

#[test]
fn invariant_position_tracking() {
    let mut portfolio = Portfolio::new(100_000.0, 10);
    
    // Initial position is zero
    assert_eq!(portfolio.get_position(0), 0);
    
    // After buy, position is positive
    portfolio.process_fill(&FillEvent {
        timestamp: 0,
        asset_id: 0,
        quantity: 100,
        price: 50.0,
        cost: 0.0,
    });
    assert_eq!(portfolio.get_position(0), 100);
    
    // After partial sell, position is reduced
    portfolio.process_fill(&FillEvent {
        timestamp: 1,
        asset_id: 0,
        quantity: -50,
        price: 55.0,
        cost: 0.0,
    });
    assert_eq!(portfolio.get_position(0), 50);
}

#[test]
fn invariant_flat_portfolio_check() {
    let mut portfolio = Portfolio::new(100_000.0, 10);
    assert!(portfolio.is_flat(), "New portfolio should be flat");
    
    portfolio.process_fill(&FillEvent {
        timestamp: 0,
        asset_id: 0,
        quantity: 100,
        price: 50.0,
        cost: 0.0,
    });
    assert!(!portfolio.is_flat(), "Portfolio with position is not flat");
    
    portfolio.process_fill(&FillEvent {
        timestamp: 1,
        asset_id: 0,
        quantity: -100,
        price: 55.0,
        cost: 0.0,
    });
    assert!(portfolio.is_flat(), "Closed position should be flat");
}
