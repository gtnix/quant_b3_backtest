//! Performance Module Stress Tests
//!
//! Validates behavior under extreme conditions.

use backtester_intelligence::performance::{
    TradeLedger, PerformanceEngine, AttributionEngine, RiskCalculator,
};
use backtester_intelligence::filters::Market;
use backtester_intelligence::performance::engine::PerformanceConfig;
use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::BTreeMap;
use std::time::Instant;

fn make_prices(data: &[(&str, Decimal)]) -> BTreeMap<String, Decimal> {
    data.iter().map(|(s, p)| (s.to_string(), *p)).collect()
}

fn make_date(day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(2025, 1, day.min(28)).unwrap()
}

// ==============================================================
// A) Flash Crash Scenarios
// ==============================================================

#[test]
fn stress_flash_crash_50pct_drop() {
    let mut engine = PerformanceEngine::new(PerformanceConfig::default(), dec!(100000));
    
    // Build positions
    for i in 0..10 {
        let symbol = format!("SYM{}", i);
        engine.record_buy(make_date(1), &symbol, 100, dec!(100), dec!(10), Market::BR);
    }
    
    // Normal prices - snapshot 1
    let prices_normal: BTreeMap<String, Decimal> = (0..10)
        .map(|i| (format!("SYM{}", i), dec!(100)))
        .collect();
    let snap1 = engine.generate_snapshot(make_date(1), dec!(0), &prices_normal);
    
    // Flash crash - 50% drop
    let prices_crash: BTreeMap<String, Decimal> = (0..10)
        .map(|i| (format!("SYM{}", i), dec!(50)))
        .collect();
    let snap2 = engine.generate_snapshot(make_date(2), dec!(0), &prices_crash);
    
    // Equity should be ~50% of initial
    assert!(snap2.equity < snap1.equity);
    assert!(snap2.drawdown.current_dd > dec!(0.4)); // At least 40% drawdown
}

#[test]
fn stress_flash_crash_recovery() {
    let mut engine = PerformanceEngine::new(PerformanceConfig::default(), dec!(100000));
    
    engine.record_buy(make_date(1), "A", 1000, dec!(100), dec!(100), Market::BR);
    
    // Initial
    let p1 = make_prices(&[("A", dec!(100))]);
    engine.generate_snapshot(make_date(1), dec!(0), &p1);
    
    // Crash
    let p2 = make_prices(&[("A", dec!(50))]);
    engine.generate_snapshot(make_date(2), dec!(0), &p2);
    
    // Recovery
    let p3 = make_prices(&[("A", dec!(120))]);
    let snap = engine.generate_snapshot(make_date(3), dec!(0), &p3);
    
    // Should have new high, no current drawdown
    assert_eq!(snap.drawdown.current_dd, Decimal::ZERO);
    assert!(snap.drawdown.max_dd > Decimal::ZERO); // Max DD from crash still recorded
}

// ==============================================================
// B) High Turnover Scenarios
// ==============================================================

#[test]
fn stress_100pct_turnover() {
    let mut ledger = TradeLedger::new();
    
    // Initial positions
    for i in 0..20 {
        let symbol = format!("SYM{}", i);
        ledger.record_buy(make_date(1), &symbol, 100, dec!(50), dec!(5), Market::BR);
    }
    
    // Full liquidation
    for i in 0..20 {
        let symbol = format!("SYM{}", i);
        ledger.record_sell(make_date(2), &symbol, 100, dec!(55), dec!(3), Market::BR);
    }
    
    // New positions
    for i in 20..40 {
        let symbol = format!("SYM{}", i);
        ledger.record_buy(make_date(2), &symbol, 100, dec!(45), dec!(5), Market::BR);
    }
    
    // Verify all original positions closed
    for i in 0..20 {
        let symbol = format!("SYM{}", i);
        assert!(ledger.positions().get(&symbol).is_none());
    }
    
    // Verify new positions exist
    for i in 20..40 {
        let symbol = format!("SYM{}", i);
        assert!(ledger.positions().get(&symbol).is_some());
    }
    
    // Costs should be significant
    assert!(ledger.costs().total > dec!(200));
}

// ==============================================================
// C) Large Portfolio Scenarios
// ==============================================================

#[test]
fn stress_large_portfolio_10k_positions() {
    let start = Instant::now();
    
    let mut ledger = TradeLedger::new();
    
    // Create 10,000 positions
    for i in 0..10_000 {
        let symbol = format!("SYM{:05}", i);
        let market = if i % 2 == 0 { Market::BR } else { Market::US };
        ledger.record_buy(make_date(1), &symbol, 100, dec!(10), dec!(1), market);
    }
    
    // Generate prices
    let prices: BTreeMap<String, Decimal> = (0..10_000)
        .map(|i| (format!("SYM{:05}", i), dec!(11)))
        .collect();
    
    // Get P&L breakdown
    let pnl = ledger.get_pnl_breakdown(&prices);
    
    let elapsed = start.elapsed();
    
    // Performance: should complete in < 500ms
    assert!(elapsed.as_millis() < 500, "Took {:?}", elapsed);
    
    // Correctness
    assert_eq!(ledger.positions().len(), 10_000);
    assert_eq!(pnl.unrealized, dec!(1000000)); // 10000 * 100 * 1 = 1M
}

#[test]
fn stress_large_equity_curve_100_rebalances() {
    let start = Instant::now();
    
    let mut engine = PerformanceEngine::new(PerformanceConfig::default(), dec!(100000));
    
    engine.record_buy(make_date(1), "A", 1000, dec!(100), dec!(100), Market::BR);
    
    // 100 rebalances
    for i in 1..=100 {
        let price = dec!(100) + Decimal::from(i);
        let prices = make_prices(&[("A", price)]);
        engine.generate_snapshot(make_date((i as u32).min(28)), dec!(0), &prices);
    }
    
    let elapsed = start.elapsed();
    
    // Performance
    assert!(elapsed.as_millis() < 100, "Took {:?}", elapsed);
    
    // Correctness
    assert_eq!(engine.equity_curve().len(), 100);
    assert_eq!(engine.daily_returns().len(), 99);
}

// ==============================================================
// D) Edge Cases
// ==============================================================

#[test]
fn stress_zero_price_positions() {
    let mut ledger = TradeLedger::new();
    
    ledger.record_buy(make_date(1), "A", 100, dec!(10), dec!(1), Market::BR);
    
    // Price goes to zero (delisting, etc.)
    let prices = make_prices(&[("A", dec!(0))]);
    let pnl = ledger.get_pnl_breakdown(&prices);
    
    // Should handle gracefully, 100% loss
    assert_eq!(pnl.unrealized, dec!(-1000)); // (0-10)*100
}

#[test]
fn stress_very_small_prices() {
    let mut ledger = TradeLedger::new();
    
    // Penny stock
    ledger.record_buy(make_date(1), "PENNY", 10000, dec!(0.01), dec!(1), Market::BR);
    
    let prices = make_prices(&[("PENNY", dec!(0.02))]);
    let pnl = ledger.get_pnl_breakdown(&prices);
    
    // 10000 * (0.02 - 0.01) = 100
    assert_eq!(pnl.unrealized, dec!(100));
}

#[test]
fn stress_very_large_prices() {
    let mut ledger = TradeLedger::new();
    
    // Expensive stock (BRK.A style)
    ledger.record_buy(make_date(1), "BRK.A", 1, dec!(500000), dec!(1000), Market::US);
    
    let prices = make_prices(&[("BRK.A", dec!(510000))]);
    let pnl = ledger.get_pnl_breakdown(&prices);
    
    assert_eq!(pnl.unrealized, dec!(10000));
}

#[test]
fn stress_missing_prices() {
    let mut ledger = TradeLedger::new();
    
    ledger.record_buy(make_date(1), "A", 100, dec!(10), dec!(1), Market::BR);
    ledger.record_buy(make_date(1), "B", 200, dec!(20), dec!(2), Market::BR);
    
    // Only A has a price
    let prices = make_prices(&[("A", dec!(15))]);
    let pnl = ledger.get_pnl_breakdown(&prices);
    
    // Only A's unrealized counted
    assert_eq!(pnl.unrealized, dec!(500)); // (15-10)*100
}

// ==============================================================
// E) Risk Calculation Stress
// ==============================================================

#[test]
fn stress_var_extreme_returns() {
    let calc = RiskCalculator::default();
    
    // Extreme daily returns
    let returns: Vec<Decimal> = (-50..50)
        .map(|i| Decimal::from(i) / Decimal::from(100))
        .collect();
    
    let var = calc.calculate_var(&returns, dec!(1000000));
    
    // VaR should be significantly negative
    assert!(var.var_95 < dec!(-400000));
    assert!(var.var_99 < var.var_95);
}

#[test]
fn stress_drawdown_many_drops() {
    let calc = RiskCalculator::default();
    
    // Sawtooth equity curve
    let mut equity: Vec<Decimal> = Vec::new();
    let mut value = dec!(100);
    
    for _ in 0..50 {
        value = value + dec!(10);
        equity.push(value);
        value = value - dec!(5);
        equity.push(value);
    }
    
    let dd = calc.calculate_drawdown(&equity);
    
    // Should track max drawdown correctly
    assert!(dd.max_dd > Decimal::ZERO);
    assert!(dd.hwm >= *equity.last().unwrap());
}

// ==============================================================
// F) Attribution Stress
// ==============================================================

#[test]
fn stress_attribution_many_techniques() {
    let mut attr = AttributionEngine::new();
    
    // 7 techniques per symbol
    for i in 0..100 {
        let symbol = format!("SYM{}", i);
        let weights: BTreeMap<String, Decimal> = [
            ("momentum", dec!(0.15)),
            ("value", dec!(0.15)),
            ("quality", dec!(0.15)),
            ("low_vol", dec!(0.15)),
            ("dividend", dec!(0.15)),
            ("size", dec!(0.15)),
            ("carry", dec!(0.10)),
        ].iter().map(|(k, v)| (k.to_string(), *v)).collect();
        
        attr.record_entry_weights(&symbol, weights);
    }
    
    // Generate P&L
    let pnl: BTreeMap<String, Decimal> = (0..100)
        .map(|i| (format!("SYM{}", i), Decimal::from(i * 100)))
        .collect();
    
    let start = Instant::now();
    let breakdown = attr.calculate_attribution(&pnl);
    let elapsed = start.elapsed();
    
    // Performance
    assert!(elapsed.as_millis() < 50, "Took {:?}", elapsed);
    
    // Correctness
    assert_eq!(breakdown.by_technique.len(), 7);
    
    let sum: Decimal = breakdown.by_technique.iter()
        .map(|t| t.pnl_contribution)
        .sum();
    assert_eq!(sum + breakdown.residual, breakdown.total_pnl);
}

// ==============================================================
// G) Performance Smoke Test
// ==============================================================

#[test]
fn perf_smoke_snapshot_100_under_10ms() {
    let start = Instant::now();
    
    let mut engine = PerformanceEngine::new(PerformanceConfig::default(), dec!(1000000));
    
    // 100 positions
    for i in 0..100 {
        let symbol = format!("SYM{}", i);
        engine.record_buy(make_date(1), &symbol, 100, dec!(100), dec!(10), Market::BR);
    }
    
    let prices: BTreeMap<String, Decimal> = (0..100)
        .map(|i| (format!("SYM{}", i), dec!(105)))
        .collect();
    
    // Generate snapshot
    let _snapshot = engine.generate_snapshot(make_date(2), dec!(0), &prices);
    
    let elapsed = start.elapsed();
    
    assert!(elapsed.as_millis() < 10, "Snapshot took {:?}, expected < 10ms", elapsed);
}



