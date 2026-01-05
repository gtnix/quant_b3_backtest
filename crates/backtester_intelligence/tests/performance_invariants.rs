//! Performance Module Invariant Tests
//!
//! Proves accounting invariants that must always hold.

use backtester_intelligence::performance::{
    TradeLedger, PerformanceEngine, AttributionEngine,
    PerformanceSnapshot, PnLBreakdown, TurnoverMetrics,
};
use backtester_intelligence::filters::Market;
use backtester_intelligence::performance::engine::PerformanceConfig;
use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::BTreeMap;

fn make_prices(data: &[(&str, Decimal)]) -> BTreeMap<String, Decimal> {
    data.iter().map(|(s, p)| (s.to_string(), *p)).collect()
}

fn make_date(day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(2025, 1, day).unwrap()
}

// ==============================================================
// A) Equity / Cash Invariants
// ==============================================================

#[test]
fn invariant_equity_equals_cash_plus_mtm() {
    let mut engine = PerformanceEngine::new(PerformanceConfig::default(), dec!(100000));
    
    engine.record_buy(make_date(1), "PETR4", 100, dec!(30), dec!(10), Market::BR);
    engine.record_buy(make_date(1), "VALE3", 200, dec!(50), dec!(15), Market::BR);
    
    let prices = make_prices(&[("PETR4", dec!(35)), ("VALE3", dec!(55))]);
    let cash = dec!(86975); // 100000 - 3000 - 10000 - 25
    
    let snapshot = engine.generate_snapshot(make_date(2), cash, &prices);
    
    // Equity should equal cash + market value
    let market_value = dec!(35) * dec!(100) + dec!(55) * dec!(200); // 3500 + 11000 = 14500
    let expected_equity = cash + market_value;
    
    assert_eq!(snapshot.equity, expected_equity);
}

#[test]
fn invariant_pnl_realized_plus_unrealized_equals_total() {
    let mut ledger = TradeLedger::new();
    
    ledger.record_buy(make_date(1), "PETR4", 100, dec!(30), dec!(10), Market::BR);
    ledger.record_buy(make_date(1), "VALE3", 200, dec!(50), dec!(15), Market::BR);
    ledger.record_sell(make_date(2), "PETR4", 50, dec!(35), dec!(5), Market::BR);
    
    let prices = make_prices(&[("PETR4", dec!(36)), ("VALE3", dec!(52))]);
    let pnl = ledger.get_pnl_breakdown(&prices);
    
    assert_eq!(pnl.total, pnl.realized + pnl.unrealized);
}

#[test]
fn invariant_costs_never_negative() {
    let mut ledger = TradeLedger::new();
    
    for i in 0..20 {
        let symbol = format!("SYM{}", i);
        ledger.record_buy(make_date(1), &symbol, 100, dec!(10), dec!(5), Market::BR);
        ledger.record_sell(make_date(2), &symbol, 50, dec!(12), dec!(3), Market::BR);
    }
    
    let costs = ledger.costs();
    
    assert!(costs.fees_br >= Decimal::ZERO);
    assert!(costs.fees_us >= Decimal::ZERO);
    assert!(costs.slippage_br >= Decimal::ZERO);
    assert!(costs.slippage_us >= Decimal::ZERO);
    assert!(costs.total >= Decimal::ZERO);
}

#[test]
fn invariant_ledger_reconciliation() {
    let mut ledger = TradeLedger::new();
    
    // Complex trading sequence
    ledger.record_buy(make_date(1), "A", 100, dec!(10), dec!(1), Market::BR);
    ledger.record_buy(make_date(2), "A", 50, dec!(12), dec!(1), Market::BR);
    ledger.record_sell(make_date(3), "A", 80, dec!(15), dec!(1), Market::BR);
    ledger.record_buy(make_date(4), "B", 200, dec!(20), dec!(2), Market::US);
    ledger.record_sell(make_date(5), "B", 100, dec!(18), dec!(1), Market::US);
    
    let prices = make_prices(&[("A", dec!(14)), ("B", dec!(22))]);
    
    assert!(ledger.verify_reconciliation(&prices));
}

// ==============================================================
// B) Attribution Invariants
// ==============================================================

#[test]
fn invariant_attribution_sums_to_total() {
    let mut attr = AttributionEngine::new();
    
    attr.record_entry_weights("PETR4", [
        ("momentum".to_string(), dec!(0.4)),
        ("value".to_string(), dec!(0.3)),
        ("quality".to_string(), dec!(0.3)),
    ].into());
    
    attr.record_entry_weights("VALE3", [
        ("momentum".to_string(), dec!(0.5)),
        ("value".to_string(), dec!(0.5)),
    ].into());
    
    let pnl = [
        ("PETR4".to_string(), dec!(1000)),
        ("VALE3".to_string(), dec!(500)),
    ].into();
    
    let breakdown = attr.calculate_attribution(&pnl);
    
    let sum: Decimal = breakdown.by_technique.iter()
        .map(|t| t.pnl_contribution)
        .sum();
    
    assert_eq!(sum + breakdown.residual, breakdown.total_pnl);
}

#[test]
fn invariant_attribution_residual_for_missing_weights() {
    let attr = AttributionEngine::new(); // No weights recorded
    
    let pnl = [
        ("PETR4".to_string(), dec!(1000)),
        ("VALE3".to_string(), dec!(500)),
    ].into();
    
    let breakdown = attr.calculate_attribution(&pnl);
    
    // All goes to residual
    assert_eq!(breakdown.residual, dec!(1500));
    assert!(breakdown.by_technique.is_empty());
}

// ==============================================================
// C) Determinism Invariants
// ==============================================================

#[test]
fn invariant_determinism_ledger() {
    fn run_scenario() -> (PnLBreakdown, Decimal) {
        let mut ledger = TradeLedger::new();
        
        ledger.record_buy(make_date(1), "PETR4", 100, dec!(30), dec!(10), Market::BR);
        ledger.record_buy(make_date(1), "VALE3", 200, dec!(50), dec!(15), Market::BR);
        ledger.record_sell(make_date(2), "PETR4", 50, dec!(35), dec!(5), Market::BR);
        
        let prices = make_prices(&[("PETR4", dec!(36)), ("VALE3", dec!(52))]);
        let pnl = ledger.get_pnl_breakdown(&prices);
        let mv = ledger.market_value(&prices);
        
        (pnl, mv)
    }
    
    let (pnl1, mv1) = run_scenario();
    let (pnl2, mv2) = run_scenario();
    
    assert_eq!(pnl1.total, pnl2.total);
    assert_eq!(pnl1.realized, pnl2.realized);
    assert_eq!(pnl1.unrealized, pnl2.unrealized);
    assert_eq!(mv1, mv2);
}

#[test]
fn invariant_determinism_attribution() {
    fn run_scenario() -> Vec<(String, Decimal)> {
        let mut attr = AttributionEngine::new();
        
        attr.record_entry_weights("PETR4", [
            ("momentum".to_string(), dec!(0.6)),
            ("value".to_string(), dec!(0.4)),
        ].into());
        
        let pnl = [("PETR4".to_string(), dec!(1000))].into();
        let breakdown = attr.calculate_attribution(&pnl);
        
        breakdown.by_technique.iter()
            .map(|t| (t.technique_name.clone(), t.pnl_contribution))
            .collect()
    }
    
    let result1 = run_scenario();
    let result2 = run_scenario();
    
    assert_eq!(result1, result2);
}

#[test]
fn invariant_determinism_snapshot() {
    fn run_scenario() -> Decimal {
        let mut engine = PerformanceEngine::new(PerformanceConfig::default(), dec!(100000));
        
        engine.record_buy(make_date(1), "PETR4", 100, dec!(30), dec!(10), Market::BR);
        
        let prices = make_prices(&[("PETR4", dec!(35))]);
        let snapshot = engine.generate_snapshot(make_date(2), dec!(97000), &prices);
        
        snapshot.equity
    }
    
    let eq1 = run_scenario();
    let eq2 = run_scenario();
    
    assert_eq!(eq1, eq2);
}

// ==============================================================
// D) WAP Correctness Invariants
// ==============================================================

#[test]
fn invariant_wap_formula_correct() {
    let mut ledger = TradeLedger::new();
    
    // Buy 100 @ 30
    ledger.record_buy(make_date(1), "A", 100, dec!(30), dec!(0), Market::BR);
    // Buy 100 @ 40
    ledger.record_buy(make_date(2), "A", 100, dec!(40), dec!(0), Market::BR);
    
    let pos = ledger.positions().get("A").unwrap();
    
    // WAP = (100*30 + 100*40) / 200 = 7000 / 200 = 35
    assert_eq!(pos.wap_cost_basis, dec!(35));
    assert_eq!(pos.shares, 200);
}

#[test]
fn invariant_realized_pnl_correct() {
    let mut ledger = TradeLedger::new();
    
    ledger.record_buy(make_date(1), "A", 100, dec!(30), dec!(0), Market::BR);
    
    // Sell 50 @ 40 -> realized = (40-30)*50 = 500
    let realized = ledger.record_sell(make_date(2), "A", 50, dec!(40), dec!(0), Market::BR);
    
    assert_eq!(realized, dec!(500));
    assert_eq!(ledger.realized_pnl(), dec!(500));
}

#[test]
fn invariant_unrealized_pnl_correct() {
    let mut ledger = TradeLedger::new();
    
    ledger.record_buy(make_date(1), "A", 100, dec!(30), dec!(0), Market::BR);
    
    let prices = make_prices(&[("A", dec!(40))]);
    let unrealized = ledger.get_unrealized_pnl(&prices);
    
    // (40-30)*100 = 1000
    assert_eq!(unrealized, dec!(1000));
}

// ==============================================================
// E) Turnover Invariants
// ==============================================================

#[test]
fn invariant_turnover_calculation() {
    let turnover = TurnoverMetrics::from_notionals(dec!(10000), dec!(8000), dec!(100000));
    
    // (10000 + 8000) / 100000 * 100 = 18%
    assert_eq!(turnover.turnover_pct, dec!(18));
}

#[test]
fn invariant_turnover_zero_portfolio() {
    let turnover = TurnoverMetrics::from_notionals(dec!(1000), dec!(500), dec!(0));
    
    // Division by zero protection
    assert_eq!(turnover.turnover_pct, Decimal::ZERO);
}
































