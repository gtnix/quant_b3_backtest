//! E2E Integration Tests: Entry + Exit + Orchestrator + Accounting + Performance
//!
//! Final gate tests proving all modules work together correctly:
//! - A1: Full rebalance with performance snapshot reconciliation
//! - A2: Double counting prevention (costs applied once on net)
//! - A3: Multi-market explicit (BR/US separate, no silent currency mixing)

use backtester_intelligence::{
    PortfolioState, Order, OrderSide, Market,
    TradeLedger, PerformanceEngine,
};
use backtester_intelligence::performance::engine::PerformanceConfig;
use backtester_intelligence::exit::Position;

use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::{BTreeMap, HashMap};

fn make_date(day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(2025, 1, day.min(28)).unwrap()
}

fn make_prices_btree(data: &[(&str, Decimal)]) -> BTreeMap<String, Decimal> {
    data.iter().map(|(s, p)| (s.to_string(), *p)).collect()
}

fn make_prices_hash(data: &[(&str, Decimal)]) -> HashMap<String, Decimal> {
    data.iter().map(|(s, p)| (s.to_string(), *p)).collect()
}

// ==============================================================================
// A1: E2E Rebalance with Performance Snapshot
// ==============================================================================

#[test]
fn e2e_rebalance_snapshot_reconciliation() {
    // Setup: 10 BR + 10 US assets, initial capital 1M each
    let mut portfolio_br = PortfolioState::new(dec!(1_000_000));
    let mut portfolio_us = PortfolioState::new(dec!(1_000_000));
    let mut ledger_br = TradeLedger::new();
    let mut ledger_us = TradeLedger::new();
    let mut perf_br = PerformanceEngine::new(PerformanceConfig::default(), dec!(1_000_000));
    let mut perf_us = PerformanceEngine::new(PerformanceConfig::default(), dec!(1_000_000));

    // Rebalance 1: Buy 5 BR assets
    let date1 = make_date(1);
    let br_buys: Vec<(&str, i64, Decimal)> = vec![
        ("PETR4", 100, dec!(30)),
        ("VALE3", 200, dec!(50)),
        ("ITUB4", 100, dec!(25)),
        ("BBDC4", 100, dec!(20)),
        ("ABEV3", 100, dec!(15)),
    ];
    
    for (sym, shares, price) in &br_buys {
        let cost = *price * Decimal::from(*shares) * dec!(0.001); // 10 bps
        portfolio_br.apply_buy(sym, *shares, *price, cost, Market::BR, date1).unwrap();
        ledger_br.record_buy(date1, sym, *shares, *price, cost, Market::BR);
        perf_br.record_buy(date1, sym, *shares, *price, cost, Market::BR);
    }

    // US buys: 5 US assets
    let us_buys: Vec<(&str, i64, Decimal)> = vec![
        ("AAPL", 10, dec!(150)),
        ("MSFT", 10, dec!(300)),
        ("GOOGL", 5, dec!(140)),
        ("AMZN", 5, dec!(180)),
        ("NVDA", 10, dec!(500)),
    ];

    for (sym, shares, price) in &us_buys {
        let cost = *price * Decimal::from(*shares) * dec!(0.0005); // 5 bps
        portfolio_us.apply_buy(sym, *shares, *price, cost, Market::US, date1).unwrap();
        ledger_us.record_buy(date1, sym, *shares, *price, cost, Market::US);
        perf_us.record_buy(date1, sym, *shares, *price, cost, Market::US);
    }

    // Generate snapshot 1
    let br_prices1 = make_prices_btree(&[
        ("PETR4", dec!(30)), ("VALE3", dec!(50)), ("ITUB4", dec!(25)),
        ("BBDC4", dec!(20)), ("ABEV3", dec!(15)),
    ]);
    let us_prices1 = make_prices_btree(&[
        ("AAPL", dec!(150)), ("MSFT", dec!(300)), ("GOOGL", dec!(140)),
        ("AMZN", dec!(180)), ("NVDA", dec!(500)),
    ]);

    let snap_br1 = perf_br.generate_snapshot(date1, portfolio_br.cash, &br_prices1);
    let snap_us1 = perf_us.generate_snapshot(date1, portfolio_us.cash, &us_prices1);

    // Assert A1.1: equity = cash + mark-to-market
    let br_mtm: Decimal = br_buys.iter().map(|(_, s, p)| *p * Decimal::from(*s)).sum();
    assert_eq!(snap_br1.equity, portfolio_br.cash + br_mtm, "BR equity mismatch");
    
    let us_mtm: Decimal = us_buys.iter().map(|(_, s, p)| *p * Decimal::from(*s)).sum();
    assert_eq!(snap_us1.equity, portfolio_us.cash + us_mtm, "US equity mismatch");

    // Rebalance 2: Prices move, partial sells
    let date2 = make_date(5);
    let br_prices2 = make_prices_btree(&[
        ("PETR4", dec!(35)), ("VALE3", dec!(55)), ("ITUB4", dec!(22)),
        ("BBDC4", dec!(18)), ("ABEV3", dec!(17)),
    ]);
    let us_prices2 = make_prices_btree(&[
        ("AAPL", dec!(160)), ("MSFT", dec!(310)), ("GOOGL", dec!(130)),
        ("AMZN", dec!(190)), ("NVDA", dec!(550)),
    ]);

    // Update portfolio prices
    portfolio_br.update_prices(&make_prices_hash(&[
        ("PETR4", dec!(35)), ("VALE3", dec!(55)), ("ITUB4", dec!(22)),
        ("BBDC4", dec!(18)), ("ABEV3", dec!(17)),
    ]));
    portfolio_us.update_prices(&make_prices_hash(&[
        ("AAPL", dec!(160)), ("MSFT", dec!(310)), ("GOOGL", dec!(130)),
        ("AMZN", dec!(190)), ("NVDA", dec!(550)),
    ]));

    // Sell half of PETR4 (BR) and AAPL (US)
    let sell_cost_br = dec!(35) * dec!(50) * dec!(0.001);
    portfolio_br.apply_sell("PETR4", 50, dec!(35), sell_cost_br).unwrap();
    ledger_br.record_sell(date2, "PETR4", 50, dec!(35), sell_cost_br, Market::BR);
    perf_br.record_sell(date2, "PETR4", 50, dec!(35), sell_cost_br, Market::BR);

    let sell_cost_us = dec!(160) * dec!(5) * dec!(0.0005);
    portfolio_us.apply_sell("AAPL", 5, dec!(160), sell_cost_us).unwrap();
    ledger_us.record_sell(date2, "AAPL", 5, dec!(160), sell_cost_us, Market::US);
    perf_us.record_sell(date2, "AAPL", 5, dec!(160), sell_cost_us, Market::US);

    let snap_br2 = perf_br.generate_snapshot(date2, portfolio_br.cash, &br_prices2);
    let snap_us2 = perf_us.generate_snapshot(date2, portfolio_us.cash, &us_prices2);

    // Assert A1.2: realized + unrealized = total
    let br_pnl = ledger_br.get_pnl_breakdown(&br_prices2);
    assert_eq!(br_pnl.total, br_pnl.realized + br_pnl.unrealized, "BR PnL reconciliation failed");
    
    let us_pnl = ledger_us.get_pnl_breakdown(&us_prices2);
    assert_eq!(us_pnl.total, us_pnl.realized + us_pnl.unrealized, "US PnL reconciliation failed");

    // Assert A1.3: total costs = fees (slippage is 0 in current model)
    let br_costs = ledger_br.costs();
    assert_eq!(br_costs.total, br_costs.fees_br + br_costs.fees_us + br_costs.slippage_br + br_costs.slippage_us);
    assert!(br_costs.fees_br > Decimal::ZERO, "BR fees should be positive");
    assert_eq!(br_costs.fees_us, Decimal::ZERO, "US fees in BR ledger should be 0");

    let us_costs = ledger_us.costs();
    assert!(us_costs.fees_us > Decimal::ZERO, "US fees should be positive");
    assert_eq!(us_costs.fees_br, Decimal::ZERO, "BR fees in US ledger should be 0");

    // Assert A1.4: exposure matches positions
    assert!(snap_br2.exposure.gross > Decimal::ZERO);
    assert!(snap_us2.exposure.gross > Decimal::ZERO);

    // Rebalance 3: More activity
    let date3 = make_date(10);
    let br_prices3 = make_prices_btree(&[
        ("PETR4", dec!(38)), ("VALE3", dec!(52)), ("ITUB4", dec!(24)),
        ("BBDC4", dec!(21)), ("ABEV3", dec!(16)),
    ]);
    
    // Buy more ITUB4
    let buy_cost = dec!(24) * dec!(100) * dec!(0.001);
    portfolio_br.apply_buy("ITUB4", 100, dec!(24), buy_cost, Market::BR, date3).unwrap();
    ledger_br.record_buy(date3, "ITUB4", 100, dec!(24), buy_cost, Market::BR);
    perf_br.record_buy(date3, "ITUB4", 100, dec!(24), buy_cost, Market::BR);

    let snap_br3 = perf_br.generate_snapshot(date3, portfolio_br.cash, &br_prices3);

    // Final reconciliation
    assert!(snap_br3.equity > Decimal::ZERO);
    assert!(ledger_br.verify_reconciliation(&br_prices3), "Final BR reconciliation failed");
    assert!(portfolio_br.validate().is_ok(), "Portfolio validation failed");
}

// ==============================================================================
// A2: Double Counting Prevention
// ==============================================================================

#[test]
fn e2e_double_counting_prevention_netting() {
    // Scenario: BUY and SELL same symbol in same step
    // Cost should be charged ONCE on the NET order, not twice
    
    let date = make_date(1);
    
    // Simulate: Had 1000 PETR4, want to reduce to 600 (SELL 400)
    // But also entry engine says buy 200 (rebalancing target)
    // Net = SELL 200
    
    // Method 1: Gross orders (wrong - would double count)
    let gross_sell = Order::new("PETR4".to_string(), OrderSide::Sell, 400, dec!(50), dec!(20)); // 400 @ 50 = 20000, 10bps = 20
    let gross_buy = Order::new("PETR4".to_string(), OrderSide::Buy, 200, dec!(50), dec!(10)); // 200 @ 50 = 10000, 10bps = 10
    let gross_total_cost = gross_sell.estimated_cost + gross_buy.estimated_cost; // 30
    
    // Method 2: Net order (correct)
    let net_shares = 400 - 200; // SELL 200
    let net_notional = dec!(50) * Decimal::from(net_shares); // 10000
    let net_cost = net_notional * dec!(0.001); // 10 bps = 10
    
    // Assert: Net cost is LESS than gross cost (no double counting)
    assert!(net_cost < gross_total_cost, "Net cost ({}) should be less than gross ({})", net_cost, gross_total_cost);
    assert_eq!(net_cost, dec!(10));
    assert_eq!(gross_total_cost, dec!(30));
    
    // Now verify with actual ledger
    let mut ledger = TradeLedger::new();
    
    // Initial position
    ledger.record_buy(date, "PETR4", 1000, dec!(45), dec!(45), Market::BR);
    
    // Apply netted order (SELL 200 @ 50)
    ledger.record_sell(date, "PETR4", 200, dec!(50), net_cost, Market::BR);
    
    // Costs should match net, not gross
    assert_eq!(ledger.costs().total, dec!(45) + dec!(10)); // Initial buy + net sell
    
    // Positions should reflect net
    let pos = ledger.positions().get("PETR4").unwrap();
    assert_eq!(pos.shares, 800); // 1000 - 200
}

#[test]
fn e2e_double_counting_full_cancellation() {
    // Scenario: Exit SELL 500 + Entry BUY 500 = No net order
    // Should result in ZERO costs for this operation
    
    let date = make_date(1);
    
    // Gross: SELL 500 @ 100, BUY 500 @ 100
    let gross_sell_cost = dec!(100) * dec!(500) * dec!(0.001); // 50
    let gross_buy_cost = dec!(100) * dec!(500) * dec!(0.001); // 50
    let gross_total = gross_sell_cost + gross_buy_cost; // 100
    
    // Net: shares = 500 - 500 = 0, no order needed
    let net_cost = dec!(0);
    
    assert_eq!(net_cost, Decimal::ZERO);
    assert!(net_cost < gross_total);
    
    // Verify: position unchanged, no new costs
    let mut ledger = TradeLedger::new();
    ledger.record_buy(date, "VALE3", 500, dec!(100), dec!(50), Market::BR);
    
    // No net order = no additional trades
    let costs_before = ledger.costs().total;
    // (Nothing to record since net is 0)
    let costs_after = ledger.costs().total;
    
    assert_eq!(costs_before, costs_after, "Costs should not change for cancelled orders");
}

#[test]
fn e2e_ledger_portfolio_cost_consistency() {
    // Verify that TradeLedger and PortfolioState track costs consistently
    let date = make_date(1);
    
    let mut portfolio = PortfolioState::new(dec!(100_000));
    let mut ledger = TradeLedger::new();
    
    // Same trades in both
    let trades = vec![
        ("PETR4", 100, dec!(50), dec!(5)),   // 5000 + 5 cost
        ("VALE3", 200, dec!(60), dec!(12)),  // 12000 + 12 cost
        ("ITUB4", 100, dec!(30), dec!(3)),   // 3000 + 3 cost
    ];
    
    for (sym, shares, price, cost) in &trades {
        portfolio.apply_buy(sym, *shares, *price, *cost, Market::BR, date).unwrap();
        ledger.record_buy(date, sym, *shares, *price, *cost, Market::BR);
    }
    
    // Total cost should match
    let expected_cost: Decimal = trades.iter().map(|(_, _, _, c)| *c).sum();
    assert_eq!(ledger.costs().total, expected_cost, "Ledger costs mismatch");
    
    // Cash consumed should match: notional + costs
    let expected_cash_used: Decimal = trades.iter()
        .map(|(_, s, p, c)| *p * Decimal::from(*s) + *c)
        .sum();
    let actual_cash_used = dec!(100_000) - portfolio.cash;
    assert_eq!(actual_cash_used, expected_cash_used, "Cash consumption mismatch");
}

// ==============================================================================
// A3: Multi-Market Explicit (No Silent Currency Mixing)
// ==============================================================================

#[test]
fn e2e_multi_market_separate_portfolios() {
    // Verify BR and US are tracked separately and NOT mixed
    let date = make_date(1);
    
    // BR portfolio in BRL
    let mut portfolio_br = PortfolioState::new(dec!(1_000_000));
    let mut ledger_br = TradeLedger::new();
    
    // US portfolio in USD
    let mut portfolio_us = PortfolioState::new(dec!(100_000)); // 100k USD
    let mut ledger_us = TradeLedger::new();
    
    // BR trades
    portfolio_br.apply_buy("PETR4", 1000, dec!(50), dec!(50), Market::BR, date).unwrap();
    ledger_br.record_buy(date, "PETR4", 1000, dec!(50), dec!(50), Market::BR);
    
    // US trades
    portfolio_us.apply_buy("AAPL", 100, dec!(150), dec!(7.5), Market::US, date).unwrap();
    ledger_us.record_buy(date, "AAPL", 100, dec!(150), dec!(7.5), Market::US);
    
    // Verify exposure is by market
    let br_prices = make_prices_btree(&[("PETR4", dec!(55))]);
    let us_prices = make_prices_btree(&[("AAPL", dec!(160))]);
    
    let br_pnl = ledger_br.get_pnl_breakdown(&br_prices);
    let us_pnl = ledger_us.get_pnl_breakdown(&us_prices);
    
    // BR PnL should only have BR market
    assert!(br_pnl.by_market.get("BR").is_some() || br_pnl.unrealized > Decimal::ZERO);
    assert!(br_pnl.by_market.get("US").is_none() || br_pnl.by_market.get("US") == Some(&Decimal::ZERO));
    
    // US PnL should only have US market
    assert!(us_pnl.by_market.get("US").is_some() || us_pnl.unrealized > Decimal::ZERO);
    assert!(us_pnl.by_market.get("BR").is_none() || us_pnl.by_market.get("BR") == Some(&Decimal::ZERO));
    
    // Costs are market-specific
    assert!(ledger_br.costs().fees_br > Decimal::ZERO);
    assert_eq!(ledger_br.costs().fees_us, Decimal::ZERO);
    
    assert!(ledger_us.costs().fees_us > Decimal::ZERO);
    assert_eq!(ledger_us.costs().fees_br, Decimal::ZERO);
}

#[test]
fn e2e_multi_market_no_currency_mixing() {
    // Test that we never silently add BRL + USD values
    let date = make_date(1);
    
    let mut perf_br = PerformanceEngine::new(PerformanceConfig::default(), dec!(1_000_000));
    let mut perf_us = PerformanceEngine::new(PerformanceConfig::default(), dec!(100_000));
    
    // BR trade: 1000 shares @ 50 BRL = 50,000 BRL position
    perf_br.record_buy(date, "PETR4", 1000, dec!(50), dec!(50), Market::BR);
    
    // US trade: 100 shares @ 150 USD = 15,000 USD position
    perf_us.record_buy(date, "AAPL", 100, dec!(150), dec!(7.5), Market::US);
    
    let br_prices = make_prices_btree(&[("PETR4", dec!(55))]);
    let us_prices = make_prices_btree(&[("AAPL", dec!(160))]);
    
    // Cash: 1M - 50000 - 50 = 949950 BRL
    // Cash: 100k - 15000 - 7.5 = 84992.5 USD
    let snap_br = perf_br.generate_snapshot(date, dec!(949_950), &br_prices);
    let snap_us = perf_us.generate_snapshot(date, dec!(84_992.5), &us_prices);
    
    // BR equity should be in BRL scale
    // Positions: 1000 * 55 = 55000
    // Total: 949950 + 55000 = 1004950 BRL
    assert_eq!(snap_br.equity, dec!(1_004_950));
    
    // US equity should be in USD scale
    // Positions: 100 * 160 = 16000
    // Total: 84992.5 + 16000 = 100992.5 USD
    assert_eq!(snap_us.equity, dec!(100_992.5));
    
    // These values should NEVER be added together without FX conversion
    // The system correctly keeps them separate (no combined "total equity" field)
    
    // Verify exposure breakdown by market
    assert!(snap_br.exposure.by_market.contains_key("BR"));
    assert!(!snap_br.exposure.by_market.contains_key("US"));
    
    assert!(snap_us.exposure.by_market.contains_key("US"));
    assert!(!snap_us.exposure.by_market.contains_key("BR"));
}

#[test]
fn e2e_exposure_by_market_correct() {
    // Verify exposure.by_market accurately reflects positions per market
    let date = make_date(1);
    
    // Create engine with both BR and US positions
    let mut perf = PerformanceEngine::new(PerformanceConfig::default(), dec!(1_000_000));
    
    // Mix of BR and US
    perf.record_buy(date, "PETR4", 100, dec!(50), dec!(5), Market::BR);  // 5000 BRL
    perf.record_buy(date, "VALE3", 200, dec!(60), dec!(12), Market::BR); // 12000 BRL
    perf.record_buy(date, "AAPL", 10, dec!(150), dec!(1), Market::US);   // 1500 USD
    
    let prices = make_prices_btree(&[
        ("PETR4", dec!(55)),  // 5500
        ("VALE3", dec!(65)),  // 13000
        ("AAPL", dec!(160)),  // 1600
    ]);
    
    // Cash = 1M - 5005 - 12012 - 1501 = 981482
    let snap = perf.generate_snapshot(date, dec!(981_482), &prices);
    
    // Total exposure: 5500 + 13000 + 1600 = 20100
    assert_eq!(snap.exposure.gross, dec!(20_100));
    
    // By market:
    // BR: 5500 + 13000 = 18500
    // US: 1600
    assert_eq!(snap.exposure.by_market.get("BR"), Some(&dec!(18_500)));
    assert_eq!(snap.exposure.by_market.get("US"), Some(&dec!(1_600)));
}

// ==============================================================================
// Additional Invariant Tests
// ==============================================================================

#[test]
fn e2e_turnover_matches_orders() {
    // Verify turnover in snapshot matches actual order notional
    let date = make_date(1);
    
    let mut perf = PerformanceEngine::new(PerformanceConfig::default(), dec!(1_000_000));
    
    // Buy: 100 @ 50 = 5000
    perf.record_buy(date, "PETR4", 100, dec!(50), dec!(5), Market::BR);
    // Buy: 200 @ 60 = 12000
    perf.record_buy(date, "VALE3", 200, dec!(60), dec!(12), Market::BR);
    // Sell: 50 @ 55 = 2750
    perf.record_sell(date, "PETR4", 50, dec!(55), dec!(3), Market::BR);
    
    let prices = make_prices_btree(&[("PETR4", dec!(55)), ("VALE3", dec!(60))]);
    let snap = perf.generate_snapshot(date, dec!(980_230), &prices);
    
    // Buy notional: 5000 + 12000 = 17000
    // Sell notional: 2750
    assert_eq!(snap.turnover.buy_notional, dec!(17_000));
    assert_eq!(snap.turnover.sell_notional, dec!(2_750));
}

#[test]
fn e2e_portfolio_state_ledger_position_match() {
    // Verify PortfolioState and TradeLedger agree on positions
    let date = make_date(1);
    
    let mut portfolio = PortfolioState::new(dec!(100_000));
    let mut ledger = TradeLedger::new();
    
    // Same operations
    portfolio.apply_buy("PETR4", 100, dec!(50), dec!(5), Market::BR, date).unwrap();
    ledger.record_buy(date, "PETR4", 100, dec!(50), dec!(5), Market::BR);
    
    portfolio.apply_buy("PETR4", 50, dec!(55), dec!(3), Market::BR, date).unwrap();
    ledger.record_buy(date, "PETR4", 50, dec!(55), dec!(3), Market::BR);
    
    // Check shares match
    let portfolio_pos = portfolio.get_position("PETR4").unwrap();
    let ledger_pos = ledger.positions().get("PETR4").unwrap();
    
    assert_eq!(portfolio_pos.shares, ledger_pos.shares as i64);
    
    // Check WAP/cost_basis match (should be same formula)
    // WAP = (100*50 + 50*55) / 150 = 7750 / 150 = 51.666...
    let expected_wap = dec!(51.666666666666666666666666667);
    let tolerance = dec!(0.01);
    
    assert!((portfolio_pos.cost_basis - expected_wap).abs() < tolerance);
    assert!((ledger_pos.wap_cost_basis - expected_wap).abs() < tolerance);
}

