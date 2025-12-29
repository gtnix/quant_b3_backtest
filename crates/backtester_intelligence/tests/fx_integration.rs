//! Integration tests for FX module.
//!
//! Tests cover:
//! - Snapshot generation with BR+US positions converted to base currency
//! - FX attribution 3-term decomposition verification
//! - Exposure by_currency aggregation

use backtester_intelligence::currency::{Currency, FxPair};
use backtester_intelligence::fx::{InMemoryFxProvider, FxRateProvider};
use backtester_intelligence::filters::Market;
use backtester_intelligence::performance::{
    PerformanceEngine, PerformanceConfig,
    FxAttributionEngine,
};
use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::BTreeMap;
use std::sync::Arc;

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

fn make_provider() -> InMemoryFxProvider {
    let mut provider = InMemoryFxProvider::new();
    
    // USD/BRL rates
    provider.add_rate(FxPair::USD_BRL, date(2024, 1, 1), dec!(5.00));
    provider.add_rate(FxPair::USD_BRL, date(2024, 1, 15), dec!(5.25));
    provider.add_rate(FxPair::USD_BRL, date(2024, 1, 31), dec!(5.50));
    
    provider
}

// =============================================================================
// SNAPSHOT TESTS
// =============================================================================

#[test]
fn test_snapshot_br_us_positions_converted_to_brl_base() {
    let provider = Arc::new(make_provider());
    let config = PerformanceConfig::default()
        .with_base_currency(Currency::BRL);
    
    let mut engine = PerformanceEngine::with_fx(config, dec!(100000), provider);
    
    // Buy BR position (in BRL)
    engine.record_buy(date(2024, 1, 1), "PETR4", 100, dec!(30), dec!(10), Market::BR);
    
    // Buy US position (in USD)
    engine.record_buy(date(2024, 1, 1), "AAPL", 10, dec!(150), dec!(5), Market::US);
    
    // Generate snapshot
    let prices: BTreeMap<String, Decimal> = [
        ("PETR4".to_string(), dec!(32)),  // BRL
        ("AAPL".to_string(), dec!(160)),  // USD
    ].into();
    
    let snapshot = engine.generate_snapshot(date(2024, 1, 15), dec!(90000), &prices);
    
    // Verify base currency is set
    assert_eq!(snapshot.base_currency, Some(Currency::BRL));
    
    // Verify equity_base is calculated
    // BR: 100 * 32 = 3200 BRL
    // US: 10 * 160 = 1600 USD = 1600 * 5.25 = 8400 BRL
    // Cash: 90000 BRL
    // Total: 90000 + 3200 + 8400 = 101600 BRL
    assert!(snapshot.equity_base.is_some());
    let equity_base = snapshot.equity_base.unwrap();
    assert_eq!(equity_base, dec!(101600));
    
    // Verify exposure by_currency
    assert!(snapshot.exposure.by_currency.contains_key("BRL"));
    assert!(snapshot.exposure.by_currency.contains_key("USD"));
}

#[test]
fn test_snapshot_br_us_positions_converted_to_usd_base() {
    let mut provider = InMemoryFxProvider::new();
    // BRL/USD rate (inverse of USD/BRL)
    // If USD/BRL = 5.00, then BRL/USD = 0.20
    provider.add_rate(FxPair::USD_BRL, date(2024, 1, 15), dec!(5.00));
    
    let provider = Arc::new(provider);
    let config = PerformanceConfig::default()
        .with_base_currency(Currency::USD);
    
    let mut engine = PerformanceEngine::with_fx(config, dec!(20000), provider);
    
    // Buy BR position (in BRL)
    engine.record_buy(date(2024, 1, 15), "PETR4", 100, dec!(30), dec!(10), Market::BR);
    
    // Buy US position (in USD) 
    engine.record_buy(date(2024, 1, 15), "AAPL", 10, dec!(150), dec!(5), Market::US);
    
    let prices: BTreeMap<String, Decimal> = [
        ("PETR4".to_string(), dec!(30)),
        ("AAPL".to_string(), dec!(150)),
    ].into();
    
    let snapshot = engine.generate_snapshot(date(2024, 1, 15), dec!(18000), &prices);
    
    assert_eq!(snapshot.base_currency, Some(Currency::USD));
    assert!(snapshot.equity_base.is_some());
    
    // BR: 100 * 30 = 3000 BRL = 3000 / 5.00 = 600 USD
    // US: 10 * 150 = 1500 USD
    // Cash: 18000 (assume USD) = 18000 USD
    // Total: 18000 + 600 + 1500 = 20100 USD
    // Note: cash is treated as base currency by default
    let equity_base = snapshot.equity_base.unwrap();
    assert_eq!(equity_base, dec!(20100));
}

// =============================================================================
// FX ATTRIBUTION TESTS
// =============================================================================

#[test]
fn test_fx_attribution_3_terms_sum_to_total() {
    let provider = make_provider();
    let engine = FxAttributionEngine::new(Currency::BRL);
    
    // Start: $1000 USD @ 5.00 = R$5000
    // End: $1100 USD @ 5.50 = R$6050
    // 
    // R_asset = 1100/1000 - 1 = 10%
    // R_fx = 5.50/5.00 - 1 = 10%
    // R_interaction = 10% * 10% = 1%
    // R_total = 10% + 10% + 1% = 21%
    
    let attr = engine.calculate_currency_attribution(
        Currency::USD,
        dec!(1000),  // start
        dec!(1100),  // end
        dec!(100),   // weight
        &provider,
        date(2024, 1, 1),
        date(2024, 1, 31),
    ).unwrap();
    
    // Verify 3 terms
    assert_eq!(attr.asset_return, dec!(0.1));    // 10%
    assert_eq!(attr.fx_return, dec!(0.1));       // 10%
    assert_eq!(attr.interaction, dec!(0.01));    // 1%
    assert_eq!(attr.total_return_base, dec!(0.21)); // 21%
    
    // Verify decomposition: asset + fx + interaction = total
    let sum = attr.asset_return + attr.fx_return + attr.interaction;
    assert_eq!(sum, attr.total_return_base);
    
    // Verify multiplicative: (1 + R_total) = (1 + R_asset) * (1 + R_fx)
    let multiplicative = (Decimal::ONE + attr.asset_return) * (Decimal::ONE + attr.fx_return);
    assert_eq!(multiplicative, Decimal::ONE + attr.total_return_base);
}

#[test]
fn test_fx_attribution_same_currency_no_fx() {
    let provider = make_provider();
    let engine = FxAttributionEngine::new(Currency::BRL);
    
    // BRL position with BRL base - no FX effect
    let attr = engine.calculate_currency_attribution(
        Currency::BRL,
        dec!(10000),  // start
        dec!(11000),  // end (10% gain)
        dec!(100),
        &provider,
        date(2024, 1, 1),
        date(2024, 1, 31),
    ).unwrap();
    
    assert_eq!(attr.asset_return, dec!(0.1));
    assert_eq!(attr.fx_return, Decimal::ZERO);
    assert_eq!(attr.interaction, Decimal::ZERO);
    assert_eq!(attr.total_return_base, dec!(0.1));
}

#[test]
fn test_fx_attribution_negative_fx() {
    let mut provider = InMemoryFxProvider::new();
    // USD weakens: 5.00 -> 4.50 (10% loss for USD holders in BRL)
    provider.add_rate(FxPair::USD_BRL, date(2024, 1, 1), dec!(5.00));
    provider.add_rate(FxPair::USD_BRL, date(2024, 1, 31), dec!(4.50));
    
    let engine = FxAttributionEngine::new(Currency::BRL);
    
    // 10% asset gain, 10% FX loss
    let attr = engine.calculate_currency_attribution(
        Currency::USD,
        dec!(1000),
        dec!(1100),  // 10% gain in USD
        dec!(100),
        &provider,
        date(2024, 1, 1),
        date(2024, 1, 31),
    ).unwrap();
    
    assert_eq!(attr.asset_return, dec!(0.1));    // +10%
    assert_eq!(attr.fx_return, dec!(-0.1));      // -10%
    assert_eq!(attr.interaction, dec!(-0.01));   // -1%
    assert_eq!(attr.total_return_base, dec!(-0.01)); // Net -1%
    
    // Verify: start $1000 @ 5.00 = R$5000
    //         end   $1100 @ 4.50 = R$4950
    //         return = 4950/5000 - 1 = -1%
}

#[test]
fn test_portfolio_fx_attribution() {
    let provider = make_provider();
    let engine = FxAttributionEngine::new(Currency::BRL);
    
    // Portfolio: 50% BRL, 50% USD (in base currency)
    let mut values_start = BTreeMap::new();
    values_start.insert(Currency::BRL, dec!(5000));  // R$5000
    values_start.insert(Currency::USD, dec!(1000));  // $1000 @ 5.00 = R$5000
    
    let mut values_end = BTreeMap::new();
    values_end.insert(Currency::BRL, dec!(5500));   // R$5500 (+10%)
    values_end.insert(Currency::USD, dec!(1100));   // $1100 @ 5.50 = R$6050
    
    let breakdown = engine.calculate_period_attribution(
        &values_start,
        &values_end,
        &provider,
        date(2024, 1, 1),
        date(2024, 1, 31),
    ).unwrap();
    
    // Verify 2 currency attributions
    assert_eq!(breakdown.by_currency.len(), 2);
    
    // Verify decomposition holds
    assert!(breakdown.verify_decomposition());
    
    // BRL: 10% asset, 0% fx
    // USD: 10% asset, 10% fx, 1% interaction = 21% total
    // Portfolio (50/50 weights): 
    //   asset = 0.5 * 10% + 0.5 * 10% = 10%
    //   fx = 0.5 * 0% + 0.5 * 10% = 5%
    //   interaction = 0.5 * 0% + 0.5 * 1% = 0.5%
    //   total = 10% + 5% + 0.5% = 15.5%
}

// =============================================================================
// EXPOSURE BY CURRENCY TESTS
// =============================================================================

#[test]
fn test_exposure_by_currency_sums_correctly() {
    let provider = Arc::new(make_provider());
    let config = PerformanceConfig::default()
        .with_base_currency(Currency::BRL);
    
    let mut engine = PerformanceEngine::with_fx(config, dec!(100000), provider);
    
    // Multiple BR positions
    engine.record_buy(date(2024, 1, 1), "PETR4", 100, dec!(30), dec!(10), Market::BR);
    engine.record_buy(date(2024, 1, 1), "VALE3", 50, dec!(60), dec!(10), Market::BR);
    
    // Multiple US positions
    engine.record_buy(date(2024, 1, 1), "AAPL", 10, dec!(150), dec!(5), Market::US);
    engine.record_buy(date(2024, 1, 1), "MSFT", 5, dec!(300), dec!(5), Market::US);
    
    let prices: BTreeMap<String, Decimal> = [
        ("PETR4".to_string(), dec!(30)),
        ("VALE3".to_string(), dec!(60)),
        ("AAPL".to_string(), dec!(150)),
        ("MSFT".to_string(), dec!(300)),
    ].into();
    
    let snapshot = engine.generate_snapshot(date(2024, 1, 15), dec!(90000), &prices);
    
    // Verify by_currency sums
    // BRL: 100*30 + 50*60 = 3000 + 3000 = 6000
    // USD: 10*150 + 5*300 = 1500 + 1500 = 3000
    assert_eq!(snapshot.exposure.by_currency.get("BRL"), Some(&dec!(6000)));
    assert_eq!(snapshot.exposure.by_currency.get("USD"), Some(&dec!(3000)));
    
    // Verify by_currency_base (USD converted at 5.25)
    // BRL: 6000 BRL
    // USD: 3000 USD * 5.25 = 15750 BRL
    assert!(snapshot.exposure.by_currency_base.contains_key("BRL"));
    assert!(snapshot.exposure.by_currency_base.contains_key("USD"));
    
    let usd_base = snapshot.exposure.by_currency_base.get("USD").unwrap();
    assert_eq!(*usd_base, dec!(15750));
}





