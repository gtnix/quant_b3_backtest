//! Property-based tests for FX module using proptest.
//!
//! These tests verify key invariants hold across a wide range of inputs:
//! - Roundtrip conversion identity (A→B→A ≈ A within tolerance)
//! - Same currency identity (always rate=1, no provider needed)
//! - Decomposition identity (R_asset + R_fx + R_interaction = R_total)

use backtester_intelligence::currency::{Currency, Money, FxPair, FxRate};
use backtester_intelligence::fx::{InMemoryFxProvider, FxRateProvider, convert_money, convert_money_with_audit};
use backtester_intelligence::performance::FxResolutionMethod;
use chrono::NaiveDate;
use proptest::prelude::*;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

// =============================================================================
// STRATEGIES
// =============================================================================

/// Generate realistic FX rates for USD/BRL (typically 4.0 to 7.0)
fn usd_brl_rate_strategy() -> impl Strategy<Value = Decimal> {
    (4000u64..7000u64).prop_map(|n| Decimal::new(n as i64, 3))
}

/// Generate realistic FX rates for EUR/USD (typically 1.0 to 1.2)
fn eur_usd_rate_strategy() -> impl Strategy<Value = Decimal> {
    (1000u64..1200u64).prop_map(|n| Decimal::new(n as i64, 3))
}

/// Generate positive monetary amounts (0.01 to 10,000,000)
fn money_amount_strategy() -> impl Strategy<Value = Decimal> {
    (1u64..10_000_000_000u64).prop_map(|n| Decimal::new(n as i64, 2))
}

/// Generate a currency
fn currency_strategy() -> impl Strategy<Value = Currency> {
    prop_oneof![
        Just(Currency::USD),
        Just(Currency::BRL),
        Just(Currency::EUR),
    ]
}

/// Generate a date in 2024
fn date_strategy() -> impl Strategy<Value = NaiveDate> {
    (1u32..366u32).prop_map(|day_of_year| {
        NaiveDate::from_yo_opt(2024, day_of_year).unwrap()
    })
}

// =============================================================================
// PROPERTY: SAME CURRENCY IDENTITY
// =============================================================================

proptest! {
    /// Converting same currency always returns identical value with rate=1.
    /// This must work without any provider data.
    #[test]
    fn prop_same_currency_identity(
        amount in money_amount_strategy(),
        currency in currency_strategy(),
        date in date_strategy(),
    ) {
        let provider = InMemoryFxProvider::new();
        let money = Money::new(amount, currency);
        
        // Same currency conversion should always succeed
        let result = convert_money_with_audit(&money, currency, date, &provider, 0);
        
        prop_assert!(result.is_ok(), "Same currency conversion should always succeed");
        
        let conversion = result.unwrap();
        prop_assert_eq!(conversion.money.amount(), amount, "Amount should be unchanged");
        prop_assert_eq!(conversion.rate_used, Decimal::ONE, "Rate should be 1.0");
        prop_assert_eq!(conversion.method, FxResolutionMethod::Identity, "Method should be Identity");
    }
}

// =============================================================================
// PROPERTY: ROUNDTRIP CONVERSION IDENTITY
// =============================================================================

proptest! {
    /// Converting A→B→A should return approximately the original amount.
    /// Due to division precision, we allow a small tolerance.
    #[test]
    fn prop_roundtrip_usd_brl_identity(
        amount in money_amount_strategy(),
        rate in usd_brl_rate_strategy(),
        date in date_strategy(),
    ) {
        let mut provider = InMemoryFxProvider::new();
        provider.add_rate(FxPair::USD_BRL, date, rate);
        
        let usd = Money::new(amount, Currency::USD);
        
        // USD → BRL
        let brl = convert_money(&usd, Currency::BRL, date, &provider, 0)?;
        
        // BRL → USD
        let usd_back = convert_money(&brl, Currency::USD, date, &provider, 0)?;
        
        // Check roundtrip is approximately equal
        // Tolerance: 1e-10 (due to division precision)
        let diff = (usd_back.amount() - amount).abs();
        let tolerance = Decimal::new(1, 10);
        
        prop_assert!(
            diff < tolerance,
            "Roundtrip conversion should be approximately identity. \
            Original: {}, After roundtrip: {}, Diff: {}",
            amount, usd_back.amount(), diff
        );
    }

    /// Converting A→B→A should return approximately the original amount (EUR/USD pair).
    #[test]
    fn prop_roundtrip_eur_usd_identity(
        amount in money_amount_strategy(),
        rate in eur_usd_rate_strategy(),
        date in date_strategy(),
    ) {
        let mut provider = InMemoryFxProvider::new();
        provider.add_rate(FxPair::EUR_USD, date, rate);
        
        let eur = Money::new(amount, Currency::EUR);
        
        // EUR → USD
        let usd = convert_money(&eur, Currency::USD, date, &provider, 0)?;
        
        // USD → EUR
        let eur_back = convert_money(&usd, Currency::EUR, date, &provider, 0)?;
        
        // Check roundtrip is approximately equal
        let diff = (eur_back.amount() - amount).abs();
        let tolerance = Decimal::new(1, 10);
        
        prop_assert!(
            diff < tolerance,
            "Roundtrip conversion should be approximately identity. \
            Original: {}, After roundtrip: {}, Diff: {}",
            amount, eur_back.amount(), diff
        );
    }
}

// =============================================================================
// PROPERTY: DECOMPOSITION IDENTITY
// =============================================================================

proptest! {
    /// The 3-term decomposition must exactly equal total return:
    /// R_asset + R_fx + R_interaction = R_total
    ///
    /// This uses exact Decimal arithmetic, so no tolerance needed.
    #[test]
    fn prop_decomposition_exact(
        asset_return_pct in -50i32..200i32,  // -50% to +200%
        fx_return_pct in -30i32..50i32,       // -30% to +50%
    ) {
        // Convert percentages to decimals
        let r_asset = Decimal::new(asset_return_pct as i64, 2);
        let r_fx = Decimal::new(fx_return_pct as i64, 2);
        
        // Calculate interaction term
        let r_interaction = r_asset * r_fx;
        
        // Additive decomposition
        let r_total_additive = r_asset + r_fx + r_interaction;
        
        // Multiplicative decomposition (for verification)
        let multiplicative = (Decimal::ONE + r_asset) * (Decimal::ONE + r_fx);
        let r_total_multiplicative = multiplicative - Decimal::ONE;
        
        // Both methods should give exactly the same result
        prop_assert_eq!(
            r_total_additive, r_total_multiplicative,
            "Additive and multiplicative decomposition should match exactly. \
            R_asset: {}, R_fx: {}, R_interaction: {}, \
            Additive total: {}, Multiplicative total: {}",
            r_asset, r_fx, r_interaction,
            r_total_additive, r_total_multiplicative
        );
    }

    /// Verify decomposition with actual FX rate changes
    #[test]
    fn prop_decomposition_with_fx_rates(
        start_value in 1000u64..1000000u64,
        asset_change_pct in -30i32..100i32,  // -30% to +100%
        fx_start in (4000u64..6000u64),
        fx_change_pct in -20i32..30i32,
    ) {
        let start_date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let end_date = NaiveDate::from_ymd_opt(2024, 1, 31).unwrap();
        
        // Set up values
        let v_start = Decimal::new(start_value as i64, 0);
        let r_asset = Decimal::new(asset_change_pct as i64, 2);
        let v_end = v_start * (Decimal::ONE + r_asset);
        
        // Set up FX rates
        let fx_start_rate = Decimal::new(fx_start as i64, 3);
        let fx_change = Decimal::new(fx_change_pct as i64, 2);
        let fx_end_rate = fx_start_rate * (Decimal::ONE + fx_change);
        
        // Calculate returns
        let r_fx = fx_end_rate / fx_start_rate - Decimal::ONE;
        let r_interaction = r_asset * r_fx;
        
        // Calculate actual total return in base currency
        let value_base_start = v_start * fx_start_rate;
        let value_base_end = v_end * fx_end_rate;
        let r_total_actual = if value_base_start.is_zero() {
            Decimal::ZERO
        } else {
            value_base_end / value_base_start - Decimal::ONE
        };
        
        // Calculate from decomposition
        let r_total_decomposition = r_asset + r_fx + r_interaction;
        
        // Should be very close (within rounding tolerance)
        let diff = (r_total_actual - r_total_decomposition).abs();
        let tolerance = Decimal::new(1, 12);  // 1e-12
        
        prop_assert!(
            diff < tolerance,
            "Decomposition should match actual total return. \
            R_asset: {}, R_fx: {}, R_interaction: {}, \
            Decomposition total: {}, Actual total: {}, Diff: {}",
            r_asset, r_fx, r_interaction,
            r_total_decomposition, r_total_actual, diff
        );
    }
}

// =============================================================================
// PROPERTY: RATE CONSISTENCY
// =============================================================================

proptest! {
    /// FxRate semantic accessors should be consistent
    #[test]
    fn prop_rate_accessors_consistent(
        rate_val in (1000u64..10000u64),
        date in date_strategy(),
    ) {
        let rate = Decimal::new(rate_val as i64, 3);
        let fx_rate = FxRate::new(FxPair::USD_BRL, rate, date);
        
        // rate_quote_per_base should equal rate
        prop_assert_eq!(
            fx_rate.rate_quote_per_base(), 
            fx_rate.rate,
            "rate_quote_per_base should equal rate field"
        );
        
        // describe should contain correct values
        let description = fx_rate.describe();
        prop_assert!(description.contains("1 USD"), "Description should show 1 USD");
        prop_assert!(description.contains("BRL"), "Description should show BRL");
    }

    /// FxRate inverse should satisfy: rate * inverse_rate ≈ 1
    #[test]
    fn prop_rate_inverse_identity(
        rate_val in (1000u64..10000u64),
        date in date_strategy(),
    ) {
        let rate = Decimal::new(rate_val as i64, 3);
        let fx_rate = FxRate::new(FxPair::USD_BRL, rate, date);
        let inverse = fx_rate.inverse();
        
        // rate * inverse_rate should equal 1
        let product = fx_rate.rate * inverse.rate;
        let diff = (product - Decimal::ONE).abs();
        let tolerance = Decimal::new(1, 15);
        
        prop_assert!(
            diff < tolerance,
            "Rate * inverse rate should equal 1. Rate: {}, Inverse: {}, Product: {}",
            fx_rate.rate, inverse.rate, product
        );
    }
}

// =============================================================================
// PROPERTY: CONVERSION AUDIT TRAIL
// =============================================================================

proptest! {
    /// Conversion audit trail should be consistent
    #[test]
    fn prop_conversion_audit_consistent(
        amount in money_amount_strategy(),
        rate in usd_brl_rate_strategy(),
        date in date_strategy(),
        gap_days in 0u32..5u32,
    ) {
        let rate_date = date.pred_opt().unwrap_or(date);  // Rate is from day before
        
        let mut provider = InMemoryFxProvider::new();
        provider.add_rate(FxPair::USD_BRL, rate_date, rate);
        
        let usd = Money::new(amount, Currency::USD);
        let result = convert_money_with_audit(&usd, Currency::BRL, date, &provider, gap_days + 1);
        
        if let Ok(conversion) = result {
            // Date resolved should be the actual rate date
            prop_assert_eq!(
                conversion.date_resolved.to_string(), 
                rate_date.to_string(),
                "Date resolved should be the actual rate date"
            );
            
            // If dates differ, method should indicate LOCF
            if conversion.date_requested != conversion.date_resolved {
                prop_assert!(
                    conversion.method.used_locf(),
                    "Method should indicate LOCF when dates differ. Method: {:?}",
                    conversion.method
                );
            }
            
            // Rate used should match the stored rate
            prop_assert_eq!(
                conversion.rate_used, rate,
                "Rate used should match stored rate"
            );
        }
    }
}
























