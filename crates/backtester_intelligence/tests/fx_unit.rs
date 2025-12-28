//! Unit tests for FX types and conversion.
//!
//! Tests cover:
//! - Currency enum operations
//! - Money arithmetic (same currency allowed, different rejected)
//! - FxPair parsing and inversion
//! - FxRate conversion
//! - LOCF (Last Observation Carried Forward) behavior
//! - Gap limit enforcement

use backtester_intelligence::currency::{Currency, Money, FxPair, FxRate};
use backtester_intelligence::fx::{InMemoryFxProvider, FxRateProvider, FxError, convert_money};
use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

// =============================================================================
// CURRENCY TESTS
// =============================================================================

#[test]
fn test_currency_exhaustive() {
    // Ensure all variants have proper codes
    assert_eq!(Currency::BRL.code(), "BRL");
    assert_eq!(Currency::USD.code(), "USD");
    assert_eq!(Currency::EUR.code(), "EUR");
    
    // Symbols
    assert_eq!(Currency::BRL.symbol(), "R$");
    assert_eq!(Currency::USD.symbol(), "$");
    assert_eq!(Currency::EUR.symbol(), "€");
}

#[test]
fn test_currency_from_str() {
    assert_eq!(Currency::from_str("BRL"), Some(Currency::BRL));
    assert_eq!(Currency::from_str("usd"), Some(Currency::USD));
    assert_eq!(Currency::from_str("EUR"), Some(Currency::EUR));
    assert_eq!(Currency::from_str("GBP"), None);
    assert_eq!(Currency::from_str(""), None);
}

// =============================================================================
// MONEY TESTS
// =============================================================================

#[test]
fn test_money_same_currency_add() {
    let a = Money::new(dec!(100), Currency::USD);
    let b = Money::new(dec!(50.50), Currency::USD);
    
    let result = a.try_add(&b).unwrap();
    assert_eq!(result.amount(), dec!(150.50));
    assert_eq!(result.currency(), Currency::USD);
}

#[test]
fn test_money_same_currency_sub() {
    let a = Money::new(dec!(100), Currency::BRL);
    let b = Money::new(dec!(30.25), Currency::BRL);
    
    let result = a.try_sub(&b).unwrap();
    assert_eq!(result.amount(), dec!(69.75));
}

#[test]
fn test_money_different_currency_rejected() {
    let usd = Money::new(dec!(100), Currency::USD);
    let brl = Money::new(dec!(500), Currency::BRL);
    
    let result = usd.try_add(&brl);
    assert!(result.is_err());
    
    let err = result.unwrap_err();
    assert_eq!(err.expected, Currency::USD);
    assert_eq!(err.actual, Currency::BRL);
}

#[test]
fn test_money_scale() {
    let m = Money::new(dec!(100), Currency::EUR);
    let scaled = m.scale(dec!(2.5));
    assert_eq!(scaled.amount(), dec!(250));
}

#[test]
fn test_money_round() {
    let m = Money::new(dec!(123.456789), Currency::USD);
    let rounded = m.round();
    assert_eq!(rounded.amount(), dec!(123.46));
}

// =============================================================================
// FXPAIR TESTS
// =============================================================================

#[test]
fn test_fx_pair_inverse() {
    let pair = FxPair::USD_BRL;
    let inverse = pair.inverse();
    
    assert_eq!(inverse.base, Currency::BRL);
    assert_eq!(inverse.quote, Currency::USD);
}

#[test]
fn test_fx_pair_from_str_slash() {
    let pair = FxPair::from_str("USD/BRL").unwrap();
    assert_eq!(pair.base, Currency::USD);
    assert_eq!(pair.quote, Currency::BRL);
}

#[test]
fn test_fx_pair_from_str_concat() {
    let pair = FxPair::from_str("EURBRL").unwrap();
    assert_eq!(pair.base, Currency::EUR);
    assert_eq!(pair.quote, Currency::BRL);
}

#[test]
fn test_fx_pair_for_conversion() {
    // Same currency
    assert!(FxPair::for_conversion(Currency::USD, Currency::USD).is_none());
    
    // Different currencies
    let pair = FxPair::for_conversion(Currency::EUR, Currency::BRL).unwrap();
    assert_eq!(pair.base, Currency::EUR);
    assert_eq!(pair.quote, Currency::BRL);
}

// =============================================================================
// FXRATE TESTS
// =============================================================================

#[test]
fn test_fx_rate_convert_to_quote() {
    // USD/BRL = 5.00 means 1 USD = 5 BRL
    let rate = FxRate::new(FxPair::USD_BRL, dec!(5), date(2024, 1, 1));
    
    // 100 USD -> 500 BRL
    let result = rate.convert_to_quote(dec!(100));
    assert_eq!(result, dec!(500));
}

#[test]
fn test_fx_rate_convert_to_base() {
    let rate = FxRate::new(FxPair::USD_BRL, dec!(5), date(2024, 1, 1));
    
    // 500 BRL -> 100 USD
    let result = rate.convert_to_base(dec!(500));
    assert_eq!(result, dec!(100));
}

#[test]
fn test_fx_rate_inverse() {
    let rate = FxRate::new(FxPair::USD_BRL, dec!(5), date(2024, 1, 1));
    let inverse = rate.inverse();
    
    assert_eq!(inverse.rate, dec!(0.2)); // 1/5
    assert_eq!(inverse.pair, FxPair::USD_BRL.inverse());
}

#[test]
fn test_fx_rate_return() {
    let start = FxRate::new(FxPair::USD_BRL, dec!(5), date(2024, 1, 1));
    let end = FxRate::new(FxPair::USD_BRL, dec!(5.50), date(2024, 1, 31));
    
    let fx_return = end.fx_return_from(&start);
    assert_eq!(fx_return, dec!(0.1)); // 10%
}

// =============================================================================
// INMEMORY PROVIDER TESTS
// =============================================================================

#[test]
fn test_provider_locf_exact_date() {
    let mut provider = InMemoryFxProvider::new();
    provider.add_rate(FxPair::USD_BRL, date(2024, 1, 15), dec!(5.00));
    
    let rate = provider.get_rate_locf(date(2024, 1, 15), FxPair::USD_BRL, 5).unwrap();
    
    assert_eq!(rate.rate, dec!(5.00));
    assert_eq!(rate.date, date(2024, 1, 15));
}

#[test]
fn test_provider_locf_gap_within_limit() {
    let mut provider = InMemoryFxProvider::new();
    // Rate on Monday
    provider.add_rate(FxPair::USD_BRL, date(2024, 1, 15), dec!(5.00));
    
    // Query on Friday (4 days later, within 5 day gap)
    let rate = provider.get_rate_locf(date(2024, 1, 19), FxPair::USD_BRL, 5).unwrap();
    
    assert_eq!(rate.rate, dec!(5.00));
    assert_eq!(rate.date, date(2024, 1, 15)); // Returns actual rate date
}

#[test]
fn test_provider_locf_gap_exceeds_limit() {
    let mut provider = InMemoryFxProvider::new();
    provider.add_rate(FxPair::USD_BRL, date(2024, 1, 15), dec!(5.00));
    
    // Query 7 days later (exceeds 5 day gap)
    let result = provider.get_rate_locf(date(2024, 1, 22), FxPair::USD_BRL, 5);
    
    assert!(matches!(result, Err(FxError::GapExceedsLimit { gap_days: 7, .. })));
}

#[test]
fn test_provider_holiday_gap_handled() {
    let mut provider = InMemoryFxProvider::new();
    
    // Rate on Friday Dec 20
    provider.add_rate(FxPair::USD_BRL, date(2024, 12, 20), dec!(5.00));
    // No rates for weekend + Christmas (Dec 21-25)
    // Rate on Dec 26
    provider.add_rate(FxPair::USD_BRL, date(2024, 12, 26), dec!(5.10));
    
    // Query on Dec 24 should use Dec 20 rate (4 day gap)
    let rate = provider.get_rate_locf(date(2024, 12, 24), FxPair::USD_BRL, 5).unwrap();
    assert_eq!(rate.rate, dec!(5.00));
    
    // Query on Dec 26 should use Dec 26 rate
    let rate = provider.get_rate_locf(date(2024, 12, 26), FxPair::USD_BRL, 5).unwrap();
    assert_eq!(rate.rate, dec!(5.10));
}

#[test]
fn test_provider_inverse_rate() {
    let mut provider = InMemoryFxProvider::new();
    // Only store USD/BRL
    provider.add_rate(FxPair::USD_BRL, date(2024, 1, 15), dec!(5.00));
    
    // Query BRL/USD (inverse)
    let pair = FxPair::new(Currency::BRL, Currency::USD);
    let rate = provider.get_rate(date(2024, 1, 15), pair).unwrap();
    
    assert_eq!(rate.rate, dec!(0.2)); // 1/5 = 0.2
}

// =============================================================================
// CONVERT_MONEY TESTS
// =============================================================================

#[test]
fn test_convert_usd_to_brl() {
    let mut provider = InMemoryFxProvider::new();
    provider.add_rate(FxPair::USD_BRL, date(2024, 1, 15), dec!(5.50));
    
    let usd = Money::new(dec!(1000), Currency::USD);
    let brl = convert_money(&usd, Currency::BRL, date(2024, 1, 15), &provider, 0).unwrap();
    
    assert_eq!(brl.amount(), dec!(5500));
    assert_eq!(brl.currency(), Currency::BRL);
}

#[test]
fn test_convert_brl_to_usd_inverse() {
    let mut provider = InMemoryFxProvider::new();
    // Only store USD/BRL
    provider.add_rate(FxPair::USD_BRL, date(2024, 1, 15), dec!(5.00));
    
    let brl = Money::new(dec!(5000), Currency::BRL);
    let usd = convert_money(&brl, Currency::USD, date(2024, 1, 15), &provider, 0).unwrap();
    
    assert_eq!(usd.amount(), dec!(1000));
    assert_eq!(usd.currency(), Currency::USD);
}

#[test]
fn test_convert_same_currency() {
    let provider = InMemoryFxProvider::new();
    let usd = Money::new(dec!(1000), Currency::USD);
    
    // No FX data needed for same currency
    let result = convert_money(&usd, Currency::USD, date(2024, 1, 15), &provider, 0).unwrap();
    
    assert_eq!(result.amount(), dec!(1000));
    assert_eq!(result.currency(), Currency::USD);
}

#[test]
fn test_convert_missing_rate() {
    let provider = InMemoryFxProvider::new();
    let usd = Money::new(dec!(1000), Currency::USD);
    
    let result = convert_money(&usd, Currency::BRL, date(2024, 1, 15), &provider, 0);
    
    assert!(matches!(result, Err(FxError::RateNotFound { .. })));
}

#[test]
fn test_convert_high_precision() {
    let mut provider = InMemoryFxProvider::new();
    provider.add_rate(FxPair::USD_BRL, date(2024, 1, 15), dec!(5.123456789));
    
    let usd = Money::new(dec!(1000.00), Currency::USD);
    let brl = convert_money(&usd, Currency::BRL, date(2024, 1, 15), &provider, 0).unwrap();
    
    // Should maintain full precision
    assert_eq!(brl.amount(), dec!(5123.456789));
}

// =============================================================================
// EDGE CASE TESTS (V1.1)
// =============================================================================

mod edge_cases {
    use super::*;

    // -------------------------------------------------------------------------
    // Empty Series Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_empty_provider_rate_not_found() {
        let provider = InMemoryFxProvider::new();
        
        let result = provider.get_rate(date(2024, 1, 15), FxPair::USD_BRL);
        
        assert!(matches!(result, Err(FxError::RateNotFound { .. })));
    }

    #[test]
    fn test_empty_provider_locf_not_found() {
        let provider = InMemoryFxProvider::new();
        
        let result = provider.get_rate_locf(date(2024, 1, 15), FxPair::USD_BRL, 30);
        
        assert!(matches!(result, Err(FxError::RateNotFound { .. })));
    }

    #[test]
    fn test_empty_provider_date_range() {
        let provider = InMemoryFxProvider::new();
        
        let range = provider.date_range(FxPair::USD_BRL);
        
        assert!(range.is_none(), "Empty provider should have no date range");
    }

    #[test]
    fn test_empty_provider_available_pairs() {
        let provider = InMemoryFxProvider::new();
        
        let pairs = provider.available_pairs();
        
        assert!(pairs.is_empty(), "Empty provider should have no pairs");
    }

    // -------------------------------------------------------------------------
    // Gap Exceeded Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_gap_exceeds_limit_by_one_day() {
        let mut provider = InMemoryFxProvider::new();
        provider.add_rate(FxPair::USD_BRL, date(2024, 1, 1), dec!(5.00));
        
        // Gap is 6 days, limit is 5
        let result = provider.get_rate_locf(date(2024, 1, 7), FxPair::USD_BRL, 5);
        
        assert!(matches!(result, Err(FxError::GapExceedsLimit { gap_days: 6, max_gap_days: 5, .. })));
    }

    #[test]
    fn test_gap_at_limit_succeeds() {
        let mut provider = InMemoryFxProvider::new();
        provider.add_rate(FxPair::USD_BRL, date(2024, 1, 1), dec!(5.00));
        
        // Gap is exactly 5 days
        let result = provider.get_rate_locf(date(2024, 1, 6), FxPair::USD_BRL, 5);
        
        assert!(result.is_ok(), "Gap at limit should succeed");
        let rate = result.unwrap();
        assert_eq!(rate.date, date(2024, 1, 1));
    }

    #[test]
    fn test_gap_error_message_contains_details() {
        let mut provider = InMemoryFxProvider::new();
        provider.add_rate(FxPair::USD_BRL, date(2024, 1, 1), dec!(5.00));
        
        let result = provider.get_rate_locf(date(2024, 1, 15), FxPair::USD_BRL, 5);
        
        if let Err(FxError::GapExceedsLimit { pair, requested_date, last_available_date, gap_days, max_gap_days }) = result {
            assert_eq!(pair, FxPair::USD_BRL);
            assert_eq!(requested_date, date(2024, 1, 15));
            assert_eq!(last_available_date, date(2024, 1, 1));
            assert_eq!(gap_days, 14);
            assert_eq!(max_gap_days, 5);
        } else {
            panic!("Expected GapExceedsLimit error");
        }
    }

    // -------------------------------------------------------------------------
    // Inverse Not Available Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_inverse_not_available_when_no_data() {
        let provider = InMemoryFxProvider::new();
        
        // No EUR/USD data, so can't get USD/EUR via inverse
        let result = provider.get_rate(date(2024, 1, 15), FxPair::new(Currency::USD, Currency::EUR));
        
        assert!(matches!(result, Err(FxError::RateNotFound { .. })));
    }

    #[test]
    fn test_inverse_works_when_available() {
        let mut provider = InMemoryFxProvider::new();
        // Only store EUR/USD
        provider.add_rate(FxPair::EUR_USD, date(2024, 1, 15), dec!(1.10));
        
        // Should be able to get USD/EUR via inverse
        let result = provider.get_rate(date(2024, 1, 15), FxPair::new(Currency::USD, Currency::EUR));
        
        assert!(result.is_ok());
        let rate = result.unwrap();
        // USD/EUR = 1 / 1.10 ≈ 0.909
        assert!((rate.rate - dec!(0.909090909090909090909090909)).abs() < dec!(0.0001));
    }

    // -------------------------------------------------------------------------
    // Base Currency Equals Local Currency Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_base_equals_local_no_fx_effect() {
        let provider = InMemoryFxProvider::new();
        
        // BRL to BRL - should always work with rate = 1
        let brl = Money::new(dec!(1000), Currency::BRL);
        let result = convert_money(&brl, Currency::BRL, date(2024, 1, 15), &provider, 0);
        
        assert!(result.is_ok());
        let converted = result.unwrap();
        assert_eq!(converted.amount(), dec!(1000));
        assert_eq!(converted.currency(), Currency::BRL);
    }

    #[test]
    fn test_identity_conversion_no_provider_needed() {
        // Empty provider - identity should still work
        let provider = InMemoryFxProvider::new();
        
        let currencies = [Currency::USD, Currency::BRL, Currency::EUR];
        
        for currency in currencies {
            let money = Money::new(dec!(12345.67), currency);
            let result = convert_money(&money, currency, date(2024, 6, 15), &provider, 0);
            
            assert!(result.is_ok(), "Identity conversion for {:?} should work", currency);
            assert_eq!(result.unwrap().amount(), dec!(12345.67));
        }
    }

    // -------------------------------------------------------------------------
    // Multiple Currencies Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_multiple_currency_pairs_coexist() {
        let mut provider = InMemoryFxProvider::new();
        provider.add_rate(FxPair::USD_BRL, date(2024, 1, 15), dec!(5.00));
        provider.add_rate(FxPair::EUR_USD, date(2024, 1, 15), dec!(1.10));
        provider.add_rate(FxPair::EUR_BRL, date(2024, 1, 15), dec!(5.50));
        
        // All pairs should be retrievable
        let usd_brl = provider.get_rate(date(2024, 1, 15), FxPair::USD_BRL).unwrap();
        let eur_usd = provider.get_rate(date(2024, 1, 15), FxPair::EUR_USD).unwrap();
        let eur_brl = provider.get_rate(date(2024, 1, 15), FxPair::EUR_BRL).unwrap();
        
        assert_eq!(usd_brl.rate, dec!(5.00));
        assert_eq!(eur_usd.rate, dec!(1.10));
        assert_eq!(eur_brl.rate, dec!(5.50));
    }

    #[test]
    fn test_different_pairs_different_dates() {
        let mut provider = InMemoryFxProvider::new();
        provider.add_rate(FxPair::USD_BRL, date(2024, 1, 10), dec!(4.90));
        provider.add_rate(FxPair::USD_BRL, date(2024, 1, 15), dec!(5.00));
        provider.add_rate(FxPair::EUR_USD, date(2024, 1, 12), dec!(1.08));
        provider.add_rate(FxPair::EUR_USD, date(2024, 1, 15), dec!(1.10));
        
        // LOCF should pick correct dates for each pair
        let usd_brl = provider.get_rate_locf(date(2024, 1, 14), FxPair::USD_BRL, 5).unwrap();
        let eur_usd = provider.get_rate_locf(date(2024, 1, 14), FxPair::EUR_USD, 5).unwrap();
        
        assert_eq!(usd_brl.rate, dec!(4.90)); // From Jan 10
        assert_eq!(usd_brl.date, date(2024, 1, 10));
        
        assert_eq!(eur_usd.rate, dec!(1.08)); // From Jan 12
        assert_eq!(eur_usd.date, date(2024, 1, 12));
    }

    // -------------------------------------------------------------------------
    // Determinism Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_conversion_deterministic() {
        let mut provider = InMemoryFxProvider::new();
        provider.add_rate(FxPair::USD_BRL, date(2024, 1, 15), dec!(5.00));
        
        let usd = Money::new(dec!(1000), Currency::USD);
        
        // Run conversion multiple times
        let results: Vec<_> = (0..10)
            .map(|_| convert_money(&usd, Currency::BRL, date(2024, 1, 15), &provider, 0).unwrap())
            .collect();
        
        // All results should be identical
        for result in &results {
            assert_eq!(result.amount(), dec!(5000));
            assert_eq!(result.currency(), Currency::BRL);
        }
    }

    #[test]
    fn test_provider_query_deterministic() {
        let mut provider = InMemoryFxProvider::new();
        provider.add_rate(FxPair::USD_BRL, date(2024, 1, 10), dec!(4.90));
        provider.add_rate(FxPair::USD_BRL, date(2024, 1, 15), dec!(5.00));
        
        // Query same date multiple times
        let results: Vec<_> = (0..10)
            .map(|_| provider.get_rate_locf(date(2024, 1, 14), FxPair::USD_BRL, 10).unwrap())
            .collect();
        
        // All results should be identical
        for result in &results {
            assert_eq!(result.rate, dec!(4.90));
            assert_eq!(result.date, date(2024, 1, 10));
        }
    }

    // -------------------------------------------------------------------------
    // Weekend/Holiday LOCF Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_locf_friday_to_monday() {
        let mut provider = InMemoryFxProvider::new();
        // Friday Jan 12, 2024
        provider.add_rate(FxPair::USD_BRL, date(2024, 1, 12), dec!(5.00));
        
        // Saturday Jan 13 - should use Friday rate (1 day gap)
        let sat = provider.get_rate_locf(date(2024, 1, 13), FxPair::USD_BRL, 3).unwrap();
        assert_eq!(sat.rate, dec!(5.00));
        
        // Sunday Jan 14 - should use Friday rate (2 day gap)
        let sun = provider.get_rate_locf(date(2024, 1, 14), FxPair::USD_BRL, 3).unwrap();
        assert_eq!(sun.rate, dec!(5.00));
        
        // Monday Jan 15 - should use Friday rate (3 day gap, at limit)
        let mon = provider.get_rate_locf(date(2024, 1, 15), FxPair::USD_BRL, 3).unwrap();
        assert_eq!(mon.rate, dec!(5.00));
    }

    #[test]
    fn test_locf_holiday_gap() {
        let mut provider = InMemoryFxProvider::new();
        // Thursday before 3-day holiday weekend
        provider.add_rate(FxPair::USD_BRL, date(2024, 7, 4), dec!(5.20));
        
        // After 4-day gap (Fri, Sat, Sun, Mon) - Tuesday Jul 9
        let result = provider.get_rate_locf(date(2024, 7, 9), FxPair::USD_BRL, 5);
        
        assert!(result.is_ok());
        let rate = result.unwrap();
        assert_eq!(rate.rate, dec!(5.20));
        assert_eq!(rate.date, date(2024, 7, 4));
    }
}

