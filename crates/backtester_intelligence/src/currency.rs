//! Currency types for multi-currency portfolio management.
//!
//! # Design Principles
//!
//! 1. **Explicit units**: Every monetary value carries its currency
//! 2. **No implicit conversion**: Conversion only via explicit `convert_money()`
//! 3. **Type safety**: Money arithmetic only between same currencies
//! 4. **Precision**: All amounts use `Decimal` (never f64 for money)
//!
//! # FX Pair Convention
//!
//! `FxPair::new(USD, BRL)` represents USD/BRL, meaning:
//! - 1 USD = rate BRL (quote per base)
//! - rate = 5.50 means 1 USD buys 5.50 BRL
//!
//! # Example
//!
//! ```
//! use backtester_intelligence::currency::{Currency, Money, FxPair};
//! use rust_decimal_macros::dec;
//!
//! let usd_amount = Money::new(dec!(1000), Currency::USD);
//! let pair = FxPair::new(Currency::USD, Currency::BRL);
//! // To convert: need FxRateProvider (see fx.rs)
//! ```

use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Sub, Neg};

use crate::filters::Market;

// =============================================================================
// CURRENCY
// =============================================================================

/// Supported currencies.
///
/// Extensible: add new variants as needed.
/// Each currency has a 3-letter ISO 4217 code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "UPPERCASE")]
pub enum Currency {
    /// Brazilian Real
    #[default]
    BRL,
    /// United States Dollar
    USD,
    /// Euro
    EUR,
}

impl Currency {
    /// ISO 4217 3-letter code.
    pub fn code(&self) -> &'static str {
        match self {
            Currency::BRL => "BRL",
            Currency::USD => "USD",
            Currency::EUR => "EUR",
        }
    }

    /// Currency symbol for display.
    pub fn symbol(&self) -> &'static str {
        match self {
            Currency::BRL => "R$",
            Currency::USD => "$",
            Currency::EUR => "€",
        }
    }

    /// Parse from string (case-insensitive).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "BRL" => Some(Currency::BRL),
            "USD" => Some(Currency::USD),
            "EUR" => Some(Currency::EUR),
            _ => None,
        }
    }

    /// Standard decimal places for this currency.
    pub fn decimal_places(&self) -> u32 {
        match self {
            Currency::BRL | Currency::USD | Currency::EUR => 2,
        }
    }
}

impl fmt::Display for Currency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code())
    }
}

// =============================================================================
// MARKET -> CURRENCY MAPPING
// =============================================================================

impl Market {
    /// Get the primary currency for this market.
    pub fn currency(&self) -> Currency {
        match self {
            Market::BR => Currency::BRL,
            Market::US => Currency::USD,
        }
    }
}

impl From<Market> for Currency {
    fn from(market: Market) -> Self {
        market.currency()
    }
}

// =============================================================================
// MONEY
// =============================================================================

/// A monetary amount with explicit currency.
///
/// # Invariants
///
/// - Arithmetic operations (add, sub) only allowed between same currencies
/// - No implicit conversion - use `convert_money()` from fx module
/// - Amount stored as Decimal for precision
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Money {
    amount: Decimal,
    currency: Currency,
}

impl Money {
    /// Create a new Money value.
    pub fn new(amount: Decimal, currency: Currency) -> Self {
        Self { amount, currency }
    }

    /// Create zero amount in specified currency.
    pub fn zero(currency: Currency) -> Self {
        Self {
            amount: Decimal::ZERO,
            currency,
        }
    }

    /// Get the amount.
    pub fn amount(&self) -> Decimal {
        self.amount
    }

    /// Get the currency.
    pub fn currency(&self) -> Currency {
        self.currency
    }

    /// Check if amount is zero.
    pub fn is_zero(&self) -> bool {
        self.amount.is_zero()
    }

    /// Check if amount is positive.
    pub fn is_positive(&self) -> bool {
        self.amount > Decimal::ZERO
    }

    /// Check if amount is negative.
    pub fn is_negative(&self) -> bool {
        self.amount < Decimal::ZERO
    }

    /// Get absolute value.
    pub fn abs(&self) -> Self {
        Self {
            amount: self.amount.abs(),
            currency: self.currency,
        }
    }

    /// Multiply by a scalar (e.g., quantity).
    pub fn scale(&self, factor: Decimal) -> Self {
        Self {
            amount: self.amount * factor,
            currency: self.currency,
        }
    }

    /// Round to currency's standard decimal places.
    pub fn round(&self) -> Self {
        let places = self.currency.decimal_places();
        Self {
            amount: self.amount.round_dp(places),
            currency: self.currency,
        }
    }

    /// Try to add another Money value. Fails if currencies differ.
    pub fn try_add(&self, other: &Money) -> Result<Money, CurrencyMismatchError> {
        if self.currency != other.currency {
            return Err(CurrencyMismatchError {
                expected: self.currency,
                actual: other.currency,
                operation: "add",
            });
        }
        Ok(Self {
            amount: self.amount + other.amount,
            currency: self.currency,
        })
    }

    /// Try to subtract another Money value. Fails if currencies differ.
    pub fn try_sub(&self, other: &Money) -> Result<Money, CurrencyMismatchError> {
        if self.currency != other.currency {
            return Err(CurrencyMismatchError {
                expected: self.currency,
                actual: other.currency,
                operation: "subtract",
            });
        }
        Ok(Self {
            amount: self.amount - other.amount,
            currency: self.currency,
        })
    }
}

impl fmt::Display for Money {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {:.2}", self.currency.symbol(), self.amount)
    }
}

impl Default for Money {
    fn default() -> Self {
        Self::zero(Currency::default())
    }
}

impl Neg for Money {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            amount: -self.amount,
            currency: self.currency,
        }
    }
}

// Implement Add/Sub that panic on currency mismatch (for convenience in tests)
// Production code should use try_add/try_sub

impl Add for Money {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.try_add(&rhs).expect("Currency mismatch in Money addition")
    }
}

impl Sub for Money {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.try_sub(&rhs).expect("Currency mismatch in Money subtraction")
    }
}

// =============================================================================
// FX PAIR
// =============================================================================

/// A currency pair for FX rates.
///
/// Convention: `FxPair { base: USD, quote: BRL }` represents USD/BRL
/// - base = "1 unit of" currency
/// - quote = "equals X units of" currency
/// - rate = 5.50 means 1 USD = 5.50 BRL
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FxPair {
    /// Base currency (1 unit of this)
    pub base: Currency,
    /// Quote currency (equals rate units of this)
    pub quote: Currency,
}

impl FxPair {
    /// Create a new FX pair.
    pub fn new(base: Currency, quote: Currency) -> Self {
        Self { base, quote }
    }

    /// USD/BRL pair (most common for BR-US portfolios).
    pub const USD_BRL: FxPair = FxPair {
        base: Currency::USD,
        quote: Currency::BRL,
    };

    /// EUR/USD pair.
    pub const EUR_USD: FxPair = FxPair {
        base: Currency::EUR,
        quote: Currency::USD,
    };

    /// EUR/BRL pair.
    pub const EUR_BRL: FxPair = FxPair {
        base: Currency::EUR,
        quote: Currency::BRL,
    };

    /// Get the inverse pair (e.g., USD/BRL -> BRL/USD).
    pub fn inverse(&self) -> Self {
        Self {
            base: self.quote,
            quote: self.base,
        }
    }

    /// Check if this is a same-currency pair (rate = 1).
    pub fn is_identity(&self) -> bool {
        self.base == self.quote
    }

    /// Get the pair needed to convert from `from` to `to`.
    /// Returns None if same currency.
    pub fn for_conversion(from: Currency, to: Currency) -> Option<Self> {
        if from == to {
            None
        } else {
            // Convention: base = from, quote = to
            // rate means 1 `from` = rate `to`
            Some(Self::new(from, to))
        }
    }

    /// Standard string representation (e.g., "USD/BRL").
    pub fn as_str(&self) -> String {
        format!("{}/{}", self.base.code(), self.quote.code())
    }

    /// Parse from string (e.g., "USD/BRL" or "USDBRL").
    pub fn from_str(s: &str) -> Option<Self> {
        let s = s.trim().to_uppercase();
        
        // Try "USD/BRL" format
        if let Some((base, quote)) = s.split_once('/') {
            let base = Currency::from_str(base)?;
            let quote = Currency::from_str(quote)?;
            return Some(Self::new(base, quote));
        }
        
        // Try "USDBRL" format (6 chars)
        if s.len() == 6 {
            let base = Currency::from_str(&s[0..3])?;
            let quote = Currency::from_str(&s[3..6])?;
            return Some(Self::new(base, quote));
        }
        
        None
    }
}

impl fmt::Display for FxPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.base.code(), self.quote.code())
    }
}

// =============================================================================
// FX RATE
// =============================================================================

/// An FX rate observation at a specific date.
///
/// # Convention
///
/// For `pair = USD/BRL`, `rate = 5.50`:
/// - 1 USD = 5.50 BRL (one unit of base equals `rate` units of quote)
/// - To convert USD -> BRL: amount * rate
/// - To convert BRL -> USD: amount / rate
///
/// # Example
///
/// ```
/// use backtester_intelligence::currency::{Currency, FxPair, FxRate};
/// use rust_decimal_macros::dec;
/// use chrono::NaiveDate;
///
/// let date = NaiveDate::from_ymd_opt(2024, 12, 27).unwrap();
/// let rate = FxRate::new(FxPair::USD_BRL, dec!(5.50), date);
///
/// // Semantic accessors prevent confusion
/// assert_eq!(rate.rate_quote_per_base(), dec!(5.50));
/// assert_eq!(rate.describe(), "1 USD = 5.50 BRL");
///
/// // Convert 100 USD to BRL
/// let brl = rate.convert_to_quote(dec!(100));
/// assert_eq!(brl, dec!(550));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FxRate {
    /// The currency pair.
    pub pair: FxPair,
    /// The exchange rate (quote per base).
    /// Interpretation: 1 unit of `pair.base` = `rate` units of `pair.quote`.
    pub rate: Decimal,
    /// The observation date.
    pub date: NaiveDate,
}

impl FxRate {
    /// Create a new FX rate.
    pub fn new(pair: FxPair, rate: Decimal, date: NaiveDate) -> Self {
        Self { pair, rate, date }
    }

    /// Create an identity rate (same currency, rate = 1).
    pub fn identity(currency: Currency, date: NaiveDate) -> Self {
        Self {
            pair: FxPair::new(currency, currency),
            rate: Decimal::ONE,
            date,
        }
    }

    // -------------------------------------------------------------------------
    // Semantic Accessors (prevent confusion)
    // -------------------------------------------------------------------------

    /// Get the rate as "quote currency per one unit of base currency".
    ///
    /// This is a semantic alias for `self.rate` that makes the meaning explicit.
    ///
    /// # Example
    ///
    /// For USD/BRL = 5.50:
    /// - `rate_quote_per_base()` returns 5.50
    /// - Meaning: 1 USD = 5.50 BRL
    #[inline]
    pub fn rate_quote_per_base(&self) -> Decimal {
        self.rate
    }

    /// Get a human-readable description of what this rate means.
    ///
    /// # Example
    ///
    /// For USD/BRL = 5.50: returns "1 USD = 5.50 BRL"
    pub fn describe(&self) -> String {
        format!(
            "1 {} = {} {}",
            self.pair.base.code(),
            self.rate,
            self.pair.quote.code()
        )
    }

    /// Get a human-readable description with the date.
    ///
    /// # Example
    ///
    /// For USD/BRL = 5.50 on 2024-12-27: returns "1 USD = 5.50 BRL (2024-12-27)"
    pub fn describe_with_date(&self) -> String {
        format!(
            "1 {} = {} {} ({})",
            self.pair.base.code(),
            self.rate,
            self.pair.quote.code(),
            self.date
        )
    }

    // -------------------------------------------------------------------------
    // Rate Operations
    // -------------------------------------------------------------------------

    /// Get the inverse rate.
    ///
    /// If this is USD/BRL = 5.50, inverse is BRL/USD = 1/5.50 ≈ 0.1818
    pub fn inverse(&self) -> Self {
        Self {
            pair: self.pair.inverse(),
            rate: Decimal::ONE / self.rate,
            date: self.date,
        }
    }

    /// Convert an amount from base to quote currency.
    ///
    /// For USD/BRL = 5.50: convert_to_quote(100 USD) = 550 BRL
    pub fn convert_to_quote(&self, amount: Decimal) -> Decimal {
        amount * self.rate
    }

    /// Convert an amount from quote to base currency.
    ///
    /// For USD/BRL = 5.50: convert_to_base(550 BRL) = 100 USD
    pub fn convert_to_base(&self, amount: Decimal) -> Decimal {
        if self.rate.is_zero() {
            Decimal::ZERO
        } else {
            amount / self.rate
        }
    }

    /// Calculate the return of the FX rate from another rate.
    ///
    /// R_fx = rate_end / rate_start - 1
    pub fn fx_return_from(&self, start_rate: &FxRate) -> Decimal {
        if start_rate.rate.is_zero() {
            Decimal::ZERO
        } else {
            self.rate / start_rate.rate - Decimal::ONE
        }
    }
}

impl fmt::Display for FxRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} = {} ({})", self.pair, self.rate, self.date)
    }
}

// =============================================================================
// ERRORS
// =============================================================================

/// Error when attempting arithmetic on Money with different currencies.
#[derive(Debug, Clone)]
pub struct CurrencyMismatchError {
    pub expected: Currency,
    pub actual: Currency,
    pub operation: &'static str,
}

impl fmt::Display for CurrencyMismatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Currency mismatch in {}: expected {}, got {}",
            self.operation, self.expected, self.actual
        )
    }
}

impl std::error::Error for CurrencyMismatchError {}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    // -------------------------------------------------------------------------
    // Currency Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_currency_code() {
        assert_eq!(Currency::BRL.code(), "BRL");
        assert_eq!(Currency::USD.code(), "USD");
        assert_eq!(Currency::EUR.code(), "EUR");
    }

    #[test]
    fn test_currency_from_str() {
        assert_eq!(Currency::from_str("BRL"), Some(Currency::BRL));
        assert_eq!(Currency::from_str("usd"), Some(Currency::USD));
        assert_eq!(Currency::from_str("Eur"), Some(Currency::EUR));
        assert_eq!(Currency::from_str("GBP"), None);
    }

    #[test]
    fn test_market_to_currency() {
        assert_eq!(Market::BR.currency(), Currency::BRL);
        assert_eq!(Market::US.currency(), Currency::USD);
    }

    // -------------------------------------------------------------------------
    // Money Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_money_creation() {
        let m = Money::new(dec!(1000.50), Currency::USD);
        assert_eq!(m.amount(), dec!(1000.50));
        assert_eq!(m.currency(), Currency::USD);
    }

    #[test]
    fn test_money_zero() {
        let m = Money::zero(Currency::BRL);
        assert!(m.is_zero());
        assert_eq!(m.currency(), Currency::BRL);
    }

    #[test]
    fn test_money_same_currency_add() {
        let a = Money::new(dec!(100), Currency::USD);
        let b = Money::new(dec!(50), Currency::USD);
        let result = a.try_add(&b).unwrap();
        assert_eq!(result.amount(), dec!(150));
        assert_eq!(result.currency(), Currency::USD);
    }

    #[test]
    fn test_money_same_currency_sub() {
        let a = Money::new(dec!(100), Currency::USD);
        let b = Money::new(dec!(30), Currency::USD);
        let result = a.try_sub(&b).unwrap();
        assert_eq!(result.amount(), dec!(70));
    }

    #[test]
    fn test_money_different_currency_add_fails() {
        let a = Money::new(dec!(100), Currency::USD);
        let b = Money::new(dec!(50), Currency::BRL);
        let result = a.try_add(&b);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.expected, Currency::USD);
        assert_eq!(err.actual, Currency::BRL);
    }

    #[test]
    fn test_money_scale() {
        let m = Money::new(dec!(100), Currency::USD);
        let scaled = m.scale(dec!(2.5));
        assert_eq!(scaled.amount(), dec!(250));
    }

    #[test]
    fn test_money_neg() {
        let m = Money::new(dec!(100), Currency::USD);
        let neg = -m;
        assert_eq!(neg.amount(), dec!(-100));
    }

    #[test]
    fn test_money_display() {
        let m = Money::new(dec!(1234.56), Currency::BRL);
        assert_eq!(format!("{}", m), "R$ 1234.56");
    }

    // -------------------------------------------------------------------------
    // FxPair Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_fx_pair_creation() {
        let pair = FxPair::new(Currency::USD, Currency::BRL);
        assert_eq!(pair.base, Currency::USD);
        assert_eq!(pair.quote, Currency::BRL);
    }

    #[test]
    fn test_fx_pair_inverse() {
        let pair = FxPair::USD_BRL;
        let inv = pair.inverse();
        assert_eq!(inv.base, Currency::BRL);
        assert_eq!(inv.quote, Currency::USD);
    }

    #[test]
    fn test_fx_pair_is_identity() {
        let identity = FxPair::new(Currency::USD, Currency::USD);
        assert!(identity.is_identity());

        let non_identity = FxPair::USD_BRL;
        assert!(!non_identity.is_identity());
    }

    #[test]
    fn test_fx_pair_for_conversion() {
        // Same currency -> None
        assert!(FxPair::for_conversion(Currency::USD, Currency::USD).is_none());

        // Different currency -> Some
        let pair = FxPair::for_conversion(Currency::USD, Currency::BRL).unwrap();
        assert_eq!(pair.base, Currency::USD);
        assert_eq!(pair.quote, Currency::BRL);
    }

    #[test]
    fn test_fx_pair_from_str() {
        assert_eq!(
            FxPair::from_str("USD/BRL"),
            Some(FxPair::new(Currency::USD, Currency::BRL))
        );
        assert_eq!(
            FxPair::from_str("USDBRL"),
            Some(FxPair::new(Currency::USD, Currency::BRL))
        );
        assert_eq!(
            FxPair::from_str("eur/usd"),
            Some(FxPair::new(Currency::EUR, Currency::USD))
        );
        assert_eq!(FxPair::from_str("INVALID"), None);
    }

    #[test]
    fn test_fx_pair_display() {
        assert_eq!(format!("{}", FxPair::USD_BRL), "USD/BRL");
    }

    // -------------------------------------------------------------------------
    // FxRate Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_fx_rate_creation() {
        let date = NaiveDate::from_ymd_opt(2024, 12, 27).unwrap();
        let rate = FxRate::new(FxPair::USD_BRL, dec!(5.50), date);
        
        assert_eq!(rate.pair, FxPair::USD_BRL);
        assert_eq!(rate.rate, dec!(5.50));
        assert_eq!(rate.date, date);
    }

    #[test]
    fn test_fx_rate_identity() {
        let date = NaiveDate::from_ymd_opt(2024, 12, 27).unwrap();
        let rate = FxRate::identity(Currency::USD, date);
        
        assert_eq!(rate.rate, Decimal::ONE);
        assert!(rate.pair.is_identity());
    }

    #[test]
    fn test_fx_rate_inverse() {
        let date = NaiveDate::from_ymd_opt(2024, 12, 27).unwrap();
        let rate = FxRate::new(FxPair::USD_BRL, dec!(5), date);
        let inv = rate.inverse();
        
        assert_eq!(inv.pair, FxPair::USD_BRL.inverse());
        assert_eq!(inv.rate, dec!(0.2)); // 1/5 = 0.2
    }

    #[test]
    fn test_fx_rate_convert_to_quote() {
        let date = NaiveDate::from_ymd_opt(2024, 12, 27).unwrap();
        let rate = FxRate::new(FxPair::USD_BRL, dec!(5), date);
        
        // 100 USD * 5 = 500 BRL
        let result = rate.convert_to_quote(dec!(100));
        assert_eq!(result, dec!(500));
    }

    #[test]
    fn test_fx_rate_convert_to_base() {
        let date = NaiveDate::from_ymd_opt(2024, 12, 27).unwrap();
        let rate = FxRate::new(FxPair::USD_BRL, dec!(5), date);
        
        // 500 BRL / 5 = 100 USD
        let result = rate.convert_to_base(dec!(500));
        assert_eq!(result, dec!(100));
    }

    #[test]
    fn test_fx_rate_fx_return() {
        let date1 = NaiveDate::from_ymd_opt(2024, 12, 26).unwrap();
        let date2 = NaiveDate::from_ymd_opt(2024, 12, 27).unwrap();
        
        let rate_start = FxRate::new(FxPair::USD_BRL, dec!(5), date1);
        let rate_end = FxRate::new(FxPair::USD_BRL, dec!(5.50), date2);
        
        // R_fx = 5.50 / 5.00 - 1 = 0.10 = 10%
        let fx_return = rate_end.fx_return_from(&rate_start);
        assert_eq!(fx_return, dec!(0.1));
    }

    #[test]
    fn test_fx_rate_display() {
        let date = NaiveDate::from_ymd_opt(2024, 12, 27).unwrap();
        let rate = FxRate::new(FxPair::USD_BRL, dec!(5.50), date);
        assert_eq!(format!("{}", rate), "USD/BRL = 5.50 (2024-12-27)");
    }

    // -------------------------------------------------------------------------
    // FxRate Semantic Accessor Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_fx_rate_quote_per_base() {
        let date = NaiveDate::from_ymd_opt(2024, 12, 27).unwrap();
        let rate = FxRate::new(FxPair::USD_BRL, dec!(5.50), date);
        
        // Semantic accessor returns same value as .rate
        assert_eq!(rate.rate_quote_per_base(), dec!(5.50));
        assert_eq!(rate.rate_quote_per_base(), rate.rate);
    }

    #[test]
    fn test_fx_rate_describe() {
        let date = NaiveDate::from_ymd_opt(2024, 12, 27).unwrap();
        
        // USD/BRL = 5.50
        let usd_brl = FxRate::new(FxPair::USD_BRL, dec!(5.50), date);
        assert_eq!(usd_brl.describe(), "1 USD = 5.50 BRL");
        
        // EUR/USD = 1.10
        let eur_usd = FxRate::new(FxPair::EUR_USD, dec!(1.10), date);
        assert_eq!(eur_usd.describe(), "1 EUR = 1.10 USD");
        
        // EUR/BRL = 6.05
        let eur_brl = FxRate::new(FxPair::EUR_BRL, dec!(6.05), date);
        assert_eq!(eur_brl.describe(), "1 EUR = 6.05 BRL");
    }

    #[test]
    fn test_fx_rate_describe_with_date() {
        let date = NaiveDate::from_ymd_opt(2024, 12, 27).unwrap();
        let rate = FxRate::new(FxPair::USD_BRL, dec!(5.50), date);
        
        assert_eq!(rate.describe_with_date(), "1 USD = 5.50 BRL (2024-12-27)");
    }

    // -------------------------------------------------------------------------
    // Realistic FX Value Tests (preventing confusion)
    // -------------------------------------------------------------------------

    #[test]
    fn test_fx_rate_realistic_usd_brl_range() {
        // USD/BRL typically ranges from 4.50 to 6.50 in recent years
        let date = NaiveDate::from_ymd_opt(2024, 12, 27).unwrap();
        
        let low = FxRate::new(FxPair::USD_BRL, dec!(4.50), date);
        assert_eq!(low.describe(), "1 USD = 4.50 BRL");
        // 1000 USD = 4500 BRL
        assert_eq!(low.convert_to_quote(dec!(1000)), dec!(4500));
        
        let high = FxRate::new(FxPair::USD_BRL, dec!(6.50), date);
        assert_eq!(high.describe(), "1 USD = 6.50 BRL");
        // 1000 USD = 6500 BRL
        assert_eq!(high.convert_to_quote(dec!(1000)), dec!(6500));
    }

    #[test]
    fn test_fx_rate_realistic_eur_usd_range() {
        // EUR/USD typically ranges from 1.05 to 1.15
        let date = NaiveDate::from_ymd_opt(2024, 12, 27).unwrap();
        
        let low = FxRate::new(FxPair::EUR_USD, dec!(1.05), date);
        assert_eq!(low.describe(), "1 EUR = 1.05 USD");
        // 1000 EUR = 1050 USD
        assert_eq!(low.convert_to_quote(dec!(1000)), dec!(1050));
        
        let high = FxRate::new(FxPair::EUR_USD, dec!(1.15), date);
        assert_eq!(high.describe(), "1 EUR = 1.15 USD");
        // 1000 EUR = 1150 USD
        assert_eq!(high.convert_to_quote(dec!(1000)), dec!(1150));
    }

    #[test]
    fn test_fx_rate_inverse_semantic_clarity() {
        let date = NaiveDate::from_ymd_opt(2024, 12, 27).unwrap();
        
        // USD/BRL = 5.00 means 1 USD = 5.00 BRL
        let usd_brl = FxRate::new(FxPair::USD_BRL, dec!(5.00), date);
        assert_eq!(usd_brl.describe(), "1 USD = 5.00 BRL");
        
        // Inverse: BRL/USD = 0.20 means 1 BRL = 0.20 USD
        let brl_usd = usd_brl.inverse();
        assert_eq!(brl_usd.describe(), "1 BRL = 0.2 USD");
        assert_eq!(brl_usd.rate_quote_per_base(), dec!(0.2));
        
        // Convert 5000 BRL to USD: 5000 * 0.20 = 1000 USD
        assert_eq!(brl_usd.convert_to_quote(dec!(5000)), dec!(1000));
    }
}

