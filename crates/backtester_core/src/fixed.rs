//! Fixed-point numeric types for high-performance backtesting.
//!
//! This module provides deterministic, bit-exact numeric types that replace
//! `rust_decimal::Decimal` in the hot path for significant performance gains.
//!
//! # Types
//!
//! - [`Price`] - Asset prices (6 decimal places, scale 1e6)
//! - [`Money`] - Cash, equity, PnL (6 decimal places, scale 1e6)
//! - [`Rate`] - Rates, dividends (8 decimal places, scale 1e8)
//!
//! # Performance
//!
//! Fixed-point i64 operations are 10-50x faster than `Decimal`:
//! - Addition/Subtraction: Single i64 operation
//! - Multiplication: Uses i128 intermediate to prevent overflow
//! - Division: Uses i128 intermediate for precision
//!
//! # Precision
//!
//! - Scale 1e6 = 6 decimal places (supports micro-units for crypto)
//! - Max value: i64::MAX / 1e6 = 9,223,372,036,854 (9.2 trillion)
//! - Min value: i64::MIN / 1e6 = -9,223,372,036,854

use std::fmt;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Neg, Sub, SubAssign};

/// Scale factor for Price and Money (6 decimal places).
pub const SCALE_6: i64 = 1_000_000;

/// Scale factor for Rate (8 decimal places).
pub const SCALE_8: i64 = 100_000_000;

// =============================================================================
// PRICE
// =============================================================================

/// Fixed-point price with 6 decimal places.
///
/// Used for asset prices (open, high, low, close, etc.).
///
/// # Examples
///
/// ```
/// use backtester_core::fixed::Price;
///
/// let price = Price::from_f64(123.456789);
/// assert_eq!(price.to_f64(), 123.456789);
///
/// // Multiply by shares to get Money
/// let shares = 100;
/// let value = price.mul_shares(shares);
/// assert_eq!(value.to_f64(), 12345.6789);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct Price(i64);

impl Price {
    /// Zero price.
    pub const ZERO: Self = Self(0);

    /// One unit (1.000000).
    pub const ONE: Self = Self(SCALE_6);

    /// Maximum representable price (~9.2 trillion).
    pub const MAX: Self = Self(i64::MAX);

    /// Minimum representable price (~-9.2 trillion).
    pub const MIN: Self = Self(i64::MIN);

    /// Create from raw scaled value.
    #[inline]
    #[must_use]
    pub const fn from_raw(raw: i64) -> Self {
        Self(raw)
    }

    /// Get raw scaled value.
    #[inline]
    #[must_use]
    pub const fn raw(self) -> i64 {
        self.0
    }

    /// Create from f64 (rounds to 6 decimal places).
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_f64(value: f64) -> Self {
        Self((value * SCALE_6 as f64).round() as i64)
    }

    /// Convert to f64.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / SCALE_6 as f64
    }

    /// Create from integer (no scaling needed for whole numbers).
    #[inline]
    #[must_use]
    pub const fn from_int(value: i64) -> Self {
        Self(value * SCALE_6)
    }

    /// Multiply price by shares to get Money.
    ///
    /// Uses i128 intermediate to prevent overflow.
    ///
    /// # Math
    ///
    /// Price is stored as `price * SCALE_6` (e.g., $50.00 = 50_000_000).
    /// Money is stored as `money * SCALE_6` (e.g., $5000 = 5_000_000_000).
    /// So: `price_scaled * shares = money_scaled` directly.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn mul_shares(self, shares: i64) -> Money {
        // price_scaled * shares = money_scaled (both use same scale)
        // Using i128 to prevent overflow: max(i64) * max(i64) fits in i128
        let result = (self.0 as i128) * (shares as i128);
        Money::from_raw(result as i64)
    }

    /// Multiply price by quantity (same as shares but clearer naming).
    #[inline]
    #[must_use]
    pub fn mul_qty(self, qty: i64) -> Money {
        self.mul_shares(qty)
    }

    /// Check if price is zero.
    #[inline]
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Check if price is positive.
    #[inline]
    #[must_use]
    pub const fn is_positive(self) -> bool {
        self.0 > 0
    }

    /// Check if price is negative.
    #[inline]
    #[must_use]
    pub const fn is_negative(self) -> bool {
        self.0 < 0
    }

    /// Absolute value.
    #[inline]
    #[must_use]
    pub const fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// Saturating addition.
    #[inline]
    #[must_use]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }

    /// Saturating subtraction.
    #[inline]
    #[must_use]
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }
}

impl Add for Price {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for Price {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub for Price {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl SubAssign for Price {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl Neg for Price {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl fmt::Debug for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Price({:.6})", self.to_f64())
    }
}

impl fmt::Display for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6}", self.to_f64())
    }
}

// =============================================================================
// MONEY
// =============================================================================

/// Fixed-point money with 6 decimal places.
///
/// Used for cash, equity, PnL, and other monetary values.
///
/// # Examples
///
/// ```
/// use backtester_core::fixed::Money;
///
/// let cash = Money::from_f64(1_000_000.0);
/// let cost = Money::from_f64(5000.50);
/// let remaining = cash - cost;
/// assert_eq!(remaining.to_f64(), 994999.5);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct Money(i64);

impl Money {
    /// Zero money.
    pub const ZERO: Self = Self(0);

    /// One unit (1.000000).
    pub const ONE: Self = Self(SCALE_6);

    /// Maximum representable money (~9.2 trillion).
    pub const MAX: Self = Self(i64::MAX);

    /// Minimum representable money (~-9.2 trillion).
    pub const MIN: Self = Self(i64::MIN);

    /// Create from raw scaled value.
    #[inline]
    #[must_use]
    pub const fn from_raw(raw: i64) -> Self {
        Self(raw)
    }

    /// Get raw scaled value.
    #[inline]
    #[must_use]
    pub const fn raw(self) -> i64 {
        self.0
    }

    /// Create from f64 (rounds to 6 decimal places).
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_f64(value: f64) -> Self {
        Self((value * SCALE_6 as f64).round() as i64)
    }

    /// Convert to f64.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / SCALE_6 as f64
    }

    /// Create from integer (no scaling needed for whole numbers).
    #[inline]
    #[must_use]
    pub const fn from_int(value: i64) -> Self {
        Self(value * SCALE_6)
    }

    /// Divide money by money to get ratio (f64).
    ///
    /// Used for calculating returns, drawdowns, etc.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn div_money(self, rhs: Self) -> f64 {
        if rhs.0 == 0 {
            return 0.0;
        }
        self.0 as f64 / rhs.0 as f64
    }

    /// Divide money by price to get quantity (i64).
    ///
    /// Returns the number of whole shares that can be bought.
    ///
    /// # Math
    ///
    /// Money is stored as `money * SCALE_6`.
    /// Price is stored as `price * SCALE_6`.
    /// So: `money_scaled / price_scaled = quantity` directly.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn div_price(self, price: Price) -> i64 {
        if price.0 == 0 {
            return 0;
        }
        // money_scaled / price_scaled = quantity (scales cancel out)
        (self.0 / price.0) as i64
    }

    /// Divide money by quantity to get average price.
    ///
    /// Used for cost averaging calculations.
    ///
    /// # Math
    ///
    /// Money is stored as `money * SCALE_6`.
    /// Price should be `price * SCALE_6`.
    /// So: `money_scaled / qty = price_scaled` directly.
    #[inline]
    #[must_use]
    pub fn div_qty(self, qty: i64) -> Price {
        if qty == 0 {
            return Price::ZERO;
        }
        Price::from_raw(self.0 / qty)
    }

    /// Check if money is zero.
    #[inline]
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Check if money is positive.
    #[inline]
    #[must_use]
    pub const fn is_positive(self) -> bool {
        self.0 > 0
    }

    /// Check if money is negative.
    #[inline]
    #[must_use]
    pub const fn is_negative(self) -> bool {
        self.0 < 0
    }

    /// Absolute value.
    #[inline]
    #[must_use]
    pub const fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// Saturating addition.
    #[inline]
    #[must_use]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }

    /// Saturating subtraction.
    #[inline]
    #[must_use]
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }

    /// Multiply money by a rate to get money.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn mul_rate(self, rate: Rate) -> Self {
        let result = (self.0 as i128) * (rate.0 as i128) / (SCALE_8 as i128);
        Self(result as i64)
    }

    /// Multiply money by f64 factor (for percentage-based calculations).
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn mul_f64(self, factor: f64) -> Self {
        Self((self.0 as f64 * factor).round() as i64)
    }
}

impl Add for Money {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for Money {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub for Money {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl SubAssign for Money {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl Neg for Money {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl Sum for Money {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<'a> Sum<&'a Money> for Money {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + *x)
    }
}

impl fmt::Debug for Money {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Money({:.6})", self.to_f64())
    }
}

impl fmt::Display for Money {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}", self.to_f64())
    }
}

// =============================================================================
// RATE
// =============================================================================

/// Fixed-point rate with 8 decimal places.
///
/// Used for dividend rates, interest rates, and other rates requiring higher precision.
///
/// # Examples
///
/// ```
/// use backtester_core::fixed::{Rate, Money};
///
/// let dividend_rate = Rate::from_f64(0.05); // 5% dividend
/// let shares = 1000;
/// let cashflow = dividend_rate.mul_shares(shares);
/// assert_eq!(cashflow.to_f64(), 50.0);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct Rate(i64);

impl Rate {
    /// Zero rate.
    pub const ZERO: Self = Self(0);

    /// One unit (1.00000000).
    pub const ONE: Self = Self(SCALE_8);

    /// Create from raw scaled value.
    #[inline]
    #[must_use]
    pub const fn from_raw(raw: i64) -> Self {
        Self(raw)
    }

    /// Get raw scaled value.
    #[inline]
    #[must_use]
    pub const fn raw(self) -> i64 {
        self.0
    }

    /// Create from f64 (rounds to 8 decimal places).
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_f64(value: f64) -> Self {
        Self((value * SCALE_8 as f64).round() as i64)
    }

    /// Convert to f64.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / SCALE_8 as f64
    }

    /// Multiply rate by shares to get Money (for dividend calculations).
    ///
    /// rate * shares = money (scaled to SCALE_6)
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn mul_shares(self, shares: i64) -> Money {
        // rate (scale 1e8) * shares -> money (scale 1e6)
        // Need to convert from scale 1e8 to scale 1e6
        let result = (self.0 as i128) * (shares as i128) / (SCALE_8 as i128 / SCALE_6 as i128);
        Money::from_raw(result as i64)
    }

    /// Check if rate is zero.
    #[inline]
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Check if rate is positive.
    #[inline]
    #[must_use]
    pub const fn is_positive(self) -> bool {
        self.0 > 0
    }
}

impl Add for Rate {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Rate {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl fmt::Debug for Rate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rate({:.8})", self.to_f64())
    }
}

impl fmt::Display for Rate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6}", self.to_f64())
    }
}

// =============================================================================
// CONVERSIONS BETWEEN TYPES
// =============================================================================

impl From<Price> for Money {
    /// Convert Price to Money (same scale, just different semantic meaning).
    #[inline]
    fn from(price: Price) -> Self {
        Self(price.0)
    }
}

impl From<Money> for Price {
    /// Convert Money to Price (same scale, just different semantic meaning).
    #[inline]
    fn from(money: Money) -> Self {
        Self(money.0)
    }
}

// =============================================================================
// DECIMAL CONVERSIONS (for API boundaries - NOT hot path)
// =============================================================================

#[cfg(feature = "decimal")]
mod decimal_conversions {
    use super::*;
    use rust_decimal::prelude::ToPrimitive;
    use rust_decimal::Decimal;

    impl From<Decimal> for Price {
        fn from(d: Decimal) -> Self {
            let scaled = d * Decimal::from(SCALE_6);
            Self(scaled.to_i64().unwrap_or(0))
        }
    }

    impl From<Price> for Decimal {
        fn from(p: Price) -> Self {
            Decimal::from(p.0) / Decimal::from(SCALE_6)
        }
    }

    impl From<Decimal> for Money {
        fn from(d: Decimal) -> Self {
            let scaled = d * Decimal::from(SCALE_6);
            Self(scaled.to_i64().unwrap_or(0))
        }
    }

    impl From<Money> for Decimal {
        fn from(m: Money) -> Self {
            Decimal::from(m.0) / Decimal::from(SCALE_6)
        }
    }

    impl From<Decimal> for Rate {
        fn from(d: Decimal) -> Self {
            let scaled = d * Decimal::from(SCALE_8);
            Self(scaled.to_i64().unwrap_or(0))
        }
    }

    impl From<Rate> for Decimal {
        fn from(r: Rate) -> Self {
            Decimal::from(r.0) / Decimal::from(SCALE_8)
        }
    }

    impl Price {
        /// Create from Decimal.
        #[must_use]
        pub fn from_decimal(d: Decimal) -> Self {
            Self::from(d)
        }

        /// Convert to Decimal.
        #[must_use]
        pub fn to_decimal(self) -> Decimal {
            Decimal::from(self)
        }
    }

    impl Money {
        /// Create from Decimal.
        #[must_use]
        pub fn from_decimal(d: Decimal) -> Self {
            Self::from(d)
        }

        /// Convert to Decimal.
        #[must_use]
        pub fn to_decimal(self) -> Decimal {
            Decimal::from(self)
        }
    }

    impl Rate {
        /// Create from Decimal.
        #[must_use]
        pub fn from_decimal(d: Decimal) -> Self {
            Self::from(d)
        }

        /// Convert to Decimal.
        #[must_use]
        pub fn to_decimal(self) -> Decimal {
            Decimal::from(self)
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn price_from_f64() {
        let p = Price::from_f64(123.456789);
        assert_eq!(p.raw(), 123_456_789);
        assert!((p.to_f64() - 123.456789).abs() < 1e-6);
    }

    #[test]
    fn price_from_int() {
        let p = Price::from_int(100);
        assert_eq!(p.raw(), 100_000_000);
        assert_eq!(p.to_f64(), 100.0);
    }

    #[test]
    fn price_arithmetic() {
        let a = Price::from_f64(10.5);
        let b = Price::from_f64(3.25);
        
        let sum = a + b;
        assert!((sum.to_f64() - 13.75).abs() < 1e-6);
        
        let diff = a - b;
        assert!((diff.to_f64() - 7.25).abs() < 1e-6);
    }

    #[test]
    fn price_mul_shares() {
        let price = Price::from_f64(50.0);
        let shares = 100;
        let value = price.mul_shares(shares);
        assert_eq!(value.to_f64(), 5000.0);
    }

    #[test]
    fn price_mul_shares_fractional() {
        let price = Price::from_f64(123.456789);
        let shares = 1000;
        let value = price.mul_shares(shares);
        // 123.456789 * 1000 = 123456.789
        assert!((value.to_f64() - 123456.789).abs() < 0.001);
    }

    #[test]
    fn money_arithmetic() {
        let cash = Money::from_f64(1_000_000.0);
        let cost = Money::from_f64(5000.50);
        let remaining = cash - cost;
        assert!((remaining.to_f64() - 994999.5).abs() < 0.001);
    }

    #[test]
    fn money_sum() {
        let values = vec![
            Money::from_f64(100.0),
            Money::from_f64(200.0),
            Money::from_f64(300.0),
        ];
        let total: Money = values.into_iter().sum();
        assert_eq!(total.to_f64(), 600.0);
    }

    #[test]
    fn money_div_money() {
        let equity = Money::from_f64(900_000.0);
        let peak = Money::from_f64(1_000_000.0);
        let ratio = equity.div_money(peak);
        assert!((ratio - 0.9).abs() < 1e-10);
    }

    #[test]
    fn money_div_price() {
        let cash = Money::from_f64(10_000.0);
        let price = Price::from_f64(50.0);
        let shares = cash.div_price(price);
        assert_eq!(shares, 200);
    }

    #[test]
    fn money_div_qty() {
        // Total cost = 5000.0 for 100 shares -> avg price = 50.0
        let total_cost = Money::from_f64(5000.0);
        let shares = 100;
        let avg_price = total_cost.div_qty(shares);
        assert_eq!(avg_price.to_f64(), 50.0);
    }

    #[test]
    fn money_div_qty_fractional() {
        // Total cost = 12345.67 for 100 shares -> avg price = 123.4567
        let total_cost = Money::from_f64(12345.67);
        let shares = 100;
        let avg_price = total_cost.div_qty(shares);
        assert!((avg_price.to_f64() - 123.4567).abs() < 0.0001);
    }

    #[test]
    fn rate_mul_shares() {
        let rate = Rate::from_f64(0.50); // R$0.50 per share
        let shares = 1000;
        let cashflow = rate.mul_shares(shares);
        assert!((cashflow.to_f64() - 500.0).abs() < 0.01);
    }

    #[test]
    fn rate_small_dividend() {
        let rate = Rate::from_f64(0.05); // R$0.05 per share
        let shares = 100;
        let cashflow = rate.mul_shares(shares);
        assert!((cashflow.to_f64() - 5.0).abs() < 0.001);
    }

    #[test]
    fn overflow_prevention_large_price() {
        // Test with large values that would overflow i64 without i128 intermediate
        let price = Price::from_f64(1_000_000.0); // 1M per share
        let shares = 1_000_000; // 1M shares
        let value = price.mul_shares(shares);
        // 1M * 1M = 1 trillion
        assert!((value.to_f64() - 1_000_000_000_000.0).abs() < 1.0);
    }

    #[test]
    fn determinism() {
        // Same inputs must produce exact same outputs
        let price1 = Price::from_f64(123.456789);
        let price2 = Price::from_f64(123.456789);
        assert_eq!(price1.raw(), price2.raw());

        let value1 = price1.mul_shares(1000);
        let value2 = price2.mul_shares(1000);
        assert_eq!(value1.raw(), value2.raw());
    }

    #[test]
    fn price_money_conversion() {
        let price = Price::from_f64(100.0);
        let money: Money = price.into();
        assert_eq!(money.to_f64(), 100.0);

        let back: Price = money.into();
        assert_eq!(back.raw(), price.raw());
    }

    #[test]
    fn comparisons() {
        let a = Money::from_f64(100.0);
        let b = Money::from_f64(200.0);
        assert!(a < b);
        assert!(b > a);
        assert!(a != b);

        let c = Money::from_f64(100.0);
        assert_eq!(a, c);
    }

    #[test]
    fn zero_checks() {
        assert!(Price::ZERO.is_zero());
        assert!(!Price::from_f64(1.0).is_zero());
        
        assert!(Money::ZERO.is_zero());
        assert!(Money::from_f64(-100.0).is_negative());
        assert!(Money::from_f64(100.0).is_positive());
    }

    #[test]
    fn saturating_ops() {
        let big = Money::from_raw(i64::MAX - 1000);
        let small = Money::from_raw(2000);
        let result = big.saturating_add(small);
        assert_eq!(result.raw(), i64::MAX);
    }

    #[test]
    fn crypto_precision() {
        // Test micro-precision for crypto (6 decimal places)
        let btc_price = Price::from_f64(0.000001);
        assert_eq!(btc_price.raw(), 1);
        assert!((btc_price.to_f64() - 0.000001).abs() < 1e-10);
    }
}

