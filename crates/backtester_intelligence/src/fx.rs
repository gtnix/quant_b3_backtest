//! FX Rate Provider and Currency Conversion Module.
//!
//! # Overview
//!
//! This module provides point-in-time FX rate lookups and currency conversion
//! utilities for multi-currency portfolio management.
//!
//! # Design Principles
//!
//! 1. **Point-in-time semantics**: Rates are looked up by date, using the
//!    most recent available rate (LOCF) within a configurable gap.
//! 2. **Explicit conversion**: No implicit currency conversion - all conversions
//!    go through `convert_money()`.
//! 3. **Audit trail**: Conversions return metadata about the rate used.
//! 4. **Inverse pair support**: If USD/BRL is stored, BRL/USD conversions work
//!    automatically via inversion.
//!
//! # Example
//!
//! ```ignore
//! use backtester_intelligence::fx::{InMemoryFxProvider, convert_money, FxRateProvider};
//! use backtester_intelligence::currency::{Currency, Money, FxPair};
//! use rust_decimal_macros::dec;
//!
//! // Create provider with USD/BRL rates
//! let mut provider = InMemoryFxProvider::new();
//! provider.add_rate(FxPair::USD_BRL, date, dec!(5.50));
//!
//! // Convert 1000 USD to BRL
//! let usd = Money::new(dec!(1000), Currency::USD);
//! let brl = convert_money(&usd, Currency::BRL, date, &provider, 5)?;
//! assert_eq!(brl.amount(), dec!(5500));
//! ```

use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use crate::currency::{Currency, FxPair, FxRate, Money};
use crate::performance::FxResolutionMethod;

// =============================================================================
// ERROR TYPES
// =============================================================================

/// Errors that can occur during FX operations.
#[derive(Debug, Clone)]
pub enum FxError {
    /// No rate available for the requested pair and date.
    RateNotFound {
        pair: FxPair,
        date: NaiveDate,
        max_gap_days: u32,
    },
    
    /// Rate gap exceeds maximum allowed.
    GapExceedsLimit {
        pair: FxPair,
        requested_date: NaiveDate,
        last_available_date: NaiveDate,
        gap_days: i64,
        max_gap_days: u32,
    },
    
    /// Cannot find a conversion path between currencies.
    NoConversionPath {
        from: Currency,
        to: Currency,
    },
    
    /// Attempted to use identity rate for different currencies.
    InvalidIdentityConversion {
        from: Currency,
        to: Currency,
    },
    
    /// Division by zero in rate calculation.
    ZeroRate {
        pair: FxPair,
        date: NaiveDate,
    },
}

impl fmt::Display for FxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FxError::RateNotFound { pair, date, max_gap_days } => {
                write!(
                    f,
                    "No FX rate found for {} on {} (searched {} days back)",
                    pair, date, max_gap_days
                )
            }
            FxError::GapExceedsLimit {
                pair,
                requested_date,
                last_available_date,
                gap_days,
                max_gap_days,
            } => {
                write!(
                    f,
                    "FX rate gap for {} exceeds limit: requested {}, last available {} ({} days > {} max)",
                    pair, requested_date, last_available_date, gap_days, max_gap_days
                )
            }
            FxError::NoConversionPath { from, to } => {
                write!(f, "No conversion path from {} to {}", from, to)
            }
            FxError::InvalidIdentityConversion { from, to } => {
                write!(
                    f,
                    "Cannot use identity conversion for different currencies: {} -> {}",
                    from, to
                )
            }
            FxError::ZeroRate { pair, date } => {
                write!(f, "Zero rate encountered for {} on {}", pair, date)
            }
        }
    }
}

impl std::error::Error for FxError {}

// =============================================================================
// FX RATE PROVIDER TRAIT
// =============================================================================

/// Trait for FX rate providers.
///
/// Implementations must be thread-safe (Send + Sync) to support
/// parallel backtest execution.
pub trait FxRateProvider: Send + Sync {
    /// Get the FX rate for a specific pair and date.
    ///
    /// Returns the exact rate if available, or an error if not.
    fn get_rate(&self, date: NaiveDate, pair: FxPair) -> Result<FxRate, FxError>;
    
    /// Get the FX rate using Last Observation Carried Forward (LOCF).
    ///
    /// If no rate is available for the exact date, uses the most recent
    /// rate available, up to `max_gap_days` in the past.
    fn get_rate_locf(
        &self,
        date: NaiveDate,
        pair: FxPair,
        max_gap_days: u32,
    ) -> Result<FxRate, FxError>;
    
    /// Check if a rate is available for the given pair and date.
    fn has_rate(&self, date: NaiveDate, pair: FxPair) -> bool {
        self.get_rate(date, pair).is_ok()
    }
    
    /// Get the date range covered by this provider for a pair.
    fn date_range(&self, pair: FxPair) -> Option<(NaiveDate, NaiveDate)>;
    
    /// List all available pairs.
    fn available_pairs(&self) -> Vec<FxPair>;
}

// =============================================================================
// IN-MEMORY FX PROVIDER
// =============================================================================

/// In-memory FX rate provider backed by BTreeMaps.
///
/// Supports:
/// - Point-in-time lookups (last rate <= date)
/// - LOCF with configurable gap limits
/// - Automatic inverse pair calculation
#[derive(Debug, Clone, Default)]
pub struct InMemoryFxProvider {
    /// Rates indexed by pair, then by date.
    /// BTreeMap ensures O(log n) lookups and range queries.
    rates: BTreeMap<FxPair, BTreeMap<NaiveDate, Decimal>>,
}

impl InMemoryFxProvider {
    /// Create a new empty provider.
    pub fn new() -> Self {
        Self {
            rates: BTreeMap::new(),
        }
    }
    
    /// Add a rate for a specific pair and date.
    pub fn add_rate(&mut self, pair: FxPair, date: NaiveDate, rate: Decimal) {
        self.rates
            .entry(pair)
            .or_default()
            .insert(date, rate);
    }
    
    /// Add multiple rates for a pair.
    pub fn add_rates(&mut self, pair: FxPair, rates: impl IntoIterator<Item = (NaiveDate, Decimal)>) {
        let entry = self.rates.entry(pair).or_default();
        for (date, rate) in rates {
            entry.insert(date, rate);
        }
    }
    
    /// Load rates from a vector of (date, rate) tuples.
    pub fn from_data(pair: FxPair, data: Vec<(NaiveDate, Decimal)>) -> Self {
        let mut provider = Self::new();
        provider.add_rates(pair, data);
        provider
    }
    
    /// Get the number of rates stored for a pair.
    pub fn rate_count(&self, pair: FxPair) -> usize {
        self.rates.get(&pair).map(|m| m.len()).unwrap_or(0)
    }
    
    /// Get total number of rates across all pairs.
    pub fn total_rate_count(&self) -> usize {
        self.rates.values().map(|m| m.len()).sum()
    }
    
    /// Try to find a rate for the pair or its inverse.
    fn find_rate_or_inverse(
        &self,
        date: NaiveDate,
        pair: FxPair,
    ) -> Option<(FxRate, bool)> {
        // Try direct pair first
        if let Some(rates) = self.rates.get(&pair) {
            if let Some((&rate_date, &rate)) = rates.range(..=date).next_back() {
                return Some((FxRate::new(pair, rate, rate_date), false));
            }
        }
        
        // Try inverse pair
        let inverse = pair.inverse();
        if let Some(rates) = self.rates.get(&inverse) {
            if let Some((&rate_date, &rate)) = rates.range(..=date).next_back() {
                if !rate.is_zero() {
                    let inverse_rate = Decimal::ONE / rate;
                    return Some((FxRate::new(pair, inverse_rate, rate_date), true));
                }
            }
        }
        
        None
    }
}

impl FxRateProvider for InMemoryFxProvider {
    fn get_rate(&self, date: NaiveDate, pair: FxPair) -> Result<FxRate, FxError> {
        // Identity pair always returns 1.0
        if pair.is_identity() {
            return Ok(FxRate::identity(pair.base, date));
        }
        
        // Try to find exact date match
        if let Some(rates) = self.rates.get(&pair) {
            if let Some(&rate) = rates.get(&date) {
                return Ok(FxRate::new(pair, rate, date));
            }
        }
        
        // Try inverse
        let inverse = pair.inverse();
        if let Some(rates) = self.rates.get(&inverse) {
            if let Some(&rate) = rates.get(&date) {
                if rate.is_zero() {
                    return Err(FxError::ZeroRate { pair: inverse, date });
                }
                return Ok(FxRate::new(pair, Decimal::ONE / rate, date));
            }
        }
        
        Err(FxError::RateNotFound {
            pair,
            date,
            max_gap_days: 0,
        })
    }
    
    fn get_rate_locf(
        &self,
        date: NaiveDate,
        pair: FxPair,
        max_gap_days: u32,
    ) -> Result<FxRate, FxError> {
        // Identity pair always returns 1.0
        if pair.is_identity() {
            return Ok(FxRate::identity(pair.base, date));
        }
        
        // Try to find rate (direct or inverse)
        if let Some((rate, _is_inverse)) = self.find_rate_or_inverse(date, pair) {
            // Check gap
            let gap = (date - rate.date).num_days();
            if gap < 0 {
                // This shouldn't happen with our range query, but defensive
                return Err(FxError::RateNotFound {
                    pair,
                    date,
                    max_gap_days,
                });
            }
            
            if gap as u32 > max_gap_days {
                return Err(FxError::GapExceedsLimit {
                    pair,
                    requested_date: date,
                    last_available_date: rate.date,
                    gap_days: gap,
                    max_gap_days,
                });
            }
            
            // Return rate with original pair but actual rate date
            Ok(FxRate::new(pair, rate.rate, rate.date))
        } else {
            Err(FxError::RateNotFound {
                pair,
                date,
                max_gap_days,
            })
        }
    }
    
    fn date_range(&self, pair: FxPair) -> Option<(NaiveDate, NaiveDate)> {
        // Check direct pair
        if let Some(rates) = self.rates.get(&pair) {
            if !rates.is_empty() {
                let min = *rates.keys().next()?;
                let max = *rates.keys().next_back()?;
                return Some((min, max));
            }
        }
        
        // Check inverse pair
        let inverse = pair.inverse();
        if let Some(rates) = self.rates.get(&inverse) {
            if !rates.is_empty() {
                let min = *rates.keys().next()?;
                let max = *rates.keys().next_back()?;
                return Some((min, max));
            }
        }
        
        None
    }
    
    fn available_pairs(&self) -> Vec<FxPair> {
        self.rates.keys().copied().collect()
    }
}

// =============================================================================
// CONVERSION UTILITIES
// =============================================================================

/// Result of a currency conversion with full audit information.
///
/// Tracks complete details of how the conversion was performed,
/// enabling audit trail and debugging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionResult {
    /// The converted money amount.
    pub money: Money,
    
    // Request details
    /// The currency pair that was requested for conversion.
    pub pair_requested: String,
    /// The date for which the rate was requested.
    pub date_requested: NaiveDate,
    
    // Resolution details
    /// The currency pair that was actually used (may differ if inverse).
    pub pair_resolved: String,
    /// The date of the actual rate observation (may differ if LOCF).
    pub date_resolved: NaiveDate,
    /// The FX rate used for conversion.
    pub rate_used: Decimal,
    /// Method used to resolve the rate.
    pub method: FxResolutionMethod,
    
    // Legacy fields (for backward compatibility)
    /// The date of the rate (alias for date_resolved).
    #[serde(skip_serializing)]
    pub rate_date: NaiveDate,
    /// The FX pair used (alias for pair_resolved).
    #[serde(skip_serializing)]
    pub pair_used: String,
    /// Whether the inverse rate was used (derived from method).
    #[serde(skip_serializing)]
    pub used_inverse: bool,
}

impl ConversionResult {
    /// Check if LOCF was used to obtain the rate.
    pub fn is_locf(&self) -> bool {
        self.method.used_locf()
    }
    
    /// Get the gap in days between requested and resolved dates.
    pub fn gap_days(&self) -> i64 {
        (self.date_requested - self.date_resolved).num_days()
    }
    
    /// Get a human-readable description of the rate used.
    pub fn rate_description(&self) -> String {
        format!(
            "1 {} = {} {} ({})",
            self.pair_resolved.split('/').next().unwrap_or("?"),
            self.rate_used,
            self.pair_resolved.split('/').last().unwrap_or("?"),
            self.method
        )
    }
}

/// Convert money from one currency to another.
///
/// # Arguments
///
/// * `money` - The amount to convert
/// * `to` - Target currency
/// * `date` - Date for rate lookup
/// * `provider` - FX rate provider
/// * `max_gap_days` - Maximum gap for LOCF (0 = exact date only)
///
/// # Returns
///
/// The converted Money value, or an error if conversion fails.
///
/// # Example
///
/// ```ignore
/// let usd = Money::new(dec!(1000), Currency::USD);
/// let brl = convert_money(&usd, Currency::BRL, date, &provider, 5)?;
/// ```
pub fn convert_money(
    money: &Money,
    to: Currency,
    date: NaiveDate,
    provider: &dyn FxRateProvider,
    max_gap_days: u32,
) -> Result<Money, FxError> {
    convert_money_with_audit(money, to, date, provider, max_gap_days)
        .map(|r| r.money)
}

/// Convert money with full audit trail.
///
/// Same as `convert_money` but returns additional metadata about
/// the conversion for audit purposes.
pub fn convert_money_with_audit(
    money: &Money,
    to: Currency,
    date: NaiveDate,
    provider: &dyn FxRateProvider,
    max_gap_days: u32,
) -> Result<ConversionResult, FxError> {
    let from = money.currency();
    let pair_requested = format!("{}/{}", from, to);
    
    // Same currency - no conversion needed
    if from == to {
        return Ok(ConversionResult {
            money: *money,
            pair_requested: pair_requested.clone(),
            date_requested: date,
            pair_resolved: pair_requested.clone(),
            date_resolved: date,
            rate_used: Decimal::ONE,
            method: FxResolutionMethod::Identity,
            // Legacy
            rate_date: date,
            pair_used: pair_requested,
            used_inverse: false,
        });
    }
    
    // Get the rate for conversion
    // Convention: we need from -> to, which means pair(from, to)
    // where rate means "1 from = rate to"
    let pair = FxPair::new(from, to);
    
    let rate = if max_gap_days == 0 {
        provider.get_rate(date, pair)?
    } else {
        provider.get_rate_locf(date, pair, max_gap_days)?
    };
    
    if rate.rate.is_zero() {
        return Err(FxError::ZeroRate { pair, date: rate.date });
    }
    
    // Determine resolution method
    let used_inverse = rate.pair != pair;
    let used_locf = rate.date != date;
    let method = match (used_inverse, used_locf) {
        (false, false) => FxResolutionMethod::Direct,
        (true, false) => FxResolutionMethod::Inverse,
        (false, true) => FxResolutionMethod::LOCF,
        (true, true) => FxResolutionMethod::InverseLOCF,
    };
    
    // Convert: amount_to = amount_from * rate
    let converted_amount = money.amount() * rate.rate;
    let converted = Money::new(converted_amount, to);
    
    let pair_resolved = pair.as_str();
    
    Ok(ConversionResult {
        money: converted,
        pair_requested: pair_requested.clone(),
        date_requested: date,
        pair_resolved: pair_resolved.clone(),
        date_resolved: rate.date,
        rate_used: rate.rate,
        method,
        // Legacy
        rate_date: rate.date,
        pair_used: pair_resolved,
        used_inverse,
    })
}

/// Convert a Decimal amount between currencies.
///
/// Convenience function when you don't need Money type safety.
pub fn convert_amount(
    amount: Decimal,
    from: Currency,
    to: Currency,
    date: NaiveDate,
    provider: &dyn FxRateProvider,
    max_gap_days: u32,
) -> Result<Decimal, FxError> {
    let money = Money::new(amount, from);
    convert_money(&money, to, date, provider, max_gap_days)
        .map(|m| m.amount())
}

// =============================================================================
// ARC WRAPPER FOR SHARING
// =============================================================================

/// Thread-safe wrapper for FxRateProvider.
pub type SharedFxProvider = Arc<dyn FxRateProvider>;

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn make_date(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).unwrap()
    }

    // -------------------------------------------------------------------------
    // InMemoryFxProvider Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_provider_add_and_get_rate() {
        let mut provider = InMemoryFxProvider::new();
        let date = make_date(2024, 12, 27);
        
        provider.add_rate(FxPair::USD_BRL, date, dec!(5.50));
        
        let rate = provider.get_rate(date, FxPair::USD_BRL).unwrap();
        assert_eq!(rate.rate, dec!(5.50));
        assert_eq!(rate.date, date);
    }

    #[test]
    fn test_provider_identity_rate() {
        let provider = InMemoryFxProvider::new();
        let date = make_date(2024, 12, 27);
        let pair = FxPair::new(Currency::USD, Currency::USD);
        
        let rate = provider.get_rate(date, pair).unwrap();
        assert_eq!(rate.rate, Decimal::ONE);
    }

    #[test]
    fn test_provider_inverse_rate() {
        let mut provider = InMemoryFxProvider::new();
        let date = make_date(2024, 12, 27);
        
        // Store USD/BRL
        provider.add_rate(FxPair::USD_BRL, date, dec!(5));
        
        // Query BRL/USD (inverse)
        let pair = FxPair::new(Currency::BRL, Currency::USD);
        let rate = provider.get_rate(date, pair).unwrap();
        
        // 1/5 = 0.2
        assert_eq!(rate.rate, dec!(0.2));
    }

    #[test]
    fn test_provider_rate_not_found() {
        let provider = InMemoryFxProvider::new();
        let date = make_date(2024, 12, 27);
        
        let result = provider.get_rate(date, FxPair::USD_BRL);
        assert!(matches!(result, Err(FxError::RateNotFound { .. })));
    }

    #[test]
    fn test_provider_locf_exact_date() {
        let mut provider = InMemoryFxProvider::new();
        let date = make_date(2024, 12, 27);
        
        provider.add_rate(FxPair::USD_BRL, date, dec!(5.50));
        
        let rate = provider.get_rate_locf(date, FxPair::USD_BRL, 5).unwrap();
        assert_eq!(rate.rate, dec!(5.50));
        assert_eq!(rate.date, date);
    }

    #[test]
    fn test_provider_locf_gap_within_limit() {
        let mut provider = InMemoryFxProvider::new();
        let rate_date = make_date(2024, 12, 23);
        let query_date = make_date(2024, 12, 27); // 4 days later
        
        provider.add_rate(FxPair::USD_BRL, rate_date, dec!(5.50));
        
        // Max gap of 5 days - should succeed
        let rate = provider.get_rate_locf(query_date, FxPair::USD_BRL, 5).unwrap();
        assert_eq!(rate.rate, dec!(5.50));
        assert_eq!(rate.date, rate_date); // Date of actual rate
    }

    #[test]
    fn test_provider_locf_gap_exceeds_limit() {
        let mut provider = InMemoryFxProvider::new();
        let rate_date = make_date(2024, 12, 20);
        let query_date = make_date(2024, 12, 27); // 7 days later
        
        provider.add_rate(FxPair::USD_BRL, rate_date, dec!(5.50));
        
        // Max gap of 5 days - should fail
        let result = provider.get_rate_locf(query_date, FxPair::USD_BRL, 5);
        assert!(matches!(result, Err(FxError::GapExceedsLimit { gap_days: 7, .. })));
    }

    #[test]
    fn test_provider_locf_uses_most_recent() {
        let mut provider = InMemoryFxProvider::new();
        
        provider.add_rate(FxPair::USD_BRL, make_date(2024, 12, 20), dec!(5.40));
        provider.add_rate(FxPair::USD_BRL, make_date(2024, 12, 24), dec!(5.50));
        provider.add_rate(FxPair::USD_BRL, make_date(2024, 12, 26), dec!(5.55));
        
        // Query for Dec 27 - should get Dec 26 rate
        let rate = provider.get_rate_locf(make_date(2024, 12, 27), FxPair::USD_BRL, 5).unwrap();
        assert_eq!(rate.rate, dec!(5.55));
        assert_eq!(rate.date, make_date(2024, 12, 26));
    }

    #[test]
    fn test_provider_date_range() {
        let mut provider = InMemoryFxProvider::new();
        
        provider.add_rate(FxPair::USD_BRL, make_date(2024, 1, 1), dec!(5.00));
        provider.add_rate(FxPair::USD_BRL, make_date(2024, 6, 15), dec!(5.25));
        provider.add_rate(FxPair::USD_BRL, make_date(2024, 12, 31), dec!(5.50));
        
        let (min, max) = provider.date_range(FxPair::USD_BRL).unwrap();
        assert_eq!(min, make_date(2024, 1, 1));
        assert_eq!(max, make_date(2024, 12, 31));
    }

    #[test]
    fn test_provider_available_pairs() {
        let mut provider = InMemoryFxProvider::new();
        
        provider.add_rate(FxPair::USD_BRL, make_date(2024, 12, 27), dec!(5.50));
        provider.add_rate(FxPair::EUR_USD, make_date(2024, 12, 27), dec!(1.10));
        
        let pairs = provider.available_pairs();
        assert_eq!(pairs.len(), 2);
        assert!(pairs.contains(&FxPair::USD_BRL));
        assert!(pairs.contains(&FxPair::EUR_USD));
    }

    // -------------------------------------------------------------------------
    // Conversion Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_convert_same_currency() {
        let provider = InMemoryFxProvider::new();
        let date = make_date(2024, 12, 27);
        let usd = Money::new(dec!(1000), Currency::USD);
        
        let result = convert_money(&usd, Currency::USD, date, &provider, 0).unwrap();
        assert_eq!(result.amount(), dec!(1000));
        assert_eq!(result.currency(), Currency::USD);
    }

    #[test]
    fn test_convert_usd_to_brl() {
        let mut provider = InMemoryFxProvider::new();
        let date = make_date(2024, 12, 27);
        
        provider.add_rate(FxPair::USD_BRL, date, dec!(5.50));
        
        let usd = Money::new(dec!(1000), Currency::USD);
        let brl = convert_money(&usd, Currency::BRL, date, &provider, 0).unwrap();
        
        // 1000 USD * 5.50 = 5500 BRL
        assert_eq!(brl.amount(), dec!(5500));
        assert_eq!(brl.currency(), Currency::BRL);
    }

    #[test]
    fn test_convert_brl_to_usd_via_inverse() {
        let mut provider = InMemoryFxProvider::new();
        let date = make_date(2024, 12, 27);
        
        // Only store USD/BRL
        provider.add_rate(FxPair::USD_BRL, date, dec!(5));
        
        let brl = Money::new(dec!(5000), Currency::BRL);
        let usd = convert_money(&brl, Currency::USD, date, &provider, 0).unwrap();
        
        // 5000 BRL / 5 = 1000 USD
        assert_eq!(usd.amount(), dec!(1000));
        assert_eq!(usd.currency(), Currency::USD);
    }

    #[test]
    fn test_convert_with_audit() {
        let mut provider = InMemoryFxProvider::new();
        let rate_date = make_date(2024, 12, 24);
        let query_date = make_date(2024, 12, 27);
        
        provider.add_rate(FxPair::USD_BRL, rate_date, dec!(5.50));
        
        let usd = Money::new(dec!(100), Currency::USD);
        let result = convert_money_with_audit(
            &usd,
            Currency::BRL,
            query_date,
            &provider,
            5, // LOCF with 5 day gap
        ).unwrap();
        
        assert_eq!(result.money.amount(), dec!(550));
        assert_eq!(result.rate_used, dec!(5.50));
        assert_eq!(result.rate_date, rate_date);
        assert_eq!(result.pair_used, "USD/BRL");
        assert!(!result.used_inverse);
    }

    #[test]
    fn test_convert_missing_rate() {
        let provider = InMemoryFxProvider::new();
        let date = make_date(2024, 12, 27);
        
        let usd = Money::new(dec!(1000), Currency::USD);
        let result = convert_money(&usd, Currency::BRL, date, &provider, 0);
        
        assert!(matches!(result, Err(FxError::RateNotFound { .. })));
    }

    #[test]
    fn test_convert_amount() {
        let mut provider = InMemoryFxProvider::new();
        let date = make_date(2024, 12, 27);
        
        provider.add_rate(FxPair::USD_BRL, date, dec!(5));
        
        let result = convert_amount(
            dec!(100),
            Currency::USD,
            Currency::BRL,
            date,
            &provider,
            0,
        ).unwrap();
        
        assert_eq!(result, dec!(500));
    }

    // -------------------------------------------------------------------------
    // Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_zero_amount_conversion() {
        let mut provider = InMemoryFxProvider::new();
        let date = make_date(2024, 12, 27);
        
        provider.add_rate(FxPair::USD_BRL, date, dec!(5.50));
        
        let zero = Money::zero(Currency::USD);
        let result = convert_money(&zero, Currency::BRL, date, &provider, 0).unwrap();
        
        assert!(result.is_zero());
        assert_eq!(result.currency(), Currency::BRL);
    }

    #[test]
    fn test_negative_amount_conversion() {
        let mut provider = InMemoryFxProvider::new();
        let date = make_date(2024, 12, 27);
        
        provider.add_rate(FxPair::USD_BRL, date, dec!(5));
        
        let negative = Money::new(dec!(-100), Currency::USD);
        let result = convert_money(&negative, Currency::BRL, date, &provider, 0).unwrap();
        
        assert_eq!(result.amount(), dec!(-500));
    }

    #[test]
    fn test_high_precision_conversion() {
        let mut provider = InMemoryFxProvider::new();
        let date = make_date(2024, 12, 27);
        
        provider.add_rate(FxPair::USD_BRL, date, dec!(5.123456789));
        
        let usd = Money::new(dec!(1000.00), Currency::USD);
        let brl = convert_money(&usd, Currency::BRL, date, &provider, 0).unwrap();
        
        // Should maintain precision
        assert_eq!(brl.amount(), dec!(5123.456789));
    }
}

