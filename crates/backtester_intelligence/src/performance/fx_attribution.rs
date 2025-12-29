//! FX Attribution Module
//!
//! Implements multiplicative return decomposition for multi-currency portfolios.
//!
//! # Formula
//!
//! For a position in local currency L converted to base currency B:
//!
//! ```text
//! Value_B(t) = Value_L(t) * FX(t)   where FX = B per 1 L
//!
//! R_total_B = Value_B(t1) / Value_B(t0) - 1
//! R_asset_L = Value_L(t1) / Value_L(t0) - 1
//! R_fx = FX(t1) / FX(t0) - 1
//!
//! Multiplicative decomposition:
//! (1 + R_total_B) = (1 + R_asset_L) * (1 + R_fx)
//!
//! Additive decomposition (3 terms):
//! R_total_B = R_asset + R_fx + R_interaction
//!           = R_asset_L + R_fx + (R_asset_L * R_fx)
//! ```
//!
//! We report all 3 terms for maximum transparency.
//!
//! # Example
//!
//! ```ignore
//! let attribution = FxAttributionEngine::new(Currency::BRL);
//! let result = attribution.calculate_period_attribution(
//!     start_values,
//!     end_values,
//!     fx_provider,
//!     start_date,
//!     end_date,
//! )?;
//! ```

use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::currency::Currency;
use crate::fx::{FxError, FxRateProvider};

// =============================================================================
// TYPES
// =============================================================================

/// Attribution result for a single currency exposure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrencyAttribution {
    /// The currency of the exposure.
    pub currency: Currency,
    /// Return from asset price changes (in local currency).
    pub asset_return: Decimal,
    /// Return from FX rate changes.
    pub fx_return: Decimal,
    /// Interaction term (asset * fx).
    pub interaction: Decimal,
    /// Total return in base currency.
    pub total_return_base: Decimal,
    /// Weight of this currency in portfolio (as of start).
    pub weight_pct: Decimal,
    
    // Audit trail
    /// FX rate at start of period.
    pub fx_rate_start: Decimal,
    /// FX rate at end of period.
    pub fx_rate_end: Decimal,
}

impl CurrencyAttribution {
    /// Verify that components sum to total (within tolerance).
    pub fn verify_decomposition(&self) -> bool {
        let reconstructed = self.asset_return + self.fx_return + self.interaction;
        let diff = (reconstructed - self.total_return_base).abs();
        diff < Decimal::new(1, 10) // 1e-10 tolerance
    }
}

/// Complete FX attribution breakdown for a portfolio.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FxAttributionBreakdown {
    /// Attribution by currency.
    pub by_currency: Vec<CurrencyAttribution>,
    
    /// Portfolio-level asset return (weighted sum).
    pub portfolio_asset_return: Decimal,
    /// Portfolio-level FX return (weighted sum).
    pub portfolio_fx_return: Decimal,
    /// Portfolio-level interaction (weighted sum).
    pub portfolio_interaction: Decimal,
    /// Portfolio-level total return in base currency.
    pub portfolio_total_return_base: Decimal,
    
    /// Base currency used for reporting.
    pub base_currency: Currency,
    /// Start date of the period.
    pub start_date: NaiveDate,
    /// End date of the period.
    pub end_date: NaiveDate,
}

impl FxAttributionBreakdown {
    /// Verify that portfolio-level components sum correctly.
    pub fn verify_decomposition(&self) -> bool {
        let reconstructed = self.portfolio_asset_return 
            + self.portfolio_fx_return 
            + self.portfolio_interaction;
        let diff = (reconstructed - self.portfolio_total_return_base).abs();
        diff < Decimal::new(1, 8) // 1e-8 tolerance for aggregates
    }
}

/// Input values for attribution calculation.
#[derive(Debug, Clone)]
pub struct CurrencyValue {
    pub currency: Currency,
    pub value_local: Decimal,
}

// =============================================================================
// ENGINE
// =============================================================================

/// Engine for calculating FX attribution.
pub struct FxAttributionEngine {
    /// Base currency for reporting.
    base_currency: Currency,
    /// Max gap for LOCF rate lookup.
    max_gap_days: u32,
}

impl FxAttributionEngine {
    /// Create a new attribution engine.
    pub fn new(base_currency: Currency) -> Self {
        Self {
            base_currency,
            max_gap_days: 5,
        }
    }
    
    /// Create with custom LOCF gap.
    pub fn with_max_gap(base_currency: Currency, max_gap_days: u32) -> Self {
        Self {
            base_currency,
            max_gap_days,
        }
    }
    
    /// Calculate attribution for a single currency exposure.
    ///
    /// # Arguments
    ///
    /// * `currency` - The currency of the exposure
    /// * `value_local_start` - Value in local currency at start
    /// * `value_local_end` - Value in local currency at end
    /// * `weight_pct` - Weight of this currency in portfolio (start)
    /// * `fx_provider` - FX rate provider
    /// * `start_date` - Start of period
    /// * `end_date` - End of period
    pub fn calculate_currency_attribution(
        &self,
        currency: Currency,
        value_local_start: Decimal,
        value_local_end: Decimal,
        weight_pct: Decimal,
        fx_provider: &dyn FxRateProvider,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Result<CurrencyAttribution, FxError> {
        // Same currency as base - no FX effect
        if currency == self.base_currency {
            let asset_return = if value_local_start.is_zero() {
                Decimal::ZERO
            } else {
                value_local_end / value_local_start - Decimal::ONE
            };
            
            return Ok(CurrencyAttribution {
                currency,
                asset_return,
                fx_return: Decimal::ZERO,
                interaction: Decimal::ZERO,
                total_return_base: asset_return,
                weight_pct,
                fx_rate_start: Decimal::ONE,
                fx_rate_end: Decimal::ONE,
            });
        }
        
        // Get FX rates
        use crate::currency::FxPair;
        let pair = FxPair::new(currency, self.base_currency);
        
        let fx_start = fx_provider.get_rate_locf(start_date, pair, self.max_gap_days)?;
        let fx_end = fx_provider.get_rate_locf(end_date, pair, self.max_gap_days)?;
        
        // Calculate returns
        // R_asset_L = V_L(t1) / V_L(t0) - 1
        let asset_return = if value_local_start.is_zero() {
            Decimal::ZERO
        } else {
            value_local_end / value_local_start - Decimal::ONE
        };
        
        // R_fx = FX(t1) / FX(t0) - 1
        let fx_return = if fx_start.rate.is_zero() {
            Decimal::ZERO
        } else {
            fx_end.rate / fx_start.rate - Decimal::ONE
        };
        
        // Interaction = R_asset * R_fx
        let interaction = asset_return * fx_return;
        
        // Total in base = R_asset + R_fx + interaction
        let total_return_base = asset_return + fx_return + interaction;
        
        Ok(CurrencyAttribution {
            currency,
            asset_return,
            fx_return,
            interaction,
            total_return_base,
            weight_pct,
            fx_rate_start: fx_start.rate,
            fx_rate_end: fx_end.rate,
        })
    }
    
    /// Calculate attribution for multiple currency exposures.
    ///
    /// # Arguments
    ///
    /// * `values_start` - Map of currency -> value at start
    /// * `values_end` - Map of currency -> value at end
    /// * `fx_provider` - FX rate provider
    /// * `start_date` - Start of period
    /// * `end_date` - End of period
    pub fn calculate_period_attribution(
        &self,
        values_start: &BTreeMap<Currency, Decimal>,
        values_end: &BTreeMap<Currency, Decimal>,
        fx_provider: &dyn FxRateProvider,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Result<FxAttributionBreakdown, FxError> {
        // Calculate total portfolio value at start (in base currency)
        let total_start_base = self.convert_total_to_base(
            values_start, 
            fx_provider, 
            start_date
        )?;
        
        let mut by_currency = Vec::new();
        let mut portfolio_asset_return = Decimal::ZERO;
        let mut portfolio_fx_return = Decimal::ZERO;
        let mut portfolio_interaction = Decimal::ZERO;
        
        // Calculate attribution for each currency
        for (&currency, &value_start) in values_start {
            let value_end = values_end.get(&currency).copied().unwrap_or(Decimal::ZERO);
            
            // Weight = value_start_base / total_start_base
            let weight_pct = if total_start_base.is_zero() {
                Decimal::ZERO
            } else {
                let value_start_base = self.convert_to_base(
                    currency, 
                    value_start, 
                    fx_provider, 
                    start_date
                )?;
                (value_start_base / total_start_base) * Decimal::from(100)
            };
            
            let attribution = self.calculate_currency_attribution(
                currency,
                value_start,
                value_end,
                weight_pct,
                fx_provider,
                start_date,
                end_date,
            )?;
            
            // Weight contributions to portfolio level
            let weight_factor = weight_pct / Decimal::from(100);
            portfolio_asset_return += attribution.asset_return * weight_factor;
            portfolio_fx_return += attribution.fx_return * weight_factor;
            portfolio_interaction += attribution.interaction * weight_factor;
            
            by_currency.push(attribution);
        }
        
        let portfolio_total_return_base = portfolio_asset_return 
            + portfolio_fx_return 
            + portfolio_interaction;
        
        Ok(FxAttributionBreakdown {
            by_currency,
            portfolio_asset_return,
            portfolio_fx_return,
            portfolio_interaction,
            portfolio_total_return_base,
            base_currency: self.base_currency,
            start_date,
            end_date,
        })
    }
    
    /// Convert a single value to base currency.
    fn convert_to_base(
        &self,
        currency: Currency,
        value: Decimal,
        fx_provider: &dyn FxRateProvider,
        date: NaiveDate,
    ) -> Result<Decimal, FxError> {
        if currency == self.base_currency {
            return Ok(value);
        }
        
        use crate::currency::FxPair;
        let pair = FxPair::new(currency, self.base_currency);
        let rate = fx_provider.get_rate_locf(date, pair, self.max_gap_days)?;
        
        Ok(value * rate.rate)
    }
    
    /// Convert total portfolio value to base currency.
    fn convert_total_to_base(
        &self,
        values: &BTreeMap<Currency, Decimal>,
        fx_provider: &dyn FxRateProvider,
        date: NaiveDate,
    ) -> Result<Decimal, FxError> {
        let mut total = Decimal::ZERO;
        
        for (&currency, &value) in values {
            total += self.convert_to_base(currency, value, fx_provider, date)?;
        }
        
        Ok(total)
    }
}

// =============================================================================
// SIMPLE ATTRIBUTION HELPER
// =============================================================================

/// Calculate simple FX attribution for a single period.
///
/// This is a convenience function for quick calculations.
pub fn calculate_fx_attribution(
    asset_return_local: Decimal,
    fx_rate_start: Decimal,
    fx_rate_end: Decimal,
) -> (Decimal, Decimal, Decimal, Decimal) {
    // R_fx = FX(t1) / FX(t0) - 1
    let fx_return = if fx_rate_start.is_zero() {
        Decimal::ZERO
    } else {
        fx_rate_end / fx_rate_start - Decimal::ONE
    };
    
    // Interaction
    let interaction = asset_return_local * fx_return;
    
    // Total in base
    let total_base = asset_return_local + fx_return + interaction;
    
    (asset_return_local, fx_return, interaction, total_base)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx::InMemoryFxProvider;
    use crate::currency::FxPair;
    use rust_decimal_macros::dec;

    fn make_date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    fn make_provider() -> InMemoryFxProvider {
        let mut provider = InMemoryFxProvider::new();
        
        // USD/BRL rates
        provider.add_rate(FxPair::USD_BRL, make_date(2024, 1, 1), dec!(5.00));
        provider.add_rate(FxPair::USD_BRL, make_date(2024, 1, 31), dec!(5.50));
        
        provider
    }

    #[test]
    fn test_same_currency_no_fx_effect() {
        let provider = make_provider();
        let engine = FxAttributionEngine::new(Currency::BRL);
        
        let attr = engine.calculate_currency_attribution(
            Currency::BRL,
            dec!(10000),  // start: R$ 10,000
            dec!(11000),  // end: R$ 11,000
            dec!(100),    // 100% weight
            &provider,
            make_date(2024, 1, 1),
            make_date(2024, 1, 31),
        ).unwrap();
        
        assert_eq!(attr.asset_return, dec!(0.1)); // 10% return
        assert_eq!(attr.fx_return, Decimal::ZERO);
        assert_eq!(attr.interaction, Decimal::ZERO);
        assert_eq!(attr.total_return_base, dec!(0.1));
    }

    #[test]
    fn test_usd_to_brl_with_fx_gain() {
        let provider = make_provider();
        let engine = FxAttributionEngine::new(Currency::BRL);
        
        // USD position: $1,000 -> $1,100 (10% asset return)
        // FX: 5.00 -> 5.50 (10% FX gain)
        let attr = engine.calculate_currency_attribution(
            Currency::USD,
            dec!(1000),
            dec!(1100),
            dec!(100),
            &provider,
            make_date(2024, 1, 1),
            make_date(2024, 1, 31),
        ).unwrap();
        
        assert_eq!(attr.asset_return, dec!(0.1)); // 10%
        assert_eq!(attr.fx_return, dec!(0.1));    // 10%
        assert_eq!(attr.interaction, dec!(0.01)); // 1% (10% * 10%)
        assert_eq!(attr.total_return_base, dec!(0.21)); // 21% total
        
        // Verify decomposition
        assert!(attr.verify_decomposition());
        
        // Verify: start $1000 @ 5.00 = R$5,000
        //         end   $1100 @ 5.50 = R$6,050
        //         return = 6050/5000 - 1 = 0.21 = 21%
    }

    #[test]
    fn test_multiplicative_decomposition() {
        // Verify: (1 + R_total) = (1 + R_asset) * (1 + R_fx)
        let asset_return = dec!(0.10);  // 10%
        let fx_return = dec!(0.10);     // 10%
        
        let multiplicative = (Decimal::ONE + asset_return) * (Decimal::ONE + fx_return);
        let total_return = multiplicative - Decimal::ONE;
        
        // Check additive decomposition
        let interaction = asset_return * fx_return;
        let additive = asset_return + fx_return + interaction;
        
        assert_eq!(total_return, additive);
        assert_eq!(total_return, dec!(0.21));
    }

    #[test]
    fn test_portfolio_attribution() {
        let provider = make_provider();
        let engine = FxAttributionEngine::new(Currency::BRL);
        
        let mut values_start = BTreeMap::new();
        values_start.insert(Currency::BRL, dec!(5000)); // R$ 5,000
        values_start.insert(Currency::USD, dec!(1000)); // $1,000 @ 5.00 = R$ 5,000
        
        let mut values_end = BTreeMap::new();
        values_end.insert(Currency::BRL, dec!(5500)); // R$ 5,500 (10% return)
        values_end.insert(Currency::USD, dec!(1100)); // $1,100 (10% return)
        
        let breakdown = engine.calculate_period_attribution(
            &values_start,
            &values_end,
            &provider,
            make_date(2024, 1, 1),
            make_date(2024, 1, 31),
        ).unwrap();
        
        assert_eq!(breakdown.by_currency.len(), 2);
        
        // Portfolio is 50% BRL, 50% USD (in base currency)
        // BRL: 10% asset, 0% fx, 0% interaction
        // USD: 10% asset, 10% fx, 1% interaction = 21% total
        // Portfolio: 0.5 * 10% + 0.5 * 21% = 15.5% total
        
        // Note: weights should be approximately 50% each
        let brl_attr = breakdown.by_currency.iter()
            .find(|a| a.currency == Currency::BRL).unwrap();
        let usd_attr = breakdown.by_currency.iter()
            .find(|a| a.currency == Currency::USD).unwrap();
        
        assert_eq!(brl_attr.asset_return, dec!(0.1));
        assert_eq!(usd_attr.asset_return, dec!(0.1));
        assert_eq!(usd_attr.total_return_base, dec!(0.21));
        
        // Verify portfolio decomposition
        assert!(breakdown.verify_decomposition());
    }

    #[test]
    fn test_simple_attribution_helper() {
        let (asset, fx, interaction, total) = calculate_fx_attribution(
            dec!(0.10),  // 10% asset return
            dec!(5.00),  // start FX
            dec!(5.50),  // end FX
        );
        
        assert_eq!(asset, dec!(0.10));
        assert_eq!(fx, dec!(0.10));
        assert_eq!(interaction, dec!(0.01));
        assert_eq!(total, dec!(0.21));
    }

    #[test]
    fn test_fx_loss_scenario() {
        let mut provider = InMemoryFxProvider::new();
        // USD weakens: 5.00 -> 4.50 (10% loss for USD holders in BRL terms)
        provider.add_rate(FxPair::USD_BRL, make_date(2024, 1, 1), dec!(5.00));
        provider.add_rate(FxPair::USD_BRL, make_date(2024, 1, 31), dec!(4.50));
        
        let engine = FxAttributionEngine::new(Currency::BRL);
        
        let attr = engine.calculate_currency_attribution(
            Currency::USD,
            dec!(1000),
            dec!(1100),  // 10% asset gain
            dec!(100),
            &provider,
            make_date(2024, 1, 1),
            make_date(2024, 1, 31),
        ).unwrap();
        
        assert_eq!(attr.asset_return, dec!(0.1));
        assert_eq!(attr.fx_return, dec!(-0.1));
        assert_eq!(attr.interaction, dec!(-0.01));
        assert_eq!(attr.total_return_base, dec!(-0.01)); // Net loss of 1%
        
        // Verify: start $1000 @ 5.00 = R$5,000
        //         end   $1100 @ 4.50 = R$4,950
        //         return = 4950/5000 - 1 = -0.01 = -1%
    }
}





