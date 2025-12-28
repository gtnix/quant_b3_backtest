//! Gating filter for asset eligibility.

use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::filters::Market;
use super::types::{EntryExclusion, ExclusionReason, ExclusionStage};
use super::universe_range::{EligibilityResult, UniverseRangeProvider};
use super::eligibility::EligibilityProvider;

/// Configuration for gating filters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatingConfig {
    /// Minimum average daily volume in BRL (for BR market)
    #[serde(default = "default_min_volume_brl")]
    pub min_avg_volume_brl: Decimal,
    
    /// Minimum average daily volume in USD (for US market)
    #[serde(default = "default_min_volume_usd")]
    pub min_avg_volume_usd: Decimal,
    
    /// Minimum price in BRL (for BR market)
    #[serde(default = "default_min_price_brl")]
    pub min_price_brl: Decimal,
    
    /// Minimum price in USD (for US market)
    #[serde(default = "default_min_price_usd")]
    pub min_price_usd: Decimal,
    
    /// Whether to require fundamentals (Value, Quality techniques)
    #[serde(default)]
    pub require_fundamentals: bool,
    
    /// Whether to require dividends (Carry, Dividend techniques)
    #[serde(default)]
    pub require_dividends: bool,
    
    /// Minimum days of price history required
    #[serde(default = "default_min_price_days")]
    pub min_price_days: usize,
}

fn default_min_volume_brl() -> Decimal { Decimal::from(500_000) }
fn default_min_volume_usd() -> Decimal { Decimal::from(1_000_000) }
fn default_min_price_brl() -> Decimal { Decimal::ONE }
fn default_min_price_usd() -> Decimal { Decimal::ONE }
fn default_min_price_days() -> usize { 20 }

impl Default for GatingConfig {
    fn default() -> Self {
        Self {
            min_avg_volume_brl: default_min_volume_brl(),
            min_avg_volume_usd: default_min_volume_usd(),
            min_price_brl: default_min_price_brl(),
            min_price_usd: default_min_price_usd(),
            require_fundamentals: false,
            require_dividends: false,
            min_price_days: default_min_price_days(),
        }
    }
}

/// Candidate asset for gating evaluation.
#[derive(Debug, Clone)]
pub struct GatingCandidate {
    pub symbol: String,
    pub market: Market,
    pub price: Option<Decimal>,
    pub avg_volume: Option<Decimal>,
    pub price_days: usize,
    pub has_fundamentals: bool,
    pub has_dividends: bool,
    pub is_tradeable: bool,
    /// Date of fundamentals snapshot (for anti-look-ahead check)
    pub fundamentals_as_of: Option<NaiveDate>,
    /// Rebalance date for point-in-time validation
    pub rebalance_date: Option<NaiveDate>,
}

impl GatingCandidate {
    pub fn new(symbol: impl Into<String>, market: Market) -> Self {
        Self {
            symbol: symbol.into(),
            market,
            price: None,
            avg_volume: None,
            price_days: 0,
            has_fundamentals: false,
            has_dividends: false,
            is_tradeable: true,
            fundamentals_as_of: None,
            rebalance_date: None,
        }
    }
}

/// Gating filter to exclude ineligible assets.
pub struct GatingFilter {
    config: GatingConfig,
    /// Optional eligibility provider for point-in-time validation (V1 or V2).
    /// When set, candidates are checked against historical existence windows.
    eligibility_provider: Option<Arc<dyn EligibilityProvider>>,
}

impl std::fmt::Debug for GatingFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatingFilter")
            .field("config", &self.config)
            .field("has_eligibility_provider", &self.eligibility_provider.is_some())
            .finish()
    }
}

impl Clone for GatingFilter {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            eligibility_provider: self.eligibility_provider.clone(),
        }
    }
}

impl GatingFilter {
    /// Create a new gating filter without universe validation.
    pub fn new(config: GatingConfig) -> Self {
        Self {
            config,
            eligibility_provider: None,
        }
    }

    /// Create a gating filter with V2 eligibility provider (supports both V1 and V2).
    ///
    /// When enabled, candidates are excluded if:
    /// - The symbol is not in the universe data (NoUniverseRangeData)
    /// - The rebalance_date is outside the symbol's eligibility window (OutsideUniverseDateRange)
    pub fn with_eligibility_provider(
        config: GatingConfig,
        provider: Arc<dyn EligibilityProvider>,
    ) -> Self {
        Self {
            config,
            eligibility_provider: Some(provider),
        }
    }

    /// Create a gating filter with V1 universe range validation (backward compatible).
    ///
    /// When enabled, candidates are excluded if:
    /// - The symbol is not in the universe data (NoUniverseRangeData)
    /// - The rebalance_date is outside the symbol's [min_date, max_date] range (OutsideUniverseDateRange)
    pub fn with_universe_provider(config: GatingConfig, provider: Arc<UniverseRangeProvider>) -> Self {
        // Wrap V1 provider in trait object for unified handling
        Self {
            config,
            eligibility_provider: Some(provider as Arc<dyn EligibilityProvider>),
        }
    }

    /// Get the eligibility provider if set.
    pub fn eligibility_provider(&self) -> Option<&Arc<dyn EligibilityProvider>> {
        self.eligibility_provider.as_ref()
    }

    /// Check if eligibility validation is enabled (V1 or V2).
    pub fn has_eligibility_provider(&self) -> bool {
        self.eligibility_provider.is_some()
    }

    /// Backward compatible: get the universe provider if set.
    /// Deprecated: use eligibility_provider() instead.
    pub fn universe_provider(&self) -> Option<&Arc<dyn EligibilityProvider>> {
        self.eligibility_provider.as_ref()
    }

    /// Apply gating filter to candidates.
    /// Returns (eligible, excluded).
    pub fn apply(&self, candidates: Vec<GatingCandidate>) -> (Vec<GatingCandidate>, Vec<EntryExclusion>) {
        let mut eligible = Vec::new();
        let mut excluded = Vec::new();

        for candidate in candidates {
            match self.evaluate(&candidate) {
                Ok(()) => eligible.push(candidate),
                Err(reason) => {
                    excluded.push(EntryExclusion {
                        symbol: candidate.symbol,
                        reason,
                        stage: ExclusionStage::Gating,
                        score: None,
                    });
                }
            }
        }

        (eligible, excluded)
    }

    /// Evaluate a single candidate.
    fn evaluate(&self, candidate: &GatingCandidate) -> Result<(), ExclusionReason> {
        // FIRST: Check universe date range (survivorship bias guard)
        // This must be the first check - if asset didn't exist at this date,
        // no other checks are meaningful.
        if let Some(ref provider) = self.eligibility_provider {
            self.check_universe_eligibility(candidate, provider.as_ref())?;
        }

        // Check tradeability
        if !candidate.is_tradeable {
            return Err(ExclusionReason::NotTradeable);
        }

        // Check price data availability
        if candidate.price_days < self.config.min_price_days {
            return Err(ExclusionReason::MissingPriceData);
        }

        // Check price level
        if let Some(price) = candidate.price {
            let min_price = match candidate.market {
                Market::BR => self.config.min_price_brl,
                Market::US => self.config.min_price_usd,
            };
            if price < min_price {
                return Err(ExclusionReason::PriceTooLow);
            }
        } else {
            return Err(ExclusionReason::MissingPriceData);
        }

        // Check liquidity
        if let Some(volume) = candidate.avg_volume {
            let min_volume = match candidate.market {
                Market::BR => self.config.min_avg_volume_brl,
                Market::US => self.config.min_avg_volume_usd,
            };
            if volume < min_volume {
                return Err(ExclusionReason::InsufficientLiquidity);
            }
        } else {
            return Err(ExclusionReason::InsufficientLiquidity);
        }

        // Check fundamentals requirement
        if self.config.require_fundamentals && !candidate.has_fundamentals {
            return Err(ExclusionReason::MissingFundamentals);
        }

        // Check for look-ahead bias: fundamentals from future date
        if let (Some(fundamentals_date), Some(rebalance_date)) = 
            (candidate.fundamentals_as_of, candidate.rebalance_date) 
        {
            if fundamentals_date > rebalance_date {
                return Err(ExclusionReason::FutureFundamentals);
            }
        }

        // Check dividends requirement
        if self.config.require_dividends && !candidate.has_dividends {
            return Err(ExclusionReason::MissingDividends);
        }

        Ok(())
    }

    /// Check if candidate is eligible based on universe date range (V1 or V2).
    ///
    /// Returns Ok(()) if eligible, Err(ExclusionReason) otherwise.
    fn check_universe_eligibility(
        &self,
        candidate: &GatingCandidate,
        provider: &dyn EligibilityProvider,
    ) -> Result<(), ExclusionReason> {
        // Require rebalance_date for universe validation
        let rebalance_date = match candidate.rebalance_date {
            Some(date) => date,
            None => {
                // If no rebalance_date provided, we can't validate.
                // Conservative approach: reject as NoUniverseRangeData.
                // This ensures callers must provide rebalance_date for proper validation.
                return Err(ExclusionReason::NoUniverseRangeData);
            }
        };

        match provider.is_eligible(&candidate.symbol, rebalance_date) {
            EligibilityResult::Eligible => Ok(()),
            EligibilityResult::OutsideDateRange { .. } => {
                Err(ExclusionReason::OutsideUniverseDateRange)
            }
            EligibilityResult::SymbolNotInUniverse => {
                Err(ExclusionReason::NoUniverseRangeData)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_valid_br_candidate() -> GatingCandidate {
        GatingCandidate {
            symbol: "PETR4".to_string(),
            market: Market::BR,
            price: Some(Decimal::from(38)),
            avg_volume: Some(Decimal::from(1_000_000)),
            price_days: 30,
            has_fundamentals: true,
            has_dividends: true,
            is_tradeable: true,
            fundamentals_as_of: None,
            rebalance_date: None,
        }
    }

    #[test]
    fn test_valid_candidate_passes() {
        let filter = GatingFilter::new(GatingConfig::default());
        let candidate = make_valid_br_candidate();
        
        let (eligible, excluded) = filter.apply(vec![candidate]);
        
        assert_eq!(eligible.len(), 1);
        assert!(excluded.is_empty());
    }

    #[test]
    fn test_low_price_excluded() {
        let filter = GatingFilter::new(GatingConfig::default());
        let mut candidate = make_valid_br_candidate();
        candidate.price = Some(Decimal::new(50, 2)); // R$ 0.50
        
        let (eligible, excluded) = filter.apply(vec![candidate]);
        
        assert!(eligible.is_empty());
        assert_eq!(excluded.len(), 1);
        assert_eq!(excluded[0].reason, ExclusionReason::PriceTooLow);
    }

    #[test]
    fn test_low_volume_excluded() {
        let filter = GatingFilter::new(GatingConfig::default());
        let mut candidate = make_valid_br_candidate();
        candidate.avg_volume = Some(Decimal::from(100_000)); // Below 500k
        
        let (eligible, excluded) = filter.apply(vec![candidate]);
        
        assert!(eligible.is_empty());
        assert_eq!(excluded[0].reason, ExclusionReason::InsufficientLiquidity);
    }

    #[test]
    fn test_missing_fundamentals_when_required() {
        let config = GatingConfig {
            require_fundamentals: true,
            ..Default::default()
        };
        let filter = GatingFilter::new(config);
        let mut candidate = make_valid_br_candidate();
        candidate.has_fundamentals = false;
        
        let (eligible, excluded) = filter.apply(vec![candidate]);
        
        assert!(eligible.is_empty());
        assert_eq!(excluded[0].reason, ExclusionReason::MissingFundamentals);
    }

    #[test]
    fn test_us_market_thresholds() {
        let filter = GatingFilter::new(GatingConfig::default());
        let candidate = GatingCandidate {
            symbol: "AAPL".to_string(),
            market: Market::US,
            price: Some(Decimal::from(150)),
            avg_volume: Some(Decimal::from(5_000_000)),
            price_days: 30,
            has_fundamentals: false, // US typically no fundamentals
            has_dividends: false,
            is_tradeable: true,
            fundamentals_as_of: None,
            rebalance_date: None,
        };
        
        let (eligible, excluded) = filter.apply(vec![candidate]);
        
        // Should pass if fundamentals not required
        assert_eq!(eligible.len(), 1);
        assert!(excluded.is_empty());
    }

    #[test]
    fn test_not_tradeable_excluded() {
        let filter = GatingFilter::new(GatingConfig::default());
        let mut candidate = make_valid_br_candidate();
        candidate.is_tradeable = false;
        
        let (eligible, excluded) = filter.apply(vec![candidate]);
        
        assert!(eligible.is_empty());
        assert_eq!(excluded[0].reason, ExclusionReason::NotTradeable);
    }

    #[test]
    fn test_future_fundamentals_excluded() {
        let filter = GatingFilter::new(GatingConfig::default());
        let mut candidate = make_valid_br_candidate();
        
        // Rebalance on 2025-01-03, but fundamentals from 2025-03-31 (future)
        candidate.rebalance_date = Some(NaiveDate::from_ymd_opt(2025, 1, 3).unwrap());
        candidate.fundamentals_as_of = Some(NaiveDate::from_ymd_opt(2025, 3, 31).unwrap());
        
        let (eligible, excluded) = filter.apply(vec![candidate]);
        
        assert!(eligible.is_empty());
        assert_eq!(excluded.len(), 1);
        assert_eq!(excluded[0].reason, ExclusionReason::FutureFundamentals);
    }

    #[test]
    fn test_past_fundamentals_passes() {
        let filter = GatingFilter::new(GatingConfig::default());
        let mut candidate = make_valid_br_candidate();
        
        // Rebalance on 2025-01-03, fundamentals from 2024-09-30 (past)
        candidate.rebalance_date = Some(NaiveDate::from_ymd_opt(2025, 1, 3).unwrap());
        candidate.fundamentals_as_of = Some(NaiveDate::from_ymd_opt(2024, 9, 30).unwrap());
        
        let (eligible, excluded) = filter.apply(vec![candidate]);
        
        assert_eq!(eligible.len(), 1);
        assert!(excluded.is_empty());
    }

    // ========================================================================
    // Universe Range Gating Tests (Survivorship Bias)
    // ========================================================================

    use super::super::universe_range::{DateRange, UniverseRangeProvider};
    use std::collections::HashMap;

    fn make_test_universe() -> Arc<UniverseRangeProvider> {
        let mut ranges = HashMap::new();
        // PETR4: existed from 2015-01-02 to 2025-12-23
        ranges.insert(
            "PETR4".to_string(),
            DateRange::new(
                NaiveDate::from_ymd_opt(2015, 1, 2).unwrap(),
                NaiveDate::from_ymd_opt(2025, 12, 23).unwrap(),
            ),
        );
        // RAIZ4: IPO'd in 2021, still active
        ranges.insert(
            "RAIZ4".to_string(),
            DateRange::new(
                NaiveDate::from_ymd_opt(2021, 8, 5).unwrap(),
                NaiveDate::from_ymd_opt(2025, 12, 23).unwrap(),
            ),
        );
        // OIBR3: existed but delisted in 2020
        ranges.insert(
            "OIBR3".to_string(),
            DateRange::new(
                NaiveDate::from_ymd_opt(2015, 1, 2).unwrap(),
                NaiveDate::from_ymd_opt(2020, 12, 31).unwrap(),
            ),
        );
        Arc::new(UniverseRangeProvider::from_map(ranges))
    }

    #[test]
    fn test_universe_gating_eligible_at_min_date() {
        let universe = make_test_universe();
        let filter = GatingFilter::with_universe_provider(GatingConfig::default(), universe);
        
        let mut candidate = make_valid_br_candidate();
        candidate.rebalance_date = Some(NaiveDate::from_ymd_opt(2015, 1, 2).unwrap());
        
        let (eligible, excluded) = filter.apply(vec![candidate]);
        
        assert_eq!(eligible.len(), 1);
        assert!(excluded.is_empty());
    }

    #[test]
    fn test_universe_gating_eligible_in_middle() {
        let universe = make_test_universe();
        let filter = GatingFilter::with_universe_provider(GatingConfig::default(), universe);
        
        let mut candidate = make_valid_br_candidate();
        candidate.rebalance_date = Some(NaiveDate::from_ymd_opt(2020, 6, 15).unwrap());
        
        let (eligible, excluded) = filter.apply(vec![candidate]);
        
        assert_eq!(eligible.len(), 1);
        assert!(excluded.is_empty());
    }

    #[test]
    fn test_universe_gating_before_min_date_excluded() {
        let universe = make_test_universe();
        let filter = GatingFilter::with_universe_provider(GatingConfig::default(), universe);
        
        let mut candidate = make_valid_br_candidate();
        // PETR4 min_date is 2015-01-02, try day before
        candidate.rebalance_date = Some(NaiveDate::from_ymd_opt(2015, 1, 1).unwrap());
        
        let (eligible, excluded) = filter.apply(vec![candidate]);
        
        assert!(eligible.is_empty());
        assert_eq!(excluded.len(), 1);
        assert_eq!(excluded[0].reason, ExclusionReason::OutsideUniverseDateRange);
    }

    #[test]
    fn test_universe_gating_after_max_date_excluded() {
        let universe = make_test_universe();
        let filter = GatingFilter::with_universe_provider(GatingConfig::default(), universe);
        
        let mut candidate = make_valid_br_candidate();
        // PETR4 max_date is 2025-12-23, try day after
        candidate.rebalance_date = Some(NaiveDate::from_ymd_opt(2025, 12, 24).unwrap());
        
        let (eligible, excluded) = filter.apply(vec![candidate]);
        
        assert!(eligible.is_empty());
        assert_eq!(excluded.len(), 1);
        assert_eq!(excluded[0].reason, ExclusionReason::OutsideUniverseDateRange);
    }

    #[test]
    fn test_universe_gating_no_resurrection() {
        // OIBR3 delisted end of 2020, should not appear in 2021
        let universe = make_test_universe();
        let filter = GatingFilter::with_universe_provider(GatingConfig::default(), universe);
        
        let mut candidate = GatingCandidate {
            symbol: "OIBR3".to_string(),
            market: Market::BR,
            price: Some(Decimal::from(5)),
            avg_volume: Some(Decimal::from(1_000_000)),
            price_days: 30,
            has_fundamentals: true,
            has_dividends: false,
            is_tradeable: true,
            fundamentals_as_of: None,
            rebalance_date: Some(NaiveDate::from_ymd_opt(2021, 1, 15).unwrap()),
        };
        
        let (eligible, excluded) = filter.apply(vec![candidate]);
        
        assert!(eligible.is_empty());
        assert_eq!(excluded.len(), 1);
        assert_eq!(excluded[0].reason, ExclusionReason::OutsideUniverseDateRange);
        assert_eq!(excluded[0].symbol, "OIBR3");
    }

    #[test]
    fn test_universe_gating_unknown_symbol_excluded() {
        let universe = make_test_universe();
        let filter = GatingFilter::with_universe_provider(GatingConfig::default(), universe);
        
        let mut candidate = GatingCandidate {
            symbol: "UNKNOWN".to_string(),
            market: Market::BR,
            price: Some(Decimal::from(10)),
            avg_volume: Some(Decimal::from(1_000_000)),
            price_days: 30,
            has_fundamentals: false,
            has_dividends: false,
            is_tradeable: true,
            fundamentals_as_of: None,
            rebalance_date: Some(NaiveDate::from_ymd_opt(2020, 6, 15).unwrap()),
        };
        
        let (eligible, excluded) = filter.apply(vec![candidate]);
        
        assert!(eligible.is_empty());
        assert_eq!(excluded.len(), 1);
        assert_eq!(excluded[0].reason, ExclusionReason::NoUniverseRangeData);
    }

    #[test]
    fn test_universe_gating_mixed_eligibility() {
        let universe = make_test_universe();
        let filter = GatingFilter::with_universe_provider(GatingConfig::default(), universe);
        
        // Rebalance date: 2021-09-01
        // PETR4: eligible (2015 - 2025)
        // RAIZ4: eligible (IPO'd 2021-08-05)
        // OIBR3: excluded (delisted 2020-12-31)
        let rebalance_date = Some(NaiveDate::from_ymd_opt(2021, 9, 1).unwrap());
        
        let candidates = vec![
            GatingCandidate {
                symbol: "PETR4".to_string(),
                market: Market::BR,
                price: Some(Decimal::from(30)),
                avg_volume: Some(Decimal::from(50_000_000)),
                price_days: 100,
                has_fundamentals: true,
                has_dividends: true,
                is_tradeable: true,
                fundamentals_as_of: None,
                rebalance_date,
            },
            GatingCandidate {
                symbol: "RAIZ4".to_string(),
                market: Market::BR,
                price: Some(Decimal::from(10)),
                avg_volume: Some(Decimal::from(10_000_000)),
                price_days: 30,
                has_fundamentals: false,
                has_dividends: false,
                is_tradeable: true,
                fundamentals_as_of: None,
                rebalance_date,
            },
            GatingCandidate {
                symbol: "OIBR3".to_string(),
                market: Market::BR,
                price: Some(Decimal::from(5)),
                avg_volume: Some(Decimal::from(1_000_000)),
                price_days: 30,
                has_fundamentals: false,
                has_dividends: false,
                is_tradeable: true,
                fundamentals_as_of: None,
                rebalance_date,
            },
        ];
        
        let (eligible, excluded) = filter.apply(candidates);
        
        assert_eq!(eligible.len(), 2);
        assert_eq!(excluded.len(), 1);
        
        let eligible_symbols: Vec<&str> = eligible.iter().map(|c| c.symbol.as_str()).collect();
        assert!(eligible_symbols.contains(&"PETR4"));
        assert!(eligible_symbols.contains(&"RAIZ4"));
        
        assert_eq!(excluded[0].symbol, "OIBR3");
        assert_eq!(excluded[0].reason, ExclusionReason::OutsideUniverseDateRange);
    }

    #[test]
    fn test_universe_gating_ipo_after_rebalance_excluded() {
        // RAIZ4 IPO'd 2021-08-05, should not appear in early 2021
        let universe = make_test_universe();
        let filter = GatingFilter::with_universe_provider(GatingConfig::default(), universe);
        
        let candidate = GatingCandidate {
            symbol: "RAIZ4".to_string(),
            market: Market::BR,
            price: Some(Decimal::from(10)),
            avg_volume: Some(Decimal::from(10_000_000)),
            price_days: 30,
            has_fundamentals: false,
            has_dividends: false,
            is_tradeable: true,
            fundamentals_as_of: None,
            rebalance_date: Some(NaiveDate::from_ymd_opt(2021, 1, 15).unwrap()),
        };
        
        let (eligible, excluded) = filter.apply(vec![candidate]);
        
        assert!(eligible.is_empty());
        assert_eq!(excluded[0].reason, ExclusionReason::OutsideUniverseDateRange);
    }

    #[test]
    fn test_universe_gating_no_rebalance_date_excluded() {
        let universe = make_test_universe();
        let filter = GatingFilter::with_universe_provider(GatingConfig::default(), universe);
        
        let mut candidate = make_valid_br_candidate();
        candidate.rebalance_date = None; // No rebalance date
        
        let (eligible, excluded) = filter.apply(vec![candidate]);
        
        assert!(eligible.is_empty());
        assert_eq!(excluded.len(), 1);
        assert_eq!(excluded[0].reason, ExclusionReason::NoUniverseRangeData);
    }

    #[test]
    fn test_without_universe_provider_passes() {
        // Without universe provider, universe check is skipped
        let filter = GatingFilter::new(GatingConfig::default());
        let mut candidate = make_valid_br_candidate();
        candidate.rebalance_date = Some(NaiveDate::from_ymd_opt(2010, 1, 1).unwrap());
        
        let (eligible, excluded) = filter.apply(vec![candidate]);
        
        // Should pass because universe check is not enabled
        assert_eq!(eligible.len(), 1);
        assert!(excluded.is_empty());
    }
}

