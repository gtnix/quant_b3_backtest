//! Gating filter for asset eligibility.

use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::filters::Market;
use super::types::{EntryExclusion, ExclusionReason, ExclusionStage};

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
#[derive(Debug, Clone)]
pub struct GatingFilter {
    config: GatingConfig,
}

impl GatingFilter {
    pub fn new(config: GatingConfig) -> Self {
        Self { config }
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
}

