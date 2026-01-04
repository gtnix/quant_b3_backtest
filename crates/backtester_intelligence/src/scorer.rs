//! Asset scoring and ranking.

use crate::config::{FilterMode, IntelligenceConfig};
use crate::filters::{AssetData, AssetFilter, FilterResult};
use crate::filters::{
    CarryFilter, DividendYieldFilter, LowVolFilter, MomentumFilter, QualityFilter, SizeFilter,
    ValueFilter,
};
use crate::risk_free::RiskFreeRateProvider;
use std::sync::Arc;

/// Scored asset with combined filter results.
#[derive(Debug, Clone)]
pub struct ScoredAsset {
    pub symbol: String,
    pub total_score: f64,
    pub passed_all: bool,
    pub filter_results: Vec<(String, FilterResult)>,
    pub rank: Option<usize>,
}

impl ScoredAsset {
    /// Check if asset passed a specific filter.
    pub fn passed_filter(&self, name: &str) -> bool {
        self.filter_results
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, r)| r.passed)
            .unwrap_or(false)
    }

    /// Get score for a specific filter.
    pub fn filter_score(&self, name: &str) -> Option<f64> {
        self.filter_results
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, r)| r.score)
    }
}

/// Asset scorer that combines multiple filters.
pub struct AssetScorer {
    filters: Vec<Box<dyn AssetFilter>>,
    mode: FilterMode,
    top_percentile: f64,
}

impl AssetScorer {
    /// Create scorer from configuration (uses config fallback values for carry).
    pub fn from_config(config: &IntelligenceConfig) -> Self {
        Self::from_config_with_provider(config, None)
    }

    /// Create scorer from configuration with optional risk-free rate provider.
    /// When provider is given, CarryFilter uses point-in-time rates from database.
    pub fn from_config_with_provider(
        config: &IntelligenceConfig,
        rate_provider: Option<Arc<dyn RiskFreeRateProvider>>,
    ) -> Self {
        let mut filters: Vec<Box<dyn AssetFilter>> = Vec::new();

        if let Some(ref m) = config.momentum {
            if m.base.enabled {
                filters.push(Box::new(MomentumFilter::new(m.clone())));
            }
        }

        if let Some(ref v) = config.value {
            if v.base.enabled {
                filters.push(Box::new(ValueFilter::new(v.clone())));
            }
        }

        if let Some(ref q) = config.quality {
            if q.base.enabled {
                filters.push(Box::new(QualityFilter::new(q.clone())));
            }
        }

        if let Some(ref l) = config.low_vol {
            if l.base.enabled {
                filters.push(Box::new(LowVolFilter::new(l.clone())));
            }
        }

        if let Some(ref d) = config.dividend_yield {
            if d.base.enabled {
                filters.push(Box::new(DividendYieldFilter::new(d.clone())));
            }
        }

        if let Some(ref s) = config.size {
            if s.base.enabled {
                filters.push(Box::new(SizeFilter::new(s.clone())));
            }
        }

        if let Some(ref c) = config.carry {
            if c.base.enabled {
                let filter = match &rate_provider {
                    Some(provider) => CarryFilter::with_provider(c.clone(), Arc::clone(provider)),
                    None => CarryFilter::new(c.clone()),
                };
                filters.push(Box::new(filter));
            }
        }

        Self {
            filters,
            mode: config.mode,
            top_percentile: config.top_percentile,
        }
    }

    /// Create empty scorer (no filters).
    pub fn empty() -> Self {
        Self {
            filters: Vec::new(),
            mode: FilterMode::Intersection,
            top_percentile: 100.0,
        }
    }

    /// Check if scorer has any active filters.
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    /// Score a single asset.
    pub fn score(&self, data: &AssetData) -> ScoredAsset {
        let mut results = Vec::new();
        let mut total_score = 0.0;
        let mut total_weight = 0.0;
        let mut all_passed = true;

        for filter in &self.filters {
            let result = filter.evaluate(data);

            if !result.passed {
                all_passed = false;
            }

            let weight = filter.weight();
            total_score += result.score * weight;
            total_weight += weight;

            results.push((filter.name().to_string(), result));
        }

        let normalized_score = if total_weight > 0.0 {
            total_score / total_weight
        } else {
            0.0
        };

        ScoredAsset {
            symbol: data.symbol.clone(),
            total_score: normalized_score,
            passed_all: all_passed,
            filter_results: results,
            rank: None,
        }
    }

    /// Score and rank multiple assets.
    pub fn score_and_rank(&self, assets: &[AssetData]) -> Vec<ScoredAsset> {
        let mut scored: Vec<ScoredAsset> = assets.iter().map(|data| self.score(data)).collect();

        // Filter based on mode
        match self.mode {
            FilterMode::Intersection => {
                // Keep only assets that passed all filters
                scored.retain(|s| s.passed_all);
            }
            FilterMode::Score => {
                // Keep all, sort by score
            }
            FilterMode::Cascade => {
                // Already filtered during evaluation
                scored.retain(|s| s.passed_all);
            }
        }

        // Sort by score (descending)
        scored.sort_by(|a, b| {
            b.total_score
                .partial_cmp(&a.total_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Assign ranks
        for (i, asset) in scored.iter_mut().enumerate() {
            asset.rank = Some(i + 1);
        }

        // Apply top percentile filter
        if self.top_percentile < 100.0 && !scored.is_empty() {
            let keep_count =
                ((scored.len() as f64) * (self.top_percentile / 100.0)).ceil() as usize;
            scored.truncate(keep_count.max(1));
        }

        scored
    }

    /// Get selected symbols (convenience method).
    pub fn select_symbols(&self, assets: &[AssetData]) -> Vec<String> {
        self.score_and_rank(assets)
            .into_iter()
            .map(|s| s.symbol)
            .collect()
    }

    /// Get number of active filters.
    pub fn filter_count(&self) -> usize {
        self.filters.len()
    }

    /// Get filter names.
    pub fn filter_names(&self) -> Vec<&str> {
        self.filters.iter().map(|f| f.name()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;

    #[test]
    fn test_scorer_empty() {
        let scorer = AssetScorer::empty();
        assert!(scorer.is_empty());
    }

    #[test]
    fn test_scorer_single_filter() {
        let config = IntelligenceConfig {
            enabled: true,
            mode: FilterMode::Score,
            top_percentile: 50.0,
            momentum: Some(MomentumConfig {
                base: AssetFilterConfig {
                    enabled: true,
                    weight: 1.0,
                    threshold_mode: crate::config::FilterThresholdMode::Quantile,
                    top_quantile: 0.20,
                },
                ..Default::default()
            }),
            ..Default::default()
        };

        let scorer = AssetScorer::from_config(&config);
        assert_eq!(scorer.filter_count(), 1);

        // Create test assets
        let mut asset1 = AssetData::new("GOOD");
        asset1.momentum_return = Some(0.20);

        let mut asset2 = AssetData::new("BAD");
        asset2.momentum_return = Some(-0.10);

        let ranked = scorer.score_and_rank(&[asset1, asset2]);
        assert_eq!(ranked.len(), 1); // top 50% = 1 asset
        assert_eq!(ranked[0].symbol, "GOOD");
    }

    #[test]
    fn test_scorer_intersection_mode() {
        let config = IntelligenceConfig {
            enabled: true,
            mode: FilterMode::Intersection,
            top_percentile: 100.0,
            momentum: Some(MomentumConfig {
                base: AssetFilterConfig {
                    enabled: true,
                    weight: 0.5,
                    threshold_mode: crate::config::FilterThresholdMode::Quantile,
                    top_quantile: 0.20,
                },
                min_return: 0.05,
                ..Default::default()
            }),
            value: Some(ValueConfig {
                base: AssetFilterConfig {
                    enabled: true,
                    weight: 0.5,
                    threshold_mode: crate::config::FilterThresholdMode::Quantile,
                    top_quantile: 0.20,
                },
                max_pe: 20.0,
                max_pb: 3.0,
                ..Default::default()
            }),
            ..Default::default()
        };

        let scorer = AssetScorer::from_config(&config);

        // Asset that passes both
        let mut good = AssetData::new("GOOD");
        good.momentum_return = Some(0.15);
        good.price_earnings = Some(12.0);
        good.price_to_book = Some(1.5);

        // Asset that fails momentum
        let mut bad = AssetData::new("BAD");
        bad.momentum_return = Some(0.02);
        bad.price_earnings = Some(10.0);

        let ranked = scorer.score_and_rank(&[good, bad]);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].symbol, "GOOD");
    }
}
