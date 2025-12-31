//! Size filter - selects assets by market capitalization.

use super::{AssetData, AssetFilter, FilterResult};
use crate::config::SizeConfig;

/// Size filter based on market capitalization.
pub struct SizeFilter {
    config: SizeConfig,
}

impl SizeFilter {
    pub fn new(config: SizeConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(SizeConfig::default())
    }
}

impl AssetFilter for SizeFilter {
    fn name(&self) -> &str {
        "size"
    }

    fn weight(&self) -> f64 {
        self.config.base.weight
    }

    fn evaluate(&self, data: &AssetData) -> FilterResult {
        let cap = match data.market_cap {
            Some(v) => v,
            None => {
                return FilterResult::fail("No market cap data available");
            }
        };

        if cap < self.config.min_market_cap {
            return FilterResult::fail(format!(
                "Market cap R${:.2}B below minimum R${:.2}B",
                cap as f64 / 1e9,
                self.config.min_market_cap as f64 / 1e9
            ))
            .with_metric(cap as f64);
        }

        if let Some(max) = self.config.max_market_cap {
            if cap > max {
                return FilterResult::fail(format!(
                    "Market cap R${:.2}B above maximum R${:.2}B",
                    cap as f64 / 1e9,
                    max as f64 / 1e9
                ))
                .with_metric(cap as f64);
            }
        }

        // Score: logarithmic scale for market cap
        // Large caps get slightly higher scores for liquidity
        let log_cap = (cap as f64).ln();
        let log_min = (self.config.min_market_cap as f64).ln();
        let log_max = self
            .config
            .max_market_cap
            .map(|v| (v as f64).ln())
            .unwrap_or_else(|| (1e12_f64).ln()); // R$ 1 trillion max

        let score = ((log_cap - log_min) / (log_max - log_min)).clamp(0.0, 1.0);

        FilterResult::pass(score, format!("Market cap R${:.2}B", cap as f64 / 1e9))
            .with_metric(cap as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_pass() {
        let filter = SizeFilter::with_defaults();
        let mut data = AssetData::new("TEST");
        data.market_cap = Some(10_000_000_000); // R$ 10B

        let result = filter.evaluate(&data);
        assert!(result.passed);
    }

    #[test]
    fn test_size_fail_small() {
        let filter = SizeFilter::with_defaults();
        let mut data = AssetData::new("TEST");
        data.market_cap = Some(1_000_000_000); // R$ 1B, below 5B

        let result = filter.evaluate(&data);
        assert!(!result.passed);
    }
}





















