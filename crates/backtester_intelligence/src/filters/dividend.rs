//! Dividend yield filter - selects dividend-paying assets.

use super::{AssetData, AssetFilter, FilterResult};
use crate::config::DividendYieldConfig;

/// Dividend yield filter.
pub struct DividendYieldFilter {
    config: DividendYieldConfig,
}

impl DividendYieldFilter {
    pub fn new(config: DividendYieldConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(DividendYieldConfig::default())
    }
}

impl AssetFilter for DividendYieldFilter {
    fn name(&self) -> &str {
        "dividend_yield"
    }

    fn weight(&self) -> f64 {
        self.config.base.weight
    }

    fn evaluate(&self, data: &AssetData) -> FilterResult {
        let dy = match data.dividend_yield {
            Some(v) => v,
            None => {
                return FilterResult::fail("No dividend yield data available");
            }
        };

        if dy < self.config.min_yield {
            return FilterResult::fail(format!(
                "Dividend yield {:.2}% below minimum {:.2}%",
                dy * 100.0,
                self.config.min_yield * 100.0
            ))
            .with_metric(dy);
        }

        // Check max yield (value trap protection)
        if let Some(max) = self.config.max_yield {
            if dy > max {
                return FilterResult::fail(format!(
                    "Dividend yield {:.2}% above maximum {:.2}% (potential value trap)",
                    dy * 100.0,
                    max * 100.0
                ))
                .with_metric(dy);
            }
        }

        // Score: higher yield = higher score (up to reasonable max)
        let max_for_score = self.config.max_yield.unwrap_or(0.15);
        let score = ((dy - self.config.min_yield) / (max_for_score - self.config.min_yield))
            .clamp(0.0, 1.0);

        FilterResult::pass(score, format!("Dividend yield {:.2}%", dy * 100.0)).with_metric(dy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dividend_pass() {
        let filter = DividendYieldFilter::with_defaults();
        let mut data = AssetData::new("TEST");
        data.dividend_yield = Some(0.05);

        let result = filter.evaluate(&data);
        assert!(result.passed);
    }

    #[test]
    fn test_dividend_fail_low() {
        let filter = DividendYieldFilter::with_defaults();
        let mut data = AssetData::new("TEST");
        data.dividend_yield = Some(0.01); // Below 3%

        let result = filter.evaluate(&data);
        assert!(!result.passed);
    }
}




























