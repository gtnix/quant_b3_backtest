//! Low volatility filter - selects less volatile assets.

use super::{AssetData, AssetFilter, FilterResult};
use crate::config::LowVolConfig;

/// Low volatility filter based on annualized volatility.
pub struct LowVolFilter {
    config: LowVolConfig,
}

impl LowVolFilter {
    pub fn new(config: LowVolConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(LowVolConfig::default())
    }
}

impl AssetFilter for LowVolFilter {
    fn name(&self) -> &str {
        "low_vol"
    }

    fn weight(&self) -> f64 {
        self.config.base.weight
    }

    fn evaluate(&self, data: &AssetData) -> FilterResult {
        let vol = match data.annualized_volatility {
            Some(v) => v,
            None => {
                return FilterResult::fail("No volatility data available");
            }
        };

        if vol > self.config.max_annualized_vol {
            return FilterResult::fail(format!(
                "Volatility {:.2}% above maximum {:.2}%",
                vol * 100.0,
                self.config.max_annualized_vol * 100.0
            ))
            .with_metric(vol);
        }

        // Score: lower volatility = higher score
        let score = 1.0 - (vol / self.config.max_annualized_vol).min(1.0);

        FilterResult::pass(score, format!("Volatility {:.2}%", vol * 100.0)).with_metric(vol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_low_vol_pass() {
        let filter = LowVolFilter::with_defaults();
        let mut data = AssetData::new("TEST");
        data.annualized_volatility = Some(0.20);

        let result = filter.evaluate(&data);
        assert!(result.passed);
        assert!(result.score > 0.0);
    }

    #[test]
    fn test_low_vol_fail_high_vol() {
        let filter = LowVolFilter::with_defaults();
        let mut data = AssetData::new("TEST");
        data.annualized_volatility = Some(0.40); // Above 30%

        let result = filter.evaluate(&data);
        assert!(!result.passed);
    }
}
















