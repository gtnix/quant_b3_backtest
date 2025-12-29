//! Momentum filter - selects assets with strong recent performance.

use super::{AssetData, AssetFilter, FilterResult};
use crate::config::MomentumConfig;

/// Momentum filter based on price returns.
pub struct MomentumFilter {
    config: MomentumConfig,
}

impl MomentumFilter {
    pub fn new(config: MomentumConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(MomentumConfig::default())
    }
}

impl AssetFilter for MomentumFilter {
    fn name(&self) -> &str {
        "momentum"
    }

    fn weight(&self) -> f64 {
        self.config.base.weight
    }

    fn evaluate(&self, data: &AssetData) -> FilterResult {
        let momentum = match data.momentum_return {
            Some(m) => m,
            None => {
                return FilterResult::fail("No momentum data available");
            }
        };

        if momentum < self.config.min_return {
            return FilterResult::fail(format!(
                "Momentum {:.2}% below minimum {:.2}%",
                momentum * 100.0,
                self.config.min_return * 100.0
            ))
            .with_metric(momentum);
        }

        // Score: normalize momentum to 0-1 range
        // Assume reasonable range is -50% to +100%
        let score = ((momentum + 0.5) / 1.5).clamp(0.0, 1.0);

        FilterResult::pass(score, format!("Momentum {:.2}%", momentum * 100.0))
            .with_metric(momentum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_momentum_pass() {
        let filter = MomentumFilter::with_defaults();
        let mut data = AssetData::new("TEST");
        data.momentum_return = Some(0.15);

        let result = filter.evaluate(&data);
        assert!(result.passed);
        assert!(result.score > 0.0);
    }

    #[test]
    fn test_momentum_fail_below_min() {
        let config = MomentumConfig {
            min_return: 0.10,
            ..Default::default()
        };
        let filter = MomentumFilter::new(config);

        let mut data = AssetData::new("TEST");
        data.momentum_return = Some(0.05);

        let result = filter.evaluate(&data);
        assert!(!result.passed);
    }
}
















