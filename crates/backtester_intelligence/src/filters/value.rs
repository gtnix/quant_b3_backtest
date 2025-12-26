//! Value filter - selects undervalued assets.

use super::{AssetData, AssetFilter, FilterResult};
use crate::config::ValueConfig;

/// Value filter based on P/E and P/B ratios.
pub struct ValueFilter {
    config: ValueConfig,
}

impl ValueFilter {
    pub fn new(config: ValueConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(ValueConfig::default())
    }
}

impl AssetFilter for ValueFilter {
    fn name(&self) -> &str {
        "value"
    }

    fn weight(&self) -> f64 {
        self.config.base.weight
    }

    fn evaluate(&self, data: &AssetData) -> FilterResult {
        let pe = data.price_earnings;
        let pb = data.price_to_book;

        // Check P/E
        if let Some(pe_val) = pe {
            if pe_val < self.config.min_pe {
                return FilterResult::fail(format!(
                    "P/E {:.2} below minimum {:.2} (negative earnings)",
                    pe_val, self.config.min_pe
                ))
                .with_metric(pe_val);
            }
            if pe_val > self.config.max_pe {
                return FilterResult::fail(format!(
                    "P/E {:.2} above maximum {:.2}",
                    pe_val, self.config.max_pe
                ))
                .with_metric(pe_val);
            }
        }

        // Check P/B
        if let Some(pb_val) = pb {
            if pb_val > self.config.max_pb {
                return FilterResult::fail(format!(
                    "P/B {:.2} above maximum {:.2}",
                    pb_val, self.config.max_pb
                ))
                .with_metric(pb_val);
            }
        }

        // Calculate score - lower ratios = higher score
        let pe_score = pe
            .map(|v| {
                if v <= 0.0 {
                    0.0
                } else {
                    1.0 - (v / self.config.max_pe).min(1.0)
                }
            })
            .unwrap_or(0.5);

        let pb_score = pb
            .map(|v| {
                if v <= 0.0 {
                    0.0
                } else {
                    1.0 - (v / self.config.max_pb).min(1.0)
                }
            })
            .unwrap_or(0.5);

        let score = (pe_score + pb_score) / 2.0;

        FilterResult::pass(score, format!("P/E: {:?}, P/B: {:?}", pe, pb))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_pass() {
        let filter = ValueFilter::with_defaults();
        let mut data = AssetData::new("TEST");
        data.price_earnings = Some(10.0);
        data.price_to_book = Some(1.5);

        let result = filter.evaluate(&data);
        assert!(result.passed);
    }

    #[test]
    fn test_value_fail_high_pe() {
        let filter = ValueFilter::with_defaults();
        let mut data = AssetData::new("TEST");
        data.price_earnings = Some(25.0); // Above max 15

        let result = filter.evaluate(&data);
        assert!(!result.passed);
    }
}





