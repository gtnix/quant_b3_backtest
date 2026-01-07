//! Quality filter - selects high-quality companies.

use super::{AssetData, AssetFilter, FilterResult};
use crate::config::QualityConfig;

/// Quality filter based on ROE, debt levels, and margins.
pub struct QualityFilter {
    config: QualityConfig,
}

impl QualityFilter {
    pub fn new(config: QualityConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(QualityConfig::default())
    }
}

impl AssetFilter for QualityFilter {
    fn name(&self) -> &str {
        "quality"
    }

    fn weight(&self) -> f64 {
        self.config.base.weight
    }

    fn evaluate(&self, data: &AssetData) -> FilterResult {
        // Check ROE
        if let Some(roe) = data.return_on_equity {
            if roe < self.config.min_roe {
                return FilterResult::fail(format!(
                    "ROE {:.2}% below minimum {:.2}%",
                    roe * 100.0,
                    self.config.min_roe * 100.0
                ))
                .with_metric(roe);
            }
        }

        // Check Debt/Equity
        if let Some(de) = data.debt_to_equity {
            if de > self.config.max_debt_equity {
                return FilterResult::fail(format!(
                    "D/E {:.2} above maximum {:.2}",
                    de, self.config.max_debt_equity
                ))
                .with_metric(de);
            }
        }

        // Check profit margin
        if let Some(pm) = data.profit_margins {
            if pm < self.config.min_profit_margin {
                return FilterResult::fail(format!(
                    "Profit margin {:.2}% below minimum {:.2}%",
                    pm * 100.0,
                    self.config.min_profit_margin * 100.0
                ))
                .with_metric(pm);
            }
        }

        // Check gross margin if configured
        if let Some(min_gm) = self.config.min_gross_margin {
            if let Some(gm) = data.gross_margins {
                if gm < min_gm {
                    return FilterResult::fail(format!(
                        "Gross margin {:.2}% below minimum {:.2}%",
                        gm * 100.0,
                        min_gm * 100.0
                    ))
                    .with_metric(gm);
                }
            }
        }

        // Calculate composite score
        let roe_score = data
            .return_on_equity
            .map(|v| (v / 0.30).min(1.0).max(0.0)) // 30% ROE = max score
            .unwrap_or(0.5);

        let de_score = data
            .debt_to_equity
            .map(|v| 1.0 - (v / 2.0).min(1.0)) // 0 D/E = max, 2+ = 0
            .unwrap_or(0.5);

        let pm_score = data
            .profit_margins
            .map(|v| (v / 0.20).min(1.0).max(0.0)) // 20% margin = max
            .unwrap_or(0.5);

        let score = (roe_score + de_score + pm_score) / 3.0;

        FilterResult::pass(
            score,
            format!(
                "ROE: {:?}, D/E: {:?}, PM: {:?}",
                data.return_on_equity.map(|v| format!("{:.1}%", v * 100.0)),
                data.debt_to_equity.map(|v| format!("{:.2}", v)),
                data.profit_margins.map(|v| format!("{:.1}%", v * 100.0))
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_pass() {
        let filter = QualityFilter::with_defaults();
        let mut data = AssetData::new("TEST");
        data.return_on_equity = Some(0.20);
        data.debt_to_equity = Some(0.5);
        data.profit_margins = Some(0.10);

        let result = filter.evaluate(&data);
        assert!(result.passed);
    }

    #[test]
    fn test_quality_fail_low_roe() {
        let filter = QualityFilter::with_defaults();
        let mut data = AssetData::new("TEST");
        data.return_on_equity = Some(0.05); // Below 12%

        let result = filter.evaluate(&data);
        assert!(!result.passed);
    }
}




































