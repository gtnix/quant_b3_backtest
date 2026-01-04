//! Selection module - picks top-N assets per market.
//!
//! Includes anti-concentration filters based on:
//! - Drawdown Beta (Ding & Uryasev, 2022)
//! - Return Correlation

use serde::{Deserialize, Serialize};

use crate::filters::Market;
use super::types::{EntryExclusion, ExclusionReason, ExclusionStage};

/// Configuration for asset selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionConfig {
    /// Number of top assets to select for BR market
    #[serde(default = "default_top_n")]
    pub top_n_br: usize,
    
    /// Number of top assets to select for US market
    #[serde(default = "default_top_n")]
    pub top_n_us: usize,
    
    /// Minimum score threshold (optional)
    #[serde(default)]
    pub min_score_threshold: Option<f64>,
    
    /// Maximum Drawdown Beta with existing portfolio (0.0 to 2.0).
    /// Assets with DD Beta > this threshold are excluded.
    /// Reference: Ding & Uryasev (2022) recommends 0.8.
    #[serde(default = "default_max_dd_beta")]
    pub max_drawdown_beta: f64,
    
    /// Maximum correlation with existing portfolio (0.0 to 1.0).
    /// Assets with correlation > this threshold are excluded.
    #[serde(default = "default_max_correlation")]
    pub max_correlation: f64,
    
    /// Enable Drawdown Beta anti-concentration filter
    #[serde(default)]
    pub check_drawdown_beta: bool,
    
    /// Enable correlation anti-concentration filter
    #[serde(default)]
    pub check_correlation: bool,
}

fn default_top_n() -> usize { 10 }
fn default_max_dd_beta() -> f64 { 0.8 }
fn default_max_correlation() -> f64 { 0.7 }

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            top_n_br: 10,
            top_n_us: 10,
            min_score_threshold: None,
            max_drawdown_beta: default_max_dd_beta(),
            max_correlation: default_max_correlation(),
            check_drawdown_beta: false,
            check_correlation: false,
        }
    }
}

/// Scored candidate for selection.
#[derive(Debug, Clone)]
pub struct ScoredCandidate {
    pub symbol: String,
    pub market: Market,
    pub score: f64,
    pub filter_scores: Vec<(String, f64)>,
    /// Historical daily drawdowns for Drawdown Beta calculation.
    /// Should be aligned with portfolio drawdowns (same dates).
    pub drawdowns: Option<Vec<f64>>,
    /// Historical daily returns for correlation calculation.
    pub returns: Option<Vec<f64>>,
}

impl ScoredCandidate {
    pub fn new(symbol: impl Into<String>, market: Market, score: f64) -> Self {
        Self {
            symbol: symbol.into(),
            market,
            score,
            filter_scores: Vec::new(),
            drawdowns: None,
            returns: None,
        }
    }

    pub fn with_filter_scores(mut self, scores: Vec<(String, f64)>) -> Self {
        self.filter_scores = scores;
        self
    }
    
    /// Add drawdown data for DD Beta calculation.
    pub fn with_drawdowns(mut self, drawdowns: Vec<f64>) -> Self {
        self.drawdowns = Some(drawdowns);
        self
    }
    
    /// Add return data for correlation calculation.
    pub fn with_returns(mut self, returns: Vec<f64>) -> Self {
        self.returns = Some(returns);
        self
    }
}

/// Selector for picking top-N assets.
#[derive(Debug, Clone)]
pub struct Selector {
    config: SelectionConfig,
}

impl Selector {
    pub fn new(config: SelectionConfig) -> Self {
        Self { config }
    }

    /// Select top-N assets per market.
    /// Returns (selected, excluded).
    pub fn select(&self, candidates: Vec<ScoredCandidate>) -> (Vec<ScoredCandidate>, Vec<EntryExclusion>) {
        self.select_with_portfolio(candidates, &[])
    }
    
    /// Select top-N assets with anti-concentration filters using existing portfolio drawdowns.
    /// 
    /// `portfolio_drawdowns`: Historical drawdowns of existing portfolio for DD Beta calculation.
    /// `portfolio_returns`: Historical returns of existing portfolio for correlation calculation.
    pub fn select_with_portfolio(
        &self, 
        candidates: Vec<ScoredCandidate>,
        portfolio_drawdowns: &[f64],
    ) -> (Vec<ScoredCandidate>, Vec<EntryExclusion>) {
        let mut selected = Vec::new();
        let mut excluded = Vec::new();

        // Separate by market
        let (br_candidates, us_candidates): (Vec<_>, Vec<_>) = candidates
            .into_iter()
            .partition(|c| c.market == Market::BR);

        // Process BR with anti-concentration
        self.select_market_with_anticonc(
            br_candidates, 
            self.config.top_n_br, 
            portfolio_drawdowns,
            &mut selected, 
            &mut excluded
        );

        // Process US with anti-concentration
        self.select_market_with_anticonc(
            us_candidates, 
            self.config.top_n_us, 
            portfolio_drawdowns,
            &mut selected, 
            &mut excluded
        );

        (selected, excluded)
    }

    fn select_market(
        &self,
        candidates: Vec<ScoredCandidate>,
        top_n: usize,
        selected: &mut Vec<ScoredCandidate>,
        excluded: &mut Vec<EntryExclusion>,
    ) {
        self.select_market_with_anticonc(candidates, top_n, &[], selected, excluded)
    }
    
    /// Select top-N from a market with anti-concentration filters.
    fn select_market_with_anticonc(
        &self,
        mut candidates: Vec<ScoredCandidate>,
        top_n: usize,
        portfolio_drawdowns: &[f64],
        selected: &mut Vec<ScoredCandidate>,
        excluded: &mut Vec<EntryExclusion>,
    ) {
        // Apply threshold filter first if configured
        if let Some(threshold) = self.config.min_score_threshold {
            let mut passing = Vec::with_capacity(candidates.len());
            for candidate in candidates {
                if candidate.score >= threshold {
                    passing.push(candidate);
                } else {
                    excluded.push(EntryExclusion {
                        symbol: candidate.symbol,
                        reason: ExclusionReason::BelowScoreThreshold,
                        stage: ExclusionStage::Selection,
                        score: Some(candidate.score),
                    });
                }
            }
            candidates = passing;
        }

        if candidates.is_empty() {
            return;
        }

        // Sort by score descending
        candidates.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Track selections for THIS market only
        let initial_selected = selected.len();
        
        // Incremental selection with anti-concentration filters
        let mut portfolio_dd: Vec<f64> = portfolio_drawdowns.to_vec();
        let mut portfolio_ret: Vec<f64> = Vec::new();
        
        for candidate in candidates {
            let market_selected = selected.len() - initial_selected;
            if market_selected >= top_n {
                excluded.push(EntryExclusion {
                    symbol: candidate.symbol,
                    reason: ExclusionReason::OutOfTopN,
                    stage: ExclusionStage::Selection,
                    score: Some(candidate.score),
                });
                continue;
            }

            // Check Drawdown Beta if enabled and data available
            if self.config.check_drawdown_beta {
                if let Some(ref asset_dd) = candidate.drawdowns {
                    if !portfolio_dd.is_empty() && !asset_dd.is_empty() {
                        let dd_beta = Self::calculate_drawdown_beta(asset_dd, &portfolio_dd);
                        if dd_beta > self.config.max_drawdown_beta {
                            excluded.push(EntryExclusion {
                                symbol: candidate.symbol,
                                reason: ExclusionReason::HighDrawdownBeta,
                                stage: ExclusionStage::Selection,
                                score: Some(candidate.score),
                            });
                            continue;
                        }
                    }
                }
            }

            // Check Correlation if enabled and data available
            if self.config.check_correlation {
                if let Some(ref asset_ret) = candidate.returns {
                    if !portfolio_ret.is_empty() && !asset_ret.is_empty() {
                        let corr = Self::calculate_correlation(asset_ret, &portfolio_ret);
                        if corr > self.config.max_correlation {
                            excluded.push(EntryExclusion {
                                symbol: candidate.symbol,
                                reason: ExclusionReason::HighCorrelation,
                                stage: ExclusionStage::Selection,
                                score: Some(candidate.score),
                            });
                            continue;
                        }
                    }
                }
            }

            // Update portfolio aggregates for next candidate check
            if let Some(ref asset_dd) = candidate.drawdowns {
                Self::update_portfolio_aggregate(&mut portfolio_dd, asset_dd);
            }
            if let Some(ref asset_ret) = candidate.returns {
                Self::update_portfolio_aggregate(&mut portfolio_ret, asset_ret);
            }

            selected.push(candidate);
        }
    }
    
    /// Calculate Drawdown Beta: Cov(DD_asset, DD_portfolio) / Var(DD_portfolio).
    /// Reference: Ding & Uryasev (2022)
    fn calculate_drawdown_beta(asset_dd: &[f64], portfolio_dd: &[f64]) -> f64 {
        let n = asset_dd.len().min(portfolio_dd.len());
        if n < 10 {
            return 0.0; // Need sufficient data
        }

        let mean_asset: f64 = asset_dd[..n].iter().sum::<f64>() / n as f64;
        let mean_port: f64 = portfolio_dd[..n].iter().sum::<f64>() / n as f64;

        let mut cov = 0.0;
        let mut var_port = 0.0;

        for i in 0..n {
            let diff_asset = asset_dd[i] - mean_asset;
            let diff_port = portfolio_dd[i] - mean_port;
            cov += diff_asset * diff_port;
            var_port += diff_port * diff_port;
        }

        if var_port.abs() < 1e-10 {
            return 0.0;
        }

        (cov / n as f64) / (var_port / n as f64)
    }
    
    /// Calculate Pearson correlation between two return series.
    fn calculate_correlation(series_a: &[f64], series_b: &[f64]) -> f64 {
        let n = series_a.len().min(series_b.len());
        if n < 10 {
            return 0.0;
        }

        let mean_a: f64 = series_a[..n].iter().sum::<f64>() / n as f64;
        let mean_b: f64 = series_b[..n].iter().sum::<f64>() / n as f64;

        let mut cov = 0.0;
        let mut var_a = 0.0;
        let mut var_b = 0.0;

        for i in 0..n {
            let diff_a = series_a[i] - mean_a;
            let diff_b = series_b[i] - mean_b;
            cov += diff_a * diff_b;
            var_a += diff_a * diff_a;
            var_b += diff_b * diff_b;
        }

        let std_a = (var_a / n as f64).sqrt();
        let std_b = (var_b / n as f64).sqrt();

        if std_a < 1e-10 || std_b < 1e-10 {
            return 0.0;
        }

        (cov / n as f64) / (std_a * std_b)
    }
    
    /// Update portfolio aggregate by averaging with new asset data.
    fn update_portfolio_aggregate(portfolio: &mut Vec<f64>, asset: &[f64]) {
        if portfolio.is_empty() {
            *portfolio = asset.to_vec();
        } else {
            let n = portfolio.len().min(asset.len());
            for i in 0..n {
                // Simple equal-weight average
                portfolio[i] = (portfolio[i] + asset[i]) / 2.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_top_n_br() {
        let config = SelectionConfig {
            top_n_br: 3,
            top_n_us: 3,
            min_score_threshold: None,
            ..Default::default()
        };
        let selector = Selector::new(config);

        let candidates = vec![
            ScoredCandidate::new("PETR4", Market::BR, 0.9),
            ScoredCandidate::new("VALE3", Market::BR, 0.85),
            ScoredCandidate::new("ITUB4", Market::BR, 0.8),
            ScoredCandidate::new("BBDC4", Market::BR, 0.75),
            ScoredCandidate::new("ABEV3", Market::BR, 0.7),
        ];

        let (selected, excluded) = selector.select(candidates);

        assert_eq!(selected.len(), 3);
        assert_eq!(excluded.len(), 2);
        
        // Top 3 should be selected
        let selected_symbols: Vec<_> = selected.iter().map(|s| s.symbol.as_str()).collect();
        assert!(selected_symbols.contains(&"PETR4"));
        assert!(selected_symbols.contains(&"VALE3"));
        assert!(selected_symbols.contains(&"ITUB4"));

        // Others excluded with OutOfTopN
        assert_eq!(excluded[0].reason, ExclusionReason::OutOfTopN);
    }

    #[test]
    fn test_select_mixed_markets() {
        let config = SelectionConfig {
            top_n_br: 2,
            top_n_us: 2,
            min_score_threshold: None,
            ..Default::default()
        };
        let selector = Selector::new(config);

        let candidates = vec![
            ScoredCandidate::new("PETR4", Market::BR, 0.9),
            ScoredCandidate::new("VALE3", Market::BR, 0.85),
            ScoredCandidate::new("ITUB4", Market::BR, 0.8),
            ScoredCandidate::new("AAPL", Market::US, 0.95),
            ScoredCandidate::new("MSFT", Market::US, 0.88),
            ScoredCandidate::new("GOOG", Market::US, 0.82),
        ];

        let (selected, excluded) = selector.select(candidates);

        // 2 BR + 2 US = 4
        assert_eq!(selected.len(), 4);
        assert_eq!(excluded.len(), 2);
    }

    #[test]
    fn test_threshold_filter() {
        let config = SelectionConfig {
            top_n_br: 10,
            top_n_us: 10,
            min_score_threshold: Some(0.5),
            ..Default::default()
        };
        let selector = Selector::new(config);

        let candidates = vec![
            ScoredCandidate::new("PETR4", Market::BR, 0.9),
            ScoredCandidate::new("VALE3", Market::BR, 0.4), // Below threshold
            ScoredCandidate::new("ITUB4", Market::BR, 0.3), // Below threshold
        ];

        let (selected, excluded) = selector.select(candidates);

        assert_eq!(selected.len(), 1);
        assert_eq!(excluded.len(), 2);
        assert!(excluded.iter().all(|e| e.reason == ExclusionReason::BelowScoreThreshold));
    }

    #[test]
    fn test_fewer_candidates_than_top_n() {
        let config = SelectionConfig {
            top_n_br: 10,
            top_n_us: 10,
            min_score_threshold: None,
            ..Default::default()
        };
        let selector = Selector::new(config);

        let candidates = vec![
            ScoredCandidate::new("PETR4", Market::BR, 0.9),
            ScoredCandidate::new("VALE3", Market::BR, 0.85),
        ];

        let (selected, excluded) = selector.select(candidates);

        // Only 2 available, both selected
        assert_eq!(selected.len(), 2);
        assert!(excluded.is_empty());
    }

    #[test]
    fn test_empty_candidates() {
        let selector = Selector::new(SelectionConfig::default());
        let (selected, excluded) = selector.select(Vec::new());

        assert!(selected.is_empty());
        assert!(excluded.is_empty());
    }

    #[test]
    fn test_drawdown_beta_calculation() {
        // Perfect correlation: beta = 1.0
        let asset_dd = vec![0.05, 0.10, 0.15, 0.08, 0.03, 0.02, 0.06, 0.09, 0.12, 0.07];
        let portfolio_dd = asset_dd.clone();
        
        let beta = Selector::calculate_drawdown_beta(&asset_dd, &portfolio_dd);
        assert!((beta - 1.0).abs() < 0.01, "Beta should be ~1.0: {}", beta);
    }
    
    #[test]
    fn test_drawdown_beta_scaled() {
        // Asset DD = 2x portfolio DD
        let portfolio_dd: Vec<f64> = vec![0.05, 0.10, 0.15, 0.08, 0.03, 0.02, 0.06, 0.09, 0.12, 0.07];
        let asset_dd: Vec<f64> = portfolio_dd.iter().map(|x| x * 2.0).collect();
        
        let beta = Selector::calculate_drawdown_beta(&asset_dd, &portfolio_dd);
        assert!((beta - 2.0).abs() < 0.01, "Beta should be ~2.0: {}", beta);
    }
    
    #[test]
    fn test_high_dd_beta_exclusion() {
        let config = SelectionConfig {
            top_n_br: 10,
            max_drawdown_beta: 0.8,
            check_drawdown_beta: true,
            ..Default::default()
        };
        let selector = Selector::new(config);

        // Create portfolio drawdowns
        let portfolio_dd: Vec<f64> = vec![0.05, 0.10, 0.15, 0.08, 0.03, 0.02, 0.06, 0.09, 0.12, 0.07];
        
        // Candidate with high DD correlation (beta > 0.8)
        let high_dd_asset: Vec<f64> = portfolio_dd.iter().map(|x| x * 1.2).collect();
        
        let candidates = vec![
            ScoredCandidate::new("HIGH_DD", Market::BR, 0.9)
                .with_drawdowns(high_dd_asset),
        ];

        let (selected, excluded) = selector.select_with_portfolio(candidates, &portfolio_dd);

        assert_eq!(selected.len(), 0);
        assert_eq!(excluded.len(), 1);
        assert_eq!(excluded[0].reason, ExclusionReason::HighDrawdownBeta);
    }
    
    #[test]
    fn test_low_dd_beta_passes() {
        let config = SelectionConfig {
            top_n_br: 10,
            max_drawdown_beta: 0.8,
            check_drawdown_beta: true,
            ..Default::default()
        };
        let selector = Selector::new(config);

        let portfolio_dd: Vec<f64> = vec![0.05, 0.10, 0.15, 0.08, 0.03, 0.02, 0.06, 0.09, 0.12, 0.07];
        
        // Candidate with low DD correlation (inverted pattern)
        let low_dd_asset: Vec<f64> = portfolio_dd.iter().rev().copied().collect();
        
        let candidates = vec![
            ScoredCandidate::new("LOW_DD", Market::BR, 0.9)
                .with_drawdowns(low_dd_asset),
        ];

        let (selected, excluded) = selector.select_with_portfolio(candidates, &portfolio_dd);

        assert_eq!(selected.len(), 1);
        assert!(excluded.is_empty());
    }
    
    #[test]
    fn test_correlation_calculation() {
        // Perfect correlation
        let a = vec![0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.01, 0.01, 0.02];
        let b = a.clone();
        
        let corr = Selector::calculate_correlation(&a, &b);
        assert!((corr - 1.0).abs() < 0.01, "Correlation should be ~1.0: {}", corr);
    }
    
    #[test]
    fn test_high_correlation_exclusion() {
        let config = SelectionConfig {
            top_n_br: 10,
            max_correlation: 0.7,
            check_correlation: true,
            ..Default::default()
        };
        let selector = Selector::new(config);

        // First candidate will be selected and set the portfolio returns
        let base_returns: Vec<f64> = vec![0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.01, 0.01, 0.02];
        
        // Second candidate with high correlation
        let high_corr_returns = base_returns.clone();
        
        let candidates = vec![
            ScoredCandidate::new("FIRST", Market::BR, 0.95)
                .with_returns(base_returns),
            ScoredCandidate::new("HIGH_CORR", Market::BR, 0.90)
                .with_returns(high_corr_returns),
        ];

        let (selected, excluded) = selector.select(candidates);

        // First should be selected, second excluded due to high correlation
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].symbol, "FIRST");
        assert_eq!(excluded.len(), 1);
        assert_eq!(excluded[0].reason, ExclusionReason::HighCorrelation);
    }
}

