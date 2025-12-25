//! Selection module - picks top-N assets per market.

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
}

fn default_top_n() -> usize { 10 }

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            top_n_br: 10,
            top_n_us: 10,
            min_score_threshold: None,
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
}

impl ScoredCandidate {
    pub fn new(symbol: impl Into<String>, market: Market, score: f64) -> Self {
        Self {
            symbol: symbol.into(),
            market,
            score,
            filter_scores: Vec::new(),
        }
    }

    pub fn with_filter_scores(mut self, scores: Vec<(String, f64)>) -> Self {
        self.filter_scores = scores;
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
        let mut selected = Vec::new();
        let mut excluded = Vec::new();

        // Separate by market
        let (br_candidates, us_candidates): (Vec<_>, Vec<_>) = candidates
            .into_iter()
            .partition(|c| c.market == Market::BR);

        // Process BR
        self.select_market(br_candidates, self.config.top_n_br, &mut selected, &mut excluded);

        // Process US
        self.select_market(us_candidates, self.config.top_n_us, &mut selected, &mut excluded);

        (selected, excluded)
    }

    fn select_market(
        &self,
        mut candidates: Vec<ScoredCandidate>,
        top_n: usize,
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

        // Optimized top-K selection using partial sort: O(N + K log K)
        // Instead of sorting all N elements, we:
        // 1. Partition around the K-th element: O(N)
        // 2. Sort only the top K elements: O(K log K)
        let k = top_n.min(candidates.len());
        
        if k > 0 && k < candidates.len() {
            // select_nth_unstable_by partitions around the k-th element
            // Elements before k will be >= elements after k (for descending)
            candidates.select_nth_unstable_by(k - 1, |a, b| {
                b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
            });
            
            // Now sort only the top-k portion for stable ordering
            candidates[..k].sort_by(|a, b| {
                b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            // k == candidates.len(), sort all
            candidates.sort_by(|a, b| {
                b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Split into selected and excluded
        for (i, candidate) in candidates.into_iter().enumerate() {
            if i < top_n {
                selected.push(candidate);
            } else {
                excluded.push(EntryExclusion {
                    symbol: candidate.symbol,
                    reason: ExclusionReason::OutOfTopN,
                    stage: ExclusionStage::Selection,
                    score: Some(candidate.score),
                });
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
}

