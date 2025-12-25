//! Weighting module - calculates position weights using risk-parity.
//!
//! Risk-parity formula: weight_i = (1/vol_i) / sum(1/vol_j)
//! Assets with lower volatility receive higher weights.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Weighting method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum WeightingMethod {
    /// Equal weight for all selected assets
    EqualWeight,
    /// Weight proportional to score
    ScoreProportional,
    /// Weight inversely proportional to volatility (risk-parity)
    #[default]
    RiskParity,
}

/// Configuration for weighting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightingConfig {
    /// Weighting method
    #[serde(default)]
    pub method: WeightingMethod,
    
    /// Volatility lookback in days (for risk-parity)
    #[serde(default = "default_vol_lookback")]
    pub vol_lookback: usize,
    
    /// Maximum weight per asset (e.g., 0.20 for 20%)
    #[serde(default = "default_max_weight")]
    pub max_weight: f64,
    
    /// Minimum weight per asset (e.g., 0.02 for 2%)
    #[serde(default = "default_min_weight")]
    pub min_weight: f64,
    
    /// Default volatility if not available (annualized)
    #[serde(default = "default_fallback_vol")]
    pub fallback_volatility: f64,
}

fn default_vol_lookback() -> usize { 20 }
fn default_max_weight() -> f64 { 0.20 }
fn default_min_weight() -> f64 { 0.02 }
fn default_fallback_vol() -> f64 { 0.30 } // 30% annualized

impl Default for WeightingConfig {
    fn default() -> Self {
        Self {
            method: WeightingMethod::RiskParity,
            vol_lookback: default_vol_lookback(),
            max_weight: default_max_weight(),
            min_weight: default_min_weight(),
            fallback_volatility: default_fallback_vol(),
        }
    }
}

/// Asset with volatility data for weighting.
#[derive(Debug, Clone)]
pub struct WeightingCandidate {
    pub symbol: String,
    pub score: f64,
    pub volatility: Option<f64>,
}

impl WeightingCandidate {
    pub fn new(symbol: impl Into<String>, score: f64, volatility: Option<f64>) -> Self {
        Self {
            symbol: symbol.into(),
            score,
            volatility,
        }
    }
}

/// Result of weighting calculation.
#[derive(Debug, Clone)]
pub struct WeightResult {
    pub symbol: String,
    pub weight: f64,
    pub raw_weight: f64,
    pub volatility: f64,
    pub capped: bool,
}

/// Weighter for calculating position weights.
#[derive(Debug, Clone)]
pub struct Weighter {
    config: WeightingConfig,
}

impl Weighter {
    pub fn new(config: WeightingConfig) -> Self {
        Self { config }
    }

    /// Calculate weights for selected assets.
    pub fn calculate_weights(&self, candidates: Vec<WeightingCandidate>) -> Vec<WeightResult> {
        if candidates.is_empty() {
            return Vec::new();
        }

        match self.config.method {
            WeightingMethod::EqualWeight => self.equal_weight(candidates),
            WeightingMethod::ScoreProportional => self.score_proportional(candidates),
            WeightingMethod::RiskParity => self.risk_parity(candidates),
        }
    }

    fn equal_weight(&self, candidates: Vec<WeightingCandidate>) -> Vec<WeightResult> {
        let n = candidates.len() as f64;
        let raw_weight = 1.0 / n;

        candidates
            .into_iter()
            .map(|c| WeightResult {
                symbol: c.symbol,
                weight: raw_weight.clamp(self.config.min_weight, self.config.max_weight),
                raw_weight,
                volatility: c.volatility.unwrap_or(self.config.fallback_volatility),
                capped: raw_weight > self.config.max_weight,
            })
            .collect()
    }

    fn score_proportional(&self, candidates: Vec<WeightingCandidate>) -> Vec<WeightResult> {
        let total_score: f64 = candidates.iter().map(|c| c.score).sum();
        
        if total_score <= 0.0 {
            return self.equal_weight(candidates);
        }

        let mut results: Vec<_> = candidates
            .into_iter()
            .map(|c| {
                let raw_weight = c.score / total_score;
                WeightResult {
                    symbol: c.symbol,
                    weight: raw_weight,
                    raw_weight,
                    volatility: c.volatility.unwrap_or(self.config.fallback_volatility),
                    capped: false,
                }
            })
            .collect();

        self.apply_caps_and_normalize(&mut results);
        results
    }

    fn risk_parity(&self, candidates: Vec<WeightingCandidate>) -> Vec<WeightResult> {
        // Calculate inverse volatility for each asset
        let inverse_vols: Vec<(usize, f64, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let vol = c.volatility.unwrap_or(self.config.fallback_volatility);
                // Ensure minimum volatility to avoid division issues
                let safe_vol = vol.max(0.01);
                (i, 1.0 / safe_vol, vol)
            })
            .collect();

        let total_inverse_vol: f64 = inverse_vols.iter().map(|(_, inv, _)| inv).sum();

        if total_inverse_vol <= 0.0 {
            return self.equal_weight(candidates);
        }

        let mut results: Vec<WeightResult> = candidates
            .into_iter()
            .zip(inverse_vols.iter())
            .map(|(c, (_, inv_vol, vol))| {
                let raw_weight = inv_vol / total_inverse_vol;
                WeightResult {
                    symbol: c.symbol,
                    weight: raw_weight,
                    raw_weight,
                    volatility: *vol,
                    capped: false,
                }
            })
            .collect();

        self.apply_caps_and_normalize(&mut results);
        results
    }

    /// Apply min/max caps and re-normalize weights.
    fn apply_caps_and_normalize(&self, results: &mut Vec<WeightResult>) {
        // First pass: apply caps
        let mut total_capped = 0.0;
        let mut uncapped_total = 0.0;
        
        for r in results.iter_mut() {
            if r.weight > self.config.max_weight {
                r.capped = true;
                r.weight = self.config.max_weight;
                total_capped += self.config.max_weight;
            } else if r.weight < self.config.min_weight {
                r.capped = true;
                r.weight = self.config.min_weight;
                total_capped += self.config.min_weight;
            } else {
                uncapped_total += r.weight;
            }
        }

        // Redistribute if needed (iterative approach)
        let target_total = 1.0;
        let current_total: f64 = results.iter().map(|r| r.weight).sum();
        
        if (current_total - target_total).abs() > 0.001 {
            // Simple rescaling of uncapped weights
            let scale = if uncapped_total > 0.0 {
                (target_total - total_capped) / uncapped_total
            } else {
                1.0
            };

            for r in results.iter_mut() {
                if !r.capped {
                    r.weight *= scale;
                    // Re-check bounds after scaling
                    r.weight = r.weight.clamp(self.config.min_weight, self.config.max_weight);
                }
            }
        }

        // Final normalization to ensure sum = 1.0
        let total: f64 = results.iter().map(|r| r.weight).sum();
        if total > 0.0 && (total - 1.0).abs() > 0.001 {
            for r in results.iter_mut() {
                r.weight /= total;
            }
        }
    }

    /// Get weights as a HashMap.
    pub fn weights_map(&self, results: &[WeightResult]) -> HashMap<String, f64> {
        results.iter().map(|r| (r.symbol.clone(), r.weight)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equal_weight() {
        let config = WeightingConfig {
            method: WeightingMethod::EqualWeight,
            max_weight: 0.50, // Increase max to allow 33%
            ..Default::default()
        };
        let weighter = Weighter::new(config);

        let candidates = vec![
            WeightingCandidate::new("A", 0.8, Some(0.20)),
            WeightingCandidate::new("B", 0.7, Some(0.25)),
            WeightingCandidate::new("C", 0.6, Some(0.15)),
        ];

        let results = weighter.calculate_weights(candidates);
        
        // Each should have ~33% weight
        for r in &results {
            assert!((r.weight - 0.333).abs() < 0.02, "weight {} should be ~0.333", r.weight);
        }
    }

    #[test]
    fn test_risk_parity_lower_vol_higher_weight() {
        let config = WeightingConfig {
            method: WeightingMethod::RiskParity,
            max_weight: 0.90, // Increase cap for this test
            min_weight: 0.05,
            ..Default::default()
        };
        let weighter = Weighter::new(config);

        let candidates = vec![
            WeightingCandidate::new("LOW_VOL", 0.5, Some(0.10)),  // 10% vol → 1/0.10 = 10
            WeightingCandidate::new("HIGH_VOL", 0.5, Some(0.30)), // 30% vol → 1/0.30 = 3.33
        ];
        // Total inverse = 13.33
        // LOW_VOL weight = 10/13.33 = 0.75 (75%)
        // HIGH_VOL weight = 3.33/13.33 = 0.25 (25%)

        let results = weighter.calculate_weights(candidates);
        
        let low_vol_weight = results.iter().find(|r| r.symbol == "LOW_VOL").unwrap().weight;
        let high_vol_weight = results.iter().find(|r| r.symbol == "HIGH_VOL").unwrap().weight;

        // Low vol should have higher weight (75% vs 25%)
        assert!(low_vol_weight > high_vol_weight, 
            "Low vol ({:.2}) should have higher weight than high vol ({:.2})", 
            low_vol_weight, high_vol_weight);
    }

    #[test]
    fn test_weight_cap_applied() {
        let config = WeightingConfig {
            method: WeightingMethod::RiskParity,
            max_weight: 0.20,
            min_weight: 0.02,
            ..Default::default()
        };
        let max_weight = config.max_weight;
        let weighter = Weighter::new(config);

        // One very low vol should try to get very high weight
        let candidates = vec![
            WeightingCandidate::new("VERY_LOW_VOL", 0.8, Some(0.05)), // 5% vol - would get 80%+
            WeightingCandidate::new("HIGH_VOL_1", 0.5, Some(0.40)),
            WeightingCandidate::new("HIGH_VOL_2", 0.5, Some(0.40)),
            WeightingCandidate::new("HIGH_VOL_3", 0.5, Some(0.40)),
            WeightingCandidate::new("HIGH_VOL_4", 0.5, Some(0.40)),
        ];

        let results = weighter.calculate_weights(candidates);

        // No weight should exceed 20%
        for r in &results {
            assert!(r.weight <= max_weight + 0.001, 
                "{} weight {} exceeds max {}", r.symbol, r.weight, max_weight);
        }
    }

    #[test]
    fn test_weights_sum_to_one() {
        let weighter = Weighter::new(WeightingConfig::default());

        let candidates = vec![
            WeightingCandidate::new("A", 0.9, Some(0.15)),
            WeightingCandidate::new("B", 0.8, Some(0.20)),
            WeightingCandidate::new("C", 0.7, Some(0.25)),
            WeightingCandidate::new("D", 0.6, Some(0.30)),
            WeightingCandidate::new("E", 0.5, Some(0.18)),
        ];

        let results = weighter.calculate_weights(candidates);
        let total: f64 = results.iter().map(|r| r.weight).sum();

        assert!((total - 1.0).abs() < 0.01, "Total weight {} should be ~1.0", total);
    }

    #[test]
    fn test_fallback_volatility() {
        let config = WeightingConfig {
            method: WeightingMethod::RiskParity,
            fallback_volatility: 0.25,
            ..Default::default()
        };
        let weighter = Weighter::new(config);

        let candidates = vec![
            WeightingCandidate::new("HAS_VOL", 0.8, Some(0.20)),
            WeightingCandidate::new("NO_VOL", 0.8, None), // Uses fallback
        ];

        let results = weighter.calculate_weights(candidates);
        
        let no_vol = results.iter().find(|r| r.symbol == "NO_VOL").unwrap();
        assert_eq!(no_vol.volatility, 0.25);
    }

    #[test]
    fn test_score_proportional() {
        let config = WeightingConfig {
            method: WeightingMethod::ScoreProportional,
            max_weight: 0.90, // Allow proportional weights
            min_weight: 0.01,
            ..Default::default()
        };
        let weighter = Weighter::new(config);

        let candidates = vec![
            WeightingCandidate::new("HIGH", 0.80, Some(0.20)), // 80% of total score
            WeightingCandidate::new("LOW", 0.20, Some(0.20)),  // 20% of total score
        ];
        // Total score = 1.0, HIGH = 80%, LOW = 20%

        let results = weighter.calculate_weights(candidates);
        
        let high_weight = results.iter().find(|r| r.symbol == "HIGH").unwrap().weight;
        let low_weight = results.iter().find(|r| r.symbol == "LOW").unwrap().weight;

        // HIGH should have ~4x the weight of LOW
        assert!(high_weight > low_weight, 
            "HIGH ({:.2}) should have higher weight than LOW ({:.2})", 
            high_weight, low_weight);
    }
}

