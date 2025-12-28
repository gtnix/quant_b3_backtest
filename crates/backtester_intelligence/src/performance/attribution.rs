//! Attribution Engine - Proportional P&L attribution to AO7 techniques.
//!
//! Allocates position P&L proportionally based on each technique's
//! weight in the entry decision score.

use rust_decimal::Decimal;
use std::collections::BTreeMap;

use super::types::{TechniqueAttribution, AttributionBreakdown};

/// Attribution engine for decomposing P&L by technique.
#[derive(Debug, Clone)]
pub struct AttributionEngine {
    /// Technique weights by symbol (symbol -> technique -> normalized_weight)
    entry_weights: BTreeMap<String, BTreeMap<String, Decimal>>,
}

impl AttributionEngine {
    pub fn new() -> Self {
        Self {
            entry_weights: BTreeMap::new(),
        }
    }

    /// Record the technique weights used for a position entry.
    ///
    /// Weights should sum to 1.0 (or will be normalized).
    pub fn record_entry_weights(&mut self, symbol: &str, weights: BTreeMap<String, Decimal>) {
        // Normalize weights to sum to 1.0
        let total: Decimal = weights.values().sum();
        let normalized = if total.is_zero() {
            weights
        } else {
            weights.into_iter()
                .map(|(k, v)| (k, v / total))
                .collect()
        };
        
        self.entry_weights.insert(symbol.to_string(), normalized);
    }

    /// Calculate attribution breakdown from P&L by symbol.
    pub fn calculate_attribution(
        &self,
        pnl_by_symbol: &BTreeMap<String, Decimal>,
    ) -> AttributionBreakdown {
        let mut technique_pnl: BTreeMap<String, Decimal> = BTreeMap::new();
        let mut technique_weight_sum: BTreeMap<String, Decimal> = BTreeMap::new();
        let mut total_pnl = Decimal::ZERO;
        let mut residual = Decimal::ZERO;

        for (symbol, &pnl) in pnl_by_symbol {
            total_pnl += pnl;
            
            if let Some(weights) = self.entry_weights.get(symbol) {
                for (technique, &weight) in weights {
                    let contribution = pnl * weight;
                    *technique_pnl.entry(technique.clone()).or_default() += contribution;
                    *technique_weight_sum.entry(technique.clone()).or_default() += weight;
                }
            } else {
                // No attribution data - add to residual
                residual += pnl;
            }
        }

        // Build attribution vector (sorted for determinism)
        let by_technique: Vec<TechniqueAttribution> = technique_pnl
            .into_iter()
            .map(|(name, pnl_contribution)| {
                let weight_pct = technique_weight_sum
                    .get(&name)
                    .copied()
                    .unwrap_or(Decimal::ZERO) * Decimal::from(100);
                let return_contribution = if total_pnl.is_zero() {
                    Decimal::ZERO
                } else {
                    pnl_contribution / total_pnl * Decimal::from(100)
                };
                TechniqueAttribution {
                    technique_name: name,
                    weight_pct,
                    pnl_contribution,
                    return_contribution,
                }
            })
            .collect();

        AttributionBreakdown {
            by_technique,
            total_pnl,
            residual,
        }
    }

    /// Clear all recorded weights.
    pub fn clear(&mut self) {
        self.entry_weights.clear();
    }

    /// Check if we have weights for a symbol.
    pub fn has_weights(&self, symbol: &str) -> bool {
        self.entry_weights.contains_key(symbol)
    }
}

impl Default for AttributionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn make_weights(data: &[(&str, Decimal)]) -> BTreeMap<String, Decimal> {
        data.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    fn make_pnl(data: &[(&str, Decimal)]) -> BTreeMap<String, Decimal> {
        data.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn test_single_position_attribution() {
        let mut engine = AttributionEngine::new();
        
        // PETR4 entry was 60% momentum, 40% value
        engine.record_entry_weights("PETR4", make_weights(&[
            ("momentum", dec!(0.6)),
            ("value", dec!(0.4)),
        ]));

        let pnl = make_pnl(&[("PETR4", dec!(1000))]);
        let attr = engine.calculate_attribution(&pnl);

        assert_eq!(attr.total_pnl, dec!(1000));
        assert_eq!(attr.residual, dec!(0));
        assert_eq!(attr.by_technique.len(), 2);

        // Check momentum contribution: 1000 * 0.6 = 600
        let momentum = attr.by_technique.iter().find(|t| t.technique_name == "momentum").unwrap();
        assert_eq!(momentum.pnl_contribution, dec!(600));

        // Check value contribution: 1000 * 0.4 = 400
        let value = attr.by_technique.iter().find(|t| t.technique_name == "value").unwrap();
        assert_eq!(value.pnl_contribution, dec!(400));
    }

    #[test]
    fn test_multiple_positions() {
        let mut engine = AttributionEngine::new();
        
        engine.record_entry_weights("PETR4", make_weights(&[
            ("momentum", dec!(0.7)),
            ("quality", dec!(0.3)),
        ]));
        
        engine.record_entry_weights("VALE3", make_weights(&[
            ("momentum", dec!(0.5)),
            ("value", dec!(0.5)),
        ]));

        let pnl = make_pnl(&[
            ("PETR4", dec!(1000)),
            ("VALE3", dec!(500)),
        ]);
        let attr = engine.calculate_attribution(&pnl);

        assert_eq!(attr.total_pnl, dec!(1500));
        
        // Momentum: PETR4 * 0.7 + VALE3 * 0.5 = 700 + 250 = 950
        let momentum = attr.by_technique.iter().find(|t| t.technique_name == "momentum").unwrap();
        assert_eq!(momentum.pnl_contribution, dec!(950));
    }

    #[test]
    fn test_sum_equals_total() {
        let mut engine = AttributionEngine::new();
        
        engine.record_entry_weights("A", make_weights(&[
            ("momentum", dec!(0.4)),
            ("value", dec!(0.3)),
            ("quality", dec!(0.3)),
        ]));

        let pnl = make_pnl(&[("A", dec!(1000))]);
        let attr = engine.calculate_attribution(&pnl);

        let sum: Decimal = attr.by_technique.iter()
            .map(|t| t.pnl_contribution)
            .sum();
        
        assert_eq!(sum + attr.residual, attr.total_pnl);
    }

    #[test]
    fn test_missing_weights_goes_to_residual() {
        let engine = AttributionEngine::new();
        
        let pnl = make_pnl(&[("PETR4", dec!(1000))]);
        let attr = engine.calculate_attribution(&pnl);

        assert_eq!(attr.residual, dec!(1000));
        assert!(attr.by_technique.is_empty());
    }

    #[test]
    fn test_weight_normalization() {
        let mut engine = AttributionEngine::new();
        
        // Weights don't sum to 1 - should be normalized
        engine.record_entry_weights("A", make_weights(&[
            ("momentum", dec!(2)),
            ("value", dec!(3)),
        ]));

        let pnl = make_pnl(&[("A", dec!(1000))]);
        let attr = engine.calculate_attribution(&pnl);

        // momentum: 1000 * (2/5) = 400
        // value: 1000 * (3/5) = 600
        let momentum = attr.by_technique.iter().find(|t| t.technique_name == "momentum").unwrap();
        assert_eq!(momentum.pnl_contribution, dec!(400));
    }

    #[test]
    fn test_deterministic_output() {
        let mut engine = AttributionEngine::new();
        
        engine.record_entry_weights("A", make_weights(&[
            ("momentum", dec!(0.5)),
            ("value", dec!(0.5)),
        ]));

        let pnl = make_pnl(&[("A", dec!(1000))]);
        
        let attr1 = engine.calculate_attribution(&pnl);
        let attr2 = engine.calculate_attribution(&pnl);

        // Same order and values
        assert_eq!(attr1.by_technique.len(), attr2.by_technique.len());
        for (t1, t2) in attr1.by_technique.iter().zip(attr2.by_technique.iter()) {
            assert_eq!(t1.technique_name, t2.technique_name);
            assert_eq!(t1.pnl_contribution, t2.pnl_contribution);
        }
    }
}









