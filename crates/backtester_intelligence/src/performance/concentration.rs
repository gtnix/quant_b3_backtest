//! Portfolio Concentration Metrics.
//!
//! Calculates HHI, top-N weights, effective-N, and Gini coefficient
//! for portfolio concentration analysis.
//!
//! # Design Decisions
//!
//! - Uses absolute weights for concentration (shorts count positively)
//! - Empty portfolio returns zeroed metrics (not None/error)
//! - Gini is optional due to O(N log N) sort cost
//! - All calculations use Decimal for precision
//!
//! # Formulas
//!
//! ## HHI (Herfindahl-Hirschman Index)
//! ```text
//! HHI = sum(w_i^2) where w_i = |value_i| / sum(|value_j|)
//! Range: [1/N, 1] where 1 = single position, 1/N = equal weight
//! ```
//!
//! ## Effective N
//! ```text
//! Effective N = 1 / HHI
//! Interpretation: Portfolio behaves like N equal-weight positions
//! ```
//!
//! ## Gini Coefficient
//! ```text
//! Gini = (2 * sum(i * w_sorted_i)) / (n * sum(w_i)) - (n+1)/n
//! Range: [0, 1] where 0 = perfect equality, 1 = max inequality
//! ```

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

// =============================================================================
// CONCENTRATION METRICS
// =============================================================================

/// Portfolio concentration metrics.
///
/// All metrics are calculated from absolute weights (|position_value|),
/// meaning short positions contribute positively to concentration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConcentrationMetrics {
    /// Herfindahl-Hirschman Index (sum of squared weights).
    ///
    /// Range: [1/N, 1] where:
    /// - 1 = single position (maximum concentration)
    /// - 1/N = equal weight distribution (minimum concentration)
    /// - 0 = empty portfolio
    pub hhi: Decimal,

    /// Effective number of positions = 1 / HHI.
    ///
    /// Interpretation: "Portfolio behaves like N equal-weight positions."
    /// For example, effective_n = 10 means the portfolio has the same
    /// concentration as 10 equal-weight positions.
    pub effective_n: Decimal,

    /// Weight of the largest position (as percentage 0-100).
    pub top_1_weight_pct: Decimal,

    /// Sum of top 5 position weights (as percentage 0-100).
    pub top_5_weight_pct: Decimal,

    /// Sum of top 10 position weights (as percentage 0-100).
    pub top_10_weight_pct: Decimal,

    /// Maximum single position weight (as percentage 0-100).
    /// Same as top_1_weight_pct, included for explicit clarity.
    pub max_position_weight_pct: Decimal,

    /// Number of positions used in calculation.
    pub n_positions: u32,

    /// Gini coefficient (0 = perfect equality, 1 = max inequality).
    ///
    /// Optional because it requires O(N log N) sorting.
    /// None when not calculated (e.g., for performance reasons).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gini: Option<Decimal>,
}

impl ConcentrationMetrics {
    /// Create empty/zero metrics.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Check if the portfolio is empty.
    pub fn is_empty(&self) -> bool {
        self.n_positions == 0
    }

    /// Check if portfolio is highly concentrated (HHI > 0.25 = 4 effective positions).
    pub fn is_highly_concentrated(&self) -> bool {
        self.hhi > Decimal::from_str_exact("0.25").unwrap_or(Decimal::ZERO)
    }

    /// Get concentration level as a human-readable category.
    pub fn concentration_level(&self) -> &'static str {
        if self.is_empty() {
            "Empty"
        } else if self.hhi >= Decimal::ONE {
            "Single Position"
        } else if self.hhi > Decimal::from_str_exact("0.5").unwrap_or(Decimal::ONE) {
            "Very High"
        } else if self.hhi > Decimal::from_str_exact("0.25").unwrap_or(Decimal::ONE) {
            "High"
        } else if self.hhi > Decimal::from_str_exact("0.15").unwrap_or(Decimal::ONE) {
            "Moderate"
        } else if self.hhi > Decimal::from_str_exact("0.1").unwrap_or(Decimal::ONE) {
            "Low"
        } else {
            "Very Low"
        }
    }
}

// =============================================================================
// CONCENTRATION CALCULATOR
// =============================================================================

/// Calculator for portfolio concentration metrics.
///
/// Stateless calculator - all methods are pure functions.
pub struct ConcentrationCalculator;

impl ConcentrationCalculator {
    /// Calculate concentration metrics from symbol-value pairs.
    ///
    /// # Arguments
    ///
    /// * `positions` - Vector of (symbol, value) pairs. Values can be negative
    ///   for short positions; absolute values are used for concentration.
    ///
    /// # Returns
    ///
    /// `ConcentrationMetrics` with all fields populated.
    /// Empty input returns zeroed metrics.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let positions = vec![
    ///     ("AAPL".to_string(), dec!(5000)),
    ///     ("MSFT".to_string(), dec!(3000)),
    ///     ("GOOG".to_string(), dec!(2000)),
    /// ];
    /// let metrics = ConcentrationCalculator::calculate(&positions);
    /// assert!(metrics.hhi > Decimal::ZERO);
    /// ```
    pub fn calculate(positions: &[(String, Decimal)]) -> ConcentrationMetrics {
        Self::calculate_with_gini(positions, true)
    }

    /// Calculate concentration metrics with optional Gini.
    ///
    /// # Arguments
    ///
    /// * `positions` - Vector of (symbol, value) pairs
    /// * `include_gini` - Whether to calculate Gini coefficient (O(N log N))
    pub fn calculate_with_gini(
        positions: &[(String, Decimal)],
        include_gini: bool,
    ) -> ConcentrationMetrics {
        if positions.is_empty() {
            return ConcentrationMetrics::empty();
        }

        // Calculate absolute weights
        let abs_values: Vec<Decimal> = positions.iter().map(|(_, v)| v.abs()).collect();
        let total: Decimal = abs_values.iter().sum();

        if total.is_zero() {
            return ConcentrationMetrics::empty();
        }

        // Calculate weights as percentages
        let mut weights: Vec<Decimal> = abs_values
            .iter()
            .map(|v| *v / total * Decimal::from(100))
            .collect();

        // Sort descending for top-N calculations
        weights.sort_by(|a, b| b.cmp(a));

        let n = weights.len();
        let n_dec = Decimal::from(n as u32);

        // HHI: sum of squared weights (normalized to 0-1 scale)
        let hhi: Decimal = weights
            .iter()
            .map(|w| {
                let w_normalized = *w / Decimal::from(100);
                w_normalized * w_normalized
            })
            .sum();

        // Effective N: 1 / HHI
        let effective_n = if hhi.is_zero() {
            Decimal::ZERO
        } else {
            Decimal::ONE / hhi
        };

        // Top-N weights
        let top_1_weight_pct = weights.first().copied().unwrap_or(Decimal::ZERO);
        let top_5_weight_pct: Decimal = weights.iter().take(5).sum();
        let top_10_weight_pct: Decimal = weights.iter().take(10).sum();
        let max_position_weight_pct = top_1_weight_pct;

        // Gini coefficient (optional)
        let gini = if include_gini && n > 1 {
            Some(Self::calculate_gini(&weights))
        } else if n == 1 {
            // Single position = maximum inequality
            Some(Decimal::ZERO) // Gini is 0 for single position (no inequality possible)
        } else {
            None
        };

        ConcentrationMetrics {
            hhi,
            effective_n,
            top_1_weight_pct,
            top_5_weight_pct,
            top_10_weight_pct,
            max_position_weight_pct,
            n_positions: n as u32,
            gini,
        }
    }

    /// Calculate Gini coefficient from sorted weights.
    ///
    /// Formula: Gini = (2 * sum(i * w_i)) / (n * sum(w_i)) - (n+1)/n
    ///
    /// Note: Weights should already be sorted in descending order,
    /// but we re-sort ascending for the Gini formula.
    fn calculate_gini(weights: &[Decimal]) -> Decimal {
        let n = weights.len();
        if n <= 1 {
            return Decimal::ZERO;
        }

        let n_dec = Decimal::from(n as u32);

        // Sort ascending for Gini formula
        let mut sorted: Vec<Decimal> = weights.to_vec();
        sorted.sort();

        let total: Decimal = sorted.iter().sum();
        if total.is_zero() {
            return Decimal::ZERO;
        }

        // Gini = (2 * sum(i * w_i)) / (n * sum(w_i)) - (n+1)/n
        // where i is 1-indexed
        let weighted_sum: Decimal = sorted
            .iter()
            .enumerate()
            .map(|(i, w)| Decimal::from((i + 1) as u32) * *w)
            .sum();

        let gini = (Decimal::from(2) * weighted_sum) / (n_dec * total)
            - (n_dec + Decimal::ONE) / n_dec;

        // Ensure Gini is in [0, 1] range
        gini.max(Decimal::ZERO).min(Decimal::ONE)
    }

    /// Calculate concentration from a simple weight map.
    ///
    /// Convenience method that converts BTreeMap to vector format.
    pub fn from_weight_map(
        weights: &std::collections::BTreeMap<String, Decimal>,
    ) -> ConcentrationMetrics {
        let positions: Vec<(String, Decimal)> = weights
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        Self::calculate(&positions)
    }

    /// Calculate concentration from position values (values, not weights).
    ///
    /// Same as `calculate`, but makes the "values not weights" semantic explicit.
    pub fn from_position_values(position_values: &[(String, Decimal)]) -> ConcentrationMetrics {
        Self::calculate(position_values)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_empty_portfolio() {
        let positions: Vec<(String, Decimal)> = vec![];
        let metrics = ConcentrationCalculator::calculate(&positions);

        assert_eq!(metrics.hhi, Decimal::ZERO);
        assert_eq!(metrics.effective_n, Decimal::ZERO);
        assert_eq!(metrics.n_positions, 0);
        assert!(metrics.is_empty());
    }

    #[test]
    fn test_single_position() {
        let positions = vec![("AAPL".to_string(), dec!(10000))];
        let metrics = ConcentrationCalculator::calculate(&positions);

        // Single position = HHI of 1 (100%^2 = 1)
        assert_eq!(metrics.hhi, Decimal::ONE);
        assert_eq!(metrics.effective_n, Decimal::ONE);
        assert_eq!(metrics.top_1_weight_pct, dec!(100));
        assert_eq!(metrics.n_positions, 1);
        assert_eq!(metrics.concentration_level(), "Single Position");
    }

    #[test]
    fn test_two_equal_positions() {
        let positions = vec![
            ("AAPL".to_string(), dec!(5000)),
            ("MSFT".to_string(), dec!(5000)),
        ];
        let metrics = ConcentrationCalculator::calculate(&positions);

        // Two equal positions: HHI = 0.5^2 + 0.5^2 = 0.5
        assert_eq!(metrics.hhi, dec!(0.5));
        assert_eq!(metrics.effective_n, dec!(2));
        assert_eq!(metrics.top_1_weight_pct, dec!(50));
        assert_eq!(metrics.top_5_weight_pct, dec!(100));
    }

    #[test]
    fn test_three_equal_positions() {
        let positions = vec![
            ("A".to_string(), dec!(1000)),
            ("B".to_string(), dec!(1000)),
            ("C".to_string(), dec!(1000)),
        ];
        let metrics = ConcentrationCalculator::calculate(&positions);

        // Three equal positions: HHI = 3 * (1/3)^2 = 1/3 ≈ 0.333
        let expected_hhi = Decimal::ONE / Decimal::from(3);
        assert!((metrics.hhi - expected_hhi).abs() < dec!(0.001));
        
        // Effective N should be 3
        assert!((metrics.effective_n - dec!(3)).abs() < dec!(0.01));
    }

    #[test]
    fn test_unequal_positions() {
        let positions = vec![
            ("A".to_string(), dec!(7000)),  // 70%
            ("B".to_string(), dec!(2000)),  // 20%
            ("C".to_string(), dec!(1000)),  // 10%
        ];
        let metrics = ConcentrationCalculator::calculate(&positions);

        // HHI = 0.7^2 + 0.2^2 + 0.1^2 = 0.49 + 0.04 + 0.01 = 0.54
        assert!((metrics.hhi - dec!(0.54)).abs() < dec!(0.001));
        
        // Top weights
        assert_eq!(metrics.top_1_weight_pct, dec!(70));
        assert_eq!(metrics.top_5_weight_pct, dec!(100));
        
        // Effective N ≈ 1.85
        assert!(metrics.effective_n < dec!(2));
        assert!(metrics.effective_n > dec!(1.8));
    }

    #[test]
    fn test_short_positions_use_absolute_value() {
        // Short positions should use absolute value for concentration
        let positions = vec![
            ("LONG".to_string(), dec!(5000)),
            ("SHORT".to_string(), dec!(-5000)),  // Short position
        ];
        let metrics = ConcentrationCalculator::calculate(&positions);

        // Both should count equally: HHI = 0.5
        assert_eq!(metrics.hhi, dec!(0.5));
        assert_eq!(metrics.top_1_weight_pct, dec!(50));
    }

    #[test]
    fn test_top_n_weights() {
        // 15 positions with decreasing weights
        let mut positions: Vec<(String, Decimal)> = Vec::new();
        for i in 1..=15 {
            positions.push((format!("POS{}", i), Decimal::from(100 - i as u32 * 5)));
        }
        let metrics = ConcentrationCalculator::calculate(&positions);

        assert_eq!(metrics.n_positions, 15);
        assert!(metrics.top_1_weight_pct > Decimal::ZERO);
        assert!(metrics.top_5_weight_pct > metrics.top_1_weight_pct);
        assert!(metrics.top_10_weight_pct > metrics.top_5_weight_pct);
        assert!(metrics.top_10_weight_pct <= dec!(100));
    }

    #[test]
    fn test_gini_equal_weights() {
        // Equal weights should have Gini close to 0
        let positions = vec![
            ("A".to_string(), dec!(100)),
            ("B".to_string(), dec!(100)),
            ("C".to_string(), dec!(100)),
            ("D".to_string(), dec!(100)),
        ];
        let metrics = ConcentrationCalculator::calculate(&positions);

        assert!(metrics.gini.is_some());
        let gini = metrics.gini.unwrap();
        assert!(gini < dec!(0.1), "Gini should be near 0 for equal weights, got {}", gini);
    }

    #[test]
    fn test_gini_unequal_weights() {
        // Very unequal weights should have high Gini
        let positions = vec![
            ("A".to_string(), dec!(9000)),
            ("B".to_string(), dec!(500)),
            ("C".to_string(), dec!(300)),
            ("D".to_string(), dec!(200)),
        ];
        let metrics = ConcentrationCalculator::calculate(&positions);

        assert!(metrics.gini.is_some());
        let gini = metrics.gini.unwrap();
        assert!(gini > dec!(0.5), "Gini should be high for unequal weights, got {}", gini);
    }

    #[test]
    fn test_without_gini() {
        let positions = vec![
            ("A".to_string(), dec!(5000)),
            ("B".to_string(), dec!(5000)),
        ];
        let metrics = ConcentrationCalculator::calculate_with_gini(&positions, false);

        assert!(metrics.gini.is_none());
        // Other metrics should still be calculated
        assert_eq!(metrics.hhi, dec!(0.5));
    }

    #[test]
    fn test_concentration_levels() {
        // Very High: HHI > 0.5
        let single = vec![("A".to_string(), dec!(100))];
        assert_eq!(
            ConcentrationCalculator::calculate(&single).concentration_level(),
            "Single Position"
        );

        // High: HHI > 0.25
        let two_unequal = vec![
            ("A".to_string(), dec!(80)),
            ("B".to_string(), dec!(20)),
        ];
        let metrics = ConcentrationCalculator::calculate(&two_unequal);
        // HHI = 0.8^2 + 0.2^2 = 0.68 -> Very High
        assert_eq!(metrics.concentration_level(), "Very High");

        // Lower concentration with more positions
        let many: Vec<(String, Decimal)> = (0..20)
            .map(|i| (format!("P{}", i), dec!(100)))
            .collect();
        let metrics = ConcentrationCalculator::calculate(&many);
        // HHI = 20 * (1/20)^2 = 1/20 = 0.05 -> Very Low
        assert_eq!(metrics.concentration_level(), "Very Low");
    }

    #[test]
    fn test_zero_values() {
        let positions = vec![
            ("A".to_string(), Decimal::ZERO),
            ("B".to_string(), Decimal::ZERO),
        ];
        let metrics = ConcentrationCalculator::calculate(&positions);

        // All zeros should return empty metrics
        assert!(metrics.is_empty() || metrics.hhi.is_zero());
    }

    #[test]
    fn test_from_weight_map() {
        use std::collections::BTreeMap;

        let mut map = BTreeMap::new();
        map.insert("AAPL".to_string(), dec!(50));
        map.insert("MSFT".to_string(), dec!(50));

        let metrics = ConcentrationCalculator::from_weight_map(&map);
        assert_eq!(metrics.hhi, dec!(0.5));
    }

    #[test]
    fn test_serialization() {
        let positions = vec![
            ("A".to_string(), dec!(6000)),
            ("B".to_string(), dec!(4000)),
        ];
        let metrics = ConcentrationCalculator::calculate(&positions);

        let json = serde_json::to_string(&metrics).unwrap();
        let parsed: ConcentrationMetrics = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.hhi, metrics.hhi);
        assert_eq!(parsed.n_positions, metrics.n_positions);
    }

    #[test]
    fn test_is_highly_concentrated() {
        // HHI > 0.25 is highly concentrated
        let concentrated = vec![
            ("A".to_string(), dec!(8000)),
            ("B".to_string(), dec!(2000)),
        ];
        let metrics = ConcentrationCalculator::calculate(&concentrated);
        // HHI = 0.8^2 + 0.2^2 = 0.68
        assert!(metrics.is_highly_concentrated());

        // 10 equal positions: HHI = 0.1 -> not highly concentrated
        let diversified: Vec<(String, Decimal)> = (0..10)
            .map(|i| (format!("P{}", i), dec!(1000)))
            .collect();
        let metrics = ConcentrationCalculator::calculate(&diversified);
        assert!(!metrics.is_highly_concentrated());
    }
}

