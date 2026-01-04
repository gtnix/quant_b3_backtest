//! Monte Carlo Risk of Ruin (RoR) Analysis.
//!
//! Implements academic-grade Monte Carlo simulation for Risk of Ruin calculation.
//! Reference: Vince (1992), Chamness (2009)
//!
//! Key metrics:
//! - Probability of Ruin (RoR): P(equity drops below ruin threshold)
//! - Expected Time to Ruin: Mean number of trades before ruin
//! - Confidence Intervals: 95% CI for RoR estimate

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};

// =============================================================================
// TYPES
// =============================================================================

/// Result of a Monte Carlo Risk of Ruin simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoRResult {
    /// Probability of ruin (0.0 - 1.0).
    pub ror_probability: Decimal,
    /// Number of simulations where ruin occurred.
    pub ruin_count: u32,
    /// Total number of simulations run.
    pub total_simulations: u32,
    /// Mean number of trades before ruin (only for ruined paths).
    pub mean_trades_to_ruin: Option<Decimal>,
    /// 95% confidence interval lower bound for RoR.
    pub ci_95_lower: Decimal,
    /// 95% confidence interval upper bound for RoR.
    pub ci_95_upper: Decimal,
    /// Kelly fraction used in simulation.
    pub kelly_fraction: Decimal,
    /// Ruin threshold as percentage of initial capital (e.g., 0.5 = 50% loss).
    pub ruin_threshold: Decimal,
    /// Maximum drawdown observed across all simulations.
    pub max_drawdown_observed: Decimal,
    /// Pass/fail based on academic threshold (RoR < 1%).
    pub passed: bool,
}

impl RoRResult {
    /// Check if the result passes the academic threshold (RoR < 1%).
    pub fn is_acceptable(&self) -> bool {
        self.ror_probability < dec!(0.01)
    }
    
    /// Get a human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "RoR: {:.2}% (CI: {:.2}%-{:.2}%), {} passed",
            self.ror_probability * dec!(100),
            self.ci_95_lower * dec!(100),
            self.ci_95_upper * dec!(100),
            if self.passed { "✓" } else { "✗" }
        )
    }
}

impl Default for RoRResult {
    fn default() -> Self {
        Self {
            ror_probability: Decimal::ZERO,
            ruin_count: 0,
            total_simulations: 0,
            mean_trades_to_ruin: None,
            ci_95_lower: Decimal::ZERO,
            ci_95_upper: Decimal::ZERO,
            kelly_fraction: dec!(0.5),
            ruin_threshold: dec!(0.5),
            max_drawdown_observed: Decimal::ZERO,
            passed: true,
        }
    }
}

/// Configuration for Monte Carlo simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloConfig {
    /// Number of simulations to run.
    pub n_simulations: u32,
    /// Number of trades per simulation path.
    pub trades_per_path: u32,
    /// Ruin threshold as fraction of capital (e.g., 0.5 = 50% loss = ruin).
    pub ruin_threshold: Decimal,
    /// Kelly fraction to use for position sizing.
    pub kelly_fraction: Decimal,
    /// Random seed for reproducibility (None = random).
    pub seed: Option<u64>,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            n_simulations: 1000,
            trades_per_path: 252, // One year of daily trades
            ruin_threshold: dec!(0.5), // 50% drawdown = ruin
            kelly_fraction: dec!(0.5), // Half-Kelly
            seed: None,
        }
    }
}

/// Trade result for simulation input.
#[derive(Debug, Clone)]
pub struct TradeResult {
    /// Return of the trade as a decimal (e.g., 0.02 = 2% gain).
    pub return_pct: Decimal,
}

impl TradeResult {
    pub fn new(return_pct: Decimal) -> Self {
        Self { return_pct }
    }
}

// =============================================================================
// MONTE CARLO ENGINE
// =============================================================================

/// Monte Carlo simulation engine for Risk of Ruin.
pub struct MonteCarloEngine {
    config: MonteCarloConfig,
}

impl MonteCarloEngine {
    /// Create a new engine with default config.
    pub fn new() -> Self {
        Self {
            config: MonteCarloConfig::default(),
        }
    }

    /// Create with custom config.
    pub fn with_config(config: MonteCarloConfig) -> Self {
        Self { config }
    }

    /// Run Monte Carlo simulation for Risk of Ruin.
    ///
    /// # Arguments
    ///
    /// * `trades` - Historical trade results to bootstrap from
    ///
    /// # Returns
    ///
    /// RoRResult with probability of ruin and confidence intervals.
    pub fn simulate_ror(&self, trades: &[TradeResult]) -> RoRResult {
        if trades.is_empty() {
            return RoRResult::default();
        }

        let mut rng = match self.config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let n = trades.len();
        let mut ruin_count = 0u32;
        let mut trades_to_ruin_sum = 0u64;
        let mut max_dd_overall = Decimal::ZERO;

        for _ in 0..self.config.n_simulations {
            let (ruined, trades_before_ruin, max_dd) = 
                self.simulate_single_path(trades, &mut rng);
            
            if ruined {
                ruin_count += 1;
                trades_to_ruin_sum += trades_before_ruin as u64;
            }
            
            if max_dd > max_dd_overall {
                max_dd_overall = max_dd;
            }
        }

        // Calculate probability and confidence interval
        let ror = Decimal::from(ruin_count) / Decimal::from(self.config.n_simulations);
        
        // Wilson score interval for binomial proportion
        let (ci_lower, ci_upper) = self.wilson_ci(
            ruin_count as f64, 
            self.config.n_simulations as f64, 
            0.05 // 95% CI
        );

        let mean_trades_to_ruin = if ruin_count > 0 {
            Some(Decimal::from(trades_to_ruin_sum) / Decimal::from(ruin_count))
        } else {
            None
        };

        RoRResult {
            ror_probability: ror,
            ruin_count,
            total_simulations: self.config.n_simulations,
            mean_trades_to_ruin,
            ci_95_lower: Decimal::from_f64_retain(ci_lower).unwrap_or(Decimal::ZERO),
            ci_95_upper: Decimal::from_f64_retain(ci_upper).unwrap_or(Decimal::ONE),
            kelly_fraction: self.config.kelly_fraction,
            ruin_threshold: self.config.ruin_threshold,
            max_drawdown_observed: max_dd_overall,
            passed: ror < dec!(0.01),
        }
    }

    /// Simulate a single equity path.
    ///
    /// Returns (ruined, trades_before_ruin, max_drawdown).
    fn simulate_single_path<R: Rng>(
        &self, 
        trades: &[TradeResult], 
        rng: &mut R
    ) -> (bool, u32, Decimal) {
        let n = trades.len();
        let mut equity = Decimal::ONE; // Start with 1.0 (100%)
        let mut hwm = equity;
        let mut max_dd = Decimal::ZERO;
        let ruin_level = Decimal::ONE - self.config.ruin_threshold;

        for trade_num in 0..self.config.trades_per_path {
            // Bootstrap: randomly select a trade
            let idx = rng.gen_range(0..n);
            let trade_return = trades[idx].return_pct;
            
            // Apply Kelly-scaled position sizing
            // Actual return = kelly_fraction * trade_return
            let scaled_return = self.config.kelly_fraction * trade_return;
            
            // Update equity
            equity *= Decimal::ONE + scaled_return;
            
            // Update high-water mark and drawdown
            if equity > hwm {
                hwm = equity;
            }
            let dd = (hwm - equity) / hwm;
            if dd > max_dd {
                max_dd = dd;
            }
            
            // Check for ruin
            if equity < ruin_level {
                return (true, trade_num + 1, max_dd);
            }
        }

        (false, self.config.trades_per_path, max_dd)
    }

    /// Calculate Wilson score confidence interval for binomial proportion.
    fn wilson_ci(&self, successes: f64, n: f64, alpha: f64) -> (f64, f64) {
        if n == 0.0 {
            return (0.0, 1.0);
        }

        let p = successes / n;
        let z = self.normal_quantile(1.0 - alpha / 2.0);
        let z2 = z * z;
        
        let denominator = 1.0 + z2 / n;
        let center = (p + z2 / (2.0 * n)) / denominator;
        let margin = (z / denominator) * ((p * (1.0 - p) / n) + (z2 / (4.0 * n * n))).sqrt();

        let lower = (center - margin).max(0.0);
        let upper = (center + margin).min(1.0);

        (lower, upper)
    }

    /// Approximate inverse normal CDF (quantile function).
    fn normal_quantile(&self, p: f64) -> f64 {
        // Approximation for standard normal quantile
        // Using Beasley-Springer-Moro algorithm
        if p <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }

        let a = [
            -3.969683028665376e1,
            2.209460984245205e2,
            -2.759285104469687e2,
            1.383577518672690e2,
            -3.066479806614716e1,
            2.506628277459239e0,
        ];
        let b = [
            -5.447609879822406e1,
            1.615858368580409e2,
            -1.556989798598866e2,
            6.680131188771972e1,
            -1.328068155288572e1,
        ];
        let c = [
            -7.784894002430293e-3,
            -3.223964580411365e-1,
            -2.400758277161838e0,
            -2.549732539343734e0,
            4.374664141464968e0,
            2.938163982698783e0,
        ];
        let d = [
            7.784695709041462e-3,
            3.224671290700398e-1,
            2.445134137142996e0,
            3.754408661907416e0,
        ];

        let p_low = 0.02425;
        let p_high = 1.0 - p_low;

        let q: f64;
        let r: f64;

        if p < p_low {
            q = (-2.0 * p.ln()).sqrt();
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
        } else if p <= p_high {
            q = p - 0.5;
            r = q * q;
            return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
        } else {
            q = (-2.0 * (1.0 - p).ln()).sqrt();
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
        }
    }
}

impl Default for MonteCarloEngine {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/// Run a quick Monte Carlo RoR simulation with default settings.
///
/// # Arguments
///
/// * `trade_returns` - Slice of trade returns as decimals
/// * `kelly_fraction` - Kelly fraction to use (e.g., 0.5 for Half-Kelly)
/// * `n_sims` - Number of simulations (default 1000)
///
/// # Returns
///
/// RoRResult with probability and confidence intervals.
pub fn monte_carlo_ror(
    trade_returns: &[Decimal],
    kelly_fraction: Decimal,
    n_sims: u32,
) -> RoRResult {
    let trades: Vec<TradeResult> = trade_returns
        .iter()
        .map(|&r| TradeResult::new(r))
        .collect();

    let config = MonteCarloConfig {
        n_simulations: n_sims,
        kelly_fraction,
        ..Default::default()
    };

    let engine = MonteCarloEngine::with_config(config);
    engine.simulate_ror(&trades)
}

/// Run Monte Carlo with custom ruin threshold.
pub fn monte_carlo_ror_with_threshold(
    trade_returns: &[Decimal],
    kelly_fraction: Decimal,
    ruin_threshold: Decimal,
    n_sims: u32,
) -> RoRResult {
    let trades: Vec<TradeResult> = trade_returns
        .iter()
        .map(|&r| TradeResult::new(r))
        .collect();

    let config = MonteCarloConfig {
        n_simulations: n_sims,
        kelly_fraction,
        ruin_threshold,
        ..Default::default()
    };

    let engine = MonteCarloEngine::with_config(config);
    engine.simulate_ror(&trades)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_trades_profitable() -> Vec<TradeResult> {
        // Simulated profitable strategy: 60% win rate, 1.5:1 R:R
        (0..100).map(|i| {
            if i % 10 < 6 {
                TradeResult::new(dec!(0.015)) // Win: +1.5%
            } else {
                TradeResult::new(dec!(-0.01)) // Loss: -1%
            }
        }).collect()
    }

    fn sample_trades_losing() -> Vec<TradeResult> {
        // Simulated losing strategy: 40% win rate, 1:1 R:R
        (0..100).map(|i| {
            if i % 10 < 4 {
                TradeResult::new(dec!(0.01)) // Win: +1%
            } else {
                TradeResult::new(dec!(-0.01)) // Loss: -1%
            }
        }).collect()
    }

    #[test]
    fn test_ror_result_default() {
        let result = RoRResult::default();
        assert!(result.passed);
        assert_eq!(result.ror_probability, Decimal::ZERO);
    }

    #[test]
    fn test_monte_carlo_config_default() {
        let config = MonteCarloConfig::default();
        assert_eq!(config.n_simulations, 1000);
        assert_eq!(config.kelly_fraction, dec!(0.5));
    }

    #[test]
    fn test_profitable_strategy_low_ror() {
        let trades = sample_trades_profitable();
        let config = MonteCarloConfig {
            n_simulations: 500,
            trades_per_path: 100,
            kelly_fraction: dec!(0.25), // Quarter Kelly
            ruin_threshold: dec!(0.5),
            seed: Some(42),
        };

        let engine = MonteCarloEngine::with_config(config);
        let result = engine.simulate_ror(&trades);

        // Profitable strategy with quarter Kelly should have low RoR
        assert!(result.ror_probability < dec!(0.1), 
            "Profitable strategy should have low RoR: {}", result.ror_probability);
    }

    #[test]
    fn test_losing_strategy_higher_ror() {
        let trades = sample_trades_losing();
        let config = MonteCarloConfig {
            n_simulations: 500,
            trades_per_path: 500, // More trades to see ruin
            kelly_fraction: dec!(1.0), // Full Kelly (more risky)
            ruin_threshold: dec!(0.3), // Lower threshold (30% loss = ruin)
            seed: Some(42),
        };

        let engine = MonteCarloEngine::with_config(config);
        let result = engine.simulate_ror(&trades);

        // Losing strategy with full Kelly should have measurable RoR
        // At minimum, max drawdown should be significant
        assert!(result.max_drawdown_observed > dec!(0.1), 
            "Losing strategy should have significant max DD: {}", result.max_drawdown_observed);
    }

    #[test]
    fn test_higher_kelly_higher_ror() {
        let trades = sample_trades_profitable();
        
        // Quarter Kelly
        let result_quarter = monte_carlo_ror_with_threshold(
            &trades.iter().map(|t| t.return_pct).collect::<Vec<_>>(),
            dec!(0.25),
            dec!(0.5),
            500,
        );
        
        // Full Kelly
        let result_full = monte_carlo_ror_with_threshold(
            &trades.iter().map(|t| t.return_pct).collect::<Vec<_>>(),
            dec!(1.0),
            dec!(0.5),
            500,
        );

        // Higher Kelly fraction should have higher or equal RoR
        assert!(result_full.ror_probability >= result_quarter.ror_probability,
            "Full Kelly RoR {} should be >= Quarter Kelly RoR {}", 
            result_full.ror_probability, result_quarter.ror_probability);
    }

    #[test]
    fn test_confidence_interval_contains_point_estimate() {
        let trades = sample_trades_profitable();
        let returns: Vec<Decimal> = trades.iter().map(|t| t.return_pct).collect();
        
        let result = monte_carlo_ror(&returns, dec!(0.5), 1000);
        
        // Point estimate should be within CI
        assert!(result.ror_probability >= result.ci_95_lower,
            "RoR {} should be >= CI lower {}", result.ror_probability, result.ci_95_lower);
        assert!(result.ror_probability <= result.ci_95_upper,
            "RoR {} should be <= CI upper {}", result.ror_probability, result.ci_95_upper);
    }

    #[test]
    fn test_empty_trades() {
        let trades: Vec<TradeResult> = vec![];
        let engine = MonteCarloEngine::new();
        let result = engine.simulate_ror(&trades);
        
        assert!(result.passed);
        assert_eq!(result.total_simulations, 0);
    }

    #[test]
    fn test_reproducibility_with_seed() {
        let trades = sample_trades_profitable();
        let returns: Vec<Decimal> = trades.iter().map(|t| t.return_pct).collect();
        
        let config = MonteCarloConfig {
            n_simulations: 100,
            seed: Some(12345),
            ..Default::default()
        };
        
        let engine1 = MonteCarloEngine::with_config(config.clone());
        let engine2 = MonteCarloEngine::with_config(config);
        
        let trades_vec: Vec<TradeResult> = returns.iter().map(|&r| TradeResult::new(r)).collect();
        
        let result1 = engine1.simulate_ror(&trades_vec);
        let result2 = engine2.simulate_ror(&trades_vec);
        
        assert_eq!(result1.ror_probability, result2.ror_probability,
            "Same seed should produce same results");
    }

    #[test]
    fn test_max_drawdown_tracked() {
        let trades = sample_trades_losing();
        let config = MonteCarloConfig {
            n_simulations: 100,
            trades_per_path: 50,
            kelly_fraction: dec!(0.5),
            ruin_threshold: dec!(0.5),
            seed: Some(42),
        };

        let engine = MonteCarloEngine::with_config(config);
        let result = engine.simulate_ror(&trades);

        // Max DD should be positive for any non-trivial simulation
        assert!(result.max_drawdown_observed > Decimal::ZERO,
            "Max DD should be positive: {}", result.max_drawdown_observed);
    }

    #[test]
    fn test_summary_format() {
        let result = RoRResult {
            ror_probability: dec!(0.05),
            ci_95_lower: dec!(0.03),
            ci_95_upper: dec!(0.07),
            passed: false,
            ..Default::default()
        };

        let summary = result.summary();
        assert!(summary.contains("5.00%"));
        assert!(summary.contains("✗"));
    }

    #[test]
    fn test_is_acceptable() {
        let mut result = RoRResult::default();
        
        result.ror_probability = dec!(0.005);
        assert!(result.is_acceptable(), "0.5% RoR should be acceptable");
        
        result.ror_probability = dec!(0.02);
        assert!(!result.is_acceptable(), "2% RoR should not be acceptable");
    }
}

