//! Multi-objective fitness for Pareto optimization.
//!
//! The fitness system evaluates strategies on multiple objectives
//! (CAGR, Sharpe, Max Drawdown) and uses Pareto dominance for selection.

use serde::{Deserialize, Serialize};

/// Direction for optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    Maximize,
    Minimize,
}

/// Specification for a single objective.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveSpec {
    /// Name of the objective (e.g., "cagr", "sharpe_ratio").
    pub name: String,
    /// Optimization direction.
    pub direction: Direction,
    /// Weight for scalarization (optional).
    #[serde(default = "default_weight")]
    pub weight: f64,
    /// Minimum threshold (filter: discard if below).
    pub threshold_min: Option<f64>,
    /// Maximum threshold (filter: discard if above, for minimization).
    pub threshold_max: Option<f64>,
}

fn default_weight() -> f64 {
    1.0
}

impl ObjectiveSpec {
    /// Create a new objective to maximize.
    pub fn maximize(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            direction: Direction::Maximize,
            weight: 1.0,
            threshold_min: None,
            threshold_max: None,
        }
    }

    /// Create a new objective to minimize.
    pub fn minimize(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            direction: Direction::Minimize,
            weight: 1.0,
            threshold_min: None,
            threshold_max: None,
        }
    }

    /// Set the weight for this objective.
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Set minimum threshold.
    pub fn with_min_threshold(mut self, threshold: f64) -> Self {
        self.threshold_min = Some(threshold);
        self
    }

    /// Set maximum threshold.
    pub fn with_max_threshold(mut self, threshold: f64) -> Self {
        self.threshold_max = Some(threshold);
        self
    }
}

/// Configuration for fitness calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitnessConfig {
    /// Objectives to optimize.
    pub objectives: Vec<ObjectiveSpec>,
    /// Minimum number of trades required.
    #[serde(default = "default_min_trades")]
    pub min_trades: u32,
    /// Maximum annual turnover.
    #[serde(default = "default_max_turnover")]
    pub max_turnover_annual: f64,
}

fn default_min_trades() -> u32 {
    30
}

fn default_max_turnover() -> f64 {
    12.0
}

impl Default for FitnessConfig {
    fn default() -> Self {
        Self {
            objectives: vec![
                ObjectiveSpec::maximize("cagr").with_min_threshold(0.0),
                ObjectiveSpec::maximize("sharpe_ratio").with_min_threshold(0.0),
                ObjectiveSpec::maximize("max_drawdown"), // Less negative = better
            ],
            min_trades: 30,
            max_turnover_annual: 12.0,
        }
    }
}

/// Multi-objective fitness scores.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MultiObjectiveFitness {
    /// Compound Annual Growth Rate.
    pub cagr: f64,
    /// Sharpe Ratio.
    pub sharpe_ratio: f64,
    /// Maximum Drawdown (negative, less negative = better).
    pub max_drawdown: f64,
    /// Calmar Ratio (CAGR / |Max DD|).
    pub calmar_ratio: f64,
    /// Sortino Ratio.
    pub sortino_ratio: f64,
    /// Profit Factor (gross profit / gross loss).
    pub profit_factor: f64,
    /// Total number of trades.
    pub total_trades: u32,
    /// Annualized volatility.
    pub volatility: f64,
    /// Annual turnover.
    pub turnover_annual: f64,

    // Penalties
    /// Penalty for low trade count.
    #[serde(default)]
    pub penalty_low_trades: f64,
    /// Penalty for extreme turnover.
    #[serde(default)]
    pub penalty_extreme_turnover: f64,

    // Pareto ranking (computed by Evolution Engine)
    /// Pareto rank (0 = non-dominated, 1 = dominated by rank 0, etc.).
    #[serde(default)]
    pub pareto_rank: u32,
    /// Crowding distance for diversity preservation.
    #[serde(default)]
    pub crowding_distance: f64,

    // Status
    /// Whether this fitness represents a valid evaluation.
    #[serde(default = "default_true")]
    pub is_valid: bool,
    /// Error message if evaluation failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    
    // Artifact tracking
    /// Run ID (UUID) for pending OBFS artifact cleanup.
    /// Populated when backtest is executed via CLI.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub run_id: Option<String>,
}

fn default_true() -> bool {
    true
}

impl MultiObjectiveFitness {
    /// Create an invalid/failed fitness.
    pub fn invalid(error: impl Into<String>) -> Self {
        Self {
            cagr: f64::NEG_INFINITY,
            sharpe_ratio: f64::NEG_INFINITY,
            max_drawdown: -1.0,
            calmar_ratio: 0.0,
            sortino_ratio: 0.0,
            profit_factor: 0.0,
            total_trades: 0,
            volatility: 0.0,
            turnover_annual: 0.0,
            penalty_low_trades: 1.0,
            penalty_extreme_turnover: 0.0,
            pareto_rank: u32::MAX,
            crowding_distance: 0.0,
            is_valid: false,
            error: Some(error.into()),
            run_id: None,
        }
    }

    /// Create fitness from raw metrics.
    pub fn from_metrics(
        cagr: f64,
        sharpe_ratio: f64,
        max_drawdown: f64,
        calmar_ratio: f64,
        sortino_ratio: f64,
        profit_factor: f64,
        total_trades: u32,
        volatility: f64,
        turnover_annual: f64,
        config: &FitnessConfig,
    ) -> Self {
        // Zero trades = invalid strategy (eliminates artificial metrics)
        if total_trades == 0 {
            return Self::invalid("Zero trades executed - strategy failed to operate");
        }

        let mut fitness = Self {
            cagr,
            sharpe_ratio,
            max_drawdown,
            calmar_ratio,
            sortino_ratio,
            profit_factor,
            total_trades,
            volatility,
            turnover_annual,
            penalty_low_trades: 0.0,
            penalty_extreme_turnover: 0.0,
            pareto_rank: 0,
            crowding_distance: 0.0,
            is_valid: true,
            error: None,
            run_id: None,
        };

        // Graduated penalty for low trade counts
        let min_trades_third = config.min_trades / 3;
        let min_trades_half = config.min_trades / 2;
        if total_trades < min_trades_third {
            fitness.penalty_low_trades = 0.9; // -90% severe penalty
        } else if total_trades < min_trades_half {
            fitness.penalty_low_trades = 0.7; // -70% heavy penalty
        } else if total_trades < config.min_trades {
            fitness.penalty_low_trades = 0.5; // -50% standard penalty
        }

        if turnover_annual > config.max_turnover_annual {
            fitness.penalty_extreme_turnover =
                (turnover_annual - config.max_turnover_annual) / config.max_turnover_annual * 0.2;
        }

        fitness
    }

    /// Check if this fitness dominates another (Pareto dominance).
    ///
    /// A dominates B if A is at least as good as B in all objectives
    /// and strictly better in at least one.
    pub fn dominates(&self, other: &Self) -> bool {
        if !self.is_valid || !other.is_valid {
            return self.is_valid && !other.is_valid;
        }

        // Apply penalties to effective values
        let self_cagr = self.cagr * (1.0 - self.penalty_low_trades - self.penalty_extreme_turnover);
        let self_sharpe =
            self.sharpe_ratio * (1.0 - self.penalty_low_trades - self.penalty_extreme_turnover);
        let self_dd = self.max_drawdown; // Less negative = better

        let other_cagr =
            other.cagr * (1.0 - other.penalty_low_trades - other.penalty_extreme_turnover);
        let other_sharpe =
            other.sharpe_ratio * (1.0 - other.penalty_low_trades - other.penalty_extreme_turnover);
        let other_dd = other.max_drawdown;

        // At least as good in all objectives
        let at_least_cagr = self_cagr >= other_cagr;
        let at_least_sharpe = self_sharpe >= other_sharpe;
        let at_least_dd = self_dd >= other_dd; // Less negative = better

        // Strictly better in at least one
        let better_cagr = self_cagr > other_cagr;
        let better_sharpe = self_sharpe > other_sharpe;
        let better_dd = self_dd > other_dd;

        at_least_cagr
            && at_least_sharpe
            && at_least_dd
            && (better_cagr || better_sharpe || better_dd)
    }

    /// Compute a scalar fitness value for simple comparisons.
    pub fn scalar_fitness(&self) -> f64 {
        if !self.is_valid {
            return f64::NEG_INFINITY;
        }

        // Weighted sum of normalized objectives
        let penalty_factor = 1.0 - self.penalty_low_trades - self.penalty_extreme_turnover;
        self.sharpe_ratio * penalty_factor
    }

    /// Get objective value by name.
    pub fn get_objective(&self, name: &str) -> Option<f64> {
        match name {
            "cagr" => Some(self.cagr),
            "sharpe_ratio" => Some(self.sharpe_ratio),
            "max_drawdown" => Some(self.max_drawdown),
            "calmar_ratio" => Some(self.calmar_ratio),
            "sortino_ratio" => Some(self.sortino_ratio),
            "profit_factor" => Some(self.profit_factor),
            "volatility" => Some(self.volatility),
            "turnover_annual" => Some(self.turnover_annual),
            _ => None,
        }
    }

    /// Check if fitness passes all threshold filters.
    pub fn passes_thresholds(&self, config: &FitnessConfig) -> bool {
        if !self.is_valid {
            return false;
        }

        for objective in &config.objectives {
            if let Some(value) = self.get_objective(&objective.name) {
                if let Some(min) = objective.threshold_min {
                    if value < min {
                        return false;
                    }
                }
                if let Some(max) = objective.threshold_max {
                    if value > max {
                        return false;
                    }
                }
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dominance() {
        let config = FitnessConfig::default();

        let better = MultiObjectiveFitness::from_metrics(
            0.15,  // cagr
            1.2,   // sharpe
            -0.10, // max_dd
            1.5,   // calmar
            1.3,   // sortino
            1.8,   // profit_factor
            100,   // trades
            0.12,  // vol
            2.5,   // turnover
            &config,
        );

        let worse = MultiObjectiveFitness::from_metrics(
            0.10,  // cagr
            0.8,   // sharpe
            -0.15, // max_dd
            0.67,  // calmar
            0.9,   // sortino
            1.2,   // profit_factor
            80,    // trades
            0.15,  // vol
            3.0,   // turnover
            &config,
        );

        assert!(better.dominates(&worse));
        assert!(!worse.dominates(&better));
    }

    #[test]
    fn test_no_dominance() {
        let config = FitnessConfig::default();

        // A: Better CAGR, worse Sharpe
        let a = MultiObjectiveFitness::from_metrics(
            0.20, -0.5, -0.10, 2.0, 1.0, 1.5, 100, 0.12, 2.5, &config,
        );

        // B: Worse CAGR, better Sharpe
        let b = MultiObjectiveFitness::from_metrics(
            0.10, 1.5, -0.10, 1.0, 1.2, 1.8, 100, 0.10, 2.0, &config,
        );

        // Neither dominates the other
        assert!(!a.dominates(&b));
        assert!(!b.dominates(&a));
    }

    #[test]
    fn test_invalid_fitness() {
        let invalid = MultiObjectiveFitness::invalid("backtest failed");
        let config = FitnessConfig::default();
        let valid = MultiObjectiveFitness::from_metrics(
            0.10, 1.0, -0.10, 1.0, 1.0, 1.5, 100, 0.12, 2.5, &config,
        );

        assert!(valid.dominates(&invalid));
        assert!(!invalid.dominates(&valid));
    }

    #[test]
    fn test_penalty_low_trades() {
        let config = FitnessConfig {
            min_trades: 30,
            ..Default::default()
        };

        let low_trades = MultiObjectiveFitness::from_metrics(
            0.15, 1.0, -0.10, 1.5, 1.0, 1.5, 15, // 15 trades (< 30, >= 15)
            0.12, 2.5, &config,
        );

        assert!(low_trades.penalty_low_trades > 0.0);
        assert_eq!(low_trades.penalty_low_trades, 0.5); // Standard penalty
    }

    #[test]
    fn test_zero_trades_invalid() {
        let config = FitnessConfig::default();

        let zero_trades = MultiObjectiveFitness::from_metrics(
            0.25, 10.0, 0.0, 0.0, 0.0, 0.0, 0, // 0 trades = artificial metrics
            0.001, 0.0, &config,
        );

        // Zero trades should be marked as invalid
        assert!(!zero_trades.is_valid);
        assert!(zero_trades.error.is_some());
        assert_eq!(zero_trades.sharpe_ratio, f64::NEG_INFINITY);
    }

    #[test]
    fn test_graduated_penalties() {
        let config = FitnessConfig {
            min_trades: 30,
            ..Default::default()
        };

        // Very low trades (< 10 = 30/3) -> 90% penalty
        let very_low = MultiObjectiveFitness::from_metrics(
            0.15, 1.0, -0.10, 1.5, 1.0, 1.5, 5, 0.12, 2.5, &config,
        );
        assert_eq!(very_low.penalty_low_trades, 0.9);

        // Low trades (< 15 = 30/2, >= 10) -> 70% penalty
        let low = MultiObjectiveFitness::from_metrics(
            0.15, 1.0, -0.10, 1.5, 1.0, 1.5, 12, 0.12, 2.5, &config,
        );
        assert_eq!(low.penalty_low_trades, 0.7);

        // Moderate trades (< 30, >= 15) -> 50% penalty
        let moderate = MultiObjectiveFitness::from_metrics(
            0.15, 1.0, -0.10, 1.5, 1.0, 1.5, 20, 0.12, 2.5, &config,
        );
        assert_eq!(moderate.penalty_low_trades, 0.5);

        // Sufficient trades (>= 30) -> no penalty
        let sufficient = MultiObjectiveFitness::from_metrics(
            0.15, 1.0, -0.10, 1.5, 1.0, 1.5, 50, 0.12, 2.5, &config,
        );
        assert_eq!(sufficient.penalty_low_trades, 0.0);
    }
}

