//! Walk-Forward Validation types.
//!
//! Core data structures for rolling window validation with purge/embargo.

use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::filters::Market;

/// Type of window (train or test).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WindowType {
    Train,
    Test,
}

/// Specification for a single time window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowSpec {
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub window_type: WindowType,
    pub index: usize,
}

impl WindowSpec {
    pub fn new(start_date: NaiveDate, end_date: NaiveDate, window_type: WindowType, index: usize) -> Self {
        Self { start_date, end_date, window_type, index }
    }

    /// Number of calendar days in this window.
    pub fn days(&self) -> i64 {
        (self.end_date - self.start_date).num_days()
    }
}

/// A complete train/test split with purge and embargo.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowSplit {
    pub train: WindowSpec,
    pub test: WindowSpec,
    pub purge_days: u32,
    pub embargo_days: u32,
    pub index: usize,
}

impl WindowSplit {
    /// Verify no overlap between train and test.
    pub fn is_valid(&self) -> bool {
        self.train.end_date < self.test.start_date
    }

    /// Gap in calendar days between train end and test start.
    pub fn gap_days(&self) -> i64 {
        (self.test.start_date - self.train.end_date).num_days()
    }
}

/// Configuration for walk-forward validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardConfig {
    pub train_months: u32,
    pub test_months: u32,
    pub step_months: u32,
    pub purge_days: u32,
    pub embargo_days: u32,
    pub market: Market,
    pub grid: Option<GridConfig>,
}

impl Default for WalkForwardConfig {
    fn default() -> Self {
        Self {
            train_months: 6,
            test_months: 3,
            step_months: 3,
            purge_days: 5,
            embargo_days: 5,
            market: Market::BR,
            grid: None,
        }
    }
}

/// Parameter set for a single grid point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParamSet {
    pub top_n: usize,
    pub stop_loss_pct: Decimal,
    pub take_profit_pct: Decimal,
    pub max_weight: Decimal,
    pub turnover_cap: Decimal,
    pub min_score: Decimal,
}

impl Default for ParamSet {
    fn default() -> Self {
        Self {
            top_n: 10,
            stop_loss_pct: Decimal::new(15, 2),      // 0.15 = 15%
            take_profit_pct: Decimal::new(30, 2),    // 0.30 = 30%
            max_weight: Decimal::new(20, 2),         // 0.20 = 20%
            turnover_cap: Decimal::new(50, 2),       // 0.50 = 50%
            min_score: Decimal::ZERO,
        }
    }
}

/// Range specification for a parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamRange {
    pub min: Decimal,
    pub max: Decimal,
    pub step: Decimal,
}

impl ParamRange {
    pub fn new(min: Decimal, max: Decimal, step: Decimal) -> Self {
        Self { min, max, step }
    }

    /// Generate all values in this range.
    pub fn values(&self) -> Vec<Decimal> {
        let mut result = Vec::new();
        let mut v = self.min;
        while v <= self.max {
            result.push(v);
            v += self.step;
        }
        result
    }

    /// Number of values in this range.
    pub fn count(&self) -> usize {
        if self.step == Decimal::ZERO {
            return 1;
        }
        let diff = self.max - self.min;
        (diff / self.step).to_string().parse::<usize>().unwrap_or(1) + 1
    }
}

/// Grid configuration for parameter search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridConfig {
    pub top_n_range: Vec<usize>,
    pub stop_loss_range: ParamRange,
    pub take_profit_range: ParamRange,
    pub max_weight_range: ParamRange,
    pub turnover_cap_range: ParamRange,
    pub min_score_range: ParamRange,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            top_n_range: vec![5, 10, 15],
            stop_loss_range: ParamRange::new(
                Decimal::new(10, 2),  // 10%
                Decimal::new(20, 2),  // 20%
                Decimal::new(5, 2),   // step 5%
            ),
            take_profit_range: ParamRange::new(
                Decimal::new(20, 2),  // 20%
                Decimal::new(40, 2),  // 40%
                Decimal::new(10, 2),  // step 10%
            ),
            max_weight_range: ParamRange::new(
                Decimal::new(15, 2),  // 15%
                Decimal::new(25, 2),  // 25%
                Decimal::new(5, 2),   // step 5%
            ),
            turnover_cap_range: ParamRange::new(
                Decimal::new(30, 2),  // 30%
                Decimal::new(70, 2),  // 70%
                Decimal::new(20, 2),  // step 20%
            ),
            min_score_range: ParamRange::new(
                Decimal::ZERO,
                Decimal::new(5, 1),   // 0.5
                Decimal::new(25, 2),  // step 0.25
            ),
        }
    }
}

impl GridConfig {
    /// Generate all parameter combinations.
    pub fn generate_combinations(&self) -> Vec<ParamSet> {
        let stop_loss_vals = self.stop_loss_range.values();
        let take_profit_vals = self.take_profit_range.values();
        let max_weight_vals = self.max_weight_range.values();
        let turnover_cap_vals = self.turnover_cap_range.values();
        let min_score_vals = self.min_score_range.values();

        let mut combinations = Vec::new();

        for &top_n in &self.top_n_range {
            for &stop_loss in &stop_loss_vals {
                for &take_profit in &take_profit_vals {
                    for &max_weight in &max_weight_vals {
                        for &turnover_cap in &turnover_cap_vals {
                            for &min_score in &min_score_vals {
                                combinations.push(ParamSet {
                                    top_n,
                                    stop_loss_pct: stop_loss,
                                    take_profit_pct: take_profit,
                                    max_weight,
                                    turnover_cap,
                                    min_score,
                                });
                            }
                        }
                    }
                }
            }
        }

        combinations
    }

    /// Total number of combinations.
    pub fn total_combinations(&self) -> usize {
        self.top_n_range.len()
            * self.stop_loss_range.count()
            * self.take_profit_range.count()
            * self.max_weight_range.count()
            * self.turnover_cap_range.count()
            * self.min_score_range.count()
    }
}

/// Metrics for a single window.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WindowMetrics {
    pub total_return_pct: Decimal,
    pub cagr_pct: Decimal,
    pub volatility_ann: Decimal,
    pub sharpe_ratio: Decimal,
    pub max_drawdown_pct: Decimal,
    pub dd_duration_days: u32,
    pub turnover_avg_pct: Decimal,
    pub total_costs: Decimal,
    pub hit_rate: Option<Decimal>,
}

/// Result for a single window (train or test).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowResult {
    pub split: WindowSplit,
    pub train_metrics: WindowMetrics,
    pub test_metrics: WindowMetrics,
    pub selected_params: ParamSet,
    pub is_oos: bool,
}

/// Aggregate metrics across all windows.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregateMetrics {
    // Sharpe stats
    pub mean_sharpe: Decimal,
    pub median_sharpe: Decimal,
    pub std_sharpe: Decimal,
    
    // Return stats
    pub mean_return: Decimal,
    pub median_return: Decimal,
    pub std_return: Decimal,
    
    // Risk stats
    pub mean_drawdown: Decimal,
    pub worst_drawdown: Decimal,
    pub mean_volatility: Decimal,
    
    // Derived scores
    pub stability_score: Decimal,
    pub robustness_score: Decimal,
    
    // Window indices
    pub best_window_idx: usize,
    pub worst_window_idx: usize,
    
    // Totals
    pub total_windows: usize,
    pub total_months_tested: u32,
}

/// Complete walk-forward report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateReport {
    pub config: WalkForwardConfig,
    pub windows: Vec<WindowResult>,
    pub aggregate: AggregateMetrics,
    pub most_selected_params: ParamSet,
    pub generated_at: NaiveDate,
}

impl AggregateReport {
    /// Get the test-only (OOS) results.
    pub fn oos_results(&self) -> Vec<&WindowResult> {
        self.windows.iter().filter(|w| w.is_oos).collect()
    }

    /// Average Sharpe across OOS windows.
    pub fn oos_sharpe(&self) -> Decimal {
        let oos: Vec<_> = self.oos_results();
        if oos.is_empty() {
            return Decimal::ZERO;
        }
        let sum: Decimal = oos.iter().map(|w| w.test_metrics.sharpe_ratio).sum();
        sum / Decimal::from(oos.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_window_spec_days() {
        let spec = WindowSpec::new(
            NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            NaiveDate::from_ymd_opt(2024, 6, 30).unwrap(),
            WindowType::Train,
            0,
        );
        assert_eq!(spec.days(), 181);
    }

    #[test]
    fn test_window_split_valid() {
        let train = WindowSpec::new(
            NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            NaiveDate::from_ymd_opt(2024, 6, 25).unwrap(),
            WindowType::Train,
            0,
        );
        let test = WindowSpec::new(
            NaiveDate::from_ymd_opt(2024, 7, 5).unwrap(),
            NaiveDate::from_ymd_opt(2024, 9, 30).unwrap(),
            WindowType::Test,
            0,
        );
        let split = WindowSplit {
            train,
            test,
            purge_days: 5,
            embargo_days: 5,
            index: 0,
        };
        assert!(split.is_valid());
        assert_eq!(split.gap_days(), 10);
    }

    #[test]
    fn test_param_range_values() {
        let range = ParamRange::new(dec!(0.10), dec!(0.20), dec!(0.05));
        let vals = range.values();
        assert_eq!(vals.len(), 3);
        assert_eq!(vals[0], dec!(0.10));
        assert_eq!(vals[1], dec!(0.15));
        assert_eq!(vals[2], dec!(0.20));
    }

    #[test]
    fn test_grid_config_combinations() {
        let mut grid = GridConfig::default();
        // Reduce for test
        grid.top_n_range = vec![5, 10];
        grid.stop_loss_range = ParamRange::new(dec!(0.10), dec!(0.15), dec!(0.05));
        grid.take_profit_range = ParamRange::new(dec!(0.20), dec!(0.20), dec!(0.10));
        grid.max_weight_range = ParamRange::new(dec!(0.20), dec!(0.20), dec!(0.05));
        grid.turnover_cap_range = ParamRange::new(dec!(0.50), dec!(0.50), dec!(0.20));
        grid.min_score_range = ParamRange::new(dec!(0.0), dec!(0.0), dec!(0.25));

        let combos = grid.generate_combinations();
        // 2 top_n * 2 stop_loss * 1 * 1 * 1 * 1 = 4
        assert_eq!(combos.len(), 4);
    }

    #[test]
    fn test_default_config() {
        let config = WalkForwardConfig::default();
        assert_eq!(config.train_months, 6);
        assert_eq!(config.test_months, 3);
        assert_eq!(config.step_months, 3);
        assert_eq!(config.purge_days, 5);
        assert_eq!(config.embargo_days, 5);
    }
}

