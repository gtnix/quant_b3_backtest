//! Walk-Forward Validation types.
//!
//! Core data structures for rolling window validation with purge/embargo.
//! Supports nested 3-segment windows (Train/Val/Test) with PSR/DSR metrics.

use chrono::{Datelike, NaiveDate};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

use crate::filters::Market;

/// Type of window segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WindowType {
    Train,
    Validation,
    Test,
}

/// Selection criteria for choosing best ParamSet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum SelectionCriteria {
    /// Select by highest Sharpe ratio on validation
    Sharpe,
    /// Select by highest PSR (probability Sharpe > threshold)
    #[default]
    PSR,
    /// Composite score with penalties for turnover, cost, drawdown
    Composite,
}

/// Reason for parameter selection with score breakdown.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectionReason {
    pub criteria: SelectionCriteria,
    pub primary_score: Decimal,
    pub psr: Decimal,
    pub dsr: Option<Decimal>,
    pub turnover_penalty: Decimal,
    pub cost_penalty: Decimal,
    pub drawdown_penalty: Decimal,
    pub slippage_penalty: Decimal,
    pub capacity_penalty: Decimal,
    pub final_score: Decimal,
    pub tiebreaker_used: Option<String>,
}

impl Default for SelectionReason {
    fn default() -> Self {
        Self {
            criteria: SelectionCriteria::PSR,
            primary_score: Decimal::ZERO,
            psr: Decimal::ZERO,
            dsr: None,
            turnover_penalty: Decimal::ZERO,
            cost_penalty: Decimal::ZERO,
            drawdown_penalty: Decimal::ZERO,
            slippage_penalty: Decimal::ZERO,
            capacity_penalty: Decimal::ZERO,
            final_score: Decimal::ZERO,
            tiebreaker_used: None,
        }
    }
}

impl std::fmt::Display for SelectionReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.criteria {
            SelectionCriteria::Sharpe => {
                write!(f, "Sharpe={:.3}", self.primary_score)
            }
            SelectionCriteria::PSR => {
                write!(f, "PSR={:.3} (Sharpe={:.3})", self.psr, self.primary_score)
            }
            SelectionCriteria::Composite => {
                write!(
                    f,
                    "Composite={:.3} (Sharpe={:.3}, PSR={:.3}, penalties: turn={:.3}, cost={:.3}, dd={:.3})",
                    self.final_score, self.primary_score, self.psr,
                    self.turnover_penalty, self.cost_penalty, self.drawdown_penalty
                )
            }
        }
    }
}

/// Penalty configuration for composite selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenaltyConfig {
    /// Weight for turnover penalty (default 0.1)
    pub turnover_weight: Decimal,
    /// Weight for cost penalty (default 0.05)
    pub cost_weight: Decimal,
    /// Weight for max drawdown penalty (default 0.2)
    pub drawdown_weight: Decimal,
    /// Weight for slippage sensitivity penalty (default 0.05)
    #[serde(default = "default_slippage_weight")]
    pub slippage_weight: Decimal,
    /// Weight for low capacity penalty (default 0.1)
    #[serde(default = "default_capacity_weight")]
    pub capacity_weight: Decimal,
    /// Max annual turnover threshold for penalty (default 12.0 = 12x)
    #[serde(default = "default_max_turnover")]
    pub max_turnover_annual: Decimal,
    /// Min capacity in USD below which penalty applies (default 5M)
    #[serde(default = "default_min_capacity")]
    pub min_capacity_usd: Decimal,
}

fn default_slippage_weight() -> Decimal { dec!(0.05) }
fn default_capacity_weight() -> Decimal { dec!(0.10) }
fn default_max_turnover() -> Decimal { dec!(12.0) }
fn default_min_capacity() -> Decimal { dec!(5_000_000) }

impl Default for PenaltyConfig {
    fn default() -> Self {
        Self {
            turnover_weight: dec!(0.10),
            cost_weight: dec!(0.05),
            drawdown_weight: dec!(0.20),
            slippage_weight: dec!(0.05),
            capacity_weight: dec!(0.10),
            max_turnover_annual: dec!(12.0),
            min_capacity_usd: dec!(5_000_000),
        }
    }
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

/// A complete train/test split with purge and embargo (legacy 2-segment).
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

/// A complete nested 3-segment split: Train/Validation/Test.
/// Used for research-grade walk-forward with parameter selection on validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NestedWindowSplit {
    /// Training period for grid search
    pub train: WindowSpec,
    /// Validation period for parameter selection (PSR/DSR)
    pub val: WindowSpec,
    /// Test period for out-of-sample evaluation
    pub test: WindowSpec,
    /// Purge days between train and val
    pub purge_train_val: u32,
    /// Purge days between val and test
    pub purge_val_test: u32,
    /// Embargo days after each purge
    pub embargo_days: u32,
    /// Window index in the walk-forward sequence
    pub index: usize,
}

impl NestedWindowSplit {
    /// Verify no overlap between any segments.
    pub fn is_valid(&self) -> bool {
        self.train.end_date < self.val.start_date
            && self.val.end_date < self.test.start_date
    }

    /// Gap in calendar days between train end and val start.
    pub fn gap_train_val(&self) -> i64 {
        (self.val.start_date - self.train.end_date).num_days()
    }

    /// Gap in calendar days between val end and test start.
    pub fn gap_val_test(&self) -> i64 {
        (self.test.start_date - self.val.end_date).num_days()
    }

    /// Total duration in calendar days (all 3 segments + gaps).
    pub fn total_days(&self) -> i64 {
        (self.test.end_date - self.train.start_date).num_days()
    }
}

/// Configuration for walk-forward validation (legacy 2-segment).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardConfig {
    pub train_months: u32,
    pub test_months: u32,
    pub step_months: u32,
    pub purge_days: u32,
    pub embargo_days: u32,
    pub market: Market,
    pub grid: Option<GridConfig>,
    /// Execution model configuration for cost/slippage modeling.
    #[serde(default)]
    pub execution_config: Option<backtester_execution::ExecutionModelConfig>,
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
            execution_config: None,
        }
    }
}

/// Configuration for nested 3-segment walk-forward validation.
/// Train/Val/Test with PSR/DSR selection on validation period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NestedWalkForwardConfig {
    /// Training period length in months (default 4)
    pub train_months: u32,
    /// Validation period length in months (default 1)
    pub val_months: u32,
    /// Test period length in months (default 1)
    pub test_months: u32,
    /// Step forward in months between windows (default 3)
    pub step_months: u32,
    /// Purge days at segment transitions (default 5)
    pub purge_days: u32,
    /// Embargo days after purge (default 5)
    pub embargo_days: u32,
    /// Market (BR or US)
    pub market: Market,
    /// Grid search configuration
    pub grid: Option<GridConfig>,
    /// Selection criteria for best ParamSet (default PSR)
    pub selection_criteria: SelectionCriteria,
    /// PSR threshold (default 0.5)
    pub psr_threshold: Decimal,
    /// Penalty configuration for composite selection
    pub penalties: PenaltyConfig,
    /// Execution model configuration for cost/slippage modeling.
    #[serde(default)]
    pub execution_config: Option<backtester_execution::ExecutionModelConfig>,
    /// Institutional gates configuration.
    #[serde(default)]
    pub gates: Option<backtester_execution::InstitutionalGatesConfig>,
}

impl Default for NestedWalkForwardConfig {
    fn default() -> Self {
        Self {
            train_months: 4,
            val_months: 1,
            test_months: 1,
            step_months: 3,
            purge_days: 5,
            embargo_days: 5,
            market: Market::BR,
            grid: None,
            selection_criteria: SelectionCriteria::PSR,
            psr_threshold: dec!(0.5),
            penalties: PenaltyConfig::default(),
            execution_config: None,
            gates: None,
        }
    }
}

impl NestedWalkForwardConfig {
    /// Total window duration in months (train + val + test).
    pub fn window_months(&self) -> u32 {
        self.train_months + self.val_months + self.test_months
    }

    /// Estimate number of windows for a date range.
    pub fn estimate_windows(&self, start: NaiveDate, end: NaiveDate) -> usize {
        let total_months = ((end.year() - start.year()) * 12 
            + (end.month() as i32 - start.month() as i32)) as u32;
        if total_months < self.window_months() {
            return 0;
        }
        let available = total_months - self.window_months();
        (available / self.step_months) as usize + 1
    }
}

/// Parameter set for a single grid point.
/// Implements Ord for deterministic tie-breaking.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

impl PartialOrd for ParamSet {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ParamSet {
    /// Lexicographic ordering for deterministic tie-breaking.
    fn cmp(&self, other: &Self) -> Ordering {
        self.top_n.cmp(&other.top_n)
            .then_with(|| self.stop_loss_pct.cmp(&other.stop_loss_pct))
            .then_with(|| self.take_profit_pct.cmp(&other.take_profit_pct))
            .then_with(|| self.max_weight.cmp(&other.max_weight))
            .then_with(|| self.turnover_cap.cmp(&other.turnover_cap))
            .then_with(|| self.min_score.cmp(&other.min_score))
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
    /// Skewness of returns (for PSR/DSR calculation)
    pub skewness: Decimal,
    /// Excess kurtosis of returns (for PSR/DSR calculation)
    pub kurtosis: Decimal,
    /// Number of return observations
    pub n_observations: usize,
    /// Probabilistic Sharpe Ratio (probability Sharpe > threshold)
    pub psr: Option<Decimal>,
    /// Deflated Sharpe Ratio (adjusted for multiple testing)
    pub dsr: Option<Decimal>,
    /// Detailed cost report (optional, for PM-ready analysis).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_report: Option<backtester_execution::cost_report::CostReport>,
}

/// Result for a single window (legacy 2-segment).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowResult {
    pub split: WindowSplit,
    pub train_metrics: WindowMetrics,
    pub test_metrics: WindowMetrics,
    pub selected_params: ParamSet,
    pub is_oos: bool,
}

/// Result for a nested 3-segment window (Train/Val/Test).
/// Used for research-grade walk-forward with PSR/DSR selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NestedWindowResult {
    /// The 3-segment split specification
    pub split: NestedWindowSplit,
    /// Metrics from training period (grid search)
    pub metrics_train: WindowMetrics,
    /// Metrics from validation period (selection)
    pub metrics_val: WindowMetrics,
    /// Metrics from test period (out-of-sample)
    pub metrics_test: WindowMetrics,
    /// Selected parameters based on validation
    pub selected_params: ParamSet,
    /// Reason for parameter selection
    pub selection_reason: SelectionReason,
    /// PSR on validation set
    pub psr_val: Decimal,
    /// DSR on validation set (if calculated)
    pub dsr_val: Option<Decimal>,
    /// Number of ParamSets tested in grid search
    pub n_trials: usize,
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
    
    // PSR/DSR stats (for nested walk-forward)
    pub mean_psr: Decimal,
    pub median_psr: Decimal,
    pub mean_dsr: Option<Decimal>,
    pub median_dsr: Option<Decimal>,
    
    // OOS stats
    pub oos_sharpe_mean: Decimal,
    pub oos_return_mean: Decimal,
    pub oos_psr_mean: Decimal,
}

/// Complete walk-forward report (legacy 2-segment).
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

/// Complete nested walk-forward report (3-segment).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NestedAggregateReport {
    pub config: NestedWalkForwardConfig,
    pub windows: Vec<NestedWindowResult>,
    pub aggregate: AggregateMetrics,
    pub most_selected_params: ParamSet,
    pub generated_at: NaiveDate,
}

impl NestedAggregateReport {
    /// Average Sharpe across all test (OOS) windows.
    pub fn oos_sharpe(&self) -> Decimal {
        if self.windows.is_empty() {
            return Decimal::ZERO;
        }
        let sum: Decimal = self.windows.iter()
            .map(|w| w.metrics_test.sharpe_ratio)
            .sum();
        sum / Decimal::from(self.windows.len())
    }

    /// Average PSR across all validation windows.
    pub fn mean_psr(&self) -> Decimal {
        if self.windows.is_empty() {
            return Decimal::ZERO;
        }
        let sum: Decimal = self.windows.iter()
            .map(|w| w.psr_val)
            .sum();
        sum / Decimal::from(self.windows.len())
    }

    /// Average DSR across all validation windows (if available).
    pub fn mean_dsr(&self) -> Option<Decimal> {
        let dsrs: Vec<_> = self.windows.iter()
            .filter_map(|w| w.dsr_val)
            .collect();
        if dsrs.is_empty() {
            return None;
        }
        Some(dsrs.iter().sum::<Decimal>() / Decimal::from(dsrs.len()))
    }

    /// Get train metrics for all windows.
    pub fn train_metrics(&self) -> Vec<&WindowMetrics> {
        self.windows.iter().map(|w| &w.metrics_train).collect()
    }

    /// Get validation metrics for all windows.
    pub fn val_metrics(&self) -> Vec<&WindowMetrics> {
        self.windows.iter().map(|w| &w.metrics_val).collect()
    }

    /// Get test metrics for all windows.
    pub fn test_metrics(&self) -> Vec<&WindowMetrics> {
        self.windows.iter().map(|w| &w.metrics_test).collect()
    }
}

/// Candidate for selection (used during grid search evaluation).
#[derive(Debug, Clone)]
pub struct SelectionCandidate {
    pub params: ParamSet,
    pub sharpe: Decimal,
    pub psr: Decimal,
    pub dsr: Option<Decimal>,
    pub turnover: Decimal,
    pub costs: Decimal,
    pub max_drawdown: Decimal,
    pub composite_score: Decimal,
}

impl SelectionCandidate {
    /// Compare with deterministic tie-breakers.
    /// Returns Ordering for sorting (higher score first, then lower turnover, etc.)
    pub fn compare_with_tiebreaker(&self, other: &Self, criteria: SelectionCriteria) -> Ordering {
        let self_score = self.get_primary_score(criteria);
        let other_score = other.get_primary_score(criteria);
        
        // Primary: score (higher is better)
        match other_score.cmp(&self_score) {
            Ordering::Equal => {}
            order => return order,
        }
        // Tie-breaker 1: turnover (lower is better)
        match self.turnover.cmp(&other.turnover) {
            Ordering::Equal => {}
            order => return order,
        }
        // Tie-breaker 2: costs (lower is better)
        match self.costs.cmp(&other.costs) {
            Ordering::Equal => {}
            order => return order,
        }
        // Tie-breaker 3: max_drawdown (lower is better)
        match self.max_drawdown.cmp(&other.max_drawdown) {
            Ordering::Equal => {}
            order => return order,
        }
        // Tie-breaker 4: lexicographic ParamSet
        self.params.cmp(&other.params)
    }

    fn get_primary_score(&self, criteria: SelectionCriteria) -> Decimal {
        match criteria {
            SelectionCriteria::Sharpe => self.sharpe,
            SelectionCriteria::PSR => self.psr,
            SelectionCriteria::Composite => self.composite_score,
        }
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
    fn test_nested_window_split_valid() {
        let train = WindowSpec::new(
            NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            NaiveDate::from_ymd_opt(2024, 4, 25).unwrap(),
            WindowType::Train,
            0,
        );
        let val = WindowSpec::new(
            NaiveDate::from_ymd_opt(2024, 5, 5).unwrap(),
            NaiveDate::from_ymd_opt(2024, 5, 30).unwrap(),
            WindowType::Validation,
            0,
        );
        let test = WindowSpec::new(
            NaiveDate::from_ymd_opt(2024, 6, 10).unwrap(),
            NaiveDate::from_ymd_opt(2024, 7, 5).unwrap(),
            WindowType::Test,
            0,
        );
        let split = NestedWindowSplit {
            train,
            val,
            test,
            purge_train_val: 5,
            purge_val_test: 5,
            embargo_days: 5,
            index: 0,
        };
        assert!(split.is_valid());
        assert_eq!(split.gap_train_val(), 10);
        assert_eq!(split.gap_val_test(), 11);
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

    #[test]
    fn test_nested_default_config() {
        let config = NestedWalkForwardConfig::default();
        assert_eq!(config.train_months, 4);
        assert_eq!(config.val_months, 1);
        assert_eq!(config.test_months, 1);
        assert_eq!(config.step_months, 3);
        assert_eq!(config.psr_threshold, dec!(0.5));
        assert_eq!(config.selection_criteria, SelectionCriteria::PSR);
    }

    #[test]
    fn test_param_set_ordering() {
        let p1 = ParamSet {
            top_n: 10,
            stop_loss_pct: dec!(0.10),
            take_profit_pct: dec!(0.20),
            max_weight: dec!(0.15),
            turnover_cap: dec!(0.50),
            min_score: dec!(0),
        };
        let p2 = ParamSet {
            top_n: 10,
            stop_loss_pct: dec!(0.15),
            take_profit_pct: dec!(0.20),
            max_weight: dec!(0.15),
            turnover_cap: dec!(0.50),
            min_score: dec!(0),
        };
        assert!(p1 < p2);  // lower stop_loss comes first
    }

    #[test]
    fn test_selection_candidate_tiebreaker() {
        let c1 = SelectionCandidate {
            params: ParamSet::default(),
            sharpe: dec!(1.0),
            psr: dec!(0.7),
            dsr: Some(dec!(0.6)),
            turnover: dec!(0.20),  // lower turnover
            costs: dec!(0.01),
            max_drawdown: dec!(0.10),
            composite_score: dec!(0.8),
        };
        let c2 = SelectionCandidate {
            params: ParamSet::default(),
            sharpe: dec!(1.0),
            psr: dec!(0.7),  // same PSR
            dsr: Some(dec!(0.6)),
            turnover: dec!(0.30),  // higher turnover
            costs: dec!(0.01),
            max_drawdown: dec!(0.10),
            composite_score: dec!(0.8),
        };
        // c1 should come before c2 (lower turnover wins)
        assert_eq!(
            c1.compare_with_tiebreaker(&c2, SelectionCriteria::PSR),
            Ordering::Less
        );
    }

    #[test]
    fn test_selection_reason_display() {
        let reason = SelectionReason {
            criteria: SelectionCriteria::PSR,
            primary_score: dec!(1.2),
            psr: dec!(0.72),
            dsr: Some(dec!(0.65)),
            turnover_penalty: dec!(0),
            cost_penalty: dec!(0),
            drawdown_penalty: dec!(0),
            final_score: dec!(0.72),
            tiebreaker_used: None,
        };
        let display = format!("{}", reason);
        assert!(display.contains("PSR=0.72"));
        assert!(display.contains("Sharpe=1.2"));
    }

    #[test]
    fn test_penalty_config_default() {
        let p = PenaltyConfig::default();
        assert_eq!(p.turnover_weight, dec!(0.10));
        assert_eq!(p.cost_weight, dec!(0.05));
        assert_eq!(p.drawdown_weight, dec!(0.20));
    }
}

