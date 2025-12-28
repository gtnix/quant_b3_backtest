//! Comparator - compare experiment runs and detect regressions.
//!
//! This module provides regression detection for strategy experiments.
//! Thresholds can be configured via:
//! - CLI flags (--sharpe-threshold, --cagr-threshold, --dd-threshold)
//! - TOML configuration file
//! - Programmatically via `RegressionThresholds::builder()`

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::artifacts::ArtifactWriter;
use super::types::{ExperimentTraceEntry, RunMetadata, RunMetrics};

/// Result of comparing two experiment runs.
#[derive(Debug, Clone)]
pub struct CompareResult {
    /// Run A identifier
    pub run_a: String,
    /// Run B identifier
    pub run_b: String,
    /// Strategy A identifier
    pub strategy_a: String,
    /// Strategy B identifier
    pub strategy_b: String,
    /// Metric differences
    pub metric_diffs: HashMap<String, MetricDiff>,
    /// Configuration differences
    pub config_diffs: Vec<ConfigDiff>,
    /// Whether this comparison indicates a regression
    pub regression: bool,
    /// Reason for regression flag (if any)
    pub regression_reason: Option<String>,
}

/// Difference in a single metric between two runs.
#[derive(Debug, Clone)]
pub struct MetricDiff {
    pub name: String,
    pub value_a: f64,
    pub value_b: f64,
    pub diff_absolute: f64,
    pub diff_pct: f64,
    pub is_improvement: bool,
}

impl MetricDiff {
    /// Create a new metric diff (higher is better).
    pub fn new_higher_better(name: impl Into<String>, a: f64, b: f64) -> Self {
        let diff_absolute = b - a;
        let diff_pct = if a.abs() > 0.0001 {
            (b - a) / a.abs() * 100.0
        } else {
            0.0
        };
        Self {
            name: name.into(),
            value_a: a,
            value_b: b,
            diff_absolute,
            diff_pct,
            is_improvement: b > a,
        }
    }

    /// Create a new metric diff (lower is better, e.g., volatility, drawdown).
    pub fn new_lower_better(name: impl Into<String>, a: f64, b: f64) -> Self {
        let diff_absolute = b - a;
        let diff_pct = if a.abs() > 0.0001 {
            (b - a) / a.abs() * 100.0
        } else {
            0.0
        };
        Self {
            name: name.into(),
            value_a: a,
            value_b: b,
            diff_absolute,
            diff_pct,
            is_improvement: b < a, // Lower is better
        }
    }
}

/// Difference in configuration between two runs.
#[derive(Debug, Clone)]
pub struct ConfigDiff {
    pub path: String,
    pub value_a: Option<String>,
    pub value_b: Option<String>,
}

/// Configuration for regression detection.
///
/// Thresholds define when a metric change is considered a "regression".
/// Values are expressed as percentages (e.g., 0.20 = 20%).
///
/// # Example Configuration (TOML)
///
/// ```toml
/// [regression_thresholds]
/// sharpe_drop_pct = 0.15  # 15% drop triggers regression
/// max_dd_increase_pct = 0.20  # 20% worse drawdown triggers regression
/// cagr_drop_pct = 0.25  # 25% drop triggers regression
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionThresholds {
    /// Sharpe ratio drop threshold (e.g., 0.20 = 20% drop triggers regression)
    #[serde(default = "default_sharpe_drop")]
    pub sharpe_drop_pct: f64,
    /// Max drawdown increase threshold
    #[serde(default = "default_dd_increase")]
    pub max_dd_increase_pct: f64,
    /// CAGR drop threshold
    #[serde(default = "default_cagr_drop")]
    pub cagr_drop_pct: f64,
}

fn default_sharpe_drop() -> f64 { 0.20 }
fn default_dd_increase() -> f64 { 0.25 }
fn default_cagr_drop() -> f64 { 0.30 }

impl Default for RegressionThresholds {
    fn default() -> Self {
        Self {
            sharpe_drop_pct: default_sharpe_drop(),
            max_dd_increase_pct: default_dd_increase(),
            cagr_drop_pct: default_cagr_drop(),
        }
    }
}

impl RegressionThresholds {
    /// Create a new builder for RegressionThresholds.
    pub fn builder() -> RegressionThresholdsBuilder {
        RegressionThresholdsBuilder::default()
    }
    
    /// Load thresholds from a TOML file.
    ///
    /// The file should contain a `[regression_thresholds]` section:
    /// ```toml
    /// [regression_thresholds]
    /// sharpe_drop_pct = 0.15
    /// max_dd_increase_pct = 0.20
    /// cagr_drop_pct = 0.25
    /// ```
    pub fn load_from_file(path: &Path) -> Result<Self, ComparatorError> {
        let content = fs::read_to_string(path)
            .map_err(|e| ComparatorError::Io(e))?;
        
        #[derive(Deserialize)]
        struct ConfigFile {
            #[serde(default)]
            regression_thresholds: RegressionThresholds,
        }
        
        let config: ConfigFile = toml::from_str(&content)
            .map_err(|e| ComparatorError::ConfigParse(e.to_string()))?;
        
        Ok(config.regression_thresholds)
    }
    
    /// Format thresholds as a human-readable string.
    pub fn format(&self) -> String {
        format!(
            "Sharpe drop: {:.0}%, Max DD increase: {:.0}%, CAGR drop: {:.0}%",
            self.sharpe_drop_pct * 100.0,
            self.max_dd_increase_pct * 100.0,
            self.cagr_drop_pct * 100.0
        )
    }
}

/// Builder for RegressionThresholds.
#[derive(Debug, Clone, Default)]
pub struct RegressionThresholdsBuilder {
    sharpe_drop_pct: Option<f64>,
    max_dd_increase_pct: Option<f64>,
    cagr_drop_pct: Option<f64>,
}

impl RegressionThresholdsBuilder {
    /// Set the Sharpe ratio drop threshold.
    pub fn sharpe_drop(mut self, pct: f64) -> Self {
        self.sharpe_drop_pct = Some(pct);
        self
    }
    
    /// Set the max drawdown increase threshold.
    pub fn max_dd_increase(mut self, pct: f64) -> Self {
        self.max_dd_increase_pct = Some(pct);
        self
    }
    
    /// Set the CAGR drop threshold.
    pub fn cagr_drop(mut self, pct: f64) -> Self {
        self.cagr_drop_pct = Some(pct);
        self
    }
    
    /// Build the RegressionThresholds.
    pub fn build(self) -> RegressionThresholds {
        RegressionThresholds {
            sharpe_drop_pct: self.sharpe_drop_pct.unwrap_or(default_sharpe_drop()),
            max_dd_increase_pct: self.max_dd_increase_pct.unwrap_or(default_dd_increase()),
            cagr_drop_pct: self.cagr_drop_pct.unwrap_or(default_cagr_drop()),
        }
    }
}

/// Comparator for experiment runs.
pub struct Comparator {
    thresholds: RegressionThresholds,
    golden_dir: String,
}

impl Comparator {
    /// Create a new comparator with default thresholds.
    pub fn new() -> Self {
        Self {
            thresholds: RegressionThresholds::default(),
            golden_dir: "output/experiments".into(),
        }
    }

    /// Create comparator with custom thresholds.
    pub fn with_thresholds(thresholds: RegressionThresholds) -> Self {
        Self {
            thresholds,
            golden_dir: "output/experiments".into(),
        }
    }

    /// Set the golden strategies directory.
    pub fn with_golden_dir(mut self, dir: impl Into<String>) -> Self {
        self.golden_dir = dir.into();
        self
    }

    /// Compare two experiment runs by their directories.
    pub fn compare(&self, run_a: &Path, run_b: &Path) -> Result<CompareResult, ComparatorError> {
        // Read metadata and metrics from both runs
        let metadata_a = ArtifactWriter::read_metadata(run_a)?;
        let metadata_b = ArtifactWriter::read_metadata(run_b)?;
        let metrics_a = ArtifactWriter::read_metrics(run_a)?;
        let metrics_b = ArtifactWriter::read_metrics(run_b)?;

        // Compute metric diffs
        let metric_diffs = self.compute_metric_diffs(&metrics_a, &metrics_b);

        // Compute config diffs from metadata and traces
        let trace_a = ArtifactWriter::read_trace(run_a).ok();
        let trace_b = ArtifactWriter::read_trace(run_b).ok();
        let config_diffs = self.compute_config_diffs(&metadata_a, &metadata_b, &trace_a, &trace_b);

        // Check for regression
        let (regression, reason) = self.check_regression(&metrics_a, &metrics_b);

        Ok(CompareResult {
            run_a: metadata_a.run_id,
            run_b: metadata_b.run_id,
            strategy_a: metadata_a.strategy_id.clone(),
            strategy_b: metadata_b.strategy_id.clone(),
            metric_diffs,
            config_diffs,
            regression,
            regression_reason: reason,
        })
    }

    /// Compare a run against a golden strategy.
    pub fn compare_to_golden(
        &self,
        run: &Path,
        golden_id: &str,
    ) -> Result<CompareResult, ComparatorError> {
        // Find the golden run directory
        let golden_path = self.find_golden_run(golden_id)?;
        self.compare(&golden_path, run)
    }

    /// Find the most recent golden strategy run.
    fn find_golden_run(&self, golden_id: &str) -> Result<std::path::PathBuf, ComparatorError> {
        let base = Path::new(&self.golden_dir);
        if !base.exists() {
            return Err(ComparatorError::GoldenNotFound(format!(
                "Golden directory not found: {}. Create runs first with 'run' or 'run-batch'.",
                self.golden_dir
            )));
        }

        // Collect all available strategies for error message
        let mut all_strategies: Vec<String> = Vec::new();
        let mut matching_runs = Vec::new();

        for entry in std::fs::read_dir(base)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                let metadata_path = entry.path().join("metadata.json");
                if metadata_path.exists() {
                    if let Ok(metadata) = ArtifactWriter::read_metadata(&entry.path()) {
                        // Track all strategies for error message
                        if !all_strategies.contains(&metadata.strategy_id) {
                            all_strategies.push(metadata.strategy_id.clone());
                        }
                        
                        // Check for match (exact or partial)
                        if metadata.strategy_id == golden_id 
                            || metadata.strategy_id.contains(golden_id) 
                        {
                            matching_runs.push((entry.path(), metadata.timestamp_utc));
                        }
                    }
                }
            }
        }

        if matching_runs.is_empty() {
            all_strategies.sort();
            let available = if all_strategies.is_empty() {
                "none (no runs found)".to_string()
            } else {
                all_strategies.join(", ")
            };
            
            return Err(ComparatorError::GoldenNotFound(format!(
                "Golden strategy '{}' not found in '{}'. Available strategies: [{}]",
                golden_id, self.golden_dir, available
            )));
        }

        // Return most recent
        matching_runs.sort_by(|a, b| b.1.cmp(&a.1));
        Ok(matching_runs[0].0.clone())
    }
    
    /// List all available strategies in the golden directory.
    pub fn list_available_strategies(&self) -> Result<Vec<String>, ComparatorError> {
        let base = Path::new(&self.golden_dir);
        if !base.exists() {
            return Ok(Vec::new());
        }
        
        let mut strategies: Vec<String> = Vec::new();
        
        for entry in std::fs::read_dir(base)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                let metadata_path = entry.path().join("metadata.json");
                if metadata_path.exists() {
                    if let Ok(metadata) = ArtifactWriter::read_metadata(&entry.path()) {
                        if !strategies.contains(&metadata.strategy_id) {
                            strategies.push(metadata.strategy_id);
                        }
                    }
                }
            }
        }
        
        strategies.sort();
        Ok(strategies)
    }

    /// Compute metric differences between two runs.
    fn compute_metric_diffs(
        &self,
        a: &RunMetrics,
        b: &RunMetrics,
    ) -> HashMap<String, MetricDiff> {
        let mut diffs = HashMap::new();

        // Higher is better metrics
        diffs.insert(
            "cagr".into(),
            MetricDiff::new_higher_better("cagr", a.cagr, b.cagr),
        );
        diffs.insert(
            "sharpe_ratio".into(),
            MetricDiff::new_higher_better("sharpe_ratio", a.sharpe_ratio, b.sharpe_ratio),
        );
        diffs.insert(
            "sortino_ratio".into(),
            MetricDiff::new_higher_better("sortino_ratio", a.sortino_ratio, b.sortino_ratio),
        );
        diffs.insert(
            "calmar_ratio".into(),
            MetricDiff::new_higher_better("calmar_ratio", a.calmar_ratio, b.calmar_ratio),
        );
        diffs.insert(
            "hit_rate".into(),
            MetricDiff::new_higher_better("hit_rate", a.hit_rate, b.hit_rate),
        );
        diffs.insert(
            "profit_factor".into(),
            MetricDiff::new_higher_better("profit_factor", a.profit_factor, b.profit_factor),
        );

        // Lower is better metrics
        diffs.insert(
            "volatility".into(),
            MetricDiff::new_lower_better("volatility", a.volatility, b.volatility),
        );
        diffs.insert(
            "max_drawdown".into(),
            MetricDiff::new_lower_better("max_drawdown", a.max_drawdown, b.max_drawdown),
        );
        diffs.insert(
            "turnover_annual".into(),
            MetricDiff::new_lower_better("turnover_annual", a.turnover_annual, b.turnover_annual),
        );

        diffs
    }

    /// Compute configuration differences between two runs.
    ///
    /// Compares:
    /// - Strategy ID and version
    /// - Execution mode
    /// - Cost configuration
    /// - Pipeline steps (via trace)
    fn compute_config_diffs(
        &self,
        meta_a: &RunMetadata,
        meta_b: &RunMetadata,
        trace_a: &Option<Vec<ExperimentTraceEntry>>,
        trace_b: &Option<Vec<ExperimentTraceEntry>>,
    ) -> Vec<ConfigDiff> {
        let mut diffs = Vec::new();

        // Strategy ID
        if meta_a.strategy_id != meta_b.strategy_id {
            diffs.push(ConfigDiff {
                path: "strategy.id".into(),
                value_a: Some(meta_a.strategy_id.clone()),
                value_b: Some(meta_b.strategy_id.clone()),
            });
        }

        // Strategy version
        if meta_a.strategy_version != meta_b.strategy_version {
            diffs.push(ConfigDiff {
                path: "strategy.version".into(),
                value_a: Some(meta_a.strategy_version.clone()),
                value_b: Some(meta_b.strategy_version.clone()),
            });
        }

        // Execution mode
        if meta_a.execution_mode != meta_b.execution_mode {
            diffs.push(ConfigDiff {
                path: "execution_mode".into(),
                value_a: Some(format!("{:?}", meta_a.execution_mode)),
                value_b: Some(format!("{:?}", meta_b.execution_mode)),
            });
        }

        // Seed
        if meta_a.seed != meta_b.seed {
            diffs.push(ConfigDiff {
                path: "seed".into(),
                value_a: meta_a.seed.map(|s| s.to_string()),
                value_b: meta_b.seed.map(|s| s.to_string()),
            });
        }

        // Cost config
        if (meta_a.costs.trading_fee_pct - meta_b.costs.trading_fee_pct).abs() > 1e-10 {
            diffs.push(ConfigDiff {
                path: "costs.trading_fee_pct".into(),
                value_a: Some(meta_a.costs.trading_fee_pct.to_string()),
                value_b: Some(meta_b.costs.trading_fee_pct.to_string()),
            });
        }
        if (meta_a.costs.slippage_pct - meta_b.costs.slippage_pct).abs() > 1e-10 {
            diffs.push(ConfigDiff {
                path: "costs.slippage_pct".into(),
                value_a: Some(meta_a.costs.slippage_pct.to_string()),
                value_b: Some(meta_b.costs.slippage_pct.to_string()),
            });
        }

        // Pipeline steps from trace
        if let (Some(ta), Some(tb)) = (trace_a, trace_b) {
            // Compare step count
            if ta.len() != tb.len() {
                diffs.push(ConfigDiff {
                    path: "pipeline.step_count".into(),
                    value_a: Some(ta.len().to_string()),
                    value_b: Some(tb.len().to_string()),
                });
            }

            // Compare each step
            let max_steps = ta.len().max(tb.len());
            for i in 0..max_steps {
                let step_a = ta.get(i);
                let step_b = tb.get(i);

                match (step_a, step_b) {
                    (Some(sa), Some(sb)) => {
                        if sa.block_id != sb.block_id {
                            diffs.push(ConfigDiff {
                                path: format!("pipeline[{}].block_id", i),
                                value_a: Some(sa.block_id.clone()),
                                value_b: Some(sb.block_id.clone()),
                            });
                        }
                        if sa.block_type != sb.block_type {
                            diffs.push(ConfigDiff {
                                path: format!("pipeline[{}].block_type", i),
                                value_a: Some(sa.block_type.clone()),
                                value_b: Some(sb.block_type.clone()),
                            });
                        }
                        // Compare params_effective
                        self.diff_params(
                            &format!("pipeline[{}].params", i),
                            &sa.params_effective,
                            &sb.params_effective,
                            &mut diffs,
                        );
                    }
                    (Some(sa), None) => {
                        diffs.push(ConfigDiff {
                            path: format!("pipeline[{}]", i),
                            value_a: Some(format!("{}:{}", sa.block_type, sa.block_id)),
                            value_b: None,
                        });
                    }
                    (None, Some(sb)) => {
                        diffs.push(ConfigDiff {
                            path: format!("pipeline[{}]", i),
                            value_a: None,
                            value_b: Some(format!("{}:{}", sb.block_type, sb.block_id)),
                        });
                    }
                    (None, None) => {}
                }
            }
        }

        diffs
    }

    /// Compare two param maps and add diffs.
    fn diff_params(
        &self,
        base_path: &str,
        a: &HashMap<String, serde_json::Value>,
        b: &HashMap<String, serde_json::Value>,
        diffs: &mut Vec<ConfigDiff>,
    ) {
        // Check all keys from a
        for (key, val_a) in a {
            let val_b = b.get(key);
            match val_b {
                Some(vb) if vb != val_a => {
                    diffs.push(ConfigDiff {
                        path: format!("{}.{}", base_path, key),
                        value_a: Some(val_a.to_string()),
                        value_b: Some(vb.to_string()),
                    });
                }
                None => {
                    diffs.push(ConfigDiff {
                        path: format!("{}.{}", base_path, key),
                        value_a: Some(val_a.to_string()),
                        value_b: None,
                    });
                }
                _ => {} // Equal values
            }
        }

        // Check for keys only in b
        for (key, val_b) in b {
            if !a.contains_key(key) {
                diffs.push(ConfigDiff {
                    path: format!("{}.{}", base_path, key),
                    value_a: None,
                    value_b: Some(val_b.to_string()),
                });
            }
        }
    }

    /// Check if run B is a regression compared to run A (baseline).
    fn check_regression(&self, baseline: &RunMetrics, candidate: &RunMetrics) -> (bool, Option<String>) {
        // Check Sharpe ratio drop
        if baseline.sharpe_ratio > 0.0 {
            let sharpe_drop = (baseline.sharpe_ratio - candidate.sharpe_ratio) / baseline.sharpe_ratio;
            if sharpe_drop > self.thresholds.sharpe_drop_pct {
                return (
                    true,
                    Some(format!(
                        "Sharpe ratio dropped {:.1}% (threshold: {:.1}%)",
                        sharpe_drop * 100.0,
                        self.thresholds.sharpe_drop_pct * 100.0
                    )),
                );
            }
        }

        // Check max drawdown increase (drawdown is negative, so more negative is worse)
        // If baseline is -10% and candidate is -15%, that's 50% worse drawdown
        if baseline.max_drawdown < 0.0 {
            // candidate.abs() > baseline.abs() means worse drawdown
            let baseline_abs = baseline.max_drawdown.abs();
            let candidate_abs = candidate.max_drawdown.abs();
            if candidate_abs > baseline_abs {
                let dd_increase = (candidate_abs - baseline_abs) / baseline_abs;
                if dd_increase > self.thresholds.max_dd_increase_pct {
                    return (
                        true,
                        Some(format!(
                            "Max drawdown increased {:.1}% (threshold: {:.1}%)",
                            dd_increase * 100.0,
                            self.thresholds.max_dd_increase_pct * 100.0
                        )),
                    );
                }
            }
        }

        // Check CAGR drop
        if baseline.cagr > 0.0 {
            let cagr_drop = (baseline.cagr - candidate.cagr) / baseline.cagr;
            if cagr_drop > self.thresholds.cagr_drop_pct {
                return (
                    true,
                    Some(format!(
                        "CAGR dropped {:.1}% (threshold: {:.1}%)",
                        cagr_drop * 100.0,
                        self.thresholds.cagr_drop_pct * 100.0
                    )),
                );
            }
        }

        (false, None)
    }

    /// Generate a textual comparison report.
    pub fn generate_report(&self, result: &CompareResult) -> String {
        let mut report = String::new();

        report.push_str(&format!("\n=== Comparison Report ===\n"));
        report.push_str(&format!("Run A: {} ({})\n", result.run_a, result.strategy_a));
        report.push_str(&format!("Run B: {} ({})\n", result.run_b, result.strategy_b));
        report.push_str(&format!("\n"));

        // Regression status
        if result.regression {
            report.push_str(&format!(
                "⚠️  REGRESSION DETECTED: {}\n\n",
                result.regression_reason.as_deref().unwrap_or("Unknown reason")
            ));
        } else {
            report.push_str("✓ No regression detected\n\n");
        }

        // Metric comparison table
        report.push_str("Metric Comparison:\n");
        report.push_str(&format!(
            "{:<20} {:>12} {:>12} {:>12} {:>8}\n",
            "Metric", "Run A", "Run B", "Diff", "Status"
        ));
        report.push_str(&format!("{}\n", "-".repeat(68)));

        let mut metrics: Vec<_> = result.metric_diffs.values().collect();
        metrics.sort_by(|a, b| a.name.cmp(&b.name));

        for diff in metrics {
            let status = if diff.is_improvement { "↑" } else { "↓" };
            report.push_str(&format!(
                "{:<20} {:>12.4} {:>12.4} {:>+11.1}% {:>8}\n",
                diff.name, diff.value_a, diff.value_b, diff.diff_pct, status
            ));
        }

        report
    }
}

impl Default for Comparator {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ComparatorError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Artifact error: {0}")]
    Artifact(#[from] super::artifacts::ArtifactError),
    #[error("Golden strategy not found: {0}")]
    GoldenNotFound(String),
    #[error("Config parse error: {0}")]
    ConfigParse(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metrics(sharpe: f64, cagr: f64, max_dd: f64) -> RunMetrics {
        RunMetrics {
            cagr,
            volatility: 0.20,
            sharpe_ratio: sharpe,
            max_drawdown: max_dd,
            max_drawdown_duration_days: 30,
            turnover_annual: 2.0,
            hit_rate: 0.55,
            profit_factor: 1.5,
            total_trades: 100,
            total_days: 252,
            sortino_ratio: 1.0,
            calmar_ratio: 1.5,
            avg_win: 100.0,
            avg_loss: 66.0,
            win_loss_ratio: 1.5,
        }
    }

    #[test]
    fn test_no_regression() {
        let comparator = Comparator::new();
        let baseline = sample_metrics(1.0, 0.15, -0.10);
        let candidate = sample_metrics(1.1, 0.16, -0.09); // All better

        let (regression, _) = comparator.check_regression(&baseline, &candidate);
        assert!(!regression);
    }

    #[test]
    fn test_sharpe_regression() {
        let comparator = Comparator::new();
        let baseline = sample_metrics(1.0, 0.15, -0.10);
        let candidate = sample_metrics(0.7, 0.15, -0.10); // 30% Sharpe drop

        let (regression, reason) = comparator.check_regression(&baseline, &candidate);
        assert!(regression);
        assert!(reason.unwrap().contains("Sharpe"));
    }

    #[test]
    fn test_drawdown_regression() {
        let comparator = Comparator::new();
        let baseline = sample_metrics(1.0, 0.15, -0.10);
        let candidate = sample_metrics(1.0, 0.15, -0.15); // 50% worse drawdown

        let (regression, reason) = comparator.check_regression(&baseline, &candidate);
        assert!(regression);
        assert!(reason.unwrap().contains("drawdown"));
    }

    #[test]
    fn test_metric_diff_higher_better() {
        let diff = MetricDiff::new_higher_better("sharpe", 1.0, 1.2);
        assert!(diff.is_improvement);
        assert!((diff.diff_pct - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_metric_diff_lower_better() {
        let diff = MetricDiff::new_lower_better("volatility", 0.20, 0.15);
        assert!(diff.is_improvement); // Lower is better
        assert!((diff.diff_pct - (-25.0)).abs() < 0.1);
    }

    #[test]
    fn test_generate_report() {
        let comparator = Comparator::new();
        
        let mut metric_diffs = HashMap::new();
        metric_diffs.insert(
            "sharpe_ratio".into(),
            MetricDiff::new_higher_better("sharpe_ratio", 1.0, 1.2),
        );

        let result = CompareResult {
            run_a: "run-001".into(),
            run_b: "run-002".into(),
            strategy_a: "momentum".into(),
            strategy_b: "momentum_v2".into(),
            metric_diffs,
            config_diffs: Vec::new(),
            regression: false,
            regression_reason: None,
        };

        let report = comparator.generate_report(&result);
        assert!(report.contains("run-001"));
        assert!(report.contains("run-002"));
        assert!(report.contains("sharpe_ratio"));
        assert!(report.contains("No regression"));
    }
}

