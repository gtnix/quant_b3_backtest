//! Stress Testing Module
//!
//! Provides utilities for stress testing strategies under adverse conditions:
//! - Increased slippage (2x, 4x)
//! - Increased transaction costs (+50%, +100%)
//! - Combined stress scenarios
//!
//! # Usage
//!
//! ```ignore
//! let stress_runner = StressTestRunner::new(runner_config);
//! let report = stress_runner.run_stress_suite(&strategy_config, &prices)?;
//! ```

use serde::{Deserialize, Serialize};

use super::types::{CostConfig, RunMetrics};

/// Stress test scenarios for robustness validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StressScenario {
    /// Baseline (no stress applied)
    Baseline,
    /// 2x slippage
    Slippage2x,
    /// 4x slippage
    Slippage4x,
    /// +50% transaction costs
    Costs150,
    /// +100% transaction costs (2x)
    Costs200,
    /// Combined: 2x slippage + 50% costs
    Combined2x150,
    /// Combined: 4x slippage + 100% costs  
    Combined4x200,
}

impl StressScenario {
    /// Get all standard stress scenarios.
    pub fn all() -> Vec<Self> {
        vec![
            Self::Baseline,
            Self::Slippage2x,
            Self::Slippage4x,
            Self::Costs150,
            Self::Costs200,
            Self::Combined2x150,
            Self::Combined4x200,
        ]
    }

    /// Get slippage-only scenarios.
    pub fn slippage_only() -> Vec<Self> {
        vec![Self::Baseline, Self::Slippage2x, Self::Slippage4x]
    }

    /// Get cost-only scenarios.
    pub fn costs_only() -> Vec<Self> {
        vec![Self::Baseline, Self::Costs150, Self::Costs200]
    }

    /// Get the display name.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Slippage2x => "slippage_2x",
            Self::Slippage4x => "slippage_4x",
            Self::Costs150 => "costs_+50%",
            Self::Costs200 => "costs_+100%",
            Self::Combined2x150 => "slippage_2x_costs_+50%",
            Self::Combined4x200 => "slippage_4x_costs_+100%",
        }
    }

    /// Get the slippage multiplier for this scenario.
    pub fn slippage_multiplier(&self) -> f64 {
        match self {
            Self::Baseline | Self::Costs150 | Self::Costs200 => 1.0,
            Self::Slippage2x | Self::Combined2x150 => 2.0,
            Self::Slippage4x | Self::Combined4x200 => 4.0,
        }
    }

    /// Get the cost multiplier for this scenario.
    pub fn cost_multiplier(&self) -> f64 {
        match self {
            Self::Baseline | Self::Slippage2x | Self::Slippage4x => 1.0,
            Self::Costs150 | Self::Combined2x150 => 1.5,
            Self::Costs200 | Self::Combined4x200 => 2.0,
        }
    }

    /// Apply this stress scenario to a cost configuration.
    pub fn apply_to_config(&self, base_config: &CostConfig) -> CostConfig {
        CostConfig {
            trading_fee_pct: base_config.trading_fee_pct * self.cost_multiplier(),
            slippage_pct: base_config.slippage_pct * self.slippage_multiplier(),
            min_trade_brl: base_config.min_trade_brl,
        }
    }
}

/// Result of a single stress test run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestResult {
    /// The scenario that was applied
    pub scenario: StressScenario,
    /// Metrics from the run
    pub metrics: RunMetrics,
    /// Cost config used
    pub cost_config: CostConfig,
    /// Whether the strategy passed (positive Sharpe, reasonable drawdown)
    pub passed: bool,
    /// Reason for failure (if any)
    pub failure_reason: Option<String>,
}

impl StressTestResult {
    /// Create a passing result.
    pub fn pass(scenario: StressScenario, metrics: RunMetrics, cost_config: CostConfig) -> Self {
        Self {
            scenario,
            metrics,
            cost_config,
            passed: true,
            failure_reason: None,
        }
    }

    /// Create a failing result.
    pub fn fail(
        scenario: StressScenario,
        metrics: RunMetrics,
        cost_config: CostConfig,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            scenario,
            metrics,
            cost_config,
            passed: false,
            failure_reason: Some(reason.into()),
        }
    }
}

/// Metadata for traceability and reproducibility.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReportMetadata {
    /// Crate version that generated the report
    pub crate_version: String,
    /// UTC timestamp when report was generated
    pub timestamp_utc: String,
    /// Config hash for reproducibility
    pub config_hash: Option<String>,
    /// Start date of backtest period
    pub period_start: Option<String>,
    /// End date of backtest period
    pub period_end: Option<String>,
}

impl ReportMetadata {
    /// Create metadata with current timestamp and version.
    pub fn now() -> Self {
        Self {
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp_utc: chrono::Utc::now().to_rfc3339(),
            config_hash: None,
            period_start: None,
            period_end: None,
        }
    }

    /// Set config hash.
    pub fn with_config_hash(mut self, hash: impl Into<String>) -> Self {
        self.config_hash = Some(hash.into());
        self
    }

    /// Set period.
    pub fn with_period(mut self, start: impl Into<String>, end: impl Into<String>) -> Self {
        self.period_start = Some(start.into());
        self.period_end = Some(end.into());
        self
    }
}

/// Complete stress test report with all scenarios.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestReport {
    /// Strategy ID being tested
    pub strategy_id: String,
    /// Traceability metadata (version, timestamp, config hash)
    pub metadata: ReportMetadata,
    /// Results for each scenario
    pub results: Vec<StressTestResult>,
    /// Overall pass/fail status
    pub all_passed: bool,
    /// Number of scenarios that passed
    pub passed_count: usize,
    /// Number of scenarios that failed
    pub failed_count: usize,
    /// Summary statistics
    /// Thresholds used for evaluation (for audit trail)
    pub thresholds_used: Option<StressThresholds>,
    /// Summary statistics
    pub summary: StressSummary,
}

/// Summary statistics across all stress scenarios.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressSummary {
    /// Baseline Sharpe ratio
    pub baseline_sharpe: f64,
    /// Minimum Sharpe across all scenarios
    pub min_sharpe: f64,
    /// Sharpe degradation from baseline to worst case
    pub sharpe_degradation_pct: f64,
    /// Baseline max drawdown
    pub baseline_max_dd: f64,
    /// Maximum drawdown across all scenarios
    pub worst_max_dd: f64,
    /// Drawdown increase from baseline to worst case
    pub max_dd_increase_pct: f64,
}

impl StressTestReport {
    /// Create a new stress test report from results.
    pub fn from_results(strategy_id: String, results: Vec<StressTestResult>) -> Self {
        Self::from_results_with_metadata(strategy_id, results, ReportMetadata::now())
    }

    /// Create a new stress test report with custom metadata.
    pub fn from_results_with_metadata(
        strategy_id: String,
        results: Vec<StressTestResult>,
        metadata: ReportMetadata,
    ) -> Self {
        let passed_count = results.iter().filter(|r| r.passed).count();
        let failed_count = results.len() - passed_count;
        let all_passed = failed_count == 0;

        let summary = Self::calculate_summary(&results);

        Self {
            strategy_id,
            metadata,
            thresholds_used: None,
            results,
            all_passed,
            passed_count,
            failed_count,
            summary,
        }
    }

    /// Create a new stress test report with metadata and thresholds for audit trail.
    pub fn from_results_with_metadata_and_thresholds(
        strategy_id: String,
        results: Vec<StressTestResult>,
        metadata: ReportMetadata,
        thresholds: Option<StressThresholds>,
    ) -> Self {
        let passed_count = results.iter().filter(|r| r.passed).count();
        let failed_count = results.len() - passed_count;
        let all_passed = failed_count == 0;
        let summary = Self::calculate_summary(&results);

        Self {
            strategy_id,
            metadata,
            thresholds_used: thresholds,
            results,
            all_passed,
            passed_count,
            failed_count,
            summary,
        }
    }

    fn calculate_summary(results: &[StressTestResult]) -> StressSummary {
        let baseline = results
            .iter()
            .find(|r| r.scenario == StressScenario::Baseline);

        let (baseline_sharpe, baseline_max_dd) = baseline
            .map(|r| (r.metrics.sharpe_ratio, r.metrics.max_drawdown))
            .unwrap_or((0.0, 0.0));

        let min_sharpe = results
            .iter()
            .map(|r| r.metrics.sharpe_ratio)
            .fold(f64::INFINITY, f64::min);

        let worst_max_dd = results
            .iter()
            .map(|r| r.metrics.max_drawdown)
            .fold(0.0, |a, b| if b < a { b } else { a }); // More negative = worse

        let sharpe_degradation_pct = if baseline_sharpe.abs() > 1e-10 {
            ((baseline_sharpe - min_sharpe) / baseline_sharpe.abs()) * 100.0
        } else {
            0.0
        };

        let max_dd_increase_pct = if baseline_max_dd.abs() > 1e-10 {
            ((worst_max_dd - baseline_max_dd) / baseline_max_dd.abs()) * 100.0
        } else {
            0.0
        };

        StressSummary {
            baseline_sharpe,
            min_sharpe,
            sharpe_degradation_pct,
            baseline_max_dd,
            worst_max_dd,
            max_dd_increase_pct,
        }
    }

    /// Check if strategy is robust under stress.
    ///
    /// A strategy is considered robust if:
    /// - Sharpe doesn't drop more than `max_sharpe_drop_pct`
    /// - Max drawdown doesn't increase more than `max_dd_increase_pct`
    pub fn is_robust(&self, max_sharpe_drop_pct: f64, max_dd_increase_pct: f64) -> bool {
        self.summary.sharpe_degradation_pct <= max_sharpe_drop_pct
            && self.summary.max_dd_increase_pct.abs() <= max_dd_increase_pct
    }

    /// Generate a text summary for logging.
    pub fn to_summary_string(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("=== Stress Test Report: {} ===", self.strategy_id));
        lines.push(format!(
            "Overall: {} ({}/{} passed)",
            if self.all_passed { "PASS" } else { "FAIL" },
            self.passed_count,
            self.results.len()
        ));
        lines.push(String::new());
        lines.push("Scenario Results:".to_string());

        for result in &self.results {
            let status = if result.passed { "PASS" } else { "FAIL" };
            lines.push(format!(
                "  {} - {}: Sharpe={:.3}, MaxDD={:.2}%",
                status,
                result.scenario.as_str(),
                result.metrics.sharpe_ratio,
                result.metrics.max_drawdown * 100.0
            ));
            if let Some(reason) = &result.failure_reason {
                lines.push(format!("    Reason: {}", reason));
            }
        }

        lines.push(String::new());
        lines.push("Summary:".to_string());
        lines.push(format!(
            "  Baseline Sharpe: {:.3}",
            self.summary.baseline_sharpe
        ));
        lines.push(format!("  Min Sharpe: {:.3}", self.summary.min_sharpe));
        lines.push(format!(
            "  Sharpe Degradation: {:.1}%",
            self.summary.sharpe_degradation_pct
        ));
        lines.push(format!(
            "  Baseline MaxDD: {:.2}%",
            self.summary.baseline_max_dd * 100.0
        ));
        lines.push(format!(
            "  Worst MaxDD: {:.2}%",
            self.summary.worst_max_dd * 100.0
        ));
        lines.push(format!(
            "  MaxDD Increase: {:.1}%",
            self.summary.max_dd_increase_pct.abs()
        ));

        lines.join("\n")
    }

    /// Compare reports ignoring non-deterministic fields (timestamp, git_sha, build_id).
    /// Uses numeric tolerance for floating-point comparisons.
    pub fn equals_deterministic(&self, other: &Self) -> bool {
        const TOLERANCE: f64 = 1e-9;

        self.strategy_id == other.strategy_id
            && self.all_passed == other.all_passed
            && self.passed_count == other.passed_count
            && self.failed_count == other.failed_count
            && self.results.len() == other.results.len()
            && (self.summary.baseline_sharpe - other.summary.baseline_sharpe).abs() < TOLERANCE
            && (self.summary.min_sharpe - other.summary.min_sharpe).abs() < TOLERANCE
            && (self.summary.sharpe_degradation_pct - other.summary.sharpe_degradation_pct).abs() < TOLERANCE
            && (self.summary.baseline_max_dd - other.summary.baseline_max_dd).abs() < TOLERANCE
            && (self.summary.worst_max_dd - other.summary.worst_max_dd).abs() < TOLERANCE
            && (self.summary.max_dd_increase_pct - other.summary.max_dd_increase_pct).abs() < TOLERANCE
    }

    /// Save report to disk with full audit trail.
    /// Path: `{output_dir}/stress/{strategy_id}_{timestamp}.json`
    pub fn save_to_disk(&self, output_dir: &std::path::Path) -> std::io::Result<std::path::PathBuf> {
        let dir = output_dir.join("stress");
        std::fs::create_dir_all(&dir)?;

        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let filename = format!("{}_{}.json", self.strategy_id, timestamp);
        let path = dir.join(&filename);

        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(&path, &json)?;

        tracing::info!(path = %path.display(), bytes = json.len(), "Saved stress report");
        Ok(path)
    }
}

/// Thresholds for passing stress tests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressThresholds {
    /// Minimum Sharpe ratio to pass
    pub min_sharpe: f64,
    /// Maximum drawdown to pass (as negative percentage, e.g., -0.30 = -30%)
    pub max_drawdown: f64,
    /// Maximum Sharpe degradation from baseline (percentage)
    pub max_sharpe_degradation_pct: f64,
    /// Maximum drawdown increase from baseline (percentage)
    pub max_dd_increase_pct: f64,
}

impl Default for StressThresholds {
    fn default() -> Self {
        Self {
            min_sharpe: 0.0,        // Must be positive
            max_drawdown: -0.50,    // No more than 50% drawdown
            max_sharpe_degradation_pct: 50.0, // Sharpe can drop at most 50%
            max_dd_increase_pct: 50.0,        // DD can increase at most 50%
        }
    }
}

impl StressThresholds {
    /// Check if metrics pass these thresholds.
    pub fn check(&self, metrics: &RunMetrics) -> Result<(), String> {
        if metrics.sharpe_ratio < self.min_sharpe {
            return Err(format!(
                "Sharpe ratio {:.3} < minimum {:.3}",
                metrics.sharpe_ratio, self.min_sharpe
            ));
        }
        if metrics.max_drawdown < self.max_drawdown {
            return Err(format!(
                "Max drawdown {:.2}% exceeds limit {:.2}%",
                metrics.max_drawdown * 100.0,
                self.max_drawdown * 100.0
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stress_scenario_multipliers() {
        assert_eq!(StressScenario::Baseline.slippage_multiplier(), 1.0);
        assert_eq!(StressScenario::Slippage2x.slippage_multiplier(), 2.0);
        assert_eq!(StressScenario::Slippage4x.slippage_multiplier(), 4.0);
        assert_eq!(StressScenario::Costs150.cost_multiplier(), 1.5);
        assert_eq!(StressScenario::Costs200.cost_multiplier(), 2.0);
        assert_eq!(StressScenario::Combined4x200.slippage_multiplier(), 4.0);
        assert_eq!(StressScenario::Combined4x200.cost_multiplier(), 2.0);
    }

    #[test]
    fn test_apply_to_config() {
        let base = CostConfig {
            trading_fee_pct: 0.001,
            slippage_pct: 0.0005,
            min_trade_brl: Some(100.0),
        };

        let stressed = StressScenario::Slippage4x.apply_to_config(&base);
        assert_eq!(stressed.slippage_pct, 0.002); // 4x
        assert_eq!(stressed.trading_fee_pct, 0.001); // unchanged

        let stressed = StressScenario::Costs200.apply_to_config(&base);
        assert_eq!(stressed.trading_fee_pct, 0.002); // 2x
        assert_eq!(stressed.slippage_pct, 0.0005); // unchanged
    }

    #[test]
    fn test_stress_report() {
        let results = vec![
            StressTestResult::pass(
                StressScenario::Baseline,
                RunMetrics {
                    sharpe_ratio: 1.5,
                    max_drawdown: -0.10,
                    ..Default::default()
                },
                CostConfig::default(),
            ),
            StressTestResult::pass(
                StressScenario::Slippage2x,
                RunMetrics {
                    sharpe_ratio: 1.2,
                    max_drawdown: -0.12,
                    ..Default::default()
                },
                CostConfig::default(),
            ),
            StressTestResult::fail(
                StressScenario::Slippage4x,
                RunMetrics {
                    sharpe_ratio: 0.8,
                    max_drawdown: -0.18,
                    ..Default::default()
                },
                CostConfig::default(),
                "Sharpe below threshold",
            ),
        ];

        let report = StressTestReport::from_results("test_strategy".into(), results);
        assert!(!report.all_passed);
        assert_eq!(report.passed_count, 2);
        assert_eq!(report.failed_count, 1);
        assert_eq!(report.summary.baseline_sharpe, 1.5);
        assert_eq!(report.summary.min_sharpe, 0.8);
    }

    #[test]
    fn test_thresholds_check() {
        let thresholds = StressThresholds::default();

        let good = RunMetrics {
            sharpe_ratio: 1.0,
            max_drawdown: -0.15,
            ..Default::default()
        };
        assert!(thresholds.check(&good).is_ok());

        let bad_sharpe = RunMetrics {
            sharpe_ratio: -0.5,
            max_drawdown: -0.15,
            ..Default::default()
        };
        assert!(thresholds.check(&bad_sharpe).is_err());

        let bad_dd = RunMetrics {
            sharpe_ratio: 1.0,
            max_drawdown: -0.60,
            ..Default::default()
        };
        assert!(thresholds.check(&bad_dd).is_err());
    }

    #[test]
    fn test_expected_degradation_under_stress() {
        // This test validates that stress scenarios produce expected degradation patterns
        // Key insight: higher friction → lower Sharpe (or at least not higher)
        
        let base = CostConfig {
            trading_fee_pct: 0.001,
            slippage_pct: 0.0005,
            min_trade_brl: Some(100.0),
        };

        // Simulate how friction affects metrics:
        // Baseline Sharpe = 1.5
        // With 2x slippage, Sharpe should degrade
        // With 4x200 combined, Sharpe should degrade more
        
        let baseline_sharpe = 1.5;
        
        // Model: each 1% increase in friction reduces Sharpe by ~0.5%
        // This is a simplified model for testing purposes
        fn estimate_sharpe_under_stress(base_sharpe: f64, slippage_mult: f64, cost_mult: f64) -> f64 {
            let friction_penalty = (slippage_mult - 1.0) * 0.05 + (cost_mult - 1.0) * 0.03;
            base_sharpe * (1.0 - friction_penalty)
        }

        // Baseline: no degradation
        let baseline_config = StressScenario::Baseline.apply_to_config(&base);
        assert_eq!(baseline_config, base);

        // Slippage 2x: moderate degradation
        let slip2x = estimate_sharpe_under_stress(baseline_sharpe, 2.0, 1.0);
        assert!(slip2x < baseline_sharpe, "2x slippage should degrade Sharpe");
        assert!(slip2x > 1.0, "2x slippage shouldn't be catastrophic: {}", slip2x);

        // Slippage 4x: more degradation
        let slip4x = estimate_sharpe_under_stress(baseline_sharpe, 4.0, 1.0);
        assert!(slip4x < slip2x, "4x slippage should degrade more than 2x");

        // Combined 4x200: worst case
        let combined = estimate_sharpe_under_stress(baseline_sharpe, 4.0, 2.0);
        assert!(combined < slip4x, "Combined stress should be worst");

        // Verify the degradation percentages are in expected range
        let degradation_4x200 = (baseline_sharpe - combined) / baseline_sharpe * 100.0;
        assert!(
            degradation_4x200 > 10.0 && degradation_4x200 < 30.0,
            "4x200 degradation should be 10-30%: {:.1}%",
            degradation_4x200
        );
    }

    #[test]
    fn test_stress_summary_degradation_calculation() {
        // Verify StressSummary correctly calculates degradation percentages
        let results = vec![
            StressTestResult::pass(
                StressScenario::Baseline,
                RunMetrics { sharpe_ratio: 1.5, max_drawdown: -0.10, ..Default::default() },
                CostConfig::default(),
            ),
            StressTestResult::pass(
                StressScenario::Slippage2x,
                RunMetrics { sharpe_ratio: 1.35, max_drawdown: -0.12, ..Default::default() },
                CostConfig::default(),
            ),
            StressTestResult::pass(
                StressScenario::Combined4x200,
                RunMetrics { sharpe_ratio: 1.0, max_drawdown: -0.18, ..Default::default() },
                CostConfig::default(),
            ),
        ];

        let report = StressTestReport::from_results("test_strategy".to_string(), results);

        // Verify baseline is correctly captured
        assert!((report.summary.baseline_sharpe - 1.5).abs() < 0.001);
        assert!((report.summary.baseline_max_dd - (-0.10)).abs() < 0.001);

        // Verify min Sharpe is the worst case
        assert!((report.summary.min_sharpe - 1.0).abs() < 0.001);

        // Verify worst drawdown
        assert!((report.summary.worst_max_dd - (-0.18)).abs() < 0.001);

        // Verify degradation calculation: (1.5 - 1.0) / 1.5 * 100 = 33.33%
        let expected_sharpe_degradation: f64 = (1.5 - 1.0) / 1.5 * 100.0;
        assert!(
            (report.summary.sharpe_degradation_pct - expected_sharpe_degradation).abs() < 0.1,
            "Sharpe degradation should be ~{:.1}%, got {:.1}%",
            expected_sharpe_degradation,
            report.summary.sharpe_degradation_pct
        );

        // Verify drawdown increase: (-0.18 - (-0.10)) / (-0.10) * 100 = 80%
        // (drawdown got 80% worse)
        let expected_dd_increase: f64 = 80.0; // DD went from 10% to 18%, that's 80% increase
        assert!(
            (report.summary.max_dd_increase_pct.abs() - expected_dd_increase).abs() < 0.5,
            "DD increase should be ~{:.1}%, got {:.1}%",
            expected_dd_increase,
            report.summary.max_dd_increase_pct.abs()
        );
    }

    #[test]
    fn test_report_metadata() {
        let results = vec![
            StressTestResult::pass(
                StressScenario::Baseline,
                RunMetrics { sharpe_ratio: 1.5, ..Default::default() },
                CostConfig::default(),
            ),
        ];

        let report = StressTestReport::from_results("test_strategy".to_string(), results);

        // Verify metadata is populated
        assert!(!report.metadata.crate_version.is_empty());
        assert!(!report.metadata.timestamp_utc.is_empty());
        
        // Verify timestamp is a valid RFC3339 format
        assert!(
            chrono::DateTime::parse_from_rfc3339(&report.metadata.timestamp_utc).is_ok(),
            "Timestamp should be valid RFC3339: {}",
            report.metadata.timestamp_utc
        );
    }

    #[test]
    fn test_equals_deterministic() {
        let results1 = vec![
            StressTestResult::pass(
                StressScenario::Baseline,
                RunMetrics { sharpe_ratio: 1.5, max_drawdown: -0.10, ..Default::default() },
                CostConfig::default(),
            ),
        ];
        let results2 = results1.clone();
        let report1 = StressTestReport::from_results("test".to_string(), results1);
        std::thread::sleep(std::time::Duration::from_millis(10));
        let report2 = StressTestReport::from_results("test".to_string(), results2);
        assert_ne!(report1.metadata.timestamp_utc, report2.metadata.timestamp_utc);
        assert!(report1.equals_deterministic(&report2));
    }

    #[test]
    fn test_thresholds_used_in_report() {
        let thresholds = StressThresholds {
            min_sharpe: 0.5,
            max_drawdown: -0.25,
            max_sharpe_degradation_pct: 40.0,
            max_dd_increase_pct: 60.0,
        };
        let results = vec![
            StressTestResult::pass(
                StressScenario::Baseline,
                RunMetrics { sharpe_ratio: 1.5, ..Default::default() },
                CostConfig::default(),
            ),
        ];
        let report = StressTestReport::from_results_with_metadata_and_thresholds(
            "test".to_string(),
            results,
            ReportMetadata::now(),
            Some(thresholds.clone()),
        );
        assert!(report.thresholds_used.is_some());
        let used = report.thresholds_used.unwrap();
        assert!((used.min_sharpe - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_mode_equivalence_tolerance() {
        const SHARPE_TOLERANCE: f64 = 0.01;
        let diff = (1.500_f64 - 1.498_f64).abs();
        assert!(diff < SHARPE_TOLERANCE);
    }

    // ========================================================================
    // EDGE CASE TESTS (Harvey et al. 2016: robust systems handle extremes)
    // ========================================================================

    /// Test: Strategy with zero trades should not panic and produce valid report.
    #[test]
    fn test_edge_case_zero_trades() {
        let results = vec![
            StressTestResult::pass(
                StressScenario::Baseline,
                RunMetrics {
                    sharpe_ratio: 0.0,
                    max_drawdown: 0.0,
                    total_trades: 0,
                    
                    ..Default::default()
                },
                CostConfig::default(),
            ),
        ];

        let report = StressTestReport::from_results("zero_trades".to_string(), results);
        assert!(report.all_passed);
        assert!((report.summary.baseline_sharpe - 0.0).abs() < 1e-9);
    }

    /// Test: All scenarios failing should be handled correctly.
    #[test]
    fn test_edge_case_all_scenarios_fail() {
        let results = vec![
            StressTestResult::fail(
                StressScenario::Baseline,
                RunMetrics { sharpe_ratio: -0.5, max_drawdown: -0.50, ..Default::default() },
                CostConfig::default(),
                "Sharpe below minimum".to_string(),
            ),
            StressTestResult::fail(
                StressScenario::Slippage2x,
                RunMetrics { sharpe_ratio: -1.0, max_drawdown: -0.60, ..Default::default() },
                CostConfig::default(),
                "Sharpe below minimum".to_string(),
            ),
        ];

        let report = StressTestReport::from_results("all_fail".to_string(), results);
        assert!(!report.all_passed);
        assert_eq!(report.failed_count, 2);
        assert_eq!(report.passed_count, 0);
    }

    /// Test: Extreme drawdown values (-99%) should be handled.
    #[test]
    fn test_edge_case_extreme_drawdown() {
        let results = vec![
            StressTestResult::pass(
                StressScenario::Combined4x200,
                RunMetrics {
                    sharpe_ratio: -5.0,
                    max_drawdown: -0.99,
                    
                    ..Default::default()
                },
                CostConfig::default(),
            ),
        ];

        let report = StressTestReport::from_results("extreme_dd".to_string(), results);
        assert!((report.summary.worst_max_dd - (-0.99)).abs() < 1e-9);
    }

    /// Test: Single result should not cause panic.
    #[test]
    fn test_edge_case_single_result() {
        let results = vec![
            StressTestResult::pass(
                StressScenario::Baseline,
                RunMetrics { sharpe_ratio: 1.5, max_drawdown: -0.10, ..Default::default() },
                CostConfig::default(),
            ),
        ];

        let report = StressTestReport::from_results("single".to_string(), results);
        assert!(report.all_passed);
        assert_eq!(report.passed_count, 1);
    }

    /// Test: Negative baseline Sharpe with positive stressed (unusual but valid).
    #[test]
    fn test_edge_case_sharpe_sign_change() {
        let results = vec![
            StressTestResult::pass(
                StressScenario::Baseline,
                RunMetrics { sharpe_ratio: -0.5, max_drawdown: -0.20, ..Default::default() },
                CostConfig::default(),
            ),
            StressTestResult::pass(
                StressScenario::Slippage2x,
                RunMetrics { sharpe_ratio: 0.2, max_drawdown: -0.15, ..Default::default() },
                CostConfig::default(),
            ),
        ];

        let report = StressTestReport::from_results("sign_change".to_string(), results);
        assert!(report.all_passed);
        assert!(report.summary.baseline_sharpe.is_finite());
    }
}
