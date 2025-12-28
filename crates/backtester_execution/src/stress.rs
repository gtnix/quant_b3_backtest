//! Execution Stress Testing Suite
//!
//! Defines mandatory stress scenarios for top candidates to verify
//! robustness under adverse execution conditions.
//!
//! All top-10 HoF candidates must pass these stress tests.

use serde::{Deserialize, Serialize};

use crate::config::ExecutionModelConfig;

// =============================================================================
// STRESS SUITE
// =============================================================================

/// Collection of stress scenarios for execution validation.
#[derive(Debug, Clone)]
pub struct StressSuite {
    /// List of stress scenarios.
    pub scenarios: Vec<StressScenario>,
}

impl StressSuite {
    /// Create a new empty stress suite.
    #[must_use]
    pub fn new() -> Self {
        Self {
            scenarios: Vec::new(),
        }
    }

    /// Create the default institutional stress suite with S1-S5.
    #[must_use]
    pub fn default_institutional() -> Self {
        Self {
            scenarios: vec![
                StressScenario::costs_2x(),
                StressScenario::delay_plus1(),
                StressScenario::spread_widen_vol(),
                StressScenario::capacity_constraint(),
                StressScenario::combined_adverse(),
            ],
        }
    }

    /// Add a stress scenario.
    pub fn add(&mut self, scenario: StressScenario) {
        self.scenarios.push(scenario);
    }

    /// Get a scenario by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&StressScenario> {
        self.scenarios.iter().find(|s| s.id == id)
    }

    /// Get number of scenarios.
    #[must_use]
    pub fn len(&self) -> usize {
        self.scenarios.len()
    }

    /// Check if suite is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.scenarios.is_empty()
    }

    /// Apply a scenario to transform an execution config.
    #[must_use]
    pub fn apply_scenario(&self, id: &str, config: &ExecutionModelConfig) -> Option<ExecutionModelConfig> {
        self.get(id).map(|s| s.apply(config))
    }
}

impl Default for StressSuite {
    fn default() -> Self {
        Self::default_institutional()
    }
}

// =============================================================================
// STRESS SCENARIO
// =============================================================================

/// A single stress scenario with transformation and acceptance criteria.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressScenario {
    /// Unique identifier (e.g., "S1", "S2").
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Description of what this stress tests.
    pub description: String,
    /// Type of transformation to apply.
    pub transform_type: StressTransformType,
    /// Acceptance criteria for this stress.
    pub acceptance: AcceptanceCriteria,
}

/// Type of stress transformation to apply.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum StressTransformType {
    /// Scale costs (slippage and fees) by a factor.
    ScaleCosts { factor: f64 },
    /// Add extra delay bars.
    AddDelay { extra_bars: u8 },
    /// Widen spread in high volatility regimes.
    SpreadWidenHighVol { multiplier: f64 },
    /// Reduce max participation (capacity constraint).
    ReduceParticipation { new_max: f64 },
    /// Combined: scale costs AND add delay.
    Combined { cost_factor: f64, extra_delay: u8 },
}

impl StressScenario {
    /// Create S1: costs x2 scenario.
    #[must_use]
    pub fn costs_2x() -> Self {
        Self {
            id: "S1".into(),
            name: "costs_2x".into(),
            description: "Double all execution costs (slippage and fees)".into(),
            transform_type: StressTransformType::ScaleCosts { factor: 2.0 },
            acceptance: AcceptanceCriteria {
                min_oos_sharpe: 0.3,
                min_execution_rate: None,
                max_oos_mdd: None,
            },
        }
    }

    /// Create S2: delay +1 bar scenario.
    #[must_use]
    pub fn delay_plus1() -> Self {
        Self {
            id: "S2".into(),
            name: "delay_plus1".into(),
            description: "Add one extra bar of execution delay".into(),
            transform_type: StressTransformType::AddDelay { extra_bars: 1 },
            acceptance: AcceptanceCriteria {
                min_oos_sharpe: 0.5,
                min_execution_rate: None,
                max_oos_mdd: None,
            },
        }
    }

    /// Create S3: spread widen in high vol regime.
    #[must_use]
    pub fn spread_widen_vol() -> Self {
        Self {
            id: "S3".into(),
            name: "spread_widen_vol".into(),
            description: "Triple slippage in high volatility regimes".into(),
            transform_type: StressTransformType::SpreadWidenHighVol { multiplier: 3.0 },
            acceptance: AcceptanceCriteria {
                min_oos_sharpe: 0.2,
                min_execution_rate: None,
                max_oos_mdd: None,
            },
        }
    }

    /// Create S4: capacity constraint scenario.
    #[must_use]
    pub fn capacity_constraint() -> Self {
        Self {
            id: "S4".into(),
            name: "capacity_constraint".into(),
            description: "Reduce max participation to 1%".into(),
            transform_type: StressTransformType::ReduceParticipation { new_max: 0.01 },
            acceptance: AcceptanceCriteria {
                min_oos_sharpe: 0.0, // Any positive
                min_execution_rate: Some(0.80), // Must execute >= 80% of orders
                max_oos_mdd: None,
            },
        }
    }

    /// Create S5: combined adverse scenario.
    #[must_use]
    pub fn combined_adverse() -> Self {
        Self {
            id: "S5".into(),
            name: "combined_adverse".into(),
            description: "Double costs AND add one bar delay".into(),
            transform_type: StressTransformType::Combined {
                cost_factor: 2.0,
                extra_delay: 1,
            },
            acceptance: AcceptanceCriteria {
                min_oos_sharpe: 0.0, // Must not be negative
                min_execution_rate: None,
                max_oos_mdd: Some(0.30), // Max 30% drawdown
            },
        }
    }

    /// Apply this stress scenario to an execution config.
    #[must_use]
    pub fn apply(&self, config: &ExecutionModelConfig) -> ExecutionModelConfig {
        match &self.transform_type {
            StressTransformType::ScaleCosts { factor } => config.scale_costs(*factor),
            StressTransformType::AddDelay { extra_bars } => config.add_delay(*extra_bars),
            StressTransformType::SpreadWidenHighVol { multiplier } => {
                // For now, just scale slippage (full regime-aware impl later)
                config.scale_costs(*multiplier)
            }
            StressTransformType::ReduceParticipation { new_max } => {
                let mut new_config = config.clone();
                new_config.fill_policy.max_participation = *new_max;
                new_config
            }
            StressTransformType::Combined {
                cost_factor,
                extra_delay,
            } => config.scale_costs(*cost_factor).add_delay(*extra_delay),
        }
    }
}

// =============================================================================
// ACCEPTANCE CRITERIA
// =============================================================================

/// Criteria for passing a stress test.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AcceptanceCriteria {
    /// Minimum OOS Sharpe ratio required.
    pub min_oos_sharpe: f64,
    /// Minimum execution rate (orders filled / orders attempted).
    pub min_execution_rate: Option<f64>,
    /// Maximum OOS drawdown allowed.
    pub max_oos_mdd: Option<f64>,
}

impl AcceptanceCriteria {
    /// Check if the given metrics pass this criteria.
    #[must_use]
    pub fn check(&self, oos_sharpe: f64, execution_rate: Option<f64>, oos_mdd: Option<f64>) -> bool {
        if oos_sharpe < self.min_oos_sharpe {
            return false;
        }

        if let Some(min_exec) = self.min_execution_rate {
            if execution_rate.map_or(true, |r| r < min_exec) {
                return false;
            }
        }

        if let Some(max_mdd) = self.max_oos_mdd {
            if oos_mdd.map_or(false, |mdd| mdd > max_mdd) {
                return false;
            }
        }

        true
    }
}

// =============================================================================
// STRESS RESULT
// =============================================================================

/// Result of running a single stress scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressResult {
    /// Scenario ID.
    pub scenario_id: String,
    /// Scenario name.
    pub scenario_name: String,
    /// Original Sharpe ratio (before stress).
    pub sharpe_original: f64,
    /// Stressed Sharpe ratio.
    pub sharpe_stressed: f64,
    /// Degradation percentage.
    pub degradation_pct: f64,
    /// Execution rate (if applicable).
    pub execution_rate: Option<f64>,
    /// OOS drawdown (if applicable).
    pub oos_mdd: Option<f64>,
    /// Whether the candidate passed this stress.
    pub passed: bool,
    /// Reason for failure (if any).
    pub failure_reason: Option<String>,
}

impl StressResult {
    /// Create a new stress result.
    #[must_use]
    pub fn new(
        scenario: &StressScenario,
        sharpe_original: f64,
        sharpe_stressed: f64,
        execution_rate: Option<f64>,
        oos_mdd: Option<f64>,
    ) -> Self {
        let degradation_pct = if sharpe_original != 0.0 {
            ((sharpe_original - sharpe_stressed) / sharpe_original.abs()) * 100.0
        } else {
            0.0
        };

        let passed = scenario
            .acceptance
            .check(sharpe_stressed, execution_rate, oos_mdd);

        let failure_reason = if !passed {
            Some(format!(
                "Failed criteria: min_sharpe={:.2}, actual={:.2}",
                scenario.acceptance.min_oos_sharpe, sharpe_stressed
            ))
        } else {
            None
        };

        Self {
            scenario_id: scenario.id.clone(),
            scenario_name: scenario.name.clone(),
            sharpe_original,
            sharpe_stressed,
            degradation_pct,
            execution_rate,
            oos_mdd,
            passed,
            failure_reason,
        }
    }
}

/// Aggregate results from running all stress scenarios.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StressSuiteResult {
    /// Individual stress results.
    pub results: Vec<StressResult>,
    /// Whether all stresses passed.
    pub all_passed: bool,
    /// Number of scenarios passed.
    pub passed_count: usize,
    /// Number of scenarios total.
    pub total_count: usize,
}

impl StressSuiteResult {
    /// Create from a list of stress results.
    #[must_use]
    pub fn from_results(results: Vec<StressResult>) -> Self {
        let passed_count = results.iter().filter(|r| r.passed).count();
        let total_count = results.len();
        let all_passed = passed_count == total_count;

        Self {
            results,
            all_passed,
            passed_count,
            total_count,
        }
    }

    /// Get results for failed scenarios.
    #[must_use]
    pub fn failures(&self) -> Vec<&StressResult> {
        self.results.iter().filter(|r| !r.passed).collect()
    }

    /// Summary string for display.
    #[must_use]
    pub fn summary(&self) -> String {
        if self.all_passed {
            format!("All {}/{} stress tests passed", self.passed_count, self.total_count)
        } else {
            let failed: Vec<_> = self.failures().iter().map(|r| r.scenario_id.as_str()).collect();
            format!(
                "{}/{} stress tests passed, failed: {}",
                self.passed_count,
                self.total_count,
                failed.join(", ")
            )
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_suite_has_5_scenarios() {
        let suite = StressSuite::default_institutional();
        assert_eq!(suite.len(), 5);
        assert!(suite.get("S1").is_some());
        assert!(suite.get("S2").is_some());
        assert!(suite.get("S3").is_some());
        assert!(suite.get("S4").is_some());
        assert!(suite.get("S5").is_some());
    }

    #[test]
    fn test_costs_2x_transform() {
        let config = ExecutionModelConfig::mvp();
        let scenario = StressScenario::costs_2x();
        let stressed = scenario.apply(&config);

        // MVP has 10 bps slippage, doubled should be 20
        assert!(stressed.slippage.base_bps() > config.slippage.base_bps());
    }

    #[test]
    fn test_delay_plus1_transform() {
        let config = ExecutionModelConfig::mvp();
        let scenario = StressScenario::delay_plus1();
        let stressed = scenario.apply(&config);

        assert_eq!(stressed.delay_bars, config.delay_bars + 1);
    }

    #[test]
    fn test_acceptance_criteria_check() {
        let criteria = AcceptanceCriteria {
            min_oos_sharpe: 0.3,
            min_execution_rate: Some(0.8),
            max_oos_mdd: Some(0.25),
        };

        // Should pass
        assert!(criteria.check(0.5, Some(0.9), Some(0.20)));

        // Should fail - low sharpe
        assert!(!criteria.check(0.2, Some(0.9), Some(0.20)));

        // Should fail - low execution rate
        assert!(!criteria.check(0.5, Some(0.7), Some(0.20)));

        // Should fail - high MDD
        assert!(!criteria.check(0.5, Some(0.9), Some(0.30)));
    }

    #[test]
    fn test_stress_result_creation() {
        let scenario = StressScenario::costs_2x();
        let result = StressResult::new(&scenario, 1.0, 0.5, None, None);

        assert!((result.degradation_pct - 50.0).abs() < 0.01);
        assert!(result.passed); // 0.5 > 0.3 threshold
    }

    #[test]
    fn test_stress_result_failure() {
        let scenario = StressScenario::costs_2x();
        let result = StressResult::new(&scenario, 1.0, 0.2, None, None);

        assert!(!result.passed); // 0.2 < 0.3 threshold
        assert!(result.failure_reason.is_some());
    }

    #[test]
    fn test_stress_suite_result_aggregation() {
        let results = vec![
            StressResult::new(&StressScenario::costs_2x(), 1.0, 0.5, None, None),
            StressResult::new(&StressScenario::delay_plus1(), 1.0, 0.6, None, None),
        ];

        let suite_result = StressSuiteResult::from_results(results);

        assert!(suite_result.all_passed);
        assert_eq!(suite_result.passed_count, 2);
        assert_eq!(suite_result.total_count, 2);
    }

    #[test]
    fn test_stress_suite_result_with_failures() {
        let results = vec![
            StressResult::new(&StressScenario::costs_2x(), 1.0, 0.5, None, None), // 0.5 >= 0.3 threshold - PASS
            StressResult::new(&StressScenario::delay_plus1(), 1.0, 0.3, None, None), // 0.3 < 0.5 threshold - FAIL
            StressResult::new(&StressScenario::combined_adverse(), 1.0, -0.1, None, None), // -0.1 < 0.0 threshold - FAIL
        ];

        let suite_result = StressSuiteResult::from_results(results);

        assert!(!suite_result.all_passed);
        assert_eq!(suite_result.passed_count, 1); // Only costs_2x passes
        assert_eq!(suite_result.failures().len(), 2); // delay_plus1 and combined_adverse fail
    }
}

