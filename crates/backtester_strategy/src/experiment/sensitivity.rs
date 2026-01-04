//! Parameter Sensitivity Analysis Module.
//!
//! Tests strategy robustness by perturbing parameters and checking if
//! performance degrades significantly with small changes.
//!
//! Reference: Anti-Overfitting Checklist - "Estabilidade dos Parâmetros"
//! A robust strategy should NOT degrade drastically with small parameter changes.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for sensitivity analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityConfig {
    /// Perturbation percentage (e.g., 0.05 for ±5%)
    #[serde(default = "default_perturbation_pct")]
    pub perturbation_pct: f64,
    
    /// Maximum allowed degradation in Sharpe (e.g., 0.20 for 20%)
    #[serde(default = "default_max_sharpe_degradation")]
    pub max_sharpe_degradation: f64,
    
    /// Maximum allowed degradation in CAGR (e.g., 0.25 for 25%)
    #[serde(default = "default_max_cagr_degradation")]
    pub max_cagr_degradation: f64,
    
    /// Maximum allowed increase in max drawdown (e.g., 0.30 for 30%)
    #[serde(default = "default_max_dd_increase")]
    pub max_drawdown_increase: f64,
    
    /// Minimum percentage of perturbations that must pass (e.g., 0.75)
    #[serde(default = "default_min_pass_rate")]
    pub min_pass_rate: f64,
    
    /// Parameters to test (if empty, test all numeric params)
    #[serde(default)]
    pub params_to_test: Vec<String>,
}

fn default_perturbation_pct() -> f64 { 0.05 }
fn default_max_sharpe_degradation() -> f64 { 0.20 }
fn default_max_cagr_degradation() -> f64 { 0.25 }
fn default_max_dd_increase() -> f64 { 0.30 }
fn default_min_pass_rate() -> f64 { 0.75 }

impl Default for SensitivityConfig {
    fn default() -> Self {
        Self {
            perturbation_pct: default_perturbation_pct(),
            max_sharpe_degradation: default_max_sharpe_degradation(),
            max_cagr_degradation: default_max_cagr_degradation(),
            max_drawdown_increase: default_max_dd_increase(),
            min_pass_rate: default_min_pass_rate(),
            params_to_test: Vec::new(),
        }
    }
}

/// A parameter perturbation to test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Perturbation {
    /// Parameter name (e.g., "lookback_days")
    pub param_name: String,
    /// Original value
    pub original: f64,
    /// Perturbed value
    pub perturbed: f64,
    /// Direction: "up" or "down"
    pub direction: String,
}

impl Perturbation {
    pub fn up(name: impl Into<String>, original: f64, pct: f64) -> Self {
        Self {
            param_name: name.into(),
            original,
            perturbed: original * (1.0 + pct),
            direction: "up".to_string(),
        }
    }
    
    pub fn down(name: impl Into<String>, original: f64, pct: f64) -> Self {
        Self {
            param_name: name.into(),
            original,
            perturbed: original * (1.0 - pct),
            direction: "down".to_string(),
        }
    }
}

/// Metrics snapshot for comparison.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub sharpe_ratio: f64,
    pub cagr: f64,
    pub max_drawdown: f64,
    pub profit_factor: f64,
    pub win_rate: f64,
}

/// Result of testing a single perturbation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationResult {
    pub perturbation: Perturbation,
    pub baseline_metrics: MetricsSnapshot,
    pub perturbed_metrics: MetricsSnapshot,
    pub sharpe_change_pct: f64,
    pub cagr_change_pct: f64,
    pub drawdown_change_pct: f64,
    pub passed: bool,
    pub failure_reasons: Vec<String>,
}

impl PerturbationResult {
    pub fn new(
        perturbation: Perturbation,
        baseline: MetricsSnapshot,
        perturbed: MetricsSnapshot,
        config: &SensitivityConfig,
    ) -> Self {
        let sharpe_change = if baseline.sharpe_ratio.abs() > 1e-6 {
            (perturbed.sharpe_ratio - baseline.sharpe_ratio) / baseline.sharpe_ratio
        } else {
            0.0
        };
        
        let cagr_change = if baseline.cagr.abs() > 1e-6 {
            (perturbed.cagr - baseline.cagr) / baseline.cagr
        } else {
            0.0
        };
        
        let dd_change = if baseline.max_drawdown.abs() > 1e-6 {
            (perturbed.max_drawdown.abs() - baseline.max_drawdown.abs()) / baseline.max_drawdown.abs()
        } else {
            0.0
        };
        
        let mut failures = Vec::new();
        
        // Check degradation (negative change for Sharpe/CAGR is bad)
        if sharpe_change < -config.max_sharpe_degradation {
            failures.push(format!(
                "Sharpe degraded {:.1}% (limit: {:.1}%)",
                sharpe_change * 100.0,
                -config.max_sharpe_degradation * 100.0
            ));
        }
        
        if cagr_change < -config.max_cagr_degradation {
            failures.push(format!(
                "CAGR degraded {:.1}% (limit: {:.1}%)",
                cagr_change * 100.0,
                -config.max_cagr_degradation * 100.0
            ));
        }
        
        // Check drawdown increase (positive change is bad)
        if dd_change > config.max_drawdown_increase {
            failures.push(format!(
                "Max DD increased {:.1}% (limit: {:.1}%)",
                dd_change * 100.0,
                config.max_drawdown_increase * 100.0
            ));
        }
        
        Self {
            perturbation,
            baseline_metrics: baseline,
            perturbed_metrics: perturbed,
            sharpe_change_pct: sharpe_change,
            cagr_change_pct: cagr_change,
            drawdown_change_pct: dd_change,
            passed: failures.is_empty(),
            failure_reasons: failures,
        }
    }
}

/// Report from sensitivity analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityReport {
    /// Configuration used
    pub config: SensitivityConfig,
    /// Baseline metrics (original strategy)
    pub baseline_metrics: MetricsSnapshot,
    /// Results for each perturbation
    pub results: Vec<PerturbationResult>,
    /// Number of tests passed
    pub passed_count: usize,
    /// Number of tests failed
    pub failed_count: usize,
    /// Pass rate (0.0 to 1.0)
    pub pass_rate: f64,
    /// Whether overall sensitivity test passed
    pub is_stable: bool,
    /// Summary of failures
    pub failure_summary: Vec<String>,
}

impl SensitivityReport {
    pub fn from_results(
        config: SensitivityConfig,
        baseline: MetricsSnapshot,
        results: Vec<PerturbationResult>,
    ) -> Self {
        let passed_count = results.iter().filter(|r| r.passed).count();
        let failed_count = results.len() - passed_count;
        let pass_rate = if results.is_empty() {
            1.0
        } else {
            passed_count as f64 / results.len() as f64
        };
        
        let is_stable = pass_rate >= config.min_pass_rate;
        
        let failure_summary: Vec<String> = results.iter()
            .filter(|r| !r.passed)
            .map(|r| {
                format!(
                    "{} {} {}: {}",
                    r.perturbation.param_name,
                    r.perturbation.direction,
                    format!("{:.0}→{:.0}", r.perturbation.original, r.perturbation.perturbed),
                    r.failure_reasons.join(", ")
                )
            })
            .collect();
        
        Self {
            config,
            baseline_metrics: baseline,
            results,
            passed_count,
            failed_count,
            pass_rate,
            is_stable,
            failure_summary,
        }
    }
    
    /// Generate summary string.
    pub fn to_summary_string(&self) -> String {
        let status = if self.is_stable { "STABLE" } else { "UNSTABLE" };
        format!(
            "{}: {}/{} passed ({:.1}%)",
            status,
            self.passed_count,
            self.passed_count + self.failed_count,
            self.pass_rate * 100.0
        )
    }
}

/// Generator for parameter perturbations.
#[derive(Debug, Clone)]
pub struct PerturbationGenerator {
    config: SensitivityConfig,
}

impl PerturbationGenerator {
    pub fn new(config: SensitivityConfig) -> Self {
        Self { config }
    }
    
    /// Generate perturbations for a set of parameters.
    pub fn generate(&self, params: &HashMap<String, f64>) -> Vec<Perturbation> {
        let mut perturbations = Vec::new();
        
        for (name, &value) in params {
            // Skip if params_to_test is specified and this param isn't in it
            if !self.config.params_to_test.is_empty() 
                && !self.config.params_to_test.contains(name) {
                continue;
            }
            
            // Skip zero/near-zero values (can't meaningfully perturb)
            if value.abs() < 1e-10 {
                continue;
            }
            
            // Generate up and down perturbations
            perturbations.push(Perturbation::up(name, value, self.config.perturbation_pct));
            perturbations.push(Perturbation::down(name, value, self.config.perturbation_pct));
        }
        
        perturbations
    }
    
    /// Generate perturbations for integer parameters.
    pub fn generate_integer(&self, params: &HashMap<String, i64>) -> Vec<Perturbation> {
        let mut perturbations = Vec::new();
        
        for (name, &value) in params {
            if !self.config.params_to_test.is_empty() 
                && !self.config.params_to_test.contains(name) {
                continue;
            }
            
            if value == 0 {
                continue;
            }
            
            let float_val = value as f64;
            let delta = (float_val * self.config.perturbation_pct).max(1.0);
            
            perturbations.push(Perturbation {
                param_name: name.clone(),
                original: float_val,
                perturbed: float_val + delta,
                direction: "up".to_string(),
            });
            
            if float_val - delta > 0.0 {
                perturbations.push(Perturbation {
                    param_name: name.clone(),
                    original: float_val,
                    perturbed: float_val - delta,
                    direction: "down".to_string(),
                });
            }
        }
        
        perturbations
    }
}

/// Sensitivity Analyzer for testing parameter stability.
#[derive(Debug, Clone)]
pub struct SensitivityAnalyzer {
    pub config: SensitivityConfig,
}

impl SensitivityAnalyzer {
    pub fn new(config: SensitivityConfig) -> Self {
        Self { config }
    }
    
    /// Check if perturbation results indicate stability.
    pub fn check_stability(&self, report: &SensitivityReport) -> (bool, Vec<String>) {
        let mut reasons = Vec::new();
        
        if report.pass_rate < self.config.min_pass_rate {
            reasons.push(format!(
                "Pass rate {:.1}% < required {:.1}%",
                report.pass_rate * 100.0,
                self.config.min_pass_rate * 100.0
            ));
        }
        
        // Check for any catastrophic failures (>50% degradation)
        for result in &report.results {
            if result.sharpe_change_pct < -0.50 {
                reasons.push(format!(
                    "Catastrophic Sharpe drop ({:.1}%) for {} {}",
                    result.sharpe_change_pct * 100.0,
                    result.perturbation.param_name,
                    result.perturbation.direction
                ));
            }
        }
        
        (reasons.is_empty(), reasons)
    }
}

impl Default for SensitivityAnalyzer {
    fn default() -> Self {
        Self::new(SensitivityConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perturbation_generation() {
        let config = SensitivityConfig {
            perturbation_pct: 0.05,
            ..Default::default()
        };
        let gen = PerturbationGenerator::new(config);
        
        let mut params = HashMap::new();
        params.insert("lookback".to_string(), 20.0);
        params.insert("threshold".to_string(), 0.5);
        
        let perturbations = gen.generate(&params);
        
        // 2 params * 2 directions = 4 perturbations
        assert_eq!(perturbations.len(), 4);
        
        // Check lookback up perturbation
        let lookback_up = perturbations.iter()
            .find(|p| p.param_name == "lookback" && p.direction == "up")
            .unwrap();
        assert!((lookback_up.perturbed - 21.0).abs() < 0.01);
        
        // Check lookback down perturbation
        let lookback_down = perturbations.iter()
            .find(|p| p.param_name == "lookback" && p.direction == "down")
            .unwrap();
        assert!((lookback_down.perturbed - 19.0).abs() < 0.01);
    }

    #[test]
    fn test_perturbation_result_pass() {
        let config = SensitivityConfig::default();
        
        let baseline = MetricsSnapshot {
            sharpe_ratio: 1.5,
            cagr: 0.15,
            max_drawdown: -0.10,
            ..Default::default()
        };
        
        // Small degradation - should pass
        let perturbed = MetricsSnapshot {
            sharpe_ratio: 1.4,
            cagr: 0.14,
            max_drawdown: -0.11,
            ..Default::default()
        };
        
        let perturbation = Perturbation::up("lookback", 20.0, 0.05);
        let result = PerturbationResult::new(perturbation, baseline, perturbed, &config);
        
        assert!(result.passed, "Should pass: {:?}", result.failure_reasons);
    }

    #[test]
    fn test_perturbation_result_fail() {
        let config = SensitivityConfig {
            max_sharpe_degradation: 0.10,
            ..Default::default()
        };
        
        let baseline = MetricsSnapshot {
            sharpe_ratio: 1.5,
            cagr: 0.15,
            max_drawdown: -0.10,
            ..Default::default()
        };
        
        // Large degradation - should fail
        let perturbed = MetricsSnapshot {
            sharpe_ratio: 1.0, // 33% drop
            cagr: 0.10,
            max_drawdown: -0.15,
            ..Default::default()
        };
        
        let perturbation = Perturbation::up("lookback", 20.0, 0.05);
        let result = PerturbationResult::new(perturbation, baseline, perturbed, &config);
        
        assert!(!result.passed);
        assert!(!result.failure_reasons.is_empty());
    }

    #[test]
    fn test_sensitivity_report() {
        let config = SensitivityConfig {
            min_pass_rate: 0.75,
            ..Default::default()
        };
        
        let baseline = MetricsSnapshot {
            sharpe_ratio: 1.5,
            cagr: 0.15,
            max_drawdown: -0.10,
            ..Default::default()
        };
        
        // 3 pass, 1 fail = 75% pass rate
        let results = vec![
            PerturbationResult {
                perturbation: Perturbation::up("a", 10.0, 0.05),
                baseline_metrics: baseline.clone(),
                perturbed_metrics: baseline.clone(),
                sharpe_change_pct: -0.05,
                cagr_change_pct: -0.05,
                drawdown_change_pct: 0.05,
                passed: true,
                failure_reasons: vec![],
            },
            PerturbationResult {
                perturbation: Perturbation::down("a", 10.0, 0.05),
                baseline_metrics: baseline.clone(),
                perturbed_metrics: baseline.clone(),
                sharpe_change_pct: -0.05,
                cagr_change_pct: -0.05,
                drawdown_change_pct: 0.05,
                passed: true,
                failure_reasons: vec![],
            },
            PerturbationResult {
                perturbation: Perturbation::up("b", 20.0, 0.05),
                baseline_metrics: baseline.clone(),
                perturbed_metrics: baseline.clone(),
                sharpe_change_pct: -0.05,
                cagr_change_pct: -0.05,
                drawdown_change_pct: 0.05,
                passed: true,
                failure_reasons: vec![],
            },
            PerturbationResult {
                perturbation: Perturbation::down("b", 20.0, 0.05),
                baseline_metrics: baseline.clone(),
                perturbed_metrics: baseline.clone(),
                sharpe_change_pct: -0.30,
                cagr_change_pct: -0.30,
                drawdown_change_pct: 0.50,
                passed: false,
                failure_reasons: vec!["Too much degradation".to_string()],
            },
        ];
        
        let report = SensitivityReport::from_results(config, baseline, results);
        
        assert_eq!(report.passed_count, 3);
        assert_eq!(report.failed_count, 1);
        assert!((report.pass_rate - 0.75).abs() < 0.01);
        assert!(report.is_stable); // 75% == min_pass_rate
    }
}

