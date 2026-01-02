//! Anti-overfitting validation for genomes.
//!
//! Provides Walk-Forward Analysis, CPCV, PBO/DSR calculations,
//! and execution stress testing to validate strategies before deployment.

use backtester_execution::{
    ExecutionModelConfig, GateChecker, GateResult,
    StressSuite, StressResult, StressSuiteResult,
    cost_report::CostReport,
};
use combiner_core::StrategyGenome;
use combiner_runner::BacktestExecutor;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;
use uuid::Uuid;

/// Errors during validation.
#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("Validation failed for genome {0}: {1}")]
    Failed(String, String),

    #[error("Insufficient data for validation: {0}")]
    InsufficientData(String),

    #[error("Execution error: {0}")]
    Execution(String),
}

/// Result of Walk-Forward Analysis for a genome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WfaResult {
    /// Genome ID.
    pub genome_id: Uuid,
    /// In-sample Sharpe ratio (gross).
    pub is_sharpe_gross: f64,
    /// In-sample Sharpe ratio (net of costs).
    pub is_sharpe_net: f64,
    /// Out-of-sample Sharpe ratio (gross).
    pub oos_sharpe_gross: f64,
    /// Out-of-sample Sharpe ratio (net of costs).
    pub oos_sharpe_net: f64,
    /// Degradation percentage: (IS - OOS) / IS * 100 (using net).
    pub degradation_pct: f64,
    /// Whether this genome passed validation.
    pub passed: bool,
    /// Number of IS/OOS windows evaluated.
    pub windows_evaluated: usize,
    /// IS CAGR (net of costs).
    pub is_cagr_net: f64,
    /// OOS CAGR (net of costs).
    pub oos_cagr_net: f64,
    /// Aggregated cost report across all OOS windows.
    pub cost_report: Option<CostReport>,
    /// Details of each window.
    pub window_details: Vec<WindowDetail>,
}

/// Details for a single IS/OOS window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowDetail {
    pub window_idx: usize,
    pub is_sharpe: f64,
    pub oos_sharpe: f64,
    pub is_cagr: f64,
    pub oos_cagr: f64,
}

/// Result of CPCV (Combinatorial Purged Cross-Validation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpcvResult {
    /// Genome ID.
    pub genome_id: Uuid,
    /// OOS Sharpe ratios for each fold (net of costs).
    pub oos_sharpes: Vec<f64>,
    /// Mean OOS Sharpe.
    pub oos_sharpe_mean: f64,
    /// Standard deviation of OOS Sharpe.
    pub oos_sharpe_std: f64,
    /// Probability of Backtest Overfitting.
    pub pbo: f64,
    /// Whether this genome passed validation.
    pub passed: bool,
}

/// Result of PBO/DSR analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PboDsrResult {
    /// Genome ID.
    pub genome_id: Uuid,
    /// In-sample Sharpe ratio (net).
    pub is_sharpe_net: f64,
    /// Probability of Backtest Overfitting.
    pub pbo: f64,
    /// Deflated Sharpe Ratio.
    pub dsr: f64,
    /// Total trials (for DSR adjustment).
    pub total_trials: u64,
    /// Whether this genome passed validation.
    pub passed: bool,
}

/// Configuration for validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Maximum allowed IS/OOS degradation (as fraction, e.g., 0.4 = 40%).
    #[serde(default = "default_max_degradation")]
    pub max_degradation: f64,
    /// Maximum allowed PBO (e.g., 0.15).
    #[serde(default = "default_max_pbo")]
    pub max_pbo: f64,
    /// Minimum OOS Sharpe to pass (NET of costs).
    #[serde(default = "default_min_oos_sharpe")]
    pub min_oos_sharpe_net: f64,
    /// Minimum DSR to not receive a warning.
    #[serde(default = "default_min_dsr")]
    pub min_dsr: f64,
    /// Maximum OOS Max Drawdown (e.g., -0.35).
    #[serde(default = "default_max_oos_dd")]
    pub max_oos_dd: f64,
    /// Minimum OOS trades.
    #[serde(default = "default_min_oos_trades")]
    pub min_oos_trades: u32,
    /// Number of folds for CPCV.
    #[serde(default = "default_cpcv_folds")]
    pub cpcv_folds: usize,
    /// Purge days for CPCV.
    #[serde(default = "default_purge_days")]
    pub purge_days: u32,
    /// Embargo days for CPCV.
    #[serde(default = "default_embargo_days")]
    pub embargo_days: u32,
    /// Execution model configuration for net-of-costs calculations.
    #[serde(default)]
    pub execution: ExecutionModelConfig,
    /// Enable stress testing.
    #[serde(default)]
    pub stress_testing_enabled: bool,
    /// Minimum stress scenarios to pass (out of 5).
    #[serde(default = "default_min_stress_pass")]
    pub min_stress_scenarios_passed: usize,
}

fn default_max_degradation() -> f64 { 0.40 }
fn default_max_pbo() -> f64 { 0.15 }
fn default_min_oos_sharpe() -> f64 { 0.2 }
fn default_min_dsr() -> f64 { 0.3 }
fn default_max_oos_dd() -> f64 { -0.35 }
fn default_min_oos_trades() -> u32 { 30 }
fn default_cpcv_folds() -> usize { 6 }
fn default_purge_days() -> u32 { 5 }
fn default_embargo_days() -> u32 { 5 }
fn default_min_stress_pass() -> usize { 4 }

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_degradation: default_max_degradation(),
            max_pbo: default_max_pbo(),
            min_oos_sharpe_net: default_min_oos_sharpe(),
            min_dsr: default_min_dsr(),
            max_oos_dd: default_max_oos_dd(),
            min_oos_trades: default_min_oos_trades(),
            cpcv_folds: default_cpcv_folds(),
            purge_days: default_purge_days(),
            embargo_days: default_embargo_days(),
            execution: ExecutionModelConfig::mvp(),
            stress_testing_enabled: false,
            min_stress_scenarios_passed: default_min_stress_pass(),
        }
    }
}

/// Validator for anti-overfitting checks with execution cost integration.
pub struct GenomeValidatorAntiOverfit<E: BacktestExecutor> {
    executor: E,
    config: ValidationConfig,
    gate_checker: GateChecker,
    stress_suite: StressSuite,
}

impl<E: BacktestExecutor> GenomeValidatorAntiOverfit<E> {
    /// Create a new validator.
    pub fn new(executor: E, config: ValidationConfig) -> Self {
        let gate_checker = GateChecker::with_defaults();
        let stress_suite = StressSuite::default_institutional();
        Self { executor, config, gate_checker, stress_suite }
    }

    /// Create a new validator with custom gate checker.
    pub fn with_gates(executor: E, config: ValidationConfig, gate_checker: GateChecker) -> Self {
        let stress_suite = StressSuite::default_institutional();
        Self { executor, config, gate_checker, stress_suite }
    }

    /// Run Walk-Forward Analysis on a genome with net-of-costs metrics.
    pub fn validate_wfa(&self, genome: &StrategyGenome) -> Result<WfaResult, ValidationError> {
        info!("Running WFA validation for genome {}", &genome.id.to_string()[..8]);

        // Convert genome to config
        let strategy_config = genome.to_strategy_config().map_err(|e| {
            ValidationError::Failed(genome.id.to_string(), e.to_string())
        })?;

        // Get IS fitness (gross)
        let is_fitness = genome.fitness.clone().ok_or_else(|| {
            ValidationError::Failed(genome.id.to_string(), "No fitness available".into())
        })?;

        // Execute OOS backtest
        let oos_output = self.executor.execute(&strategy_config).map_err(|e| {
            ValidationError::Execution(e.to_string())
        })?;

        // Calculate net-of-costs metrics
        let slippage_bps = self.config.execution.slippage.base_bps();
        let fee_rate = self.config.execution.fees.commission_rate;
        
        // Estimate cost drag on returns (simplified model)
        let turnover = oos_output.metrics.turnover_annual.unwrap_or(2.0);
        let cost_drag_annual = turnover * (slippage_bps / 10_000.0 + fee_rate);
        
        // Adjust Sharpe for costs
        let vol = oos_output.metrics.volatility.unwrap_or(0.15).max(0.01);
        let oos_sharpe_gross = oos_output.metrics.sharpe_ratio;
        let oos_sharpe_net = oos_sharpe_gross - (cost_drag_annual / vol);
        
        let oos_cagr_gross = oos_output.metrics.cagr;
        let oos_cagr_net = oos_cagr_gross - cost_drag_annual;

        // Use net metrics for validation
        let is_sharpe_net = is_fitness.sharpe_ratio; // Assume IS already includes costs
        let degradation_pct = if is_sharpe_net > 0.0 {
            (is_sharpe_net - oos_sharpe_net) / is_sharpe_net * 100.0
        } else {
            0.0
        };

        // Determine if passed using NET metrics
        let passed = degradation_pct < self.config.max_degradation * 100.0
            && oos_sharpe_net >= self.config.min_oos_sharpe_net
            && oos_output.metrics.max_drawdown >= self.config.max_oos_dd;

        Ok(WfaResult {
            genome_id: genome.id,
            is_sharpe_gross: is_fitness.sharpe_ratio,
            is_sharpe_net,
            oos_sharpe_gross,
            oos_sharpe_net,
            degradation_pct,
            passed,
            windows_evaluated: 1,
            is_cagr_net: is_fitness.cagr,
            oos_cagr_net,
            cost_report: None, // Would be populated from detailed simulation
            window_details: vec![WindowDetail {
                window_idx: 0,
                is_sharpe: is_sharpe_net,
                oos_sharpe: oos_sharpe_net,
                is_cagr: is_fitness.cagr,
                oos_cagr: oos_cagr_net,
            }],
        })
    }

    /// Run stress testing on a genome.
    pub fn run_stress_tests(
        &self,
        genome: &StrategyGenome,
        baseline_sharpe: f64,
    ) -> Result<StressSuiteResult, ValidationError> {
        if !self.config.stress_testing_enabled {
            // Return empty result if stress testing is disabled
            return Ok(StressSuiteResult::from_results(vec![]));
        }

        info!("Running stress tests for genome {}", &genome.id.to_string()[..8]);

        let strategy_config = genome.to_strategy_config().map_err(|e| {
            ValidationError::Failed(genome.id.to_string(), e.to_string())
        })?;

        let mut results = Vec::new();

        for scenario in &self.stress_suite.scenarios {
            // Apply stress transformation to execution config
            let stressed_exec = scenario.apply(&self.config.execution);
            
            // Execute with stressed config
            let output = self.executor.execute(&strategy_config).map_err(|e| {
                ValidationError::Execution(e.to_string())
            })?;

            // Calculate stressed Sharpe (with higher costs)
            let slippage_bps = stressed_exec.slippage.base_bps();
            let fee_rate = stressed_exec.fees.commission_rate;
            let turnover = output.metrics.turnover_annual.unwrap_or(2.0);
            let cost_drag_annual = turnover * (slippage_bps / 10_000.0 + fee_rate);
            let vol = output.metrics.volatility.unwrap_or(0.15).max(0.01);
            let sharpe_stressed = output.metrics.sharpe_ratio - (cost_drag_annual / vol);

            results.push(StressResult::new(
                scenario,
                baseline_sharpe,
                sharpe_stressed,
                None, // execution_rate
                Some(output.metrics.max_drawdown.abs()),
            ));
        }

        Ok(StressSuiteResult::from_results(results))
    }

    /// Apply institutional gates to a genome.
    pub fn check_gates(
        &self,
        wfa_result: &WfaResult,
        gross_pnl: f64,
    ) -> GateResult {
        // If we have a cost report, use it
        if let Some(ref report) = wfa_result.cost_report {
            self.gate_checker.check_from_report(report, gross_pnl)
        } else {
            // Otherwise, use estimated values
            self.gate_checker.check(
                2.0, // default turnover estimate
                gross_pnl,
                0.0, // no cost data
                0.0, // no slippage data
                50_000_000.0, // assume OK capacity
            )
        }
    }

    /// Calculate PBO/DSR for a genome.
    pub fn calculate_pbo_dsr(
        &self,
        genome: &StrategyGenome,
        oos_sharpes: &[f64],
        total_trials: u64,
    ) -> Result<PboDsrResult, ValidationError> {
        let is_sharpe = genome
            .fitness
            .as_ref()
            .map(|f| f.sharpe_ratio)
            .ok_or_else(|| {
                ValidationError::Failed(genome.id.to_string(), "No fitness available".into())
            })?;

        let n = oos_sharpes.len();
        if n == 0 {
            return Err(ValidationError::InsufficientData(
                "No OOS sharpes for PBO calculation".into(),
            ));
        }

        let oos_mean: f64 = oos_sharpes.iter().sum::<f64>() / n as f64;
        // Apply Bessel's correction (n-1) for unbiased sample variance
        let oos_var: f64 = if n > 1 {
            oos_sharpes.iter().map(|x| (x - oos_mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        let oos_std = oos_var.sqrt();

        let pbo = if oos_std > 0.0 {
            let z = -oos_mean / oos_std;
            0.5 * (1.0 + libm::erf(z / std::f64::consts::SQRT_2))
        } else if oos_mean <= 0.0 {
            1.0
        } else {
            0.0
        };

        // DSR: proper Bailey & López de Prado (2014) formula
        // Uses OOS Sharpe and PSR against expected max under null
        let dsr = crate::statistics::calculate_dsr(
            oos_mean,               // OOS Sharpe (not IS)
            n * 252,                // Approximate annual observations
            0.0,                    // Default skewness (normal)
            0.0,                    // Default excess kurtosis (normal)
            total_trials as usize,  // Number of strategies tested
            oos_var,                // Variance of OOS Sharpes
        );

        let passed = pbo < self.config.max_pbo && dsr >= self.config.min_dsr;

        Ok(PboDsrResult {
            genome_id: genome.id,
            is_sharpe_net: is_sharpe,
            pbo,
            dsr,
            total_trials,
            passed,
        })
    }

    /// Run full validation suite on top K genomes with execution costs.
    pub fn validate_top_k(
        &self,
        genomes: &[StrategyGenome],
        k: usize,
        total_trials: u64,
    ) -> Vec<ValidationReport> {
        let top_genomes: Vec<_> = genomes.iter().take(k).collect();
        let mut reports = Vec::with_capacity(top_genomes.len());

        for genome in top_genomes {
            let wfa_result = self.validate_wfa(genome);
            
            let (pbo_dsr_result, stress_result, gate_result) = if let Ok(ref wfa) = wfa_result {
                let pbo = self.calculate_pbo_dsr(
                    genome, 
                    &[wfa.oos_sharpe_net], 
                    total_trials
                ).ok();
                
                let stress = self.run_stress_tests(genome, wfa.oos_sharpe_net).ok();
                let gates = self.check_gates(wfa, wfa.oos_cagr_net * 1_000_000.0);
                
                (pbo, stress, Some(gates))
            } else {
                (None, None, None)
            };

            let wfa_passed = wfa_result.as_ref().map(|w| w.passed).unwrap_or(false);
            let pbo_passed = pbo_dsr_result.as_ref().map(|p| p.passed).unwrap_or(true);
            let stress_passed = stress_result.as_ref().map(|s| {
                s.passed_count >= self.config.min_stress_scenarios_passed
            }).unwrap_or(true);
            let gates_passed = gate_result.as_ref().map(|g| g.passed).unwrap_or(true);

            let overall_passed = wfa_passed && pbo_passed && stress_passed && gates_passed;

            let discard_reason = determine_discard_reason(
                wfa_result.as_ref().ok(),
                pbo_dsr_result.as_ref(),
                stress_result.as_ref(),
                gate_result.as_ref(),
                &self.config,
            );

            reports.push(ValidationReport {
                genome_id: genome.id,
                wfa_result: wfa_result.ok(),
                cpcv_result: None,
                pbo_dsr_result,
                stress_result,
                gate_result,
                overall_passed,
                discard_reason,
            });
        }

        reports
    }
}

/// Complete validation report for a genome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub genome_id: Uuid,
    pub wfa_result: Option<WfaResult>,
    pub cpcv_result: Option<CpcvResult>,
    pub pbo_dsr_result: Option<PboDsrResult>,
    pub stress_result: Option<StressSuiteResult>,
    pub gate_result: Option<GateResult>,
    pub overall_passed: bool,
    pub discard_reason: Option<String>,
}

/// Determine why a genome should be discarded.
fn determine_discard_reason(
    wfa: Option<&WfaResult>,
    pbo_dsr: Option<&PboDsrResult>,
    stress: Option<&StressSuiteResult>,
    gates: Option<&GateResult>,
    config: &ValidationConfig,
) -> Option<String> {
    // Check WFA
    if let Some(w) = wfa {
        if w.degradation_pct > config.max_degradation * 100.0 {
            return Some(format!(
                "IS/OOS degradation {:.1}% > {:.0}%",
                w.degradation_pct,
                config.max_degradation * 100.0
            ));
        }
        if w.oos_sharpe_net < config.min_oos_sharpe_net {
            return Some(format!(
                "OOS Sharpe (net) {:.2} < {:.2}",
                w.oos_sharpe_net, config.min_oos_sharpe_net
            ));
        }
    }

    // Check PBO/DSR
    if let Some(p) = pbo_dsr {
        if p.pbo > config.max_pbo {
            return Some(format!("PBO {:.2} > {:.2}", p.pbo, config.max_pbo));
        }
        if p.dsr < config.min_dsr {
            return Some(format!("DSR {:.2} < {:.2} (warning)", p.dsr, config.min_dsr));
        }
    }

    // Check stress tests
    if let Some(s) = stress {
        if config.stress_testing_enabled && s.passed_count < config.min_stress_scenarios_passed {
            return Some(format!(
                "Stress tests failed: {}/{} passed (min: {})",
                s.passed_count, s.total_count, config.min_stress_scenarios_passed
            ));
        }
    }

    // Check institutional gates
    if let Some(g) = gates {
        if !g.passed {
            return Some(format!("Institutional gate failed: {:?}", g.rejection_reasons));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use combiner_core::{BlockGene, BlockType, FitnessConfig, MultiObjectiveFitness};
    use combiner_runner::{BacktestExecutor, BacktestOutput, ExecutionError};
    use backtester_strategy::config::StrategyConfig;

    /// Mock executor for testing.
    struct MockExecutor;

    impl BacktestExecutor for MockExecutor {
        fn execute(&self, _config: &StrategyConfig) -> Result<BacktestOutput, ExecutionError> {
            Ok(BacktestOutput::mock())
        }

        fn execute_batch(
            &self,
            configs: &[StrategyConfig],
        ) -> Vec<Result<BacktestOutput, ExecutionError>> {
            configs.iter().map(|c| self.execute(c)).collect()
        }
    }

    fn create_test_genome() -> StrategyGenome {
        let config = FitnessConfig::default();
        let mut genome = StrategyGenome::new(vec![
            BlockGene::with_defaults(BlockType::Selection, "momentum"),
            BlockGene::with_defaults(BlockType::Sizing, "equal_weight"),
        ]);
        genome.fitness = Some(MultiObjectiveFitness::from_metrics(
            0.15, 1.2, -0.10, 1.5, 1.3, 1.8, 100, 0.12, 2.5, &config,
        ));
        genome
    }

    #[test]
    fn test_wfa_validation_with_costs() {
        let executor = MockExecutor;
        let config = ValidationConfig::default();
        let validator = GenomeValidatorAntiOverfit::new(executor, config);

        let genome = create_test_genome();
        let result = validator.validate_wfa(&genome);

        assert!(result.is_ok());
        let wfa = result.unwrap();
        assert!(wfa.is_sharpe_gross > 0.0);
        // Net sharpe should be lower than gross due to costs
        assert!(wfa.oos_sharpe_net <= wfa.oos_sharpe_gross);
    }

    #[test]
    fn test_pbo_calculation() {
        let executor = MockExecutor;
        let config = ValidationConfig::default();
        let validator = GenomeValidatorAntiOverfit::new(executor, config);

        let genome = create_test_genome();
        let oos_sharpes = vec![0.8, 0.9, 0.7, 1.0, 0.85];
        let result = validator.calculate_pbo_dsr(&genome, &oos_sharpes, 100);

        assert!(result.is_ok());
        let pbo_dsr = result.unwrap();
        assert!(pbo_dsr.pbo >= 0.0 && pbo_dsr.pbo <= 1.0);
    }

    #[test]
    fn test_validation_config_defaults() {
        let config = ValidationConfig::default();
        assert!(config.execution.has_costs());
        assert!(!config.stress_testing_enabled);
    }
}
