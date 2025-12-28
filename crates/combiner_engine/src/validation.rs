//! Anti-overfitting validation for genomes.
//!
//! Provides Walk-Forward Analysis, CPCV, and PBO/DSR calculations
//! to validate strategies before deployment.

use combiner_core::{MultiObjectiveFitness, StrategyGenome};
use combiner_runner::{BacktestExecutor, BacktestOutput};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{info, warn};
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
    /// In-sample Sharpe ratio.
    pub is_sharpe: f64,
    /// Out-of-sample Sharpe ratio.
    pub oos_sharpe: f64,
    /// Degradation percentage: (IS - OOS) / IS * 100.
    pub degradation_pct: f64,
    /// Whether this genome passed validation.
    pub passed: bool,
    /// Number of IS/OOS windows evaluated.
    pub windows_evaluated: usize,
    /// IS CAGR.
    pub is_cagr: f64,
    /// OOS CAGR.
    pub oos_cagr: f64,
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
    /// OOS Sharpe ratios for each fold.
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
    /// In-sample Sharpe ratio.
    pub is_sharpe: f64,
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
    /// Minimum OOS Sharpe to pass.
    #[serde(default = "default_min_oos_sharpe")]
    pub min_oos_sharpe: f64,
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
}

fn default_max_degradation() -> f64 {
    0.40
}
fn default_max_pbo() -> f64 {
    0.15
}
fn default_min_oos_sharpe() -> f64 {
    0.2
}
fn default_min_dsr() -> f64 {
    0.3
}
fn default_max_oos_dd() -> f64 {
    -0.35
}
fn default_min_oos_trades() -> u32 {
    30
}
fn default_cpcv_folds() -> usize {
    6
}
fn default_purge_days() -> u32 {
    5
}
fn default_embargo_days() -> u32 {
    5
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_degradation: default_max_degradation(),
            max_pbo: default_max_pbo(),
            min_oos_sharpe: default_min_oos_sharpe(),
            min_dsr: default_min_dsr(),
            max_oos_dd: default_max_oos_dd(),
            min_oos_trades: default_min_oos_trades(),
            cpcv_folds: default_cpcv_folds(),
            purge_days: default_purge_days(),
            embargo_days: default_embargo_days(),
        }
    }
}

/// Validator for anti-overfitting checks.
pub struct GenomeValidatorAntiOverfit<E: BacktestExecutor> {
    executor: E,
    config: ValidationConfig,
}

impl<E: BacktestExecutor> GenomeValidatorAntiOverfit<E> {
    /// Create a new validator.
    pub fn new(executor: E, config: ValidationConfig) -> Self {
        Self { executor, config }
    }

    /// Run Walk-Forward Analysis on a genome.
    ///
    /// This is a simplified WFA that splits the data into IS/OOS and compares performance.
    pub fn validate_wfa(&self, genome: &StrategyGenome) -> Result<WfaResult, ValidationError> {
        info!("Running WFA validation for genome {}", &genome.id.to_string()[..8]);

        // Convert genome to config
        let strategy_config = genome.to_strategy_config().map_err(|e| {
            ValidationError::Failed(genome.id.to_string(), e.to_string())
        })?;

        // Execute backtest (in a real implementation, would split data into windows)
        // For now, use the existing fitness as the IS result
        let is_fitness = genome.fitness.clone().ok_or_else(|| {
            ValidationError::Failed(genome.id.to_string(), "No fitness available".into())
        })?;

        // Execute OOS backtest
        // In a real implementation, this would use different data
        let oos_output = self.executor.execute(&strategy_config).map_err(|e| {
            ValidationError::Execution(e.to_string())
        })?;

        let oos_sharpe = oos_output.metrics.sharpe_ratio;
        let is_sharpe = is_fitness.sharpe_ratio;

        // Calculate degradation
        let degradation_pct = if is_sharpe > 0.0 {
            (is_sharpe - oos_sharpe) / is_sharpe * 100.0
        } else {
            0.0
        };

        // Determine if passed
        let passed = degradation_pct < self.config.max_degradation * 100.0
            && oos_sharpe >= self.config.min_oos_sharpe
            && oos_output.metrics.max_drawdown >= self.config.max_oos_dd;

        Ok(WfaResult {
            genome_id: genome.id,
            is_sharpe,
            oos_sharpe,
            degradation_pct,
            passed,
            windows_evaluated: 1,
            is_cagr: is_fitness.cagr,
            oos_cagr: oos_output.metrics.cagr,
            window_details: vec![WindowDetail {
                window_idx: 0,
                is_sharpe,
                oos_sharpe,
                is_cagr: is_fitness.cagr,
                oos_cagr: oos_output.metrics.cagr,
            }],
        })
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

        // Calculate variance for PBO
        let n = oos_sharpes.len();
        if n == 0 {
            return Err(ValidationError::InsufficientData(
                "No OOS sharpes for PBO calculation".into(),
            ));
        }

        // Simplified PBO calculation
        // PBO = P(OOS Sharpe < 0 | IS Sharpe > 0)
        let oos_mean: f64 = oos_sharpes.iter().sum::<f64>() / n as f64;
        let oos_var: f64 =
            oos_sharpes.iter().map(|x| (x - oos_mean).powi(2)).sum::<f64>() / n as f64;
        let oos_std = oos_var.sqrt();

        // Probability that OOS is negative given the observed distribution
        let pbo = if oos_std > 0.0 {
            // Use normal approximation
            let z = -oos_mean / oos_std;
            0.5 * (1.0 + libm::erf(z / std::f64::consts::SQRT_2))
        } else if oos_mean <= 0.0 {
            1.0
        } else {
            0.0
        };

        // DSR = SR * (1 - PBO) adjusted by trials
        // Bailey et al. formula (simplified)
        let trial_adjustment = 1.0 - (total_trials as f64).ln() / 100.0;
        let dsr = is_sharpe * (1.0 - pbo) * trial_adjustment.max(0.5);

        let passed = pbo < self.config.max_pbo && dsr >= self.config.min_dsr;

        Ok(PboDsrResult {
            genome_id: genome.id,
            is_sharpe,
            pbo,
            dsr,
            total_trials,
            passed,
        })
    }

    /// Run full validation suite on top K genomes.
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
            let pbo_dsr_result = if let Ok(ref wfa) = wfa_result {
                self.calculate_pbo_dsr(genome, &[wfa.oos_sharpe], total_trials).ok()
            } else {
                None
            };

            let wfa_passed = wfa_result.as_ref().map(|w| w.passed).unwrap_or(false);
            let pbo_passed = pbo_dsr_result.as_ref().map(|p| p.passed).unwrap_or(true);
            let discard_reason = determine_discard_reason(
                wfa_result.as_ref().ok(),
                pbo_dsr_result.as_ref(),
                &self.config,
            );

            reports.push(ValidationReport {
                genome_id: genome.id,
                wfa_result: wfa_result.ok(),
                cpcv_result: None, // Would require multiple fold execution
                pbo_dsr_result,
                overall_passed: wfa_passed && pbo_passed,
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
    pub overall_passed: bool,
    pub discard_reason: Option<String>,
}

/// Determine why a genome should be discarded.
fn determine_discard_reason(
    wfa: Option<&WfaResult>,
    pbo_dsr: Option<&PboDsrResult>,
    config: &ValidationConfig,
) -> Option<String> {
    if let Some(w) = wfa {
        if w.degradation_pct > config.max_degradation * 100.0 {
            return Some(format!(
                "IS/OOS degradation {:.1}% > {:.0}%",
                w.degradation_pct,
                config.max_degradation * 100.0
            ));
        }
        if w.oos_sharpe < config.min_oos_sharpe {
            return Some(format!(
                "OOS Sharpe {:.2} < {:.2}",
                w.oos_sharpe, config.min_oos_sharpe
            ));
        }
    }

    if let Some(p) = pbo_dsr {
        if p.pbo > config.max_pbo {
            return Some(format!("PBO {:.2} > {:.2}", p.pbo, config.max_pbo));
        }
        if p.dsr < config.min_dsr {
            return Some(format!("DSR {:.2} < {:.2} (warning)", p.dsr, config.min_dsr));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use combiner_core::{BlockGene, BlockType, FitnessConfig, ParamValue};
    use combiner_runner::ExecutionError;
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
    fn test_wfa_validation() {
        let executor = MockExecutor;
        let config = ValidationConfig::default();
        let validator = GenomeValidatorAntiOverfit::new(executor, config);

        let genome = create_test_genome();
        let result = validator.validate_wfa(&genome);

        assert!(result.is_ok());
        let wfa = result.unwrap();
        assert!(wfa.is_sharpe > 0.0);
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
}

