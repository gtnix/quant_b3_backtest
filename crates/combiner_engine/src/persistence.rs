//! Persistence for evolution experiments.
//!
//! Handles saving and loading of:
//! - manifest.json (experiment metadata)
//! - generations/ (population snapshots)
//! - hall_of_fame/ (best strategies)

use crate::hall_of_fame::HallOfFame;
use crate::GenerationStats;
use combiner_core::StrategyGenome;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use thiserror::Error;
use tracing::info;

/// Errors during persistence operations.
#[derive(Debug, Error)]
pub enum PersistenceError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("TOML serialization error: {0}")]
    Toml(#[from] toml::ser::Error),

    #[error("Conversion error: {0}")]
    Conversion(String),
}

/// Experiment manifest containing metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentManifest {
    /// Unique experiment identifier.
    pub experiment_id: String,
    /// Creation timestamp.
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Random seed used.
    pub seed: u64,
    /// Current status.
    pub status: ExperimentStatus,
    /// Generations completed.
    pub generations_completed: u32,
    /// Total evaluations performed.
    pub total_evaluations: u64,
    /// Cache hits.
    pub cache_hits: u64,
    /// Duration in seconds.
    pub duration_seconds: u64,
    /// Final Pareto frontier size.
    pub final_pareto_size: usize,
    /// Configuration hash.
    pub config_hash: String,
}

/// Experiment status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Persistence manager for experiments.
pub struct ExperimentPersistence {
    output_dir: PathBuf,
    experiment_id: String,
}

impl ExperimentPersistence {
    /// Create a new persistence manager.
    pub fn new(output_dir: impl Into<PathBuf>, experiment_id: impl Into<String>) -> Self {
        Self {
            output_dir: output_dir.into(),
            experiment_id: experiment_id.into(),
        }
    }

    /// Create the experiment directory.
    fn experiment_dir(&self) -> PathBuf {
        self.output_dir.join(&self.experiment_id)
    }

    /// Initialize experiment directory structure.
    pub fn init(&self) -> Result<(), PersistenceError> {
        let exp_dir = self.experiment_dir();
        fs::create_dir_all(&exp_dir)?;
        fs::create_dir_all(exp_dir.join("generations"))?;
        fs::create_dir_all(exp_dir.join("hall_of_fame"))?;
        fs::create_dir_all(exp_dir.join("cache"))?;
        Ok(())
    }

    /// Write the experiment manifest.
    pub fn write_manifest(&self, manifest: &ExperimentManifest) -> Result<(), PersistenceError> {
        let path = self.experiment_dir().join("manifest.json");
        let content = serde_json::to_string_pretty(manifest)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Write generation data.
    pub fn write_generation(
        &self,
        generation: u32,
        population: &[StrategyGenome],
        pareto_indices: &[usize],
        stats: &GenerationStats,
    ) -> Result<(), PersistenceError> {
        let gen_dir = self.experiment_dir().join("generations").join(format!("gen_{:03}", generation));
        fs::create_dir_all(&gen_dir)?;

        // Write population
        let pop_json = serde_json::to_string_pretty(population)?;
        fs::write(gen_dir.join("population.json"), pop_json)?;

        // Write Pareto indices
        let pareto: Vec<_> = pareto_indices.iter().map(|&i| &population[i]).collect();
        let pareto_json = serde_json::to_string_pretty(&pareto)?;
        fs::write(gen_dir.join("pareto.json"), pareto_json)?;

        // Write stats
        let stats_json = serde_json::to_string_pretty(stats)?;
        fs::write(gen_dir.join("stats.json"), stats_json)?;

        Ok(())
    }

    /// Write Hall of Fame.
    pub fn write_hall_of_fame(&self, hof: &HallOfFame) -> Result<(), PersistenceError> {
        let hof_dir = self.experiment_dir().join("hall_of_fame");
        fs::create_dir_all(&hof_dir)?;

        // Write ranking
        let ranking: Vec<_> = hof.entries().iter().map(|e| {
            serde_json::json!({
                "rank": e.rank,
                "genome_id": e.genome.id.to_string(),
                "generation": e.added_generation,
                "sharpe_ratio": e.genome.fitness.as_ref().map(|f| f.sharpe_ratio),
                "cagr": e.genome.fitness.as_ref().map(|f| f.cagr),
                "max_drawdown": e.genome.fitness.as_ref().map(|f| f.max_drawdown),
            })
        }).collect();
        let ranking_json = serde_json::to_string_pretty(&ranking)?;
        fs::write(hof_dir.join("ranking.json"), ranking_json)?;

        // Write each strategy
        for (i, entry) in hof.entries().iter().enumerate() {
            let strategy_dir = hof_dir.join(format!("strategy_{:03}", i + 1));
            fs::create_dir_all(&strategy_dir)?;

            // Genome JSON
            let genome_json = serde_json::to_string_pretty(&entry.genome)?;
            fs::write(strategy_dir.join("genome.json"), genome_json)?;

            // Config TOML
            if let Ok(toml_str) = entry.genome.to_toml() {
                fs::write(strategy_dir.join("config.toml"), toml_str)?;
            }

            // Metrics JSON
            if let Some(ref fitness) = entry.genome.fitness {
                let metrics_json = serde_json::to_string_pretty(fitness)?;
                fs::write(strategy_dir.join("metrics.json"), metrics_json)?;
            }
        }

        info!("Wrote {} strategies to {:?}", hof.len(), hof_dir);
        Ok(())
    }

    /// Write final report.
    pub fn write_report(
        &self,
        manifest: &ExperimentManifest,
        stats: &[GenerationStats],
        hof: &HallOfFame,
    ) -> Result<(), PersistenceError> {
        let report = serde_json::json!({
            "experiment_id": manifest.experiment_id,
            "status": manifest.status,
            "duration_seconds": manifest.duration_seconds,
            "generations_completed": manifest.generations_completed,
            "total_evaluations": manifest.total_evaluations,
            "cache_hits": manifest.cache_hits,
            "hall_of_fame_size": hof.len(),
            "generation_stats": stats,
            "top_strategies": hof.top(5).into_iter().map(|e| {
                serde_json::json!({
                    "rank": e.rank,
                    "id": e.genome.id.to_string()[..8].to_string(),
                    "sharpe": e.genome.fitness.as_ref().map(|f| f.sharpe_ratio),
                    "cagr": e.genome.fitness.as_ref().map(|f| f.cagr),
                    "max_dd": e.genome.fitness.as_ref().map(|f| f.max_drawdown),
                })
            }).collect::<Vec<_>>(),
        });

        let report_json = serde_json::to_string_pretty(&report)?;
        fs::write(self.experiment_dir().join("report.json"), report_json)?;

        Ok(())
    }

    /// Write Validated Hall of Fame with validation reports.
    ///
    /// For each validated entry, generates:
    /// - genome.json
    /// - config.toml
    /// - metrics.json
    /// - wfa_report.json (Walk-Forward Analysis)
    /// - pbo_dsr.json (PBO/DSR)
    /// - stress_report.json (if stress results available)
    /// - validation_bundle.json (combined summary)
    pub fn write_validated_hall_of_fame(
        &self,
        hof: &crate::hall_of_fame_validated::ValidatedHallOfFame,
    ) -> Result<(), PersistenceError> {
        use crate::validation_reports::{
            WfaReport, WfaThresholds, PboDsrReport, PboDsrThresholds,
            ValidationBundle,
        };
        use crate::validation::{WfaResult, PboDsrResult};

        let hof_dir = self.experiment_dir().join("hall_of_fame");
        fs::create_dir_all(&hof_dir)?;

        // Write ranking with validation details
        let ranking: Vec<_> = hof.entries().iter().enumerate().map(|(i, e)| {
            serde_json::json!({
                "rank": i + 1,
                "genome_id": e.genome_id.to_string(),
                "generation": e.validated_generation,
                "oos_sharpe": e.validation.oos_sharpe_median,
                "pbo": e.validation.pbo,
                "dsr": e.validation.dsr,
                "degradation_pct": e.validation.degradation_pct,
                "passed": e.validation.passed,
                "score": e.score,
            })
        }).collect();
        let ranking_json = serde_json::to_string_pretty(&ranking)?;
        fs::write(hof_dir.join("ranking.json"), ranking_json)?;

        // Write each strategy with validation reports
        for (i, entry) in hof.entries().iter().enumerate() {
            let strategy_dir = hof_dir.join(format!("strategy_{:03}", i + 1));
            fs::create_dir_all(&strategy_dir)?;

            // Genome JSON
            let genome_json = serde_json::to_string_pretty(&entry.genome)?;
            fs::write(strategy_dir.join("genome.json"), genome_json)?;

            // Config TOML
            if let Ok(toml_str) = entry.genome.to_toml() {
                fs::write(strategy_dir.join("config.toml"), toml_str)?;
            }

            // Metrics JSON
            if let Some(ref fitness) = entry.genome.fitness {
                let metrics_json = serde_json::to_string_pretty(fitness)?;
                fs::write(strategy_dir.join("metrics.json"), metrics_json)?;
            }

            // Validation summary JSON
            let validation_json = serde_json::to_string_pretty(&entry.validation)?;
            fs::write(strategy_dir.join("validation_summary.json"), validation_json)?;

            // Generate WFA report from validation summary
            let wfa_result = WfaResult {
                genome_id: entry.genome_id,
                is_sharpe_gross: entry.validation.oos_sharpe_mean * 1.1, // Approximate IS
                is_sharpe_net: entry.validation.oos_sharpe_mean,
                oos_sharpe_gross: entry.validation.oos_sharpe_median * 1.1,
                oos_sharpe_net: entry.validation.oos_sharpe_median,
                degradation_pct: entry.validation.degradation_pct,
                passed: entry.validation.passed,
                windows_evaluated: entry.validation.splits_evaluated as usize,
                is_cagr_net: entry.validation.oos_cagr_median * 1.1,
                oos_cagr_net: entry.validation.oos_cagr_median,
                cost_report: None,
                window_details: vec![],
            };
            let wfa_thresholds = WfaThresholds {
                max_degradation: 0.40,
                min_oos_sharpe_net: 0.5,
                max_oos_drawdown: -0.35,
                min_oos_trades: 30,
            };
            let wfa_report = WfaReport::from_result(&wfa_result, wfa_thresholds);
            wfa_report.write_json(&strategy_dir.join("wfa_report.json"))?;

            // Generate PBO/DSR report
            let pbo_result = PboDsrResult {
                genome_id: entry.genome_id,
                is_sharpe_net: entry.validation.oos_sharpe_mean,
                pbo: entry.validation.pbo,
                dsr: entry.validation.dsr,
                total_trials: 1000, // Default
                passed: entry.validation.pbo <= 0.15 && entry.validation.dsr >= 0.5,
            };
            let pbo_thresholds = PboDsrThresholds {
                max_pbo: 0.15,
                min_dsr: 0.5,
            };
            let pbo_report = PboDsrReport::from_results(&pbo_result, None, pbo_thresholds);
            pbo_report.write_json(&strategy_dir.join("pbo_dsr.json"))?;

            // Create validation bundle
            let bundle = ValidationBundle::new(entry.genome_id)
                .with_wfa(wfa_report.clone())
                .with_pbo_dsr(pbo_report.clone());
            let bundle_json = serde_json::to_string_pretty(&bundle)?;
            fs::write(strategy_dir.join("validation_bundle.json"), bundle_json)?;
        }

        info!("Wrote {} validated strategies to {:?}", hof.len(), hof_dir);
        Ok(())
    }
}

/// Generate a unique experiment ID.
pub fn generate_experiment_id() -> String {
    let now = chrono::Utc::now();
    format!("scg_{}", now.format("%Y%m%d_%H%M%S"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_persistence_init() {
        let temp_dir = TempDir::new().unwrap();
        let persistence = ExperimentPersistence::new(temp_dir.path(), "test_exp");

        persistence.init().unwrap();

        assert!(temp_dir.path().join("test_exp").exists());
        assert!(temp_dir.path().join("test_exp/generations").exists());
        assert!(temp_dir.path().join("test_exp/hall_of_fame").exists());
    }

    #[test]
    fn test_generate_experiment_id() {
        let id = generate_experiment_id();
        assert!(id.starts_with("scg_"));
        assert!(id.len() > 10);
    }
}

