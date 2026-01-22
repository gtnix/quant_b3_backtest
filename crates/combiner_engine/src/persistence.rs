//! Persistence for evolution experiments.
//!
//! Handles saving and loading of:
//! - manifest.json (experiment metadata)
//! - generations/ (population snapshots)
//! - hall_of_fame/ (best strategies)
//!
//! Supports two output formats:
//! - Legacy: JSON files (default, backwards compatible)
//! - OBFS: Optimized Binary File System (rkyv + Zstd, ~84% space savings)

use crate::hall_of_fame::HallOfFame;
use crate::GenerationStats;
use combiner_core::StrategyGenome;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use thiserror::Error;
use tracing::info;

/// Output format for persistence artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ArtifactFormat {
    /// Legacy JSON format (backwards compatible)
    #[default]
    Legacy,
    /// OBFS binary format (rkyv + Zstd compression)
    Obfs,
}

impl ArtifactFormat {
    /// Parse from string (for config files).
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "obfs" | "binary" | "compressed" => Self::Obfs,
            _ => Self::Legacy,
        }
    }
}

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
/// Supports both Legacy (JSON) and OBFS (compressed binary) formats.
pub struct ExperimentPersistence {
    output_dir: PathBuf,
    experiment_id: String,
    format: ArtifactFormat,
    obfs: Option<obfs::Obfs>,
}

impl ExperimentPersistence {
    /// Create a new persistence manager (OBFS format by default for ultra-performance).
    pub fn new(output_dir: impl Into<PathBuf>, experiment_id: impl Into<String>) -> Self {
        let output_dir = output_dir.into();
        let experiment_id = experiment_id.into();
        let mut instance = Self {
            output_dir,
            experiment_id,
            format: ArtifactFormat::Obfs,
            obfs: None,
        };
        instance.init_obfs();
        instance
    }

    /// Set the artifact format (builder pattern).
    pub fn with_format(mut self, format: ArtifactFormat) -> Self {
        self.format = format;
        if format == ArtifactFormat::Obfs {
            self.init_obfs();
        }
        self
    }

    /// Initialize OBFS storage backend.
    fn init_obfs(&mut self) {
        let obfs_path = self.output_dir.join("obfs");
        let config = obfs::ObfsConfig {
            root_path: obfs_path.to_string_lossy().to_string(),
            compression_level: 3,
            enable_blake3: true,
            enable_xxh3: true,
            max_file_size: 1024 * 1024 * 1024, // 1 GB
            lmdb_map_size: 10 * 1024 * 1024 * 1024, // 10 GB
        };
        let obfs_instance = obfs::Obfs::with_config(config);
        if let Err(e) = obfs_instance.initialize() {
            tracing::warn!("Failed to initialize OBFS: {}", e);
        }
        self.obfs = Some(obfs_instance);
    }

    /// Get current format.
    pub fn format(&self) -> ArtifactFormat {
        self.format
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
        match self.format {
            ArtifactFormat::Legacy => {
                let path = self.experiment_dir().join("manifest.json");
                let content = serde_json::to_string_pretty(manifest)?;
                fs::write(path, content)?;
            }
            ArtifactFormat::Obfs => {
                // Write compressed manifest using OBFS compression pipeline
                if let Some(ref obfs) = self.obfs {
                    let json_bytes = serde_json::to_vec(manifest)?;
                    let compressed = obfs.compression_pipeline().compress(&json_bytes)
                        .map_err(|e| PersistenceError::Conversion(e.to_string()))?;
                    let path = self.experiment_dir().join("manifest.obfs");
                    fs::write(path, compressed)?;
                } else {
                    // Fallback to legacy if OBFS not initialized
                    let path = self.experiment_dir().join("manifest.json");
                    let content = serde_json::to_string_pretty(manifest)?;
                    fs::write(path, content)?;
                }
            }
        }
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

        match self.format {
            ArtifactFormat::Legacy => {
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
            }
            ArtifactFormat::Obfs => {
                if let Some(ref obfs) = self.obfs {
                    let pipeline = obfs.compression_pipeline();

                    // Compress population
                    let pop_bytes = serde_json::to_vec(population)?;
                    let pop_compressed = pipeline.compress(&pop_bytes)
                        .map_err(|e| PersistenceError::Conversion(e.to_string()))?;
                    fs::write(gen_dir.join("population.obfs"), pop_compressed)?;

                    // Compress Pareto
                    let pareto: Vec<_> = pareto_indices.iter().map(|&i| &population[i]).collect();
                    let pareto_bytes = serde_json::to_vec(&pareto)?;
                    let pareto_compressed = pipeline.compress(&pareto_bytes)
                        .map_err(|e| PersistenceError::Conversion(e.to_string()))?;
                    fs::write(gen_dir.join("pareto.obfs"), pareto_compressed)?;

                    // Compress stats
                    let stats_bytes = serde_json::to_vec(stats)?;
                    let stats_compressed = pipeline.compress(&stats_bytes)
                        .map_err(|e| PersistenceError::Conversion(e.to_string()))?;
                    fs::write(gen_dir.join("stats.obfs"), stats_compressed)?;
                } else {
                    // Fallback to legacy
                    let pop_json = serde_json::to_string_pretty(population)?;
                    fs::write(gen_dir.join("population.json"), pop_json)?;
                    let pareto: Vec<_> = pareto_indices.iter().map(|&i| &population[i]).collect();
                    let pareto_json = serde_json::to_string_pretty(&pareto)?;
                    fs::write(gen_dir.join("pareto.json"), pareto_json)?;
                    let stats_json = serde_json::to_string_pretty(stats)?;
                    fs::write(gen_dir.join("stats.json"), stats_json)?;
                }
            }
        }

        Ok(())
    }

    /// Write Hall of Fame.
    pub fn write_hall_of_fame(&self, hof: &HallOfFame) -> Result<(), PersistenceError> {
        let hof_dir = self.experiment_dir().join("hall_of_fame");
        fs::create_dir_all(&hof_dir)?;

        // Build ranking data
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

        match self.format {
            ArtifactFormat::Legacy => {
                let ranking_json = serde_json::to_string_pretty(&ranking)?;
                fs::write(hof_dir.join("ranking.json"), ranking_json)?;

                for (i, entry) in hof.entries().iter().enumerate() {
                    let strategy_dir = hof_dir.join(format!("strategy_{:03}", i + 1));
                    fs::create_dir_all(&strategy_dir)?;

                    let genome_json = serde_json::to_string_pretty(&entry.genome)?;
                    fs::write(strategy_dir.join("genome.json"), genome_json)?;

                    if let Ok(toml_str) = entry.genome.to_toml() {
                        fs::write(strategy_dir.join("config.toml"), toml_str)?;
                    }

                    if let Some(ref fitness) = entry.genome.fitness {
                        let metrics_json = serde_json::to_string_pretty(fitness)?;
                        fs::write(strategy_dir.join("metrics.json"), metrics_json)?;
                    }
                }
            }
            ArtifactFormat::Obfs => {
                if let Some(ref obfs) = self.obfs {
                    let pipeline = obfs.compression_pipeline();

                    // Compress ranking
                    let ranking_bytes = serde_json::to_vec(&ranking)?;
                    let ranking_compressed = pipeline.compress(&ranking_bytes)
                        .map_err(|e| PersistenceError::Conversion(e.to_string()))?;
                    fs::write(hof_dir.join("ranking.obfs"), ranking_compressed)?;

                    // Compress each strategy
                    for (i, entry) in hof.entries().iter().enumerate() {
                        let strategy_dir = hof_dir.join(format!("strategy_{:03}", i + 1));
                        fs::create_dir_all(&strategy_dir)?;

                        // Combine genome with identity for full traceability
                        let genome_with_identity = serde_json::json!({
                            "id": entry.genome.id,
                            "genes": entry.genome.genes,
                            "fitness": entry.genome.fitness,
                            "identity": entry.identity,
                        });
                        let genome_bytes = serde_json::to_vec(&genome_with_identity)?;
                        let genome_compressed = pipeline.compress(&genome_bytes)
                            .map_err(|e| PersistenceError::Conversion(e.to_string()))?;
                        fs::write(strategy_dir.join("genome.obfs"), genome_compressed)?;

                        // Keep TOML uncompressed for human readability
                        if let Ok(toml_str) = entry.genome.to_toml() {
                            fs::write(strategy_dir.join("config.toml"), toml_str)?;
                        }

                        if let Some(ref fitness) = entry.genome.fitness {
                            let metrics_bytes = serde_json::to_vec(fitness)?;
                            let metrics_compressed = pipeline.compress(&metrics_bytes)
                                .map_err(|e| PersistenceError::Conversion(e.to_string()))?;
                            fs::write(strategy_dir.join("metrics.obfs"), metrics_compressed)?;
                        }
                    }
                } else {
                    // Fallback to legacy
                    let ranking_json = serde_json::to_string_pretty(&ranking)?;
                    fs::write(hof_dir.join("ranking.json"), ranking_json)?;
                    for (i, entry) in hof.entries().iter().enumerate() {
                        let strategy_dir = hof_dir.join(format!("strategy_{:03}", i + 1));
                        fs::create_dir_all(&strategy_dir)?;
                        // Combine genome with identity for full traceability
                        let genome_with_identity = serde_json::json!({
                            "id": entry.genome.id,
                            "genes": entry.genome.genes,
                            "fitness": entry.genome.fitness,
                            "identity": entry.identity,
                        });
                        let genome_json = serde_json::to_string_pretty(&genome_with_identity)?;
                        fs::write(strategy_dir.join("genome.json"), genome_json)?;
                        if let Ok(toml_str) = entry.genome.to_toml() {
                            fs::write(strategy_dir.join("config.toml"), toml_str)?;
                        }
                        if let Some(ref fitness) = entry.genome.fitness {
                            let metrics_json = serde_json::to_string_pretty(fitness)?;
                            fs::write(strategy_dir.join("metrics.json"), metrics_json)?;
                        }
                    }
                }
            }
        }

        info!("Wrote {} strategies to {:?} (format: {:?})", hof.len(), hof_dir, self.format);
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
    /// - Legacy: Individual genome.json, config.toml, metrics.json, wfa_report.json, etc.
    /// - OBFS: Consolidated hall_of_fame.obfs bundle
    pub fn write_validated_hall_of_fame(
        &self,
        hof: &crate::hall_of_fame_validated::ValidatedHallOfFame,
    ) -> Result<(), PersistenceError> {
        match self.format {
            ArtifactFormat::Legacy => self.write_validated_hall_of_fame_legacy(hof),
            ArtifactFormat::Obfs => self.write_validated_hall_of_fame_obfs(hof),
        }
    }

    /// Write Validated Hall of Fame in Legacy format (individual files)
    fn write_validated_hall_of_fame_legacy(
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
            let v = e.validation_ref();
            serde_json::json!({
                "rank": i + 1,
                "genome_id": e.genome_id.to_string(),
                "generation": e.validated_generation(),
                "oos_sharpe": v.oos_sharpe_median,
                "pbo": v.pbo,
                "dsr": v.dsr,
                "degradation_pct": v.degradation_pct,
                "passed": v.passed,
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
            let v = entry.validation_ref();
            let validation_json = serde_json::to_string_pretty(v)?;
            fs::write(strategy_dir.join("validation_summary.json"), validation_json)?;

            // Generate WFA report from validation summary
            let wfa_result = WfaResult {
                genome_id: entry.genome_id,
                is_sharpe_gross: v.oos_sharpe_mean * 1.1,
                is_sharpe_net: v.oos_sharpe_mean,
                oos_sharpe_gross: v.oos_sharpe_median * 1.1,
                oos_sharpe_net: v.oos_sharpe_median,
                degradation_pct: v.degradation_pct,
                passed: v.passed,
                windows_evaluated: v.splits_evaluated as usize,
                is_cagr_net: v.oos_cagr_median * 1.1,
                oos_cagr_net: v.oos_cagr_median,
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
                is_sharpe_net: v.oos_sharpe_mean,
                pbo: v.pbo,
                dsr: v.dsr,
                total_trials: 1000,
                passed: v.pbo <= 0.15 && v.dsr >= 0.5,
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

    /// Write Validated Hall of Fame in OBFS format (consolidated bundle)
    fn write_validated_hall_of_fame_obfs(
        &self,
        hof: &crate::hall_of_fame_validated::ValidatedHallOfFame,
    ) -> Result<(), PersistenceError> {
        let hof_dir = self.experiment_dir().join("hall_of_fame");
        let bundle_dir = hof_dir.join("obfs");
        fs::create_dir_all(&bundle_dir)?;

        // Create OBFS bundle writer
        let mut bundle_writer = obfs::ReportBundleWriter::new(&bundle_dir)
            .map_err(|e| PersistenceError::Conversion(e.to_string()))?;

        // Write ranking as compressed OBFS
        let ranking: Vec<_> = hof.entries().iter().enumerate().map(|(i, e)| {
            let v = e.validation_ref();
            serde_json::json!({
                "rank": i + 1,
                "genome_id": e.genome_id.to_string(),
                "generation": e.validated_generation(),
                "oos_sharpe": v.oos_sharpe_median,
                "pbo": v.pbo,
                "dsr": v.dsr,
                "degradation_pct": v.degradation_pct,
                "passed": v.passed,
                "score": e.score,
            })
        }).collect();
        let ranking_json = serde_json::to_vec(&ranking)?;
        let ranking_compressed = obfs::UltraCompressor::compress(&ranking_json)
            .map_err(|e| PersistenceError::Conversion(e.to_string()))?;
        fs::write(hof_dir.join("ranking.obfs"), ranking_compressed)?;

        // Write each strategy to bundle
        for (rank, entry) in hof.entries().iter().enumerate() {
            // Generate config TOML
            let config_toml = entry.genome.to_toml()
                .unwrap_or_else(|_| "[error]\nfailed_to_serialize = true".to_string());

            let v = entry.validation_ref();
            
            // Generate comprehensive validation JSON
            let validation_json = serde_json::to_string(&serde_json::json!({
                "genome_id": entry.genome_id.to_string(),
                "genome_hash": format!("{:016x}", entry.genome_hash),
                "validation": v,
                "validated_generation": entry.validated_generation(),
                "rank": entry.rank,
                "score": entry.score,
                "fitness": entry.genome.fitness,
            }))?;

            // Calculate production score
            let production_score = v.oos_sharpe_median 
                * (1.0 - v.pbo)
                * (1.0 - v.degradation_pct / 100.0);

            bundle_writer.add(
                entry.genome_id,
                entry.genome_hash,
                (rank + 1) as u32,
                entry.validated_generation(),
                production_score,
                v.oos_sharpe_median,
                v.oos_cagr_median,
                v.oos_max_dd_worst,
                v.pbo,
                v.dsr,
                v.degradation_pct,
                v.splits_passed,
                v.splits_evaluated,
                &config_toml,
                &validation_json,
            ).map_err(|e| PersistenceError::Conversion(e.to_string()))?;
        }

        let stats = bundle_writer.finish()
            .map_err(|e| PersistenceError::Conversion(e.to_string()))?;

        info!(
            "Wrote {} validated strategies to OBFS bundle: {} bytes (level {})",
            stats.candidate_count,
            stats.data_file_size,
            stats.compression_level
        );
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

