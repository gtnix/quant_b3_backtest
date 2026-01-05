//! Final report generation for SCG experiments.
//!
//! This module generates comprehensive final reports including:
//! - Hall of Fame strategies with validation evidence
//! - Performance metrics and timing breakdown
//! - Evolution history and convergence analysis
//! - Overfitting risk assessment
//!
//! Supports two output formats:
//! - Legacy: Individual JSON/TOML files (backwards compatible)
//! - OBFS: Consolidated ultra-compressed binary bundle (~10x space reduction)

use std::path::{Path, PathBuf};
use std::fs;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

use combiner_core::{GenomeConverter, ConversionError};
use crate::config::EvolutionConfig;
use crate::hall_of_fame_validated::{ValidatedHallOfFame, ValidatedHofEntry};
use crate::performance_metrics::{PerformanceMetrics, PerformanceMetricsSummary, GenerationSnapshot};

/// Output format for reports
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReportFormat {
    /// Legacy JSON/TOML files (backwards compatible)
    #[default]
    Legacy,
    /// OBFS consolidated bundle (ultra-compressed)
    Obfs,
}

/// Final report structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalReport {
    /// Experiment metadata
    pub experiment_id: String,
    pub timestamp: DateTime<Utc>,
    pub config_summary: ConfigSummary,
    
    /// Performance summary
    pub performance: PerformanceMetricsSummary,
    
    /// Hall of Fame summary
    pub hall_of_fame: HallOfFameSummary,
    
    /// Evolution history
    pub evolution_history: EvolutionHistory,
    
    /// Overfitting assessment
    pub overfitting_assessment: OverfittingAssessment,
    
    /// Top candidates for production
    pub production_candidates: Vec<ProductionCandidate>,
    
    /// Warnings and recommendations
    pub warnings: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Summary of configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSummary {
    pub population_size: usize,
    pub generations: u32,
    pub blocks_used: Vec<String>,
    pub date_range: (String, String),
    pub validation_method: String,
    pub wfa_splits: usize,
    pub parallel_threads: usize,
}

/// Summary of Hall of Fame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HallOfFameSummary {
    pub total_entries: usize,
    pub validated_entries: usize,
    pub avg_oos_sharpe: f64,
    pub avg_pbo: f64,
    pub avg_degradation_pct: f64,
    pub best_oos_sharpe: f64,
    pub best_pbo: f64,
    pub best_genome_id: String,
}

/// Evolution history summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionHistory {
    pub total_generations: u32,
    pub total_genomes_evaluated: usize,
    pub total_validation_budget_used: usize,
    pub convergence_generation: Option<u32>,
    pub pareto_front_size_final: usize,
    pub diversity_final: f64,
    pub generation_metrics: Vec<GenerationMetric>,
}

/// Metrics for a single generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationMetric {
    pub generation: u32,
    pub best_is_sharpe: f64,
    pub best_oos_sharpe: f64,
    pub avg_is_sharpe: f64,
    pub pareto_size: usize,
    pub hof_additions: usize,
    pub stage_a_time_ms: f64,
    pub stage_b_time_ms: f64,
}

/// Overfitting risk assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverfittingAssessment {
    pub overall_risk: OverfittingRisk,
    pub avg_pbo: f64,
    pub avg_dsr: f64,
    pub avg_degradation: f64,
    pub candidates_above_threshold: usize,
    pub pbo_distribution: DistributionStats,
    pub degradation_distribution: DistributionStats,
    pub risk_factors: Vec<String>,
}

/// Risk level classification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OverfittingRisk {
    Low,
    Medium,
    High,
    Critical,
}

/// Distribution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DistributionStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    pub std: f64,
    pub p25: f64,
    pub p75: f64,
}

/// Production-ready candidate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductionCandidate {
    pub rank: usize,
    pub genome_id: String,
    pub genome_hash: String,
    pub oos_sharpe: f64,
    pub oos_cagr: f64,
    pub max_drawdown: f64,
    pub pbo: f64,
    pub dsr: f64,
    pub degradation_pct: f64,
    pub splits_passed: String, // e.g., "5/6"
    pub toml_path: String,
    pub validation_evidence_path: String,
    pub production_score: f64,
}

/// Final report generator
pub struct FinalReportGenerator {
    output_dir: PathBuf,
    experiment_id: String,
    config: EvolutionConfig,
    converter: GenomeConverter,
    format: ReportFormat,
}

impl FinalReportGenerator {
    /// Create a new report generator (legacy format by default)
    pub fn new(
        output_dir: impl Into<PathBuf>,
        experiment_id: impl Into<String>,
        config: EvolutionConfig,
    ) -> Self {
        Self {
            output_dir: output_dir.into(),
            experiment_id: experiment_id.into(),
            config,
            converter: GenomeConverter::new(),
            format: ReportFormat::Legacy,
        }
    }

    /// Set the output format (builder pattern)
    pub fn with_format(mut self, format: ReportFormat) -> Self {
        self.format = format;
        self
    }

    /// Generate the final report
    pub fn generate(
        &self,
        hof: &ValidatedHallOfFame,
        metrics: &PerformanceMetrics,
        snapshots: &[GenerationSnapshot],
    ) -> Result<FinalReport, ReportError> {
        let performance = metrics.summary();
        
        // Build Hall of Fame summary
        let hof_summary = self.build_hof_summary(hof);
        
        // Build evolution history
        let evolution_history = self.build_evolution_history(snapshots, metrics);
        
        // Build overfitting assessment
        let overfitting_assessment = self.build_overfitting_assessment(hof);
        
        // Build production candidates
        let production_candidates = self.build_production_candidates(hof)?;
        
        // Generate warnings and recommendations
        let (warnings, recommendations) = self.generate_insights(
            &performance, 
            &hof_summary, 
            &overfitting_assessment
        );

        let report = FinalReport {
            experiment_id: self.experiment_id.clone(),
            timestamp: Utc::now(),
            config_summary: self.build_config_summary(),
            performance,
            hall_of_fame: hof_summary,
            evolution_history,
            overfitting_assessment,
            production_candidates,
            warnings,
            recommendations,
        };

        Ok(report)
    }

    /// Generate and save the report to disk
    pub fn generate_and_save(
        &self,
        hof: &ValidatedHallOfFame,
        metrics: &PerformanceMetrics,
        snapshots: &[GenerationSnapshot],
    ) -> Result<PathBuf, ReportError> {
        match self.format {
            ReportFormat::Legacy => self.generate_and_save_legacy(hof, metrics, snapshots),
            ReportFormat::Obfs => self.generate_and_save_obfs(hof, metrics, snapshots),
        }
    }

    /// Generate and save in Legacy format (individual files)
    fn generate_and_save_legacy(
        &self,
        hof: &ValidatedHallOfFame,
        metrics: &PerformanceMetrics,
        snapshots: &[GenerationSnapshot],
    ) -> Result<PathBuf, ReportError> {
        let report = self.generate(hof, metrics, snapshots)?;
        
        let report_dir = self.output_dir.join("report");
        fs::create_dir_all(&report_dir)?;
        
        // Save main report
        let report_path = report_dir.join("final_report.json");
        let json = serde_json::to_string_pretty(&report)?;
        fs::write(&report_path, json)?;
        
        // Save production candidate TOMLs
        for candidate in &report.production_candidates {
            if let Some(entry) = hof.entries().iter().find(|e| e.genome_id.to_string() == candidate.genome_id) {
                self.save_production_candidate(&report_dir, entry)?;
            }
        }
        
        // Save snapshots
        let snapshots_path = report_dir.join("generation_snapshots.json");
        let snapshots_json = serde_json::to_string_pretty(snapshots)?;
        fs::write(snapshots_path, snapshots_json)?;
        
        Ok(report_path)
    }

    /// Generate and save in OBFS format (consolidated bundle)
    fn generate_and_save_obfs(
        &self,
        hof: &ValidatedHallOfFame,
        metrics: &PerformanceMetrics,
        snapshots: &[GenerationSnapshot],
    ) -> Result<PathBuf, ReportError> {
        let report = self.generate(hof, metrics, snapshots)?;
        
        let report_dir = self.output_dir.join("report");
        let bundle_dir = report_dir.join("obfs");
        fs::create_dir_all(&bundle_dir)?;

        // Write candidates to OBFS bundle
        let mut bundle_writer = obfs::ReportBundleWriter::new(&bundle_dir)
            .map_err(|e| ReportError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;

        for (rank, entry) in hof.entries().iter().enumerate() {
            // Generate TOML config
            let strategy_config = self.converter.to_strategy_config(&entry.genome)?;
            let toml_content = toml::to_string_pretty(&strategy_config)?;

            // Generate validation JSON
            let evidence = ValidationEvidence {
                genome_id: entry.genome_id.to_string(),
                genome_hash: format!("{:016x}", entry.genome_hash),
                validation: entry.validation.clone(),
                validated_generation: entry.validated_generation,
                rank: entry.rank,
                score: entry.score,
            };
            let validation_json = serde_json::to_string(&evidence)?;

            // Calculate production score
            let production_score = entry.validation.oos_sharpe_median 
                * (1.0 - entry.validation.pbo)
                * (1.0 - entry.validation.degradation_pct / 100.0);

            bundle_writer.add(
                entry.genome_id,
                entry.genome_hash,
                (rank + 1) as u32,
                entry.validated_generation,
                production_score,
                entry.validation.oos_sharpe_median,
                entry.validation.oos_cagr_median,
                entry.validation.oos_max_dd_worst,
                entry.validation.pbo,
                entry.validation.dsr,
                entry.validation.degradation_pct,
                entry.validation.splits_passed as u16,
                entry.validation.splits_evaluated as u16,
                &toml_content,
                &validation_json,
            ).map_err(|e| ReportError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
        }

        let stats = bundle_writer.finish()
            .map_err(|e| ReportError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;

        tracing::info!(
            "OBFS bundle written: {} candidates, {} bytes (level {})",
            stats.candidate_count,
            stats.data_file_size,
            stats.compression_level
        );

        // Also save main report as compressed OBFS
        let report_json = serde_json::to_vec(&report)?;
        let compressed_report = obfs::UltraCompressor::compress(&report_json)
            .map_err(|e| ReportError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
        fs::write(report_dir.join("final_report.obfs"), compressed_report)?;

        // Save snapshots as compressed OBFS
        let snapshots_json = serde_json::to_vec(snapshots)?;
        let compressed_snapshots = obfs::UltraCompressor::compress(&snapshots_json)
            .map_err(|e| ReportError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
        fs::write(report_dir.join("generation_snapshots.obfs"), compressed_snapshots)?;

        Ok(report_dir.join("final_report.obfs"))
    }

    fn build_config_summary(&self) -> ConfigSummary {
        // DESCONHECIDO: Date range should come from experiment manifest
        let date_range = (
            "N/A".to_string(),
            "N/A".to_string(),
        );

        // DESCONHECIDO: Validation config should be stored with experiment
        let validation_method = "Walk-Forward Analysis".to_string();
        let wfa_splits = 6; // Default

        ConfigSummary {
            population_size: self.config.population_size,
            generations: self.config.max_generations,
            blocks_used: vec![], // DESCONHECIDO: Block types not in current config
            date_range,
            validation_method,
            wfa_splits,
            parallel_threads: self.config.workers,
        }
    }

    fn build_hof_summary(&self, hof: &ValidatedHallOfFame) -> HallOfFameSummary {
        let hof_stats = hof.summary();
        
        let best_genome_id = hof.best()
            .map(|e| e.genome_id.to_string())
            .unwrap_or_default();

        HallOfFameSummary {
            total_entries: hof.len(),
            validated_entries: hof.len(), // All entries are validated
            avg_oos_sharpe: hof_stats.avg_oos_sharpe,
            avg_pbo: hof_stats.avg_pbo,
            avg_degradation_pct: hof_stats.avg_degradation_pct,
            best_oos_sharpe: hof_stats.best_oos_sharpe,
            best_pbo: hof_stats.best_pbo,
            best_genome_id,
        }
    }

    fn build_evolution_history(
        &self,
        snapshots: &[GenerationSnapshot],
        metrics: &PerformanceMetrics,
    ) -> EvolutionHistory {
        let summary = metrics.summary();
        
        let generation_metrics: Vec<GenerationMetric> = snapshots.iter().map(|s| {
            GenerationMetric {
                generation: s.generation,
                best_is_sharpe: 0.0, // DESCONHECIDO: Need to track this in snapshots
                best_oos_sharpe: 0.0, // DESCONHECIDO: Need to track this
                avg_is_sharpe: 0.0, // DESCONHECIDO: Need to track this
                pareto_size: 0, // DESCONHECIDO: Need to track this
                hof_additions: 0, // DESCONHECIDO: Need to track this
                stage_a_time_ms: s.stage_a_time_ms,
                stage_b_time_ms: s.stage_b_time_ms,
            }
        }).collect();

        EvolutionHistory {
            total_generations: summary.total_generations as u32,
            total_genomes_evaluated: summary.total_genomes_evaluated,
            total_validation_budget_used: summary.total_genomes_validated,
            convergence_generation: None, // DESCONHECIDO: Need convergence detection
            pareto_front_size_final: 0, // DESCONHECIDO: Need to track
            diversity_final: 0.0, // DESCONHECIDO: Need to calculate
            generation_metrics,
        }
    }

    fn build_overfitting_assessment(&self, hof: &ValidatedHallOfFame) -> OverfittingAssessment {
        let entries = hof.entries();
        
        if entries.is_empty() {
            return OverfittingAssessment {
                overall_risk: OverfittingRisk::Critical,
                avg_pbo: 1.0,
                avg_dsr: 0.0,
                avg_degradation: 100.0,
                candidates_above_threshold: 0,
                pbo_distribution: DistributionStats::default(),
                degradation_distribution: DistributionStats::default(),
                risk_factors: vec!["No validated candidates".to_string()],
            };
        }

        let pbos: Vec<f64> = entries.iter().map(|e| e.validation.pbo).collect();
        let degradations: Vec<f64> = entries.iter().map(|e| e.validation.degradation_pct).collect();
        let dsrs: Vec<f64> = entries.iter().map(|e| e.validation.dsr).collect();

        let avg_pbo = pbos.iter().sum::<f64>() / pbos.len() as f64;
        let avg_dsr = dsrs.iter().sum::<f64>() / dsrs.len() as f64;
        let avg_degradation = degradations.iter().sum::<f64>() / degradations.len() as f64;

        let pbo_distribution = self.compute_distribution(&pbos);
        let degradation_distribution = self.compute_distribution(&degradations);

        // Count candidates meeting institutional criteria
        let criteria = hof.criteria();
        let candidates_above_threshold = entries.iter()
            .filter(|e| e.validation.pbo < criteria.max_pbo && 
                       e.validation.oos_sharpe_median >= criteria.min_oos_sharpe)
            .count();

        // Classify overall risk
        let overall_risk = if avg_pbo > 0.5 || avg_degradation > 60.0 {
            OverfittingRisk::Critical
        } else if avg_pbo > 0.3 || avg_degradation > 40.0 {
            OverfittingRisk::High
        } else if avg_pbo > 0.15 || avg_degradation > 25.0 {
            OverfittingRisk::Medium
        } else {
            OverfittingRisk::Low
        };

        // Identify risk factors
        let mut risk_factors = Vec::new();
        if avg_pbo > criteria.max_pbo {
            risk_factors.push(format!("Average PBO ({:.2}) exceeds threshold ({:.2})", avg_pbo, criteria.max_pbo));
        }
        if avg_degradation > criteria.max_degradation_pct {
            risk_factors.push(format!("Average degradation ({:.1}%) exceeds threshold ({:.1}%)", avg_degradation, criteria.max_degradation_pct));
        }
        if pbo_distribution.std > 0.2 {
            risk_factors.push("High variance in PBO across candidates".to_string());
        }

        OverfittingAssessment {
            overall_risk,
            avg_pbo,
            avg_dsr,
            avg_degradation,
            candidates_above_threshold,
            pbo_distribution,
            degradation_distribution,
            risk_factors,
        }
    }

    fn compute_distribution(&self, values: &[f64]) -> DistributionStats {
        if values.is_empty() {
            return DistributionStats::default();
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let mean = sorted.iter().sum::<f64>() / n as f64;
        let variance: f64 = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std = variance.sqrt();

        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        let p25_idx = (n as f64 * 0.25) as usize;
        let p75_idx = (n as f64 * 0.75) as usize;

        DistributionStats {
            min: sorted[0],
            max: sorted[n - 1],
            mean,
            median,
            std,
            p25: sorted[p25_idx.min(n - 1)],
            p75: sorted[p75_idx.min(n - 1)],
        }
    }

    fn build_production_candidates(&self, hof: &ValidatedHallOfFame) -> Result<Vec<ProductionCandidate>, ReportError> {
        let mut candidates = Vec::new();

        for (rank, entry) in hof.entries().iter().enumerate() {
            // Calculate production score (emphasizes robustness)
            let production_score = entry.validation.oos_sharpe_median 
                * (1.0 - entry.validation.pbo)
                * (1.0 - entry.validation.degradation_pct / 100.0);

            let toml_path = format!("candidates/{}.toml", entry.genome_id);
            let evidence_path = format!("candidates/{}_validation.json", entry.genome_id);

            candidates.push(ProductionCandidate {
                rank: rank + 1,
                genome_id: entry.genome_id.to_string(),
                genome_hash: format!("{:016x}", entry.genome_hash),
                oos_sharpe: entry.validation.oos_sharpe_median,
                oos_cagr: entry.validation.oos_cagr_median,
                max_drawdown: entry.validation.oos_max_dd_worst,
                pbo: entry.validation.pbo,
                dsr: entry.validation.dsr,
                degradation_pct: entry.validation.degradation_pct,
                splits_passed: format!("{}/{}", entry.validation.splits_passed, entry.validation.splits_evaluated),
                toml_path,
                validation_evidence_path: evidence_path,
                production_score,
            });
        }

        // Sort by production score
        candidates.sort_by(|a, b| b.production_score.partial_cmp(&a.production_score).unwrap_or(std::cmp::Ordering::Equal));

        Ok(candidates)
    }

    fn save_production_candidate(&self, report_dir: &Path, entry: &ValidatedHofEntry) -> Result<(), ReportError> {
        let candidates_dir = report_dir.join("candidates");
        fs::create_dir_all(&candidates_dir)?;

        // Save TOML
        let strategy_config = self.converter.to_strategy_config(&entry.genome)?;
        let toml_content = toml::to_string_pretty(&strategy_config)?;
        let toml_path = candidates_dir.join(format!("{}.toml", entry.genome_id));
        fs::write(&toml_path, toml_content)?;

        // Save validation evidence
        let evidence = ValidationEvidence {
            genome_id: entry.genome_id.to_string(),
            genome_hash: format!("{:016x}", entry.genome_hash),
            validation: entry.validation.clone(),
            validated_generation: entry.validated_generation,
            rank: entry.rank,
            score: entry.score,
        };
        let evidence_json = serde_json::to_string_pretty(&evidence)?;
        let evidence_path = candidates_dir.join(format!("{}_validation.json", entry.genome_id));
        fs::write(&evidence_path, evidence_json)?;

        Ok(())
    }

    fn generate_insights(
        &self,
        performance: &PerformanceMetricsSummary,
        hof: &HallOfFameSummary,
        overfitting: &OverfittingAssessment,
    ) -> (Vec<String>, Vec<String>) {
        let mut warnings = Vec::new();
        let mut recommendations = Vec::new();

        // Performance warnings
        if performance.stage_a_cache_hit_rate < 30.0 {
            warnings.push(format!(
                "Low cache hit rate ({:.1}%). Consider increasing population diversity or adjusting mutation rates.",
                performance.stage_a_cache_hit_rate
            ));
        }

        if performance.early_exit_rate > 50.0 {
            warnings.push(format!(
                "High early exit rate ({:.1}%). Many candidates fail validation early.",
                performance.early_exit_rate
            ));
            recommendations.push("Consider loosening initial screening criteria to explore more candidates.".to_string());
        }

        // HoF warnings
        if hof.total_entries == 0 {
            warnings.push("No candidates met institutional criteria. Experiment may need more generations or relaxed criteria.".to_string());
        } else if hof.validated_entries < 5 {
            warnings.push(format!(
                "Only {} validated candidates. Consider running more generations or increasing population size.",
                hof.validated_entries
            ));
        }

        // Overfitting warnings
        match overfitting.overall_risk {
            OverfittingRisk::Critical => {
                warnings.push("CRITICAL: Very high overfitting risk detected. Results may not generalize.".to_string());
                recommendations.push("Consider using more conservative parameter ranges and longer OOS periods.".to_string());
            }
            OverfittingRisk::High => {
                warnings.push("HIGH: Significant overfitting risk. Review candidates carefully.".to_string());
                recommendations.push("Use paper trading validation before live deployment.".to_string());
            }
            OverfittingRisk::Medium => {
                recommendations.push("Consider additional out-of-sample validation on unseen data.".to_string());
            }
            OverfittingRisk::Low => {
                recommendations.push("Candidates show good robustness. Proceed with paper trading validation.".to_string());
            }
        }

        // Add risk factors as warnings
        for factor in &overfitting.risk_factors {
            warnings.push(factor.clone());
        }

        // General recommendations
        if performance.throughput_genomes_per_sec < 10.0 {
            recommendations.push("Consider enabling --ultra mode for faster evaluation.".to_string());
        }

        if hof.avg_degradation_pct > 30.0 {
            recommendations.push("High IS->OOS degradation suggests overfitting. Consider simpler strategies.".to_string());
        }

        (warnings, recommendations)
    }
}

/// Validation evidence for audit trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationEvidence {
    pub genome_id: String,
    pub genome_hash: String,
    pub validation: crate::hall_of_fame_validated::ValidationResultSummary,
    pub validated_generation: u32,
    pub rank: usize,
    pub score: f64,
}

/// Report generation errors
#[derive(Debug, thiserror::Error)]
pub enum ReportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
    
    #[error("TOML serialization error: {0}")]
    Toml(#[from] toml::ser::Error),
    
    #[error("Genome conversion error: {0}")]
    Conversion(#[from] ConversionError),
    
    #[error("OBFS error: {0}")]
    Obfs(String),

    #[error("Not found: {0}")]
    NotFound(String),
}

// ============================================================================
// Report Reader (On-demand decompression)
// ============================================================================

/// Reader for loading reports from disk (supports both Legacy and OBFS formats)
pub struct ReportReader {
    report_dir: PathBuf,
    format: ReportFormat,
    bundle_reader: Option<obfs::ReportBundleReader>,
}

impl ReportReader {
    /// Open a report directory (auto-detects format)
    pub fn open(report_dir: impl Into<PathBuf>) -> Result<Self, ReportError> {
        let report_dir = report_dir.into();
        
        // Check if OBFS bundle exists
        let obfs_bundle_dir = report_dir.join("obfs");
        let has_obfs = obfs_bundle_dir.exists() && obfs_bundle_dir.join("candidates.obfs").exists();
        
        let (format, bundle_reader) = if has_obfs {
            let reader = obfs::ReportBundleReader::open(&obfs_bundle_dir)
                .map_err(|e| ReportError::Obfs(e.to_string()))?;
            (ReportFormat::Obfs, Some(reader))
        } else {
            (ReportFormat::Legacy, None)
        };

        Ok(Self {
            report_dir,
            format,
            bundle_reader,
        })
    }

    /// Get the detected format
    pub fn format(&self) -> ReportFormat {
        self.format
    }

    /// Load the final report
    pub fn load_final_report(&self) -> Result<FinalReport, ReportError> {
        match self.format {
            ReportFormat::Legacy => {
                let path = self.report_dir.join("final_report.json");
                let content = fs::read_to_string(&path)?;
                Ok(serde_json::from_str(&content)?)
            }
            ReportFormat::Obfs => {
                let path = self.report_dir.join("final_report.obfs");
                let compressed = fs::read(&path)?;
                let decompressed = obfs::UltraCompressor::decompress(&compressed)
                    .map_err(|e| ReportError::Obfs(e.to_string()))?;
                Ok(serde_json::from_slice(&decompressed)?)
            }
        }
    }

    /// Load generation snapshots
    pub fn load_snapshots(&self) -> Result<Vec<GenerationSnapshot>, ReportError> {
        match self.format {
            ReportFormat::Legacy => {
                let path = self.report_dir.join("generation_snapshots.json");
                let content = fs::read_to_string(&path)?;
                Ok(serde_json::from_str(&content)?)
            }
            ReportFormat::Obfs => {
                let path = self.report_dir.join("generation_snapshots.obfs");
                let compressed = fs::read(&path)?;
                let decompressed = obfs::UltraCompressor::decompress(&compressed)
                    .map_err(|e| ReportError::Obfs(e.to_string()))?;
                Ok(serde_json::from_slice(&decompressed)?)
            }
        }
    }

    /// Get config TOML for a candidate by UUID
    pub fn get_candidate_config(&self, uuid: uuid::Uuid) -> Result<String, ReportError> {
        match self.format {
            ReportFormat::Legacy => {
                let path = self.report_dir.join("candidates").join(format!("{}.toml", uuid));
                if path.exists() {
                    Ok(fs::read_to_string(&path)?)
                } else {
                    Err(ReportError::NotFound(format!("Candidate config: {}", uuid)))
                }
            }
            ReportFormat::Obfs => {
                if let Some(ref reader) = self.bundle_reader {
                    reader.get_config(uuid)
                        .map_err(|e| ReportError::Obfs(e.to_string()))?
                        .ok_or_else(|| ReportError::NotFound(format!("Candidate config: {}", uuid)))
                } else {
                    Err(ReportError::Obfs("Bundle reader not initialized".to_string()))
                }
            }
        }
    }

    /// Get validation JSON for a candidate by UUID
    pub fn get_candidate_validation(&self, uuid: uuid::Uuid) -> Result<String, ReportError> {
        match self.format {
            ReportFormat::Legacy => {
                let path = self.report_dir.join("candidates").join(format!("{}_validation.json", uuid));
                if path.exists() {
                    Ok(fs::read_to_string(&path)?)
                } else {
                    Err(ReportError::NotFound(format!("Candidate validation: {}", uuid)))
                }
            }
            ReportFormat::Obfs => {
                if let Some(ref reader) = self.bundle_reader {
                    reader.get_validation(uuid)
                        .map_err(|e| ReportError::Obfs(e.to_string()))?
                        .ok_or_else(|| ReportError::NotFound(format!("Candidate validation: {}", uuid)))
                } else {
                    Err(ReportError::Obfs("Bundle reader not initialized".to_string()))
                }
            }
        }
    }

    /// List all candidate UUIDs
    pub fn list_candidates(&self) -> Result<Vec<uuid::Uuid>, ReportError> {
        match self.format {
            ReportFormat::Legacy => {
                let candidates_dir = self.report_dir.join("candidates");
                if !candidates_dir.exists() {
                    return Ok(Vec::new());
                }
                
                let mut uuids = Vec::new();
                for entry in fs::read_dir(&candidates_dir)? {
                    let entry = entry?;
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    if name_str.ends_with(".toml") {
                        if let Some(uuid_str) = name_str.strip_suffix(".toml") {
                            if let Ok(uuid) = uuid::Uuid::parse_str(uuid_str) {
                                uuids.push(uuid);
                            }
                        }
                    }
                }
                Ok(uuids)
            }
            ReportFormat::Obfs => {
                if let Some(ref reader) = self.bundle_reader {
                    reader.list()
                        .map_err(|e| ReportError::Obfs(e.to_string()))
                } else {
                    Err(ReportError::Obfs("Bundle reader not initialized".to_string()))
                }
            }
        }
    }

    /// Get bundle statistics (OBFS only)
    pub fn bundle_stats(&self) -> Option<obfs::BundleStats> {
        self.bundle_reader.as_ref().and_then(|r| r.stats().ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_config_summary() {
        let config = EvolutionConfig::default();
        let generator = FinalReportGenerator::new(
            tempdir().unwrap().path(),
            "test-exp",
            config,
        );

        let summary = generator.build_config_summary();
        assert!(summary.population_size > 0);
    }

    #[test]
    fn test_distribution_stats() {
        let config = EvolutionConfig::default();
        let generator = FinalReportGenerator::new(
            tempdir().unwrap().path(),
            "test-exp",
            config,
        );

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = generator.compute_distribution(&values);

        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.median, 3.0);
    }

    #[test]
    fn test_empty_distribution() {
        let config = EvolutionConfig::default();
        let generator = FinalReportGenerator::new(
            tempdir().unwrap().path(),
            "test-exp",
            config,
        );

        let stats = generator.compute_distribution(&[]);
        assert_eq!(stats.mean, 0.0);
    }
}

