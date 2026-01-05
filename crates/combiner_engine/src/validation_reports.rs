//! Validation Report Export
//!
//! Standardized export for anti-overfitting validation results.
//! Supports two output formats:
//! - Legacy JSON: Individual files (wfa_report.json, pbo_dsr.json, stress_report.json)
//! - OBFS: Consolidated ultra-compressed binary bundle (~10x compression)
//!
//! These files provide evidence for Marco 3 (Validation) and Marco 4 (Promotion Gates).

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use zstd::stream::{Decoder, Encoder};

use crate::validation::{WfaResult, CpcvResult, PboDsrResult};
use backtester_execution::StressSuiteResult;

/// Ultra compression level (Zstd max with LDM)
const ULTRA_COMPRESSION_LEVEL: i32 = 19;

// =============================================================================
// WFA REPORT
// =============================================================================

/// Standardized Walk-Forward Analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WfaReport {
    /// Report metadata
    pub metadata: ReportMetadata,
    /// Genome identification
    pub genome_id: String,
    /// Overall verdict
    pub verdict: ValidationVerdict,
    /// Summary metrics
    pub summary: WfaSummary,
    /// Detailed window results
    pub windows: Vec<WfaWindowResult>,
    /// Thresholds used for validation
    pub thresholds: WfaThresholds,
}

/// Summary of WFA results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WfaSummary {
    /// Number of windows evaluated
    pub windows_evaluated: usize,
    /// In-sample Sharpe (gross)
    pub is_sharpe_gross: f64,
    /// In-sample Sharpe (net of costs)
    pub is_sharpe_net: f64,
    /// Out-of-sample Sharpe (gross)
    pub oos_sharpe_gross: f64,
    /// Out-of-sample Sharpe (net of costs)
    pub oos_sharpe_net: f64,
    /// IS → OOS degradation percentage
    pub degradation_pct: f64,
    /// In-sample CAGR (net)
    pub is_cagr_net: f64,
    /// Out-of-sample CAGR (net)
    pub oos_cagr_net: f64,
}

/// Results for a single WFA window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WfaWindowResult {
    /// Window index (0-based)
    pub window_idx: usize,
    /// In-sample Sharpe
    pub is_sharpe: f64,
    /// Out-of-sample Sharpe
    pub oos_sharpe: f64,
    /// In-sample CAGR
    pub is_cagr: f64,
    /// Out-of-sample CAGR
    pub oos_cagr: f64,
}

/// Thresholds used for WFA validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WfaThresholds {
    /// Maximum allowed degradation (e.g., 0.40 = 40%)
    pub max_degradation: f64,
    /// Minimum OOS Sharpe (net)
    pub min_oos_sharpe_net: f64,
    /// Maximum OOS drawdown
    pub max_oos_drawdown: f64,
    /// Minimum OOS trades
    pub min_oos_trades: u32,
}

impl WfaReport {
    /// Create from WfaResult.
    pub fn from_result(result: &WfaResult, thresholds: WfaThresholds) -> Self {
        let verdict = if result.passed {
            ValidationVerdict::Pass
        } else {
            ValidationVerdict::Fail
        };

        Self {
            metadata: ReportMetadata::new("wfa_report", "1.0"),
            genome_id: result.genome_id.to_string(),
            verdict,
            summary: WfaSummary {
                windows_evaluated: result.windows_evaluated,
                is_sharpe_gross: result.is_sharpe_gross,
                is_sharpe_net: result.is_sharpe_net,
                oos_sharpe_gross: result.oos_sharpe_gross,
                oos_sharpe_net: result.oos_sharpe_net,
                degradation_pct: result.degradation_pct,
                is_cagr_net: result.is_cagr_net,
                oos_cagr_net: result.oos_cagr_net,
            },
            windows: result
                .window_details
                .iter()
                .map(|w| WfaWindowResult {
                    window_idx: w.window_idx,
                    is_sharpe: w.is_sharpe,
                    oos_sharpe: w.oos_sharpe,
                    is_cagr: w.is_cagr,
                    oos_cagr: w.oos_cagr,
                })
                .collect(),
            thresholds,
        }
    }

    /// Write to JSON file.
    pub fn write_json(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let mut file = fs::File::create(path)?;
        file.write_all(json.as_bytes())
    }

    /// Write to OBFS file (ultra-compressed).
    pub fn write_obfs(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_vec(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let compressed = ultra_compress(&json)?;
        fs::write(path, compressed)
    }

    /// Read from OBFS file.
    pub fn read_obfs(path: &Path) -> std::io::Result<Self> {
        let compressed = fs::read(path)?;
        let decompressed = ultra_decompress(&compressed)?;
        serde_json::from_slice(&decompressed)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

// =============================================================================
// PBO/DSR REPORT
// =============================================================================

/// Standardized PBO/DSR report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PboDsrReport {
    /// Report metadata
    pub metadata: ReportMetadata,
    /// Genome identification
    pub genome_id: String,
    /// Overall verdict
    pub verdict: ValidationVerdict,
    /// PBO/DSR metrics
    pub metrics: PboDsrMetrics,
    /// CPCV fold results (if available)
    pub cpcv_folds: Option<CpcvFoldsResult>,
    /// Thresholds used
    pub thresholds: PboDsrThresholds,
    /// Warnings
    pub warnings: Vec<String>,
}

/// PBO/DSR metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PboDsrMetrics {
    /// In-sample Sharpe (net)
    pub is_sharpe_net: f64,
    /// Probability of Backtest Overfitting
    pub pbo: f64,
    /// Deflated Sharpe Ratio
    pub dsr: f64,
    /// Total trials used for DSR calculation
    pub total_trials: u64,
    /// PBO percentile interpretation
    pub pbo_interpretation: String,
    /// DSR interpretation
    pub dsr_interpretation: String,
}

/// CPCV fold results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpcvFoldsResult {
    /// Number of folds
    pub num_folds: usize,
    /// OOS Sharpe for each fold
    pub fold_sharpes: Vec<f64>,
    /// Mean OOS Sharpe
    pub mean_sharpe: f64,
    /// Std deviation of OOS Sharpe
    pub std_sharpe: f64,
}

/// Thresholds for PBO/DSR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PboDsrThresholds {
    /// Maximum allowed PBO
    pub max_pbo: f64,
    /// Minimum DSR for warning
    pub min_dsr: f64,
}

impl PboDsrReport {
    /// Create from PboDsrResult and optional CpcvResult.
    pub fn from_results(
        pbo_result: &PboDsrResult,
        cpcv_result: Option<&CpcvResult>,
        thresholds: PboDsrThresholds,
    ) -> Self {
        let verdict = if pbo_result.passed {
            ValidationVerdict::Pass
        } else {
            ValidationVerdict::Fail
        };

        let pbo_interpretation = if pbo_result.pbo <= 0.05 {
            "Excellent - very low overfitting risk".to_string()
        } else if pbo_result.pbo <= 0.15 {
            "Good - acceptable overfitting risk".to_string()
        } else if pbo_result.pbo <= 0.30 {
            "Moderate - some overfitting risk".to_string()
        } else {
            "High - significant overfitting risk".to_string()
        };

        let dsr_interpretation = if pbo_result.dsr >= 2.0 {
            "Strong - statistically significant".to_string()
        } else if pbo_result.dsr >= 1.0 {
            "Moderate - marginally significant".to_string()
        } else if pbo_result.dsr >= 0.5 {
            "Weak - low statistical significance".to_string()
        } else {
            "Very weak - likely noise".to_string()
        };

        let mut warnings = Vec::new();
        if pbo_result.pbo > thresholds.max_pbo {
            warnings.push(format!(
                "PBO ({:.2}) exceeds threshold ({:.2})",
                pbo_result.pbo, thresholds.max_pbo
            ));
        }
        if pbo_result.dsr < thresholds.min_dsr {
            warnings.push(format!(
                "DSR ({:.2}) below threshold ({:.2})",
                pbo_result.dsr, thresholds.min_dsr
            ));
        }

        let cpcv_folds = cpcv_result.map(|c| CpcvFoldsResult {
            num_folds: c.oos_sharpes.len(),
            fold_sharpes: c.oos_sharpes.clone(),
            mean_sharpe: c.oos_sharpe_mean,
            std_sharpe: c.oos_sharpe_std,
        });

        Self {
            metadata: ReportMetadata::new("pbo_dsr", "1.0"),
            genome_id: pbo_result.genome_id.to_string(),
            verdict,
            metrics: PboDsrMetrics {
                is_sharpe_net: pbo_result.is_sharpe_net,
                pbo: pbo_result.pbo,
                dsr: pbo_result.dsr,
                total_trials: pbo_result.total_trials,
                pbo_interpretation,
                dsr_interpretation,
            },
            cpcv_folds,
            thresholds,
            warnings,
        }
    }

    /// Write to JSON file.
    pub fn write_json(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let mut file = fs::File::create(path)?;
        file.write_all(json.as_bytes())
    }

    /// Write to OBFS file (ultra-compressed).
    pub fn write_obfs(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_vec(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let compressed = ultra_compress(&json)?;
        fs::write(path, compressed)
    }

    /// Read from OBFS file.
    pub fn read_obfs(path: &Path) -> std::io::Result<Self> {
        let compressed = fs::read(path)?;
        let decompressed = ultra_decompress(&compressed)?;
        serde_json::from_slice(&decompressed)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

// =============================================================================
// STRESS REPORT
// =============================================================================

/// Standardized stress test report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressReport {
    /// Report metadata
    pub metadata: ReportMetadata,
    /// Genome identification
    pub genome_id: String,
    /// Overall verdict
    pub verdict: ValidationVerdict,
    /// Summary statistics
    pub summary: StressSummary,
    /// Individual scenario results
    pub scenarios: Vec<StressScenarioDetail>,
    /// Thresholds used
    pub thresholds: StressThresholds,
}

/// Summary of stress test results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressSummary {
    /// Total scenarios run
    pub total_scenarios: usize,
    /// Scenarios passed
    pub passed_count: usize,
    /// Scenarios failed
    pub failed_count: usize,
    /// Pass rate percentage
    pub pass_rate_pct: f64,
    /// Average stressed Sharpe
    pub avg_stressed_sharpe: f64,
    /// Worst stressed Sharpe
    pub worst_stressed_sharpe: f64,
}

/// Detail for a single stress scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressScenarioDetail {
    /// Scenario ID (e.g., "S1", "S2")
    pub scenario_id: String,
    /// Scenario name
    pub name: String,
    /// Description of the stress
    pub description: String,
    /// Whether scenario passed
    pub passed: bool,
    /// Baseline Sharpe (no stress)
    pub baseline_sharpe: f64,
    /// Stressed Sharpe
    pub stressed_sharpe: f64,
    /// Sharpe degradation percentage
    pub sharpe_degradation_pct: f64,
    /// Reason if failed
    pub failure_reason: Option<String>,
}

/// Thresholds for stress testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressThresholds {
    /// Minimum scenarios to pass (out of total)
    pub min_pass_count: usize,
    /// Total scenarios
    pub total_scenarios: usize,
    /// Minimum stressed Sharpe to pass
    pub min_stressed_sharpe: f64,
}

impl StressReport {
    /// Create from StressSuiteResult.
    pub fn from_result(
        genome_id: Uuid,
        baseline_sharpe: f64,
        result: &StressSuiteResult,
        thresholds: StressThresholds,
    ) -> Self {
        let verdict = if result.passed_count >= thresholds.min_pass_count {
            ValidationVerdict::Pass
        } else {
            ValidationVerdict::Fail
        };

        let stressed_sharpes: Vec<f64> = result
            .results
            .iter()
            .filter_map(|r| {
                if r.passed {
                    Some(r.sharpe_stressed)
                } else {
                    None
                }
            })
            .collect();

        let avg_stressed_sharpe = if stressed_sharpes.is_empty() {
            0.0
        } else {
            stressed_sharpes.iter().sum::<f64>() / stressed_sharpes.len() as f64
        };

        let worst_stressed_sharpe = stressed_sharpes
            .iter()
            .cloned()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let scenarios: Vec<StressScenarioDetail> = result
            .results
            .iter()
            .map(|r| {
                let degradation = if baseline_sharpe != 0.0 {
                    (baseline_sharpe - r.sharpe_stressed) / baseline_sharpe.abs() * 100.0
                } else {
                    0.0
                };

                StressScenarioDetail {
                    scenario_id: r.scenario_id.clone(),
                    name: r.scenario_name.clone(),
                    description: format!("Stress scenario: {}", r.scenario_name),
                    passed: r.passed,
                    baseline_sharpe,
                    stressed_sharpe: r.sharpe_stressed,
                    sharpe_degradation_pct: degradation,
                    failure_reason: if r.passed {
                        None
                    } else {
                        Some(r.failure_reason.clone().unwrap_or_else(|| "Sharpe below threshold".to_string()))
                    },
                }
            })
            .collect();

        Self {
            metadata: ReportMetadata::new("stress_report", "1.0"),
            genome_id: genome_id.to_string(),
            verdict,
            summary: StressSummary {
                total_scenarios: result.total_count,
                passed_count: result.passed_count,
                failed_count: result.total_count - result.passed_count,
                pass_rate_pct: (result.passed_count as f64 / result.total_count as f64) * 100.0,
                avg_stressed_sharpe,
                worst_stressed_sharpe,
            },
            scenarios,
            thresholds,
        }
    }

    /// Write to JSON file.
    pub fn write_json(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let mut file = fs::File::create(path)?;
        file.write_all(json.as_bytes())
    }

    /// Write to OBFS file (ultra-compressed).
    pub fn write_obfs(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_vec(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let compressed = ultra_compress(&json)?;
        fs::write(path, compressed)
    }

    /// Read from OBFS file.
    pub fn read_obfs(path: &Path) -> std::io::Result<Self> {
        let compressed = fs::read(path)?;
        let decompressed = ultra_decompress(&compressed)?;
        serde_json::from_slice(&decompressed)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

// =============================================================================
// COMMON TYPES
// =============================================================================

/// Report metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    /// Report type
    pub report_type: String,
    /// Schema version
    pub schema_version: String,
    /// Generation timestamp
    pub generated_at: DateTime<Utc>,
    /// Generator
    pub generator: String,
}

impl ReportMetadata {
    /// Create new metadata.
    pub fn new(report_type: &str, schema_version: &str) -> Self {
        Self {
            report_type: report_type.to_string(),
            schema_version: schema_version.to_string(),
            generated_at: Utc::now(),
            generator: "combiner_engine".to_string(),
        }
    }
}

/// Validation verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum ValidationVerdict {
    /// Passed validation
    Pass,
    /// Failed validation
    Fail,
    /// Passed with warnings
    Warn,
}

// =============================================================================
// COMBINED VALIDATION BUNDLE
// =============================================================================

/// Complete validation bundle for a candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationBundle {
    /// Genome ID
    pub genome_id: String,
    /// Overall verdict
    pub overall_verdict: ValidationVerdict,
    /// WFA report
    pub wfa: Option<WfaReport>,
    /// PBO/DSR report
    pub pbo_dsr: Option<PboDsrReport>,
    /// Stress report
    pub stress: Option<StressReport>,
    /// Summary
    pub summary: ValidationBundleSummary,
}

/// Summary of validation bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationBundleSummary {
    /// WFA verdict
    pub wfa_verdict: Option<ValidationVerdict>,
    /// PBO/DSR verdict
    pub pbo_dsr_verdict: Option<ValidationVerdict>,
    /// Stress verdict
    pub stress_verdict: Option<ValidationVerdict>,
    /// Key metrics
    pub oos_sharpe_net: Option<f64>,
    pub pbo: Option<f64>,
    pub dsr: Option<f64>,
    pub stress_pass_rate: Option<f64>,
}

impl ValidationBundle {
    /// Create a new bundle.
    pub fn new(genome_id: Uuid) -> Self {
        Self {
            genome_id: genome_id.to_string(),
            overall_verdict: ValidationVerdict::Pass,
            wfa: None,
            pbo_dsr: None,
            stress: None,
            summary: ValidationBundleSummary {
                wfa_verdict: None,
                pbo_dsr_verdict: None,
                stress_verdict: None,
                oos_sharpe_net: None,
                pbo: None,
                dsr: None,
                stress_pass_rate: None,
            },
        }
    }

    /// Add WFA report.
    pub fn with_wfa(mut self, wfa: WfaReport) -> Self {
        self.summary.wfa_verdict = Some(wfa.verdict);
        self.summary.oos_sharpe_net = Some(wfa.summary.oos_sharpe_net);
        self.wfa = Some(wfa);
        self.update_overall_verdict();
        self
    }

    /// Add PBO/DSR report.
    pub fn with_pbo_dsr(mut self, pbo_dsr: PboDsrReport) -> Self {
        self.summary.pbo_dsr_verdict = Some(pbo_dsr.verdict);
        self.summary.pbo = Some(pbo_dsr.metrics.pbo);
        self.summary.dsr = Some(pbo_dsr.metrics.dsr);
        self.pbo_dsr = Some(pbo_dsr);
        self.update_overall_verdict();
        self
    }

    /// Add stress report.
    pub fn with_stress(mut self, stress: StressReport) -> Self {
        self.summary.stress_verdict = Some(stress.verdict);
        self.summary.stress_pass_rate = Some(stress.summary.pass_rate_pct);
        self.stress = Some(stress);
        self.update_overall_verdict();
        self
    }

    /// Update overall verdict based on component verdicts.
    fn update_overall_verdict(&mut self) {
        let verdicts = [
            self.summary.wfa_verdict,
            self.summary.pbo_dsr_verdict,
            self.summary.stress_verdict,
        ];

        // If any is Fail, overall is Fail
        if verdicts.iter().any(|v| *v == Some(ValidationVerdict::Fail)) {
            self.overall_verdict = ValidationVerdict::Fail;
        } else if verdicts.iter().any(|v| *v == Some(ValidationVerdict::Warn)) {
            self.overall_verdict = ValidationVerdict::Warn;
        } else {
            self.overall_verdict = ValidationVerdict::Pass;
        }
    }

    /// Write all reports to a directory (Legacy JSON format).
    pub fn write_to_dir(&self, dir: &Path) -> std::io::Result<()> {
        fs::create_dir_all(dir)?;

        if let Some(ref wfa) = self.wfa {
            wfa.write_json(&dir.join("wfa_report.json"))?;
        }

        if let Some(ref pbo_dsr) = self.pbo_dsr {
            pbo_dsr.write_json(&dir.join("pbo_dsr.json"))?;
        }

        if let Some(ref stress) = self.stress {
            stress.write_json(&dir.join("stress_report.json"))?;
        }

        // Write summary bundle
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let mut file = fs::File::create(dir.join("validation_bundle.json"))?;
        file.write_all(json.as_bytes())?;

        Ok(())
    }

    /// Write all reports as single consolidated OBFS bundle (~10x compression).
    /// Creates: validation.obfs (single file containing all reports)
    pub fn write_obfs_bundle(&self, dir: &Path) -> std::io::Result<()> {
        fs::create_dir_all(dir)?;

        // Serialize entire bundle to JSON
        let json = serde_json::to_vec(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Ultra-compress with Zstd level 19 + LDM
        let compressed = ultra_compress(&json)?;

        // Write single file
        let path = dir.join("validation.obfs");
        fs::write(&path, &compressed)?;

        tracing::debug!(
            "Validation bundle written: {} bytes -> {} bytes ({:.1}x)",
            json.len(),
            compressed.len(),
            json.len() as f64 / compressed.len() as f64
        );

        Ok(())
    }

    /// Read from OBFS bundle.
    pub fn read_obfs_bundle(path: &Path) -> std::io::Result<Self> {
        let compressed = fs::read(path)?;
        let decompressed = ultra_decompress(&compressed)?;
        serde_json::from_slice(&decompressed)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

// =============================================================================
// OBFS COMPRESSION HELPERS
// =============================================================================

/// Ultra-compress data using Zstd level 19 with LDM and checksum.
fn ultra_compress(data: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut encoder = Encoder::new(Vec::new(), ULTRA_COMPRESSION_LEVEL)?;
    encoder.include_checksum(true)?;
    encoder.long_distance_matching(true)?;
    encoder.write_all(data)?;
    encoder.finish()
}

/// Decompress ultra-compressed data.
fn ultra_decompress(compressed: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut decoder = Decoder::new(compressed)?;
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;
    Ok(decompressed)
}

// =============================================================================
// VALIDATION REPORT BUNDLE (OBFS Native)
// =============================================================================

/// rkyv-compatible entry for consolidated validation storage.
#[derive(Archive, RkyvSerialize, RkyvDeserialize, Debug, Clone)]
#[rkyv(derive(Debug))]
pub struct ValidationReportEntry {
    /// UUID as bytes
    pub uuid_bytes: [u8; 16],
    /// WFA passed
    pub wfa_passed: bool,
    /// OOS Sharpe (net)
    pub oos_sharpe_net: f64,
    /// Degradation percentage
    pub degradation_pct: f64,
    /// PBO value
    pub pbo: f64,
    /// DSR value
    pub dsr: f64,
    /// Stress pass rate
    pub stress_pass_rate: f64,
    /// Overall verdict (0=Fail, 1=Pass, 2=Warn)
    pub verdict: u8,
}

impl ValidationReportEntry {
    pub fn uuid(&self) -> Uuid {
        Uuid::from_bytes(self.uuid_bytes)
    }

    pub fn from_bundle(bundle: &ValidationBundle) -> Option<Self> {
        let uuid = Uuid::parse_str(&bundle.genome_id).ok()?;
        Some(Self {
            uuid_bytes: *uuid.as_bytes(),
            wfa_passed: bundle.summary.wfa_verdict == Some(ValidationVerdict::Pass),
            oos_sharpe_net: bundle.summary.oos_sharpe_net.unwrap_or(0.0),
            degradation_pct: bundle.wfa.as_ref().map(|w| w.summary.degradation_pct).unwrap_or(0.0),
            pbo: bundle.summary.pbo.unwrap_or(1.0),
            dsr: bundle.summary.dsr.unwrap_or(0.0),
            stress_pass_rate: bundle.summary.stress_pass_rate.unwrap_or(0.0),
            verdict: match bundle.overall_verdict {
                ValidationVerdict::Fail => 0,
                ValidationVerdict::Pass => 1,
                ValidationVerdict::Warn => 2,
            },
        })
    }
}

/// Location of data within the bundle file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleLocation {
    pub offset: u64,
    pub compressed_size: u32,
    pub original_size: u32,
}

/// Writer for consolidated validation report bundles.
pub struct ValidationBundleWriter {
    root_path: PathBuf,
    data_file: Option<File>,
    entries: Vec<(Uuid, BundleLocation)>,
    written_count: u64,
}

impl ValidationBundleWriter {
    /// Create a new bundle writer.
    pub fn new(root_path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let root_path = root_path.into();
        fs::create_dir_all(&root_path)?;

        Ok(Self {
            root_path,
            data_file: None,
            entries: Vec::new(),
            written_count: 0,
        })
    }

    fn data_file_path(&self) -> PathBuf {
        self.root_path.join("validations.obfs")
    }

    fn ensure_data_file(&mut self) -> std::io::Result<&mut File> {
        if self.data_file.is_none() {
            let file = fs::OpenOptions::new()
                .create(true)
                .read(true)
                .append(true)
                .open(self.data_file_path())?;
            self.data_file = Some(file);
        }
        Ok(self.data_file.as_mut().unwrap())
    }

    /// Add a validation bundle.
    pub fn add(&mut self, bundle: &ValidationBundle) -> std::io::Result<()> {
        let uuid = Uuid::parse_str(&bundle.genome_id)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Serialize bundle to JSON
        let json = serde_json::to_vec(bundle)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Ultra-compress
        let compressed = ultra_compress(&json)?;

        // Write to data file
        let file = self.ensure_data_file()?;
        let offset = file.seek(SeekFrom::End(0))?;

        // Write length prefix + compressed data
        let len_bytes = (compressed.len() as u32).to_le_bytes();
        file.write_all(&len_bytes)?;
        file.write_all(&compressed)?;

        // Track location
        self.entries.push((
            uuid,
            BundleLocation {
                offset,
                compressed_size: compressed.len() as u32,
                original_size: json.len() as u32,
            },
        ));
        self.written_count += 1;

        Ok(())
    }

    /// Finish writing and return stats.
    pub fn finish(mut self) -> std::io::Result<ValidationBundleStats> {
        // Sync data file
        if let Some(ref mut file) = self.data_file {
            file.sync_all()?;
        }

        // Write index
        let index_path = self.root_path.join("index.json");
        let index: std::collections::HashMap<String, BundleLocation> = self
            .entries
            .iter()
            .map(|(uuid, loc)| (uuid.to_string(), loc.clone()))
            .collect();
        let index_json = serde_json::to_string_pretty(&index)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        fs::write(index_path, index_json)?;

        // Get data file size
        let data_size = if self.data_file_path().exists() {
            fs::metadata(self.data_file_path())?.len()
        } else {
            0
        };

        Ok(ValidationBundleStats {
            bundle_count: self.written_count,
            data_file_size: data_size,
            compression_level: ULTRA_COMPRESSION_LEVEL,
        })
    }

    pub fn count(&self) -> u64 {
        self.written_count
    }
}

/// Reader for consolidated validation report bundles.
pub struct ValidationBundleReader {
    root_path: PathBuf,
    index: std::collections::HashMap<String, BundleLocation>,
}

impl ValidationBundleReader {
    /// Open an existing bundle.
    pub fn open(root_path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let root_path = root_path.into();
        let index_path = root_path.join("index.json");

        let index: std::collections::HashMap<String, BundleLocation> = if index_path.exists() {
            let content = fs::read_to_string(&index_path)?;
            serde_json::from_str(&content)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?
        } else {
            std::collections::HashMap::new()
        };

        Ok(Self { root_path, index })
    }

    fn data_file_path(&self) -> PathBuf {
        self.root_path.join("validations.obfs")
    }

    /// Get a validation bundle by UUID.
    pub fn get(&self, uuid: Uuid) -> std::io::Result<Option<ValidationBundle>> {
        let key = uuid.to_string();
        let loc = match self.index.get(&key) {
            Some(l) => l,
            None => return Ok(None),
        };

        let mut file = File::open(self.data_file_path())?;
        file.seek(SeekFrom::Start(loc.offset))?;

        // Read length prefix
        let mut len_bytes = [0u8; 4];
        file.read_exact(&mut len_bytes)?;
        let compressed_len = u32::from_le_bytes(len_bytes) as usize;

        // Read compressed data
        let mut compressed = vec![0u8; compressed_len];
        file.read_exact(&mut compressed)?;

        // Decompress
        let decompressed = ultra_decompress(&compressed)?;

        // Deserialize
        let bundle: ValidationBundle = serde_json::from_slice(&decompressed)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        Ok(Some(bundle))
    }

    /// List all UUIDs in the bundle.
    pub fn list(&self) -> Vec<Uuid> {
        self.index
            .keys()
            .filter_map(|s| Uuid::parse_str(s).ok())
            .collect()
    }

    /// Get bundle count.
    pub fn count(&self) -> usize {
        self.index.len()
    }
}

/// Statistics for validation bundle.
#[derive(Debug, Clone)]
pub struct ValidationBundleStats {
    pub bundle_count: u64,
    pub data_file_size: u64,
    pub compression_level: i32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::WindowDetail;

    #[test]
    fn test_wfa_report_creation() {
        let wfa_result = WfaResult {
            genome_id: Uuid::new_v4(),
            is_sharpe_gross: 1.5,
            is_sharpe_net: 1.2,
            oos_sharpe_gross: 0.9,
            oos_sharpe_net: 0.7,
            degradation_pct: 41.7,
            passed: true,
            windows_evaluated: 5,
            is_cagr_net: 0.15,
            oos_cagr_net: 0.08,
            cost_report: None,
            window_details: vec![
                WindowDetail {
                    window_idx: 0,
                    is_sharpe: 1.3,
                    oos_sharpe: 0.8,
                    is_cagr: 0.14,
                    oos_cagr: 0.07,
                },
            ],
        };

        let thresholds = WfaThresholds {
            max_degradation: 0.40,
            min_oos_sharpe_net: 0.5,
            max_oos_drawdown: -0.35,
            min_oos_trades: 30,
        };

        let report = WfaReport::from_result(&wfa_result, thresholds);
        assert_eq!(report.verdict, ValidationVerdict::Pass);
        assert_eq!(report.summary.windows_evaluated, 5);
    }

    #[test]
    fn test_pbo_dsr_interpretation() {
        let pbo_result = PboDsrResult {
            genome_id: Uuid::new_v4(),
            is_sharpe_net: 1.5,
            pbo: 0.10,
            dsr: 1.5,
            total_trials: 1000,
            passed: true,
        };

        let thresholds = PboDsrThresholds {
            max_pbo: 0.15,
            min_dsr: 0.3,
        };

        let report = PboDsrReport::from_results(&pbo_result, None, thresholds);
        assert_eq!(report.verdict, ValidationVerdict::Pass);
        assert!(report.metrics.pbo_interpretation.contains("Good"));
        assert!(report.metrics.dsr_interpretation.contains("Moderate"));
    }

    #[test]
    fn test_validation_bundle() {
        let genome_id = Uuid::new_v4();
        let bundle = ValidationBundle::new(genome_id);

        assert_eq!(bundle.overall_verdict, ValidationVerdict::Pass);
        assert!(bundle.wfa.is_none());
        assert!(bundle.pbo_dsr.is_none());
        assert!(bundle.stress.is_none());
    }
}

