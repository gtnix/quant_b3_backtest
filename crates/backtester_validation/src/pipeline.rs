//! Validation pipeline orchestration.

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::{
    BacktestArtifacts, ValidationError, ValidationResult, ValidationWarning,
    schema::{SchemaValidator, SchemaCheckResult},
    sanity::{SanityChecker, SanityCheckResult, SanityConfig},
    crosscheck::{CrossChecker, CrosscheckConfig, CrosscheckResult, ReportedMetrics},
    attribution::{AttributionCalculator, AttributionConfig, AttributionResult},
    report::ReportGenerator,
    summary::{ValidationSummary, Verdict},
};

/// Configuration for the validation pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Enable schema validation.
    #[serde(default = "default_true")]
    pub schema_check_enabled: bool,
    /// Enable sanity checks.
    #[serde(default = "default_true")]
    pub sanity_check_enabled: bool,
    /// Enable cross-check (recompute and compare).
    #[serde(default = "default_true")]
    pub crosscheck_enabled: bool,
    /// Enable attribution calculation.
    #[serde(default = "default_true")]
    pub attribution_enabled: bool,
    /// Generate markdown report.
    #[serde(default = "default_true")]
    pub report_enabled: bool,
    /// Sanity check configuration.
    #[serde(default)]
    pub sanity: SanityConfig,
    /// Cross-check configuration.
    #[serde(default)]
    pub crosscheck: CrosscheckConfig,
    /// Attribution configuration.
    #[serde(default)]
    pub attribution: AttributionConfig,
    /// Strict mode: fail on any warning.
    #[serde(default)]
    pub strict_mode: bool,
}

fn default_true() -> bool { true }

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            schema_check_enabled: true,
            sanity_check_enabled: true,
            crosscheck_enabled: true,
            attribution_enabled: true,
            report_enabled: true,
            sanity: SanityConfig::default(),
            crosscheck: CrosscheckConfig::default(),
            attribution: AttributionConfig::default(),
            strict_mode: false,
        }
    }
}

/// Validation pipeline that orchestrates all checks.
pub struct ValidationPipeline {
    config: ValidationConfig,
    schema_validator: SchemaValidator,
    sanity_checker: SanityChecker,
    crosschecker: CrossChecker,
    attribution_calculator: AttributionCalculator,
}

impl Default for ValidationPipeline {
    fn default() -> Self {
        Self::new(ValidationConfig::default())
    }
}

impl ValidationPipeline {
    /// Create a new validation pipeline.
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            schema_validator: SchemaValidator::new(true),
            sanity_checker: SanityChecker::new(config.sanity.clone()),
            crosschecker: CrossChecker::new(config.crosscheck.clone()),
            attribution_calculator: AttributionCalculator::new(config.attribution.clone()),
            config,
        }
    }

    /// Run the full validation pipeline.
    pub fn validate(&self, artifacts: &BacktestArtifacts) -> Result<ValidationResult, ValidationError> {
        let mut errors: Vec<String> = Vec::new();
        let mut warnings: Vec<ValidationWarning> = Vec::new();

        // Check files exist
        if !artifacts.metrics_path.exists() {
            return Err(ValidationError::MissingField("metrics.json".to_string()));
        }

        // Load metrics JSON
        let metrics_content = std::fs::read_to_string(&artifacts.metrics_path)?;
        let metrics_json: serde_json::Value = serde_json::from_str(&metrics_content)?;

        // Stage A: Schema validation
        let schema_check = if self.config.schema_check_enabled {
            self.schema_validator.validate_metrics(&metrics_json)
        } else {
            SchemaCheckResult::default()
        };

        if schema_check.has_failures() {
            let error_msg = format!(
                "Schema validation failed: missing {:?}, null {:?}",
                schema_check.missing_fields, schema_check.null_fields
            );
            return Ok(ValidationResult {
                run_id: artifacts.run_id.clone(),
                verdict: Verdict::Fail,
                schema_check,
                sanity_check: SanityCheckResult::default(),
                crosscheck: None,
                attribution: None,
                errors: vec![error_msg],
                warnings: vec![],
                validated_at: chrono::Utc::now(),
            });
        }

        // Stage B: Sanity checks
        let sanity_check = if self.config.sanity_check_enabled {
            self.sanity_checker.check_json(&metrics_json)
        } else {
            SanityCheckResult::default()
        };

        warnings.extend(sanity_check.warnings.clone());

        if sanity_check.verdict == Verdict::Fail {
            errors.push(sanity_check.message.clone());
        }

        // Stage C: Cross-check (if nav_history exists)
        let crosscheck = if self.config.crosscheck_enabled && artifacts.nav_history_path.exists() {
            match self.run_crosscheck(&artifacts.nav_history_path, &metrics_json) {
                Ok(cc) => {
                    warnings.extend(cc.warnings.clone());
                    if cc.verdict == Verdict::Fail {
                        errors.push("Cross-check validation failed".to_string());
                    }
                    Some(cc)
                }
                Err(e) => {
                    warnings.push(ValidationWarning::new(
                        "CROSSCHECK_ERROR",
                        format!("Could not run cross-check: {}", e),
                    ));
                    None
                }
            }
        } else {
            None
        };

        // Stage D: Attribution (if trades exist)
        let attribution = if self.config.attribution_enabled && artifacts.trades_path.exists() {
            match self.run_attribution(&artifacts.trades_path) {
                Ok(attr) => {
                    warnings.extend(attr.warnings.clone());
                    Some(attr)
                }
                Err(e) => {
                    warnings.push(ValidationWarning::new(
                        "ATTRIBUTION_ERROR",
                        format!("Could not calculate attribution: {}", e),
                    ));
                    None
                }
            }
        } else {
            None
        };

        // Determine final verdict
        let verdict = self.determine_verdict(&schema_check, &sanity_check, &crosscheck);

        Ok(ValidationResult {
            run_id: artifacts.run_id.clone(),
            verdict,
            schema_check,
            sanity_check,
            crosscheck,
            attribution,
            errors,
            warnings,
            validated_at: chrono::Utc::now(),
        })
    }

    /// Run cross-check validation.
    fn run_crosscheck(
        &self,
        nav_path: &Path,
        metrics_json: &serde_json::Value,
    ) -> Result<CrosscheckResult, ValidationError> {
        let nav_rows = self.crosschecker.load_nav_history(nav_path)?;
        
        if nav_rows.is_empty() {
            return Err(ValidationError::InvalidValue {
                field: "nav_history".to_string(),
                message: "No NAV data found".to_string(),
            });
        }

        let nav_series: Vec<f64> = nav_rows.iter().map(|r| r.nav).collect();
        let reported = ReportedMetrics::from_json(metrics_json);

        Ok(self.crosschecker.crosscheck(&reported, &nav_series))
    }

    /// Run attribution calculation.
    fn run_attribution(&self, trades_path: &Path) -> Result<AttributionResult, ValidationError> {
        let trades = self.attribution_calculator.load_trades(trades_path)?;
        Ok(self.attribution_calculator.calculate(&trades))
    }

    /// Determine final verdict from all checks.
    fn determine_verdict(
        &self,
        schema: &SchemaCheckResult,
        sanity: &SanityCheckResult,
        crosscheck: &Option<CrosscheckResult>,
    ) -> Verdict {
        // Fail if schema failed
        if schema.has_failures() {
            return Verdict::Fail;
        }

        // Fail if sanity failed
        if sanity.verdict == Verdict::Fail {
            return Verdict::Fail;
        }

        // Fail if crosscheck failed
        if let Some(cc) = crosscheck {
            if cc.verdict == Verdict::Fail {
                return Verdict::Fail;
            }
        }

        // Strict mode: warnings become failures
        if self.config.strict_mode {
            if sanity.verdict == Verdict::Warn {
                return Verdict::Fail;
            }
            if let Some(cc) = crosscheck {
                if cc.verdict == Verdict::Warn {
                    return Verdict::Fail;
                }
            }
        }

        // Warn if any warnings
        if sanity.verdict == Verdict::Warn {
            return Verdict::Warn;
        }
        if let Some(cc) = crosscheck {
            if cc.verdict == Verdict::Warn {
                return Verdict::Warn;
            }
        }

        Verdict::Pass
    }

    /// Generate all output artifacts.
    pub fn generate_artifacts(
        &self,
        result: &ValidationResult,
        output_dir: &Path,
    ) -> Result<(), ValidationError> {
        std::fs::create_dir_all(output_dir)?;

        // Generate validation_summary.json
        let mut summary = ValidationSummary::new(&result.run_id);
        summary
            .with_schema(&result.schema_check)
            .with_sanity(&result.sanity_check)
            .with_crosscheck(result.crosscheck.as_ref())
            .with_attribution(result.attribution.as_ref());

        summary.write_json(&output_dir.join("validation_summary.json"))?;

        // Generate sanity.json
        summary.write_sanity_json(&result.sanity_check, &output_dir.join("sanity.json"))?;

        // Generate asset_attribution.csv
        if let Some(ref attr) = result.attribution {
            self.attribution_calculator.write_csv(attr, &output_dir.join("asset_attribution.csv"))?;
        }

        // Generate backtest_report.md
        if self.config.report_enabled {
            let report_gen = ReportGenerator::new(&result.run_id);
            let report = report_gen.generate(
                &result.sanity_check,
                result.crosscheck.as_ref(),
                result.attribution.as_ref(),
                result.verdict,
                None, // TODO: extract from manifest
                None,
                None,
            );
            report_gen.write_to_file(&report, &output_dir.join("backtest_report.md"))?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_artifacts(dir: &Path) -> BacktestArtifacts {
        // Create minimal test files
        let metrics = serde_json::json!({
            "cagr": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.12,
            "total_trades": 100,
            "volatility": 0.18
        });
        std::fs::write(dir.join("metrics.json"), serde_json::to_string(&metrics).unwrap()).unwrap();

        // Create minimal nav_history
        std::fs::write(dir.join("nav_history.csv"), "date,nav\n2020-01-01,1000000\n2020-01-02,1005000\n").unwrap();

        // Create minimal trades
        std::fs::write(dir.join("trades.csv"), "symbol,net_pnl\nPETR4,100\nVALE3,-50\n").unwrap();

        BacktestArtifacts::from_dir(dir, "test_run")
    }

    #[test]
    fn test_pipeline_basic() {
        let temp_dir = TempDir::new().unwrap();
        let artifacts = create_test_artifacts(temp_dir.path());

        // Disable crosscheck for this test since NAV data is minimal
        let config = ValidationConfig {
            crosscheck_enabled: false,
            ..Default::default()
        };
        let pipeline = ValidationPipeline::new(config);
        let result = pipeline.validate(&artifacts).unwrap();

        // Should pass schema and sanity checks
        assert!(!result.schema_check.has_failures());
        // Note: may have warnings due to minimal data (< 30 trades in attribution)
    }

    #[test]
    fn test_pipeline_missing_metrics() {
        let temp_dir = TempDir::new().unwrap();
        let artifacts = BacktestArtifacts::from_dir(temp_dir.path(), "test_run");

        let pipeline = ValidationPipeline::default();
        let result = pipeline.validate(&artifacts);

        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_generates_artifacts() {
        let temp_dir = TempDir::new().unwrap();
        let artifacts = create_test_artifacts(temp_dir.path());

        let pipeline = ValidationPipeline::default();
        let result = pipeline.validate(&artifacts).unwrap();

        let output_dir = temp_dir.path().join("validation");
        pipeline.generate_artifacts(&result, &output_dir).unwrap();

        assert!(output_dir.join("validation_summary.json").exists());
        assert!(output_dir.join("sanity.json").exists());
        assert!(output_dir.join("asset_attribution.csv").exists());
        assert!(output_dir.join("backtest_report.md").exists());
    }
}

