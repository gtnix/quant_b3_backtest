//! Validation summary generation.

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;

use crate::{
    SchemaCheckResult, SanityCheckResult, CrosscheckResult, AttributionResult,
    ValidationError,
};

/// Validation verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Verdict {
    /// All checks passed.
    #[default]
    Pass,
    /// Passed with warnings.
    Warn,
    /// Failed validation.
    Fail,
}

/// Complete validation summary for JSON output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    /// Run identifier.
    pub run_id: String,
    /// Overall verdict.
    pub verdict: Verdict,
    /// Check results summary.
    pub checks: ChecksSummary,
    /// All warnings.
    pub warnings: Vec<String>,
    /// All errors.
    pub errors: Vec<String>,
    /// Timestamp.
    pub generated_at: String,
}

/// Summary of individual checks.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChecksSummary {
    /// Schema validation result.
    pub schema: CheckStatus,
    /// Numeric invariants result.
    pub invariants: CheckStatus,
    /// Sanity checks result.
    pub sanity: CheckStatus,
    /// Cross-check result.
    pub crosscheck: CheckStatus,
    /// Attribution result.
    pub attribution: CheckStatus,
}

/// Status of a single check.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CheckStatus {
    /// Check verdict.
    pub verdict: Verdict,
    /// Message.
    pub message: String,
}

impl CheckStatus {
    /// Create a passed check.
    pub fn pass(message: impl Into<String>) -> Self {
        Self { verdict: Verdict::Pass, message: message.into() }
    }

    /// Create a warning check.
    pub fn warn(message: impl Into<String>) -> Self {
        Self { verdict: Verdict::Warn, message: message.into() }
    }

    /// Create a failed check.
    pub fn fail(message: impl Into<String>) -> Self {
        Self { verdict: Verdict::Fail, message: message.into() }
    }

    /// Create a skipped check.
    pub fn skipped() -> Self {
        Self { verdict: Verdict::Pass, message: "Skipped".to_string() }
    }
}

impl ValidationSummary {
    /// Create a new summary.
    pub fn new(run_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            verdict: Verdict::Pass,
            checks: ChecksSummary::default(),
            warnings: vec![],
            errors: vec![],
            generated_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Add schema check result.
    pub fn with_schema(&mut self, result: &SchemaCheckResult) -> &mut Self {
        if result.has_failures() {
            let msg = format!(
                "Missing: {:?}, Null: {:?}",
                result.missing_fields, result.null_fields
            );
            self.checks.schema = CheckStatus::fail(&msg);
            self.errors.push(format!("Schema validation failed: {}", msg));
            self.verdict = Verdict::Fail;
        } else if !result.warnings.is_empty() {
            self.checks.schema = CheckStatus::warn(result.warnings.join("; "));
            for w in &result.warnings {
                self.warnings.push(w.clone());
            }
            if self.verdict == Verdict::Pass {
                self.verdict = Verdict::Warn;
            }
        } else {
            self.checks.schema = CheckStatus::pass("All required fields present");
        }
        self
    }

    /// Add sanity check result.
    pub fn with_sanity(&mut self, result: &SanityCheckResult) -> &mut Self {
        self.checks.sanity = CheckStatus {
            verdict: result.verdict,
            message: result.message.clone(),
        };

        for warning in &result.warnings {
            self.warnings.push(warning.message.clone());
        }

        if result.verdict == Verdict::Fail {
            self.verdict = Verdict::Fail;
        } else if result.verdict == Verdict::Warn && self.verdict == Verdict::Pass {
            self.verdict = Verdict::Warn;
        }

        self
    }

    /// Add cross-check result.
    pub fn with_crosscheck(&mut self, result: Option<&CrosscheckResult>) -> &mut Self {
        match result {
            Some(cc) => {
                self.checks.crosscheck = CheckStatus {
                    verdict: cc.verdict,
                    message: if cc.passed {
                        "All metrics match within tolerance".to_string()
                    } else {
                        format!("{} metrics failed cross-check", 
                            cc.comparisons.iter().filter(|c| !c.passed).count())
                    },
                };

                for warning in &cc.warnings {
                    self.warnings.push(warning.message.clone());
                }

                if cc.verdict == Verdict::Fail {
                    self.verdict = Verdict::Fail;
                    self.errors.push("Cross-check validation failed".to_string());
                }
            }
            None => {
                self.checks.crosscheck = CheckStatus::skipped();
            }
        }
        self
    }

    /// Add attribution result.
    pub fn with_attribution(&mut self, result: Option<&AttributionResult>) -> &mut Self {
        match result {
            Some(attr) => {
                self.checks.attribution = CheckStatus {
                    verdict: attr.verdict,
                    message: format!(
                        "{} assets, total PnL: {:.2}",
                        attr.attributions.len(),
                        attr.total_net_pnl
                    ),
                };

                for warning in &attr.warnings {
                    self.warnings.push(warning.message.clone());
                }

                if attr.verdict == Verdict::Warn && self.verdict == Verdict::Pass {
                    self.verdict = Verdict::Warn;
                }
            }
            None => {
                self.checks.attribution = CheckStatus::skipped();
            }
        }
        self
    }

    /// Set invariants check.
    pub fn with_invariants(&mut self, passed: bool, message: impl Into<String>) -> &mut Self {
        if passed {
            self.checks.invariants = CheckStatus::pass(message);
        } else {
            self.checks.invariants = CheckStatus::fail(message.into());
            self.verdict = Verdict::Fail;
        }
        self
    }

    /// Build the summary.
    pub fn build(self) -> Self {
        self
    }

    /// Write to JSON file.
    pub fn write_json(&self, path: &Path) -> Result<(), ValidationError> {
        let json = serde_json::to_string_pretty(self)?;
        let mut file = std::fs::File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Write sanity.json (subset for quick checks).
    pub fn write_sanity_json(&self, sanity: &SanityCheckResult, path: &Path) -> Result<(), ValidationError> {
        let json = serde_json::to_string_pretty(sanity)?;
        let mut file = std::fs::File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ValidationWarning;
    use crate::sanity::{SanityFlags, MetricsSnapshot};

    #[test]
    fn test_summary_pass() {
        let summary = ValidationSummary::new("test_123")
            .build();

        assert_eq!(summary.verdict, Verdict::Pass);
        assert!(summary.errors.is_empty());
    }

    #[test]
    fn test_summary_with_schema_failure() {
        let mut summary = ValidationSummary::new("test_123");
        
        let schema = SchemaCheckResult {
            passed: false,
            missing_fields: vec!["sharpe_ratio".to_string()],
            null_fields: vec![],
            validated_fields: vec![],
            warnings: vec![],
        };

        summary.with_schema(&schema);

        assert_eq!(summary.verdict, Verdict::Fail);
        assert!(!summary.errors.is_empty());
    }

    #[test]
    fn test_summary_with_sanity_warning() {
        let mut summary = ValidationSummary::new("test_123");
        
        let sanity = SanityCheckResult {
            flags: SanityFlags { sharpe_suspicious: true, ..Default::default() },
            verdict: Verdict::Warn,
            message: "High Sharpe detected".to_string(),
            warnings: vec![ValidationWarning::new("SHARPE_HIGH", "Sharpe > 10")],
            metrics_snapshot: MetricsSnapshot::default(),
        };

        summary.with_sanity(&sanity);

        assert_eq!(summary.verdict, Verdict::Warn);
        assert!(!summary.warnings.is_empty());
    }
}

