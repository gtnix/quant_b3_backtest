//! # Backtester Validation
//!
//! Validation module for backtester outputs with sanity checks, cross-validation,
//! asset attribution, and human-readable reporting.
//!
//! ## Pipeline Stages
//!
//! 1. **Schema Validation** - Verify required fields are present and non-null
//! 2. **Numeric Invariants** - Check for NaN/Inf, consistency
//! 3. **Sanity Checks** - Detect suspicious metrics (Sharpe > 20, etc.)
//! 4. **Cross-check** - Recompute metrics and compare
//! 5. **Attribution** - Calculate PnL by asset
//!
//! ## Usage
//!
//! ```ignore
//! use backtester_validation::{ValidationPipeline, ValidationConfig};
//!
//! let config = ValidationConfig::default();
//! let pipeline = ValidationPipeline::new(config);
//! let result = pipeline.validate(&artifacts)?;
//!
//! if result.verdict == Verdict::Fail {
//!     eprintln!("Validation failed: {:?}", result.errors);
//! }
//! ```

#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod schema;
pub mod sanity;
pub mod crosscheck;
pub mod attribution;
pub mod report;
pub mod summary;
pub mod pipeline;

// Re-exports
pub use pipeline::{ValidationPipeline, ValidationConfig};
pub use schema::{SchemaValidator, SchemaCheckResult};
pub use sanity::{SanityChecker, SanityCheckResult, SanityFlags};
pub use crosscheck::{CrossChecker, CrosscheckResult};
pub use attribution::{AttributionCalculator, AssetAttribution, AttributionResult};
pub use report::ReportGenerator;
pub use summary::{ValidationSummary, Verdict};

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

/// Errors during validation.
#[derive(Debug, Error)]
pub enum ValidationError {
    /// IO error reading files.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parse error.
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    /// CSV parse error.
    #[error("CSV parse error: {0}")]
    Csv(#[from] csv::Error),

    /// Missing required field.
    #[error("Missing required field: {0}")]
    MissingField(String),

    /// Invalid value.
    #[error("Invalid value for {field}: {message}")]
    InvalidValue {
        /// Field name.
        field: String,
        /// Error message.
        message: String,
    },

    /// Schema validation failed.
    #[error("Schema validation failed: {0}")]
    SchemaFailed(String),

    /// Sanity check failed.
    #[error("Sanity check failed: {0}")]
    SanityFailed(String),

    /// Cross-check failed.
    #[error("Cross-check failed: {0}")]
    CrosscheckFailed(String),
}

/// Warning during validation (non-fatal).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    /// Warning code.
    pub code: String,
    /// Warning message.
    pub message: String,
    /// Related field or metric.
    pub field: Option<String>,
}

impl ValidationWarning {
    /// Create a new warning.
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            field: None,
        }
    }

    /// Create a warning with field context.
    pub fn with_field(
        code: impl Into<String>,
        message: impl Into<String>,
        field: impl Into<String>,
    ) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            field: Some(field.into()),
        }
    }
}

/// Artifacts from a backtest run to be validated.
#[derive(Debug, Clone)]
pub struct BacktestArtifacts {
    /// Run identifier.
    pub run_id: String,
    /// Path to metrics.json.
    pub metrics_path: PathBuf,
    /// Path to nav_history.csv.
    pub nav_history_path: PathBuf,
    /// Path to trades.csv.
    pub trades_path: PathBuf,
    /// Path to manifest.json (optional).
    pub manifest_path: Option<PathBuf>,
}

impl BacktestArtifacts {
    /// Create from a run directory.
    pub fn from_dir(run_dir: &std::path::Path, run_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            metrics_path: run_dir.join("metrics.json"),
            nav_history_path: run_dir.join("nav_history.csv"),
            trades_path: run_dir.join("trades.csv"),
            manifest_path: Some(run_dir.join("manifest.json")),
        }
    }

    /// Check if all required files exist.
    pub fn files_exist(&self) -> bool {
        self.metrics_path.exists() && self.nav_history_path.exists()
    }
}

/// Complete validation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Run ID.
    pub run_id: String,
    /// Overall verdict.
    pub verdict: Verdict,
    /// Schema check result.
    pub schema_check: SchemaCheckResult,
    /// Sanity check result.
    pub sanity_check: SanityCheckResult,
    /// Cross-check result.
    pub crosscheck: Option<CrosscheckResult>,
    /// Attribution result.
    pub attribution: Option<AttributionResult>,
    /// All errors encountered.
    pub errors: Vec<String>,
    /// All warnings encountered.
    pub warnings: Vec<ValidationWarning>,
    /// Timestamp of validation.
    pub validated_at: chrono::DateTime<chrono::Utc>,
}

impl ValidationResult {
    /// Create a failed result.
    pub fn failed(run_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            verdict: Verdict::Fail,
            schema_check: SchemaCheckResult::default(),
            sanity_check: SanityCheckResult::default(),
            crosscheck: None,
            attribution: None,
            errors: vec![error.into()],
            warnings: vec![],
            validated_at: chrono::Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_warning_creation() {
        let warn = ValidationWarning::new("SHARPE_HIGH", "Sharpe ratio > 10");
        assert_eq!(warn.code, "SHARPE_HIGH");
        assert!(warn.field.is_none());

        let warn2 = ValidationWarning::with_field("NULL_FIELD", "Field is null", "volatility");
        assert_eq!(warn2.field, Some("volatility".to_string()));
    }

    #[test]
    fn test_artifacts_from_dir() {
        let dir = std::path::Path::new("/tmp/test_run");
        let artifacts = BacktestArtifacts::from_dir(dir, "test_123");
        assert_eq!(artifacts.run_id, "test_123");
        assert_eq!(artifacts.metrics_path, dir.join("metrics.json"));
    }
}


