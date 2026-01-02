//! Schema validation for metrics and outputs.
//!
//! Ensures required fields are present and non-null.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

use crate::ValidationError;

/// Required fields that cannot be null in metrics.json.
pub const REQUIRED_METRICS_FIELDS: &[&str] = &[
    "cagr",
    "sharpe_ratio",
    "max_drawdown",
    "total_trades",
];

/// Fields that should exist but can have fallback defaults.
pub const RECOMMENDED_METRICS_FIELDS: &[&str] = &[
    "volatility",
    "sortino_ratio",
    "calmar_ratio",
    "profit_factor",
    "turnover_annual",
    "final_nav",
    "initial_capital",
];

/// Result of schema validation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchemaCheckResult {
    /// Whether schema validation passed.
    pub passed: bool,
    /// Fields that were validated.
    pub validated_fields: Vec<String>,
    /// Fields that were null (failures).
    pub null_fields: Vec<String>,
    /// Fields that were missing (failures).
    pub missing_fields: Vec<String>,
    /// Warnings for recommended but missing fields.
    pub warnings: Vec<String>,
}

impl SchemaCheckResult {
    /// Check if there are any critical failures.
    pub fn has_failures(&self) -> bool {
        !self.null_fields.is_empty() || !self.missing_fields.is_empty()
    }
}

/// Schema validator for backtest outputs.
pub struct SchemaValidator {
    /// Strict mode: fail on any null/missing required field.
    strict: bool,
}

impl Default for SchemaValidator {
    fn default() -> Self {
        Self::new(true)
    }
}

impl SchemaValidator {
    /// Create a new validator.
    pub fn new(strict: bool) -> Self {
        Self { strict }
    }

    /// Validate metrics.json content.
    pub fn validate_metrics(&self, json: &Value) -> SchemaCheckResult {
        let mut result = SchemaCheckResult {
            passed: true,
            validated_fields: vec![],
            null_fields: vec![],
            missing_fields: vec![],
            warnings: vec![],
        };

        // Check required fields
        for field in REQUIRED_METRICS_FIELDS {
            match json.get(*field) {
                None => {
                    result.missing_fields.push((*field).to_string());
                    result.passed = false;
                }
                Some(v) if v.is_null() => {
                    result.null_fields.push((*field).to_string());
                    result.passed = false;
                }
                Some(_) => {
                    result.validated_fields.push((*field).to_string());
                }
            }
        }

        // Check recommended fields (warnings only)
        for field in RECOMMENDED_METRICS_FIELDS {
            match json.get(*field) {
                None => {
                    result.warnings.push(format!("Recommended field '{}' is missing", field));
                }
                Some(v) if v.is_null() => {
                    result.warnings.push(format!("Recommended field '{}' is null", field));
                }
                Some(_) => {
                    result.validated_fields.push((*field).to_string());
                }
            }
        }

        result
    }

    /// Validate metrics from file.
    pub fn validate_metrics_file(&self, path: &Path) -> Result<SchemaCheckResult, ValidationError> {
        let content = std::fs::read_to_string(path)?;
        let json: Value = serde_json::from_str(&content)?;
        Ok(self.validate_metrics(&json))
    }

    /// Validate nav_history.csv exists and has data.
    pub fn validate_nav_history(&self, path: &Path) -> Result<SchemaCheckResult, ValidationError> {
        let mut result = SchemaCheckResult {
            passed: true,
            validated_fields: vec![],
            null_fields: vec![],
            missing_fields: vec![],
            warnings: vec![],
        };

        if !path.exists() {
            result.missing_fields.push("nav_history.csv".to_string());
            result.passed = false;
            return Ok(result);
        }

        // Check if file has content
        let content = std::fs::read_to_string(path)?;
        let lines: Vec<&str> = content.lines().collect();
        
        if lines.len() <= 1 {
            result.warnings.push("nav_history.csv has no data rows".to_string());
            if self.strict {
                result.passed = false;
            }
        } else {
            result.validated_fields.push(format!("nav_history: {} rows", lines.len() - 1));
        }

        Ok(result)
    }

    /// Validate trades.csv exists.
    pub fn validate_trades(&self, path: &Path) -> Result<SchemaCheckResult, ValidationError> {
        let mut result = SchemaCheckResult {
            passed: true,
            validated_fields: vec![],
            null_fields: vec![],
            missing_fields: vec![],
            warnings: vec![],
        };

        if !path.exists() {
            // trades.csv can be empty for buy-and-hold strategies
            result.warnings.push("trades.csv not found (may be intentional for buy-and-hold)".to_string());
            return Ok(result);
        }

        let content = std::fs::read_to_string(path)?;
        let lines: Vec<&str> = content.lines().collect();
        result.validated_fields.push(format!("trades: {} rows", lines.len().saturating_sub(1)));

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_valid_metrics() {
        let validator = SchemaValidator::default();
        let json = json!({
            "cagr": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.15,
            "total_trades": 100,
            "volatility": 0.18,
            "sortino_ratio": 1.5,
            "calmar_ratio": 1.0,
            "profit_factor": 1.8,
            "turnover_annual": 2.5,
            "final_nav": 1150000.0,
            "initial_capital": 1000000.0
        });

        let result = validator.validate_metrics(&json);
        assert!(result.passed);
        assert!(result.null_fields.is_empty());
        assert!(result.missing_fields.is_empty());
    }

    #[test]
    fn test_missing_required_field() {
        let validator = SchemaValidator::default();
        let json = json!({
            "cagr": 0.15,
            // missing sharpe_ratio
            "max_drawdown": -0.15,
            "total_trades": 100
        });

        let result = validator.validate_metrics(&json);
        assert!(!result.passed);
        assert!(result.missing_fields.contains(&"sharpe_ratio".to_string()));
    }

    #[test]
    fn test_null_required_field() {
        let validator = SchemaValidator::default();
        let json = json!({
            "cagr": 0.15,
            "sharpe_ratio": null,
            "max_drawdown": -0.15,
            "total_trades": 100
        });

        let result = validator.validate_metrics(&json);
        assert!(!result.passed);
        assert!(result.null_fields.contains(&"sharpe_ratio".to_string()));
    }

    #[test]
    fn test_missing_recommended_field_warns() {
        let validator = SchemaValidator::default();
        let json = json!({
            "cagr": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.15,
            "total_trades": 100
            // missing volatility and others
        });

        let result = validator.validate_metrics(&json);
        assert!(result.passed); // Still passes
        assert!(!result.warnings.is_empty()); // But has warnings
    }
}

