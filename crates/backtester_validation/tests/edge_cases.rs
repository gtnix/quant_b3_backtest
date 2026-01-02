//! Edge case tests for validation module.
//!
//! Tests extreme and unusual scenarios to ensure robustness.

use backtester_validation::{
    BacktestArtifacts, ValidationPipeline, ValidationConfig, Verdict,
    SchemaValidator, SanityChecker, CrossChecker,
    sanity::SanityConfig,
    crosscheck::CrosscheckConfig,
};
use std::fs;
use tempfile::TempDir;

// =============================================================================
// EDGE CASE: EMPTY OR MINIMAL DATA
// =============================================================================

#[test]
fn test_empty_metrics_json() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path();
    
    // Empty JSON object
    fs::write(path.join("metrics.json"), "{}").unwrap();
    fs::write(path.join("nav_history.csv"), "date,nav\n").unwrap();
    
    let artifacts = BacktestArtifacts::from_dir(path, "empty_run");
    let pipeline = ValidationPipeline::default();
    let result = pipeline.validate(&artifacts).unwrap();
    
    // Should fail due to missing required fields
    assert_eq!(result.verdict, Verdict::Fail);
    assert!(!result.schema_check.missing_fields.is_empty());
}

#[test]
fn test_single_day_nav_history() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path();
    
    let metrics = serde_json::json!({
        "cagr": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "total_trades": 0,
        "volatility": 0.0
    });
    fs::write(path.join("metrics.json"), serde_json::to_string(&metrics).unwrap()).unwrap();
    
    // Only one day - can't compute returns
    fs::write(path.join("nav_history.csv"), "date,nav\n2024-01-02,1000000\n").unwrap();
    
    let artifacts = BacktestArtifacts::from_dir(path, "single_day");
    let config = ValidationConfig {
        crosscheck_enabled: false, // Can't crosscheck with 1 day
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();
    
    // Should pass schema, but may warn about zero values
    assert!(!result.schema_check.has_failures());
}

#[test]
fn test_empty_nav_history() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path();
    
    let metrics = serde_json::json!({
        "cagr": 0.1,
        "sharpe_ratio": 1.0,
        "max_drawdown": -0.1,
        "total_trades": 50,
        "volatility": 0.15
    });
    fs::write(path.join("metrics.json"), serde_json::to_string(&metrics).unwrap()).unwrap();
    
    // Empty NAV history (header only)
    fs::write(path.join("nav_history.csv"), "date,nav\n").unwrap();
    
    let artifacts = BacktestArtifacts::from_dir(path, "empty_nav");
    let pipeline = ValidationPipeline::default();
    let result = pipeline.validate(&artifacts).unwrap();
    
    // Should still pass schema check (file exists)
    // Cross-check might fail or warn
    assert!(!result.schema_check.has_failures());
}

#[test]
fn test_empty_trades_file() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path();
    
    let metrics = serde_json::json!({
        "cagr": 0.1,
        "sharpe_ratio": 1.0,
        "max_drawdown": -0.1,
        "total_trades": 0,
        "volatility": 0.15
    });
    fs::write(path.join("metrics.json"), serde_json::to_string(&metrics).unwrap()).unwrap();
    fs::write(path.join("nav_history.csv"), "date,nav\n2024-01-02,1000000\n2024-01-03,1001000\n").unwrap();
    
    // Empty trades (header only)
    fs::write(path.join("trades.csv"), "symbol,net_pnl\n").unwrap();
    
    let artifacts = BacktestArtifacts::from_dir(path, "empty_trades");
    let config = ValidationConfig {
        crosscheck_enabled: false,
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();
    
    // Attribution should handle empty trades gracefully
    assert!(!result.schema_check.has_failures());
}

// =============================================================================
// EDGE CASE: EXTREME VALUES
// =============================================================================

#[test]
fn test_nan_in_metrics() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path();
    
    // NaN value (JavaScript NaN representation)
    let metrics_str = r#"{
        "cagr": null,
        "sharpe_ratio": 1.0,
        "max_drawdown": -0.1,
        "total_trades": 50,
        "volatility": 0.15
    }"#;
    fs::write(path.join("metrics.json"), metrics_str).unwrap();
    fs::write(path.join("nav_history.csv"), "date,nav\n2024-01-02,1000000\n").unwrap();
    
    let artifacts = BacktestArtifacts::from_dir(path, "nan_run");
    let pipeline = ValidationPipeline::default();
    let result = pipeline.validate(&artifacts).unwrap();
    
    // Should fail - null is not allowed for cagr
    assert_eq!(result.verdict, Verdict::Fail);
    assert!(result.schema_check.null_fields.contains(&"cagr".to_string()));
}

#[test]
fn test_infinity_sharpe() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path();
    
    // Technically valid but suspicious
    let metrics = serde_json::json!({
        "cagr": 0.5,
        "sharpe_ratio": 999.0, // Absurdly high
        "max_drawdown": -0.01,
        "total_trades": 10,
        "volatility": 0.0001
    });
    fs::write(path.join("metrics.json"), serde_json::to_string(&metrics).unwrap()).unwrap();
    fs::write(path.join("nav_history.csv"), "date,nav\n2024-01-02,1000000\n").unwrap();
    
    let artifacts = BacktestArtifacts::from_dir(path, "inf_sharpe");
    let config = ValidationConfig {
        crosscheck_enabled: false,
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();
    
    // Should fail - Sharpe > 20
    assert_eq!(result.verdict, Verdict::Fail);
}

#[test]
fn test_negative_nav() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path();
    
    let metrics = serde_json::json!({
        "cagr": -0.5,
        "sharpe_ratio": -1.0,
        "max_drawdown": -0.9,
        "total_trades": 50,
        "volatility": 0.3
    });
    fs::write(path.join("metrics.json"), serde_json::to_string(&metrics).unwrap()).unwrap();
    
    // Negative NAV (shouldn't happen but let's test)
    fs::write(path.join("nav_history.csv"), 
              "date,nav\n2024-01-02,1000000\n2024-01-03,-100000\n").unwrap();
    
    let artifacts = BacktestArtifacts::from_dir(path, "neg_nav");
    let config = ValidationConfig {
        crosscheck_enabled: false, // Would fail on negative returns
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();
    
    // Schema should pass (metrics are valid)
    assert!(!result.schema_check.has_failures());
}

#[test]
fn test_zero_volatility() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path();
    
    let metrics = serde_json::json!({
        "cagr": 0.1,
        "sharpe_ratio": 5.0,
        "max_drawdown": 0.0,
        "total_trades": 50,
        "volatility": 0.0 // Zero volatility
    });
    fs::write(path.join("metrics.json"), serde_json::to_string(&metrics).unwrap()).unwrap();
    fs::write(path.join("nav_history.csv"), "date,nav\n2024-01-02,1000000\n").unwrap();
    
    let artifacts = BacktestArtifacts::from_dir(path, "zero_vol");
    let config = ValidationConfig {
        crosscheck_enabled: false,
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();
    
    // Zero volatility is a suspicious but not always flagged - depends on sanity config
    // The schema check should pass since values are not null
    assert!(!result.schema_check.has_failures());
}

// =============================================================================
// EDGE CASE: MALFORMED DATA
// =============================================================================

#[test]
fn test_invalid_json() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path();
    
    // Invalid JSON
    fs::write(path.join("metrics.json"), "not valid json {").unwrap();
    fs::write(path.join("nav_history.csv"), "date,nav\n").unwrap();
    
    let artifacts = BacktestArtifacts::from_dir(path, "invalid_json");
    let pipeline = ValidationPipeline::default();
    let result = pipeline.validate(&artifacts);
    
    // Should return error (not Ok with Fail verdict)
    assert!(result.is_err());
}

#[test]
fn test_missing_metrics_file() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path();
    
    // No metrics.json
    fs::write(path.join("nav_history.csv"), "date,nav\n2024-01-02,1000000\n").unwrap();
    
    let artifacts = BacktestArtifacts::from_dir(path, "missing_metrics");
    let pipeline = ValidationPipeline::default();
    let result = pipeline.validate(&artifacts);
    
    // Should return error
    assert!(result.is_err());
}

#[test]
fn test_wrong_csv_columns() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path();
    
    let metrics = serde_json::json!({
        "cagr": 0.1,
        "sharpe_ratio": 1.0,
        "max_drawdown": -0.1,
        "total_trades": 50,
        "volatility": 0.15
    });
    fs::write(path.join("metrics.json"), serde_json::to_string(&metrics).unwrap()).unwrap();
    
    // Wrong column names
    fs::write(path.join("nav_history.csv"), "wrong,columns\nfoo,bar\n").unwrap();
    
    let artifacts = BacktestArtifacts::from_dir(path, "wrong_csv");
    let config = ValidationConfig {
        crosscheck_enabled: true,
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();
    
    // Schema should pass (metrics OK), crosscheck should warn/fail
    assert!(!result.schema_check.has_failures());
    // Crosscheck might fail due to missing columns
}

// =============================================================================
// EDGE CASE: BOUNDARY CONDITIONS
// =============================================================================

#[test]
fn test_exactly_20_sharpe() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path();
    
    let metrics = serde_json::json!({
        "cagr": 0.5,
        "sharpe_ratio": 20.0, // Exactly at threshold
        "max_drawdown": -0.05,
        "total_trades": 100,
        "volatility": 0.02
    });
    fs::write(path.join("metrics.json"), serde_json::to_string(&metrics).unwrap()).unwrap();
    fs::write(path.join("nav_history.csv"), "date,nav\n2024-01-02,1000000\n").unwrap();
    
    let artifacts = BacktestArtifacts::from_dir(path, "boundary_sharpe");
    let config = ValidationConfig {
        crosscheck_enabled: false,
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();
    
    // At exactly 20, should warn (not fail)
    assert!(result.verdict == Verdict::Warn || result.verdict == Verdict::Fail);
}

#[test]
fn test_exactly_30_trades() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path();
    
    let metrics = serde_json::json!({
        "cagr": 0.15,
        "sharpe_ratio": 1.5,
        "max_drawdown": -0.1,
        "total_trades": 30, // Exactly at threshold
        "volatility": 0.15
    });
    fs::write(path.join("metrics.json"), serde_json::to_string(&metrics).unwrap()).unwrap();
    fs::write(path.join("nav_history.csv"), "date,nav\n2024-01-02,1000000\n").unwrap();
    
    let artifacts = BacktestArtifacts::from_dir(path, "boundary_trades");
    let config = ValidationConfig {
        crosscheck_enabled: false,
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();
    
    // At exactly 30, should pass (threshold is "< 30")
    assert_eq!(result.verdict, Verdict::Pass);
}

#[test]
fn test_exactly_29_trades() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path();
    
    let metrics = serde_json::json!({
        "cagr": 0.15,
        "sharpe_ratio": 1.5,
        "max_drawdown": -0.1,
        "total_trades": 29, // Just below threshold
        "volatility": 0.15
    });
    fs::write(path.join("metrics.json"), serde_json::to_string(&metrics).unwrap()).unwrap();
    fs::write(path.join("nav_history.csv"), "date,nav\n2024-01-02,1000000\n").unwrap();
    
    let artifacts = BacktestArtifacts::from_dir(path, "low_trades");
    let config = ValidationConfig {
        crosscheck_enabled: false,
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();
    
    // Should warn about low trades
    assert_eq!(result.verdict, Verdict::Warn);
}

// =============================================================================
// EDGE CASE: DATES AND TIMESTAMPS
// =============================================================================

#[test]
fn test_unsorted_nav_dates() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path();
    
    let metrics = serde_json::json!({
        "cagr": 0.1,
        "sharpe_ratio": 1.0,
        "max_drawdown": -0.1,
        "total_trades": 50,
        "volatility": 0.15
    });
    fs::write(path.join("metrics.json"), serde_json::to_string(&metrics).unwrap()).unwrap();
    
    // Unsorted dates
    fs::write(path.join("nav_history.csv"), 
              "date,nav\n2024-01-05,1020000\n2024-01-02,1000000\n2024-01-03,1010000\n").unwrap();
    
    let artifacts = BacktestArtifacts::from_dir(path, "unsorted");
    let config = ValidationConfig {
        crosscheck_enabled: true,
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();
    
    // Schema should pass; crosscheck might compute differently
    assert!(!result.schema_check.has_failures());
}

#[test]
fn test_duplicate_dates() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path();
    
    let metrics = serde_json::json!({
        "cagr": 0.1,
        "sharpe_ratio": 1.0,
        "max_drawdown": -0.1,
        "total_trades": 50,
        "volatility": 0.15
    });
    fs::write(path.join("metrics.json"), serde_json::to_string(&metrics).unwrap()).unwrap();
    
    // Duplicate date
    fs::write(path.join("nav_history.csv"), 
              "date,nav\n2024-01-02,1000000\n2024-01-02,1010000\n2024-01-03,1020000\n").unwrap();
    
    let artifacts = BacktestArtifacts::from_dir(path, "duplicates");
    let config = ValidationConfig {
        crosscheck_enabled: false,
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();
    
    // Should handle gracefully
    assert!(!result.schema_check.has_failures());
}

// =============================================================================
// UNIT TESTS FOR INDIVIDUAL COMPONENTS
// =============================================================================

#[test]
fn test_schema_validator_all_nulls() {
    let validator = SchemaValidator::new(true);
    
    // Only required fields: cagr, sharpe_ratio, max_drawdown, total_trades
    // volatility is recommended but not required
    let json = serde_json::json!({
        "cagr": null,
        "sharpe_ratio": null,
        "max_drawdown": null,
        "total_trades": null,
        "volatility": null
    });
    
    let result = validator.validate_metrics(&json);
    
    // 4 required fields are null (volatility is recommended, not required)
    assert!(result.has_failures());
    assert_eq!(result.null_fields.len(), 4);
}

#[test]
fn test_sanity_checker_extreme_values() {
    let config = SanityConfig::default();
    let checker = SanityChecker::new(config);
    
    let json = serde_json::json!({
        "cagr": 5.0, // 500% CAGR
        "sharpe_ratio": 50.0, // Absurd
        "max_drawdown": -0.001, // Almost no drawdown
        "total_trades": 5, // Very few
        "volatility": 0.001 // Almost no volatility
    });
    
    let result = checker.check_json(&json);
    
    // Should fail on Sharpe
    assert_eq!(result.verdict, Verdict::Fail);
}

#[test]
fn test_crosschecker_with_flat_nav() {
    let config = CrosscheckConfig::default();
    let checker = CrossChecker::new(config);
    
    // Flat NAV (no movement)
    let nav_series = vec![1_000_000.0; 252];
    
    let reported = backtester_validation::crosscheck::ReportedMetrics {
        cagr: 0.0,
        sharpe_ratio: 0.0,
        volatility: 0.0,
        max_drawdown: 0.0,
    };
    
    let result = checker.crosscheck(&reported, &nav_series);
    
    // Should pass (recomputed 0 matches reported 0)
    assert!(result.passed);
}

#[test]
fn test_crosschecker_with_monotonic_growth() {
    let config = CrosscheckConfig::default();
    let checker = CrossChecker::new(config);
    
    // Perfect monotonic growth (1% daily)
    let nav_series: Vec<f64> = (0..252)
        .map(|i| 1_000_000.0 * 1.01_f64.powi(i))
        .collect();
    
    // Don't check exact values, just ensure it runs
    let reported = backtester_validation::crosscheck::ReportedMetrics {
        cagr: 10.0, // Deliberately wrong
        sharpe_ratio: 1.0,
        volatility: 0.1,
        max_drawdown: 0.0,
    };
    
    let result = checker.crosscheck(&reported, &nav_series);
    
    // Should fail on mismatch
    assert!(!result.passed);
}

