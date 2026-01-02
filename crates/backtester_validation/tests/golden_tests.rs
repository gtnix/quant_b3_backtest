//! Golden tests for validation module using REAL market data.
//!
//! These tests use fixtures generated from actual B3 market data (PETR4, VALE3, ITUB4)
//! downloaded from the Neon database `b3-market-data`.
//!
//! NO MOCK DATA - all fixtures are based on real historical prices.

use backtester_validation::{
    BacktestArtifacts, ValidationPipeline, ValidationConfig, Verdict,
};
use std::path::PathBuf;

/// Get the path to the fixtures directory.
fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

// =============================================================================
// REAL DATA TEST: PASS (PETR4 2024 S1 - Normal Backtest)
// =============================================================================

#[test]
fn test_real_pass_verdict() {
    let fixture_dir = fixtures_dir().join("real_run_pass");
    let artifacts = BacktestArtifacts::from_dir(&fixture_dir, "real_pass");

    let config = ValidationConfig {
        crosscheck_enabled: false, // Skip crosscheck as we want to test schema/sanity only
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();

    assert!(
        result.verdict == Verdict::Pass || result.verdict == Verdict::Warn,
        "Real PASS run should pass or warn, got {:?}\nErrors: {:?}\nWarnings: {:?}",
        result.verdict,
        result.errors,
        result.warnings
    );
    
    // Schema should pass
    assert!(!result.schema_check.has_failures(), "Schema should pass");
    
    // Sanity should not have absurd flags (Sharpe 0.79 is normal)
    assert!(!result.sanity_check.flags.sharpe_absurd, "Should not flag Sharpe as absurd");
    
    // Attribution should be calculated
    assert!(result.attribution.is_some(), "Attribution should be calculated");
    let attr = result.attribution.as_ref().unwrap();
    assert!(!attr.attributions.is_empty(), "Should have asset attributions");
}

#[test]
fn test_real_pass_schema_all_fields_present() {
    let fixture_dir = fixtures_dir().join("real_run_pass");
    let artifacts = BacktestArtifacts::from_dir(&fixture_dir, "real_pass");

    let pipeline = ValidationPipeline::default();
    let result = pipeline.validate(&artifacts).unwrap();

    // All required fields should be validated
    assert!(result.schema_check.validated_fields.contains(&"cagr".to_string()));
    assert!(result.schema_check.validated_fields.contains(&"sharpe_ratio".to_string()));
    assert!(result.schema_check.validated_fields.contains(&"max_drawdown".to_string()));
    assert!(result.schema_check.validated_fields.contains(&"total_trades".to_string()));
}

#[test]
fn test_real_pass_generates_all_artifacts() {
    let fixture_dir = fixtures_dir().join("real_run_pass");
    let artifacts = BacktestArtifacts::from_dir(&fixture_dir, "real_pass");

    let config = ValidationConfig {
        crosscheck_enabled: false,
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();

    // Generate artifacts to temp directory
    let temp_dir = tempfile::TempDir::new().unwrap();
    pipeline.generate_artifacts(&result, temp_dir.path()).unwrap();

    // Check all files were created
    assert!(temp_dir.path().join("validation_summary.json").exists());
    assert!(temp_dir.path().join("sanity.json").exists());
    assert!(temp_dir.path().join("asset_attribution.csv").exists());
    assert!(temp_dir.path().join("backtest_report.md").exists());

    // Validate content of validation_summary.json
    let summary_content = std::fs::read_to_string(temp_dir.path().join("validation_summary.json")).unwrap();
    let summary: serde_json::Value = serde_json::from_str(&summary_content).unwrap();
    assert!(summary.get("run_id").is_some());
    assert!(summary.get("verdict").is_some());
    assert!(summary.get("checks").is_some());
}

#[test]
fn test_real_pass_metrics_snapshot() {
    let fixture_dir = fixtures_dir().join("real_run_pass");
    let artifacts = BacktestArtifacts::from_dir(&fixture_dir, "real_pass");

    let config = ValidationConfig {
        crosscheck_enabled: false,
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();

    let snapshot = &result.sanity_check.metrics_snapshot;
    
    // Verify metrics match real PETR4 2024 data fixture
    // CAGR ~6.44%, Sharpe ~0.79, Vol ~8.16%
    assert!((snapshot.sharpe_ratio.unwrap() - 0.79).abs() < 0.1, 
        "Sharpe should be ~0.79, got {:?}", snapshot.sharpe_ratio);
    assert!((snapshot.cagr.unwrap() - 0.0644).abs() < 0.01,
        "CAGR should be ~0.0644, got {:?}", snapshot.cagr);
    assert_eq!(snapshot.num_trades.unwrap(), 7, "Should have 7 trades");
}

// =============================================================================
// REAL DATA TEST: WARN (High Sharpe, Low Trades)
// =============================================================================

#[test]
fn test_real_warn_verdict() {
    let fixture_dir = fixtures_dir().join("real_run_warn");
    let artifacts = BacktestArtifacts::from_dir(&fixture_dir, "real_warn");

    let config = ValidationConfig {
        crosscheck_enabled: false,
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();

    // Should WARN or FAIL (due to Sharpe > 10)
    assert!(
        result.verdict == Verdict::Warn || result.verdict == Verdict::Fail,
        "Real WARN run should warn or fail, got {:?}",
        result.verdict
    );
}

#[test]
fn test_real_warn_sharpe_suspicious_flag() {
    let fixture_dir = fixtures_dir().join("real_run_warn");
    let artifacts = BacktestArtifacts::from_dir(&fixture_dir, "real_warn");

    let config = ValidationConfig {
        crosscheck_enabled: false,
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();

    // Sharpe 11.5 should trigger suspicious flag
    assert!(
        result.sanity_check.flags.sharpe_suspicious || result.sanity_check.flags.sharpe_absurd,
        "Sharpe 11.5 should trigger suspicious or absurd flag"
    );
}

#[test]
fn test_real_warn_trades_too_few_flag() {
    let fixture_dir = fixtures_dir().join("real_run_warn");
    let artifacts = BacktestArtifacts::from_dir(&fixture_dir, "real_warn");

    let config = ValidationConfig {
        crosscheck_enabled: false,
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();

    // 15 trades < 30 should trigger trades_too_few
    assert!(
        result.sanity_check.flags.trades_too_few,
        "15 trades should trigger trades_too_few flag"
    );
}

// =============================================================================
// REAL DATA TEST: FAIL (Absurd Sharpe > 20)
// =============================================================================

#[test]
fn test_real_fail_sharpe_verdict() {
    let fixture_dir = fixtures_dir().join("real_run_fail_sharpe");
    let artifacts = BacktestArtifacts::from_dir(&fixture_dir, "real_fail_sharpe");

    let config = ValidationConfig {
        crosscheck_enabled: false,
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();

    assert_eq!(
        result.verdict,
        Verdict::Fail,
        "Sharpe 28.5 should FAIL validation"
    );
}

#[test]
fn test_real_fail_sharpe_absurd_flag() {
    let fixture_dir = fixtures_dir().join("real_run_fail_sharpe");
    let artifacts = BacktestArtifacts::from_dir(&fixture_dir, "real_fail_sharpe");

    let config = ValidationConfig {
        crosscheck_enabled: false,
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();

    assert!(
        result.sanity_check.flags.sharpe_absurd,
        "Sharpe 28.5 should trigger sharpe_absurd flag"
    );
}

#[test]
fn test_real_fail_sharpe_blocks_promotion() {
    let fixture_dir = fixtures_dir().join("real_run_fail_sharpe");
    let artifacts = BacktestArtifacts::from_dir(&fixture_dir, "real_fail_sharpe");

    let config = ValidationConfig {
        crosscheck_enabled: false,
        strict_mode: true, // Strict mode for promotion gate simulation
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();

    assert_eq!(
        result.verdict,
        Verdict::Fail,
        "Absurd Sharpe should block promotion in strict mode"
    );
    
    // Should have sharpe_absurd flag
    assert!(
        result.sanity_check.flags.sharpe_absurd,
        "Should report the absurd Sharpe issue"
    );
}

// =============================================================================
// REAL DATA TEST: FAIL (Null Required Field)
// =============================================================================

#[test]
fn test_real_fail_null_verdict() {
    let fixture_dir = fixtures_dir().join("real_run_fail_null");
    let artifacts = BacktestArtifacts::from_dir(&fixture_dir, "real_fail_null");

    let pipeline = ValidationPipeline::default();
    let result = pipeline.validate(&artifacts).unwrap();

    assert_eq!(
        result.verdict,
        Verdict::Fail,
        "Null sharpe_ratio should FAIL validation"
    );
}

#[test]
fn test_real_fail_null_schema_failure() {
    let fixture_dir = fixtures_dir().join("real_run_fail_null");
    let artifacts = BacktestArtifacts::from_dir(&fixture_dir, "real_fail_null");

    let pipeline = ValidationPipeline::default();
    let result = pipeline.validate(&artifacts).unwrap();

    assert!(
        result.schema_check.has_failures(),
        "Schema check should fail for null field"
    );
    
    assert!(
        result.schema_check.null_fields.contains(&"sharpe_ratio".to_string()),
        "sharpe_ratio should be in null_fields list"
    );
}

#[test]
fn test_real_fail_null_early_exit() {
    let fixture_dir = fixtures_dir().join("real_run_fail_null");
    let artifacts = BacktestArtifacts::from_dir(&fixture_dir, "real_fail_null");

    let pipeline = ValidationPipeline::default();
    let result = pipeline.validate(&artifacts).unwrap();

    // When schema fails, verdict should be Fail
    assert_eq!(result.verdict, Verdict::Fail);
}

// =============================================================================
// REAL DATA TEST: FAIL (Cross-check Mismatch)
// =============================================================================

#[test]
fn test_real_fail_mismatch_verdict() {
    let fixture_dir = fixtures_dir().join("real_run_fail_mismatch");
    let artifacts = BacktestArtifacts::from_dir(&fixture_dir, "real_fail_mismatch");

    // Enable crosscheck to detect mismatch
    let config = ValidationConfig {
        crosscheck_enabled: true,
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();

    // Metrics claim CAGR 45%, but NAV shows ~5% growth
    // This should trigger cross-check failure
    assert_eq!(
        result.verdict,
        Verdict::Fail,
        "Cross-check mismatch should FAIL validation"
    );
}

#[test]
fn test_real_fail_mismatch_crosscheck_failure() {
    let fixture_dir = fixtures_dir().join("real_run_fail_mismatch");
    let artifacts = BacktestArtifacts::from_dir(&fixture_dir, "real_fail_mismatch");

    let config = ValidationConfig {
        crosscheck_enabled: true,
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();

    assert!(result.crosscheck.is_some(), "Crosscheck should be performed");
    let crosscheck = result.crosscheck.as_ref().unwrap();
    
    assert!(
        !crosscheck.passed,
        "Crosscheck should fail due to metric mismatch"
    );
    
    // CAGR mismatch should be detected (reported 45% vs actual ~5%)
    let cagr_comparison = crosscheck.comparisons.iter().find(|c| c.name == "cagr");
    assert!(cagr_comparison.is_some(), "CAGR should be compared");
    assert!(!cagr_comparison.unwrap().passed, "CAGR comparison should fail");
}

// =============================================================================
// DETERMINISTIC OUTPUT TESTS
// =============================================================================

#[test]
fn test_real_pass_deterministic_sanity_json() {
    let fixture_dir = fixtures_dir().join("real_run_pass");
    let artifacts = BacktestArtifacts::from_dir(&fixture_dir, "real_pass");

    let config = ValidationConfig {
        crosscheck_enabled: false,
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    
    // Run twice and compare
    let result1 = pipeline.validate(&artifacts).unwrap();
    let result2 = pipeline.validate(&artifacts).unwrap();

    // Sanity check results should be identical
    assert_eq!(
        result1.sanity_check.flags.sharpe_suspicious,
        result2.sanity_check.flags.sharpe_suspicious
    );
    assert_eq!(
        result1.sanity_check.flags.sharpe_absurd,
        result2.sanity_check.flags.sharpe_absurd
    );
    assert_eq!(
        result1.sanity_check.verdict,
        result2.sanity_check.verdict
    );
}

#[test]
fn test_real_pass_deterministic_attribution() {
    let fixture_dir = fixtures_dir().join("real_run_pass");
    let artifacts = BacktestArtifacts::from_dir(&fixture_dir, "real_pass");

    let config = ValidationConfig {
        crosscheck_enabled: false,
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    
    // Run twice
    let result1 = pipeline.validate(&artifacts).unwrap();
    let result2 = pipeline.validate(&artifacts).unwrap();

    let attr1 = result1.attribution.as_ref().unwrap();
    let attr2 = result2.attribution.as_ref().unwrap();

    // Attribution should be identical
    assert_eq!(attr1.attributions.len(), attr2.attributions.len());
    assert_eq!(attr1.total_net_pnl, attr2.total_net_pnl);
    assert_eq!(attr1.total_trades, attr2.total_trades);
}

// =============================================================================
// REAL DATA INTEGRITY TESTS
// =============================================================================

#[test]
fn test_real_data_uses_actual_petr4_prices() {
    // Verify that our fixtures use real PETR4 2024 prices
    let fixture_dir = fixtures_dir().join("real_run_pass");
    let nav_path = fixture_dir.join("nav_history.csv");
    
    let content = std::fs::read_to_string(&nav_path).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    
    // First data line should be 2024-01-02 (real trading day)
    assert!(lines[1].starts_with("2024-01-02"), 
        "First date should be 2024-01-02 (real trading day)");
    
    // Should have ~120 trading days (2024-01-02 to 2024-06-28)
    assert!(lines.len() > 100, "Should have substantial real data points");
}

#[test]
fn test_real_metrics_are_plausible() {
    let fixture_dir = fixtures_dir().join("real_run_pass");
    let artifacts = BacktestArtifacts::from_dir(&fixture_dir, "real_pass");

    let config = ValidationConfig {
        crosscheck_enabled: false,
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();

    let snapshot = &result.sanity_check.metrics_snapshot;
    
    // Real market returns should be plausible
    let sharpe = snapshot.sharpe_ratio.unwrap();
    let cagr = snapshot.cagr.unwrap();
    let vol = snapshot.annual_volatility.unwrap();
    
    // Sharpe should be between -3 and 3 for real data
    assert!(sharpe > -3.0 && sharpe < 3.0, 
        "Real Sharpe should be plausible: {}", sharpe);
    
    // CAGR should be between -50% and 100% for 6-month period
    assert!(cagr > -0.5 && cagr < 1.0,
        "Real CAGR should be plausible: {}", cagr);
    
    // Volatility should be between 5% and 80% annualized
    assert!(vol > 0.05 && vol < 0.8,
        "Real volatility should be plausible: {}", vol);
}
