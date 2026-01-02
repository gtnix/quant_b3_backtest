//! Integration tests for the validation module.

use backtester_validation::{
    BacktestArtifacts, ValidationPipeline, ValidationConfig, Verdict,
    sanity::{SanityChecker, SanityConfig, ParsedMetrics},
    schema::SchemaValidator,
    crosscheck::CrossChecker,
};
use tempfile::TempDir;
use std::fs;

/// Create a complete test run with realistic data.
fn create_realistic_run(dir: &std::path::Path) -> BacktestArtifacts {
    // Create metrics.json with all fields
    let metrics = serde_json::json!({
        "cagr": 0.152,
        "volatility": 0.185,
        "sharpe_ratio": 0.82,
        "max_drawdown": -0.223,
        "sortino_ratio": 1.15,
        "calmar_ratio": 0.68,
        "profit_factor": 1.45,
        "turnover_annual": 2.3,
        "total_trades": 342,
        "winning_trades": 185,
        "losing_trades": 157,
        "hit_rate": 0.541,
        "final_nav": 1520000.0,
        "initial_capital": 1000000.0
    });
    fs::write(dir.join("metrics.json"), serde_json::to_string_pretty(&metrics).unwrap()).unwrap();

    // Create nav_history.csv with 252 days of data (1 year)
    let mut nav_csv = String::from("date,nav\n");
    let mut nav = 1_000_000.0;
    for day in 0..252 {
        let date = format!("2023-{:02}-{:02}", (day / 22) + 1, (day % 22) + 1);
        // Simulate realistic daily returns with some noise
        let daily_return = 0.0006 + 0.012 * (day as f64 / 252.0 * 6.28).sin() * 0.1;
        nav *= 1.0 + daily_return;
        nav_csv.push_str(&format!("{},{:.2}\n", date, nav));
    }
    fs::write(dir.join("nav_history.csv"), nav_csv).unwrap();

    // Create trades.csv
    let mut trades_csv = String::from("date,symbol,side,quantity,price,net_pnl,gross_pnl,cost\n");
    let symbols = ["PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3"];
    for i in 0..342 {
        let symbol = symbols[i % symbols.len()];
        let pnl = if i % 3 == 0 { 500.0 + (i as f64 * 10.0) } else { -300.0 - (i as f64 * 5.0) };
        let cost = 15.0;
        trades_csv.push_str(&format!(
            "2023-{:02}-{:02},{},BUY,100,25.50,{:.2},{:.2},{:.2}\n",
            (i / 30) + 1, (i % 28) + 1, symbol, pnl, pnl + cost, cost
        ));
    }
    fs::write(dir.join("trades.csv"), trades_csv).unwrap();

    // Create manifest.json
    let manifest = serde_json::json!({
        "run_id": "test_realistic_001",
        "strategy_id": "momentum_quality",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "initial_capital": 1000000.0,
        "config_hash": "sha256:abc123",
        "created_at": "2024-01-01T12:00:00Z"
    });
    fs::write(dir.join("manifest.json"), serde_json::to_string_pretty(&manifest).unwrap()).unwrap();

    BacktestArtifacts::from_dir(dir, "test_realistic_001")
}

/// Create a run with suspicious metrics (Sharpe > 10).
fn create_suspicious_run(dir: &std::path::Path) -> BacktestArtifacts {
    let metrics = serde_json::json!({
        "cagr": 0.85,
        "volatility": 0.03,  // Suspiciously low
        "sharpe_ratio": 15.5, // > 10, suspicious
        "max_drawdown": -0.05,
        "total_trades": 25  // Too few trades
    });
    fs::write(dir.join("metrics.json"), serde_json::to_string_pretty(&metrics).unwrap()).unwrap();

    // Minimal nav_history
    fs::write(dir.join("nav_history.csv"), "date,nav\n2023-01-01,1000000\n2023-12-31,1850000\n").unwrap();

    BacktestArtifacts::from_dir(dir, "test_suspicious_001")
}

/// Create a run with absurd metrics (Sharpe > 20).
fn create_absurd_run(dir: &std::path::Path) -> BacktestArtifacts {
    let metrics = serde_json::json!({
        "cagr": 1.5,
        "volatility": 0.005,  // Extremely low
        "sharpe_ratio": 25.0, // > 20, almost certainly a bug
        "max_drawdown": -0.02,
        "total_trades": 10  // Very few trades
    });
    fs::write(dir.join("metrics.json"), serde_json::to_string_pretty(&metrics).unwrap()).unwrap();

    // Minimal nav_history
    fs::write(dir.join("nav_history.csv"), "date,nav\n2023-01-01,1000000\n2023-12-31,2500000\n").unwrap();

    BacktestArtifacts::from_dir(dir, "test_absurd_001")
}

/// Create a run with null required fields.
fn create_null_fields_run(dir: &std::path::Path) -> BacktestArtifacts {
    let metrics = serde_json::json!({
        "cagr": 0.15,
        "sharpe_ratio": null,  // This is a required field!
        "max_drawdown": -0.12,
        "total_trades": 100
    });
    fs::write(dir.join("metrics.json"), serde_json::to_string(&metrics).unwrap()).unwrap();

    fs::write(dir.join("nav_history.csv"), "date,nav\n2023-01-01,1000000\n").unwrap();

    BacktestArtifacts::from_dir(dir, "test_null_001")
}

#[test]
fn test_realistic_run_passes() {
    let temp_dir = TempDir::new().unwrap();
    let artifacts = create_realistic_run(temp_dir.path());

    let config = ValidationConfig {
        crosscheck_enabled: false, // Skip crosscheck for this test
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();

    assert!(
        result.verdict == Verdict::Pass || result.verdict == Verdict::Warn,
        "Realistic run should pass or warn, got {:?}",
        result.verdict
    );
    assert!(!result.schema_check.has_failures());
    assert!(!result.sanity_check.flags.sharpe_absurd);
}

#[test]
fn test_suspicious_sharpe_warns() {
    let temp_dir = TempDir::new().unwrap();
    let artifacts = create_suspicious_run(temp_dir.path());

    let config = ValidationConfig {
        crosscheck_enabled: false,
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();

    assert!(
        result.sanity_check.flags.sharpe_suspicious,
        "Sharpe 15.5 should trigger suspicious flag"
    );
    assert!(
        result.verdict == Verdict::Warn || result.verdict == Verdict::Fail,
        "Should warn or fail on suspicious metrics"
    );
}

#[test]
fn test_absurd_sharpe_fails() {
    let temp_dir = TempDir::new().unwrap();
    let artifacts = create_absurd_run(temp_dir.path());

    let config = ValidationConfig {
        crosscheck_enabled: false,
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();

    assert!(
        result.sanity_check.flags.sharpe_absurd,
        "Sharpe 25.0 should trigger absurd flag"
    );
    assert_eq!(
        result.verdict,
        Verdict::Fail,
        "Absurd Sharpe should cause validation failure"
    );
}

#[test]
fn test_null_required_field_fails() {
    let temp_dir = TempDir::new().unwrap();
    let artifacts = create_null_fields_run(temp_dir.path());

    let pipeline = ValidationPipeline::default();
    let result = pipeline.validate(&artifacts).unwrap();

    assert!(
        result.schema_check.has_failures(),
        "Null required field should fail schema check"
    );
    assert!(
        result.schema_check.null_fields.contains(&"sharpe_ratio".to_string()),
        "sharpe_ratio should be in null fields"
    );
    assert_eq!(result.verdict, Verdict::Fail);
}

#[test]
fn test_strict_mode_fails_on_warnings() {
    let temp_dir = TempDir::new().unwrap();
    let artifacts = create_suspicious_run(temp_dir.path());

    let config = ValidationConfig {
        strict_mode: true,
        crosscheck_enabled: false,
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();

    // In strict mode, warnings become failures
    assert_eq!(
        result.verdict,
        Verdict::Fail,
        "Strict mode should fail on warnings"
    );
}

#[test]
fn test_attribution_generates_csv() {
    let temp_dir = TempDir::new().unwrap();
    let artifacts = create_realistic_run(temp_dir.path());

    let config = ValidationConfig {
        crosscheck_enabled: false,
        ..Default::default()
    };
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts).unwrap();

    // Check attribution was calculated
    assert!(result.attribution.is_some());
    let attr = result.attribution.as_ref().unwrap();
    assert!(!attr.attributions.is_empty());
    assert!(attr.total_trades > 0);

    // Generate artifacts
    let output_dir = temp_dir.path().join("validation");
    pipeline.generate_artifacts(&result, &output_dir).unwrap();

    // Check files were created
    assert!(output_dir.join("validation_summary.json").exists());
    assert!(output_dir.join("sanity.json").exists());
    assert!(output_dir.join("asset_attribution.csv").exists());
    assert!(output_dir.join("backtest_report.md").exists());
}

#[test]
fn test_sanity_checker_thresholds() {
    let config = SanityConfig {
        sharpe_warn_threshold: 5.0,
        sharpe_fail_threshold: 10.0,
        min_volatility: 0.02,
        min_trades: 50,
        max_cagr: 1.0,
        max_calmar: 5.0,
    };
    let checker = SanityChecker::new(config);

    // Normal case
    let normal = ParsedMetrics {
        sharpe_ratio: 2.0,
        volatility: 0.15,
        cagr: 0.20,
        total_trades: 100,
        max_drawdown: -0.15,
        calmar_ratio: Some(1.3),
    };
    let result = checker.check(&normal);
    assert_eq!(result.verdict, Verdict::Pass);

    // Warn case: Sharpe > 5
    let warn = ParsedMetrics {
        sharpe_ratio: 7.0,
        volatility: 0.15,
        cagr: 0.20,
        total_trades: 100,
        max_drawdown: -0.15,
        calmar_ratio: None,
    };
    let result = checker.check(&warn);
    assert_eq!(result.verdict, Verdict::Warn);
    assert!(result.flags.sharpe_suspicious);

    // Fail case: Sharpe > 10
    let fail = ParsedMetrics {
        sharpe_ratio: 15.0,
        volatility: 0.15,
        cagr: 0.20,
        total_trades: 100,
        max_drawdown: -0.15,
        calmar_ratio: None,
    };
    let result = checker.check(&fail);
    assert_eq!(result.verdict, Verdict::Fail);
    assert!(result.flags.sharpe_absurd);
}

#[test]
fn test_crosscheck_detects_mismatch() {
    let checker = CrossChecker::default();

    // Create NAV series
    let nav: Vec<f64> = (0..252)
        .map(|i| 1_000_000.0_f64 * (1.0_f64 + 0.15_f64 / 252.0_f64).powi(i))
        .collect();

    // Correct metrics should match
    let recomputed = checker.compute_metrics(&nav);
    
    // Reported metrics with wrong values
    let wrong_reported = backtester_validation::crosscheck::ReportedMetrics {
        cagr: 0.50, // Way off from actual ~0.15
        volatility: 0.30,
        sharpe_ratio: 2.5,
        max_drawdown: -0.10,
    };

    let result = checker.crosscheck(&wrong_reported, &nav);
    assert!(!result.passed, "Should detect mismatch");
    assert!(!result.warnings.is_empty());
}

#[test]
fn test_schema_validator_required_fields() {
    let validator = SchemaValidator::default();

    // All required fields present
    let valid = serde_json::json!({
        "cagr": 0.15,
        "sharpe_ratio": 1.2,
        "max_drawdown": -0.12,
        "total_trades": 100
    });
    let result = validator.validate_metrics(&valid);
    assert!(result.passed);
    assert!(result.missing_fields.is_empty());
    assert!(result.null_fields.is_empty());

    // Missing sharpe_ratio
    let missing = serde_json::json!({
        "cagr": 0.15,
        "max_drawdown": -0.12,
        "total_trades": 100
    });
    let result = validator.validate_metrics(&missing);
    assert!(!result.passed);
    assert!(result.missing_fields.contains(&"sharpe_ratio".to_string()));

    // Null sharpe_ratio
    let null_field = serde_json::json!({
        "cagr": 0.15,
        "sharpe_ratio": null,
        "max_drawdown": -0.12,
        "total_trades": 100
    });
    let result = validator.validate_metrics(&null_field);
    assert!(!result.passed);
    assert!(result.null_fields.contains(&"sharpe_ratio".to_string()));
}

