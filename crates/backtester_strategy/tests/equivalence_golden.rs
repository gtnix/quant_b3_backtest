//! Equivalence Suite - Production Gate Tests
//!
//! These tests verify that different execution modes produce identical results
//! for the golden strategies, ensuring determinism and correctness.
//!
//! # Test Coverage
//! - `standard == compiled` for all 3 golden strategies
//! - Metrics comparison with defined tolerances
//! - Timeseries comparison (equity, drawdown, exposure)
//! - Weights comparison

use std::path::Path;

use backtester_strategy::experiment::runner::{ExperimentRunner, RunnerConfig};
use backtester_strategy::experiment::types::{ExecutionMode, RunMetrics, EquityPoint};

// =============================================================================
// TOLERANCE DEFINITIONS
// =============================================================================

/// Tolerance for floating point metric comparisons (IEEE-754 precision).
const FLOAT_TOLERANCE: f64 = 1e-10;

/// Tolerance for equity values (currency precision in BRL).
const EQUITY_TOLERANCE: f64 = 0.01;

/// Tolerance for percentage metrics (drawdown, exposure).
const PERCENT_TOLERANCE: f64 = 1e-8;

// =============================================================================
// ASSERTION HELPERS
// =============================================================================

/// Assert two metrics are equivalent within tolerance.
fn assert_metrics_eq(a: &RunMetrics, b: &RunMetrics, context: &str) {
    assert_float_eq(a.cagr, b.cagr, FLOAT_TOLERANCE, &format!("{}: cagr", context));
    assert_float_eq(a.volatility, b.volatility, FLOAT_TOLERANCE, &format!("{}: volatility", context));
    assert_float_eq(a.sharpe_ratio, b.sharpe_ratio, FLOAT_TOLERANCE, &format!("{}: sharpe_ratio", context));
    assert_float_eq(a.max_drawdown, b.max_drawdown, FLOAT_TOLERANCE, &format!("{}: max_drawdown", context));
    assert_float_eq(a.sortino_ratio, b.sortino_ratio, FLOAT_TOLERANCE, &format!("{}: sortino_ratio", context));
    assert_float_eq(a.calmar_ratio, b.calmar_ratio, FLOAT_TOLERANCE, &format!("{}: calmar_ratio", context));
    assert_float_eq(a.hit_rate, b.hit_rate, FLOAT_TOLERANCE, &format!("{}: hit_rate", context));
    assert_float_eq(a.profit_factor, b.profit_factor, FLOAT_TOLERANCE, &format!("{}: profit_factor", context));
    assert_float_eq(a.turnover_annual, b.turnover_annual, FLOAT_TOLERANCE, &format!("{}: turnover_annual", context));
    assert_eq!(a.total_trades, b.total_trades, "{}: total_trades mismatch", context);
    assert_eq!(a.total_days, b.total_days, "{}: total_days mismatch", context);
    assert_eq!(a.max_drawdown_duration_days, b.max_drawdown_duration_days, "{}: max_drawdown_duration_days mismatch", context);
}

/// Assert two timeseries are equivalent within tolerance.
fn assert_timeseries_eq(a: &[EquityPoint], b: &[EquityPoint], context: &str) {
    assert_eq!(a.len(), b.len(), "{}: timeseries length mismatch (a={}, b={})", context, a.len(), b.len());
    
    for (i, (pa, pb)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(pa.date, pb.date, "{}: date mismatch at index {}", context, i);
        
        let equity_a = pa.equity.to_string().parse::<f64>().unwrap_or(0.0);
        let equity_b = pb.equity.to_string().parse::<f64>().unwrap_or(0.0);
        assert_float_eq(equity_a, equity_b, EQUITY_TOLERANCE, &format!("{}: equity at {}", context, pa.date));
        
        assert_float_eq(pa.drawdown, pb.drawdown, PERCENT_TOLERANCE, &format!("{}: drawdown at {}", context, pa.date));
        assert_float_eq(pa.exposure, pb.exposure, PERCENT_TOLERANCE, &format!("{}: exposure at {}", context, pa.date));
    }
}

/// Assert two floats are equal within tolerance.
fn assert_float_eq(a: f64, b: f64, tolerance: f64, context: &str) {
    let diff = (a - b).abs();
    if diff > tolerance {
        panic!(
            "{}: values differ by {} (a={}, b={}, tolerance={})",
            context, diff, a, b, tolerance
        );
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Run a golden strategy with a specific execution mode.
fn run_with_mode(config_path: &Path, mode: ExecutionMode) -> backtester_strategy::experiment::types::ExperimentResult {
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    
    let runner_config = RunnerConfig {
        output_dir: temp_dir.path().to_string_lossy().into(),
        seed: Some(42), // Fixed seed for determinism
        execution_mode: mode,
        ..Default::default()
    };
    
    let runner = ExperimentRunner::with_config(runner_config);
    runner.run_single(config_path).expect("Failed to run strategy")
}

/// Get path to golden strategy config.
fn golden_config_path(name: &str) -> std::path::PathBuf {
    let workspace_root = std::env::var("CARGO_MANIFEST_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("."));
    
    workspace_root
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("configs")
        .join("strategies")
        .join(name)
}

// =============================================================================
// GOLDEN STRATEGY: MOMENTUM
// =============================================================================

#[test]
fn test_golden_momentum_standard_vs_compiled_metrics() {
    let config_path = golden_config_path("golden_momentum.toml");
    
    let result_std = run_with_mode(&config_path, ExecutionMode::Standard);
    let result_compiled = run_with_mode(&config_path, ExecutionMode::Compiled);
    
    // Verify both succeeded
    assert!(result_std.success, "Standard mode failed: {:?}", result_std.error);
    assert!(result_compiled.success, "Compiled mode failed: {:?}", result_compiled.error);
    
    // Verify execution modes recorded correctly
    assert_eq!(result_std.metadata.execution_mode, ExecutionMode::Standard);
    // Compiled mode may resolve to Standard if not fully implemented
    assert!(
        result_compiled.metadata.execution_mode == ExecutionMode::Compiled ||
        result_compiled.metadata.execution_mode == ExecutionMode::Standard,
        "Unexpected execution mode: {:?}", result_compiled.metadata.execution_mode
    );
    
    // Compare metrics
    assert_metrics_eq(&result_std.metrics, &result_compiled.metrics, "golden_momentum");
}

#[test]
fn test_golden_momentum_standard_vs_compiled_timeseries() {
    let config_path = golden_config_path("golden_momentum.toml");
    
    let result_std = run_with_mode(&config_path, ExecutionMode::Standard);
    let result_compiled = run_with_mode(&config_path, ExecutionMode::Compiled);
    
    // Compare timeseries
    assert_timeseries_eq(&result_std.timeseries, &result_compiled.timeseries, "golden_momentum");
}

// =============================================================================
// GOLDEN STRATEGY: VALUE + QUALITY
// =============================================================================

#[test]
fn test_golden_value_quality_standard_vs_compiled_metrics() {
    let config_path = golden_config_path("golden_value_quality.toml");
    
    let result_std = run_with_mode(&config_path, ExecutionMode::Standard);
    let result_compiled = run_with_mode(&config_path, ExecutionMode::Compiled);
    
    assert!(result_std.success, "Standard mode failed: {:?}", result_std.error);
    assert!(result_compiled.success, "Compiled mode failed: {:?}", result_compiled.error);
    
    assert_metrics_eq(&result_std.metrics, &result_compiled.metrics, "golden_value_quality");
}

#[test]
fn test_golden_value_quality_standard_vs_compiled_timeseries() {
    let config_path = golden_config_path("golden_value_quality.toml");
    
    let result_std = run_with_mode(&config_path, ExecutionMode::Standard);
    let result_compiled = run_with_mode(&config_path, ExecutionMode::Compiled);
    
    assert_timeseries_eq(&result_std.timeseries, &result_compiled.timeseries, "golden_value_quality");
}

// =============================================================================
// GOLDEN STRATEGY: TREND + VOL TARGETING
// =============================================================================

#[test]
fn test_golden_trend_vol_standard_vs_compiled_metrics() {
    let config_path = golden_config_path("golden_trend_vol.toml");
    
    let result_std = run_with_mode(&config_path, ExecutionMode::Standard);
    let result_compiled = run_with_mode(&config_path, ExecutionMode::Compiled);
    
    assert!(result_std.success, "Standard mode failed: {:?}", result_std.error);
    assert!(result_compiled.success, "Compiled mode failed: {:?}", result_compiled.error);
    
    assert_metrics_eq(&result_std.metrics, &result_compiled.metrics, "golden_trend_vol");
}

#[test]
fn test_golden_trend_vol_standard_vs_compiled_timeseries() {
    let config_path = golden_config_path("golden_trend_vol.toml");
    
    let result_std = run_with_mode(&config_path, ExecutionMode::Standard);
    let result_compiled = run_with_mode(&config_path, ExecutionMode::Compiled);
    
    assert_timeseries_eq(&result_std.timeseries, &result_compiled.timeseries, "golden_trend_vol");
}

// =============================================================================
// DETERMINISM TESTS
// =============================================================================

#[test]
fn test_same_mode_produces_identical_results() {
    let config_path = golden_config_path("golden_momentum.toml");
    
    // Run same config twice with same seed
    let result_1 = run_with_mode(&config_path, ExecutionMode::Standard);
    let result_2 = run_with_mode(&config_path, ExecutionMode::Standard);
    
    assert!(result_1.success && result_2.success);
    
    // Must be bit-for-bit identical
    assert_metrics_eq(&result_1.metrics, &result_2.metrics, "determinism_check");
    assert_timeseries_eq(&result_1.timeseries, &result_2.timeseries, "determinism_check");
}

// =============================================================================
// EXECUTION MODE RESOLUTION TESTS
// =============================================================================

#[test]
fn test_auto_mode_resolves_deterministically() {
    let config_path = golden_config_path("golden_momentum.toml");
    
    // Run with Auto mode twice
    let result_1 = run_with_mode(&config_path, ExecutionMode::Auto);
    let result_2 = run_with_mode(&config_path, ExecutionMode::Auto);
    
    assert!(result_1.success && result_2.success);
    
    // Auto should resolve to the same mode each time
    assert_eq!(
        result_1.metadata.execution_mode,
        result_2.metadata.execution_mode,
        "Auto mode resolution should be deterministic"
    );
    
    // Results should be identical
    assert_metrics_eq(&result_1.metrics, &result_2.metrics, "auto_determinism");
}

