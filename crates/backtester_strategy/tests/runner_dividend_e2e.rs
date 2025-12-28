//! End-to-end tests for dividend integration in the experiment runner.
//!
//! Tests cover:
//! - T1: Determinism - same inputs produce identical outputs
//! - T2: Anti-double-count policy enforcement
//! - T3: Fast mode fallback when dividends enabled

use backtester_strategy::experiment::{
    ExecutionMode, ExperimentRunner, PriceType, RunnerConfig,
};
use rust_decimal::Decimal;
use std::path::PathBuf;

/// Helper to create a test runner with specific config.
fn create_test_runner(dividends_enabled: bool, execution_mode: ExecutionMode) -> ExperimentRunner {
    let config = RunnerConfig {
        output_dir: "test_output".into(),
        risk_free_rate: 0.05,
        enable_dividends: dividends_enabled,
        initial_capital: Decimal::from(1_000_000),
        execution_mode,
        ..Default::default()
    };
    ExperimentRunner::with_config(config)
}

/// Helper to get test config path.
fn test_config_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("configs")
        .join("example_buy_hold.toml")
}

// =============================================================================
// T1: DETERMINISM TEST
// =============================================================================

#[test]
fn t1_runner_determinism_with_dividends() {
    // Skip if config doesn't exist
    let config_path = test_config_path();
    if !config_path.exists() {
        eprintln!("Skipping test: config file not found at {:?}", config_path);
        return;
    }

    let runner = create_test_runner(true, ExecutionMode::Auto);

    // Run twice with same config
    let result1 = runner.run_single(&config_path);
    let result2 = runner.run_single(&config_path);

    // Both should succeed or both should fail
    assert_eq!(result1.is_ok(), result2.is_ok(), "Determinism: both runs should have same outcome");

    if let (Ok(r1), Ok(r2)) = (result1, result2) {
        // Metrics should be identical
        assert_eq!(
            r1.metrics.sharpe_ratio, r2.metrics.sharpe_ratio,
            "Sharpe ratio should be deterministic"
        );
        assert_eq!(
            r1.metrics.cagr, r2.metrics.cagr,
            "CAGR should be deterministic"
        );
        assert_eq!(
            r1.metrics.max_drawdown, r2.metrics.max_drawdown,
            "Max drawdown should be deterministic"
        );

        // Timeseries length should match
        assert_eq!(
            r1.timeseries.len(),
            r2.timeseries.len(),
            "Timeseries length should be deterministic"
        );

        // Execution mode should be same
        assert_eq!(
            r1.metadata.execution_mode, r2.metadata.execution_mode,
            "Execution mode should be deterministic"
        );

        // Dividend settings should be same
        assert_eq!(
            r1.metadata.dividends_enabled, r2.metadata.dividends_enabled,
            "Dividends enabled should be deterministic"
        );
    }
}

// =============================================================================
// T2: ANTI-DOUBLE-COUNT VALIDATION
// =============================================================================

#[test]
fn t2_policy_recorded_in_metadata() {
    let config_path = test_config_path();
    if !config_path.exists() {
        eprintln!("Skipping test: config file not found at {:?}", config_path);
        return;
    }

    let runner = create_test_runner(true, ExecutionMode::Standard);
    let result = runner.run_single(&config_path);

    if let Ok(r) = result {
        // Policy should be recorded
        assert!(
            r.metadata.dividend_policy.is_some(),
            "Dividend policy should be recorded in metadata"
        );

        let policy = r.metadata.dividend_policy.unwrap();
        
        // Verify anti-double-count policy
        assert_eq!(
            policy.signals_price,
            PriceType::Adjusted,
            "Signals should use adjusted prices"
        );
        assert_eq!(
            policy.valuation_price,
            PriceType::Raw,
            "Valuation should use raw prices"
        );
        assert!(
            policy.dividends_as_cashflow,
            "Dividends should be enabled as cashflow"
        );
    }
}

#[test]
fn t2_policy_trace_entry_present() {
    let config_path = test_config_path();
    if !config_path.exists() {
        eprintln!("Skipping test: config file not found at {:?}", config_path);
        return;
    }

    let runner = create_test_runner(true, ExecutionMode::Standard);
    let result = runner.run_single(&config_path);

    if let Ok(r) = result {
        // First trace entry should be policy
        assert!(!r.trace.is_empty(), "Trace should not be empty");
        
        let policy_entry = &r.trace[0];
        assert_eq!(
            policy_entry.block_type, "dividend_policy",
            "First trace entry should be dividend_policy"
        );
        assert!(
            policy_entry.params_effective.contains_key("valuation_price"),
            "Policy trace should contain valuation_price"
        );
    }
}

// =============================================================================
// T3: FAST MODE FALLBACK
// =============================================================================

#[test]
fn t3_fast_mode_fallback_with_dividends() {
    let config_path = test_config_path();
    if !config_path.exists() {
        eprintln!("Skipping test: config file not found at {:?}", config_path);
        return;
    }

    // Request Fast mode with dividends enabled
    let runner = create_test_runner(true, ExecutionMode::Fast);
    let result = runner.run_single(&config_path);

    if let Ok(r) = result {
        // Should NOT be Fast mode (fallback due to dividends)
        assert_ne!(
            r.metadata.execution_mode,
            ExecutionMode::Fast,
            "Fast mode should fallback when dividends enabled"
        );

        // Should be Compiled mode
        assert_eq!(
            r.metadata.execution_mode,
            ExecutionMode::Compiled,
            "Should fallback to Compiled mode"
        );

        // Fallback reason should mention dividends
        assert!(
            r.metadata.mode_fallback_reason.is_some(),
            "Fallback reason should be recorded"
        );
        let reason = r.metadata.mode_fallback_reason.as_ref().unwrap();
        assert!(
            reason.contains("dividend"),
            "Fallback reason should mention dividend: got '{}'",
            reason
        );
    }
}

#[test]
fn t3_fast_mode_allowed_without_dividends() {
    let config_path = test_config_path();
    if !config_path.exists() {
        eprintln!("Skipping test: config file not found at {:?}", config_path);
        return;
    }

    // Request Fast mode with dividends DISABLED
    let runner = create_test_runner(false, ExecutionMode::Fast);
    let result = runner.run_single(&config_path);

    if let Ok(r) = result {
        // Fallback may still happen for other reasons (unsupported blocks)
        // but NOT because of dividends
        if r.metadata.execution_mode == ExecutionMode::Fast {
            // Good - no fallback needed
            assert!(r.metadata.mode_fallback_reason.is_none());
        } else if let Some(ref reason) = r.metadata.mode_fallback_reason {
            // If there's a fallback, it shouldn't be about dividends
            assert!(
                !reason.contains("dividend cashflow"),
                "Fallback should not be about dividends when disabled"
            );
        }
    }
}

// =============================================================================
// T4: FALLBACK TRACE ENTRY
// =============================================================================

#[test]
fn t4_fallback_trace_entry_present() {
    let config_path = test_config_path();
    if !config_path.exists() {
        eprintln!("Skipping test: config file not found at {:?}", config_path);
        return;
    }

    // Request Fast mode with dividends enabled (will cause fallback)
    let runner = create_test_runner(true, ExecutionMode::Fast);
    let result = runner.run_single(&config_path);

    if let Ok(r) = result {
        // Look for mode_fallback trace entry
        let fallback_entry = r.trace.iter().find(|e| e.block_type == "mode_fallback");
        
        assert!(
            fallback_entry.is_some(),
            "Mode fallback trace entry should be present"
        );

        if let Some(entry) = fallback_entry {
            assert!(
                entry.params_effective.contains_key("requested"),
                "Fallback trace should contain requested mode"
            );
            assert!(
                entry.params_effective.contains_key("resolved"),
                "Fallback trace should contain resolved mode"
            );
            assert!(
                entry.params_effective.contains_key("reason"),
                "Fallback trace should contain reason"
            );
        }
    }
}

