//! Property-based tests for validation module using proptest.
//!
//! Tests invariants that should hold for any valid input.

use proptest::prelude::*;
use backtester_validation::{
    SchemaValidator, SanityChecker, CrossChecker, 
    sanity::SanityConfig,
    crosscheck::{CrosscheckConfig, ReportedMetrics},
};

// =============================================================================
// STRATEGIES FOR GENERATING TEST DATA
// =============================================================================

/// Generate a random f64 in a reasonable range for financial metrics.
fn metric_value() -> impl Strategy<Value = f64> {
    prop_oneof![
        // Normal values
        (-2.0..2.0_f64),
        // Edge cases
        Just(0.0),
        Just(-0.0),
        Just(1.0),
        Just(-1.0),
    ]
}

/// Generate a plausible NAV value.
fn nav_value() -> impl Strategy<Value = f64> {
    100_000.0..10_000_000.0_f64
}

/// Generate a NAV series of given length.
fn nav_series(len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(nav_value(), len)
}

/// Generate a monotonically increasing NAV series (simulating positive returns).
#[allow(dead_code)]
fn monotonic_nav_series(len: usize) -> impl Strategy<Value = Vec<f64>> {
    (1_000_000.0..2_000_000.0_f64)
        .prop_map(move |start| {
            // Simple deterministic growth pattern
            (0..len).map(|i| start * (1.001_f64).powi(i as i32)).collect()
        })
}

/// Generate valid reported metrics.
fn valid_metrics() -> impl Strategy<Value = serde_json::Value> {
    (
        metric_value(), // cagr
        (-5.0..5.0_f64), // sharpe (reasonable range)
        (-0.5..0.0_f64), // max_drawdown (negative)
        (10u32..500), // total_trades
        (0.05..0.5_f64), // volatility
    )
        .prop_map(|(cagr, sharpe, dd, trades, vol)| {
            serde_json::json!({
                "cagr": cagr,
                "sharpe_ratio": sharpe,
                "max_drawdown": dd,
                "total_trades": trades,
                "volatility": vol
            })
        })
}

// =============================================================================
// PROPERTY: Schema validation is deterministic
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_schema_validation_deterministic(
        cagr in metric_value(),
        sharpe in metric_value(),
        dd in metric_value(),
        trades in 0u32..1000,
        vol in metric_value()
    ) {
        let validator = SchemaValidator::new(true);
        
        let json = serde_json::json!({
            "cagr": cagr,
            "sharpe_ratio": sharpe,
            "max_drawdown": dd,
            "total_trades": trades,
            "volatility": vol
        });
        
        // Run validation twice
        let result1 = validator.validate_metrics(&json);
        let result2 = validator.validate_metrics(&json);
        
        // Results should be identical
        prop_assert_eq!(result1.passed, result2.passed);
        prop_assert_eq!(result1.null_fields, result2.null_fields);
        prop_assert_eq!(result1.missing_fields, result2.missing_fields);
    }
}

// =============================================================================
// PROPERTY: Valid metrics always pass schema
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_valid_metrics_pass_schema(json in valid_metrics()) {
        let validator = SchemaValidator::new(true);
        let result = validator.validate_metrics(&json);
        
        // Non-null values for required fields should always pass schema
        prop_assert!(!result.has_failures(), 
            "Valid metrics should pass schema: {:?}", result);
    }
}

// =============================================================================
// PROPERTY: CrossChecker produces finite Sharpe for valid NAV series
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_crosscheck_finite_sharpe(nav in nav_series(252)) {
        // Skip if nav is empty or has only one element
        prop_assume!(nav.len() >= 2);
        
        // Skip if any value is zero or negative
        prop_assume!(nav.iter().all(|&v| v > 0.0));
        
        let config = CrosscheckConfig::default();
        let checker = CrossChecker::new(config);
        
        let reported = ReportedMetrics {
            cagr: 0.1,
            sharpe_ratio: 1.0,
            volatility: 0.15,
            max_drawdown: -0.1,
        };
        
        let result = checker.crosscheck(&reported, &nav);
        
        // Recomputed Sharpe should be finite (not NaN or Inf)
        prop_assert!(result.recomputed.sharpe_ratio.is_finite(),
            "Sharpe should be finite, got: {}", result.recomputed.sharpe_ratio);
    }
}

// =============================================================================
// PROPERTY: CrossCheck is consistent (same input -> same output)
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_crosscheck_consistent(nav in nav_series(100)) {
        prop_assume!(nav.len() >= 2);
        prop_assume!(nav.iter().all(|&v| v > 0.0));
        
        let config = CrosscheckConfig::default();
        let checker = CrossChecker::new(config);
        
        let reported = ReportedMetrics {
            cagr: 0.1,
            sharpe_ratio: 1.0,
            volatility: 0.15,
            max_drawdown: -0.1,
        };
        
        let result1 = checker.crosscheck(&reported, &nav);
        let result2 = checker.crosscheck(&reported, &nav);
        
        // Results should be identical
        prop_assert_eq!(result1.passed, result2.passed);
        prop_assert!((result1.recomputed.cagr - result2.recomputed.cagr).abs() < 1e-10);
        prop_assert!((result1.recomputed.sharpe_ratio - result2.recomputed.sharpe_ratio).abs() < 1e-10);
        prop_assert!((result1.recomputed.volatility - result2.recomputed.volatility).abs() < 1e-10);
    }
}

// =============================================================================
// PROPERTY: Sanity check flags absurd Sharpe
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_sanity_flags_absurd_sharpe(sharpe in 25.0..100.0_f64) {
        let config = SanityConfig::default();
        let checker = SanityChecker::new(config);
        
        let json = serde_json::json!({
            "cagr": 0.5,
            "sharpe_ratio": sharpe,
            "max_drawdown": -0.1,
            "total_trades": 100,
            "volatility": 0.15
        });
        
        let result = checker.check_json(&json);
        
        // Sharpe > 20 should always be flagged as Fail
        prop_assert_eq!(result.verdict, backtester_validation::Verdict::Fail,
            "Sharpe {} should be flagged as Fail", sharpe);
    }
}

// =============================================================================
// PROPERTY: Sanity check allows reasonable metrics
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_sanity_allows_reasonable(
        cagr in -0.3..0.5_f64,
        sharpe in -2.0..3.0_f64,
        dd in -0.4..-0.05_f64,
        trades in 50u32..500,
        vol in 0.1..0.3_f64
    ) {
        let config = SanityConfig::default();
        let checker = SanityChecker::new(config);
        
        let json = serde_json::json!({
            "cagr": cagr,
            "sharpe_ratio": sharpe,
            "max_drawdown": dd,
            "total_trades": trades,
            "volatility": vol
        });
        
        let result = checker.check_json(&json);
        
        // Reasonable metrics should Pass (or maybe Warn for edge cases, but not Fail)
        prop_assert_ne!(result.verdict, backtester_validation::Verdict::Fail,
            "Reasonable metrics should not Fail: cagr={}, sharpe={}, dd={}, trades={}, vol={}",
            cagr, sharpe, dd, trades, vol);
    }
}

// =============================================================================
// PROPERTY: Monotonically increasing NAV produces positive CAGR
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn prop_monotonic_nav_positive_returns(start in 1_000_000.0..2_000_000.0_f64) {
        // Create strictly increasing NAV
        let nav: Vec<f64> = (0..252)
            .map(|i| start * (1.0 + 0.001_f64).powi(i))
            .collect();
        
        let config = CrosscheckConfig::default();
        let checker = CrossChecker::new(config);
        
        let reported = ReportedMetrics {
            cagr: 0.0,
            sharpe_ratio: 0.0,
            volatility: 0.0,
            max_drawdown: 0.0,
        };
        
        let result = checker.crosscheck(&reported, &nav);
        
        // CAGR should be positive for strictly increasing NAV
        prop_assert!(result.recomputed.cagr > 0.0,
            "Monotonically increasing NAV should have positive CAGR, got: {}",
            result.recomputed.cagr);
        
        // Max drawdown should be zero or very close (no declines)
        prop_assert!(result.recomputed.max_drawdown >= -0.001,
            "Monotonic NAV should have near-zero drawdown, got: {}",
            result.recomputed.max_drawdown);
    }
}

// =============================================================================
// PROPERTY: Max drawdown is always non-positive
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_max_drawdown_non_positive(nav in nav_series(100)) {
        prop_assume!(nav.len() >= 2);
        prop_assume!(nav.iter().all(|&v| v > 0.0));
        
        let config = CrosscheckConfig::default();
        let checker = CrossChecker::new(config);
        
        let reported = ReportedMetrics {
            cagr: 0.0,
            sharpe_ratio: 0.0,
            volatility: 0.0,
            max_drawdown: 0.0,
        };
        
        let result = checker.crosscheck(&reported, &nav);
        
        // Max drawdown is defined as negative (or zero if no decline)
        prop_assert!(result.recomputed.max_drawdown <= 0.0,
            "Max drawdown should be <= 0, got: {}", result.recomputed.max_drawdown);
    }
}

