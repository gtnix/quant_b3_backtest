//! Golden snapshot tests for Entry Module audit output.
//!
//! These tests detect accidental changes to the audit output format.
//! If the format changes intentionally, update the golden file.

use backtester_core::{Money, Price};
use backtester_intelligence::entry::{
    AssetCandidate, EntryContext, EntryEngine, EntryEngineConfig,
    SelectionConfig, WeightingConfig,
};
use backtester_intelligence::filters::Market;
use chrono::NaiveDate;
use std::collections::HashMap;

/// Golden audit summary - update this if format changes intentionally.
const GOLDEN_AUDIT_SUMMARY: &str = r#"=== REBALANCE 2025-01-03 (BR) ===

SELECIONADOS (5):
  PETR4: 19.0% (score=0.850, vol=25.0%, weight=19.0%)
  VALE3: 15.9% (score=0.820, vol=30.0%, weight=15.9%)
  ITUB4: 21.6% (score=0.780, vol=22.0%, weight=21.6%)
  BBDC4: 17.0% (score=0.750, vol=28.0%, weight=17.0%)
  WEGE3: 26.4% (score=0.720, vol=18.0%, weight=26.4%)

EXCLUÍDOS GATING (1):
  SMALL: liquidez insuficiente

ORDENS (5):
  COMPRA PETR4 x 4900 @ 38.000000 (custo: 209.47)
  COMPRA VALE3 x 2300 @ 65.000000 (custo: 168.19)
  COMPRA ITUB4 x 6600 @ 32.000000 (custo: 237.60)
  COMPRA BBDC4 x 9300 @ 18.000000 (custo: 188.32)
  COMPRA WEGE3 x 5700 @ 45.000000 (custo: 288.56)

MÉTRICAS:
  Candidatos: 6
  Excluídos gating: 1
  Excluídos seleção: 0
  Selecionados: 5
  Peso total: 100.0%
  Turnover: 97.1%
  Custos estimados: 1092.15
  Cash residual: 9400.00
"#;

/// Fixed scenario for golden snapshot testing.
/// This produces deterministic output for comparison.
fn run_golden_scenario() -> String {
    let config = EntryEngineConfig {
        selection: SelectionConfig {
            top_n_br: 5,
            top_n_us: 5,
            min_score_threshold: None,
            ..Default::default()
        },
        weighting: WeightingConfig {
            max_weight: 0.30,
            min_weight: 0.05,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = EntryEngine::new(config);

    // Fixed candidates with deterministic data
    let candidates: Vec<AssetCandidate> = vec![
        {
            let mut c = AssetCandidate::new("PETR4", Market::BR);
            c.price = Some(Price::from_int(38));
            c.avg_volume = Some(Money::from_int(5_000_000));
            c.price_days = 30;
            c.has_fundamentals = true;
            c.volatility = Some(0.25);
            c.score = Some(0.85);
            c
        },
        {
            let mut c = AssetCandidate::new("VALE3", Market::BR);
            c.price = Some(Price::from_int(65));
            c.avg_volume = Some(Money::from_int(8_000_000));
            c.price_days = 30;
            c.has_fundamentals = true;
            c.volatility = Some(0.30);
            c.score = Some(0.82);
            c
        },
        {
            let mut c = AssetCandidate::new("ITUB4", Market::BR);
            c.price = Some(Price::from_int(32));
            c.avg_volume = Some(Money::from_int(4_000_000));
            c.price_days = 30;
            c.has_fundamentals = true;
            c.volatility = Some(0.22);
            c.score = Some(0.78);
            c
        },
        {
            let mut c = AssetCandidate::new("BBDC4", Market::BR);
            c.price = Some(Price::from_int(18));
            c.avg_volume = Some(Money::from_int(3_500_000));
            c.price_days = 30;
            c.has_fundamentals = true;
            c.volatility = Some(0.28);
            c.score = Some(0.75);
            c
        },
        {
            let mut c = AssetCandidate::new("WEGE3", Market::BR);
            c.price = Some(Price::from_int(45));
            c.avg_volume = Some(Money::from_int(2_000_000));
            c.price_days = 30;
            c.has_fundamentals = true;
            c.volatility = Some(0.18);
            c.score = Some(0.72);
            c
        },
        // Low volume candidate - will be excluded
        {
            let mut c = AssetCandidate::new("SMALL", Market::BR);
            c.price = Some(Price::from_int(10));
            c.avg_volume = Some(Money::from_int(100_000)); // Below 500k threshold
            c.price_days = 30;
            c.has_fundamentals = true;
            c.volatility = Some(0.35);
            c.score = Some(0.90);
            c
        },
    ];

    let capital = Money::from_int(1_000_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let date = NaiveDate::from_ymd_opt(2025, 1, 3).unwrap();
    let ctx = EntryContext::new(date, capital, Market::BR);

    let (_, _, audit) = engine.evaluate(&ctx, &candidates, &positions);
    audit.to_summary()
}

/// Golden snapshot test - validates audit output format stability.
/// 
/// The expected output is embedded as a constant to avoid file I/O issues.
/// If the format changes intentionally, update GOLDEN_AUDIT_SUMMARY.
#[test]
fn golden_audit_summary() {
    let summary = run_golden_scenario();
    
    // Expected golden output (update this if format changes intentionally)
    let expected = GOLDEN_AUDIT_SUMMARY;
    
    // Compare line by line for better error messages
    let summary_lines: Vec<&str> = summary.lines().collect();
    let expected_lines: Vec<&str> = expected.lines().collect();
    
    assert_eq!(
        summary_lines.len(),
        expected_lines.len(),
        "Line count mismatch: got {} lines, expected {}\n\nActual output:\n{}",
        summary_lines.len(),
        expected_lines.len(),
        summary
    );
    
    for (i, (got, want)) in summary_lines.iter().zip(expected_lines.iter()).enumerate() {
        assert_eq!(
            got, want,
            "Line {} mismatch:\n  got:  {:?}\n  want: {:?}",
            i + 1, got, want
        );
    }
}

/// Helper test to regenerate golden output (run with --nocapture)
#[test]
#[ignore]
fn generate_golden_output() {
    let summary = run_golden_scenario();
    println!("=== GOLDEN OUTPUT (copy this to GOLDEN_AUDIT_SUMMARY) ===");
    println!("{}", summary);
    println!("=== END GOLDEN OUTPUT ===");
}

/// Test that exclusion_counts_by_reason returns machine-readable data.
#[test]
fn golden_exclusion_counts() {
    let config = EntryEngineConfig {
        selection: SelectionConfig {
            top_n_br: 2, // Only top 2, so others excluded
            top_n_us: 2,
            min_score_threshold: None,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = EntryEngine::new(config);

    let candidates: Vec<AssetCandidate> = (0..5).map(|i| {
        let mut c = AssetCandidate::new(format!("SYM{}", i), Market::BR);
        c.price = Some(Price::from_int(50));
        c.avg_volume = Some(Money::from_int(2_000_000));
        c.price_days = 30;
        c.volatility = Some(0.25);
        c.score = Some(0.90 - (i as f64 * 0.05));
        c
    }).collect();

    let capital = Money::from_int(500_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let date = NaiveDate::from_ymd_opt(2025, 1, 3).unwrap();
    let ctx = EntryContext::new(date, capital, Market::BR);

    let (_, _, audit) = engine.evaluate(&ctx, &candidates, &positions);
    let counts = audit.exclusion_counts_by_reason();

    // Should have 3 excluded for OutOfTopN (5 candidates - 2 selected)
    use backtester_intelligence::entry::ExclusionReason;
    assert_eq!(
        counts.get(&ExclusionReason::OutOfTopN).copied().unwrap_or(0),
        3,
        "Should have 3 OutOfTopN exclusions"
    );
}

/// Test that cash_residual is tracked correctly.
#[test]
fn golden_cash_residual() {
    let config = EntryEngineConfig::default();
    let engine = EntryEngine::new(config);

    let candidates: Vec<AssetCandidate> = vec![{
        let mut c = AssetCandidate::new("TEST", Market::BR);
        c.price = Some(Price::from_int(100));
        c.avg_volume = Some(Money::from_int(5_000_000));
        c.price_days = 30;
        c.volatility = Some(0.25);
        c.score = Some(0.90);
        c
    }];

    let capital = Money::from_int(100_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let date = NaiveDate::from_ymd_opt(2025, 1, 3).unwrap();
    let ctx = EntryContext::new(date, capital, Market::BR);

    let (result, _, audit) = engine.evaluate(&ctx, &candidates, &positions);

    // Cash residual should be >= 0
    assert!(
        !result.diagnostics.cash_residual.is_negative(),
        "Cash residual should be non-negative"
    );

    // Should match audit accessor
    assert_eq!(
        result.diagnostics.cash_residual,
        audit.cash_residual(),
        "Cash residual should match between result and audit"
    );

    // With one asset at $100 and 99% max allocation, we should have some residual
    // Capital = 100k, max allocation = 99k, so we buy floor(99000/100) = 990 shares
    // But BR lot = 100, so 900 shares * $100 = $90,000 allocated
    // Cash residual = $100,000 - $90,000 = $10,000
    // Note: If weight normalization pushes to 100%, we might get 0 residual
    // The key invariant is that cash_residual >= 0
    println!("Cash residual: {}", result.diagnostics.cash_residual);
}
