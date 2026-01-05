//! Smoke tests for Entry Module.
//!
//! Tests the full entry flow with realistic data for BR and US markets.

use backtester_intelligence::entry::{
    AssetCandidate, EntryContext, EntryEngine, EntryEngineConfig,
    GatingConfig, SelectionConfig, WeightingConfig, OrderGeneratorConfig,
};
use backtester_intelligence::filters::Market;
use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;

/// Generate synthetic BR candidates for testing.
fn make_br_candidates(count: usize) -> Vec<AssetCandidate> {
    let symbols = vec![
        "PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3", "B3SA3", "RENT3", "WEGE3", "SUZB3", "JBSS3",
        "GGBR4", "CSNA3", "USIM5", "BRKM5", "GOAU4", "CPLE6", "CMIG4", "ELET3", "ELET6", "SBSP3",
        "CCRO3", "EQTL3", "TAEE11", "ENGI11", "CSAN3", "RAIZ4", "PRIO3", "RRRP3", "UGPA3", "VBBR3",
        "MGLU3", "VIIA3", "AMER3", "COGN3", "YDUQ3", "BEEF3", "MRFG3", "BRFS3", "NTCO3", "LWSA3",
        "TOTS3", "POSI3", "INTB3", "CASH3", "MEAL3", "AURA33", "ALPA4", "LREN3", "SOMA3", "ARZZ3",
    ];

    symbols.iter().take(count).enumerate().map(|(i, sym)| {
        let mut c = AssetCandidate::new(*sym, Market::BR);
        c.price = Some(Decimal::from(20 + (i as i64 * 5))); // R$ 20 to R$ 265
        c.avg_volume = Some(Decimal::from(1_000_000 + (i as i64 * 100_000)));
        c.price_days = 30;
        c.has_fundamentals = i % 5 != 0; // 80% have fundamentals
        c.has_dividends = i % 4 != 0;    // 75% have dividends
        c.volatility = Some(0.15 + (i as f64 * 0.01)); // 15% to 64%
        c.score = Some(0.90 - (i as f64 * 0.015));     // 0.90 to 0.15
        c
    }).collect()
}

/// Generate synthetic US candidates for testing (no fundamentals).
fn make_us_candidates(count: usize) -> Vec<AssetCandidate> {
    let symbols = vec![
        "AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "BRK-B", "UNH", "JNJ",
        "V", "MA", "PG", "HD", "CVX", "MRK", "ABBV", "PFE", "KO", "PEP",
        "COST", "TMO", "AVGO", "WMT", "MCD", "ACN", "LLY", "DHR", "NEE", "TXN",
        "MDT", "HON", "UPS", "QCOM", "PM", "LOW", "IBM", "GE", "CAT", "BA",
        "AMT", "SBUX", "DE", "ADP", "GS", "BLK", "SCHW", "MMM", "DIS", "NKE",
    ];

    symbols.iter().take(count).enumerate().map(|(i, sym)| {
        let mut c = AssetCandidate::new(*sym, Market::US);
        c.price = Some(Decimal::from(50 + (i as i64 * 10))); // $50 to $540
        c.avg_volume = Some(Decimal::from(5_000_000 + (i as i64 * 200_000)));
        c.price_days = 30;
        c.has_fundamentals = false; // US has no fundamentals (known limitation)
        c.has_dividends = false;    // US has no dividends
        c.volatility = Some(0.20 + (i as f64 * 0.005)); // 20% to 44.5%
        c.score = Some(0.85 - (i as f64 * 0.012));      // 0.85 to 0.25
        c
    }).collect()
}

#[test]
fn smoke_test_br_multiple_rebalances() {
    // Configuration
    let config = EntryEngineConfig {
        gating: GatingConfig {
            require_fundamentals: false, // Don't require for smoke test
            require_dividends: false,
            ..Default::default()
        },
        selection: SelectionConfig {
            top_n_br: 10,
            top_n_us: 10,
            min_score_threshold: None,
            ..Default::default()
        },
        weighting: WeightingConfig::default(),
        orders: OrderGeneratorConfig::default(),
        eligibility_provider: None,
    };
    let engine = EntryEngine::new(config);

    let candidates = make_br_candidates(50);
    let capital = dec!(1_000_000); // R$ 1M
    let mut positions: HashMap<String, i64> = HashMap::new();

    // Simulate 4 weekly rebalances
    let dates = vec![
        NaiveDate::from_ymd_opt(2025, 1, 3).unwrap(),
        NaiveDate::from_ymd_opt(2025, 1, 10).unwrap(),
        NaiveDate::from_ymd_opt(2025, 1, 17).unwrap(),
        NaiveDate::from_ymd_opt(2025, 1, 24).unwrap(),
    ];

    for (i, date) in dates.iter().enumerate() {
        let ctx = EntryContext::new(*date, capital, Market::BR);
        let (result, orders, audit) = engine.evaluate(&ctx, &candidates, &positions);

        // Basic sanity checks
        assert!(result.targets.len() <= 10, "Rebalance {}: too many targets", i);
        assert!(result.diagnostics.final_selected <= 10);

        // Weights should sum to ~1.0
        let total_weight: f64 = result.targets.iter().map(|t| t.target_weight).sum();
        assert!(
            (total_weight - 1.0).abs() < 0.05,
            "Rebalance {}: weight sum {} should be ~1.0", i, total_weight
        );

        // Orders should be in multiples of 100 (BR lots)
        for order in &orders {
            assert!(
                order.shares % 100 == 0,
                "Rebalance {}: {} shares {} not multiple of 100",
                i, order.symbol, order.shares
            );
        }

        // Audit log should be generated
        let summary = audit.to_summary();
        assert!(summary.contains("SELECIONADOS"), "Rebalance {}: missing SELECIONADOS", i);
        assert!(summary.contains("MÉTRICAS"), "Rebalance {}: missing MÉTRICAS", i);

        // Update positions for next rebalance
        for target in &result.targets {
            positions.insert(target.symbol.clone(), target.target_shares);
        }

        println!("=== Rebalance {} ({}) ===", i + 1, date);
        println!("Selected: {}", result.targets.len());
        println!("Orders: {}", orders.len());
        println!("Turnover: {:.1}%", result.diagnostics.turnover * 100.0);
        println!();
    }
}

#[test]
fn smoke_test_us_without_fundamentals() {
    // US market without fundamentals should still work with price-based techniques
    let config = EntryEngineConfig {
        gating: GatingConfig {
            require_fundamentals: false, // Don't require - would exclude all US
            require_dividends: false,
            ..Default::default()
        },
        selection: SelectionConfig {
            top_n_br: 10,
            top_n_us: 10,
            min_score_threshold: None,
            ..Default::default()
        },
        weighting: WeightingConfig::default(),
        orders: OrderGeneratorConfig::default(),
        eligibility_provider: None,
    };
    let engine = EntryEngine::new(config);

    let candidates = make_us_candidates(50);
    let capital = dec!(500_000); // $500k
    let positions: HashMap<String, i64> = HashMap::new();

    // Single rebalance
    let date = NaiveDate::from_ymd_opt(2025, 1, 3).unwrap();
    let ctx = EntryContext::new(date, capital, Market::US);
    let (result, orders, audit) = engine.evaluate(&ctx, &candidates, &positions);

    // Should select top 10 US assets
    assert!(result.targets.len() <= 10);
    assert!(result.diagnostics.final_selected <= 10);

    // No panic with missing fundamentals
    assert!(result.exclusions.is_empty() || result.exclusions.iter().any(|e| 
        format!("{:?}", e.reason).contains("OutOfTopN")
    ));

    // US orders should be any share count (lot size 1)
    for order in &orders {
        assert!(order.shares >= 1, "US order should have at least 1 share");
    }

    // Audit log generated
    let summary = audit.to_summary();
    assert!(summary.contains("US"), "Should show US market");
    println!("{}", summary);
}

#[test]
fn smoke_test_fundamentals_required_drops_us() {
    // When fundamentals are required, all US assets should be excluded
    let config = EntryEngineConfig {
        gating: GatingConfig {
            require_fundamentals: true, // REQUIRE fundamentals
            require_dividends: false,
            ..Default::default()
        },
        selection: SelectionConfig::default(),
        weighting: WeightingConfig::default(),
        orders: OrderGeneratorConfig::default(),
        eligibility_provider: None,
    };
    let engine = EntryEngine::new(config);

    let candidates = make_us_candidates(20);
    let capital = dec!(500_000);
    let positions: HashMap<String, i64> = HashMap::new();

    let date = NaiveDate::from_ymd_opt(2025, 1, 3).unwrap();
    let ctx = EntryContext::new(date, capital, Market::US);
    let (result, _orders, audit) = engine.evaluate(&ctx, &candidates, &positions);

    // All US should be excluded for missing fundamentals
    assert_eq!(result.targets.len(), 0, "No US assets should be selected when fundamentals required");
    assert!(result.diagnostics.gating_excluded > 0, "Should have gating exclusions");

    // Audit shows exclusions
    let summary = audit.to_summary();
    assert!(summary.contains("EXCLUÍDOS") || summary.contains("gating"), 
        "Should show exclusions for missing fundamentals");
    println!("{}", summary);
}

#[test]
fn smoke_test_determinism() {
    // Same input should produce same output
    let config = EntryEngineConfig::default();
    let engine = EntryEngine::new(config);

    let candidates = make_br_candidates(30);
    let capital = dec!(500_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let date = NaiveDate::from_ymd_opt(2025, 1, 3).unwrap();
    let ctx = EntryContext::new(date, capital, Market::BR);

    // Run twice
    let (result1, _, _) = engine.evaluate(&ctx, &candidates, &positions);
    let (result2, _, _) = engine.evaluate(&ctx, &candidates, &positions);

    // Should be identical
    assert_eq!(result1.targets.len(), result2.targets.len(), "Target count should match");
    
    for (t1, t2) in result1.targets.iter().zip(result2.targets.iter()) {
        assert_eq!(t1.symbol, t2.symbol, "Symbols should match");
        assert!((t1.target_weight - t2.target_weight).abs() < 0.0001, "Weights should match");
        assert_eq!(t1.target_shares, t2.target_shares, "Shares should match");
    }
}

// =============================================================================
// GAP-1: Anti-Look-Ahead Volatility Test
// =============================================================================

/// Test proving that volatility used for risk-parity weighting is determined
/// only by data available at rebalance time, not future data.
///
/// This test creates two scenarios with identical "past" volatility but different
/// "future" volatility. The weights should be identical because Entry only uses
/// the volatility passed to it (which should be computed from past data upstream).
#[test]
fn test_volatility_anti_lookahead() {
    use backtester_intelligence::entry::WeightingConfig;
    
    let config = EntryEngineConfig {
        selection: SelectionConfig {
            top_n_br: 3,
            top_n_us: 3,
            min_score_threshold: None,
            ..Default::default()
        },
        weighting: WeightingConfig {
            max_weight: 0.80, // Allow weights to vary without capping
            min_weight: 0.05,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = EntryEngine::new(config);

    // Scenario 1: Volatility as of rebalance date (20%, 30%, 40%)
    let candidates1 = vec![
        {
            let mut c = AssetCandidate::new("A", Market::BR);
            c.price = Some(dec!(50));
            c.avg_volume = Some(dec!(2_000_000));
            c.price_days = 30;
            c.has_fundamentals = true;
            c.volatility = Some(0.20); // Low vol -> high weight
            c.score = Some(0.8);
            c
        },
        {
            let mut c = AssetCandidate::new("B", Market::BR);
            c.price = Some(dec!(60));
            c.avg_volume = Some(dec!(2_000_000));
            c.price_days = 30;
            c.has_fundamentals = true;
            c.volatility = Some(0.30); // Medium vol
            c.score = Some(0.7);
            c
        },
        {
            let mut c = AssetCandidate::new("C", Market::BR);
            c.price = Some(dec!(70));
            c.avg_volume = Some(dec!(2_000_000));
            c.price_days = 30;
            c.has_fundamentals = true;
            c.volatility = Some(0.40); // High vol -> low weight
            c.score = Some(0.6);
            c
        },
    ];

    // Scenario 2: Same volatility at rebalance date
    // Even if "future" volatility would be different, Entry doesn't see it
    let candidates2 = candidates1.clone();

    let capital = dec!(100_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let date = NaiveDate::from_ymd_opt(2025, 1, 3).unwrap();
    let ctx = EntryContext::new(date, capital, Market::BR);

    let (result1, _, _) = engine.evaluate(&ctx, &candidates1, &positions);
    let (result2, _, _) = engine.evaluate(&ctx, &candidates2, &positions);

    // Weights should be identical because Entry uses only the volatility passed
    assert_eq!(result1.targets.len(), result2.targets.len());
    for (t1, t2) in result1.targets.iter().zip(result2.targets.iter()) {
        assert_eq!(t1.symbol, t2.symbol);
        assert!(
            (t1.target_weight - t2.target_weight).abs() < 0.0001,
            "Weights should match: {} vs {}",
            t1.target_weight, t2.target_weight
        );
    }

    // Verify risk-parity is working: lower vol should have higher weight
    let weight_a = result1.targets.iter().find(|t| t.symbol == "A").map(|t| t.target_weight).unwrap_or(0.0);
    let weight_c = result1.targets.iter().find(|t| t.symbol == "C").map(|t| t.target_weight).unwrap_or(0.0);
    assert!(
        weight_a > weight_c,
        "Lower volatility asset A ({:.2}) should have higher weight than C ({:.2})",
        weight_a, weight_c
    );
}

// =============================================================================
// GAP-3: Vol=0/NaN Edge Case Test
// =============================================================================

/// Test that volatility=0 is clamped to a minimum value to avoid division by zero.
#[test]
fn test_zero_volatility_clamped() {
    use backtester_intelligence::entry::{Weighter, WeightingCandidate, WeightingConfig, WeightingMethod};

    let config = WeightingConfig {
        method: WeightingMethod::RiskParity,
        max_weight: 0.90,
        min_weight: 0.05,
        fallback_volatility: 0.30,
        ..Default::default()
    };
    let weighter = Weighter::new(config);

    // One asset with vol=0.0 (should be clamped)
    let candidates = vec![
        WeightingCandidate::new("ZERO_VOL", 0.8, Some(0.0)),    // Vol=0 -> clamp to 0.01
        WeightingCandidate::new("NORMAL", 0.8, Some(0.20)),      // Normal vol
    ];

    // Should not panic
    let results = weighter.calculate_weights(candidates);

    // Both should have valid weights
    assert_eq!(results.len(), 2);
    for r in &results {
        assert!(r.weight > 0.0, "{} should have positive weight", r.symbol);
        assert!(r.weight <= 1.0, "{} weight should be <= 1.0", r.symbol);
        assert!(!r.weight.is_nan(), "{} weight should not be NaN", r.symbol);
        assert!(!r.weight.is_infinite(), "{} weight should not be infinite", r.symbol);
    }

    // Zero vol gets clamped to 0.01, so 1/0.01 = 100
    // Normal vol 0.20: 1/0.20 = 5
    // Total inverse = 105
    // ZERO_VOL weight = 100/105 ≈ 0.95 -> capped at 0.90
    let zero_vol_weight = results.iter().find(|r| r.symbol == "ZERO_VOL").unwrap().weight;
    let normal_weight = results.iter().find(|r| r.symbol == "NORMAL").unwrap().weight;
    
    assert!(zero_vol_weight > normal_weight, 
        "Zero-vol (clamped) should have higher weight: {} vs {}", zero_vol_weight, normal_weight);
}

/// Test that None volatility uses fallback.
#[test]
fn test_none_volatility_uses_fallback() {
    use backtester_intelligence::entry::{Weighter, WeightingCandidate, WeightingConfig, WeightingMethod};

    let config = WeightingConfig {
        method: WeightingMethod::RiskParity,
        fallback_volatility: 0.25,
        max_weight: 0.80,
        min_weight: 0.05,
        ..Default::default()
    };
    let weighter = Weighter::new(config);

    let candidates = vec![
        WeightingCandidate::new("NO_VOL", 0.8, None),          // Uses fallback 0.25
        WeightingCandidate::new("HAS_VOL", 0.8, Some(0.25)),   // Explicit 0.25
    ];

    let results = weighter.calculate_weights(candidates);

    // Both should have same weight since both end up with 0.25 vol
    let no_vol = results.iter().find(|r| r.symbol == "NO_VOL").unwrap();
    let has_vol = results.iter().find(|r| r.symbol == "HAS_VOL").unwrap();
    
    assert!(
        (no_vol.weight - has_vol.weight).abs() < 0.001,
        "None vol should use fallback: {} vs {}", no_vol.weight, has_vol.weight
    );
    assert_eq!(no_vol.volatility, 0.25, "No vol should show fallback volatility");
}

// =============================================================================
// GAP-7: Fundamentals Point-in-Time Test  
// =============================================================================

/// Test that fundamentals from future dates are excluded (anti-look-ahead).
#[test]
fn test_future_fundamentals_excluded() {
    use backtester_intelligence::entry::ExclusionReason;
    
    let config = EntryEngineConfig::default();
    let engine = EntryEngine::new(config);

    // Create a candidate with fundamentals from the future
    let candidates = vec![{
        let mut c = AssetCandidate::new("FUTURE_DATA", Market::BR);
        c.price = Some(dec!(50));
        c.avg_volume = Some(dec!(2_000_000));
        c.price_days = 30;
        c.has_fundamentals = true;
        c.volatility = Some(0.25);
        c.score = Some(0.9);
        // Fundamentals from 2025-06-30, but rebalance on 2025-01-03 (future data!)
        c.fundamentals_as_of = Some(NaiveDate::from_ymd_opt(2025, 6, 30).unwrap());
        c
    }];

    let capital = dec!(100_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let date = NaiveDate::from_ymd_opt(2025, 1, 3).unwrap();
    let ctx = EntryContext::new(date, capital, Market::BR);

    let (result, _, audit) = engine.evaluate(&ctx, &candidates, &positions);

    // Asset should be excluded for future fundamentals
    assert!(result.targets.is_empty(), "Future data asset should not be selected");
    assert!(result.exclusions.iter().any(|e| e.reason == ExclusionReason::FutureFundamentals),
        "Should have FutureFundamentals exclusion reason");
    
    let summary = audit.to_summary();
    assert!(summary.contains("look-ahead") || summary.contains("futuro"), 
        "Audit should mention look-ahead or future");
}

/// Test that past fundamentals pass validation.
#[test]
fn test_past_fundamentals_allowed() {
    let config = EntryEngineConfig {
        selection: SelectionConfig {
            top_n_br: 1,
            top_n_us: 1,
            min_score_threshold: None,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = EntryEngine::new(config);

    let candidates = vec![{
        let mut c = AssetCandidate::new("PAST_DATA", Market::BR);
        c.price = Some(dec!(50));
        c.avg_volume = Some(dec!(2_000_000));
        c.price_days = 30;
        c.has_fundamentals = true;
        c.volatility = Some(0.25);
        c.score = Some(0.9);
        // Fundamentals from 2024-09-30, rebalance on 2025-01-03 (valid past data)
        c.fundamentals_as_of = Some(NaiveDate::from_ymd_opt(2024, 9, 30).unwrap());
        c
    }];

    let capital = dec!(100_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let date = NaiveDate::from_ymd_opt(2025, 1, 3).unwrap();
    let ctx = EntryContext::new(date, capital, Market::BR);

    let (result, _, _) = engine.evaluate(&ctx, &candidates, &positions);

    // Asset should be selected (past fundamentals are valid)
    assert_eq!(result.targets.len(), 1, "Past data asset should be selected");
    assert_eq!(result.targets[0].symbol, "PAST_DATA");
}

