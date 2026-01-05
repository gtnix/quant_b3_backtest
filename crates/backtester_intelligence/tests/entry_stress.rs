//! Stress tests for Entry Module - Extreme Scenarios.
//!
//! Tests validate robustness under extreme conditions:
//! - Extreme volatility values
//! - Threshold edge cases
//! - Penny stocks
//! - Large universe (10k assets)
//! - High turnover scenarios

use backtester_core::{Money, Price};
use backtester_intelligence::entry::{
    AssetCandidate, EntryContext, EntryEngine, EntryEngineConfig,
    GatingConfig, SelectionConfig, WeightingConfig,
    OrderSide,
};
use backtester_intelligence::filters::Market;
use chrono::NaiveDate;
use std::collections::HashMap;
use std::time::{Duration, Instant};

fn fixed_date() -> NaiveDate {
    NaiveDate::from_ymd_opt(2025, 1, 3).unwrap()
}

// =============================================================================
// Stress Test 1: Extreme Volatility
// =============================================================================

/// Stress: Volatility ranging from near-zero to extremely high.
/// Validates clamps prevent weight explosion.
#[test]
fn stress_extreme_volatility() {
    let config = EntryEngineConfig {
        selection: SelectionConfig {
            top_n_br: 20,
            top_n_us: 20,
            min_score_threshold: None,
            ..Default::default()
        },
        weighting: WeightingConfig {
            max_weight: 0.30,    // Allow higher weight to test capping
            min_weight: 0.02,
            fallback_volatility: 0.30,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = EntryEngine::new(config);

    // Create candidates with extreme volatilities
    let candidates: Vec<AssetCandidate> = vec![
        // Near-zero volatility (should be clamped)
        {
            let mut c = AssetCandidate::new("ZERO_VOL", Market::BR);
            c.price = Some(Price::from_int(50));
            c.avg_volume = Some(Money::from_int(2_000_000));
            c.price_days = 30;
            c.has_fundamentals = true;
            c.volatility = Some(0.0001); // Nearly zero
            c.score = Some(0.85);
            c
        },
        // Very low volatility
        {
            let mut c = AssetCandidate::new("LOW_VOL", Market::BR);
            c.price = Some(Price::from_int(50));
            c.avg_volume = Some(Money::from_int(2_000_000));
            c.price_days = 30;
            c.has_fundamentals = true;
            c.volatility = Some(0.05);
            c.score = Some(0.80);
            c
        },
        // Normal volatility
        {
            let mut c = AssetCandidate::new("NORMAL", Market::BR);
            c.price = Some(Price::from_int(50));
            c.avg_volume = Some(Money::from_int(2_000_000));
            c.price_days = 30;
            c.has_fundamentals = true;
            c.volatility = Some(0.25);
            c.score = Some(0.75);
            c
        },
        // High volatility
        {
            let mut c = AssetCandidate::new("HIGH_VOL", Market::BR);
            c.price = Some(Price::from_int(50));
            c.avg_volume = Some(Money::from_int(2_000_000));
            c.price_days = 30;
            c.has_fundamentals = true;
            c.volatility = Some(1.50); // 150% annual vol
            c.score = Some(0.70);
            c
        },
        // Extreme volatility (500%)
        {
            let mut c = AssetCandidate::new("EXTREME_VOL", Market::BR);
            c.price = Some(Price::from_int(50));
            c.avg_volume = Some(Money::from_int(2_000_000));
            c.price_days = 30;
            c.has_fundamentals = true;
            c.volatility = Some(5.0); // 500% annual vol
            c.score = Some(0.65);
            c
        },
    ];

    let capital = Money::from_int(500_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let ctx = EntryContext::new(fixed_date(), capital, Market::BR);

    let (result, orders, _) = engine.evaluate(&ctx, &candidates, &positions);

    // All should be selected (5 candidates, top_n=20)
    assert_eq!(result.targets.len(), 5, "All 5 should be selected");

    // No single weight should dominate completely (explosion from vol~0)
    // With 5 assets, max reasonable weight is ~0.50 even with risk-parity extremes
    for target in &result.targets {
        assert!(
            target.target_weight <= 0.60,
            "{} has weight {} showing weight explosion", target.symbol, target.target_weight
        );
    }

    // No weight explosion from near-zero vol - should be capped reasonably
    let zero_vol = result.targets.iter().find(|t| t.symbol == "ZERO_VOL");
    assert!(
        zero_vol.map(|t| t.target_weight).unwrap_or(0.0) <= 0.60,
        "Near-zero vol weight should be bounded"
    );

    // Weights should sum to ~1.0
    let weight_sum: f64 = result.targets.iter().map(|t| t.target_weight).sum();
    assert!(
        (weight_sum - 1.0).abs() < 0.01,
        "Weight sum {} should be ~1.0", weight_sum
    );

    // Orders generated successfully
    assert!(!orders.is_empty(), "Should generate orders");
}

// =============================================================================
// Stress Test 2: Threshold Edge Cases
// =============================================================================

/// Stress: Volume exactly at threshold ± 1.
/// Tests >= vs > boundary conditions.
#[test]
fn stress_threshold_edges() {
    // Use default GatingConfig (min_avg_volume_brl = 500_000)
    let threshold = Money::from_int(500_000);
    let engine = EntryEngine::new(EntryEngineConfig::default());

    let candidates: Vec<AssetCandidate> = vec![
        // Volume below threshold by 1
        {
            let mut c = AssetCandidate::new("BELOW", Market::BR);
            c.price = Some(Price::from_int(50));
            c.avg_volume = Some(Money::from_int(499_999)); // Below threshold
            c.price_days = 30;
            c.has_fundamentals = true;
            c.volatility = Some(0.25);
            c.score = Some(0.90);
            c
        },
        // Volume exactly at threshold
        {
            let mut c = AssetCandidate::new("EXACT", Market::BR);
            c.price = Some(Price::from_int(50));
            c.avg_volume = Some(threshold);
            c.price_days = 30;
            c.has_fundamentals = true;
            c.volatility = Some(0.25);
            c.score = Some(0.85);
            c
        },
        // Volume above threshold by 1
        {
            let mut c = AssetCandidate::new("ABOVE", Market::BR);
            c.price = Some(Price::from_int(50));
            c.avg_volume = Some(Money::from_int(500_001));
            c.price_days = 30;
            c.has_fundamentals = true;
            c.volatility = Some(0.25);
            c.score = Some(0.80);
            c
        },
    ];

    let capital = Money::from_int(500_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let ctx = EntryContext::new(fixed_date(), capital, Market::BR);

    let (result, _, _) = engine.evaluate(&ctx, &candidates, &positions);

    // BELOW should be excluded
    assert!(
        !result.targets.iter().any(|t| t.symbol == "BELOW"),
        "Below threshold should be excluded"
    );

    // EXACT should be included (>= threshold)
    assert!(
        result.targets.iter().any(|t| t.symbol == "EXACT"),
        "Exact threshold should be included"
    );

    // ABOVE should be included
    assert!(
        result.targets.iter().any(|t| t.symbol == "ABOVE"),
        "Above threshold should be included"
    );
}

/// Stress: Price exactly at min_price ± epsilon.
#[test]
fn stress_price_threshold_edges() {
    // Use default GatingConfig (min_price_brl = 1.0)
    let engine = EntryEngine::new(EntryEngineConfig::default());

    let candidates: Vec<AssetCandidate> = vec![
        // Price below by 0.01
        {
            let mut c = AssetCandidate::new("BELOW", Market::BR);
            c.price = Some(Price::from_f64(0.99));
            c.avg_volume = Some(Money::from_int(1_000_000));
            c.price_days = 30;
            c.has_fundamentals = true;
            c.volatility = Some(0.25);
            c.score = Some(0.90);
            c
        },
        // Price exactly at threshold
        {
            let mut c = AssetCandidate::new("EXACT", Market::BR);
            c.price = Some(Price::from_f64(1.00));
            c.avg_volume = Some(Money::from_int(1_000_000));
            c.price_days = 30;
            c.has_fundamentals = true;
            c.volatility = Some(0.25);
            c.score = Some(0.85);
            c
        },
        // Price just above
        {
            let mut c = AssetCandidate::new("ABOVE", Market::BR);
            c.price = Some(Price::from_f64(1.01));
            c.avg_volume = Some(Money::from_int(1_000_000));
            c.price_days = 30;
            c.has_fundamentals = true;
            c.volatility = Some(0.25);
            c.score = Some(0.80);
            c
        },
    ];

    let capital = Money::from_int(500_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let ctx = EntryContext::new(fixed_date(), capital, Market::BR);

    let (result, _, _) = engine.evaluate(&ctx, &candidates, &positions);

    // BELOW should be excluded
    assert!(
        !result.targets.iter().any(|t| t.symbol == "BELOW"),
        "Below min_price should be excluded"
    );

    // EXACT and ABOVE should be included
    assert!(
        result.targets.iter().any(|t| t.symbol == "EXACT"),
        "Exact min_price should be included"
    );
    assert!(
        result.targets.iter().any(|t| t.symbol == "ABOVE"),
        "Above min_price should be included"
    );
}

// =============================================================================
// Stress Test 3: Penny Stocks
// =============================================================================

/// Stress: Very low-priced stocks near min_price.
/// Tests lot sizing with small notionals.
#[test]
fn stress_penny_stocks() {
    // Use custom gating config via serde for lower min_price
    let gating: GatingConfig = serde_json::from_str(r#"{"min_price_brl_f64": 0.50}"#).unwrap();
    let config = EntryEngineConfig {
        gating,
        selection: SelectionConfig {
            top_n_br: 5,
            top_n_us: 5,
            min_score_threshold: None,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = EntryEngine::new(config);

    let candidates: Vec<AssetCandidate> = (0..10).map(|i| {
        let mut c = AssetCandidate::new(format!("PENNY{}", i), Market::BR);
        // Prices from R$0.50 to R$5.00
        c.price = Some(Price::from_f64(0.50 + (i as f64 * 0.50)));
        c.avg_volume = Some(Money::from_int(1_000_000));
        c.price_days = 30;
        c.has_fundamentals = true;
        c.volatility = Some(0.40 + (i as f64 * 0.02));
        c.score = Some(0.80 - (i as f64 * 0.03));
        c
    }).collect();

    let capital = Money::from_int(100_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let ctx = EntryContext::new(fixed_date(), capital, Market::BR);

    let (result, orders, _) = engine.evaluate(&ctx, &candidates, &positions);

    // Should select top 5
    assert_eq!(result.targets.len(), 5, "Should select 5");

    // All BR orders should be multiples of 100
    for order in &orders {
        assert!(
            order.shares % 100 == 0,
            "Penny stock {} has shares {} not multiple of 100",
            order.symbol, order.shares
        );
    }

    // Orders should have positive shares (low price = more shares per lot)
    for order in &orders {
        assert!(
            order.shares > 0,
            "Penny stock {} should have positive shares", order.symbol
        );
    }
}

// =============================================================================
// Stress Test 4: Large Universe (10,000 assets)
// =============================================================================

/// Stress: 10,000 assets - validates performance and correctness at scale.
#[test]
fn stress_large_universe() {
    let config = EntryEngineConfig {
        selection: SelectionConfig {
            top_n_br: 50,
            top_n_us: 50,
            min_score_threshold: None,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = EntryEngine::new(config);

    // Generate 10,000 candidates
    let candidates: Vec<AssetCandidate> = (0..10_000).map(|i| {
        let mut c = AssetCandidate::new(format!("ASSET{:05}", i), Market::BR);
        c.price = Some(Price::from_int(20 + (i as i64 % 100)));
        c.avg_volume = Some(Money::from_int(1_000_000 + (i as i64 * 1_000)));
        c.price_days = 30;
        c.has_fundamentals = i % 10 != 0;
        c.has_dividends = i % 5 != 0;
        c.volatility = Some(0.15 + ((i % 50) as f64) * 0.01);
        c.score = Some(0.95 - ((i % 100) as f64) * 0.009);
        c
    }).collect();

    let capital = Money::from_int(10_000_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let ctx = EntryContext::new(fixed_date(), capital, Market::BR);

    // Measure time
    let start = Instant::now();
    let (result, orders, _) = engine.evaluate(&ctx, &candidates, &positions);
    let elapsed = start.elapsed();

    // Performance: should complete in reasonable time (< 100ms)
    assert!(
        elapsed < Duration::from_millis(100),
        "10k assets took {:?}, should be < 100ms", elapsed
    );

    // Correctness: should select exactly 50
    assert_eq!(
        result.targets.len(), 50,
        "Should select exactly 50 from 10k"
    );

    // All weights should be valid
    let weight_sum: f64 = result.targets.iter().map(|t| t.target_weight).sum();
    assert!(
        (weight_sum - 1.0).abs() < 0.01,
        "Weight sum {} should be ~1.0", weight_sum
    );

    // All orders should be valid
    for order in &orders {
        assert!(order.shares % 100 == 0, "BR lot invariant");
        assert!(order.shares > 0, "No zero shares");
        assert!(!order.estimated_cost.is_negative(), "No negative costs");
    }

    println!("Large universe (N=10000) completed in {:?}", elapsed);
}

// =============================================================================
// Stress Test 5: High Turnover Scenarios
// =============================================================================

/// Stress: Existing positions with complete portfolio replacement.
/// Tests order generation and turnover calculation.
#[test]
fn stress_high_turnover() {
    let config = EntryEngineConfig {
        selection: SelectionConfig {
            top_n_br: 10,
            top_n_us: 10,
            min_score_threshold: None,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = EntryEngine::new(config);

    // Current holdings: OLD_0 to OLD_9
    let mut positions: HashMap<String, i64> = HashMap::new();
    for i in 0..10 {
        positions.insert(format!("OLD_{}", i), 1000 + (i as i64 * 100));
    }

    // New candidates: NEW_0 to NEW_19 (different from current holdings)
    let candidates: Vec<AssetCandidate> = (0..20).map(|i| {
        let mut c = AssetCandidate::new(format!("NEW_{}", i), Market::BR);
        c.price = Some(Price::from_int(50 + i as i64 * 5));
        c.avg_volume = Some(Money::from_int(2_000_000));
        c.price_days = 30;
        c.has_fundamentals = true;
        c.volatility = Some(0.20 + (i as f64 * 0.01));
        c.score = Some(0.90 - (i as f64 * 0.02));
        c
    }).collect();

    let capital = Money::from_int(1_000_000);
    let ctx = EntryContext::new(fixed_date(), capital, Market::BR);

    let (result, orders, audit) = engine.evaluate(&ctx, &candidates, &positions);

    // Should select 10 new assets
    assert_eq!(result.targets.len(), 10, "Should select 10 new assets");

    // Should generate BUY orders for new assets
    let buy_orders: Vec<_> = orders.iter().filter(|o| o.side == OrderSide::Buy).collect();
    assert!(!buy_orders.is_empty(), "Should have buy orders for new assets");

    // Turnover should be high (buying new positions)
    assert!(
        result.diagnostics.turnover > 0.0,
        "Turnover should be positive"
    );

    // Total costs should be calculated
    assert!(
        !result.diagnostics.estimated_costs.is_negative(),
        "Estimated costs should be non-negative"
    );

    // Audit should report turnover
    let summary = audit.to_summary();
    assert!(summary.contains("Turnover"), "Audit should mention turnover");

    println!("Turnover: {:.1}%", result.diagnostics.turnover * 100.0);
    println!("Estimated costs: {}", result.diagnostics.estimated_costs);
}

/// Stress: Partial portfolio update (some holdings stay, some change).
#[test]
fn stress_partial_turnover() {
    let config = EntryEngineConfig {
        selection: SelectionConfig {
            top_n_br: 5,
            top_n_us: 5,
            min_score_threshold: None,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = EntryEngine::new(config);

    // Current holdings: SYM0, SYM1, SYM2
    let mut positions: HashMap<String, i64> = HashMap::new();
    positions.insert("SYM0000".to_string(), 500);
    positions.insert("SYM0001".to_string(), 600);
    positions.insert("SYM0002".to_string(), 400);

    // Candidates: SYM0 to SYM9 (SYM0-2 overlap with holdings)
    let candidates: Vec<AssetCandidate> = (0..10).map(|i| {
        let mut c = AssetCandidate::new(format!("SYM{:04}", i), Market::BR);
        c.price = Some(Price::from_int(40 + i as i64 * 5));
        c.avg_volume = Some(Money::from_int(2_000_000));
        c.price_days = 30;
        c.has_fundamentals = true;
        c.volatility = Some(0.20 + (i as f64 * 0.01));
        c.score = Some(0.90 - (i as f64 * 0.02));
        c
    }).collect();

    let capital = Money::from_int(500_000);
    let ctx = EntryContext::new(fixed_date(), capital, Market::BR);

    let (result, orders, _) = engine.evaluate(&ctx, &candidates, &positions);

    // Should select 5 assets
    assert_eq!(result.targets.len(), 5, "Should select 5 assets");

    // Orders may include adjustments to existing positions
    // and buys for new positions
    for order in &orders {
        assert!(order.shares % 100 == 0, "BR lot invariant");
        assert!(order.shares > 0, "No zero shares");
    }

    // Turnover should be less than 100% with overlap
    // (but could still be high depending on weight changes)
    println!("Partial turnover: {:.1}%", result.diagnostics.turnover * 100.0);
}

// =============================================================================
// Performance Smoke Test (C2)
// =============================================================================

/// Performance smoke: N=1000 should complete under 50ms.
/// This is a generous margin to avoid flakiness.
#[test]
fn perf_smoke_1k_under_50ms() {
    let config = EntryEngineConfig {
        selection: SelectionConfig {
            top_n_br: 20,
            top_n_us: 20,
            min_score_threshold: None,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = EntryEngine::new(config);

    let candidates: Vec<AssetCandidate> = (0..1_000).map(|i| {
        let mut c = AssetCandidate::new(format!("PERF{:04}", i), Market::BR);
        c.price = Some(Price::from_int(20 + (i as i64 % 100)));
        c.avg_volume = Some(Money::from_int(1_000_000 + (i as i64 * 1_000)));
        c.price_days = 30;
        c.has_fundamentals = true;
        c.volatility = Some(0.15 + ((i % 50) as f64) * 0.01);
        c.score = Some(0.90 - ((i % 100) as f64) * 0.008);
        c
    }).collect();

    let capital = Money::from_int(1_000_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let ctx = EntryContext::new(fixed_date(), capital, Market::BR);

    let start = Instant::now();
    let (result, _, _) = engine.evaluate(&ctx, &candidates, &positions);
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_millis(50),
        "N=1000 took {:?}, should be < 50ms", elapsed
    );

    // Sanity check
    assert_eq!(result.targets.len(), 20, "Should select 20");
    println!("Perf smoke N=1000: {:?}", elapsed);
}
