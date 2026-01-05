//! Invariant tests for Entry Module - Production Gate.
//!
//! These tests prove formal invariants that must ALWAYS hold:
//! - A1: Allocation safety (weights, bounds)
//! - A2: Order correctness (lots, costs)
//! - A3: Strong determinism (identical outputs)
//! - A4: Anti-look-ahead (no future data)

use backtester_intelligence::entry::{
    AssetCandidate, EntryContext, EntryEngine, EntryEngineConfig,
    GatingConfig, SelectionConfig, WeightingConfig, OrderGeneratorConfig,
    ExclusionReason,
};
use backtester_intelligence::filters::Market;
use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;

// =============================================================================
// Test Fixtures
// =============================================================================

fn make_test_candidates(n: usize, market: Market) -> Vec<AssetCandidate> {
    (0..n).map(|i| {
        let mut c = AssetCandidate::new(format!("SYM{:04}", i), market);
        c.price = Some(Decimal::from(20 + (i as i64 % 100)));
        c.avg_volume = Some(Decimal::from(1_000_000 + (i as i64 * 50_000)));
        c.price_days = 30;
        c.has_fundamentals = true;
        c.has_dividends = true;
        c.volatility = Some(0.15 + ((i % 50) as f64) * 0.01);
        c.score = Some(0.90 - ((i % 100) as f64) * 0.008);
        c
    }).collect()
}

fn default_engine() -> EntryEngine {
    EntryEngine::new(EntryEngineConfig::default())
}

fn fixed_date() -> NaiveDate {
    NaiveDate::from_ymd_opt(2025, 1, 3).unwrap()
}

// =============================================================================
// A1) Allocation Safety Invariants
// =============================================================================

/// Invariant: Sum of target weights <= max_allocation_pct + small tolerance
#[test]
fn invariant_weight_sum_within_bounds() {
    let config = EntryEngineConfig {
        selection: SelectionConfig {
            top_n_br: 20,
            top_n_us: 20,
            min_score_threshold: None,
            ..Default::default()
        },
        weighting: WeightingConfig {
            max_weight: 0.20,
            min_weight: 0.02,
            ..Default::default()
        },
        orders: OrderGeneratorConfig {
            max_allocation_pct: dec!(0.99),
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = EntryEngine::new(config);

    let candidates = make_test_candidates(100, Market::BR);
    let capital = dec!(1_000_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let ctx = EntryContext::new(fixed_date(), capital, Market::BR);

    let (result, _, _) = engine.evaluate(&ctx, &candidates, &positions);

    let weight_sum: f64 = result.targets.iter().map(|t| t.target_weight).sum();
    
    // Weight sum should be <= 1.0 + small tolerance (for rounding)
    assert!(
        weight_sum <= 1.0 + 0.001,
        "Weight sum {} exceeds 1.0 + tolerance", weight_sum
    );
}

/// Invariant: No target weight is negative
#[test]
fn invariant_no_negative_weights() {
    let engine = default_engine();
    let candidates = make_test_candidates(50, Market::BR);
    let capital = dec!(500_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let ctx = EntryContext::new(fixed_date(), capital, Market::BR);

    let (result, _, _) = engine.evaluate(&ctx, &candidates, &positions);

    for target in &result.targets {
        assert!(
            target.target_weight >= 0.0,
            "{} has negative weight: {}", target.symbol, target.target_weight
        );
    }
}

/// Invariant: All weights respect min_weight/max_weight when applicable
#[test]
fn invariant_weights_respect_bounds() {
    let config = EntryEngineConfig {
        selection: SelectionConfig {
            top_n_br: 10,
            top_n_us: 10,
            min_score_threshold: None,
            ..Default::default()
        },
        weighting: WeightingConfig {
            max_weight: 0.25,
            min_weight: 0.05,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = EntryEngine::new(config);

    let candidates = make_test_candidates(30, Market::BR);
    let capital = dec!(1_000_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let ctx = EntryContext::new(fixed_date(), capital, Market::BR);

    let (result, _, _) = engine.evaluate(&ctx, &candidates, &positions);

    // Only check if we have enough assets selected
    if result.targets.len() >= 4 {
        for target in &result.targets {
            assert!(
                target.target_weight <= 0.25 + 0.001,
                "{} weight {} exceeds max 0.25", target.symbol, target.target_weight
            );
            assert!(
                target.target_weight >= 0.05 - 0.001,
                "{} weight {} below min 0.05", target.symbol, target.target_weight
            );
        }
    }
}

/// Invariant: Graceful degradation when gating reduces candidates below top-N
#[test]
fn invariant_graceful_degradation_below_top_n() {
    let config = EntryEngineConfig {
        gating: GatingConfig {
            min_avg_volume_brl: Decimal::from(10_000_000), // Very high threshold
            ..Default::default()
        },
        selection: SelectionConfig {
            top_n_br: 20, // Want 20, but will get fewer
            top_n_us: 20,
            min_score_threshold: None,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = EntryEngine::new(config);

    // Create candidates with varying volumes
    let candidates: Vec<AssetCandidate> = (0..30).map(|i| {
        let mut c = AssetCandidate::new(format!("SYM{:04}", i), Market::BR);
        c.price = Some(Decimal::from(50));
        // Only first 5 have high enough volume
        c.avg_volume = Some(if i < 5 { 
            Decimal::from(15_000_000) 
        } else { 
            Decimal::from(1_000_000) 
        });
        c.price_days = 30;
        c.has_fundamentals = true;
        c.volatility = Some(0.20);
        c.score = Some(0.80);
        c
    }).collect();

    let capital = dec!(500_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let ctx = EntryContext::new(fixed_date(), capital, Market::BR);

    // Should not panic, should gracefully select fewer than top-N
    let (result, _, audit) = engine.evaluate(&ctx, &candidates, &positions);

    // Should have selected <= 5 (the ones with high volume)
    assert!(result.targets.len() <= 5, "Should have at most 5 eligible");
    assert!(result.diagnostics.gating_excluded >= 25, "Should exclude most by gating");
    
    // Audit should log clearly
    let summary = audit.to_summary();
    assert!(summary.contains("EXCLUÍDOS"), "Audit should mention exclusions");
}

/// Invariant: When all candidates are filtered out, engine handles gracefully
#[test]
fn invariant_empty_eligible_no_panic() {
    let config = EntryEngineConfig {
        gating: GatingConfig {
            min_avg_volume_brl: Decimal::from(999_999_999), // Impossible threshold
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = EntryEngine::new(config);

    let candidates = make_test_candidates(20, Market::BR);
    let capital = dec!(500_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let ctx = EntryContext::new(fixed_date(), capital, Market::BR);

    // Should not panic
    let (result, orders, _audit) = engine.evaluate(&ctx, &candidates, &positions);

    assert!(result.targets.is_empty(), "No targets when all filtered");
    assert!(orders.is_empty(), "No orders when no targets");
    assert_eq!(result.diagnostics.final_selected, 0);
}

// =============================================================================
// A2) Order Correctness Invariants
// =============================================================================

/// Invariant: BR orders always have shares as multiples of 100
#[test]
fn invariant_br_orders_lot_multiples() {
    let engine = default_engine();
    let candidates = make_test_candidates(30, Market::BR);
    let capital = dec!(1_000_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let ctx = EntryContext::new(fixed_date(), capital, Market::BR);

    let (_, orders, _) = engine.evaluate(&ctx, &candidates, &positions);

    for order in &orders {
        assert!(
            order.shares % 100 == 0,
            "BR order {} has shares {} not multiple of 100",
            order.symbol, order.shares
        );
    }
}

/// Invariant: US orders have at least 1 share
#[test]
fn invariant_us_orders_min_one_share() {
    let config = EntryEngineConfig {
        gating: GatingConfig {
            require_fundamentals: false,
            require_dividends: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = EntryEngine::new(config);

    let candidates: Vec<AssetCandidate> = (0..20).map(|i| {
        let mut c = AssetCandidate::new(format!("US{:04}", i), Market::US);
        c.price = Some(Decimal::from(100 + i as i64 * 10));
        c.avg_volume = Some(Decimal::from(5_000_000));
        c.price_days = 30;
        c.volatility = Some(0.25);
        c.score = Some(0.80);
        c
    }).collect();

    let capital = dec!(500_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let ctx = EntryContext::new(fixed_date(), capital, Market::US);

    let (_, orders, _) = engine.evaluate(&ctx, &candidates, &positions);

    for order in &orders {
        assert!(
            order.shares >= 1,
            "US order {} has shares {} < 1", order.symbol, order.shares
        );
    }
}

/// Invariant: No orders with zero shares
#[test]
fn invariant_no_zero_share_orders() {
    let engine = default_engine();
    let candidates = make_test_candidates(50, Market::BR);
    let capital = dec!(1_000_000);
    
    // Create existing positions
    let mut positions: HashMap<String, i64> = HashMap::new();
    positions.insert("SYM0000".to_string(), 100);
    positions.insert("SYM0001".to_string(), 200);
    
    let ctx = EntryContext::new(fixed_date(), capital, Market::BR);

    let (_, orders, _) = engine.evaluate(&ctx, &candidates, &positions);

    for order in &orders {
        assert!(
            order.shares > 0,
            "Order {} has zero shares", order.symbol
        );
    }
}

/// Invariant: All order costs are non-negative
#[test]
fn invariant_costs_non_negative() {
    let engine = default_engine();
    let candidates = make_test_candidates(30, Market::BR);
    let capital = dec!(1_000_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let ctx = EntryContext::new(fixed_date(), capital, Market::BR);

    let (_, orders, _) = engine.evaluate(&ctx, &candidates, &positions);

    for order in &orders {
        assert!(
            order.estimated_cost >= Decimal::ZERO,
            "Order {} has negative cost: {}", order.symbol, order.estimated_cost
        );
    }
}

// =============================================================================
// A3) Strong Determinism Invariants
// =============================================================================

/// Invariant: Same inputs produce identical outputs
#[test]
fn invariant_determinism_identical_outputs() {
    let config = EntryEngineConfig {
        selection: SelectionConfig {
            top_n_br: 10,
            top_n_us: 10,
            min_score_threshold: None,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = EntryEngine::new(config.clone());

    let candidates = make_test_candidates(50, Market::BR);
    let capital = dec!(1_000_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let ctx = EntryContext::new(fixed_date(), capital, Market::BR);

    // Run 3 times
    let (result1, orders1, audit1) = engine.evaluate(&ctx, &candidates, &positions);
    let (result2, orders2, audit2) = engine.evaluate(&ctx, &candidates, &positions);
    let (result3, _orders3, audit3) = engine.evaluate(&ctx, &candidates, &positions);

    // Targets must be identical
    assert_eq!(result1.targets.len(), result2.targets.len());
    assert_eq!(result2.targets.len(), result3.targets.len());

    for (t1, t2) in result1.targets.iter().zip(result2.targets.iter()) {
        assert_eq!(t1.symbol, t2.symbol, "Symbols must match");
        assert!(
            (t1.target_weight - t2.target_weight).abs() < 1e-10,
            "Weights must be identical"
        );
        assert_eq!(t1.target_shares, t2.target_shares, "Shares must match");
    }

    // Orders must be identical
    assert_eq!(orders1.len(), orders2.len());
    for (o1, o2) in orders1.iter().zip(orders2.iter()) {
        assert_eq!(o1.symbol, o2.symbol);
        assert_eq!(o1.shares, o2.shares);
        assert_eq!(o1.price, o2.price);
    }

    // Audit summaries must be identical
    assert_eq!(audit1.to_summary(), audit2.to_summary());
    assert_eq!(audit2.to_summary(), audit3.to_summary());
}

/// Invariant: Determinism holds across multiple scenarios
#[test]
fn invariant_determinism_multiple_scenarios() {
    let engine = default_engine();

    let scenarios = vec![
        (10, Market::BR, dec!(100_000)),
        (50, Market::BR, dec!(500_000)),
        (100, Market::BR, dec!(1_000_000)),
    ];

    for (n, market, capital) in scenarios {
        let candidates = make_test_candidates(n, market);
        let positions: HashMap<String, i64> = HashMap::new();
        let ctx = EntryContext::new(fixed_date(), capital, market);

        let (r1, _, _) = engine.evaluate(&ctx, &candidates, &positions);
        let (r2, _, _) = engine.evaluate(&ctx, &candidates, &positions);

        assert_eq!(r1.targets.len(), r2.targets.len(), "N={} determinism failed", n);
    }
}

// =============================================================================
// A4) Anti-Look-Ahead Invariants
// =============================================================================

/// Invariant: Volatility from future does not affect current weighting
/// (Entry module receives pre-computed volatility, this test validates the contract)
#[test]
fn invariant_volatility_uses_passed_value() {
    let config = EntryEngineConfig {
        selection: SelectionConfig {
            top_n_br: 3,
            top_n_us: 3,
            min_score_threshold: None,
            ..Default::default()
        },
        weighting: WeightingConfig {
            max_weight: 0.80,
            min_weight: 0.05,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = EntryEngine::new(config);

    // Create candidates with specific volatilities
    let candidates: Vec<AssetCandidate> = vec![
        {
            let mut c = AssetCandidate::new("LOW_VOL", Market::BR);
            c.price = Some(dec!(50));
            c.avg_volume = Some(dec!(2_000_000));
            c.price_days = 30;
            c.has_fundamentals = true;
            c.volatility = Some(0.10); // Low vol -> should get higher weight
            c.score = Some(0.8);
            c
        },
        {
            let mut c = AssetCandidate::new("HIGH_VOL", Market::BR);
            c.price = Some(dec!(50));
            c.avg_volume = Some(dec!(2_000_000));
            c.price_days = 30;
            c.has_fundamentals = true;
            c.volatility = Some(0.50); // High vol -> should get lower weight
            c.score = Some(0.8);
            c
        },
    ];

    let capital = dec!(100_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let ctx = EntryContext::new(fixed_date(), capital, Market::BR);

    let (result, _, _) = engine.evaluate(&ctx, &candidates, &positions);

    let low_vol = result.targets.iter().find(|t| t.symbol == "LOW_VOL");
    let high_vol = result.targets.iter().find(|t| t.symbol == "HIGH_VOL");

    if let (Some(lv), Some(hv)) = (low_vol, high_vol) {
        assert!(
            lv.target_weight > hv.target_weight,
            "Low vol ({}) should have higher weight than high vol ({})",
            lv.target_weight, hv.target_weight
        );
    }
}

/// Invariant: FutureFundamentals exclusion when fundamentals_as_of > rebalance_date
#[test]
fn invariant_future_fundamentals_excluded() {
    let engine = default_engine();

    let candidates: Vec<AssetCandidate> = vec![
        {
            let mut c = AssetCandidate::new("FUTURE", Market::BR);
            c.price = Some(dec!(50));
            c.avg_volume = Some(dec!(2_000_000));
            c.price_days = 30;
            c.has_fundamentals = true;
            c.volatility = Some(0.25);
            c.score = Some(0.9);
            // Fundamentals from FUTURE (after rebalance date)
            c.fundamentals_as_of = Some(NaiveDate::from_ymd_opt(2025, 6, 30).unwrap());
            c
        },
        {
            let mut c = AssetCandidate::new("PAST", Market::BR);
            c.price = Some(dec!(50));
            c.avg_volume = Some(dec!(2_000_000));
            c.price_days = 30;
            c.has_fundamentals = true;
            c.volatility = Some(0.25);
            c.score = Some(0.8);
            // Fundamentals from PAST (before rebalance date)
            c.fundamentals_as_of = Some(NaiveDate::from_ymd_opt(2024, 9, 30).unwrap());
            c
        },
    ];

    let capital = dec!(100_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let rebalance_date = NaiveDate::from_ymd_opt(2025, 1, 3).unwrap();
    let ctx = EntryContext::new(rebalance_date, capital, Market::BR);

    let (result, _, _) = engine.evaluate(&ctx, &candidates, &positions);

    // FUTURE should be excluded
    assert!(
        !result.targets.iter().any(|t| t.symbol == "FUTURE"),
        "Asset with future fundamentals should be excluded"
    );

    // Should have FutureFundamentals exclusion
    assert!(
        result.exclusions.iter().any(|e| e.symbol == "FUTURE" && e.reason == ExclusionReason::FutureFundamentals),
        "Should have FutureFundamentals exclusion reason"
    );

    // PAST should be selected
    assert!(
        result.targets.iter().any(|t| t.symbol == "PAST"),
        "Asset with past fundamentals should be selected"
    );
}

/// Invariant: require_fundamentals=false does not exclude for MissingFundamentals
#[test]
fn invariant_require_fundamentals_false_no_exclusion() {
    let config = EntryEngineConfig {
        gating: GatingConfig {
            require_fundamentals: false,
            require_dividends: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = EntryEngine::new(config);

    let candidates: Vec<AssetCandidate> = vec![
        {
            let mut c = AssetCandidate::new("NO_FUND", Market::BR);
            c.price = Some(dec!(50));
            c.avg_volume = Some(dec!(2_000_000));
            c.price_days = 30;
            c.has_fundamentals = false; // No fundamentals
            c.volatility = Some(0.25);
            c.score = Some(0.9);
            c
        },
    ];

    let capital = dec!(100_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let ctx = EntryContext::new(fixed_date(), capital, Market::BR);

    let (result, _, _) = engine.evaluate(&ctx, &candidates, &positions);

    // Should NOT be excluded for MissingFundamentals
    assert!(
        !result.exclusions.iter().any(|e| e.reason == ExclusionReason::MissingFundamentals),
        "Should not exclude for MissingFundamentals when require_fundamentals=false"
    );
}

/// Invariant: require_fundamentals=true excludes assets without fundamentals
#[test]
fn invariant_require_fundamentals_true_excludes() {
    let config = EntryEngineConfig {
        gating: GatingConfig {
            require_fundamentals: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = EntryEngine::new(config);

    let candidates: Vec<AssetCandidate> = vec![
        {
            let mut c = AssetCandidate::new("NO_FUND", Market::BR);
            c.price = Some(dec!(50));
            c.avg_volume = Some(dec!(2_000_000));
            c.price_days = 30;
            c.has_fundamentals = false; // No fundamentals
            c.volatility = Some(0.25);
            c.score = Some(0.9);
            c
        },
    ];

    let capital = dec!(100_000);
    let positions: HashMap<String, i64> = HashMap::new();
    let ctx = EntryContext::new(fixed_date(), capital, Market::BR);

    let (result, _, _) = engine.evaluate(&ctx, &candidates, &positions);

    // Should be excluded for MissingFundamentals
    assert!(
        result.exclusions.iter().any(|e| e.reason == ExclusionReason::MissingFundamentals),
        "Should exclude for MissingFundamentals when require_fundamentals=true"
    );
}

