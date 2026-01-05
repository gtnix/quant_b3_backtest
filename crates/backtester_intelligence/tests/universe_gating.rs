//! Integration Tests for Universe Gating (Survivorship Bias Prevention)
//!
//! Tests the V1 pragmatic universe implementation that uses `cache/universe.csv`
//! as source of truth for asset existence periods.
//!
//! # Key Invariants Tested:
//! - Assets outside their date range are excluded
//! - No "resurrection" of delisted assets
//! - Mixed eligibility scenarios work correctly
//! - Universe validation integrates with other gating checks

use backtester_core::{Money, Price};
use backtester_intelligence::{
    AssetCandidate, EntryEngine, EntryEngineConfig, EntryContext, ExclusionReason,
    DateRange, UniverseRangeProvider, Market, EligibilityProvider,
};
use chrono::NaiveDate;
use std::collections::HashMap;
use std::sync::Arc;

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

/// Create a test universe with varied date ranges.
fn make_test_universe() -> Arc<dyn EligibilityProvider> {
    let mut ranges = HashMap::new();
    
    // PETR4: Long-lived stock, 2015 - 2025
    ranges.insert(
        "PETR4".to_string(),
        DateRange::new(date(2015, 1, 2), date(2025, 12, 23)),
    );
    
    // VALE3: Long-lived stock, 2015 - 2025
    ranges.insert(
        "VALE3".to_string(),
        DateRange::new(date(2015, 1, 2), date(2025, 12, 23)),
    );
    
    // RAIZ4: IPO in August 2021, still active
    ranges.insert(
        "RAIZ4".to_string(),
        DateRange::new(date(2021, 8, 5), date(2025, 12, 23)),
    );
    
    // OIBR3: Existed but delisted at end of 2020
    ranges.insert(
        "OIBR3".to_string(),
        DateRange::new(date(2015, 1, 2), date(2020, 12, 31)),
    );
    
    // MGLU3: IPO in 2017, still active
    ranges.insert(
        "MGLU3".to_string(),
        DateRange::new(date(2017, 5, 1), date(2025, 12, 23)),
    );
    
    Arc::new(UniverseRangeProvider::from_map(ranges))
}

/// Create valid candidate with all gating requirements met.
fn make_valid_candidate(symbol: &str, _rebalance_date: NaiveDate) -> AssetCandidate {
    let mut c = AssetCandidate::new(symbol, Market::BR);
    c.price = Some(Price::from_int(30));
    c.avg_volume = Some(Money::from_int(5_000_000));
    c.price_days = 100;
    c.has_fundamentals = true;
    c.has_dividends = true;
    c.volatility = Some(0.25);
    c.score = Some(0.80);
    c
}

// ============================================================================
// Mixed Eligibility Tests
// ============================================================================

#[test]
fn test_mixed_eligibility_mid_2021() {
    // Rebalance date: 2021-09-01
    // Expected eligibility:
    // - PETR4: eligible (2015 - 2025)
    // - VALE3: eligible (2015 - 2025)
    // - RAIZ4: eligible (IPO'd 2021-08-05)
    // - OIBR3: excluded (delisted 2020-12-31)
    // - MGLU3: eligible (IPO'd 2017-05-01)
    
    let universe = make_test_universe();
    let config = EntryEngineConfig {
        eligibility_provider: Some(universe),
        ..Default::default()
    };
    let engine = EntryEngine::new(config);
    
    let rebalance_date = date(2021, 9, 1);
    let ctx = EntryContext::new(rebalance_date, Money::from_int(100_000), Market::BR);
    
    let candidates = vec![
        make_valid_candidate("PETR4", rebalance_date),
        make_valid_candidate("VALE3", rebalance_date),
        make_valid_candidate("RAIZ4", rebalance_date),
        make_valid_candidate("OIBR3", rebalance_date),
        make_valid_candidate("MGLU3", rebalance_date),
    ];
    
    let (result, _, _) = engine.evaluate(&ctx, &candidates, &HashMap::new());
    
    // Should exclude only OIBR3
    let excluded_symbols: Vec<&str> = result.exclusions.iter()
        .filter(|e| e.reason == ExclusionReason::OutsideUniverseDateRange)
        .map(|e| e.symbol.as_str())
        .collect();
    
    assert_eq!(excluded_symbols, vec!["OIBR3"], 
        "Only OIBR3 should be excluded for OutsideUniverseDateRange");
    
    // Total eligible after gating should be 4
    // (may be further filtered by selection)
    let selected_symbols: Vec<&str> = result.targets.iter()
        .map(|t| t.symbol.as_str())
        .collect();
    
    assert!(!selected_symbols.contains(&"OIBR3"),
        "OIBR3 should not be in selected targets");
}

#[test]
fn test_mixed_eligibility_early_2021() {
    // Rebalance date: 2021-01-15
    // Expected eligibility:
    // - PETR4: eligible
    // - VALE3: eligible
    // - RAIZ4: excluded (IPO not until August 2021)
    // - OIBR3: excluded (delisted 2020-12-31)
    // - MGLU3: eligible
    
    let universe = make_test_universe();
    let config = EntryEngineConfig {
        eligibility_provider: Some(universe),
        ..Default::default()
    };
    let engine = EntryEngine::new(config);
    
    let rebalance_date = date(2021, 1, 15);
    let ctx = EntryContext::new(rebalance_date, Money::from_int(100_000), Market::BR);
    
    let candidates = vec![
        make_valid_candidate("PETR4", rebalance_date),
        make_valid_candidate("VALE3", rebalance_date),
        make_valid_candidate("RAIZ4", rebalance_date),
        make_valid_candidate("OIBR3", rebalance_date),
        make_valid_candidate("MGLU3", rebalance_date),
    ];
    
    let (result, _, _) = engine.evaluate(&ctx, &candidates, &HashMap::new());
    
    // Should exclude RAIZ4 and OIBR3
    let excluded_universe: Vec<&str> = result.exclusions.iter()
        .filter(|e| e.reason == ExclusionReason::OutsideUniverseDateRange)
        .map(|e| e.symbol.as_str())
        .collect();
    
    assert!(excluded_universe.contains(&"OIBR3"),
        "OIBR3 should be excluded (delisted)");
    assert!(excluded_universe.contains(&"RAIZ4"),
        "RAIZ4 should be excluded (not yet IPO'd)");
    assert_eq!(excluded_universe.len(), 2,
        "Exactly 2 symbols should be excluded for date range");
}

// ============================================================================
// No Resurrection Tests
// ============================================================================

#[test]
fn test_no_resurrection_after_delisting() {
    // OIBR3 delisted at end of 2020, should never appear after that
    let universe = make_test_universe();
    let config = EntryEngineConfig {
        eligibility_provider: Some(universe),
        ..Default::default()
    };
    let engine = EntryEngine::new(config);
    
    // Try multiple dates after delisting
    let test_dates = vec![
        date(2021, 1, 1),
        date(2021, 6, 15),
        date(2022, 1, 1),
        date(2023, 6, 15),
        date(2024, 12, 1),
    ];
    
    for rebalance_date in test_dates {
        let ctx = EntryContext::new(rebalance_date, Money::from_int(100_000), Market::BR);
        let candidates = vec![make_valid_candidate("OIBR3", rebalance_date)];
        
        let (result, _, _) = engine.evaluate(&ctx, &candidates, &HashMap::new());
        
        assert!(result.targets.is_empty(),
            "OIBR3 should not be selected on {}", rebalance_date);
        assert_eq!(result.exclusions.len(), 1,
            "OIBR3 should be excluded on {}", rebalance_date);
        assert_eq!(result.exclusions[0].reason, ExclusionReason::OutsideUniverseDateRange,
            "Exclusion reason should be OutsideUniverseDateRange on {}", rebalance_date);
    }
}

#[test]
fn test_no_resurrection_before_ipo() {
    // RAIZ4 IPO'd August 2021, should never appear before that
    let universe = make_test_universe();
    let config = EntryEngineConfig {
        eligibility_provider: Some(universe),
        ..Default::default()
    };
    let engine = EntryEngine::new(config);
    
    // Try multiple dates before IPO
    let test_dates = vec![
        date(2020, 1, 1),
        date(2020, 12, 31),
        date(2021, 1, 1),
        date(2021, 7, 1),
        date(2021, 8, 4), // Day before IPO
    ];
    
    for rebalance_date in test_dates {
        let ctx = EntryContext::new(rebalance_date, Money::from_int(100_000), Market::BR);
        let candidates = vec![make_valid_candidate("RAIZ4", rebalance_date)];
        
        let (result, _, _) = engine.evaluate(&ctx, &candidates, &HashMap::new());
        
        assert!(result.targets.is_empty(),
            "RAIZ4 should not be selected on {}", rebalance_date);
        assert_eq!(result.exclusions.len(), 1,
            "RAIZ4 should be excluded on {}", rebalance_date);
        assert_eq!(result.exclusions[0].reason, ExclusionReason::OutsideUniverseDateRange,
            "Exclusion reason should be OutsideUniverseDateRange on {}", rebalance_date);
    }
}

// ============================================================================
// Boundary Date Tests
// ============================================================================

#[test]
fn test_eligible_at_exact_min_date() {
    let universe = make_test_universe();
    let config = EntryEngineConfig {
        eligibility_provider: Some(universe),
        ..Default::default()
    };
    let engine = EntryEngine::new(config);
    
    // RAIZ4 IPO'd exactly on 2021-08-05
    let rebalance_date = date(2021, 8, 5);
    let ctx = EntryContext::new(rebalance_date, Money::from_int(100_000), Market::BR);
    let candidates = vec![make_valid_candidate("RAIZ4", rebalance_date)];
    
    let (result, _, _) = engine.evaluate(&ctx, &candidates, &HashMap::new());
    
    // Should be eligible on exact IPO date
    let universe_exclusions: Vec<_> = result.exclusions.iter()
        .filter(|e| matches!(e.reason, 
            ExclusionReason::OutsideUniverseDateRange | 
            ExclusionReason::NoUniverseRangeData))
        .collect();
    
    assert!(universe_exclusions.is_empty(),
        "RAIZ4 should be eligible on its IPO date 2021-08-05");
}

#[test]
fn test_eligible_at_exact_max_date() {
    let universe = make_test_universe();
    let config = EntryEngineConfig {
        eligibility_provider: Some(universe),
        ..Default::default()
    };
    let engine = EntryEngine::new(config);
    
    // OIBR3 last day is 2020-12-31
    let rebalance_date = date(2020, 12, 31);
    let ctx = EntryContext::new(rebalance_date, Money::from_int(100_000), Market::BR);
    let candidates = vec![make_valid_candidate("OIBR3", rebalance_date)];
    
    let (result, _, _) = engine.evaluate(&ctx, &candidates, &HashMap::new());
    
    // Should be eligible on exact last day
    let universe_exclusions: Vec<_> = result.exclusions.iter()
        .filter(|e| matches!(e.reason, 
            ExclusionReason::OutsideUniverseDateRange | 
            ExclusionReason::NoUniverseRangeData))
        .collect();
    
    assert!(universe_exclusions.is_empty(),
        "OIBR3 should be eligible on its last day 2020-12-31");
}

// ============================================================================
// Unknown Symbol Tests
// ============================================================================

#[test]
fn test_unknown_symbol_excluded() {
    let universe = make_test_universe();
    let config = EntryEngineConfig {
        eligibility_provider: Some(universe),
        ..Default::default()
    };
    let engine = EntryEngine::new(config);
    
    let rebalance_date = date(2021, 6, 15);
    let ctx = EntryContext::new(rebalance_date, Money::from_int(100_000), Market::BR);
    let candidates = vec![make_valid_candidate("FAKE99", rebalance_date)];
    
    let (result, _, _) = engine.evaluate(&ctx, &candidates, &HashMap::new());
    
    assert!(result.targets.is_empty(),
        "Unknown symbol should not be selected");
    assert_eq!(result.exclusions.len(), 1);
    assert_eq!(result.exclusions[0].reason, ExclusionReason::NoUniverseRangeData,
        "Unknown symbol should be excluded with NoUniverseRangeData");
}

// ============================================================================
// Invariant Tests
// ============================================================================

#[test]
fn test_invariant_selected_within_range() {
    // For any selected candidate, rebalance_date must be within [min_date, max_date]
    let universe = make_test_universe();
    let config = EntryEngineConfig {
        eligibility_provider: Some(Arc::clone(&universe) as Arc<dyn EligibilityProvider>),
        ..Default::default()
    };
    let engine = EntryEngine::new(config);
    
    // Test across multiple dates
    let test_dates = vec![
        date(2016, 6, 15),
        date(2018, 3, 10),
        date(2020, 9, 1),
        date(2021, 10, 15),
        date(2023, 5, 20),
    ];
    
    for rebalance_date in test_dates {
        let ctx = EntryContext::new(rebalance_date, Money::from_int(100_000), Market::BR);
        let candidates = vec![
            make_valid_candidate("PETR4", rebalance_date),
            make_valid_candidate("VALE3", rebalance_date),
            make_valid_candidate("RAIZ4", rebalance_date),
            make_valid_candidate("OIBR3", rebalance_date),
            make_valid_candidate("MGLU3", rebalance_date),
        ];
        
        let (result, _, _) = engine.evaluate(&ctx, &candidates, &HashMap::new());
        
        // Verify invariant for all selected targets
        for target in &result.targets {
            let details = universe.get_details(&target.symbol)
                .expect(&format!("Selected symbol {} must be in universe", target.symbol));
            
            let listing_date = details.listing_date.expect("Must have listing date");
            assert!(rebalance_date >= listing_date,
                "INVARIANT VIOLATION: {} selected on {} but listing_date is {}",
                target.symbol, rebalance_date, listing_date);
            
            if let Some(delisting_date) = details.delisting_date {
                assert!(rebalance_date <= delisting_date,
                    "INVARIANT VIOLATION: {} selected on {} but delisting_date is {}",
                    target.symbol, rebalance_date, delisting_date);
            }
        }
    }
}

#[test]
fn test_invariant_eligible_count_decreases_over_time() {
    // After a symbol's max_date, the pool of eligible candidates should not increase
    // (unless new IPOs occur)
    
    let universe = make_test_universe();
    let config = EntryEngineConfig {
        eligibility_provider: Some(universe),
        ..Default::default()
    };
    let engine = EntryEngine::new(config);
    
    // Only test OIBR3 delisting impact
    // Before: OIBR3 is eligible
    let before_date = date(2020, 12, 31);
    let after_date = date(2021, 1, 1);
    
    let candidates_before: Vec<_> = ["PETR4", "OIBR3"]
        .iter()
        .map(|s| make_valid_candidate(s, before_date))
        .collect();
    
    let candidates_after: Vec<_> = ["PETR4", "OIBR3"]
        .iter()
        .map(|s| make_valid_candidate(s, after_date))
        .collect();
    
    let ctx_before = EntryContext::new(before_date, Money::from_int(100_000), Market::BR);
    let ctx_after = EntryContext::new(after_date, Money::from_int(100_000), Market::BR);
    
    let (result_before, _, _) = engine.evaluate(&ctx_before, &candidates_before, &HashMap::new());
    let (result_after, _, _) = engine.evaluate(&ctx_after, &candidates_after, &HashMap::new());
    
    // Count universe-eligible candidates (not excluded for date range)
    let eligible_before = result_before.diagnostics.total_candidates 
        - result_before.exclusions.iter()
            .filter(|e| e.reason == ExclusionReason::OutsideUniverseDateRange)
            .count();
    
    let eligible_after = result_after.diagnostics.total_candidates
        - result_after.exclusions.iter()
            .filter(|e| e.reason == ExclusionReason::OutsideUniverseDateRange)
            .count();
    
    assert!(eligible_before >= eligible_after,
        "Eligible count should not increase after delisting: before={}, after={}",
        eligible_before, eligible_after);
}

// ============================================================================
// Without Universe Provider (Backward Compatibility)
// ============================================================================

#[test]
fn test_without_universe_provider_no_filtering() {
    // Without universe provider, no universe filtering should occur
    let config = EntryEngineConfig::default();
    let engine = EntryEngine::new(config);
    
    // Even with invalid date (before any reasonable IPO), should pass
    let rebalance_date = date(2010, 1, 1);
    let ctx = EntryContext::new(rebalance_date, Money::from_int(100_000), Market::BR);
    let candidates = vec![make_valid_candidate("PETR4", rebalance_date)];
    
    let (result, _, _) = engine.evaluate(&ctx, &candidates, &HashMap::new());
    
    // Should not have any universe-related exclusions
    let universe_exclusions: Vec<_> = result.exclusions.iter()
        .filter(|e| matches!(e.reason, 
            ExclusionReason::OutsideUniverseDateRange | 
            ExclusionReason::NoUniverseRangeData))
        .collect();
    
    assert!(universe_exclusions.is_empty(),
        "Without universe provider, no universe exclusions should occur");
}

#[test]
fn test_has_universe_provider_flag() {
    let universe = make_test_universe();
    
    let config_with = EntryEngineConfig {
        eligibility_provider: Some(universe),
        ..Default::default()
    };
    let engine_with = EntryEngine::new(config_with);
    
    let config_without = EntryEngineConfig::default();
    let engine_without = EntryEngine::new(config_without);
    
    assert!(engine_with.has_universe_provider(),
        "Engine with provider should report has_universe_provider = true");
    assert!(!engine_without.has_universe_provider(),
        "Engine without provider should report has_universe_provider = false");
}
