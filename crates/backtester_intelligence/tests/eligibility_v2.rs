//! Integration Tests for V2 Eligibility (Timeline-based with V1 Fallback)
//!
//! Tests the V2 eligibility system that uses database listing_date/delisting_date
//! with automatic fallback to V1 (cache/universe.csv min_date/max_date).
//!
//! # Key Invariants Tested:
//! - V2 timeline takes precedence when available
//! - V1 fallback works when V2 data is missing
//! - Statistics track V2 hits vs V1 fallbacks
//! - No resurrection of delisted assets (same as V1)
//! - No pre-IPO trading (same as V1)

use backtester_intelligence::{
    DateRange, EligibilityDetails, EligibilityProvider, EligibilityResult, EligibilitySource,
    EligibilityStatsSnapshot, Timeline, TimelineEligibilityProvider, UniverseRangeProvider,
};
use chrono::NaiveDate;
use std::collections::HashMap;
use std::sync::Arc;

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

/// Create V1 fallback provider.
fn make_v1_fallback() -> Arc<UniverseRangeProvider> {
    let mut ranges = HashMap::new();
    
    // PETR4: V1 data 2015-01-02 to 2025-12-23
    ranges.insert(
        "PETR4".to_string(),
        DateRange::new(date(2015, 1, 2), date(2025, 12, 23)),
    );
    
    // VALE3: V1 data 2015-01-02 to 2025-12-23
    ranges.insert(
        "VALE3".to_string(),
        DateRange::new(date(2015, 1, 2), date(2025, 12, 23)),
    );
    
    // OIBR3: Delisted, V1 data 2015-01-02 to 2020-12-31
    ranges.insert(
        "OIBR3".to_string(),
        DateRange::new(date(2015, 1, 2), date(2020, 12, 31)),
    );
    
    // RAIZ4: IPO 2021-08-05, V1 data to 2025-12-23
    ranges.insert(
        "RAIZ4".to_string(),
        DateRange::new(date(2021, 8, 5), date(2025, 12, 23)),
    );
    
    Arc::new(UniverseRangeProvider::from_map(ranges))
}

/// Create V2 timelines (from database).
fn make_v2_timelines() -> HashMap<String, Timeline> {
    let mut timelines = HashMap::new();
    
    // PETR4: V2 has earlier listing date (IPO was actually 2000)
    timelines.insert(
        "PETR4".to_string(),
        Timeline::new(date(2000, 8, 10), None), // Active, real IPO date
    );
    
    // MGLU3: Only in V2 (newer stock, not in V1 CSV)
    timelines.insert(
        "MGLU3".to_string(),
        Timeline::new(date(2017, 5, 2), None), // Active
    );
    
    // OIBR3: V2 confirms delisting date
    timelines.insert(
        "OIBR3".to_string(),
        Timeline::new(date(2012, 1, 2), Some(date(2020, 12, 31))), // Delisted
    );
    
    timelines
}

// ============================================================================
// Precedence Tests
// ============================================================================

#[test]
fn test_v2_takes_precedence_over_v1() {
    let fallback = make_v1_fallback();
    let timelines = make_v2_timelines();
    let provider = TimelineEligibilityProvider::from_maps(timelines, fallback);
    
    // PETR4: V2 has listing_date = 2000-08-10
    // V1 has min_date = 2015-01-02
    
    // Date in 2010: V2 should allow it, V1 would reject it
    let result = provider.is_eligible("PETR4", date(2010, 6, 15));
    assert_eq!(result, EligibilityResult::Eligible);
    
    // Check source
    assert_eq!(provider.get_source("PETR4"), EligibilitySource::V2Timeline);
}

#[test]
fn test_v1_fallback_when_no_v2_data() {
    let fallback = make_v1_fallback();
    let timelines = make_v2_timelines();
    let provider = TimelineEligibilityProvider::from_maps(timelines, fallback);
    
    // VALE3 only has V1 data
    assert_eq!(provider.get_source("VALE3"), EligibilitySource::V1Range);
    
    // Should be eligible at date within V1 range
    let result = provider.is_eligible("VALE3", date(2020, 6, 15));
    assert_eq!(result, EligibilityResult::Eligible);
    
    // Should be ineligible before V1 min_date
    let result = provider.is_eligible("VALE3", date(2014, 12, 31));
    assert!(matches!(result, EligibilityResult::OutsideDateRange { .. }));
}

#[test]
fn test_v2_only_symbol() {
    let fallback = make_v1_fallback();
    let timelines = make_v2_timelines();
    let provider = TimelineEligibilityProvider::from_maps(timelines, fallback);
    
    // MGLU3 only exists in V2
    assert_eq!(provider.get_source("MGLU3"), EligibilitySource::V2Timeline);
    
    // Eligible after IPO
    assert_eq!(
        provider.is_eligible("MGLU3", date(2020, 1, 1)),
        EligibilityResult::Eligible
    );
    
    // Not eligible before IPO
    assert!(matches!(
        provider.is_eligible("MGLU3", date(2017, 5, 1)),
        EligibilityResult::OutsideDateRange { .. }
    ));
}

#[test]
fn test_unknown_symbol_not_found() {
    let fallback = make_v1_fallback();
    let timelines = make_v2_timelines();
    let provider = TimelineEligibilityProvider::from_maps(timelines, fallback);
    
    // Symbol not in V1 or V2
    assert_eq!(provider.get_source("UNKNOWN"), EligibilitySource::Unknown);
    assert_eq!(
        provider.is_eligible("UNKNOWN", date(2020, 1, 1)),
        EligibilityResult::SymbolNotInUniverse
    );
}

// ============================================================================
// Statistics Tests
// ============================================================================

#[test]
fn test_stats_track_v2_hits_vs_v1_fallbacks() {
    let fallback = make_v1_fallback();
    let timelines = make_v2_timelines();
    let provider = TimelineEligibilityProvider::from_maps(timelines, fallback);
    
    // V2 hit (PETR4)
    let _ = provider.is_eligible("PETR4", date(2020, 1, 1));
    // V2 hit (MGLU3)
    let _ = provider.is_eligible("MGLU3", date(2020, 1, 1));
    // V1 fallback (VALE3)
    let _ = provider.is_eligible("VALE3", date(2020, 1, 1));
    // V1 fallback (RAIZ4)
    let _ = provider.is_eligible("RAIZ4", date(2022, 1, 1));
    // Unknown (not found)
    let _ = provider.is_eligible("UNKNOWN", date(2020, 1, 1));
    
    let stats = provider.stats();
    
    // V2 hits: PETR4, MGLU3, OIBR3 is also V2 but we didn't check it
    assert_eq!(stats.v2_hits, 2, "Expected 2 V2 hits");
    // V1 fallbacks: VALE3, RAIZ4, UNKNOWN (UNKNOWN goes to V1 first, then not found)
    assert_eq!(stats.v1_fallbacks, 3, "Expected 3 V1 fallbacks");
    assert_eq!(stats.not_found, 1, "Expected 1 not found");
}

#[test]
fn test_v2_percentage() {
    let fallback = make_v1_fallback();
    let timelines = make_v2_timelines();
    let provider = TimelineEligibilityProvider::from_maps(timelines, fallback);
    
    // 3 V2 hits, 1 V1 fallback = 75%
    let _ = provider.is_eligible("PETR4", date(2020, 1, 1));
    let _ = provider.is_eligible("MGLU3", date(2020, 1, 1));
    let _ = provider.is_eligible("OIBR3", date(2020, 1, 1));
    let _ = provider.is_eligible("VALE3", date(2020, 1, 1));
    
    let stats = provider.stats();
    assert_eq!(stats.v2_hits, 3);
    assert_eq!(stats.v1_fallbacks, 1);
    assert!((stats.v2_percentage() - 0.75).abs() < 0.001);
}

// ============================================================================
// Details Tests
// ============================================================================

#[test]
fn test_get_details_returns_correct_source() {
    let fallback = make_v1_fallback();
    let timelines = make_v2_timelines();
    let provider = TimelineEligibilityProvider::from_maps(timelines, fallback);
    
    // V2 symbol
    let petr4 = provider.get_details("PETR4").unwrap();
    assert_eq!(petr4.source, EligibilitySource::V2Timeline);
    assert_eq!(petr4.listing_date, Some(date(2000, 8, 10)));
    assert_eq!(petr4.delisting_date, None); // Active
    
    // V1 fallback symbol
    let vale3 = provider.get_details("VALE3").unwrap();
    assert_eq!(vale3.source, EligibilitySource::V1Range);
    assert_eq!(vale3.listing_date, Some(date(2015, 1, 2)));
    assert_eq!(vale3.delisting_date, Some(date(2025, 12, 23)));
    
    // Unknown symbol
    let unknown = provider.get_details("UNKNOWN");
    assert!(unknown.is_none());
}

// ============================================================================
// Invariant Tests
// ============================================================================

#[test]
fn test_invariant_no_resurrection() {
    let fallback = make_v1_fallback();
    let timelines = make_v2_timelines();
    let provider = TimelineEligibilityProvider::from_maps(timelines, fallback);
    
    // OIBR3 delisted 2020-12-31
    // Should not be eligible in 2021+
    for year in 2021..=2025 {
        let result = provider.is_eligible("OIBR3", date(year, 6, 15));
        assert!(
            matches!(result, EligibilityResult::OutsideDateRange { .. }),
            "OIBR3 should not be eligible in {} (delisted 2020-12-31)",
            year
        );
    }
}

#[test]
fn test_invariant_no_pre_ipo() {
    let fallback = make_v1_fallback();
    let timelines = make_v2_timelines();
    let provider = TimelineEligibilityProvider::from_maps(timelines, fallback);
    
    // MGLU3 IPO 2017-05-02
    // Should not be eligible before IPO
    for year in 2010..=2016 {
        let result = provider.is_eligible("MGLU3", date(year, 6, 15));
        assert!(
            matches!(result, EligibilityResult::OutsideDateRange { .. }),
            "MGLU3 should not be eligible in {} (IPO 2017-05-02)",
            year
        );
    }
    
    // Day before IPO
    assert!(matches!(
        provider.is_eligible("MGLU3", date(2017, 5, 1)),
        EligibilityResult::OutsideDateRange { .. }
    ));
}

#[test]
fn test_v2_matches_v1_when_data_identical() {
    // When V2 and V1 have the same dates, behavior should match
    let mut ranges = HashMap::new();
    ranges.insert(
        "TEST".to_string(),
        DateRange::new(date(2015, 1, 1), date(2025, 12, 31)),
    );
    let fallback = Arc::new(UniverseRangeProvider::from_map(ranges));
    
    let mut timelines = HashMap::new();
    timelines.insert(
        "TEST".to_string(),
        Timeline::new(date(2015, 1, 1), Some(date(2025, 12, 31))),
    );
    
    let provider = TimelineEligibilityProvider::from_maps(timelines, fallback.clone());
    
    // Check same behavior at boundaries
    assert_eq!(
        provider.is_eligible("TEST", date(2015, 1, 1)),
        fallback.is_eligible("TEST", date(2015, 1, 1))
    );
    assert_eq!(
        provider.is_eligible("TEST", date(2020, 6, 15)),
        fallback.is_eligible("TEST", date(2020, 6, 15))
    );
    assert_eq!(
        provider.is_eligible("TEST", date(2025, 12, 31)),
        fallback.is_eligible("TEST", date(2025, 12, 31))
    );
}

#[test]
fn test_v2_divergence_uses_timeline() {
    // When V2 and V1 have different dates, V2 should take precedence
    let mut ranges = HashMap::new();
    ranges.insert(
        "DIVERGENT".to_string(),
        DateRange::new(date(2015, 1, 1), date(2025, 12, 31)), // V1
    );
    let fallback = Arc::new(UniverseRangeProvider::from_map(ranges));
    
    let mut timelines = HashMap::new();
    timelines.insert(
        "DIVERGENT".to_string(),
        Timeline::new(date(2010, 1, 1), Some(date(2020, 12, 31))), // V2: earlier start, earlier end
    );
    
    let provider = TimelineEligibilityProvider::from_maps(timelines, fallback);
    
    // Date in 2012: V2 allows, V1 would reject
    assert_eq!(
        provider.is_eligible("DIVERGENT", date(2012, 6, 15)),
        EligibilityResult::Eligible
    );
    
    // Date in 2022: V2 rejects (delisted), V1 would allow
    assert!(matches!(
        provider.is_eligible("DIVERGENT", date(2022, 6, 15)),
        EligibilityResult::OutsideDateRange { .. }
    ));
}

// ============================================================================
// V1-Only Mode Tests
// ============================================================================

#[test]
fn test_v1_only_mode() {
    let fallback = make_v1_fallback();
    let provider = TimelineEligibilityProvider::v1_only(fallback);
    
    assert_eq!(provider.v2_count(), 0);
    
    // All checks go to V1
    let _ = provider.is_eligible("PETR4", date(2020, 1, 1));
    let _ = provider.is_eligible("VALE3", date(2020, 1, 1));
    
    let stats = provider.stats();
    assert_eq!(stats.v2_hits, 0);
    assert_eq!(stats.v1_fallbacks, 2);
    
    // V1 only provides V1Range source
    assert_eq!(provider.get_source("PETR4"), EligibilitySource::V1Range);
}

#[test]
fn test_empty_provider() {
    let provider = TimelineEligibilityProvider::empty();
    
    assert_eq!(provider.v2_count(), 0);
    assert_eq!(provider.v1_count(), 0);
    
    // Everything is unknown
    assert_eq!(
        provider.is_eligible("ANY", date(2020, 1, 1)),
        EligibilityResult::SymbolNotInUniverse
    );
    assert_eq!(provider.get_source("ANY"), EligibilitySource::Unknown);
}

