//! Timeline Eligibility Provider (V2) - Event-based eligibility with V1 fallback.
//!
//! This provider uses database `listing_date`/`delisting_date` when available,
//! falling back to V1's `min_date`/`max_date` from CSV when not.
//!
//! # Precedence
//!
//! 1. If symbol has `listing_date` in DB → use V2 timeline
//! 2. Else if symbol exists in CSV → use V1 range
//! 3. Else → exclude (not found)
//!
//! # Usage
//!
//! ```ignore
//! let provider = TimelineEligibilityProvider::from_maps(timelines, fallback);
//! let result = provider.is_eligible("PETR4", date);
//! ```

use chrono::NaiveDate;
use std::collections::HashMap;
use std::sync::Arc;

use super::eligibility::{
    EligibilityDetails, EligibilityProvider, EligibilitySource, EligibilityStats,
    EligibilityStatsSnapshot,
};
use super::universe_range::{EligibilityResult, UniverseRangeProvider};

// ============================================================================
// Timeline
// ============================================================================

/// Timeline entry for a single symbol.
#[derive(Debug, Clone)]
pub struct Timeline {
    /// First trading date (IPO or first data)
    pub listing_date: NaiveDate,
    /// Last trading date (None if still active)
    pub delisting_date: Option<NaiveDate>,
}

impl Timeline {
    /// Create a new timeline.
    pub fn new(listing_date: NaiveDate, delisting_date: Option<NaiveDate>) -> Self {
        Self {
            listing_date,
            delisting_date,
        }
    }

    /// Check if date is within the timeline.
    pub fn contains(&self, date: NaiveDate) -> bool {
        if date < self.listing_date {
            return false;
        }
        match self.delisting_date {
            Some(delisting) => date <= delisting,
            None => true, // Still active
        }
    }

    /// Check if date is before listing.
    pub fn is_before_listing(&self, date: NaiveDate) -> bool {
        date < self.listing_date
    }

    /// Check if date is after delisting.
    pub fn is_after_delisting(&self, date: NaiveDate) -> bool {
        match self.delisting_date {
            Some(delisting) => date > delisting,
            None => false,
        }
    }
}

// ============================================================================
// Timeline Eligibility Provider
// ============================================================================

/// V2 Timeline-based eligibility provider with V1 fallback.
pub struct TimelineEligibilityProvider {
    /// V2 data: symbol -> timeline
    timelines: HashMap<String, Timeline>,
    /// V1 fallback provider
    fallback: Arc<UniverseRangeProvider>,
    /// Statistics for telemetry
    stats: EligibilityStats,
}

impl TimelineEligibilityProvider {
    /// Create from pre-built maps (for testing or offline use).
    pub fn from_maps(
        timelines: HashMap<String, Timeline>,
        fallback: Arc<UniverseRangeProvider>,
    ) -> Self {
        Self {
            timelines,
            fallback,
            stats: EligibilityStats::new(),
        }
    }

    /// Create with only V1 fallback (V2 disabled).
    pub fn v1_only(fallback: Arc<UniverseRangeProvider>) -> Self {
        Self {
            timelines: HashMap::new(),
            fallback,
            stats: EligibilityStats::new(),
        }
    }

    /// Create empty provider (for testing).
    pub fn empty() -> Self {
        Self {
            timelines: HashMap::new(),
            fallback: Arc::new(UniverseRangeProvider::empty()),
            stats: EligibilityStats::new(),
        }
    }

    /// Get number of V2 timelines loaded.
    pub fn v2_count(&self) -> usize {
        self.timelines.len()
    }

    /// Get number of V1 symbols in fallback.
    pub fn v1_count(&self) -> usize {
        self.fallback.len()
    }

    /// Check if symbol has V2 timeline data.
    pub fn has_v2_data(&self, symbol: &str) -> bool {
        self.timelines.contains_key(symbol)
    }

    /// Get the V2 timeline for a symbol if it exists.
    pub fn get_timeline(&self, symbol: &str) -> Option<&Timeline> {
        self.timelines.get(symbol)
    }

    /// Get the V1 fallback provider.
    pub fn fallback(&self) -> &UniverseRangeProvider {
        &self.fallback
    }

    /// Check timeline eligibility (internal).
    fn check_timeline(&self, timeline: &Timeline, date: NaiveDate) -> EligibilityResult {
        if timeline.is_before_listing(date) {
            self.stats.record_pre_listing();
            EligibilityResult::OutsideDateRange {
                min_date: timeline.listing_date,
                max_date: timeline.delisting_date.unwrap_or(timeline.listing_date),
            }
        } else if timeline.is_after_delisting(date) {
            self.stats.record_post_delisting();
            EligibilityResult::OutsideDateRange {
                min_date: timeline.listing_date,
                max_date: timeline.delisting_date.unwrap_or(timeline.listing_date),
            }
        } else {
            EligibilityResult::Eligible
        }
    }

    /// Create an Arc-wrapped provider for sharing.
    pub fn into_arc(self) -> Arc<Self> {
        Arc::new(self)
    }
}

impl EligibilityProvider for TimelineEligibilityProvider {
    fn is_eligible(&self, symbol: &str, date: NaiveDate) -> EligibilityResult {
        // 1. Try V2 timeline first
        if let Some(timeline) = self.timelines.get(symbol) {
            self.stats.record_v2_hit();
            return self.check_timeline(timeline, date);
        }

        // 2. Fallback to V1 range
        self.stats.record_v1_fallback();
        let result = self.fallback.is_eligible(symbol, date);

        // Track exclusion reasons from V1
        match &result {
            EligibilityResult::OutsideDateRange { min_date, .. } => {
                if date < *min_date {
                    self.stats.record_pre_listing();
                } else {
                    self.stats.record_post_delisting();
                }
            }
            EligibilityResult::SymbolNotInUniverse => {
                self.stats.record_not_found();
            }
            EligibilityResult::Eligible => {}
        }

        result
    }

    fn get_details(&self, symbol: &str) -> Option<EligibilityDetails> {
        // 1. Try V2 timeline first
        if let Some(timeline) = self.timelines.get(symbol) {
            return Some(EligibilityDetails::from_v2(
                timeline.listing_date,
                timeline.delisting_date,
            ));
        }

        // 2. Try V1 fallback
        if let Some(range) = self.fallback.get_range(symbol) {
            return Some(EligibilityDetails::from_v1(range.min_date, range.max_date));
        }

        None
    }

    fn stats(&self) -> EligibilityStatsSnapshot {
        self.stats.snapshot()
    }

    fn get_source(&self, symbol: &str) -> EligibilitySource {
        if self.timelines.contains_key(symbol) {
            EligibilitySource::V2Timeline
        } else if self.fallback.get_range(symbol).is_some() {
            EligibilitySource::V1Range
        } else {
            EligibilitySource::Unknown
        }
    }
}

// Make it thread-safe
unsafe impl Send for TimelineEligibilityProvider {}
unsafe impl Sync for TimelineEligibilityProvider {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entry::universe_range::DateRange;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    fn make_v1_fallback() -> Arc<UniverseRangeProvider> {
        let mut ranges = HashMap::new();
        // PETR4: V1 range 2015-01-02 to 2025-12-23
        ranges.insert(
            "PETR4".to_string(),
            DateRange::new(date(2015, 1, 2), date(2025, 12, 23)),
        );
        // VALE3: V1 range 2015-01-02 to 2025-12-23
        ranges.insert(
            "VALE3".to_string(),
            DateRange::new(date(2015, 1, 2), date(2025, 12, 23)),
        );
        // OIBR3: V1 range 2015-01-02 to 2020-12-31 (delisted)
        ranges.insert(
            "OIBR3".to_string(),
            DateRange::new(date(2015, 1, 2), date(2020, 12, 31)),
        );
        Arc::new(UniverseRangeProvider::from_map(ranges))
    }

    fn make_v2_timelines() -> HashMap<String, Timeline> {
        let mut timelines = HashMap::new();
        // PETR4: V2 timeline with slightly different listing date
        timelines.insert(
            "PETR4".to_string(),
            Timeline::new(date(2000, 1, 3), None), // Active, earlier listing
        );
        // MGLU3: Only in V2, not in V1
        timelines.insert(
            "MGLU3".to_string(),
            Timeline::new(date(2017, 5, 1), None),
        );
        timelines
    }

    // ========================================================================
    // Precedence Tests
    // ========================================================================

    #[test]
    fn test_v2_takes_precedence_over_v1() {
        let fallback = make_v1_fallback();
        let timelines = make_v2_timelines();
        let provider = TimelineEligibilityProvider::from_maps(timelines, fallback);

        // PETR4 has V2 listing_date = 2000-01-03
        // V1 has min_date = 2015-01-02
        // V2 should allow dates before V1 min_date
        let result = provider.is_eligible("PETR4", date(2010, 1, 1));
        assert_eq!(result, EligibilityResult::Eligible);

        // But V1 would reject this date
        let v1_result = provider.fallback.is_eligible("PETR4", date(2010, 1, 1));
        assert!(matches!(v1_result, EligibilityResult::OutsideDateRange { .. }));
    }

    #[test]
    fn test_v1_fallback_when_v2_missing() {
        let fallback = make_v1_fallback();
        let timelines = make_v2_timelines();
        let provider = TimelineEligibilityProvider::from_maps(timelines, fallback);

        // VALE3 only has V1 data
        assert!(!provider.has_v2_data("VALE3"));

        // Should use V1 fallback
        let result = provider.is_eligible("VALE3", date(2020, 1, 1));
        assert_eq!(result, EligibilityResult::Eligible);

        let source = provider.get_source("VALE3");
        assert_eq!(source, EligibilitySource::V1Range);
    }

    #[test]
    fn test_v2_only_symbol_works() {
        let fallback = make_v1_fallback();
        let timelines = make_v2_timelines();
        let provider = TimelineEligibilityProvider::from_maps(timelines, fallback);

        // MGLU3 only has V2 data
        assert!(provider.has_v2_data("MGLU3"));
        assert!(provider.fallback.get_range("MGLU3").is_none());

        let result = provider.is_eligible("MGLU3", date(2020, 1, 1));
        assert_eq!(result, EligibilityResult::Eligible);

        let source = provider.get_source("MGLU3");
        assert_eq!(source, EligibilitySource::V2Timeline);
    }

    // ========================================================================
    // Boundary Tests
    // ========================================================================

    #[test]
    fn test_listing_date_boundary() {
        let fallback = Arc::new(UniverseRangeProvider::empty());
        let mut timelines = HashMap::new();
        timelines.insert(
            "TEST".to_string(),
            Timeline::new(date(2020, 1, 15), None),
        );
        let provider = TimelineEligibilityProvider::from_maps(timelines, fallback);

        // At listing date: eligible
        assert_eq!(
            provider.is_eligible("TEST", date(2020, 1, 15)),
            EligibilityResult::Eligible
        );

        // Day before: not eligible
        assert!(matches!(
            provider.is_eligible("TEST", date(2020, 1, 14)),
            EligibilityResult::OutsideDateRange { .. }
        ));
    }

    #[test]
    fn test_delisting_date_boundary() {
        let fallback = Arc::new(UniverseRangeProvider::empty());
        let mut timelines = HashMap::new();
        timelines.insert(
            "TEST".to_string(),
            Timeline::new(date(2020, 1, 1), Some(date(2023, 6, 30))),
        );
        let provider = TimelineEligibilityProvider::from_maps(timelines, fallback);

        // At delisting date: eligible
        assert_eq!(
            provider.is_eligible("TEST", date(2023, 6, 30)),
            EligibilityResult::Eligible
        );

        // Day after: not eligible
        assert!(matches!(
            provider.is_eligible("TEST", date(2023, 7, 1)),
            EligibilityResult::OutsideDateRange { .. }
        ));
    }

    #[test]
    fn test_active_ticker_no_delisting() {
        let fallback = Arc::new(UniverseRangeProvider::empty());
        let mut timelines = HashMap::new();
        timelines.insert(
            "ACTIVE".to_string(),
            Timeline::new(date(2015, 1, 1), None), // No delisting = still active
        );
        let provider = TimelineEligibilityProvider::from_maps(timelines, fallback);

        // Far future should still be eligible
        assert_eq!(
            provider.is_eligible("ACTIVE", date(2099, 12, 31)),
            EligibilityResult::Eligible
        );
    }

    // ========================================================================
    // Stats Tests
    // ========================================================================

    #[test]
    fn test_stats_tracking() {
        let fallback = make_v1_fallback();
        let timelines = make_v2_timelines();
        let provider = TimelineEligibilityProvider::from_maps(timelines, fallback);

        // V2 hit
        let _ = provider.is_eligible("PETR4", date(2020, 1, 1));
        // V1 fallback
        let _ = provider.is_eligible("VALE3", date(2020, 1, 1));
        // Not found
        let _ = provider.is_eligible("UNKNOWN", date(2020, 1, 1));

        let stats = provider.stats();
        assert_eq!(stats.v2_hits, 1);
        assert_eq!(stats.v1_fallbacks, 2); // VALE3 + UNKNOWN both go to V1
        assert_eq!(stats.not_found, 1);
    }

    #[test]
    fn test_v2_percentage() {
        let fallback = make_v1_fallback();
        let timelines = make_v2_timelines();
        let provider = TimelineEligibilityProvider::from_maps(timelines, fallback);

        // 2 V2 hits, 1 V1 fallback
        let _ = provider.is_eligible("PETR4", date(2020, 1, 1));
        let _ = provider.is_eligible("MGLU3", date(2020, 1, 1));
        let _ = provider.is_eligible("VALE3", date(2020, 1, 1));

        let stats = provider.stats();
        assert_eq!(stats.v2_hits, 2);
        assert_eq!(stats.v1_fallbacks, 1);
        assert!((stats.v2_percentage() - 0.666).abs() < 0.01);
    }

    // ========================================================================
    // Details Tests
    // ========================================================================

    #[test]
    fn test_get_details_v2() {
        let fallback = make_v1_fallback();
        let timelines = make_v2_timelines();
        let provider = TimelineEligibilityProvider::from_maps(timelines, fallback);

        let details = provider.get_details("PETR4").unwrap();
        assert_eq!(details.source, EligibilitySource::V2Timeline);
        assert_eq!(details.listing_date, Some(date(2000, 1, 3)));
        assert_eq!(details.delisting_date, None);
    }

    #[test]
    fn test_get_details_v1() {
        let fallback = make_v1_fallback();
        let timelines = make_v2_timelines();
        let provider = TimelineEligibilityProvider::from_maps(timelines, fallback);

        let details = provider.get_details("VALE3").unwrap();
        assert_eq!(details.source, EligibilitySource::V1Range);
        assert_eq!(details.listing_date, Some(date(2015, 1, 2)));
        assert_eq!(details.delisting_date, Some(date(2025, 12, 23)));
    }

    #[test]
    fn test_get_details_unknown() {
        let fallback = make_v1_fallback();
        let timelines = make_v2_timelines();
        let provider = TimelineEligibilityProvider::from_maps(timelines, fallback);

        let details = provider.get_details("UNKNOWN");
        assert!(details.is_none());
    }

    // ========================================================================
    // Invariant Tests
    // ========================================================================

    #[test]
    fn test_invariant_no_resurrection() {
        let fallback = make_v1_fallback();
        let mut timelines = HashMap::new();
        // Delisted at end of 2020
        timelines.insert(
            "DELISTED".to_string(),
            Timeline::new(date(2015, 1, 1), Some(date(2020, 12, 31))),
        );
        let provider = TimelineEligibilityProvider::from_maps(timelines, fallback);

        // Should never be eligible after delisting
        for year in 2021..=2025 {
            let result = provider.is_eligible("DELISTED", date(year, 6, 15));
            assert!(
                matches!(result, EligibilityResult::OutsideDateRange { .. }),
                "DELISTED should not be eligible in {}",
                year
            );
        }
    }

    #[test]
    fn test_invariant_no_pre_ipo() {
        let fallback = Arc::new(UniverseRangeProvider::empty());
        let mut timelines = HashMap::new();
        // IPO in 2021
        timelines.insert(
            "IPO2021".to_string(),
            Timeline::new(date(2021, 8, 5), None),
        );
        let provider = TimelineEligibilityProvider::from_maps(timelines, fallback);

        // Should never be eligible before IPO
        for year in 2015..=2020 {
            let result = provider.is_eligible("IPO2021", date(year, 6, 15));
            assert!(
                matches!(result, EligibilityResult::OutsideDateRange { .. }),
                "IPO2021 should not be eligible in {}",
                year
            );
        }

        // Day before IPO
        let result = provider.is_eligible("IPO2021", date(2021, 8, 4));
        assert!(matches!(result, EligibilityResult::OutsideDateRange { .. }));
    }

    // ========================================================================
    // Timeline Tests
    // ========================================================================

    #[test]
    fn test_timeline_contains() {
        let timeline = Timeline::new(date(2020, 1, 1), Some(date(2023, 12, 31)));

        assert!(!timeline.contains(date(2019, 12, 31)));
        assert!(timeline.contains(date(2020, 1, 1)));
        assert!(timeline.contains(date(2022, 6, 15)));
        assert!(timeline.contains(date(2023, 12, 31)));
        assert!(!timeline.contains(date(2024, 1, 1)));
    }

    #[test]
    fn test_timeline_active_contains() {
        let timeline = Timeline::new(date(2020, 1, 1), None);

        assert!(!timeline.contains(date(2019, 12, 31)));
        assert!(timeline.contains(date(2020, 1, 1)));
        assert!(timeline.contains(date(2099, 12, 31))); // Active = no end
    }

    #[test]
    fn test_v1_only_mode() {
        let fallback = make_v1_fallback();
        let provider = TimelineEligibilityProvider::v1_only(fallback);

        assert_eq!(provider.v2_count(), 0);
        assert!(provider.v1_count() > 0);

        // Everything should use V1
        let _ = provider.is_eligible("PETR4", date(2020, 1, 1));
        let stats = provider.stats();
        assert_eq!(stats.v2_hits, 0);
        assert_eq!(stats.v1_fallbacks, 1);
    }
}

