//! Eligibility Provider Abstraction - V2 Event-Based Universe.
//!
//! Provides a trait-based abstraction for eligibility checking, supporting
//! both V1 (range-based from CSV) and V2 (timeline-based from database).
//!
//! # Precedence Rules
//!
//! 1. V2 Timeline: If DB has `listing_date`/`delisting_date`, use those
//! 2. V1 Range: Fallback to `cache/universe.csv` min_date/max_date
//! 3. Unknown: Exclude with `NoUniverseRangeData`
//!
//! # Telemetry
//!
//! All providers track statistics for auditing:
//! - V2 hits vs V1 fallbacks
//! - Exclusion breakdown by reason

use chrono::NaiveDate;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::universe_range::EligibilityResult;

// ============================================================================
// Eligibility Source
// ============================================================================

/// Source of eligibility data for a symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EligibilitySource {
    /// V2: From database listing_date/delisting_date
    V2Timeline,
    /// V1: From CSV min_date/max_date
    V1Range,
    /// Symbol not found in any source
    Unknown,
}

impl std::fmt::Display for EligibilitySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::V2Timeline => write!(f, "V2_TIMELINE"),
            Self::V1Range => write!(f, "V1_RANGE"),
            Self::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

// ============================================================================
// Eligibility Details
// ============================================================================

/// Detailed eligibility information for auditing.
#[derive(Debug, Clone)]
pub struct EligibilityDetails {
    /// First trading date (IPO or first data)
    pub listing_date: Option<NaiveDate>,
    /// Last trading date (delisting or None if active)
    pub delisting_date: Option<NaiveDate>,
    /// Source of this eligibility data
    pub source: EligibilitySource,
}

impl EligibilityDetails {
    /// Create new eligibility details.
    pub fn new(
        listing_date: Option<NaiveDate>,
        delisting_date: Option<NaiveDate>,
        source: EligibilitySource,
    ) -> Self {
        Self {
            listing_date,
            delisting_date,
            source,
        }
    }

    /// Create from V2 timeline data.
    pub fn from_v2(listing_date: NaiveDate, delisting_date: Option<NaiveDate>) -> Self {
        Self {
            listing_date: Some(listing_date),
            delisting_date,
            source: EligibilitySource::V2Timeline,
        }
    }

    /// Create from V1 range data.
    pub fn from_v1(min_date: NaiveDate, max_date: NaiveDate) -> Self {
        Self {
            listing_date: Some(min_date),
            delisting_date: Some(max_date),
            source: EligibilitySource::V1Range,
        }
    }

    /// Create for unknown symbol.
    pub fn unknown() -> Self {
        Self {
            listing_date: None,
            delisting_date: None,
            source: EligibilitySource::Unknown,
        }
    }

    /// Check if symbol is eligible at date.
    pub fn is_eligible_at(&self, date: NaiveDate) -> bool {
        match (self.listing_date, self.delisting_date) {
            (Some(listing), Some(delisting)) => date >= listing && date <= delisting,
            (Some(listing), None) => date >= listing,
            (None, Some(delisting)) => date <= delisting,
            (None, None) => false, // Unknown = not eligible
        }
    }
}

// ============================================================================
// Eligibility Statistics
// ============================================================================

/// Statistics for eligibility checks (telemetry).
#[derive(Debug, Default)]
pub struct EligibilityStats {
    /// Checks resolved by V2 timeline data
    pub v2_hits: AtomicUsize,
    /// Checks that fell back to V1 range data
    pub v1_fallbacks: AtomicUsize,
    /// Symbols not found in any source
    pub not_found: AtomicUsize,
    /// Excluded because date < listing_date
    pub excluded_pre_listing: AtomicUsize,
    /// Excluded because date > delisting_date
    pub excluded_post_delisting: AtomicUsize,
}

impl EligibilityStats {
    /// Create new stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get snapshot of current stats.
    pub fn snapshot(&self) -> EligibilityStatsSnapshot {
        EligibilityStatsSnapshot {
            v2_hits: self.v2_hits.load(Ordering::Relaxed),
            v1_fallbacks: self.v1_fallbacks.load(Ordering::Relaxed),
            not_found: self.not_found.load(Ordering::Relaxed),
            excluded_pre_listing: self.excluded_pre_listing.load(Ordering::Relaxed),
            excluded_post_delisting: self.excluded_post_delisting.load(Ordering::Relaxed),
        }
    }

    /// Record a V2 hit.
    pub fn record_v2_hit(&self) {
        self.v2_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a V1 fallback.
    pub fn record_v1_fallback(&self) {
        self.v1_fallbacks.fetch_add(1, Ordering::Relaxed);
    }

    /// Record not found.
    pub fn record_not_found(&self) {
        self.not_found.fetch_add(1, Ordering::Relaxed);
    }

    /// Record pre-listing exclusion.
    pub fn record_pre_listing(&self) {
        self.excluded_pre_listing.fetch_add(1, Ordering::Relaxed);
    }

    /// Record post-delisting exclusion.
    pub fn record_post_delisting(&self) {
        self.excluded_post_delisting.fetch_add(1, Ordering::Relaxed);
    }
}

/// Immutable snapshot of eligibility statistics.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct EligibilityStatsSnapshot {
    pub v2_hits: usize,
    pub v1_fallbacks: usize,
    pub not_found: usize,
    pub excluded_pre_listing: usize,
    pub excluded_post_delisting: usize,
}

impl EligibilityStatsSnapshot {
    /// Total checks performed.
    pub fn total_checks(&self) -> usize {
        self.v2_hits + self.v1_fallbacks + self.not_found
    }

    /// Total exclusions.
    pub fn total_exclusions(&self) -> usize {
        self.excluded_pre_listing + self.excluded_post_delisting + self.not_found
    }

    /// Percentage using V2 (0.0 to 1.0).
    pub fn v2_percentage(&self) -> f64 {
        let total = self.v2_hits + self.v1_fallbacks;
        if total == 0 {
            0.0
        } else {
            self.v2_hits as f64 / total as f64
        }
    }
}

// ============================================================================
// Eligibility Provider Trait
// ============================================================================

/// Trait for eligibility providers (V1 range-based or V2 event-based).
///
/// Implementations must be thread-safe (Send + Sync) for use in parallel backtests.
pub trait EligibilityProvider: Send + Sync {
    /// Check if symbol is eligible at date.
    ///
    /// Returns the same `EligibilityResult` enum as V1 for compatibility.
    fn is_eligible(&self, symbol: &str, date: NaiveDate) -> EligibilityResult;

    /// Get eligibility details for auditing.
    ///
    /// Returns None if symbol not found in any source.
    fn get_details(&self, symbol: &str) -> Option<EligibilityDetails>;

    /// Get statistics snapshot for telemetry.
    fn stats(&self) -> EligibilityStatsSnapshot;

    /// Get the source that would be used for a symbol (without checking date).
    fn get_source(&self, symbol: &str) -> EligibilitySource {
        self.get_details(symbol)
            .map(|d| d.source)
            .unwrap_or(EligibilitySource::Unknown)
    }
}

// ============================================================================
// Audit Summary
// ============================================================================

/// Summary of eligibility checks for audit log.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct EligibilityAuditSummary {
    /// Checks resolved by V2 timeline
    pub v2_timeline_checks: usize,
    /// Checks that used V1 fallback
    pub v1_fallback_checks: usize,
    /// Excluded before listing/IPO
    pub excluded_pre_listing: usize,
    /// Excluded after delisting
    pub excluded_post_delisting: usize,
    /// Excluded due to unknown symbol
    pub excluded_unknown: usize,
}

impl From<EligibilityStatsSnapshot> for EligibilityAuditSummary {
    fn from(stats: EligibilityStatsSnapshot) -> Self {
        Self {
            v2_timeline_checks: stats.v2_hits,
            v1_fallback_checks: stats.v1_fallbacks,
            excluded_pre_listing: stats.excluded_pre_listing,
            excluded_post_delisting: stats.excluded_post_delisting,
            excluded_unknown: stats.not_found,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    #[test]
    fn test_eligibility_details_is_eligible_at() {
        let details = EligibilityDetails::from_v2(date(2015, 1, 2), Some(date(2025, 12, 31)));

        assert!(details.is_eligible_at(date(2015, 1, 2))); // At listing
        assert!(details.is_eligible_at(date(2020, 6, 15))); // In middle
        assert!(details.is_eligible_at(date(2025, 12, 31))); // At delisting
        assert!(!details.is_eligible_at(date(2015, 1, 1))); // Before listing
        assert!(!details.is_eligible_at(date(2026, 1, 1))); // After delisting
    }

    #[test]
    fn test_eligibility_details_active_no_delisting() {
        let details = EligibilityDetails::from_v2(date(2015, 1, 2), None);

        assert!(details.is_eligible_at(date(2015, 1, 2))); // At listing
        assert!(details.is_eligible_at(date(2099, 12, 31))); // Far future (still active)
        assert!(!details.is_eligible_at(date(2015, 1, 1))); // Before listing
    }

    #[test]
    fn test_eligibility_details_unknown_not_eligible() {
        let details = EligibilityDetails::unknown();

        assert!(!details.is_eligible_at(date(2020, 1, 1)));
        assert!(!details.is_eligible_at(date(2025, 1, 1)));
    }

    #[test]
    fn test_eligibility_stats_tracking() {
        let stats = EligibilityStats::new();

        stats.record_v2_hit();
        stats.record_v2_hit();
        stats.record_v1_fallback();
        stats.record_not_found();
        stats.record_pre_listing();
        stats.record_post_delisting();

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.v2_hits, 2);
        assert_eq!(snapshot.v1_fallbacks, 1);
        assert_eq!(snapshot.not_found, 1);
        assert_eq!(snapshot.excluded_pre_listing, 1);
        assert_eq!(snapshot.excluded_post_delisting, 1);
        assert_eq!(snapshot.total_checks(), 4);
        assert_eq!(snapshot.total_exclusions(), 3);
    }

    #[test]
    fn test_v2_percentage() {
        let stats = EligibilityStats::new();

        // 3 V2 hits, 1 V1 fallback = 75%
        stats.record_v2_hit();
        stats.record_v2_hit();
        stats.record_v2_hit();
        stats.record_v1_fallback();

        let snapshot = stats.snapshot();
        assert!((snapshot.v2_percentage() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_v2_percentage_empty() {
        let stats = EligibilityStats::new();
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.v2_percentage(), 0.0);
    }

    #[test]
    fn test_eligibility_source_display() {
        assert_eq!(EligibilitySource::V2Timeline.to_string(), "V2_TIMELINE");
        assert_eq!(EligibilitySource::V1Range.to_string(), "V1_RANGE");
        assert_eq!(EligibilitySource::Unknown.to_string(), "UNKNOWN");
    }

    #[test]
    fn test_audit_summary_from_stats() {
        let stats = EligibilityStats::new();
        stats.record_v2_hit();
        stats.record_v1_fallback();
        stats.record_pre_listing();

        let summary: EligibilityAuditSummary = stats.snapshot().into();
        assert_eq!(summary.v2_timeline_checks, 1);
        assert_eq!(summary.v1_fallback_checks, 1);
        assert_eq!(summary.excluded_pre_listing, 1);
    }
}

