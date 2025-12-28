//! Shadow Mode Migration - Compare old TradingCalendar with new MarketSessionCalendar.
//!
//! This module provides tools to run both calendars in parallel and identify
//! any divergences before migrating.

use chrono::{DateTime, Datelike, NaiveDate, Utc, Weekday};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{DayClassification, Market, MarketSessionCalendar};

// ============================================================================
// Legacy TradingCalendar (reproduced for comparison)
// ============================================================================

/// Legacy trading calendar (only checks weekends, no holidays).
pub struct LegacyTradingCalendar;

impl LegacyTradingCalendar {
    /// Check if a date is a trading day (not weekend) - legacy behavior.
    pub fn is_trading_day(date: DateTime<Utc>) -> bool {
        let weekday = date.weekday();
        weekday != Weekday::Sat && weekday != Weekday::Sun
    }

    /// Check if a date is a trading day using NaiveDate.
    pub fn is_trading_day_naive(date: NaiveDate) -> bool {
        use chrono::Datelike;
        let weekday = date.weekday();
        weekday != Weekday::Sat && weekday != Weekday::Sun
    }
}

// ============================================================================
// Divergence Types
// ============================================================================

/// Type of divergence between old and new calendar.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DivergenceType {
    /// Old calendar said trading day, new calendar says closed (holiday)
    NewlyIdentifiedHoliday {
        date: NaiveDate,
        holiday_name: String,
    },
    /// Old calendar said trading day, new calendar says half-day
    NewlyIdentifiedHalfDay {
        date: NaiveDate,
        name: String,
    },
    /// Old calendar said trading day, new calendar says extraordinary closure
    NewlyIdentifiedClosure {
        date: NaiveDate,
        reason: String,
    },
    /// New calendar said trading day but old would have missed something
    /// (This shouldn't happen with the current implementation)
    LegacyMissedTradingDay { date: NaiveDate },
}

impl DivergenceType {
    /// Get severity of the divergence.
    pub fn severity(&self) -> DivergenceSeverity {
        match self {
            DivergenceType::NewlyIdentifiedHoliday { .. } => DivergenceSeverity::High,
            DivergenceType::NewlyIdentifiedHalfDay { .. } => DivergenceSeverity::Medium,
            DivergenceType::NewlyIdentifiedClosure { .. } => DivergenceSeverity::High,
            DivergenceType::LegacyMissedTradingDay { .. } => DivergenceSeverity::Low,
        }
    }

    /// Format for logging.
    pub fn to_log(&self) -> String {
        match self {
            DivergenceType::NewlyIdentifiedHoliday { date, holiday_name } => {
                format!("DIVERGENCE:HOLIDAY:{}:{}", date, holiday_name)
            }
            DivergenceType::NewlyIdentifiedHalfDay { date, name } => {
                format!("DIVERGENCE:HALFDAY:{}:{}", date, name)
            }
            DivergenceType::NewlyIdentifiedClosure { date, reason } => {
                format!("DIVERGENCE:CLOSURE:{}:{}", date, reason)
            }
            DivergenceType::LegacyMissedTradingDay { date } => {
                format!("DIVERGENCE:MISSED_TRADING:{}:legacy_said_closed", date)
            }
        }
    }
}

/// Severity level for divergences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DivergenceSeverity {
    Low,
    Medium,
    High,
}

// ============================================================================
// Shadow Mode Comparison
// ============================================================================

/// Result of shadow mode comparison.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShadowModeResult {
    pub market: Option<Market>,
    pub start_date: Option<NaiveDate>,
    pub end_date: Option<NaiveDate>,
    pub total_days_checked: u32,
    pub divergences: Vec<DivergenceType>,
    pub legacy_trading_days: u32,
    pub new_trading_days: u32,
    pub holidays_identified: u32,
    pub half_days_identified: u32,
}

impl ShadowModeResult {
    /// Check if there are any divergences.
    pub fn has_divergences(&self) -> bool {
        !self.divergences.is_empty()
    }

    /// Get count of high-severity divergences.
    pub fn high_severity_count(&self) -> usize {
        self.divergences
            .iter()
            .filter(|d| d.severity() == DivergenceSeverity::High)
            .count()
    }

    /// Generate summary report.
    pub fn summary(&self) -> String {
        format!(
            "Shadow Mode: {} days checked, {} divergences ({} high severity)\n\
             Legacy: {} trading days, New: {} trading days\n\
             Identified: {} holidays, {} half-days",
            self.total_days_checked,
            self.divergences.len(),
            self.high_severity_count(),
            self.legacy_trading_days,
            self.new_trading_days,
            self.holidays_identified,
            self.half_days_identified
        )
    }
}

/// Shadow mode comparator.
#[derive(Debug)]
pub struct ShadowModeComparator {
    new_calendar: MarketSessionCalendar,
}

impl ShadowModeComparator {
    /// Create a new shadow mode comparator.
    pub fn new() -> Self {
        Self {
            new_calendar: MarketSessionCalendar::new(),
        }
    }

    /// Compare calendars for a date range.
    pub fn compare(&self, market: Market, start: NaiveDate, end: NaiveDate) -> ShadowModeResult {
        let mut result = ShadowModeResult {
            market: Some(market),
            start_date: Some(start),
            end_date: Some(end),
            ..Default::default()
        };

        let mut current = start;

        while current <= end {
            result.total_days_checked += 1;

            let legacy_is_trading = LegacyTradingCalendar::is_trading_day_naive(current);
            let new_classification = self.new_calendar.classify_date(market, current);
            let new_is_trading = new_classification.is_trading_day();

            if legacy_is_trading {
                result.legacy_trading_days += 1;
            }
            if new_is_trading {
                result.new_trading_days += 1;
            }

            // Check for divergences
            match &new_classification {
                DayClassification::Holiday { name, .. } => {
                    result.holidays_identified += 1;
                    if legacy_is_trading {
                        // Old calendar would have said this is a trading day
                        result.divergences.push(DivergenceType::NewlyIdentifiedHoliday {
                            date: current,
                            holiday_name: name.clone(),
                        });
                    }
                }
                DayClassification::HalfDay { name, .. } => {
                    result.half_days_identified += 1;
                    if legacy_is_trading {
                        // Old calendar didn't know about half-days
                        result.divergences.push(DivergenceType::NewlyIdentifiedHalfDay {
                            date: current,
                            name: name.clone(),
                        });
                    }
                }
                DayClassification::ExtraordinaryClosure { reason, .. } => {
                    if legacy_is_trading {
                        result.divergences.push(DivergenceType::NewlyIdentifiedClosure {
                            date: current,
                            reason: reason.clone(),
                        });
                    }
                }
                DayClassification::TradingDay(_) | DayClassification::Weekend => {
                    // These should match between old and new
                    if !legacy_is_trading && new_is_trading {
                        result.divergences.push(DivergenceType::LegacyMissedTradingDay {
                            date: current,
                        });
                    }
                }
            }

            current += chrono::Duration::days(1);
        }

        result
    }

    /// Compare calendars for a single year.
    pub fn compare_year(&self, market: Market, year: i32) -> ShadowModeResult {
        let start = NaiveDate::from_ymd_opt(year, 1, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(year, 12, 31).unwrap();
        self.compare(market, start, end)
    }

    /// Run full comparison across multiple years for both markets.
    pub fn full_comparison(&self) -> HashMap<(Market, i32), ShadowModeResult> {
        let mut results = HashMap::new();

        for market in [Market::BR, Market::US] {
            for year in [2024, 2025] {
                let result = self.compare_year(market, year);
                results.insert((market, year), result);
            }
        }

        results
    }

    /// Print divergence report to stdout.
    pub fn print_report(&self, result: &ShadowModeResult) {
        println!("=== Shadow Mode Report ===");
        println!("{}", result.summary());
        println!();

        if result.has_divergences() {
            println!("Divergences:");
            for div in &result.divergences {
                println!("  {}", div.to_log());
            }
        } else {
            println!("No divergences found!");
        }
    }
}

impl Default for ShadowModeComparator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Migration Helpers
// ============================================================================

/// Check if we're ready to migrate (no high-severity divergences).
pub fn migration_safe(results: &[ShadowModeResult]) -> bool {
    results.iter().all(|r| r.high_severity_count() == 0)
}

/// Get total divergence count across all results.
pub fn total_divergences(results: &[ShadowModeResult]) -> usize {
    results.iter().map(|r| r.divergences.len()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    #[test]
    fn test_legacy_calendar_weekends() {
        // Saturday
        assert!(!LegacyTradingCalendar::is_trading_day_naive(date(2024, 12, 21)));
        // Sunday
        assert!(!LegacyTradingCalendar::is_trading_day_naive(date(2024, 12, 22)));
        // Monday
        assert!(LegacyTradingCalendar::is_trading_day_naive(date(2024, 12, 23)));
    }

    #[test]
    fn test_legacy_calendar_no_holidays() {
        // Christmas - legacy calendar doesn't know it's a holiday
        assert!(LegacyTradingCalendar::is_trading_day_naive(date(2024, 12, 25)));
    }

    #[test]
    fn test_shadow_mode_identifies_holidays() {
        let comparator = ShadowModeComparator::new();

        // Compare Christmas 2024
        let result = comparator.compare(Market::BR, date(2024, 12, 25), date(2024, 12, 25));

        assert_eq!(result.total_days_checked, 1);
        assert_eq!(result.legacy_trading_days, 1); // Legacy says it's a trading day
        assert_eq!(result.new_trading_days, 0); // New calendar knows it's a holiday
        assert!(result.has_divergences());
        assert_eq!(result.divergences.len(), 1);
        assert!(matches!(
            &result.divergences[0],
            DivergenceType::NewlyIdentifiedHoliday { .. }
        ));
    }

    #[test]
    fn test_shadow_mode_no_divergence_weekend() {
        let comparator = ShadowModeComparator::new();

        // Compare a weekend
        let result = comparator.compare(Market::BR, date(2024, 12, 21), date(2024, 12, 22));

        assert_eq!(result.total_days_checked, 2);
        assert_eq!(result.legacy_trading_days, 0);
        assert_eq!(result.new_trading_days, 0);
        assert!(!result.has_divergences()); // Both agree it's not a trading day
    }

    #[test]
    fn test_shadow_mode_identifies_half_days() {
        let comparator = ShadowModeComparator::new();

        // July 3, 2024 - US half-day before Independence Day
        let result = comparator.compare(Market::US, date(2024, 7, 3), date(2024, 7, 3));

        assert!(result.has_divergences());
        assert!(matches!(
            &result.divergences[0],
            DivergenceType::NewlyIdentifiedHalfDay { .. }
        ));
    }

    #[test]
    fn test_shadow_mode_year_comparison() {
        let comparator = ShadowModeComparator::new();

        let result = comparator.compare_year(Market::BR, 2024);

        assert_eq!(result.total_days_checked, 366); // 2024 is a leap year
        assert!(result.holidays_identified > 0);
        assert!(result.has_divergences()); // There are holidays
    }

    #[test]
    fn test_shadow_mode_summary() {
        let comparator = ShadowModeComparator::new();
        let result = comparator.compare_year(Market::BR, 2024);

        let summary = result.summary();
        assert!(summary.contains("Shadow Mode"));
        assert!(summary.contains("divergences"));
    }

    #[test]
    fn test_divergence_severity() {
        let holiday = DivergenceType::NewlyIdentifiedHoliday {
            date: date(2024, 12, 25),
            holiday_name: "Natal".to_string(),
        };
        assert_eq!(holiday.severity(), DivergenceSeverity::High);

        let half_day = DivergenceType::NewlyIdentifiedHalfDay {
            date: date(2024, 7, 3),
            name: "Independence Day Eve".to_string(),
        };
        assert_eq!(half_day.severity(), DivergenceSeverity::Medium);
    }

    #[test]
    fn test_divergence_log_format() {
        let holiday = DivergenceType::NewlyIdentifiedHoliday {
            date: date(2024, 12, 25),
            holiday_name: "Natal".to_string(),
        };
        let log = holiday.to_log();
        assert!(log.contains("DIVERGENCE:HOLIDAY"));
        assert!(log.contains("2024-12-25"));
        assert!(log.contains("Natal"));
    }

    #[test]
    fn test_full_comparison() {
        let comparator = ShadowModeComparator::new();
        let results = comparator.full_comparison();

        // Should have results for both markets and both years
        assert!(results.contains_key(&(Market::BR, 2024)));
        assert!(results.contains_key(&(Market::BR, 2025)));
        assert!(results.contains_key(&(Market::US, 2024)));
        assert!(results.contains_key(&(Market::US, 2025)));
    }

    #[test]
    fn test_migration_safety_check() {
        let safe_result = ShadowModeResult {
            divergences: vec![
                DivergenceType::LegacyMissedTradingDay { date: date(2024, 1, 1) }
            ],
            ..Default::default()
        };

        let unsafe_result = ShadowModeResult {
            divergences: vec![
                DivergenceType::NewlyIdentifiedHoliday {
                    date: date(2024, 12, 25),
                    holiday_name: "Natal".to_string(),
                }
            ],
            ..Default::default()
        };

        // Safe result has no high-severity divergences
        assert!(migration_safe(&[safe_result.clone()]));

        // Unsafe result has high-severity divergence
        assert!(!migration_safe(&[unsafe_result]));

        // Mixed is still unsafe
        assert!(!migration_safe(&[safe_result, ShadowModeResult {
            divergences: vec![
                DivergenceType::NewlyIdentifiedClosure {
                    date: date(2024, 1, 1),
                    reason: "Test".to_string(),
                }
            ],
            ..Default::default()
        }]));
    }
}

