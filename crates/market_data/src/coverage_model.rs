//! Coverage Model - defines what "complete" means for each (ticker, interval).
//!
//! This module provides the canonical rules for determining data completeness.

use chrono::{DateTime, Datelike, Duration, NaiveTime, Utc, Weekday};
use serde::{Deserialize, Serialize};

// ============================================================================
// Coverage Rules
// ============================================================================

/// Coverage expectations for different intervals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageRule {
    pub interval: String,
    pub max_range: String,
    pub expected_bars_per_day: u32,
    pub trading_start: NaiveTime,
    pub trading_end: NaiveTime,
    pub stale_threshold_days: i64,
    pub gap_tolerance: GapTolerance,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GapTolerance {
    /// Allow gaps on weekends and holidays
    WeekendHoliday,
    /// Allow gaps outside trading hours
    OutsideTradingHours,
    /// No gaps allowed
    None,
}

impl CoverageRule {
    /// Get the canonical rule for an interval.
    pub fn for_interval(interval: &str) -> Self {
        let trading_start = NaiveTime::from_hms_opt(10, 0, 0).unwrap();
        let trading_end = NaiveTime::from_hms_opt(17, 30, 0).unwrap();

        match interval {
            "1d" => CoverageRule {
                interval: "1d".to_string(),
                max_range: "max".to_string(),
                expected_bars_per_day: 1,
                trading_start,
                trading_end,
                stale_threshold_days: 3,
                gap_tolerance: GapTolerance::WeekendHoliday,
            },
            "60m" | "1h" => CoverageRule {
                interval: "60m".to_string(),
                max_range: "1y".to_string(),
                expected_bars_per_day: 7,
                trading_start,
                trading_end,
                stale_threshold_days: 2,
                gap_tolerance: GapTolerance::OutsideTradingHours,
            },
            "15m" => CoverageRule {
                interval: "15m".to_string(),
                max_range: "1mo".to_string(),
                expected_bars_per_day: 30,
                trading_start,
                trading_end,
                stale_threshold_days: 2,
                gap_tolerance: GapTolerance::OutsideTradingHours,
            },
            "5m" => CoverageRule {
                interval: "5m".to_string(),
                max_range: "1mo".to_string(),
                expected_bars_per_day: 90,
                trading_start,
                trading_end,
                stale_threshold_days: 1,
                gap_tolerance: GapTolerance::OutsideTradingHours,
            },
            "1m" => CoverageRule {
                interval: "1m".to_string(),
                max_range: "5d".to_string(),
                expected_bars_per_day: 450,
                trading_start,
                trading_end,
                stale_threshold_days: 1,
                gap_tolerance: GapTolerance::OutsideTradingHours,
            },
            _ => CoverageRule {
                interval: interval.to_string(),
                max_range: "1mo".to_string(),
                expected_bars_per_day: 1,
                trading_start,
                trading_end,
                stale_threshold_days: 7,
                gap_tolerance: GapTolerance::None,
            },
        }
    }

    /// Get all standard intervals in priority order.
    pub fn standard_intervals() -> Vec<&'static str> {
        vec!["1d", "60m", "15m", "5m", "1m"]
    }

    /// Get interval priority (lower = higher priority).
    pub fn priority(interval: &str) -> u32 {
        match interval {
            "1d" => 0,
            "60m" | "1h" => 1,
            "15m" => 2,
            "5m" => 3,
            "1m" => 4,
            _ => 99,
        }
    }
}

// ============================================================================
// Trading Calendar
// ============================================================================

/// B3 trading calendar helpers.
pub struct TradingCalendar;

impl TradingCalendar {
    /// Check if a date is a trading day (not weekend).
    pub fn is_trading_day(date: DateTime<Utc>) -> bool {
        let weekday = date.weekday();
        weekday != Weekday::Sat && weekday != Weekday::Sun
    }

    /// Count trading days between two dates.
    pub fn trading_days_between(start: DateTime<Utc>, end: DateTime<Utc>) -> i64 {
        let mut count = 0i64;
        let mut current = start;

        while current <= end {
            if Self::is_trading_day(current) {
                count += 1;
            }
            current = current + Duration::days(1);
        }

        count
    }

    /// Check if a timestamp is within trading hours.
    pub fn is_trading_hours(ts: DateTime<Utc>, rule: &CoverageRule) -> bool {
        // Convert to BRT (UTC-3) for B3
        let brt = ts - Duration::hours(3);
        let time = brt.time();

        time >= rule.trading_start && time <= rule.trading_end
    }

    /// Calculate expected bars for a date range.
    pub fn expected_bars(start: DateTime<Utc>, end: DateTime<Utc>, rule: &CoverageRule) -> i64 {
        let trading_days = Self::trading_days_between(start, end);
        trading_days * rule.expected_bars_per_day as i64
    }
}

// ============================================================================
// Coverage Evaluator
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageEvaluation {
    pub ticker: String,
    pub interval: String,
    pub actual_bars: i64,
    pub expected_bars: i64,
    pub coverage_ratio: f64,
    pub is_complete: bool,
    pub is_stale: bool,
    pub staleness_days: i64,
    pub unacceptable_gaps: i32,
}

pub struct CoverageEvaluator;

impl CoverageEvaluator {
    /// Evaluate coverage for a (ticker, interval) pair.
    pub fn evaluate(
        ticker: &str,
        interval: &str,
        first_ts: Option<DateTime<Utc>>,
        last_ts: Option<DateTime<Utc>>,
        actual_bars: i64,
        now: DateTime<Utc>,
    ) -> CoverageEvaluation {
        let rule = CoverageRule::for_interval(interval);

        let (expected_bars, coverage_ratio) = match (first_ts, last_ts) {
            (Some(first), Some(_last)) => {
                let expected = TradingCalendar::expected_bars(first, now, &rule);
                let ratio = if expected > 0 {
                    (actual_bars as f64 / expected as f64).min(1.0)
                } else {
                    0.0
                };
                (expected, ratio)
            }
            _ => (0, 0.0),
        };

        let staleness_days = last_ts.map(|last| (now - last).num_days()).unwrap_or(999);

        let is_stale = staleness_days > rule.stale_threshold_days;
        let is_complete = coverage_ratio >= 0.90 && !is_stale;

        CoverageEvaluation {
            ticker: ticker.to_string(),
            interval: interval.to_string(),
            actual_bars,
            expected_bars,
            coverage_ratio,
            is_complete,
            is_stale,
            staleness_days,
            unacceptable_gaps: 0, // Would require detailed gap analysis
        }
    }

    /// Check if a gap is acceptable based on the rule.
    pub fn is_gap_acceptable(
        gap_start: DateTime<Utc>,
        gap_end: DateTime<Utc>,
        rule: &CoverageRule,
    ) -> bool {
        match rule.gap_tolerance {
            GapTolerance::WeekendHoliday => {
                // Gap is OK if it spans only weekends
                let mut current = gap_start;
                while current <= gap_end {
                    if TradingCalendar::is_trading_day(current) {
                        return false;
                    }
                    current = current + Duration::days(1);
                }
                true
            }
            GapTolerance::OutsideTradingHours => {
                // Gap is OK if outside trading hours
                !TradingCalendar::is_trading_hours(gap_start, rule)
                    || !TradingCalendar::is_trading_hours(gap_end, rule)
            }
            GapTolerance::None => false,
        }
    }
}

// ============================================================================
// Range Parser
// ============================================================================

/// Parse provider range strings into durations.
pub fn parse_range_to_days(range: &str) -> i64 {
    match range {
        "1d" => 1,
        "5d" => 5,
        "1mo" => 30,
        "3mo" => 90,
        "6mo" => 180,
        "1y" => 365,
        "2y" => 730,
        "5y" => 1825,
        "10y" => 3650,
        "ytd" => {
            let now = Utc::now();
            (now - Utc::now().with_ordinal(1).unwrap_or(now)).num_days()
        }
        "max" => 3650 * 3, // ~30 years max
        _ => 30,           // Default 1 month
    }
}

/// Get the range string for backfill based on interval.
pub fn get_backfill_range(interval: &str) -> &'static str {
    match interval {
        "1d" => "max",
        "60m" | "1h" => "1y",
        "15m" => "1mo",
        "5m" => "1mo",
        "1m" => "5d",
        _ => "1mo",
    }
}



























