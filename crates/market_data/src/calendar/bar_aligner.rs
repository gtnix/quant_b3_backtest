//! Bar Aligner - Validates and aligns bar timestamps per market rules.
//!
//! Ensures bar timestamps follow the correct convention for each market
//! and data type (EOD vs intraday).

use chrono::{DateTime, NaiveDate, NaiveTime, Utc};
use chrono::Timelike;

use super::{Market, MarketSessionCalendar, SessionInfo, TimezoneResolver};

/// Bar alignment convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarConvention {
    /// End-of-day bars: timestamp is midnight UTC of the trading date
    EndOfDay,
    /// Intraday bars: timestamp is the start of the bar interval
    IntradayStart,
    /// Intraday bars: timestamp is the end of the bar interval
    IntradayEnd,
}

/// Result of bar alignment validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlignmentResult {
    /// Bar is correctly aligned
    Valid,
    /// Bar is outside trading session
    OutsideSession {
        bar_time: NaiveTime,
        session_start: NaiveTime,
        session_end: NaiveTime,
    },
    /// Bar timestamp doesn't align with expected interval grid
    MisalignedInterval {
        bar_time: NaiveTime,
        expected_alignment: NaiveTime,
    },
    /// Bar is on a non-trading day
    NonTradingDay { reason: String },
    /// EOD bar has incorrect timestamp
    IncorrectEodTimestamp {
        expected: DateTime<Utc>,
        actual: DateTime<Utc>,
    },
}

impl AlignmentResult {
    /// Check if the alignment is valid.
    pub fn is_valid(&self) -> bool {
        matches!(self, AlignmentResult::Valid)
    }
}

/// Bar aligner for validating and correcting bar timestamps.
#[derive(Debug)]
pub struct BarAligner {
    calendar: MarketSessionCalendar,
    timezone_resolver: TimezoneResolver,
}

impl BarAligner {
    /// Create a new bar aligner.
    pub fn new() -> Self {
        Self {
            calendar: MarketSessionCalendar::new(),
            timezone_resolver: TimezoneResolver::new(),
        }
    }

    /// Create with a custom calendar.
    pub fn with_calendar(calendar: MarketSessionCalendar) -> Self {
        Self {
            timezone_resolver: TimezoneResolver::new(),
            calendar,
        }
    }

    // ========================================================================
    // EOD Bar Alignment
    // ========================================================================

    /// Get the expected timestamp for an EOD bar.
    ///
    /// Convention: EOD bars use midnight UTC of the trading date.
    pub fn expected_eod_timestamp(&self, date: NaiveDate) -> DateTime<Utc> {
        date.and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
    }

    /// Validate an EOD bar timestamp.
    pub fn validate_eod(&self, market: Market, date: NaiveDate, timestamp: DateTime<Utc>) -> AlignmentResult {
        // Check if it's a trading day
        if !self.calendar.is_trading_day(market, date) {
            let classification = self.calendar.classify_date(market, date);
            return AlignmentResult::NonTradingDay {
                reason: classification.closure_reason().unwrap_or_default(),
            };
        }

        // Check timestamp matches expected
        let expected = self.expected_eod_timestamp(date);
        if timestamp != expected {
            return AlignmentResult::IncorrectEodTimestamp {
                expected,
                actual: timestamp,
            };
        }

        AlignmentResult::Valid
    }

    /// Normalize an EOD bar timestamp to the expected format.
    ///
    /// Takes any timestamp for a trading day and returns the normalized
    /// EOD timestamp (midnight UTC of that date).
    pub fn normalize_eod(&self, timestamp: DateTime<Utc>) -> DateTime<Utc> {
        let date = timestamp.date_naive();
        self.expected_eod_timestamp(date)
    }

    // ========================================================================
    // Intraday Bar Alignment
    // ========================================================================

    /// Validate an intraday bar timestamp.
    ///
    /// Checks if the bar falls within the trading session.
    pub fn validate_intraday(
        &self,
        market: Market,
        timestamp: DateTime<Utc>,
        interval_minutes: u32,
    ) -> AlignmentResult {
        let local = self.timezone_resolver.to_local(market, timestamp);
        let date = local.date_naive();
        let time = local.time();

        // Check if it's a trading day
        if !self.calendar.is_trading_day(market, date) {
            let classification = self.calendar.classify_date(market, date);
            return AlignmentResult::NonTradingDay {
                reason: classification.closure_reason().unwrap_or_default(),
            };
        }

        // Get session info
        let session = match self.calendar.get_session(market, date) {
            Some(s) => s,
            None => {
                return AlignmentResult::NonTradingDay {
                    reason: "No session info".to_string(),
                }
            }
        };

        // Check if within session
        if !self.is_within_session(&session, time) {
            return AlignmentResult::OutsideSession {
                bar_time: time,
                session_start: session.regular.start,
                session_end: session.regular.end,
            };
        }

        // Check alignment to interval grid
        let expected = self.aligned_bar_time(session.regular.start, time, interval_minutes);
        if time != expected {
            return AlignmentResult::MisalignedInterval {
                bar_time: time,
                expected_alignment: expected,
            };
        }

        AlignmentResult::Valid
    }

    /// Check if a time is within any trading period.
    fn is_within_session(&self, session: &SessionInfo, time: NaiveTime) -> bool {
        // Check regular session with some tolerance
        let start = session.regular.start;
        let end = session.regular.end;

        // Allow bars up to the session end (closing bar)
        time >= start && time <= end
    }

    /// Get the aligned bar time for a given time and interval.
    ///
    /// Returns the start of the bar that contains this time.
    fn aligned_bar_time(&self, session_start: NaiveTime, time: NaiveTime, interval_minutes: u32) -> NaiveTime {
        let session_start_mins = session_start.num_seconds_from_midnight() / 60;
        let time_mins = time.num_seconds_from_midnight() / 60;

        // Calculate minutes since session start
        let since_start = time_mins.saturating_sub(session_start_mins);

        // Round down to interval boundary
        let aligned_mins = (since_start / interval_minutes) * interval_minutes;

        // Add back to session start
        let total_mins = session_start_mins + aligned_mins;

        NaiveTime::from_num_seconds_from_midnight_opt(total_mins * 60, 0)
            .unwrap_or(time)
    }

    /// Get the expected intraday bar timestamp in UTC.
    ///
    /// Given a trading date and bar index, returns the expected UTC timestamp
    /// using the start-of-bar convention.
    pub fn expected_intraday_timestamp(
        &self,
        market: Market,
        date: NaiveDate,
        bar_index: u32,
        interval_minutes: u32,
    ) -> Option<DateTime<Utc>> {
        let session = self.calendar.get_session(market, date)?;

        // Calculate local time for this bar
        let session_start_mins = session.regular.start.num_seconds_from_midnight() / 60;
        let bar_mins = session_start_mins + (bar_index * interval_minutes);

        let local_time = NaiveTime::from_num_seconds_from_midnight_opt(bar_mins * 60, 0)?;

        // Convert to UTC
        self.timezone_resolver.to_utc(market, date, local_time)
    }

    /// Get the number of expected bars for a trading day.
    pub fn expected_bar_count(
        &self,
        market: Market,
        date: NaiveDate,
        interval_minutes: u32,
    ) -> u32 {
        let session = match self.calendar.get_session(market, date) {
            Some(s) => s,
            None => return 0,
        };

        let duration_mins = session.regular.duration_minutes() as u32;
        duration_mins / interval_minutes
    }
}

impl Default for BarAligner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Datelike, Timelike};

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    fn time(h: u32, m: u32) -> NaiveTime {
        NaiveTime::from_hms_opt(h, m, 0).unwrap()
    }

    #[test]
    fn test_eod_expected_timestamp() {
        let aligner = BarAligner::new();

        let expected = aligner.expected_eod_timestamp(date(2024, 12, 20));

        assert_eq!(expected.year(), 2024);
        assert_eq!(expected.month(), 12);
        assert_eq!(expected.day(), 20);
        assert_eq!(expected.hour(), 0);
        assert_eq!(expected.minute(), 0);
    }

    #[test]
    fn test_eod_validation_valid() {
        let aligner = BarAligner::new();

        let timestamp = date(2024, 12, 23).and_hms_opt(0, 0, 0).unwrap().and_utc();
        let result = aligner.validate_eod(Market::BR, date(2024, 12, 23), timestamp);

        assert!(result.is_valid());
    }

    #[test]
    fn test_eod_validation_wrong_timestamp() {
        let aligner = BarAligner::new();

        // Using close time instead of midnight
        let timestamp = date(2024, 12, 23).and_hms_opt(20, 55, 0).unwrap().and_utc();
        let result = aligner.validate_eod(Market::BR, date(2024, 12, 23), timestamp);

        assert!(matches!(result, AlignmentResult::IncorrectEodTimestamp { .. }));
    }

    #[test]
    fn test_eod_validation_non_trading_day() {
        let aligner = BarAligner::new();

        let timestamp = date(2024, 12, 25).and_hms_opt(0, 0, 0).unwrap().and_utc();
        let result = aligner.validate_eod(Market::BR, date(2024, 12, 25), timestamp);

        assert!(matches!(result, AlignmentResult::NonTradingDay { .. }));
    }

    #[test]
    fn test_normalize_eod() {
        let aligner = BarAligner::new();

        // Any time should be normalized to midnight
        let timestamp = date(2024, 12, 23).and_hms_opt(15, 30, 0).unwrap().and_utc();
        let normalized = aligner.normalize_eod(timestamp);

        assert_eq!(normalized.hour(), 0);
        assert_eq!(normalized.minute(), 0);
        assert_eq!(normalized.day(), 23);
    }

    #[test]
    fn test_intraday_validation_valid() {
        let aligner = BarAligner::new();

        // 10:00 BRT = 13:00 UTC - first bar of B3 session
        let timestamp = date(2024, 12, 23).and_hms_opt(13, 0, 0).unwrap().and_utc();
        let result = aligner.validate_intraday(Market::BR, timestamp, 60);

        assert!(result.is_valid());
    }

    #[test]
    fn test_intraday_validation_outside_session() {
        let aligner = BarAligner::new();

        // 22:00 BRT = 01:00 UTC next day - outside session
        let timestamp = date(2024, 12, 24).and_hms_opt(1, 0, 0).unwrap().and_utc();
        let result = aligner.validate_intraday(Market::BR, timestamp, 60);

        // This should be outside session since 22:00 BRT is after close
        assert!(matches!(result, AlignmentResult::OutsideSession { .. } | AlignmentResult::NonTradingDay { .. }));
    }

    #[test]
    fn test_expected_bar_count_b3() {
        let aligner = BarAligner::new();

        // B3 regular session: 10:00-17:55 = 475 minutes
        // 1-hour bars: 475 / 60 = 7 bars (truncated)
        let count_60m = aligner.expected_bar_count(Market::BR, date(2024, 12, 23), 60);
        assert_eq!(count_60m, 7);

        // 15-minute bars: 475 / 15 = 31 bars
        let count_15m = aligner.expected_bar_count(Market::BR, date(2024, 12, 23), 15);
        assert_eq!(count_15m, 31);
    }

    #[test]
    fn test_expected_bar_count_nyse() {
        let aligner = BarAligner::new();

        // NYSE regular session: 09:30-16:00 = 390 minutes
        // 1-hour bars: 390 / 60 = 6 bars
        let count_60m = aligner.expected_bar_count(Market::US, date(2024, 12, 23), 60);
        assert_eq!(count_60m, 6);
    }

    #[test]
    fn test_expected_bar_count_non_trading_day() {
        let aligner = BarAligner::new();

        // Weekend - no bars
        let count = aligner.expected_bar_count(Market::BR, date(2024, 12, 21), 60);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_aligned_bar_time() {
        let aligner = BarAligner::new();
        let session_start = time(10, 0);

        // 10:00 should stay 10:00 (first bar)
        assert_eq!(aligner.aligned_bar_time(session_start, time(10, 0), 60), time(10, 0));

        // 10:30 should align to 10:00 for 60m bars
        assert_eq!(aligner.aligned_bar_time(session_start, time(10, 30), 60), time(10, 0));

        // 11:15 should align to 11:00 for 60m bars
        assert_eq!(aligner.aligned_bar_time(session_start, time(11, 15), 60), time(11, 0));

        // 10:30 should stay 10:30 for 15m bars
        assert_eq!(aligner.aligned_bar_time(session_start, time(10, 30), 15), time(10, 30));

        // 10:37 should align to 10:30 for 15m bars
        assert_eq!(aligner.aligned_bar_time(session_start, time(10, 37), 15), time(10, 30));
    }

    #[test]
    fn test_expected_intraday_timestamp_b3() {
        let aligner = BarAligner::new();

        // First 1-hour bar at 10:00 BRT = 13:00 UTC
        let ts = aligner.expected_intraday_timestamp(Market::BR, date(2024, 12, 23), 0, 60);
        assert!(ts.is_some());
        assert_eq!(ts.unwrap().hour(), 13);
        assert_eq!(ts.unwrap().minute(), 0);

        // Second 1-hour bar at 11:00 BRT = 14:00 UTC
        let ts = aligner.expected_intraday_timestamp(Market::BR, date(2024, 12, 23), 1, 60);
        assert!(ts.is_some());
        assert_eq!(ts.unwrap().hour(), 14);
    }
}

