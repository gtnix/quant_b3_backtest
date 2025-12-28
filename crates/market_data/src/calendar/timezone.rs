//! Timezone Resolver - DST-aware timezone conversions.
//!
//! Uses chrono-tz for IANA timezone database support, handling historical
//! DST transitions correctly (including Brazil's DST abolition in 2019).

use chrono::{DateTime, Datelike, NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Utc};
use chrono_tz::Tz;

use super::Market;

/// Timezone resolver for market-specific time conversions.
///
/// Handles DST transitions correctly using the IANA timezone database.
#[derive(Debug, Clone)]
pub struct TimezoneResolver {
    tz_br: Tz,
    tz_us: Tz,
}

impl TimezoneResolver {
    /// Create a new timezone resolver.
    pub fn new() -> Self {
        Self {
            tz_br: chrono_tz::America::Sao_Paulo,
            tz_us: chrono_tz::America::New_York,
        }
    }

    /// Get the timezone for a market.
    pub fn timezone(&self, market: Market) -> Tz {
        match market {
            Market::BR => self.tz_br,
            Market::US => self.tz_us,
        }
    }

    /// Convert UTC datetime to local market time.
    pub fn to_local(&self, market: Market, utc: DateTime<Utc>) -> DateTime<Tz> {
        let tz = self.timezone(market);
        utc.with_timezone(&tz)
    }

    /// Convert local market time to UTC.
    ///
    /// Returns None if the local time is ambiguous (DST transition) or invalid.
    /// For ambiguous times, this returns the earlier interpretation.
    pub fn to_utc(&self, market: Market, date: NaiveDate, time: NaiveTime) -> Option<DateTime<Utc>> {
        let tz = self.timezone(market);
        let naive = NaiveDateTime::new(date, time);

        match tz.from_local_datetime(&naive) {
            chrono::LocalResult::Single(dt) => Some(dt.with_timezone(&Utc)),
            chrono::LocalResult::Ambiguous(earlier, _later) => {
                // During fall-back, prefer the earlier time
                Some(earlier.with_timezone(&Utc))
            }
            chrono::LocalResult::None => None, // Invalid time during spring-forward
        }
    }

    /// Get the UTC offset in hours for a market at a specific date.
    ///
    /// This accounts for DST. For example:
    /// - BR: -3 (no DST since 2019)
    /// - US: -5 (EST) or -4 (EDT)
    pub fn utc_offset_hours(&self, market: Market, date: NaiveDate) -> i32 {
        let tz = self.timezone(market);
        let noon = NaiveDateTime::new(date, NaiveTime::from_hms_opt(12, 0, 0).unwrap());

        match tz.from_local_datetime(&noon) {
            chrono::LocalResult::Single(dt) => {
                let utc_dt = dt.with_timezone(&Utc);
                let diff = dt.naive_local().signed_duration_since(utc_dt.naive_utc());
                (diff.num_seconds() / 3600) as i32
            }
            chrono::LocalResult::Ambiguous(dt, _) => {
                let utc_dt = dt.with_timezone(&Utc);
                let diff = dt.naive_local().signed_duration_since(utc_dt.naive_utc());
                (diff.num_seconds() / 3600) as i32
            }
            chrono::LocalResult::None => {
                // This shouldn't happen at noon, but fallback to standard offset
                match market {
                    Market::BR => -3,
                    Market::US => -5,
                }
            }
        }
    }

    /// Check if DST is in effect for a market on a given date.
    pub fn is_dst(&self, market: Market, date: NaiveDate) -> bool {
        match market {
            Market::BR => {
                // Brazil abolished DST in 2019, so no DST after that
                if date.year() >= 2019 {
                    return false;
                }
                // Historical DST check
                self.utc_offset_hours(market, date) == -2
            }
            Market::US => {
                // US has DST: EDT is UTC-4, EST is UTC-5
                self.utc_offset_hours(market, date) == -4
            }
        }
    }

    /// Get the start of trading session in UTC for a given date.
    ///
    /// This converts the local market open time to UTC, accounting for DST.
    pub fn session_start_utc(
        &self,
        market: Market,
        date: NaiveDate,
        local_start: NaiveTime,
    ) -> Option<DateTime<Utc>> {
        self.to_utc(market, date, local_start)
    }

    /// Get the end of trading session in UTC for a given date.
    pub fn session_end_utc(
        &self,
        market: Market,
        date: NaiveDate,
        local_end: NaiveTime,
    ) -> Option<DateTime<Utc>> {
        self.to_utc(market, date, local_end)
    }
}

impl Default for TimezoneResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Timelike;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    fn time(h: u32, m: u32, s: u32) -> NaiveTime {
        NaiveTime::from_hms_opt(h, m, s).unwrap()
    }

    #[test]
    fn test_brazil_no_dst_since_2019() {
        let resolver = TimezoneResolver::new();

        // After 2019, Brazil should always be UTC-3
        assert_eq!(resolver.utc_offset_hours(Market::BR, date(2024, 1, 15)), -3);
        assert_eq!(resolver.utc_offset_hours(Market::BR, date(2024, 7, 15)), -3);
        assert_eq!(resolver.utc_offset_hours(Market::BR, date(2024, 12, 15)), -3);

        // No DST after 2019
        assert!(!resolver.is_dst(Market::BR, date(2024, 1, 15)));
        assert!(!resolver.is_dst(Market::BR, date(2024, 7, 15)));
    }

    #[test]
    fn test_us_dst_transitions() {
        let resolver = TimezoneResolver::new();

        // Winter (EST = UTC-5)
        assert_eq!(resolver.utc_offset_hours(Market::US, date(2024, 1, 15)), -5);
        assert!(!resolver.is_dst(Market::US, date(2024, 1, 15)));

        // Summer (EDT = UTC-4)
        assert_eq!(resolver.utc_offset_hours(Market::US, date(2024, 7, 15)), -4);
        assert!(resolver.is_dst(Market::US, date(2024, 7, 15)));
    }

    #[test]
    fn test_to_local_br() {
        let resolver = TimezoneResolver::new();

        // 13:00 UTC should be 10:00 BRT (UTC-3)
        let utc = DateTime::parse_from_rfc3339("2024-12-20T13:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let local = resolver.to_local(Market::BR, utc);

        assert_eq!(local.hour(), 10);
        assert_eq!(local.minute(), 0);
    }

    #[test]
    fn test_to_local_us_winter() {
        let resolver = TimezoneResolver::new();

        // 14:30 UTC should be 09:30 EST (UTC-5) in winter
        let utc = DateTime::parse_from_rfc3339("2024-01-15T14:30:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let local = resolver.to_local(Market::US, utc);

        assert_eq!(local.hour(), 9);
        assert_eq!(local.minute(), 30);
    }

    #[test]
    fn test_to_local_us_summer() {
        let resolver = TimezoneResolver::new();

        // 13:30 UTC should be 09:30 EDT (UTC-4) in summer
        let utc = DateTime::parse_from_rfc3339("2024-07-15T13:30:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let local = resolver.to_local(Market::US, utc);

        assert_eq!(local.hour(), 9);
        assert_eq!(local.minute(), 30);
    }

    #[test]
    fn test_to_utc_br() {
        let resolver = TimezoneResolver::new();

        // 10:00 BRT should be 13:00 UTC
        let utc = resolver.to_utc(Market::BR, date(2024, 12, 20), time(10, 0, 0));
        assert!(utc.is_some());

        let utc = utc.unwrap();
        assert_eq!(utc.hour(), 13);
        assert_eq!(utc.minute(), 0);
    }

    #[test]
    fn test_to_utc_us_winter() {
        let resolver = TimezoneResolver::new();

        // 09:30 EST should be 14:30 UTC
        let utc = resolver.to_utc(Market::US, date(2024, 1, 15), time(9, 30, 0));
        assert!(utc.is_some());

        let utc = utc.unwrap();
        assert_eq!(utc.hour(), 14);
        assert_eq!(utc.minute(), 30);
    }

    #[test]
    fn test_to_utc_us_summer() {
        let resolver = TimezoneResolver::new();

        // 09:30 EDT should be 13:30 UTC
        let utc = resolver.to_utc(Market::US, date(2024, 7, 15), time(9, 30, 0));
        assert!(utc.is_some());

        let utc = utc.unwrap();
        assert_eq!(utc.hour(), 13);
        assert_eq!(utc.minute(), 30);
    }

    #[test]
    fn test_session_times_br() {
        let resolver = TimezoneResolver::new();

        // B3 opens at 10:00 BRT = 13:00 UTC
        let open = resolver.session_start_utc(Market::BR, date(2024, 12, 20), time(10, 0, 0));
        assert!(open.is_some());
        assert_eq!(open.unwrap().hour(), 13);

        // B3 closes at 17:55 BRT = 20:55 UTC
        let close = resolver.session_end_utc(Market::BR, date(2024, 12, 20), time(17, 55, 0));
        assert!(close.is_some());
        assert_eq!(close.unwrap().hour(), 20);
        assert_eq!(close.unwrap().minute(), 55);
    }

    #[test]
    fn test_historical_brazil_dst_2018() {
        let resolver = TimezoneResolver::new();

        // November 2018 - Brazil still had DST
        // DST started on first Sunday of November 2018 (Nov 4)
        // Before DST started: UTC-3
        assert_eq!(resolver.utc_offset_hours(Market::BR, date(2018, 11, 1)), -3);

        // After DST started (Nov 4, 2018): UTC-2
        assert_eq!(resolver.utc_offset_hours(Market::BR, date(2018, 11, 5)), -2);
    }
}

