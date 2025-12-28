//! Market Hours Provider - Trading session times by market.
//!
//! Provides official trading hours for B3 and NYSE markets,
//! including half-day and late-open sessions.

use chrono::{Datelike, NaiveDate, NaiveTime, Weekday};

use super::{Holiday, HolidayProvider, HolidayType, Market, SessionInfo, TimeRange};

/// Provider for market trading hours.
#[derive(Debug, Clone)]
pub struct MarketHoursProvider {
    holiday_provider: HolidayProvider,
}

impl MarketHoursProvider {
    /// Create a new market hours provider.
    pub fn new() -> Self {
        Self {
            holiday_provider: HolidayProvider::new(),
        }
    }

    /// Create with a custom holiday provider.
    pub fn with_holidays(holiday_provider: HolidayProvider) -> Self {
        Self { holiday_provider }
    }

    /// Get the session info for a specific date.
    ///
    /// Returns None if the market is closed (weekend or holiday).
    pub fn get_session(&self, market: Market, date: NaiveDate) -> Option<SessionInfo> {
        // Check for special holiday sessions first
        if let Some(holiday) = self.holiday_provider.get_holiday(market, date) {
            return self.session_for_holiday(market, holiday);
        }

        // Check weekend
        if self.is_weekend(date) {
            return None;
        }

        // Regular session
        Some(self.regular_session(market))
    }

    /// Get the regular (non-holiday) session for a market.
    pub fn regular_session(&self, market: Market) -> SessionInfo {
        match market {
            Market::BR => self.b3_regular_session(),
            Market::US => self.nyse_regular_session(),
        }
    }

    /// Check if a date is a weekend.
    fn is_weekend(&self, date: NaiveDate) -> bool {
        matches!(date.weekday(), Weekday::Sat | Weekday::Sun)
    }

    /// Get session for a holiday (may be half-day or late-open).
    fn session_for_holiday(&self, market: Market, holiday: &Holiday) -> Option<SessionInfo> {
        match &holiday.holiday_type {
            HolidayType::National | HolidayType::MarketSpecific | HolidayType::ExtraordinaryClosure => {
                // Full closure
                None
            }
            HolidayType::HalfDay { close_time } => {
                // Early close
                let regular = self.regular_session(market);
                Some(SessionInfo {
                    pre_market: regular.pre_market,
                    regular: TimeRange::new(regular.regular.start, *close_time),
                    closing_auction: None, // Usually no auction on half-days
                    after_hours: None,
                })
            }
            HolidayType::LateOpen { open_time } => {
                // Late open (e.g., Ash Wednesday in Brazil)
                let regular = self.regular_session(market);
                Some(SessionInfo {
                    pre_market: None, // No pre-market on late-open days
                    regular: TimeRange::new(*open_time, regular.regular.end),
                    closing_auction: regular.closing_auction,
                    after_hours: regular.after_hours,
                })
            }
        }
    }

    // ========================================================================
    // B3 Sessions
    // ========================================================================

    /// B3 regular trading session.
    ///
    /// Source: https://www.b3.com.br/pt_br/solucoes/plataformas/puma-trading-system/
    ///
    /// - Pre-market auction: 09:45-10:00
    /// - Regular session: 10:00-17:55
    /// - Closing auction: 17:55-18:00
    /// - After-market: 18:25-18:45
    fn b3_regular_session(&self) -> SessionInfo {
        SessionInfo {
            pre_market: Some(TimeRange::new(
                NaiveTime::from_hms_opt(9, 45, 0).unwrap(),
                NaiveTime::from_hms_opt(10, 0, 0).unwrap(),
            )),
            regular: TimeRange::new(
                NaiveTime::from_hms_opt(10, 0, 0).unwrap(),
                NaiveTime::from_hms_opt(17, 55, 0).unwrap(),
            ),
            closing_auction: Some(TimeRange::new(
                NaiveTime::from_hms_opt(17, 55, 0).unwrap(),
                NaiveTime::from_hms_opt(18, 0, 0).unwrap(),
            )),
            after_hours: Some(TimeRange::new(
                NaiveTime::from_hms_opt(18, 25, 0).unwrap(),
                NaiveTime::from_hms_opt(18, 45, 0).unwrap(),
            )),
        }
    }

    // ========================================================================
    // NYSE Sessions
    // ========================================================================

    /// NYSE regular trading session.
    ///
    /// Source: https://www.nyse.com/markets/hours-calendars
    ///
    /// - Pre-market: 04:00-09:30
    /// - Regular session: 09:30-16:00
    /// - After-hours: 16:00-20:00
    fn nyse_regular_session(&self) -> SessionInfo {
        SessionInfo {
            pre_market: Some(TimeRange::new(
                NaiveTime::from_hms_opt(4, 0, 0).unwrap(),
                NaiveTime::from_hms_opt(9, 30, 0).unwrap(),
            )),
            regular: TimeRange::new(
                NaiveTime::from_hms_opt(9, 30, 0).unwrap(),
                NaiveTime::from_hms_opt(16, 0, 0).unwrap(),
            ),
            closing_auction: None, // NYSE doesn't have a separate closing auction period
            after_hours: Some(TimeRange::new(
                NaiveTime::from_hms_opt(16, 0, 0).unwrap(),
                NaiveTime::from_hms_opt(20, 0, 0).unwrap(),
            )),
        }
    }
}

impl Default for MarketHoursProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    fn time(h: u32, m: u32) -> NaiveTime {
        NaiveTime::from_hms_opt(h, m, 0).unwrap()
    }

    #[test]
    fn test_b3_regular_session() {
        let provider = MarketHoursProvider::new();
        let session = provider.regular_session(Market::BR);

        assert_eq!(session.regular.start, time(10, 0));
        assert_eq!(session.regular.end, time(17, 55));
        assert!(session.pre_market.is_some());
        assert!(session.closing_auction.is_some());
        assert!(session.after_hours.is_some());
    }

    #[test]
    fn test_nyse_regular_session() {
        let provider = MarketHoursProvider::new();
        let session = provider.regular_session(Market::US);

        assert_eq!(session.regular.start, time(9, 30));
        assert_eq!(session.regular.end, time(16, 0));
        assert!(session.pre_market.is_some());
        assert!(session.closing_auction.is_none());
        assert!(session.after_hours.is_some());
    }

    #[test]
    fn test_regular_trading_day() {
        let provider = MarketHoursProvider::new();

        // A regular Monday
        let session = provider.get_session(Market::BR, date(2024, 12, 23));
        assert!(session.is_some());

        let session = session.unwrap();
        assert_eq!(session.regular.start, time(10, 0));
        assert_eq!(session.regular.end, time(17, 55));
    }

    #[test]
    fn test_weekend_no_session() {
        let provider = MarketHoursProvider::new();

        // Saturday
        assert!(provider.get_session(Market::BR, date(2024, 12, 21)).is_none());
        // Sunday
        assert!(provider.get_session(Market::BR, date(2024, 12, 22)).is_none());
    }

    #[test]
    fn test_holiday_no_session() {
        let provider = MarketHoursProvider::new();

        // Christmas 2024
        assert!(provider.get_session(Market::BR, date(2024, 12, 25)).is_none());
        assert!(provider.get_session(Market::US, date(2024, 12, 25)).is_none());
    }

    #[test]
    fn test_nyse_half_day() {
        let provider = MarketHoursProvider::new();

        // July 3, 2024 - half-day before Independence Day
        let session = provider.get_session(Market::US, date(2024, 7, 3));
        assert!(session.is_some());

        let session = session.unwrap();
        assert_eq!(session.regular.start, time(9, 30));
        assert_eq!(session.regular.end, time(13, 0)); // Early close
        assert!(session.after_hours.is_none()); // No after-hours on half-days
    }

    #[test]
    fn test_b3_late_open_ash_wednesday() {
        let provider = MarketHoursProvider::new();

        // Ash Wednesday 2025 - late open at 13:00
        let session = provider.get_session(Market::BR, date(2025, 3, 5));
        assert!(session.is_some());

        let session = session.unwrap();
        assert_eq!(session.regular.start, time(13, 0)); // Late open
        assert_eq!(session.regular.end, time(17, 55));
        assert!(session.pre_market.is_none()); // No pre-market on late-open days
    }

    #[test]
    fn test_session_duration() {
        let provider = MarketHoursProvider::new();

        // B3 regular session: 10:00-17:55 = 7h 55m = 475 minutes
        let b3_session = provider.regular_session(Market::BR);
        assert_eq!(b3_session.regular.duration_minutes(), 475);

        // NYSE regular session: 09:30-16:00 = 6h 30m = 390 minutes
        let nyse_session = provider.regular_session(Market::US);
        assert_eq!(nyse_session.regular.duration_minutes(), 390);
    }

    #[test]
    fn test_earliest_and_latest_times() {
        let provider = MarketHoursProvider::new();

        let b3_session = provider.regular_session(Market::BR);
        assert_eq!(b3_session.earliest_start(), time(9, 45)); // Pre-market
        assert_eq!(b3_session.latest_end(), time(18, 45)); // After-hours

        let nyse_session = provider.regular_session(Market::US);
        assert_eq!(nyse_session.earliest_start(), time(4, 0)); // Pre-market
        assert_eq!(nyse_session.latest_end(), time(20, 0)); // After-hours
    }
}

