//! Trading Day Classifier - Classifies dates for markets.
//!
//! Determines whether a given date is a trading day, weekend, holiday,
//! half-day, or extraordinary closure.

use chrono::{Datelike, NaiveDate, Weekday};

use super::{DayClassification, HolidayProvider, HolidayType, Market, MarketHoursProvider, SessionInfo};

/// Trading day classifier.
///
/// Classifies any (market, date) pair into a `DayClassification`.
#[derive(Debug, Clone)]
pub struct TradingDayClassifier {
    holiday_provider: HolidayProvider,
    hours_provider: MarketHoursProvider,
}

impl TradingDayClassifier {
    /// Create a new classifier with default holiday data.
    pub fn new() -> Self {
        let holiday_provider = HolidayProvider::new();
        let hours_provider = MarketHoursProvider::with_holidays(holiday_provider.clone());
        Self {
            holiday_provider,
            hours_provider,
        }
    }

    /// Create with a custom holiday provider.
    pub fn with_holidays(holiday_provider: HolidayProvider) -> Self {
        let hours_provider = MarketHoursProvider::with_holidays(holiday_provider.clone());
        Self {
            holiday_provider,
            hours_provider,
        }
    }

    /// Classify a date for a market.
    pub fn classify(&self, market: Market, date: NaiveDate) -> DayClassification {
        // 1. Check weekend first
        if self.is_weekend(date) {
            return DayClassification::Weekend;
        }

        // 2. Check for holidays
        if let Some(holiday) = self.holiday_provider.get_holiday(market, date) {
            match &holiday.holiday_type {
                HolidayType::National | HolidayType::MarketSpecific => {
                    return DayClassification::Holiday {
                        name: holiday.name.clone(),
                        official_source: holiday.source.clone(),
                    };
                }
                HolidayType::ExtraordinaryClosure => {
                    return DayClassification::ExtraordinaryClosure {
                        reason: holiday.name.clone(),
                        source: holiday.source.clone(),
                    };
                }
                HolidayType::HalfDay { close_time } => {
                    if let Some(session) = self.hours_provider.get_session(market, date) {
                        return DayClassification::HalfDay {
                            name: holiday.name.clone(),
                            close_time: *close_time,
                            session,
                        };
                    }
                }
                HolidayType::LateOpen { .. } => {
                    // Late open is still a trading day with modified hours
                    if let Some(session) = self.hours_provider.get_session(market, date) {
                        return DayClassification::TradingDay(session);
                    }
                }
            }
        }

        // 3. Regular trading day
        let session = self.hours_provider.regular_session(market);
        DayClassification::TradingDay(session)
    }

    /// Check if a date is a weekend.
    fn is_weekend(&self, date: NaiveDate) -> bool {
        matches!(date.weekday(), Weekday::Sat | Weekday::Sun)
    }

    /// Check if a date is a trading day (including half-days).
    pub fn is_trading_day(&self, market: Market, date: NaiveDate) -> bool {
        self.classify(market, date).is_trading_day()
    }

    /// Get the session info for a trading day (None if closed).
    pub fn get_session(&self, market: Market, date: NaiveDate) -> Option<SessionInfo> {
        self.hours_provider.get_session(market, date)
    }

    /// Get a reference to the holiday provider.
    pub fn holiday_provider(&self) -> &HolidayProvider {
        &self.holiday_provider
    }

    /// Get a reference to the hours provider.
    pub fn hours_provider(&self) -> &MarketHoursProvider {
        &self.hours_provider
    }
}

impl Default for TradingDayClassifier {
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

    #[test]
    fn test_classify_weekend() {
        let classifier = TradingDayClassifier::new();

        // Saturday
        let sat = classifier.classify(Market::BR, date(2024, 12, 21));
        assert!(matches!(sat, DayClassification::Weekend));

        // Sunday
        let sun = classifier.classify(Market::BR, date(2024, 12, 22));
        assert!(matches!(sun, DayClassification::Weekend));
    }

    #[test]
    fn test_classify_trading_day() {
        let classifier = TradingDayClassifier::new();

        // Monday Dec 23, 2024
        let classification = classifier.classify(Market::BR, date(2024, 12, 23));
        assert!(matches!(classification, DayClassification::TradingDay(_)));
        assert!(classification.is_trading_day());
    }

    #[test]
    fn test_classify_holiday_br() {
        let classifier = TradingDayClassifier::new();

        // Christmas 2024
        let christmas = classifier.classify(Market::BR, date(2024, 12, 25));
        assert!(matches!(christmas, DayClassification::Holiday { .. }));

        if let DayClassification::Holiday { name, .. } = christmas {
            assert_eq!(name, "Natal");
        }
    }

    #[test]
    fn test_classify_holiday_us() {
        let classifier = TradingDayClassifier::new();

        // Independence Day 2024
        let july4 = classifier.classify(Market::US, date(2024, 7, 4));
        assert!(matches!(july4, DayClassification::Holiday { .. }));

        if let DayClassification::Holiday { name, .. } = july4 {
            assert_eq!(name, "Independence Day");
        }
    }

    #[test]
    fn test_classify_half_day_us() {
        let classifier = TradingDayClassifier::new();

        // July 3, 2024 - half day
        let july3 = classifier.classify(Market::US, date(2024, 7, 3));
        assert!(matches!(july3, DayClassification::HalfDay { .. }));

        if let DayClassification::HalfDay { name, close_time, session } = july3 {
            assert!(name.contains("Independence Day"));
            assert_eq!(close_time, chrono::NaiveTime::from_hms_opt(13, 0, 0).unwrap());
            assert_eq!(session.regular.end, close_time);
        }
    }

    #[test]
    fn test_classify_late_open_br() {
        let classifier = TradingDayClassifier::new();

        // Ash Wednesday 2025 - late open
        let ash_wed = classifier.classify(Market::BR, date(2025, 3, 5));

        // Late open is still a trading day
        assert!(ash_wed.is_trading_day());

        if let DayClassification::TradingDay(session) = ash_wed {
            assert_eq!(session.regular.start, chrono::NaiveTime::from_hms_opt(13, 0, 0).unwrap());
        }
    }

    #[test]
    fn test_cross_market_classification() {
        let classifier = TradingDayClassifier::new();

        // Brazilian Carnival 2025 (Mar 3) - B3 closed, NYSE open
        let carnival_br = classifier.classify(Market::BR, date(2025, 3, 3));
        let carnival_us = classifier.classify(Market::US, date(2025, 3, 3));

        assert!(matches!(carnival_br, DayClassification::Holiday { .. }));
        assert!(matches!(carnival_us, DayClassification::TradingDay(_)));
    }

    #[test]
    fn test_is_trading_day() {
        let classifier = TradingDayClassifier::new();

        // Regular trading day
        assert!(classifier.is_trading_day(Market::BR, date(2024, 12, 23)));

        // Weekend
        assert!(!classifier.is_trading_day(Market::BR, date(2024, 12, 21)));

        // Holiday
        assert!(!classifier.is_trading_day(Market::BR, date(2024, 12, 25)));

        // Half-day is still a trading day
        assert!(classifier.is_trading_day(Market::US, date(2024, 7, 3)));
    }

    #[test]
    fn test_closure_reason() {
        let classifier = TradingDayClassifier::new();

        let weekend = classifier.classify(Market::BR, date(2024, 12, 21));
        assert_eq!(weekend.closure_reason(), Some("Weekend".to_string()));

        let holiday = classifier.classify(Market::BR, date(2024, 12, 25));
        assert_eq!(holiday.closure_reason(), Some("Holiday: Natal".to_string()));

        let trading_day = classifier.classify(Market::BR, date(2024, 12, 23));
        assert_eq!(trading_day.closure_reason(), None);
    }
}













