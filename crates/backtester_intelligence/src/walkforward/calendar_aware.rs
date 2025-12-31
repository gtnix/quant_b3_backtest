//! Calendar-Aware Walk-Forward Splitter.
//!
//! Wraps the existing splitters to ensure all window boundaries
//! land on valid trading days, adjusting for holidays and weekends.

use chrono::{Datelike, NaiveDate, Weekday};

use crate::filters::Market;

use super::splitter::{NestedSplitter, RollingSplitter, TimeSplitter};
use super::types::{
    NestedWalkForwardConfig, NestedWindowSplit, WalkForwardConfig, WindowSplit, WindowSpec,
    WindowType,
};

// ============================================================================
// Embedded Holiday Data (minimal for calendar-aware splitting)
// ============================================================================

/// Simple trading day calendar for walk-forward adjustments.
///
/// This is a minimal calendar implementation for the walk-forward module
/// that can check if a date is a trading day without requiring the full
/// market_data dependency.
#[derive(Debug, Clone)]
pub struct TradingDayCalendar {
    market: Market,
    holidays_br: Vec<NaiveDate>,
    holidays_us: Vec<NaiveDate>,
}

impl TradingDayCalendar {
    /// Create a new calendar with embedded holiday data.
    pub fn new(market: Market) -> Self {
        Self {
            market,
            holidays_br: Self::b3_holidays(),
            holidays_us: Self::nyse_holidays(),
        }
    }

    /// Check if a date is a trading day.
    pub fn is_trading_day(&self, date: NaiveDate) -> bool {
        // Check weekend
        if matches!(date.weekday(), Weekday::Sat | Weekday::Sun) {
            return false;
        }

        // Check holidays
        let holidays = match self.market {
            Market::BR => &self.holidays_br,
            Market::US => &self.holidays_us,
        };

        !holidays.contains(&date)
    }

    /// Get the next trading day on or after the given date.
    pub fn next_trading_day(&self, date: NaiveDate) -> NaiveDate {
        let mut current = date;
        while !self.is_trading_day(current) {
            current += chrono::Duration::days(1);
            // Safety limit
            if (current - date).num_days() > 30 {
                break;
            }
        }
        current
    }

    /// Get the previous trading day on or before the given date.
    pub fn prev_trading_day(&self, date: NaiveDate) -> NaiveDate {
        let mut current = date;
        while !self.is_trading_day(current) {
            current -= chrono::Duration::days(1);
            // Safety limit
            if (date - current).num_days() > 30 {
                break;
            }
        }
        current
    }

    /// Count trading days between two dates (inclusive of start, exclusive of end).
    pub fn trading_days_between(&self, start: NaiveDate, end: NaiveDate) -> i64 {
        let mut count = 0;
        let mut current = start;
        while current < end {
            if self.is_trading_day(current) {
                count += 1;
            }
            current += chrono::Duration::days(1);
        }
        count
    }

    /// B3 holidays for 2024-2025.
    fn b3_holidays() -> Vec<NaiveDate> {
        fn date(y: i32, m: u32, d: u32) -> NaiveDate {
            NaiveDate::from_ymd_opt(y, m, d).unwrap()
        }

        vec![
            // 2024
            date(2024, 1, 1),
            date(2024, 2, 12),
            date(2024, 2, 13),
            date(2024, 3, 29),
            date(2024, 4, 21),
            date(2024, 5, 1),
            date(2024, 5, 30),
            date(2024, 11, 15),
            date(2024, 11, 20),
            date(2024, 12, 24),
            date(2024, 12, 25),
            date(2024, 12, 31),
            // 2025
            date(2025, 1, 1),
            date(2025, 3, 3),
            date(2025, 3, 4),
            date(2025, 4, 18),
            date(2025, 4, 21),
            date(2025, 5, 1),
            date(2025, 6, 19),
            date(2025, 11, 20),
            date(2025, 12, 24),
            date(2025, 12, 25),
            date(2025, 12, 31),
        ]
    }

    /// NYSE holidays for 2024-2025.
    fn nyse_holidays() -> Vec<NaiveDate> {
        fn date(y: i32, m: u32, d: u32) -> NaiveDate {
            NaiveDate::from_ymd_opt(y, m, d).unwrap()
        }

        vec![
            // 2024
            date(2024, 1, 1),
            date(2024, 1, 15),
            date(2024, 2, 19),
            date(2024, 3, 29),
            date(2024, 5, 27),
            date(2024, 6, 19),
            date(2024, 7, 4),
            date(2024, 9, 2),
            date(2024, 11, 28),
            date(2024, 12, 25),
            // 2025
            date(2025, 1, 1),
            date(2025, 1, 20),
            date(2025, 2, 17),
            date(2025, 4, 18),
            date(2025, 5, 26),
            date(2025, 6, 19),
            date(2025, 7, 4),
            date(2025, 9, 1),
            date(2025, 11, 27),
            date(2025, 12, 25),
        ]
    }
}

// ============================================================================
// Calendar-Aware Splitter
// ============================================================================

/// A calendar-aware wrapper for the rolling splitter.
///
/// Adjusts window boundaries to land on valid trading days.
#[derive(Debug, Clone)]
pub struct CalendarAwareRollingSplitter {
    inner: RollingSplitter,
    calendar: TradingDayCalendar,
}

impl CalendarAwareRollingSplitter {
    /// Create a new calendar-aware splitter.
    pub fn new(config: &WalkForwardConfig) -> Self {
        Self {
            inner: RollingSplitter::new(config),
            calendar: TradingDayCalendar::new(config.market),
        }
    }

    /// Adjust a window split so all dates land on trading days.
    fn adjust_split(&self, split: WindowSplit) -> WindowSplit {
        WindowSplit {
            train: self.adjust_window_spec(&split.train, true),
            test: self.adjust_window_spec(&split.test, true),
            purge_days: split.purge_days,
            embargo_days: split.embargo_days,
            index: split.index,
        }
    }

    /// Adjust a window spec to land on trading days.
    ///
    /// - Start dates are adjusted forward to the next trading day
    /// - End dates are adjusted backward to the previous trading day
    fn adjust_window_spec(&self, spec: &WindowSpec, _is_train: bool) -> WindowSpec {
        let adjusted_start = self.calendar.next_trading_day(spec.start_date);
        let adjusted_end = self.calendar.prev_trading_day(spec.end_date);

        WindowSpec {
            start_date: adjusted_start,
            end_date: adjusted_end,
            window_type: spec.window_type,
            index: spec.index,
        }
    }
}

impl TimeSplitter for CalendarAwareRollingSplitter {
    fn generate_splits(&self, start: NaiveDate, end: NaiveDate) -> Vec<WindowSplit> {
        // Adjust start to first trading day
        let adjusted_start = self.calendar.next_trading_day(start);
        // Adjust end to last trading day
        let adjusted_end = self.calendar.prev_trading_day(end);

        // Generate splits with the inner splitter
        let raw_splits = self.inner.generate_splits(adjusted_start, adjusted_end);

        // Adjust each split
        raw_splits
            .into_iter()
            .map(|s| self.adjust_split(s))
            .filter(|s| s.train.end_date > s.train.start_date && s.test.end_date > s.test.start_date)
            .collect()
    }
}

/// A calendar-aware wrapper for the nested splitter.
#[derive(Debug, Clone)]
pub struct CalendarAwareNestedSplitter {
    inner: NestedSplitter,
    calendar: TradingDayCalendar,
}

impl CalendarAwareNestedSplitter {
    /// Create a new calendar-aware nested splitter.
    pub fn new(config: &NestedWalkForwardConfig) -> Self {
        Self {
            inner: NestedSplitter::new(config),
            calendar: TradingDayCalendar::new(config.market),
        }
    }

    /// Generate calendar-adjusted nested splits.
    pub fn generate_nested_splits(&self, start: NaiveDate, end: NaiveDate) -> Vec<NestedWindowSplit> {
        // Adjust start to first trading day
        let adjusted_start = self.calendar.next_trading_day(start);
        // Adjust end to last trading day
        let adjusted_end = self.calendar.prev_trading_day(end);

        // Generate splits with the inner splitter
        let raw_splits = self.inner.generate_nested_splits(adjusted_start, adjusted_end);

        // Adjust each split
        raw_splits
            .into_iter()
            .map(|s| self.adjust_split(s))
            .filter(|s| s.is_valid())
            .collect()
    }

    /// Adjust a nested window split so all dates land on trading days.
    fn adjust_split(&self, split: NestedWindowSplit) -> NestedWindowSplit {
        NestedWindowSplit {
            train: self.adjust_window_spec(&split.train),
            val: self.adjust_window_spec(&split.val),
            test: self.adjust_window_spec(&split.test),
            purge_train_val: split.purge_train_val,
            purge_val_test: split.purge_val_test,
            embargo_days: split.embargo_days,
            index: split.index,
        }
    }

    fn adjust_window_spec(&self, spec: &WindowSpec) -> WindowSpec {
        let adjusted_start = self.calendar.next_trading_day(spec.start_date);
        let adjusted_end = self.calendar.prev_trading_day(spec.end_date);

        WindowSpec {
            start_date: adjusted_start,
            end_date: adjusted_end,
            window_type: spec.window_type,
            index: spec.index,
        }
    }
}

// ============================================================================
// WindowSpec extension for trading days
// ============================================================================

/// Extension trait for WindowSpec to add trading day calculations.
pub trait WindowSpecExt {
    /// Get the number of trading days in this window.
    fn trading_days(&self, calendar: &TradingDayCalendar) -> i64;
}

impl WindowSpecExt for WindowSpec {
    fn trading_days(&self, calendar: &TradingDayCalendar) -> i64 {
        calendar.trading_days_between(self.start_date, self.end_date)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    #[test]
    fn test_calendar_weekend_detection() {
        let calendar = TradingDayCalendar::new(Market::BR);

        // Saturday
        assert!(!calendar.is_trading_day(date(2024, 12, 21)));
        // Sunday
        assert!(!calendar.is_trading_day(date(2024, 12, 22)));
        // Monday
        assert!(calendar.is_trading_day(date(2024, 12, 23)));
    }

    #[test]
    fn test_calendar_holiday_detection() {
        let calendar = TradingDayCalendar::new(Market::BR);

        // Christmas
        assert!(!calendar.is_trading_day(date(2024, 12, 25)));
        // New Year
        assert!(!calendar.is_trading_day(date(2025, 1, 1)));
    }

    #[test]
    fn test_calendar_next_trading_day_from_weekend() {
        let calendar = TradingDayCalendar::new(Market::BR);

        // Saturday -> Monday
        let next = calendar.next_trading_day(date(2024, 12, 21));
        assert_eq!(next, date(2024, 12, 23));
    }

    #[test]
    fn test_calendar_next_trading_day_from_holiday() {
        let calendar = TradingDayCalendar::new(Market::BR);

        // Christmas (Wed) -> Thurs (but 26 is also closed in some years)
        // In 2024, Dec 25 is holiday, Dec 24 is holiday too
        let next = calendar.next_trading_day(date(2024, 12, 25));
        // Dec 26, 2024 is Thursday and should be open
        assert!(calendar.is_trading_day(next));
    }

    #[test]
    fn test_calendar_prev_trading_day() {
        let calendar = TradingDayCalendar::new(Market::BR);

        // Monday -> Friday
        let prev = calendar.prev_trading_day(date(2024, 12, 22));
        assert_eq!(prev, date(2024, 12, 20));
    }

    #[test]
    fn test_calendar_trading_days_between() {
        let calendar = TradingDayCalendar::new(Market::BR);

        // Week with no holidays: Mon-Fri = 5 trading days
        let count = calendar.trading_days_between(date(2024, 12, 16), date(2024, 12, 21));
        assert_eq!(count, 5);
    }

    #[test]
    fn test_calendar_aware_splitter_adjusts_dates() {
        let config = WalkForwardConfig {
            market: Market::BR,
            ..WalkForwardConfig::default()
        };
        let splitter = CalendarAwareRollingSplitter::new(&config);

        let start = date(2024, 1, 1); // New Year's Day (holiday)
        let end = date(2024, 12, 31);

        let splits = splitter.generate_splits(start, end);

        // All splits should have trading days as boundaries
        for split in &splits {
            let calendar = &splitter.calendar;
            assert!(
                calendar.is_trading_day(split.train.start_date),
                "Train start {} is not a trading day",
                split.train.start_date
            );
            assert!(
                calendar.is_trading_day(split.train.end_date),
                "Train end {} is not a trading day",
                split.train.end_date
            );
            assert!(
                calendar.is_trading_day(split.test.start_date),
                "Test start {} is not a trading day",
                split.test.start_date
            );
            assert!(
                calendar.is_trading_day(split.test.end_date),
                "Test end {} is not a trading day",
                split.test.end_date
            );
        }
    }

    #[test]
    fn test_calendar_aware_nested_splitter() {
        let config = NestedWalkForwardConfig {
            market: Market::US,
            ..NestedWalkForwardConfig::default()
        };
        let splitter = CalendarAwareNestedSplitter::new(&config);

        let start = date(2024, 1, 1);
        let end = date(2024, 12, 31);

        let splits = splitter.generate_nested_splits(start, end);

        for split in &splits {
            let calendar = &splitter.calendar;
            assert!(calendar.is_trading_day(split.train.start_date));
            assert!(calendar.is_trading_day(split.train.end_date));
            assert!(calendar.is_trading_day(split.val.start_date));
            assert!(calendar.is_trading_day(split.val.end_date));
            assert!(calendar.is_trading_day(split.test.start_date));
            assert!(calendar.is_trading_day(split.test.end_date));
        }
    }

    #[test]
    fn test_cross_market_holidays() {
        let calendar_br = TradingDayCalendar::new(Market::BR);
        let calendar_us = TradingDayCalendar::new(Market::US);

        // Brazilian Carnival (Mar 3, 2025) - B3 closed, NYSE open
        assert!(!calendar_br.is_trading_day(date(2025, 3, 3)));
        assert!(calendar_us.is_trading_day(date(2025, 3, 3)));

        // US MLK Day (Jan 20, 2025) - NYSE closed, B3 open
        assert!(calendar_br.is_trading_day(date(2025, 1, 20)));
        assert!(!calendar_us.is_trading_day(date(2025, 1, 20)));
    }

    #[test]
    fn test_window_spec_trading_days() {
        let calendar = TradingDayCalendar::new(Market::BR);
        let spec = WindowSpec::new(date(2024, 12, 16), date(2024, 12, 20), WindowType::Train, 0);

        // Mon-Fri = 4 trading days (end exclusive)
        let trading_days = spec.trading_days(&calendar);
        assert_eq!(trading_days, 4);
    }
}













