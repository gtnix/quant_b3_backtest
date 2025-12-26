//! Time-based window splitting with purge and embargo.
//!
//! Implements rolling splits for walk-forward validation.

use chrono::{Datelike, NaiveDate};

use super::types::{WalkForwardConfig, WindowSplit, WindowSpec, WindowType};

/// Trait for generating time-based splits.
pub trait TimeSplitter {
    /// Generate all train/test splits for a date range.
    fn generate_splits(&self, start: NaiveDate, end: NaiveDate) -> Vec<WindowSplit>;
}

/// Rolling window splitter with fixed train period.
#[derive(Debug, Clone)]
pub struct RollingSplitter {
    train_months: u32,
    test_months: u32,
    step_months: u32,
    purge_days: u32,
    embargo_days: u32,
}

impl RollingSplitter {
    pub fn new(config: &WalkForwardConfig) -> Self {
        Self {
            train_months: config.train_months,
            test_months: config.test_months,
            step_months: config.step_months,
            purge_days: config.purge_days,
            embargo_days: config.embargo_days,
        }
    }

    pub fn from_parts(
        train_months: u32,
        test_months: u32,
        step_months: u32,
        purge_days: u32,
        embargo_days: u32,
    ) -> Self {
        Self {
            train_months,
            test_months,
            step_months,
            purge_days,
            embargo_days,
        }
    }

    /// Add months to a date, handling month-end edge cases.
    fn add_months(date: NaiveDate, months: u32) -> NaiveDate {
        let months = months as i32;
        let mut new_year = date.year();
        let mut new_month = date.month() as i32 + months;

        while new_month > 12 {
            new_month -= 12;
            new_year += 1;
        }
        while new_month < 1 {
            new_month += 12;
            new_year -= 1;
        }

        // Get last day of target month
        let days_in_month = days_in_month(new_year, new_month as u32);
        let new_day = date.day().min(days_in_month);

        NaiveDate::from_ymd_opt(new_year, new_month as u32, new_day)
            .unwrap_or(date)
    }

    /// Subtract days from a date.
    fn sub_days(date: NaiveDate, days: u32) -> NaiveDate {
        date - chrono::Duration::days(days as i64)
    }

    /// Add days to a date.
    fn add_days(date: NaiveDate, days: u32) -> NaiveDate {
        date + chrono::Duration::days(days as i64)
    }
}

/// Get the number of days in a month.
fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0) {
                29
            } else {
                28
            }
        }
        _ => 30,
    }
}

impl TimeSplitter for RollingSplitter {
    fn generate_splits(&self, start: NaiveDate, end: NaiveDate) -> Vec<WindowSplit> {
        let mut splits = Vec::new();
        let mut idx = 0;
        let mut current_start = start;

        loop {
            // Train window: [current_start, current_start + train_months - purge_days]
            let train_end_raw = Self::add_months(current_start, self.train_months);
            let train_end = Self::sub_days(train_end_raw, self.purge_days);

            // Test window: [train_end + purge + embargo, train_end + purge + embargo + test_months]
            let gap_days = self.purge_days + self.embargo_days;
            let test_start = Self::add_days(train_end_raw, self.embargo_days as u32);
            let test_end = Self::add_months(test_start, self.test_months);

            // Stop if test window exceeds the end date
            if test_end > end {
                break;
            }

            let train = WindowSpec::new(current_start, train_end, WindowType::Train, idx);
            let test = WindowSpec::new(test_start, test_end, WindowType::Test, idx);

            splits.push(WindowSplit {
                train,
                test,
                purge_days: self.purge_days,
                embargo_days: self.embargo_days,
                index: idx,
            });

            // Move forward by step_months
            current_start = Self::add_months(current_start, self.step_months);
            idx += 1;

            // Safety limit
            if idx > 200 {
                break;
            }
        }

        splits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    #[test]
    fn test_rolling_splitter_basic() {
        let splitter = RollingSplitter::from_parts(6, 3, 3, 5, 5);
        let start = date(2020, 1, 1);
        let end = date(2021, 12, 31);

        let splits = splitter.generate_splits(start, end);

        // 2020-01-01 to 2021-12-31 = 24 months
        // With 6mo train + 3mo test + 3mo step:
        // Window 0: train 2020-01-01 to 2020-06-26, test 2020-07-06 to 2020-10-06
        // Window 1: train 2020-04-01 to 2020-09-26, test 2020-10-06 to 2021-01-06
        // ...
        assert!(!splits.is_empty());
        
        // All splits should be valid (no overlap)
        for split in &splits {
            assert!(split.is_valid(), "Split {} has overlap", split.index);
            assert!(split.gap_days() >= 5, "Split {} gap too small", split.index);
        }
    }

    #[test]
    fn test_no_train_test_overlap() {
        let splitter = RollingSplitter::from_parts(6, 3, 3, 5, 5);
        let start = date(2010, 1, 1);
        let end = date(2025, 1, 1);

        let splits = splitter.generate_splits(start, end);

        for split in &splits {
            assert!(
                split.train.end_date < split.test.start_date,
                "Window {}: train ends {} but test starts {}",
                split.index,
                split.train.end_date,
                split.test.start_date
            );
        }
    }

    #[test]
    fn test_purge_creates_gap() {
        let splitter = RollingSplitter::from_parts(6, 3, 3, 10, 5);
        let start = date(2020, 1, 1);
        let end = date(2021, 12, 31);

        let splits = splitter.generate_splits(start, end);

        for split in &splits {
            // Gap should be at least embargo days (purge removes from train)
            let gap = (split.test.start_date - split.train.end_date).num_days();
            assert!(
                gap >= 5,
                "Window {}: expected gap >= 5, got {}",
                split.index,
                gap
            );
        }
    }

    #[test]
    fn test_embargo_creates_gap() {
        let splitter = RollingSplitter::from_parts(6, 3, 3, 5, 10);
        let start = date(2020, 1, 1);
        let end = date(2021, 12, 31);

        let splits = splitter.generate_splits(start, end);

        for split in &splits {
            let gap = (split.test.start_date - split.train.end_date).num_days();
            // Gap = purge (5) absorbed in train_end + embargo (10)
            assert!(
                gap >= 10,
                "Window {}: expected gap >= 10, got {}",
                split.index,
                gap
            );
        }
    }

    #[test]
    fn test_20_years_coverage() {
        let splitter = RollingSplitter::from_parts(6, 3, 3, 5, 5);
        let start = date(2005, 1, 1);
        let end = date(2025, 1, 1);

        let splits = splitter.generate_splits(start, end);

        // 20 years / 0.25 year step = ~80 windows
        // But need 6+3 = 9 months minimum, so expect ~76-80 windows
        assert!(splits.len() >= 70, "Expected >= 70 windows, got {}", splits.len());
        assert!(splits.len() <= 85, "Expected <= 85 windows, got {}", splits.len());
    }

    #[test]
    fn test_consecutive_splits_overlap_train() {
        let splitter = RollingSplitter::from_parts(6, 3, 3, 5, 5);
        let start = date(2020, 1, 1);
        let end = date(2021, 12, 31);

        let splits = splitter.generate_splits(start, end);

        if splits.len() >= 2 {
            // With 3-month step and 6-month train, consecutive trains should overlap
            let s0 = &splits[0];
            let s1 = &splits[1];

            // Window 1 starts 3 months after window 0
            let step_days = (s1.train.start_date - s0.train.start_date).num_days();
            assert!(step_days >= 85 && step_days <= 95, "Expected ~90 days step, got {}", step_days);
        }
    }

    #[test]
    fn test_add_months_edge_cases() {
        // February to March
        let feb = date(2020, 2, 29);  // leap year
        let result = RollingSplitter::add_months(feb, 1);
        assert_eq!(result.month(), 3);
        assert!(result.day() <= 31);

        // January 31 + 1 month
        let jan31 = date(2020, 1, 31);
        let result = RollingSplitter::add_months(jan31, 1);
        assert_eq!(result.month(), 2);
        assert!(result.day() <= 29);  // leap year

        // Cross year boundary
        let dec = date(2020, 12, 15);
        let result = RollingSplitter::add_months(dec, 3);
        assert_eq!(result.year(), 2021);
        assert_eq!(result.month(), 3);
    }

    #[test]
    fn test_determinism() {
        let splitter = RollingSplitter::from_parts(6, 3, 3, 5, 5);
        let start = date(2015, 1, 1);
        let end = date(2020, 1, 1);

        let splits1 = splitter.generate_splits(start, end);
        let splits2 = splitter.generate_splits(start, end);

        assert_eq!(splits1.len(), splits2.len());
        for (s1, s2) in splits1.iter().zip(splits2.iter()) {
            assert_eq!(s1.train.start_date, s2.train.start_date);
            assert_eq!(s1.train.end_date, s2.train.end_date);
            assert_eq!(s1.test.start_date, s2.test.start_date);
            assert_eq!(s1.test.end_date, s2.test.end_date);
        }
    }
}

