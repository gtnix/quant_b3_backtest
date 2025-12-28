//! Pre-computed validation split plans.
//!
//! This module generates and stores split configurations for Walk-Forward
//! and CPCV validation. Splits are computed once at startup and reused
//! for all genome evaluations.

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

use super::split_data::{SplitDataRef, SplitPair, NestedSplitTriplet, SplitType};

/// Configuration for generating validation splits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitPlanConfig {
    /// Overall data start date
    pub start_date: NaiveDate,
    /// Overall data end date
    pub end_date: NaiveDate,
    /// Training period in months
    pub train_months: u32,
    /// Validation period in months (for nested WFA)
    pub val_months: u32,
    /// Test period in months
    pub test_months: u32,
    /// Step size in months (how much to advance between splits)
    pub step_months: u32,
    /// Purge period in days
    pub purge_days: u32,
    /// Embargo period in days
    pub embargo_days: u32,
    /// Minimum trading days for a valid split
    pub min_trading_days: u32,
    /// Use nested (3-segment) splits
    pub use_nested: bool,
}

impl Default for SplitPlanConfig {
    fn default() -> Self {
        Self {
            start_date: NaiveDate::from_ymd_opt(2010, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2020, 12, 31).unwrap(),
            train_months: 36, // 3 years
            val_months: 12,   // 1 year (for nested)
            test_months: 12,  // 1 year
            step_months: 12,  // 1 year steps
            purge_days: 5,
            embargo_days: 5,
            min_trading_days: 126, // 6 months
            use_nested: false,
        }
    }
}

/// Pre-computed validation split plan
#[derive(Debug, Clone)]
pub struct ValidationSplitPlan {
    /// Configuration used to generate this plan
    pub config: SplitPlanConfig,
    /// 2-segment splits (train/test)
    pub splits: Vec<SplitPair>,
    /// 3-segment splits (train/val/test) for nested WFA
    pub nested_splits: Vec<NestedSplitTriplet>,
    /// Total data rows available
    pub total_rows: usize,
    /// Rows per trading day estimate
    pub rows_per_day: f64,
}

impl ValidationSplitPlan {
    /// Generate a new split plan from configuration
    pub fn generate(config: SplitPlanConfig, total_rows: usize) -> Self {
        let total_days = (config.end_date - config.start_date).num_days() as f64;
        let rows_per_day = if total_days > 0.0 {
            total_rows as f64 / total_days * (365.0 / 252.0) // Adjust for trading days
        } else {
            1.0
        };

        let mut plan = Self {
            config: config.clone(),
            splits: Vec::new(),
            nested_splits: Vec::new(),
            total_rows,
            rows_per_day,
        };

        if config.use_nested {
            plan.generate_nested_splits();
        } else {
            plan.generate_simple_splits();
        }

        plan
    }

    /// Generate simple 2-segment splits
    fn generate_simple_splits(&mut self) {
        let config = &self.config;
        let mut split_index: u16 = 0;
        let mut current_start = config.start_date;

        loop {
            // Calculate train period
            let train_end = add_months(current_start, config.train_months);
            let train_end_with_purge = subtract_days(train_end, config.purge_days);

            // Calculate test period
            let test_start = add_days(train_end, config.embargo_days);
            let test_end = add_months(test_start, config.test_months);

            // Check if we've exceeded the data range
            if test_end > config.end_date {
                break;
            }

            // Convert dates to row ranges
            let train_row_start = self.date_to_row(current_start);
            let train_row_end = self.date_to_row(train_end_with_purge);
            let test_row_start = self.date_to_row(test_start);
            let test_row_end = self.date_to_row(test_end);

            // Create split references
            let train = SplitDataRef::new(
                split_index,
                (current_start, train_end_with_purge),
                train_row_start..train_row_end,
                SplitType::Train,
            );

            let test = SplitDataRef::new(
                split_index,
                (test_start, test_end),
                test_row_start..test_row_end,
                SplitType::Test,
            );

            // Check minimum data requirements
            if train.has_sufficient_data(config.min_trading_days)
                && test.has_sufficient_data(config.min_trading_days / 2)
            {
                self.splits.push(SplitPair::new(
                    train,
                    test,
                    config.purge_days,
                    config.embargo_days,
                ));
            }

            // Advance to next split
            current_start = add_months(current_start, config.step_months);
            split_index += 1;

            // Safety limit
            if split_index > 100 {
                break;
            }
        }
    }

    /// Generate nested 3-segment splits for research-grade validation
    fn generate_nested_splits(&mut self) {
        let config = &self.config;
        let mut split_index: u16 = 0;
        let mut current_start = config.start_date;

        loop {
            // Calculate train period
            let train_end_raw = add_months(current_start, config.train_months);
            let train_end = subtract_days(train_end_raw, config.purge_days);

            // Calculate validation period
            let val_start = add_days(train_end_raw, config.embargo_days);
            let val_end_raw = add_months(val_start, config.val_months);
            let val_end = subtract_days(val_end_raw, config.purge_days);

            // Calculate test period
            let test_start = add_days(val_end_raw, config.embargo_days);
            let test_end = add_months(test_start, config.test_months);

            // Check if we've exceeded the data range
            if test_end > config.end_date {
                break;
            }

            // Convert dates to row ranges
            let train_row_start = self.date_to_row(current_start);
            let train_row_end = self.date_to_row(train_end);
            let val_row_start = self.date_to_row(val_start);
            let val_row_end = self.date_to_row(val_end);
            let test_row_start = self.date_to_row(test_start);
            let test_row_end = self.date_to_row(test_end);

            // Create split references
            let train = SplitDataRef::new(
                split_index,
                (current_start, train_end),
                train_row_start..train_row_end,
                SplitType::Train,
            );

            let validation = SplitDataRef::new(
                split_index,
                (val_start, val_end),
                val_row_start..val_row_end,
                SplitType::Validation,
            );

            let test = SplitDataRef::new(
                split_index,
                (test_start, test_end),
                test_row_start..test_row_end,
                SplitType::Test,
            );

            // Check minimum data requirements
            if train.has_sufficient_data(config.min_trading_days)
                && validation.has_sufficient_data(config.min_trading_days / 2)
                && test.has_sufficient_data(config.min_trading_days / 2)
            {
                self.nested_splits.push(NestedSplitTriplet::new(
                    train,
                    validation,
                    test,
                    config.purge_days,
                    config.purge_days,
                    config.embargo_days,
                ));
            }

            // Advance to next split
            current_start = add_months(current_start, config.step_months);
            split_index += 1;

            // Safety limit
            if split_index > 100 {
                break;
            }
        }
    }

    /// Convert a date to an approximate row index
    fn date_to_row(&self, date: NaiveDate) -> usize {
        let days_from_start = (date - self.config.start_date).num_days().max(0) as f64;
        let trading_days = days_from_start * (252.0 / 365.0); // Approximate trading days
        (trading_days * self.rows_per_day / (252.0 / 365.0)) as usize
    }

    /// Get the number of 2-segment splits
    pub fn num_splits(&self) -> usize {
        self.splits.len()
    }

    /// Get the number of 3-segment nested splits
    pub fn num_nested_splits(&self) -> usize {
        self.nested_splits.len()
    }

    /// Get split pair by index
    pub fn get_split(&self, index: usize) -> Option<&SplitPair> {
        self.splits.get(index)
    }

    /// Get nested split by index
    pub fn get_nested_split(&self, index: usize) -> Option<&NestedSplitTriplet> {
        self.nested_splits.get(index)
    }

    /// Iterator over all split pairs
    pub fn iter_splits(&self) -> impl Iterator<Item = &SplitPair> {
        self.splits.iter()
    }

    /// Iterator over all nested splits
    pub fn iter_nested_splits(&self) -> impl Iterator<Item = &NestedSplitTriplet> {
        self.nested_splits.iter()
    }

    /// Get total OOS trading days across all splits
    pub fn total_oos_days(&self) -> u32 {
        self.splits.iter().map(|s| s.test.trading_days).sum()
    }

    /// Summary statistics for the split plan
    pub fn summary(&self) -> SplitPlanSummary {
        SplitPlanSummary {
            num_splits: self.num_splits(),
            num_nested_splits: self.num_nested_splits(),
            total_oos_days: self.total_oos_days(),
            first_oos_date: self.splits.first().map(|s| s.test.date_range.0),
            last_oos_date: self.splits.last().map(|s| s.test.date_range.1),
        }
    }
}

/// Summary of a split plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitPlanSummary {
    pub num_splits: usize,
    pub num_nested_splits: usize,
    pub total_oos_days: u32,
    pub first_oos_date: Option<NaiveDate>,
    pub last_oos_date: Option<NaiveDate>,
}

// ============================================================================
// Helper functions for date arithmetic
// ============================================================================

/// Add months to a date
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

    NaiveDate::from_ymd_opt(new_year, new_month as u32, new_day).unwrap_or(date)
}

/// Subtract days from a date
fn subtract_days(date: NaiveDate, days: u32) -> NaiveDate {
    date - chrono::Duration::days(days as i64)
}

/// Add days to a date
fn add_days(date: NaiveDate, days: u32) -> NaiveDate {
    date + chrono::Duration::days(days as i64)
}

/// Get the number of days in a month
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

use chrono::Datelike;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SplitPlanConfig::default();
        assert_eq!(config.train_months, 36);
        assert_eq!(config.test_months, 12);
        assert_eq!(config.step_months, 12);
    }

    #[test]
    fn test_generate_simple_splits() {
        let config = SplitPlanConfig {
            start_date: NaiveDate::from_ymd_opt(2010, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2020, 12, 31).unwrap(),
            train_months: 24, // 2 years
            val_months: 6,
            test_months: 6, // 6 months
            step_months: 6, // 6 month steps
            purge_days: 5,
            embargo_days: 5,
            min_trading_days: 63, // 3 months
            use_nested: false,
        };

        let plan = ValidationSplitPlan::generate(config, 2520); // ~10 years of daily data

        assert!(plan.num_splits() > 0, "Should generate at least one split");
        assert!(plan.num_splits() <= 20, "Should not exceed reasonable number of splits");

        // First split should start at data start
        let first = plan.get_split(0).unwrap();
        assert_eq!(first.train.split_index, 0);
    }

    #[test]
    fn test_generate_nested_splits() {
        let config = SplitPlanConfig {
            start_date: NaiveDate::from_ymd_opt(2010, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2020, 12, 31).unwrap(),
            train_months: 24,
            val_months: 6,
            test_months: 6,
            step_months: 6,
            purge_days: 5,
            embargo_days: 5,
            min_trading_days: 63,
            use_nested: true,
        };

        let plan = ValidationSplitPlan::generate(config, 2520);

        assert!(plan.num_nested_splits() > 0, "Should generate nested splits");
        
        // Each nested split should have 3 segments
        let first = plan.get_nested_split(0).unwrap();
        assert_eq!(first.train.split_type, SplitType::Train);
        assert_eq!(first.validation.split_type, SplitType::Validation);
        assert_eq!(first.test.split_type, SplitType::Test);
    }

    #[test]
    fn test_add_months() {
        let date = NaiveDate::from_ymd_opt(2020, 1, 15).unwrap();
        
        let plus_1 = add_months(date, 1);
        assert_eq!(plus_1.month(), 2);
        
        let plus_12 = add_months(date, 12);
        assert_eq!(plus_12.year(), 2021);
        assert_eq!(plus_12.month(), 1);
    }
}

