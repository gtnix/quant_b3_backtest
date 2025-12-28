//! Zero-copy split data references for memory-efficient validation.
//!
//! This module provides data structures for referencing slices of market data
//! without copying. Used for Walk-Forward and CPCV validation where multiple
//! splits need to reference the same underlying data.

use std::ops::Range;
use chrono::NaiveDate;

/// Zero-copy reference to a data split.
///
/// Stores byte ranges into memory-mapped data rather than copying.
/// Used for ultra-fast validation with minimal memory footprint.
#[derive(Debug, Clone)]
pub struct SplitDataRef {
    /// Split index (0-based)
    pub split_index: u16,
    /// Date range for this split (inclusive start, exclusive end)
    pub date_range: (NaiveDate, NaiveDate),
    /// Row range in the underlying data (start, end)
    pub row_range: Range<usize>,
    /// Whether this is a train, validation, or test split
    pub split_type: SplitType,
    /// Number of trading days in this split
    pub trading_days: u32,
    /// Offset from data start in bytes (for mmap)
    pub byte_offset: usize,
    /// Length in bytes
    pub byte_length: usize,
}

/// Type of split segment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitType {
    /// Training data (in-sample)
    Train,
    /// Validation data (for nested WFA)
    Validation,
    /// Test data (out-of-sample)
    Test,
}

impl SplitDataRef {
    /// Create a new split reference
    pub fn new(
        split_index: u16,
        date_range: (NaiveDate, NaiveDate),
        row_range: Range<usize>,
        split_type: SplitType,
    ) -> Self {
        let trading_days = (row_range.end - row_range.start) as u32;
        Self {
            split_index,
            date_range,
            row_range,
            split_type,
            trading_days,
            byte_offset: 0,
            byte_length: 0,
        }
    }

    /// Create with byte offsets for mmap access
    pub fn with_byte_range(
        split_index: u16,
        date_range: (NaiveDate, NaiveDate),
        row_range: Range<usize>,
        split_type: SplitType,
        byte_offset: usize,
        byte_length: usize,
    ) -> Self {
        let trading_days = (row_range.end - row_range.start) as u32;
        Self {
            split_index,
            date_range,
            row_range,
            split_type,
            trading_days,
            byte_offset,
            byte_length,
        }
    }

    /// Get the number of rows in this split
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.row_range.end - self.row_range.start
    }

    /// Check if this split has enough data for meaningful analysis
    pub fn has_sufficient_data(&self, min_days: u32) -> bool {
        self.trading_days >= min_days
    }

    /// Get years covered by this split
    pub fn years(&self) -> f64 {
        self.trading_days as f64 / 252.0
    }
}

/// A pair of train/test splits for 2-segment WFA
#[derive(Debug, Clone)]
pub struct SplitPair {
    /// Training (in-sample) data reference
    pub train: SplitDataRef,
    /// Test (out-of-sample) data reference
    pub test: SplitDataRef,
    /// Purge period in days (gap between train and test)
    pub purge_days: u32,
    /// Embargo period in days (buffer after test)
    pub embargo_days: u32,
}

impl SplitPair {
    /// Create a new split pair
    pub fn new(train: SplitDataRef, test: SplitDataRef, purge_days: u32, embargo_days: u32) -> Self {
        Self {
            train,
            test,
            purge_days,
            embargo_days,
        }
    }

    /// Get the split index
    pub fn index(&self) -> u16 {
        self.train.split_index
    }

    /// Total trading days across both segments
    pub fn total_days(&self) -> u32 {
        self.train.trading_days + self.test.trading_days
    }

    /// Check if both segments have sufficient data
    pub fn has_sufficient_data(&self, min_train_days: u32, min_test_days: u32) -> bool {
        self.train.has_sufficient_data(min_train_days) && self.test.has_sufficient_data(min_test_days)
    }
}

/// A triplet of train/validation/test splits for 3-segment nested WFA
#[derive(Debug, Clone)]
pub struct NestedSplitTriplet {
    /// Training (in-sample) data reference
    pub train: SplitDataRef,
    /// Validation data reference (for parameter selection)
    pub validation: SplitDataRef,
    /// Test (out-of-sample) data reference
    pub test: SplitDataRef,
    /// Purge period between train and validation
    pub purge_train_val: u32,
    /// Purge period between validation and test
    pub purge_val_test: u32,
    /// Embargo period after test
    pub embargo_days: u32,
}

impl NestedSplitTriplet {
    /// Create a new nested split triplet
    pub fn new(
        train: SplitDataRef,
        validation: SplitDataRef,
        test: SplitDataRef,
        purge_train_val: u32,
        purge_val_test: u32,
        embargo_days: u32,
    ) -> Self {
        Self {
            train,
            validation,
            test,
            purge_train_val,
            purge_val_test,
            embargo_days,
        }
    }

    /// Get the split index
    pub fn index(&self) -> u16 {
        self.train.split_index
    }

    /// Total trading days across all segments
    pub fn total_days(&self) -> u32 {
        self.train.trading_days + self.validation.trading_days + self.test.trading_days
    }
}

/// Slice returns from a larger buffer using a SplitDataRef
pub fn slice_returns<'a>(data: &'a [f64], split: &SplitDataRef) -> &'a [f64] {
    let start = split.row_range.start;
    let end = split.row_range.end.min(data.len());
    if start >= end {
        return &[];
    }
    &data[start..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_data_ref_creation() {
        let split = SplitDataRef::new(
            0,
            (NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(), NaiveDate::from_ymd_opt(2020, 12, 31).unwrap()),
            0..252,
            SplitType::Train,
        );

        assert_eq!(split.split_index, 0);
        assert_eq!(split.trading_days, 252);
        assert_eq!(split.num_rows(), 252);
        assert!(split.has_sufficient_data(200));
        assert!(!split.has_sufficient_data(300));
    }

    #[test]
    fn test_split_pair() {
        let train = SplitDataRef::new(
            0,
            (NaiveDate::from_ymd_opt(2019, 1, 1).unwrap(), NaiveDate::from_ymd_opt(2019, 12, 31).unwrap()),
            0..252,
            SplitType::Train,
        );
        let test = SplitDataRef::new(
            0,
            (NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(), NaiveDate::from_ymd_opt(2020, 6, 30).unwrap()),
            257..383,
            SplitType::Test,
        );

        let pair = SplitPair::new(train, test, 5, 5);

        assert_eq!(pair.index(), 0);
        assert!(pair.total_days() > 300);
    }

    #[test]
    fn test_slice_returns() {
        let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
        let split = SplitDataRef::new(
            0,
            (NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(), NaiveDate::from_ymd_opt(2020, 6, 30).unwrap()),
            100..200,
            SplitType::Train,
        );

        let sliced = slice_returns(&data, &split);
        
        assert_eq!(sliced.len(), 100);
        assert!((sliced[0] - 0.1).abs() < 0.0001);
    }
}

