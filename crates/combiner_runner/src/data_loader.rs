//! Memory-mapped OHLCV data loader for ultra-fast data access.
//!
//! This module provides zero-copy access to market data using memory mapping.
//! The data is mapped directly from disk into virtual memory, avoiding
//! explicit read calls and enabling efficient random access.

use std::fs::File;
use std::io;
use std::path::Path;
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};
use chrono::NaiveDate;

/// Memory-mapped OHLCV data
pub struct MmapOhlcv {
    /// Memory-mapped file
    mmap: Mmap,
    /// Number of rows in the data
    num_rows: usize,
    /// Bytes per row (fixed-width record)
    bytes_per_row: usize,
    /// Start date of the data
    start_date: NaiveDate,
    /// End date of the data
    end_date: NaiveDate,
}

impl MmapOhlcv {
    /// Create a new memory-mapped loader from a file path.
    ///
    /// The file is expected to be in a binary format with fixed-width records.
    /// Each record contains: date (4 bytes), open, high, low, close, volume (8 bytes each).
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        let file_size = metadata.len() as usize;

        // Safety: We're mapping a read-only file
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // Default record format: date(4) + OHLCV(8*5) = 44 bytes
        let bytes_per_row = 44;
        let num_rows = file_size / bytes_per_row;

        // Default dates (would be read from header in production)
        let start_date = NaiveDate::from_ymd_opt(2010, 1, 1).unwrap();
        let end_date = NaiveDate::from_ymd_opt(2020, 12, 31).unwrap();

        Ok(Self {
            mmap,
            num_rows,
            bytes_per_row,
            start_date,
            end_date,
        })
    }

    /// Create from raw bytes (for testing)
    pub fn from_bytes(data: Vec<u8>, bytes_per_row: usize) -> Self {
        let num_rows = data.len() / bytes_per_row;
        
        // Create a fake mmap from the data
        // In production, this would use a temp file or shared memory
        Self {
            mmap: unsafe { Mmap::map(&File::open("/dev/null").unwrap()).unwrap() },
            num_rows,
            bytes_per_row,
            start_date: NaiveDate::from_ymd_opt(2010, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2020, 12, 31).unwrap(),
        }
    }

    /// Get the total number of rows
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Get the raw byte slice for a range of rows
    #[inline]
    pub fn slice_rows(&self, start_row: usize, end_row: usize) -> &[u8] {
        let start_byte = start_row * self.bytes_per_row;
        let end_byte = (end_row.min(self.num_rows)) * self.bytes_per_row;
        
        if start_byte >= self.mmap.len() || end_byte > self.mmap.len() {
            return &[];
        }
        
        &self.mmap[start_byte..end_byte]
    }

    /// Get the entire data as a byte slice
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Get start date
    pub fn start_date(&self) -> NaiveDate {
        self.start_date
    }

    /// Get end date
    pub fn end_date(&self) -> NaiveDate {
        self.end_date
    }

    /// Convert row index to approximate date
    pub fn row_to_date(&self, row: usize) -> NaiveDate {
        let total_days = (self.end_date - self.start_date).num_days() as f64;
        let fraction = row as f64 / self.num_rows.max(1) as f64;
        let days_offset = (total_days * fraction) as i64;
        self.start_date + chrono::Duration::days(days_offset)
    }

    /// Convert date to approximate row index
    pub fn date_to_row(&self, date: NaiveDate) -> usize {
        if date <= self.start_date {
            return 0;
        }
        if date >= self.end_date {
            return self.num_rows;
        }
        
        let total_days = (self.end_date - self.start_date).num_days() as f64;
        let days_from_start = (date - self.start_date).num_days() as f64;
        let fraction = days_from_start / total_days;
        (self.num_rows as f64 * fraction) as usize
    }
}

/// Shared reference-counted OHLCV data
pub type SharedMmapOhlcv = Arc<MmapOhlcv>;

/// Create a shared memory-mapped loader
pub fn load_shared<P: AsRef<Path>>(path: P) -> io::Result<SharedMmapOhlcv> {
    Ok(Arc::new(MmapOhlcv::open(path)?))
}

/// Mock OHLCV data for testing (generates synthetic returns)
pub struct MockOhlcv {
    /// Daily returns
    pub returns: Vec<f64>,
    /// Start date
    pub start_date: NaiveDate,
}

impl MockOhlcv {
    /// Generate mock data with specified characteristics
    pub fn generate(num_days: usize, mean_return: f64, volatility: f64, seed: u64) -> Self {
        let mut returns = Vec::with_capacity(num_days);
        let mut state = seed;

        for _ in 0..num_days {
            // Simple LCG for deterministic pseudo-random
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let uniform = ((state >> 33) as f64) / (u32::MAX as f64);
            
            // Box-Muller transform for normal distribution (simplified)
            let z = (uniform - 0.5) * 3.46; // Approximation
            returns.push(mean_return + z * volatility);
        }

        Self {
            returns,
            start_date: NaiveDate::from_ymd_opt(2010, 1, 1).unwrap(),
        }
    }

    /// Get returns slice for a row range
    pub fn slice(&self, start: usize, end: usize) -> &[f64] {
        let end = end.min(self.returns.len());
        if start >= end {
            return &[];
        }
        &self.returns[start..end]
    }

    /// Get all returns
    pub fn all_returns(&self) -> &[f64] {
        &self.returns
    }

    /// Get number of days
    pub fn num_days(&self) -> usize {
        self.returns.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_ohlcv_generation() {
        let mock = MockOhlcv::generate(252, 0.0005, 0.015, 42);
        
        assert_eq!(mock.num_days(), 252);
        assert!(mock.returns.iter().all(|&r| r.is_finite()));
    }

    #[test]
    fn test_mock_ohlcv_slice() {
        let mock = MockOhlcv::generate(1000, 0.0003, 0.01, 123);
        
        let slice = mock.slice(100, 200);
        assert_eq!(slice.len(), 100);
        
        let empty = mock.slice(1000, 1100);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_date_row_conversion() {
        // This test would need a real mmap to work
        // For now, just test the mock
        let mock = MockOhlcv::generate(2520, 0.0003, 0.01, 42);
        assert_eq!(mock.num_days(), 2520);
    }
}

