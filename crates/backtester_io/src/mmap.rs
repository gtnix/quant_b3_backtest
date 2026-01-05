//! # Memory-Mapped File Stream
//!
//! Zero-copy CSV parsing using memory-mapped files for maximum I/O performance.
//! Provides streaming iterator over market events without loading entire file into memory.

use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

use backtester_core::{Bar, MarketEvent};

use crate::{DataError, Normalizer};

/// Memory-mapped CSV reader for zero-copy access.
pub struct MmapReader {
    mmap: Mmap,
    line_offsets: Vec<usize>,
}

impl MmapReader {
    /// Open a file with memory mapping.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, DataError> {
        let file = File::open(path.as_ref())
            .map_err(|e| DataError::IoError(format!("{}: {e}", path.as_ref().display())))?;

        // SAFETY: We only read from the mmap, file is kept open
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| DataError::IoError(format!("mmap failed: {e}")))?
        };

        // Pre-compute line offsets for fast access
        let mut line_offsets = vec![0];
        for (i, &byte) in mmap.iter().enumerate() {
            if byte == b'\n' {
                line_offsets.push(i + 1);
            }
        }

        // Add final offset if file doesn't end with newline
        let last_offset = *line_offsets.last().unwrap_or(&0);
        if !mmap.is_empty() && last_offset < mmap.len() {
            line_offsets.push(mmap.len());
        }

        Ok(Self { mmap, line_offsets })
    }

    /// Get total number of lines.
    #[must_use]
    pub fn line_count(&self) -> usize {
        self.line_offsets.len().saturating_sub(1)
    }

    /// Get a specific line as bytes (zero-copy).
    #[must_use]
    pub fn get_line(&self, index: usize) -> Option<&[u8]> {
        if index >= self.line_offsets.len() - 1 {
            return None;
        }
        let start = self.line_offsets[index];
        let end = self.line_offsets[index + 1];
        // Trim newline
        let end = if end > start && self.mmap.get(end - 1) == Some(&b'\n') {
            end - 1
        } else {
            end
        };
        // Trim carriage return
        let end = if end > start && self.mmap.get(end - 1) == Some(&b'\r') {
            end - 1
        } else {
            end
        };
        Some(&self.mmap[start..end])
    }

    /// Get raw bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }
}

/// Zero-copy streaming iterator over market events.
pub struct MmapStream {
    reader: MmapReader,
    normalizer: Normalizer,
    current_line: usize,
    has_header: bool,
}

impl MmapStream {
    /// Create a new mmap stream from a CSV file.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, DataError> {
        let reader = MmapReader::open(path)?;

        // Check if first line is header
        let has_header = if let Some(first_line) = reader.get_line(0) {
            let line_str = std::str::from_utf8(first_line).unwrap_or("");
            line_str.to_lowercase().contains("timestamp")
        } else {
            false
        };

        Ok(Self {
            reader,
            normalizer: Normalizer::new(),
            current_line: if has_header { 1 } else { 0 },
            has_header,
        })
    }

    /// Get total number of data lines (excluding header).
    #[must_use]
    pub fn len(&self) -> usize {
        let total = self.reader.line_count();
        if self.has_header && total > 0 {
            total - 1
        } else {
            total
        }
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reset to beginning.
    pub fn reset(&mut self) {
        self.current_line = if self.has_header { 1 } else { 0 };
    }

    /// Parse a line into a MarketEvent.
    /// 
    /// # Performance
    /// Uses split iterator instead of collecting to Vec, avoiding allocation.
    /// Returns parsed data tuple to decouple from mmap borrow.
    fn parse_line_data(line: &[u8]) -> Option<(i64, String, f64, f64, f64, f64, f64)> {
        let line_str = std::str::from_utf8(line).ok()?;
        if line_str.trim().is_empty() {
            return None;
        }

        // Zero-alloc: use split iterator with nth() instead of collect
        let mut fields = line_str.split(',');
        
        let timestamp = parse_timestamp_fast(fields.next()?.trim())?;
        let ticker = fields.next()?.trim().to_string(); // Copy ticker to avoid borrow
        let open = fields.next()?.trim().parse::<f64>().ok()?;
        let high = fields.next()?.trim().parse::<f64>().ok()?;
        let low = fields.next()?.trim().parse::<f64>().ok()?;
        let close = fields.next()?.trim().parse::<f64>().ok()?;
        let volume = fields.next()?.trim().parse::<f64>().ok()?;

        // Validate OHLC
        if high < low || open < 0.0 || close < 0.0 {
            return None;
        }

        Some((timestamp, ticker, open, high, low, close, volume))
    }
    
    /// Parse line at given index without advancing iterator.
    fn parse_line_at(&mut self, line_idx: usize) -> Option<MarketEvent> {
        let line = self.reader.get_line(line_idx)?;
        let (timestamp, ticker, open, high, low, close, volume) = Self::parse_line_data(line)?;
        
        let asset_id = self.normalizer.register_ticker(ticker);

        Some(MarketEvent {
            asset_id,
            bar: Bar {
                timestamp,
                open,
                high,
                low,
                close,
                volume,
            },
        })
    }

    /// Get normalizer reference.
    #[must_use]
    pub fn normalizer(&self) -> &Normalizer {
        &self.normalizer
    }

    /// Consume and get normalizer.
    #[must_use]
    pub fn into_normalizer(self) -> Normalizer {
        self.normalizer
    }

    /// Prefetch next N lines for cache warming.
    #[inline]
    pub fn prefetch(&self, count: usize) {
        let end = (self.current_line + count).min(self.reader.line_count());
        for i in self.current_line..end {
            // Touch the memory to bring it into cache
            let _ = self.reader.get_line(i);
        }
    }

    /// Load all events into a vector (useful for parallel processing).
    pub fn load_all(&mut self) -> Vec<MarketEvent> {
        self.reset();
        let mut events = Vec::with_capacity(self.len());

        for event in self.by_ref() {
            events.push(event);
        }

        // Sort by timestamp for deterministic order
        events.sort_by_key(|e| e.bar.timestamp);
        events
    }
}

impl Iterator for MmapStream {
    type Item = MarketEvent;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_line >= self.reader.line_count() {
                return None;
            }

            let line_idx = self.current_line;
            self.current_line += 1;

            // Zero-copy: parse directly from mmap without to_vec()
            if let Some(event) = self.parse_line_at(line_idx) {
                return Some(event);
            }
            // Skip invalid lines
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.reader.line_count().saturating_sub(self.current_line);
        (0, Some(remaining))
    }
}

/// Fast timestamp parser optimized for common formats.
fn parse_timestamp_fast(s: &str) -> Option<i64> {
    // Try epoch nanoseconds first (most common in our data)
    if let Ok(ts) = s.parse::<i64>() {
        return Some(ts);
    }

    // Try epoch seconds
    if s.len() <= 12 {
        if let Ok(secs) = s.parse::<i64>() {
            return Some(secs * 1_000_000_000);
        }
    }

    // Try date format YYYY-MM-DD
    if s.len() == 10 && s.chars().nth(4) == Some('-') {
        let year: i32 = s[0..4].parse().ok()?;
        let month: u32 = s[5..7].parse().ok()?;
        let day: u32 = s[8..10].parse().ok()?;

        // Simple date to timestamp (approximate, but fast)
        let days_since_epoch = days_from_date(year, month, day)?;
        return Some(days_since_epoch * 86_400_000_000_000);
    }

    // Fallback to chrono parsing
    use chrono::{DateTime, NaiveDate, Utc};

    if let Ok(dt) = s.parse::<DateTime<Utc>>() {
        return dt.timestamp_nanos_opt();
    }

    if let Ok(date) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let dt = date.and_hms_opt(23, 59, 59)?;
        let utc = DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc);
        return utc.timestamp_nanos_opt();
    }

    None
}

/// Fast days since Unix epoch calculation.
fn days_from_date(year: i32, month: u32, day: u32) -> Option<i64> {
    if month < 1 || month > 12 || day < 1 || day > 31 {
        return None;
    }

    // Algorithm from https://howardhinnant.github.io/date_algorithms.html
    let y = if month <= 2 { year - 1 } else { year } as i64;
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u32;
    let m = month;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe as i64 - 719468;

    Some(days)
}

/// Batch reader that loads data in chunks for parallel processing.
pub struct BatchReader {
    stream: MmapStream,
    batch_size: usize,
}

impl BatchReader {
    /// Create a new batch reader.
    pub fn new(stream: MmapStream, batch_size: usize) -> Self {
        Self {
            stream,
            batch_size: batch_size.max(1),
        }
    }

    /// Get next batch of events.
    pub fn next_batch(&mut self) -> Option<Vec<MarketEvent>> {
        let mut batch = Vec::with_capacity(self.batch_size);

        for _ in 0..self.batch_size {
            match self.stream.next() {
                Some(event) => batch.push(event),
                None => break,
            }
        }

        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }

    /// Reset to beginning.
    pub fn reset(&mut self) {
        self.stream.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_csv(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn mmap_reader_opens_file() {
        let csv = "timestamp,ticker,open,high,low,close,volume\n1700000000000000000,PETR4,35.0,36.0,34.0,35.5,1000000";
        let file = create_test_csv(csv);
        let reader = MmapReader::open(file.path()).unwrap();
        assert_eq!(reader.line_count(), 2);
    }

    #[test]
    fn mmap_stream_parses_events() {
        let csv = "timestamp,ticker,open,high,low,close,volume\n1700000000000000000,PETR4,35.0,36.0,34.0,35.5,1000000\n1700000000000000000,VALE3,70.0,72.0,69.0,71.0,500000";
        let file = create_test_csv(csv);
        let mut stream = MmapStream::open(file.path()).unwrap();

        let events: Vec<_> = stream.by_ref().collect();
        assert_eq!(events.len(), 2);
        assert!((events[0].bar.close - 35.5).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_timestamp_fast_works() {
        // Nanoseconds
        assert_eq!(
            parse_timestamp_fast("1700000000000000000"),
            Some(1700000000000000000)
        );

        // Date format
        let ts = parse_timestamp_fast("2023-11-14").unwrap();
        assert!(ts > 0);
    }

    #[test]
    fn batch_reader_chunks_data() {
        let csv = "timestamp,ticker,open,high,low,close,volume\n\
            1700000000000000000,A,1.0,1.0,1.0,1.0,100\n\
            1700000000000000000,B,2.0,2.0,2.0,2.0,200\n\
            1700000000000000000,C,3.0,3.0,3.0,3.0,300";
        let file = create_test_csv(csv);
        let stream = MmapStream::open(file.path()).unwrap();
        let mut batch_reader = BatchReader::new(stream, 2);

        let batch1 = batch_reader.next_batch().unwrap();
        assert_eq!(batch1.len(), 2);

        let batch2 = batch_reader.next_batch().unwrap();
        assert_eq!(batch2.len(), 1);

        assert!(batch_reader.next_batch().is_none());
    }
}
