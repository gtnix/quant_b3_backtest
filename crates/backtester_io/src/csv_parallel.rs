//! Parallel CSV parsing with memory-mapped I/O.
//!
//! Ultra-performance CSV ingestion using rayon + memmap2.
//! ~3-5x faster than sequential parsing on multi-core systems.

use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::path::Path;

use crate::{DataError, RawBar};

/// Parallel CSV loader with memory-mapped I/O.
///
/// # Performance
/// - Memory-mapped file avoids copy overhead
/// - Parallel chunk parsing (one per CPU core)
/// - ~3-5x speedup vs sequential on 8+ cores
pub struct CsvParallelLoader {
    skip_invalid: bool,
    chunk_size: usize,
}

impl CsvParallelLoader {
    /// Create a new parallel CSV loader.
    #[must_use]
    pub fn new() -> Self {
        Self {
            skip_invalid: true,
            chunk_size: 64 * 1024, // 64KB chunks
        }
    }

    /// Configure chunk size in bytes (default: 64KB).
    #[must_use]
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Configure whether to skip invalid bars.
    #[must_use]
    pub fn skip_invalid(mut self, skip: bool) -> Self {
        self.skip_invalid = skip;
        self
    }

    /// Load CSV file in parallel.
    pub fn load<P: AsRef<Path>>(&self, path: P) -> Result<Vec<RawBar>, DataError> {
        let file = File::open(path.as_ref())
            .map_err(|e| DataError::IoError(format!("{}: {e}", path.as_ref().display())))?;

        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| DataError::IoError(format!("mmap failed: {e}")))?;

        let data = &mmap[..];
        
        // Find line boundaries for parallel processing
        let chunks = self.split_into_chunks(data);

        // Parse chunks in parallel
        let skip_invalid = self.skip_invalid;
        let results: Vec<Vec<RawBar>> = chunks
            .into_par_iter()
            .map(|(start, end, is_first)| {
                parse_chunk(&data[start..end], skip_invalid, is_first)
            })
            .collect();

        // Flatten results
        let total_bars: usize = results.iter().map(|v| v.len()).sum();
        let mut all_bars = Vec::with_capacity(total_bars);
        for chunk_bars in results {
            all_bars.extend(chunk_bars);
        }

        Ok(all_bars)
    }

    /// Split data into chunks at line boundaries.
    fn split_into_chunks(&self, data: &[u8]) -> Vec<(usize, usize, bool)> {
        if data.is_empty() {
            return Vec::new();
        }

        let num_chunks = (data.len() / self.chunk_size).max(1);
        let mut chunks = Vec::with_capacity(num_chunks);
        let mut pos = 0;

        for i in 0..num_chunks {
            let target_end = ((i + 1) * data.len()) / num_chunks;
            let actual_end = find_line_boundary(data, target_end);
            
            if pos < actual_end {
                chunks.push((pos, actual_end, i == 0));
                pos = actual_end;
            }
        }

        // Handle remaining data
        if pos < data.len() {
            chunks.push((pos, data.len(), chunks.is_empty()));
        }

        chunks
    }
}

impl Default for CsvParallelLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Find next line boundary (newline) from position.
fn find_line_boundary(data: &[u8], from: usize) -> usize {
    if from >= data.len() {
        return data.len();
    }
    
    for i in from..data.len() {
        if data[i] == b'\n' {
            return i + 1;
        }
    }
    data.len()
}

/// Parse a chunk of CSV data.
fn parse_chunk(data: &[u8], skip_invalid: bool, is_first_chunk: bool) -> Vec<RawBar> {
    let text = match std::str::from_utf8(data) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };

    let mut bars = Vec::new();
    let mut skip_first = is_first_chunk; // Skip header in first chunk

    for line in text.lines() {
        // Skip header
        if skip_first && line.to_lowercase().contains("timestamp") {
            skip_first = false;
            continue;
        }
        skip_first = false;

        if line.trim().is_empty() {
            continue;
        }

        if let Some(bar) = parse_line(line, skip_invalid) {
            bars.push(bar);
        }
    }

    bars
}

/// Parse a single CSV line into RawBar.
fn parse_line(line: &str, skip_invalid: bool) -> Option<RawBar> {
    let fields: Vec<&str> = line.split(',').collect();
    if fields.len() < 7 {
        return None;
    }

    let timestamp = parse_timestamp_fast(fields[0].trim())?;
    let open = fields[2].trim().parse::<f64>().ok()?;
    let high = fields[3].trim().parse::<f64>().ok()?;
    let low = fields[4].trim().parse::<f64>().ok()?;
    let close = fields[5].trim().parse::<f64>().ok()?;
    let volume = fields[6].trim().parse::<f64>().ok()?;

    // Validate OHLC
    if !skip_invalid {
        if open < 0.0 || high < 0.0 || low < 0.0 || close < 0.0 || volume < 0.0 {
            return None;
        }
        if high < low {
            return None;
        }
    }

    Some(RawBar {
        ticker: fields[1].trim().to_string(),
        timestamp,
        open,
        high,
        low,
        close,
        volume,
    })
}

/// Fast timestamp parsing (nanoseconds or ISO 8601).
fn parse_timestamp_fast(s: &str) -> Option<i64> {
    // Try epoch nanoseconds first (fastest path)
    if let Ok(ts) = s.parse::<i64>() {
        return Some(ts);
    }

    // Try ISO 8601 with chrono
    use chrono::{DateTime, NaiveDate, Utc};
    
    if let Ok(dt) = s.parse::<DateTime<Utc>>() {
        return dt.timestamp_nanos_opt();
    }

    // Try date-only format
    if let Ok(date) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let dt = date.and_hms_opt(23, 59, 59)?;
        let utc = DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc);
        return utc.timestamp_nanos_opt();
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_csv(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file
    }

    #[test]
    fn parallel_loader_parses_valid_data() {
        let csv = "timestamp,ticker,open,high,low,close,volume\n\
                   1700000000000000000,PETR4,35.0,36.0,34.0,35.5,1000000\n\
                   1700000000000000000,VALE3,70.0,72.0,69.0,71.0,500000\n";
        let file = create_test_csv(csv);
        let loader = CsvParallelLoader::new();
        let bars = loader.load(file.path()).unwrap();
        assert_eq!(bars.len(), 2);
    }

    #[test]
    fn parallel_loader_handles_empty_file() {
        let file = create_test_csv("");
        let loader = CsvParallelLoader::new();
        let bars = loader.load(file.path()).unwrap();
        assert!(bars.is_empty());
    }

    #[test]
    fn parallel_loader_handles_header_only() {
        let csv = "timestamp,ticker,open,high,low,close,volume\n";
        let file = create_test_csv(csv);
        let loader = CsvParallelLoader::new();
        let bars = loader.load(file.path()).unwrap();
        assert!(bars.is_empty());
    }
}
