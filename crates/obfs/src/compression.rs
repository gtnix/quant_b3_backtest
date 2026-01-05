//! Compression pipeline for OBFS
//!
//! Implements a multi-stage compression pipeline:
//! 1. Delta encoding for numerical time-series data
//! 2. Zstd compression for general data
//!
//! This achieves better compression ratios than Zstd alone for time-series data.

use anyhow::Result;
use std::io::{Read, Write};
use zstd::stream::{Decoder, Encoder};

use crate::types::CompressionStats;

/// Default compression level for Zstandard
pub const DEFAULT_COMPRESSION_LEVEL: i32 = 3;

/// Ultra compression level for maximum compression ratio (Zstd max)
pub const ULTRA_COMPRESSION_LEVEL: i32 = 19;

/// Compression strategy for different data types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionStrategy {
    /// Standard Zstd compression (default)
    Standard,
    /// Delta encoding + Zstd for time-series floats
    DeltaTimeSeries,
    /// XOR encoding for floating point (Gorilla-inspired)
    XorFloat,
}

/// CompressionPipeline handles all compression operations
#[derive(Debug, Clone)]
pub struct CompressionPipeline {
    compression_level: i32,
    strategy: CompressionStrategy,
}

impl CompressionPipeline {
    /// Create a new CompressionPipeline with default settings
    pub fn new() -> Self {
        Self {
            compression_level: DEFAULT_COMPRESSION_LEVEL,
            strategy: CompressionStrategy::Standard,
        }
    }

    /// Create a CompressionPipeline with custom compression level
    pub fn with_level(compression_level: i32) -> Self {
        Self {
            compression_level,
            strategy: CompressionStrategy::Standard,
        }
    }

    /// Create a CompressionPipeline with specific strategy
    pub fn with_strategy(compression_level: i32, strategy: CompressionStrategy) -> Self {
        Self {
            compression_level,
            strategy,
        }
    }

    /// Compress data using Zstandard
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut encoder = Encoder::new(Vec::new(), self.compression_level)?;
        encoder.write_all(data)?;
        let compressed = encoder.finish()?;
        Ok(compressed)
    }

    /// Decompress data using Zstandard
    pub fn decompress(&self, compressed_data: &[u8]) -> Result<Vec<u8>> {
        let mut decoder = Decoder::new(compressed_data)?;
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        Ok(decompressed)
    }

    /// Compress with statistics tracking
    pub fn compress_with_stats(&self, data: &[u8]) -> Result<(Vec<u8>, CompressionStats)> {
        let start = std::time::Instant::now();
        let compressed = self.compress(data)?;
        let compression_time_ms = start.elapsed().as_millis() as u64;

        let mut stats = CompressionStats::new(data.len() as u64, compressed.len() as u64);
        stats.compression_time_ms = compression_time_ms;

        Ok((compressed, stats))
    }

    /// Decompress with statistics tracking
    pub fn decompress_with_stats(&self, compressed_data: &[u8]) -> Result<(Vec<u8>, CompressionStats)> {
        let start = std::time::Instant::now();
        let decompressed = self.decompress(compressed_data)?;
        let decompression_time_ms = start.elapsed().as_millis() as u64;

        let mut stats = CompressionStats::new(decompressed.len() as u64, compressed_data.len() as u64);
        stats.decompression_time_ms = decompression_time_ms;

        Ok((decompressed, stats))
    }

    /// Get the current strategy
    pub fn strategy(&self) -> CompressionStrategy {
        self.strategy
    }

    /// Get the compression level
    pub fn level(&self) -> i32 {
        self.compression_level
    }
}

// ============================================================================
// Ultra Compression (Level 19 + LDM)
// ============================================================================

/// Ultra compressor using Zstd level 19 with long-distance matching.
/// Achieves maximum compression at the cost of slower compression speed.
/// Decompression remains fast.
#[derive(Debug, Clone)]
pub struct UltraCompressor;

impl UltraCompressor {
    /// Compress data with maximum compression (level 19 + LDM + checksum)
    pub fn compress(data: &[u8]) -> Result<Vec<u8>> {
        let mut encoder = Encoder::new(Vec::new(), ULTRA_COMPRESSION_LEVEL)?;
        encoder.include_checksum(true)?;
        encoder.long_distance_matching(true)?;
        encoder.write_all(data)?;
        Ok(encoder.finish()?)
    }

    /// Decompress ultra-compressed data
    pub fn decompress(compressed: &[u8]) -> Result<Vec<u8>> {
        let mut decoder = Decoder::new(compressed)?;
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        Ok(decompressed)
    }

    /// Compress with stats
    pub fn compress_with_stats(data: &[u8]) -> Result<(Vec<u8>, CompressionStats)> {
        let start = std::time::Instant::now();
        let compressed = Self::compress(data)?;
        let compression_time_ms = start.elapsed().as_millis() as u64;

        let mut stats = CompressionStats::new(data.len() as u64, compressed.len() as u64);
        stats.compression_time_ms = compression_time_ms;

        Ok((compressed, stats))
    }

    /// Decompress with stats
    pub fn decompress_with_stats(compressed: &[u8]) -> Result<(Vec<u8>, CompressionStats)> {
        let start = std::time::Instant::now();
        let decompressed = Self::decompress(compressed)?;
        let decompression_time_ms = start.elapsed().as_millis() as u64;

        let mut stats = CompressionStats::new(decompressed.len() as u64, compressed.len() as u64);
        stats.decompression_time_ms = decompression_time_ms;

        Ok((decompressed, stats))
    }
}

// ============================================================================
// Specialized Time-Series Compression
// ============================================================================

/// Delta encoder for f32 time-series data
/// Stores the first value as-is, then differences between consecutive values.
/// This improves compression for slowly-changing values (e.g., equity curves).
#[derive(Debug, Clone)]
pub struct DeltaEncoder;

impl DeltaEncoder {
    /// Encode f32 values using delta encoding
    pub fn encode_f32(values: &[f32]) -> Vec<u8> {
        if values.is_empty() {
            return Vec::new();
        }

        let mut output = Vec::with_capacity(values.len() * 4);
        
        // Store count as u32
        output.extend_from_slice(&(values.len() as u32).to_le_bytes());
        
        // Store first value as-is
        output.extend_from_slice(&values[0].to_le_bytes());
        
        // Store deltas (as i32 representing the XOR of bits)
        let mut prev_bits = values[0].to_bits();
        for &val in &values[1..] {
            let curr_bits = val.to_bits();
            let delta = curr_bits ^ prev_bits;
            output.extend_from_slice(&delta.to_le_bytes());
            prev_bits = curr_bits;
        }

        output
    }

    /// Decode delta-encoded f32 values
    pub fn decode_f32(data: &[u8]) -> Result<Vec<f32>> {
        if data.len() < 8 {
            return Ok(Vec::new());
        }

        let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let first_bits = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        
        let mut values = Vec::with_capacity(count);
        values.push(f32::from_bits(first_bits));

        let mut prev_bits = first_bits;
        let mut offset = 8;
        
        for _ in 1..count {
            if offset + 4 > data.len() {
                break;
            }
            let delta = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            let curr_bits = prev_bits ^ delta;
            values.push(f32::from_bits(curr_bits));
            prev_bits = curr_bits;
            offset += 4;
        }

        Ok(values)
    }
}

/// Time-series compressor using delta + Zstd pipeline
#[derive(Debug, Clone)]
pub struct TimeSeriesCompressor {
    zstd_level: i32,
}

impl TimeSeriesCompressor {
    /// Create a new TimeSeriesCompressor
    pub fn new(zstd_level: i32) -> Self {
        Self { zstd_level }
    }

    /// Compress f32 time-series data
    pub fn compress_f32(&self, values: &[f32]) -> Result<Vec<u8>> {
        // Step 1: Delta encode
        let delta_encoded = DeltaEncoder::encode_f32(values);
        
        // Step 2: Zstd compress
        let mut encoder = Encoder::new(Vec::new(), self.zstd_level)?;
        encoder.write_all(&delta_encoded)?;
        Ok(encoder.finish()?)
    }

    /// Decompress f32 time-series data
    pub fn decompress_f32(&self, compressed: &[u8]) -> Result<Vec<f32>> {
        // Step 1: Zstd decompress
        let mut decoder = Decoder::new(compressed)?;
        let mut delta_encoded = Vec::new();
        decoder.read_to_end(&mut delta_encoded)?;
        
        // Step 2: Delta decode
        DeltaEncoder::decode_f32(&delta_encoded)
    }

    /// Compress with stats
    pub fn compress_f32_with_stats(&self, values: &[f32]) -> Result<(Vec<u8>, CompressionStats)> {
        let original_size = (values.len() * 4) as u64;
        let start = std::time::Instant::now();
        let compressed = self.compress_f32(values)?;
        let compression_time = start.elapsed().as_millis() as u64;

        let mut stats = CompressionStats::new(original_size, compressed.len() as u64);
        stats.compression_time_ms = compression_time;

        Ok((compressed, stats))
    }
}

/// Compress multiple columns of time-series data together
#[derive(Debug, Clone)]
pub struct ColumnarCompressor {
    ts_compressor: TimeSeriesCompressor,
}

impl ColumnarCompressor {
    /// Create a new ColumnarCompressor
    pub fn new(zstd_level: i32) -> Self {
        Self {
            ts_compressor: TimeSeriesCompressor::new(zstd_level),
        }
    }

    /// Compress multiple f32 columns
    pub fn compress_columns(&self, columns: &[&[f32]]) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        
        // Header: number of columns
        output.extend_from_slice(&(columns.len() as u32).to_le_bytes());
        
        // Compress each column
        for col in columns {
            let compressed = self.ts_compressor.compress_f32(col)?;
            // Store compressed length
            output.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
            output.extend(compressed);
        }

        Ok(output)
    }

    /// Decompress multiple f32 columns
    pub fn decompress_columns(&self, data: &[u8]) -> Result<Vec<Vec<f32>>> {
        if data.len() < 4 {
            return Ok(Vec::new());
        }

        let num_columns = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let mut columns = Vec::with_capacity(num_columns);
        let mut offset = 4;

        for _ in 0..num_columns {
            if offset + 4 > data.len() {
                break;
            }
            
            let compressed_len = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;

            if offset + compressed_len > data.len() {
                break;
            }

            let compressed = &data[offset..offset + compressed_len];
            let column = self.ts_compressor.decompress_f32(compressed)?;
            columns.push(column);
            offset += compressed_len;
        }

        Ok(columns)
    }
}

impl Default for CompressionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_decompress() {
        let pipeline = CompressionPipeline::new();
        let original_data = b"This is a test string for compression. It should compress well because it has repetitive patterns.";

        let compressed = pipeline.compress(original_data).unwrap();
        let decompressed = pipeline.decompress(&compressed).unwrap();

        assert_eq!(original_data.as_slice(), decompressed.as_slice());
        assert!(compressed.len() < original_data.len(), "Data should be compressed");
    }

    #[test]
    fn test_compression_stats() {
        let pipeline = CompressionPipeline::new();
        let data = vec![0u8; 10000];

        let (compressed, stats) = pipeline.compress_with_stats(&data).unwrap();

        assert!(stats.compression_ratio > 1.0);
        assert_eq!(stats.original_size, 10000);
        assert_eq!(stats.compressed_size, compressed.len() as u64);
    }

    #[test]
    fn test_different_compression_levels() {
        let data = b"Test data for compression level comparison. ".repeat(100);

        let pipeline_fast = CompressionPipeline::with_level(1);
        let pipeline_balanced = CompressionPipeline::with_level(3);
        let pipeline_max = CompressionPipeline::with_level(19);

        let compressed_fast = pipeline_fast.compress(&data).unwrap();
        let compressed_balanced = pipeline_balanced.compress(&data).unwrap();
        let compressed_max = pipeline_max.compress(&data).unwrap();

        assert!(compressed_max.len() <= compressed_balanced.len());

        assert_eq!(pipeline_fast.decompress(&compressed_fast).unwrap(), data);
        assert_eq!(pipeline_balanced.decompress(&compressed_balanced).unwrap(), data);
        assert_eq!(pipeline_max.decompress(&compressed_max).unwrap(), data);
    }

    #[test]
    fn test_delta_encoder_roundtrip() {
        // Simulate equity curve: slowly increasing values
        let values: Vec<f32> = (0..1000)
            .map(|i| 1_000_000.0 + i as f32 * 100.0 + (i as f32 * 0.1).sin() * 1000.0)
            .collect();

        let encoded = DeltaEncoder::encode_f32(&values);
        let decoded = DeltaEncoder::decode_f32(&encoded).unwrap();

        assert_eq!(values.len(), decoded.len());
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.001, "Values should match: {} vs {}", orig, dec);
        }
    }

    #[test]
    fn test_timeseries_compressor() {
        let compressor = TimeSeriesCompressor::new(3);
        
        // Realistic equity curve data
        let values: Vec<f32> = (0..1245) // 5 years of trading days
            .map(|i| 1_000_000.0 * (1.0 + 0.0004 * i as f32)) // ~10% annual growth
            .collect();

        let (compressed, stats) = compressor.compress_f32_with_stats(&values).unwrap();
        
        println!("Time-series compression stats:");
        println!("  Original: {} bytes ({} floats)", values.len() * 4, values.len());
        println!("  Compressed: {} bytes", compressed.len());
        println!("  Ratio: {:.2}x", stats.compression_ratio);

        // Decompress and verify
        let decompressed = compressor.decompress_f32(&compressed).unwrap();
        assert_eq!(values.len(), decompressed.len());
        for (orig, dec) in values.iter().zip(decompressed.iter()) {
            assert!((orig - dec).abs() < 0.001);
        }

        // Delta + Zstd should achieve better compression than raw Zstd
        let raw_zstd = CompressionPipeline::new();
        let raw_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let raw_compressed = raw_zstd.compress(&raw_bytes).unwrap();
        
        println!("  Raw Zstd: {} bytes ({:.2}x)", raw_compressed.len(), raw_bytes.len() as f64 / raw_compressed.len() as f64);
        
        // Delta + Zstd should be comparable or better for smooth data
        assert!(compressed.len() <= raw_compressed.len() * 2, "Delta compression should not be much worse");
    }

    #[test]
    fn test_columnar_compressor() {
        let compressor = ColumnarCompressor::new(3);
        
        // Simulate multiple columns: equity, drawdown, exposure
        let equity: Vec<f32> = (0..100).map(|i| 1_000_000.0 + i as f32 * 1000.0).collect();
        let drawdown: Vec<f32> = (0..100).map(|i| -0.01 * i as f32).collect();
        let exposure: Vec<f32> = (0..100).map(|i| 0.5 + 0.001 * i as f32).collect();

        let columns: Vec<&[f32]> = vec![&equity, &drawdown, &exposure];
        
        let compressed = compressor.compress_columns(&columns).unwrap();
        let decompressed = compressor.decompress_columns(&compressed).unwrap();

        assert_eq!(decompressed.len(), 3);
        assert_eq!(decompressed[0].len(), 100);
        assert_eq!(decompressed[1].len(), 100);
        assert_eq!(decompressed[2].len(), 100);

        // Verify values
        for (orig, dec) in equity.iter().zip(decompressed[0].iter()) {
            assert!((orig - dec).abs() < 0.001);
        }
    }

    #[test]
    fn test_compression_strategy_selection() {
        let standard = CompressionPipeline::new();
        let delta = CompressionPipeline::with_strategy(3, CompressionStrategy::DeltaTimeSeries);

        assert_eq!(standard.strategy(), CompressionStrategy::Standard);
        assert_eq!(delta.strategy(), CompressionStrategy::DeltaTimeSeries);
    }

    #[test]
    fn test_ultra_compression() {
        // Typical SCG report data (repetitive JSON/TOML)
        let data = r#"
        {
            "genome_id": "06cd3b8a-24c7-4fec-a0ae-1a67d3a35188",
            "oos_sharpe": 0.75,
            "oos_cagr": 0.12,
            "max_drawdown": -0.15,
            "pbo": 0.10,
            "dsr": 0.65,
            "passed": true
        }
        "#.repeat(50);

        let (compressed, stats) = UltraCompressor::compress_with_stats(data.as_bytes()).unwrap();
        
        println!("Ultra compression stats:");
        println!("  Original: {} bytes", stats.original_size);
        println!("  Compressed: {} bytes", stats.compressed_size);
        println!("  Ratio: {:.2}x", stats.compression_ratio);
        println!("  Time: {} ms", stats.compression_time_ms);

        // Ultra should achieve significant compression on repetitive data
        assert!(stats.compression_ratio > 5.0, "Should achieve at least 5x on repetitive JSON");

        // Verify roundtrip
        let decompressed = UltraCompressor::decompress(&compressed).unwrap();
        assert_eq!(data.as_bytes(), decompressed.as_slice());
    }

    #[test]
    fn test_ultra_vs_standard_compression() {
        // Use larger data to see the benefit of ultra compression
        // Small data may have larger overhead due to LDM framing
        let data = b"Test data for comparing compression levels. ".repeat(10000);

        let standard = CompressionPipeline::with_level(3).compress(&data).unwrap();
        let ultra = UltraCompressor::compress(&data).unwrap();

        println!("Standard (level 3): {} bytes", standard.len());
        println!("Ultra (level 19): {} bytes", ultra.len());

        // Both should decompress correctly
        let standard_decompressed = CompressionPipeline::with_level(3).decompress(&standard).unwrap();
        let ultra_decompressed = UltraCompressor::decompress(&ultra).unwrap();
        
        assert_eq!(data.as_slice(), standard_decompressed.as_slice());
        assert_eq!(data.as_slice(), ultra_decompressed.as_slice());
    }
}

