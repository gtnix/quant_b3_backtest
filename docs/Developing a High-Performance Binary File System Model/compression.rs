// compression.rs - Compression pipeline for OBFS

use anyhow::Result;
use std::io::{Read, Write};
use zstd::stream::{Encoder, Decoder};

use crate::types::CompressionStats;

/// Compression level for Zstandard
/// 
/// - Level 1: Fast compression, lower ratio (~2.5:1)
/// - Level 3: Balanced (default) (~2.8:1)
/// - Level 10: High compression, slower (~3.2:1)
/// - Level 19: Maximum compression, very slow (~3.5:1)
pub const DEFAULT_COMPRESSION_LEVEL: i32 = 3;

/// CompressionPipeline handles all compression operations
pub struct CompressionPipeline {
    compression_level: i32,
}

impl CompressionPipeline {
    /// Create a new CompressionPipeline with default settings
    pub fn new() -> Self {
        Self {
            compression_level: DEFAULT_COMPRESSION_LEVEL,
        }
    }

    /// Create a CompressionPipeline with custom compression level
    pub fn with_level(compression_level: i32) -> Self {
        Self { compression_level }
    }

    /// Compress data using Zstandard
    /// 
    /// Zstandard provides excellent balance between compression ratio (2.5-3.0:1)
    /// and decompression speed (1800-2100 MB/s).
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
}

impl Default for CompressionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Specialized compression for time-series data
/// 
/// This would integrate with specialized crates like `tms` or `tsink`
/// for Gorilla compression or SIMD-optimized delta encoding.
pub struct TimeSeriesCompressor;

impl TimeSeriesCompressor {
    /// Compress time-series data using specialized algorithms
    /// 
    /// In a full implementation, this would:
    /// 1. Apply delta-of-delta encoding to timestamps
    /// 2. Apply delta encoding + XOR to floating-point values
    /// 3. Use SIMD bit-packing for small integers
    /// 4. Achieve compression ratios up to 73:1 for metrics data
    pub fn compress_timeseries(&self, _data: &[f32]) -> Result<Vec<u8>> {
        // Placeholder for specialized time-series compression
        // In production, this would use `tms` or `tsink` crates
        todo!("Implement specialized time-series compression")
    }

    /// Decompress time-series data
    pub fn decompress_timeseries(&self, _compressed: &[u8]) -> Result<Vec<f32>> {
        // Placeholder for specialized time-series decompression
        todo!("Implement specialized time-series decompression")
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
        let data = vec![0u8; 10000]; // Highly compressible data
        
        let (compressed, stats) = pipeline.compress_with_stats(&data).unwrap();
        
        assert!(stats.compression_ratio > 1.0);
        assert_eq!(stats.original_size, 10000);
        assert_eq!(stats.compressed_size, compressed.len() as u64);
        assert!(stats.compression_time_ms > 0);
    }

    #[test]
    fn test_different_compression_levels() {
        let data = b"Test data for compression level comparison";
        
        let pipeline_fast = CompressionPipeline::with_level(1);
        let pipeline_balanced = CompressionPipeline::with_level(3);
        let pipeline_max = CompressionPipeline::with_level(19);
        
        let compressed_fast = pipeline_fast.compress(data).unwrap();
        let compressed_balanced = pipeline_balanced.compress(data).unwrap();
        let compressed_max = pipeline_max.compress(data).unwrap();
        
        // Higher compression level should result in smaller size
        assert!(compressed_max.len() <= compressed_balanced.len());
        assert!(compressed_balanced.len() <= compressed_fast.len());
        
        // All should decompress to the same original data
        assert_eq!(pipeline_fast.decompress(&compressed_fast).unwrap(), data);
        assert_eq!(pipeline_balanced.decompress(&compressed_balanced).unwrap(), data);
        assert_eq!(pipeline_max.decompress(&compressed_max).unwrap(), data);
    }
}
