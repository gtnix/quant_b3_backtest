//! OBFS - Optimized Binary File System for high-performance backtest artifact storage
//!
//! This crate provides a high-performance storage system for backtest artifacts with:
//! - Zero-copy serialization using rkyv
//! - Dual-hashing integrity (XXH3 + BLAKE3)
//! - Zstd compression (2.5-3.0x ratio)
//! - LMDB-based metadata store for fast lookups

pub mod compression;
pub mod integrity;
pub mod reader;
pub mod store;
pub mod timeseries;
pub mod types;
pub mod writer;
pub mod adapters;

pub use compression::{
    CompressionPipeline, CompressionStrategy, DeltaEncoder, TimeSeriesCompressor, ColumnarCompressor,
};
pub use integrity::IntegrityEngine;
pub use reader::ArtifactReader;
pub use store::MetadataStore;
pub use timeseries::{TimeSeriesStore, TimeSeriesPoint, TimeSeriesRef, ParquetStats};
pub use types::*;
pub use writer::ArtifactWriter;

use std::path::Path;
use std::sync::Arc;

/// OBFS Configuration
#[derive(Debug, Clone)]
pub struct ObfsConfig {
    /// Root directory for OBFS storage
    pub root_path: String,
    /// Compression level (1-19 for Zstandard)
    pub compression_level: i32,
    /// Enable BLAKE3 hashing for all artifacts
    pub enable_blake3: bool,
    /// Enable XXH3 checksums for fast validation
    pub enable_xxh3: bool,
    /// Maximum size of a single data file before rotation (in bytes)
    pub max_file_size: u64,
}

impl Default for ObfsConfig {
    fn default() -> Self {
        Self {
            root_path: "./artifacts".to_string(),
            compression_level: 3,
            enable_blake3: true,
            enable_xxh3: true,
            max_file_size: 1024 * 1024 * 1024, // 1 GB
        }
    }
}

/// OBFS (Optimized Binary File System) main interface
pub struct Obfs {
    config: ObfsConfig,
    integrity_engine: IntegrityEngine,
    compression_pipeline: CompressionPipeline,
    metadata_store: Arc<MetadataStore>,
    timeseries_store: TimeSeriesStore,
}

impl Obfs {
    /// Create a new OBFS instance with default configuration
    pub fn new() -> Self {
        Self::with_config(ObfsConfig::default())
    }

    /// Create a new OBFS instance with custom configuration
    pub fn with_config(config: ObfsConfig) -> Self {
        let integrity_engine = IntegrityEngine::new();
        let compression_pipeline = CompressionPipeline::with_level(config.compression_level);

        // Initialize LMDB store
        let lmdb_path = Path::new(&config.root_path).join("lmdb");
        std::fs::create_dir_all(&lmdb_path).expect("Failed to create LMDB directory");
        let metadata_store =
            Arc::new(MetadataStore::open(&lmdb_path).expect("Failed to open metadata store"));

        // Initialize TimeSeriesStore
        let root_path = Path::new(&config.root_path);
        let timeseries_store =
            TimeSeriesStore::new(root_path).expect("Failed to initialize time-series store");

        Self {
            config,
            integrity_engine,
            compression_pipeline,
            metadata_store,
            timeseries_store,
        }
    }

    /// Get a writer for creating new artifacts
    pub fn writer(&self) -> ArtifactWriter {
        ArtifactWriter::new(
            self.config.clone(),
            self.integrity_engine.clone(),
            self.compression_pipeline.clone(),
            Arc::clone(&self.metadata_store),
        )
    }

    /// Get a reader for querying artifacts
    pub fn reader(&self) -> ArtifactReader {
        ArtifactReader::new(
            self.config.clone(),
            self.integrity_engine.clone(),
            self.compression_pipeline.clone(),
            Arc::clone(&self.metadata_store),
        )
    }

    /// Initialize the OBFS storage structure
    pub fn initialize(&self) -> anyhow::Result<()> {
        let root = Path::new(&self.config.root_path);

        std::fs::create_dir_all(root.join("data"))?;
        std::fs::create_dir_all(root.join("wal"))?;
        std::fs::create_dir_all(root.join("integrity"))?;
        std::fs::create_dir_all(root.join("lmdb"))?;

        Ok(())
    }

    /// Get the configuration
    pub fn config(&self) -> &ObfsConfig {
        &self.config
    }

    /// Get the compression pipeline for direct compression operations
    pub fn compression_pipeline(&self) -> &CompressionPipeline {
        &self.compression_pipeline
    }

    /// Sync all data to disk
    pub fn sync(&self) -> anyhow::Result<()> {
        self.metadata_store.sync()
    }

    /// Get a mutable reference to the time-series store
    pub fn timeseries_store_mut(&mut self) -> &mut TimeSeriesStore {
        &mut self.timeseries_store
    }

    /// Get an immutable reference to the time-series store
    pub fn timeseries_store(&self) -> &TimeSeriesStore {
        &self.timeseries_store
    }
}

impl Default for Obfs {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obfs_initialization() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ObfsConfig {
            root_path: temp_dir.path().to_str().unwrap().to_string(),
            ..Default::default()
        };

        let obfs = Obfs::with_config(config);
        assert!(obfs.initialize().is_ok());

        assert!(temp_dir.path().join("data").exists());
        assert!(temp_dir.path().join("wal").exists());
        assert!(temp_dir.path().join("integrity").exists());
        assert!(temp_dir.path().join("lmdb").exists());
    }

    #[test]
    fn test_obfs_default_config() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ObfsConfig {
            root_path: temp_dir.path().to_str().unwrap().to_string(),
            ..Default::default()
        };
        let obfs = Obfs::with_config(config);
        assert_eq!(obfs.config.compression_level, 3);
        assert!(obfs.config.enable_blake3);
        assert!(obfs.config.enable_xxh3);
    }
}

