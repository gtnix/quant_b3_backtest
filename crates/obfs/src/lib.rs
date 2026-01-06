//! OBFS - Optimized Binary File System for high-performance backtest artifact storage
//!
//! This crate provides a high-performance storage system for backtest artifacts with:
//! - Zero-copy serialization using rkyv
//! - Dual-hashing integrity (XXH3 + BLAKE3)
//! - Zstd compression (2.5-3.0x ratio for standard, ~10-15x for UltraCompressor)
//! - LMDB-based metadata store for fast O(1) lookups
//!
//! ## Bundle Types (using OBFS infrastructure)
//!
//! The following bundle types leverage OBFS compression across the codebase:
//!
//! | Bundle | Location | Compression | Use Case |
//! |--------|----------|-------------|----------|
//! | `ReportBundle` | obfs | ~10x | SCG candidates/configs |
//! | `ValidationBundle` | combiner_engine | ~10x | WFA/PBO/Stress reports |
//! | `ResultBundle` | backtester_reports | ~8x | Backtest results + NAV |
//! | `IntegrityBundle` | backtester_intelligence | ~15x | Data integrity audits |
//! | Market data reports | market_data | ~15x | Coverage/Freshness |
//!
//! All bundles use UltraCompressor (Zstd level 19 + LDM + checksum).

pub mod compression;
pub mod consolidator;
pub mod integrity;
pub mod pending_store;
pub mod reader;
pub mod report_bundle;
pub mod store;
pub mod timeseries;
pub mod types;
pub mod writer;
pub mod adapters;

pub use compression::{
    CompressionPipeline, CompressionStrategy, DeltaEncoder, TimeSeriesCompressor, ColumnarCompressor,
    UltraCompressor, ULTRA_COMPRESSION_LEVEL as COMPRESSION_LEVEL_ULTRA,
};
pub use consolidator::{Consolidator, ConsolidationStats, consolidate};
pub use integrity::IntegrityEngine;
pub use reader::ArtifactReader;
pub use report_bundle::{
    ReportBundle, ReportBundleReader, ReportBundleWriter, BundleStats, CandidateEntry,
    ULTRA_COMPRESSION_LEVEL,
};
pub use store::MetadataStore;
pub use pending_store::{PendingArtifact, PendingStore, TimeseriesPoint as PendingTimeseriesPoint};
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

// =============================================================================
// STANDARDIZED ARTIFACT HELPERS
// =============================================================================

use std::sync::OnceLock;

/// Global compression pipeline for artifact helpers (lazily initialized).
static GLOBAL_PIPELINE: OnceLock<CompressionPipeline> = OnceLock::new();

/// Get the global compression pipeline.
fn global_pipeline() -> &'static CompressionPipeline {
    GLOBAL_PIPELINE.get_or_init(|| CompressionPipeline::with_level(3))
}

/// Write any serializable data to a file with OBFS compression.
///
/// This is the primary helper for writing artifacts across all modules.
/// It automatically handles JSON serialization and Zstd compression.
///
/// # Arguments
/// * `path` - Output file path (will have .obfs extension)
/// * `data` - Any type that implements Serialize
///
/// # Returns
/// * `Ok(CompressionStats)` - Compression statistics
/// * `Err` - If serialization or I/O fails
///
/// # Example
/// ```ignore
/// use obfs::write_artifact;
/// 
/// let report = MyReport { ... };
/// write_artifact("output/report", &report)?;
/// // Creates: output/report.obfs
/// ```
pub fn write_artifact<T: serde::Serialize>(
    path: impl AsRef<Path>,
    data: &T,
) -> anyhow::Result<CompressionStats> {
    let path = path.as_ref();
    let path_with_ext = if path.extension().map_or(true, |e| e != "obfs") {
        path.with_extension("obfs")
    } else {
        path.to_path_buf()
    };
    
    // Ensure parent directory exists
    if let Some(parent) = path_with_ext.parent() {
        std::fs::create_dir_all(parent)?;
    }
    
    let json = serde_json::to_vec(data)?;
    let (compressed, stats) = global_pipeline().compress_with_stats(&json)?;
    std::fs::write(&path_with_ext, compressed)?;
    
    Ok(stats)
}

/// Read any deserializable data from an OBFS-compressed file.
///
/// # Arguments
/// * `path` - Input file path (.obfs extension optional)
///
/// # Returns
/// * `Ok(T)` - Deserialized data
/// * `Err` - If I/O, decompression, or deserialization fails
pub fn read_artifact<T: serde::de::DeserializeOwned>(
    path: impl AsRef<Path>,
) -> anyhow::Result<T> {
    let path = path.as_ref();
    let path_with_ext = if path.extension().map_or(true, |e| e != "obfs") {
        path.with_extension("obfs")
    } else {
        path.to_path_buf()
    };
    
    let compressed = std::fs::read(&path_with_ext)?;
    let decompressed = global_pipeline().decompress(&compressed)?;
    let data: T = serde_json::from_slice(&decompressed)?;
    
    Ok(data)
}

/// Write JSON data with optional OBFS compression based on format flag.
///
/// This helper maintains backward compatibility with legacy JSON output
/// while allowing a smooth transition to OBFS.
///
/// # Arguments
/// * `base_path` - Base path without extension
/// * `name` - File name without extension
/// * `data` - Serializable data
/// * `use_obfs` - If true, writes .obfs; if false, writes .json
pub fn write_artifact_conditional<T: serde::Serialize>(
    base_path: impl AsRef<Path>,
    name: &str,
    data: &T,
    use_obfs: bool,
) -> anyhow::Result<()> {
    let base = base_path.as_ref();
    
    if use_obfs {
        let path = base.join(format!("{}.obfs", name));
        write_artifact(&path, data)?;
    } else {
        let path = base.join(format!("{}.json", name));
        let json = serde_json::to_string_pretty(data)?;
        std::fs::write(path, json)?;
    }
    
    Ok(())
}

/// Batch write multiple artifacts to a directory.
///
/// # Arguments
/// * `dir` - Output directory
/// * `artifacts` - List of (name, data) pairs
pub fn write_artifacts_batch<T: serde::Serialize>(
    dir: impl AsRef<Path>,
    artifacts: &[(&str, &T)],
) -> anyhow::Result<Vec<CompressionStats>> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir)?;
    
    let mut all_stats = Vec::with_capacity(artifacts.len());
    for (name, data) in artifacts {
        let path = dir.join(format!("{}.obfs", name));
        let stats = write_artifact(&path, data)?;
        all_stats.push(stats);
    }
    
    Ok(all_stats)
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

