// lib.rs - Main library for OBFS (Optimized Binary File System)

pub mod types;
pub mod integrity;
pub mod compression;
pub mod writer;
pub mod reader;

pub use types::*;
pub use integrity::IntegrityEngine;
pub use compression::CompressionPipeline;
pub use writer::ArtifactWriter;
pub use reader::ArtifactReader;

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

        Self {
            config,
            integrity_engine,
            compression_pipeline,
        }
    }

    /// Get a writer for creating new artifacts
    pub fn writer(&self) -> ArtifactWriter {
        ArtifactWriter::new(
            self.config.clone(),
            self.integrity_engine.clone(),
            self.compression_pipeline.clone(),
        )
    }

    /// Get a reader for querying artifacts
    pub fn reader(&self) -> ArtifactReader {
        ArtifactReader::new(
            self.config.clone(),
            self.integrity_engine.clone(),
            self.compression_pipeline.clone(),
        )
    }

    /// Initialize the OBFS storage structure
    pub fn initialize(&self) -> anyhow::Result<()> {
        use std::fs;
        use std::path::Path;

        let root = Path::new(&self.config.root_path);
        
        // Create directory structure
        fs::create_dir_all(root.join("data"))?;
        fs::create_dir_all(root.join("wal"))?;
        fs::create_dir_all(root.join("integrity"))?;

        println!("OBFS initialized at: {}", self.config.root_path);
        println!("  - Compression level: {}", self.config.compression_level);
        println!("  - BLAKE3 enabled: {}", self.config.enable_blake3);
        println!("  - XXH3 enabled: {}", self.config.enable_xxh3);

        Ok(())
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

        // Verify directory structure
        assert!(temp_dir.path().join("data").exists());
        assert!(temp_dir.path().join("wal").exists());
        assert!(temp_dir.path().join("integrity").exists());
    }

    #[test]
    fn test_obfs_default_config() {
        let obfs = Obfs::new();
        assert_eq!(obfs.config.compression_level, 3);
        assert!(obfs.config.enable_blake3);
        assert!(obfs.config.enable_xxh3);
    }
}
