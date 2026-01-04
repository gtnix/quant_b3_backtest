//! Artifact writer for OBFS

use anyhow::Result;
use rkyv::rancor::Error as RancorError;
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;

use crate::compression::CompressionPipeline;
use crate::integrity::IntegrityEngine;
use crate::store::MetadataStore;
use crate::types::{ArtifactLocation, ArtifactMetadata, BacktestArtifact};
use crate::ObfsConfig;

/// ArtifactWriter handles writing backtest artifacts to OBFS
pub struct ArtifactWriter {
    config: ObfsConfig,
    integrity_engine: IntegrityEngine,
    compression_pipeline: CompressionPipeline,
    metadata_store: Arc<MetadataStore>,
    /// Current data file index for rotation
    current_file_index: u32,
}

impl ArtifactWriter {
    /// Create a new ArtifactWriter
    pub fn new(
        config: ObfsConfig,
        integrity_engine: IntegrityEngine,
        compression_pipeline: CompressionPipeline,
        metadata_store: Arc<MetadataStore>,
    ) -> Self {
        // Detect the current file index by scanning existing files
        let data_dir = Path::new(&config.root_path).join("data");
        let current_file_index = Self::detect_current_file_index(&data_dir);

        Self {
            config,
            integrity_engine,
            compression_pipeline,
            metadata_store,
            current_file_index,
        }
    }

    /// Detect the highest data file index in the data directory
    fn detect_current_file_index(data_dir: &Path) -> u32 {
        if !data_dir.exists() {
            return 0;
        }

        let mut max_index = 0u32;
        if let Ok(entries) = std::fs::read_dir(data_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                // Pattern: data_XXXX.obfs
                if name_str.starts_with("data_") && name_str.ends_with(".obfs") {
                    if let Some(num_str) = name_str.strip_prefix("data_").and_then(|s| s.strip_suffix(".obfs")) {
                        if let Ok(index) = num_str.parse::<u32>() {
                            max_index = max_index.max(index);
                        }
                    }
                }
            }
        }
        max_index
    }

    /// Write a backtest artifact to OBFS
    pub fn write_artifact(&mut self, artifact: &BacktestArtifact) -> Result<ArtifactMetadata> {
        // Step 1: Serialize artifact using rkyv
        let serialized = self.serialize_artifact(artifact)?;

        // Step 2: Compress serialized data
        let (compressed, _compression_stats) = self.compression_pipeline.compress_with_stats(&serialized)?;

        // Step 3: Compute integrity seals
        let seal = if self.config.enable_blake3 && self.config.enable_xxh3 {
            self.integrity_engine.create_seal(&compressed)
        } else {
            artifact.integrity.clone()
        };

        // Step 4: Write to data file (with automatic rotation if needed)
        let location = self.write_to_data_file(&compressed)?;

        // Step 5: Create metadata entry with both BLAKE3 and XXH3 for integrity
        let metadata = ArtifactMetadata {
            uuid: artifact.uuid(),
            artifact_location: location,
            blake3_hash: seal.artifact_blake3,
            xxh3_checksum: seal.block_xxh3,
            metrics: artifact.metrics.clone(),
            created_at: chrono::Utc::now().timestamp(),
        };

        // Step 6: Store in LMDB
        self.metadata_store.put(&metadata)?;

        // Step 7: Log to WAL for durability
        self.write_wal_entry(&metadata)?;

        Ok(metadata)
    }

    /// Serialize artifact using rkyv (zero-copy format)
    fn serialize_artifact(&self, artifact: &BacktestArtifact) -> Result<Vec<u8>> {
        let bytes = rkyv::to_bytes::<RancorError>(artifact)
            .map_err(|e| anyhow::anyhow!("Serialization failed: {}", e))?;
        Ok(bytes.to_vec())
    }

    /// Write compressed data to data file with automatic rotation
    fn write_to_data_file(&mut self, data: &[u8]) -> Result<ArtifactLocation> {
        let data_dir = Path::new(&self.config.root_path).join("data");
        std::fs::create_dir_all(&data_dir)?;

        // Check if current file needs rotation
        let file_path = self.get_current_data_file_path(&data_dir);
        let current_size = if file_path.exists() {
            std::fs::metadata(&file_path)?.len()
        } else {
            0
        };

        // Rotate if adding this data would exceed max_file_size
        if current_size > 0 && current_size + data.len() as u64 > self.config.max_file_size {
            self.current_file_index += 1;
        }

        let file_path = self.get_current_data_file_path(&data_dir);

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)?;

        let offset = file.seek(SeekFrom::End(0))?;
        file.write_all(data)?;
        file.sync_all()?;

        Ok(ArtifactLocation {
            file_path: file_path.to_str().unwrap().to_string(),
            offset,
            size: data.len() as u64,
        })
    }

    /// Get the path to the current data file based on index
    fn get_current_data_file_path(&self, data_dir: &Path) -> std::path::PathBuf {
        data_dir.join(format!("data_{:04}.obfs", self.current_file_index))
    }

    /// Get the current file index (useful for testing)
    pub fn current_file_index(&self) -> u32 {
        self.current_file_index
    }

    /// Write entry to Write-Ahead Log (WAL)
    fn write_wal_entry(&self, metadata: &ArtifactMetadata) -> Result<()> {
        let wal_dir = Path::new(&self.config.root_path).join("wal");
        std::fs::create_dir_all(&wal_dir)?;
        let wal_file = wal_dir.join("segment_0000.wal");

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(wal_file)?;

        let entry = format!(
            "{},{},{}\n",
            metadata.uuid, metadata.artifact_location.offset, metadata.artifact_location.size
        );

        file.write_all(entry.as_bytes())?;
        file.sync_all()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use uuid::Uuid;

    fn create_test_artifact() -> BacktestArtifact {
        let uuid = Uuid::new_v4();
        BacktestArtifact {
            uuid_bytes: *uuid.as_bytes(),
            metadata: Metadata {
                strategy_id: "test_strategy".to_string(),
                strategy_version: "1.0.0".to_string(),
                run_id: "test_run".to_string(),
                timestamp: chrono::Utc::now().timestamp(),
                universe: "B3_IBOV".to_string(),
                start_date: "2020-01-01".to_string(),
                end_date: "2024-12-31".to_string(),
                initial_capital: 1_000_000.0,
                mode: "fast".to_string(),
            },
            metrics: Metrics {
                cagr: 0.15,
                volatility: 0.20,
                sharpe_ratio: 0.75,
                sortino_ratio: 1.10,
                max_drawdown: -0.25,
                max_drawdown_duration_days: 180,
                hit_rate: 0.55,
                profit_factor: 1.5,
                turnover_annual: 2.0,
                total_trades: 500,
            },
            timeseries_ref: TimeseriesReference {
                parquet_file: "timeseries_0000.parquet".to_string(),
                row_group: 0,
                start_row: 0,
                num_rows: 1245,
            },
            trace: vec![TraceEvent {
                timestamp: chrono::Utc::now().timestamp(),
                event_type: "start".to_string(),
                message: "Backtest started".to_string(),
            }],
            integrity: IntegritySeal::default(),
        }
    }

    #[test]
    fn test_write_artifact() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ObfsConfig {
            root_path: temp_dir.path().to_str().unwrap().to_string(),
            ..Default::default()
        };

        let obfs = crate::Obfs::with_config(config.clone());
        obfs.initialize().unwrap();

        let mut writer = obfs.writer();
        let artifact = create_test_artifact();
        let uuid = artifact.uuid();
        let metadata = writer.write_artifact(&artifact).unwrap();

        assert_eq!(metadata.uuid, uuid);
        assert!(metadata.artifact_location.size > 0);

        let data_file = Path::new(&metadata.artifact_location.file_path);
        assert!(data_file.exists());
    }

    #[test]
    fn test_file_rotation() {
        let temp_dir = tempfile::tempdir().unwrap();
        // Set very small max_file_size to force rotation
        let config = ObfsConfig {
            root_path: temp_dir.path().to_str().unwrap().to_string(),
            max_file_size: 500, // 500 bytes - will trigger rotation quickly
            ..Default::default()
        };

        let obfs = crate::Obfs::with_config(config.clone());
        obfs.initialize().unwrap();

        let mut writer = obfs.writer();
        
        // Write multiple artifacts to trigger rotation
        let mut file_paths = std::collections::HashSet::new();
        for _ in 0..5 {
            let artifact = create_test_artifact();
            let metadata = writer.write_artifact(&artifact).unwrap();
            file_paths.insert(metadata.artifact_location.file_path);
        }

        // Should have created multiple files due to rotation
        assert!(
            file_paths.len() > 1,
            "Expected multiple files due to rotation, got: {:?}",
            file_paths
        );

        // Verify files exist
        let data_dir = temp_dir.path().join("data");
        let mut obfs_files: Vec<_> = std::fs::read_dir(&data_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "obfs"))
            .collect();
        obfs_files.sort_by_key(|e| e.file_name());

        println!("Created {} data files:", obfs_files.len());
        for file in &obfs_files {
            let size = std::fs::metadata(file.path()).unwrap().len();
            println!("  {:?}: {} bytes", file.file_name(), size);
        }

        assert!(obfs_files.len() > 1, "Expected multiple .obfs files");
    }

    #[test]
    fn test_file_index_detection() {
        let temp_dir = tempfile::tempdir().unwrap();
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        // Create some dummy files with different indices
        std::fs::write(data_dir.join("data_0000.obfs"), b"test").unwrap();
        std::fs::write(data_dir.join("data_0003.obfs"), b"test").unwrap();
        std::fs::write(data_dir.join("data_0007.obfs"), b"test").unwrap();

        let detected = ArtifactWriter::detect_current_file_index(&data_dir);
        assert_eq!(detected, 7, "Should detect highest index as 7");
    }
}
