// writer.rs - Artifact writer for OBFS

use anyhow::Result;
use rkyv::ser::{serializers::AllocSerializer, Serializer};
use std::fs::{File, OpenOptions};
use std::io::{Write, Seek, SeekFrom};
use std::path::Path;

use crate::types::{BacktestArtifact, ArtifactLocation, ArtifactMetadata};
use crate::{ObfsConfig, IntegrityEngine, CompressionPipeline};

/// ArtifactWriter handles writing backtest artifacts to OBFS
pub struct ArtifactWriter {
    config: ObfsConfig,
    integrity_engine: IntegrityEngine,
    compression_pipeline: CompressionPipeline,
}

impl ArtifactWriter {
    /// Create a new ArtifactWriter
    pub fn new(
        config: ObfsConfig,
        integrity_engine: IntegrityEngine,
        compression_pipeline: CompressionPipeline,
    ) -> Self {
        Self {
            config,
            integrity_engine,
            compression_pipeline,
        }
    }

    /// Write a backtest artifact to OBFS
    /// 
    /// This implements the complete write path:
    /// 1. Serialize artifact using rkyv (zero-copy format)
    /// 2. Compress serialized data using Zstandard
    /// 3. Compute integrity seals (XXH3 + BLAKE3)
    /// 4. Write to data file
    /// 5. Update metadata store (LMDB)
    /// 6. Log to WAL for durability
    pub fn write_artifact(&mut self, artifact: &BacktestArtifact) -> Result<ArtifactMetadata> {
        // Step 1: Serialize artifact using rkyv
        let serialized = self.serialize_artifact(artifact)?;
        
        // Step 2: Compress serialized data
        let (compressed, compression_stats) = self.compression_pipeline
            .compress_with_stats(&serialized)?;
        
        println!("Compression stats:");
        println!("  Original size: {} bytes", compression_stats.original_size);
        println!("  Compressed size: {} bytes", compression_stats.compressed_size);
        println!("  Compression ratio: {:.2}x", compression_stats.compression_ratio);
        println!("  Compression time: {} ms", compression_stats.compression_time_ms);
        
        // Step 3: Compute integrity seals
        let seal = if self.config.enable_blake3 && self.config.enable_xxh3 {
            self.integrity_engine.create_seal(&compressed)
        } else {
            artifact.integrity.clone()
        };
        
        // Step 4: Write to data file
        let location = self.write_to_data_file(&compressed)?;
        
        // Step 5: Create metadata entry
        let metadata = ArtifactMetadata {
            uuid: artifact.uuid,
            artifact_location: location,
            blake3_hash: seal.artifact_blake3,
            metrics: artifact.metrics.clone(),
            created_at: chrono::Utc::now().timestamp(),
        };
        
        // Step 6: Update metadata store (LMDB)
        // In a full implementation, this would write to LMDB
        self.write_metadata(&metadata)?;
        
        // Step 7: Log to WAL for durability
        // In a full implementation, this would write to WAL
        self.write_wal_entry(&metadata)?;
        
        Ok(metadata)
    }

    /// Serialize artifact using rkyv (zero-copy format)
    fn serialize_artifact(&self, artifact: &BacktestArtifact) -> Result<Vec<u8>> {
        let mut serializer = AllocSerializer::<256>::default();
        serializer.serialize_value(artifact)
            .map_err(|e| anyhow::anyhow!("Serialization failed: {}", e))?;
        Ok(serializer.into_serializer().into_inner().to_vec())
    }

    /// Write compressed data to data file
    fn write_to_data_file(&self, data: &[u8]) -> Result<ArtifactLocation> {
        let data_dir = Path::new(&self.config.root_path).join("data");
        let file_path = data_dir.join("data_0000.obfs");
        
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)?;
        
        // Get current file position (offset)
        let offset = file.seek(SeekFrom::End(0))?;
        
        // Write data
        file.write_all(data)?;
        file.sync_all()?;
        
        Ok(ArtifactLocation {
            file_path: file_path.to_str().unwrap().to_string(),
            offset,
            size: data.len() as u64,
        })
    }

    /// Write metadata to LMDB
    /// 
    /// In a full implementation, this would:
    /// 1. Open LMDB transaction
    /// 2. Write uuid -> artifact_location mapping
    /// 3. Write uuid -> blake3_hash mapping
    /// 4. Write uuid -> metrics mapping
    /// 5. Commit transaction
    fn write_metadata(&self, metadata: &ArtifactMetadata) -> Result<()> {
        // Placeholder for LMDB write
        let metadata_json = serde_json::to_string_pretty(metadata)?;
        let metadata_file = Path::new(&self.config.root_path)
            .join("data")
            .join(format!("{}.metadata.json", metadata.uuid));
        
        std::fs::write(metadata_file, metadata_json)?;
        
        Ok(())
    }

    /// Write entry to Write-Ahead Log (WAL)
    /// 
    /// In a full implementation, this would use `walrus-rust` crate
    /// to ensure atomic durability with 1M ops/sec throughput.
    fn write_wal_entry(&self, metadata: &ArtifactMetadata) -> Result<()> {
        // Placeholder for WAL write
        let wal_dir = Path::new(&self.config.root_path).join("wal");
        let wal_file = wal_dir.join("segment_0000.wal");
        
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(wal_file)?;
        
        let entry = format!("{},{},{}\n", 
            metadata.uuid,
            metadata.artifact_location.offset,
            metadata.artifact_location.size
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
        BacktestArtifact {
            uuid: Uuid::new_v4(),
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
            trace: vec![
                TraceEvent {
                    timestamp: chrono::Utc::now().timestamp(),
                    event_type: "start".to_string(),
                    message: "Backtest started".to_string(),
                },
            ],
            integrity: IntegritySeal {
                block_xxh3: 0,
                artifact_blake3: [0u8; 32],
                sealed_at: chrono::Utc::now().timestamp(),
            },
        }
    }

    #[test]
    fn test_write_artifact() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ObfsConfig {
            root_path: temp_dir.path().to_str().unwrap().to_string(),
            ..Default::default()
        };

        // Initialize OBFS
        let obfs = crate::Obfs::with_config(config.clone());
        obfs.initialize().unwrap();

        // Create writer
        let mut writer = ArtifactWriter::new(
            config,
            IntegrityEngine::new(),
            CompressionPipeline::new(),
        );

        // Write artifact
        let artifact = create_test_artifact();
        let metadata = writer.write_artifact(&artifact).unwrap();

        // Verify metadata
        assert_eq!(metadata.uuid, artifact.uuid);
        assert!(metadata.artifact_location.size > 0);
        
        // Verify file was created
        let data_file = Path::new(&metadata.artifact_location.file_path);
        assert!(data_file.exists());
    }
}
