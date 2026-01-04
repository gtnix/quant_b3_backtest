//! Artifact reader for OBFS

use anyhow::Result;
use memmap2::Mmap;
use rkyv::rancor::Error as RancorError;
use std::fs::File;
use std::sync::Arc;
use uuid::Uuid;

use crate::compression::CompressionPipeline;
use crate::integrity::IntegrityEngine;
use crate::store::MetadataStore;
use crate::types::{ArtifactLocation, ArtifactMetadata, BacktestArtifact, Metrics};
use crate::ObfsConfig;

/// ArtifactReader handles reading backtest artifacts from OBFS
pub struct ArtifactReader {
    config: ObfsConfig,
    integrity_engine: IntegrityEngine,
    compression_pipeline: CompressionPipeline,
    metadata_store: Arc<MetadataStore>,
}

impl ArtifactReader {
    /// Create a new ArtifactReader
    pub fn new(
        config: ObfsConfig,
        integrity_engine: IntegrityEngine,
        compression_pipeline: CompressionPipeline,
        metadata_store: Arc<MetadataStore>,
    ) -> Self {
        Self {
            config,
            integrity_engine,
            compression_pipeline,
            metadata_store,
        }
    }

    /// Read a backtest artifact from OBFS by UUID
    pub fn read_artifact(&self, uuid: Uuid) -> Result<BacktestArtifact> {
        // Step 1: Query metadata from LMDB
        let metadata = self
            .metadata_store
            .get(uuid)?
            .ok_or_else(|| anyhow::anyhow!("Artifact not found: {}", uuid))?;

        // Step 2: Memory-map the data file
        let compressed_data = self.mmap_artifact(&metadata.artifact_location)?;

        // Step 3: Validate integrity (XXH3 for fast validation)
        if self.config.enable_xxh3 {
            let computed_xxh3 = self.integrity_engine.compute_xxh3(&compressed_data);
            let stored_xxh3 = self
                .metadata_store
                .get_xxh3(uuid)?
                .ok_or_else(|| anyhow::anyhow!("XXH3 checksum not found for: {}", uuid))?;
            
            if computed_xxh3 != stored_xxh3 {
                return Err(anyhow::anyhow!(
                    "Integrity validation failed for {}: expected XXH3 {:016x}, got {:016x}",
                    uuid,
                    stored_xxh3,
                    computed_xxh3
                ));
            }
        }

        // Step 4: Decompress data
        let (decompressed, _stats) = self.compression_pipeline.decompress_with_stats(&compressed_data)?;

        // Step 5: Zero-copy deserialize using rkyv
        let artifact = self.deserialize_artifact(&decompressed)?;

        Ok(artifact)
    }

    /// Memory-map artifact data for zero-copy access
    fn mmap_artifact(&self, location: &ArtifactLocation) -> Result<Vec<u8>> {
        let file = File::open(&location.file_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let start = location.offset as usize;
        let end = start + location.size as usize;
        let data = &mmap[start..end];

        Ok(data.to_vec())
    }

    /// Zero-copy deserialize artifact using rkyv
    fn deserialize_artifact(&self, data: &[u8]) -> Result<BacktestArtifact> {
        let artifact: BacktestArtifact = rkyv::from_bytes::<BacktestArtifact, RancorError>(data)
            .map_err(|e| anyhow::anyhow!("Deserialization failed: {}", e))?;

        Ok(artifact)
    }

    /// Get metrics for an artifact without reading the full artifact
    pub fn get_metrics(&self, uuid: Uuid) -> Result<Metrics> {
        self.metadata_store
            .get_metrics(uuid)?
            .ok_or_else(|| anyhow::anyhow!("Artifact not found: {}", uuid))
    }

    /// Get artifact metadata
    pub fn get_metadata(&self, uuid: Uuid) -> Result<ArtifactMetadata> {
        self.metadata_store
            .get(uuid)?
            .ok_or_else(|| anyhow::anyhow!("Artifact not found: {}", uuid))
    }

    /// Check if an artifact exists
    pub fn exists(&self, uuid: Uuid) -> Result<bool> {
        self.metadata_store.exists(uuid)
    }

    /// List all artifact UUIDs
    pub fn list(&self) -> Result<Vec<Uuid>> {
        self.metadata_store.list_uuids()
    }

    /// Count total artifacts
    pub fn count(&self) -> Result<u64> {
        self.metadata_store.count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

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
    fn test_write_and_read_artifact() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ObfsConfig {
            root_path: temp_dir.path().to_str().unwrap().to_string(),
            ..Default::default()
        };

        let obfs = crate::Obfs::with_config(config.clone());
        obfs.initialize().unwrap();

        let mut writer = obfs.writer();
        let original_artifact = create_test_artifact();
        let uuid = original_artifact.uuid();
        writer.write_artifact(&original_artifact).unwrap();

        let reader = obfs.reader();
        let read_artifact = reader.read_artifact(uuid).unwrap();

        assert_eq!(read_artifact.uuid(), original_artifact.uuid());
        assert_eq!(
            read_artifact.metadata.strategy_id,
            original_artifact.metadata.strategy_id
        );
        assert_eq!(read_artifact.metrics.cagr, original_artifact.metrics.cagr);
    }

    #[test]
    fn test_get_metrics_only() {
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
        writer.write_artifact(&artifact).unwrap();

        let reader = obfs.reader();
        let metrics = reader.get_metrics(uuid).unwrap();
        assert_eq!(metrics.cagr, artifact.metrics.cagr);
    }

    #[test]
    fn test_exists_and_count() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ObfsConfig {
            root_path: temp_dir.path().to_str().unwrap().to_string(),
            ..Default::default()
        };

        let obfs = crate::Obfs::with_config(config.clone());
        obfs.initialize().unwrap();

        let reader = obfs.reader();
        let uuid = Uuid::new_v4();
        assert!(!reader.exists(uuid).unwrap());
        assert_eq!(reader.count().unwrap(), 0);

        let mut writer = obfs.writer();
        let artifact = create_test_artifact();
        let uuid = artifact.uuid();
        writer.write_artifact(&artifact).unwrap();

        assert!(reader.exists(uuid).unwrap());
        assert_eq!(reader.count().unwrap(), 1);
    }
}
