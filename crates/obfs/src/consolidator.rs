//! Consolidator - Single-thread merge of pending artifacts into Parquet + LMDB.
//!
//! After evolution completes with parallel workers writing isolated `.obfs` files,
//! the Consolidator merges all pending artifacts into:
//! - `timeseries.parquet` - Columnar storage with Zstd compression
//! - `index.lmdb` - O(1) lookup by UUID
//!
//! This is a single-threaded operation to avoid Parquet concurrent write issues.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{ArrayRef, Float32Array, StringArray, UInt16Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use crate::pending_store::{PendingArtifact, PendingStore};
use crate::store::MetadataStore;
use crate::types::{ArtifactLocation, ArtifactMetadata};

/// Statistics from a consolidation run.
#[derive(Debug, Clone, Default)]
pub struct ConsolidationStats {
    /// Number of pending artifacts processed.
    pub artifacts_processed: usize,
    /// Number of timeseries rows written.
    pub timeseries_rows: usize,
    /// Size of output Parquet file in bytes.
    pub parquet_size_bytes: u64,
    /// Size of LMDB index in bytes.
    pub lmdb_size_bytes: u64,
    /// Total time in milliseconds.
    pub duration_ms: u64,
    /// Compression ratio achieved.
    pub compression_ratio: f64,
}

/// Consolidator for merging pending artifacts into Parquet.
pub struct Consolidator {
    pending_dir: PathBuf,
    output_dir: PathBuf,
}

impl Consolidator {
    /// Create a new consolidator.
    pub fn new(pending_dir: impl Into<PathBuf>, output_dir: impl Into<PathBuf>) -> Self {
        Self {
            pending_dir: pending_dir.into(),
            output_dir: output_dir.into(),
        }
    }

    /// Consolidate all pending artifacts into Parquet + LMDB.
    /// 
    /// This is a single-threaded operation to ensure no concurrent writes.
    pub fn consolidate(&self) -> Result<ConsolidationStats> {
        let start = std::time::Instant::now();
        let mut stats = ConsolidationStats::default();

        // Create output directories
        fs::create_dir_all(&self.output_dir)?;
        let data_dir = self.output_dir.join("data");
        let lmdb_dir = self.output_dir.join("lmdb");
        fs::create_dir_all(&data_dir)?;
        fs::create_dir_all(&lmdb_dir)?;

        // Open pending store
        let pending_store = PendingStore::new(&self.pending_dir)?;
        let pending_uuids = pending_store.list_pending()?;

        if pending_uuids.is_empty() {
            return Ok(stats);
        }

        // Collect all artifacts
        let mut artifacts: Vec<PendingArtifact> = Vec::with_capacity(pending_uuids.len());
        for uuid in &pending_uuids {
            match pending_store.read_pending(*uuid) {
                Ok(artifact) => artifacts.push(artifact),
                Err(_) => continue,
            }
        }

        stats.artifacts_processed = artifacts.len();

        // Write timeseries to Parquet
        let parquet_path = data_dir.join("timeseries.parquet");
        let ts_rows = self.write_timeseries_parquet(&artifacts, &parquet_path)?;
        stats.timeseries_rows = ts_rows;

        if parquet_path.exists() {
            stats.parquet_size_bytes = fs::metadata(&parquet_path)?.len();
        }

        // Index in LMDB
        let metadata_store = MetadataStore::open(&lmdb_dir)?;
        self.index_artifacts(&artifacts, &parquet_path, &metadata_store)?;

        // Calculate LMDB size
        if let Ok(entries) = fs::read_dir(&lmdb_dir) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata() {
                    stats.lmdb_size_bytes += meta.len();
                }
            }
        }

        // Clean up pending files
        for artifact in &artifacts {
            let _ = pending_store.remove_pending(artifact.run_id);
        }

        stats.duration_ms = start.elapsed().as_millis() as u64;

        // Calculate compression ratio
        let original_estimate = stats.artifacts_processed * 60_000; // ~60KB per legacy run
        if stats.parquet_size_bytes > 0 {
            stats.compression_ratio = original_estimate as f64 / stats.parquet_size_bytes as f64;
        }

        Ok(stats)
    }

    /// Write all timeseries data to a single Parquet file.
    fn write_timeseries_parquet(
        &self,
        artifacts: &[PendingArtifact],
        output_path: &Path,
    ) -> Result<usize> {
        // Collect all timeseries points with UUIDs
        let mut all_uuids: Vec<String> = Vec::new();
        let mut all_date_offsets: Vec<u16> = Vec::new();
        let mut all_equity: Vec<f32> = Vec::new();
        let mut all_drawdown: Vec<f32> = Vec::new();
        let mut all_exposure: Vec<f32> = Vec::new();

        for artifact in artifacts {
            let uuid_str = artifact.run_id.to_string();
            for point in &artifact.timeseries {
                all_uuids.push(uuid_str.clone());
                all_date_offsets.push(point.date_offset);
                all_equity.push(point.equity);
                all_drawdown.push(point.drawdown);
                all_exposure.push(point.exposure);
            }
        }

        let total_rows = all_uuids.len();
        if total_rows == 0 {
            return Ok(0);
        }

        // Create Arrow arrays
        let uuid_array: ArrayRef = Arc::new(StringArray::from(all_uuids));
        let date_offset_array: ArrayRef = Arc::new(UInt16Array::from(all_date_offsets));
        let equity_array: ArrayRef = Arc::new(Float32Array::from(all_equity));
        let drawdown_array: ArrayRef = Arc::new(Float32Array::from(all_drawdown));
        let exposure_array: ArrayRef = Arc::new(Float32Array::from(all_exposure));

        // Define schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("backtest_uuid", DataType::Utf8, false),
            Field::new("date_offset", DataType::UInt16, false),
            Field::new("equity", DataType::Float32, false),
            Field::new("drawdown", DataType::Float32, false),
            Field::new("exposure", DataType::Float32, false),
        ]));

        // Create RecordBatch
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                uuid_array,
                date_offset_array,
                equity_array,
                drawdown_array,
                exposure_array,
            ],
        )?;

        // Write Parquet with Zstd compression
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(Default::default()))
            .set_dictionary_enabled(true)
            .set_max_row_group_size(100_000)
            .build();

        let file = fs::File::create(output_path)
            .with_context(|| format!("Failed to create Parquet file: {:?}", output_path))?;

        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;

        Ok(total_rows)
    }

    /// Index artifacts in LMDB for O(1) lookup.
    fn index_artifacts(
        &self,
        artifacts: &[PendingArtifact],
        parquet_path: &Path,
        store: &MetadataStore,
    ) -> Result<()> {
        let parquet_path_str = parquet_path.to_string_lossy().to_string();
        let mut row_offset: u64 = 0;

        for artifact in artifacts {
            let ts_count = artifact.timeseries.len() as u64;

            let metadata = ArtifactMetadata {
                uuid: artifact.run_id,
                artifact_location: ArtifactLocation {
                    file_path: parquet_path_str.clone(),
                    offset: row_offset,
                    size: ts_count,
                },
                blake3_hash: [0u8; 32], // Could compute if needed
                xxh3_checksum: 0,
                metrics: artifact.metrics.clone(),
                created_at: chrono::Utc::now().timestamp(),
            };

            store.put(&metadata)?;
            row_offset += ts_count;
        }

        store.sync()?;
        Ok(())
    }
}

/// Convenience function to consolidate pending artifacts.
pub fn consolidate(pending_dir: impl Into<PathBuf>, output_dir: impl Into<PathBuf>) -> Result<ConsolidationStats> {
    let consolidator = Consolidator::new(pending_dir, output_dir);
    consolidator.consolidate()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Metadata;
    use tempfile::tempdir;

    fn create_test_artifacts(pending_dir: &Path, count: usize) -> Vec<Uuid> {
        let store = PendingStore::new(pending_dir).unwrap();
        let mut uuids = Vec::new();

        for i in 0..count {
            let uuid = Uuid::new_v4();
            let artifact = PendingArtifact::new(
                uuid,
                Metadata {
                    strategy_id: format!("strategy_{}", i),
                    strategy_version: "1.0.0".into(),
                    run_id: uuid.to_string(),
                    timestamp: 1704067200 + i as i64,
                    universe: "B3_IBOV".into(),
                    start_date: "2020-01-01".into(),
                    end_date: "2024-01-01".into(),
                    initial_capital: 1_000_000.0,
                    mode: "standard".into(),
                },
                crate::types::Metrics {
                    cagr: 0.10 + i as f64 * 0.01,
                    volatility: 0.20,
                    sharpe_ratio: 0.5 + i as f64 * 0.1,
                    sortino_ratio: 1.0,
                    max_drawdown: -0.12,
                    max_drawdown_duration_days: 30,
                    hit_rate: 0.55,
                    profit_factor: 1.5,
                    turnover_annual: 2.5,
                    total_trades: 100 + i as i32,
                },
            )
            .with_timeseries(
                (0..100)
                    .map(|j| TimeseriesPoint {
                        date_offset: j as u16,
                        equity: 1_000_000.0 + j as f32 * 100.0,
                        drawdown: -0.001 * j as f32,
                        exposure: 0.95,
                    })
                    .collect(),
            );

            store.write_pending(&artifact).unwrap();
            uuids.push(uuid);
        }

        uuids
    }

    #[test]
    fn test_consolidate_single() {
        let temp = tempdir().unwrap();
        let pending_dir = temp.path().join("pending");
        let output_dir = temp.path().join("output");

        create_test_artifacts(&pending_dir, 1);

        let stats = consolidate(&pending_dir, &output_dir).unwrap();
        assert_eq!(stats.artifacts_processed, 1);
        assert_eq!(stats.timeseries_rows, 100);
        assert!(stats.parquet_size_bytes > 0);
    }

    #[test]
    fn test_consolidate_multiple() {
        let temp = tempdir().unwrap();
        let pending_dir = temp.path().join("pending");
        let output_dir = temp.path().join("output");

        create_test_artifacts(&pending_dir, 10);

        let stats = consolidate(&pending_dir, &output_dir).unwrap();
        assert_eq!(stats.artifacts_processed, 10);
        assert_eq!(stats.timeseries_rows, 1000);
        assert!(output_dir.join("data/timeseries.parquet").exists());
        assert!(output_dir.join("lmdb").exists());
    }

    #[test]
    fn test_consolidate_cleans_pending() {
        let temp = tempdir().unwrap();
        let pending_dir = temp.path().join("pending");
        let output_dir = temp.path().join("output");

        create_test_artifacts(&pending_dir, 5);

        let store = PendingStore::new(&pending_dir).unwrap();
        assert_eq!(store.count().unwrap(), 5);

        consolidate(&pending_dir, &output_dir).unwrap();

        assert_eq!(store.count().unwrap(), 0);
    }

    #[test]
    fn test_consolidate_empty() {
        let temp = tempdir().unwrap();
        let pending_dir = temp.path().join("pending");
        let output_dir = temp.path().join("output");

        fs::create_dir_all(&pending_dir).unwrap();

        let stats = consolidate(&pending_dir, &output_dir).unwrap();
        assert_eq!(stats.artifacts_processed, 0);
    }
}

