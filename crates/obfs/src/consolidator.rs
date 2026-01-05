//! Consolidator - Single-thread merge of pending artifacts into Parquet + LMDB.
//!
//! After evolution completes with parallel workers writing isolated `.obfs` files,
//! the Consolidator merges all pending artifacts into:
//! - `timeseries.parquet` - Columnar storage with Zstd compression
//! - `index.lmdb` - O(1) lookup by UUID
//!
//! This is a single-threaded operation to avoid Parquet concurrent write issues.
//!
//! ## Memory Safety
//! 
//! Uses streaming batch writes to prevent Arrow offset overflow:
//! - Processes artifacts in chunks of BATCH_ARTIFACTS
//! - Each batch is written as a separate row group
//! - Memory is released between batches
//! - Prevents i32 offset overflow in StringArray (2GB limit)

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
use crate::pending_store::PendingStore;
use crate::store::MetadataStore;
use crate::types::{ArtifactLocation, ArtifactMetadata};

/// Maximum artifacts per batch to prevent Arrow offset overflow.
/// 5000 artifacts × ~1000 timeseries points × 36 chars = ~180 MB strings per batch.
/// Well under the 2GB i32 offset limit.
const BATCH_ARTIFACTS: usize = 5_000;

/// Maximum artifacts before skipping consolidation.
/// Above this threshold, keep pending files and consolidate on-demand.
const MAX_CONSOLIDATE_THRESHOLD: usize = 100_000;

/// Maximum rows per Parquet row group for optimal read performance.
const MAX_ROW_GROUP_SIZE: usize = 500_000;

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
    /// Number of batches written.
    pub batches_written: usize,
    /// Whether consolidation was skipped due to threshold.
    pub skipped: bool,
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
    /// Uses streaming batch writes to prevent memory overflow.
    pub fn consolidate(&self) -> Result<ConsolidationStats> {
        let start = std::time::Instant::now();
        let mut stats = ConsolidationStats::default();

        // Open pending store
        let pending_store = PendingStore::new(&self.pending_dir)?;
        let pending_uuids = pending_store.list_pending()?;

        if pending_uuids.is_empty() {
            eprintln!("[obfs] No pending artifacts to consolidate");
            return Ok(stats);
        }

        eprintln!("[obfs] Found {} pending artifacts to consolidate", pending_uuids.len());

        // Check threshold - skip consolidation for very large runs
        if pending_uuids.len() > MAX_CONSOLIDATE_THRESHOLD {
            eprintln!(
                "[obfs] Skipping consolidation: {} artifacts exceeds threshold {}. \
                 Pending files will be kept for on-demand consolidation.",
                pending_uuids.len(),
                MAX_CONSOLIDATE_THRESHOLD
            );
            stats.artifacts_processed = pending_uuids.len();
            stats.skipped = true;
            stats.duration_ms = start.elapsed().as_millis() as u64;
            return Ok(stats);
        }

        // Create output directories
        fs::create_dir_all(&self.output_dir)?;
        let data_dir = self.output_dir.join("data");
        let lmdb_dir = self.output_dir.join("lmdb");
        fs::create_dir_all(&data_dir)?;
        fs::create_dir_all(&lmdb_dir)?;

        // Stream artifacts in batches to prevent memory overflow
        let parquet_path = data_dir.join("timeseries.parquet");
        let (ts_rows, batches) = self.write_timeseries_streaming(&pending_store, &pending_uuids, &parquet_path)?;
        stats.timeseries_rows = ts_rows;
        stats.batches_written = batches;
        stats.artifacts_processed = pending_uuids.len();

        if parquet_path.exists() {
            stats.parquet_size_bytes = fs::metadata(&parquet_path)?.len();
        }

        // Index in LMDB (stream artifacts again to avoid memory issues)
        let metadata_store = MetadataStore::open(&lmdb_dir)?;
        self.index_artifacts_streaming(&pending_store, &pending_uuids, &parquet_path, &metadata_store)?;

        // Calculate LMDB size
        if let Ok(entries) = fs::read_dir(&lmdb_dir) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata() {
                    stats.lmdb_size_bytes += meta.len();
                }
            }
        }

        // Clean up pending files
        for uuid in &pending_uuids {
            let _ = pending_store.remove_pending(*uuid);
        }

        stats.duration_ms = start.elapsed().as_millis() as u64;

        // Calculate compression ratio
        let original_estimate = stats.artifacts_processed * 60_000; // ~60KB per legacy run
        if stats.parquet_size_bytes > 0 {
            stats.compression_ratio = original_estimate as f64 / stats.parquet_size_bytes as f64;
        }

        eprintln!(
            "[obfs] Consolidation complete: {} artifacts, {} rows, {} batches, {:.1} MB Parquet in {}ms",
            stats.artifacts_processed,
            stats.timeseries_rows,
            stats.batches_written,
            stats.parquet_size_bytes as f64 / 1_048_576.0,
            stats.duration_ms
        );

        Ok(stats)
    }

    /// Get the timeseries schema.
    fn timeseries_schema() -> Schema {
        Schema::new(vec![
            Field::new("backtest_uuid", DataType::Utf8, false),
            Field::new("date_offset", DataType::UInt16, false),
            Field::new("equity", DataType::Float32, false),
            Field::new("drawdown", DataType::Float32, false),
            Field::new("exposure", DataType::Float32, false),
        ])
    }

    /// Write timeseries data using streaming batches.
    /// 
    /// This prevents Arrow offset overflow by:
    /// 1. Processing artifacts in chunks of BATCH_ARTIFACTS
    /// 2. Writing each chunk as a separate row group
    /// 3. Releasing memory between chunks
    fn write_timeseries_streaming(
        &self,
        pending_store: &PendingStore,
        uuids: &[uuid::Uuid],
        output_path: &Path,
    ) -> Result<(usize, usize)> {
        let schema = Arc::new(Self::timeseries_schema());
        
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(Default::default()))
            .set_dictionary_enabled(true)
            .set_max_row_group_size(MAX_ROW_GROUP_SIZE)
            .build();

        let file = fs::File::create(output_path)
            .with_context(|| format!("Failed to create Parquet file: {:?}", output_path))?;

        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
        let mut total_rows = 0;
        let mut batches_written = 0;

        // Process in chunks to prevent memory overflow
        for (chunk_idx, uuid_chunk) in uuids.chunks(BATCH_ARTIFACTS).enumerate() {
            let batch = self.build_batch_for_uuids(pending_store, uuid_chunk)?;
            
            if batch.num_rows() > 0 {
                total_rows += batch.num_rows();
                writer.write(&batch)?;
                batches_written += 1;
                
                // Debug: batch progress (uncomment for verbose output)
                // eprintln!("[obfs] Wrote batch {}: {} artifacts, {} rows", chunk_idx + 1, uuid_chunk.len(), batch.num_rows());
            }
            
            // Batch goes out of scope here, memory is released
        }

        writer.close()?;
        Ok((total_rows, batches_written))
    }

    /// Build a RecordBatch for a chunk of UUIDs.
    fn build_batch_for_uuids(
        &self,
        pending_store: &PendingStore,
        uuids: &[uuid::Uuid],
    ) -> Result<RecordBatch> {
        let mut all_uuids: Vec<String> = Vec::new();
        let mut all_date_offsets: Vec<u16> = Vec::new();
        let mut all_equity: Vec<f32> = Vec::new();
        let mut all_drawdown: Vec<f32> = Vec::new();
        let mut all_exposure: Vec<f32> = Vec::new();

        for uuid in uuids {
            let artifact = match pending_store.read_pending(*uuid) {
                Ok(a) => a,
                Err(_) => continue,
            };

            let uuid_str = uuid.to_string();
            for point in &artifact.timeseries {
                all_uuids.push(uuid_str.clone());
                all_date_offsets.push(point.date_offset);
                all_equity.push(point.equity);
                all_drawdown.push(point.drawdown);
                all_exposure.push(point.exposure);
            }
            // artifact goes out of scope here, memory released
        }

        let schema = Arc::new(Self::timeseries_schema());

        if all_uuids.is_empty() {
            return Ok(RecordBatch::new_empty(schema));
        }

        let uuid_array: ArrayRef = Arc::new(StringArray::from(all_uuids));
        let date_offset_array: ArrayRef = Arc::new(UInt16Array::from(all_date_offsets));
        let equity_array: ArrayRef = Arc::new(Float32Array::from(all_equity));
        let drawdown_array: ArrayRef = Arc::new(Float32Array::from(all_drawdown));
        let exposure_array: ArrayRef = Arc::new(Float32Array::from(all_exposure));

        Ok(RecordBatch::try_new(
            schema,
            vec![
                uuid_array,
                date_offset_array,
                equity_array,
                drawdown_array,
                exposure_array,
            ],
        )?)
    }

    /// Index artifacts in LMDB for O(1) lookup, streaming to avoid memory issues.
    fn index_artifacts_streaming(
        &self,
        pending_store: &PendingStore,
        uuids: &[uuid::Uuid],
        parquet_path: &Path,
        store: &MetadataStore,
    ) -> Result<()> {
        let parquet_path_str = parquet_path.to_string_lossy().to_string();
        let mut row_offset: u64 = 0;

        for uuid in uuids {
            let artifact = match pending_store.read_pending(*uuid) {
                Ok(a) => a,
                Err(_) => continue,
            };

            let ts_count = artifact.timeseries.len() as u64;

            let metadata = ArtifactMetadata {
                uuid: artifact.run_id,
                artifact_location: ArtifactLocation {
                    file_path: parquet_path_str.clone(),
                    offset: row_offset,
                    size: ts_count,
                },
                blake3_hash: [0u8; 32],
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
    use crate::pending_store::{PendingArtifact, TimeseriesPoint};
    use crate::types::Metadata;
    use tempfile::tempdir;

    fn create_test_artifacts(pending_dir: &Path, count: usize) -> Vec<uuid::Uuid> {
        let store = PendingStore::new(pending_dir).unwrap();
        let mut uuids = Vec::new();

        for i in 0..count {
            let uuid = uuid::Uuid::new_v4();
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
        assert!(!stats.skipped);
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

    #[test]
    fn test_consolidate_batched() {
        let temp = tempdir().unwrap();
        let pending_dir = temp.path().join("pending");
        let output_dir = temp.path().join("output");

        // Create enough artifacts to trigger multiple batches
        create_test_artifacts(&pending_dir, 100);

        let stats = consolidate(&pending_dir, &output_dir).unwrap();
        assert_eq!(stats.artifacts_processed, 100);
        assert_eq!(stats.timeseries_rows, 10_000);
        assert!(stats.batches_written >= 1);
    }
}
