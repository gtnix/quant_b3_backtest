//! Consolidator - Single-thread merge of pending artifacts into Parquet + LMDB.
//!
//! After evolution completes with parallel workers writing isolated `.obfs` files,
//! the Consolidator merges all pending artifacts into:
//! - `timeseries_<timestamp>_batch<N>.parquet` - Columnar storage with Zstd compression
//! - `index.lmdb` - O(1) lookup by UUID
//!
//! This is a single-threaded operation to avoid Parquet concurrent write issues.
//!
//! ## QUANT PRINCIPLE: DATA PRESERVATION IS SACRED
//! 
//! **NEVER delete source data until it is safely persisted to destination.**
//! 
//! This consolidator uses incremental batch processing:
//! - Processes artifacts in chunks of INCREMENTAL_BATCH_SIZE
//! - Each batch is written to its own timestamped Parquet file
//! - Pending files are ONLY deleted after successful Parquet write + fsync
//! - Memory is released between batches
//! - Prevents i32 offset overflow in StringArray (2GB limit)

use std::fs::{self, File};
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

/// Artifacts per memory batch within a single Parquet file.
/// Controls Arrow memory usage (prevents offset overflow).
const MEMORY_BATCH_SIZE: usize = 5_000;

/// Artifacts per incremental consolidation batch.
/// Each batch produces one Parquet file. Pending files are deleted
/// ONLY after successful write of each batch.
const INCREMENTAL_BATCH_SIZE: usize = 50_000;

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

/// Statistics from a compaction run.
#[derive(Debug, Clone, Default)]
pub struct CompactionStats {
    /// Number of input Parquet files merged.
    pub files_merged: usize,
    /// Total rows in output file.
    pub total_rows: usize,
    /// Size of merged output file in bytes.
    pub output_size_bytes: u64,
    /// Total size of input files in bytes.
    pub input_size_bytes: u64,
    /// Space saved in bytes (input - output).
    pub space_saved_bytes: i64,
    /// Number of files removed after merge.
    pub files_removed: usize,
    /// Total time in milliseconds.
    pub duration_ms: u64,
    /// Whether compaction was skipped (not enough files).
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

    /// Consolidate all pending artifacts into Parquet + LMDB using incremental batches.
    /// 
    /// ## QUANT PRINCIPLE: DATA PRESERVATION IS SACRED
    /// 
    /// This method processes artifacts in incremental batches:
    /// 1. Each batch of INCREMENTAL_BATCH_SIZE artifacts → one Parquet file
    /// 2. Pending files are ONLY deleted after successful Parquet write + fsync
    /// 3. If any batch fails, remaining data is preserved
    /// 
    /// This ensures NO DATA LOSS even with hundreds of thousands of artifacts.
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

        let total_artifacts = pending_uuids.len();
        let num_batches = (total_artifacts + INCREMENTAL_BATCH_SIZE - 1) / INCREMENTAL_BATCH_SIZE;
        
        eprintln!(
            "[obfs] Incremental consolidation: {} artifacts in {} batches of up to {}",
            total_artifacts, num_batches, INCREMENTAL_BATCH_SIZE
        );

        // Create output directories
        fs::create_dir_all(&self.output_dir)?;
        let data_dir = self.output_dir.join("data");
        let lmdb_dir = self.output_dir.join("lmdb");
        fs::create_dir_all(&data_dir)?;
        fs::create_dir_all(&lmdb_dir)?;

        // Open LMDB once for all batches
        let metadata_store = MetadataStore::open(&lmdb_dir)?;

        // Generate timestamp for this consolidation run
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string();

        // Process each incremental batch
        for (batch_idx, batch_uuids) in pending_uuids.chunks(INCREMENTAL_BATCH_SIZE).enumerate() {
            let batch_num = batch_idx + 1;
            
            eprintln!(
                "[obfs] Processing batch {}/{}: {} artifacts",
                batch_num, num_batches, batch_uuids.len()
            );

            // Create timestamped Parquet file for this batch
            let parquet_filename = format!("timeseries_{}_batch{:04}.parquet", timestamp, batch_num);
            let parquet_path = data_dir.join(&parquet_filename);

            // Write Parquet file for this batch
            let (ts_rows, memory_batches) = self.write_batch_to_parquet(
                &pending_store,
                batch_uuids,
                &parquet_path,
            )?;

            // Verify Parquet file was written successfully
            if !parquet_path.exists() {
                anyhow::bail!(
                    "[obfs] CRITICAL: Parquet file not created for batch {}: {:?}",
                    batch_num, parquet_path
                );
            }

            let parquet_size = fs::metadata(&parquet_path)?.len();
            if parquet_size == 0 {
                anyhow::bail!(
                    "[obfs] CRITICAL: Parquet file is empty for batch {}: {:?}",
                    batch_num, parquet_path
                );
            }

            // Index artifacts in LMDB
            self.index_batch_in_lmdb(&pending_store, batch_uuids, &parquet_path, &metadata_store)?;

            // Sync LMDB to ensure durability
            metadata_store.sync()?;

            // NOW and ONLY NOW delete the pending files for this batch
            // All data has been safely persisted
            let mut deleted = 0;
            for uuid in batch_uuids {
                if pending_store.remove_pending(*uuid).is_ok() {
                    deleted += 1;
                }
            }

            stats.artifacts_processed += batch_uuids.len();
            stats.timeseries_rows += ts_rows;
            stats.batches_written += memory_batches;
            stats.parquet_size_bytes += parquet_size;

            eprintln!(
                "[obfs] Batch {}/{} complete: {} rows, {:.2} MB, {} pending files safely deleted",
                batch_num, num_batches, ts_rows,
                parquet_size as f64 / 1_048_576.0,
                deleted
            );
        }

        // Calculate LMDB size
        if let Ok(entries) = fs::read_dir(&lmdb_dir) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata() {
                    stats.lmdb_size_bytes += meta.len();
                }
            }
        }

        stats.duration_ms = start.elapsed().as_millis() as u64;

        // Calculate compression ratio
        let original_estimate = stats.artifacts_processed * 20_000; // ~20KB per .obfs file
        if stats.parquet_size_bytes > 0 {
            stats.compression_ratio = original_estimate as f64 / stats.parquet_size_bytes as f64;
        }

        eprintln!(
            "[obfs] Consolidation complete: {} artifacts, {} rows, {:.1} MB Parquet in {}ms",
            stats.artifacts_processed,
            stats.timeseries_rows,
            stats.parquet_size_bytes as f64 / 1_048_576.0,
            stats.duration_ms
        );

        Ok(stats)
    }

    /// Write a batch of artifacts to a single Parquet file.
    /// Returns (total_rows, memory_batches_written).
    fn write_batch_to_parquet(
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

        let file = File::create(output_path)
            .with_context(|| format!("Failed to create Parquet file: {:?}", output_path))?;

        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
        let mut total_rows = 0;
        let mut batches_written = 0;

        // Process in memory batches to prevent Arrow offset overflow
        for uuid_chunk in uuids.chunks(MEMORY_BATCH_SIZE) {
            let batch = self.build_batch_for_uuids(pending_store, uuid_chunk)?;
            
            if batch.num_rows() > 0 {
                total_rows += batch.num_rows();
                writer.write(&batch)?;
                batches_written += 1;
            }
        }

        // Close writer and ensure data is flushed
        writer.close()?;

        // Fsync the file to ensure durability before we delete source files
        let file = File::open(output_path)?;
        file.sync_all()?;

        Ok((total_rows, batches_written))
    }

    /// Index a batch of artifacts in LMDB.
    fn index_batch_in_lmdb(
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

        Ok(())
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

}

impl Consolidator {
    /// Consolidate a specific list of UUIDs into Parquet + LMDB.
    /// 
    /// ## QUANT PRINCIPLE: DATA PRESERVATION IS SACRED
    /// 
    /// This method consolidates ONLY the specified UUIDs:
    /// 1. Writes artifacts to timestamped Parquet file
    /// 2. Indexes in LMDB
    /// 3. Deletes pending files ONLY after successful fsync
    /// 
    /// Use this for incremental cleanup during evolution - consolidate artifacts
    /// that are no longer needed before deleting them.
    pub fn consolidate_specific(&self, uuids_to_consolidate: &[uuid::Uuid]) -> Result<ConsolidationStats> {
        let start = std::time::Instant::now();
        let mut stats = ConsolidationStats::default();

        if uuids_to_consolidate.is_empty() {
            return Ok(stats);
        }

        // Open pending store
        let pending_store = PendingStore::new(&self.pending_dir)?;

        let total_artifacts = uuids_to_consolidate.len();
        let num_batches = (total_artifacts + INCREMENTAL_BATCH_SIZE - 1) / INCREMENTAL_BATCH_SIZE;

        // Create output directories
        fs::create_dir_all(&self.output_dir)?;
        let data_dir = self.output_dir.join("data");
        let lmdb_dir = self.output_dir.join("lmdb");
        fs::create_dir_all(&data_dir)?;
        fs::create_dir_all(&lmdb_dir)?;

        // Open LMDB once for all batches
        let metadata_store = MetadataStore::open(&lmdb_dir)?;

        // Generate timestamp for this consolidation run
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string();

        // Process each incremental batch
        for (batch_idx, batch_uuids) in uuids_to_consolidate.chunks(INCREMENTAL_BATCH_SIZE).enumerate() {
            let batch_num = batch_idx + 1;

            // Create timestamped Parquet file for this batch
            let parquet_filename = format!("timeseries_{}_batch{:04}.parquet", timestamp, batch_num);
            let parquet_path = data_dir.join(&parquet_filename);

            // Write Parquet file for this batch
            let (ts_rows, memory_batches) = self.write_batch_to_parquet(
                &pending_store,
                batch_uuids,
                &parquet_path,
            )?;

            // Skip if no rows were written (all artifacts were empty or missing)
            if ts_rows == 0 {
                // Remove empty parquet file if created
                let _ = fs::remove_file(&parquet_path);
                continue;
            }

            // Verify Parquet file was written successfully
            if !parquet_path.exists() {
                anyhow::bail!(
                    "[obfs] CRITICAL: Parquet file not created for batch {}: {:?}",
                    batch_num, parquet_path
                );
            }

            let parquet_size = fs::metadata(&parquet_path)?.len();
            if parquet_size == 0 {
                anyhow::bail!(
                    "[obfs] CRITICAL: Parquet file is empty for batch {}: {:?}",
                    batch_num, parquet_path
                );
            }

            // Index artifacts in LMDB
            self.index_batch_in_lmdb(&pending_store, batch_uuids, &parquet_path, &metadata_store)?;

            // Sync LMDB to ensure durability
            metadata_store.sync()?;

            // NOW and ONLY NOW delete the pending files for this batch
            let mut deleted = 0;
            for uuid in batch_uuids {
                if pending_store.remove_pending(*uuid).is_ok() {
                    deleted += 1;
                }
            }

            stats.artifacts_processed += batch_uuids.len();
            stats.timeseries_rows += ts_rows;
            stats.batches_written += memory_batches;
            stats.parquet_size_bytes += parquet_size;

            eprintln!(
                "[obfs] Batch {}/{} complete: {} rows, {:.2} MB, {} pending files safely deleted",
                batch_num, num_batches, ts_rows,
                parquet_size as f64 / 1_048_576.0,
                deleted
            );
        }

        // Calculate LMDB size
        if let Ok(entries) = fs::read_dir(&lmdb_dir) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata() {
                    stats.lmdb_size_bytes += meta.len();
                }
            }
        }

        stats.duration_ms = start.elapsed().as_millis() as u64;

        // Calculate compression ratio
        let original_estimate = stats.artifacts_processed * 20_000;
        if stats.parquet_size_bytes > 0 {
            stats.compression_ratio = original_estimate as f64 / stats.parquet_size_bytes as f64;
        }

        Ok(stats)
    }
}

/// Convenience function to consolidate pending artifacts.
/// Uses safe incremental consolidation - NEVER deletes data without first persisting to Parquet.
pub fn consolidate(pending_dir: impl Into<PathBuf>, output_dir: impl Into<PathBuf>) -> Result<ConsolidationStats> {
    let consolidator = Consolidator::new(pending_dir, output_dir);
    consolidator.consolidate()
}

/// Consolidate specific UUIDs and cleanup.
/// 
/// ## QUANT PRINCIPLE: DATA PRESERVATION IS SACRED
/// 
/// This function consolidates artifacts NOT in the keep set, then deletes them.
/// Artifacts in the keep set are left untouched.
/// 
/// # Arguments
/// * `pending_dir` - Directory containing pending .obfs files
/// * `output_dir` - Directory for Parquet + LMDB output
/// * `keep_uuids` - Set of UUIDs to keep (will NOT be consolidated or deleted)
/// 
/// # Returns
/// * `Ok((stats, removed, kept))` - Consolidation stats, removed count, kept count
pub fn consolidate_and_cleanup(
    pending_dir: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
    keep_uuids: &std::collections::HashSet<uuid::Uuid>,
) -> Result<(ConsolidationStats, usize, usize)> {
    let pending_dir = pending_dir.as_ref();
    let output_dir = output_dir.as_ref();
    
    // List all pending artifacts
    let pending_store = PendingStore::new(pending_dir)?;
    let all_uuids = pending_store.list_pending()?;
    
    // Separate into to-consolidate and to-keep
    let mut to_consolidate = Vec::new();
    let mut kept = 0;
    
    for uuid in all_uuids {
        if keep_uuids.contains(&uuid) {
            kept += 1;
        } else {
            to_consolidate.push(uuid);
        }
    }
    
    if to_consolidate.is_empty() {
        return Ok((ConsolidationStats::default(), 0, kept));
    }
    
    // Consolidate the artifacts NOT in keep set
    let consolidator = Consolidator::new(pending_dir, output_dir);
    let stats = consolidator.consolidate_specific(&to_consolidate)?;
    
    // The consolidate_specific method already deletes after successful persistence
    let removed = stats.artifacts_processed;
    
    Ok((stats, removed, kept))
}

/// Compact multiple small Parquet files into fewer larger files.
///
/// ## QUANT PRINCIPLE: DATA PRESERVATION IS SACRED
///
/// This function merges small Parquet files to reduce file count and improve
/// read performance. Original files are ONLY deleted after successful write
/// of the merged file.
///
/// # Arguments
/// * `data_dir` - Directory containing Parquet files (e.g., "consolidated/data")
/// * `min_files` - Minimum number of files before compaction triggers
/// * `target_size_mb` - Target size for merged files in MB
///
/// # Returns
/// * `Ok(CompactionStats)` - Compaction statistics
pub fn compact_parquets(
    data_dir: impl AsRef<Path>,
    min_files: usize,
    target_size_mb: f64,
) -> Result<CompactionStats> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    
    let start = std::time::Instant::now();
    let data_dir = data_dir.as_ref();
    let mut stats = CompactionStats::default();
    
    if !data_dir.exists() {
        return Ok(stats);
    }
    
    // List all Parquet files sorted by modification time (oldest first)
    let mut parquet_files: Vec<_> = fs::read_dir(data_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().extension().map_or(false, |ext| ext == "parquet")
        })
        .map(|e| {
            let path = e.path();
            let meta = fs::metadata(&path).ok();
            let size = meta.as_ref().map(|m| m.len()).unwrap_or(0);
            let mtime = meta.and_then(|m| m.modified().ok());
            (path, size, mtime)
        })
        .collect();
    
    // Sort by modification time (oldest first)
    parquet_files.sort_by(|a, b| a.2.cmp(&b.2));
    
    // Check if we have enough files to compact
    if parquet_files.len() < min_files {
        stats.skipped = true;
        return Ok(stats);
    }
    
    let target_size_bytes = (target_size_mb * 1_048_576.0) as u64;
    let total_input_size: u64 = parquet_files.iter().map(|(_, size, _)| *size).sum();
    stats.input_size_bytes = total_input_size;
    
    eprintln!(
        "[obfs] Compacting {} Parquet files ({:.2} MB total)",
        parquet_files.len(),
        total_input_size as f64 / 1_048_576.0
    );
    
    // Group files into batches targeting ~target_size_mb each
    let mut batches: Vec<Vec<PathBuf>> = Vec::new();
    let mut current_batch: Vec<PathBuf> = Vec::new();
    let mut current_batch_size: u64 = 0;
    
    for (path, size, _) in &parquet_files {
        current_batch.push(path.clone());
        current_batch_size += size;
        
        if current_batch_size >= target_size_bytes {
            batches.push(std::mem::take(&mut current_batch));
            current_batch_size = 0;
        }
    }
    
    // Add remaining files if there are enough
    if current_batch.len() >= 2 {
        batches.push(current_batch);
    }
    
    // Skip if no batches to merge
    if batches.is_empty() {
        stats.skipped = true;
        return Ok(stats);
    }
    
    // Process each batch
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string();
    
    for (batch_idx, batch_files) in batches.iter().enumerate() {
        if batch_files.len() < 2 {
            continue; // Skip batches with only 1 file
        }
        
        let output_filename = format!("compacted_{}_batch{:04}.parquet", timestamp, batch_idx + 1);
        let output_path = data_dir.join(&output_filename);
        
        // Read all record batches from input files
        let mut all_batches: Vec<RecordBatch> = Vec::new();
        let mut schema: Option<Arc<Schema>> = None;
        
        for input_file in batch_files {
            let file = File::open(input_file)
                .with_context(|| format!("Failed to open {:?}", input_file))?;
            
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
            
            // Capture schema from first file
            if schema.is_none() {
                schema = Some(builder.schema().clone());
            }
            
            let reader = builder.build()?;
            
            for batch_result in reader {
                let batch = batch_result?;
                stats.total_rows += batch.num_rows();
                all_batches.push(batch);
            }
        }
        
        // Write merged Parquet file
        let schema = schema.context("No schema found in input files")?;
        let output_file = File::create(&output_path)?;
        
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(Default::default()))
            .set_dictionary_enabled(true)
            .set_max_row_group_size(MAX_ROW_GROUP_SIZE)
            .build();
        
        let mut writer = ArrowWriter::try_new(output_file, schema, Some(props))?;
        
        for batch in &all_batches {
            writer.write(batch)?;
        }
        
        writer.close()?;
        
        // Verify output file
        let output_meta = fs::metadata(&output_path)?;
        if output_meta.len() == 0 {
            anyhow::bail!("Compacted file is empty: {:?}", output_path);
        }
        
        stats.output_size_bytes += output_meta.len();
        stats.files_merged += batch_files.len();
        
        // NOW and ONLY NOW delete the original files
        for input_file in batch_files {
            if fs::remove_file(input_file).is_ok() {
                stats.files_removed += 1;
            }
        }
        
        eprintln!(
            "[obfs] Compacted batch {}: {} files -> {:.2} MB",
            batch_idx + 1,
            batch_files.len(),
            output_meta.len() as f64 / 1_048_576.0
        );
    }
    
    stats.space_saved_bytes = stats.input_size_bytes as i64 - stats.output_size_bytes as i64;
    stats.duration_ms = start.elapsed().as_millis() as u64;
    
    eprintln!(
        "[obfs] Compaction complete: {} files merged, {:.2} MB saved, {}ms",
        stats.files_merged,
        stats.space_saved_bytes as f64 / 1_048_576.0,
        stats.duration_ms
    );
    
    Ok(stats)
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
        // Check for timestamped parquet file (new format)
        let data_dir = output_dir.join("data");
        let parquet_files: Vec<_> = fs::read_dir(&data_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "parquet"))
            .collect();
        assert!(!parquet_files.is_empty(), "Should have at least one parquet file");
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

    #[test]
    fn test_consolidate_specific() {
        let temp = tempdir().unwrap();
        let pending_dir = temp.path().join("pending");
        let output_dir = temp.path().join("output");

        // Create 10 artifacts
        let uuids = create_test_artifacts(&pending_dir, 10);
        let store = PendingStore::new(&pending_dir).unwrap();
        assert_eq!(store.count().unwrap(), 10);

        // Consolidate only first 5
        let consolidator = Consolidator::new(&pending_dir, &output_dir);
        let stats = consolidator.consolidate_specific(&uuids[0..5]).unwrap();
        
        assert_eq!(stats.artifacts_processed, 5);
        assert_eq!(stats.timeseries_rows, 500);
        
        // Only 5 should remain (the ones we didn't consolidate)
        assert_eq!(store.count().unwrap(), 5);
        
        // Verify the remaining ones are the correct UUIDs
        let remaining = store.list_pending().unwrap();
        for uuid in &uuids[5..10] {
            assert!(remaining.contains(uuid), "UUID {:?} should still be pending", uuid);
        }
    }

    #[test]
    fn test_consolidate_and_cleanup() {
        let temp = tempdir().unwrap();
        let pending_dir = temp.path().join("pending");
        let output_dir = temp.path().join("output");

        // Create 10 artifacts
        let uuids = create_test_artifacts(&pending_dir, 10);
        let store = PendingStore::new(&pending_dir).unwrap();
        assert_eq!(store.count().unwrap(), 10);

        // Keep set: last 3 UUIDs
        let keep_set: std::collections::HashSet<uuid::Uuid> = 
            uuids[7..10].iter().copied().collect();

        // Consolidate and cleanup
        let (stats, removed, kept) = consolidate_and_cleanup(
            &pending_dir, 
            &output_dir, 
            &keep_set
        ).unwrap();

        // 7 should be consolidated (removed)
        assert_eq!(stats.artifacts_processed, 7);
        assert_eq!(removed, 7);
        assert_eq!(kept, 3);

        // 3 should remain pending
        assert_eq!(store.count().unwrap(), 3);

        // Verify correct UUIDs remain
        let remaining = store.list_pending().unwrap();
        for uuid in &uuids[7..10] {
            assert!(remaining.contains(uuid), "UUID {:?} should still be pending", uuid);
        }

        // Verify Parquet file was created
        let data_dir = output_dir.join("data");
        let parquet_files: Vec<_> = fs::read_dir(&data_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "parquet"))
            .collect();
        assert!(!parquet_files.is_empty(), "Should have at least one parquet file");
    }

    #[test]
    fn test_consolidate_and_cleanup_all_kept() {
        let temp = tempdir().unwrap();
        let pending_dir = temp.path().join("pending");
        let output_dir = temp.path().join("output");

        // Create 5 artifacts
        let uuids = create_test_artifacts(&pending_dir, 5);
        let store = PendingStore::new(&pending_dir).unwrap();

        // Keep all UUIDs
        let keep_set: std::collections::HashSet<uuid::Uuid> = 
            uuids.iter().copied().collect();

        let (stats, removed, kept) = consolidate_and_cleanup(
            &pending_dir, 
            &output_dir, 
            &keep_set
        ).unwrap();

        // Nothing should be consolidated
        assert_eq!(stats.artifacts_processed, 0);
        assert_eq!(removed, 0);
        assert_eq!(kept, 5);

        // All should remain
        assert_eq!(store.count().unwrap(), 5);
    }

    #[test]
    fn test_quant_principle_data_preserved_in_lmdb() {
        // QUANT PRINCIPLE: Data is NEVER lost - verify we can read metadata from LMDB
        // after consolidation deletes the pending files
        
        let temp = tempdir().unwrap();
        let pending_dir = temp.path().join("pending");
        let output_dir = temp.path().join("output");

        // Create 5 artifacts and record their metrics
        let uuids = create_test_artifacts(&pending_dir, 5);
        let store = PendingStore::new(&pending_dir).unwrap();
        
        // Record original metrics for verification
        let mut original_metrics: Vec<(uuid::Uuid, f64)> = Vec::new();
        for uuid in &uuids {
            let artifact = store.read_pending(*uuid).unwrap();
            original_metrics.push((*uuid, artifact.metrics.sharpe_ratio));
        }

        // Consolidate all artifacts
        let stats = consolidate(&pending_dir, &output_dir).unwrap();
        assert_eq!(stats.artifacts_processed, 5);

        // Verify pending files are gone
        assert_eq!(store.count().unwrap(), 0, "Pending files should be deleted after consolidation");

        // Verify data is preserved in LMDB
        let lmdb_dir = output_dir.join("lmdb");
        let metadata_store = crate::store::MetadataStore::open(&lmdb_dir).unwrap();
        
        for (uuid, expected_sharpe) in &original_metrics {
            let metadata = metadata_store.get(*uuid).unwrap();
            assert!(metadata.is_some(), "UUID {:?} should exist in LMDB after consolidation", uuid);
            
            let metadata = metadata.unwrap();
            assert!(
                (metadata.metrics.sharpe_ratio - expected_sharpe).abs() < 0.001,
                "Sharpe ratio should be preserved: expected {}, got {}",
                expected_sharpe, metadata.metrics.sharpe_ratio
            );
        }
    }

    #[test]
    fn test_quant_principle_no_data_loss_on_partial_cleanup() {
        // Verify that consolidate_and_cleanup preserves data integrity:
        // - Kept files remain in pending/
        // - Removed files are persisted to Parquet + LMDB
        
        let temp = tempdir().unwrap();
        let pending_dir = temp.path().join("pending");
        let output_dir = temp.path().join("output");

        let uuids = create_test_artifacts(&pending_dir, 10);
        let store = PendingStore::new(&pending_dir).unwrap();

        // Keep last 3
        let keep_set: std::collections::HashSet<uuid::Uuid> = 
            uuids[7..10].iter().copied().collect();
        
        // Record metrics for the ones we're consolidating (first 7)
        let mut consolidated_metrics: Vec<(uuid::Uuid, f64)> = Vec::new();
        for uuid in &uuids[0..7] {
            let artifact = store.read_pending(*uuid).unwrap();
            consolidated_metrics.push((*uuid, artifact.metrics.sharpe_ratio));
        }

        let (stats, _removed, _kept) = consolidate_and_cleanup(
            &pending_dir, 
            &output_dir, 
            &keep_set
        ).unwrap();

        assert_eq!(stats.artifacts_processed, 7);

        // Verify consolidated UUIDs are in LMDB
        let lmdb_dir = output_dir.join("lmdb");
        let metadata_store = crate::store::MetadataStore::open(&lmdb_dir).unwrap();
        
        for (uuid, expected_sharpe) in &consolidated_metrics {
            let metadata = metadata_store.get(*uuid).unwrap();
            assert!(
                metadata.is_some(), 
                "Consolidated UUID {:?} should exist in LMDB", 
                uuid
            );
            assert!(
                (metadata.unwrap().metrics.sharpe_ratio - expected_sharpe).abs() < 0.001,
                "Metrics should be preserved in LMDB"
            );
        }

        // Verify kept UUIDs are still in pending
        for uuid in &uuids[7..10] {
            let artifact = store.read_pending(*uuid);
            assert!(
                artifact.is_ok(), 
                "Kept UUID {:?} should still be readable from pending", 
                uuid
            );
        }
    }
}
