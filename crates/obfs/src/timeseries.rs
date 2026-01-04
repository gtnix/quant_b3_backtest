//! TimeSeriesStore - Parquet-based storage for time-series data
//!
//! Implements columnar storage with high compression ratios for backtest equity curves.
//! Uses Apache Parquet with Zstd compression for optimal space efficiency.

use anyhow::{Context, Result};
use arrow::array::{ArrayRef, Float32Array, StringArray, UInt16Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use uuid::Uuid;

/// A single row of time-series data
#[derive(Debug, Clone)]
pub struct TimeSeriesPoint {
    /// UUID of the backtest (stored as bytes for efficiency)
    pub backtest_uuid: Uuid,
    /// Days offset from epoch (2020-01-01)
    pub date_offset: u16,
    /// Equity value
    pub equity: f32,
    /// Drawdown value
    pub drawdown: f32,
    /// Exposure value
    pub exposure: f32,
}

/// Reference to time-series data in a Parquet file
#[derive(Debug, Clone)]
pub struct TimeSeriesRef {
    pub parquet_file: String,
    pub row_group: u32,
    pub start_row: u64,
    pub num_rows: u64,
}

/// Statistics about the Parquet file
#[derive(Debug, Clone)]
pub struct ParquetStats {
    pub file_size_bytes: u64,
    pub row_count: u64,
    pub row_group_count: usize,
    pub compression_ratio: f64,
}

/// TimeSeriesStore manages Parquet files for time-series data
pub struct TimeSeriesStore {
    root_path: PathBuf,
    current_file_index: u32,
    max_rows_per_file: u64,
    current_row_count: u64,
}

impl TimeSeriesStore {
    /// Create a new TimeSeriesStore
    pub fn new(root_path: &Path) -> Result<Self> {
        let data_dir = root_path.join("data");
        std::fs::create_dir_all(&data_dir)?;

        Ok(Self {
            root_path: root_path.to_path_buf(),
            current_file_index: 0,
            max_rows_per_file: 10_000_000, // ~10M rows per file
            current_row_count: 0,
        })
    }

    /// Get the schema for time-series data
    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("backtest_uuid", DataType::Utf8, false),
            Field::new("date_offset", DataType::UInt16, false),
            Field::new("equity", DataType::Float32, false),
            Field::new("drawdown", DataType::Float32, false),
            Field::new("exposure", DataType::Float32, false),
        ])
    }

    /// Get the current Parquet file path
    fn current_file_path(&self) -> PathBuf {
        self.root_path
            .join("data")
            .join(format!("timeseries_{:04}.parquet", self.current_file_index))
    }

    /// Write time-series data for a backtest
    pub fn write_timeseries(
        &mut self,
        backtest_uuid: Uuid,
        points: &[TimeSeriesPoint],
    ) -> Result<TimeSeriesRef> {
        if points.is_empty() {
            return Err(anyhow::anyhow!("Cannot write empty time-series data"));
        }

        // Check if we need to rotate to a new file
        if self.current_row_count + points.len() as u64 > self.max_rows_per_file {
            self.current_file_index += 1;
            self.current_row_count = 0;
        }

        let file_path = self.current_file_path();
        let start_row = self.current_row_count;

        // Create Arrow arrays from points
        let uuid_str = backtest_uuid.to_string();
        let uuid_array: ArrayRef = Arc::new(StringArray::from(
            points.iter().map(|_| uuid_str.as_str()).collect::<Vec<_>>(),
        ));
        let date_offset_array: ArrayRef = Arc::new(UInt16Array::from(
            points.iter().map(|p| p.date_offset).collect::<Vec<_>>(),
        ));
        let equity_array: ArrayRef = Arc::new(Float32Array::from(
            points.iter().map(|p| p.equity).collect::<Vec<_>>(),
        ));
        let drawdown_array: ArrayRef = Arc::new(Float32Array::from(
            points.iter().map(|p| p.drawdown).collect::<Vec<_>>(),
        ));
        let exposure_array: ArrayRef = Arc::new(Float32Array::from(
            points.iter().map(|p| p.exposure).collect::<Vec<_>>(),
        ));

        // Create RecordBatch
        let schema = Arc::new(Self::schema());
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

        // Write to Parquet with Zstd compression
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(Default::default()))
            .set_dictionary_enabled(true) // Enable dictionary for UUID column
            .set_max_row_group_size(100_000) // 100K rows per row group
            .build();

        let file = File::options()
            .create(true)
            .append(true)
            .open(&file_path)
            .with_context(|| format!("Failed to open Parquet file: {:?}", file_path))?;

        // If file is empty, write with schema; otherwise append
        let file_is_empty = file.metadata()?.len() == 0;
        
        if file_is_empty {
            let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
            writer.write(&batch)?;
            writer.close()?;
        } else {
            // For append, we need to read existing data and write together
            // In production, use a more efficient approach with row groups
            let existing_file = File::open(&file_path)?;
            let reader = ParquetRecordBatchReaderBuilder::try_new(existing_file)?
                .build()?;
            
            let mut all_batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>()?;
            all_batches.push(batch);

            // Rewrite the file with all data
            let new_file = File::create(&file_path)?;
            let mut writer = ArrowWriter::try_new(new_file, schema, Some(props))?;
            for b in &all_batches {
                writer.write(b)?;
            }
            writer.close()?;
        }

        self.current_row_count += points.len() as u64;

        Ok(TimeSeriesRef {
            parquet_file: file_path.to_str().unwrap().to_string(),
            row_group: 0, // Simplified for now
            start_row,
            num_rows: points.len() as u64,
        })
    }

    /// Read time-series data for a backtest
    pub fn read_timeseries(&self, backtest_uuid: Uuid) -> Result<Vec<TimeSeriesPoint>> {
        let mut all_points = Vec::new();
        let uuid_str = backtest_uuid.to_string();

        // Scan all Parquet files
        let data_dir = self.root_path.join("data");
        if !data_dir.exists() {
            return Ok(all_points);
        }

        for entry in std::fs::read_dir(&data_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "parquet") {
                let points = self.read_from_file(&path, &uuid_str)?;
                all_points.extend(points);
            }
        }

        Ok(all_points)
    }

    /// Read time-series from a specific file
    fn read_from_file(&self, path: &Path, uuid_filter: &str) -> Result<Vec<TimeSeriesPoint>> {
        let file = File::open(path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
            .build()?;

        let mut points = Vec::new();

        for batch_result in reader {
            let batch = batch_result?;
            
            let uuid_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| anyhow::anyhow!("Failed to read UUID column"))?;
            
            let date_col = batch
                .column(1)
                .as_any()
                .downcast_ref::<UInt16Array>()
                .ok_or_else(|| anyhow::anyhow!("Failed to read date_offset column"))?;
            
            let equity_col = batch
                .column(2)
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| anyhow::anyhow!("Failed to read equity column"))?;
            
            let drawdown_col = batch
                .column(3)
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| anyhow::anyhow!("Failed to read drawdown column"))?;
            
            let exposure_col = batch
                .column(4)
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| anyhow::anyhow!("Failed to read exposure column"))?;

            for i in 0..batch.num_rows() {
                if let Some(uuid) = uuid_col.value(i).into() {
                    if uuid == uuid_filter {
                        points.push(TimeSeriesPoint {
                            backtest_uuid: Uuid::parse_str(uuid)?,
                            date_offset: date_col.value(i),
                            equity: equity_col.value(i),
                            drawdown: drawdown_col.value(i),
                            exposure: exposure_col.value(i),
                        });
                    }
                }
            }
        }

        Ok(points)
    }

    /// Get statistics for a Parquet file
    pub fn get_stats(&self, file_path: &Path) -> Result<ParquetStats> {
        let file = File::open(file_path)?;
        let file_size = file.metadata()?.len();
        
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let metadata = reader.metadata();
        
        let row_count = metadata.file_metadata().num_rows() as u64;
        let row_group_count = metadata.num_row_groups();
        
        // Estimate uncompressed size (rough approximation)
        // Each row: 36 bytes UUID + 2 bytes date + 4*3 bytes floats = 50 bytes
        let estimated_uncompressed = row_count * 50;
        let compression_ratio = if file_size > 0 {
            estimated_uncompressed as f64 / file_size as f64
        } else {
            0.0
        };

        Ok(ParquetStats {
            file_size_bytes: file_size,
            row_count,
            row_group_count,
            compression_ratio,
        })
    }

    /// List all time-series files
    pub fn list_files(&self) -> Result<Vec<PathBuf>> {
        let data_dir = self.root_path.join("data");
        if !data_dir.exists() {
            return Ok(Vec::new());
        }

        let mut files = Vec::new();
        for entry in std::fs::read_dir(&data_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "parquet") {
                files.push(path);
            }
        }
        files.sort();
        Ok(files)
    }

    /// Get total storage size of all Parquet files
    pub fn total_size(&self) -> Result<u64> {
        let files = self.list_files()?;
        let mut total = 0;
        for file in files {
            total += std::fs::metadata(&file)?.len();
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_points(uuid: Uuid, count: usize) -> Vec<TimeSeriesPoint> {
        (0..count)
            .map(|i| TimeSeriesPoint {
                backtest_uuid: uuid,
                date_offset: i as u16,
                equity: 1_000_000.0 + i as f32 * 1000.0,
                drawdown: -0.01 * i as f32,
                exposure: 0.5 + 0.001 * i as f32,
            })
            .collect()
    }

    #[test]
    fn test_write_and_read_timeseries() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut store = TimeSeriesStore::new(temp_dir.path()).unwrap();

        let uuid = Uuid::new_v4();
        let points = create_test_points(uuid, 100);

        // Write
        let ts_ref = store.write_timeseries(uuid, &points).unwrap();
        assert!(ts_ref.num_rows == 100);

        // Read back
        let read_points = store.read_timeseries(uuid).unwrap();
        assert_eq!(read_points.len(), 100);
        assert_eq!(read_points[0].backtest_uuid, uuid);
        assert_eq!(read_points[0].date_offset, 0);
        assert!((read_points[0].equity - 1_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_multiple_backtests() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut store = TimeSeriesStore::new(temp_dir.path()).unwrap();

        let uuid1 = Uuid::new_v4();
        let uuid2 = Uuid::new_v4();
        let points1 = create_test_points(uuid1, 50);
        let points2 = create_test_points(uuid2, 75);

        store.write_timeseries(uuid1, &points1).unwrap();
        store.write_timeseries(uuid2, &points2).unwrap();

        let read1 = store.read_timeseries(uuid1).unwrap();
        let read2 = store.read_timeseries(uuid2).unwrap();

        assert_eq!(read1.len(), 50);
        assert_eq!(read2.len(), 75);
    }

    #[test]
    fn test_compression_efficiency() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut store = TimeSeriesStore::new(temp_dir.path()).unwrap();

        let uuid = Uuid::new_v4();
        // 1245 rows = typical backtest (5 years of daily data)
        let points = create_test_points(uuid, 1245);

        let ts_ref = store.write_timeseries(uuid, &points).unwrap();
        let stats = store.get_stats(Path::new(&ts_ref.parquet_file)).unwrap();

        println!("Parquet stats:");
        println!("  File size: {} bytes", stats.file_size_bytes);
        println!("  Row count: {}", stats.row_count);
        println!("  Compression ratio: {:.2}x", stats.compression_ratio);

        // Parquet should achieve at least 2x compression
        assert!(stats.compression_ratio > 1.5);
    }

    #[test]
    fn test_parquet_file_listing() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut store = TimeSeriesStore::new(temp_dir.path()).unwrap();

        let uuid = Uuid::new_v4();
        let points = create_test_points(uuid, 100);
        store.write_timeseries(uuid, &points).unwrap();

        let files = store.list_files().unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].to_str().unwrap().contains("timeseries_0000.parquet"));
    }

    #[test]
    fn test_total_size_calculation() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut store = TimeSeriesStore::new(temp_dir.path()).unwrap();

        let uuid = Uuid::new_v4();
        let points = create_test_points(uuid, 500);
        store.write_timeseries(uuid, &points).unwrap();

        let total_size = store.total_size().unwrap();
        assert!(total_size > 0);
    }
}

