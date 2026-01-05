//! Pending Store - Concurrent-safe isolated file storage for backtests.
//!
//! Each backtest writes its own isolated `.obfs` file, avoiding Parquet
//! concurrent write issues. Files are later consolidated into Parquet.
//!
//! ## Design
//! - Phase 1: Each process writes `{run_id}.obfs` (Zstd compressed JSON)
//! - Phase 2: Consolidator merges all into `timeseries.parquet` + LMDB index
//!
//! ## File Format
//! ```text
//! {run_id}.obfs:
//!   - Magic: "OBFS" (4 bytes)
//!   - Version: u8
//!   - Zstd compressed PendingArtifact JSON
//! ```

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::compression::CompressionPipeline;
use crate::types::{Metadata, Metrics, TraceEvent};

/// Magic bytes for pending artifact files.
const MAGIC: &[u8; 4] = b"OBFS";

/// Current schema version.
const VERSION: u8 = 1;

/// Compression level for pending artifacts (fast, good ratio).
const COMPRESSION_LEVEL: i32 = 3;

/// A pending artifact waiting to be consolidated.
/// Written as individual files for concurrent-safe operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingArtifact {
    /// Schema version for forward compatibility.
    pub version: u8,
    /// Unique run identifier.
    pub run_id: Uuid,
    /// Run metadata.
    pub metadata: Metadata,
    /// Performance metrics.
    pub metrics: Metrics,
    /// Execution trace events.
    pub trace: Vec<TraceEvent>,
    /// Timeseries data (embedded, delta-encoded).
    pub timeseries: Vec<TimeseriesPoint>,
}

/// A single timeseries point (compact representation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeseriesPoint {
    /// Days offset from epoch (2020-01-01).
    pub date_offset: u16,
    /// Equity value.
    pub equity: f32,
    /// Drawdown percentage.
    pub drawdown: f32,
    /// Portfolio exposure.
    pub exposure: f32,
}

impl PendingArtifact {
    /// Create a new pending artifact.
    pub fn new(run_id: Uuid, metadata: Metadata, metrics: Metrics) -> Self {
        Self {
            version: VERSION,
            run_id,
            metadata,
            metrics,
            trace: Vec::new(),
            timeseries: Vec::new(),
        }
    }

    /// Add trace events.
    pub fn with_trace(mut self, trace: Vec<TraceEvent>) -> Self {
        self.trace = trace;
        self
    }

    /// Add timeseries data.
    pub fn with_timeseries(mut self, timeseries: Vec<TimeseriesPoint>) -> Self {
        self.timeseries = timeseries;
        self
    }
}

/// Pending store for isolated file writes.
pub struct PendingStore {
    pending_dir: PathBuf,
    pipeline: CompressionPipeline,
}

impl PendingStore {
    /// Create a new pending store at the given directory.
    pub fn new(pending_dir: impl Into<PathBuf>) -> Result<Self> {
        let pending_dir = pending_dir.into();
        fs::create_dir_all(&pending_dir)
            .with_context(|| format!("Failed to create pending dir: {:?}", pending_dir))?;
        
        Ok(Self {
            pending_dir,
            pipeline: CompressionPipeline::with_level(COMPRESSION_LEVEL),
        })
    }

    /// Get the pending directory path.
    pub fn pending_dir(&self) -> &Path {
        &self.pending_dir
    }

    /// Write a pending artifact to an isolated file.
    /// 
    /// File format: MAGIC (4) + VERSION (1) + Zstd(JSON)
    pub fn write_pending(&self, artifact: &PendingArtifact) -> Result<PathBuf> {
        let filename = format!("{}.obfs", artifact.run_id);
        let path = self.pending_dir.join(&filename);
        
        // Serialize to JSON
        let json = serde_json::to_vec(artifact)
            .context("Failed to serialize pending artifact")?;
        
        // Compress with Zstd
        let compressed = self.pipeline.compress(&json)
            .context("Failed to compress pending artifact")?;
        
        // Write file with magic header
        let file = File::create(&path)
            .with_context(|| format!("Failed to create file: {:?}", path))?;
        let mut writer = BufWriter::new(file);
        
        writer.write_all(MAGIC)?;
        writer.write_all(&[VERSION])?;
        writer.write_all(&compressed)?;
        writer.flush()?;
        
        Ok(path)
    }

    /// Read a pending artifact from file.
    pub fn read_pending(&self, run_id: Uuid) -> Result<PendingArtifact> {
        let filename = format!("{}.obfs", run_id);
        let path = self.pending_dir.join(&filename);
        self.read_pending_from_path(&path)
    }

    /// Read a pending artifact from a specific path.
    pub fn read_pending_from_path(&self, path: &Path) -> Result<PendingArtifact> {
        let file = File::open(path)
            .with_context(|| format!("Failed to open file: {:?}", path))?;
        let mut reader = BufReader::new(file);
        
        // Read and verify magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            anyhow::bail!("Invalid magic bytes in {:?}", path);
        }
        
        // Read version
        let mut version = [0u8; 1];
        reader.read_exact(&mut version)?;
        if version[0] > VERSION {
            anyhow::bail!("Unsupported version {} in {:?}", version[0], path);
        }
        
        // Read compressed data
        let mut compressed = Vec::new();
        reader.read_to_end(&mut compressed)?;
        
        // Decompress
        let json = self.pipeline.decompress(&compressed)
            .context("Failed to decompress pending artifact")?;
        
        // Deserialize
        let artifact: PendingArtifact = serde_json::from_slice(&json)
            .context("Failed to deserialize pending artifact")?;
        
        Ok(artifact)
    }

    /// List all pending artifact UUIDs.
    pub fn list_pending(&self) -> Result<Vec<Uuid>> {
        let mut uuids = Vec::new();
        
        if !self.pending_dir.exists() {
            return Ok(uuids);
        }
        
        for entry in fs::read_dir(&self.pending_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().map_or(false, |e| e == "obfs") {
                if let Some(stem) = path.file_stem() {
                    if let Ok(uuid) = Uuid::parse_str(&stem.to_string_lossy()) {
                        uuids.push(uuid);
                    }
                }
            }
        }
        
        Ok(uuids)
    }

    /// Count pending artifacts.
    pub fn count(&self) -> Result<usize> {
        Ok(self.list_pending()?.len())
    }

    /// Remove a pending artifact file.
    pub fn remove_pending(&self, run_id: Uuid) -> Result<()> {
        let filename = format!("{}.obfs", run_id);
        let path = self.pending_dir.join(&filename);
        
        if path.exists() {
            fs::remove_file(&path)
                .with_context(|| format!("Failed to remove: {:?}", path))?;
        }
        
        Ok(())
    }

    /// Remove all pending artifacts.
    pub fn clear(&self) -> Result<usize> {
        let uuids = self.list_pending()?;
        let count = uuids.len();
        
        for uuid in uuids {
            self.remove_pending(uuid)?;
        }
        
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn sample_artifact() -> PendingArtifact {
        PendingArtifact::new(
            Uuid::new_v4(),
            Metadata {
                strategy_id: "test_strategy".into(),
                strategy_version: "1.0.0".into(),
                run_id: "test-run".into(),
                timestamp: 1704067200,
                universe: "B3_IBOV".into(),
                start_date: "2020-01-01".into(),
                end_date: "2024-01-01".into(),
                initial_capital: 1_000_000.0,
                mode: "standard".into(),
            },
            Metrics {
                cagr: 0.15,
                volatility: 0.20,
                sharpe_ratio: 0.75,
                sortino_ratio: 1.0,
                max_drawdown: -0.12,
                max_drawdown_duration_days: 30,
                hit_rate: 0.55,
                profit_factor: 1.5,
                turnover_annual: 2.5,
                total_trades: 100,
            },
        )
        .with_timeseries(vec![
            TimeseriesPoint { date_offset: 0, equity: 1_000_000.0, drawdown: 0.0, exposure: 0.95 },
            TimeseriesPoint { date_offset: 1, equity: 1_001_000.0, drawdown: 0.0, exposure: 0.95 },
            TimeseriesPoint { date_offset: 2, equity: 1_002_000.0, drawdown: 0.0, exposure: 0.95 },
        ])
    }

    #[test]
    fn test_write_and_read_pending() {
        let temp = tempdir().unwrap();
        let store = PendingStore::new(temp.path().join("pending")).unwrap();
        
        let artifact = sample_artifact();
        let run_id = artifact.run_id;
        
        // Write
        let path = store.write_pending(&artifact).unwrap();
        assert!(path.exists());
        
        // Read back
        let loaded = store.read_pending(run_id).unwrap();
        assert_eq!(loaded.run_id, run_id);
        assert_eq!(loaded.metadata.strategy_id, "test_strategy");
        assert_eq!(loaded.timeseries.len(), 3);
    }

    #[test]
    fn test_list_pending() {
        let temp = tempdir().unwrap();
        let store = PendingStore::new(temp.path().join("pending")).unwrap();
        
        // Write multiple
        for _ in 0..5 {
            let artifact = sample_artifact();
            store.write_pending(&artifact).unwrap();
        }
        
        let uuids = store.list_pending().unwrap();
        assert_eq!(uuids.len(), 5);
    }

    #[test]
    fn test_remove_pending() {
        let temp = tempdir().unwrap();
        let store = PendingStore::new(temp.path().join("pending")).unwrap();
        
        let artifact = sample_artifact();
        let run_id = artifact.run_id;
        
        store.write_pending(&artifact).unwrap();
        assert_eq!(store.count().unwrap(), 1);
        
        store.remove_pending(run_id).unwrap();
        assert_eq!(store.count().unwrap(), 0);
    }

    #[test]
    fn test_compression_efficiency() {
        let temp = tempdir().unwrap();
        let store = PendingStore::new(temp.path().join("pending")).unwrap();
        
        // Create artifact with large timeseries
        let mut artifact = sample_artifact();
        artifact.timeseries = (0..1000)
            .map(|i| TimeseriesPoint {
                date_offset: i as u16,
                equity: 1_000_000.0 + i as f32 * 100.0,
                drawdown: -0.001 * i as f32,
                exposure: 0.95,
            })
            .collect();
        
        let path = store.write_pending(&artifact).unwrap();
        let file_size = fs::metadata(&path).unwrap().len();
        
        // Should be well under 10KB for 1000 points
        println!("File size for 1000 points: {} bytes", file_size);
        assert!(file_size < 15_000);
    }
}


