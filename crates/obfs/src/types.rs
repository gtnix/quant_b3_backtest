//! Core data structures for OBFS (Optimized Binary File System)

use rkyv::{Archive, Deserialize, Serialize};
use serde::{Deserialize as SerdeDeserialize, Serialize as SerdeSerialize};
use uuid::Uuid;

/// Represents a complete backtest artifact
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
#[rkyv(derive(Debug))]
pub struct BacktestArtifact {
    /// UUID stored as bytes for rkyv compatibility
    pub uuid_bytes: [u8; 16],
    pub metadata: Metadata,
    pub metrics: Metrics,
    pub timeseries_ref: TimeseriesReference,
    pub trace: Vec<TraceEvent>,
    pub integrity: IntegritySeal,
}

impl BacktestArtifact {
    /// Get UUID from bytes
    pub fn uuid(&self) -> Uuid {
        Uuid::from_bytes(self.uuid_bytes)
    }

    /// Set UUID from Uuid type
    pub fn set_uuid(&mut self, uuid: Uuid) {
        self.uuid_bytes = *uuid.as_bytes();
    }

    /// Create with UUID
    pub fn with_uuid(uuid: Uuid) -> Self {
        Self {
            uuid_bytes: *uuid.as_bytes(),
            metadata: Metadata::default(),
            metrics: Metrics::default(),
            timeseries_ref: TimeseriesReference::default(),
            trace: Vec::new(),
            integrity: IntegritySeal::default(),
        }
    }
}

/// Metadata for a backtest run
#[derive(Archive, Serialize, Deserialize, Debug, Clone, SerdeSerialize, SerdeDeserialize, Default)]
#[rkyv(derive(Debug))]
pub struct Metadata {
    pub strategy_id: String,
    pub strategy_version: String,
    pub run_id: String,
    pub timestamp: i64,
    pub universe: String,
    pub start_date: String,
    pub end_date: String,
    pub initial_capital: f64,
    pub mode: String,
}

/// Performance metrics for a backtest
#[derive(Archive, Serialize, Deserialize, Debug, Clone, SerdeSerialize, SerdeDeserialize, Default)]
#[rkyv(derive(Debug))]
pub struct Metrics {
    pub cagr: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub max_drawdown_duration_days: i32,
    pub hit_rate: f64,
    pub profit_factor: f64,
    pub turnover_annual: f64,
    pub total_trades: i32,
}

/// Reference to time-series data stored in Parquet
#[derive(Archive, Serialize, Deserialize, Debug, Clone, Default)]
#[rkyv(derive(Debug))]
pub struct TimeseriesReference {
    pub parquet_file: String,
    pub row_group: u32,
    pub start_row: u64,
    pub num_rows: u64,
}

/// A single trace event in the backtest execution
#[derive(Archive, Serialize, Deserialize, Debug, Clone, SerdeSerialize, SerdeDeserialize)]
#[rkyv(derive(Debug))]
pub struct TraceEvent {
    pub timestamp: i64,
    pub event_type: String,
    pub message: String,
}

/// Integrity seal containing checksums and hashes
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
#[rkyv(derive(Debug))]
pub struct IntegritySeal {
    /// XXH3 checksum for fast validation (8 bytes)
    pub block_xxh3: u64,
    /// BLAKE3 hash for cryptographic integrity (32 bytes)
    pub artifact_blake3: [u8; 32],
    /// Timestamp when the seal was created
    pub sealed_at: i64,
}

impl Default for IntegritySeal {
    fn default() -> Self {
        Self {
            block_xxh3: 0,
            artifact_blake3: [0u8; 32],
            sealed_at: 0,
        }
    }
}

/// Time-series data point (stored in Parquet)
#[derive(Debug, Clone, SerdeSerialize, SerdeDeserialize)]
pub struct TimeseriesRow {
    pub backtest_uuid: Uuid,
    pub date_offset: i32,
    pub equity: f32,
    pub drawdown: f32,
    pub exposure: f32,
    pub vol_exante: Option<f32>,
    pub vol_expost: Option<f32>,
    pub dividend_cashflow: Option<f32>,
    pub dividend_cumulative: Option<f32>,
}

/// Metadata stored in LMDB for fast lookups
#[derive(Debug, Clone, SerdeSerialize, SerdeDeserialize)]
pub struct ArtifactMetadata {
    pub uuid: Uuid,
    pub artifact_location: ArtifactLocation,
    pub blake3_hash: [u8; 32],
    /// XXH3 checksum of compressed data for fast integrity validation
    pub xxh3_checksum: u64,
    pub metrics: Metrics,
    pub created_at: i64,
}

/// Physical location of an artifact in storage
#[derive(Debug, Clone, SerdeSerialize, SerdeDeserialize)]
pub struct ArtifactLocation {
    pub file_path: String,
    pub offset: u64,
    pub size: u64,
}

/// Statistics for compression and storage
#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub original_size: u64,
    pub compressed_size: u64,
    pub compression_ratio: f64,
    pub compression_time_ms: u64,
    pub decompression_time_ms: u64,
}

impl CompressionStats {
    pub fn new(original_size: u64, compressed_size: u64) -> Self {
        let compression_ratio = if compressed_size > 0 {
            original_size as f64 / compressed_size as f64
        } else {
            0.0
        };
        Self {
            original_size,
            compressed_size,
            compression_ratio,
            compression_time_ms: 0,
            decompression_time_ms: 0,
        }
    }
}
