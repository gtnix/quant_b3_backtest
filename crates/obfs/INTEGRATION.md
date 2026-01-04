# OBFS Integration Guide

## Overview

OBFS (Optimized Binary File System) is a high-performance storage system for backtest artifacts, now integrated as an internal crate in this workspace.

## Location

```
crates/obfs/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Public API
│   ├── types.rs            # Core data structures (rkyv-compatible)
│   ├── writer.rs           # Write path (serialize → compress → store) with auto-rotation
│   ├── reader.rs           # Read path (mmap → decompress → deserialize) with XXH3 validation
│   ├── compression.rs      # Multi-stage compression pipeline (Delta + Zstd)
│   ├── timeseries.rs       # Parquet-based columnar time-series storage
│   ├── integrity.rs        # XXH3 + BLAKE3 integrity validation
│   ├── store/
│   │   └── mod.rs          # LMDB-based metadata store (with XXH3)
│   └── adapters/
│       ├── mod.rs
│       └── project.rs      # RunMetadata/RunMetrics → OBFS converters
├── benches/
│   ├── compression_benchmark.rs
│   └── read_write_benchmark.rs
└── tests/
    ├── smoke_roundtrip.rs      # Basic write/read/validate tests
    └── real_artifacts_test.rs  # Tests with real data from output/scg/
```

## Usage

### Basic Example

```rust
use obfs::{Obfs, ObfsConfig, BacktestArtifact, Metrics, Metadata};
use uuid::Uuid;

// Initialize OBFS
let config = ObfsConfig {
    root_path: "./artifacts".to_string(),
    compression_level: 3,
    enable_blake3: true,
    enable_xxh3: true,
    ..Default::default()
};

let obfs = Obfs::with_config(config);
obfs.initialize()?;

// Create and write artifact
let uuid = Uuid::new_v4();
let artifact = BacktestArtifact {
    uuid_bytes: *uuid.as_bytes(),
    metadata: Metadata { /* ... */ },
    metrics: Metrics { /* ... */ },
    // ...
};

let mut writer = obfs.writer();
let metadata = writer.write_artifact(&artifact)?;

// Read artifact back
let reader = obfs.reader();
let recovered = reader.read_artifact(uuid)?;

// Fast metrics-only read (from LMDB, no decompression)
let metrics = reader.get_metrics(uuid)?;
```

### Converting Existing Artifacts

```rust
use obfs::adapters::{ProjectArtifactLoader, ProjectRunMetadata, ProjectRunMetrics};

// Load from project's directory structure
let (metadata, metrics, trace) = ProjectArtifactLoader::load_from_dir(backtest_dir)?;
let ts_rows = ProjectArtifactLoader::count_timeseries_rows(backtest_dir);

// Convert to OBFS format
let artifact = ProjectArtifactLoader::convert_to_obfs(&metadata, &metrics, &trace, ts_rows);

// Write to OBFS
let mut writer = obfs.writer();
writer.write_artifact(&artifact)?;
```

## Commands

```bash
# Build
cargo build -p obfs

# Test (including real data tests)
cargo test -p obfs

# Benchmarks
cargo bench -p obfs
```

## Design Decisions (Non-Negotiable)

1. **Zero-copy serialization**: `rkyv` for direct memory access
2. **Dual-hashing integrity**: XXH3 (fast) + BLAKE3 (cryptographic), both stored in LMDB
3. **Multi-stage compression**: Delta encoding + Zstd for time-series data
4. **LMDB metadata store**: Fast key-value lookups (O(1)) with XXH3/BLAKE3
5. **Memory-mapped reads**: `memmap2` for zero-copy access
6. **Fixed-size types**: All struct fields use fixed-size types (no `usize`)
7. **Automatic file rotation**: Data files rotate when `max_file_size` (default 1GB) is reached
8. **Parquet columnar storage**: TimeSeriesStore for high-compression time-series data

## New Features (v0.2)

### XXH3 Integrity Validation
- XXH3 checksum now stored in LMDB alongside BLAKE3 hash
- Read path validates XXH3 and returns error if mismatch detected

### Automatic File Rotation
- Data files automatically rotate when `max_file_size` is reached
- Files named `data_0000.obfs`, `data_0001.obfs`, etc.
- Detection of existing files on startup

### TimeSeriesStore (Parquet)
```rust
// Write time-series to Parquet
let mut obfs = Obfs::with_config(config);
let ts_store = obfs.timeseries_store_mut();
let points = vec![TimeSeriesPoint { ... }];
ts_store.write_timeseries(uuid, &points)?;

// Read back
let points = obfs.timeseries_store().read_timeseries(uuid)?;
```

### Specialized Compression
```rust
use obfs::{TimeSeriesCompressor, DeltaEncoder, ColumnarCompressor};

// Delta + Zstd for smooth time-series
let compressor = TimeSeriesCompressor::new(3);
let (compressed, stats) = compressor.compress_f32_with_stats(&equity_values)?;

// Multi-column compression
let columnar = ColumnarCompressor::new(3);
let compressed = columnar.compress_columns(&[&equity, &drawdown, &exposure])?;
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Storage reduction | >10x vs JSON |
| Read latency | <100 µs |
| Write throughput | >10,000/s |
| Read throughput | >100,000/s (16 threads) |
| Integrity validation | >50 GB/s |

## Dependencies Added

- `rkyv = "0.8"` - Zero-copy serialization
- `blake3 = "1.5"` - Cryptographic hashing
- `xxhash-rust = "0.8"` - Fast checksums
- `zstd = "0.13"` - Compression
- `heed = "0.20"` - LMDB wrapper
- `arrow = "57"`, `parquet = "57"` - Columnar storage
- `anyhow = "1.0"` - Error handling

## Reference Specification

Based on prototype from:
- `docs/Developing a High-Performance Binary File System Model/`

