# OBFS Integration Guide

**Version**: 1.0.0  
**Last Updated**: 2026-01-05  
**Status**: Production

## Overview

OBFS (Optimized Binary File System) is a high-performance storage system for backtest artifacts with a **two-phase write strategy** for concurrent-safe operations.

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
│   ├── pending_store.rs    # Phase 1: Isolated pending artifact storage
│   ├── consolidator.rs     # Phase 2: Streaming batch consolidation
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

---

## Two-Phase Write Strategy

OBFS uses a two-phase write strategy to handle concurrent writes from multiple workers without data corruption.

### The Problem

Parquet files do not support concurrent writes. When 14+ workers attempt to write to the same `timeseries.parquet`, file corruption occurs (`EOF: Invalid page header`).

### The Solution

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     TWO-PHASE WRITE STRATEGY                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PHASE 1: ISOLATED WRITES (Concurrent)                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                   │
│  │ Worker 1 │ │ Worker 2 │ │ Worker 3 │ │ Worker N │                   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘                   │
│       │            │            │            │                          │
│       ▼            ▼            ▼            ▼                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    pending/ directory                            │   │
│  │  <uuid1>.obfs  <uuid2>.obfs  <uuid3>.obfs  ...  <uuidN>.obfs   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  PHASE 2: CONSOLIDATION (Single-Threaded)                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                       Consolidator                               │   │
│  │  - Reads all .obfs files in batches (5,000 artifacts/batch)     │   │
│  │  - Writes streaming Parquet row groups                           │   │
│  │  - Updates LMDB index                                            │   │
│  │  - Cleans up pending files                                       │   │
│  └────────────────────────────────┬────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   consolidated/ directory                        │   │
│  │  ├── data/timeseries.parquet  (columnar, Zstd compressed)       │   │
│  │  └── lmdb/data.mdb            (index for O(1) lookups)          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: PendingStore

Each worker writes isolated `.obfs` files during concurrent execution.

### PendingArtifact Struct

```rust
use obfs::{PendingArtifact, PendingStore};
use uuid::Uuid;

// Create pending artifact
let artifact = PendingArtifact::new(
    Uuid::new_v4(),
    metadata,  // obfs::Metadata
    metrics,   // obfs::Metrics
)
.with_trace(trace_events)
.with_timeseries(timeseries_points);

// Write to pending directory
let pending_store = PendingStore::new(&pending_dir)?;
let path = pending_store.write_pending(&artifact)?;
// Result: pending/<uuid>.obfs (Zstd compressed)
```

### TimeseriesPoint (Delta-Encoded)

```rust
pub struct TimeseriesPoint {
    pub date_offset: u16,  // Days since 2020-01-01 (compact)
    pub equity: f32,
    pub drawdown: f32,
    pub exposure: f32,
}
```

---

## Phase 2: Consolidator

After all workers complete, a single-threaded consolidator merges pending files.

### Streaming Batch Processing

To avoid Arrow offset overflow (>2GB strings), the consolidator processes artifacts in batches:

```rust
use obfs::{Consolidator, ConsolidationStats};

let consolidator = Consolidator::new(&pending_dir, &output_dir);
let stats: ConsolidationStats = consolidator.consolidate()?;

println!("Processed: {} artifacts", stats.artifacts_processed);
println!("Timeseries rows: {}", stats.timeseries_rows);
println!("Duration: {}ms", stats.duration_ms);
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_ARTIFACTS` | 5,000 | Artifacts per batch (memory-safe) |
| `MAX_CONSOLIDATE_THRESHOLD` | 100,000 | Skip consolidation above this (keep pending) |
| Row group size | 500,000 | Parquet row group max rows |
| Compression | Zstd | Dictionary encoding for UUIDs |

### Output Structure

```
consolidated/
├── data/
│   └── timeseries.parquet    # Columnar storage (Zstd + dictionary)
└── lmdb/
    ├── data.mdb              # Index: UUID → offset
    └── lock.mdb
```

### Parquet Schema

```
backtest_uuid: Utf8 (dictionary encoded)
date_offset: UInt16
equity: Float32
drawdown: Float32
exposure: Float32
```

---

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

---

## Commands

```bash
# Build
cargo build -p obfs

# Test (including real data tests)
cargo test -p obfs

# Benchmarks
cargo bench -p obfs
```

---

## Design Decisions (Non-Negotiable)

1. **Zero-copy serialization**: `rkyv` for direct memory access
2. **Dual-hashing integrity**: XXH3 (fast) + BLAKE3 (cryptographic), both stored in LMDB
3. **Multi-stage compression**: Delta encoding + Zstd for time-series data
4. **LMDB metadata store**: Fast key-value lookups (O(1)) with XXH3/BLAKE3
5. **Memory-mapped reads**: `memmap2` for zero-copy access
6. **Fixed-size types**: All struct fields use fixed-size types (no `usize`)
7. **Automatic file rotation**: Data files rotate when `max_file_size` (default 1GB) is reached
8. **Parquet columnar storage**: TimeSeriesStore for high-compression time-series data
9. **Two-phase writes**: Isolated pending + single-threaded consolidation for concurrency safety
10. **Streaming batches**: 5K artifacts per batch to prevent Arrow offset overflow

---

## Production Performance (v1.0)

Benchmark from 5-hour overnight campaign (2026-01-05):

| Metric | Value |
|--------|-------|
| **Strategies generated** | 176,672 |
| **Storage/strategy** | 8.01 KB |
| **Throughput** | 210 strategies/s |
| **Time/strategy** | 4.75 ms |
| **Compression ratio** | 7.1x vs Legacy JSON |

### Per-Run Consolidation

| Run | Artifacts | Rows | Batches | Parquet Size | Time |
|-----|-----------|------|---------|--------------|------|
| 1 | 59,325 | 73.8M | 12 | 355 MB | 95s |
| 2 | 58,331 | 72.6M | 12 | 432 MB | 97s |
| 3 | 59,016 | 73.4M | 12 | 370 MB | 97s |

---

## Troubleshooting

### Arrow Offset Overflow

**Error**: `offset overflow in arrow-array StringArray`

**Cause**: Too many UUID strings (>2GB) loaded into a single Arrow StringArray.

**Solution**: OBFS v1.0+ uses streaming batch consolidation (5K artifacts/batch). Each batch is written as a separate Parquet row group, freeing memory between batches.

### Parquet EOF Invalid Page Header

**Error**: `EOF: Invalid page header`

**Cause**: Concurrent writes to Parquet from multiple workers.

**Solution**: Use two-phase write strategy. Workers write to `pending/`, consolidator merges after all workers complete.

### LMDB Map Full

**Error**: `MDB_MAP_FULL`

**Cause**: LMDB map size exceeded.

**Solution**: Increase `max_map_size` in `ObfsConfig` (default: 10GB).

---

## Dependencies

- `rkyv = "0.8"` - Zero-copy serialization
- `blake3 = "1.5"` - Cryptographic hashing
- `xxhash-rust = "0.8"` - Fast checksums
- `zstd = "0.13"` - Compression
- `heed = "0.20"` - LMDB wrapper
- `arrow = "57"`, `parquet = "57"` - Columnar storage
- `anyhow = "1.0"` - Error handling
- `tracing = "0.1"` - Logging

---

## Reference Specification

Based on prototype from:
- `docs/Developing a High-Performance Binary File System Model/`
