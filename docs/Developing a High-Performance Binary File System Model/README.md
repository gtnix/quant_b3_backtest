# OBFS Prototype - Optimized Binary File System

This is a functional prototype of the Optimized Binary File System (OBFS) for high-performance backtesting artifact storage.

## Quick Start

### Build the Project

```bash
cargo build --release
```

### Run Tests

```bash
cargo test
```

### Run Benchmarks

```bash
cargo bench
```

The benchmark results will be available in `target/criterion/`.

## Project Structure

```
prototype/
├── Cargo.toml         # Dependencies and project configuration
├── benches/           # Performance benchmarks (Criterion)
│   ├── compression_benchmark.rs
│   └── read_write_benchmark.rs
└── src/               # OBFS source code
    ├── lib.rs         # Main module and public interface
    ├── types.rs       # Core data structures
    ├── integrity.rs   # Hashing engine (BLAKE3 + XXH3)
    ├── compression.rs # Compression pipeline (Zstd)
    ├── writer.rs      # Artifact writing logic
    └── reader.rs      # Artifact reading logic
```

## Usage Example

```rust
use obfs_prototype::*;

// Initialize OBFS
let config = ObfsConfig {
    root_path: "./artifacts".to_string(),
    compression_level: 3,
    ..Default::default()
};

let obfs = Obfs::with_config(config);
obfs.initialize().unwrap();

// Write an artifact
let mut writer = obfs.writer();
let artifact = BacktestArtifact { /* ... */ };
writer.write_artifact(&artifact).unwrap();

// Read an artifact
let reader = obfs.reader();
let loaded_artifact = reader.read_artifact(artifact.uuid).unwrap();
```

## Performance Targets

| Metric | Target |
|---|---|
| Storage Reduction | > 10x |
| Read Latency | < 100 µs |
| Write Throughput | > 10,000/s |
| Read Throughput (16 threads) | > 100,000/s |
| Integrity Validation | > 50 GB/s |

## Dependencies

- **rkyv**: Zero-copy serialization
- **blake3**: Cryptographic hashing
- **xxhash-rust**: Fast checksums
- **zstd**: Compression
- **heed**: LMDB wrapper for metadata
- **parquet**: Columnar storage
- **memmap2**: Memory mapping

## License

This prototype is part of the OBFS research and development project.
