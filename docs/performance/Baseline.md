# Performance Baseline Report

**Date**: 2026-01-18  
**Commit**: (pre-optimization)

## Local Environment

| Metric | Value | Notes |
|--------|-------|-------|
| `target/` size (post-build) | 9.0 GB | Caused disk exhaustion during test runs |
| `output/scg/` size | 467 MB | Campaign outputs |
| `.cache/` size | 19 MB | Market data cache |
| `artifacts/` size | 328 KB | Data integrity reports |
| Test compile time (no-run) | ~81s | `cargo test --workspace --no-run` |
| Disk available | 37 GB total, 6.9 GB free (81% used) | Critical - tests can exhaust disk |

## Identified Issues

1. **Disk exhaustion during tests**: The `target/` directory grew from 1.5GB to 9GB during test compilation, exhausting disk space
2. **No incremental compilation**: `profile.dev.incremental = false` means full rebuilds
3. **Debug symbols**: `profile.dev.debug = 1` still generates some symbols

## CI Workflow (from ci.yml analysis)

| Job | Cache Key | Notes |
|-----|-----------|-------|
| check | `cargo-$hash` | Main test job |
| real-data-tests | `cargo-realdata-$hash` | Separate cache (redundant) |
| bench | `cargo-bench-$hash` | Separate cache (redundant) |
| calendar-check | `cargo-calendar-$hash` | Separate cache (redundant) |

**Problem**: 4 different cache keys = 4x redundant compilation

## Artifact Retention (from workflows)

| Artifact | Retention | Size (est.) |
|----------|-----------|-------------|
| benchmark-results | 14 days | 10-50 MB (includes HTML reports) |
| calendar-reports | 7 days | < 1 MB |
| freshness-report | 30 days | < 1 KB |
| integrity-report | 30 days | < 1 KB |
| fx-rates | 30 days | Variable |

## Crate Count

- **Workspace crates**: 17
- **Test files**: ~245 with 2185+ test annotations
- **Benchmark files**: 14

## Next Steps

See optimization plan for improvements targeting:
- 30% reduction in CI time
- 50% reduction in artifact storage
- Prevention of disk exhaustion
