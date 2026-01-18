# Performance Optimization Report - After

**Date**: 2026-01-18  
**Commit**: (post-optimization)

## Changes Implemented

### P1 - Low Risk, High Impact

| Change | File(s) | Expected Gain |
|--------|---------|---------------|
| Unified cargo cache keys | `.github/workflows/ci.yml` | 30-50% build time reduction |
| Reduced benchmark artifacts | `.github/workflows/ci.yml` | ~80% artifact size reduction |
| Reduced retention days (30 -> 7) | `.github/workflows/monitoring.yml` | ~75% storage reduction |
| Added cargo-nextest | `.github/workflows/ci.yml`, `.config/nextest.toml` | 20-40% faster tests |
| Added restore-keys fallback | All workflow cache steps | Better cache hit rate |

### P2 - Medium Risk

| Change | File(s) | Expected Gain |
|--------|---------|---------------|
| Added `[profile.ci]` | `Cargo.toml` | Incremental builds in CI |
| Clippy with `--workspace` | `.github/workflows/ci.yml` | Consistent linting |

## Files Modified

1. **`.github/workflows/ci.yml`**
   - Unified cache keys across all jobs
   - Added `restore-keys` for partial cache matches
   - Installed `cargo-nextest` via `taiki-e/install-action`
   - Changed `cargo test` to `cargo nextest run --profile ci`
   - Reduced benchmark artifact paths (removed HTML reports)
   - Reduced retention from 14 to 7 days

2. **`.github/workflows/monitoring.yml`**
   - Unified cache key with main CI
   - Reduced artifact retention from 30 to 7 days

3. **`Cargo.toml`**
   - Added `[profile.ci]` with incremental compilation enabled

4. **`.config/nextest.toml`** (NEW)
   - Nextest configuration with CI and release profiles

5. **`.cursor/skills/perf-machine/SKILL.md`** (NEW)
   - Performance optimization runbook

6. **`docs/performance/Baseline.md`** (NEW)
   - Baseline metrics documentation

## Expected Metrics (Post-Implementation)

| Metric | Before | After (Expected) | Change |
|--------|--------|------------------|--------|
| CI cache sharing | 4 separate caches | 1 unified cache | -75% redundancy |
| Benchmark artifact size | ~50 MB (HTML) | ~1 MB (JSON only) | -98% |
| Artifact retention | 14-30 days | 7 days | -50-75% storage |
| Test execution | `cargo test` | `cargo nextest` | -20-40% time |
| Build time (cached) | Full rebuild | Incremental | -40-60% time |

## Verification

To verify the optimizations work:

1. **Local test with nextest**:
   ```bash
   cargo install cargo-nextest
   cargo nextest run --workspace --profile ci
   ```

2. **Check CI runs** (after merge):
   - Verify cache is being restored in all jobs
   - Check artifact sizes in Actions UI
   - Compare total workflow time

3. **Disk usage**:
   ```bash
   du -sh target/
   # Should grow more slowly with incremental + no debug symbols
   ```

## Rollback Plan

If issues arise:

1. **Nextest fails**: Revert to `cargo test --workspace` in ci.yml
2. **Cache issues**: Remove `restore-keys` to force exact match
3. **Profile issues**: Remove `[profile.ci]` from Cargo.toml

## Next Steps (Deferred)

- **sccache**: Shared compilation cache (requires S3/Redis)
- **Path filters**: Conditional job execution with `dorny/paths-filter`
- **Matrix builds**: Parallel jobs for different crates
