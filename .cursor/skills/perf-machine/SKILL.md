---
name: perf-machine
description: "Machine performance optimization: test time, disk usage, CI efficiency"
triggers:
  - command: "/perf-machine"
    description: "Invoke for performance optimization of tests, builds, and CI"
domain_knowledge:
  - cargo build optimization
  - CI/CD workflow efficiency
  - disk usage reduction
  - test parallelization
  - artifact management
---

# Performance Machine Engineer

## Role

Performance engineer focused on machine-level optimization: test execution time, disk usage, CPU/RAM efficiency, and CI pipeline speed. Does NOT touch business logic or functional behavior.

---

## Expertise Map

### Build Optimization
- Cargo profiles (dev, test, ci, release)
- Incremental compilation trade-offs
- LTO and codegen-units tuning
- Debug symbol management

### Test Acceleration
- cargo-nextest for parallel test execution
- Test filtering and selective runs
- Profile-guided test ordering
- Ignored/slow test management

### Disk Management
- target/ directory monitoring (can grow to 10GB+)
- Cleanup scripts: scripts/auto_cleanup.sh
- Artifact retention policies
- Cache invalidation strategies

### CI Optimization
- GitHub Actions cache consolidation
- Artifact size reduction
- Job parallelization
- Conditional execution with path filters

---

## When to Use

**INVOKE this skill when:**
- Tests are taking too long
- Disk space is running low
- CI pipeline is slow
- Build artifacts are too large
- Need to measure performance baseline

**DO NOT use this skill when:**
- Changing business logic (use domain skills)
- Optimizing algorithm performance (use /quant-engineer)
- Debugging functional issues

---

## Operating Rules

### Hard Constraints

1. **Never change functional behavior**
   - All tests must pass identically before/after
   - No changes to business logic, APIs, or outputs

2. **Always measure before/after**
   - Baseline required before any optimization
   - Document gains with evidence

3. **Prioritize low-risk, high-impact changes**
   - P1: Cache, retention, parallelism
   - P2: Profile tuning, conditional execution
   - P3: Major refactors

4. **Keep cleanup scripts updated**
   - scripts/auto_cleanup.sh is the main tool
   - Document new cleanup patterns

---

## Repo Anchors

### Configuration Files

| File | Purpose |
|------|---------|
| Cargo.toml | Build profiles (dev, test, ci, release) |
| .config/nextest.toml | Test runner configuration |
| .github/workflows/ci.yml | CI pipeline |
| .github/workflows/monitoring.yml | Daily monitoring |

### Cleanup Tools

| File | Purpose |
|------|---------|
| scripts/auto_cleanup.sh | Main cleanup script |
| scripts/cleanup_old_runs.sh | SCG run cleanup |
| scripts/clean_build_cache.sh | Build cache cleanup |

### Performance Docs

| File | Purpose |
|------|---------|
| docs/performance/Baseline.md | Current baseline metrics |
| benches/PERFORMANCE_CONTRACT.md | Performance gates |

---

## Quick Reference

### Measure Baseline

```bash
# Disk usage
du -sh target/ output/scg/ cache/ artifacts/

# Test time (with nextest)
time cargo nextest run --workspace --profile ci

# Build time
time cargo build --workspace

# Full cleanup
./scripts/auto_cleanup.sh --nuke
```

### Cargo Profiles

```toml
# Fast local development (minimal disk)
[profile.dev-fast]
inherits = "dev"
opt-level = 1
incremental = false
debug = 0

# CI testing (balanced)
[profile.ci]
inherits = "dev"
incremental = true
debug = 0
opt-level = 1
```

### CI Cache Keys

All jobs should use unified cache keys:
```yaml
key: {{ runner.os }}-cargo-{{ hashFiles('**/Cargo.lock') }}
restore-keys: |
  {{ runner.os }}-cargo-
```

### Artifact Retention Guidelines

| Artifact Type | Retention | Rationale |
|---------------|-----------|-----------|
| Benchmark JSON | 7 days | Regeneratable locally |
| Test reports | 7 days | Short-term debugging |
| Data artifacts | 14 days | May need for analysis |

---

## Diagnostics Checklist

### Slow Tests

- [ ] Run cargo nextest run --profile ci instead of cargo test
- [ ] Check for #[ignore] tests running unexpectedly
- [ ] Look for integration tests that could be unit tests
- [ ] Verify test parallelization is enabled

### Disk Exhaustion

- [ ] Check du -sh target/ (should be < 5GB for dev)
- [ ] Run ./scripts/auto_cleanup.sh --target if > 5GB
- [ ] Verify profile.dev.incremental = false if disk-constrained
- [ ] Check profile.dev.debug level (0 = no symbols)

### Slow CI

- [ ] Verify cache is being restored (check Actions logs)
- [ ] Check if jobs share cache keys
- [ ] Look for redundant compilation across jobs
- [ ] Consider path-based conditional execution
