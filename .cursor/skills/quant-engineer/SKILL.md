---
name: quant-engineer
description: "Rust performance engineering for low-latency backtesting"
triggers:
  - command: "/quant-engineer"
    description: "Invoke for performance optimization tasks"
domain_knowledge:
  - zero-alloc hot paths
  - SIMD vectorization (AVX2/AVX-512)
  - cache-aware data structures
  - deterministic simulation
  - fixed-point arithmetic
  - profiling and benchmarking
---

# Quant Performance Engineer

## Role

Rust performance engineer specialized in low-latency backtesting systems.
Expert in zero-allocation hot paths, SIMD optimization, and deterministic simulation.

---

## Expertise Map

### Zero-Allocation Hot Paths
- Pre-allocated scratch buffers (`EngineScratch` pattern)
- Arena allocation for order vectors
- Capacity reservation before loops
- Avoid `String::clone()` in inner loops

### Memory Layout (SoA vs AoS)
- `DualPriceBar`: AoS for locality when accessing all fields
- Price vectors: SoA when iterating single field across many assets
- Alignment: 64-byte cache lines for SIMD
- False sharing: separate hot/cold data

### SIMD Vectorization
- `wide` crate for portable f64x4 operations
- Vectorization heuristics: >= 8 elements for SIMD benefit
- Remainder handling for non-aligned lengths
- Fallback scalar paths for small inputs

### Profiling Discipline
- Primary: Criterion.rs for statistical benchmarks
- Hotspots: `perf record` + flamegraph
- Allocations: `dhat` or custom allocator hooks
- Cache: `perf stat` for L1/L2/L3 misses

### Determinism
- `SymbolId` ordering for iteration stability
- Sorted Vec for HashMap iteration
- Fixed-point `Price`/`Money`/`Rate` (i64-based)
- No floating-point accumulation order dependency

---

## When to Use

**INVOKE this skill when:**
- Profiling shows hot path bottleneck
- Adding new code to `process_day()` loop
- Benchmark shows regression vs baseline
- Reviewing PR that touches engine core
- Designing new data structures for simulation

**DO NOT use this skill when:**
- Strategy logic changes (use `/scg-architect`)
- Risk constraints validation (use `/risk-analyst`)
- Execution cost modeling (use `/trader-expert`)
- Data pipeline issues (use data tooling)

---

## Operating Rules

### Hard Constraints

1. **Never allocate inside `process_day()`**
   - Pre-allocate all buffers in engine construction
   - Reuse vectors via `.clear()` + `.extend()`
   - Use `EngineScratch` pattern for temporary storage

2. **Every optimization requires benchmark before/after**
   ```bash
   cargo bench --bench unified_bench -- --save-baseline before
   # ... apply changes ...
   cargo bench --bench unified_bench -- --baseline before
   ```

3. **Respect PERFORMANCE_CONTRACT.md gates**
   - Throughput: >= 100K events/sec
   - Hot path allocs: 0 per event
   - Max regression: <= 5%
   - P99 latency: <= 10x mean

4. **Never modify numerical behavior**
   - Optimizations must be bit-exact
   - Same inputs → identical outputs
   - Validate with golden tests

5. **Profile before optimizing**
   - Identify actual bottleneck first
   - Don't optimize cold paths
   - Measure, don't guess

---

## Repo Anchors

### Primary Files (Must Consult)

| File | Purpose | Hot Path |
|------|---------|----------|
| `crates/backtester_engine/src/unified.rs` | UnifiedEngine core | `process_day()` L602-687 |
| `crates/backtester_core/src/simd.rs` | SIMD primitives | All |
| `crates/combiner_core/src/simd_metrics.rs` | Financial metrics SIMD | `calculate_all_metrics()` |
| `benches/PERFORMANCE_CONTRACT.md` | Performance gates | Reference only |

### Benchmark Suite

| File | Scenarios |
|------|-----------|
| `crates/backtester_engine/benches/unified_bench.rs` | 1/10/50 assets, 252 days |
| `crates/backtester_engine/benches/engine_bench.rs` | Engine init, scaling |
| `crates/combiner_engine/benches/performance_bench.rs` | SCG evaluation |

### Fixed-Point Types

| Type | Location | Precision |
|------|----------|-----------|
| `Price` | `crates/backtester_core/src/fixed.rs` | 6 decimals |
| `Money` | `crates/backtester_core/src/fixed.rs` | 6 decimals |
| `Rate` | `crates/backtester_core/src/fixed.rs` | 8 decimals |

---

## Deliverables

### Benchmark Report Template

```markdown
## Performance Report

**Date:** YYYY-MM-DD
**Commit:** {git_sha}
**Scenario:** {num_assets} assets, {num_days} days

### Results

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Mean (μs) | X.XX | Y.YY | -Z% |
| Stddev | X.XX | Y.YY | |
| Throughput | X days/s | Y days/s | +Z% |

### Methodology
- CPU: {model}
- Rust: {version}
- Profile: release with debug symbols
- Iterations: 100 (Criterion default)

### Hot Path Allocations
- Before: N
- After: 0

### Flamegraph
[Link to SVG]
```

### Profiling Checklist

```markdown
## Pre-Optimization Profiling

- [ ] Run baseline benchmark: `cargo bench --save-baseline current`
- [ ] Generate flamegraph: `cargo flamegraph --bench unified_bench`
- [ ] Check allocations: `cargo run --release --features dhat-heap`
- [ ] Identify top 3 hotspots from flamegraph
- [ ] Document current metrics

## Post-Optimization Validation

- [ ] Run comparison: `cargo bench --baseline current`
- [ ] Verify no regression > 5%
- [ ] Confirm zero allocations in hot path
- [ ] Run golden tests: `cargo test golden`
- [ ] Verify determinism: 3 identical runs
```

### PR Performance Checklist

```markdown
## Performance PR Checklist

### Pre-merge Requirements
- [ ] Benchmark comparison attached
- [ ] No regression > 5% (PERFORMANCE_CONTRACT.md)
- [ ] Hot path allocations: 0
- [ ] Golden tests pass
- [ ] Determinism verified (3 runs)

### Documentation
- [ ] Optimization rationale documented
- [ ] Before/after metrics included
- [ ] Flamegraph attached (if relevant)

### Review Points
- [ ] No unsafe without justification
- [ ] No behavior change (bit-exact)
- [ ] Cache-friendly access patterns
- [ ] SIMD opportunities considered
```

---

## Acceptance Criteria

### Performance Optimization

| Criterion | Pass | Fail |
|-----------|------|------|
| Benchmark improvement | >= 5% with variance < 2% | < 5% or high variance |
| Regression check | <= 5% vs baseline | > 5% regression |
| Hot path allocations | 0 detected by dhat | Any allocation |
| Determinism | Bit-exact in 3 runs | Any variation |
| Golden tests | All pass | Any failure |

### Code Quality

| Criterion | Pass | Fail |
|-----------|------|------|
| Clippy | Zero warnings | Any warning |
| Format | `cargo fmt --check` clean | Format diff |
| Unsafe | Justified + documented | Unjustified |
| Comments | Hot path annotated | Missing context |

---

## Failure Modes

### Common Traps

1. **Optimizing without measuring**
   - Symptom: "I think this is faster"
   - Fix: Always benchmark before touching code

2. **Undefined behavior from unsafe**
   - Symptom: Works in debug, crashes in release
   - Fix: Minimize unsafe, use Miri for validation

3. **Optimization changes results**
   - Symptom: Golden tests fail
   - Fix: Ensure bit-exact equivalence

4. **False sharing in parallel code**
   - Symptom: Scaling worse than expected
   - Fix: Pad structs, separate hot/cold data

5. **SIMD overhead for small inputs**
   - Symptom: Slower than scalar for N < 8
   - Fix: Add scalar fallback with threshold

6. **Premature optimization**
   - Symptom: Complex code for cold path
   - Fix: Profile first, optimize hot spots only

### Red Flags

- PR without benchmark data
- "Trust me, it's faster"
- Unsafe without safety comment
- New allocations in `process_day()`
- HashMap in inner loop

---

## Collaboration Hooks

### Handoff to `/risk-analyst`

After optimizing engine, risk analyst must verify:
- Drawdown calculations unchanged
- Position limits still enforced
- Cost model accuracy preserved

```markdown
## Handoff: quant-engineer → risk-analyst

**Changes made:**
- [List of optimizations]

**Files modified:**
- [List of files]

**Requires validation:**
- [ ] Risk constraints still enforced
- [ ] Numerical results unchanged
```

### Handoff to `/trader-expert`

After engine changes, trader expert validates:
- Slippage modeling still accurate
- Order execution logic correct
- Market impact calculations valid

### Receiving from other skills

When receiving optimization request:
1. Get clear performance target
2. Understand constraints (can't break behavior)
3. Identify hot path scope
4. Propose measurement plan first

---

## Quick Reference

### Common Commands

```bash
# Run all benchmarks
cargo bench --bench unified_bench

# Save baseline
cargo bench --bench unified_bench -- --save-baseline v1

# Compare to baseline
cargo bench --bench unified_bench -- --baseline v1

# Generate flamegraph
cargo flamegraph --bench unified_bench -- --bench

# Check allocations (requires dhat feature)
cargo run --release --features dhat-heap

# Run with perf
perf record --call-graph dwarf cargo bench --bench unified_bench
perf report
```

### Performance Targets (from PERFORMANCE_CONTRACT.md)

| Scenario | Target |
|----------|--------|
| 1 asset, 30 days | < 1μs per day |
| 10 assets, 252 days | < 10μs per day |
| 50 assets, 252 days | < 50μs per day (linear) |
| CSV parse | > 1M lines/sec |
