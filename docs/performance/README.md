# Performance Testing Guide

This document describes how to run performance benchmarks and golden tests for the `quant_b3_backtest` system.

## Quick Start

```bash
# Run all benchmarks
cargo bench --bench unified_bench

# Run golden tests (determinism validation)
cargo test --package backtester_engine --test golden_tests

# Run all tests including golden tests
cargo test
```

## Benchmark Suite

### UnifiedEngine Benchmarks

Located in `crates/backtester_engine/benches/unified_bench.rs`

| Benchmark | Description | Target |
|-----------|-------------|--------|
| `process_day_1_asset` | Single asset, single day | < 10 μs |
| `process_day_10_assets` | 10 assets, single day | < 100 μs |
| `full_backtest_252d_10_assets` | Full year, 10 assets | < 25 ms |
| `scaling/*` | Asset scaling analysis | Sub-linear |
| `engine_init` | Engine initialization | < 5 μs |

### Running Benchmarks

```bash
# Full benchmark suite
cargo bench --bench unified_bench

# Specific benchmark group
cargo bench --bench unified_bench -- unified_engine

# Save baseline for comparison
cargo bench --bench unified_bench -- --save-baseline milestone3

# Compare against baseline
cargo bench --bench unified_bench -- --baseline milestone3

# Generate HTML report
cargo bench --bench unified_bench -- --plotting-backend plotters
# Reports saved to target/criterion/
```

### Interpreting Results

Criterion outputs three key metrics:

- **time**: Mean execution time with confidence interval
- **thrpt**: Throughput (elements/second)
- **change**: Percentage change from baseline (if available)

Example output:
```
unified_engine/process_day_10_assets
                        time:   [20.7 µs 20.87 µs 21.0 µs]
                        thrpt:  [47.6 Kelem/s 47.9 Kelem/s 48.3 Kelem/s]
```

## Golden Tests

Located in `crates/backtester_engine/tests/golden_tests.rs`

Golden tests verify that the engine produces consistent results:

| Test | Purpose |
|------|---------|
| `golden_test_determinism_basic` | Same inputs → consistent DayProcessed events |
| `golden_test_baseline_values` | Structural validation |
| `golden_test_trace_consistency` | Trace event validation |
| `golden_test_no_dividends` | Dividend handling |
| `golden_test_equity_bounds` | Reasonable output bounds |
| `golden_test_multiple_runs_consistent_structure` | Multiple runs consistency |

### Determinism (Milestone 3)

The UnifiedEngine hot path is now fully deterministic:
- **SymbolId** replaces String for O(1) array indexing
- **Price/Money/Rate** fixed-point types ensure bit-exact arithmetic
- **Vec<Option<DualPriceBar>>** replaces HashMap for price storage
- **Sorted iteration** ensures consistent DayProcessed event ordering

## Performance Contract

See `benches/PERFORMANCE_CONTRACT.md` for:
- Performance gates and thresholds
- Regression policy
- Hot path definition
- Measurement methodology

## Baseline Results

### Milestone 6 (Current)

| Scenario | Mean Time | Throughput | vs M5 | vs M1 |
|----------|-----------|------------|-------|-------|
| 1 asset/day | 1.48 μs | 675K elem/s | ~0% | **4.5x faster** |
| 10 assets/day | 14.2 μs | 70K elem/s | **23% faster** | **5.3x faster** |
| 252 days × 10 assets | 2.84 ms | 89K elem/s | **32% faster** | **6.8x faster** |
| 20 assets (252d) | 3.65 ms | 69K elem/s | - | - |
| 50 assets (252d) | 5.66 ms | 45K elem/s | - | - |
| engine_init | 2.28 μs | - | +5% | - |

### Milestone 5

| Scenario | Mean Time | Throughput | vs M4 | vs M1 |
|----------|-----------|------------|-------|-------|
| 1 asset/day | 1.87 μs | 535K elem/s | ~0% | **3.6x faster** |
| 10 assets/day | 18.4 μs | 54K elem/s | **8.7% faster** | **4.1x faster** |
| 252 days × 10 assets | 4.20 ms | 60K elem/s | **4.8% faster** | **4.6x faster** |
| 25 assets/day | 24.5 μs | 1.02M elem/s | **16% faster** | - |
| 50 assets/day | 34.2 μs | 1.46M elem/s | **20% faster** | - |
| engine_init | 2.17 μs | - | **6% faster** | - |

### Milestone 4

| Scenario | Mean Time | Throughput | vs M3 | vs M1 |
|----------|-----------|------------|-------|-------|
| 1 asset/day | 1.87 μs | 535K elem/s | ~0% | **3.6x faster** |
| 10 assets/day | 20.16 μs | 50K elem/s | **4% faster** | **3.7x faster** |
| 252 days × 10 assets | 4.41 ms | 57K elem/s | **6% faster** | **4.4x faster** |
| engine_init | 2.30 μs | - | +8% | - |

### Milestone 3

| Scenario | Mean Time | Throughput | vs M2 |
|----------|-----------|------------|-------|
| 1 asset/day | 1.79 μs | 559K elem/s | **5% faster** |
| 10 assets/day | 20.87 μs | 48K elem/s | **4% faster** |
| 252 days × 10 assets | 4.67 ms | 54K elem/s | **5% faster** |
| engine_init | 2.13 μs | - | **12% faster** |

### Milestone 2

| Scenario | Mean Time | Throughput |
|----------|-----------|------------|
| 1 asset/day | 1.92 μs | 520K elem/s |
| 10 assets/day | 22.68 μs | 44K elem/s |
| 252 days × 10 assets | 4.92 ms | 51K elem/s |

### Milestone 1 (Initial Baseline)

| Scenario | Mean Time | Throughput |
|----------|-----------|------------|
| 1 asset/day | 6.65 μs | 150K elem/s |
| 10 assets/day | 75.6 μs | 132K elem/s |
| 252 days × 10 assets | 19.4 ms | 13K days/s |

### Milestone 6 Improvements (Current)

**What changed:**
1. Migrated `Order` struct to fixed-point:
   - `price: Price` (was `Decimal`)
   - `estimated_cost: Money` (was `Decimal`)
   - `notional: Money` (was `Decimal`)
2. Migrated `AssetCandidate` to fixed-point:
   - `price: Option<Price>` (was `Option<Decimal>`)
   - `avg_volume: Option<Money>` (was `Option<Decimal>`)
3. Migrated `EntryContext`, `EntryTarget`, `EntryDiagnostics` to `Money`/`Price`
4. Migrated `GatingConfig` thresholds to fixed-point:
   - Internal f64 with getters returning `Price`/`Money`
5. Migrated `OrderGeneratorConfig` to fixed-point:
   - `br_brokerage: Rate`, `us_per_share_fee: Money`, `max_allocation_pct: Rate`
6. Migrated `RebalanceOrchestrator`:
   - `execute_rebalance()` now takes `Money` for cash/equity/peak_equity
   - `RebalanceStepAudit` uses `Money` for all monetary fields
7. Updated `EntryEngine.evaluate()` and `ExitEngine` to use fixed-point throughout
8. `unified.rs` now passes `Money` directly to orchestrator (no `.to_decimal()`)

**Performance gains:**
- 10 assets/day: **23% faster** (14.2 μs vs 18.4 μs)
- Full backtest (252 × 10): **32% faster** (2.84 ms vs 4.20 ms)

**Where Decimal is still used (boundaries only):**
- Config parsing (serde): `initial_capital`, `cost_bps`
- External reporting: `DayResult.equity`, `BacktestResult`
- PnL display formatting

**New benchmark: Rebalance Scaling**
```bash
cargo bench --bench unified_bench -- rebalance_scaling
```
Measures full backtest with 10/20/50 assets to isolate rebalance path cost.

### Milestone 5 Improvements

**What changed:**
1. `process_day()` now takes `candidates: &[AssetCandidate]` instead of `Vec`
   - Callers no longer need to clone candidates every day
   - Zero-copy pass-through of candidate slices
2. Added `EngineScratch` buffer for position collection
   - Reusable buffer with `clear()` + reuse pattern
   - Avoids Vec allocation per day in steady state
3. `orchestrator.execute_rebalance()` takes slice instead of Vec
4. `entry_engine.evaluate()` takes slice instead of Vec
5. Removed `gating_candidates.clone()` in entry engine
6. Changed `positions.sort_by()` to `sort_unstable_by()` (faster, same determinism)

**Performance gains scale with portfolio size:**
- 10 assets: 8.7% faster
- 25 assets: 16% faster  
- 50 assets: 20% faster

**Proof of allocation reduction:**
- Before: `candidates.clone()` on every `process_day()` call (N × sizeof(AssetCandidate))
- After: Zero-copy slice reference
- Before: `Vec::new()` + collect for positions every day
- After: `scratch.positions.clear()` + reuse (no realloc in steady state)

**Remaining bottleneck:**
- `RebalanceOrchestrator` still uses `Decimal` for order netting (not hot path)
- `Entry` module (`Order`, `AssetCandidate`) still uses `Decimal`
- Candidate → GatingCandidate conversion still allocates (could use arena)

### Milestone 4 Improvements

**What changed:**
1. `Position` now uses `Price` for:
   - `cost_basis` - Average cost per share
   - `current_price` - Current market price
   - `high_water_mark` - Trailing stop calculation
2. `PortfolioState` now uses `Money` for:
   - `cash`, `equity`, `peak_equity`, `initial_capital`
3. New `update_prices_with_fast()` method - direct `Price` assignment
4. `DividendEvent` uses `Rate`, `DividendApplication` uses `Money`
5. Zero Decimal in hot path between `process_day` and `PortfolioState`

**Hot path evidence:**
- `portfolio.update_prices_with_fast(|sym| prices.get(id).map(|b| b.raw_close))`
- `market_value_fast()` → `Price.mul_shares()` → pure i64 arithmetic
- `calculate_equity_fast()` → `Money` summation → pure i64 addition
- These are NOT hot path (only run on rebalance days)

### Milestone 3 Improvements

**What changed:**
1. New fixed-point types in `backtester_core/src/fixed.rs`:
   - `Price(i64)` - 6 decimal places (scale 1e6) for asset prices
   - `Money(i64)` - 6 decimal places for cash/equity/PnL
   - `Rate(i64)` - 8 decimal places for dividend rates
2. `DualPriceBar` now uses `Price` internally instead of `Decimal`
3. 6x smaller memory footprint per price bar
4. Bit-exact determinism via integer arithmetic

### Cumulative Improvement (M1 → M6)

| Scenario | M1 | M6 | Total Gain |
|----------|----|----|------------|
| 1 asset/day | 6.65 μs | 1.48 μs | **4.5x** |
| 10 assets/day | 75.6 μs | 14.2 μs | **5.3x** |
| 252 days × 10 assets | 19.4 ms | 2.84 ms | **6.8x** |

## Profiling

### CPU Profile with perf

```bash
# Build with debug symbols
cargo build --release --bench unified_bench

# Record profile
perf record --call-graph dwarf \
  target/release/deps/unified_bench-* --bench

# Generate flamegraph
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

### Memory Allocation Analysis

```bash
# Using dhat (requires dhat-rs dependency)
cargo run --release --features dhat-heap

# Or using heaptrack
heaptrack cargo bench --bench unified_bench
heaptrack_gui heaptrack.*.gz
```

## CI Integration

Benchmarks should be run in CI with:

1. **Nightly**: Full suite with baseline update
2. **PR checks**: Quick smoke test (1 iteration)
3. **Release gates**: Full suite must pass thresholds

Example CI command:
```bash
# Quick check (sample-size reduces statistical rigor but is faster)
cargo bench --bench unified_bench -- --sample-size 10

# Full check with comparison
cargo bench --bench unified_bench -- --baseline milestone3
```

## Optimization Roadmap

| Milestone | Focus | Status | Gain |
|-----------|-------|--------|------|
| 1 | Baseline + Guards | Done | - |
| 2 | Symbol ID Mapping | Done | **3-4x** |
| 3 | Fixed-Point DualPriceBar | Done | **4-5%** |
| 4 | Fixed-Point Position/Portfolio | Done | **6%** |
| 5 | Zero-Alloc Hot Path (clone/sort removal) | Done | **8-20%** |
| 6 | Fixed-Point Orchestrator/Entry | Done | **23-32%** |
| 7 | SIMD Metrics | Pending | 2-4x |
| 8 | I/O Fast Path (OBFS) | Pending | 2-3x |
| 9 | Lock-free Parallelism | Pending | 1.5-2x |

**Current Status**: 6.8x faster than baseline (M1 → M6)
**Target**: 10-100x total improvement through combined optimizations.

## Fixed-Point Types Reference

### Price(i64)

For asset prices with 6 decimal places (supports crypto micro-units):

```rust
use backtester_core::Price;

let price = Price::from_f64(123.456789);  // Stored as 123_456_789
let shares = 1000;
let value = price.mul_shares(shares);      // Returns Money
```

### Money(i64)

For cash, equity, PnL with 6 decimal places:

```rust
use backtester_core::Money;

let cash = Money::from_f64(1_000_000.0);
let cost = Money::from_f64(5000.50);
let remaining = cash - cost;  // Money arithmetic is i64 addition
```

### Rate(i64)

For dividend rates with 8 decimal places:

```rust
use backtester_core::Rate;

let div_rate = Rate::from_f64(0.05);  // R$0.05 per share
let shares = 1000;
let cashflow = div_rate.mul_shares(shares);  // Returns Money(50.0)
```
