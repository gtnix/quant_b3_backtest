# Performance Contract

**Version**: 1.0.0  
**Last Updated**: 2026-01-05

This document defines the performance gates and measurement methodology for the `quant_b3_backtest` system. All optimizations must be validated against these gates.

## Performance Gates

| Metric | Threshold | Scenario | Rationale |
|--------|-----------|----------|-----------|
| Throughput | >= 100K events/sec | 10 assets, 252 days | Minimum viable for strategy optimization |
| Hot path allocs | 0 per event | `process_day` loop | Cache efficiency |
| Max regression | <= 5% | vs previous baseline | CI gate |
| P99 latency | <= 10x mean | single `process_day` | Tail latency control |
| **GA Backtest** | **< 20ms** | 10 assets, 252 days | GA population evaluation |

## GA Backtest Performance Gate

**Added**: 2026-01-21

The Genetic Algorithm (GA) backtest evaluation is a critical performance gate for the Strategy Combiner (SCG). Each genome in the population requires a full backtest evaluation.

### Requirements

| Mode | Threshold | Executor | Notes |
|------|-----------|----------|-------|
| Production | < 20ms | `InProcessExecutor` | **Mandatory for SCG runs** |
| CI Gate | < 30ms | `InProcessExecutor` | Allows CI variance |
| Legacy | ~1000ms | `CliExecutor` | External process, not for production |

### Enabling Fast GA Backtests

Use the `--in-process` flag with the `combiner run` command:

```bash
# Fast mode (recommended for production)
cargo run --release --bin combiner -- run --config config.toml --in-process

# Legacy mode (slow, external process)
cargo run --release --bin combiner -- run --config config.toml
```

### Prerequisites

1. `dataset.market_data_path` must be set in the config file
2. Market data CSV must be pre-generated (see `scripts/export_market_data.py`)

### CI Test

A performance gate test exists in `combiner_runner::in_process::tests`:

```rust
#[test]
fn test_performance_gate_ga_backtest_under_30ms()
```

This test:
- Creates a 10 asset x 252 day dataset
- Executes 5 backtests and measures median time
- **Fails if median > 30ms**

### Implementation Details

The `InProcessExecutor`:
- Pre-loads market data once from CSV
- Reuses data across all backtest evaluations (via `Arc`)
- Uses SIMD-optimized metrics calculation
- Zero file I/O in the hot path

## Benchmark Scenarios

### 1. Minimal Baseline (`unified_process_day_1_asset`)
- **Purpose**: Measure pure engine overhead without asset scaling
- **Configuration**: 1 asset, 30 days, no dividends
- **Target**: < 1μs per day

### 2. Realistic Workload (`unified_process_day_10_assets`)
- **Purpose**: Representative production scenario
- **Configuration**: 10 assets, 252 days, with dividends
- **Target**: < 10μs per day

### 3. Stress Test (`unified_full_backtest_50_assets`)
- **Purpose**: Scaling behavior validation
- **Configuration**: 50 assets, 252 days, with dividends
- **Target**: < 50μs per day (linear scaling)

### 4. I/O Isolation (`io_csv_parse`)
- **Purpose**: Measure data loading overhead separately
- **Configuration**: 10K lines CSV
- **Target**: > 1M lines/sec

## Measurement Methodology

### Tools
- **Primary**: Criterion.rs (statistical benchmarking)
- **Profiling**: `perf` + flamegraph for hotspot analysis
- **Allocations**: `dhat` or custom allocator hooks

### Execution
```bash
# Run all benchmarks with baseline comparison
cargo bench --bench unified_bench -- --save-baseline current

# Compare against previous baseline
cargo bench --bench unified_bench -- --baseline milestone1

# Generate flamegraph (requires perf)
cargo flamegraph --bench unified_bench -- --bench
```

### Environment Requirements
- Profile: `release` with `debug = true` (for profiling symbols)
- CPU: Isolated cores preferred (`taskset`)
- Turbo boost: Disabled for reproducibility
- Memory: Cold start (drop caches before run)

## Baseline Format

Baselines are stored in `benches/results/baseline.json`:

```json
{
  "version": "1.0.0",
  "created_at": "ISO8601",
  "git_commit": "hash",
  "scenarios": {
    "unified_engine": {
      "process_day_1_asset": {
        "mean_ns": 850,
        "stddev_ns": 50,
        "throughput_days_per_sec": 1176470
      }
    }
  },
  "gates": {
    "hot_path_allocations": 0,
    "max_regression_percent": 5.0
  }
}
```

## CI Integration

The benchmark suite should be integrated into CI with:

1. **Nightly runs**: Full benchmark suite with baseline update
2. **PR checks**: Quick smoke test (1 iteration) to catch major regressions
3. **Release gates**: Full suite must pass all thresholds

## Regression Policy

If a regression is detected:
1. **< 5%**: Warning, investigate but don't block
2. **5-10%**: Block PR, require justification
3. **> 10%**: Block PR, require fix or explicit approval

## Hot Path Definition

The hot path for the `UnifiedEngine` is:

```
UnifiedEngine::process_day()
├── Update current_prices (HashMap insert)
├── apply_dividends() [if enabled]
│   └── DividendIndex::get_by_date()
├── portfolio.update_prices()
│   └── HashMap iteration + Decimal math
├── orchestrator.execute_rebalance()
│   └── Entry/Exit evaluation
└── apply_orders()
    └── Portfolio state updates
```

**Zero-allocation zones**: The entire `process_day` loop should not allocate except for:
- Order vector (pre-allocated or arena)
- Trace events (optional, can be disabled)

## Next Steps

After baseline collection, optimization priorities are:
1. Symbol ID mapping (HashMap → Vec)
2. SoA layout for price data
3. Decimal → f64 hybrid for hot path






