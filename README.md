# Quant B3 Backtester

High-performance, deterministic backtesting engine for B3 (Brazilian Stock Exchange) trading strategies.

## Workspace Structure

| Crate | Purpose |
|-------|---------|
| `backtester_core` | Fundamental types, traits, and events |
| `backtester_io` | Data ingestion and normalization |
| `backtester_engine` | Simulation motor and order router |
| `backtester_portfolio` | Portfolio state, PnL, drawdown |
| `backtester_execution` | Order execution with slippage/costs |
| `backtester_reports` | Report generation |
| `strategy_lib` | User strategy implementations |

## Quick Start

```bash
# Build (release)
cargo build --release

# Run all tests
cargo test

# Run specific test suite
cargo test --test determinism
cargo test --test invariants
cargo test --test anti_look_ahead
```

## Quality Checks

```bash
# Format check
cargo fmt --check

# Lint (strict)
cargo clippy --all-targets -- -D warnings

# Dependency audit (requires cargo-deny)
cargo deny check
```

## Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific crate benchmark
cargo bench -p backtester_engine
```

## Design Principles

1. **Determinism-First**: Identical inputs produce bit-identical outputs
2. **Performance-First**: Zero allocations in hot path
3. **Hot Path Sacred**: No I/O, no `dyn Trait`, no allocations in simulation loop

See `/docs` for full architecture documentation.

