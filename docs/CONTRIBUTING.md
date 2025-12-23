# Contributing

## Core Principles

1. **Determinism-First**: No change may break bit-reproducibility (AC-03)
2. **Performance-First**: Hot path must remain allocation-free
3. **Hot Path Sacred**: No I/O, `dyn Trait`, or heap allocations in simulation loop
4. **Measure Before Optimize**: Profile before optimizing

## Canonical Commands

```bash
# Must pass before any PR
cargo build --release
cargo test
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

## PR Gates

- [ ] `cargo test` passes (all suites)
- [ ] `cargo fmt --check` passes
- [ ] `cargo clippy -- -D warnings` passes
- [ ] No regressions in `cargo bench` (if touching hot path)
- [ ] Determinism hash unchanged for existing scenarios

## Crate Boundaries

Dependencies must follow acyclic graph:

- `backtester_core`: No internal dependencies
- `backtester_io`: Only `backtester_core`
- `backtester_engine`: Only `backtester_core`
- `backtester_portfolio`: Only `backtester_core`
- `backtester_execution`: Only `backtester_core`
- `backtester_reports`: `backtester_core` + `backtester_portfolio`
- `strategy_lib`: Only `backtester_core`

## Hot Path Rules

The following are **forbidden** in the simulation loop:

- Heap allocations (`Vec::push`, `Box::new`, `String`)
- Any I/O (file, network, logging)
- `dyn Trait` dispatch
- Unbounded loops
- System calls

## Benchmark Protocol

1. Run benchmark on isolated CPU core (`taskset`)
2. Fix CPU governor to `performance`
3. Disable turbo boost
4. Run 10+ iterations, report median
5. Compare against baseline in `/benches/results/`

