# Rust Backtesting Engine Performance Analysis and Optimization Roadmap

## Analysis Title: Architectural Optimization for 1000x Speedup in Quant Backtesting Engine

## Key Findings

The `quant_b3_backtest` repository is a high-performance Rust project that already incorporates advanced optimization techniques, but the architecture reveals a potential bottleneck in the simulation core that prevents the target 1000x speedup.

*   **Existing Vectorization/Parallelism:** The project already utilizes **`rayon`** for multi-threading (`backtester_engine/src/parallel.rs:158`) and **`wide`** for SIMD operations (`combiner_engine/src/pareto_simd.rs:7`), confirming a strong foundation in high-performance computing.
*   **Vectorized Genetic Algorithm Core:** The genetic algorithm's multi-objective optimization (NSGA-II) is highly optimized, leveraging **SIMD** for dominance and crowding distance calculations (`combiner_engine/src/pareto_simd.rs:100-164`), processing 4 comparisons at a time. This is a world-class optimization pattern.
*   **Hybrid Event-Driven/Parallel Simulation:** The `ParallelEngine` in `backtester_engine/src/parallel.rs` uses a hybrid approach: **parallel signal generation** (`par_iter` on events) followed by **sequential portfolio update** (`Phase 2: Sequential update`). This sequential bottleneck is necessary for portfolio state consistency but limits the maximum speedup.
*   **Data Structure for Simulation:** The `ParallelEngine` uses a `MarketState` with `Vec<f64>` for prices and volumes (`backtester_engine/src/parallel.rs:22`), which is a Structure-of-Arrays (SoA) layout, aligning with SIMD and cache-friendly access.
*   **Precision Overhead:** The `UnifiedEngine` in `backtester_engine/src/unified.rs` uses **`rust_decimal::Decimal`** for all financial calculations to ensure institutional-grade precision and determinism (`backtester_engine/src/unified.rs:18`). While correct, this introduces a significant performance overhead compared to native `f64` operations, especially in hot loops.
*   **Batching for GA Evaluation:** The `StageABatchEvaluator` in the genetic algorithm (`combiner_engine/src/evaluation/stage_a.rs:77`) is designed for high-throughput screening, using **caching** and **parallel execution** to evaluate genomes, mirroring the "Reduce border crossings" pattern from QuantConnect LEAN.

## Optimization Opportunities

The path to 1000x speedup requires a fundamental architectural shift from the current hybrid event-driven model to a pure, end-to-end vectorized model, similar to VectorBT, for the initial screening phase.

1.  **Implement Pure Vectorized Backtest Engine (Stage A):**
    *   **Rationale:** The current hybrid model is bottlenecked by the sequential portfolio update. For the initial, high-throughput screening (Stage A) in the genetic algorithm, a pure vectorized approach can eliminate the sequential loop entirely.
    *   **Approach:** Create a new `VectorizedEngine` that calculates all indicators, signals, and portfolio returns (NAV) across the entire dataset for *all* assets and *all* strategies in a single, massive array operation. This is possible for simple strategies that are not path-dependent. This is the core architectural pattern of VectorBT.
    *   **Implementation Detail:** Leverage the existing SoA data layout and `wide` for SIMD operations on indicator calculations. Use `rayon` to parallelize the outer loop over strategies/assets.

2.  **Profile and Optimize `rust_decimal` Usage:**
    *   **Rationale:** `rust_decimal` is critical for final, high-fidelity backtests (Stage B/UnifiedEngine) but is a major performance sink in the high-throughput Stage A.
    *   **Approach:** Introduce a **dual-precision policy**. Use native `f64` for all calculations in the high-speed **Stage A** screening phase, where slight floating-point inaccuracies are acceptable for fitness ranking. Only use `rust_decimal` for the final, validated **Stage B** backtests and reporting.
    *   **Implementation Detail:** Refactor the `UnifiedEngine` to accept a `PrecisionMode` enum (`HighFidelityDecimal` | `HighSpeedF64`) and switch the underlying math accordingly.

3.  **Optimize Data Loading and Access with `memmap2`:**
    *   **Rationale:** The existing use of `memmap2` is excellent, but ensure that the data access patterns are fully optimized to avoid page faults and unnecessary copies.
    *   **Approach:** Ensure all market data is loaded into a single, contiguous memory-mapped file with an optimized SoA layout. Use **`madvise`** (via Rust's `libc` or a wrapper crate) to advise the kernel on access patterns (e.g., `MADV_SEQUENTIAL` or `MADV_WILLNEED`) to pre-fetch data and minimize I/O latency.

4.  **Fine-Grained Parallelism in Sequential Phase:**
    *   **Rationale:** The sequential portfolio update in the `ParallelEngine` is unavoidable for path-dependent logic. However, individual operations within the loop (e.g., mark-to-market, commission calculation) can be micro-optimized.
    *   **Approach:** Use **SIMD** (`wide`) for the Mark-to-Market calculation across all open positions. Since the positions are stored in a portfolio state, ensure the underlying position data is also in an SoA layout to facilitate SIMD vectorization.

## Performance Impact Estimate

| Optimization | Expected Speedup | Confidence | Measurement Method |
| :--- | :--- | :--- | :--- |
| Pure Vectorized Engine (Stage A) | 10x - 100x | High | Micro-benchmark against current `ParallelEngine` |
| Dual-Precision Policy (`f64` for Stage A) | 5x - 10x | High | Profiling hot loops with `rust_decimal` vs. `f64` |
| Optimized Data Loading (`madvise`) | 1.5x - 3x | Medium | I/O profiling with `perf` and `strace` |
| SIMD Mark-to-Market | 2x - 4x | Medium | Micro-benchmark on position valuation loop |

**Cumulative Potential Speedup:** The multiplicative effect of these optimizations, particularly the architectural shift to a pure vectorized engine for the high-volume Stage A, could easily exceed the 1000x target for the overall strategy generation process.

## Implementation Complexity Assessment

| Optimization | Effort (Low/Medium/High) | Risk | Dependencies |
| :--- | :--- | :--- | :--- |
| Pure Vectorized Engine (Stage A) | High | High | Requires significant re-architecture of the backtest core. |
| Dual-Precision Policy (`f64` for Stage A) | Medium | Medium | Requires careful management of data types and potential precision bugs. |
| Optimized Data Loading (`madvise`) | Low | Low | Requires a small wrapper around `libc` or a dedicated crate. |
| SIMD Mark-to-Market | Medium | Medium | Requires ensuring position data is in SoA format for vectorization. |

## Trade-offs and Risks

### 1. Architectural Shift to Vectorized Engine (Risk: Loss of Flexibility)
*   **Trade-off:** A pure vectorized engine (like VectorBT) is extremely fast but **cannot handle path-dependent logic** (e.g., order book simulation, complex slippage models, dynamic position sizing based on real-time P&L).
*   **Mitigation:** The current hybrid `ParallelEngine` (which is event-driven at the portfolio level) must be retained as the **Stage B** engine for final, high-fidelity validation. The vectorized engine is strictly for the high-throughput **Stage A** screening phase, where speed is paramount and simple strategies are evaluated.

### 2. Dual-Precision Policy (Risk: Precision Errors)
*   **Trade-off:** Using `f64` in the hot path (Stage A) sacrifices the absolute financial precision guaranteed by `rust_decimal`.
*   **Mitigation:** The `f64` results are only used for **relative ranking** (Pareto sorting) in the genetic algorithm. The top-performing strategies must be re-validated using the high-fidelity `rust_decimal`-based `UnifiedEngine` (Stage B) before being accepted into the Hall of Fame. This is already partially implemented in the Stage A/B structure.

### 3. SIMD Optimization (Risk: Code Complexity and Portability)
*   **Trade-off:** Direct SIMD programming with crates like `wide` significantly increases code complexity and reduces readability.
*   **Mitigation:** Encapsulate all SIMD logic within dedicated, well-tested utility functions (e.g., `pareto_simd.rs`). Where possible, prefer higher-level vectorized operations that the Rust compiler can auto-vectorize, or use crates like `nalgebra` or `ndarray` which are designed for array-based computation.

### 4. Data Loading Optimization (Risk: OS/Platform Dependency)
*   **Trade-off:** Using OS-specific calls like `madvise` introduces platform-dependent code.
*   **Mitigation:** Abstract the `madvise` calls behind a feature flag or a dedicated `data_loader` module that falls back to standard I/O on unsupported platforms. Given the target is a high-performance server environment (likely Linux), this is a low-risk trade-off for a significant performance gain.

## Full Markdown Report

/home/ubuntu/performance_analysis_report.md