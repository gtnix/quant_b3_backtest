# SIMD Performance Analysis: Wide Crate, Vectorization, and Portable SIMD Migration

This report provides a detailed analysis of the SIMD implementation within the `quant_b3_backtest` repository, focusing on the use of the `wide` crate, identifying vectorization bottlenecks, and assessing the potential for migration to `std::simd` and multi-instruction-set optimization (AVX2/AVX512). The analysis is aimed at achieving the project's goal of a 1000x performance improvement for strategy generation by maximizing data parallelism in core computational kernels.

## Current SIMD Implementation Overview

The backtesting engine currently utilizes the `wide` crate for portable SIMD operations, specifically employing `f64x4` vectors. This approach successfully vectorizes several fundamental financial calculations, such as mean, variance, and dot product, which are crucial for performance metric calculation.

## Key Findings

The following specific findings highlight areas where the current SIMD implementation can be significantly improved:

*   **Scalar Bottleneck in Drawdown Calculation:** The `simd_drawdown` function in `backtester_core/src/simd.rs` (lines 85-97) drops back to a scalar loop for peak tracking and max drawdown calculation after loading the vector. This is a critical performance bottleneck, as the core logic is not vectorized.
*   **Anti-Pattern in Pareto Dominance:** The `compute_dominance_simd` function in `combiner_engine/src/pareto_simd.rs` loads vectors (lines 103-120) but immediately converts them back to arrays (`.into()`) and processes them in a scalar loop (lines 135-161) for the complex dominance check. This negates the benefit of vectorization.
*   **Non-Vectorized Filtering in Sortino:** The `simd_sortino` function in `backtester_core/src/simd.rs` (line 246) uses a scalar filter to extract downside returns, which involves heap allocation (`Vec<f64>`) and is a significant performance hit in a hot loop.
*   **Hardcoded Vector Size:** The code is hardcoded to process chunks of 4 (`f64x4`), which limits performance to 256-bit SIMD (AVX/AVX2) and prevents automatic scaling to 512-bit SIMD (AVX-512) on compatible hardware.
*   **Manual Horizontal Reduction:** Horizontal sums (e.g., in `simd_mean` line 148) are performed manually by converting the vector back to an array and summing the elements. While functional, this is less efficient than dedicated horizontal sum intrinsics or functions provided by `std::simd`.
*   **Sub-Optimal Dominance Check:** The dominance check logic in `pareto_simd.rs` (lines 142-152) is implemented using multiple scalar comparisons on the array elements, which is inherently difficult to vectorize and should be refactored to use bitmasks and vector comparison operations.

## Optimization Opportunities

The following concrete recommendations are proposed to address the identified bottlenecks and achieve higher performance:

1.  **Full Vectorization of Drawdown and Peak Tracking:**
    *   **Rationale:** The current scalar loop in `simd_drawdown` is a major performance inhibitor for a core financial metric.
    *   **Approach:** Implement a fully vectorized running maximum (peak tracking) algorithm. This typically involves a combination of `max` operations and masked blends to update the running peak vector, followed by a vectorized calculation of the drawdown against the peak. This is a complex operation that may require using `std::simd`'s masked operations for efficient conditional updates.

2.  **Migration to `std::simd` (Portable SIMD):**
    *   **Rationale:** The `std::simd` module is the official, future-proof path for portable SIMD in Rust, offering better compiler integration and access to advanced features like masked operations and horizontal reductions.
    *   **Approach:** Replace all `wide::f64x4` usage with `std::simd::f64x4` and utilize `std::simd`'s built-in functions for horizontal sums (`.reduce_sum()`) and masked operations. This will also simplify the path for multi-instruction-set optimization.

3.  **Vectorized Downside Filtering for Sortino Ratio:**
    *   **Rationale:** The scalar filtering and heap allocation in `simd_sortino` is inefficient.
    *   **Approach:** Use vector comparison operations to create a mask for negative returns. Then, use masked operations (e.g., masked load/store or masked blend) to accumulate the sum of squares for only the negative returns, avoiding the creation of an intermediate `Vec<f64>`.

4.  **Multi-Instruction-Set Optimization (AVX-512):**
    *   **Rationale:** Leveraging 512-bit vectors (`f64x8`) can potentially double the throughput of vectorized operations on modern CPUs.
    *   **Approach:** After migrating to `std::simd`, use Rust's target feature system (`#[target_feature(enable = "avx512f")]`) and conditional compilation to define an `f64x8` implementation for AVX-512 targets. This requires refactoring the chunking logic to process 8 elements at a time. A dynamic dispatch mechanism (e.g., using the `multiversion` crate or manual function pointers) should be implemented to select the best available instruction set at runtime.

5.  **Refactor Pareto Dominance Check:**
    *   **Rationale:** The current scalar loop within the SIMD function is a severe bottleneck in the genetic algorithm's core.
    *   **Approach:** Refactor the dominance check to perform the comparisons (Sharpe, CAGR, Drawdown) fully in parallel using vector comparisons (`.cmp_ge()`, `.cmp_gt()`). The resulting masks should be combined using bitwise operations (`&`, `|`) to determine dominance for all 4 solutions simultaneously. The updates to `domination_count` and `dominated_by` will still require some scalar logic, but the comparison phase will be vectorized.

## Performance Impact Estimate

| Optimization | Expected Speedup | Confidence | Measurement Method |
| :--- | :--- | :--- | :--- |
| Full Vectorization of Drawdown | 2x - 4x | High | Micro-benchmark against scalar implementation |
| `std::simd` Migration & Horizontal Sums | 1.1x - 1.5x | Medium | Micro-benchmark of reduction kernels |
| Vectorized Downside Filtering | 3x - 5x | High | Micro-benchmark of `simd_sortino` with large data sets |
| Multi-Instruction-Set (AVX-512) | 1.5x - 2x (over AVX2) | Medium | Full backtest run on AVX-512 enabled hardware |
| Refactor Pareto Dominance Check | 1.5x - 3x | Medium | Micro-benchmark of `compute_dominance_simd` |

## Implementation Complexity

| Optimization | Effort (Low/Medium/High) | Risk | Dependencies |
| :--- | :--- | :--- | :--- |
| Full Vectorization of Drawdown | High | Medium | Requires complex masked operations |
| `std::simd` Migration & Horizontal Sums | Low | Low | `std::simd` (Rust nightly/stable with feature) |
| Vectorized Downside Filtering | Medium | Medium | Requires careful handling of masked accumulation |
| Multi-Instruction-Set (AVX-512) | High | High | `multiversion` crate or manual runtime dispatch |
| Refactor Pareto Dominance Check | High | High | Complex logic, potential for correctness bugs |

## Trade-offs and Risks

### 1. Complexity of Full Vectorization for Conditional Logic (Drawdown/Pareto)

*   **Trade-off:** Achieving full vectorization for algorithms with complex conditional logic (like peak tracking in drawdown or the multi-criteria dominance check) significantly increases code complexity compared to the current scalar-in-SIMD approach.
*   **Risk:** Increased risk of subtle bugs related to floating-point precision, mask handling, and correctness, which are difficult to debug in SIMD code.
*   **Mitigation:** Implement comprehensive unit tests for the vectorized kernels, including edge cases (e.g., all-zero data, single-element data, boundary conditions). Use a reference scalar implementation to cross-validate results.

### 2. Migration to `std::simd`

*   **Trade-off:** Requires updating all SIMD-related code across the workspace, which is a non-trivial refactoring effort.
*   **Risk:** `std::simd` is still evolving, and while stable, some advanced features might be less mature than in specialized crates.
*   **Mitigation:** Perform the migration incrementally, starting with the simplest functions (`simd_sum`, `simd_mean`). Ensure the target Rust toolchain is compatible with the required `std::simd` features.

### 3. Multi-Instruction-Set Optimization (AVX-512)

*   **Trade-off:** Requires implementing and maintaining multiple versions of the same function (e.g., one for SSE, one for AVX2, one for AVX-512). This adds significant build and runtime complexity.
*   **Risk:** AVX-512 can lead to CPU clock throttling on some processors, potentially negating the performance gains for mixed workloads. Incorrect feature detection can lead to runtime crashes.
*   **Mitigation:** Use a robust runtime dispatch library like `multiversion` to handle feature detection automatically. Profile the AVX-512 version carefully to ensure the performance gain outweighs any potential thermal throttling or increased power consumption. Limit AVX-512 usage to the most critical, data-intensive kernels.

### 4. Floating-Point Precision and Associativity


*   **Trade-off:** SIMD operations often change the order of floating-point additions and multiplications (e.g., horizontal sums), which can lead to minor, but measurable, differences in the final result compared to the scalar version due to the non-associativity of floating-point arithmetic.
*   **Risk:** Test failures due to precision differences, or subtle errors in financial calculations.
*   **Mitigation:** Use fuzzy comparison assertions (e.g., `assert!((a - b).abs() < 1e-9)`) in all unit tests. Document the expected precision trade-off and ensure the differences are within acceptable financial tolerance limits.

***

**File Path References:**

*   `backtester_core/src/simd.rs:85-97`
*   `backtester_core/src/simd.rs:246`
*   `combiner_engine/src/pareto_simd.rs:135-161`
*   `combiner_engine/src/pareto_simd.rs:142-152`
*   `backtester_core/src/simd.rs:148`