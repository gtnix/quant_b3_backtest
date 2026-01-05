# Performance Analysis of `combiner_engine` Genetic Algorithm Implementation

## Overview
This report provides a detailed performance analysis of the `combiner_engine` crate within the `quant_b3_backtest` repository, focusing on the implementation of the Non-dominated Sorting Genetic Algorithm II (NSGA-II) and related components. The primary goal is to identify bottlenecks and propose concrete optimizations to contribute to the overall target of a **1000x performance improvement** for strategy generation.

The analysis confirms a strong foundation, utilizing Rust's performance features, including Structure-of-Arrays (SoA) layout, `rayon` for parallelism, and explicit SIMD instructions via the `wide` crate. The core bottleneck remains the external backtest execution, but significant gains can be achieved by optimizing the genetic algorithm's overhead.

## Key Findings

*   **Parallel Evaluation is Correctly Implemented:** The `StageABatchEvaluator::evaluate_batch` function uses `rayon::par_iter()` to parallelize the execution of backtests for cache misses. This correctly addresses the primary bottleneck (backtest execution) by distributing the workload across available cores.
    *   File: `/home/ubuntu/quant_b3_backtest/crates/combiner_engine/src/evaluation/stage_a.rs` (L136-140)
*   **SIMD for Dominance Check:** The most computationally intensive part of NSGA-II, the pairwise dominance comparison, is vectorized using `wide::f64x4` in `compute_dominance_simd`. This is a high-quality optimization.
    *   File: `/home/ubuntu/quant_b3_backtest/crates/combiner_engine/src/pareto_simd.rs` (L77-187)
*   **Sequential Rank Assignment Bottleneck:** The subsequent rank assignment phase (`assign_ranks`) in NSGA-II is implemented sequentially, relying on dynamic `Vec<Vec<usize>>` (`dominated_by`) and a `while` loop to process fronts. For large populations, this sequential step can become the new bottleneck after SIMD optimization of the dominance check.
    *   File: `/home/ubuntu/quant_b3_backtest/crates/combiner_engine/src/pareto_simd.rs` (L208-251)
*   **Sequential Next Generation Creation:** The selection, crossover, and mutation operators in `EvolutionEngine::create_next_generation` are executed sequentially within a `while` loop to fill the new population. This is a minor bottleneck compared to evaluation but represents an easy opportunity for parallelization.
    *   File: `/home/ubuntu/quant_b3_backtest/crates/combiner_engine/src/engine.rs` (L299-328)
*   **Crowding Distance Sorting Overhead:** The crowding distance calculation (`compute_crowding_for_objective`) requires sorting indices for each objective (`sharpe_ratios`, `cagrs`, `max_drawdowns`) for every Pareto front. This sorting operation is sequential and repeated, adding significant overhead.
    *   File: `/home/ubuntu/quant_b3_backtest/crates/combiner_engine/src/pareto_simd.rs` (L299-303)
*   **Unused Batching in Parallel Evaluation:** The `StageABatchEvaluator` configuration includes a `batch_size: 16` (L40), but the `evaluate_batch` implementation uses `par_iter().enumerate().map(...)` (L136-140), which processes single genomes in parallel, not batches of 16. The batch size is effectively ignored, which might lead to excessive thread spawning overhead if the backtest execution time is very short.
    *   File: `/home/ubuntu/quant_b3_backtest/crates/combiner_engine/src/evaluation/stage_a.rs` (L40, L136-140)

## Optimization Opportunities

1.  **Parallelize Next Generation Creation (Selection/Crossover/Mutation)**
    *   **Rationale:** The creation of new genomes is currently sequential. Parallelizing this loop will reduce the GA overhead, especially for large population sizes.
    *   **Approach:** Replace the sequential `while new_genomes.len() < self.config.population_size` loop with a parallel approach using `rayon::scope` or `par_iter` over the required number of children. The selection step must be thread-safe (e.g., using a lock-free random number generator or pre-generating selection indices).

2.  **Parallelize NSGA-II Rank Assignment**
    *   **Rationale:** The rank assignment phase is a known sequential bottleneck in NSGA-II. Parallelizing this step is crucial for large-scale multi-objective optimization.
    *   **Approach:** Implement a parallel version of the Fast Non-Dominated Sorting Algorithm (e.g., using a parallel prefix sum or a parallel merge-sort based approach) to assign ranks concurrently. Alternatively, investigate a tree-based NSGA-II variant.

3.  **Optimize Crowding Distance Calculation**
    *   **Rationale:** The repeated sorting of indices for each objective is computationally expensive.
    *   **Approach:** Instead of sorting indices for each objective, consider pre-sorting the entire population once by each objective and storing the sorted indices. The crowding distance calculation can then iterate over these pre-sorted lists. This trades memory for significant time savings.

4.  **Implement Batching in `StageABatchEvaluator`**
    *   **Rationale:** The current parallel evaluation processes single genomes, potentially leading to high thread-spawning overhead. The configured `batch_size` is unused.
    *   **Approach:** Modify `StageABatchEvaluator::evaluate_batch` to use `par_chunks(self.config.batch_size)` instead of `par_iter()`. This will process a configurable number of genomes per thread, reducing synchronization and thread management overhead.

5.  **Explore SIMD for Fitness Metric Aggregation**
    *   **Rationale:** While the backtest is external, the aggregation of metrics from multiple backtests (if they were batched) or the calculation of summary statistics (mean, variance) could benefit from SIMD.
    *   **Approach:** If the `combiner_core`'s `PopulationFitnessSoA` is used for summary statistics, ensure that calculations like mean Sharpe ratio are vectorized using `wide` or Rust's native SIMD intrinsics.

## Quantitative Performance Impact Estimates

| Optimization | Expected Speedup | Confidence | Measurement Method |
| :--- | :--- | :--- | :--- |
| Parallelize Next Generation | 1.5x - 3x | Medium | Benchmarking `create_next_generation` with large population sizes (e.g., 10,000) |
| Parallelize NSGA-II Rank Assignment | 5x - 10x | High | Profiling NSGA-II on a large, diverse population (e.g., 5,000 genomes) |
| Optimize Crowding Distance | 3x - 5x | High | Benchmarking `compute_crowding_distance_simd` against a pre-sorted index approach |
| Implement Batching in Evaluation | 1.1x - 1.5x | Medium | Micro-benchmarking `evaluate_batch` with varying batch sizes (1, 16, 32) |
| SIMD for Metric Aggregation | 1.2x - 2x | Low | Profiling summary statistics calculation on `PopulationFitnessSoA` |

## Implementation Complexity Assessment

| Optimization | Effort (Low/Medium/High) | Risk | Dependencies |
| :--- | :--- | :--- | :--- |
| Parallelize Next Generation | Low | Low | `rayon` (already used) |
| Parallelize NSGA-II Rank Assignment | High | High | Custom parallel algorithm implementation |
| Optimize Crowding Distance | Medium | Medium | Changes to `pareto_simd.rs` and `PopulationFitnessSoA` |
| Implement Batching in Evaluation | Low | Low | `rayon` (already used) |
| SIMD for Metric Aggregation | Medium | Medium | `wide` crate, `combiner_core` changes |

## Trade-offs and Risks

### Trade-off 1: Parallel NSGA-II Complexity vs. Sequential Simplicity
The current sequential rank assignment is simple, robust, and easy to debug. Moving to a parallel NSGA-II implementation introduces significant complexity.

*   **Risk:** Increased code complexity, potential for subtle race conditions or deadlocks, and difficulty in verifying the correctness of the non-dominated fronts.
*   **Mitigation Approach:** Start with a well-vetted parallel NSGA-II algorithm (e.g., a parallelized version of the original NSGA-II) and implement it in a separate module with extensive unit tests that compare results against the existing scalar implementation.

### Trade-off 2: Memory vs. Speed in Crowding Distance
The proposed optimization for crowding distance involves pre-sorting the population by each objective and storing the indices. This requires additional memory proportional to $3 \times N$ (where $N$ is the population size) to store the sorted index vectors.

*   **Risk:** Increased memory consumption, which could be a concern if the population size is extremely large or if the system is memory-constrained.
*   **Mitigation Approach:** Profile the memory usage before and after the change. If memory is a concern, implement the optimization only for the largest fronts, or use an in-place sorting algorithm that minimizes memory allocation.

### Trade-off 3: Thread Overhead vs. Batching Granularity
Implementing batching in `StageABatchEvaluator` reduces thread overhead but increases the latency of individual backtest results. An improperly chosen batch size can lead to underutilization of cores or excessive waiting.

*   **Risk:** Suboptimal batch size selection could hurt performance instead of helping.
*   **Mitigation Approach:** Make the batch size configurable and perform rigorous benchmarking to find the optimal value for the target execution environment. The default of 16 is a good starting point, but it should be validated.

### Trade-off 4: SIMD Intrinsics Portability

The use of the `wide` crate for SIMD is generally portable, but explicit SIMD can sometimes lead to issues on non-x86 architectures or older CPUs.

*   **Risk:** Reduced portability or reliance on target-specific CPU features.
*   **Mitigation Approach:** Ensure the SIMD implementation has a scalar fallback (which `wide` generally provides) and that the build process correctly targets the desired CPU features. The existing use of `wide` suggests this is already a known trade-off.

## Conclusion
The `combiner_engine` is well-structured and already employs advanced performance techniques. The path to the 1000x goal lies in a combination of optimizing the backtest execution (external to this crate) and aggressively parallelizing the remaining GA overhead. The most impactful internal optimizations are the parallelization of the NSGA-II rank assignment and the optimization of the crowding distance calculation. These two areas represent the largest remaining sequential bottlenecks in the core genetic algorithm loop.