# Memory Allocation Pattern Analysis: quant_b3_backtest

## Introduction

This analysis focuses on identifying and mitigating heap allocation patterns within the `quant_b3_backtest` repository, particularly in the performance-critical daily simulation loop of the `UnifiedEngine`. The goal is to reduce memory pressure, minimize cache misses, and eliminate continuous reallocations to achieve the target of a 1000x performance improvement for strategy generation.

The backtesting engine, located primarily in `backtester_engine` and `backtester_intelligence`, is a high-throughput system where even small, repeated allocations can become a significant bottleneck over long backtest periods.

## Key Findings

The following are the most critical areas of heap allocation identified in the core backtesting path:

*   **Continuous Trace History Growth:** The `UnifiedEngine` stores `trace: Vec<TraceEvent>` and `daily_dividend_cashflow: Vec<(NaiveDate, Decimal)>` in `/home/ubuntu/quant_b3_backtest/crates/backtester_engine/src/unified.rs` (lines 280, 282). These vectors grow linearly with the backtest duration, leading to continuous reallocations and memory fragmentation in the main engine struct.
*   **Per-Order String Clones in Hot Path:** Within the `apply_orders` function, which is called daily, every executed order results in two heap allocations for the audit trail:
    *   `symbol: order.symbol.clone()` in `/home/ubuntu/quant_b3_backtest/crates/backtester_engine/src/unified.rs` (line 517).
    *   `side: format!("{:?}", order.side)` in `/home/ubuntu/quant_b3_backtest/crates/backtester_engine/src/unified.rs` (line 518).
*   **Temporary Vector Allocations:** Short-lived temporary vectors are created inside the daily processing loop and its helpers, such as `let mut applications = Vec::new()` (line 453) and `let mut applied = Vec::new()` (line 489) in `/home/ubuntu/quant_b3_backtest/crates/backtester_engine/src/unified.rs`. Given the typically small number of daily orders/candidates, these allocations are likely unnecessary.
*   **`Box<dyn Trait>` for Filters/Checks:** The strategy and monitoring systems use dynamic dispatch via `Box<dyn Trait>`, e.g., `Vec<Box<dyn AssetFilter>>` in `/home/ubuntu/quant_b3_backtest/crates/backtester_intelligence/src/scorer.rs` (line 60) and `Vec<Box<dyn DataHealthCheck>>` in `/home/ubuntu/quant_b3_backtest/crates/backtester_intelligence/src/monitoring/data_health.rs` (line 821). The creation of these objects involves heap allocation.
*   **`to_vec` and `to_owned` in Performance Metrics:** Multiple instances of `returns.to_vec()` and `data.to_vec()` are found in performance and risk calculation modules, such as `/home/ubuntu/quant_b3_backtest/crates/backtester_intelligence/src/performance/risk.rs` (lines 57, 241, 285) and `/home/ubuntu/quant_b3_backtest/crates/backtester_intelligence/src/monitoring/statistics.rs` (lines 29, 78). These create full copies of large data arrays for sorting or further processing.

## Optimization Opportunities

1.  **Replace `Vec<T>` with Pre-allocated/Fixed-size Structures:**
    *   **Rationale:** Eliminate continuous reallocations of the main history vectors (`trace`, `daily_dividend_cashflow`) and temporary vectors in the hot path.
    *   **Approach:** For the history, consider using a **slab allocator** or a **pre-allocated ring buffer** if the maximum backtest length is known. For temporary vectors like `applications` and `applied`, replace `Vec<T>` with **`ArrayVec`** or **`SmallVec`** with a reasonable inline capacity (e.g., 8 or 16) to move the allocation to the stack for the common case.

2.  **Eliminate Per-Order String Allocations:**
    *   **Rationale:** The `String` clones for `symbol` and `format!("{:?}", side)` are executed for every single trade, creating significant allocation overhead.
    *   **Approach:** Change the `TraceEvent` struct to use **`Arc<str>`** or **`&'static str`** for the symbol if symbols are interned, or use **`Cow<'static, str>`** if they are not. For the `side` field, use a simple `enum` or a fixed-size array of bytes (`[u8; 4]`) instead of a `String` to represent the side (Buy/Sell).

3.  **Implement Arena Allocation for Strategy Generation:**
    *   **Rationale:** The `combiner_engine` likely creates many short-lived objects (e.g., strategy candidates, intermediate calculation results) during genetic algorithm runs. Using a standard allocator for these will lead to high fragmentation and overhead.
    *   **Approach:** Introduce a dedicated **`bumpalo`** or similar **arena allocator** for the entire strategy generation process. All temporary objects created during a single generation/evaluation run should be allocated from this arena, which can be cleared in a single operation when the generation is complete.

4.  **Optimize `to_vec` with `sort_unstable_by` on Slices:**
    *   **Rationale:** The `to_vec()` calls in performance metrics create full copies of large data arrays just to sort them. This is wasteful.
    *   **Approach:** Pass slices (`&[T]`) to the calculation functions and use methods that operate on slices, such as `sort_unstable_by` on a temporary slice, or use an **index-based sort** to avoid moving the data entirely. If a copy is unavoidable, use `Vec::with_capacity` to avoid the initial small allocation/reallocation cycle.

5.  **Explore Static Dispatch for Filters/Checks:**
    *   **Rationale:** Dynamic dispatch (`Box<dyn Trait>`) introduces heap allocation and vtable lookups.
    *   **Approach:** If the set of filters/checks is fixed or small, use **`enum`** or **`const generics`** to achieve static dispatch. For example, a `FilterList<F1, F2, F3>` struct where the filters are type parameters, or a single `enum` that covers all filter types. This eliminates the `Box` allocation and allows the compiler to inline the calls.

## Performance Impact Estimate

| Optimization | Expected Speedup | Confidence | Measurement Method |
| :--- | :--- | :--- | :--- |
| Arena Allocation for Strategy | 5x - 10x | High | Profiling with `dhat` or `valgrind` on `combiner_engine` execution. |
| SmallVec/ArrayVec for Daily Vectors | 1.5x - 3x | Medium | Micro-benchmarking `process_day` with `criterion` and comparing allocation counts. |
| Eliminate Per-Order String Clones | 2x - 5x | High | Profiling `apply_orders` with `perf` to measure time spent in `malloc`/`free`. |
| Slice-based Sorting (Remove `to_vec`) | 1.2x - 2x | Medium | Benchmarking risk/performance calculation functions with and without full vector copy. |

## Implementation Complexity Assessment

| Optimization | Effort (Low/Medium/High) | Risk | Dependencies |
| :--- | :--- | :--- | :--- |
| Arena Allocation for Strategy | Medium | Medium | `bumpalo` or similar crate. Requires refactoring object creation in `combiner_engine`. |
| SmallVec/ArrayVec for Daily Vectors | Low | Low | `smallvec` or `arrayvec` crates. Requires changing type definitions in `UnifiedEngine`. |
| Eliminate Per-Order String Clones | Medium | Medium | Refactoring `TraceEvent` and ensuring symbol interning/fixed-size representation is sound. |
| Slice-based Sorting (Remove `to_vec`) | Low | Low | No new dependencies. Requires minor refactoring of function signatures and logic. |
| Static Dispatch for Filters/Checks | High | High | Requires significant refactoring of the `backtester_intelligence` trait system. |

## Trade-offs and Risks

### 1. Arena Allocation for Strategy Objects

**Trade-off:** Simplifies memory management for short-lived objects but introduces a new memory model. Objects allocated in the arena cannot be easily moved or dropped individually; the entire arena is cleared at once.
**Mitigation:** Carefully define the scope of the arena. The arena should only be used for objects whose lifetime is strictly bound to a single strategy evaluation or genetic generation run. Use a dedicated type alias (e.g., `ArenaVec<T>`) to clearly mark arena-allocated types.

### 2. Fixed-Size Collections (SmallVec/ArrayVec)

**Trade-off:** Requires a commitment to a maximum inline size. If the number of daily orders or candidates exceeds the inline capacity (e.g., 16), the collection will fall back to a heap allocation, which is slightly slower than a standard `Vec` allocation due to the internal logic.
**Mitigation:** Profile the maximum and 99th percentile of daily orders/candidates to choose a safe and efficient inline capacity. Add runtime assertions or logging to detect and monitor heap fallback events.

### 3. Symbol Interning/Fixed-Size Symbol Representation

**Trade-off:** Moving from `String` to a fixed-size representation (e.g., `[u8; 8]`) or interning (`Arc<str>`) complicates string handling. Interning adds a small overhead on symbol creation/lookup, and fixed-size limits the symbol length.
**Mitigation:** Given the domain (B3 assets), symbol length is fixed and short (e.g., `PETR4`, `VALE3`). A fixed-size array is a high-reward, low-risk solution. If interning is chosen, ensure the interning mechanism is thread-safe and highly optimized, possibly using a crate like `string-cache` or `internment`.

### 4. Refactoring to Static Dispatch

**Trade-off:** Eliminates runtime polymorphism, which is the primary benefit of the current `Box<dyn Trait>` design. This can lead to significant code bloat and increased compilation times due to monomorphization, especially if the number of filter/check combinations is large.
**Mitigation:** Only apply static dispatch to the most performance-critical, frequently-called components (e.g., the core `AssetFilter` trait). For less critical components (e.g., monitoring checks), retain dynamic dispatch or use a simpler, more contained enum-based approach.