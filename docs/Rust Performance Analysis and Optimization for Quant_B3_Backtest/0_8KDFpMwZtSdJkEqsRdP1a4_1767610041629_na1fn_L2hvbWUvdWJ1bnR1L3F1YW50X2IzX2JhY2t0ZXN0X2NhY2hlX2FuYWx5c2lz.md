# Rust Backtester Core CPU Cache Optimization Analysis

## Introduction
This analysis focuses on the `backtester_core` module of the `quant_b3_backtest` repository, specifically examining core data structures for CPU cache optimization opportunities. The goal is to identify potential issues related to data structure layouts, memory access patterns, cache line alignment, false sharing, and struct padding inefficiencies, which are critical for achieving the target 1000x performance improvement in a high-frequency, data-intensive backtesting environment.

The core data structures analyzed are `Bar`, `MarketEvent`, `FillEvent`, and the main event enum `Event`.

## Key Findings

*   **Inefficient Struct Padding in `MarketEvent`**: The `MarketEvent` struct, which is on the hot path (`Strategy::on_market`), is 50 bytes in size (`AssetId` (2 bytes) + `Bar` (48 bytes)). Due to 8-byte alignment requirements for the internal `f64` fields in `Bar`, the struct is likely padded to 56 bytes, resulting in 6 bytes of wasted space and an inefficient use of a 64-byte cache line.
    *   File: `/home/ubuntu/quant_b3_backtest/crates/backtester_core/src/lib.rs` (lines 332-337)
*   **Cache Line Overflow in `FillEvent`**: The `FillEvent` struct is approximately 65 bytes in size. This slightly exceeds the common 64-byte cache line size, meaning a single `FillEvent` instance will span two cache lines. When an array of `FillEvent` is processed, this can lead to increased cache misses (L1/L2) and higher memory latency.
    *   File: `/home/ubuntu/quant_b3_backtest/crates/backtester_core/src/lib.rs` (lines 660-682)
*   **Sub-optimal Enum Sizing for `Event`**: The `Event` enum, which encapsulates all core events, is sized based on its largest variant (`FillEvent`). The current size is likely 72 or 80 bytes (as `FillEvent` is 65 bytes, padded to the next 8-byte boundary, and then the enum is padded). The test `assert!(std::mem::size_of::<Event>() <= 128)` suggests an awareness of size, but a size of 64 bytes or 128 bytes would be more cache-friendly. The current size of 72-80 bytes is an inefficient use of two cache lines.
    *   File: `/home/ubuntu/quant_b3_backtest/crates/backtester_core/src/lib.rs` (lines 741-753, 877-879)
*   **`Bar` Structure and `f64` Usage**: The `Bar` struct uses 5 `f64` fields (40 bytes) and one `i64` (8 bytes), totaling 48 bytes. While the struct itself is well-aligned, the heavy use of `f64` (8 bytes each) for market data is a memory-intensive choice. Given the context of B3 (Brazilian market), which often deals with lower-precision fixed-point arithmetic, using `rust_decimal` or a custom fixed-point type (as mentioned in the repository context) for hot path calculations could reduce memory footprint and improve cache utilization.
    *   File: `/home/ubuntu/quant_b3_backtest/crates/backtester_core/src/lib.rs` (lines 262-276)
*   **Potential False Sharing in Parallel Contexts**: Given the use of `rayon` for parallelism, any array or vector of the identified large structs (`FillEvent`, `Event`) that are accessed by different threads could suffer from **false sharing**. Since `FillEvent` is 65 bytes (spanning two cache lines), two adjacent elements in an array could reside on the same cache line, causing unnecessary cache line invalidations between cores.
    *   File: `/home/ubuntu/quant_b3_backtest/crates/backtester_core/src/lib.rs` (General data structures)
*   **Unnecessary `i64` for `Bar::timestamp`**: The `Bar::timestamp` is an `i64` (8 bytes), while the wrapper `Timestamp` struct is also an `i64`. The `Bar` struct is the primary data source. If the `Timestamp` struct were used directly in `Bar`, it would not change the size, but it would improve type safety. However, the `i64` itself is a full 8 bytes. If the backtest duration is limited, a smaller type like `u32` (seconds since epoch) or `i32` (offset from start time) could save 4 bytes per bar, which is significant when dealing with millions of bars.
    *   File: `/home/ubuntu/quant_b3_backtest/crates/backtester_core/src/lib.rs` (line 265)

## Optimization Opportunities

1.  **Reorder `MarketEvent` Fields for Optimal Padding**
    *   **Rationale**: The current layout (`AssetId` (u16) followed by `Bar` (48 bytes)) creates 6 bytes of internal padding. Reordering the fields to place the smaller `AssetId` at the end can eliminate or minimize this padding.
    *   **Approach**: Change the order of fields in `MarketEvent` to:
        ```rust
        pub struct MarketEvent {
            pub bar: Bar,      // 48 bytes (8-byte aligned)
            pub asset_id: AssetId, // 2 bytes (2-byte aligned)
            // 6 bytes of padding are now at the end, making the struct 56 bytes.
            // This is still better than 64 bytes, but we can do better.
        }
        ```
    *   **Further Approach**: If `AssetId` is changed to `u64` (8 bytes) for alignment purposes (even though it only holds a `u16`), the struct size becomes 56 bytes, which is perfectly 8-byte aligned with no internal padding. This trades 6 bytes of padding for 6 bytes of actual data, but ensures perfect alignment and cache line utilization.

2.  **Compact `FillEvent` to Fit Within a Single Cache Line**
    *   **Rationale**: `FillEvent`'s 65-byte size causes it to span two cache lines, leading to a mandatory L1 cache miss for the second part of the struct on first access. Reducing its size to 64 bytes or less is crucial.
    *   **Approach**: The struct has 4 `f64` fields. Replacing these with a smaller, high-precision fixed-point type (e.g., a custom 32-bit or 48-bit fixed-point representation, or leveraging the existing `rust_decimal` if possible with smaller storage) could save 8-16 bytes, bringing the total size below 64 bytes. For example, replacing two `f64` with two `f32` would save 8 bytes, making the struct 57 bytes, which pads to 64 bytes.

3.  **Optimize `Event` Enum Size and Layout**
    *   **Rationale**: The large size of the `Event` enum (72-80 bytes) is driven by the largest variant (`FillEvent`). This size is inefficient for array processing.
    *   **Approach**: Implement **Enum Layout Optimization** by reducing the size of the largest variant (`FillEvent`) as described in opportunity 2. If `FillEvent` is reduced to 57 bytes (padded to 64), the entire `Event` enum will fit within a single 64-byte cache line, assuming a 1-byte discriminant. This is a significant win for event processing throughput.

4.  **Adopt Fixed-Point Arithmetic for Market Data in `Bar`**
    *   **Rationale**: The use of `f64` for all price and volume data in `Bar` is memory-intensive (40 bytes). High-frequency backtesting involves processing massive arrays of `Bar` data.
    *   **Approach**: Replace `f64` fields in `Bar` with a custom fixed-point type, potentially a 64-bit integer (`i64`) that stores the price scaled by a large factor (e.g., 10^8). This is a common pattern in financial systems for precision and speed. This would not change the size of `Bar` (still 48 bytes), but it would eliminate the overhead of floating-point operations and ensure deterministic, high-precision calculations, which is often faster than `f64` on modern CPUs when SIMD is not fully utilized.

## Performance Impact Estimate

| Optimization | Expected Speedup | Confidence | Measurement Method |
| :--- | :--- | :--- | :--- |
| Reorder `MarketEvent` Fields | 5% - 15% | Medium | Micro-benchmark on `on_market` loop with array of events |
| Compact `FillEvent` | 10% - 25% | High | Micro-benchmark on `FillEvent` array iteration and memory access latency |
| Optimize `Event` Enum Size | 15% - 30% | High | Macro-benchmark on full backtest run with high event volume |
| Fixed-Point Arithmetic in `Bar` | 20% - 50% | Medium | Micro-benchmark on `Bar` array processing (e.g., calculating moving averages) |

## Implementation Complexity

| Optimization | Effort (Low/Medium/High) | Risk | Dependencies |
| :--- | :--- | :--- | :--- |
| Reorder `MarketEvent` Fields | Low | Low | None (internal struct change) |
| Compact `FillEvent` | Medium | Medium | Requires careful selection of replacement type (e.g., `f32` or fixed-point) |
| Optimize `Event` Enum Size | Low | Low | Depends on `FillEvent` compaction; mostly a size check |
| Fixed-Point Arithmetic in `Bar` | High | High | Requires changing all price/volume logic across the entire backtester; potential for precision bugs |

## Trade-offs and Risks

### Trade-off: Precision vs. Memory Footprint (Fixed-Point Arithmetic)
*   **Description**: Replacing `f64` with a fixed-point integer type (e.g., `i64` scaled) reduces memory usage and improves cache locality, but requires a strict definition of precision and range.
*   **Mitigation Approach**: Define the fixed-point type with a high enough scale factor (e.g., 10^8) to cover the required precision (e.g., 8 decimal places) and range of prices. Implement robust conversion functions between the fixed-point type and `f64` for external interfaces. Use a dedicated crate like `rust_decimal` or a custom type with clear documentation on its limits.

### Trade-off: Code Readability vs. Micro-Optimization (Field Reordering)
*   **Description**: Reordering fields in structs like `MarketEvent` to minimize padding can make the code less intuitive (e.g., placing the smallest field last, or using a larger type than necessary for alignment).
*   **Mitigation Approach**: Add clear comments to the struct definition explaining the reordering is for cache alignment (`#[repr(C)]` or `#[repr(align(64))]` if necessary, though Rust's default is usually sufficient if fields are ordered correctly). Use `#[cfg(debug_assertions)]` assertions to check struct sizes in tests to ensure the optimization holds.

### Trade-off: Loss of Floating-Point Range/Performance (Compacting `FillEvent`)
*   **Description**: Reducing the size of `f64` fields to `f32` or a smaller fixed-point type in `FillEvent` may limit the range or precision of prices, commissions, and slippage.
*   **Mitigation Approach**: Analyze the maximum expected values for these fields. If `f32` is sufficient for the required precision (e.g., 6-7 significant decimal digits), the trade-off is acceptable. If not, a custom 48-bit fixed-point type might be necessary, which adds complexity but maintains precision. The risk is mitigated by ensuring all calculations involving these fields are done using the appropriate type.

### Risk: False Sharing Introduction

*   **Description**: If the data structures are not perfectly aligned and sized, and are accessed concurrently by different threads (e.g., in `rayon` loops), false sharing can occur, negating performance gains.
*   **Mitigation Approach**: Use the `#[repr(align(64))]` attribute on structs that are frequently stored in arrays and accessed in parallel (e.g., `Event`, `FillEvent`) to force alignment to the cache line boundary. This will introduce padding, but guarantees that adjacent elements do not share a cache line, eliminating false sharing.

## Conclusion
The `backtester_core` module shows several opportunities for CPU cache optimization, primarily centered around struct padding and cache line alignment in the hot path data structures (`MarketEvent`, `FillEvent`, `Event`). Implementing the proposed changes, particularly the compaction of `FillEvent` and the subsequent size optimization of the `Event` enum, is expected to yield significant performance improvements by maximizing cache line utilization and reducing memory latency. The highest risk and highest reward opportunity is the transition to fixed-point arithmetic for market data, which requires a broader refactoring effort but aligns with the goal of 1000x performance.