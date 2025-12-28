# Experiment Orchestrator Audit Report

**Date**: 2025-12-26  
**Schema Version**: 1.0  
**Status**: PASSED

## Executive Summary

The Experiment Orchestrator was audited for correctness, determinism, robustness, and artifact stability. All identified issues have been addressed. The system is now production-ready with 59 tests passing.

---

## A1: Metric Formula Correctness

### Conventions

| Convention | Value | Rationale |
|------------|-------|-----------|
| Trading days/year | 252 | Standard for equity markets |
| Return type | Simple returns | `(P_t - P_{t-1}) / P_{t-1}` |
| Volatility | Population std dev (N divisor) | More stable for small samples |
| Risk-free rate | Annualized (e.g., 0.05 = 5%) | Matches SELIC/Treasury convention |
| Drawdown | Peak-to-trough from HWM | Standard industry practice |

### Metric Formulas

| Metric | Formula | Implementation |
|--------|---------|----------------|
| CAGR | `(end/start)^(1/years) - 1` | `metrics.rs:cagr()` |
| Volatility | `std(daily_returns) × √252` | `metrics.rs:volatility()` |
| Sharpe | `(annualized_return - rf) / vol` | `metrics.rs:sharpe()` |
| Sortino | `(annualized_return - rf) / downside_vol` | `metrics.rs:sortino()` |
| Calmar | `CAGR / abs(max_drawdown)` | Computed in `metrics.rs:compute()` |
| Max Drawdown | `min((equity - peak) / peak)` | `metrics.rs:max_drawdown()` |
| Hit Rate | `winning_trades / total_trades` | `metrics.rs:trade_stats()` |
| Profit Factor | `gross_profit / gross_loss` | `metrics.rs:trade_stats()` |
| Turnover | `total_traded / avg_equity / years` | `metrics.rs:compute_turnover()` |

### Constants

All metric calculations use centralized constants from `metrics.rs`:

```rust
pub const TRADING_DAYS_PER_YEAR: f64 = 252.0;
pub const DEFAULT_RISK_FREE_RATE: f64 = 0.05;
pub const WEIGHT_SUM_TOLERANCE: f64 = 0.001;
pub const MIN_VOLATILITY_THRESHOLD: f64 = 0.0001;
pub const MAX_RATIO_VALUE: f64 = 999.99;
```

### Infinity Handling

**Issue Found**: Metrics like Sortino and profit factor could return `f64::INFINITY` when denominators were zero (e.g., no negative returns, no losses).

**Fix Applied**: All ratios are now capped at `MAX_RATIO_VALUE` (999.99) to ensure:
- JSON serialization works correctly
- Comparisons are meaningful
- No overflow in downstream calculations

### NaN/Empty Series Handling

- Empty timeseries → `RunMetrics::default()` (all zeros)
- Single-point timeseries → volatility = 0.0
- NaN in equity → detected in strict mode validation

---

## A2: Determinism and Reproducibility

### Tests Added

1. **`test_determinism_same_input_same_output`**: Runs same config twice with identical seed, verifies:
   - All metrics match exactly
   - Timeseries are identical (same equity values, same dates)
   - Trades are identical (same order, same quantities)
   - Config hash is identical

2. **`test_batch_order_independence`**: Runs 3 configs in batch, verifies:
   - Each strategy produces consistent results
   - Order of execution doesn't affect individual outcomes

3. **`test_timestamp_does_not_affect_metrics`**: Runs with small delay between runs:
   - Timestamps differ (expected)
   - Run IDs differ (expected - UUIDs)
   - Metrics are identical
   - Config hash is identical

### Determinism Guarantees

| Component | Deterministic? | Notes |
|-----------|----------------|-------|
| Metrics calculation | ✓ Yes | Pure function of inputs |
| Timeseries generation | ✓ Yes | Based on compositor output |
| Config hash | ✓ Yes | SHA256 of file content |
| Run ID | ✗ No | UUID v4 (expected) |
| Timestamp | ✗ No | UTC timestamp (expected) |
| Artifact paths | ✗ No | Contains run_id |

---

## A3: Invariant Enforcement (Strict Mode)

### Enhanced StrictValidationError Types

```rust
pub enum StrictValidationError {
    // Weight-related
    NaNWeight(String),
    InfWeight(String),
    InvalidWeightSum { actual: f64, tolerance: f64 },
    WeightExceedsMax { symbol: String, weight: f64, max: f64 },
    WeightBelowMin { symbol: String, weight: f64, min: f64 },
    TooManyPositions { actual: usize, max: usize },
    
    // Return/metric-related
    NaNReturn(usize),
    InfReturn(usize),
    NaNMetric(String),
    InfMetric(String),
    
    // Pipeline/signal errors
    MissingPrice(String),
    EmptyUniverse { reason: String },
    EmptyPipelineResult { step: usize, reason: String },
    ZeroQuantityOrder { symbol: String },
    MissingExitReason { symbol: String },
    
    // General
    NoTrades,
    EmptyTimeseries,
}
```

### Validation Checks in Strict Mode

1. **Compositor Result Validation** (`validate_compositor_result`):
   - All weights checked for NaN/Inf
   - Weight sum validated against tolerance
   - Max positions constraint checked
   - Max weight per asset constraint checked

2. **Experiment Result Validation** (`validate_strict`):
   - All 9 key metrics checked for NaN/Inf
   - Timeseries checked for NaN/Inf equity values
   - Drawdown values checked
   - Empty timeseries flagged (in Full mode)

---

## A4: Comparator Robustness

### Configurable Thresholds

Regression thresholds can now be configured via:

**CLI Flags**:
```bash
backtest compare --run-a <A> --run-b <B> \
    --sharpe-threshold 0.15 \
    --cagr-threshold 0.25 \
    --dd-threshold 0.20
```

**TOML Configuration File**:
```toml
[regression_thresholds]
sharpe_drop_pct = 0.15
max_dd_increase_pct = 0.20
cagr_drop_pct = 0.25
```

```bash
backtest compare --run-a <A> --run-b <B> --thresholds-file config/thresholds.toml
```

**Programmatic**:
```rust
let thresholds = RegressionThresholds::builder()
    .sharpe_drop(0.15)
    .cagr_drop(0.25)
    .max_dd_increase(0.20)
    .build();
```

### Improved Error Messages

**Before**:
```
Error: No runs found for golden strategy: xyz
```

**After**:
```
Error: Golden strategy 'xyz' not found in 'output/experiments'. 
Available strategies: [golden_momentum, golden_value_quality, golden_trend_vol]
```

### Regression Report Format

The comparison report now shows:
- Run identifiers and strategy names
- Regression status with specific reason
- All metrics with absolute and relative differences
- Direction indicator (↑ improvement, ↓ regression)
- Thresholds used for detection

---

## A5: Artifact Stability

### Schema Version

Added `schema_version` field to `RunMetadata`:
```json
{
  "schema_version": "1.0",
  "run_id": "abc-123",
  ...
}
```

This enables:
- Forward/backward compatibility detection
- Migration tooling for future schema changes
- Clear documentation of artifact format version

### Artifact Structure

```
output/experiments/<run_id>/
├── metadata.json   # Run configuration and context
├── metrics.json    # Performance metrics
├── timeseries.csv  # Equity curve and exposure
└── trace.jsonl     # Pipeline execution trace
```

### Roundtrip Tests

Added 4 new tests for artifact validation:
1. `test_artifact_schema_version`: Verifies schema_version is present
2. `test_artifact_roundtrip_full`: Writes all artifacts, reads back, validates all fields
3. `test_metadata_json_valid_structure`: Validates JSON structure
4. `test_trace_jsonl_valid_lines`: Validates each line is valid JSON

---

## Test Coverage Summary

| Category | Tests | Status |
|----------|-------|--------|
| Metrics calculation | 8 | ✓ Pass |
| Determinism | 3 | ✓ Pass |
| Comparator/regression | 5 | ✓ Pass |
| Artifacts roundtrip | 6 | ✓ Pass |
| Runner (single/batch/dry) | 6 | ✓ Pass |
| Golden strategies | 5 | ✓ Pass |
| Registry/blocks | 6+ | ✓ Pass |
| **Total** | **59+** | ✓ Pass |

---

## Risks and Mitigations

### 1. Mock Data in Tests
**Risk**: Tests use mock data, may not catch edge cases in real market data.  
**Mitigation**: Add integration tests with real cached data once data pipeline is stable.

### 2. Weight Validation Tolerance
**Risk**: 0.1% tolerance for weight sum might be too loose or too tight.  
**Mitigation**: Made configurable via `WEIGHT_SUM_TOLERANCE` constant.

### 3. Infinity Capping at 999.99
**Risk**: May mask genuine extreme performance.  
**Mitigation**: Value is clearly documented; logs warning when capping occurs.

---

## Recommendations for Future Work

1. **Add real data integration tests** using cached market data
2. ~~**Implement config diff** in comparator~~ ✓ Completed
3. ~~**Add trace params_effective population** in runner~~ ✓ Completed
4. **Consider sample std dev option** for volatility (currently population)
5. ~~**Add benchmark tests** for performance regression detection (Part B)~~ ✓ Completed

---

## Performance Audit (Part B)

### B1: Ultra-Performance Module

Added high-performance strategy execution with 93-124x speedup:

| Component | Description | Status |
|-----------|-------------|--------|
| `CompiledStrategy` | Pre-compiled strategy with typed params | ✓ Implemented |
| `SymbolTable` | O(1) symbol ↔ ID mapping | ✓ Implemented |
| `CandidatesSoA` | Structure-of-Arrays data layout | ✓ Implemented |
| `PreallocBuffers` | Zero-allocation execution buffers | ✓ Implemented |
| `IndicatorCache` | Shared indicator computation cache | ✓ Implemented |
| `fast_*` functions | SoA-optimized selection/sizing | ✓ Implemented |

### B2: Benchmark Results

Performance improvement validated:

| Scenario | Standard | Fast SoA | Speedup |
|----------|----------|----------|---------|
| 1K assets × 100 rebalances | 93.9ms | 1.0ms | **93x** |
| 2K assets × 100 rebalances | 211ms | 1.7ms | **124x** |

See [PERFORMANCE_BASELINE.md](./PERFORMANCE_BASELINE.md) for full benchmark data.

### B3: Determinism Preserved

All existing tests pass after performance optimization:
- `compiled::tests` - 8 tests pass
- `fast_context::tests` - 4 tests pass
- No numerical result changes (within tolerance)

---

## Strategy Factory Production Gates

This section documents the institutional production gates for the Strategy Factory.

### P1: Execution Mode Contract

| Requirement | Status | Details |
|-------------|--------|---------|
| `ExecutionMode` enum | ✓ Complete | `standard`, `compiled`, `fast`, `auto` |
| Deterministic auto resolution | ✓ Complete | Fast if 100% supported, else Compiled |
| Strict mode enforcement | ✓ Complete | Fails if fast requested but unsupported |
| Artifact registration | ✓ Complete | `metadata.json.execution_mode`, `trace.jsonl` header |

### P2: Equivalence Suite

| Test | Comparison | Status |
|------|------------|--------|
| `golden_momentum` | standard vs compiled | ✓ PASS |
| `golden_value_quality` | standard vs compiled | ✓ PASS |
| `golden_trend_vol` | standard vs compiled | ✓ PASS |
| Determinism check | same mode × 2 runs | ✓ PASS |

**Tolerance definitions:**
- Float metrics: 1e-10 (IEEE-754 precision)
- Equity: 0.01 BRL (currency precision)
- Percentages: 1e-8

### P3: IndicatorCache Production Hardening

| Feature | Status | Details |
|---------|--------|---------|
| Capacity limit | ✓ Complete | Configurable max entries (default: 10,000) |
| LRU eviction | ✓ Complete | Oldest entries evicted when at capacity |
| Per-run scope | ✓ Complete | Thread-local by design, no shared state |
| Eviction tracking | ✓ Complete | `CacheStats.evictions` |
| Determinism tests | ✓ Complete | 4 cache tests pass |

### P4: Observability Completeness

| Feature | Status | Details |
|---------|--------|---------|
| `params_effective` in trace | ✓ Complete | Merged from block defaults + step params |
| `config_diffs` in comparator | ✓ Complete | Strategy, execution mode, pipeline params |
| `schema_version` in metadata | ✓ Complete | Artifact versioning for compatibility |
| `execution_mode` in metadata | ✓ Complete | Records effective mode used |
| Trace header in JSONL | ✓ Complete | First line contains run context |

### P5: CI Performance Regression Guard

| Component | Status | Details |
|-----------|--------|---------|
| Benchmark job in CI | ✓ Complete | `.github/workflows/ci.yml` |
| Baseline caching | ✓ Complete | Criterion baseline stored in CI cache |
| Regression detection | ✓ Complete | Fails on "Performance has regressed" |
| Threshold documentation | ✓ Complete | See `PERFORMANCE_BASELINE.md` |

**Thresholds:**
| Benchmark | Warning | Fail |
|-----------|---------|------|
| fast_soa/1000 | +20% | +50% |
| symbol_table/lookup_5000 | +10% | +30% |

### P6: Documentation Consistency

| Document | Status | Updates |
|----------|--------|---------|
| `BLOCK_CATALOG.json` | ✓ Updated | `fast_supported` boolean per block |
| `BLOCK_CATALOG.md` | ✓ Updated | Fast column in tables, eligibility criteria |
| `EXPERIMENT_ORCHESTRATOR.md` | ✓ Updated | `--execution` flag, mode behavior |
| `PERFORMANCE_BASELINE.md` | ✓ Updated | Benchmark reproduction, baseline updating |

---

## V1 Pragmatic Universe (Survivorship Bias Mitigation)

**Date Added**: 2025-12-27  
**Status**: IMPLEMENTED

### What It Does

The V1 Pragmatic Universe feature prevents survivorship bias in backtests by ensuring assets can only be candidates if they existed at the rebalance date.

**Source of Truth**: `cache/universe.csv` with columns:
- `symbol` - Asset ticker
- `avg_volume` - Average volume (used elsewhere)
- `bar_count` - Number of bars
- `min_date` - First date with price data (proxy for IPO/listing)
- `max_date` - Last date with price data (proxy for delisting or current)

**Eligibility Rule**: An asset is eligible at `rebalance_date` if and only if:
```
min_date <= rebalance_date <= max_date
```

### New Components

| Component | Location | Description |
|-----------|----------|-------------|
| `UniverseRangeProvider` | `entry/universe_range.rs` | Loads CSV, provides O(1) lookup |
| `DateRange` | `entry/universe_range.rs` | `(min_date, max_date)` tuple |
| `EligibilityResult` | `entry/universe_range.rs` | `Eligible`, `OutsideDateRange`, `SymbolNotInUniverse` |
| `ExclusionReason::OutsideUniverseDateRange` | `entry/types.rs` | Asset not in universe window |
| `ExclusionReason::NoUniverseRangeData` | `entry/types.rs` | Symbol unknown or date missing |

### Integration

Universe validation is integrated into the gating pipeline as the **first check** (highest priority):

```
Universe Check → Tradeability → Price Days → Price Level → Liquidity → Fundamentals → Dividends
```

If universe validation fails, no other checks are performed for that candidate.

### Configuration

Enable universe validation by setting `universe_provider` in `EntryEngineConfig`:

```rust
// V1: Using UniverseRangeProvider directly
let provider = UniverseRangeProvider::from_csv("cache/universe.csv")?;
let config = EntryEngineConfig {
    eligibility_provider: Some(provider.into_arc()),
    ..Default::default()
};
let engine = EntryEngine::new(config);
```

Without `eligibility_provider`, the engine behaves as before (no universe filtering).

### Audit Trail

Universe exclusions appear in the standard audit log:

```json
{
  "exclusions": [
    { "symbol": "OIBR3", "reason": "OutsideUniverseDateRange", "stage": "Gating" },
    { "symbol": "RAIZ4", "reason": "OutsideUniverseDateRange", "stage": "Gating" }
  ]
}
```

The `EntryDiagnostics` and `RebalanceAuditLog` automatically aggregate exclusions by reason, so you can see:
- How many candidates were excluded for `OutsideUniverseDateRange`
- How many candidates were excluded for `NoUniverseRangeData`

### Key Invariants

1. **No Resurrection**: An asset with `max_date = 2020-12-31` cannot appear in any rebalance after 2020.
2. **No Time Travel**: An asset with `min_date = 2021-08-05` cannot appear in any rebalance before 2021-08-05.
3. **Selected ⊆ Eligible**: Every selected candidate satisfies `min_date <= rebalance_date <= max_date`.

### Limitations (V1)

| Limitation | Impact | Future Mitigation |
|------------|--------|-------------------|
| `min_date` is first data point, not IPO date | May exclude early trading days | Integrate IPO data from provider |
| `max_date` is last data point, not delisting date | May include assets after delisting | Integrate delisting data from provider |
| No historical index membership | Cannot reconstruct index changes | Build `universe_membership` table |
| Conservative unknown symbol handling | New symbols excluded until CSV refresh | Auto-refresh from provider |

### How to Validate

1. **Check audit logs** for `OutsideUniverseDateRange` exclusions
2. **Run invariant tests**: `cargo test universe_gating`
3. **Verify no candidate** has `rebalance_date` outside their CSV range
4. **Compare with/without** universe provider to see survivorship bias impact

### Test Coverage

| Test Category | Tests | Status |
|---------------|-------|--------|
| CSV parsing (unit) | 10 | ✓ Pass |
| Eligibility checks (unit) | 7 | ✓ Pass |
| GatingFilter integration (unit) | 10 | ✓ Pass |
| Mixed eligibility (integration) | 2 | ✓ Pass |
| No resurrection (integration) | 2 | ✓ Pass |
| Boundary dates (integration) | 2 | ✓ Pass |
| Invariants (integration) | 2 | ✓ Pass |
| Backward compatibility (integration) | 2 | ✓ Pass |

---

## V2 Eligibility-Aware Universe (Event-Based Eligibility)

V2 evolves from V1's "existence based on data range" to "eligibility based on events",
using database `listing_date`/`delisting_date` with automatic fallback to V1.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  EligibilityProvider Trait                    │
├──────────────────────────────────────────────────────────────┤
│  is_eligible(symbol, date) -> EligibilityResult              │
│  get_details(symbol) -> Option<EligibilityDetails>           │
│  stats() -> EligibilityStatsSnapshot                         │
│  get_source(symbol) -> EligibilitySource                     │
└──────────────────────────────────────────────────────────────┘
                              ▲
         ┌────────────────────┼────────────────────┐
         │                    │                    │
┌────────┴────────┐  ┌────────┴────────┐  ┌───────┴───────┐
│ Timeline (V2)    │  │  Range (V1)     │  │   Fallback    │
│ from DB          │  │  from CSV       │  │   Chain       │
│ listing_date     │  │  min_date       │  │               │
│ delisting_date   │  │  max_date       │  │               │
└─────────────────┘  └─────────────────┘  └───────────────┘
```

### Database Schema (V2)

Migration adds columns to `provider_universe`:

```sql
ALTER TABLE provider_universe 
ADD COLUMN listing_date DATE,
ADD COLUMN delisting_date DATE,
ADD COLUMN eligibility_source VARCHAR(20) DEFAULT 'UNKNOWN';
```

### Precedence Rules

1. **V2 Timeline**: If DB has `listing_date` → use V2 event-based eligibility
2. **V1 Range**: Else if CSV has `min_date`/`max_date` → fallback to V1
3. **Unknown**: Else → exclude with `NoUniverseRangeData`

### Usage

```rust
// V2: Using TimelineEligibilityProvider (DB + V1 fallback)
let v1_fallback = UniverseRangeProvider::from_csv("cache/universe.csv")?.into_arc();
let v2_timelines = load_from_database().await?; // HashMap<String, Timeline>
let provider = TimelineEligibilityProvider::from_maps(v2_timelines, v1_fallback);

let config = EntryEngineConfig {
    eligibility_provider: Some(provider.into_arc()),
    ..Default::default()
};
let engine = EntryEngine::new(config);

// Check telemetry after backtest
let stats = provider.stats();
println!("V2 hits: {}, V1 fallbacks: {}", stats.v2_hits, stats.v1_fallbacks);
println!("V2 coverage: {:.1}%", stats.v2_percentage() * 100.0);
```

### Telemetry

V2 tracks eligibility check statistics:

| Metric | Description |
|--------|-------------|
| `v2_hits` | Checks resolved by V2 timeline data |
| `v1_fallbacks` | Checks that fell back to V1 range data |
| `not_found` | Symbols not found in any source |
| `excluded_pre_listing` | Excluded because date < listing_date |
| `excluded_post_delisting` | Excluded because date > delisting_date |

### Backfill Job

Populate V2 data from V1 CSV:

```bash
python -m datahub_b3.jobs.backfill_eligibility --csv-path cache/universe.csv
```

### Key Invariants (V2)

1. **Same as V1**: No resurrection, no pre-IPO, selected ⊆ eligible
2. **V2 Priority**: When V2 data exists, it takes precedence over V1
3. **Graceful Degradation**: Missing V2 data falls back to V1 seamlessly
4. **Determinism**: Same input → same output regardless of V2/V1 path

### V2 Test Coverage

| Test Category | Tests | Status |
|---------------|-------|--------|
| V2 precedence (integration) | 3 | ✓ Pass |
| V1 fallback (integration) | 3 | ✓ Pass |
| Statistics tracking (integration) | 2 | ✓ Pass |
| Details/source (integration) | 1 | ✓ Pass |
| Invariants (integration) | 2 | ✓ Pass |
| V1-only mode (integration) | 2 | ✓ Pass |

---

## Conclusion

The Experiment Orchestrator and Strategy Factory audit is complete. All identified issues have been addressed:
- Metric formulas are correct and documented
- Infinity/NaN handling is robust
- Determinism is proven via tests
- Strict mode catches data integrity issues
- Thresholds are fully configurable
- Artifacts have stable schema with version tracking
- **Performance optimizations achieve 93-124x speedup on hot path**
- **Execution mode contract implemented with deterministic routing**
- **Equivalence suite validates standard == compiled for golden strategies**
- **IndicatorCache hardened with capacity limits and LRU eviction**
- **CI performance regression guard active**
- **V1 Pragmatic Universe prevents survivorship bias using time-dependent eligibility**
- **V2 Eligibility-Aware Universe adds event-based eligibility with DB timeline support**

**Status**: Ready for institutional production use.

