# Dividend Integration Migration Guide

This document describes how to use the new dividend cashflow feature and migrate from the deprecated `SimulationEngine` to the new `UnifiedEngine`.

## What's New

### UnifiedEngine

The `UnifiedEngine` in `backtester_engine` now provides:

- **Dividend cashflow processing**: Dividends are credited as cash on `ex_date`
- **Anti-double-count policy**: Signals use adjusted prices, valuation uses raw prices
- **Full observability**: Dividend events tracked in trace and timeseries

### New Artifacts

**timeseries.csv** now includes:
- `dividend_cashflow`: Daily dividend cashflow received
- `dividend_cumulative`: Cumulative dividend cashflow to date

**trace.jsonl** now includes:
- `dividend_policy` entry: Records the anti-double-count policy applied
- `mode_fallback` entry: Records execution mode fallback (if any)

**metadata.json** now includes:
```json
{
  "dividends_enabled": true,
  "dividend_policy": {
    "signals_price": "adjusted",
    "valuation_price": "raw",
    "dividends_as_cashflow": true
  },
  "total_dividend_cashflow": "1234.56",
  "dividend_count": 4,
  "mode_fallback_reason": null
}
```

## Anti-Double-Count Policy

**Critical Rule**: Never use adjusted prices for valuation when dividends are enabled as cashflow.

| Component | Price Type | Reason |
|-----------|-----------|--------|
| Signals/Indicators | Adjusted | Smooth returns for ranking |
| Mark-to-Market | Raw | Dividends enter via cashflow |
| Equity Curve | Raw + Cashflow | Accurate economic return |

### Policy Enforcement

The `UnifiedEngine` validates this policy:

```rust
// This will return an error
let config = UnifiedEngineConfig {
    enable_dividends: true,
    valuation_price_type: PriceType::Signals, // WRONG: adjusted + cashflow = double count
    ..Default::default()
};
let engine = UnifiedEngine::with_config(config);
engine.validate_anti_double_count()?; // Err(PolicyViolation)
```

## Enabling Dividends

### RunnerConfig

```rust
let config = RunnerConfig {
    enable_dividends: true,  // Enable dividend cashflow
    initial_capital: Decimal::from(1_000_000),
    ..Default::default()
};
let runner = ExperimentRunner::with_config(config);
```

### Direct UnifiedEngine Usage

```rust
use backtester_engine::{UnifiedEngine, UnifiedEngineConfig, DividendEvent, PriceType};

let config = UnifiedEngineConfig {
    initial_capital: dec!(1_000_000),
    enable_dividends: true,
    valuation_price_type: PriceType::Valuation, // Raw prices
    ..Default::default()
};

let mut engine = UnifiedEngine::with_config(config);

// Load dividends for the simulation period
engine.load_dividends(vec![
    DividendEvent {
        symbol: "TAEE11".to_string(),
        ex_date: NaiveDate::from_ymd_opt(2024, 3, 15).unwrap(),
        rate: dec!(0.50),
    },
]);

// Process each day
for date in trading_days {
    let day_result = engine.process_day(date, &bars, candidates);
    // day_result.dividend_cashflow contains today's dividend credit
}

// Get final result
let result = engine.get_result();
// result.total_dividend_cashflow contains sum of all dividends
```

## Execution Mode and Dividends

**Fast mode does NOT support dividend cashflow**. When dividends are enabled and Fast mode is requested, the engine automatically falls back to Compiled mode.

```rust
// This will fallback to Compiled
let config = RunnerConfig {
    enable_dividends: true,
    execution_mode: ExecutionMode::Fast,
    ..Default::default()
};
// metadata.execution_mode will be Compiled
// metadata.mode_fallback_reason will explain why
```

The fallback is deterministic and recorded in:
- `metadata.mode_fallback_reason`
- `trace.jsonl` with `mode_fallback` entry

## Breaking Changes

### Deprecated Types

```rust
// OLD (deprecated)
use backtester_engine::SimulationEngine;  // ❌

// NEW (recommended)
use backtester_engine::UnifiedEngine;     // ✓
```

### New Required Fields

If you manually construct `RunMetadata`, you must include:

```rust
RunMetadata {
    // ... existing fields ...
    dividends_enabled: bool,
    dividend_policy: Option<DividendPolicyInfo>,
    total_dividend_cashflow: Option<Decimal>,
    dividend_count: Option<usize>,
    mode_fallback_reason: Option<String>,
}
```

### New Required Fields in EquityPoint

```rust
EquityPoint {
    // ... existing fields ...
    dividend_cashflow: Option<Decimal>,
    dividend_cumulative: Option<Decimal>,
}
```

## Testing

Run the dividend integration tests:

```bash
cargo test -p backtester_strategy --test runner_dividend_e2e
```

This validates:
- T1: Determinism with dividends enabled
- T2: Policy correctly recorded in metadata and trace
- T3: Fast mode fallback when dividends enabled
- T4: Fallback trace entry present

## See Also

- `docs/policies/corporate_actions_pnl.md` - Full policy documentation
- `crates/backtester_engine/src/unified.rs` - UnifiedEngine implementation
- `crates/backtester_intelligence/src/dividends/` - Dividend index and types


