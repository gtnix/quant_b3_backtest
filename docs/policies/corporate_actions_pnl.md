# Corporate Actions & Dividends: PnL Policy

## Overview

This document defines the **official policy** for handling corporate actions (primarily dividends) in the backtester to ensure economically correct PnL calculation **without double-counting**.

## Policy Decisions (FIXED)

### P1: Adjusted Prices for Signals: YES

**Status**: Implemented

Indicators, filters, and asset selection use `adjustedClose` prices by default. This:
- Avoids artificial gaps in momentum/trend signals
- Ensures continuity in return calculations
- Follows industry standard for quantitative analysis

**Code Reference**: `backtester_io/src/lib.rs` - BrapiLoader uses `adjustedClose` as the close price.

### P2: Dividends as Cashflow: YES

**Status**: Implemented

Dividends enter the portfolio as explicit cashflow events on the **ex-date**. This:
- Provides clear audit trail
- Separates capital gains from income
- Enables accurate tax-lot tracking

**Code Reference**: `backtester_intelligence/src/dividends/` - Dividend engine implementation.

### P3: Anti-Double-Count Rule: CRITICAL

**Status**: Implemented and tested

This is the most important policy decision.

#### The Problem

If we use **adjusted prices** for PnL calculation AND add dividends as cashflow, dividends are counted twice:
1. Once via the adjusted price series (which "absorbs" dividends)
2. Again via explicit cashflow

#### The Solution

| Use Case | Price Type | Dividend Handling |
|----------|------------|-------------------|
| Signals/Indicators | Adjusted (`adjustedClose`) | Implicit in price |
| Mark-to-Market | Raw (`close`) | Explicit cashflow |
| Equity Curve | Raw (`close`) | Explicit cashflow |
| Order Execution | Raw (`close`) | N/A |

**Key Invariant**:
```
equity_raw(T) + Σ dividends(0..T) ≈ equity_adjusted(T)
```

This is proven via test `t1_buyhold_economic_return_matches_adjusted`.

## Implementation Details

### Dividend Crediting: Ex-Date (Not Payment Date)

Dividends are credited on the **ex_date** because:

1. **Market Consistency**: Share price drops by dividend amount on ex-date
2. **Series Alignment**: Adjusted price series reflect dividends on ex-date
3. **Simplicity**: No need to track "dividend receivable" between ex-date and payment

```rust
// Shares held = position at END of T-1 (day before ex-date)
// Dividend credited = rate × shares_held
if date == ex_date {
    let cashflow = div.rate * Decimal::from(position.shares);
    portfolio.add_cash(cashflow);
}
```

### Shares Eligible for Dividend

**Convention**: Shares held at end of day T-1 receive dividends on ex-date T.

| Scenario | Receives Dividend? |
|----------|-------------------|
| Held position before ex-date | YES |
| Buy on ex-date | NO |
| Sell on ex-date | YES (sold after record) |

### DualPriceBar Structure

```rust
pub struct DualPriceBar {
    pub symbol: String,
    pub date: NaiveDate,
    pub adjusted_close: Decimal,  // For signals
    pub raw_close: Decimal,       // For valuation
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub volume: Decimal,
}
```

### Price Type Enum

```rust
pub enum PriceType {
    Signals,    // Use adjusted
    Valuation,  // Use raw
}
```

## Artifacts

### timeseries.csv

New columns added:
```csv
date,equity,drawdown,exposure,vol_exante,vol_expost,dividend_cashflow,dividend_cumulative
```

### trace.jsonl

Dividend events are logged:
```json
{"type": "dividend", "date": "2024-03-15", "symbol": "TAEE11", "rate": 0.45, "shares": 1000, "cashflow": 450.00}
```

## Numerical Example

### Setup
- Buy 1000 shares of TAEE11 at R$40.00 on 2024-01-01
- Initial equity: R$100,000

### Ex-Date: 2024-03-15
- Dividend: R$0.50/share
- Raw price drops: R$40.50 → R$40.00
- Adjusted price: R$40.50 (no visible drop)

### Calculations

**RAW + Cashflow Method** (Correct):
```
Position value (raw):    1000 × 40.00 = R$40,000
Dividend cashflow:       1000 × 0.50  = R$   500
Cash balance:            R$60,000 + R$500 = R$60,500
Total equity:            R$60,500 + R$40,000 = R$100,500
```

**Adjusted Method** (Also Correct):
```
Position value (adj):    1000 × 40.50 = R$40,500
Cash balance:            R$60,000 (no dividend added)
Total equity:            R$60,000 + R$40,500 = R$100,500
```

**WRONG: Adjusted + Cashflow** (Double Count!):
```
Position value (adj):    1000 × 40.50 = R$40,500
Dividend cashflow:       R$500 (WRONG to add)
Cash balance:            R$60,000 + R$500 = R$60,500
Total equity:            R$60,500 + R$40,500 = R$101,000  ← WRONG!
```

## Gotchas

### 1. Data Source Consistency

Ensure raw and adjusted prices come from the same source and are properly aligned. Mixing sources can cause subtle discrepancies.

### 2. Corporate Actions Beyond Dividends

This policy currently covers:
- ✅ Cash dividends
- ⚠️ Stock dividends (TODO: share adjustment)
- ⚠️ Stock splits (TODO: price/share adjustment)
- ⚠️ Mergers/Spinoffs (TODO)

### 3. Ex-Date vs Record Date

We use **ex-date** consistently. Some data sources provide record date instead. Ensure your data uses ex-date or convert appropriately (typically ex-date = record-date - 1 business day).

### 4. Partial Day Trading

Our model assumes positions are held overnight. Intraday trading on ex-date may require additional logic.

## Policy Enforcement

### Anti-Double-Count Guard

The `UnifiedEngine` validates configuration at runtime:

```rust
impl UnifiedEngine {
    pub fn validate_anti_double_count(&self) -> Result<(), PolicyViolation> {
        if self.config.enable_dividends && 
           self.config.valuation_price_type == PriceType::Signals {
            return Err(PolicyViolation {
                message: "Cannot use adjusted prices for valuation with dividends enabled"
            });
        }
        Ok(())
    }
}
```

### Fast Mode Restriction

Fast execution mode does NOT support dividend cashflow tracking. When dividends are enabled:

1. Requested: `ExecutionMode::Fast` with `enable_dividends: true`
2. Resolved: Automatic fallback to `ExecutionMode::Compiled`
3. Recorded: `metadata.mode_fallback_reason` explains why

This is deterministic and never silently ignores dividends.

### Trace Observability

Every run records the policy in `trace.jsonl`:

```json
{"step": 0, "block_type": "dividend_policy", "message": "Anti-double-count policy applied", 
 "params_effective": {"signals_price": "adjusted", "valuation_price": "raw", "dividends_as_cashflow": true}}
```

## Tests

All policy assertions are verified via integration tests:

**Unit/Integration Tests** (`tests/dividend_integration.rs`):

| Test | Description |
|------|-------------|
| `t1_buyhold_economic_return_matches_adjusted` | RAW+cashflow ≈ ADJ return |
| `t2_anti_double_count_validation` | ADJ+cashflow > correct (proves double-count) |
| `t3_determinism_with_dividends` | Same inputs → same outputs |
| `t4_*` | Edge cases (no position, partial, buy on ex-date) |

**End-to-End Tests** (`tests/runner_dividend_e2e.rs`):

| Test | Description |
|------|-------------|
| `t1_runner_determinism_with_dividends` | Same config → identical metrics |
| `t2_policy_recorded_in_metadata` | Policy in metadata.json |
| `t2_policy_trace_entry_present` | Policy entry in trace.jsonl |
| `t3_fast_mode_fallback_with_dividends` | Fast → Compiled when dividends on |
| `t4_fallback_trace_entry_present` | Fallback reason in trace |

## References

- `crates/backtester_engine/src/unified.rs` - UnifiedEngine implementation
- `crates/backtester_intelligence/src/dividends/` - Dividend module
- `crates/backtester_intelligence/tests/dividend_integration.rs` - Policy tests
- `crates/backtester_strategy/tests/runner_dividend_e2e.rs` - End-to-end tests
- `docs/MIGRATION_DIVIDENDS.md` - Migration guide

