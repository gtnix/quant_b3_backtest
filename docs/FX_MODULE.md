# FX Module - Multi-Currency Performance Measurement

**Version**: V1.1 "Fund-Grade"  
**Schema**: `fx_report_v1.1`

## Overview

The FX Module enables multi-currency portfolio measurement by providing:

1. **Currency Types**: Type-safe currency handling with `Currency`, `Money`, `FxPair`, and `FxRate`
2. **FX Provider**: Point-in-time rate lookups with LOCF (Last Observation Carried Forward)
3. **NAV Conversion**: Convert portfolio values to a configurable base currency
4. **FX Attribution**: Decompose returns into asset, FX, and interaction components
5. **Complete Audit Trail**: Track resolution method, dates, and rates for compliance
6. **Auto-updating Data Pipeline**: Python scrapers for BCB and FRED APIs

## V1.1 Changes (Fund-Grade Hardening)

### What's New

| Feature | Description |
|---------|-------------|
| **FxResolutionMethod** | Tracks how each rate was resolved: Direct, Inverse, LOCF, InverseLOCF, Identity, Triangulated |
| **Full Audit Trail** | Records `pair_requested`, `date_requested`, `pair_resolved`, `date_resolved`, `method` |
| **FxGapMode** | Formalized gap counting: CalendarDays (default) or BusinessDays |
| **FxMissingAction** | Action when gap exceeded: Error (default) or MarkIncomplete |
| **schema_version** | JSON reports include version for backward compatibility |
| **Property Tests** | proptest-based invariant verification |
| **Golden Tests** | Schema stability tests against golden JSON files |

---

## FX Convention (CRITICAL)

### Rate Semantics

```
FxPair { base: USD, quote: BRL }

USD/BRL = 5.50 means:
  - 1 USD = 5.50 BRL
  - rate_quote_per_base() returns 5.50
  - To convert 100 USD to BRL: 100 × 5.50 = 550 BRL
```

### Semantic Helpers

```rust
let rate = FxRate::new(FxPair::USD_BRL, dec!(5.50), date);

// Unambiguous accessors
assert_eq!(rate.rate_quote_per_base(), dec!(5.50));
assert_eq!(rate.describe(), "1 USD = 5.50 BRL");
assert_eq!(rate.describe_with_date(), "1 USD = 5.50 BRL (2024-12-27)");
```

### Never Confuse Directions

| Operation | Formula |
|-----------|---------|
| USD → BRL | `amount_usd × rate` |
| BRL → USD | `amount_brl ÷ rate` |
| Get inverse | `rate.inverse()` returns BRL/USD = 0.1818 |

---

## Data Model

### Currency

```rust
pub enum Currency {
    BRL,  // Brazilian Real
    USD,  // US Dollar
    EUR,  // Euro
}
```

Each market maps to a currency:
- `Market::BR` → `Currency::BRL`
- `Market::US` → `Currency::USD`

### Money

```rust
pub struct Money {
    amount: Decimal,
    currency: Currency,
}
```

**Invariants**:
- Arithmetic only allowed between same currencies
- Use `convert_money()` for cross-currency operations
- All amounts use `Decimal` for precision

### FxPair

```rust
pub struct FxPair {
    base: Currency,   // "1 unit of this"
    quote: Currency,  // "equals rate units of this"
}
```

**Convention**: `USD/BRL = 5.50` means 1 USD = 5.50 BRL

### FxRate

```rust
pub struct FxRate {
    pair: FxPair,
    rate: Decimal,    // quote per base
    date: NaiveDate,
}
```

---

## Resolution Method (V1.1)

### FxResolutionMethod Enum

```rust
pub enum FxResolutionMethod {
    Direct,       // Rate found for exact pair and date
    Inverse,      // Rate obtained by inverting stored pair
    LOCF,         // Last Observation Carried Forward
    InverseLOCF,  // Both inverse and LOCF applied
    Identity,     // Same currency (rate = 1)
    Triangulated, // Cross-rate (V2)
}
```

### Audit Trail

Every rate lookup records:

| Field | Description |
|-------|-------------|
| `pair_requested` | The pair originally requested (e.g., "USD/BRL") |
| `date_requested` | The date for which rate was requested |
| `pair_resolved` | The pair actually used (may differ if inverse) |
| `date_resolved` | The date of the actual rate observation |
| `rate` | The exchange rate used |
| `method` | How the rate was resolved |

### Example Audit JSON

```json
{
  "fx_rates_used": [
    {
      "pair_requested": "USD/BRL",
      "date_requested": "2024-12-27",
      "pair_resolved": "USD/BRL",
      "date_resolved": "2024-12-24",
      "rate": "5.50",
      "method": "LOCF"
    }
  ]
}
```

---

## Missing Data Policy (V1.1)

### Formalized Configuration

```rust
pub struct PerformanceReportingConfig {
    pub base_currency: Currency,           // Default: BRL
    pub fx_missing_policy: FxMissingPolicy,
    pub fx_max_gap_days: u32,              // Default: 5
    pub fx_gap_mode: FxGapMode,            // Default: CalendarDays
    pub fx_missing_action: FxMissingAction, // Default: Error
}
```

### FxMissingPolicy

| Policy | Behavior |
|--------|----------|
| `LastObservationCarriedForward` | Use most recent rate within gap limit (default) |
| `ErrorOnMissing` | Require exact date match |
| `UseOne` | Use rate=1 (same currency only) |

### FxGapMode

| Mode | Counting |
|------|----------|
| `CalendarDays` | Count all days including weekends (default) |
| `BusinessDays` | Count weekdays only (no holidays) |

### FxMissingAction

| Action | Behavior |
|--------|----------|
| `Error` | Hard failure when gap exceeded (default) |
| `MarkIncomplete` | Warn and continue with incomplete snapshot |

### Weekend/Holiday Behavior

With `CalendarDays` (default) and `max_gap_days = 5`:

| Scenario | Gap | Result |
|----------|-----|--------|
| Friday → Saturday | 1 day | LOCF ✓ |
| Friday → Sunday | 2 days | LOCF ✓ |
| Friday → Monday | 3 days | LOCF ✓ |
| Thursday → Tuesday (holiday Monday) | 5 days | LOCF ✓ |
| Monday → Tuesday (no Monday data) | 6 days | Error ✗ |

---

## FX Data Pipeline

### Sources

| Source | Pairs | API |
|--------|-------|-----|
| BCB (Banco Central do Brasil) | USD/BRL, EUR/BRL | `api.bcb.gov.br` |
| FRED (Federal Reserve) | EUR/USD | `api.stlouisfed.org` |

### Storage

Files stored in `cache/fx/` as CSV:

```csv
date,rate,source
2024-01-02,4.8521,BCB
2024-01-03,4.8934,BCB
```

### Usage

```bash
# Full sync from inception
python -m datahub_fx sync

# Incremental update
python -m datahub_fx update

# Check status
python -m datahub_fx status

# Show recent rates
python -m datahub_fx show USD/BRL --tail 10
```

---

## FX Attribution

### Formula

For a position in local currency L, reporting in base currency B:

```
Value_B(t) = Value_L(t) × FX(t)

where FX = B per 1 L (e.g., USD/BRL = 5.50)
```

### Return Decomposition (3-Term)

**Multiplicative**:
```
(1 + R_total_B) = (1 + R_asset_L) × (1 + R_fx)
```

**Additive (3 terms)**:
```
R_total_B = R_asset + R_fx + R_interaction

where:
  R_asset = V_L(t1) / V_L(t0) - 1
  R_fx = FX(t1) / FX(t0) - 1
  R_interaction = R_asset × R_fx
```

### Example

| Metric | Value |
|--------|-------|
| Start value | $1,000 USD @ 5.00 = R$5,000 |
| End value | $1,100 USD @ 5.50 = R$6,050 |
| Asset return | 10% ($1,100 / $1,000 - 1) |
| FX return | 10% (5.50 / 5.00 - 1) |
| Interaction | 1% (10% × 10%) |
| **Total return** | **21%** (R$6,050 / R$5,000 - 1) |

### Verification (Property Test)

```rust
// Multiplicative identity
assert_eq!(
    (1 + asset_return) * (1 + fx_return),
    1 + total_return
);

// Additive decomposition
assert_eq!(
    asset_return + fx_return + interaction,
    total_return
);
```

---

## Rounding Policy

### Core Principle

**No rounding in core calculations.** Round only in display/export.

| Layer | Precision |
|-------|-----------|
| Internal calculations | Full Decimal precision |
| Money display | 2 decimal places |
| FX rates display | 4 decimal places |
| Percentages display | 2 decimal places |

### Rationale

- Prevents accumulation of rounding errors
- Enables exact verification of invariants
- Allows consumers to round as needed

---

## API Usage

### Setting Up FX-Enabled Engine

```rust
use backtester_intelligence::currency::Currency;
use backtester_intelligence::fx::InMemoryFxProvider;
use backtester_intelligence::performance::{PerformanceEngine, PerformanceConfig};
use std::sync::Arc;

// Load FX data
let mut provider = InMemoryFxProvider::new();
provider.add_rate(FxPair::USD_BRL, date, dec!(5.50));

// Create engine with FX support
let config = PerformanceConfig::default()
    .with_base_currency(Currency::BRL);
let mut engine = PerformanceEngine::with_fx(
    config,
    initial_capital,
    Arc::new(provider),
);

// Record trades (unchanged)
engine.record_buy(date, "AAPL", 10, dec!(150), dec!(1.5), Market::US);
engine.record_buy(date, "PETR4", 100, dec!(30), dec!(3), Market::BR);

// Generate snapshot with FX fields
let snapshot = engine.generate_snapshot(date, cash, &prices);

// Access consolidated view
println!("Equity (local): {}", snapshot.equity);
println!("Equity (base):  {:?}", snapshot.equity_base);
println!("Base currency:  {:?}", snapshot.base_currency);

// Generate FX attribution
let attribution = engine.generate_fx_attribution(start_date, end_date)?;
println!("Asset return:  {}%", attribution.portfolio_asset_return * 100);
println!("FX return:     {}%", attribution.portfolio_fx_return * 100);
println!("Interaction:   {}%", attribution.portfolio_interaction * 100);
```

### Converting Money with Audit

```rust
use backtester_intelligence::fx::{convert_money_with_audit, InMemoryFxProvider};
use backtester_intelligence::currency::{Currency, Money};

let provider = /* load provider */;
let usd = Money::new(dec!(1000), Currency::USD);

// Convert with full audit trail
let result = convert_money_with_audit(&usd, Currency::BRL, date, &provider, 5)?;

println!("Amount: {}", result.money);
println!("Rate: {}", result.rate_used);
println!("Method: {:?}", result.method);
println!("Gap days: {}", result.gap_days());
```

---

## Reporting

### JSON Output (V1.1)

```json
{
  "schema_version": "fx_report_v1.1",
  "date": "2024-12-27",
  "equity": "99993.50",
  "base_currency": "BRL",
  "equity_base": "105993.50",
  "fx_attribution": {
    "base_currency": "BRL",
    "asset_return_pct": "10.00",
    "fx_return_pct": "5.00",
    "interaction_pct": "0.50",
    "total_return_base_pct": "15.50",
    "by_currency": [
      {
        "currency": "BRL",
        "asset_return_pct": "10.00",
        "fx_return_pct": "0.00",
        "interaction_pct": "0.00",
        "weight_pct": "50.00"
      },
      {
        "currency": "USD",
        "asset_return_pct": "10.00",
        "fx_return_pct": "10.00",
        "interaction_pct": "1.00",
        "weight_pct": "50.00"
      }
    ]
  },
  "exposure_by_currency": [
    {"currency": "BRL", "value_local": "5000.00", "value_base": "5000.00", "weight_pct": "40.00"},
    {"currency": "USD", "value_local": "1500.00", "value_base": "7500.00", "weight_pct": "60.00"}
  ],
  "fx_rates_used": [
    {
      "pair_requested": "USD/BRL",
      "date_requested": "2024-12-27",
      "pair_resolved": "USD/BRL",
      "date_resolved": "2024-12-27",
      "rate": "5.00",
      "method": "Direct"
    }
  ]
}
```

### CIO View

```
CIO VIEW 2024-12-27
========================================
Total Return: 5.00%
Annualized Return: 12.00%
Max Drawdown: 5.00%
Sharpe Ratio: 1.25
VaR 95%: -2500.00
Total Costs: 180.00
Turnover: 17.10%
Positions: 10

FX ATTRIBUTION (Base: BRL)
----------------------------------------
Total Return (BRL): 15.50%
Asset Return: 10.00%
FX Return: 5.00%
Interaction: 0.50%
```

---

## Invariants

1. **No implicit conversion**: All currency conversion goes through explicit `convert_money()` or `FxRateProvider`

2. **Separation preserved**: Internal BR/US portfolios remain separate; consolidation only via explicit FX conversion

3. **Precision**: All monetary values use `Decimal` (never `f64`)

4. **Complete audit trail**: Snapshots record the FX rates, dates, and methods used for conversion

5. **Decomposition verified**: `R_asset + R_fx + R_interaction = R_total` (exact with Decimal)

6. **Determinism**: Same inputs always produce identical outputs

7. **Schema stability**: JSON format versioned for backward compatibility

---

## Testing

### Test Types

| Test Type | File | Purpose |
|-----------|------|---------|
| Unit tests | `fx_unit.rs` | Currency, Money, FxPair, FxRate |
| Integration | `fx_integration.rs` | Multi-currency scenarios |
| Property | `fx_proptest.rs` | Invariant verification |
| Golden | `fx_golden.rs` | Schema stability |
| E2E | `integration_e2e_performance.rs` | Full backtest flow |

### Running Tests

```bash
# All FX tests
cargo test --package backtester_intelligence fx

# Property tests
cargo test --package backtester_intelligence --test fx_proptest

# Golden tests
cargo test --package backtester_intelligence --test fx_golden

# Edge cases
cargo test --package backtester_intelligence --test fx_unit edge_cases

# E2E tests
cargo test --test integration_e2e_performance e2e_consolidated
cargo test --test integration_e2e_performance e2e_fx_attribution
```

---

## Future Enhancements (V2+)

1. **Cross rates**: Support EUR→BRL via EUR/USD × USD/BRL (Triangulated method)
2. **Hedging**: Model FX forward contracts and hedged positions
3. **Multi-currency cash buckets**: Track cash in multiple currencies
4. **Intraday rates**: Support intraday FX data for HFT scenarios
5. **Additional currencies**: GBP, JPY, CHF, etc.
6. **Business day calendar**: Holiday-aware gap calculation
