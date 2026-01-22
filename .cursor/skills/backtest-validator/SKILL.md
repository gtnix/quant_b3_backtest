---
name: backtest-validator
description: "Metrics validation and sanity checks for backtest results"
triggers:
  - command: "/backtest-validator"
    description: "Invoke for metrics validation and anomaly detection"
domain_knowledge:
  - sharpe ratio validation
  - metrics sanity checks
  - zero-trade detection
  - artificial metrics detection
  - numerical edge cases
---

# Backtest Validator

## Role

Backtest Metrics Validator / Sanity Check Expert.
Responsible for detecting invalid, artificial, or suspicious metrics in backtest results.

---

## Expertise Map

### Sharpe Ratio Validation
- **Valid range**: Clamped to [-10.0, 10.0]
- **Zero trades**: Must return 0.0 or mark as INVALID
- **Zero variance**: Returns 0.0 (undefined Sharpe)
- **Suspicious**: Sharpe = 10.0 with trades = 0 is a BUG

### Zero Trade Detection
- Strategies with 0 trades are INVALID
- Never accept artificial metrics from zero-trade runs
- Check `total_trades` field in all fitness evaluations
- Source: `crates/combiner_core/src/fitness.rs` line 207

### Metrics Clamping Policy
| Metric | Clamp Range | Zero/Undefined |
|--------|-------------|----------------|
| Sharpe | [-10, 10] | 0.0 |
| Sortino | [-20, 20] | 0.0 |
| Calmar | [-10, 10] | 0.0 |
| Profit Factor | [0, 100] | 0.0 |

### Numerical Edge Cases
- `std_dev = 0` → Sharpe = 0.0
- `downside_dev = 0` → Sortino = 0.0
- `max_drawdown = 0` → Calmar = 0.0
- `gross_loss = 0` → Profit Factor = 0.0

---

## When to Use

**INVOKE this skill when:**
- Sharpe/Sortino/Calmar shows suspicious values (10.0, -10.0)
- Strategy reports 0 trades but non-zero metrics
- Metrics seem artificially high or low
- Debugging metrics calculation code
- Validating SIMD metrics implementations

**DO NOT use this skill when:**
- Running WFA/CPCV validation (use `/risk-analyst`)
- Optimizing metrics calculation performance (use `/quant-engineer`)
- Investigating data quality issues (use `/data-engineer`)

---

## Operating Rules

### Hard Constraints

1. **Zero trades = INVALID**
   - Never accept any metrics from 0-trade runs
   - Mark fitness as invalid immediately

2. **Clamped values require investigation**
   - Sharpe = 10.0 or -10.0 must be investigated
   - Often indicates numerical issues, not real performance

3. **Variance = 0 means undefined, not infinite**
   - Sharpe(returns with no variance) = 0.0, not ±∞
   - This was the root cause of the Sharpe=10.0 bug

4. **All metrics must be NET of costs**
   - GROSS metrics can hide cost-dependent issues
   - Validate that costs were applied

---

## Repo Anchors

### Critical Files

| File | Purpose |
|------|---------|
| `crates/combiner_core/src/simd_metrics.rs` | SIMD metrics calculations |
| `crates/backtester_core/src/simd.rs` | Core SIMD utilities |
| `crates/combiner_core/src/fitness.rs` | Fitness evaluation (zero trade check) |
| `crates/backtester_reports/src/lib.rs` | BacktestResult calculation |
| `crates/backtester_strategy/src/experiment/types.rs` | RunMetrics validation |

### Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `sharpe_simd()` | simd_metrics.rs | SIMD Sharpe calculation |
| `calculate_all_metrics()` | simd_metrics.rs | Batch metrics |
| `Fitness::calculate()` | fitness.rs | Zero trade check |
| `is_suspicious()` | experiment/types.rs | Anomaly detection |

---

## Validation Checklist

### Pre-Backtest
```
[ ] Data has sufficient bars for lookback period
[ ] Universe has tradeable symbols
[ ] Cost model is configured
```

### Post-Backtest
```
[ ] total_trades > 0 (or marked invalid)
[ ] Sharpe not clamped to boundary (±10.0)
[ ] Sortino not clamped to boundary (±20.0)
[ ] Calmar reasonable (< 5.0 for most strategies)
[ ] Profit Factor < 100 (unless perfect strategy)
```

### Red Flags
- Sharpe = 10.0 with trades = 0 → **BUG**
- Sharpe = 10.0 with variance = 0 → **BUG (fixed)**
- Sortino = 10.0 with no downside → **Now returns 0.0**
- All metrics = 0.0 → Check if strategy executed

---

## Bug History

### Sharpe=10.0 Bug (Fixed 2026-01-21)

**Symptom**: US mining returned Sharpe=10.000 with trades=0

**Root Cause**: When no downside deviation existed, Sortino/Calmar returned 10.0 instead of 0.0

**Fix Applied**:
- `simd_metrics.rs`: Changed fallback from 10.0 to 0.0
- `backtester_core/src/simd.rs`: Same fix

**Affected Functions**:
- `sortino_simd_batch()` 
- `sortino_scalar()`
- `calmar_ratio()`
- `calculate_all_metrics()`
- `calculate_all_metrics_avx512()`

---

## Collaboration Hooks

### Handoff to `/risk-analyst`
When metrics are validated, hand off for WFA/PBO/DSR analysis.

### Handoff to `/quant-engineer`
When performance issues in metrics calculation are found.

### Receiving from `/omp-operator`
When mining produces suspicious results for investigation.
