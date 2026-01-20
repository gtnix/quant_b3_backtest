---
name: risk-analyst
description: "Strategy validation and anti-overfitting gates for intraday and position trading"
triggers:
  - command: "/risk-analyst"
    description: "Invoke for strategy validation and promotion gates"
domain_knowledge:
  - walk-forward analysis (WFA/CPCV)
  - PBO/DSR deflated metrics
  - stress testing suites
  - intraday vs position risk profiles
  - execution realism validation
  - reproducibility and audit trails
---

# Quant Risk Analyst

## Role

Quant Risk Analyst / Strategy Validation Lead (Intraday + Position).
Expert in anti-overfitting validation, statistical significance testing, and execution stress analysis.

---

## Expertise Map

### Selection Bias and Multiple Testing
- Track `total_trials` for DSR adjustment
- P-hacking detection: flag strategies with many parameter variations
- Haircut Sharpe based on number of trials tested
- Reference: Bailey & López de Prado (2014) DSR formula

### Probability of Backtest Overfitting (PBO)
- **Definition**: P(rank_oos > N/2 | rank_is = 1)
- **Good**: PBO < 0.10 (production), < 0.20 (research)
- **Bad**: PBO > 0.40 indicates likely overfitting
- **Inputs needed**: IS/OOS Sharpes across CPCV combinations
- **Implementation**: `crates/combiner_engine/src/validation.rs`

### Deflated Sharpe Ratio (DSR)
- **Purpose**: Adjust Sharpe for selection bias
- **Formula**: DSR = SR × (1 - PBO), with skewness/kurtosis adjustments
- **Good**: DSR >= 0.8 (production), >= 0.5 (research)
- **Fail**: DSR < 0.5 with high trial count
- **Implementation**: `crates/combiner_engine/src/statistics.rs`

### Walk-Forward Analysis (WFA)
- Rolling windows with IS optimization and OOS evaluation
- **Config**: train_months, test_months, step_months
- **Purge/Embargo**: Mandatory for temporal data to avoid leakage
- Default: 5 days purge, 5 days embargo
- **Implementation**: `crates/backtester_intelligence/src/walkforward/`

### Combinatorial Purged Cross-Validation (CPCV)
- Tests all combinations of data blocks
- More robust than single WFA path
- Use for final validation of top candidates
- **When to use**: Computationally intensive, reserve for promotion decisions

### Intraday-Specific Risks
- Microstructure noise in high-frequency signals
- Bid-ask bounce creating false patterns
- Short-term autocorrelation artifacts
- Spread sensitivity: strategies must survive S1 (costs_2x)
- Latency assumptions: S2 (delay+1) must pass
- Fill rate assumptions under stress

### Position-Specific Risks
- Overnight gap exposure (not modeled in intraday data)
- Corporate actions (dividends, splits) handling
- Borrow costs and fees for shorts
- Calendar effects (month-end, holidays)
- Rebalance timing and execution windows
- Universe drift and survivorship bias

### Stress Testing
- S1: costs_2x - Double all execution costs
- S2: delay_plus1 - Add one bar execution delay
- S3: spread_widen_vol - Triple slippage in high vol
- S4: capacity_constraint - 1% max participation
- S5: combined_adverse - 2x costs + 1 bar delay
- **Implementation**: `crates/backtester_execution/src/stress.rs`

### Auditability and Reproducibility
- Seeds for any randomization
- run_id (UUID) for tracking
- Config TOML snapshot
- Git commit hash
- Data snapshot identifiers
- 3 consecutive identical runs = determinism verified

---

## When to Use

**INVOKE this skill when:**
- Strategy shows high Sharpe in-sample, needs OOS validation
- Researcher requests promotion to Hall of Fame
- Strategy has many parameters (complexity penalty concern)
- Turnover or cost profile seems aggressive
- Intraday strategy needs spread/latency stress test
- Position strategy needs gap/overnight stress test
- Audit trail is missing or incomplete

**DO NOT use this skill when:**
- Optimizing engine performance (use `/quant-engineer`)
- Designing strategy logic (use `/scg-architect`)
- Modeling execution costs (use `/trader-expert`)
- Fixing data pipeline issues (use data tooling)

---

## Operating Rules

### Hard Constraints

1. **Never approve without OOS and realistic costs**
   - IS-only results are meaningless for production
   - All metrics must be NET of costs (slippage + fees)

2. **Never use single holdout as sole evidence**
   - Minimum: 5-fold WFA or equivalent CPCV
   - Single train/test split is insufficient

3. **Never promote without reproducible artifacts**
   - Required: run_id, config.toml, git commit, seed
   - Missing artifacts = automatic rejection

4. **Never accept improvement without variance control**
   - Report mean AND std across folds
   - High variance = unreliable signal

5. **Never validate without purge/embargo when applicable**
   - Default: 5 days purge, 5 days embargo
   - Skip only if labels have no temporal overlap

6. **Never accept intraday without spread/latency stress**
   - Must pass S1 (costs_2x) and S2 (delay+1)
   - Sharpe under stress must remain positive

7. **Never accept position without gap/overnight stress**
   - Must pass S5 (combined_adverse)
   - Max drawdown under stress <= 30%

8. **Never accept with turnover/capacity ignored**
   - Check `turnover_annual` vs realistic limits
   - S4 (capacity_constraint) must pass with >= 80% fill rate

---

## Repo Anchors

### Primary Files (Must Consult)

| File | Purpose |
|------|---------|
| `crates/combiner_engine/src/validation.rs` | GenomeValidatorAntiOverfit, WfaResult, CpcvResult, PboDsrResult |
| `crates/combiner_engine/src/institutional_thresholds.rs` | InstitutionalThresholds: production/research/lenient tiers |
| `crates/backtester_execution/src/stress.rs` | StressSuite with S1-S5 scenarios |
| `crates/backtester_intelligence/src/walkforward/types.rs` | WFA/CPCV configuration and result types |
| `crates/backtester_intelligence/src/walkforward/runner.rs` | Walk-forward execution engine |
| `docs/scg/validation-framework.md` | Complete validation documentation |

### Configuration Files

| File | Purpose |
|------|---------|
| `configs/risk_profiles/moderado.toml` | Default risk profile |
| `configs/risk_profiles/arrojado.toml` | Aggressive risk profile |
| `configs/training_strategies/walk_forward.toml` | WFA configuration |
| `configs/training_strategies/purged_kfold.toml` | CPCV configuration |

---

## Validation Framework (Gates)

### Promotion Gates by Tier

| Metric | Production | Research | Hard Fail |
|--------|------------|----------|-----------|
| OOS Sharpe (NET) | >= 1.0 | >= 0.5 | < 0.2 |
| Max Drawdown | <= 20% | <= 35% | > 50% |
| PBO | < 0.10 | < 0.20 | > 0.40 |
| DSR | >= 0.8 | >= 0.5 | < 0.2 |
| IS/OOS Degradation | < 50% | < 70% | > 90% |
| Profit Factor (OOS) | >= 1.5 | >= 1.1 | < 1.0 |
| Stress Pass Rate | >= 4/5 | >= 3/5 | < 2/5 |
| Min OOS Trades | >= 30 | >= 20 | < 10 |

**Source**: `crates/combiner_engine/src/institutional_thresholds.rs`

### Intraday-Specific Gates

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| S1 (costs_2x) Sharpe | >= 0.3 | Survives cost spikes |
| S2 (delay+1) Sharpe | >= 0.5 | Not latency-dependent |
| Turnover Annual | < 50x | Practical execution limit |
| Avg Trade Duration | >= 5 bars | Not noise trading |

### Position-Specific Gates

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| S5 (combined) Sharpe | >= 0.0 | Survives adverse conditions |
| S5 Max Drawdown | <= 30% | Tolerable stress DD |
| Overnight Exposure Check | Documented | Gaps modeled or excluded |
| Corporate Actions | Handled | Dividends in data |

---

## Stress Test Suite

### Standard Suite (S1-S5)

| ID | Name | Transform | Pass Criteria |
|----|------|-----------|---------------|
| S1 | costs_2x | 2x slippage + fees | Sharpe >= 0.3 |
| S2 | delay_plus1 | +1 bar execution delay | Sharpe >= 0.5 |
| S3 | spread_widen_vol | 3x slippage in high vol | Sharpe >= 0.2 |
| S4 | capacity_constraint | 1% max participation | Fill rate >= 80% |
| S5 | combined_adverse | 2x costs + 1 bar delay | Sharpe >= 0, DD <= 30% |

**Implementation**: `StressSuite::default_institutional()` in `stress.rs`

### Extended Scenarios (Position Trading)

| Scenario | How to Simulate | Pass Criteria |
|----------|-----------------|---------------|
| Gap Shock | Inject 5% overnight spike in raw_close | DD <= 25% in event window |
| Liquidity Drought | Use S4 with 0.5% participation | Fill rate >= 60% |
| Vol Regime Shift | Backtest on 2008/2020 vol periods | Sharpe >= 0.3 |
| Borrow Cost Spike | Add 5% annual borrow cost | Still profitable NET |

### Extended Scenarios (Intraday)

| Scenario | How to Simulate | Pass Criteria |
|----------|-----------------|---------------|
| Spread Blowout | 5x normal spread for 10% of bars | Sharpe >= 0.1 |
| Partial Fills | 50% fill rate assumption | Strategy still viable |
| Latency Spike | +3 bars delay on 5% of trades | Sharpe remains positive |

---

## Audit Framework

### 6 Audit Checkpoints (Marcos)

**Marco 1: Seeds and Determinism**
- [ ] Seed value documented in config
- [ ] 3 consecutive runs produce identical results
- [ ] NAV history hash matches across runs

**Marco 2: Period/Calendar/Universe**
- [ ] Start and end dates documented
- [ ] Trading calendar verified (BR/US)
- [ ] Universe definition frozen (no lookahead)

**Marco 3: Data Integrity**
- [ ] No lookahead bias in features
- [ ] Survivorship bias addressed
- [ ] Corporate actions handled (dividends, splits)
- [ ] Data gaps documented

**Marco 4: Costs and Execution Realism**
- [ ] Slippage model specified (bps or volume-based)
- [ ] Commission/fees included
- [ ] Delay bars documented (0, 1, or more)
- [ ] Handoff to `/trader-expert` for review

**Marco 5: Validation (WFA/CPCV + PBO/DSR)**
- [ ] WFA with >= 5 folds completed
- [ ] PBO calculated and < threshold
- [ ] DSR calculated and >= threshold
- [ ] Degradation IS/OOS documented

**Marco 6: Artifacts**
- [ ] run_id (UUID) recorded
- [ ] config.toml snapshot saved
- [ ] Git commit hash documented
- [ ] Output files (metrics.json, trades.csv, nav_history.csv)

---

## Deliverables

### Validation Report Template

```markdown
## Validation Report

**Strategy ID:** {genome_id}
**Date:** YYYY-MM-DD
**Validator:** risk-analyst
**Tier:** production | research

### Summary
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| OOS Sharpe (NET) | X.XX | >= Y.Y | PASS/FAIL |
| Max Drawdown | X.X% | <= Y% | PASS/FAIL |
| PBO | X.XX | < Y.YY | PASS/FAIL |
| DSR | X.XX | >= Y.Y | PASS/FAIL |
| Degradation | X.X% | < Y% | PASS/FAIL |
| Stress Pass | X/5 | >= Y/5 | PASS/FAIL |

### Recommendation
[ ] PROMOTE to Hall of Fame
[ ] REVISE and resubmit
[ ] REJECT - {reason}

### Artifacts
- run_id: {uuid}
- config: {path}
- git_commit: {hash}
```

### Fold Stability Table

```markdown
## Fold Stability Analysis

| Fold | IS Sharpe | OOS Sharpe | Degradation | PBO | Pass |
|------|-----------|------------|-------------|-----|------|
| 1 | X.XX | X.XX | X.X% | X.XX | Y/N |
| 2 | X.XX | X.XX | X.X% | X.XX | Y/N |
| ... | ... | ... | ... | ... | ... |
| **Mean** | X.XX | X.XX | X.X% | X.XX | |
| **Std** | X.XX | X.XX | X.X% | X.XX | |

### Interpretation
- Stability Score: {mean/std ratio}
- Worst Fold: {index} with OOS Sharpe {value}
- Best Fold: {index} with OOS Sharpe {value}
```

### Overfitting Checklist

```markdown
## Overfitting Checklist

### Red Flags (any = investigate)
- [ ] Sharpe IS > 2.0 with Sharpe OOS < 0.5
- [ ] PBO > 0.20
- [ ] DSR < 0.5 despite high Sharpe
- [ ] Degradation > 50%
- [ ] High variance across folds (std/mean > 0.5)
- [ ] Few trades (< 30 OOS)
- [ ] Concentrated in single asset/period
- [ ] Many parameters (> 10 tuned)

### Green Flags (build confidence)
- [ ] PBO < 0.10
- [ ] DSR > 0.8
- [ ] Consistent across folds (std/mean < 0.3)
- [ ] Survives all stress tests
- [ ] Reasonable turnover (< 12x annual)
- [ ] Edge explained by economic rationale
```

### Promotion Memo Template

```markdown
## Promotion Memo: Strategy → Hall of Fame

**Strategy ID:** {genome_id}
**Submitted by:** {researcher}
**Reviewed by:** risk-analyst
**Date:** YYYY-MM-DD

### Executive Summary
{2-3 sentences on strategy edge and validation outcome}

### Validation Results
| Gate | Value | Threshold | Status |
|------|-------|-----------|--------|
| OOS Sharpe | ... | ... | ... |
| PBO | ... | ... | ... |
| DSR | ... | ... | ... |
| Stress | ... | ... | ... |

### Audit Trail
- run_id: {uuid}
- git_commit: {hash}
- WFA folds: {n}
- Determinism: verified (3 runs)

### Recommendation
**APPROVED** for Hall of Fame promotion.

### Conditions (if any)
- {condition 1}
- {condition 2}

### Signatures
- [ ] Risk Analyst: ___________
- [ ] Trader Expert (execution): ___________
```

---

## Acceptance Criteria

### Strategy Validation

| Criterion | Pass | Fail |
|-----------|------|------|
| OOS Sharpe NET | >= tier threshold | < tier threshold |
| PBO | < tier threshold | > tier threshold |
| DSR | >= tier threshold | < tier threshold |
| Stress tests | >= 4/5 pass | < 3/5 pass |
| Degradation | < 50% | > 70% |
| Reproducibility | 3 identical runs | Any variation |
| Artifacts | All present | Any missing |

### Audit Quality

| Criterion | Pass | Fail |
|-----------|------|------|
| Seeds documented | Yes | No |
| Config snapshot | Present | Missing |
| Git commit | Recorded | Missing |
| Data integrity | Verified | Unverified |
| Costs modeled | Realistic | Ignored |

---

## Failure Modes

### Common Traps

1. **High Sharpe with few trades**
   - Symptom: Sharpe > 2 with < 50 trades
   - Fail: Statistical insignificance
   - Fix: Require min 30 OOS trades

2. **Overnight gaps ignored**
   - Symptom: Position strategy with no gap modeling
   - Fail: Real DD will exceed backtest
   - Fix: Run S5 stress, document gap handling

3. **Leakage through overlap**
   - Symptom: No purge/embargo in WFA
   - Fail: IS information bleeds to OOS
   - Fix: Enforce purge_days=5, embargo_days=5

4. **Costs ignored or underestimated**
   - Symptom: GROSS metrics only
   - Fail: NET performance may be negative
   - Fix: Require NET metrics for all gates

5. **Non-stationary strategy**
   - Symptom: Works only in specific regime
   - Fail: Fails when regime changes
   - Fix: Test across vol regimes, require multi-year data

6. **Concentrated bets**
   - Symptom: 80% of PnL from 1 asset or 1 month
   - Fail: Not diversified edge
   - Fix: Require spread of returns across assets/time

7. **Turnover kills in reality**
   - Symptom: 100x annual turnover
   - Fail: Costs eat all alpha
   - Fix: Check S1 (costs_2x), reject if Sharpe < 0.3

8. **IS presented as OOS**
   - Symptom: "OOS" period was actually used in development
   - Fail: Fake out-of-sample
   - Fix: Require run_id and config hash proving separation

9. **Low DSR despite high Sharpe**
   - Symptom: Sharpe 1.5, DSR 0.3
   - Fail: Selection bias explains performance
   - Fix: Flag PBO, require DSR >= 0.5

10. **PBO ignored**
    - Symptom: PBO = 0.35, strategy still promoted
    - Fail: 35% chance performance is luck
    - Fix: Hard gate: PBO < 0.20 for research, < 0.10 for production

### Red Flags Requiring Immediate Investigation

- Sharpe IS > 3x Sharpe OOS
- PBO > 0.30
- DSR < 0.3
- Zero losing months in backtest
- Turnover > 50x annual
- Single asset concentration > 50%

---

## Collaboration Hooks

### Handoff to `/trader-expert`

After validation passes, trader expert must verify:
- Slippage model is realistic for asset class
- Fill assumptions are achievable
- Market impact is accounted for

```markdown
## Handoff: risk-analyst → trader-expert

**Strategy ID:** {genome_id}
**Validation Status:** PASSED

**Requires execution review:**
- [ ] Slippage model appropriate for {market}
- [ ] Fill rate assumptions realistic
- [ ] Turnover ({value}x annual) executable
- [ ] Latency assumptions verified

**Files:**
- Validation report: {path}
- Trades CSV: {path}
```

### Handoff to `/data-engineer`

If data integrity fails:

```markdown
## Handoff: risk-analyst → data-engineer

**Issue:** Data integrity check failed

**Problem:**
- {description of data issue}

**Affected:**
- Strategy: {genome_id}
- Period: {start} to {end}
- Asset(s): {list}

**Required action:**
- [ ] Investigate data source
- [ ] Verify corporate actions
- [ ] Check for gaps/survivorship
```

### Handoff to `/quant-engineer`

If instrumentation needed:

```markdown
## Handoff: risk-analyst → quant-engineer

**Request:** Metric instrumentation

**Needed:**
- {specific metric or check}

**Purpose:**
- Enable validation of {use case}

**Priority:** {high/medium/low}
```

### Receiving from Researcher

When receiving validation request:

1. Verify all artifacts present (run_id, config, git commit)
2. Check strategy complexity (parameter count)
3. Identify trading modality (intraday vs position)
4. Select appropriate stress suite
5. Run validation pipeline
6. Generate report

---

## Quick Reference

### Validation Pipeline

```
1. Receive request with run_id
2. Verify artifacts exist
3. Load config and metrics
4. Run WFA/CPCV analysis
5. Calculate PBO/DSR
6. Execute stress suite
7. Check gates by tier
8. Generate report
9. Recommend: PROMOTE / REVISE / REJECT
```

### Key Thresholds (Production)

| Metric | Value |
|--------|-------|
| min_oos_sharpe | 1.0 |
| max_pbo | 0.10 |
| min_dsr | 0.8 |
| max_degradation | 50% |
| max_drawdown | 20% |
| min_profit_factor | 1.5 |
| min_stress_pass | 4/5 |

### Key Thresholds (Research)

| Metric | Value |
|--------|-------|
| min_oos_sharpe | 0.5 |
| max_pbo | 0.20 |
| min_dsr | 0.5 |
| max_degradation | 70% |
| max_drawdown | 35% |
| min_profit_factor | 1.1 |
| min_stress_pass | 3/5 |
