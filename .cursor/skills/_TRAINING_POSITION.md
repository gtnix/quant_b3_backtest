# Training: Position Strategy Lifecycle

Step-by-step operational guide for developing and promoting a position (multi-day) strategy using the Quant Finance Team.

---

## Mandate Definition

Before starting, define:

| Parameter | Position Default | Your Value |
|-----------|------------------|------------|
| Timeframe | Daily bars | __________ |
| Universe | IBOV / SP500 / custom | __________ |
| Holding Period | Days to weeks | __________ |
| Rebalance Cadence | Daily / Weekly / Monthly | __________ |
| Base Slippage | 5-15 bps | __________ |
| Commission Tier | B3Retail / B3Prime / USRetail / USPrime | __________ |
| Max Turnover | < 12x annual | __________ |
| OOS Sharpe Target | >= 0.5 (research) / >= 1.0 (production) | __________ |
| Short Positions | Yes / No | __________ |

---

## Phase 0: Data Readiness

**Owner:** `/data-engineer`

### Prompt Snippet

```
/data-engineer
Validate data quality for IBOV daily bars from 2015-01-01 to 2024-12-31.
Check OHLCV invariants, overnight gaps, corporate actions (dividends, splits),
and point-in-time universe. Provide snapshot_id.
```

### Checklist

```
[ ] OHLC invariants: low <= open/close <= high
[ ] Volume >= 0, no negatives
[ ] No duplicate bars
[ ] Gaps explained with GapReason (holiday vs missing)
[ ] Corporate actions in ledger:
    [ ] Dividends with ex_date, rate, type
    [ ] Splits with ratio and direction
[ ] Adjusted vs raw prices per policy:
    - Signals use adjusted_close
    - Valuation uses raw_close
[ ] Universe is point-in-time (no survivorship bias)
[ ] Delisted stocks tracked (INACTIVE status)
[ ] Data freshness < 24h for active period
[ ] snapshot_id documented: __________
```

### Position-Specific Data Checks

```
[ ] Overnight gaps present in data (not intraday only)
[ ] Dividends properly credited on ex_date
[ ] Splits correctly applied to adjusted prices
[ ] Universe membership table populated with ref_date
[ ] Ticker status (ACTIVE/INACTIVE/SUSPECT) verified
```

### Artifacts Produced

| Artifact | Location |
|----------|----------|
| Data Quality Report | inline or `artifacts/data/dq_report_{snapshot_id}.md` |
| CA Ledger Snapshot | inline |
| Universe Snapshot Card | inline |
| snapshot_id | UUID recorded |

### Go/No-Go

| Criterion | Status |
|-----------|--------|
| All BLOCKER gates pass | [ ] GO / [ ] NO-GO |
| Corporate actions verified | [ ] GO / [ ] NO-GO |
| Point-in-time universe confirmed | [ ] GO / [ ] NO-GO |
| snapshot_id documented | [ ] GO / [ ] NO-GO |

---

## Phase 1: Research

**Owner:** `/quant-researcher`

### Prompt Snippet

```
/quant-researcher
Design a value+momentum position campaign for IBOV daily bars.
Universe: IBOV point-in-time. Period: 2015-2024.
Max blocks: 5. Complexity budget: 40 DoF.
Holding period: 5-20 days. Rebalance: weekly.
Cost assumptions: slippage=10bps, commission=B3Retail.
Include overnight gap risk in mandate.
```

### Checklist

```
[ ] Mandate documented (hypothesis, universe, timeframe, holding)
[ ] Overnight gap risk acknowledged in mandate
[ ] Rebalance cadence defined
[ ] Corporate actions handling documented:
    - Dividends: included in total return
    - Splits: using adjusted prices for signals
[ ] Search space defined (block types, parameter bounds)
[ ] Complexity budget set (<= 40 DoF recommended)
[ ] Evolution parameters configured
[ ] Diversity settings enabled
[ ] Modality declared: POSITION
[ ] Seeds documented: [42, 123, 456]
[ ] If shorts allowed, borrow costs documented (limitation if not modeled)
```

### Artifacts Produced

| Artifact | Location |
|----------|----------|
| Research Hypothesis Card | inline |
| Campaign Config | `configs/campaigns/{campaign_name}.toml` |
| run_id | `artifacts/runs/{run_id}/` |
| Pareto Front Summary | `artifacts/runs/{run_id}/pareto_summary.md` |
| Top K Genomes | `artifacts/runs/{run_id}/top_k_genomes.json` |
| Metrics JSON | `artifacts/runs/{run_id}/metrics.json` |

### Go/No-Go

| Criterion | Status |
|-----------|--------|
| run_id generated | [ ] GO / [ ] NO-GO |
| config.toml frozen | [ ] GO / [ ] NO-GO |
| Overnight gap handling documented | [ ] GO / [ ] NO-GO |
| >= 10 diverse candidates on Pareto front | [ ] GO / [ ] NO-GO |

---

## Phase 2: Validation

**Owner:** `/risk-analyst`

### Prompt Snippet

```
/risk-analyst
Validate run_id {run_id} for position strategy.
Modality: position. Market: BR. Rebalance: weekly. Turnover: ~6x annual.
Run WFA with 5 folds, calculate PBO/DSR, execute stress suite S1-S5.
Focus on S5 (combined_adverse) for position-specific gates.
```

### Checklist

```
[ ] Artifacts verified (run_id, config, seed, git_sha)
[ ] WFA completed with >= 5 folds
[ ] Purge/embargo applied (5 days each)
[ ] PBO calculated: __________ (threshold: < 0.20)
[ ] DSR calculated: __________ (threshold: >= 0.5)
[ ] OOS Sharpe NET: __________ (threshold: >= 0.5)
[ ] Max Drawdown: __________ (threshold: <= 35%)
[ ] Degradation IS->OOS: __________ (threshold: < 70%)
[ ] Stress tests executed:
    [ ] S1 (costs_2x): Sharpe = __________
    [ ] S2 (delay+1): Sharpe = __________
    [ ] S3 (spread_widen_vol): Sharpe = __________
    [ ] S4 (capacity_constraint): Fill rate = __________
    [ ] S5 (combined_adverse): Sharpe = __________ (>= 0.0), DD = __________ (<= 30%)
[ ] Stress pass rate: __/5 (threshold: >= 3/5)
[ ] Determinism verified (3 identical runs)
```

### Position-Specific Gates

```
[ ] S5 (combined_adverse) Sharpe >= 0.0
[ ] S5 Max Drawdown <= 30%
[ ] Overnight exposure documented
[ ] Corporate actions handled correctly
[ ] Turnover annual < 12x (or justified)
```

### Artifacts Produced

| Artifact | Location |
|----------|----------|
| Validation Report | inline or `artifacts/runs/{run_id}/validation_report.md` |
| Fold Stability Table | part of report |
| Overfitting Checklist | part of report |

### Go/No-Go

| Criterion | Status |
|-----------|--------|
| OOS Sharpe >= threshold | [ ] GO / [ ] NO-GO |
| PBO < threshold | [ ] GO / [ ] NO-GO |
| S5 passed (Sharpe >= 0, DD <= 30%) | [ ] GO / [ ] NO-GO |
| Handoff to trader-expert if turnover > 12x | [ ] REQUIRED / [ ] SKIP |

---

## Phase 3: Execution Realism

**Owner:** `/trader-expert`

### Prompt Snippet

```
/trader-expert
Review execution assumptions for position genome {genome_id} from run_id {run_id}.
Market: B3. Modality: position. Rebalance: weekly. Turnover: ~6x annual.
Verify rebalance timing, capacity, and overnight gap handling.
If shorts allowed, verify borrow cost assumptions.
```

### Checklist

```
[ ] Order type documented (Market/Limit)
[ ] Rebalance timing documented (open / close / VWAP window)
[ ] Delay bars documented (0 / 1)
[ ] Slippage model selected (Constant / VolumeLinear)
[ ] Base slippage: __________ bps
[ ] Fee tier selected: B3Retail / B3Prime / USRetail / USPrime
[ ] 3 cost scenarios tested:
    [ ] Base: Sharpe = __________
    [ ] Pessimist (2x): Sharpe = __________
    [ ] Stress (3x): Sharpe = __________
[ ] max_participation set: __________%
[ ] capacity_proxy_usd: $__________
[ ] Rebalance window documented: __________
```

### Position-Specific Checks

```
[ ] Overnight gap risk documented (or excluded)
[ ] Rebalance execution window realistic
[ ] Settlement timing noted (T+1/T+2 limitation)
[ ] If shorts:
    [ ] Borrow cost assumption: __________ bps annual
    [ ] Borrow availability documented (or limitation declared)
```

### Limitations Declared

```
[ ] Settlement not modeled (T+1/T+2)
[ ] Borrow costs not fully modeled (if applicable)
[ ] Overnight gaps modeled via daily data (not intraday)
[ ] Corporate actions handled per policy
[ ] Other: __________
```

### Artifacts Produced

| Artifact | Location |
|----------|----------|
| Execution Assumptions Card | inline or `artifacts/runs/{run_id}/execution_card.md` |
| Cost Report | optional |

### Go/No-Go

| Criterion | Status |
|-----------|--------|
| Assumptions Card complete and signed | [ ] GO / [ ] NO-GO |
| S5 passed with realistic costs | [ ] GO / [ ] NO-GO |
| Capacity >= $5M (or documented exception) | [ ] GO / [ ] NO-GO |
| Borrow limitations declared (if shorts) | [ ] GO / [ ] NO-GO |

---

## Phase 4: Performance

**Owner:** `/quant-engineer`

### Prompt Snippet

```
/quant-engineer
Profile backtester performance for position strategy genome {genome_id}.
Daily rebalance with 30+ assets.
Check for allocations in hot path and benchmark against PERFORMANCE_CONTRACT.md.
```

### Checklist

```
[ ] Baseline benchmark captured: cargo bench --save-baseline before
[ ] Flamegraph generated: cargo flamegraph --bench unified_bench
[ ] Hot path allocations checked: 0 in process_day()
[ ] Performance targets met (per PERFORMANCE_CONTRACT.md):
    [ ] 50 assets, 252 days: < 50μs per day (linear scaling)
    [ ] Regression vs baseline: <= 5%
[ ] Golden tests pass after any changes
[ ] Determinism verified (3 identical runs)
```

### Artifacts Produced

| Artifact | Location |
|----------|----------|
| Benchmark Report | inline |
| Flamegraph | `artifacts/runs/{run_id}/flamegraph.svg` |

### Go/No-Go

| Criterion | Status |
|-----------|--------|
| No regression > 5% | [ ] GO / [ ] NO-GO |
| Zero hot path allocations | [ ] GO / [ ] NO-GO |

---

## Phase 5: Deploy

**Owner:** `/devops-infra`

### Prompt Snippet

```
/devops-infra
Prepare deployment for updated combiner with position strategy support.
Run full deploy cycle with verification and document rollback plan.
```

### Checklist (execute via scripts/deploy.sh)

```
[ ] Build successful: ./scripts/deploy.sh build
[ ] Pre-deploy health check: ./scripts/vps/health-check.sh
[ ] Deploy executed: ./scripts/deploy.sh deploy
[ ] Post-deploy verification: ./scripts/deploy.sh verify
[ ] PM2 services online: pm2 list
[ ] Logs flowing: pm2 logs --lines 20
[ ] Rollback tested: ./scripts/deploy.sh rollback (then re-deploy)
[ ] Config frozen with git sha: __________
```

### Artifacts Produced

| Artifact | Location |
|----------|----------|
| Deployment Checklist | inline |
| git_sha | recorded |

### Go/No-Go

| Criterion | Status |
|-----------|--------|
| Health check PASS | [ ] GO / [ ] NO-GO |
| Rollback plan verified | [ ] GO / [ ] NO-GO |

---

## Phase 6: Mining

**Owner:** `/omp-operator`

### Prompt Snippet

```
/omp-operator
Add position campaign {campaign_name} to mining queue.
Config: configs/campaigns/{campaign_name}.toml
Priority: 2. Market: BR. Repeat: true (continuous research).
Verify resource budget and watchdog policy.
```

### Checklist

```
[ ] Campaign config exists and validates
[ ] Queue entry created in dashboard/campaign_queue.json:
    - id: camp_________
    - config_path: __________
    - market: BR
    - priority: __
    - enabled: true
    - repeat: true (for ongoing research)
[ ] Resource budget acceptable:
    [ ] CPU < 85%
    [ ] RAM > 400MB free
    [ ] Disk > 1GB free
[ ] Watchdog policy active (check dashboard/omp_config.toml)
[ ] Variance sanity gate enabled
[ ] Promotion thresholds configured:
    - min_oos_sharpe_net: 0.5
    - max_pbo: 0.20
    - min_dsr: 0.4
    - max_drawdown_net: 0.30
```

### Monitoring Commands

```bash
# Check OMP status
curl -s http://localhost:3001/api/omp/status | jq

# List queue
curl -s http://localhost:3001/api/omp/queue | jq

# Health check
./scripts/vps/health-check.sh
```

### Artifacts Produced

| Artifact | Location |
|----------|----------|
| Campaign Spec Card | inline |
| Queue entry | `dashboard/campaign_queue.json` |
| Mining Ops Log | daily |

### Go/No-Go

| Criterion | Status |
|-----------|--------|
| Queue entry valid | [ ] GO / [ ] NO-GO |
| Resources within budget | [ ] GO / [ ] NO-GO |
| Watchdog active | [ ] GO / [ ] NO-GO |

---

## Phase 7: Hall of Fame Promotion

**Owners:** `/omp-operator` + `/risk-analyst`

### Prompt Snippet (OMP Operator)

```
/omp-operator
Prepare promotion packet for position candidate {candidate_id} from run_id {run_id}.
Verify overnight gap handling documented and pass to risk-analyst for final approval.
```

### Prompt Snippet (Risk Analyst)

```
/risk-analyst
Review promotion packet for position candidate {candidate_id}.
Verify S5 stress passed and corporate actions handled correctly.
Issue Promotion Memo or rejection.
```

### Promotion Packet Checklist

```
[ ] Provenance complete:
    - candidate_id: __________
    - genome_hash: __________
    - run_id: __________
    - campaign_id: __________
    - config_hash: __________
    - git_sha: __________
    - snapshot_id: __________
[ ] Validation gates:
    - OOS Sharpe >= 0.5: __________
    - PBO <= 0.20: __________
    - DSR >= 0.4: __________
    - Max DD <= 30%: __________
    - Variance sanity: PASS
    - Stress tests: __/5
[ ] Position-specific:
    - [ ] S5 (combined_adverse) passed
    - [ ] Overnight exposure documented
    - [ ] Corporate actions handled per policy
    - [ ] Borrow costs noted (if shorts)
[ ] Artifacts present:
    - strategy.toml
    - metrics.obfs
    - trades.csv (if applicable)
[ ] Reviews:
    - [ ] Risk-analyst gate passed
    - [ ] Trader-expert execution reviewed (if turnover > 12x or shorts)
    - [ ] Data snapshot documented
[ ] Approval:
    - [ ] Ready for Hall of Fame promotion
```

### Artifacts Produced

| Artifact | Location |
|----------|----------|
| Promotion Packet | inline |
| Promotion Memo | from risk-analyst |
| HoF Entry | `artifacts/hall_of_fame/{candidate_id}/` |

### Go/No-Go

| Criterion | Status |
|-----------|--------|
| All provenance fields present | [ ] GO / [ ] NO-GO |
| All validation gates passed | [ ] GO / [ ] NO-GO |
| S5 stress passed | [ ] GO / [ ] NO-GO |
| Trader-expert reviewed (if required) | [ ] GO / [ ] NO-GO |
| Risk-analyst Promotion Memo issued | [ ] GO / [ ] NO-GO |

---

## Summary: Position Closed Pipeline

```
data-engineer (snapshot_id, CA ledger, PIT universe)
       |
       v
quant-researcher (run_id, config, seed, candidates + gap handling)
       |
       v
risk-analyst (WFA/PBO/DSR + S5 stress)
       |
       v [if turnover > 12x or shorts]
trader-expert (Execution Assumptions Card + borrow)
       |
       v
risk-analyst (final validation)
       |
       v
quant-engineer (performance verification) [optional]
       |
       v
devops-infra (deploy + health check)
       |
       v
omp-operator (queue + mining + promotion packet)
       |
       v
risk-analyst (Promotion Memo)
       |
       v
Hall of Fame
```

**No bypass allowed at any gate.**

---

## Position vs Intraday Key Differences

| Aspect | Intraday | Position |
|--------|----------|----------|
| Holding Period | < 1 day | Days to weeks |
| Overnight Risk | None (flat EOD) | Gap exposure |
| Critical Stress | S1 (costs_2x), S2 (delay+1) | S5 (combined_adverse) |
| Corporate Actions | Less relevant | Critical (dividends, splits) |
| Universe | Current | Point-in-time (survivorship) |
| Borrow Costs | Rare | Common if shorts |
| Rebalance | Continuous | Scheduled |
| Slippage Focus | Microstructure, latency | Market impact, capacity |

---

## Related Documents

- [_TEAM_INDEX.md](_TEAM_INDEX.md)
- [_TEAM_PLAYBOOK.md](_TEAM_PLAYBOOK.md)
- [_HANDOFF_CONTRACTS.md](_HANDOFF_CONTRACTS.md)
- [_RUNBOOK_PROMOTION.md](_RUNBOOK_PROMOTION.md)
- [_TRAINING_INTRADAY.md](_TRAINING_INTRADAY.md)
