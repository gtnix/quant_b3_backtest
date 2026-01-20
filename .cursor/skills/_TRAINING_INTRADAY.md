# Training: Intraday Strategy Lifecycle

Step-by-step operational guide for developing and promoting an intraday strategy using the Quant Finance Team.

---

## Mandate Definition

Before starting, define:

| Parameter | Intraday Default | Your Value |
|-----------|------------------|------------|
| Timeframe | 5-min / 30-min bars | __________ |
| Universe | IBOV / custom | __________ |
| Holding Period | < 1 day (no overnight) | __________ |
| Base Slippage | 5-10 bps | __________ |
| Commission Tier | B3Retail / B3Prime | __________ |
| Max Turnover | < 50x annual | __________ |
| OOS Sharpe Target | >= 0.5 (research) / >= 1.0 (production) | __________ |

---

## Phase 0: Data Readiness

**Owner:** `/data-engineer`

### Prompt Snippet

```
/data-engineer
Validate data quality for IBOV intraday (5-min) from 2020-01-01 to 2024-12-31.
Check OHLCV invariants, gaps, corporate actions, and provide snapshot_id.
```

### Checklist

```
[ ] OHLC invariants: low <= open/close <= high
[ ] Volume >= 0, no negatives
[ ] No duplicate bars
[ ] Timestamps monotonic within each symbol
[ ] Gaps explained with GapReason (holiday vs missing)
[ ] Corporate actions in ledger (dividends, splits)
[ ] Adjusted/raw prices per policy (docs/policies/dividend-policy.md)
[ ] Data freshness < 24h for active period
[ ] snapshot_id documented: __________
```

### Artifacts Produced

| Artifact | Location |
|----------|----------|
| Data Quality Report | inline or `artifacts/data/dq_report_{snapshot_id}.md` |
| snapshot_id | UUID recorded |

### Go/No-Go

| Criterion | Status |
|-----------|--------|
| All BLOCKER gates pass | [ ] GO / [ ] NO-GO |
| snapshot_id documented | [ ] GO / [ ] NO-GO |

---

## Phase 1: Research

**Owner:** `/quant-researcher`

### Prompt Snippet

```
/quant-researcher
Design a momentum+mean-reversion intraday campaign for B3 5-min bars.
Universe: IBOV. Period: 2020-2024.
Max blocks: 6. Complexity budget: 50 DoF.
Cost assumptions: slippage=10bps, commission=B3Retail.
```

### Checklist

```
[ ] Mandate documented (hypothesis, universe, timeframe)
[ ] Search space defined (block types, parameter bounds)
[ ] Complexity budget set (<= 50 DoF recommended)
[ ] Evolution parameters configured (population, generations)
[ ] Diversity settings enabled (sigma_share, fitness_sharing)
[ ] Stagnation detection enabled
[ ] Modality declared: INTRADAY
[ ] Seeds documented: [42, 123, 456]
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
| >= 10 diverse candidates on Pareto front | [ ] GO / [ ] NO-GO |
| Handoff Packet ready for risk-analyst | [ ] GO / [ ] NO-GO |

---

## Phase 2: Validation

**Owner:** `/risk-analyst`

### Prompt Snippet

```
/risk-analyst
Validate run_id {run_id} for intraday strategy.
Modality: intraday. Market: BR. Turnover: ~20x annual.
Run WFA with 5 folds, calculate PBO/DSR, execute stress suite S1-S5.
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
    [ ] S1 (costs_2x): Sharpe = __________ (>= 0.3)
    [ ] S2 (delay+1): Sharpe = __________ (>= 0.5)
    [ ] S3 (spread_widen_vol): Sharpe = __________
    [ ] S4 (capacity_constraint): Fill rate = __________
    [ ] S5 (combined_adverse): Sharpe = __________
[ ] Stress pass rate: __/5 (threshold: >= 3/5)
[ ] Determinism verified (3 identical runs)
```

### Intraday-Specific Gates

```
[ ] S1 (costs_2x) Sharpe >= 0.3 - survives cost spikes
[ ] S2 (delay+1) Sharpe >= 0.5 - not latency-dependent
[ ] Turnover annual < 50x
[ ] Avg trade duration >= 5 bars (not noise trading)
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
| S1 + S2 passed | [ ] GO / [ ] NO-GO |
| Handoff to trader-expert if turnover > 12x | [ ] REQUIRED / [ ] SKIP |

---

## Phase 3: Execution Realism

**Owner:** `/trader-expert`

### Prompt Snippet

```
/trader-expert
Review execution assumptions for genome {genome_id} from run_id {run_id}.
Market: B3. Modality: intraday. Turnover: ~20x annual.
Current slippage assumption: 10bps. Verify capacity and session constraints.
```

### Checklist

```
[ ] Order type documented (Market/Limit)
[ ] Fill assumption documented (100% / Partial / Volume-constrained)
[ ] Delay bars documented (0 / 1 / 2)
[ ] Slippage model selected (Constant / VolumeLinear / VolatilityAdaptive)
[ ] Base slippage: __________ bps
[ ] Fee tier selected: B3Retail / B3Prime
[ ] 3 cost scenarios tested:
    [ ] Base: Sharpe = __________
    [ ] Pessimist (2x): Sharpe = __________
    [ ] Stress (3x): Sharpe = __________
[ ] max_participation set: __________%
[ ] capacity_proxy_usd: $__________
[ ] Session constraints documented (regular only / includes auction)
[ ] Timezone verified: America/Sao_Paulo
```

### Intraday-Specific Checks

```
[ ] No after-hours execution assumed
[ ] No closing auction dependency (or documented)
[ ] Session boundaries respected (10:00-17:55 B3)
[ ] Latency assumptions realistic for bar interval
```

### Limitations Declared

```
[ ] Settlement not modeled (T+1/T+2)
[ ] Auction mechanics not simulated
[ ] Extended hours not executable
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
| S1 + S2 passed with realistic costs | [ ] GO / [ ] NO-GO |
| Capacity >= $5M (or documented exception) | [ ] GO / [ ] NO-GO |

---

## Phase 4: Performance

**Owner:** `/quant-engineer`

### Prompt Snippet

```
/quant-engineer
Profile backtester performance for strategy genome {genome_id}.
Target: < 20μs per day per asset.
Check for allocations in hot path and benchmark against PERFORMANCE_CONTRACT.md.
```

### Checklist

```
[ ] Baseline benchmark captured: cargo bench --save-baseline before
[ ] Flamegraph generated: cargo flamegraph --bench unified_bench
[ ] Hot path allocations checked: 0 in process_day()
[ ] Performance targets met (per PERFORMANCE_CONTRACT.md):
    [ ] 10 assets, 252 days: < 10μs per day
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
Prepare deployment for updated combiner with new strategy support.
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
Add campaign {campaign_name} to mining queue.
Config: configs/campaigns/{campaign_name}.toml
Priority: 1. Market: BR. Repeat: false.
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
    - repeat: false
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
Prepare promotion packet for candidate {candidate_id} from run_id {run_id}.
Verify all provenance fields and pass to risk-analyst for final approval.
```

### Prompt Snippet (Risk Analyst)

```
/risk-analyst
Review promotion packet for candidate {candidate_id}.
Confirm all gates passed and issue Promotion Memo or rejection.
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
[ ] Artifacts present:
    - strategy.toml
    - metrics.obfs
    - trades.csv (if applicable)
[ ] Reviews:
    - [ ] Risk-analyst gate passed
    - [ ] Trader-expert execution reviewed (if turnover > 12x)
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
| Trader-expert reviewed (if required) | [ ] GO / [ ] NO-GO |
| Risk-analyst Promotion Memo issued | [ ] GO / [ ] NO-GO |

---

## Summary: Intraday Closed Pipeline

```
data-engineer (snapshot_id)
       |
       v
quant-researcher (run_id, config, seed, candidates)
       |
       v
risk-analyst (WFA/PBO/DSR + S1/S2 stress)
       |
       v [if turnover > 12x]
trader-expert (Execution Assumptions Card)
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

## Related Documents

- [_TEAM_INDEX.md](_TEAM_INDEX.md)
- [_TEAM_PLAYBOOK.md](_TEAM_PLAYBOOK.md)
- [_HANDOFF_CONTRACTS.md](_HANDOFF_CONTRACTS.md)
- [_RUNBOOK_PROMOTION.md](_RUNBOOK_PROMOTION.md)
- [_TRAINING_POSITION.md](_TRAINING_POSITION.md)
