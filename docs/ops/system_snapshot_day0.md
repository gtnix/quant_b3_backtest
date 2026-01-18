# System Snapshot Report - Day 0

**Generated**: 2026-01-18  
**Version**: 1.0.0  
**Status**: Diagnostic (Read-Only Discovery)

---

## A) Executive Summary

1. **System Purpose**: Institutional-grade quantitative strategy backtesting and evolutionary discovery platform for B3 (Brazil) and US equity markets.

2. **Core Architecture**: Rust-based engine with 17 crates organized into backtester (engine, core, execution, portfolio, strategy, validation, intelligence), combiner/SCG (core, engine, runner, cli), and market_data layers.

3. **Performance Profile**: Zero-allocation hot path in `process_day()`, SIMD vectorization, fixed-point arithmetic (Price/Money/Rate), targets >= 100K events/sec throughput.

4. **Evolutionary Search**: NSGA-II multiobjective optimization with Pareto ranking, fitness sharing, diversity preservation, and stagnation detection/restart.

5. **Validation Rigor**: Walk-Forward Analysis (WFA/CPCV), PBO/DSR anti-overfitting metrics, 5-scenario stress suite (S1-S5), institutional gates.

6. **Perpetual Mining**: OMP daemon runs 24/7 locally on Ubuntu workstation, orchestrates campaigns, auto-promotes to Hall of Fame.

7. **Data Infrastructure**: Neon PostgreSQL for persistence, Python DataHubs (B3/US/FX) for ingestion, point-in-time universe, trading calendars.

8. **Current Strengths**:
   - Comprehensive skill-based team documentation with handoff contracts
   - Deterministic simulation with run_id/seed/config_hash/git_sha provenance
   - Automated HoF promotion with variance sanity gates
   - Health check scripts and process auto-restart

9. **Key Fragilities**:
   - Disk growth from run artifacts requires proactive cleanup
   - No automated alerting/notifications implemented
   - Settlement (T+1/T+2) and borrow costs not modeled

10. **Top 5 Efficiency Opportunities (No Core Change)**:
    - **Config Tuning**: Reduce `population_size` or `max_generations` for faster iteration
    - **Retention Policy**: Implement aggressive artifact cleanup (keep 3-5 runs vs accumulating)
    - **Log Rotation**: PM2 logs need rotation policy to prevent disk fill
    - **Batch I/O**: Consolidate database writes during HoF sync
    - **Instrumentation**: Add SSE metrics for real-time resource monitoring

---

## B) Repo Anchors Map

| Subsystem | Key Paths | Config Files | Outputs/Artifacts | Notes |
|-----------|-----------|--------------|-------------------|-------|
| **Engine Core** | `crates/backtester_engine/src/unified.rs` | - | - | `process_day()` L602-687 hot path |
| **Fixed-Point** | `crates/backtester_core/src/fixed.rs` | - | - | Price/Money/Rate 6-8 decimal precision |
| **SIMD** | `crates/backtester_core/src/simd.rs`, `crates/combiner_core/src/simd_metrics.rs` | - | - | AVX2/AVX-512 vectorization |
| **Execution** | `crates/backtester_execution/src/` | `configs/execution_institutional.toml` | - | Slippage models, stress tests S1-S5 |
| **Strategy Blocks** | `crates/backtester_strategy/src/blocks/` | `configs/parameter_bounds/*.toml` | - | Selection/Entry/Exit/Sizing blocks |
| **SCG Engine** | `crates/combiner_engine/src/engine.rs` | `configs/campaigns/*.toml` | `output/scg/run_*/` | EvolutionEngine main loop |
| **Pareto/Diversity** | `crates/combiner_engine/src/pareto_unified.rs`, `diversity.rs` | - | - | NSGA-II, fitness sharing |
| **Validation** | `crates/combiner_engine/src/validation.rs`, `institutional_thresholds.rs` | - | - | PBO/DSR, WFA gates |
| **Walk-Forward** | `crates/backtester_intelligence/src/walkforward/` | `configs/training_strategies/*.toml` | - | WFA/CPCV engine |
| **Data Health** | `crates/backtester_intelligence/src/monitoring/data_health.rs` | - | - | 8 core data quality checks |
| **Market Data** | `crates/market_data/src/` | - | `.cache/market_data/*.obfs` | Brapi, calendar, universe_gate |
| **Calendar** | `crates/market_data/src/calendar/` | - | - | B3/NYSE sessions, holidays, gaps |
| **DataHub B3** | `datahub_b3/` | - | Neon tables | Python CLI for B3 ingestion |
| **DataHub US** | `datahub_us/` | - | Neon tables | Python CLI for US ingestion |
| **DataHub FX** | `datahub_fx/` | - | Neon tables | BCB/FRED FX rates |
| **OMP Daemon** | `dashboard/server/routes/omp.js`, `state.js` | `dashboard/omp_config.toml` | `artifacts/hall_of_fame/` | Perpetual mining orchestrator |
| **Campaign Queue** | - | `dashboard/campaign_queue.json` | - | Priority queue for campaigns |
| **HoF Sync** | `dashboard/server/services/hofSync.js` | - | Neon `scg_hall_of_fame` | Local-to-DB sync |
| **Dashboard** | `dashboard/src/pages/*.tsx` | - | - | React/Vite frontend |
| **Deploy** | `scripts/deploy.sh` | - | - | Build/deploy (DEFERRED: VPS in `scripts/vps/`) |
| **Health Check** | `scripts/local-health-check.sh` (planned) | - | - | Local system health |
| **CI/CD** | `.github/workflows/ci.yml` | - | - | Tests, benchmarks |
| **Benchmarks** | `benches/`, `crates/*/benches/` | `benches/PERFORMANCE_CONTRACT.md` | `benches/results/` | Criterion.rs benchmarks |
| **Benchmarks** | `benches/`, `crates/*/benches/` | `benches/PERFORMANCE_CONTRACT.md` | `benches/results/` | Criterion.rs benchmarks |
| **Audits** | `crates/combiner_cli/artifacts/audits/` | - | `audit_*/` | Marco 0-5 audit artifacts |
| **Data Integrity** | - | - | `artifacts/data_integrity/camp_*/` | Per-campaign integrity reports |

---

## C) Current Workflow Map

### Phase Diagram (ASCII)

```
Phase 0        Phase 1       Phase 2         Phase 3            Phase 4        Phase 5      Phase 6        Phase 7
   |              |             |               |                  |              |            |              |
   v              v             v               v                  v              v            v              v
+----------+  +----------+  +----------+  +------------------+  +----------+  +----------+  +----------+  +----------+
|   Data   |  | Research |  |Validation|  |Execution Realism |  |Performnce|  |  Deploy  |  |  Mining  |  |Hall Fame |
| Readiness|->|          |->|          |->|                  |->|          |->|          |->|   24/7   |->| Promotion|
+----------+  +----------+  +----------+  +------------------+  +----------+  +----------+  +----------+  +----------+
     |              |             |               |                  |              |            |              |
data-engineer quant-researcher risk-analyst  trader-expert    quant-engineer devops-infra omp-operator  risk-analyst
                                                                                                        omp-operator
```

### Phase Details

| Phase | Name | Governing Skill | Artifacts Generated | Gates Today |
|-------|------|-----------------|---------------------|-------------|
| 0 | Data Readiness | data-engineer | snapshot_id, Data Quality Report | OHLCV invariants, coverage, freshness, CA ledger |
| 1 | Research | quant-researcher | run_id, config.toml, Pareto Front Summary, top_k_genomes.json | Diversity > critical_threshold, DoF budget |
| 2 | Validation | risk-analyst | Validation Report, PBO/DSR metrics | OOS Sharpe >= 0.5, PBO <= 0.20, DSR >= 0.4 |
| 3 | Execution Realism | trader-expert | Execution Assumptions Card, Cost Report | 3 cost scenarios, S1/S2 (intraday), S5 (position) |
| 4 | Performance | quant-engineer | Benchmark Report, Flamegraph | Regression <= 5%, hot path allocs = 0 |
| 5 | Deploy | devops-infra | Deployment Checklist | health-check.sh PASS, PM2 online |
| 6 | Mining 24/7 | omp-operator | Campaign logs, SSE events | CPU < 85%, RAM > 400MB, Disk > 1GB |
| 7 | HoF Promotion | omp-operator + risk-analyst | Promotion Packet, HoF entry | Variance gate, OOS Sharpe >= 0.5, PBO <= 0.20 |

### Current Gate Status

| Gate | Implemented | Location | Automated |
|------|-------------|----------|-----------|
| Data Quality (OHLCV invariants) | Yes | `market_data/src/audit_integrity.rs` | Yes |
| Diversity threshold | Yes | `combiner_engine/src/diversity.rs` | Yes |
| PBO/DSR calculation | Yes | `combiner_engine/src/validation.rs` | Yes |
| Stress suite S1-S5 | Yes | `backtester_execution/src/stress.rs` | Yes |
| Variance sanity gate | Yes | `/api/omp/promote-check` | Yes |
| HoF promotion criteria | Yes | `omp_config.toml` | Yes |
| Performance regression | Partial | CI benchmark job | On PR only |
| Health check | Yes | `health-check.sh` | Manual/cron |

---

## D) Observed/Inferable Constraints

### Infrastructure (Local Ubuntu - MEASURE AT RUNTIME)

| Resource | Value | Source |
|----------|-------|--------|
| Environment | Local Ubuntu Workstation | See `docs/ops/local_only_policy.md` |
| CPU | UNKNOWN (run `lscpu`) | Measure at runtime |
| RAM | UNKNOWN (run `free -h`) | Measure at runtime |
| Disk | UNKNOWN (run `df -h`) | Measure at runtime |
| Filesystem | UNKNOWN (run `mount`) | Measure at runtime |

> **Note**: VPS infrastructure (Vultr, PM2, nginx) is DEFERRED. See `docs/ops/local_only_policy.md`.

### OMP Budgets (from omp_config.toml)

| Limit | Value | Effect |
|-------|-------|--------|
| max_cpu_util_pct | 85% | Block new campaigns |
| min_mem_available_mb | 400MB | Block new campaigns |
| min_disk_free_gb | 1.0GB | Auto-stop mining |
| max_concurrent_campaigns | 1 | Queue additional |
| loop_interval_secs | 30 | Check frequency |
| max_runtime_seconds | 900 (15min) | Default campaign timeout |

### Model Limitations (documented)

| Limitation | Impact | Source |
|------------|--------|--------|
| Auctions not modeled | Opening/closing prices may differ | trader-expert SKILL.md |
| Settlement T+1/T+2 not modeled | Cash availability assumed instant | trader-expert SKILL.md |
| Borrow costs not fully modeled | Short positions underestimate costs | trader-expert SKILL.md |
| Circuit breakers/halts not modeled | Missing market stress events | trader-expert SKILL.md |
| Extended hours not executable | Sessions defined but limited support | trader-expert SKILL.md |

---

## E) Efficiency Heatmap

| Area | Suspected Hotspots | Evidence Available | Measurement Plan | Risk if Ignored |
|------|-------------------|-------------------|------------------|-----------------|
| **CPU** | `process_day()` loop, Pareto ranking | Yes: `benches/unified_bench.rs` | `cargo flamegraph --bench unified_bench` | Slow campaigns, missed opportunities |
| **CPU** | SIMD threshold (N < 8 falls to scalar) | Partial: code inspection | Profile with varying asset counts | Suboptimal for small universes |
| **RAM** | Population storage in SCG | No direct profiling | `dhat` or `heaptrack` during campaign | OOM, campaign crashes |
| **RAM** | Process memory growth | Monitor via `htop` | Check during campaigns | Memory pressure |
| **Disk** | Run artifacts `output/scg/run_*/` | Yes: cleanup script exists | `du -sh output/scg/` | Disk full, auto-stop |
| **Disk** | Local logs | No rotation configured | `du -sh` on log directories | Disk fill |
| **Disk** | `.cache/market_data/*.obfs` | Yes: files exist | Size audit | Stale cache accumulation |
| **DB** | HoF sync batch size | No: single-row inserts | Add batch insert timing | Slow promotion under load |
| **DB** | Connection pool exhaustion | No evidence | Monitor `pg_stat_activity` | API timeouts |
| **Network** | Brapi/yfinance API rate limits | Documented in data-engineer | Log 429 responses | Incomplete ingestion |

---

## F) Artifact and Disk Growth Model

### Artifacts Generated Per Run/Campaign

| Artifact Type | Location | Size Estimate | Growth Rate | Deletable |
|---------------|----------|---------------|-------------|-----------|
| Run output (full) | `output/scg/run_{id}/` | 10-100MB | Per campaign | Yes (after analysis) |
| Generations JSON | `output/scg/run_{id}/generations/` | 1-10MB | Per generation | Yes |
| Top candidates | `output/scg/run_{id}/top_k.json` | 100KB-1MB | Per run | Yes |
| Hall of Fame local | `output/scg/run_{id}/hall_of_fame/` | 1-5MB | Per elite | Yes (if synced to DB) |
| Hall of Fame promoted | `artifacts/hall_of_fame/` | 1-5MB per strategy | Permanent | **No** |
| Data integrity reports | `artifacts/data_integrity/camp_*/` | 10KB-100KB | Per campaign | Yes (after 30 days) |
| Audit artifacts | `crates/combiner_cli/artifacts/audits/` | 50KB-200KB | Per audit | Yes (after review) |
| Local logs | `dashboard/logs/` (if configured) | 1-10MB/day | Continuous | Yes (rotate) |
| Criterion baselines | `target/criterion/` | 10-50MB | Per benchmark | Yes (keep latest) |
| Market data cache | `.cache/market_data/` | 10-100MB | Per sync | Yes (refresh) |

### Estimated Disk Pressure

| Scenario | Daily Growth | Weekly Growth | Mitigation |
|----------|--------------|---------------|------------|
| 1 campaign/day | 50-100MB | 350-700MB | cleanup_old_runs.sh (keep 5) |
| 5 campaigns/day | 250-500MB | 1.75-3.5GB | Aggressive cleanup (keep 3) |
| Logs unmanaged | 10MB/day | 70MB/week | Add logrotate |
| With retention | Net ~0 (steady state) | ~0 | All policies active |

### Duplications Observed

| Duplication | Locations | Action |
|-------------|-----------|--------|
| HoF artifacts | `output/scg/run_*/hall_of_fame/` + `artifacts/hall_of_fame/` | Delete local after sync |
| Campaign configs | Embedded in run output + original in `configs/campaigns/` | Reference only, don't duplicate |
| Benchmark baselines | Multiple in `target/criterion/` | Keep only `main` baseline |

---

## G) Operational Risk Register (Top 10)

| ID | Description | Impact | Probability | Detection | Mitigation (No Core Change) |
|----|-------------|--------|-------------|-----------|----------------------------|
| R1 | Disk full stops mining | High | Medium | `health-check.sh` disk % | Cron job for `cleanup_old_runs.sh` hourly |
| R2 | OOM kills campaign | High | Low | Process killed (dmesg) | Reduce population_size, monitor with htop |
| R3 | PM2 log disk fill | Medium | High | Manual inspection | Configure PM2 log rotation (--log-date-format) |
| R4 | Stuck run blocks queue | High | Low | Watchdog timeout | Implement campaign timeout enforcement |
| R5 | HoF sync failure loses promotions | High | Low | API error logs | Add retry logic, local backup |
| R6 | Config drift between runs | Medium | Low | No detection | Git SHA check, config_hash in run metadata |
| R7 | Data freshness degradation | Medium | Medium | monitoring.yml workflow | Add freshness alerts to health-check |
| R8 | Determinism divergence undetected | High | Low | Golden tests in CI | Add determinism check to campaign finish |
| R9 | Neon connection pool exhaustion | Medium | Low | API timeout errors | Configure pool limits in dashboard/db.js |
| R10 | Secrets exposed in logs | High | Low | Code review | Audit log output, mask DATABASE_URL |

---

## H) Quality and Validity Posture (Per Skill)

### quant-engineer

**Strengths**:
1. PERFORMANCE_CONTRACT.md defines clear gates (100K events/sec, 0 hot path allocs, <= 5% regression)
2. Benchmark suite exists with Criterion.rs (`unified_bench.rs`, `engine_bench.rs`)
3. CI includes performance regression check on PRs

**Risks/Gaps**:
1. No continuous profiling in production - flamegraphs are manual (UNKNOWN: production latency distribution)
2. SIMD fallback thresholds not documented per-function
3. `dhat` allocation checking not integrated into CI

**Recommended Actions (Read-Only)**:
1. Add flamegraph generation to nightly CI for trend tracking
2. Document SIMD threshold constants in code comments
3. Create dhat-based allocation test for `process_day()`

---

### quant-researcher

**Strengths**:
1. Block DSL with documented catalog (`docs/strategies/block-catalog.md`)
2. Diversity monitoring with critical_threshold restart mechanism
3. Multiobjective optimization (Sharpe/CAGR/MaxDD) prevents single-metric collapse

**Risks/Gaps**:
1. DoF budget tracking mentioned but enforcement unclear (UNKNOWN: per-campaign DoF tracking)
2. No phenotypic distance filter for HoF insertion documented
3. Champion's curse mitigation (validate top 5-10) is guidance only, not enforced

**Recommended Actions (Read-Only)**:
1. Add DoF count to campaign output metrics
2. Implement genome hash similarity check before HoF insert
3. Configure `top_k` to 10+ in all production campaigns

---

### risk-analyst

**Strengths**:
1. PBO/DSR metrics implemented in `combiner_engine/src/validation.rs`
2. Institutional thresholds defined by tier (Production/Research)
3. Stress suite S1-S5 with clear pass criteria

**Risks/Gaps**:
1. CPCV computationally expensive - usage frequency unclear (UNKNOWN: when CPCV vs WFA)
2. Degradation IS/OOS < 50% gate exists but enforcement in OMP unclear
3. Trader-expert handoff for turnover > 12x is manual process

**Recommended Actions (Read-Only)**:
1. Document CPCV vs WFA decision tree in validation-framework.md
2. Add degradation check to OMP promotion criteria
3. Add turnover flag to candidate metadata for auto-handoff

---

### data-engineer

**Strengths**:
1. 8 core data health checks in `data_health.rs`
2. Point-in-time universe with `universe_gate.rs` (ACTIVE/INACTIVE/SUSPECT)
3. Trading calendar with GapReason deterministic classification

**Risks/Gaps**:
1. Corporate actions coverage for US market unclear (UNKNOWN: yfinance dividend quality)
2. Data freshness monitoring in CI but no automated alerting
3. Snapshot_id linkage to runs is documented but not enforced in OMP

**Recommended Actions (Read-Only)**:
1. Audit US dividend coverage vs B3
2. Add Slack/email notification to monitoring.yml on failure
3. Add snapshot_id to campaign config validation

---

### trader-expert

**Strengths**:
1. Multiple slippage models (Constant, VolumeLinear, VolatilityAdaptive)
2. Fee tiers documented for B3 and US (Retail/Prime)
3. Execution Assumptions Card template with 3-scenario requirement

**Risks/Gaps**:
1. Capacity proxy methodology documented but not automated
2. Limit order fill assumptions flagged but no fill rate simulation exists
3. Borrow costs explicitly marked as "not modeled" limitation

**Recommended Actions (Read-Only)**:
1. Add capacity_proxy_usd to campaign output metrics
2. Document fill rate assumption in all campaign configs
3. Add borrow cost placeholder field for future implementation

---

### devops-infra

**Strengths**:
1. deploy.sh with build/deploy/verify/rollback commands
2. PM2 ecosystem with autorestart and memory limits
3. Health check script with JSON output for automation

**Risks/Gaps**:
1. No log rotation configured (disk risk)
2. No automated alerting on health check failure
3. Binary rollback exists but database rollback plan undocumented

**Recommended Actions (Read-Only)**:
1. Add PM2 log rotation via ecosystem.config.cjs `log_date_format`
2. Integrate health-check.sh with cron + notification
3. Document Neon point-in-time recovery procedure

---

### omp-operator

**Strengths**:
1. Variance sanity gate blocks collapsed metrics
2. Resource budget enforcement (CPU/RAM/Disk limits)
3. Activity log with SSE real-time broadcast

**Risks/Gaps**:
1. Notifications disabled (`notifications.enabled = false`)
2. Repeat mode campaigns may accumulate if not completing
3. Cleanup script exists but not integrated with OMP lifecycle

**Recommended Actions (Read-Only)**:
1. Enable webhook notifications for HoF promotions
2. Add max_repeat_failures limit to queue governance
3. Call cleanup_old_runs.sh before campaign start if disk < 2GB

---

## I) Next Prompt Payload

```text
SYSTEM_SNAPSHOT_PAYLOAD
- stack_summary: Rust backtester (17 crates) + SCG evolutionary search + React dashboard + Neon PostgreSQL + Local Ubuntu
- key_anchors:
  - engine: crates/backtester_engine/src/unified.rs (process_day hot path)
  - scg: crates/combiner_engine/src/engine.rs (EvolutionEngine)
  - validation: crates/combiner_engine/src/validation.rs (PBO/DSR)
  - execution: crates/backtester_execution/src/stress.rs (S1-S5)
  - omp: dashboard/server/routes/omp.js, dashboard/omp_config.toml
  - data: crates/market_data/src/, datahub_b3/, datahub_us/
- outputs_locations:
  - runs: output/scg/run_{id}/
  - hof: artifacts/hall_of_fame/
  - logs: /opt/alpha-forge/logs/
  - integrity: artifacts/data_integrity/
  - cache: .cache/market_data/
- current_gates:
  - data: OHLCV invariants, freshness, coverage
  - research: diversity > 0.25, DoF budget
  - validation: OOS Sharpe >= 0.5, PBO <= 0.20, DSR >= 0.4, stress >= 4/5
  - execution: 3 cost scenarios, S1/S2 (intraday), S5 (position)
  - performance: regression <= 5%, hot path allocs = 0
  - omp: CPU < 85%, RAM > 400MB, Disk > 1GB, variance sanity
- top_hotspots:
  - CPU: process_day(), Pareto ranking
  - RAM: SCG population storage, PM2 growth
  - Disk: run artifacts, PM2 logs (no rotation)
  - DB: single-row HoF inserts, no pool monitoring
- top_disk_drivers:
  - output/scg/run_*/ (10-100MB per campaign)
  - PM2 logs (10MB/day, no rotation)
  - target/criterion/ (stale baselines)
- top_risks:
  - R1: Disk full stops mining (cron cleanup mitigates)
  - R2: OOM kills campaign (reduce pop_size, monitor with htop)
  - R3: Logs fill disk (add rotation)
  - R4: Stuck run blocks queue (timeout watchdog needed)
  - R5: HoF sync failure loses data (retry + backup)
- quick_wins_no_core_change:
  - Enable cron for cleanup_old_runs.sh (hourly)
  - Add PM2 log rotation in ecosystem.config.cjs
  - Enable notifications.enabled in omp_config.toml
  - Add snapshot_id to campaign config validation
  - Reduce keep_runs from 5 to 3 for tighter disk
- unknowns_to_resolve:
  - UNKNOWN: Production latency distribution (no continuous profiling)
  - UNKNOWN: Per-campaign DoF tracking enforcement
  - UNKNOWN: CPCV vs WFA decision criteria in practice
  - UNKNOWN: US dividend/CA coverage quality vs B3
  - UNKNOWN: Fill rate simulation for limit orders
  - UNKNOWN: Database pool saturation under load
END_PAYLOAD
```

---

## Appendix: Skill File Locations

| Skill | File Path |
|-------|-----------|
| Team Index | `.cursor/skills/_TEAM_INDEX.md` |
| Team Playbook | `.cursor/skills/_TEAM_PLAYBOOK.md` |
| Handoff Contracts | `.cursor/skills/_HANDOFF_CONTRACTS.md` |
| Quality Bar | `.cursor/skills/_QUALITY_BAR.md` |
| quant-engineer | `.cursor/skills/quant-engineer/SKILL.md` |
| quant-researcher | `.cursor/skills/quant-researcher/SKILL.md` |
| risk-analyst | `.cursor/skills/risk-analyst/SKILL.md` |
| data-engineer | `.cursor/skills/data-engineer/SKILL.md` |
| trader-expert | `.cursor/skills/trader-expert/SKILL.md` |
| devops-infra | `.cursor/skills/devops-infra/SKILL.md` |
| omp-operator | `.cursor/skills/omp-operator/SKILL.md` |
