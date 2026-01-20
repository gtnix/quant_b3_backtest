# Team Index

Master index for the Quant Finance Team agent skills.

---

## What is This Team

The Quant Finance Team is a coordinated set of AI agent skills designed for institutional-grade quantitative strategy development. Each skill operates as a domain expert with strict boundaries, hard constraints, and explicit handoff protocols. The team covers the full lifecycle: data quality, strategy research, anti-overfitting validation, execution realism, performance engineering, infrastructure operations, and 24/7 perpetual mining with Hall of Fame governance. All skills enforce reproducibility (run_id, seed, config_hash, git_sha, snapshot_id) and audit trails.

---

## Skills Inventory

| Skill | Trigger | Purpose | Key Outputs |
|-------|---------|---------|-------------|
| [quant-engineer](quant-engineer/SKILL.md) | `/quant-engineer` | Rust performance engineering, zero-alloc hot paths, SIMD | Benchmark Report, Flamegraph, PR Checklist |
| [quant-researcher](quant-researcher/SKILL.md) | `/quant-researcher` | Evolutionary strategy discovery, Block DSL, Pareto optimization | Research Hypothesis Card, Pareto Front Summary, Handoff Packet |
| [risk-analyst](risk-analyst/SKILL.md) | `/risk-analyst` | Strategy validation, WFA/CPCV, PBO/DSR, stress testing | Validation Report, Promotion Memo, Overfitting Checklist |
| [data-engineer](data-engineer/SKILL.md) | `/data-engineer` | Data quality, OHLCV ingestion, corporate actions, calendars | Data Quality Report, CA Ledger, Universe Snapshot |
| [trader-expert](trader-expert/SKILL.md) | `/trader-expert` | Execution realism, slippage/costs, sessions, capacity | Execution Assumptions Card, Cost Report |
| [devops-infra](devops-infra/SKILL.md) | `/devops-infra` | Local infrastructure, CI/CD, monitoring, health checks | Deployment Checklist, Runbook, Postmortem |
| [omp-operator](omp-operator/SKILL.md) | `/omp-operator` | Mining 24/7, queue governance, Hall of Fame, watchdog | Campaign Spec Card, Promotion Packet, Ops Log |

**Placeholder (future):** `scg-architect` - Strategy Combination Genome design

---

## Trigger Quick Reference

| Need | Invoke | Example |
|------|--------|---------|
| Hot path bottleneck | `/quant-engineer` | "Profile process_day() and reduce allocations" |
| Design new campaign | `/quant-researcher` | "Create momentum+value search space for IBOV" |
| Validate before promotion | `/risk-analyst` | "Validate run_id X with WFA and stress suite" |
| Data looks wrong | `/data-engineer` | "Check PETR4 for missing bars and CA issues" |
| Costs seem unrealistic | `/trader-expert` | "Review slippage assumptions for intraday strategy" |
| Deploy failed | `/devops-infra` | "Rollback to previous version and investigate" |
| Mining stuck | `/omp-operator` | "Check campaign queue and restart daemon" |

---

## End-to-End Workflow

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

### Phase Gates

| Phase | Gate | Owner | Artifact |
|-------|------|-------|----------|
| 0 -> 1 | Data Quality Report PASS | data-engineer | snapshot_id documented |
| 1 -> 2 | Repro Pack complete | quant-researcher | run_id, config.toml, seed |
| 2 -> 3 | Validation Report PASS | risk-analyst | WFA/PBO/DSR metrics |
| 3 -> 4 | Execution Assumptions Card signed | trader-expert | Cost scenarios tested |
| 4 -> 5 | Benchmark no regression | quant-engineer | Before/after comparison |
| 5 -> 6 | Health checks PASS | devops-infra | deploy.sh verify |
| 6 -> 7 | Promotion Packet complete | omp-operator + risk-analyst | All gates confirmed |

---

## Quick Start

### How to Choose a Skill

1. **Identify the domain** of your current task
2. **Check the trigger table** above for the closest match
3. **Invoke with context**: Include run_id, file paths, or specific questions
4. **Expect handoff**: Skills will hand off to others when scope changes

### Invocation Examples

```markdown
/quant-researcher
Design a momentum campaign for B3 5-min bars with 6 blocks max.
Universe: IBOV. Timeframe: 2020-2024. Cost assumption: 10bps.

/risk-analyst
Validate run_id abc123 for Hall of Fame promotion.
Strategy is intraday, turnover ~20x annual.

/trader-expert
Review execution assumptions for genome xyz789.
Market: B3. Current slippage: 5bps. Concerned about capacity.

/omp-operator
Campaign stuck in queue for 2 hours. Check daemon status and resources.
```

### Cross-Reference Documents

- [_QUALITY_BAR.md](_QUALITY_BAR.md) - Structure requirements for all skills
- [_TEAM_PLAYBOOK.md](_TEAM_PLAYBOOK.md) - Operational workflow and checklists
- [_HANDOFF_CONTRACTS.md](_HANDOFF_CONTRACTS.md) - Formal handoff requirements
- [_TRAINING_INTRADAY.md](_TRAINING_INTRADAY.md) - Intraday strategy lifecycle training
- [_TRAINING_POSITION.md](_TRAINING_POSITION.md) - Position strategy lifecycle training
- [_RUNBOOK_PROMOTION.md](_RUNBOOK_PROMOTION.md) - Hall of Fame promotion runbook

---

## Prop-Firm Ready Checklist

Final verification that the system meets institutional standards.

### Data Correctness

```
[ ] Point-in-time universe (no lookahead in composition)
[ ] Corporate actions policy documented (adjusted vs raw)
[ ] OHLCV invariants pass (low <= open/close <= high)
[ ] Data Quality Report with snapshot_id for each dataset
[ ] Trading calendar verified for all markets (B3/US)
```

### Validation Rigor

```
[ ] WFA/CPCV with purge=5, embargo=5 days minimum
[ ] PBO < 0.20 (research) or < 0.10 (production)
[ ] DSR >= 0.5 (research) or >= 0.8 (production)
[ ] Stress tests: >= 4/5 passed (S1-S5)
[ ] Degradation IS->OOS < 50% (production)
```

### Execution Realism

```
[ ] Execution Assumptions Card completed for each strategy
[ ] 3 cost scenarios tested (base, pessimist, stress)
[ ] S1 (costs_2x) and S2 (delay+1) passed for intraday
[ ] S5 (combined_adverse) passed for position
[ ] Turnover <= 12x annual or justified exception
[ ] Capacity documented (>= $5M for institutional)
```

### Performance Engineering

```
[ ] PERFORMANCE_CONTRACT.md gates met (regression <= 5%)
[ ] Zero allocations in hot path verified (dhat)
[ ] Determinism: 3 identical runs produce same output
[ ] Golden tests pass after any optimization
```

### Infrastructure

```
[ ] Deploy script tested (deploy.sh full + verify)
[ ] Rollback tested (deploy.sh rollback + verify)
[ ] Health checks automated (health-check.sh)
[ ] PM2 autorestart and memory limits configured
[ ] Secrets in GitHub Secrets, not in git
[ ] Monitoring workflow enabled
```

### Mining Governance

```
[ ] OMP watchdog policy active (auto-stop on disk < 1GB)
[ ] Resource budget enforced (CPU < 85%, RAM > 400MB)
[ ] Validation automation active (variance gate)
[ ] Hall of Fame provenance complete (genome_hash, run_id, config_hash, git_sha)
[ ] Retention policy enforced (cleanup script)
[ ] Audit trail populated (activity logs)
```

### Reproducibility

```
[ ] Every run has: run_id, seed, config.toml, git_sha
[ ] Data snapshot_id documented for each campaign
[ ] Config hash recorded with promoted strategies
[ ] Artifacts retained per policy (HoF = permanent)
```

---

## Terminology Reference

| Term | Definition | Example |
|------|------------|---------|
| `run_id` | UUID v4 identifying a single execution | `a1b2c3d4-...` |
| `seed` | Random seed for determinism | `[42, 123, 456]` |
| `config.toml` | Frozen configuration snapshot | `configs/campaigns/momentum.toml` |
| `git_sha` | Git commit hash (same as `git_commit`) | `abc123def...` |
| `snapshot_id` | Data version identifier | UUID or date-based |
| `genome_id` | Human-readable strategy identifier | `mom_val_001` |
| `genome_hash` | Cryptographic hash of genome | `sha256:...` |
| `config_hash` | Cryptographic hash of config | `sha256:...` |

---

## Threshold Reference

### Validation Tiers (source: risk-analyst)

| Metric | Production | Research | Hard Fail |
|--------|------------|----------|-----------|
| OOS Sharpe (NET) | >= 1.0 | >= 0.5 | < 0.2 |
| Max Drawdown | <= 20% | <= 35% | > 50% |
| PBO | < 0.10 | < 0.20 | > 0.40 |
| DSR | >= 0.8 | >= 0.5 | < 0.2 |
| Stress Pass Rate | >= 4/5 | >= 3/5 | < 2/5 |

### Stage B Automation (source: omp_config.toml)

| Metric | Threshold |
|--------|-----------|
| min_oos_sharpe_net | 0.5 |
| max_pbo | 0.20 |
| min_dsr | 0.4 |
| max_drawdown_net | 0.30 |

**Note:** Stage B thresholds are for automated queue promotion. Final Hall of Fame promotion requires risk-analyst validation at the appropriate tier.
