# Team Playbook

Operational guide for the Quant Finance Team.

---

## Hardening Notes (P8)

Harmonization applied during team hardening:

- **scg-architect** is a future placeholder (not yet implemented)
- **Terminology**: `git_sha` and `git_commit` are synonymous (prefer `git_sha`)
- **Terminology**: `genome_id` is display name, `genome_hash` is cryptographic identifier
- **Thresholds**: `risk-analyst/SKILL.md` defines validation tiers (Production/Research); `omp_config.toml` defines Stage B automation thresholds
- **Promotion pipeline** requires all gates: data-engineer (data pass) -> trader-expert (execution, if turnover > 12x) -> risk-analyst (validation) -> omp-operator (HoF)
- **Data versioning**: All runs must have `snapshot_id` linking to specific data version
- **Reproducibility**: Every run requires run_id, seed, config.toml, git_sha

**Related documents:**
- [_TEAM_INDEX.md](_TEAM_INDEX.md) - Master index and Prop-Firm Ready checklist
- [_HANDOFF_CONTRACTS.md](_HANDOFF_CONTRACTS.md) - Formal handoff requirements

---


## Skill Invocation

### Available Skills

| Skill | Trigger | Purpose |
|-------|---------|---------|
| `quant-engineer` | `/quant-engineer` | Performance optimization, low-latency tuning |
| `quant-researcher` | `/quant-researcher` | Strategy discovery, genome design, campaign setup |
| `risk-analyst` | `/risk-analyst` | Strategy validation, anti-overfitting gates, promotion |
| `data-engineer` | `/data-engineer` | Data quality, ingestion, calendar, universe, corporate actions |
| `trader-expert` | `/trader-expert` | Execution realism, slippage, costs, sessions |
| `devops-infra` | `/devops-infra` | Local infrastructure, CI/CD, monitoring, health checks |
| `omp-operator` | `/omp-operator` | Mining 24/7, queue, Hall of Fame, resources, watchdog |
| `scg-architect` | `/scg-architect` | Strategy Combination Genome design (future) |

### When to Use Each

**Use `/quant-engineer` when:**
- Optimizing hot paths in `backtester_engine`
- Profiling CPU or memory bottlenecks
- Reviewing SIMD vectorization opportunities
- Validating performance against `PERFORMANCE_CONTRACT.md`

**Use `/quant-researcher` when:**
- Designing new search space or campaign
- Debugging premature convergence or diversity collapse
- Reviewing Pareto front diversity
- Preparing candidates for risk-analyst validation
- Choosing blocks for intraday vs position trading

**Use `/risk-analyst` when:**
- Validating strategy before promotion to Hall of Fame
- Reviewing Sharpe/PBO/DSR after Walk-Forward Analysis
- Stress testing intraday or position strategies
- Auditing reproducibility of backtest results
- Checking for overfitting red flags

**Use `/data-engineer` when:**
- Backtest results changed without strategy modification
- Suspecting data gaps, outliers, or corporate action issues
- Setting up new dataset or ingestion pipeline
- Investigating calendar or universe problems
- Query performance issues on Neon

**Use `/trader-expert` when:**
- Strategy has high turnover needing cost validation
- Execution assumptions need verification
- Intraday strategy operates near session boundaries
- Result degrades significantly with slippage increase
- Limit order fills assumed at 100%
- Short positions or borrow costs involved

**Use `/devops-infra` when:**
- Deploy failed or service not starting
- Latency increased or memory exhausted
- DB connections saturated
- Need rollback to previous version
- Setup or troubleshoot CI/CD
- Secrets exposed or environment drift

**Use `/omp-operator` when:**
- Mining stuck or campaigns failing
- Queue needs reordering or prioritization
- Promoting strategy to Hall of Fame
- Resource limits exceeded (CPU/RAM/disk)
- Disk space running low
- Pausing/resuming/stopping mining
- Investigating run reproducibility
- Cleaning up old runs and artifacts

**Handoff patterns:**
- After `/quant-engineer` optimizes engine → `/trader-expert` validates execution realism
- After `/quant-researcher` produces candidates → `/risk-analyst` validates
- After `/quant-researcher` designs campaign → `/trader-expert` reviews cost assumptions
- After `/quant-researcher` designs campaign → `/omp-operator` queues it for mining
- After `/risk-analyst` sets constraints → `/quant-engineer` ensures no perf regression
- Before promotion → `/risk-analyst` validates anti-overfitting gates
- After `/risk-analyst` flags data issue → `/data-engineer` investigates
- After `/data-engineer` fixes data issue → `/risk-analyst` re-validates
- After `/data-engineer` sets up new dataset → `/quant-researcher` uses it
- After `/data-engineer` flags data incident → `/omp-operator` pauses mining
- After `/trader-expert` reviews execution → `/risk-analyst` validates with stress suite
- After `/devops-infra` deploys → `/risk-analyst` verifies artifacts retained
- After `/devops-infra` detects resource pressure → `/omp-operator` adjusts mining load
- After `/data-engineer` prepares migration → `/devops-infra` coordinates deploy
- After `/devops-infra` detects resource pressure → `/quant-engineer` investigates hot paths
- After `/omp-operator` promotes candidate → `/risk-analyst` confirms HoF integrity

---

## Standard Workflow

```
Data Readiness → Research → Validation → Execution Realism → Performance → Deploy → Mining
```

### Phase Details

**0. Data Readiness** (gate before Research)
- `/data-engineer` validates data quality
- Data readiness checklist passed
- Snapshot_id documented

**1. Research**
- `/quant-researcher` designs search space and campaign
- Define hypothesis and mandate
- Configure evolution parameters
- Run campaign and select Pareto front
- Research output must include: run_id, config.toml, seed, complexity_budget

**2. Validation**
- Unit tests pass
- Golden tests unchanged
- Property tests hold
- `/risk-analyst` validates OOS performance (for strategies)

**3. Execution Realism** (gate before Performance)
- `/trader-expert` reviews execution assumptions
- Execution Assumptions Card filled
- Transaction costs modeled with sensitivity
- Slippage and market impact accounted for
- Session/calendar constraints verified

**4. Performance**
- Benchmark baseline captured
- Optimization applied
- Regression check passed

**5. Deploy** (gate before Mining)
- `/devops-infra` coordinates deployment
- CI gates green
- Config frozen and versioned with git sha
- Artifacts versioned
- Health checks pass post-deploy
- Rollback plan documented and tested

**6. Mining** (24/7 operation)
- `/omp-operator` manages perpetual mining
- Resource budget enforced (CPU < 85%, RAM > 400MB free, Disk > 1GB)
- Watchdog policy active (auto-stop on disk < 1GB)
- Queue governance: priority, fairness, repeat mode
- Validation automation: variance gate, threshold checks
- Hall of Fame promotion with full provenance
- Metrics logged via SSE events
- Drift monitored, incidents escalated

---

## Standard Artifacts

### Run Identification

Every run produces:

| Artifact | Format | Purpose |
|----------|--------|---------|
| `run_id` | UUID v4 | Unique identifier |
| `config.toml` | TOML | Exact parameters used |
| `baseline.json` | JSON | Benchmark metrics |
| `flamegraph.svg` | SVG | CPU profile visualization |

### Directory Layout

```
artifacts/
└── runs/
    └── {run_id}/
        ├── config.toml
        ├── metrics.json
        ├── trades.csv
        └── nav_history.csv
```

### Metadata Schema

```json
{
  "run_id": "uuid",
  "git_commit": "sha256",
  "timestamp": "ISO8601",
  "seed": 42,
  "scenario": {
    "assets": 10,
    "days": 252,
    "market": "BR"
  },
  "performance": {
    "wall_time_ms": 150,
    "throughput_days_per_sec": 1680
  }
}
```

---

## Communication Protocol

### Skill-to-Skill Handoff

When handing off to another skill:

```markdown
## Handoff: quant-engineer → risk-analyst

**Context:**
- Optimized `process_day()` from 15μs to 8μs
- Changed HashMap to Vec for price lookup

**Requires validation:**
- [ ] Risk constraints still enforced
- [ ] Drawdown calculation unchanged
- [ ] Cost model accuracy preserved

**Files modified:**
- `crates/backtester_engine/src/unified.rs`
```

### Escalation

If blocked:
1. Document the blocker clearly
2. Identify which skill can resolve it
3. Provide minimal reproduction steps

---

## Quality Checkpoints

Data readiness before validation:

```
[ ] OHLC invariants: low <= open/close <= high
[ ] Volume >= 0, no negatives
[ ] No duplicate bars
[ ] Gaps explained (holiday vs missing)
[ ] Corporate actions applied per policy
[ ] Universe is point-in-time (no lookahead)
[ ] Data freshness < 24h for active period
[ ] Snapshot_id documented
```

Execution realism before promotion:

```
[ ] Execution Assumptions Card completed
[ ] Slippage model specified (bps or volume-based)
[ ] Fee tier selected and justified
[ ] 3 cost scenarios tested (base/pessimist/stress)
[ ] Intraday: S1 (costs_2x) and S2 (delay+1) passed
[ ] Position: S5 (combined_adverse) passed
[ ] Capacity proxy documented
[ ] Session constraints noted (regular/auction/after)
[ ] Limitations declared (settlement, borrow, etc.)
[ ] Handoff to risk-analyst signed
```

Production readiness before perpetual mining:

```
[ ] Deploy script tested (deploy.sh full + verify)
[ ] Rollback tested (deploy.sh rollback + verify)
[ ] PM2 ecosystem config reviewed
[ ] Health checks passing (health-check.sh)
[ ] Nginx config validated (nginx -t)
[ ] Secrets in GitHub Secrets, not in git
[ ] Logs accessible (pm2 logs)
[ ] Resource budget respected (RAM < 80%)
[ ] Monitoring workflow enabled
[ ] Runbook documented for each service
```

Before adding campaign to queue:

```
[ ] Campaign config exists and validates
[ ] Config path correct in queue entry
[ ] Market specified (br/us)
[ ] Priority assigned (lower = higher precedence)
[ ] Data readiness confirmed for target period
[ ] Resource budget acceptable for campaign size
[ ] Seeds documented for reproducibility
[ ] Expected runtime within limits
[ ] Repeat mode set correctly (true for continuous)
[ ] Tags applied for filtering
```

Before Hall of Fame promotion:

```
[ ] Variance sanity gate passed (sharpeVar > 1e-6)
[ ] OOS Sharpe >= threshold (0.5 default)
[ ] PBO <= threshold (0.20 default)
[ ] DSR >= threshold (0.4 default)
[ ] Max Drawdown <= 30%
[ ] Provenance complete: genome_hash, run_id, config_hash, git_sha
[ ] Artifacts present: strategy.toml, metrics.obfs
[ ] Risk-analyst gate confirmed
[ ] Trader-expert reviewed (if turnover > 12x annual)
[ ] No duplicate genome_hash in HoF
```

Before any PR:

```
[ ] Tests pass: cargo test --release
[ ] Benchmarks run: cargo bench
[ ] Lints clean: cargo clippy -- -D warnings
[ ] Format: cargo fmt --check
[ ] Determinism: 3 identical runs
```

Before merge to main:

```
[ ] All CI checks green
[ ] No regression > 5%
[ ] Reviewer approved
[ ] Artifacts archived
```

Before strategy promotion:

```
[ ] /data-engineer data quality passed
[ ] /trader-expert execution realism passed
[ ] /quant-researcher produced candidates with repro pack
[ ] /risk-analyst validation passed
[ ] OOS Sharpe >= tier threshold
[ ] PBO < tier threshold
[ ] Stress tests: >= 4/5 passed
[ ] Artifacts complete (run_id, config, git commit)
```
