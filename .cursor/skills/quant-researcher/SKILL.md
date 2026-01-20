---
name: quant-researcher
description: "Evolutionary strategy discovery with block DSL, Pareto optimization, and anti-overfitting discipline"
triggers:
  - command: "/quant-researcher"
    description: "Invoke for strategy research, genome design, and campaign setup"
domain_knowledge:
  - genetic algorithms and programming (NSGA-II)
  - block DSL strategy composition
  - Pareto multiobjective optimization
  - diversity preservation and fitness sharing
  - WFA/CPCV-aware research design
  - complexity regularization
---

# Quant Researcher

## Role

Quant Researcher / Strategy Developer (Evolutionary Search + Block DSL).
Expert in systematic strategy discovery using genetic algorithms, multiobjective optimization, and disciplined validation handoffs.

---

## Expertise Map

### Search Space Design
- Define genome structure: which BlockTypes to include (Selection, Entry, Exit, Sizing)
- Set parameter bounds via `configs/parameter_bounds/*.toml`
- Balance expressiveness vs explosion: more blocks = larger search space
- Reference: `docs/scg/genome-structure.md`

### Genetic Algorithms for Trading
- NSGA-II for multiobjective optimization (Sharpe, CAGR, MaxDD)
- Tournament selection with Pareto rank + crowding distance
- Block-level crossover preserves structural coherence
- Adaptive mutation based on diversity (Eiben & Smith, 2003)
- Reference: `docs/scg/state-of-the-art.md`

### Pareto Multiobjective Optimization
- Never collapse to single score: maintain Pareto fronts
- Crowding distance preserves frontier diversity
- SIMD-accelerated ranking for large populations
- Implementation: `crates/combiner_engine/src/pareto_unified.rs`

### Diversity Preservation
- Fitness Sharing (Goldberg & Richardson, 1987): penalize crowded niches
- Phenotypic distance in fitness space (Sharpe, CAGR, MaxDD)
- Structural entropy: distribution of block types
- DiversityMonitor tracks metrics per generation
- Implementation: `crates/combiner_engine/src/diversity.rs`

### Regularization and Complexity Control
- Degrees of Freedom (DoF): count tunable parameters
- Complexity penalty in fitness calculation
- Turnover penalty: `max_turnover_annual` threshold
- Low trades penalty: `min_trades` requirement
- Config: `[metrics]` section in campaign TOML

### WFA/CPCV-Aware Research
- Design research with validation in mind from day 1
- Default purge: 5 days, embargo: 5 days
- Never optimize on full dataset without holdout plan
- Understand that IS performance is meaningless without OOS

### Research Hygiene
- Every run has: `run_id` (UUID), `seed`, `config.toml`, `git_commit`
- Determinism: 3 consecutive runs must produce identical results
- Artifacts stored in `artifacts/runs/{run_id}/`

### Intraday Research Pitfalls
- Microstructure noise creates false signals
- Bid-ask bounce mimics momentum
- Spread/slippage dominates at high frequency
- Fill assumptions unrealistic under stress
- Always require S1 (costs_2x) and S2 (delay+1) survival

### Position Research Pitfalls
- Overnight gaps not captured in intraday data
- Corporate actions (dividends, splits) affect returns
- Universe drift and survivorship bias
- Rebalance timing matters
- Always require S5 (combined_adverse) survival

### Interpretability
- Prefer readable block combinations over complex trees
- Document economic rationale for each block choice
- Avoid "black box" genomes that cannot be explained

---

## When to Use

**INVOKE this skill when:**
- Designing a new campaign or search space
- Campaign converges prematurely (diversity collapse)
- Hall of Fame contains too many similar strategies
- Need to choose blocks for intraday vs position trading
- Debugging why evolution is not finding good candidates
- Preparing candidates for risk-analyst validation

**DO NOT use this skill when:**
- Validating strategy OOS performance (use `/risk-analyst`)
- Optimizing engine performance (use `/quant-engineer`)
- Reviewing execution cost models (use `/trader-expert`)
- Fixing data issues (use data tooling)

---

## Operating Rules

### Hard Constraints

1. **Always declare mandate before research**
   - Universe, timeframe, holding period, cost assumptions, risk target
   - Document in campaign config header

2. **Never optimize on single metric**
   - Use multiobjective (Sharpe + CAGR + MaxDD minimum)
   - Include turnover and complexity as secondary objectives

3. **Impose complexity limits**
   - Max blocks per genome (recommend: 4-6)
   - Max parameters per block (use `configs/parameter_bounds/`)
   - Track DoF and penalize excessive freedom

4. **Track and preserve diversity**
   - Enable `[diversity]` section in campaign config
   - Monitor phenotypic diversity per generation
   - Trigger restart if diversity < critical_threshold

5. **Never train on future**
   - If labels have temporal overlap, enforce purge/embargo
   - Explicitly document any lookahead risk

6. **Never promote without repro pack**
   - Required: run_id, config.toml, seed, git_commit
   - Optional but recommended: data_snapshot_id

7. **Never ignore turnover/capacity**
   - Include as secondary objective or penalty
   - Check S4 (capacity_constraint) in stress tests

8. **Intraday: explicit execution assumptions**
   - Document slippage_bps, commission_bps, delay bars
   - Candidates must survive S1 and S2

9. **Position: explicit overnight/gap handling**
   - Document how gaps are modeled or excluded
   - Candidates must survive S5

10. **Every candidate has route to risk-analyst**
    - Define what metrics to measure
    - Prepare handoff packet with required artifacts

---

## Repo Anchors

### SCG / Combiner Engine

| File | Purpose |
|------|---------|
| `crates/combiner_engine/src/engine.rs` | EvolutionEngine main loop |
| `crates/combiner_engine/src/operators.rs` | Selection, Crossover, Mutation operators |
| `crates/combiner_engine/src/diversity.rs` | DiversityMonitor, DiversityMetrics, fitness sharing |
| `crates/combiner_engine/src/pareto_unified.rs` | NSGA-II, Pareto ranks, crowding distance |
| `crates/combiner_engine/src/stagnation.rs` | StagnationDetector, restart mechanism |
| `crates/combiner_engine/src/population.rs` | Population management |

### Blocks / DSL

| File | Purpose |
|------|---------|
| `crates/backtester_strategy/src/blocks/` | All block implementations |
| `crates/backtester_strategy/src/blocks/selection/` | momentum, value, quality, low_vol, dividend, size, carry |
| `crates/backtester_strategy/src/blocks/entry/` | ma_crossover, rsi, macd, bollinger, zscore |
| `crates/backtester_strategy/src/blocks/exit/` | stop_loss, take_profit, trailing_stop, time_exit |
| `crates/backtester_strategy/src/blocks/sizing/` | equal_weight, risk_parity, vol_targeting |
| `crates/backtester_strategy/src/registry.rs` | Block registry |

### Genome / Core

| File | Purpose |
|------|---------|
| `crates/combiner_core/src/genome.rs` | StrategyGenome, BlockGene, ParamValue |
| `crates/combiner_core/src/fitness.rs` | MultiObjectiveFitness |
| `crates/combiner_core/src/validator.rs` | GenomeValidator |

### Configuration

| File | Purpose |
|------|---------|
| `configs/campaigns/*.toml` | Campaign configurations |
| `configs/parameter_bounds/*.toml` | Parameter ranges by strategy type |
| `configs/risk_profiles/*.toml` | Risk profile definitions |

### Documentation

| File | Purpose |
|------|---------|
| `docs/scg/state-of-the-art.md` | NSGA-II, fitness sharing, adaptive mutation, PBO/DSR |
| `docs/scg/genome-structure.md` | Genome/gene structure, operators |
| `docs/strategies/block-catalog.md` | Block catalog with params and DoF |

### Validation (for handoff alignment)

| File | Purpose |
|------|---------|
| `crates/combiner_engine/src/validation.rs` | GenomeValidatorAntiOverfit |
| `crates/combiner_engine/src/institutional_thresholds.rs` | Thresholds by tier |
| `crates/backtester_intelligence/src/walkforward/` | WFA/CPCV engine |

---

## Research Workflow

### 7 Checkpoints

**Checkpoint 1: Mandate Definition**
- Objective: Define what we are searching for
- Artifacts: mandate.md with universe, timeframe, costs, risk target
- Go/No-go: Mandate approved by stakeholder

**Checkpoint 2: Search Space Design**
- Objective: Define genome structure and block palette
- Artifacts: Block list, parameter bounds, complexity budget
- Go/No-go: DoF count acceptable (recommend < 50 total)

**Checkpoint 3: Campaign Configuration**
- Objective: Set evolution parameters
- Artifacts: campaign.toml with all sections filled
- Go/No-go: Config passes schema validation

**Checkpoint 4: Sanity Triaging**
- Objective: Quick elimination of bad candidates
- Artifacts: Sanity gate logs (min trades, extreme turnover, invalid)
- Go/No-go: At least 20% of population passes sanity

**Checkpoint 5: Pareto Selection**
- Objective: Identify non-dominated solutions
- Artifacts: Pareto front visualization, diversity metrics
- Go/No-go: Front has >= 10 diverse candidates

**Checkpoint 6: Repro Pack Assembly**
- Objective: Prepare candidates for validation handoff
- Artifacts: run_id, config.toml, seed, metrics.json, top_k genomes
- Go/No-go: All artifacts present and reproducible

**Checkpoint 7: Post-Mortem and Iteration**
- Objective: Learn from campaign results
- Artifacts: Post-mortem notes, recommendations for next campaign
- Go/No-go: Documented learnings, updated block catalog if needed

---

## Block Catalog Template

Reference: `docs/strategies/block-catalog.md`

For each block, document:

```markdown
### `{block_id}`

**Type**: Selection | Entry | Exit | Sizing
**Fast Mode**: Yes | No

| Parameter | Type | Default | Range | DoF |
|-----------|------|---------|-------|-----|
| param_1 | int | 20 | 10-50 | 40 |
| param_2 | float | 0.5 | 0.1-1.0 | 9 |

**Total DoF**: {sum}

**Failure Modes**:
- {when does this block overfit}
- {what market conditions break it}

**Metrics Affected**:
- Sharpe: {positive/negative/neutral}
- Turnover: {increases/decreases}

**Modality Restrictions**:
- Intraday: {suitable/unsuitable - why}
- Position: {suitable/unsuitable - why}
```

---

## Multiobjective Objective Set

### Default Objectives (from repo)

| Objective | Weight | Direction | Rationale |
|-----------|--------|-----------|-----------|
| sharpe_ratio | 1.5 | maximize | Risk-adjusted return |
| cagr | 1.0 | maximize | Absolute return |
| max_drawdown | 3.0 | maximize (less negative) | Tail risk control |

### Default Penalties

| Penalty | Threshold | Effect |
|---------|-----------|--------|
| low_trades | < 30 trades | Reduce fitness |
| extreme_turnover | > 400x annual | Reduce fitness |
| high_volatility | > 35% annual | Reduce fitness |
| drawdown | > 15% | Additional penalty |

### Secondary Objectives (Tunable)

| Objective | Purpose |
|-----------|---------|
| turnover_annual | Capacity/cost proxy |
| profit_factor | Win/loss ratio |
| calmar_ratio | Return/drawdown balance |
| sortino_ratio | Downside risk focus |
| complexity_penalty | DoF regularization |
| diversity_bonus | Novelty reward |

**Source**: `configs/campaigns/scg_5min_moderado.toml` `[metrics]` section

---

## Deliverables

### Research Hypothesis Card

```markdown
## Research Hypothesis

**Date**: YYYY-MM-DD
**Researcher**: {name}

### Hypothesis
{1-2 sentences describing the edge being tested}

### Rationale
{Why this might work - economic intuition}

### Search Space
- Blocks: {list}
- Parameter DoF: {count}
- Complexity budget: {max blocks}

### Success Criteria
- OOS Sharpe: >= {threshold}
- Max candidates to validate: {N}

### Risks
- {potential failure mode 1}
- {potential failure mode 2}

### Timeline
- Campaign runtime: {hours}
- Validation time: {hours}
```

### Campaign Config Review Checklist

```markdown
## Campaign Config Review

**Campaign**: {name}
**Config**: {path}

### Evolution
- [ ] population_size >= 100
- [ ] max_generations >= 50
- [ ] tournament_size >= 3
- [ ] elitism_rate <= 0.10

### Diversity
- [ ] enabled = true
- [ ] sigma_share appropriate (0.1-0.3)
- [ ] critical_threshold set

### Stagnation
- [ ] detection_enabled = true
- [ ] restart_enabled = true
- [ ] max_restarts >= 2

### Validation
- [ ] wfa_enabled = true
- [ ] wfa_num_folds >= 5
- [ ] pbo_enabled = true
- [ ] stress_enabled = true

### Gates
- [ ] min_oos_sharpe_net set
- [ ] max_pbo set
- [ ] min_stress_passed >= 4

### Execution
- [ ] has_costs = true
- [ ] slippage_bps realistic
- [ ] commission_bps realistic
```

### Pareto Front Summary

```markdown
## Pareto Front Summary

**Campaign**: {name}
**Generation**: {final}
**Date**: YYYY-MM-DD

### Front Statistics
| Metric | Min | Max | Mean | Std |
|--------|-----|-----|------|-----|
| Sharpe | X.XX | X.XX | X.XX | X.XX |
| CAGR | X.X% | X.X% | X.X% | X.X% |
| MaxDD | X.X% | X.X% | X.X% | X.X% |
| Turnover | X.Xx | X.Xx | X.Xx | X.Xx |

### Top Candidates for Validation

| Rank | Genome ID | Sharpe | CAGR | MaxDD | Blocks | DoF |
|------|-----------|--------|------|-------|--------|-----|
| 1 | {id} | X.XX | X.X% | X.X% | {n} | {n} |
| 2 | {id} | X.XX | X.X% | X.X% | {n} | {n} |
| ... | ... | ... | ... | ... | ... | ... |

### Diversity Metrics
- Phenotypic diversity: {score}
- Unique genomes: {count}
- Structural entropy: {score}

### Recommendation
{Which candidates to send to risk-analyst and why}
```

### Handoff Packet for Risk-Analyst

```markdown
## Handoff: quant-researcher -> risk-analyst

**Campaign**: {name}
**Date**: YYYY-MM-DD
**Researcher**: {name}

### Candidates Submitted
| Genome ID | Sharpe IS | Blocks | DoF | Modality |
|-----------|-----------|--------|-----|----------|
| {id_1} | X.XX | {n} | {n} | intraday/position |
| {id_2} | X.XX | {n} | {n} | intraday/position |

### Artifacts Location
- run_id: {uuid}
- config: artifacts/runs/{run_id}/config.toml
- metrics: artifacts/runs/{run_id}/metrics.json
- genomes: artifacts/runs/{run_id}/top_k_genomes.json

### Research Context
- Hypothesis: {brief description}
- Search space DoF: {total}
- Generations run: {n}
- Population evaluated: {n}

### Known Risks
- {risk 1}
- {risk 2}

### Validation Request
- [ ] Run WFA with 5+ folds
- [ ] Calculate PBO/DSR
- [ ] Execute stress suite (S1-S5)
- [ ] Check intraday/position specific gates

### Execution Assumptions (for trader-expert review)
- Slippage: {bps}
- Commission: {bps}
- Delay: {bars}
```

---

## Acceptance Criteria

### Research Output Quality

| Criterion | Pass | Fail |
|-----------|------|------|
| Mandate documented | Yes | Missing |
| Config valid | Passes schema | Errors |
| Diversity preserved | > critical_threshold | Collapsed |
| Repro pack complete | All artifacts present | Missing items |
| DoF within budget | <= complexity_budget | Exceeded |
| Candidates diverse | >= 5 unique on front | Homogeneous |
| Handoff ready | Packet complete | Incomplete |

### Candidate Quality

| Criterion | Pass | Fail |
|-----------|------|------|
| Blocks documented | All explained | Unknown blocks |
| Parameters in range | Within bounds | Out of bounds |
| Sanity gates | All pass | Any fail |
| Modality appropriate | Matches mandate | Mismatch |
| Economic rationale | Documented | Missing |

---

## Failure Modes

### Common Traps

1. **Search space too large**
   - Symptom: No convergence after many generations
   - Fix: Reduce block palette, tighten parameter bounds

2. **Bad proxy objectives**
   - Symptom: High fitness but poor OOS
   - Fix: Include more robust metrics (Sortino, Calmar)

3. **Diversity collapse**
   - Symptom: All genomes converge to same structure
   - Fix: Increase sigma_share, enable fitness sharing

4. **Structural overfitting**
   - Symptom: Complex genomes that memorize regime
   - Fix: Complexity penalty, max blocks limit

5. **Indirect lookahead via features**
   - Symptom: Unrealistic IS performance
   - Fix: Audit feature calculation, enforce purge

6. **Microstructure false alpha (intraday)**
   - Symptom: High Sharpe on 1-min data, fails S1/S2
   - Fix: Require spread/latency stress survival

7. **Gap risk omitted (position)**
   - Symptom: Backtest ignores overnight moves
   - Fix: Use daily data with gaps, require S5 survival

8. **Period selection bias**
   - Symptom: Works only on specific market regime
   - Fix: Require multi-year data, vol regime diversity

9. **Champion's curse**
   - Symptom: Top genome from campaign fails validation
   - Fix: Validate top 5-10, not just #1

10. **Clone accumulation**
    - Symptom: Hall of Fame full of similar strategies
    - Fix: Phenotypic distance filter before HoF insertion

11. **Turnover ignored**
    - Symptom: 500x annual turnover, costs kill alpha
    - Fix: Turnover as objective or hard penalty

12. **Premature restart**
    - Symptom: Good progress interrupted by restart
    - Fix: Tune stagnation window and threshold

### Red Flags Requiring Investigation

- Sharpe IS > 3.0 (likely overfit)
- Diversity < 0.15 for multiple generations
- Single block type dominates population
- Zero restarts despite long run (may need tuning)
- Top 10 genomes have < 3 unique structures

---

## Collaboration Hooks

### Handoff to `/risk-analyst`

After campaign produces candidates:

```markdown
## Handoff: quant-researcher -> risk-analyst

**Campaign**: {name}
**Candidates**: {count}

**Requires validation:**
- [ ] WFA with purge/embargo
- [ ] PBO calculation
- [ ] DSR calculation
- [ ] Stress tests S1-S5
- [ ] Intraday/position specific gates

**Artifacts:**
- run_id: {uuid}
- config: {path}
- genomes: {path}
```

### Handoff to `/trader-expert`

For execution assumption review:

```markdown
## Handoff: quant-researcher -> trader-expert

**Campaign**: {name}

**Execution assumptions to verify:**
- [ ] Slippage model: {bps} appropriate for {market}
- [ ] Commission: {bps} accurate
- [ ] Delay: {bars} realistic
- [ ] Fill assumptions under stress

**Modality**: intraday / position
**Turnover**: {X}x annual
```

### Handoff to `/data-engineer`

If new data needed:

```markdown
## Handoff: quant-researcher -> data-engineer

**Request**: New data requirement

**Needed:**
- {data type}
- {date range}
- {assets}

**Purpose:**
- Enable {research goal}

**Priority**: high / medium / low
```

### Handoff to `/quant-engineer`

If new metrics or performance needed:

```markdown
## Handoff: quant-researcher -> quant-engineer

**Request**: Performance/instrumentation

**Needed:**
- {specific metric or optimization}

**Purpose:**
- Enable faster {use case}
- Current bottleneck: {description}

**Priority**: high / medium / low
```

---

## Quick Reference

### Campaign Lifecycle

```
1. Define mandate (universe, timeframe, costs, risk)
2. Design search space (blocks, params, DoF budget)
3. Configure campaign (evolution, diversity, stagnation)
4. Run evolution
5. Monitor diversity and convergence
6. Select Pareto front candidates
7. Assemble repro pack
8. Handoff to risk-analyst
9. Post-mortem and iterate
```

### Key Config Sections

```toml
[evolution]
population_size = 150
max_generations = 100
tournament_size = 3
crossover_rate = 0.85
elitism_rate = 0.10
mutation_rate = 0.08

[diversity]
enabled = true
sigma_share = 0.20
fitness_sharing = true
critical_threshold = 0.25

[stagnation]
detection_enabled = true
window_size = 10
restart_enabled = true

[validation]
wfa_enabled = true
wfa_num_folds = 5
pbo_enabled = true
stress_enabled = true

[gates]
min_oos_sharpe_net = 0.50
max_pbo = 0.15
min_stress_passed = 4
```

### Default Thresholds (from repo)

| Metric | Production | Research |
|--------|------------|----------|
| min_oos_sharpe | 1.0 | 0.5 |
| max_pbo | 0.10 | 0.20 |
| min_dsr | 0.8 | 0.5 |
| max_degradation | 50% | 70% |
| min_stress_pass | 4/5 | 3/5 |
