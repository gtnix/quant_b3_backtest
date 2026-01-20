# Handoff Contracts

Formal contracts for skill-to-skill handoffs. Each contract defines mandatory inputs, required artifacts, and acceptance criteria.

---

## Contract Index

| ID | Handoff | Direction |
|----|---------|-----------|
| H1 | [quant-researcher -> risk-analyst](#h1-quant-researcher--risk-analyst) | Validation request |
| H2 | [risk-analyst -> trader-expert](#h2-risk-analyst--trader-expert) | Execution review |
| H3 | [trader-expert -> risk-analyst](#h3-trader-expert--risk-analyst) | Execution sign-off |
| H4 | [data-engineer -> risk-analyst](#h4-data-engineer--risk-analyst) | Data issue resolved |
| H5 | [data-engineer -> omp-operator](#h5-data-engineer--omp-operator) | Data incident |
| H6 | [omp-operator -> risk-analyst](#h6-omp-operator--risk-analyst) | Promotion request |
| H7 | [devops-infra <-> omp-operator](#h7-devops-infra--omp-operator) | Resources/incidents |
| H8 | [quant-engineer <-> any](#h8-quant-engineer--any) | Performance/instrumentation |

---

## H1: quant-researcher -> risk-analyst

**When:** Candidates from campaign are ready for OOS validation.

### Entry Minimum (Required Fields)

| Field | Description | Example |
|-------|-------------|---------|
| `run_id` | UUID of the campaign run | `a1b2c3d4-...` |
| `config.toml` | Path to frozen config | `artifacts/runs/{run_id}/config.toml` |
| `seed` | Random seeds used | `[42, 123, 456]` |
| `git_sha` | Code version | `abc123...` |
| `genome_list` | Top N candidates | List of genome_id/genome_hash |
| `modality` | intraday or position | `intraday` |
| `market` | Target market | `BR` or `US` |

### Artifacts Required

| Artifact | Location | Format |
|----------|----------|--------|
| Handoff Packet | inline or file | Markdown template |
| Pareto Front Summary | `artifacts/runs/{run_id}/pareto_summary.md` | Markdown |
| Top K Genomes | `artifacts/runs/{run_id}/top_k_genomes.json` | JSON |
| Metrics JSON | `artifacts/runs/{run_id}/metrics.json` | JSON |

### Acceptance Criteria

| Criterion | Pass | Fail |
|-----------|------|------|
| run_id present | Yes | Missing |
| Config exists at path | Yes | File not found |
| Seed documented | Yes | Missing |
| Modality declared | Yes | Unspecified |
| >= 1 candidate | Yes | Empty list |

### Common Failures

| Failure | Symptom | Resolution |
|---------|---------|------------|
| Missing run_id | Cannot trace artifacts | Re-run with proper logging |
| Config mutated | Reproducibility fails | Use config_hash verification |
| No modality | Wrong stress tests | Researcher must declare |

---

## H2: risk-analyst -> trader-expert

**When:** Validation passed but turnover > 12x annual or execution concerns flagged.

### Entry Minimum (Required Fields)

| Field | Description | Example |
|-------|-------------|---------|
| `genome_id` | Strategy identifier | `mom_val_001` |
| `run_id` | Source run | `a1b2c3d4-...` |
| `turnover_annual` | Annual turnover ratio | `18.5x` |
| `market` | Trading market | `BR` |
| `modality` | intraday or position | `intraday` |
| `slippage_assumed` | Current slippage bps | `5 bps` |

### Artifacts Required

| Artifact | Location | Format |
|----------|----------|--------|
| Validation Report | inline or file | Markdown template |
| Trades CSV | `artifacts/runs/{run_id}/trades.csv` | CSV |

### Acceptance Criteria

| Criterion | Pass | Fail |
|-----------|------|------|
| Validation Report attached | Yes | Missing |
| Turnover documented | Yes | Unknown |
| Trades file exists | Yes | Missing |

### Common Failures

| Failure | Symptom | Resolution |
|---------|---------|------------|
| No trades file | Cannot analyze execution | Re-run backtest with trade logging |
| Turnover wrong | Capacity analysis fails | Recalculate from trades |

---

## H3: trader-expert -> risk-analyst

**When:** Execution review completed, Assumptions Card ready for sign-off.

### Entry Minimum (Required Fields)

| Field | Description | Example |
|-------|-------------|---------|
| `genome_id` | Strategy identifier | `mom_val_001` |
| `execution_review_status` | PASSED / FAILED | `PASSED` |
| `slippage_model` | Model used | `VolumeLinear` |
| `fee_tier` | Cost tier | `B3Prime` |
| `capacity_usd` | Estimated capacity | `$8M` |

### Artifacts Required

| Artifact | Location | Format |
|----------|----------|--------|
| Execution Assumptions Card | inline or file | Markdown template (signed) |
| Cost Report | optional | Markdown |

### Acceptance Criteria

| Criterion | Pass | Fail |
|-----------|------|------|
| Assumptions Card complete | All fields filled | Missing sections |
| 3 cost scenarios tested | Yes | Single scenario |
| S1/S2 passed (intraday) | Yes | Failed or not run |
| S5 passed (position) | Yes | Failed or not run |
| Limitations declared | Explicit | Assumed away |

### Common Failures

| Failure | Symptom | Resolution |
|---------|---------|------------|
| Unsigned card | Missing sign-off | Trader-expert must sign |
| No stress results | Cannot verify robustness | Re-run stress suite |
| Capacity missing | Institutional gate unknown | Calculate from trades |

---

## H4: data-engineer -> risk-analyst

**When:** Data issue has been investigated and resolved.

### Entry Minimum (Required Fields)

| Field | Description | Example |
|-------|-------------|---------|
| `incident_id` | Issue tracking ID | `DATA-001` |
| `affected_period` | Date range | `2024-01-01 to 2024-03-15` |
| `affected_symbols` | Impacted assets | `PETR4, VALE3` |
| `resolution_status` | FIXED / PARTIAL / WONTFIX | `FIXED` |
| `new_snapshot_id` | Corrected data version | `uuid-...` |

### Artifacts Required

| Artifact | Location | Format |
|----------|----------|--------|
| Data Quality Report | inline or file | Markdown template |
| Incident Report | if applicable | Markdown |

### Acceptance Criteria

| Criterion | Pass | Fail |
|-----------|------|------|
| Incident documented | Yes | No record |
| New snapshot_id | Present | Same as before |
| OHLCV invariants pass | Yes | Still failing |
| Re-validation required | Declared | Unclear |

### Common Failures

| Failure | Symptom | Resolution |
|---------|---------|------------|
| Silent fix | No snapshot_id change | Create new snapshot |
| Partial fix | Some symbols still bad | Re-investigate |
| No incident report | Cannot audit | Document the issue |

---

## H5: data-engineer -> omp-operator

**When:** Data incident detected that affects ongoing mining.

### Entry Minimum (Required Fields)

| Field | Description | Example |
|-------|-------------|---------|
| `incident_type` | Type of issue | `data_gap`, `ca_missing`, `integrity_fail` |
| `severity` | SEV-0 / SEV-1 / SEV-2 | `SEV-1` |
| `affected_market` | BR / US | `BR` |
| `affected_period` | Date range | `2024-06-01 to now` |
| `action_required` | pause / stop / investigate | `pause` |

### Artifacts Required

| Artifact | Location | Format |
|----------|----------|--------|
| Incident Alert | inline | Markdown |

### Acceptance Criteria

| Criterion | Pass | Fail |
|-----------|------|------|
| Severity declared | Yes | Unknown |
| Market specified | Yes | Unclear scope |
| Action clear | Yes | Ambiguous |

### Common Failures

| Failure | Symptom | Resolution |
|---------|---------|------------|
| Mining continues | Garbage data ingested | OMP must pause immediately |
| Unclear scope | Wrong campaigns affected | Specify market and period |

### OMP Response

On receiving this handoff, omp-operator MUST:
1. Pause affected campaigns immediately (fail closed)
2. Log the pause with reason
3. Wait for data-engineer resolution
4. Only resume after new snapshot_id confirmed

---

## H6: omp-operator -> risk-analyst

**When:** Candidate passed automated gates, ready for HoF promotion review.

### Entry Minimum (Required Fields)

| Field | Description | Example |
|-------|-------------|---------|
| `candidate_id` | Strategy identifier | `cand_abc123` |
| `run_id` | Source run | `a1b2c3d4-...` |
| `campaign_id` | Source campaign | `camp_xyz789` |
| `genome_hash` | Cryptographic hash | `sha256:...` |
| `config_hash` | Config hash | `sha256:...` |
| `git_sha` | Code version | `abc123...` |
| `snapshot_id` | Data version | `uuid-...` |

### Metrics Summary

| Metric | Value | Threshold |
|--------|-------|-----------|
| OOS Sharpe NET | {value} | >= 0.5 |
| PBO | {value} | <= 0.20 |
| DSR | {value} | >= 0.4 |
| Max DD | {value} | <= 30% |
| Variance sanity | PASS | sharpeVar > 1e-6 |

### Artifacts Required

| Artifact | Location | Format |
|----------|----------|--------|
| Promotion Packet Checklist | inline or file | Markdown template |
| Strategy artifacts | `artifacts/hall_of_fame/{candidate_id}/` | Various |

### Acceptance Criteria

| Criterion | Pass | Fail |
|-----------|------|------|
| All provenance fields present | Yes | Any missing |
| Variance sanity passed | Yes | Blocked |
| Automated thresholds met | Yes | Any fail |
| Artifacts copied | Yes | Missing files |

### Common Failures

| Failure | Symptom | Resolution |
|---------|---------|------------|
| Missing genome_hash | Cannot dedupe | Regenerate hash |
| Variance gate failed | Metrics collapsed | Investigate run |
| No trader-expert review | High turnover unreviewed | Handoff to trader-expert first |

### Risk-Analyst Response

On receiving this handoff, risk-analyst MUST:
1. Verify promotion packet completeness
2. Check if turnover requires trader-expert review
3. Apply tier-appropriate thresholds (not just Stage B)
4. Issue Promotion Memo or rejection

---

## H7: devops-infra <-> omp-operator

**When:** Resource pressure, infrastructure issues, or incident response.

### Direction: omp-operator -> devops-infra

**Trigger:** Resource limits exceeded, watchdog fired, service unhealthy.

| Field | Description | Example |
|-------|-------------|---------|
| `issue_type` | resource / service / incident | `resource` |
| `cpu_usage` | Current CPU % | `92%` |
| `memory_usage` | Current RAM % | `85%` |
| `disk_free_gb` | Free disk | `0.8 GB` |
| `omp_status` | running / paused / offline | `paused` |
| `pm2_status` | Service states | `alpha-api: online` |

**Artifacts:** Health check output (`health-check.sh --json`)

### Direction: devops-infra -> omp-operator

**Trigger:** Cleanup completed, resources restored, service recovered.

| Field | Description | Example |
|-------|-------------|---------|
| `action_taken` | What was done | `cleanup_old_runs.sh executed` |
| `disk_free_gb` | New free disk | `8.5 GB` |
| `services_status` | Current state | `all online` |
| `safe_to_resume` | Yes / No | `Yes` |

**Artifacts:** Ops Change Log entry

### Acceptance Criteria

| Criterion | Pass | Fail |
|-----------|------|------|
| Metrics provided | Yes | Vague description |
| Health check run | Yes | No evidence |
| Action documented | Yes | Silent fix |

---

## H8: quant-engineer <-> any

**When:** Performance issue or instrumentation request.

### Direction: any -> quant-engineer

**Trigger:** Bottleneck identified, new metric needed, optimization request.

| Field | Description | Example |
|-------|-------------|---------|
| `request_type` | optimization / instrumentation / investigation | `optimization` |
| `bottleneck` | What is slow | `process_day() taking 50us` |
| `target` | Goal | `< 20us` |
| `priority` | high / medium / low | `high` |
| `files_involved` | Relevant paths | `crates/backtester_engine/src/unified.rs` |

**Artifacts:** Profiling data if available (flamegraph, benchmark)

### Direction: quant-engineer -> requester

**Trigger:** Optimization complete, instrumentation added.

| Field | Description | Example |
|-------|-------------|---------|
| `change_summary` | What was done | `Replaced HashMap with Vec` |
| `improvement` | Measured delta | `-47% latency` |
| `files_modified` | Changed paths | List |
| `validation_needed` | What to check | `risk constraints unchanged` |

**Artifacts:** Benchmark Report with before/after

### Acceptance Criteria

| Criterion | Pass | Fail |
|-----------|------|------|
| Benchmark before/after | Yes | No baseline |
| Regression <= 5% | Yes | Worse performance |
| Hot path allocs = 0 | Yes | New allocations |
| Golden tests pass | Yes | Behavior changed |

---

## Handoff Template

Use this template when creating a handoff:

```markdown
## Handoff: {source_skill} -> {target_skill}

**Date:** YYYY-MM-DD
**Type:** {H1-H8}

### Context
{Brief description of what triggered this handoff}

### Required Fields
| Field | Value |
|-------|-------|
| {field} | {value} |

### Artifacts
- {artifact_1}: {location}
- {artifact_2}: {location}

### Request
{What the target skill needs to do}

### Priority
{high / medium / low}
```

---

## Promotion Pipeline (Closed)

No strategy can reach Hall of Fame without passing through this sequence:

```
quant-researcher                    (produces candidates)
        |
        | H1: Handoff Packet
        v
   risk-analyst                     (validates OOS)
        |
        | H2: if turnover > 12x
        v
  trader-expert                     (reviews execution)
        |
        | H3: Execution Assumptions Card
        v
   risk-analyst                     (stress + final gates)
        |
        | data-engineer must have PASS for snapshot_id
        v
   omp-operator                     (automates Stage B)
        |
        | H6: Promotion Packet
        v
   risk-analyst                     (final HoF approval)
        |
        v
   Hall of Fame                     (permanent record)
```

**No bypass allowed.** Every promotion must have:
- data-engineer: Data Quality Report PASS
- trader-expert: Execution Assumptions Card (if turnover > 12x)
- risk-analyst: Validation Report PASS + Promotion Memo
- omp-operator: Promotion Packet complete
