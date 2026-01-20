# Runbook: Hall of Fame Promotion

Operational runbook for promoting strategies to the Hall of Fame with full audit trail.

---

## Definition of Promotion

**Promotion** means a strategy candidate has:

1. **Passed all validation gates** (WFA, PBO, DSR, stress tests)
2. **Passed execution realism review** (costs, capacity, sessions)
3. **Complete provenance** (run_id, genome_hash, config_hash, git_sha, snapshot_id)
4. **All required artifacts** stored permanently
5. **Formal approval** via Promotion Memo from risk-analyst
6. **Entry in Hall of Fame** with permanent retention

A promoted strategy is considered **production-ready** and may be deployed for live trading (subject to additional operational approvals outside this system).

---

## Inputs Minimum (per _HANDOFF_CONTRACTS.md H6)

| Field | Description | Required |
|-------|-------------|----------|
| `candidate_id` | Unique strategy identifier | Yes |
| `genome_hash` | Cryptographic hash of genome | Yes |
| `run_id` | Source run UUID | Yes |
| `campaign_id` | Source campaign | Yes |
| `config_hash` | Hash of frozen config | Yes |
| `git_sha` | Code version at execution | Yes |
| `snapshot_id` | Data version used | Yes |
| `promoted_at` | Timestamp of promotion | Yes |

---

## Pre-Promotion Verification Sequence

### 1. Data Engineer Checks (5 items)

**Owner:** `/data-engineer`

```
[ ] 1.1 Data Quality Report exists with PASS status
[ ] 1.2 snapshot_id documented and matches run metadata
[ ] 1.3 OHLCV invariants verified for backtest period
[ ] 1.4 Corporate actions (dividends, splits) correctly applied
[ ] 1.5 Universe is point-in-time (no survivorship bias)
```

**Evidence:** Data Quality Report with snapshot_id

---

### 2. Risk Analyst Checks (10 items)

**Owner:** `/risk-analyst`

```
[ ] 2.1 run_id verified and artifacts exist
[ ] 2.2 config.toml frozen and config_hash computed
[ ] 2.3 WFA completed with >= 5 folds
[ ] 2.4 Purge/embargo applied (5 days minimum each)
[ ] 2.5 OOS Sharpe NET >= threshold (0.5 research / 1.0 production)
[ ] 2.6 PBO < threshold (0.20 research / 0.10 production)
[ ] 2.7 DSR >= threshold (0.5 research / 0.8 production)
[ ] 2.8 Max Drawdown <= threshold (35% research / 20% production)
[ ] 2.9 Stress tests passed (>= 3/5 research / >= 4/5 production)
[ ] 2.10 Determinism verified (3 identical runs)
```

**Evidence:** Validation Report

---

### 3. Trader Expert Checks (6 items)

**Owner:** `/trader-expert`

**Required if:** turnover > 12x annual OR shorts involved OR risk-analyst flags execution concerns

```
[ ] 3.1 Execution Assumptions Card complete
[ ] 3.2 Slippage model documented and justified
[ ] 3.3 3 cost scenarios tested (base, pessimist, stress)
[ ] 3.4 S1 (costs_2x) passed for intraday
[ ] 3.5 S5 (combined_adverse) passed for position
[ ] 3.6 Capacity documented (>= $5M institutional)
```

**If shorts:**
```
[ ] 3.7 Borrow cost assumption documented
[ ] 3.8 Borrow availability limitation declared
```

**Evidence:** Execution Assumptions Card (signed)

---

### 4. Quant Engineer Checks (4 items)

**Owner:** `/quant-engineer`

**Required if:** New strategy type or performance-critical changes

```
[ ] 4.1 Benchmark comparison shows no regression > 5%
[ ] 4.2 Hot path allocations = 0
[ ] 4.3 Golden tests pass
[ ] 4.4 Determinism verified
```

**Evidence:** Benchmark Report

---

### 5. DevOps Infra Checks (4 items)

**Owner:** `/devops-infra`

```
[ ] 5.1 Current deployment healthy (./scripts/vps/health-check.sh)
[ ] 5.2 git_sha recorded and traceable
[ ] 5.3 Rollback plan documented and tested
[ ] 5.4 Artifacts storage accessible (artifacts/hall_of_fame/)
```

**Evidence:** Deployment Checklist, health-check.sh output

---

### 6. OMP Operator Checks (6 items)

**Owner:** `/omp-operator`

```
[ ] 6.1 Variance sanity gate passed (sharpeVar > 1e-6)
[ ] 6.2 Automated promotion thresholds met:
      - min_oos_sharpe_net >= 0.5
      - max_pbo <= 0.20
      - min_dsr >= 0.4
      - max_drawdown_net <= 0.30
[ ] 6.3 Provenance complete (all 7 fields)
[ ] 6.4 Artifacts copied to artifacts/hall_of_fame/{candidate_id}/
[ ] 6.5 No duplicate genome_hash in Hall of Fame
[ ] 6.6 Promotion Packet assembled and sent to risk-analyst
```

**Evidence:** Promotion Packet Checklist

---

## Promotion Memo Format

**Template from risk-analyst/SKILL.md:**

```markdown
## Promotion Memo: Strategy -> Hall of Fame

**Candidate ID:** {candidate_id}
**Submitted by:** {omp-operator}
**Reviewed by:** risk-analyst
**Date:** YYYY-MM-DD

### Executive Summary
{2-3 sentences on strategy edge and validation outcome}

### Validation Results
| Gate | Value | Threshold | Status |
|------|-------|-----------|--------|
| OOS Sharpe | X.XX | >= Y.Y | PASS |
| PBO | X.XX | < Y.Y | PASS |
| DSR | X.XX | >= Y.Y | PASS |
| Max DD | X.X% | <= Y% | PASS |
| Stress Pass | X/5 | >= Y/5 | PASS |

### Modality-Specific
- [ ] Intraday: S1 + S2 passed
- [ ] Position: S5 passed

### Audit Trail
- run_id: {uuid}
- genome_hash: {hash}
- config_hash: {hash}
- git_sha: {sha}
- snapshot_id: {uuid}
- Determinism: verified (3 runs)

### Recommendation
**APPROVED** for Hall of Fame promotion.

### Conditions (if any)
- {condition 1}
- {condition 2}

### Signatures
- [ ] Risk Analyst: ___________
- [ ] Trader Expert (if required): ___________
```

---

## Post-Promotion Actions

1. **Artifact Storage**
   - Location: `artifacts/hall_of_fame/{candidate_id}/`
   - Contents: strategy.toml, metrics.obfs, trades.csv, config.toml, promotion_memo.md
   - Retention: **Permanent** (never delete)

2. **Database Entry**
   - Table: `scg_hall_of_fame` (Neon PostgreSQL)
   - Sync: `POST /api/omp/hof-sync` or `dashboard/server/services/hofSync.js`

3. **Notification**
   - Log in OMP activity log
   - SSE event broadcast

4. **Audit Trail Update**
   - Promotion timestamp recorded
   - All artifacts checksummed

---

## Revalidation Policy

### When to Revalidate

| Trigger | Frequency | Action |
|---------|-----------|--------|
| New data available | Monthly | Re-run on extended period |
| Code change affecting strategy | On release | Re-validate affected strategies |
| Performance decay detected | Quarterly review | Flag for investigation |
| Threshold change | Immediate | Re-apply gates |
| Data quality issue | On incident | Quarantine and re-validate |

### Revalidation Process

1. **Identify scope**: Which strategies affected?
2. **Re-run validation**: Same gates, new data/code
3. **Compare results**: Metrics still meet thresholds?
4. **Document**: Revalidation Report with comparison
5. **Decision**: Confirm / Flag / Demote

### Revalidation Report Format

```markdown
## Revalidation Report

**Candidate ID:** {candidate_id}
**Original Promotion:** YYYY-MM-DD
**Revalidation Date:** YYYY-MM-DD
**Trigger:** {new data / code change / periodic / incident}

### Comparison

| Metric | Original | Current | Delta | Status |
|--------|----------|---------|-------|--------|
| OOS Sharpe | X.XX | Y.YY | +/-Z% | OK/FLAG |
| PBO | X.XX | Y.YY | +/-Z% | OK/FLAG |
| DSR | X.XX | Y.YY | +/-Z% | OK/FLAG |
| Max DD | X.X% | Y.Y% | +/-Z% | OK/FLAG |

### Data Period
- Original: {start} to {end}
- Current: {start} to {end}

### Verdict
[ ] CONFIRMED - Remains in Hall of Fame
[ ] FLAGGED - Requires investigation
[ ] DEMOTED - Removed from Hall of Fame

### Notes
{observations}
```

---

## Demotion Criteria

A strategy should be **demoted** (removed from Hall of Fame) when:

| Condition | Threshold | Action |
|-----------|-----------|--------|
| OOS Sharpe drops significantly on new data | < 0.3 | Flag for review |
| PBO exceeds threshold on revalidation | > 0.30 | Demote to research |
| Strategy logic bug discovered | Any | Remove immediately |
| Data quality issue affected original validation | Material impact | Quarantine |
| Execution assumptions invalidated | Cannot execute as designed | Demote |
| Duplicate detected (same genome_hash) | Exact duplicate | Remove newer entry |

### Demotion Process

1. **Document reason**: Clear explanation of why demotion is warranted
2. **Create Demotion Memo**: Similar to Promotion Memo but with rejection
3. **Update database**: Mark status as DEMOTED with timestamp
4. **Preserve artifacts**: Do NOT delete (audit trail)
5. **Notify**: Log in activity, SSE event

### Demotion Memo Format

```markdown
## Demotion Memo

**Candidate ID:** {candidate_id}
**Original Promotion:** YYYY-MM-DD
**Demotion Date:** YYYY-MM-DD
**Demoted by:** risk-analyst

### Reason
{Clear explanation}

### Evidence
- {metric that failed}
- {data or analysis supporting demotion}

### Artifacts
- Preserved at: artifacts/hall_of_fame/{candidate_id}/
- Status: DEMOTED (not deleted)

### Signature
- [ ] Risk Analyst: ___________
```

---

## Promotion Pipeline Diagram

```
                            START
                              |
                              v
                    +-------------------+
                    | data-engineer     |
                    | Data Quality PASS |
                    | snapshot_id       |
                    +-------------------+
                              |
                              v
                    +-------------------+
                    | quant-researcher  |
                    | Candidates ready  |
                    | run_id, config    |
                    +-------------------+
                              |
                              v
                    +-------------------+
                    | risk-analyst      |
                    | WFA/PBO/DSR       |
                    | Stress tests      |
                    +-------------------+
                              |
                    turnover > 12x?
                    or shorts?
                      /          \
                    YES           NO
                    /              \
                   v                |
         +------------------+       |
         | trader-expert    |       |
         | Execution Card   |       |
         +------------------+       |
                   \               /
                    \             /
                     v           v
                    +-------------------+
                    | risk-analyst      |
                    | Final validation  |
                    +-------------------+
                              |
                              v
                    +-------------------+
                    | omp-operator      |
                    | Promotion Packet  |
                    | Provenance check  |
                    +-------------------+
                              |
                              v
                    +-------------------+
                    | risk-analyst      |
                    | Promotion Memo    |
                    | APPROVED          |
                    +-------------------+
                              |
                              v
                    +-------------------+
                    | HALL OF FAME      |
                    | Permanent storage |
                    +-------------------+
```

---

## Quick Reference: Verification Summary

| Phase | Owner | Items | Evidence |
|-------|-------|-------|----------|
| Data | data-engineer | 5 | Data Quality Report |
| Validation | risk-analyst | 10 | Validation Report |
| Execution | trader-expert | 6-8 | Execution Assumptions Card |
| Performance | quant-engineer | 4 | Benchmark Report |
| Infra | devops-infra | 4 | Health Check |
| Mining | omp-operator | 6 | Promotion Packet |
| **Total** | | **35-37** | |

---

## Anchored Paths Summary

| Artifact | Path |
|----------|------|
| Run outputs | `output/scg/run_{id}/` |
| Run artifacts | `artifacts/runs/{run_id}/` |
| Hall of Fame (permanent) | `artifacts/hall_of_fame/{candidate_id}/` |
| OMP Config | `dashboard/omp_config.toml` |
| Campaign Queue | `dashboard/campaign_queue.json` |
| Health Check | `scripts/vps/health-check.sh` |
| Cleanup Script | `scripts/cleanup_old_runs.sh` |
| Deploy Script | `scripts/deploy.sh` |

---

## Related Documents

- [_TEAM_INDEX.md](_TEAM_INDEX.md) - Master index and Prop-Firm Ready checklist
- [_TEAM_PLAYBOOK.md](_TEAM_PLAYBOOK.md) - Operational workflow
- [_HANDOFF_CONTRACTS.md](_HANDOFF_CONTRACTS.md) - Formal handoff requirements (especially H6)
- [_TRAINING_INTRADAY.md](_TRAINING_INTRADAY.md) - Intraday lifecycle training
- [_TRAINING_POSITION.md](_TRAINING_POSITION.md) - Position lifecycle training
- [risk-analyst/SKILL.md](risk-analyst/SKILL.md) - Validation gates and Promotion Memo template
- [omp-operator/SKILL.md](omp-operator/SKILL.md) - HoF governance and Promotion Packet
