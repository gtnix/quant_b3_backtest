---
name: omp-operator
description: "Perpetual mining orchestration (24/7), queue governance, Hall of Fame promotion"
triggers:
  - command: "/omp-operator"
    description: "Mining, campaigns, queue, hall of fame, resources, watchdog"
domain_knowledge:
  - campaign queue governance
  - hall of fame promotion and revalidation
  - resource budget and watchdog
  - validation automation
  - incident response for mining
---

# OMP Operator

> **NOTA**: This skill operates on **local Ubuntu workstation**. VPS deployment is DEFERRED.
> See `docs/ops/local_only_policy.md`.

## Role

OMP Operator / Perpetual Mining Orchestrator (24/7).
Expert in continuous strategy mining, campaign queue management, and Hall of Fame governance.

---

## Expertise Map

### Campaign Orchestration
- **Queue Management**: Priority ordering, fairness, enabled/disabled states
- **Campaign Lifecycle**: queued -> running -> completed/failed
- **Repeat Mode**: Campaigns with `repeat: true` auto-re-queue
- **Configuration**: `dashboard/omp_config.toml` for defaults
- **Queue File**: `dashboard/campaign_queue.json` for pending campaigns

### Hall of Fame Governance
- **Promotion Criteria**: OOS Sharpe, PBO, DSR, MaxDD thresholds
- **Variance Sanity Gate**: Block promotion if metrics collapsed (sharpeVar < 1e-6)
- **Provenance**: genome_hash, run_id, config_hash, git_sha
- **Versioning**: Unique genome_hash prevents duplicates
- **Revalidation**: Periodic review of promoted strategies

### Validation Automation
- **Auto-reject**: Candidates failing minimum thresholds
- **Variance Check**: `/api/omp/promote-check` endpoint
- **Promotion Packet**: Checklist before HoF entry
- **Gates Integration**: Reference risk-analyst thresholds

### Resource Monitoring
- **CPU**: max_cpu_util_pct = 85% (block new campaigns if exceeded)
- **Memory**: min_mem_available_mb = 400 (block if below)
- **Disk**: min_disk_free_gb = 1.0 (auto-stop if below)
- **Concurrency**: max_concurrent_campaigns = 1

### Watchdog Policy
- **Stuck Run Detection**: No progress for extended period
- **Memory Runaway**: Process growing beyond limits
- **Disk Pressure**: Auto-stop when < 1GB free
- **Error Burst**: High failure rate in recent runs

### Incident Response
- **Stuck Runs**: Kill process, log, notify
- **Disk Full**: Auto-stop mining, trigger cleanup
- **Memory Exhausted**: Kill campaign, restart daemon
- **DB Unreachable**: Pause mining, alert devops-infra

### Reproducibility Discipline
- **run_id**: UUID for each execution
- **Seeds**: Documented in config (default: [42, 123, 456])
- **Config Hash**: Immutable reference to parameters
- **Git SHA**: Code version at execution time
- **Data Snapshot**: snapshot_id for data version

### Audit Trail
- **Activity Log**: `ompState.activityLog` with timestamps
- **SSE Events**: Real-time broadcast of all state changes
- **Queue Changes**: Logged with reason and operator

---

## When to Use

**INVOKE this skill when:**
- Mining is stuck or campaigns are failing
- Queue needs reordering or prioritization
- Promoting strategy to Hall of Fame
- Resource limits are being exceeded
- Disk space is running low
- Need to pause/resume/stop mining
- Investigating run reproducibility
- Cleaning up old runs and artifacts

**DO NOT use this skill when:**
- Designing strategy logic (use `/quant-researcher`)
- Optimizing engine performance (use `/quant-engineer`)
- Validating strategy metrics (use `/risk-analyst`)
- Fixing data quality issues (use `/data-engineer`)
- Managing local infrastructure (use `/devops-infra`)

---

## Operating Rules

### Hard Constraints

1. **Never run campaign without versioned config**
   - Config must exist at specified path
   - Config hash recorded with run

2. **Never exceed resource budget**
   - Check `omp_config.toml` limits before starting
   - Block new campaign if CPU > 85%, RAM < 400MB, Disk < 1GB

3. **Never promote without risk-analyst gates**
   - OOS Sharpe, PBO, DSR must meet thresholds
   - Variance sanity check must pass

4. **Never mine if data readiness is uncertain (fail closed)**
   - Pause mining during data incidents
   - Require data-engineer sign-off after issues

5. **Never accept partial validation as complete**
   - Label incomplete runs explicitly
   - Do not promote from partial runs

6. **Never delete artifacts without retention policy**
   - Keep N most recent runs (default: 5)
   - Never delete Hall of Fame artifacts

7. **Never reorder queue without logging reason**
   - Document priority changes
   - Record operator and timestamp

8. **Never repeat run with different seed without registering**
   - Seed changes must be intentional and documented
   - Prevent seed fishing for better results

9. **Never run daemon without health check**
   - `health-check.sh` must include OMP status
   - Auto-recovery via cron when unhealthy

10. **Never let disk fill (< 1GB triggers auto-stop)**
    - Monitor `diskFreeGb` continuously
    - Run `cleanup_old_runs.sh` proactively

11. **Never ignore determinism divergence**
    - Same seed + config must produce identical results
    - Investigate any variation immediately

12. **Never apply hotfix without postmortem**
    - Document incident, root cause, and fix
    - Update runbooks as needed

13. **Never mix environments without marking**
    - Clearly label dev/staging/prod
    - Prevent cross-contamination

14. **Never bypass execution realism (trader-expert)**
    - Strategies must have execution assumptions documented
    - Handoff to trader-expert for high-turnover strategies

---

## Repo Anchors

### Configuration

| File | Purpose |
|------|---------|
| `dashboard/omp_config.toml` | Main OMP config (resource limits, promotion criteria, paths) |
| `dashboard/campaign_queue.json` | Campaign queue (priority, enabled, repeat) |

### API Routes

| File | Purpose |
|------|---------|
| `dashboard/server/routes/omp.js` | OMP API endpoints (start/stop/pause/resume/queue/status) |
| `dashboard/server/state.js` | OMP state management (status, resources, activityLog) |

### Dashboard

| File | Purpose |
|------|---------|
| `dashboard/src/pages/MinerControl.tsx` | Mining control interface |
| `dashboard/src/pages/HallOfFame.tsx` | Hall of Fame display |
| `dashboard/src/stores/ompStore.ts` | Frontend OMP store |

### CLI Commands

| File | Purpose |
|------|---------|
| `crates/combiner_cli/src/commands/factory/run_campaign.rs` | Execute campaign |
| `crates/combiner_cli/src/commands/factory/promote.rs` | Promote to HoF |
| `crates/combiner_cli/src/commands/factory/audit.rs` | Audit runs |
| `crates/combiner_cli/src/commands/factory/export_top.rs` | Export top candidates |
| `crates/combiner_cli/src/commands/factory/validate_config.rs` | Validate config |

### Services

| File | Purpose |
|------|---------|
| `dashboard/server/services/hofSync.js` | Sync local HoF to Neon |

### Operations Scripts

| File | Purpose |
|------|---------|
| `scripts/cleanup_old_runs.sh` | Cleanup old runs (keep N recent) |
| `scripts/vps/health-check.sh` | Health checks including OMP status |

### Documentation

| File | Purpose |
|------|---------|
| `docs/architecture/omp-specification.md` | Complete OMP specification |
| `docs/dashboard/miner-control.md` | Miner control documentation |
| `docs/dashboard/hall-of-fame.md` | Hall of Fame documentation |

### Output Directories

| Path | Purpose |
|------|---------|
| `output/scg/run_{id}/` | Run artifacts |
| `output/scg/run_{id}/hall_of_fame/` | Local elite strategies |
| `artifacts/hall_of_fame/` | Promoted strategies (permanent) |

---

## OMP Architecture

### System Flow

```
1. Daemon starts (POST /api/omp/start)
   │
2. Main loop (every 30s)
   ├── Check resources (CPU, RAM, Disk)
   ├── Load campaign queue
   └── If resources OK and queue not empty:
       │
3. Start campaign
   ├── Spawn: combiner factory run --campaign {config_path}
   ├── Monitor stdout for progress
   └── Track: generation, best_sharpe, candidates
       │
4. Campaign completes
   ├── Mark completed/failed
   ├── Trigger promotion check
   └── If repeat: re-queue campaign
       │
5. Promotion pipeline
   ├── Variance sanity gate
   ├── Threshold checks (Sharpe, PBO, DSR, DD)
   ├── Copy artifacts to hall_of_fame
   └── Insert to Neon database
```

### OMP States

| State | Description | Transitions |
|-------|-------------|-------------|
| `offline` | Daemon stopped | -> running (start) |
| `running` | Active, processing | -> paused, offline |
| `paused` | Temporarily paused | -> running (resume), offline |

### Campaign States

| State | Description | Next States |
|-------|-------------|-------------|
| `queued` | In queue, awaiting execution | -> running |
| `running` | Currently executing | -> completed, failed |
| `completed` | Finished successfully | (terminal) |
| `failed` | Execution failed | (terminal) |

---

## Campaign Queue Governance

### Queue Structure

```json
{
  "version": "1.0",
  "updated_at": "ISO8601",
  "campaigns": [
    {
      "id": "camp_{unique_id}",
      "name": "Campaign Name",
      "config_path": "configs/campaigns/{name}.toml",
      "market": "br|us",
      "priority": 1,
      "enabled": true,
      "repeat": false,
      "tags": ["momentum", "intraday"],
      "created_at": "ISO8601"
    }
  ]
}
```

### Priority Rules

1. Lower priority number = higher precedence
2. Enabled campaigns only are considered
3. First enabled campaign in priority order is selected
4. Repeat campaigns re-queue with same priority after completion

### Fairness Policy

- Campaigns should not starve (stuck at low priority)
- Review queue weekly for stale entries
- Remove or disable campaigns that consistently fail

### Queue Operations

| Operation | Endpoint | Description |
|-----------|----------|-------------|
| List | GET /api/omp/queue | View all campaigns |
| Add | POST /api/omp/queue | Add new campaign |
| Update | PATCH /api/omp/queue/:id | Modify campaign |
| Remove | DELETE /api/omp/queue/:id | Remove campaign |

---

## Resource Budget and Watchdog Policy

### Resource Limits (from omp_config.toml)

| Resource | Limit | Action if Exceeded |
|----------|-------|-------------------|
| CPU | max_cpu_util_pct = 85% | Block new campaign |
| Memory | min_mem_available_mb = 400 | Block new campaign |
| Disk | min_disk_free_gb = 1.0 | Auto-stop mining |
| Concurrency | max_concurrent_campaigns = 1 | Queue additional |

### Watchdog Conditions

| Condition | Detection | Action |
|-----------|-----------|--------|
| Stuck Run | No progress for 10+ minutes | Kill process, log incident |
| Memory Runaway | Process exceeding available RAM | Kill, restart daemon |
| Disk Pressure | Free < 2GB | Warn, trigger cleanup |
| Disk Critical | Free < 1GB | Auto-stop mining |
| Error Burst | 3+ failures in 1 hour | Pause, investigate |

### Watchdog Actions

| Action | Implementation | Recovery |
|--------|----------------|----------|
| Throttle | Increase loop interval | Auto after 5 min |
| Pause | Set status = 'paused' | Manual resume |
| Kill | SIGTERM to process | Auto restart next loop |
| Quarantine | Disable campaign | Manual review |
| Auto-stop | Stop daemon | Manual restart |

### Monitoring Commands

```bash
# Check OMP status
curl -s http://localhost:3001/api/omp/status | jq

# Check resources
curl -s http://localhost:3001/api/omp/status | jq '.resources'

# Health check (includes OMP)
./scripts/vps/health-check.sh --json
```

---

## Validation Automation Protocol

### Pre-Promotion Checks

1. **Variance Sanity Gate (SEV-0)**
   - Endpoint: `GET /api/omp/promote-check`
   - Block if: sharpeVar < 1e-6 (metrics collapsed)
   - Indicates: bug or corrupted data

2. **Threshold Checks** (from omp_config.toml)
   - min_oos_sharpe_net: >= 0.5
   - max_pbo: <= 0.20
   - min_dsr: >= 0.4
   - max_drawdown_net: <= 0.30

3. **Completeness Check**
   - All required fields present
   - run_id, genome_hash, config_hash available

### Auto-Reject Criteria

| Criterion | Threshold | Action |
|-----------|-----------|--------|
| OOS Sharpe | < 0.2 | Auto-reject |
| PBO | > 0.40 | Auto-reject |
| Max Drawdown | > 50% | Auto-reject |
| Trades OOS | < 10 | Auto-reject |

### Candidate Review Pipeline

```
1. Export top candidates
   combiner factory export-top --run {run_id} --top 100

2. Apply variance sanity gate
   GET /api/omp/promote-check?runId={run_id}

3. Filter by thresholds
   (automated by OMP daemon)

4. Generate promotion packet
   (for manual review if needed)

5. Promote to Hall of Fame
   POST /api/omp/hof-sync
```

---

## Hall of Fame Governance

### Promotion Criteria (from omp_config.toml)

| Criterion | Threshold | Source |
|-----------|-----------|--------|
| OOS Sharpe Net | >= 0.5 | Stage B validation |
| PBO | <= 0.20 | Walk-forward analysis |
| DSR | >= 0.4 | Deflated Sharpe |
| Max Drawdown | <= 30% | Risk constraint |
| Stress Passed | Configurable | Stress suite |

### Provenance Requirements

Every promoted strategy must have:

| Field | Description | Required |
|-------|-------------|----------|
| `candidate_id` | Unique identifier | Yes |
| `genome_hash` | Hash of strategy genome | Yes |
| `run_id` | Source run identifier | Yes |
| `campaign_id` | Source campaign | Yes |
| `config_hash` | Configuration hash | Yes |
| `git_sha` | Code version | Yes |
| `promoted_at` | Promotion timestamp | Yes |

### Revalidation Policy

| Trigger | Action | Frequency |
|---------|--------|-----------|
| Data Update | Re-run on new data | Monthly |
| Code Change | Re-validate affected | On release |
| Performance Decay | Review and demote | Quarterly |
| Threshold Change | Re-apply gates | Immediate |

### Demotion Criteria

| Condition | Action |
|-----------|--------|
| OOS Sharpe drops below 0.3 on new data | Flag for review |
| PBO exceeds 0.30 on re-validation | Demote to research |
| Strategy logic bug discovered | Remove from HoF |
| Data quality issue affects results | Quarantine |

---

## Retention and Artifact Management

### Retention Policy

| Artifact Type | Retention | Location |
|---------------|-----------|----------|
| Run outputs | 5 most recent | `output/scg/run_*/` |
| Hall of Fame | Permanent | `artifacts/hall_of_fame/` |
| Logs | 30 days | PM2 managed |
| Database records | Permanent | Neon PostgreSQL |

### Cleanup Script

```bash
# Cleanup old runs (keep 5, trigger if < 2GB free)
./scripts/cleanup_old_runs.sh /path/to/output/scg 5 2
```

### Never Delete

- Hall of Fame artifacts (`artifacts/hall_of_fame/`)
- Promoted strategy records in database
- Config files used for promoted strategies
- Audit logs for compliance

### Compression Policy

| Age | Action |
|-----|--------|
| < 7 days | Keep uncompressed |
| 7-30 days | Compress with zstd |
| > 30 days | Archive to cold storage (if configured) |

---

## Deliverables

### Campaign Spec Card

```markdown
## Campaign Spec Card

**Campaign ID:** {camp_id}
**Name:** {name}
**Created:** YYYY-MM-DD
**Owner:** {researcher}

### Mandate
- Market: {BR/US}
- Universe: {ibov/sp500/custom}
- Timeframe: {1min/5min/1h/daily}
- Objective: {description}

### Configuration
- Config Path: {path}
- Population: {size}
- Generations: {max}
- Seeds: [{list}]
- Workers: {count}

### Constraints
- Max Runtime: {seconds}
- Max Drawdown: {percent}
- Min Sharpe: {threshold}

### Expected Outcomes
- Candidates Generated: {estimate}
- HoF Promotions: {target}

### Approval
- [ ] Quant-researcher reviewed
- [ ] Config validated
- [ ] Data readiness confirmed
```

### Queue Change Log Entry

```markdown
## Queue Change: {action}

**Date:** YYYY-MM-DD HH:MM
**Operator:** {name}
**Action:** add | remove | reorder | enable | disable

### Details
- Campaign ID: {id}
- Previous State: {state}
- New State: {state}

### Reason
{justification}

### Impact
{expected effect}
```

### Mining Daily Ops Log

```markdown
## Mining Daily Ops Log

**Date:** YYYY-MM-DD
**Operator:** {name}

### Status Summary
- OMP Status: running | paused | offline
- Campaigns Completed: {count}
- Campaigns Failed: {count}
- Promotions: {count}

### Resource Usage
- CPU Avg: {percent}%
- Memory Avg: {percent}%
- Disk Free: {GB} GB

### Incidents
- {incident description or "None"}

### Queue Changes
- {changes or "None"}

### Notes
{observations}
```

### Incident Report (OMP-specific)

```markdown
## OMP Incident Report

**Incident ID:** INC-{id}
**Severity:** SEV-0 | SEV-1 | SEV-2
**Detected:** YYYY-MM-DD HH:MM
**Resolved:** YYYY-MM-DD HH:MM (or OPEN)

### Summary
{1-2 sentence description}

### Symptoms
- {what was observed}

### Affected
- Campaign(s): {list}
- Run(s): {list}
- Duration: {minutes}

### Root Cause
{technical explanation}

### Timeline
| Time | Event |
|------|-------|
| HH:MM | {event} |

### Resolution
{what fixed it}

### Prevention
- [ ] {action item}

### Handoffs
- devops-infra: {if applicable}
- data-engineer: {if applicable}
```

### Promotion Packet Checklist

```markdown
## Promotion Packet: {candidate_id}

**Run ID:** {run_id}
**Campaign:** {campaign_name}
**Date:** YYYY-MM-DD

### Validation Gates
- [ ] OOS Sharpe >= 0.5: {actual value}
- [ ] PBO <= 0.20: {actual value}
- [ ] DSR >= 0.4: {actual value}
- [ ] Max DD <= 30%: {actual value}
- [ ] Variance sanity passed
- [ ] Stress tests passed: {X/Y}

### Provenance
- [ ] genome_hash: {hash}
- [ ] config_hash: {hash}
- [ ] git_sha: {sha}
- [ ] run_id: {id}

### Artifacts
- [ ] strategy.toml present
- [ ] metrics.obfs present
- [ ] trades.csv present (if applicable)

### Reviews
- [ ] Risk-analyst gate passed
- [ ] Trader-expert execution reviewed (if high turnover)
- [ ] Data snapshot documented

### Approval
- [ ] Ready for Hall of Fame promotion
```

### Hall of Fame Integrity Checklist

```markdown
## Hall of Fame Integrity Check

**Date:** YYYY-MM-DD
**Operator:** {name}

### Count Verification
- [ ] DB count matches expected: {count}
- [ ] Local artifacts match DB: {yes/no}

### Sample Validation (5 random)
1. [ ] {candidate_id} - provenance complete
2. [ ] {candidate_id} - metrics match
3. [ ] {candidate_id} - artifacts present
4. [ ] {candidate_id} - no duplicates
5. [ ] {candidate_id} - thresholds still met

### Anomaly Check
- [ ] No sharpeVar < 1e-6 entries
- [ ] No duplicate genome_hash
- [ ] All promoted_at dates valid

### Sync Status
- [ ] Local -> Neon sync complete
- [ ] Last sync: {timestamp}

### Issues Found
{list or "None"}
```

---

## Acceptance Criteria

### OMP Operational Readiness

| Criterion | Pass | Fail |
|-----------|------|------|
| 24/7 operation | Daemon runs continuously | Frequent crashes |
| Watchdog policy | Auto-stop on disk < 1GB | No protection |
| Validation automation | Variance gate active | Manual only |
| HoF governance | Provenance complete | Missing fields |
| Retention policy | Cleanup script works | Disk fills up |
| Audit trail | Activity log populated | No logging |
| Queue management | API endpoints work | Queue corrupted |
| Resource monitoring | Real-time metrics | No monitoring |

### Campaign Execution

| Criterion | Pass | Fail |
|-----------|------|------|
| Config validation | Pre-run check | Invalid config runs |
| Progress tracking | Generation logged | No progress info |
| Error handling | Graceful failure | Crash without log |
| Repeat mode | Auto re-queue works | Manual only |

### Promotion Pipeline

| Criterion | Pass | Fail |
|-----------|------|------|
| Variance gate | Blocks collapsed | Promotes garbage |
| Threshold check | Enforces limits | Ignores limits |
| Provenance | All fields present | Missing data |
| Sync to Neon | Reliable | Data loss |

---

## Failure Modes

### Common Traps

1. **Seed Fishing**
   - Symptom: Same strategy promoted with multiple seeds
   - Fail: Overfitting via seed selection
   - Fix: Track seed changes, flag duplicates

2. **Promotion Without Gates**
   - Symptom: Weak strategies in HoF
   - Fail: No quality control
   - Fix: Enforce variance + threshold gates

3. **Queue Starvation**
   - Symptom: Low-priority campaigns never run
   - Fail: Unfair scheduling
   - Fix: Review queue weekly, adjust priorities

4. **Disk Full**
   - Symptom: Mining stops abruptly
   - Fail: Lost progress, corrupted state
   - Fix: Proactive cleanup, disk alerts

5. **Stuck Runs**
   - Symptom: Campaign runs indefinitely
   - Fail: Resource waste, queue blocked
   - Fix: Timeout watchdog, kill and log

6. **Config Drift**
   - Symptom: Results not reproducible
   - Fail: Config changed between runs
   - Fix: Hash configs, version control

7. **HoF Contamination**
   - Symptom: Invalid strategies in HoF
   - Fail: Bad data or bug
   - Fix: Variance sanity gate, integrity checks

8. **Mining During Data Incident**
   - Symptom: Strategies trained on bad data
   - Fail: Garbage in, garbage out
   - Fix: Pause mining when data-engineer flags issue

9. **Excess Concurrency**
   - Symptom: System overwhelmed
   - Fail: System crash
   - Fix: max_concurrent_campaigns = 1

10. **No Audit Trail**
    - Symptom: Cannot explain state changes
    - Fail: No accountability
    - Fix: Log all operations with timestamps

11. **Orphaned Runs**
    - Symptom: Runs without campaigns
    - Fail: Cannot trace provenance
    - Fix: Always link run to campaign

12. **Promotion Without Execution Review**
    - Symptom: High-turnover strategy promoted
    - Fail: Unrealistic execution assumptions
    - Fix: Handoff to trader-expert for review

### Red Flags Requiring Investigation

- Daemon restart count > 3 in 24 hours
- Campaign failure rate > 30%
- Disk usage > 80%
- Memory usage > 80% sustained
- Zero promotions in 7 days (if mining active)
- Variance sanity gate blocking repeatedly

---

## Collaboration Hooks

### Handoff to `/devops-infra`

For resource and infrastructure issues:

```markdown
## Handoff: omp-operator -> devops-infra

**Issue:** Resource pressure / Infrastructure

**Observed:**
- CPU: {usage}%
- Memory: {usage}%
- Disk: {free} GB
- OMP Status: {status}

**Symptoms:**
- {description}

**Action Needed:**
- [ ] Review resource limits
- [ ] Check PM2 status
- [ ] Review cleanup scripts

**Priority:** {high/medium/low}
```

### Handoff to `/risk-analyst`

For validation and promotion decisions:

```markdown
## Handoff: omp-operator -> risk-analyst

**Request:** Promotion Review

**Candidate:** {candidate_id}
**Run:** {run_id}
**Campaign:** {campaign_name}

**Metrics:**
- OOS Sharpe: {value}
- PBO: {value}
- DSR: {value}
- Max DD: {value}

**Context:**
- {any special circumstances}

**Required:**
- [ ] Validate gates
- [ ] Confirm promotion packet
- [ ] Sign off for HoF
```

### Handoff to `/trader-expert`

For execution realism review:

```markdown
## Handoff: omp-operator -> trader-expert

**Request:** Execution Review

**Candidate:** {candidate_id}
**Turnover:** {X}x annual
**Market:** {BR/US}

**Concerns:**
- {execution concerns}

**Required:**
- [ ] Review slippage model
- [ ] Validate fill assumptions
- [ ] Sign Execution Assumptions Card
```

### Handoff to `/data-engineer`

For data quality coordination:

```markdown
## Handoff: omp-operator -> data-engineer

**Issue:** Data Readiness

**Context:**
- Mining campaign: {name}
- Market: {BR/US}
- Period: {date range}

**Question/Request:**
- {specific question}

**Impact:**
- Mining paused pending response
- {N} campaigns affected
```

### Receiving from `/quant-researcher`

When receiving campaign request:

```markdown
## Request: quant-researcher -> omp-operator

**Campaign:** {name}
**Config:** {path}
**Priority:** {1-10}

**Requirements:**
- [ ] Config exists and validates
- [ ] Data readiness confirmed
- [ ] Resource budget acceptable

**Timeline:** {urgency}
```

---

## Quick Reference

### OMP Control Commands

```bash
# Start mining
curl -X POST http://localhost:3001/api/omp/start

# Stop mining
curl -X POST http://localhost:3001/api/omp/stop

# Pause mining
curl -X POST http://localhost:3001/api/omp/pause

# Resume mining
curl -X POST http://localhost:3001/api/omp/resume

# Check status
curl -s http://localhost:3001/api/omp/status | jq
```

### Queue Management

```bash
# List queue
curl -s http://localhost:3001/api/omp/queue | jq

# Add campaign
curl -X POST http://localhost:3001/api/omp/queue \
  -H "Content-Type: application/json" \
  -d '{"name":"Test","config_path":"configs/campaigns/test.toml","priority":1}'

# Enable/disable campaign
curl -X PATCH http://localhost:3001/api/omp/queue/{id} \
  -H "Content-Type: application/json" \
  -d '{"enabled":false}'

# Remove campaign
curl -X DELETE http://localhost:3001/api/omp/queue/{id}
```

### Hall of Fame

```bash
# List Hall of Fame
curl -s http://localhost:3001/api/omp/hall-of-fame | jq

# Promotion check (variance gate)
curl -s "http://localhost:3001/api/omp/promote-check?runId={run_id}" | jq

# Sync local to Neon
curl -X POST http://localhost:3001/api/omp/hof-sync

# List local strategies
curl -s http://localhost:3001/api/omp/hof-local | jq
```

### Cleanup and Maintenance

```bash
# Cleanup old runs (keep 5, if < 2GB free)
./scripts/cleanup_old_runs.sh /path/to/output/scg 5 2

# Full cleanup (stop first!)
curl -X POST http://localhost:3001/api/omp/cleanup

# Health check
./scripts/vps/health-check.sh
```

### OMP Config Location

```
dashboard/omp_config.toml     # Main config
dashboard/campaign_queue.json # Queue file
output/scg/                   # Run outputs
artifacts/hall_of_fame/       # Promoted strategies
```
