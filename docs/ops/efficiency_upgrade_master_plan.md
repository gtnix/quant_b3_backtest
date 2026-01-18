# Efficiency Upgrade Master Plan

**Generated**: 2026-01-18  
**Version**: 1.0.0  
**Status**: Active  
**Environment**: LOCAL Ubuntu 100%

---

## 1) Baseline (Day 0) - Local Ubuntu

### Hardware Discovery Commands

Run these commands to establish baseline metrics:

```bash
# CPU
lscpu | grep -E "Model name|CPU\(s\)|Thread|Core"

# RAM
free -h

# Disk
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT
df -h /home

# Filesystem
mount | grep "on /home"
```

### Baseline Metrics Template

| Metric | Value | Command | Status |
|--------|-------|---------|--------|
| CPU Model | UNKNOWN | `lscpu` | Measure at runtime |
| CPU Cores | UNKNOWN | `nproc` | Measure at runtime |
| RAM Total | UNKNOWN | `free -h` | Measure at runtime |
| Disk Free | UNKNOWN | `df -h` | Measure at runtime |
| Filesystem | UNKNOWN | `mount` | Measure at runtime |

### Key Directories to Monitor

| Directory | Purpose | Command |
|-----------|---------|---------|
| `output/scg/` | Run artifacts | `du -sh output/scg/` |
| `target/` | Build cache | `du -sh target/` |
| `artifacts/` | Permanent (HoF) | `du -sh artifacts/` |
| `.cache/market_data/` | Market data cache | `du -sh .cache/market_data/` |

### Throughput Baseline

| Metric | How to Measure |
|--------|----------------|
| Campaigns/hour | Count completed in 1 hour |
| Time per campaign | OMP logs, campaign duration |
| CPU utilization | `htop` during campaign |
| RAM peak usage | `htop` during campaign |

---

## 2) Upgrade Backlog by Waves

### Wave 0: Immediate (< 1 hour, zero risk)

| ID | Description | Category | Files Affected | Measurement | Rollback |
|----|-------------|----------|----------------|-------------|----------|
| W0-1 | Enable auto_cleanup.sh cron | Ops/Disk | crontab entry | `df -h` before/after 7 days | Remove cron entry |
| W0-2 | Set `log_level = "WARN"` in omp_config.toml | Ops | `dashboard/omp_config.toml` L10 | Log file size reduction | Revert to "INFO" |
| W0-3 | Increase disk threshold to 5GB | Ops | `scripts/cleanup_old_runs.sh` L8 | Disk never < 5GB | Change back to 2 |

**Implementation:**

```bash
# W0-1: Add to crontab
crontab -e
# Add: 0 * * * * /path/to/scripts/auto_cleanup.sh --runs --days 3

# W0-2: Edit omp_config.toml
# Change log_level = "INFO" to log_level = "WARN"

# W0-3: Edit cleanup_old_runs.sh line 8
# Change MIN_FREE_GB="${3:-2}" to MIN_FREE_GB="${3:-5}"
```

### Wave 1: Quick Wins (1-4 hours, low risk)

| ID | Description | Category | Files Affected | Measurement | Rollback |
|----|-------------|----------|----------------|-------------|----------|
| W1-1 | Reduce `keep_runs` from 5 to 3 | Disk | `scripts/cleanup_old_runs.sh` L7 | Disk savings ~40% | Change back to 5 |
| W1-2 | Add systemd timer for cleanup | Ops | New user systemd files | Timer runs hourly | Disable timer |
| W1-3 | Compress old runs with zstd | Disk | New script | ~60% space savings | Decompress |
| W1-4 | Add local log rotation | Ops | Dashboard server config | Log files < 100MB | Remove rotation |

**Implementation:**

```bash
# W1-1: Edit cleanup_old_runs.sh line 7
# Change KEEP_RUNS="${2:-5}" to KEEP_RUNS="${2:-3}"

# W1-2: Create systemd timer
mkdir -p ~/.config/systemd/user/
# Create quant-cleanup.service and quant-cleanup.timer

# W1-3: Compression script
# Create scripts/compress_old_runs.sh using zstd

# W1-4: Log rotation
# Configure max log file size in dashboard/server.js
```

### Wave 2: Config Tuning (2-8 hours, medium risk)

| ID | Description | Category | Files Affected | Measurement | Rollback |
|----|-------------|----------|----------------|-------------|----------|
| W2-1 | Tune `population_size` based on cores | Throughput | `dashboard/omp_config.toml` L52 | campaigns/hour improvement | Revert config |
| W2-2 | Tune `workers` to match CPU cores | Throughput | `dashboard/omp_config.toml` L66 | CPU utilization ~85% | Revert config |
| W2-3 | Reduce `max_generations` for faster iteration | Throughput | Campaign configs | Time per campaign | Revert config |
| W2-4 | Adjust resource_limits for local specs | Ops | `dashboard/omp_config.toml` L17-21 | Resource matching | Revert config |

**Implementation:**

```bash
# W2-1 & W2-2: Determine optimal values
nproc  # Get core count
# Set workers = nproc, population_size = 100-200

# W2-3: Edit campaign configs
# Reduce max_generations from 50 to 30 for faster iteration

# W2-4: Adjust resource limits based on local hardware
# min_mem_available_mb = based on actual RAM
# max_cpu_util_pct = 85% is good default
```

### Wave 3: Observability (4-16 hours, low risk)

| ID | Description | Category | Files Affected | Measurement | Rollback |
|----|-------------|----------|----------------|-------------|----------|
| W3-1 | Create local health-check script | Observability | New `scripts/local-health-check.sh` | Health visible | Delete script |
| W3-2 | Add disk usage metrics to SSE | Observability | `dashboard/server/routes/omp.js` | Dashboard shows disk | Revert code |
| W3-3 | Add simple alerting (stdout/log) | Observability | `dashboard/server/state.js` | Alerts visible in log | Revert code |
| W3-4 | Campaign timing instrumentation | Observability | Campaign output | Timing metrics | Revert code |

### Wave 4: Advanced Optimization (8-40 hours, medium risk)

| ID | Description | Category | Files Affected | Measurement | Rollback |
|----|-------------|----------|----------------|-------------|----------|
| W4-1 | Batch HoF sync writes | DB | `dashboard/server/services/hofSync.js` | Sync time reduction | Revert code |
| W4-2 | Pre-campaign disk cleanup trigger | Ops | OMP daemon code | Zero disk-full incidents | Revert code |
| W4-3 | Stuck campaign watchdog | Ops | OMP daemon code | Zero stuck runs | Revert code |
| W4-4 | Parallel campaign prep (SIMD-friendly) | Throughput | Campaign setup | Startup time reduction | Revert code |

---

## 3) Quick Wins (Up to 1 Day) - LOCAL

Top 5 zero-risk, high-impact changes:

1. **Cron cleanup** (W0-1)
   ```bash
   crontab -e
   # Add: 0 * * * * /home/bahuan/Documents/GitHub/quant_b3_backtest/scripts/auto_cleanup.sh --runs --days 3
   ```

2. **Reduce keep_runs** (W1-1)
   - Edit `scripts/cleanup_old_runs.sh` line 7
   - Change `KEEP_RUNS="${2:-5}"` to `KEEP_RUNS="${2:-3}"`

3. **Log level WARN** (W0-2)
   - Edit `dashboard/omp_config.toml` line 10
   - Change `log_level = "INFO"` to `log_level = "WARN"`

4. **Disk threshold increase** (W0-3)
   - Edit `scripts/cleanup_old_runs.sh` line 8
   - Change `MIN_FREE_GB="${3:-2}"` to `MIN_FREE_GB="${3:-5}"`

5. **Workers tuning** (W2-2)
   - Run `nproc` to get CPU cores
   - Edit `dashboard/omp_config.toml` line 66
   - Set `workers = N` where N = physical cores

---

## 4) Retention and Audit Policy - LOCAL

### Permanent (Never Delete)

| Artifact | Location | Reason |
|----------|----------|--------|
| Hall of Fame strategies | `artifacts/hall_of_fame/` | Promoted elites |
| Neon PostgreSQL records | Cloud DB | HoF, candidates, campaigns |
| Audit logs (archived) | Archived after 90 days | Compliance |

### Deletable (With TTL)

| Artifact | TTL | Location | Cleanup Trigger |
|----------|-----|----------|-----------------|
| Run outputs | 3 days | `output/scg/run_*/` | Age > TTL |
| Generation JSONs | On run delete | Within runs | Parent deleted |
| Server logs | 7 days | `dashboard/logs/` | Size > 50MB or age > TTL |
| Build cache | On demand | `target/` | `--nuke` flag |
| Criterion baselines | Keep latest only | `target/criterion/` | Manual cleanup |
| Market data cache | 30 days | `.cache/market_data/` | Manual refresh |

### Automatic Triggers

| Condition | Action |
|-----------|--------|
| Disk < 5GB | Run `auto_cleanup.sh --runs --days 1` |
| Disk < 2GB | PAUSE mining, alert, run aggressive cleanup |
| Disk < 1GB | STOP mining (existing watchdog behavior) |

### Audit Trail Preservation

- Keep `artifacts/data_integrity/camp_*/` for 90 days minimum
- Keep `crates/combiner_cli/artifacts/audits/` for 90 days
- **NEVER** delete `artifacts/hall_of_fame/`

---

## 5) Governance of Operation (24/7) - LOCAL

### Runbook: Common Incidents (Local)

| Incident | Detection | Mitigation | Prevention |
|----------|-----------|------------|------------|
| Disk Full | OMP watchdog (< 1GB) | `auto_cleanup.sh --runs --days 1` | Hourly cron cleanup |
| OOM | Process killed (check `dmesg`) | Reduce `population_size`, cleanup | Monitor with `htop` |
| Stuck Run | No progress 10+ min | `kill -9` process, restart daemon | Timeout watchdog |
| HoF Sync Fail | API error log | Retry, check Neon status | Add retry logic |
| Process Crash | Check if server running | Restart `node server.js` | Autorestart script |

### Local Alerting (Minimal)

Options for local alerting:
- Write to `~/.local/share/quant_b3_backtest/alerts.log`
- Desktop notification via `notify-send` (if desktop active)
- Email via local `sendmail` (if configured)

### Fail-Closed Rules

| Condition | Action |
|-----------|--------|
| Disk < 2GB | Mining PAUSES |
| RAM < 500MB free | Mining PAUSES |
| Disk < 1GB | Mining STOPS |
| 3+ consecutive failures | Mining STOPS |
| No progress 15 min | Campaign KILLED |

### Checklists

**Before Campaign Start:**
```
[ ] Disk free > 10GB (df -h)
[ ] RAM free > 2GB (free -h)
[ ] No stuck processes (htop)
[ ] Data freshness confirmed
[ ] Config validated (combiner factory validate-config)
```

**After Campaign Finish:**
```
[ ] Check exit code (0 = success)
[ ] Review candidates generated
[ ] Verify HoF sync if promotions
[ ] Check disk usage (df -h)
[ ] Review any errors in log
```

---

## 6) 2-Week Execution Plan

### Week 1

| Day | Wave | Tasks | Gate | Stop Condition |
|-----|------|-------|------|----------------|
| 1 | 0 | W0-1, W0-2, W0-3 | Disk stable, logs reduced | Any regression |
| 2 | 1 | W1-1, W1-2 | Cron running, cleanup verified | Cleanup too aggressive |
| 3 | 1 | W1-3, W1-4 | Compression working, logs rotating | Corruption detected |
| 4-5 | 2 | W2-1, W2-2 | Throughput measured, baseline established | Performance regression |
| 6-7 | 2 | W2-3, W2-4 | Config tuned for local hardware | Instability |

### Week 2

| Day | Wave | Tasks | Gate | Stop Condition |
|-----|------|-------|------|----------------|
| 8-9 | 3 | W3-1, W3-2 | Health check working, metrics visible | Dashboard broken |
| 10 | 3 | W3-3, W3-4 | Alerts working, timing logged | False positives |
| 11-12 | 4 | W4-1, W4-2 | Batch sync working, pre-cleanup active | Data loss risk |
| 13-14 | 4 | W4-3, W4-4 (optional) | Watchdog working | Stuck detection fails |

### Promotion Gates

- **Wave N -> Wave N+1**: All items in Wave N verified, no regressions
- **Each wave**: Run 24h stable before proceeding
- **If any issue**: Stop, diagnose, fix, restart wave

### When to Stop and Reevaluate

- Throughput degrades > 10% from baseline
- Disk growth exceeds expected steady-state
- Any data loss or corruption
- HoF sync fails more than once
- OMP daemon crashes repeatedly

---

## Key Paths Reference

| Purpose | Path |
|---------|------|
| OMP Config | `dashboard/omp_config.toml` |
| Cleanup Script | `scripts/cleanup_old_runs.sh` |
| Auto Cleanup | `scripts/auto_cleanup.sh` |
| OMP Routes | `dashboard/server/routes/omp.js` |
| OMP State | `dashboard/server/state.js` |
| HoF Sync | `dashboard/server/services/hofSync.js` |
| Run Output | `output/scg/run_*/` |
| HoF Artifacts | `artifacts/hall_of_fame/` |
| Data Cache | `.cache/market_data/` |
| Local Policy | `docs/ops/local_only_policy.md` |

---

## Success Criteria

- [ ] Baseline metrics measured and documented
- [ ] Wave 0 items implemented (Day 1)
- [ ] Wave 1 items implemented (Days 2-3)
- [ ] Wave 2 items implemented (Days 4-7)
- [ ] Wave 3 items implemented (Days 8-10)
- [ ] Wave 4 items implemented (Days 11-14)
- [ ] Disk steady-state achieved (growth ~0)
- [ ] Throughput improved from baseline
- [ ] No data loss or corruption
- [ ] HoF and auditability preserved

---

## Notes

- **Local-first**: All operations assume local Ubuntu workstation
- **VPS DEFERRED**: See `docs/ops/local_only_policy.md`
- **No core changes**: Engine hot path unchanged
- **Rollback ready**: Every change has documented rollback
