---
name: devops-infra
description: "Local Ubuntu infrastructure, reliability, observability, and CI/CD"
triggers:
  - command: "/devops-infra"
    description: "Invoke for local infrastructure, CI/CD, monitoring, health checks"
domain_knowledge:
  - local development environment (Ubuntu workstation)
  - build pipelines (cargo, npm)
  - config and secrets management
  - observability (logs, metrics, health checks)
  - incident response and runbooks
  - CI/CD workflows (GitHub Actions)
---

# DevOps Infra

> **NOTA**: VPS deployment is DEFERRED. See `docs/ops/local_only_policy.md`.
> This skill now focuses on **local Ubuntu workstation** infrastructure.

## Role

DevOps / Infrastructure Lead (Local Environment, Reliability, Observability, CI/CD).
Expert in local development environment, build pipelines, and operational reliability.

---

## Expertise Map

### Build Pipeline (Local)
- **build**: `cargo build --release` for Rust binaries
- **dashboard**: `npm run build` for React frontend
- **dev**: `npm run dev` for local development server
- **verify**: Local health checks and endpoint testing

### Local Process Management
- Direct process execution (no PM2 for local)
- Dashboard API: `node server.js` (port 3001)
- Frontend: `npm run dev` (port 5173)
- OMP daemon via dashboard API

### Config and Secrets Management
- **Local Secrets**: `dashboard/.env` (never in git)
- **CI Secrets**: GitHub Secrets (DATABASE_URL, BRAPI_API_KEY)
- **Validation**: Check required vars before running

### Observability (Local)
- **Health Checks**: `scripts/local-health-check.sh` (planned)
  - API health, OMP status
  - System resources: CPU%, Memory%, Disk%
- **Logs**: Dashboard server logs, campaign output
- **Monitoring**: `.github/workflows/monitoring.yml` (daily data checks)

### Reliability (Local)
- **Cleanup**: `scripts/auto_cleanup.sh` for disk management
- **Health Checks**: Cron-compatible for monitoring
- **Graceful Shutdown**: Signal handlers

### Local Resource Management
- Monitor with `htop`, `df -h`, `free -h`
- Cleanup with `scripts/cleanup_old_runs.sh`
- Configure workers based on CPU cores

### CI/CD
- **CI**: `.github/workflows/ci.yml` (tests, benchmarks, calendar)
- **Monitoring**: `.github/workflows/monitoring.yml` (daily data checks)

### VPS Infrastructure - DEFERRED
> VPS deployment is DEFERRED. Scripts in `scripts/vps/` are for historical reference.
> See `docs/ops/local_only_policy.md`.

---

## When to Use

**INVOKE this skill when:**
- Local build failed or service not starting
- Memory or disk issues on local workstation
- DB connections saturated
- Setup or troubleshoot CI/CD
- Secrets exposed or environment drift
- Health checks failing
- Cleanup needed for disk space

**DO NOT use this skill when:**
- Optimizing application code (use `/quant-engineer`)
- Investigating data quality (use `/data-engineer`)
- Validating strategy performance (use `/risk-analyst`)
- Designing strategy logic (use `/quant-researcher`)

---

## Operating Rules

### Hard Constraints

1. **Never run without traceable git sha**
   - Use `git log -1 --format="%H"` before major operations

2. **Never modify env without logging change**
   - Document what changed, when, why
   - Keep ops change log

3. **Never commit secrets to git**
   - Use env vars and GitHub Secrets
   - Template files only in repo

4. **Never update schema without migration + rollback plan**
   - Coordinate with data-engineer
   - Test rollback before applying

5. **Never ignore disk full / memory pressure alerts**
   - Monitor with `df -h` and `free -m`
   - Clean artifacts: `scripts/cleanup_old_runs.sh`

6. **Never operate without backup/restore plan**
   - Neon PostgreSQL has automatic backups
   - Local artifacts backed up before deletion

7. **Never change configs without local validation first**
   - Test configs before running campaigns

---

## Repo Anchors

### Local Scripts

| File | Purpose |
|------|---------|
| `scripts/start.sh` | Local development start |
| `scripts/stop.sh` | Local development stop |
| `scripts/cleanup_old_runs.sh` | Clean old run artifacts |
| `scripts/auto_cleanup.sh` | Automated cleanup with options |

### CI/CD Workflows

| File | Purpose |
|------|---------|
| `.github/workflows/ci.yml` | Tests, benchmarks, calendar integrity |
| `.github/workflows/monitoring.yml` | Daily data freshness and integrity |

### VPS Scripts (DEFERRED)

| File | Purpose |
|------|---------|
| `scripts/vps/*` | VPS scripts - DEFERRED (historical reference) |
| `scripts/deploy.sh` | Deploy script - DEFERRED |
| `scripts/setup-vps.sh` | VPS setup - DEFERRED |

---

## Quick Reference

### Local Commands

```bash
# Start dashboard API
cd dashboard && node server.js

# Start frontend (dev)
cd dashboard && npm run dev

# Build frontend
cd dashboard && npm run build

# Check API health
curl http://localhost:3001/api/health

# Cleanup old runs
./scripts/auto_cleanup.sh --runs --days 3
```

### Monitoring Commands

```bash
# System resources
htop
free -h
df -h

# Disk usage by directory
du -sh output/scg/
du -sh target/
du -sh artifacts/
```
