# Local-Only Policy

**Effective Date**: 2026-01-18  
**Status**: ACTIVE

---

## Official Declaration

**Target Environment: LOCAL Ubuntu 100%**

All development, testing, and production operations run exclusively on local Ubuntu Linux workstations. VPS (Virtual Private Server) deployment is explicitly out of scope for all current work.

---

## Policy Rules

### 1. No VPS Tasks in Active Backlog

- No VPS-related tasks, TODOs, or work items shall appear in the active backlog
- No VPS deployment planning or preparation work is permitted
- No VPS infrastructure costs or provisioning discussions

### 2. Documentation Standards

- General documentation must not reference VPS as a current or planned target
- VPS content in existing docs is marked as **DEFERRED - not in scope**
- No new VPS documentation shall be created

### 3. Historical Content

VPS-related content that exists for historical reference:
- Must be clearly marked with `DEFERRED - VPS NOT IN SCOPE`
- Must not generate any actionable tasks
- May be retained for future reference only

### 4. Affected Components

The following VPS-related components are DEFERRED (not actively maintained):

| Component | Location | Status |
|-----------|----------|--------|
| VPS deploy scripts | `scripts/vps/` | DEFERRED |
| VPS setup script | `scripts/setup-vps.sh` | DEFERRED |
| Deploy workflow | `.github/workflows/deploy-vps.yml` | DEFERRED |
| VPS deployment guide | `docs/dashboard/vps-deployment.md` | DEFERRED |
| PM2 ecosystem config | `scripts/vps/ecosystem.config.cjs` | DEFERRED |
| Health check (VPS) | `scripts/vps/health-check.sh` | DEFERRED |
| Nginx config | `scripts/vps/nginx-dashboard.conf` | DEFERRED |

### 5. Local-First Requirements

All system operations must:
- Run on local Ubuntu workstation
- Use local process management (not PM2 on VPS)
- Store artifacts locally
- Connect to Neon PostgreSQL for persistence (cloud DB, accessed locally)

---

## Rationale

1. **Simplified Operations**: Local execution eliminates VPS maintenance overhead
2. **Full Resources**: Local workstations have more CPU/RAM than $5/month VPS
3. **Development Velocity**: No deploy cycles, immediate iteration
4. **Cost Reduction**: No VPS hosting costs

---

## Scope Exclusions

The following are explicitly OUT OF SCOPE:

- VPS provisioning (Vultr, DigitalOcean, Linode, Hetzner, AWS EC2)
- VPS sizing recommendations (1vCPU, 1GB RAM, etc.)
- VPS deployment automation
- VPS monitoring and alerting
- VPS nginx/reverse proxy configuration
- VPS PM2 process management
- VPS SSH access and security

---

## Future Considerations

If VPS deployment becomes necessary in the future:
1. This policy must be explicitly revoked
2. DEFERRED components must be reviewed and updated
3. New deployment documentation must be created
4. Resource sizing must be re-evaluated based on actual workloads

---

## Compliance

All team members and AI agents must:
- Verify no VPS assumptions in proposed changes
- Flag any VPS references in code reviews
- Update existing VPS references to DEFERRED status when encountered

---

```
VPS_POLICY_STATUS: DEFERRED (not in scope)
```
