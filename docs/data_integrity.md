# Data Integrity System

The Data Integrity System ensures that strategy backtests are not contaminated by data issues such as lookahead bias, corporate action artifacts, or temporal inconsistencies.

## Overview

The system operates as a **hard gate** at three critical points:

1. **Factory Run/Resume** - Blocks campaign execution if data fails audit
2. **Stage B Validation** - Rejects strategies validated on bad data (implicit via run gate)
3. **Promotion Pipeline** - Prevents promoting candidates from runs without PASS verdict

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Integrity Gate                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │ TemporalCheck    │  │ LookaheadCheck   │  │ CorpActionChk │ │
│  │ - Monotonic TS   │  │ - delay_bars>=1  │  │ - Price jumps │ │
│  │ - No duplicates  │  │ - Policy check   │  │ - >30% moves  │ │
│  │ - Gap analysis   │  │                  │  │               │ │
│  └────────┬─────────┘  └────────┬─────────┘  └───────┬───────┘ │
│           │                     │                    │         │
│           └─────────┬───────────┴──────────┬─────────┘         │
│                     │                      │                    │
│                     v                      v                    │
│              ┌────────────────────────────────────┐             │
│              │       DataIntegrityReport          │             │
│              │  - verdict: PASS/FAIL              │             │
│              │  - score: 0.0-1.0                  │             │
│              │  - hard_fails: [reasons]           │             │
│              │  - warnings: [notes]               │             │
│              └────────────────────────────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Checks

### 1. Temporal Integrity Check

Validates that OHLCV data is temporally consistent:

- **Monotonic timestamps**: Bars must be in strictly increasing order
- **No duplicates**: No two bars can have the same timestamp for the same symbol
- **Gap analysis**: Large gaps (>max_gap_days) trigger warnings

**Severity:**
- Duplicates or out-of-order: **CRITICAL** (blocks)
- Excessive gaps: **WARNING** (allows but flags)

### 2. Lookahead Policy Check

Ensures execution model prevents lookahead bias:

- **delay_bars >= 1**: Signals on bar[t] can only execute on bar[t+delay]
- Validates against campaign's execution config

**Severity:**
- delay_bars < 1: **CRITICAL** (blocks)
- delay_bars >= 1: **PASS**

### 3. Corporate Action Check

Detects suspicious price jumps that may indicate unhandled corporate actions:

- Scans for returns > jump_threshold_pct (default: 30%)
- Flags potential splits/dividends without metadata

**Severity:**
- Detected jumps: **WARNING** (allows but flags for review)
- No jumps: **PASS**

### 4. Survivorship Check

Validates universe composition methodology:

- **PointInTime**: Historical constituents available (best)
- **Static**: Fixed list (survivorship bias risk)
- **Unknown**: Unable to determine (warning)

**Severity:**
- PointInTime: **PASS**
- Static/Unknown: **WARNING**

## Configuration

Add to your campaign config:

```toml
[data_integrity]
# Audit mode: "fast" (sampling) or "strict" (full scan)
mode = "fast"

# Maximum allowed gap in days without explanation
max_gap_days = 5

# Threshold for detecting suspicious price jumps (percent)
jump_threshold_pct = 30.0

# Price adjustment type: "raw", "adjusted", or "total_return"
price_adjustment = "adjusted"

# Universe type: "point_in_time", "static", or "unknown"
universe_type = "unknown"

# Enable data integrity check (default: true)
enabled = true
```

## CLI Commands

### Standalone Audit

Run a data integrity audit without starting a campaign:

```bash
combiner factory audit-data --campaign configs/factory_campaign.toml --mode fast
```

### Integrated with Factory Run

The audit runs automatically before campaign execution:

```bash
combiner factory run --campaign configs/factory_campaign.toml
```

If the audit fails, the campaign will not start and you'll see:

```
[Data Integrity] FAILED - Blocking campaign execution
  ❌ Lookahead policy violation: delay_bars=0 but required >= 1
```

## Reports

Reports are saved to:
- `artifacts/data_integrity/<campaign_id>/report.json`
- `artifacts/data_integrity/audit_report.json` (standalone)

### Report Format

```json
{
  "verdict": "PASS",
  "score": 0.85,
  "dataset_hash": "sha256:abc123...",
  "market": "BR",
  "timezone": "America/Sao_Paulo",
  "checks": [...],
  "hard_fails": [],
  "warnings": ["Static universe detected..."],
  "stats": {
    "total_checks": 4,
    "passed": 3,
    "warnings": 1,
    "critical": 0,
    "duration_ms": 42
  },
  "created_at": "2024-12-28T12:00:00Z",
  "audit_mode": "fast",
  "version": "1.0.0"
}
```

## Database Schema

The registry tracks data integrity status per run:

```sql
ALTER TABLE scg_runs 
    ADD COLUMN data_integrity_verdict TEXT,
    ADD COLUMN data_integrity_score REAL,
    ADD COLUMN data_integrity_report_path TEXT;
```

## Best Practices

1. **Always run audits in CI/CD** before deploying campaigns
2. **Use strict mode** for production campaigns
3. **Review warnings** even if verdict is PASS
4. **Set appropriate thresholds** for your market (B3 vs US)
5. **Keep price_adjustment consistent** across train/test splits

## Troubleshooting

### "Lookahead policy violation"

Fix: Ensure `delay_bars >= 1` in your execution config:

```toml
[execution]
delay_bars = 1
```

### "Static universe detected"

This is a warning about survivorship bias risk. To suppress:
- Provide point-in-time historical constituents, OR
- Explicitly acknowledge the limitation in campaign notes

### "Excessive gap detected"

Check if the gaps align with known market holidays. If not:
- Verify your data pipeline
- Check for missing files in the data directory
