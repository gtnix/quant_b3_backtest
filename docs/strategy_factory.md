# Strategy Factory - Runbook

The Strategy Factory is a campaign orchestration system for running, tracking, and promoting SCG strategy candidates. It provides:

- **Multi-seed campaigns**: Run experiments with multiple seeds for robustness
- **Experiment registry**: Track all runs in Neon PostgreSQL
- **Resume capability**: Restart interrupted campaigns without duplication
- **Promotion pipeline**: Move validated candidates from research to paper trading
- **Reproducibility**: Full provenance tracking with config/dataset hashes

## Prerequisites

### Environment Setup

Set the Neon PostgreSQL connection string:

```bash
export NEON_DATABASE_URL="postgresql://user:password@ep-xxx.us-east-2.aws.neon.tech/neondb?sslmode=require"
```

### Database Tables

The following tables are automatically created in Neon:

- `scg_campaigns` - Campaign metadata and status
- `scg_runs` - Individual run records with metrics
- `scg_candidates` - Top candidates from each run
- `scg_promotions` - Promotion tracking

## Quick Start

### 1. Initialize a Campaign

Create a new campaign configuration:

```bash
combiner factory init --name my_momentum_campaign
```

This creates:
- `configs/campaigns/my_momentum_campaign.toml` - Campaign config template
- `artifacts/candidates/` - Directory for promoted candidate bundles

### 2. Edit Campaign Configuration

Edit `configs/campaigns/my_momentum_campaign.toml`:

```toml
[campaign]
name = "momentum_exploration_q1"
tag = "momentum"
owner = "quant_team"
notes = "Exploring momentum strategies for B3"

[dataset]
market = "BR"
start_date = "2018-01-01"
end_date = "2024-12-01"
universe = "ibov"

[evolution]
population_size = 100
max_generations = 50

[execution]
config_path = "configs/execution_institutional.toml"

[seeds]
count = 5        # Run 5 different seeds
base_seed = 42   # Seeds: 42, 43, 44, 45, 46

[budget]
max_runs = 5
top_k = 10       # Validate top 10 per run
timeout_per_run_secs = 3600
stress_enabled = true

[promotion]
min_oos_sharpe_net = 0.5
max_pbo = 0.15
min_stress_passed = 4
gates_required = true
```

### 3. Run the Campaign

Execute all seeds:

```bash
combiner factory run --campaign configs/campaigns/my_momentum_campaign.toml
```

Output:
```
╔══════════════════════════════════════════════════════════════╗
║              STRATEGY FACTORY - CAMPAIGN RUN                 ║
╠══════════════════════════════════════════════════════════════╣
║ Campaign:    momentum_exploration_q1                         ║
║ Config Hash: sha256:a1b2c3d4                                 ║
║ Seeds:       [42, 43, 44, 45, 46]                            ║
║ Mode:        NEW                                             ║
╚══════════════════════════════════════════════════════════════╝

⠋ [00:05:32] [################>-----------------------] 2/5 seeds (seed 44)
```

### 4. Resume if Interrupted

If the campaign is interrupted, resume from where it left off:

```bash
combiner factory resume --campaign configs/campaigns/my_momentum_campaign.toml
```

Only incomplete seeds will be re-run.

### 5. View Results

List all campaigns:

```bash
combiner factory list
```

Show campaign details:

```bash
combiner factory show camp_abc123
```

Show run details:

```bash
combiner factory show run_xyz789
```

### 6. Compare Runs

Compare top candidates across multiple runs:

```bash
combiner factory compare --runs run_1,run_2,run_3 --top 5
```

Output:
```
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    CANDIDATE COMPARISON                                          ║
╠════════════════╦════════════════╦═══════╦═══════════╦═══════════╦═══════╦═════════╦══════════════╣
║ Run ID         ║ Candidate ID   ║ Rank  ║ OOS SR    ║ Gross SR  ║ PBO   ║ Stress  ║ Gates        ║
╠════════════════╬════════════════╬═══════╬═══════════╬═══════════╬═══════╬═════════╬══════════════╣
║ run_abc123     ║ cand_def456    ║ 0     ║ 0.872     ║ 0.945     ║ 0.08  ║ 5/5     ║ PASS         ║
║ run_ghi789     ║ cand_jkl012    ║ 0     ║ 0.756     ║ 0.823     ║ 0.12  ║ 4/5     ║ PASS         ║
╚════════════════╩════════════════╩═══════╩═══════════╩═══════════╩═══════╩═════════╩══════════════╝
```

### 7. Promote Candidates

Promote top candidates from a run to the "candidate" stage:

```bash
combiner factory promote --run run_abc123 --top 3 --stage candidate
```

Or promote from an entire campaign:

```bash
combiner factory promote --campaign camp_abc123 --top 3 --stage candidate
```

Promotion criteria (from campaign config):
- OOS Sharpe NET >= 0.5
- PBO <= 0.15
- Stress passed >= 4/5
- All institutional gates passed

### 8. Candidate Bundles

Promoted candidates are saved to `artifacts/candidates/<candidate_id>/`:

```
artifacts/candidates/cand_def456/
├── strategy.toml           # Strategy configuration
├── execution_config.toml   # Execution/cost model used
├── validation_summary.json # NET metrics, PBO, stress results
├── provenance.json         # Full audit trail
└── replay.sh               # Deterministic replay script
```

#### provenance.json

```json
{
  "candidate_id": "cand_def456",
  "genome_hash": "sha256:a1b2c3d4",
  "run_id": "run_abc123",
  "campaign_id": "camp_xyz789",
  "seed": 42,
  "git_sha": "abc123",
  "git_branch": "main",
  "config_hash": "sha256:e5f6g7h8",
  "dataset_hash": "sha256:i9j0k1l2",
  "created_at": "2024-12-28T12:00:00Z",
  "scg_version": "0.1.0",
  "original_report_path": "output/scg/run_abc123/final_report.json"
}
```

## CLI Reference

### factory init

Initialize a new campaign:

```bash
combiner factory init --name <campaign_name>
```

### factory run

Run a campaign:

```bash
combiner factory run --campaign <path/to/campaign.toml>
```

### factory resume

Resume an interrupted campaign:

```bash
combiner factory resume --campaign <path/to/campaign.toml>
```

### factory list

List campaigns:

```bash
combiner factory list [--tag <tag>]
```

### factory show

Show campaign or run details:

```bash
combiner factory show <campaign_id_or_run_id>
```

### factory compare

Compare candidates across runs:

```bash
combiner factory compare --runs <run1,run2,...> [--top <n>]
```

### factory promote

Promote candidates:

```bash
combiner factory promote --run <run_id> [--top <n>] [--stage <stage>] [--force]
combiner factory promote --campaign <campaign_id> [--top <n>] [--stage <stage>] [--force]
```

Stages: `research`, `candidate`, `paper`

## Best Practices

### Seed Selection

- Use at least 3-5 seeds per campaign for robustness
- Different seeds help identify strategies that are robust vs. lucky

### Promotion Criteria

Conservative thresholds for institutional-quality strategies:

| Metric | Minimum/Maximum |
|--------|-----------------|
| OOS Sharpe (NET) | >= 0.5 |
| PBO | <= 0.15 |
| Stress Scenarios | >= 4/5 passed |
| Institutional Gates | All PASS |

### Reproducibility

Every campaign run captures:
- Git SHA and branch
- Config hash (deterministic)
- Dataset hash (if data_path specified)
- Seed used

Use `replay.sh` in candidate bundles to reproduce results.

### Structured Logging

Enable JSON logs for production monitoring:

```bash
export FACTORY_JSON_LOGS=1
combiner factory run --campaign ...
```

## Troubleshooting

### "Environment variable NEON_DATABASE_URL not set"

Set the connection string:

```bash
export NEON_DATABASE_URL="postgresql://..."
```

### "Failed to connect to PostgreSQL"

- Check network connectivity to Neon
- Verify connection string includes `sslmode=require`
- Check Neon dashboard for IP allowlist settings

### "Campaign already exists"

Use `factory resume` to continue an existing campaign, or choose a different name.

### Duplicate Promotion Prevented

The registry tracks promoted genome hashes to prevent duplicates. Use `--force` to override:

```bash
combiner factory promote --run <id> --force
```


## Export Top Candidates

The `factory export-top` command exports the top N candidates from a run with a deterministic ranking.

### Usage

```bash
# Export top 1000 candidates in JSON and CSV
combiner factory export-top --run <run_id> --top 1000 --format json,csv

# Export top 100 in JSON only
combiner factory export-top --run <run_id> --top 100 --format json
```

### Ranking Rules (Deterministic)

Candidates are ranked using the following criteria, applied in order:

| Priority | Field | Order | Fallback |
|----------|-------|-------|----------|
| 1 | oos_sharpe_net | DESC (higher is better) | -∞ |
| 2 | pbo | ASC (lower is better) | 1.0 |
| 3 | oos_cagr_net | DESC (higher is better) | 0.0 |
| 4 | max_drawdown_net | ASC (less negative is better) | 0.0 |
| 5 | genome_hash | ASC (alphabetical) | - |

**Tie-breaker**: `genome_hash` ensures stable ordering across runs.

### Filters Applied

1. **Data Integrity**: Run must have `data_integrity_verdict = 'PASS'`
2. **Gates**: Candidate must have `gates_passed = true` (or NULL, which is treated as passed)

### Output

Files are saved to `artifacts/top_candidates/<run_id>/`:

- `top1000.json` - Full export with metadata
- `top1000.csv` - Tabular format for analysis

### JSON Schema

```json
{
  "metadata": {
    "schema_version": "1.0.0",
    "run_id": "run_abc123",
    "campaign_id": "camp_xyz",
    "data_integrity_verdict": "PASS",
    "top_n": 1000,
    "actual_count": 847,
    "exported_at": "2024-12-28T12:00:00Z",
    "ranking_rules": [...],
    "filters_applied": [...]
  },
  "candidates": [
    {
      "rank": 1,
      "candidate_id": "cand_001",
      "genome_hash": "sha256:abc123",
      "oos_sharpe_net": 1.234,
      "pbo": 0.08,
      "oos_cagr_net": 0.15,
      "max_drawdown_net": -0.12,
      ...
    }
  ]
}
```

### CSV Columns

| Column | Description |
|--------|-------------|
| rank | Position in ranking (1 = best) |
| candidate_id | Unique ID |
| genome_hash | Hash for deduplication |
| oos_sharpe_net | Out-of-sample Sharpe ratio |
| pbo | Probability of Backtest Overfitting |
| oos_cagr_net | Out-of-sample CAGR |
| max_drawdown_net | Maximum drawdown |
| dsr | Deflated Sharpe Ratio |
| stress_passed | Stress scenarios passed |
| stress_total | Total stress scenarios |
| gates_passed | Institutional gates passed |
| turnover_annual | Annual turnover |
| capacity_usd | Estimated capacity (USD) |
| created_at | Timestamp |

### Reproducibility Guarantee

Given the same:
- Database state
- Run ID
- Top N parameter

The export will produce **identical output** (same candidates, same order).

---

## Consumo de Recursos

### Espaço em Disco

> ⚠️ **Importante**: Campanhas SCG consomem espaço significativo em disco.

| Duração | Backtests | Espaço Estimado |
|---------|-----------|-----------------|
| 5 min | ~97k | 6.7 GB |
| 30 min | ~580k | 40 GB |
| 1 hora | ~1.16M | 80 GB |
| 4 horas | ~4.64M | 320 GB |

**Causa principal**: O arquivo `timeseries.csv` (57KB por backtest) é gerado para todos os backtests e representa 94% do espaço consumido.

Para análise detalhada, consulte [Análise de Armazenamento](operations/storage-analysis.md).

### Gerenciamento de Espaço

```bash
# Ver uso de espaço por run
du -sh output/scg/*/

# Limpar runs antigos (> 7 dias)
find output/scg -type d -mtime +7 -exec rm -rf {} \;

# Manter apenas Hall of Fame e relatórios
find output/scg/*/backtests -type d -exec rm -rf {} \;
```
