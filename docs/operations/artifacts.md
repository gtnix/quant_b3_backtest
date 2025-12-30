# Artefatos e Estrutura de Output

**Versão**: 1.1.0  
**Última Atualização**: 2025-12-29

## Visão Geral

O sistema gera artefatos em dois diretórios principais:

1. **`output/`** - Resultados de backtests e experimentos SCG
2. **`artifacts/`** - Candidatos promovidos e dados de validação

---

## Estrutura do Diretório `output/`

```
output/
├── experiments/           # Backtests individuais
│   └── <experiment_id>/
│       ├── manifest.json
│       ├── metrics.json
│       ├── trades.csv
│       └── nav_history.csv
│
├── replay/                # Replays de backtests
│   └── <run_id>/
│       └── <experiment_id>/
│           └── ...
│
└── scg/                   # Experimentos SCG
    └── <experiment_id>/
        ├── manifest.json
        ├── report.json
        ├── generations/
        ├── hall_of_fame/
        └── cache/
```

---

## Artefatos de Backtest Individual

### `manifest.json`

Metadados do run para audit trail.

```json
{
  "run_id": "20251228_143022_build_run_momentum_a1b2c3d4",
  "run_type": "build_run",
  "created_at_utc": "2025-12-28T14:30:22Z",
  "git_commit": "abc123def456",
  "build_profile": "release",
  "dataset_signature": "sha256:...",
  "config_signature": "sha256:...",
  "strategy_id": "momentum_v3",
  "machine_fingerprint": "linux-x86_64"
}
```

### `metrics.json`

Métricas completas do backtest.

```json
{
  "total_return": 0.2534,
  "annual_return": 0.1245,
  "annual_volatility": 0.1523,
  "sharpe_ratio": 0.8175,
  "sortino_ratio": 1.2341,
  "calmar_ratio": 1.5623,
  "max_drawdown": 0.0797,
  "max_drawdown_duration": 45,
  "current_drawdown": 0.0234,
  "win_rate": 0.5432,
  "profit_factor": 1.3421,
  "num_trades": 156,
  "num_winning_trades": 85,
  "num_losing_trades": 71,
  "avg_trade_return": 0.0016,
  "avg_winning_trade": 0.0234,
  "avg_losing_trade": -0.0123,
  "max_consecutive_wins": 8,
  "max_consecutive_losses": 5,
  "gross_profit": 12345.67,
  "gross_loss": 9200.45,
  "net_pnl": 3145.22,
  "total_costs": 234.56,
  "final_nav": 125340.00,
  "initial_capital": 100000.00,
  "events_processed": 45678,
  "fills_executed": 312
}
```

### `trades.csv`

Log de todas as trades executadas.

```csv
trade_id,asset_id,direction,entry_price,exit_price,quantity,entry_time,exit_time,gross_pnl,net_pnl,costs
1,PETR4,Buy,28.45,29.12,1000,2023-01-15,2023-02-20,670.00,665.50,4.50
2,VALE3,Buy,68.90,67.20,500,2023-01-18,2023-01-25,-850.00,-854.25,4.25
...
```

### `nav_history.csv`

Série temporal do NAV.

```csv
timestamp,nav,drawdown,peak
2023-01-02,100000.00,0.0000,100000.00
2023-01-03,100234.56,0.0000,100234.56
2023-01-04,99876.43,0.0036,100234.56
...
```

---

## Artefatos SCG

### Estrutura do Experimento SCG

```
output/scg/<experiment_id>/
├── manifest.json           # Metadados do experimento
├── report.json             # Relatório final
├── generations/            # Estatísticas por geração
│   ├── gen_000.json
│   ├── gen_001.json
│   └── ...
├── hall_of_fame/           # Top estratégias
│   ├── ranking.json
│   └── strategy_001/
│       ├── config.toml     # Configuração executável
│       ├── genome.json     # Genoma completo
│       └── metrics.json    # Métricas de fitness
└── cache/                  # Cache de avaliações
    └── <genome_hash>.json
```

### `manifest.json` (SCG)

```json
{
  "experiment_id": "scg_20251228_143022",
  "status": "completed",
  "created_at": "2025-12-28T14:30:22Z",
  "completed_at": "2025-12-28T15:45:33Z",
  "config_hash": "sha256:a1b2c3d4...",
  "git_sha": "abc123def456",
  "seed": 42,
  "population_size": 100,
  "max_generations": 50,
  "generations_completed": 50,
  "total_evaluations": 5000,
  "cache_hits": 1234,
  "duration_seconds": 4511
}
```

### `report.json` (SCG)

```json
{
  "experiment_id": "scg_20251228_143022",
  "status": "completed",
  "generations_completed": 50,
  "total_evaluations": 5000,
  "cache_hits": 1234,
  "duration_seconds": 4511,
  "hall_of_fame_size": 25,
  "generation_stats": [
    {
      "generation": 0,
      "population_size": 100,
      "evaluated": 100,
      "cache_hits": 0,
      "best_sharpe": 0.45,
      "best_cagr": 0.08,
      "mean_sharpe": 0.23,
      "pareto_size": 15,
      "duration_ms": 2345
    },
    ...
  ],
  "top_strategies": [
    {
      "rank": 0,
      "id": "a1b2c3d4",
      "sharpe": 1.23,
      "cagr": 0.15,
      "max_dd": -0.08
    },
    ...
  ]
}
```

### `ranking.json` (Hall of Fame)

```json
{
  "schema_version": "1.0.0",
  "experiment_id": "scg_20251228_143022",
  "generated_at": "2025-12-28T15:45:33Z",
  "strategies": [
    {
      "rank": 1,
      "genome_hash": "sha256:a1b2c3d4...",
      "sharpe_ratio": 1.23,
      "cagr": 0.15,
      "max_drawdown": -0.08,
      "calmar_ratio": 1.875,
      "pareto_rank": 0,
      "crowding_distance": 0.456
    },
    ...
  ]
}
```

### `genome.json` (Estratégia individual)

```json
{
  "id": "a1b2c3d4",
  "genome_hash": "sha256:a1b2c3d4...",
  "generation": 42,
  "origin": "crossover",
  "genes": [
    {
      "block_type": "Selection",
      "block_id": "momentum",
      "params": {
        "lookback_days": 126,
        "top_n": 20
      }
    },
    {
      "block_type": "Sizing",
      "block_id": "equal_weight",
      "params": {}
    },
    {
      "block_type": "Exit",
      "block_id": "stop_loss",
      "params": {
        "pct": 0.08
      }
    }
  ]
}
```

---

## Estrutura do Diretório `artifacts/`

```
artifacts/
├── site/                  # Dashboard index files (JSON)
│   ├── index.json         # Campaign list for dashboard
│   ├── campaign_<id>.json # Campaign detail + runs
│   └── run_<id>.json      # Run detail + top candidates
│
├── candidates/            # Candidatos promovidos
│   └── cand_<hash>/
│       ├── strategy.toml
│       ├── execution_config.toml
│       ├── validation_summary.json
│       ├── provenance.json
│       ├── replay.sh
│       └── output/        # Replay output (opcional)
│
├── top_candidates/        # Exports de top candidatos
│   └── <run_id>/
│       ├── top1000.json
│       └── top1000.csv
│
├── backtests/             # Backtest timeseries data
│   └── <candidate_id>/
│       ├── timeseries.csv
│       └── metadata.json
│
├── cockpit_runs/          # Runs iniciados via Dashboard Cockpit
│   └── <run_id>/
│       ├── config.json    # Configuração usada (presets, gates)
│       ├── progress.json  # Último estado de progresso
│       └── candidates/    # Candidatos descobertos
│
├── campaigns/             # Campanhas completas
│
└── data_integrity/        # Relatórios de integridade
    ├── audit_report.json
    └── <campaign_id>/
        └── integrity_check.json
```

---

## Dashboard Site Index (`artifacts/site/`)

Os arquivos em `site/` são gerados pelo Strategy Factory para consumo pelo dashboard Tauri.

### `index.json`

Índice global de campanhas.

```json
{
  "schema_version": "1.0",
  "generated_at": "2024-12-28T10:00:00Z",
  "campaigns": [
    {
      "campaign_id": "camp_001",
      "name": "IBOV Momentum Q4",
      "tag": "production",
      "status": "completed",
      "runs_count": 3,
      "created_at": "2024-12-15T10:00:00Z",
      "detail_path": "campaign_camp_001.json"
    }
  ]
}
```

### `campaign_<id>.json`

Detalhes da campanha e lista de runs.

```json
{
  "schema_version": "1.0",
  "campaign": {
    "campaign_id": "camp_001",
    "name": "IBOV Momentum Q4",
    "tag": "production",
    "owner": "quant_team",
    "status": "completed",
    "config_hash": "abc123",
    "git_sha": "def456",
    "created_at": "2024-12-15T10:00:00Z",
    "notes": "Production momentum strategy"
  },
  "runs": [
    {
      "run_id": "run_001",
      "seed": 42,
      "status": "completed",
      "data_integrity_verdict": "PASS",
      "data_integrity_score": 0.98,
      "candidates_count": 1250,
      "research_candidates_count": 1000,
      "validated_candidates_count": 250,
      "best_oos_sharpe_net": 1.85,
      "duration_secs": 3600,
      "detail_path": "run_run_001.json",
      "export_path": "top_candidates/run_001/"
    }
  ]
}
```

### `run_<id>.json`

Detalhes completos do run e top candidatos.

```json
{
  "schema_version": "1.0",
  "run": {
    "run_id": "run_001",
    "campaign_id": "camp_001",
    "seed": 42,
    "status": "completed",
    "started_at": "2024-12-15T10:00:00Z",
    "completed_at": "2024-12-15T11:00:00Z",
    "duration_secs": 3600,
    "artifact_path": "output/scg/scg_20241215_100000"
  },
  "config_snapshot": {},
  "metrics": {
    "total_evaluated": 5000,
    "research_candidates": 1000,
    "validated_candidates": 250,
    "promoted_candidates": 50,
    "best_oos_sharpe_net": 1.85,
    "best_oos_cagr_net": 0.32,
    "data_integrity_verdict": "PASS",
    "data_integrity_score": 0.98
  },
  "top_candidates": [
    {
      "rank": 1,
      "candidate_id": "cand_a1b2c3d4",
      "candidate_class": "validated",
      "display_name": "Sel:IBOV-Top20 | Entry:RSI(14) | Exit:TP/SL",
      "oos_sharpe_net": 1.85,
      "oos_cagr_net": 0.28,
      "max_drawdown_net": -0.12,
      "pbo": 0.08,
      "dsr": 1.65,
      "gates_passed": true,
      "stress_passed": true,
      "data_integrity_ok": true
    }
  ],
  "exports": {
    "top1000_json": "top_candidates/run_001/top1000.json",
    "top1000_csv": "top_candidates/run_001/top1000.csv",
    "pareto_json": "top_candidates/run_001/pareto.json"
  }
}
```

---

## Backtest Timeseries (`artifacts/backtests/`)

### `timeseries.csv`

Série temporal diária do backtest.

```csv
date,equity,drawdown,exposure,vol_exante,vol_expost
2023-01-02,1.0000,0.0000,0.80,0.15,0.14
2023-01-03,1.0234,0.0000,0.85,0.15,0.14
2023-01-04,0.9987,-0.0241,0.82,0.15,0.15
...
```

### `metadata.json`

Metadados do backtest.

```json
{
  "schema_version": "1.0",
  "run_id": "run_001",
  "config_hash": "abc123",
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "mode": "standard"
}
```

---

## Artefatos de Candidatos Promovidos

### `strategy.toml`

Configuração executável da estratégia.

```toml
[strategy]
name = "cand_a1b2c3d4"
version = "1.0"
description = "SCG discovered strategy"

[[pipeline.selection]]
block = "momentum"
lookback_days = 126
top_n = 20

[[pipeline.sizing]]
block = "equal_weight"

[[pipeline.exit]]
block = "stop_loss"
pct = 0.08
```

### `execution_config.toml`

Modelo de execução usado na validação.

```toml
[execution]
delay_bars = 1
slippage_model = "volume_linear"
slippage_bps = 10.0

[costs]
commission_bps = 5.0
emolument_bps = 2.5
```

### `validation_summary.json`

Resumo da validação com métricas NET.

```json
{
  "candidate_id": "cand_a1b2c3d4",
  "validated_at": "2025-12-28T16:00:00Z",
  "gross_metrics": {
    "sharpe_ratio": 1.23,
    "cagr": 0.15,
    "max_drawdown": -0.08
  },
  "net_metrics": {
    "sharpe_ratio": 0.89,
    "cagr": 0.12,
    "max_drawdown": -0.09
  },
  "wfa": {
    "num_folds": 5,
    "oos_sharpe": 0.89,
    "oos_cagr": 0.12,
    "degradation_sharpe": 0.28
  },
  "pbo_dsr": {
    "pbo": 0.08,
    "dsr": 0.82,
    "num_trials": 5000
  },
  "stress_tests": {
    "total": 5,
    "passed": 5,
    "results": [
      {"scenario": "HighSlippage", "passed": true, "sharpe": 0.45},
      {"scenario": "HighCosts", "passed": true, "sharpe": 0.52},
      {"scenario": "DelayedExecution", "passed": true, "sharpe": 0.38},
      {"scenario": "LowLiquidity", "passed": true, "sharpe": 0.41},
      {"scenario": "AdverseConditions", "passed": true, "sharpe": 0.28}
    ]
  },
  "gates_passed": true,
  "promotion_eligible": true
}
```

### `provenance.json`

Rastreabilidade completa do candidato.

```json
{
  "candidate_id": "cand_a1b2c3d4",
  "genome_hash": "sha256:a1b2c3d4...",
  "run_id": "run_abc123",
  "campaign_id": "camp_xyz789",
  "seed": 42,
  "git_sha": "abc123def456",
  "git_branch": "main",
  "config_hash": "sha256:e5f6g7h8...",
  "dataset_hash": "sha256:i9j0k1l2...",
  "created_at": "2025-12-28T16:00:00Z",
  "scg_version": "0.1.0",
  "original_experiment_id": "scg_20251228_143022",
  "original_report_path": "output/scg/scg_20251228_143022/report.json"
}
```

### `replay.sh`

Script para replay determinístico.

```bash
#!/bin/bash
# Replay script for cand_a1b2c3d4
# Generated: 2025-12-28T16:00:00Z

set -e

# Ensure same environment
export RUST_LOG=info

# Run backtest with exact config
cargo run --release -p backtester_cli -- run \
    --config ./strategy.toml \
    --execution-config ./execution_config.toml \
    --output ./output

# Verify hash
EXPECTED_HASH="sha256:result_hash_here"
ACTUAL_HASH=$(sha256sum ./output/metrics.json | cut -d' ' -f1)

if [ "$EXPECTED_HASH" != "$ACTUAL_HASH" ]; then
    echo "WARNING: Result hash mismatch!"
    echo "Expected: $EXPECTED_HASH"
    echo "Actual:   $ACTUAL_HASH"
    exit 1
fi

echo "Replay successful - results match original"
```

---

## Convenções de Nomenclatura

| Prefixo | Tipo | Exemplo |
|---------|------|---------|
| `scg_` | Experimento SCG | `scg_20251228_143022` |
| `run_` | Run de campanha | `run_a1b2c3d4` |
| `camp_` | Campanha | `camp_momentum_q1` |
| `cand_` | Candidato promovido | `cand_a1b2c3d4` |

### Formato de Experiment ID

```
scg_YYYYMMDD_HHMMSS
     │       │
     │       └── Hora de criação
     └── Data de criação
```

### Formato de Genome Hash

```
<8 caracteres hex do SHA256>
Exemplo: a1b2c3d4
```

---

## Limpeza de Artefatos

Script automático de limpeza:

```bash
# Limpar experimentos SCG antigos (> 30 dias)
./scripts/auto_cleanup.sh --days 30 --type scg

# Limpar cache de avaliações
./scripts/auto_cleanup.sh --cache-only

# Dry run (mostra o que seria deletado)
./scripts/auto_cleanup.sh --days 30 --dry-run
```

---

## Integração com Dashboard

O dashboard Tauri lê os artefatos via Tauri Commands (Rust backend):

### Fluxo de Dados

```
┌─────────────────────────────────────────────────────────────┐
│                    Dashboard (Tauri)                         │
│  ┌─────────────┐                                            │
│  │  React UI   │ ←── invoke('load_index')                   │
│  │  (Zustand)  │ ←── invoke('load_campaign', {campaignId})  │
│  └──────┬──────┘ ←── invoke('list_candidates_v2', {...})    │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Rust Backend (ArtifactState)               │ │
│  │  - LRU cache para candidatos (100 items)                │ │
│  │  - HashMap cache para campaigns/runs                     │ │
│  │  - File watcher para hot-reload                         │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
              │
              ▼
    artifacts/site/*.json
    artifacts/candidates/*/
    artifacts/top_candidates/*/
    artifacts/backtests/*/
```

### API de Comandos Tauri

```typescript
// Inicializa pasta de artefatos
const root = await invoke('set_artifacts_root', { 
  path: '/path/to/project' 
});

// Carrega índice de campanhas
const index = await invoke('load_index');

// Carrega detalhes da campanha
const campaign = await invoke('load_campaign', { 
  campaignId: 'camp_001' 
});

// Carrega detalhes do run
const run = await invoke('load_run', { 
  runId: 'run_001' 
});

// Lista candidatos com filtros
const candidates = await invoke('list_candidates_v2', {
  runId: 'run_001',
  candidateClass: 'validated',
  maxPbo: 0.15,
  limit: 100
});

// Carrega detalhe completo do candidato
const detail = await invoke('load_candidate_detail', {
  candidateId: 'cand_a1b2c3d4'
});

// Carrega timeseries do backtest
const backtest = await invoke('load_backtest_series', {
  candidateId: 'cand_a1b2c3d4'
});

// Inicia file watcher
await invoke('watch_artifacts');

// Invalida cache
await invoke('invalidate_cache');
```

---

## Geração de Artefatos Site

Os arquivos em `artifacts/site/` são gerados pelo Strategy Factory:

```bash
# Gera index.json e arquivos de campanha/run
combiner factory export-site --output artifacts/site
```

Ou automaticamente no final de cada campanha:

```bash
combiner factory run --campaign configs/momentum.toml --export-site
```

---

## Localização no Código

| Componente | Localização |
|------------|-------------|
| Persistência SCG | `combiner_engine/src/persistence.rs` |
| Reports backtester | `backtester_reports/src/lib.rs` |
| Factory artifacts | `combiner_cli/src/commands/factory.rs` |
| Dashboard backend | `dashboard/src-tauri/src/lib.rs` |
| Dashboard state | `dashboard/src/stores/dataStore.ts` |
| Site export | `combiner_cli/src/commands/site_export.rs` |
