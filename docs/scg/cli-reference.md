# CLI Reference - Combiner

**Versão**: 2.0.0  
**Última Atualização**: 2026-01-18

## Visão Geral

O `combiner` CLI fornece comandos para execução de evolução genética, validação de estratégias, auditoria institucional, e orquestração de campanhas via Strategy Factory.

O SCG também pode ser controlado via **Dashboard Cockpit** - ver [Cockpit Documentation](../dashboard/cockpit.md).

```bash
combiner <COMMAND> [OPTIONS]

Commands:
  run         Run evolution experiment
  status      Check experiment status
  export-top  Export top strategies
  validate    Validate with Walk-Forward Analysis
  extract     Extract OBFS artifacts to JSON
  audit       Institutional 6-marco audit
  factory     Strategy Factory commands
  help        Print help
```

---

## Comandos Principais

### `combiner run`

Executa um experimento de evolução genética.

```bash
combiner run --config <PATH> [OPTIONS]
```

#### Opções

| Flag | Descrição | Default |
|------|-----------|---------|
| `-c, --config <PATH>` | Caminho para arquivo de configuração SCG | Obrigatório |
| `-o, --output <PATH>` | Diretório de saída | `output/scg` |
| `-s, --seed <N>` | Seed para reprodutibilidade | Random |
| `--dry-run` | Validar configuração sem executar | false |
| `--ultra` | Modo ultra-performance (SIMD + batch) | false |
| `--top-k <N>` | Número de genomas para validar Stage B | 10 |

#### Overrides de Execução

| Flag | Descrição |
|------|-----------|
| `--execution-delay <N>` | Delay de execução em bars (0=same, 1=next) |
| `--slippage-bps <N>` | Slippage em basis points |
| `--fee-tier <TIER>` | Preset: b3-retail, b3-prime, us-retail, us-prime |
| `--stress-enabled` | Ativar stress testing |
| `--min-stress-pass <N>` | Mínimo de cenários stress para passar |
| `--bypass-costs` | Bypass custos (DEBUG ONLY) |

#### Exemplos

```bash
# Execução básica
combiner run --config configs/scg.toml

# Ultra-performance com validação
combiner run --config configs/scg.toml --ultra --top-k 25

# Com seed para reprodutibilidade
combiner run --config configs/scg.toml --seed 42

# Validação de configuração
combiner run --config configs/scg.toml --dry-run

# Com custos institucionais
combiner run --config configs/scg.toml --fee-tier b3-prime --stress-enabled
```

---

### `combiner status`

Verifica status de um experimento.

```bash
combiner status <EXPERIMENT_ID>
```

#### Exemplo

```bash
combiner status scg_20251228_143022

# Output:
# Experiment: scg_20251228_143022
# Status: Running
# Generation: 32/50
# Best Sharpe: 1.23
# Hall of Fame: 25 strategies
# ETA: ~5 minutes
```

---

### `combiner export-top`

Exporta as top estratégias de um experimento.

```bash
combiner export-top <EXPERIMENT_ID> [OPTIONS]
```

#### Opções

| Flag | Descrição | Default |
|------|-----------|---------|
| `-n <N>` | Número de estratégias | 10 |
| `-o, --output <PATH>` | Diretório de saída | Auto |
| `--include-execution-config` | Incluir config de execução nos TOMLs | false |

#### Exemplo

```bash
combiner export-top scg_20251228_143022 -n 20 -o ./my_strategies

# Output:
# Exported 20 strategies to ./my_strategies/
# - strategy_001.toml (Sharpe: 1.45)
# - strategy_002.toml (Sharpe: 1.38)
# ...
```

---

### `combiner validate`

Valida estratégias com Walk-Forward Analysis completo.

```bash
combiner validate <EXPERIMENT_ID> [OPTIONS]
```

#### Opções

| Flag | Descrição | Default |
|------|-----------|---------|
| `-k, --top-k <N>` | Número de estratégias para validar | 10 |
| `--full` | Validação completa (CPCV + PBO/DSR) | false |
| `--stress-enabled` | Incluir stress testing | false |

#### Exemplo

```bash
# Validação rápida
combiner validate scg_20251228_143022 --top-k 10

# Validação completa institucional
combiner validate scg_20251228_143022 --top-k 10 --full --stress-enabled

# Output:
# Validating top 10 strategies...
# ┌────────────────────────────────────────────────────────────┐
# │                    VALIDATION REPORT                       │
# ├──────────┬─────────┬─────────┬───────┬───────┬────────────┤
# │ Strategy │ OOS SR  │ GROSS   │ PBO   │ Stress│ Status     │
# ├──────────┼─────────┼─────────┼───────┼───────┼────────────┤
# │ 001      │ 0.89    │ 1.23    │ 0.08  │ 5/5   │ ✓ PASS     │
# │ 002      │ 0.72    │ 1.15    │ 0.12  │ 4/5   │ ✓ PASS     │
# │ 003      │ 0.45    │ 0.98    │ 0.22  │ 3/5   │ ✗ FAIL     │
# └──────────┴─────────┴─────────┴───────┴───────┴────────────┘
```

---

### `combiner extract`

Extrai artefatos OBFS (formato binário) para JSON legível.

```bash
combiner extract --pending-dir <PATH> [OPTIONS]
```

#### Opções

| Flag | Descrição | Default |
|------|-----------|---------|
| `--pending-dir <PATH>` | Diretório com arquivos .obfs | Obrigatório |
| `--run-ids <ID1,ID2>` | UUIDs específicos para extrair | Todos |
| `--output-dir <PATH>` | Diretório de saída | `./extracted` |
| `--top-n <N>` | Extrair apenas top N por Sharpe | Todos |
| `--full <UUID>` | Extrair artefato completo com timeseries | - |

#### Exemplos

```bash
# Extrair todos os artefatos pendentes
combiner extract --pending-dir artifacts/pending --output-dir ./extracted

# Extrair top 100 por Sharpe
combiner extract --pending-dir artifacts/pending --top-n 100

# Extrair artefato completo (inclui timeseries)
combiner extract --pending-dir artifacts/pending \
  --full abc12345-6789-abcd-ef01-234567890abc \
  --output-dir ./full_artifacts

# Output:
# Found 5000 artifacts to process
# Successfully read 4998 artifacts (2 errors)
# Exporting 100 artifacts to ./extracted
#   [1] abc12345... - Sharpe: 1.45, CAGR: 18.50%, Trades: 245
#   [2] def67890... - Sharpe: 1.38, CAGR: 16.20%, Trades: 189
#   ...
# Extraction complete.
```

---

### `combiner audit`

Executa auditoria institucional de 6 marcos em um run do SCG.

```bash
combiner audit --run-dir <PATH> [OPTIONS]
```

#### Opções

| Flag | Descrição | Default |
|------|-----------|---------|
| `--run-dir <PATH>` | Diretório do run SCG | Obrigatório |
| `--output <PATH>` | Diretório para relatórios | `artifacts/audits` |
| `--strict` | Tratar warnings como failures | false |
| `--stop-on-fail` | Parar no primeiro marco que falhar | false |
| `--verbose` | Output detalhado | false |

#### Marcos de Auditoria

| Marco | Nome | Verificações |
|-------|------|--------------|
| 0 | Initialization | Seeds, hashes, dates, output structure |
| 1 | Data Integrity | Anti-lookahead, universe, timestamps |
| 2 | Evolution | Diversity >10%, fitness variance, convergence |
| 3 | Validation | WFA, OOS Sharpe, PBO, DSR, stress tests |
| 4 | Promotion Gates | Thresholds, bundles completos |
| 5 | Artifacts | Provenance, files, ranking consistency |

#### Códigos de Saída

| Código | Significado |
|--------|-------------|
| 0 | Todos os marcos PASS |
| 1 | Um ou mais marcos FAIL |
| 2 | Erro (arquivos faltando, input inválido) |

#### Exemplo

```bash
# Auditoria básica
combiner audit --run-dir output/scg/run_abc123

# Auditoria estrita (warnings = failures)
combiner audit --run-dir output/scg/run_abc123 --strict --verbose

# Output:
# ======================================================================
#   AUDIT COMPLETE
# ======================================================================
#   Audit ID:     audit_79946199
#   Output:       artifacts/audits/audit_79946199
#   Duration:     12.34s
#   Final Verdict: Pass
# ======================================================================
#
#   Marcos Summary:
#     ✓ Marco 0: Initialization - Pass
#     ✓ Marco 1: DataIntegrity - Pass
#     ✓ Marco 2: Evolution - Pass
#     ✓ Marco 3: Validation - Pass
#     ✓ Marco 4: PromotionGates - Pass
#     ✓ Marco 5: Artifacts - Pass
#
#   Recomendação: APROVAR - Estratégia passou em todos os marcos
```

---

## Strategy Factory Commands

### `combiner factory init`

Inicializa uma nova campanha.

```bash
combiner factory init --name <NAME>
```

#### Exemplo

```bash
combiner factory init --name momentum_q1_2025

# Output:
# Created campaign config: configs/campaigns/momentum_q1_2025.toml
# Edit this file to configure your campaign, then run:
#   combiner factory run --campaign configs/campaigns/momentum_q1_2025.toml
```

---

### `combiner factory run`

Executa uma campanha multi-seed.

```bash
combiner factory run --campaign <PATH>
```

#### Exemplo

```bash
combiner factory run --campaign configs/campaigns/momentum_q1.toml

# Output:
# ╔══════════════════════════════════════════════════════════════╗
# ║              STRATEGY FACTORY - CAMPAIGN RUN                 ║
# ╠══════════════════════════════════════════════════════════════╣
# ║ Campaign:    momentum_q1_2025                                 ║
# ║ Config Hash: sha256:a1b2c3d4                                 ║
# ║ Seeds:       [42, 43, 44, 45, 46]                            ║
# ║ Mode:        NEW                                             ║
# ╚══════════════════════════════════════════════════════════════╝
#
# ⠋ [00:05:32] [################>-----------------------] 2/5 seeds
```

---

### `combiner factory resume`

Retoma uma campanha interrompida.

```bash
combiner factory resume --campaign <PATH>
```

---

### `combiner factory list`

Lista campanhas registradas.

```bash
combiner factory list [--tag <TAG>]
```

#### Exemplo

```bash
combiner factory list

# Output:
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                              CAMPAIGNS                                    ║
# ╠═══════════════╦════════════════════╦═════════╦═══════════╦═══════════════╣
# ║ Campaign ID   ║ Name               ║ Status  ║ Runs      ║ Created       ║
# ╠═══════════════╬════════════════════╬═════════╬═══════════╬═══════════════╣
# ║ camp_abc123   ║ momentum_q1_2025   ║ DONE    ║ 5/5       ║ 2025-12-28    ║
# ║ camp_def456   ║ value_explorer     ║ RUNNING ║ 2/5       ║ 2025-12-27    ║
# ╚═══════════════╩════════════════════╩═════════╩═══════════╩═══════════════╝
```

---

### `combiner factory show`

Mostra detalhes de campanha ou run.

```bash
combiner factory show <ID>
```

#### Exemplo

```bash
combiner factory show camp_abc123

# Output:
# ╔══════════════════════════════════════════════════════════════╗
# ║                    CAMPAIGN DETAILS                          ║
# ╠══════════════════════════════════════════════════════════════╣
# ║ ID:          camp_abc123                                     ║
# ║ Name:        momentum_q1_2025                                ║
# ║ Status:      COMPLETED                                       ║
# ║ Created:     2025-12-28T10:00:00Z                           ║
# ╠══════════════════════════════════════════════════════════════╣
# ║ Runs: 5                                                      ║
# ║ ├── run_001 (seed=42) - DONE - Top Sharpe: 1.23             ║
# ║ ├── run_002 (seed=43) - DONE - Top Sharpe: 1.18             ║
# ║ ├── run_003 (seed=44) - DONE - Top Sharpe: 1.31             ║
# ║ ├── run_004 (seed=45) - DONE - Top Sharpe: 1.15             ║
# ║ └── run_005 (seed=46) - DONE - Top Sharpe: 1.27             ║
# ╚══════════════════════════════════════════════════════════════╝
```

---

### `combiner factory compare`

Compara candidatos entre múltiplos runs.

```bash
combiner factory compare --runs <RUN1,RUN2,...> [--top <N>]
```

#### Exemplo

```bash
combiner factory compare --runs run_001,run_002,run_003 --top 5

# Output:
# ╔════════════════════════════════════════════════════════════════════════════════════╗
# ║                                CANDIDATE COMPARISON                                 ║
# ╠════════════════╦════════════════╦═══════╦═══════════╦═══════════╦═══════╦══════════╣
# ║ Run ID         ║ Candidate ID   ║ Rank  ║ OOS SR    ║ GROSS SR  ║ PBO   ║ Stress   ║
# ╠════════════════╬════════════════╬═══════╬═══════════╬═══════════╬═══════╬══════════╣
# ║ run_001        ║ cand_abc123    ║ 1     ║ 0.89      ║ 1.23      ║ 0.08  ║ 5/5      ║
# ║ run_003        ║ cand_def456    ║ 1     ║ 0.87      ║ 1.31      ║ 0.09  ║ 5/5      ║
# ║ run_002        ║ cand_ghi789    ║ 1     ║ 0.82      ║ 1.18      ║ 0.11  ║ 4/5      ║
# ╚════════════════╩════════════════╩═══════╩═══════════╩═══════════╩═══════╩══════════╝
```

---

### `combiner factory promote`

Promove candidatos para próximo estágio.

```bash
combiner factory promote [OPTIONS]
```

#### Opções

| Flag | Descrição |
|------|-----------|
| `--run <ID>` | Promover de um run específico |
| `--campaign <ID>` | Promover de toda a campanha |
| `--top <N>` | Número de candidatos (default: 3) |
| `--stage <STAGE>` | Estágio: research, candidate, paper |
| `--force` | Forçar re-promoção |

#### Exemplo

```bash
combiner factory promote --run run_001 --top 3 --stage candidate

# Output:
# Promoting top 3 candidates from run_001...
# ✓ cand_abc123 → artifacts/candidates/cand_abc123/
# ✓ cand_def456 → artifacts/candidates/cand_def456/
# ✓ cand_ghi789 → artifacts/candidates/cand_ghi789/
# 
# Candidate bundles created with:
# - strategy.toml
# - execution_config.toml
# - validation_summary.json
# - provenance.json
# - replay.sh
```

---

### `combiner factory audit-data`

Auditoria de integridade dos dados.

```bash
combiner factory audit-data --campaign <PATH> [--mode fast|strict]
```

---

### `combiner factory export-top`

Exporta top N candidatos com ranking determinístico.

```bash
combiner factory export-top --run <ID> --top <N> [--format json,csv]
```

#### Exemplo

```bash
combiner factory export-top --run run_001 --top 1000 --format json,csv

# Output:
# Exporting top 1000 candidates from run_001...
# Saved to:
# - artifacts/top_candidates/run_001/top1000.json
# - artifacts/top_candidates/run_001/top1000.csv
```

---

## Configuração SCG

### Exemplo Completo

```toml
[evolution]
population_size = 100
max_generations = 50
tournament_size = 3
crossover_rate = 0.85
mutation_rate = 0.05
elitism_rate = 0.1
hall_of_fame_size = 25

[fitness]
objectives = ["cagr", "sharpe", "max_drawdown"]
weights = [1.0, 1.5, 1.0]

[fitness.penalties]
low_trades_threshold = 30
low_trades_penalty = 0.5
extreme_turnover_threshold = 5.0
extreme_turnover_penalty = 0.1

[dataset]
market = "BR"
start_date = "2018-01-01"
end_date = "2024-12-01"
universe = "ibov"

[validation]
enabled = true
top_k = 10

[validation.wfa]
num_folds = 5
is_ratio = 0.6
purge_days = 5

[validation.thresholds]
min_oos_sharpe_net = 0.5
max_pbo = 0.15
min_stress_pass = 4
```

---

## Variáveis de Ambiente

| Variável | Descrição |
|----------|-----------|
| `NEON_DATABASE_URL` | Connection string PostgreSQL (Strategy Factory) |
| `FACTORY_JSON_LOGS` | Ativar logs JSON estruturados |
| `RUST_LOG` | Nível de log (combiner=info, combiner=debug) |

---

## Dashboard Cockpit

O SCG pode ser controlado via Dashboard Cockpit, que fornece uma interface gráfica para:

- **Presets** - Configurações pré-definidas (Rapid, Institutional, Exhaustive)
- **Compute Budget** - Controle de tempo e workers
- **Risk Gates** - Thresholds de validação configuráveis
- **Ranking** - Métodos de ordenação de candidatos
- **Live Progress** - Monitoramento em tempo real
- **Results** - Tabela de top strategies

### Equivalência CLI ↔ Cockpit

| Cockpit Preset | Equivalente CLI |
|----------------|-----------------|
| Rapid (3min) | `combiner run --config scg.toml --seed 42` com `max_runtime=180` |
| Institutional (15min) | `combiner factory run --campaign institutional.toml` |
| Exhaustive (1h) | `combiner factory run --campaign exhaustive.toml` |

### API Server

Em Browser Mode, o Cockpit comunica com o SCG via API Server (`server.js`):

```bash
# API spawna o combiner
POST /api/scg/start
  → spawn combiner factory run --campaign <config>

# Progress é parseado do stdout
GET /api/scg/progress/:runId
  → retorna generation, sharpe, candidates
```

Ver [API Server Documentation](../dashboard/api-server.md) para detalhes.

---

## Localização no Código

- Crate: `combiner_cli`
- Entry point: `crates/combiner_cli/src/main.rs`
- Commands: `crates/combiner_cli/src/commands/`




