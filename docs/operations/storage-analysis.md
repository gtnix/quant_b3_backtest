# Análise de Armazenamento SCG

**Versão**: 1.0.0  
**Última Atualização**: 2026-01-04  
**Status**: Diagnóstico Completo

---

## Sumário Executivo

Uma campanha SCG de **5 minutos** consome aproximadamente **6.7 GB** de espaço em disco. Este documento analisa as causas raiz desse consumo e fornece dados para tomada de decisão sobre otimizações.

---

## Métricas de Referência

### Campanha de 5 Minutos (`scg_5min_maxpower`)

| Métrica | Valor |
|---------|-------|
| **Espaço total consumido** | 6.7 GB |
| **Backtests executados** | 96,995 |
| **Estratégias no Hall of Fame** | 1,000 |
| **Estratégias validadas (Stage B)** | 51 |
| **Diretórios criados** | 96,995 |
| **Arquivos criados** | 387,980 |
| **Gerações evolutivas** | 500 |
| **Tempo por backtest** | ~25ms |

---

## Breakdown do Consumo de Espaço

### Por Componente

| Componente | Tamanho Total | % do Total | Por Backtest |
|------------|---------------|------------|--------------|
| **timeseries.csv** | **5.2 GB** | **94%** | 57 KB |
| Overhead de diretórios (ext4) | 379 MB | 5.6% | 4 KB |
| trace.jsonl | 163 MB | 2.4% | 1.8 KB |
| metadata.json | 76 MB | 1.1% | 820 B |
| metrics.json | 47 MB | 0.7% | 502 B |

### Visualização

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  timeseries.csv ████████████████████████████████████████████████████ 94%   │
│  FS Overhead    ███ 5.6%                                                   │
│  trace.jsonl    █ 2.4%                                                     │
│  metadata.json  ░ 1.1%                                                     │
│  metrics.json   ░ 0.7%                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Estrutura de Arquivos por Backtest

Cada um dos 96,995 backtests gera uma pasta com 4 arquivos:

```
output/scg/run_XXXX/backtests/
└── <uuid>/                          # 96,995 pastas
    ├── metadata.json     (820 B)    # Metadados do run
    ├── metrics.json      (502 B)    # Métricas calculadas
    ├── timeseries.csv    (57 KB)    # Série temporal diária ← 94% do espaço
    └── trace.jsonl       (1.8 KB)   # Log de execução
```

---

## Análise Detalhada: timeseries.csv

### Estrutura do Arquivo

```csv
date,equity,drawdown,exposure,vol_exante,vol_expost,dividend_cashflow,dividend_cumulative
2020-01-02,1000000.00,0.000000,0.131745,,,,
2020-01-03,998317.675,-0.001682,0.131745,,,,
...
2024-12-30,1101803.295,-0.048829,0.131737,,,,
```

### Características

| Característica | Valor |
|----------------|-------|
| Linhas por arquivo | 1,245 (dias de trading) |
| Colunas | 8 |
| Colunas **sempre vazias** | 4 (`vol_exante`, `vol_expost`, `dividend_cashflow`, `dividend_cumulative`) |
| Caracteres médios por linha | 44 |
| Formato | Texto CSV (não comprimido) |

### Ineficiência do Formato Texto

| Dado | Formato Texto | Formato Binário | Overhead |
|------|---------------|-----------------|----------|
| `1101803.295` | 11 bytes | 4 bytes (float32) | 2.75x |
| `-0.048829` | 9 bytes | 4 bytes (float32) | 2.25x |
| `2020-01-02` | 10 bytes | 2 bytes (days offset) | 5x |

### Teste de Compressão

| Formato | Tamanho | Redução |
|---------|---------|---------|
| CSV original | 57 KB | - |
| CSV + gzip | 16 KB | 71% |
| CSV + zstd | 16 KB | 71% |

---

## Análise: Overhead de Filesystem

### O Problema

O sistema cria um **diretório separado** para cada backtest executado.

| Métrica | Valor |
|---------|-------|
| Diretórios criados | 96,995 |
| Inodes consumidos | 484,975 (5 por backtest) |
| Tamanho mínimo por diretório (ext4) | 4 KB |
| **Overhead total** | **379 MB** |

### Cálculo

```
96,995 backtests × 5 inodes/backtest = 484,975 inodes
96,995 diretórios × 4 KB/diretório = 379 MB de overhead
```

---

## Análise: Dados Redundantes

### Coluna `date` Duplicada

A coluna `date` contém as **mesmas 1,245 datas** em todos os 96,995 arquivos:

| Métrica | Valor |
|---------|-------|
| Datas únicas | 1,245 |
| Arquivos com datas | 96,995 |
| Bytes por data (texto) | 10 |
| **Total duplicado** | 1.2 GB |

### Colunas Vazias

4 das 8 colunas estão **sempre vazias** neste run:

- `vol_exante`
- `vol_expost`
- `dividend_cashflow`
- `dividend_cumulative`

Cada linha tem 4 vírgulas extras = ~4 bytes × 1,245 linhas × 96,995 arquivos = **470 MB** de overhead.

---

## Análise: Uso vs. Necessidade

### Backtests Salvos vs. Utilizados

| Categoria | Quantidade | Precisa de timeseries.csv? |
|-----------|------------|---------------------------|
| Backtests executados | 96,995 | Todos salvos |
| Hall of Fame (Stage A) | 1,000 | Sim (para análise) |
| Validados (Stage B) | 51 | Sim (para produção) |
| **Descartados** | **95,944** | **Só precisavam de metrics.json** |

### Uso Real do timeseries.csv

O `timeseries.csv` é usado por:

1. **Crosscheck** (`crosscheck.rs`) - Recalcula métricas a partir do NAV
2. **Asset Attribution** (`run_campaign.rs`) - Analisa composição

Ambos os usos ocorrem **apenas para estratégias validadas** (51), não para todos os 96,995 backtests.

---

## Projeção para Campanhas Maiores

| Duração | Backtests (est.) | Espaço | Inodes |
|---------|------------------|--------|--------|
| 5 min | 97k | 6.7 GB | 485k |
| 30 min | 580k | 40 GB | 2.9M |
| 1 hora | 1.16M | 80 GB | 5.8M |
| 4 horas | 4.64M | 320 GB | 23M |
| 24 horas | 28M | 1.9 TB | 140M |

---

## Código Fonte Relacionado

### Onde os arquivos são gerados

| Arquivo | Localização |
|---------|-------------|
| timeseries.csv | `crates/backtester_cli/src/output.rs` |
| metrics.json | `crates/backtester_cli/src/output.rs` |
| metadata.json | `crates/backtester_cli/src/output.rs` |
| trace.jsonl | `crates/backtester_cli/src/output.rs` |

### Onde são consumidos

| Uso | Localização |
|-----|-------------|
| Crosscheck | `crates/combiner_cli/src/commands/factory/crosscheck.rs` |
| Asset Attribution | `crates/combiner_cli/src/commands/factory/run_campaign.rs` |
| Dashboard | `dashboard/src-tauri/src/lib.rs` |

### Configuração atual

```toml
# configs/campaigns/scg_5min_maxpower.toml
[output]
save_generations = true
save_all_genomes = false      # Existe flag para genomas
save_diversity_metrics = true
save_restart_events = true
# save_all_timeseries = ???   # NÃO EXISTE flag para timeseries
```

---

## Resumo das Causas

| Causa | Impacto | % do Total |
|-------|---------|------------|
| timeseries.csv para **todos** os backtests | 5.2 GB | 78% |
| Formato texto (não comprimido) | +3.7 GB (vs binário) | 55% |
| Overhead de 96k diretórios | 379 MB | 6% |
| Colunas vazias incluídas | 470 MB | 7% |
| Coluna date duplicada 96k vezes | 1.2 GB | 18% |

---

## Arquivos de Referência

- Configuração: `configs/campaigns/scg_5min_maxpower.toml`
- Output: `output/scg/run_b9b2cdf6f410/`
- Código gerador: `crates/backtester_cli/src/output.rs`
- Código consumidor: `crates/combiner_cli/src/commands/factory/`

