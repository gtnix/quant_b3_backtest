# Análise de Armazenamento SCG

**Versão**: 2.0.0  
**Última Atualização**: 2026-01-05  
**Status**: OBFS Ativo (Produção)

---

## Sumário Executivo

Com a implementação do **OBFS (Optimized Binary File System)**, o consumo de armazenamento foi reduzido em **7.1x** comparado ao formato Legacy (JSON/CSV).

| Formato | Storage/Estratégia | 176K Estratégias |
|---------|-------------------|------------------|
| **Legacy (JSON/CSV)** | 57 KB | ~10 GB |
| **OBFS (Parquet/Zstd)** | 8.01 KB | 1.4 GB |
| **Redução** | **7.1x** | **7.1x** |

---

## Benchmark de Produção (5 Horas)

Campanha overnight executada em 2026-01-05:

| Métrica | Valor |
|---------|-------|
| **Estratégias geradas** | 176,672 |
| **Runs completados** | 3/3 (100%) |
| **Storage total** | 2.1 GB |
| **Storage/estratégia** | 8.01 KB |
| **Throughput** | 210 estratégias/s |
| **Tempo/estratégia** | 4.75 ms |

### Consolidação por Run

| Run | Artifacts | Rows | Parquet | Tempo |
|-----|-----------|------|---------|-------|
| run_1a3d5a5f | 59,325 | 73.8M | 355 MB | 95s |
| run_7acb4c23 | 58,331 | 72.6M | 432 MB | 97s |
| run_393ae6ad | 59,016 | 73.4M | 370 MB | 97s |

---

## Comparativo: Legacy vs OBFS

### Storage por Componente

| Componente | Legacy | OBFS | Redução |
|------------|--------|------|---------|
| Timeseries | 57 KB (CSV) | 6 KB (Parquet) | 9.5x |
| Metadata | 820 B (JSON) | 200 B (binary) | 4.1x |
| Metrics | 502 B (JSON) | 150 B (binary) | 3.3x |
| Trace | 1.8 KB (JSONL) | 400 B (binary) | 4.5x |
| **Total** | **60 KB** | **8.01 KB** | **7.5x** |

### Overhead de Filesystem

| Formato | Arquivos/Backtest | Diretórios | Inodes |
|---------|-------------------|------------|--------|
| Legacy | 4 | 1 | 5 |
| OBFS | 1 (pending) → 0 (consolidado) | 0 | 1 → 0 |

### Visualização

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  LEGACY (57 KB/estratégia)                                                 │
│  timeseries.csv ████████████████████████████████████████████████████ 94%   │
│  FS Overhead    ███ 5.6%                                                   │
│  trace.jsonl    █ 2.4%                                                     │
│                                                                             │
│  OBFS (8.01 KB/estratégia)                                                 │
│  Parquet        ██████████ 75%                                             │
│  Metadata       ██ 2.5%                                                    │
│  Metrics        █ 1.9%                                                     │
│  Trace          █ 5%                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Projeção para Campanhas (OBFS)

| Duração | Estratégias (est.) | OBFS | Legacy |
|---------|-------------------|------|--------|
| 5 min | 59K | 473 MB | 3.4 GB |
| 30 min | 350K | 2.8 GB | 20 GB |
| 1 hora | 700K | 5.6 GB | 40 GB |
| 5 horas | 180K × 3 runs | 2.1 GB | 15 GB |
| 24 horas | 3.5M | 28 GB | 200 GB |

---

## Estrutura OBFS

### Durante Execução (Phase 1: Pending)

```
output/scg/run_XXXX/backtests/
└── pending/
    ├── <uuid1>.obfs    (1-2 KB, Zstd compressed)
    ├── <uuid2>.obfs
    └── ...             (59K arquivos temporários)
```

### Após Consolidação (Phase 2: Consolidated)

```
output/scg/run_XXXX/backtests/
├── pending/            (vazio após consolidação)
└── consolidated/
    ├── data/
    │   └── timeseries.parquet    (355-432 MB, ~59K estratégias)
    └── lmdb/
        ├── data.mdb              (76 MB, índice UUID → offset)
        └── lock.mdb
```

---

## Análise Legacy (Referência Histórica)

### Campanha de 5 Minutos (Formato Antigo)

| Métrica | Valor |
|---------|-------|
| **Espaço total consumido** | 6.7 GB |
| **Backtests executados** | 96,995 |
| **Diretórios criados** | 96,995 |
| **Arquivos criados** | 387,980 |

### Por Componente (Legacy)

| Componente | Tamanho Total | % do Total | Por Backtest |
|------------|---------------|------------|--------------|
| **timeseries.csv** | **5.2 GB** | **94%** | 57 KB |
| Overhead de diretórios (ext4) | 379 MB | 5.6% | 4 KB |
| trace.jsonl | 163 MB | 2.4% | 1.8 KB |
| metadata.json | 76 MB | 1.1% | 820 B |
| metrics.json | 47 MB | 0.7% | 502 B |

### Estrutura de Arquivos (Legacy)

```
output/scg/run_XXXX/backtests/
└── <uuid>/                          # 96,995 pastas
    ├── metadata.json     (820 B)
    ├── metrics.json      (502 B)
    ├── timeseries.csv    (57 KB)    # 94% do espaço
    └── trace.jsonl       (1.8 KB)
```

---

## Ineficiências do Legacy (Resolvidas pelo OBFS)

| Problema | Impacto Legacy | Solução OBFS |
|----------|---------------|--------------|
| Formato texto CSV | +3.7 GB overhead | Parquet binário + Zstd |
| 96k diretórios | 379 MB inode overhead | Zero diretórios |
| Colunas vazias | 470 MB overhead | Schema otimizado |
| Date duplicado 96k× | 1.2 GB | Delta encoding (u16 offset) |
| Arquivos separados | 387k arquivos | 1 Parquet consolidado |

---

## Código Fonte

### OBFS (Novo)

| Componente | Localização |
|------------|-------------|
| PendingStore | `crates/obfs/src/pending_store.rs` |
| Consolidator | `crates/obfs/src/consolidator.rs` |
| TimeSeriesStore | `crates/obfs/src/timeseries.rs` |
| Artifact Writer | `crates/backtester_strategy/src/experiment/artifacts.rs` |

### Legacy (Referência)

| Arquivo | Localização |
|---------|-------------|
| timeseries.csv | `crates/backtester_cli/src/output.rs` |
| metrics.json | `crates/backtester_cli/src/output.rs` |
| metadata.json | `crates/backtester_cli/src/output.rs` |
| trace.jsonl | `crates/backtester_cli/src/output.rs` |

---

## Configuração

```toml
# configs/campaigns/*.toml
[output]
artifact_format = "obfs"    # "obfs" ou "legacy"
save_generations = true
save_all_genomes = false
save_diversity_metrics = true
save_restart_events = true
```

---

## Comandos de Manutenção

```bash
# Verificar tamanho de uma campanha
du -sh output/scg/run_*/

# Contar arquivos pending
ls output/scg/run_*/backtests/pending/*.obfs | wc -l

# Verificar Parquet consolidado
ls -lh output/scg/run_*/backtests/consolidated/data/timeseries.parquet

# Limpar pending após consolidação (automático)
rm -rf output/scg/run_*/backtests/pending/
```
