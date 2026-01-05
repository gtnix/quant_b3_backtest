# Quant B3 Backtester

Sistema de backtesting institucional de alta performance para os mercados B3 (Brasil) e US, construído em Rust, com descoberta evolutiva de estratégias via algoritmos genéticos.

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                      QUANT B3 BACKTESTER                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐  │
│   │ Backtester  │   │  Combiner   │   │     Dashboard       │  │
│   │    CLI      │   │    CLI      │   │  (Tauri + React)    │  │
│   └──────┬──────┘   └──────┬──────┘   └──────────┬──────────┘  │
│          │                 │                     │              │
│   ┌──────▼──────────────────▼─────────────────────▼──────────┐  │
│   │              ENGINE LAYER                                │  │
│   │  backtester_engine │ combiner_engine │ backtester_intel  │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Características

- **Determinismo**: Mesmos inputs → outputs bit-identical
- **Performance**: Hot path zero-alloc, até 124x speedup via SoA + SIMD
- **Precisão**: Cálculos financeiros com `rust_decimal`
- **Auditabilidade**: Rastreabilidade total de decisões e artefatos
- **SCG**: Sistema Combinador Generativo com algoritmos genéticos
- **OBFS**: Armazenamento binário otimizado (7.1x compressão, 8 KB/estratégia)
- **Dashboard**: Visualização terminal-style (Tauri + React)

---

## Quick Start

### Backtester

```bash
# Build
cargo build --release

# Executar estratégia
cargo run -p backtester_cli -- run --config configs/strategies/golden_momentum.toml

# Testes
cargo test --workspace
```

### SCG (Sistema Combinador Generativo)

```bash
# Executar evolução
cargo run -p combiner_cli -- run --config configs/scg.toml --ultra --top-k 25

# Strategy Factory - Campanhas multi-seed
cargo run -p combiner_cli -- factory run --campaign configs/campaigns/momentum.toml
```

### Dashboard

```bash
cd dashboard
npm install
npm run tauri dev
```

---

## Documentação

**Documentação completa em [`/docs`](docs/README.md)**

| Seção | Descrição |
|-------|-----------|
| [Visão Geral](docs/architecture/system-overview.md) | Arquitetura do sistema |
| [Mapa de Crates](docs/architecture/crate-map.md) | Responsabilidades de cada crate |
| [SCG Overview](docs/scg/overview.md) | Sistema Combinador Generativo |
| [Strategy Factory](docs/strategy_factory.md) | Orquestração de campanhas |
| [OBFS Integration](crates/obfs/INTEGRATION.md) | Sistema de armazenamento binário |
| [CLI Reference](docs/operations/cli-reference.md) | Comandos backtester |
| [Combiner CLI](docs/scg/cli-reference.md) | Comandos combiner |
| [Artefatos](docs/operations/artifacts.md) | Estrutura de output |

---

## Workspace Structure

| Grupo | Crate | Responsabilidade |
|-------|-------|------------------|
| **Core** | `backtester_core` | Tipos fundamentais, traits, SIMD |
| **Engine** | `backtester_engine` | UnifiedEngine (simulação) |
| **Strategy** | `backtester_strategy` | Strategy Factory (DSL) |
| **Intelligence** | `backtester_intelligence` | Entry/Exit, WFA, Performance |
| **SCG** | `combiner_core` | Genome, Fitness, SIMD metrics |
| **SCG** | `combiner_engine` | Evolution, Pareto, Hall of Fame |
| **SCG** | `combiner_cli` | CLI + Strategy Factory |
| **Storage** | `obfs` | Binary storage (Parquet + Zstd + LMDB) |
| **Data** | `market_data` | Calendars, FX, Universe |
| **Frontend** | `dashboard/` | Tauri + React dashboard |

---

## Dashboard

Terminal-style dashboard para visualização de estratégias e evolução do SCG.

**Features:**
- Dashboard Overview com KPIs
- Evolution Monitor (tempo real)
- Candidate Explorer (tabela interativa)
- Pareto Frontier 3D
- Backtest Drilldown

```bash
cd dashboard
npm run tauri dev
```

![Dashboard Preview](docs/images/dashboard-preview.png)

---

## Benchmarks

```bash
# Benchmarks de estratégia
cargo bench --bench strategy_bench

# Benchmarks de engine
cargo bench --bench scenarios_bench
```

| Cenário | Tempo | Speedup |
|---------|-------|---------|
| Standard (1K assets) | 75.7ms | 1x |
| Fast SoA (1K assets) | 1.0ms | **93x** |
| SCG Population (100) | 850ms | - |
| SCG Ultra Mode | 320ms | **2.7x** |

---

## Princípios de Design

1. **Determinism-First**: Outputs idênticos para inputs idênticos
2. **Performance-First**: Zero alocações no hot path, SIMD everywhere
3. **Hot Path Sacred**: Sem I/O, sem `dyn Trait` no loop de simulação
4. **Anti-Overfitting**: WFA, PBO, DSR integrados no SCG
5. **Institutional Quality**: Custos realistas, stress testing, gates

---

## Requisitos

- **Rust**: 1.75+
- **Node.js**: 18+ (para dashboard)
- **PostgreSQL**: Neon (opcional, para Strategy Factory)

---

## Licença

MIT
