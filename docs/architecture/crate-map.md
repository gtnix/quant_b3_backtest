# Mapa de Crates

**Versão**: 4.0.0  
**Última Atualização**: 2026-01-18

## Workspace Overview

O sistema é organizado em um workspace Rust com 17 crates especializados, divididos em três subsistemas:

1. **Backtester** - Motor de simulação de estratégias
2. **Combiner (SCG)** - Sistema Combinador Generativo para descoberta evolutiva

---

## Diagrama de Dependências

```
                                    ORQUESTRAÇÃO
                    ┌─────────────────────┬─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
            backtester_cli          combiner_cli         market_data
                    │                     │                     │
                    │                     │                     │
        ┌───────────┼───────────┐         │                     │
        │           │           │         │                     │
        ▼           ▼           ▼         ▼                     │
backtester_strategy │  backtester_tests   │                     │
        │           │           │         │                     │
        │           │           │         │                     │
        └───────────┼───────────┘         │                     │
                    │                     │                     │
                    ▼                     ▼                     │
        backtester_intelligence    combiner_runner              │
                    │                     │                     │
        ┌───────────┼───────────┐         │                     │
        │           │           │         ▼                     │
        ▼           ▼           ▼   combiner_engine             │
backtester_engine   │   backtester_reports    │                 │
        │  backtester_portfolio   │           │                 │
        │           │             │           ▼                 │
        │  backtester_execution   │     combiner_core           │
        │           │             │           │                 │
        └───────────┼─────────────┘           │                 │
                    │                         │                 │
                    ▼                         │                 │
            backtester_core ◄─────────────────┘                 │
                    │                                           │
                    ▼                                           │
             backtester_io ◄────────────────────────────────────┘
```

---

## Crates do Backtester

### `backtester_core`

**Responsabilidade**: Tipos fundamentais e traits base.

**Símbolos Principais**:
- `AssetId`, `OrderId`, `FillId`, `Timestamp`
- `Bar` (OHLCV)
- `MarketEvent`, `SignalEvent`, `OrderEvent`, `FillEvent`
- `Strategy` trait
- `ExecutionModel` trait
- `BacktestConfig`, `ExecutionConfig`
- `simd` - Módulo com funções SIMD otimizadas

**Localização**: `crates/backtester_core/src/lib.rs`

---

### `backtester_io`

**Responsabilidade**: Data ingestion e normalização.

**Símbolos Principais**:
- `MmapLoader` - Memory-mapped file loading
- Data loaders (CSV, API)
- Cache management
- Normalização de datas/preços

**Localização**: `crates/backtester_io/src/`

---

### `backtester_engine`

**Responsabilidade**: Motor de simulação.

**Símbolos Principais**:
- `UnifiedEngine` (CANÔNICO)
- `ParallelEngine` - Execução paralela multi-asset
- `Rebalancer` - Lógica de rebalanceamento
- `MarketState`, `OrderRouter`
- `DualPriceBar`, `PriceType`
- `DividendEvent`, `DividendIndex`

**Localização**: `crates/backtester_engine/src/`

**Nota**: `SimulationEngine` está deprecated desde v0.2.0. Use `UnifiedEngine`.

---

### `backtester_portfolio`

**Responsabilidade**: Estado do portfólio.

**Símbolos Principais**:
- `Portfolio`
- `Position`, `Trade`
- PnL tracking
- Drawdown calculation

**Localização**: `crates/backtester_portfolio/src/`

---

### `backtester_execution`

**Responsabilidade**: Modelos de execução e custos.

**Símbolos Principais**:
- `SlippageModel` (Constant, VolumeLinear, Volatility)
- `CostModel` (fixed, commission, emolument B3)
- `LiquidityModel` (max participation, partial fills)
- `ExecutionGates` - Controles institucionais
- `StressTest` - Testes de stress de execução

**Módulos**:
| Módulo | Responsabilidade |
|--------|------------------|
| `config.rs` | Configuração de execução |
| `cost_report.rs` | Relatório detalhado de custos |
| `gates.rs` | Gates institucionais |
| `stress.rs` | Stress testing |

**Localização**: `crates/backtester_execution/src/`

---

### `backtester_reports`

**Responsabilidade**: Geração de reports e métricas.

**Símbolos Principais**:
- `NavHistory` - Histórico de NAV com drawdowns
- `BacktestResult` - Resultado completo com todas as métricas
- `BacktestReport` - Sumário do backtest
- `RunManifest` - Audit trail
- `ResultsCalculator` - Cálculo de métricas SIMD

**Localização**: `crates/backtester_reports/src/`

---

### `backtester_intelligence`

**Responsabilidade**: Lógica de inteligência de seleção e decisão.

**Módulos**:
| Módulo | Responsabilidade |
|--------|------------------|
| `entry/` | EntryEngine, gating filters, universe eligibility |
| `exit/` | ExitEngine, stop-loss, take-profit, trailing |
| `orchestrator/` | RebalanceOrchestrator, order netting |
| `performance/` | PerformanceEngine, métricas, atribuição |
| `monitoring/` | Monitoring e alertas |
| `walkforward/` | Walk-Forward Analysis |
| `dividends/` | Dividend processing |
| `fx/` | FxProvider, multi-currency |
| `currency/` | Currency, Money, FxPair types |
| `filters/` | Filtros de universo |
| `risk_free.rs` | Taxa livre de risco |
| `scorer.rs` | Scoring de ativos |

**Localização**: `crates/backtester_intelligence/src/`

---

### `backtester_strategy`

**Responsabilidade**: Strategy Factory (DSL declarativa).

**Módulos**:
| Módulo | Responsabilidade |
|--------|------------------|
| `blocks/` | 19 blocos de estratégia (selection, entry, exit, sizing) |
| `compositor.rs` | Executor de pipeline |
| `registry.rs` | BlockRegistry |
| `compiled.rs` | CompiledStrategy, SymbolTable |
| `fast_context.rs` | CandidatesSoA, PreallocBuffers |
| `experiment/` | Runner, Metrics, Artifacts, Comparator |
| `config/` | Configuração de estratégias |

**Localização**: `crates/backtester_strategy/src/`

---

### `backtester_cli`

**Responsabilidade**: Interface de linha de comando do backtester.

**Comandos**:
- `run` - Executar estratégia única
- `run-batch` - Executar múltiplas estratégias
- `compare` - Comparar dois runs
- `compare-to-golden` - Comparar contra baseline
- `generate-catalog` - Gerar catálogo de blocos

**Localização**: `crates/backtester_cli/src/main.rs`

---

### `backtester_tests`

**Responsabilidade**: Testes de integração e invariantes.

**Suites**:
- `determinism` - Testes de determinismo (bit-identical)
- `invariants` - Invariantes do sistema
- `anti_look_ahead` - Prevenção de look-ahead bias
- `real_data_integration` - Testes com dados reais
- `scenarios_bench` - Benchmarks de cenários

**Localização**: `crates/backtester_tests/`

---

## Crates do SCG (Sistema Combinador Generativo)

### `combiner_core`

**Responsabilidade**: Tipos fundamentais para descoberta evolutiva de estratégias.

**Símbolos Principais**:
- `StrategyGenome` - Genoma completo de uma estratégia
- `BlockGene` - Gene individual (bloco + parâmetros)
- `BlockType` - Selection, Entry, Exit, Sizing
- `ParamValue` - Valores de parâmetros com ranges
- `MultiObjectiveFitness` - Fitness multi-objetivo (CAGR, Sharpe, MaxDD)
- `PopulationFitnessSoA` - Layout SoA para batch processing SIMD
- `GenomeConverter` - Genoma → TOML
- `GenomeValidator` - Validação de genomas

**Módulos**:
| Módulo | Responsabilidade |
|--------|------------------|
| `genome.rs` | StrategyGenome, BlockGene, ParamValue |
| `fitness.rs` | MultiObjectiveFitness, FitnessConfig |
| `fitness_soa.rs` | PopulationFitnessSoA, AlignedVec |
| `simd_metrics.rs` | Sharpe, MaxDD, Sortino vetorizados |
| `converter.rs` | Genoma → TOML |
| `validator.rs` | Validação de genomas |
| `param_ranges.rs` | Ranges de parâmetros |

**Localização**: `crates/combiner_core/src/`

---

### `combiner_engine`

**Responsabilidade**: Motor de evolução genética.

**Símbolos Principais**:
- `EvolutionEngine` - Motor principal de evolução
- `EvolutionConfig` - Configuração do AG
- `Population` - Gerenciamento de população
- `ParetoFrontier` - Fronteira de Pareto (NSGA-II)
- `HallOfFame` - Melhores estratégias
- `ValidatedHallOfFame` - Hall of Fame com validação institucional
- `Selection`, `Crossover`, `Mutation` - Operadores genéticos
- `GenerationStats` - Estatísticas por geração
- `StageABatchEvaluator` - Avaliação rápida em batch
- `StageBParallelValidator` - Validação paralela completa

**Módulos**:
| Módulo | Responsabilidade |
|--------|------------------|
| `engine.rs` | EvolutionEngine, UltraEvolutionResult |
| `config.rs` | EvolutionConfig |
| `population.rs` | Population management |
| `operators.rs` | Selection, Crossover, Mutation |
| `pareto.rs` | ParetoFrontier |
| `pareto_simd.rs` | Pareto ranks SIMD |
| `hall_of_fame.rs` | HallOfFame |
| `hall_of_fame_validated.rs` | ValidatedHallOfFame |
| `validation.rs` | WFA, CPCV, PBO/DSR |
| `evaluation/` | Stage A/B evaluation |
| `persistence.rs` | Persistência de experimentos |
| `report.rs` | Geração de relatórios |
| `performance_metrics.rs` | Métricas de performance do AG |

**Localização**: `crates/combiner_engine/src/`

---

### `combiner_runner`

**Responsabilidade**: Executor paralelo de backtests.

**Símbolos Principais**:
- `BatchExecutor` - Execução paralela via rayon
- `DataLoader` - Carregamento de dados para backtests
- `MetricsCollector` - Coleta de métricas
- `CacheManager` - Cache de resultados

**Módulos**:
| Módulo | Responsabilidade |
|--------|------------------|
| `executor.rs` | BatchExecutor |
| `data_loader.rs` | Carregamento de dados |
| `metrics.rs` | Coleta de métricas |
| `cache.rs` | Cache de resultados |

**Localização**: `crates/combiner_runner/src/`

---

### `combiner_cli`

**Responsabilidade**: Interface de linha de comando do SCG.

**Comandos Principais**:
| Comando | Descrição |
|---------|-----------|
| `run` | Executar evolução |
| `status` | Verificar status de experimento |
| `export-top` | Exportar top estratégias |
| `validate` | Validar com Walk-Forward |
| `extract` | Extrair artefatos OBFS para JSON |
| `audit` | Auditoria institucional 6-marcos |

**Subcomandos Factory**:
| Comando | Descrição |
|---------|-----------|
| `factory init` | Inicializar campanha |
| `factory run` | Executar campanha multi-seed |
| `factory resume` | Retomar campanha |
| `factory list` | Listar campanhas |
| `factory show` | Detalhes de campanha/run |
| `factory compare` | Comparar candidatos |
| `factory promote` | Promover candidatos |
| `factory audit-data` | Auditoria de integridade de dados |
| `factory export-top` | Exportar top N candidatos |

**Localização**: `crates/combiner_cli/src/`

---

## Crate de Dados

### `market_data`

**Responsabilidade**: Acesso a dados de mercado, calendários e FX.

**Módulos**:
| Módulo | Responsabilidade |
|--------|------------------|
| `calendar/` | Calendários de trading (B3, US) |
| `fx_loader.rs` | Carregamento de taxas FX |
| `interest_rates.rs` | Taxas de juros |
| `universe_gate.rs` | Filtros de universo |
| `inventory.rs` | Inventário de dados |
| `validator.rs` | Validação de dados |
| `audit_integrity.rs` | Auditoria de integridade |
| `ingest.rs` | Ingestão de dados |
| `reports.rs` | Relatórios de dados |

**Binários**:
- `market_data` - CLI para gestão de dados
- `calendar_builder` - Construtor de calendários
- `integrity_checker` - Verificador de integridade

**Localização**: `crates/market_data/src/`

---

## Crate de Storage

### `obfs`

**Responsabilidade**: Sistema de armazenamento binário de alta performance para artefatos de backtest.

**Características**:
- Two-phase write strategy (concurrent-safe)
- Compressão 7.1x via Parquet + Zstd
- 8 KB/estratégia média
- XXH3 + BLAKE3 integrity validation
- LMDB metadata store

**Módulos**:
| Módulo | Responsabilidade |
|--------|------------------|
| `types.rs` | Core data structures (rkyv-compatible) |
| `writer.rs` | Write path with auto-rotation |
| `reader.rs` | Mmap read with validation |
| `compression.rs` | Delta + Zstd pipeline |
| `timeseries.rs` | Parquet columnar storage |
| `pending_store.rs` | Phase 1: Isolated pending storage |
| `consolidator.rs` | Phase 2: Streaming consolidation |
| `store/` | LMDB-based metadata |
| `adapters/` | Project artifact converters |

**Localização**: `crates/obfs/src/`

Ver [OBFS Integration Guide](../../crates/obfs/INTEGRATION.md) para documentação completa.

---

## Crate de Validação

### `backtester_validation`

**Responsabilidade**: Golden tests, crosscheck e validação de artefatos.

**Símbolos Principais**:
- `GoldenTest` - Testes contra baselines conhecidos
- `Crosscheck` - Validação cruzada de métricas
- `ArtifactValidator` - Validação de estrutura de artefatos

**Localização**: `crates/backtester_validation/src/`

---

## Perfis de Build

```toml
[profile.release]
lto = "fat"           # Full LTO para máxima otimização
codegen-units = 1     # Single codegen unit
panic = "abort"       # Sem overhead de unwinding
strip = "symbols"     # Binário menor
opt-level = 3         # Otimização máxima

[profile.bench]
inherits = "release"
debug = true          # Símbolos para profiling

[profile.ultra]
inherits = "release"
lto = "fat"
codegen-units = 1
panic = "abort"
strip = "symbols"
opt-level = 3
```

---

## Comandos de Build

```bash
# Build completo (release)
cargo build --release

# Build crate específico
cargo build -p backtester_engine --release
cargo build -p combiner_engine --release

# Build ultra-otimizado
cargo build --profile ultra

# Check sem compilar
cargo check --workspace

# Testes
cargo test --workspace
cargo test -p backtester_strategy
cargo test -p combiner_engine

# Benchmarks
cargo bench --bench strategy_bench
cargo bench --bench performance_bench
```

---

## Dashboard (Tauri Application)

### `dashboard`

**Responsabilidade**: Interface gráfica desktop para visualização de estratégias.

**Stack**:
- **Framework**: Tauri 2.x (Rust backend + Web frontend)
- **Frontend**: React 18 + TypeScript + Vite
- **Styling**: Tailwind CSS (terminal theme)
- **State**: Zustand
- **Charts**: Recharts

**Estrutura**:
```
dashboard/
├── src/                    # React frontend
│   ├── pages/              # 17 páginas
│   ├── components/         # Charts, layout, UI
│   ├── stores/             # Zustand dataStore, ompStore
│   └── lib/                # Utilities
├── server/                 # Express API Server
│   ├── routes/             # API endpoints (omp, analytics, etc)
│   └── services/           # Background services (hofSync)
├── src-tauri/              # Rust backend
│   └── src/lib.rs          # Tauri commands
└── index.html
```

**Tauri Commands**:
| Command | Descrição |
|---------|-----------|
| `set_artifacts_root` | Inicializa pasta de artefatos |
| `load_index` | Carrega índice de campanhas |
| `load_campaign` | Carrega detalhes da campanha |
| `load_run` | Carrega detalhes do run |
| `list_candidates_v2` | Lista candidatos com filtros |
| `load_candidate_detail` | Carrega candidato completo |
| `load_backtest_series` | Carrega timeseries |
| `watch_artifacts` | File watcher para hot-reload |
| `invalidate_cache` | Limpa cache |

**Páginas**:
| Página | Descrição |
|--------|-----------|
| Campaigns | Browser de campanhas e runs |
| Candidates | Tabela de candidatos com filtros |
| Backtest | Drilldown de backtest |
| Risk Analytics | VaR, CVaR, rolling metrics |
| Comparison | Comparação multi-estratégia |
| Walk-Forward | Validação OOS |
| Monte Carlo | Simulação bootstrap |
| Regime Analysis | Análise por regime de mercado |
| **Hall of Fame** | Estratégias elite promovidas |
| **Miner Control** | Controle do OMP |
| **Strategy Selector** | Seletor de estratégias |
| **Audit Report** | Relatórios de auditoria |
| Config Universe | Configuração de universo |
| Config Budget | Configuração de compute |
| Config Gates | Configuração de gates |
| Config Trading | Configuração de trading |

**Localização**: `dashboard/`

Ver [Dashboard README](../dashboard/README.md) para documentação completa.

---

## Resumo do Workspace

| Grupo | Crate | LOC (aprox) | Responsabilidade |
|-------|-------|-------------|------------------|
| Core | `backtester_core` | ~800 | Tipos, traits, SIMD |
| Core | `backtester_io` | ~400 | I/O, mmap |
| Engine | `backtester_engine` | ~1200 | Simulação |
| Engine | `backtester_portfolio` | ~600 | Portfolio |
| Engine | `backtester_execution` | ~800 | Custos, gates |
| Engine | `backtester_reports` | ~700 | Métricas |
| Strategy | `backtester_strategy` | ~2500 | DSL, blocos |
| Strategy | `backtester_intelligence` | ~4000 | Entry/Exit, WFA |
| CLI | `backtester_cli` | ~400 | CLI backtester |
| Tests | `backtester_tests` | ~800 | Testes |
| Validation | `backtester_validation` | ~600 | Golden tests |
| **SCG** | `combiner_core` | ~1500 | Genome, fitness |
| **SCG** | `combiner_engine` | ~3500 | Evolução, Pareto, Audit |
| **SCG** | `combiner_runner` | ~600 | Executor paralelo |
| **SCG** | `combiner_cli` | ~1500 | CLI + Factory + Audit |
| Data | `market_data` | ~2000 | Calendars, FX |
| **Storage** | `obfs` | ~1500 | Binary storage |
| **Dashboard** | `dashboard` | ~5000 | UI Tauri (Rust+React) |

**Total**: ~28.000 linhas de código (Rust + TypeScript)
