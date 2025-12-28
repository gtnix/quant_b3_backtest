# Mapa de Crates

**Versão**: 2.0.0  
**Última Atualização**: 2025-12-28

## Workspace Overview

O backtester é organizado em um workspace Rust com crates especializados.

---

## Diagrama de Dependências

```
                           backtester_cli
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
           backtester_strategy   │   backtester_tests
                    │            │            │
                    │            │            │
                    └────────────┼────────────┘
                                 │
                                 ▼
                      backtester_intelligence
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
          backtester_engine  backtester_portfolio  backtester_reports
                    │            │            │
                    └────────────┼────────────┘
                                 │
                                 ▼
                         backtester_core
                                 │
                                 ▼
                          backtester_io
```

---

## Crates do Workspace

### `backtester_core`

**Responsabilidade**: Tipos fundamentais e traits base.

**Símbolos Principais**:
- `AssetId`, `OrderId`, `FillId`, `Timestamp`
- `Bar` (OHLCV)
- `MarketEvent`, `SignalEvent`, `OrderEvent`, `FillEvent`
- `Strategy` trait
- `ExecutionModel` trait
- `BacktestConfig`, `ExecutionConfig`

**Localização**: `crates/backtester_core/src/lib.rs`

---

### `backtester_io`

**Responsabilidade**: Data ingestion e normalização.

**Símbolos Principais**:
- Data loaders (CSV, API)
- Cache management
- Normalização de datas/preços

**Localização**: `crates/backtester_io/src/`

---

### `backtester_engine`

**Responsabilidade**: Motor de simulação.

**Símbolos Principais**:
- `UnifiedEngine` (CANÔNICO)
- `SimulationEngine` (DEPRECATED)
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

**Responsabilidade**: Modelos de execução.

**Símbolos Principais**:
- `SlippageModel` (Constant, VolumeLinear, Volatility)
- `CostModel` (fixed, commission, emolument B3)
- `LiquidityModel` (max participation, partial fills)

**Localização**: `crates/backtester_execution/src/`

---

### `backtester_reports`

**Responsabilidade**: Geração de reports.

**Símbolos Principais**:
- `NavHistory`
- `BacktestResult`
- SIMD-optimized metrics

**Localização**: `crates/backtester_reports/src/`

---

### `backtester_intelligence`

**Responsabilidade**: Lógica de inteligência de seleção.

**Módulos**:
| Módulo | Responsabilidade |
|--------|------------------|
| `entry/` | EntryEngine, gating filters, universe eligibility |
| `exit/` | ExitEngine, stop-loss, take-profit, trailing |
| `orchestrator/` | RebalanceOrchestrator, order netting |
| `performance/` | PerformanceEngine, métricas, atribuição |
| `dividends/` | Dividend processing |
| `fx/` | FxProvider, multi-currency |
| `currency/` | Currency, Money, FxPair types |

**Localização**: `crates/backtester_intelligence/src/`

---

### `backtester_strategy`

**Responsabilidade**: Strategy Factory (DSL declarativa).

**Módulos**:
| Módulo | Responsabilidade |
|--------|------------------|
| `blocks/` | 19 blocos de estratégia |
| `compositor.rs` | Executor de pipeline |
| `registry.rs` | BlockRegistry |
| `compiled.rs` | CompiledStrategy, SymbolTable |
| `fast_context.rs` | CandidatesSoA, PreallocBuffers |
| `experiment/` | Runner, Metrics, Artifacts, Comparator |

**Localização**: `crates/backtester_strategy/src/`

---

### `backtester_cli`

**Responsabilidade**: Interface de linha de comando.

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
- `determinism` - Testes de determinismo
- `invariants` - Invariantes do sistema
- `anti_look_ahead` - Prevenção de look-ahead bias
- `scenarios_bench` - Benchmarks de cenários

**Localização**: `crates/backtester_tests/`

---

## Perfis de Build

```toml
[profile.release]
opt-level = 3
lto = true
codegen-units = 1

[profile.bench]
inherits = "release"
debug = true
```

---

## Comandos de Build

```bash
# Build completo (release)
cargo build --release

# Build crate específico
cargo build -p backtester_engine --release

# Check sem compilar
cargo check --workspace

# Testes
cargo test --workspace
cargo test -p backtester_strategy
```

