# Visão Geral do Sistema

**Versão**: 2.0.0  
**Última Atualização**: 2025-12-28

## Resumo Executivo

O Quant B3 Backtester é um sistema de simulação de estratégias quantitativas de alta performance para o mercado brasileiro (B3). Construído em Rust, prioriza determinismo, performance e precisão em cálculos financeiros.

---

## Arquitetura em Camadas

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Layer                                │
│                    backtester_cli                                │
│         (run, run-batch, compare, generate-catalog)             │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    Strategy Factory                              │
│                  backtester_strategy                             │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐     │
│  │ Compositor  │  │ BlockRegistry │  │ ExperimentRunner    │     │
│  │ (DSL exec)  │  │ (19 blocks)   │  │ (artifacts)         │     │
│  └─────────────┘  └──────────────┘  └─────────────────────┘     │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                   Intelligence Layer                             │
│                backtester_intelligence                           │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐     │
│  │ EntryEngine │  │ ExitEngine   │  │ PerformanceEngine   │     │
│  │ (gating)    │  │ (stop/profit)│  │ (métricas)          │     │
│  └─────────────┘  └──────────────┘  └─────────────────────┘     │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐     │
│  │ Orchestrator│  │ FxProvider   │  │ UniverseProvider    │     │
│  │ (netting)   │  │ (multi-curr) │  │ (survivorship)      │     │
│  └─────────────┘  └──────────────┘  └─────────────────────┘     │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    Engine Layer                                  │
│                  backtester_engine                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   UnifiedEngine                          │    │
│  │  - Precisão Decimal (rust_decimal)                       │    │
│  │  - Anti-double-count (dividends)                         │    │
│  │  - Dual-price bars (signals vs valuation)                │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │         SimulationEngine (DEPRECATED)                    │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     Core Layer                                   │
│   backtester_core | backtester_portfolio | backtester_execution │
│                                                                  │
│   Tipos fundamentais: AssetId, OrderId, Bar, Events             │
│   Portfolio: Positions, Cash, PnL, Drawdown                     │
│   Execution: Slippage, CostModel, LiquidityModel                │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      I/O Layer                                   │
│                    backtester_io                                 │
│                                                                  │
│   Data loading, normalization, caching                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Componentes Principais

### UnifiedEngine (Canônico)

Engine de simulação unificado que substitui o deprecated `SimulationEngine`.

**Localização**: `backtester_engine/src/unified.rs`

**Características**:
- Precisão decimal via `rust_decimal`
- Suporte a dividendos com política anti-double-count
- Separação de preços: adjusted (signals) vs raw (valuation)
- Determinismo garantido

### Strategy Factory

DSL declarativa para composição de estratégias via TOML.

**Localização**: `backtester_strategy/`

**Componentes**:
- `Compositor`: Executor de pipelines
- `BlockRegistry`: Registro de 19 blocos disponíveis
- `ExperimentRunner`: Executor com geração de artefatos
- `CompiledStrategy`: Estratégia pré-compilada para performance

### Experiment Orchestrator

Sistema de execução de experimentos com artefatos padronizados.

**Localização**: `backtester_strategy/src/experiment/`

**Artefatos gerados**:
- `metadata.json`: Configuração e contexto
- `metrics.json`: Métricas de performance
- `timeseries.csv`: Curva de equity
- `trace.jsonl`: Trace de execução

---

## Fluxo de Dados Principal

```
TOML Config → Compositor → Pipeline Blocks → Candidates
                                                │
                                                ▼
                                        EntryEngine (gating)
                                                │
                                                ▼
                                        ExitEngine (stops)
                                                │
                                                ▼
                                        Orchestrator (netting)
                                                │
                                                ▼
                                        UnifiedEngine (simulation)
                                                │
                                                ▼
                                        Artifacts (JSON/CSV)
```

---

## Invariantes do Sistema

### Determinismo

- **Garantia**: Mesmos inputs → mesmos outputs (bit-identical)
- **Validação**: `cargo test determinism`
- **Componentes não-determinísticos**: `run_id` (UUID), `timestamp` (UTC)

### Anti-Double-Count

- **Política**: Signals usam adjusted prices, valuation usa raw prices
- **Validação**: `UnifiedEngine::validate_anti_double_count()`
- **Erro**: `PolicyViolation` se configuração inválida

### Performance

- **Hot path**: Zero alocações após warmup
- **Speedup**: 93-124x via SoA + PreallocBuffers
- **Validação**: `cargo bench --bench strategy_bench`

---

## Localização no Código

| Componente | Crate | Arquivo Principal |
|------------|-------|-------------------|
| UnifiedEngine | `backtester_engine` | `src/unified.rs` |
| Compositor | `backtester_strategy` | `src/compositor.rs` |
| BlockRegistry | `backtester_strategy` | `src/registry.rs` |
| ExperimentRunner | `backtester_strategy` | `src/experiment/runner.rs` |
| MetricsCalculator | `backtester_strategy` | `src/experiment/metrics.rs` |
| EntryEngine | `backtester_intelligence` | `src/entry/engine.rs` |
| ExitEngine | `backtester_intelligence` | `src/exit/engine.rs` |
| PerformanceEngine | `backtester_intelligence` | `src/performance/engine.rs` |
| CLI | `backtester_cli` | `src/main.rs` |






