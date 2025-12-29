# Decisões de Design (ADRs)

**Versão**: 2.0.0  
**Última Atualização**: 2025-12-28

Este documento registra as decisões arquiteturais significativas (Architecture Decision Records).

---

## ADR-001: UnifiedEngine como Engine Canônico

**Status**: Aceito  
**Data**: 2025-12

### Contexto

Existiam múltiplas implementações de engine de simulação:
- `SimulationEngine`: Original, usa f64
- `Engine`: Alternativa simplificada
- `IntelligentEngine`: Com features de inteligência

### Decisão

Unificar em `UnifiedEngine` como implementação canônica, deprecando `SimulationEngine`.

### Consequências

- ✅ Uma única fonte de verdade
- ✅ Precisão decimal com `rust_decimal`
- ✅ Suporte a dividendos com política anti-double-count
- ❌ Migração necessária para código legado

### Evidência

```rust
// backtester_engine/src/lib.rs:292
#[deprecated(since = "0.2.0", note = "Use UnifiedEngine instead for dividend support and Decimal precision")]
pub struct SimulationEngine { ... }
```

---

## ADR-002: Anti-Double-Count para Dividendos

**Status**: Aceito  
**Data**: 2025-12

### Contexto

Ao usar preços ajustados para valuation E adicionar dividendos como cashflow, ocorre double-counting.

### Decisão

Separar tipos de preço:
- **Signals/Indicators**: Preços ajustados (continuidade)
- **Valuation**: Preços raw (dividendos via cashflow)

### Consequências

- ✅ Retorno econômico correto
- ✅ Audit trail claro
- ❌ Complexidade adicional (dois tipos de preço)

### Evidência

```rust
// backtester_engine/src/unified.rs
pub enum PriceType {
    Signals,    // Adjusted
    Valuation,  // Raw
}
```

---

## ADR-003: SoA (Structure-of-Arrays) para Hot Path

**Status**: Aceito  
**Data**: 2025-12

### Contexto

O hot path (seleção + sizing) era gargalo de performance devido a:
- Alocações por iteração
- Cache misses em AoS layout

### Decisão

Implementar `CandidatesSoA` e `FastContext` com:
- Dados contíguos por campo (cache-friendly)
- `PreallocBuffers` para zero alocações
- Funções `fast_*` otimizadas

### Consequências

- ✅ 93-124x speedup medido
- ✅ Cache-friendly iteration
- ❌ Código duplicado (standard vs fast)
- ❌ Nem todos os blocks suportam fast mode

### Evidência

```rust
// backtester_strategy/src/fast_context.rs
pub struct CandidatesSoA {
    prices: Vec<f64>,      // Contíguos
    volatilities: Vec<f64>, // Contíguos
    momentums: Vec<f64>,    // Contíguos
    valid: BitVec,
}
```

---

## ADR-004: 252 Dias de Trading/Ano

**Status**: Aceito  
**Data**: 2025-12

### Contexto

Diferentes convenções existem para anualização:
- 252: Padrão equity markets
- 260: Business days
- 365: Calendar days

### Decisão

Usar 252 dias como constante centralizada.

### Consequências

- ✅ Consistência com indústria
- ✅ Comparabilidade com benchmarks
- ❌ Pode diferir de outras jurisdições

### Evidência

```rust
// backtester_strategy/src/experiment/metrics.rs:63
pub const TRADING_DAYS_PER_YEAR: f64 = 252.0;
```

---

## ADR-005: Population Std Dev para Volatilidade

**Status**: Aceito  
**Data**: 2025-12

### Contexto

Duas opções para cálculo de volatilidade:
- Population (divisor N): Mais estável
- Sample (divisor N-1): Teoricamente correto para amostras

### Decisão

Default para Population, com opção configurável.

### Consequências

- ✅ Estabilidade para amostras pequenas
- ✅ Configurável via `VolatilityType`
- ❌ Pode subestimar volatilidade "real"

### Evidência

```rust
// backtester_strategy/src/experiment/metrics.rs
pub enum VolatilityType {
    #[default]
    Population,
    Sample,
}
```

---

## ADR-006: Capping de Ratios Infinitos

**Status**: Aceito  
**Data**: 2025-12

### Contexto

Métricas como Sortino (sem retornos negativos) ou Profit Factor (sem perdas) retornariam infinito.

### Decisão

Cap em `MAX_RATIO_VALUE = 999.99`.

### Consequências

- ✅ JSON serializável
- ✅ Comparações possíveis
- ❌ Informação perdida (era "infinito")

### Evidência

```rust
// backtester_strategy/src/experiment/metrics.rs:76
pub const MAX_RATIO_VALUE: f64 = 999.99;
```

---

## ADR-007: Ex-Date para Dividendos

**Status**: Aceito  
**Data**: 2025-12

### Contexto

Dividendos podem ser creditados em:
- Ex-date: Data que preço ajusta
- Payment date: Data que dinheiro entra na conta

### Decisão

Creditar em ex-date.

### Consequências

- ✅ Alinhado com séries de preço ajustado
- ✅ Simplifica lógica (sem "dividend receivable")
- ❌ Diferente da realidade de cashflow

### Evidência

```rust
// backtester_engine/src/unified.rs
// Dividends credited on ex_date
if date == div.ex_date { ... }
```

---

## ADR-008: Survivorship Bias via Universe Eligibility

**Status**: Aceito  
**Data**: 2025-12

### Contexto

Backtests podem sofrer de survivorship bias ao incluir ativos que não existiam em datas passadas.

### Decisão

Implementar `UniverseRangeProvider` (V1) e `TimelineEligibilityProvider` (V2):
- V1: min_date/max_date de dados
- V2: listing_date/delisting_date de eventos

### Consequências

- ✅ Previne "ressurreição" de ativos
- ✅ Previne "viagem no tempo"
- ❌ Dados de IPO/delisting podem ser imprecisos

### Evidência

```rust
// backtester_intelligence/src/entry/universe_range.rs
pub struct UniverseRangeProvider { ... }
```

---

## ADR-009: TOML para Configuração de Estratégias

**Status**: Aceito  
**Data**: 2025-12

### Contexto

Estratégias precisam ser configuráveis sem modificar código.

### Decisão

DSL declarativa em TOML com:
- Seção `[strategy]` para metadata
- Array `[[pipeline]]` para steps
- Seção `[rebalance]` para frequência
- Seção `[constraints]` para limites

### Consequências

- ✅ Não requer recompilação
- ✅ Versionável
- ✅ Human-readable
- ❌ Limitado a combinações predefinidas

### Evidência

```toml
[strategy]
id = "momentum_v1"

[[pipeline]]
type = "selection"
block_id = "momentum"
params = { top_pct = 20 }
```

---

## ADR-010: Artefatos Padronizados por Run

**Status**: Aceito  
**Data**: 2025-12

### Contexto

Experimentos precisam de outputs rastreáveis e comparáveis.

### Decisão

Cada run gera pasta com:
- `metadata.json`: Contexto e configuração
- `metrics.json`: Métricas calculadas
- `timeseries.csv`: Série temporal
- `trace.jsonl`: Trace de execução

### Consequências

- ✅ Reprodutibilidade
- ✅ Comparabilidade entre runs
- ✅ Audit trail
- ❌ Espaço em disco

### Evidência

```
output/experiments/<run_id>/
├── metadata.json
├── metrics.json
├── timeseries.csv
└── trace.jsonl
```




