# Engines de Simulação

**Versão**: 2.0.0  
**Última Atualização**: 2025-12-28

## UnifiedEngine (Canônico)

O `UnifiedEngine` é o motor de simulação canônico do backtester, unificando todas as funcionalidades de simulação em uma única implementação.

### Localização no Código

- **Crate**: `backtester_engine`
- **Arquivo**: `src/unified.rs`
- **Símbolos**: `UnifiedEngine`, `UnifiedEngineConfig`, `DayResult`, `UnifiedBacktestResult`

### Características

| Feature | Descrição |
|---------|-----------|
| Precisão Decimal | Usa `rust_decimal` para cálculos financeiros |
| Dividendos | Processa dividends como cashflow em ex-date |
| Anti-Double-Count | Separa preços signals (adjusted) vs valuation (raw) |
| Determinismo | Mesmos inputs → mesmos outputs |

### Configuração

```rust
use backtester_engine::{UnifiedEngine, UnifiedEngineConfig, PriceType};
use rust_decimal_macros::dec;

let config = UnifiedEngineConfig {
    initial_capital: dec!(1_000_000),
    enable_dividends: true,
    valuation_price_type: PriceType::Valuation, // Raw prices
    ..Default::default()
};

let mut engine = UnifiedEngine::with_config(config);
```

### Tipos de Preço

```rust
pub enum PriceType {
    /// Adjusted prices for signals/indicators
    Signals,
    /// Raw prices for valuation (dividends via cashflow)
    Valuation,
}
```

| Uso | Price Type | Razão |
|-----|------------|-------|
| Signals/Indicators | `Signals` (adjusted) | Continuidade de retornos |
| Mark-to-Market | `Valuation` (raw) | Dividends entram via cashflow |
| Order Execution | `Valuation` (raw) | Preço real de mercado |

### DualPriceBar

```rust
pub struct DualPriceBar {
    pub symbol: String,
    pub date: NaiveDate,
    pub adjusted_close: Decimal,  // Para signals
    pub raw_close: Decimal,       // Para valuation
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub volume: Decimal,
}
```

### Processamento de Dividendos

```rust
// Carregar dividendos
engine.load_dividends(vec![
    DividendEvent {
        symbol: "TAEE11".to_string(),
        ex_date: NaiveDate::from_ymd_opt(2024, 3, 15).unwrap(),
        rate: dec!(0.50),
        ..Default::default()
    },
]);

// Processar cada dia
for date in trading_days {
    let day_result = engine.process_day(date, &bars, candidates);
    // day_result.dividend_cashflow contém dividendo do dia
}
```

### Validação Anti-Double-Count

```rust
// Valida configuração
engine.validate_anti_double_count()?;

// Erro se: enable_dividends == true && valuation_price_type == Signals
// Retorna: Err(PolicyViolation { ... })
```

### DayResult

```rust
pub struct DayResult {
    pub date: NaiveDate,
    pub equity: Decimal,
    pub cash: Decimal,
    pub positions_value: Decimal,
    pub dividend_cashflow: Decimal,
    pub trades_executed: usize,
    pub drawdown: Decimal,
}
```

### UnifiedBacktestResult

```rust
pub struct UnifiedBacktestResult {
    pub final_equity: Decimal,
    pub total_return: Decimal,
    pub total_dividend_cashflow: Decimal,
    pub total_trades: usize,
    pub max_drawdown: Decimal,
    pub timeseries: Vec<DayResult>,
    pub trace_events: Vec<TraceEvent>,
}
```

---

## SimulationEngine (DEPRECATED)

**⚠️ DEPRECATED desde v0.2.0**

Use `UnifiedEngine` em vez de `SimulationEngine`.

### Razões da Deprecação

1. Usa `f64` em vez de `Decimal` (imprecisão)
2. Não suporta dividendos
3. Não implementa anti-double-count

### Migração

```rust
// ANTIGO (deprecated)
use backtester_engine::SimulationEngine;
let mut engine = SimulationEngine::with_defaults(strategy, 100_000.0, 10);

// NOVO (recomendado)
use backtester_engine::{UnifiedEngine, UnifiedEngineConfig};
let config = UnifiedEngineConfig {
    initial_capital: dec!(100_000),
    ..Default::default()
};
let mut engine = UnifiedEngine::with_config(config);
```

---

## Invariantes

### Anti-Double-Count

**Invariante**: `equity_raw + Σ dividends ≈ equity_adjusted`

**Teste**: `t1_buyhold_economic_return_matches_adjusted`

### Determinismo

**Invariante**: Mesmos inputs → mesmos outputs (bit-identical)

**Teste**: `test_determinism_same_input_same_output`

### Validação de Política

**Invariante**: `enable_dividends + Signals price → PolicyViolation`

**Teste**: `t2_anti_double_count_validation`

---

## Testes Relacionados

```bash
# Testes de UnifiedEngine
cargo test -p backtester_engine unified

# Testes de integração de dividendos
cargo test -p backtester_intelligence dividend_integration

# Testes E2E de dividendos
cargo test -p backtester_strategy runner_dividend_e2e
```

