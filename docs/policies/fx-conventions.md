# Convenções FX (Multi-Currency)

**Versão**: 2.0.0  
**Última Atualização**: 2025-12-28

## Visão Geral

O módulo FX permite medição de performance multi-currency com:
- Conversão de NAV para moeda base
- Atribuição de retorno FX (3 termos)
- Audit trail completo

---

## Convenção de Rate (CRÍTICO)

### Semântica

```
FxPair { base: USD, quote: BRL }

USD/BRL = 5.50 significa:
  - 1 USD = 5.50 BRL
  - rate_quote_per_base() retorna 5.50
  - Converter 100 USD para BRL: 100 × 5.50 = 550 BRL
```

### Conversões

| Operação | Fórmula |
|----------|---------|
| USD → BRL | `amount_usd × rate` |
| BRL → USD | `amount_brl ÷ rate` |
| Get inverse | `rate.inverse()` retorna BRL/USD |

---

## Tipos

### Currency

```rust
pub enum Currency {
    BRL,  // Real Brasileiro
    USD,  // Dólar Americano
    EUR,  // Euro
}
```

### Money

```rust
pub struct Money {
    amount: Decimal,
    currency: Currency,
}
```

**Invariantes**:
- Aritmética só entre mesma moeda
- Usar `convert_money()` para cross-currency

### FxPair

```rust
pub struct FxPair {
    base: Currency,   // "1 unidade disso"
    quote: Currency,  // "= rate unidades disso"
}
```

### FxRate

```rust
pub struct FxRate {
    pair: FxPair,
    rate: Decimal,
    date: NaiveDate,
}
```

---

## FxResolutionMethod

```rust
pub enum FxResolutionMethod {
    Direct,       // Rate encontrada para par/data exatos
    Inverse,      // Rate obtida invertendo par armazenado
    LOCF,         // Last Observation Carried Forward
    InverseLOCF,  // Ambos inverse e LOCF
    Identity,     // Mesma moeda (rate = 1)
    Triangulated, // Cross-rate (V2)
}
```

---

## Missing Data Policy

### FxMissingPolicy

| Policy | Comportamento |
|--------|---------------|
| `LastObservationCarriedForward` | Usa rate mais recente (default) |
| `ErrorOnMissing` | Requer match exato de data |
| `UseOne` | Usa rate=1 (só mesma moeda) |

### FxGapMode

| Mode | Contagem |
|------|----------|
| `CalendarDays` | Todos os dias incluindo weekends (default) |
| `BusinessDays` | Apenas dias úteis |

### FxMissingAction

| Action | Comportamento |
|--------|---------------|
| `Error` | Falha hard quando gap excedido (default) |
| `MarkIncomplete` | Warn e continua com snapshot incompleto |

### Weekend/Holiday

Com `CalendarDays` e `max_gap_days = 5`:

| Cenário | Gap | Resultado |
|---------|-----|-----------|
| Friday → Saturday | 1 | LOCF ✓ |
| Friday → Monday | 3 | LOCF ✓ |
| Thursday → Tuesday (holiday Monday) | 5 | LOCF ✓ |
| Monday → Tuesday (sem dado Monday) | 6 | Error ✗ |

---

## FX Attribution

### Decomposição (3 termos)

**Multiplicativo**:
```
(1 + R_total_B) = (1 + R_asset_L) × (1 + R_fx)
```

**Aditivo**:
```
R_total_B = R_asset + R_fx + R_interaction

onde:
  R_asset = V_L(t1) / V_L(t0) - 1
  R_fx = FX(t1) / FX(t0) - 1
  R_interaction = R_asset × R_fx
```

### Exemplo

| Métrica | Valor |
|---------|-------|
| Start | $1,000 USD @ 5.00 = R$5,000 |
| End | $1,100 USD @ 5.50 = R$6,050 |
| Asset return | 10% |
| FX return | 10% |
| Interaction | 1% |
| **Total return** | **21%** |

---

## Audit Trail

Cada lookup registra:

| Campo | Descrição |
|-------|-----------|
| `pair_requested` | Par originalmente requisitado |
| `date_requested` | Data requisitada |
| `pair_resolved` | Par efetivamente usado |
| `date_resolved` | Data do rate usado |
| `rate` | Taxa de câmbio |
| `method` | Como foi resolvido |

### Exemplo JSON

```json
{
  "fx_rates_used": [
    {
      "pair_requested": "USD/BRL",
      "date_requested": "2024-12-27",
      "pair_resolved": "USD/BRL",
      "date_resolved": "2024-12-24",
      "rate": "5.50",
      "method": "LOCF"
    }
  ]
}
```

---

## Data Pipeline

### Sources

| Source | Pares | API |
|--------|-------|-----|
| BCB | USD/BRL, EUR/BRL | `api.bcb.gov.br` |
| FRED | EUR/USD | `api.stlouisfed.org` |

### Storage

Arquivos em `cache/fx/`:

```csv
date,rate,source
2024-01-02,4.8521,BCB
2024-01-03,4.8934,BCB
```

### Comandos

```bash
# Full sync
python -m datahub_fx sync

# Incremental update
python -m datahub_fx update

# Check status
python -m datahub_fx status

# Show rates
python -m datahub_fx show USD/BRL --tail 10
```

---

## Uso

### Setup Engine com FX

```rust
use backtester_intelligence::currency::Currency;
use backtester_intelligence::fx::InMemoryFxProvider;
use backtester_intelligence::performance::{PerformanceEngine, PerformanceConfig};

let mut provider = InMemoryFxProvider::new();
provider.add_rate(FxPair::USD_BRL, date, dec!(5.50));

let config = PerformanceConfig::default()
    .with_base_currency(Currency::BRL);
let engine = PerformanceEngine::with_fx(config, capital, Arc::new(provider));
```

### Converter Money

```rust
use backtester_intelligence::fx::convert_money_with_audit;

let usd = Money::new(dec!(1000), Currency::USD);
let result = convert_money_with_audit(&usd, Currency::BRL, date, &provider, 5)?;

println!("Amount: {}", result.money);
println!("Rate: {}", result.rate_used);
println!("Method: {:?}", result.method);
```

---

## Invariantes

1. **No implicit conversion**: Toda conversão via `convert_money()` ou `FxRateProvider`
2. **Separation preserved**: Portfolios BR/US separados; consolidação via FX explícito
3. **Precision**: Todos valores monetários usam `Decimal`
4. **Complete audit trail**: Snapshots registram rates, dates e methods
5. **Decomposition verified**: `R_asset + R_fx + R_interaction = R_total`
6. **Determinism**: Mesmos inputs → outputs idênticos
7. **Schema stability**: JSON versionado

---

## Testes

```bash
# Unit tests
cargo test -p backtester_intelligence fx_unit

# Integration
cargo test -p backtester_intelligence fx_integration

# Property tests
cargo test -p backtester_intelligence fx_proptest

# Golden tests
cargo test -p backtester_intelligence fx_golden

# E2E
cargo test e2e_consolidated
cargo test e2e_fx_attribution
```

---

## Localização no Código

| Componente | Arquivo |
|------------|---------|
| Currency, Money, FxPair | `backtester_intelligence/src/currency.rs` |
| FxRate, FxProvider | `backtester_intelligence/src/fx/` |
| PerformanceEngine | `backtester_intelligence/src/performance/engine.rs` |
| FX Attribution | `backtester_intelligence/src/performance/fx_attribution.rs` |

