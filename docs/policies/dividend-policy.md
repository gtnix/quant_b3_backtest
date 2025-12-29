# Política de Dividendos

**Versão**: 2.0.0  
**Última Atualização**: 2025-12-28

## Política Oficial

Esta é a política oficial para tratamento de corporate actions (primariamente dividendos) no backtester, garantindo cálculo de PnL economicamente correto **sem double-counting**.

---

## Decisões de Política

### P1: Preços Ajustados para Signals ✓

**Status**: Implementado

Indicadores, filtros e seleção de ativos usam `adjustedClose` por default.

**Razão**:
- Evita gaps artificiais em sinais momentum/trend
- Garante continuidade em cálculos de retorno
- Segue padrão da indústria para análise quantitativa

**Referência**: `backtester_io/src/lib.rs`

### P2: Dividendos como Cashflow ✓

**Status**: Implementado

Dividendos entram no portfólio como eventos de cashflow explícitos no **ex-date**.

**Razão**:
- Audit trail claro
- Separa ganhos de capital de renda
- Permite tax-lot tracking

**Referência**: `backtester_intelligence/src/dividends/`

### P3: Regra Anti-Double-Count ✓ CRÍTICO

**Status**: Implementado e testado

**O Problema**:

Se usar preços ajustados para valuation E adicionar dividendos como cashflow, dividendos são contados duas vezes:
1. Via série de preços ajustados (que "absorve" dividendos)
2. Via cashflow explícito

**A Solução**:

| Uso | Tipo de Preço | Tratamento de Dividendos |
|-----|---------------|--------------------------|
| Signals/Indicators | Adjusted | Implícito no preço |
| Mark-to-Market | Raw | Cashflow explícito |
| Equity Curve | Raw | Cashflow explícito |
| Order Execution | Raw | N/A |

---

## Invariante Chave

```
equity_raw(T) + Σ dividends(0..T) ≈ equity_adjusted(T)
```

**Teste**: `t1_buyhold_economic_return_matches_adjusted`

---

## Credenciamento de Dividendos

### Por que Ex-Date?

Dividendos são creditados no **ex_date** (não payment date):

1. **Consistência de Mercado**: Preço cai pelo valor do dividendo no ex-date
2. **Alinhamento de Série**: Preços ajustados refletem dividendos no ex-date
3. **Simplicidade**: Sem necessidade de tracking "dividend receivable"

### Ações Elegíveis

| Cenário | Recebe Dividendo? |
|---------|-------------------|
| Posição mantida antes do ex-date | SIM |
| Compra no ex-date | NÃO |
| Venda no ex-date | SIM |

**Convenção**: Ações mantidas no fim do dia T-1 recebem dividendos no ex-date T.

---

## Implementação

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

### PriceType

```rust
pub enum PriceType {
    Signals,    // Usar adjusted
    Valuation,  // Usar raw
}
```

### DividendEvent

```rust
pub struct DividendEvent {
    pub symbol: String,
    pub ex_date: NaiveDate,
    pub payment_date: Option<NaiveDate>,
    pub rate: Decimal,
    pub dividend_type: String,
}
```

---

## Configuração

### Via RunnerConfig

```rust
let config = RunnerConfig {
    enable_dividends: true,
    initial_capital: Decimal::from(1_000_000),
    ..Default::default()
};
```

### Via UnifiedEngine

```rust
let config = UnifiedEngineConfig {
    initial_capital: dec!(1_000_000),
    enable_dividends: true,
    valuation_price_type: PriceType::Valuation,
    ..Default::default()
};

let mut engine = UnifiedEngine::with_config(config);
engine.load_dividends(dividend_events);
```

---

## Validação de Política

### Guard Anti-Double-Count

```rust
impl UnifiedEngine {
    pub fn validate_anti_double_count(&self) -> Result<(), PolicyViolation> {
        if self.config.enable_dividends && 
           self.config.valuation_price_type == PriceType::Signals {
            return Err(PolicyViolation {
                message: "Cannot use adjusted prices for valuation with dividends enabled"
            });
        }
        Ok(())
    }
}
```

### Restrição Fast Mode

Fast mode **NÃO suporta** dividend cashflow tracking.

Quando dividends habilitados + Fast mode:
1. Fallback automático para Compiled
2. Registrado em `metadata.mode_fallback_reason`
3. Entrada no trace com `mode_fallback`

---

## Artefatos

### timeseries.csv

```csv
date,equity,dividend_cashflow,dividend_cumulative
2024-01-02,100000.00,0.00,0.00
2024-03-15,105000.00,450.00,450.00
```

### trace.jsonl

```json
{"type": "dividend", "date": "2024-03-15", "symbol": "TAEE11", "rate": 0.45, "shares": 1000, "cashflow": 450.00}
{"type": "dividend_policy", "message": "Anti-double-count policy applied", "params": {"signals_price": "adjusted", "valuation_price": "raw"}}
```

### metadata.json

```json
{
  "dividends_enabled": true,
  "dividend_policy": {
    "signals_price": "adjusted",
    "valuation_price": "raw",
    "dividends_as_cashflow": true
  },
  "total_dividend_cashflow": "450.00",
  "dividend_count": 1
}
```

---

## Exemplo Numérico

### Setup

- Comprar 1000 ações de TAEE11 a R$40.00 em 2024-01-01
- Capital inicial: R$100,000

### Ex-Date: 2024-03-15

- Dividendo: R$0.50/ação
- Preço raw cai: R$40.50 → R$40.00
- Preço adjusted: R$40.50 (sem queda visível)

### Cálculos

**RAW + Cashflow (CORRETO)**:
```
Valor da posição (raw):   1000 × 40.00 = R$40,000
Dividend cashflow:        1000 × 0.50  = R$   500
Saldo de caixa:           R$60,000 + R$500 = R$60,500
Equity total:             R$60,500 + R$40,000 = R$100,500
```

**Adjusted (TAMBÉM CORRETO)**:
```
Valor da posição (adj):   1000 × 40.50 = R$40,500
Saldo de caixa:           R$60,000 (sem dividend)
Equity total:             R$60,000 + R$40,500 = R$100,500
```

**Adjusted + Cashflow (ERRADO - Double Count!)**:
```
Valor da posição (adj):   1000 × 40.50 = R$40,500
Dividend cashflow:        R$500 (ERRADO adicionar)
Saldo de caixa:           R$60,000 + R$500 = R$60,500
Equity total:             R$60,500 + R$40,500 = R$101,000 ← ERRADO!
```

---

## Testes

```bash
# Testes de política de dividendos
cargo test -p backtester_intelligence dividend_integration

# E2E
cargo test -p backtester_strategy runner_dividend_e2e

# Testes específicos
cargo test t1_buyhold_economic_return_matches_adjusted
cargo test t2_anti_double_count_validation
```

---

## Localização no Código

| Componente | Arquivo |
|------------|---------|
| UnifiedEngine | `backtester_engine/src/unified.rs` |
| DividendEvent | `backtester_engine/src/unified.rs` |
| DividendIndex | `backtester_engine/src/unified.rs` |
| Dividend module | `backtester_intelligence/src/dividends/` |



