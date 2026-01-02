# Performance Engine

**Versão**: 2.0.0  
**Última Atualização**: 2025-12-28

## Visão Geral

O Performance Engine é responsável pelo cálculo de métricas de performance, atribuição de retornos e geração de reports.

### Localização no Código

- **Crate**: `backtester_intelligence`
- **Diretório**: `src/performance/`
- **Arquivos Principais**:
  - `engine.rs` - PerformanceEngine
  - `reporter.rs` - Geração de JSON
  - `concentration.rs` - HHI, top-N, Gini
  - `regime.rs` - Detecção de regime
  - `sector.rs` - Exposição setorial

---

## MetricsCalculator

Calculador de métricas de performance.

### Localização

- **Crate**: `backtester_strategy`
- **Arquivo**: `src/experiment/metrics.rs`

### Constantes

```rust
pub const TRADING_DAYS_PER_YEAR: f64 = 252.0;
pub const DEFAULT_RISK_FREE_RATE: f64 = 0.05;
pub const WEIGHT_SUM_TOLERANCE: f64 = 0.001;
pub const MIN_VOLATILITY_THRESHOLD: f64 = 0.0001;
pub const MAX_RATIO_VALUE: f64 = 999.99;
```

### Métricas Calculadas

| Métrica | Fórmula | Referência |
|---------|---------|------------|
| CAGR | `(end/start)^(1/years) - 1` | `metrics.rs:cagr()` |
| Volatility | `std(daily_returns) × √252` | `metrics.rs:volatility()` |
| Sharpe | `(ann_return - rf) / vol` | `metrics.rs:sharpe()` |
| Sortino | `(ann_return - rf) / downside_vol` | `metrics.rs:sortino()` |
| Calmar | `CAGR / abs(max_drawdown)` | `metrics.rs:compute()` |
| Max Drawdown | `min((equity - peak) / peak)` | `metrics.rs:max_drawdown()` |
| Hit Rate | `winning_trades / total_trades` | `metrics.rs:trade_stats()` |
| Profit Factor | `gross_profit / gross_loss` | `metrics.rs:trade_stats()` |
| Turnover | `total_traded / avg_equity / years` | `metrics.rs:compute_turnover()` |

### Uso

```rust
use backtester_strategy::experiment::MetricsCalculator;

let metrics = MetricsCalculator::compute(
    &timeseries,
    &trades,
    0.05, // risk-free rate
);

println!("CAGR: {:.2}%", metrics.cagr * 100.0);
println!("Sharpe: {:.2}", metrics.sharpe_ratio);
println!("Max DD: {:.2}%", metrics.max_drawdown * 100.0);
```

### RunMetrics

```rust
pub struct RunMetrics {
    pub cagr: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub max_drawdown: f64,
    pub max_drawdown_duration_days: u32,
    pub hit_rate: f64,
    pub profit_factor: f64,
    pub turnover_annual: f64,
    pub total_trades: u32,
    pub winning_trades: u32,
    pub losing_trades: u32,
}
```

---

## VolatilityType

```rust
pub enum VolatilityType {
    #[default]
    Population, // Divisor N
    Sample,     // Divisor N-1
}
```

**Default**: Population (mais estável para amostras pequenas)

---

## Tratamento de Infinito

Métricas que resultariam em infinito são capped em `MAX_RATIO_VALUE = 999.99`:

- Sortino sem retornos negativos
- Profit Factor sem perdas
- Calmar com drawdown zero

---

## Concentração

### HHI (Herfindahl-Hirschman Index)

```
HHI = Σ(w_i)²
```

| HHI | Interpretação |
|-----|---------------|
| 1.0 | Posição única |
| 0.5 | 2 posições iguais |
| 0.1 | 10 posições iguais |

### Effective N

```
Effective_N = 1 / HHI
```

### Top-N Weights

- `top_1_weight_pct`: Maior posição
- `top_5_weight_pct`: Soma das 5 maiores
- `top_10_weight_pct`: Soma das 10 maiores

### Gini Coefficient

```
Gini ∈ [0, 1]
0 = Igualdade perfeita
1 = Desigualdade máxima
```

---

## Regime Detection

### Trend State

```rust
pub enum TrendState {
    Uptrend,
    Downtrend,
    Sideways,
}
```

**Algoritmo**: Slope de regressão linear normalizado por volatilidade.

### Volatility Quantile

| Quantile | Percentil |
|----------|-----------|
| Q1 | 0-20% (menor vol) |
| Q2 | 20-40% |
| Q3 | 40-60% (mediana) |
| Q4 | 60-80% |
| Q5 | 80-100% (maior vol) |

---

## FX Attribution

### Decomposição de Retorno (3 termos)

```
R_total = R_asset + R_fx + R_interaction

onde:
  R_asset = V_L(t1) / V_L(t0) - 1
  R_fx = FX(t1) / FX(t0) - 1
  R_interaction = R_asset × R_fx
```

### Verificação

```
(1 + R_asset) × (1 + R_fx) = 1 + R_total
```

---

## Schema de Report

```json
{
  "schema_version": "fx_report_v1.3",
  "date": "2024-12-27",
  "equity": "100000.00",
  "return_pct": "15.50",
  "drawdown_pct": "-5.00",
  "concentration": {
    "hhi": "0.25",
    "effective_n": "4.00",
    "top_1_weight_pct": "30.00"
  },
  "regime_summary": {
    "by_regime": [...]
  },
  "compliance": {
    "summary": {
      "total_breaches": 0
    }
  }
}
```

---

## Testes

```bash
# Testes de métricas
cargo test -p backtester_strategy metrics

# Testes de performance
cargo test -p backtester_intelligence performance

# Testes de FX
cargo test -p backtester_intelligence fx
```
















