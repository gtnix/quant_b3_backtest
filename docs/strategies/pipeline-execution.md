# Execução de Pipeline

**Versão**: 2.0.0  
**Última Atualização**: 2025-12-28

## Visão Geral

O pipeline de estratégia executa uma sequência de blocos para transformar um universo de ativos em posições ponderadas.

---

## Fluxo de Execução

```
Universe → Selection → Entry → Exit → Sizing → Weights
```

### Detalhamento

```
┌─────────────────────────────────────────────────────────────────┐
│                       UNIVERSE                                   │
│              Todos os ativos disponíveis                        │
│              (ex: IBrA-100, ~100 ativos)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SELECTION                                   │
│           Filtra e rankeia ativos                               │
│                                                                  │
│   Blocks: momentum, value, quality, low_vol, dividend, etc.    │
│   Output: candidates filtrados (ex: top 20%)                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ENTRY                                     │
│           Gera sinais de entrada                                │
│                                                                  │
│   Blocks: ma_crossover, rsi, macd, bollinger, zscore            │
│   Output: sinais de compra/venda                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         EXIT                                     │
│           Determina condições de saída                          │
│                                                                  │
│   Blocks: stop_loss, take_profit, trailing_stop, time_exit      │
│   Output: sinais de saída                                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        SIZING                                    │
│           Calcula pesos das posições                            │
│                                                                  │
│   Blocks: equal_weight, risk_parity, vol_targeting              │
│   Output: HashMap<Symbol, Weight>                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       WEIGHTS                                    │
│           Portfólio final ponderado                             │
│                                                                  │
│   Invariantes:                                                  │
│   - Σ weights ≈ 1.0 (±0.1%)                                     │
│   - Cada weight ≤ max_weight                                    │
│   - N posições ≤ max_positions                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tipos de Blocos

### Selection

Filtra e rankeia ativos do universo.

| block_id | Descrição | Parâmetros |
|----------|-----------|------------|
| `momentum` | Retornos 6-12 meses | `lookback_days`, `top_pct` |
| `value` | P/E, P/B baixos | `max_pe`, `max_pb`, `top_pct` |
| `quality` | ROE alto, dívida baixa | `min_roe`, `max_debt_equity` |
| `low_vol` | Baixa volatilidade | `max_annualized_vol`, `top_pct` |
| `dividend` | Alto dividend yield | `min_yield`, `top_pct` |
| `size` | Market cap | `min_market_cap`, `max_market_cap` |
| `carry` | Dividend vs risk-free | `min_carry` |

**Múltiplos Selection**: Podem ser encadeados (interseção).

### Entry

Gera sinais de entrada baseados em indicadores técnicos.

| block_id | Descrição | Parâmetros |
|----------|-----------|------------|
| `ma_crossover` | Cruzamento de médias | `fast_period`, `slow_period` |
| `rsi` | RSI oversold/overbought | `period`, `oversold`, `overbought` |
| `macd` | MACD signal crossover | `fast_ema`, `slow_ema`, `signal` |
| `bollinger` | Breakout de bandas | `period`, `std_dev` |
| `zscore` | Mean reversion | `period`, `threshold` |

### Exit

Determina condições de saída de posições.

| block_id | Descrição | Parâmetros |
|----------|-----------|------------|
| `stop_loss` | Saída em perda | `threshold_pct` |
| `take_profit` | Saída em ganho | `target_pct` |
| `trailing_stop` | Stop móvel | `trailing_pct`, `activation_pct` |
| `time_exit` | Saída por tempo | `max_days` |

**Múltiplos Exit**: Podem ser encadeados (primeiro a disparar executa).

### Sizing

Calcula pesos das posições.

| block_id | Descrição | Parâmetros |
|----------|-----------|------------|
| `equal_weight` | 1/N alocação | `max_weight`, `min_weight`, `max_positions` |
| `risk_parity` | Inverso de volatilidade | `max_weight`, `fallback_vol` |
| `vol_targeting` | Target de vol do portfólio | `target_vol`, `max_leverage` |

---

## Exemplos de Pipeline

### Momentum Puro

```toml
[[pipeline]]
type = "selection"
block_id = "momentum"
params = { lookback_days = 126, top_pct = 20 }

[[pipeline]]
type = "sizing"
block_id = "equal_weight"
params = { max_weight = 0.20 }
```

### Multi-Factor

```toml
[[pipeline]]
type = "selection"
block_id = "value"
params = { top_pct = 40 }

[[pipeline]]
type = "selection"
block_id = "quality"
params = { top_pct = 50 }

[[pipeline]]
type = "sizing"
block_id = "risk_parity"
params = { max_weight = 0.15 }
```

### Trend Following com Stops

```toml
[[pipeline]]
type = "entry"
block_id = "ma_crossover"
params = { fast_period = 20, slow_period = 50 }

[[pipeline]]
type = "exit"
block_id = "stop_loss"
params = { threshold_pct = 0.08 }

[[pipeline]]
type = "exit"
block_id = "trailing_stop"
params = { trailing_pct = 0.12, activation_pct = 0.05 }

[[pipeline]]
type = "sizing"
block_id = "vol_targeting"
params = { target_vol = 0.12 }
```

---

## StrategyContext

Contexto passado a cada bloco:

```rust
pub struct StrategyContext {
    pub date: NaiveDate,
    pub market: Market,
    pub capital: Decimal,
    pub candidates: Vec<StrategyCandidate>,
    pub positions: HashMap<String, Position>,
    pub signals: Vec<Signal>,
}
```

### StrategyCandidate

```rust
pub struct StrategyCandidate {
    pub symbol: String,
    pub price: Decimal,
    pub volume: Decimal,
    pub momentum: Option<f64>,
    pub volatility: Option<f64>,
    pub fundamentals: Option<Fundamentals>,
}
```

---

## CompositorResult

Resultado da execução do pipeline:

```rust
pub struct CompositorResult {
    pub selected: Vec<String>,
    pub weights: HashMap<String, f64>,
    pub signals: Vec<Signal>,
    pub trace: Vec<TraceEntry>,
}
```

---

## Trace de Execução

Cada step gera entrada no trace:

```json
{"step": 0, "block_id": "momentum", "block_type": "selection", "message": "Selected 10 assets from 100", "params_effective": {"lookback_days": 126, "top_pct": 20}}
{"step": 1, "block_id": "equal_weight", "block_type": "sizing", "message": "Weights assigned to 10 assets", "params_effective": {"max_weight": 0.20}}
```

---

## Validação Strict Mode

Em strict mode, validações adicionais:

- Σ weights ≈ 1.0 (±0.1%)
- Cada weight ≤ max_weight
- N posições ≤ max_positions
- Nenhum NaN/Inf em pesos
- Pipeline não-vazio

---

## Localização no Código

| Componente | Arquivo |
|------------|---------|
| Compositor | `backtester_strategy/src/compositor.rs` |
| StrategyContext | `backtester_strategy/src/context.rs` |
| Selection blocks | `backtester_strategy/src/blocks/selection/` |
| Entry blocks | `backtester_strategy/src/blocks/entry/` |
| Exit blocks | `backtester_strategy/src/blocks/exit/` |
| Sizing blocks | `backtester_strategy/src/blocks/sizing/` |






