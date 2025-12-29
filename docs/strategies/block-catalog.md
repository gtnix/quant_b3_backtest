# Catálogo de Blocos

**Versão**: 2.0.0  
**Última Atualização**: 2025-12-28  
**Geração**: Este documento deve ser gerado do código

> **Nota**: Este documento deve ser regenerado sempre que blocos forem adicionados ou modificados.
> ```bash
> cargo run -p backtester_cli -- generate-catalog --output docs/strategies/block-catalog.md
> ```

---

## Sumário

| Categoria | Quantidade | Fast SoA |
|-----------|------------|----------|
| Selection | 7 | 2 (momentum, low_vol) |
| Entry | 5 | 0 |
| Exit | 4 | 0 |
| Sizing | 3 | 1 (equal_weight) |
| **Total** | **19** | **3** |

---

## Fast Mode Eligibility

Pipeline é elegível para Fast mode se **TODOS** os blocks têm `fast_supported: true`.

Se qualquer block não suporta:
- `--execution auto` → fallback para `compiled`
- `--execution fast --strict` → erro
- `--execution fast` → warning + fallback

---

## Selection Blocks

Filtram e rankeiam ativos do universo.

### `momentum`

**Descrição**: Momentum selection - rankeia ativos por retornos 6-12 meses.

**Fast**: ✓ Sim

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `lookback_days` | int | 126 | Período de lookback em dias |
| `top_pct` | float | 20 | Top % de ativos a selecionar |

```toml
[[pipeline]]
type = "selection"
block_id = "momentum"
params = { lookback_days = 126, top_pct = 20 }
```

---

### `value`

**Descrição**: Value selection - seleciona P/E, P/B baixos.

**Fast**: ✗ Não

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `max_pe` | float | 15 | P/E máximo |
| `max_pb` | float | 2 | P/B máximo |
| `top_pct` | float | 20 | Top % de ativos |

```toml
[[pipeline]]
type = "selection"
block_id = "value"
params = { max_pe = 12, max_pb = 1.5, top_pct = 20 }
```

---

### `quality`

**Descrição**: Quality selection - alto ROE, baixa dívida.

**Fast**: ✗ Não

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `min_roe` | float | 0.15 | ROE mínimo |
| `max_debt_equity` | float | 0.5 | D/E máximo |
| `top_pct` | float | 20 | Top % de ativos |

```toml
[[pipeline]]
type = "selection"
block_id = "quality"
params = { min_roe = 0.15, top_pct = 30 }
```

---

### `low_vol`

**Descrição**: Low volatility selection - ativos estáveis.

**Fast**: ✓ Sim

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `max_annualized_vol` | float | 0.25 | Vol máxima anualizada |
| `lookback_days` | int | 60 | Período de lookback |
| `top_pct` | float | 20 | Top % de ativos |

```toml
[[pipeline]]
type = "selection"
block_id = "low_vol"
params = { max_annualized_vol = 0.20, top_pct = 30 }
```

---

### `dividend`

**Descrição**: Dividend yield selection - alto rendimento de dividendos.

**Fast**: ✗ Não

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `min_yield` | float | 0.04 | Yield mínimo |
| `top_pct` | float | 20 | Top % de ativos |

```toml
[[pipeline]]
type = "selection"
block_id = "dividend"
params = { min_yield = 0.05, top_pct = 20 }
```

---

### `size`

**Descrição**: Size selection - filtro por market cap.

**Fast**: ✗ Não

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `min_market_cap` | float | - | Market cap mínimo (BRL) |
| `max_market_cap` | float | - | Market cap máximo (BRL) |
| `top_pct` | float | 20 | Top % de ativos |

```toml
[[pipeline]]
type = "selection"
block_id = "size"
params = { min_market_cap = 5000000000, top_pct = 20 }
```

---

### `carry`

**Descrição**: Carry selection - dividend yield vs risk-free.

**Fast**: ✗ Não

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `min_carry` | float | 0.02 | Carry mínimo |
| `top_pct` | float | 20 | Top % de ativos |

```toml
[[pipeline]]
type = "selection"
block_id = "carry"
params = { min_carry = 0.03, top_pct = 20 }
```

---

## Entry Blocks

Geram sinais de entrada baseados em indicadores técnicos.

### `ma_crossover`

**Descrição**: MA Crossover - long quando MA rápida cruza acima da lenta.

**Fast**: ✗ Não

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `fast_period` | int | 50 | Período MA rápida |
| `slow_period` | int | 200 | Período MA lenta |

```toml
[[pipeline]]
type = "entry"
block_id = "ma_crossover"
params = { fast_period = 20, slow_period = 50 }
```

---

### `rsi`

**Descrição**: RSI - long em oversold, exit em overbought.

**Fast**: ✗ Não

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `period` | int | 14 | Período RSI |
| `oversold` | float | 30 | Threshold oversold |
| `overbought` | float | 70 | Threshold overbought |

```toml
[[pipeline]]
type = "entry"
block_id = "rsi"
params = { period = 14, oversold = 30, overbought = 70 }
```

---

### `macd`

**Descrição**: MACD - long em bullish crossover.

**Fast**: ✗ Não

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `fast_ema` | int | 12 | Período EMA rápida |
| `slow_ema` | int | 26 | Período EMA lenta |
| `signal` | int | 9 | Período linha de sinal |

```toml
[[pipeline]]
type = "entry"
block_id = "macd"
params = { fast_ema = 12, slow_ema = 26, signal = 9 }
```

---

### `bollinger`

**Descrição**: Bollinger Bands - signal em breakouts.

**Fast**: ✗ Não

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `period` | int | 20 | Período MA |
| `std_dev` | float | 2 | Multiplicador std dev |

```toml
[[pipeline]]
type = "entry"
block_id = "bollinger"
params = { period = 20, std_dev = 2 }
```

---

### `zscore`

**Descrição**: Z-Score - mean reversion.

**Fast**: ✗ Não

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `period` | int | 20 | Período lookback |
| `threshold` | float | 2 | Threshold z-score |

```toml
[[pipeline]]
type = "entry"
block_id = "zscore"
params = { period = 20, threshold = 2 }
```

---

## Exit Blocks

Determinam condições de saída de posições.

### `stop_loss`

**Descrição**: Stop-loss - exit em perda.

**Fast**: ✗ Não

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `threshold_pct` | float | 0.10 | Threshold de perda |

```toml
[[pipeline]]
type = "exit"
block_id = "stop_loss"
params = { threshold_pct = 0.08 }
```

---

### `take_profit`

**Descrição**: Take-profit - exit em ganho.

**Fast**: ✗ Não

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `target_pct` | float | 0.30 | Target de ganho |

```toml
[[pipeline]]
type = "exit"
block_id = "take_profit"
params = { target_pct = 0.20 }
```

---

### `trailing_stop`

**Descrição**: Trailing stop - exit em drawdown do high-water mark.

**Fast**: ✗ Não

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `trailing_pct` | float | 0.15 | Trailing % |
| `activation_pct` | float | 0.10 | % de ganho para ativar |

```toml
[[pipeline]]
type = "exit"
block_id = "trailing_stop"
params = { trailing_pct = 0.12, activation_pct = 0.05 }
```

---

### `time_exit`

**Descrição**: Time exit - exit após N dias.

**Fast**: ✗ Não

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `max_days` | int | 20 | Máximo dias em posição |

```toml
[[pipeline]]
type = "exit"
block_id = "time_exit"
params = { max_days = 15 }
```

---

## Sizing Blocks

Calculam pesos das posições.

### `equal_weight`

**Descrição**: Equal weight - 1/N alocação.

**Fast**: ✓ Sim

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `max_weight` | float | 0.20 | Peso máximo por posição |
| `min_weight` | float | 0.02 | Peso mínimo por posição |
| `max_positions` | int | 20 | Número máximo de posições |

```toml
[[pipeline]]
type = "sizing"
block_id = "equal_weight"
params = { max_weight = 0.20, max_positions = 10 }
```

---

### `risk_parity`

**Descrição**: Risk parity - inversamente proporcional à volatilidade.

**Fast**: ✗ Não

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `max_weight` | float | 0.20 | Peso máximo |
| `min_weight` | float | 0.02 | Peso mínimo |
| `max_positions` | int | 20 | Máximo posições |
| `fallback_vol` | float | 0.25 | Vol fallback se faltando |

```toml
[[pipeline]]
type = "sizing"
block_id = "risk_parity"
params = { max_weight = 0.15 }
```

---

### `vol_targeting`

**Descrição**: Vol targeting - escala para volatilidade alvo.

**Fast**: ✗ Não

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `target_vol` | float | 0.12 | Vol alvo do portfólio |
| `max_weight` | float | 0.30 | Peso máximo |
| `max_leverage` | float | 1.0 | Leverage máximo |
| `correlation` | float | 0.5 | Correlação assumida |

```toml
[[pipeline]]
type = "sizing"
block_id = "vol_targeting"
params = { target_vol = 0.15, max_leverage = 1.2 }
```

---

## Localização no Código

| Categoria | Diretório |
|-----------|-----------|
| Selection | `backtester_strategy/src/blocks/selection/` |
| Entry | `backtester_strategy/src/blocks/entry/` |
| Exit | `backtester_strategy/src/blocks/exit/` |
| Sizing | `backtester_strategy/src/blocks/sizing/` |
| Registry | `backtester_strategy/src/registry.rs` |



