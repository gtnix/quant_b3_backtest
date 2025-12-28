# Block Catalog

**Version**: 1.1  
**Date**: 2025-12-26  
**Schema Version**: 1.1

This document lists all available strategy blocks in the Strategy Factory.

## Overview

| Category | Count | Fast SoA Available |
|----------|-------|-------------------|
| Selection | 7 | ✓ momentum, low_vol |
| Entry | 5 | - |
| Exit | 4 | - |
| Sizing | 3 | ✓ equal_weight |
| **Total** | **19** | **3** |

> **Performance Note**: Blocks with "Fast SoA" support (`fast_supported: true`) have optimized
> implementations using Structure-of-Arrays layout, achieving 93-124x speedup.
> See [Performance Baseline](./PERFORMANCE_BASELINE.md).

### Fast Mode Eligibility

A pipeline is eligible for Fast mode (`--execution fast`) if and only if ALL blocks in the
pipeline have `fast_supported: true`. If any block lacks fast support:
- `--execution auto` falls back to `compiled`
- `--execution fast` with `--strict` fails with error
- `--execution fast` without `--strict` logs warning and falls back

---

## Selection Blocks

Selection blocks filter and rank assets from the universe.

| block_id | Description | Key Parameters | Fast |
|----------|-------------|----------------|------|
| `size` | Size selection: filters by market cap (small/mid/large) | top_pct | |
| `dividend` | Dividend yield selection: high dividend stocks | top_pct | |
| `quality` | Quality selection: high ROE, low debt companies | top_pct | |
| `momentum` | Momentum selection: ranks assets by 6-12 month returns | top_pct | ✓ |
| `low_vol` | Low volatility selection: selects stable, low-vol assets | top_pct | ✓ |
| `value` | Value selection: selects low P/E, low P/B stocks | top_pct | |
| `carry` | Carry selection: dividend yield vs risk-free rate | top_pct | |

### `size`

Size selection: filters by market cap (small/mid/large)

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_pct` | float | 20 | Top percentage of assets to select |

**Example:**

```toml
[[pipeline]]
type = "selection"
block_id = "size"
params = { top_pct = 20 }
```

### `dividend`

Dividend yield selection: high dividend stocks

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_pct` | float | 20 | Top percentage of assets to select |

**Example:**

```toml
[[pipeline]]
type = "selection"
block_id = "dividend"
params = { top_pct = 20 }
```

### `quality`

Quality selection: high ROE, low debt companies

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_pct` | float | 20 | Top percentage of assets to select |

**Example:**

```toml
[[pipeline]]
type = "selection"
block_id = "quality"
params = { top_pct = 20 }
```

### `momentum`

Momentum selection: ranks assets by 6-12 month returns

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_pct` | float | 20 | Top percentage of assets to select |

**Example:**

```toml
[[pipeline]]
type = "selection"
block_id = "momentum"
params = { top_pct = 20 }
```

### `low_vol`

Low volatility selection: selects stable, low-vol assets

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_pct` | float | 20 | Top percentage of assets to select |

**Example:**

```toml
[[pipeline]]
type = "selection"
block_id = "low_vol"
params = { top_pct = 20 }
```

### `value`

Value selection: selects low P/E, low P/B stocks

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_pct` | float | 20 | Top percentage of assets to select |

**Example:**

```toml
[[pipeline]]
type = "selection"
block_id = "value"
params = { top_pct = 20 }
```

### `carry`

Carry selection: dividend yield vs risk-free rate

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_pct` | float | 20 | Top percentage of assets to select |

**Example:**

```toml
[[pipeline]]
type = "selection"
block_id = "carry"
params = { top_pct = 20 }
```

## Entry Blocks

Entry blocks generate buy/sell signals based on technical indicators.

| block_id | Description | Key Parameters | Fast |
|----------|-------------|----------------|------|
| `macd` | MACD: Long on bullish crossover, exit on bearish crossover | fast_ema, slow_ema, signal | |
| `rsi` | RSI: Long on oversold (<30), exit on overbought (>70) | period, oversold, overbought | |
| `bollinger` | Bollinger Bands: Signal on breakouts above/below bands | period, std_dev | |
| `ma_crossover` | MA Crossover: Long when fast MA crosses above slow MA | fast_period, slow_period | |
| `zscore` | Z-Score: Long on z < -2 (oversold), exit on z > 2 (overbought) | period, threshold | |

### `macd`

MACD: Long on bullish crossover, exit on bearish crossover

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fast_ema` | int | 12 | Parameter for block configuration |
| `slow_ema` | int | 26 | Parameter for block configuration |
| `signal` | int | 9 | Parameter for block configuration |

**Example:**

```toml
[[pipeline]]
type = "entry"
block_id = "macd"
params = { fast_ema = 12, slow_ema = 26, signal = 9 }
```

### `rsi`

RSI: Long on oversold (<30), exit on overbought (>70)

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `overbought` | float | 70 | Overbought threshold |
| `period` | int | 14 | Indicator period |
| `oversold` | float | 30 | Oversold threshold |

**Example:**

```toml
[[pipeline]]
type = "entry"
block_id = "rsi"
params = { overbought = 70, period = 14, oversold = 30 }
```

### `bollinger`

Bollinger Bands: Signal on breakouts above/below bands

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period` | int | 20 | Indicator period |
| `std_dev` | float | 2 | Standard deviation multiplier |

**Example:**

```toml
[[pipeline]]
type = "entry"
block_id = "bollinger"
params = { period = 20, std_dev = 2 }
```

### `ma_crossover`

MA Crossover: Long when fast MA crosses above slow MA

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `slow_period` | int | 200 | Slow moving average period |
| `fast_period` | int | 50 | Fast moving average period |

**Example:**

```toml
[[pipeline]]
type = "entry"
block_id = "ma_crossover"
params = { slow_period = 200, fast_period = 50 }
```

### `zscore`

Z-Score: Long on z < -2 (oversold), exit on z > 2 (overbought)

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 2 | Parameter for block configuration |
| `period` | int | 20 | Indicator period |

**Example:**

```toml
[[pipeline]]
type = "entry"
block_id = "zscore"
params = { threshold = 2, period = 20 }
```

## Exit Blocks

Exit blocks determine when to close positions.

| block_id | Description | Key Parameters | Fast |
|----------|-------------|----------------|------|
| `take_profit` | Take-profit: Exit on gain exceeding target | target_pct | |
| `trailing_stop` | Trailing stop: Exit on drawdown from high-water mark | activation_pct, trailing_pct | |
| `time_exit` | Time exit: Exit after holding for max days | max_days | |
| `stop_loss` | Stop-loss: Exit on loss exceeding threshold | threshold_pct | |

### `take_profit`

Take-profit: Exit on gain exceeding target

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_pct` | float | 0.3 | Parameter for block configuration |

**Example:**

```toml
[[pipeline]]
type = "exit"
block_id = "take_profit"
params = { target_pct = 0.3 }
```

### `trailing_stop`

Trailing stop: Exit on drawdown from high-water mark

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trailing_pct` | float | 0.15 | Trailing stop percentage |
| `activation_pct` | float | 0.1 | Parameter for block configuration |

**Example:**

```toml
[[pipeline]]
type = "exit"
block_id = "trailing_stop"
params = { trailing_pct = 0.15, activation_pct = 0.1 }
```

### `time_exit`

Time exit: Exit after holding for max days

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_days` | int | 20 | Maximum holding period in days |

**Example:**

```toml
[[pipeline]]
type = "exit"
block_id = "time_exit"
params = { max_days = 20 }
```

### `stop_loss`

Stop-loss: Exit on loss exceeding threshold

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold_pct` | float | 0.1 | Exit threshold as percentage |

**Example:**

```toml
[[pipeline]]
type = "exit"
block_id = "stop_loss"
params = { threshold_pct = 0.1 }
```

## Sizing Blocks

Sizing blocks determine position weights.

| block_id | Description | Key Parameters | Fast |
|----------|-------------|----------------|------|
| `equal_weight` | Equal weight: 1/N allocation across selected assets | max_positions, max_weight, min_weight | ✓ |
| `risk_parity` | Risk parity: Inverse volatility weighting | fallback_vol, max_weight, min_weight, max_positions | |
| `vol_targeting` | Vol targeting: Scale positions to achieve target portfolio volatility | fallback_vol, max_weight, min_weight, max_positions, target_vol, max_leverage, correlation | |

### `equal_weight`

Equal weight: 1/N allocation across selected assets

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_weight` | float | 0.02 | Minimum weight per position |
| `max_positions` | int | 20 | Maximum number of positions |
| `max_weight` | float | 0.2 | Maximum weight per position |

**Example:**

```toml
[[pipeline]]
type = "sizing"
block_id = "equal_weight"
params = { min_weight = 0.02, max_positions = 20, max_weight = 0.2 }
```

### `risk_parity`

Risk parity: Inverse volatility weighting

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_positions` | int | 20 | Maximum number of positions |
| `max_weight` | float | 0.2 | Maximum weight per position |
| `min_weight` | float | 0.02 | Minimum weight per position |
| `fallback_vol` | float | 0.25 | Parameter for block configuration |

**Example:**

```toml
[[pipeline]]
type = "sizing"
block_id = "risk_parity"
params = { max_positions = 20, max_weight = 0.2, min_weight = 0.02, fallback_vol = 0.25 }
```

### `vol_targeting`

Vol targeting: Scale positions to achieve target portfolio volatility

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_weight` | float | 0.3 | Maximum weight per position |
| `max_positions` | int | 20 | Maximum number of positions |
| `min_weight` | float | 0.02 | Minimum weight per position |
| `max_leverage` | float | 1 | Parameter for block configuration |
| `correlation` | float | 0.5 | Parameter for block configuration |
| `fallback_vol` | float | 0.25 | Parameter for block configuration |
| `target_vol` | float | 0.12 | Target portfolio volatility |

**Example:**

```toml
[[pipeline]]
type = "sizing"
block_id = "vol_targeting"
params = { max_weight = 0.3, max_positions = 20, min_weight = 0.02, max_leverage = 1, correlation = 0.5, fallback_vol = 0.25, target_vol = 0.12 }
```

---

## Usage Examples

### Momentum Strategy

```toml
[strategy]
id = "momentum_pure"
version = "1.0.0"
description = "Pure momentum with equal weights"

[[pipeline]]
type = "selection"
block_id = "momentum"
params = { lookback_days = 126, top_pct = 20 }

[[pipeline]]
type = "sizing"
block_id = "equal_weight"
params = { max_weight = 0.20 }
```

### Multi-Factor Strategy

```toml
[strategy]
id = "value_quality"
version = "1.0.0"
description = "Value + Quality with risk parity"

[[pipeline]]
type = "selection"
block_id = "value"

[[pipeline]]
type = "selection"
block_id = "quality"

[[pipeline]]
type = "sizing"
block_id = "risk_parity"
params = { max_weight = 0.20 }
```

### Trend Following with Exits

```toml
[strategy]
id = "trend_following"
version = "1.0.0"
description = "MA crossover with trailing stop"

[[pipeline]]
type = "entry"
block_id = "ma_crossover"
params = { fast_period = 20, slow_period = 50 }

[[pipeline]]
type = "exit"
block_id = "trailing_stop"
params = { trailing_pct = 0.10 }

[[pipeline]]
type = "sizing"
block_id = "vol_targeting"
params = { target_vol = 0.15 }
```

