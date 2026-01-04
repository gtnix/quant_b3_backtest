# Risk Profiles User Guide

**Versão**: 2.0.0  
**Última Atualização**: 2026-01-04

## Overview

The risk profiles system provides 5 predefined risk configurations for backtesting:

| Profile | Risk Level | Target Use Case |
|---------|------------|-----------------|
| `muito_conservador` | Very Low | Capital preservation, pension funds |
| `conservador` | Low | Stable growth, low volatility funds |
| `moderado` | Medium | Balanced growth (default) |
| `arrojado` | High | Aggressive growth, hedge funds |
| `muito_arrojado` | Very High | Speculative, high conviction |

## Quick Start

### Via CLI

```bash
# Run backtest with specific profile
cargo run --release -- backtest --risk-profile conservador

# Run with market-specific adjustments
cargo run --release -- backtest --risk-profile moderado --market BR
```

### Via Config File

```toml
# In your strategy config
[risk]
profile = "conservador"
market_adjustments = true

# Optional: Override specific parameters
[risk.overrides]
kelly_fraction = 0.35
max_drawdown_pct = -0.10
```

### Via Environment Variable

```bash
export BACKTEST_RISK_PROFILE=arrojado
cargo run --release -- backtest
```

## Profile Details

### Muito Conservador

**Goal**: Beat inflation with negligible risk of ruin.

- Kelly Fraction: 15% (Quarter-Kelly)
- Risk per Trade: 0.5% max
- ATR Stop: 3.5x (wide)
- Max Drawdown: 8%
- Vol Target: 10% BR / 8% US

**Best for**: Retirement accounts, endowments, risk-averse investors.

### Conservador

**Goal**: Consistent growth with controlled drawdowns.

- Kelly Fraction: 30%
- Risk per Trade: 0.75% max
- ATR Stop: 3.0x
- Max Drawdown: 12%
- Vol Target: 12% BR / 10% US

**Best for**: Conservative funds, income-focused portfolios.

### Moderado (Default)

**Goal**: Long-term geometric growth near Half-Kelly optimal.

- Kelly Fraction: 40%
- Risk per Trade: 1.25% max
- ATR Stop: 2.5x
- Max Drawdown: 20%
- Vol Target: 16% BR / 12% US

**Best for**: General-purpose backtesting, balanced portfolios.

### Arrojado

**Goal**: High returns with elevated risk tolerance.

- Kelly Fraction: 50% (Half-Kelly)
- Risk per Trade: 1.75% max
- ATR Stop: 2.0x (tighter)
- Max Drawdown: 25%
- Vol Target: 20% BR / 16% US

**Best for**: Aggressive funds, momentum strategies.

### Muito Arrojado

**Goal**: Maximum returns within academically defensible bounds.

- Kelly Fraction: 50% (Half-Kelly max)
- Risk per Trade: 2.25% max
- ATR Stop: 1.75x (tight)
- Max Drawdown: 30%
- Vol Target: 25% BR / 20% US

**Warning**: Significant drawdown risk. Only for proven high-conviction strategies.

## Market Adjustments

When `market_adjustments = true` (default), parameters are automatically adjusted for BR vs US markets:

| Parameter | BR Adjustment | Reason |
|-----------|---------------|--------|
| ATR Multiplier | +20% | Higher volatility |
| Vol Target | +25% | Emerging market premium |
| Min Liquidity | -75% | Smaller market |
| Max Spread | +100% | Lower liquidity |

## Overriding Parameters

### Single Parameter Override

```toml
[risk]
profile = "moderado"

[risk.overrides]
max_drawdown_pct = -0.15  # Tighter than default
```

### Multiple Overrides

```toml
[risk]
profile = "conservador"

[risk.overrides.sizing]
kelly_fraction = 0.35
max_risk_per_trade_pct = 0.01

[risk.overrides.stops]
atr_multiplier_br = 3.5
enable_trailing = false

[risk.overrides.circuit_breakers]
daily_loss_limit_pct = -0.02
```

## Logging and Observability

At backtest start, the system logs effective parameters:

```
[INFO] [RISK PROFILE] Effective Parameters
    profile: conservador
    market: BR

[INFO] [SIZING]
    kelly_fraction: 30.0%
    max_risk_per_trade: 0.75%
    max_positions: 25

[INFO] [PORTFOLIO RISK]
    volatility_target: 12.0%
    max_drawdown: 12.0%
```

### Universe Filter Logging

Each filter step is logged with before/after counts:

```
[INFO] [FILTER] momentum (min_return=0.05)
    stage: gating
    before: 42
    after: 28
    excluded: 14 (PETR4, VALE3, ITUB4, BBDC4, ABEV3 +9 more)
```

## Guardrails and Warnings

### Empty Universe Detection

If filters exclude all candidates:

```
[WARN] [GUARDRAIL] Empty universe for BR on 2024-01-15
    candidates: 85
    gating_excluded: 85
    top_reasons: liquidez insuficiente (40), sem dados de preço (30), fora do top-N (15)
```

### Zero Trades Warning

If strategy generates no trades:

```
[CRITICAL] ZERO_TRADES: Strategy executed 0 trades. Metrics are artificial and unreliable.
```

The backtest result will have `is_valid = false`.

---

## Recursos Avançados (v2.0)

### CVaR (Conditional Value-at-Risk)

Circuit breaker baseado em CVaR 95% para controle de risco de cauda.

```toml
[risk.overrides.circuit_breakers]
cvar_limit_95 = -0.20  # Limite de 20% de perda esperada no pior 5%
check_cvar = true
```

| Perfil | CVaR Limit 95% |
|--------|----------------|
| Muito Conservador | -10% |
| Conservador | -15% |
| Moderado | -20% |
| Arrojado | -25% |
| Muito Arrojado | -30% |

### Kelly Dinâmico

Cálculo automático da fração de Kelly baseado no histórico de trades.

```toml
[risk.overrides.sizing]
use_dynamic_kelly = true
kelly_lookback_trades = 100
kelly_min_trades = 20
kelly_max_fraction = 0.50  # Nunca excede Half-Kelly
```

### Anti-Concentração via Drawdown Beta

Previne novas posições quando correlação de drawdown é muito alta.

```toml
[risk.overrides.portfolio]
check_drawdown_beta = true
max_drawdown_beta = 0.80  # Máximo de 0.8 de correlação de drawdown
```

### Limite de Correlação com Portfolio

Rejeita ativos com correlação excessiva com o portfolio existente.

```toml
[risk.overrides.portfolio]
check_correlation = true
max_correlation = 0.70  # Máximo de 0.7 de correlação
```

### Clusters de Risco

Classificação automática de ativos por volatilidade e liquidez.

```toml
[risk.clustering]
enabled = true
vol_threshold_high = 0.40   # >40% vol = high risk
vol_threshold_low = 0.15    # <15% vol = low risk
liquidity_threshold_high = 50_000_000  # >50M = high liquidity
liquidity_threshold_low = 5_000_000    # <5M = low liquidity
```

Clusters resultantes:
- `BlueChip`: Baixa vol, alta liquidez
- `GrowthStock`: Alta vol, alta liquidez
- `ValueStock`: Baixa vol, baixa liquidez
- `Speculative`: Alta vol, baixa liquidez
- `Standard`: Intermediário

### Análise de Sensibilidade

Testa robustez da estratégia via perturbação de parâmetros.

```toml
[validation.sensitivity]
enabled = true
perturbation_pct = 0.10  # ±10% de variação nos parâmetros
num_perturbations = 20   # 20 variações por parâmetro
min_stability_score = 0.70  # 70% das perturbações devem manter performance
```

---

## Best Practices

1. **Start Conservative**: Begin with `moderado` or `conservador` and adjust based on results.

2. **Match to Strategy**: 
   - Mean reversion → tighter stops (arrojado/muito_arrojado)
   - Trend following → wider stops (conservador/moderado)

3. **Respect Kelly Limits**: Never exceed Half-Kelly (0.5) - academic consensus.

4. **Monitor Warnings**: Empty universe and zero trades indicate configuration issues.

5. **BR vs US**: Always enable market adjustments for fair comparison.

## Troubleshooting

### "Selected 0 assets"

**Cause**: Filter thresholds too restrictive for available data.

**Solutions**:
1. Enable quantile mode in filters
2. Use a more aggressive profile
3. Check data quality (missing prices, volumes)

### Unrealistic Sharpe Ratio (> 3.0)

**Cause**: Likely survivorship bias, look-ahead bias, or overfitting.

**Solutions**:
1. Enable eligibility validation for survivorship bias
2. Check fundamentals_as_of dates for look-ahead
3. Use walk-forward validation

### High Drawdown Despite Conservative Profile

**Cause**: Market conditions exceed historical calibration.

**Solutions**:
1. Enable circuit breakers
2. Reduce position concentration
3. Consider reducing vol target



