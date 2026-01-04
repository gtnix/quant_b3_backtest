# Evidence Map: Risk Parameters

**Versão**: 2.0.0  
**Última Atualização**: 2026-01-04

This document maps each risk parameter to its academic/practitioner source and implementation location.

## Position Sizing

| Parameter | Range | Internal Source | External Source | BR vs US |
|-----------|-------|-----------------|-----------------|----------|
| `kelly_fraction` | 0.10-0.50 | `risk_profiles/profile.rs` | Ziemba & MacLean (2011), Thorp (2006) | Same (0.5 max) |
| `max_risk_per_trade_pct` | 0.25-2.5% | `risk_profiles/types.rs` | Vince (1992), Chan (2021) | BR: higher allowed due to vol |
| `max_exposure_per_asset_pct` | 5-30% | `risk_profiles/profile.rs` | Portfolio Theory | Same |
| `max_sector_concentration_pct` | 20-45% | `risk_profiles/profile.rs` | Industry practice | Same |
| `max_positions` | 10-30 | `risk_profiles/profile.rs` | Diversification theory | Same |

### Academic Sources

1. **Ziemba & MacLean (2011)** - "The Kelly Capital Growth Investment Criterion"
   - Recommends Half-Kelly (0.5) as maximum for practical use
   - Quarter-Kelly (0.25) for conservative investors
   - Section on "Fractional Kelly Strategies"

2. **Thorp (2006)** - "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"
   - Demonstrates Kelly betting leads to maximum geometric growth
   - Warns against full Kelly due to volatility of returns

3. **Vince (1992)** - "The Mathematics of Money Management"
   - Risk per trade guidelines: 0.5-2% of capital
   - Position sizing based on historical win rate and payoff ratio

## Stop-Loss

| Parameter | Range | Internal Source | External Source | BR vs US |
|-----------|-------|-----------------|-----------------|----------|
| `stop_type` | ATR/Pct/Vol | `risk_profiles/types.rs` | Wilder (1978), Chan (2013) | Same |
| `atr_period` | 14-20 | `risk_profiles/types.rs` | Wilder (1978) | Same (14 default) |
| `atr_multiplier` | 1.5-4.0x | `risk_profiles/profile.rs` | Chan (2013), Kaufman (2013) | BR: 1.2x higher |
| `trailing_activation_pct` | 5-15% | `risk_profiles/types.rs` | Industry practice | Same |

### Academic Sources

1. **Wilder (1978)** - "New Concepts in Technical Trading Systems"
   - Introduced ATR (Average True Range)
   - Recommended 14-period ATR

2. **Chan (2013)** - "Algorithmic Trading: Winning Strategies and Their Rationale"
   - ATR-based stops: 2-3x ATR for trend following
   - Wider stops for volatile markets

3. **Kaufman (2013)** - "Trading Systems and Methods"
   - Volatility-adjusted stops adapt to market conditions
   - Trailing stops to lock in profits

## Portfolio Risk

| Parameter | Range | Internal Source | External Source | BR vs US |
|-----------|-------|-----------------|-----------------|----------|
| `volatility_target` | 6-28% ann | `risk_profiles/profile.rs` | Harvey et al. (2017) | BR: 25% higher |
| `max_drawdown_pct` | -8% to -30% | `risk_profiles/profile.rs` | Chekhlov et al. (2003) | Same |
| `max_leverage` | 1.0-1.5x | `risk_profiles/profile.rs` | Industry practice | Same |
| `cvar_limit_95` | -2% to -6% | `risk_profiles/types.rs` | Rockafellar (2000) | Same |

### Academic Sources

1. **Harvey et al. (2017)** - "The Impact of Volatility Targeting"
   - Volatility targeting improves Sharpe ratios
   - Recommended targets: 10-20% annualized

2. **Chekhlov, Uryasev, Zabarankin (2003)** - "Drawdown Measure in Portfolio Optimization"
   - Maximum drawdown as risk measure
   - Integration with portfolio optimization

3. **Rockafellar & Uryasev (2000)** - "Optimization of Conditional Value-at-Risk"
   - CVaR as coherent risk measure
   - Superior to VaR for tail risk

## Circuit Breakers

| Parameter | Range | Internal Source | External Source | BR vs US |
|-----------|-------|-----------------|-----------------|----------|
| `daily_loss_limit_pct` | -1% to -5% | `risk_profiles/profile.rs` | Industry practice | Same |
| `weekly_loss_limit_pct` | -2% to -10% | `risk_profiles/profile.rs` | Industry practice | Same |
| `drawdown_action` | CashOut/ReduceRisk/Alert | `risk_profiles/types.rs` | Industry practice | Same |
| `cooldown_days` | 0-2 | `risk_profiles/types.rs` | Behavioral finance | Same |

### Practitioner Sources

1. **Institutional Trading Desks** - Daily and weekly loss limits are standard
2. **Risk Management Best Practices** - Circuit breakers prevent catastrophic losses

## Operational/Microstructure

| Parameter | Range | Internal Source | External Source | BR vs US |
|-----------|-------|-----------------|-----------------|----------|
| `min_liquidity_usd` | $3M-$20M/day | `filters/mod.rs` | Market microstructure | BR: 4x lower |
| `max_spread_bps` | 15-45 bps | `filters/mod.rs` | Transaction cost analysis | BR: 2x higher |
| `slippage_cost_bps` | 8-15 bps | `risk_profiles/profile.rs` | Almgren & Chriss (2001) | BR: 2x higher |
| `max_participation_rate` | 3-8% | `risk_profiles/types.rs` | Market impact models | Same |

### Academic Sources

1. **Almgren & Chriss (2001)** - "Optimal Execution of Portfolio Transactions"
   - Market impact model: temporary and permanent impact
   - Slippage increases with order size

2. **Kyle (1985)** - "Continuous Auctions and Insider Trading"
   - Price impact proportional to order flow
   - Lambda as measure of market depth

## Universe Filters

| Parameter | Range | Internal Source | External Source | BR vs US |
|-----------|-------|-----------------|-----------------|----------|
| `min_market_cap` | R$200M-R$2B / $0.5B-$10B | `filters/mod.rs` | Size factor literature | BR: 10x lower |
| `max_annualized_vol` | 35-60% | `filters/mod.rs` | Low-vol anomaly | BR: 1.4x higher |
| `min_dividend_yield` | 1.5-2% | `filters/mod.rs` | Yield factor | Same |
| `min_carry` | -8% to 0% | `filters/carry.rs` | Carry factor literature | BR: allows negative |

### Academic Sources

1. **Fama & French (1992)** - Size factor
2. **Blitz & van Vliet (2007)** - Low-volatility anomaly
3. **Koijen et al. (2018)** - Carry factor across asset classes

---

## Recursos Avançados (v2.0) - Implementados

### CVaR Circuit Breaker

| Parâmetro | Range | Arquivo | Fonte Acadêmica |
|-----------|-------|---------|-----------------|
| `cvar_limit_95` | -10% a -30% | `exit/risk_guard.rs` | Rockafellar & Uryasev (2000) |
| `check_cvar` | bool | `exit/risk_guard.rs` | - |

**Implementação**: `RiskGuard::check_cvar()` - Calcula CVaR 95% a partir dos retornos diários e compara com limite.

### Kelly Dinâmico

| Parâmetro | Range | Arquivo | Fonte Acadêmica |
|-----------|-------|---------|-----------------|
| `kelly_lookback_trades` | 20-200 | `performance/kelly.rs` | Thorp (2006) |
| `kelly_min_trades` | 10-50 | `performance/kelly.rs` | Statistical significance |
| `kelly_max_fraction` | 0.25-0.50 | `performance/kelly.rs` | Ziemba (2011) |

**Implementação**: `KellyCalculator::calculate_kelly_fraction()` - Calcula fração ótima de Kelly baseado em win rate e payoff.

### Anti-Concentração via Drawdown Beta

| Parâmetro | Range | Arquivo | Fonte Acadêmica |
|-----------|-------|---------|-----------------|
| `check_drawdown_beta` | bool | `entry/selection.rs` | Daniel & Moskowitz (2016) |
| `max_drawdown_beta` | 0.5-1.0 | `entry/selection.rs` | Momentum crashes |

**Implementação**: `Selector::select_with_portfolio()` - Filtra candidatos por correlação de drawdown com portfolio.

### Limite de Correlação

| Parâmetro | Range | Arquivo | Fonte Acadêmica |
|-----------|-------|---------|-----------------|
| `check_correlation` | bool | `entry/selection.rs` | Portfolio theory |
| `max_correlation` | 0.5-0.8 | `entry/selection.rs` | Diversification |

**Implementação**: Rejeita novos ativos se correlação com portfolio existente excede threshold.

### Clusters de Risco

| Parâmetro | Range | Arquivo | Fonte Acadêmica |
|-----------|-------|---------|-----------------|
| `vol_threshold_high` | 0.30-0.50 | `risk_clusters.rs` | Volatility regimes |
| `vol_threshold_low` | 0.10-0.20 | `risk_clusters.rs` | Low-vol anomaly |
| `liquidity_threshold_high` | $20M-$100M | `risk_clusters.rs` | Market microstructure |
| `liquidity_threshold_low` | $1M-$10M | `risk_clusters.rs` | Illiquidity premium |

**Implementação**: `RiskClusterer::classify_asset()` - Classifica ativos em 5 clusters (BlueChip, GrowthStock, ValueStock, Speculative, Standard).

### Análise de Sensibilidade

| Parâmetro | Range | Arquivo | Fonte Acadêmica |
|-----------|-------|---------|-----------------|
| `perturbation_pct` | 5-20% | `experiment/sensitivity.rs` | Bailey & López de Prado (2014) |
| `num_perturbations` | 10-50 | `experiment/sensitivity.rs` | Statistical robustness |
| `min_stability_score` | 0.5-0.8 | `experiment/sensitivity.rs` | Anti-overfitting |

**Implementação**: `SensitivityAnalyzer::run_analysis()` - Perturba parâmetros e mede degradação de performance.

---

## Code Locations

| Component | File Path |
|-----------|-----------|
| Risk Profile Enum | `crates/backtester_intelligence/src/risk_profiles/profile.rs` |
| Profile Parameters | `crates/backtester_intelligence/src/risk_profiles/types.rs` |
| Profile Loader | `crates/backtester_intelligence/src/risk_profiles/loader.rs` |
| TOML Configs | `configs/risk_profiles/*.toml` |
| Market Defaults | `crates/backtester_intelligence/src/filters/mod.rs` |
| Entry Guardrails | `crates/backtester_intelligence/src/entry/engine.rs` |
| Exit Risk Guard | `crates/backtester_intelligence/src/exit/risk_guard.rs` |
| Report Warnings | `crates/backtester_reports/src/lib.rs` |
| **Kelly Calculator** | `crates/backtester_intelligence/src/performance/kelly.rs` |
| **Risk Clusters** | `crates/backtester_intelligence/src/risk_clusters.rs` |
| **Sensitivity Analysis** | `crates/backtester_strategy/src/experiment/sensitivity.rs` |

## Parameter Validation Rules

1. **Kelly Fraction**: Must be in (0, 0.5] - Half-Kelly maximum per Thorp (2006)
2. **Risk per Trade**: Must be in (0, 0.03] - 3% absolute maximum
3. **ATR Multiplier**: Must be >= 1.0 - below 1x ATR causes excessive whipsaws
4. **Max Drawdown**: Must be negative - represents loss limit
5. **Volatility Target**: Must be positive - represents risk budget

## BR vs US Calibration Rationale

### Why BR Needs Different Parameters

1. **Higher Volatility**: B3 average annualized vol ~30% vs S&P 500 ~15%
2. **Lower Liquidity**: Smaller market with fewer participants
3. **Wider Spreads**: Less competition among market makers
4. **Currency Risk**: BRL volatility adds to total position risk
5. **Interest Rate Environment**: High Selic makes carry calculation different

### Adjustment Factors

| Metric | BR Adjustment | Justification |
|--------|---------------|---------------|
| ATR Multiplier | +20% | Wider stops for higher vol |
| Vol Target | +25% | Accept higher portfolio vol |
| Min Liquidity | -75% | Smaller market |
| Max Spread | +100% | Less liquid market |
| Min Carry | -6% to -8% | Selic typically > DY |



