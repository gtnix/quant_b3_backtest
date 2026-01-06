# Timeframe Mapping for TPM Module

## Overview

This document defines how trading strategies map to data windows, computational requirements, and expected performance characteristics. This mapping is critical for optimizing the genetic algorithm's search space and computational efficiency.

## Timeframe Definitions

### 1. Ultra-Short Term (Intraday 1h)
- **Data Window**: 1-6 months of 1-hour bars
- **Lookback**: 20-100 bars (20-100 hours)
- **Holding Period**: 1-8 hours
- **Rebalancing**: Every bar (hourly)
- **Minimum Trades**: 50+ for statistical significance
- **Computational Load**: HIGH (frequent rebalancing)

**Applicable Strategies**:
- Opening Range Breakout (ORB)
- VWAP Trading
- Intraday Mean Reversion
- Intraday Momentum
- Gap Trading
- Volume Profile Trading
- News-Based Trading

**Data Requirements**:
- OHLCV at 1-hour intervals
- Volume data essential
- Optional: News timestamps, market microstructure

**Optimization Considerations**:
- Transaction costs are critical (multiple trades per day)
- Slippage modeling essential
- Market hours constraints
- Overnight gap risk (if holding past close)

---

### 2. Short Term (2-10 days)
- **Data Window**: 1-3 years of daily bars
- **Lookback**: 20-60 days
- **Holding Period**: 2-10 days
- **Rebalancing**: Daily
- **Minimum Trades**: 100+ for statistical significance
- **Computational Load**: MEDIUM

**Applicable Strategies**:
- Swing Trading (all variants)
- Short-term Pair Trading
- Bollinger Band Mean Reversion
- RSI Mean Reversion
- Short-term Momentum

**Data Requirements**:
- Daily OHLCV
- Volume data important
- Optional: Fundamental data for filters

**Optimization Considerations**:
- Balance between trade frequency and costs
- Overnight risk exposure
- Weekend gap risk
- Moderate transaction costs

---

### 3. Medium Term (2-12 weeks)
- **Data Window**: 3-5 years of daily bars
- **Lookback**: 60-252 days (3-12 months)
- **Holding Period**: 2-12 weeks
- **Rebalancing**: Weekly or bi-weekly
- **Minimum Trades**: 50+ for statistical significance
- **Computational Load**: MEDIUM

**Applicable Strategies**:
- Position Trading (trend following)
- Standard Pair Trading
- Statistical Arbitrage
- Sector Rotation
- Cross-Sectional Momentum
- Donchian Channel Breakout
- Volatility Expansion Breakout

**Data Requirements**:
- Daily OHLCV
- Fundamental data recommended
- Sector/industry classification
- Economic indicators (for sector rotation)

**Optimization Considerations**:
- Lower transaction costs impact
- Focus on trend quality and persistence
- Drawdown management critical
- Correlation between positions

---

### 4. Long Term (3+ months)
- **Data Window**: 5-10 years of daily bars
- **Lookback**: 252-1260 days (1-5 years)
- **Holding Period**: 3+ months
- **Rebalancing**: Monthly or quarterly
- **Minimum Trades**: 20+ for statistical significance
- **Computational Load**: LOW to MEDIUM

**Applicable Strategies**:
- Position Trading (fundamental)
- Factor Investing (all types)
- Buy and Hold variants
- Business Cycle Sector Rotation
- Seasonal Trading
- Event-Driven (M&A)

**Data Requirements**:
- Daily OHLCV
- Fundamental data essential
- Economic indicators
- Alternative data (optional)

**Optimization Considerations**:
- Transaction costs minimal
- Focus on fundamental quality
- Long-term risk management
- Tax efficiency (if applicable)

---

## Data Window Recommendations by Strategy Family

| Strategy Family | Min Data Window | Optimal Data Window | Max Useful Window | Rationale |
|----------------|-----------------|---------------------|-------------------|-----------|
| **Intraday** | 3 months | 6 months | 12 months | Recent market microstructure most relevant |
| **Swing** | 1 year | 2-3 years | 5 years | Balance between recency and sample size |
| **Position** | 2 years | 3-5 years | 10 years | Capture full market cycles |
| **Pair Trading** | 2 years | 3 years | 5 years | Need stable cointegration relationships |
| **Portfolio** | 2 years | 3-5 years | 10 years | Correlation stability requires longer history |
| **Momentum** | 1 year | 2-3 years | 5 years | Momentum persistence varies by timeframe |
| **Mean Reversion** | 1 year | 2 years | 3 years | Recent volatility regime most relevant |
| **Breakout** | 1 year | 2-3 years | 5 years | Pattern recognition across market conditions |
| **Sector Rotation** | 5 years | 10 years | 20 years | Full business cycle coverage essential |
| **Factor Investing** | 5 years | 10 years | 20+ years | Long-term factor persistence |
| **Seasonal** | 10 years | 15-20 years | 30+ years | Statistical significance of seasonal patterns |
| **Volatility** | 2 years | 3-5 years | 10 years | Volatility regime changes |
| **Event-Driven** | 2 years | 3-5 years | 10 years | Event frequency and market reaction |
| **Buy and Hold** | 10 years | 20+ years | All available | Long-term compounding and drawdown analysis |

---

## Train/Test Split Recommendations

### General Guidelines
- **In-Sample (Training)**: 60-70% of data
- **Out-of-Sample (Testing)**: 30-40% of data
- **Walk-Forward**: Use for strategies with > 3 years of data

### By Strategy Type

**High-Frequency Strategies (Intraday)**:
- Train: 70% (4-5 months if 6-month window)
- Test: 30% (1-2 months)
- Rationale: Recent market microstructure changes quickly

**Medium-Frequency Strategies (Swing, Position)**:
- Train: 60-65%
- Test: 35-40%
- Walk-Forward: 5-10 folds recommended

**Low-Frequency Strategies (Factor, Seasonal)**:
- Train: 50-60%
- Test: 40-50%
- Walk-Forward: 10+ folds if data permits
- Rationale: Need robust out-of-sample validation

---

## Computational Efficiency Mapping

### Strategy Complexity Tiers

**Tier 1: Low Complexity (Fast Evaluation)**
- Single-asset directional strategies
- Simple technical indicators (MA, RSI, Bollinger Bands)
- Examples: Swing Momentum, Mean Reversion RSI
- **Evaluation Time**: < 100ms per backtest
- **Parallel Capacity**: 100+ concurrent evaluations

**Tier 2: Medium Complexity**
- Multi-asset strategies (2-10 assets)
- Pair trading, statistical arbitrage
- Portfolio optimization (equal weight, risk parity)
- Examples: Pair Trading, Portfolio Equal Weight
- **Evaluation Time**: 100-500ms per backtest
- **Parallel Capacity**: 50-100 concurrent evaluations

**Tier 3: High Complexity (Slow Evaluation)**
- Large portfolio optimization (20+ assets)
- Complex statistical models (cointegration, regime detection)
- Machine learning-based strategies
- Examples: Multi-Factor, Minimum Variance Portfolio
- **Evaluation Time**: 500ms-5s per backtest
- **Parallel Capacity**: 10-50 concurrent evaluations

**Tier 4: Very High Complexity**
- Full portfolio optimization with constraints
- Monte Carlo simulations
- Walk-forward analysis with multiple folds
- Examples: Maximum Sharpe Portfolio, Multi-Strategy Adaptive
- **Evaluation Time**: 5s-30s per backtest
- **Parallel Capacity**: 1-10 concurrent evaluations

---

## Strategy-Specific Timeframe Configurations

### Intraday Strategies

```toml
[timeframe]
bar_size = "1h"
data_window_months = 6
lookback_bars = 50
min_history_bars = 100

[validation]
train_test_split = 0.70
min_trades_is = 30
min_trades_oos = 15
wfa_enabled = false  # Not recommended for short windows
```

### Swing Trading

```toml
[timeframe]
bar_size = "1D"
data_window_years = 2
lookback_days = 60
min_history_days = 252

[validation]
train_test_split = 0.65
min_trades_is = 60
min_trades_oos = 30
wfa_enabled = true
wfa_folds = 5
```

### Position Trading

```toml
[timeframe]
bar_size = "1D"
data_window_years = 5
lookback_days = 252
min_history_days = 504

[validation]
train_test_split = 0.60
min_trades_is = 30
min_trades_oos = 15
wfa_enabled = true
wfa_folds = 10
```

### Pair Trading

```toml
[timeframe]
bar_size = "1D"
data_window_years = 3
lookback_days = 252  # For cointegration test
min_history_days = 504

[validation]
train_test_split = 0.65
min_trades_is = 40
min_trades_oos = 20
wfa_enabled = true
wfa_folds = 5
rolling_cointegration_test = true
```

### Portfolio Strategies

```toml
[timeframe]
bar_size = "1D"
data_window_years = 5
lookback_days = 252  # For correlation/covariance
min_history_days = 504

[validation]
train_test_split = 0.60
rebalance_frequency = "monthly"
min_rebalances_is = 24
min_rebalances_oos = 12
wfa_enabled = true
wfa_folds = 10
```

### Factor Investing

```toml
[timeframe]
bar_size = "1D"
data_window_years = 10
lookback_days = 252
min_history_days = 1260

[validation]
train_test_split = 0.60
rebalance_frequency = "quarterly"
min_rebalances_is = 16
min_rebalances_oos = 8
wfa_enabled = true
wfa_folds = 20
```

---

## Integration with Genetic Algorithm

### Population Size Recommendations

Based on strategy complexity and data window:

| Complexity Tier | Population Size | Generations | Total Evaluations |
|----------------|-----------------|-------------|-------------------|
| Tier 1 (Fast) | 200-500 | 100-200 | 20,000-100,000 |
| Tier 2 (Medium) | 100-200 | 50-100 | 5,000-20,000 |
| Tier 3 (Slow) | 50-100 | 30-50 | 1,500-5,000 |
| Tier 4 (Very Slow) | 20-50 | 20-30 | 400-1,500 |

### Fitness Function Adjustments

**Short-Term Strategies (Intraday, Swing)**:
- Higher weight on Sharpe ratio
- Penalty for excessive trading
- Reward for consistent daily/weekly returns

**Medium-Term Strategies (Position, Pair)**:
- Balance between Sharpe and Calmar ratio
- Reward for low correlation with market
- Penalty for large drawdowns

**Long-Term Strategies (Factor, Buy & Hold)**:
- Higher weight on CAGR and Calmar ratio
- Reward for factor purity
- Lower penalty for volatility

---

## Next Steps

This timeframe mapping will be used to:
1. Auto-configure data windows when user selects a strategy type
2. Set appropriate validation parameters
3. Optimize computational resource allocation
4. Guide the genetic algorithm's search space
