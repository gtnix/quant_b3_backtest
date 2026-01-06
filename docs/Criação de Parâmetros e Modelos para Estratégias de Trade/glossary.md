# Trade Parameters Module (TPM) - Glossary & Taxonomy

## Module Purpose

The **Trade Parameters Module (TPM)** provides pre-configured trading strategy templates that guide the genetic algorithm strategy generator. Instead of generating completely generic strategies, TPM constrains the search space to proven trading methodologies, improving the quality and relevance of generated strategies.

## Strategy Classification System

### By Timeframe

| Category | Timeframe | Holding Period | Data Requirements | Use Cases |
|----------|-----------|----------------|-------------------|-----------|
| **Intraday** | 1-hour bars | 1-8 hours | Intraday OHLCV | Day trading, scalping (limited) |
| **Short-Term** | Daily bars | 2-10 days | Daily OHLCV | Swing trading, tactical |
| **Medium-Term** | Daily bars | 2-12 weeks | Daily OHLCV + fundamentals | Position trading, sector rotation |
| **Long-Term** | Daily/weekly bars | 3+ months | Daily OHLCV + fundamentals | Strategic allocation, factor investing |

### By Market Hypothesis

| Type | Hypothesis | Logic | Best Timeframe |
|------|-----------|-------|----------------|
| **Momentum** | Trends persist | Follow the trend | All timeframes |
| **Mean Reversion** | Prices revert to mean | Fade extremes | Short to medium |
| **Breakout** | Consolidation precedes moves | Trade range breaks | All timeframes |
| **Statistical Arbitrage** | Relationships are stable | Trade deviations | Daily+ |
| **Fundamental** | Price follows value | Buy undervalued | Long-term |

### By Position Structure

| Type | Structure | Risk Profile | Complexity |
|------|-----------|--------------|------------|
| **Directional** | Long or short single asset | Market exposure | Low |
| **Pair** | Long + short related assets | Market neutral | Medium |
| **Portfolio** | Multiple positions | Diversified | High |
| **Multi-Strategy** | Combined approaches | Adaptive | Very high |

---

## Complete Strategy Taxonomy

### INTRADAY STRATEGIES (1h bars)

#### 1. Opening Range Breakout (ORB)

**Description**: Trades breakouts from the first 30-60 minutes of the trading session.

**Core Parameters**:
- `range_period`: 30 or 60 minutes
- `breakout_threshold`: % above/below range (0.1-0.5%)
- `volume_filter`: Minimum volume vs average (1.5-3x)
- `stop_loss`: ATR-based or fixed %
- `profit_target`: Risk/reward ratio (1.5-3x)

**Typical Configuration**:
```toml
[strategy]
type = "orb_breakout"
timeframe = "1h"
holding_period_hours = [1, 6]

[parameters]
range_period_minutes = 60
breakout_threshold_pct = 0.2
volume_multiplier = 2.0
stop_loss_atr = 2.0
profit_target_rr = 2.0
```

#### 2. VWAP Trading

**Description**: Mean reversion or trend following based on Volume Weighted Average Price.

**Core Parameters**:
- `vwap_period`: Intraday reset or rolling
- `deviation_bands`: Standard deviations (1-3)
- `entry_mode`: "mean_reversion" or "trend_following"
- `exit_mode`: "vwap_cross" or "opposite_band"

**Variants**:
- VWAP Mean Reversion: Buy at lower band, sell at upper band
- VWAP Trend: Buy above VWAP with momentum, sell below

#### 3. Intraday Mean Reversion

**Description**: Exploits short-term oversold/overbought conditions within the day.

**Core Parameters**:
- `rsi_period`: 14 typical
- `rsi_oversold`: 20-30
- `rsi_overbought`: 70-80
- `bb_period`: 20
- `bb_std`: 2.0
- `trend_filter`: Use MA to avoid counter-trend

**Indicators**: RSI, Bollinger Bands, Stochastic

#### 4. Intraday Momentum

**Description**: Follows strong intraday price movements with volume confirmation.

**Core Parameters**:
- `momentum_period`: 3-10 bars
- `volume_threshold`: 1.5-3x average
- `trend_strength`: ADX > 25
- `entry_trigger`: MACD crossover or RSI > 60

**Indicators**: MACD, RSI, ADX, Volume

#### 5. Gap Trading

**Description**: Trades gaps between previous close and current open.

**Core Parameters**:
- `gap_size_min`: Minimum gap % (0.5-2%)
- `gap_type`: "gap_fill" or "gap_continuation"
- `volume_confirmation`: Required volume spike
- `time_limit`: Maximum hours to hold (2-4)

**Variants**:
- Gap Fill: Fade the gap (mean reversion)
- Gap Continuation: Follow the gap direction (momentum)

#### 6. Volume Profile Trading

**Description**: Trades based on volume distribution and value areas.

**Core Parameters**:
- `poc_proximity`: Distance from Point of Control
- `vah_val_breakout`: Trade breaks of value area
- `volume_node_strength`: Minimum volume at level

**Indicators**: Volume Profile, POC, VAH/VAL

#### 7. News-Based Intraday

**Description**: Trades volatility expansion following news events.

**Core Parameters**:
- `news_impact_threshold`: High/medium impact only
- `volatility_expansion`: ATR increase threshold
- `entry_delay`: Minutes after news (5-30)
- `max_holding`: Hours after event (2-6)

---

### DAILY+ STRATEGIES

#### 8-10. Swing Trading Family

**8. Swing Momentum**
- Logic: Ride 2-10 day trends
- Indicators: MA crossover (20/50), RSI, MACD
- Holding: 2-10 days
- Entry: MA crossover + RSI > 50
- Exit: Opposite crossover or profit target

**9. Swing Mean Reversion**
- Logic: Buy dips in uptrend, sell rallies in downtrend
- Indicators: Bollinger Bands, RSI, support/resistance
- Holding: 3-7 days
- Entry: Price at lower BB + RSI < 30 + uptrend
- Exit: Price at upper BB or RSI > 70

**10. Swing Breakout**
- Logic: Trade consolidation breakouts
- Indicators: Price channels, volume, ATR
- Holding: 3-15 days
- Entry: Breakout with volume > 2x average
- Exit: Trailing stop or channel reversal

#### 11-12. Position Trading Family

**11. Position Trend Following**
- Logic: Follow major trends using moving averages
- Indicators: MA (50/200), ADX, MACD
- Holding: Weeks to months
- Entry: Golden cross (50 MA > 200 MA) + ADX > 25
- Exit: Death cross or trailing stop

**12. Position Fundamental**
- Logic: Buy undervalued, hold until fair value
- Indicators: P/E, P/B, dividend yield, technical confirmation
- Holding: Months to years
- Entry: Low P/E + positive technical setup
- Exit: Target valuation or technical breakdown

#### 13-15. Pair Trading Family

**13. Pair Trading - Cointegration**
- Logic: Trade mean-reverting spread between cointegrated pairs
- Statistical Test: Engle-Granger or Johansen test (p < 0.05)
- Entry: Z-score > +2 (short spread) or < -2 (long spread)
- Exit: Z-score crosses 0 or stop at ±3
- Parameters:
  - `lookback_period`: 60-252 days
  - `z_score_entry`: ±2.0
  - `z_score_exit`: 0.0
  - `z_score_stop`: ±3.0
  - `half_life_max`: 30 days

**14. Pair Trading - Distance Method**
- Logic: Trade when price ratio exceeds historical bounds
- Entry: Ratio > mean + 2*std or < mean - 2*std
- Exit: Ratio returns to mean
- Simpler than cointegration, no statistical test required

**15. Statistical Arbitrage - Multi-Pair**
- Logic: Portfolio of multiple pair trades
- Diversification: 5-20 pairs simultaneously
- Risk: Maximum 2-5% per pair
- Correlation: Avoid highly correlated pairs

#### 16-19. Portfolio Trading Family

**16. Equal Weight Portfolio**
- Logic: Equal dollar allocation across N assets
- Rebalancing: Daily, weekly, or monthly
- Selection: Top N by momentum, quality, or other factor
- Parameters:
  - `num_assets`: 5-30
  - `rebalance_frequency`: "daily" | "weekly" | "monthly"
  - `selection_method`: "momentum" | "quality" | "volatility"

**17. Risk Parity Portfolio**
- Logic: Equal risk contribution from each asset
- Calculation: Weight inversely proportional to volatility
- Leverage: May use leverage to target volatility
- Parameters:
  - `target_volatility`: 10-20% annualized
  - `vol_lookback`: 60-252 days
  - `max_leverage`: 1.0-3.0

**18. Minimum Variance Portfolio**
- Logic: Minimize portfolio variance using optimization
- Method: Quadratic programming
- Constraints: Long-only or long-short
- Rebalancing: Weekly or monthly (computationally intensive)

**19. Maximum Sharpe Portfolio**
- Logic: Maximize risk-adjusted returns
- Method: Mean-variance optimization
- Inputs: Expected returns, covariance matrix
- Challenge: Sensitive to input estimates

#### 20-21. Momentum Family

**20. Cross-Sectional Momentum**
- Logic: Long top decile, short bottom decile by past returns
- Lookback: 3, 6, or 12 months (skip last month)
- Rebalancing: Monthly
- Universe: Broad market or sector-specific

**21. Time-Series Momentum**
- Logic: Long if positive trend, short if negative trend
- Signal: Price vs MA (50/200 day)
- Absolute strategy: Each asset independent
- Rebalancing: Daily or weekly

#### 22-23. Mean Reversion Family

**22. Bollinger Band Mean Reversion**
- Entry: Price touches lower band (buy) or upper band (sell)
- Exit: Price returns to middle band or opposite band
- Filter: Trend filter to avoid counter-trend
- Parameters: BB(20, 2), trend MA(50)

**23. RSI Mean Reversion**
- Entry: RSI < 30 (buy) or RSI > 70 (sell)
- Exit: RSI crosses 50 or opposite extreme
- Filter: Only trade in direction of longer-term trend
- Parameters: RSI(14), trend MA(100)

#### 24-25. Breakout Family

**24. Donchian Channel Breakout**
- Entry: Price breaks above N-day high (buy) or below N-day low (sell)
- Exit: Opposite signal or trailing stop
- Classic: Turtle Traders used 20-day and 55-day channels
- Parameters: Channel period (20/55), ATR stop

**25. Volatility Expansion Breakout**
- Entry: Breakout from low volatility consolidation
- Signal: ATR expanding + price breakout + volume
- Logic: "Coiled spring" releases energy
- Parameters: ATR threshold, consolidation period

#### 26-27. Sector Rotation Family

**26. Business Cycle Rotation**
- Logic: Rotate sectors based on economic cycle phase
- Phases: Early expansion, mid expansion, late expansion, recession
- Sectors: Technology (early), industrials (mid), energy (late), utilities (recession)
- Indicators: GDP growth, yield curve, unemployment

**27. Relative Strength Rotation**
- Logic: Invest in sectors with strongest relative performance
- Calculation: Sector ETF / Market Index
- Rebalancing: Monthly
- Hold: Top 3-5 sectors by RS

#### 28-31. Factor Investing Family

**28. Value Factor**
- Metrics: Low P/E, low P/B, high dividend yield
- Logic: Undervalued stocks outperform long-term
- Rebalancing: Quarterly or annually
- Universe: Large/mid cap

**29. Quality Factor**
- Metrics: High ROE, low debt/equity, stable earnings
- Logic: High-quality companies are more resilient
- Rebalancing: Quarterly or annually
- Combine with: Value or momentum

**30. Low Volatility Factor**
- Metrics: Low historical volatility, low beta
- Logic: Low-vol stocks have better risk-adjusted returns
- Paradox: Contradicts CAPM
- Rebalancing: Monthly or quarterly

**31. Multi-Factor**
- Logic: Combine value, momentum, quality, low-vol
- Method: Composite score or separate sleeves
- Weights: Equal or optimized
- Diversification: Reduces factor-specific risk

#### 32-33. Seasonal Trading Family

**32. Calendar Effects**
- Patterns: January effect, sell in May, Santa rally
- Logic: Recurring behavioral patterns
- Entry/Exit: Fixed dates with confirmation
- Examples:
  - January Effect: Buy small caps in late December
  - Sell in May: Exit May 1, re-enter November 1
  - Santa Rally: Buy mid-December

**33. Commodity Seasonality**
- Logic: Supply/demand cycles (harvest, weather, etc.)
- Examples:
  - Natural gas: High demand in winter
  - Grains: Harvest season effects
  - Energy: Summer driving season
- Confirmation: Combine with technical signals

#### 34-35. Volatility Trading Family

**34. VIX Mean Reversion**
- Logic: VIX tends to revert to long-term mean (~15-20)
- Entry: VIX > 30 (buy stocks) or VIX < 12 (sell stocks)
- Instruments: VIX futures, volatility ETFs, or equity index
- Holding: 3-15 days

**35. Volatility Breakout**
- Logic: Trade volatility regime changes
- Signal: ATR expansion, Bollinger Band width increase
- Entry: Volatility breaks above threshold
- Direction: Combine with momentum or breakout signal

#### 36-37. Event-Driven Family

**36. Earnings Trading**
- Logic: Trade pre/post earnings announcements
- Strategies:
  - Pre-earnings: Buy positive momentum stocks
  - Post-earnings: Trade earnings surprise
- Risk: High volatility, gap risk
- Holding: 1-5 days around event

**37. M&A Arbitrage**
- Logic: Profit from merger spreads
- Entry: Buy target at discount to deal price
- Risk: Deal break risk
- Holding: Until deal closes (days to months)
- Advanced: Hedge with short acquirer

#### 38-39. Buy and Hold Family

**38. Index Buy and Hold**
- Logic: Passive index tracking
- Instruments: Broad market ETFs (IBOV, S&P 500)
- Rebalancing: Only when index changes
- Costs: Minimal

**39. Dividend Growth Buy and Hold**
- Logic: Long-term compounding via dividend growth
- Selection: Dividend aristocrats, high yield + growth
- Holding: Years to decades
- Rebalancing: Minimal, only on fundamental deterioration

#### 40. Multi-Strategy Adaptive

**Description**: Combines multiple strategies and switches based on market regime.

**Components**:
- Momentum strategy for trending markets
- Mean reversion for ranging markets
- Defensive allocation for high volatility

**Regime Detection**:
- Trend strength: ADX, MA slope
- Volatility: ATR, VIX
- Correlation: Market correlation regime

**Allocation**:
- Dynamic weights based on regime
- Risk parity across strategies
- Drawdown-based switching

---

## Parameter Categories Reference

### Entry/Exit Signals

| Parameter Type | Examples | Typical Range |
|----------------|----------|---------------|
| **Moving Average** | SMA, EMA period | 10-200 days |
| **Oscillator** | RSI, Stochastic | Period: 14, Levels: 30/70 |
| **Volatility** | ATR, Bollinger Bands | Period: 20, Std: 2 |
| **Volume** | Volume MA, volume ratio | 20-50 period, 1.5-3x threshold |
| **Statistical** | Z-score, p-value | Entry: ±2, Stop: ±3 |

### Position Sizing Methods

| Method | Description | Parameters |
|--------|-------------|------------|
| **Fixed Dollar** | Same $ amount per trade | `position_size_usd` |
| **Fixed Percent** | Same % of capital | `position_size_pct`: 1-10% |
| **Volatility-Based** | Inverse to volatility | `target_risk_pct`: 1-2% |
| **Risk Parity** | Equal risk contribution | `target_volatility`: 10-20% |
| **Kelly Criterion** | Optimal growth | `kelly_fraction`: 0.25-0.5 |

### Risk Management

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| **Stop Loss** | Maximum loss per trade | 1-5% or 2-3 ATR |
| **Take Profit** | Profit target | 2-5% or 1.5-3x risk |
| **Max Position** | Per-asset limit | 5-20% of capital |
| **Max Drawdown** | Portfolio limit | 15-25% |
| **Correlation Limit** | Between positions | < 0.7 |

### Time Management

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| **Holding Period Min** | Minimum hold time | 1-5 days |
| **Holding Period Max** | Maximum hold time | 10-60 days |
| **Rebalancing Frequency** | Portfolio adjustment | Daily/weekly/monthly |
| **Lookback Period** | Historical data window | 20-252 days |

---

## Timeframe-Strategy Mapping

### Intraday (1h bars) - Best Strategies
1. Opening Range Breakout
2. VWAP Trading
3. Mean Reversion Intraday
4. Momentum Intraday
5. Gap Trading
6. Volume Profile
7. News-Based

### Short-Term (2-10 days) - Best Strategies
1. Swing Momentum
2. Swing Mean Reversion
3. Swing Breakout
4. Pair Trading (short-term)
5. Bollinger Band Mean Reversion
6. RSI Mean Reversion

### Medium-Term (2-12 weeks) - Best Strategies
1. Position Trend Following
2. Pair Trading (standard)
3. Statistical Arbitrage
4. Sector Rotation
5. Momentum (cross-sectional)
6. Breakout (Donchian)

### Long-Term (3+ months) - Best Strategies
1. Position Fundamental
2. Factor Investing (all types)
3. Buy and Hold variants
4. Sector Rotation (business cycle)
5. Seasonal Trading
6. Multi-Strategy Adaptive

---

## Data Requirements by Strategy Type

| Strategy Type | OHLCV | Volume Profile | Fundamentals | Alternative Data |
|---------------|-------|----------------|--------------|------------------|
| Intraday | Required | Optional | No | Optional (news) |
| Swing | Required | No | No | No |
| Pair Trading | Required | No | Optional | No |
| Portfolio | Required | No | Optional | No |
| Factor Investing | Required | No | Required | Optional |
| Event-Driven | Required | No | Optional | Required |

---

## Next Steps

This glossary will be used to generate 100+ specific strategy configurations in the next phase, with each configuration containing:
- Complete parameter specifications
- Timeframe mappings
- Risk profiles
- Computational requirements
- Expected performance characteristics
