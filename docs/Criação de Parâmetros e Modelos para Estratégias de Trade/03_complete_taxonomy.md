# Complete Trading Strategies Taxonomy

## INTRADAY STRATEGIES (1-Hour Bars)

### 1. Opening Range Breakout (ORB)
- **Timeframe**: First 30-60 minutes of trading
- **Holding**: 1-6 hours
- **Logic**: Trade breakout above/below opening range
- **Parameters**: Range period (30/60 min), breakout threshold, volume filter
- **Indicators**: VWAP, volume, ATR

### 2. VWAP Trading
- **Timeframe**: Intraday (1h bars)
- **Holding**: 1-4 hours
- **Logic**: Mean reversion to VWAP or trend following from VWAP
- **Parameters**: VWAP period, deviation bands, entry/exit thresholds
- **Indicators**: VWAP, standard deviation bands

### 3. Mean Reversion Intraday
- **Timeframe**: 1-hour bars
- **Holding**: 2-6 hours
- **Logic**: Buy oversold, sell overbought conditions
- **Parameters**: RSI period (14), Bollinger Bands (20,2), entry/exit levels
- **Indicators**: RSI, Bollinger Bands, Stochastic

### 4. Momentum Intraday
- **Timeframe**: 1-hour bars
- **Holding**: 2-8 hours
- **Logic**: Follow strong price movements with high volume
- **Parameters**: Momentum period, volume threshold, trend strength
- **Indicators**: MACD, RSI, ADX, Volume

### 5. Gap Trading
- **Timeframe**: First hours after open
- **Holding**: 1-4 hours
- **Logic**: Trade gap fill or gap continuation
- **Parameters**: Gap size threshold, volume confirmation
- **Indicators**: Opening gap %, volume, previous day range

### 6. Volume Profile Trading
- **Timeframe**: 1-hour bars
- **Holding**: 2-6 hours
- **Logic**: Trade based on volume clusters and value areas
- **Parameters**: POC (Point of Control), VAH/VAL, volume nodes
- **Indicators**: Volume Profile, VWAP

### 7. News-Based Trading
- **Timeframe**: Event-driven (1h bars)
- **Holding**: 1-6 hours
- **Logic**: Trade volatility expansion after news
- **Parameters**: News impact score, volatility threshold
- **Indicators**: ATR, volume spike, price momentum

---

## DAILY+ STRATEGIES

### 8. Swing Trading - Momentum
- **Timeframe**: Daily bars
- **Holding**: 2-10 days
- **Logic**: Ride short-term trends
- **Parameters**: MA crossover (20/50), RSI (14), trend strength
- **Indicators**: Moving averages, RSI, MACD

### 9. Swing Trading - Mean Reversion
- **Timeframe**: Daily bars
- **Holding**: 3-7 days
- **Logic**: Buy dips in uptrend, sell rallies in downtrend
- **Parameters**: Bollinger Bands (20,2), RSI (14), support/resistance
- **Indicators**: Bollinger Bands, RSI, Stochastic

### 10. Swing Trading - Breakout
- **Timeframe**: Daily bars
- **Holding**: 3-15 days
- **Logic**: Trade breakouts from consolidation patterns
- **Parameters**: Consolidation period, breakout volume, ATR
- **Indicators**: Price channels, volume, ATR

### 11. Position Trading - Trend Following
- **Timeframe**: Daily/weekly bars
- **Holding**: Weeks to months
- **Logic**: Follow major trends using moving averages
- **Parameters**: MA periods (50/200), ADX threshold, trend confirmation
- **Indicators**: Moving averages, ADX, MACD

### 12. Position Trading - Fundamental
- **Timeframe**: Daily bars
- **Holding**: Months to years
- **Logic**: Buy undervalued, sell overvalued based on fundamentals
- **Parameters**: P/E ratio, P/B ratio, dividend yield, growth metrics
- **Indicators**: Fundamental ratios, technical confirmation

### 13. Pair Trading - Cointegration
- **Timeframe**: Daily bars
- **Holding**: 5-30 days
- **Logic**: Trade spread between cointegrated pairs
- **Parameters**: Cointegration p-value, z-score threshold (±2), half-life
- **Indicators**: Spread, z-score, correlation

### 14. Pair Trading - Distance Method
- **Timeframe**: Daily bars
- **Holding**: 5-20 days
- **Logic**: Trade when spread exceeds historical threshold
- **Parameters**: Lookback period (60-252 days), entry/exit thresholds
- **Indicators**: Price ratio, moving average of ratio

### 15. Statistical Arbitrage - Multi-Pair
- **Timeframe**: Daily bars
- **Holding**: 3-15 days
- **Logic**: Trade multiple cointegrated pairs simultaneously
- **Parameters**: Number of pairs, correlation matrix, risk allocation
- **Indicators**: Spread z-scores, portfolio correlation

### 16. Portfolio Trading - Equal Weight
- **Timeframe**: Daily bars
- **Rebalancing**: Daily/weekly
- **Logic**: Equal allocation across selected assets
- **Parameters**: Number of assets, rebalancing frequency, universe selection
- **Indicators**: Momentum, volatility for selection

### 17. Portfolio Trading - Risk Parity
- **Timeframe**: Daily bars
- **Rebalancing**: Daily/weekly
- **Logic**: Equal risk contribution from each asset
- **Parameters**: Volatility lookback, target volatility, leverage
- **Indicators**: Historical volatility, correlation matrix

### 18. Portfolio Trading - Minimum Variance
- **Timeframe**: Daily bars
- **Rebalancing**: Weekly/monthly
- **Logic**: Minimize portfolio variance
- **Parameters**: Covariance lookback, constraints, optimization method
- **Indicators**: Covariance matrix, expected returns

### 19. Portfolio Trading - Maximum Sharpe
- **Timeframe**: Daily bars
- **Rebalancing**: Weekly/monthly
- **Logic**: Maximize risk-adjusted returns
- **Parameters**: Return lookback, risk-free rate, constraints
- **Indicators**: Expected returns, covariance matrix

### 20. Momentum - Cross-Sectional
- **Timeframe**: Daily bars
- **Holding**: 1-3 months
- **Logic**: Long top performers, short bottom performers
- **Parameters**: Lookback period (3/6/12 months), rebalancing frequency
- **Indicators**: Price momentum, volume

### 21. Momentum - Time-Series
- **Timeframe**: Daily bars
- **Holding**: 1-6 months
- **Logic**: Long assets with positive trend, short negative trend
- **Parameters**: Trend period (50/200 days), signal smoothing
- **Indicators**: Moving averages, trend strength

### 22. Mean Reversion - Bollinger Bands
- **Timeframe**: Daily bars
- **Holding**: 2-10 days
- **Logic**: Buy at lower band, sell at upper band
- **Parameters**: BB period (20), standard deviations (2), exit rules
- **Indicators**: Bollinger Bands, RSI

### 23. Mean Reversion - RSI
- **Timeframe**: Daily bars
- **Holding**: 2-7 days
- **Logic**: Buy oversold (RSI<30), sell overbought (RSI>70)
- **Parameters**: RSI period (14), thresholds, trend filter
- **Indicators**: RSI, moving averages

### 24. Breakout - Donchian Channel
- **Timeframe**: Daily bars
- **Holding**: 5-30 days
- **Logic**: Buy new highs, sell new lows
- **Parameters**: Channel period (20/55), exit rules
- **Indicators**: Donchian Channel, ATR

### 25. Breakout - Volatility Expansion
- **Timeframe**: Daily bars
- **Holding**: 3-15 days
- **Logic**: Trade breakouts with expanding volatility
- **Parameters**: ATR threshold, consolidation period, volume
- **Indicators**: ATR, Bollinger Band width, volume

### 26. Sector Rotation - Business Cycle
- **Timeframe**: Daily/weekly bars
- **Holding**: 1-6 months
- **Logic**: Rotate sectors based on economic cycle phase
- **Parameters**: Cycle indicators, sector strength metrics
- **Indicators**: Economic indicators, relative strength

### 27. Sector Rotation - Relative Strength
- **Timeframe**: Daily bars
- **Holding**: 2-12 weeks
- **Logic**: Invest in strongest sectors vs benchmark
- **Parameters**: RS lookback period, rebalancing frequency
- **Indicators**: Relative strength ratio, momentum

### 28. Factor Investing - Value
- **Timeframe**: Daily bars
- **Holding**: Months to years
- **Logic**: Long low P/E, P/B stocks
- **Parameters**: Factor metrics, rebalancing frequency, universe
- **Indicators**: P/E, P/B, dividend yield

### 29. Factor Investing - Quality
- **Timeframe**: Daily bars
- **Holding**: Months to years
- **Logic**: Long high ROE, low debt stocks
- **Parameters**: Quality metrics, scoring system
- **Indicators**: ROE, debt/equity, profit margins

### 30. Factor Investing - Low Volatility
- **Timeframe**: Daily bars
- **Holding**: Months
- **Logic**: Long low volatility stocks
- **Parameters**: Volatility lookback, rebalancing frequency
- **Indicators**: Historical volatility, beta

### 31. Factor Investing - Multi-Factor
- **Timeframe**: Daily bars
- **Holding**: Months
- **Logic**: Combine value, momentum, quality, low vol
- **Parameters**: Factor weights, scoring system, rebalancing
- **Indicators**: Composite factor score

### 32. Seasonal Trading - Calendar Effects
- **Timeframe**: Daily bars
- **Holding**: Days to weeks
- **Logic**: Trade based on seasonal patterns (January effect, sell in May)
- **Parameters**: Entry/exit dates, historical pattern strength
- **Indicators**: Historical seasonality, trend confirmation

### 33. Seasonal Trading - Commodity Seasonality
- **Timeframe**: Daily bars
- **Holding**: Weeks to months
- **Logic**: Trade commodities based on seasonal supply/demand
- **Parameters**: Seasonal windows, confirmation filters
- **Indicators**: Historical seasonal patterns, fundamentals

### 34. Volatility Trading - VIX Mean Reversion
- **Timeframe**: Daily bars
- **Holding**: 3-15 days
- **Logic**: Trade volatility reversion to mean
- **Parameters**: VIX thresholds, mean lookback period
- **Indicators**: VIX, VIX term structure

### 35. Volatility Trading - Vol Breakout
- **Timeframe**: Daily bars
- **Holding**: 5-20 days
- **Logic**: Trade volatility expansion events
- **Parameters**: ATR threshold, Bollinger Band width
- **Indicators**: ATR, BB width, volume

### 36. Event-Driven - Earnings
- **Timeframe**: Daily bars
- **Holding**: 1-5 days
- **Logic**: Trade pre/post earnings announcements
- **Parameters**: Earnings surprise threshold, volatility filter
- **Indicators**: Historical earnings reaction, IV

### 37. Event-Driven - M&A Arbitrage
- **Timeframe**: Daily bars
- **Holding**: Days to months
- **Logic**: Trade merger spreads
- **Parameters**: Deal spread, completion probability
- **Indicators**: Spread to deal price, deal risk factors

### 38. Buy and Hold - Index
- **Timeframe**: Daily bars
- **Holding**: Years
- **Logic**: Passive index tracking
- **Parameters**: Index composition, rebalancing frequency
- **Indicators**: None (passive)

### 39. Buy and Hold - Dividend Growth
- **Timeframe**: Daily bars
- **Holding**: Years
- **Logic**: Long-term dividend growth stocks
- **Parameters**: Dividend yield, growth rate, payout ratio
- **Indicators**: Dividend metrics, fundamental quality

### 40. Multi-Strategy - Momentum + Mean Reversion
- **Timeframe**: Daily bars
- **Holding**: Variable
- **Logic**: Combine trend following and mean reversion
- **Parameters**: Regime detection, strategy allocation
- **Indicators**: Trend strength, volatility regime

---

## TIMEFRAME DEFINITIONS

### Ultra-Short (NOT APPLICABLE - requires <1h data)
- Scalping: seconds to minutes
- High-frequency: milliseconds to seconds

### Short-Term (1h bars - LIMITED APPLICATION)
- Intraday: 1-8 hours
- Day trading: within single day

### Short to Medium (PRIMARY FOCUS)
- Swing trading: 2-10 days
- Short-term position: 1-4 weeks

### Medium-Term
- Position trading: 1-6 months
- Tactical allocation: 3-12 months

### Long-Term
- Strategic allocation: 1+ years
- Buy and hold: multi-year

---

## KEY PARAMETER CATEGORIES

### 1. Entry/Exit Rules
- Technical indicators and thresholds
- Fundamental criteria
- Statistical signals (z-score, p-value)

### 2. Position Sizing
- Fixed dollar amount
- Fixed percentage
- Volatility-based (ATR, historical vol)
- Risk parity
- Kelly criterion

### 3. Risk Management
- Stop-loss (fixed, trailing, ATR-based)
- Take-profit targets
- Maximum position size
- Maximum drawdown limit
- Correlation limits

### 4. Time Management
- Holding period constraints
- Time-based exits
- Rebalancing frequency
- Lookback periods

### 5. Universe Selection
- Market (BR, US, both)
- Asset class (stocks, futures, FX)
- Sector/industry filters
- Liquidity requirements
- Market cap filters

### 6. Cost Assumptions
- Commission structure
- Slippage model
- Borrowing costs (for shorts)
- Market impact

### 7. Optimization Targets
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- Maximum drawdown
- Win rate
- Profit factor
