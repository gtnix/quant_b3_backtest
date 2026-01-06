# Pair Trading Strategy - Research Notes

## Source: Medium/Auquan - Pairs Trading using Data-Driven Techniques

### Core Concept

Pair trading is a **market-neutral strategy** that exploits temporary divergences in the spread between two cointegrated securities. The strategy profits from mean reversion of the price ratio, regardless of overall market direction.

### Underlying Principle

**Economic Link**: Two securities (X and Y) share underlying economic factors (e.g., Pepsi and Coca Cola, two banks, two oil companies).

**Spread Relationship**: The ratio or difference in prices (the "spread") remains relatively constant over time.

**Temporary Divergence**: Supply/demand changes, large orders, or news can cause temporary spread divergence.

**Mean Reversion**: The spread eventually reverts to its historical mean.

### Mathematical Foundation

The relationship follows: **Y = α·X + e**

Where:
- α = constant ratio (mean)
- e = white noise (random fluctuation)

### Cointegration vs Correlation

**Cointegration** (required for pairs trading):
- The ratio between two series varies around a mean
- The spread is stationary and mean-reverting
- Tested using cointegration test (p-value < 0.05)

**Correlation** (NOT sufficient):
- Two series can be highly correlated but not cointegrated (e.g., two diverging trends)
- Two series can be cointegrated but not correlated (e.g., normal distribution + square wave)

### Trading Logic

**Going Long the Ratio** (ratio is low):
- Buy the underperforming stock (Y)
- Sell the outperforming stock (X)
- Expect ratio to increase back to mean

**Going Short the Ratio** (ratio is high):
- Sell the overperforming stock (Y)
- Buy the underperforming stock (X)
- Expect ratio to decrease back to mean

### Market Neutrality

The strategy is **hedged**:
- Long position gains if security increases
- Short position gains if security decreases
- Immune to overall market movement
- Only profits from relative movement between the pair

### Key Parameters for TPM Module

1. **Pair Selection**:
   - Economic linkage (same sector, same industry, competitors)
   - Statistical cointegration (p-value threshold)
   - Correlation coefficient
   - Historical spread stability

2. **Entry Signals**:
   - Z-score threshold (typically ±2 standard deviations)
   - Spread deviation from mean
   - Lookback period for mean calculation (e.g., 20-60 days)

3. **Exit Signals**:
   - Spread reverts to mean (z-score near 0)
   - Stop-loss threshold (e.g., z-score > ±3)
   - Time-based exit (maximum holding period)
   - Half-life of mean reversion

4. **Position Sizing**:
   - Dollar-neutral (equal dollar amounts long and short)
   - Beta-neutral (adjust for different volatilities)
   - Risk parity approach

5. **Risk Management**:
   - Maximum position size per pair (2-5% of capital)
   - Maximum number of concurrent pairs
   - Correlation between pairs (avoid multiple pairs in same sector)
   - Stop-loss on spread widening

6. **Timeframe Considerations**:
   - Lookback period: 60-252 days (3 months to 1 year)
   - Holding period: 5-30 days typical
   - Rebalancing frequency: daily
   - Cointegration test frequency: weekly/monthly

7. **Transaction Costs**:
   - Higher sensitivity due to two-leg trades
   - Slippage on both entry and exit
   - Borrowing costs for short positions

### Strategy Variants

1. **Distance Method**: Trade when spread exceeds threshold
2. **Cointegration Method**: Use Engle-Granger or Johansen test
3. **Stochastic Spread Method**: Model spread as mean-reverting process
4. **Machine Learning Method**: Predict spread direction

### Data Requirements

- Daily OHLC prices
- Volume data
- Sector/industry classification
- Fundamental data for economic linkage
- Historical correlation and cointegration metrics

### Performance Metrics

- Sharpe ratio (should be high due to market neutrality)
- Maximum drawdown
- Win rate
- Average holding period
- Number of round trips
- Spread convergence rate

### Challenges

- **Multiple comparison bias**: Testing many pairs increases false positives
- **Regime changes**: Cointegration can break down
- **Transaction costs**: Two-leg trades double the costs
- **Liquidity**: Need sufficient liquidity for both legs
- **Execution risk**: Timing mismatch between legs
