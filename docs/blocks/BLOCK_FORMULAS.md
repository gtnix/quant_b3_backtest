# Block Mathematical Formulas

This document contains the mathematical foundations for all strategy blocks
implemented in the backtester. Each block includes references to academic papers
and the exact formulas used.

## Entry Blocks

### Donchian Channel (donchian)
**Reference:** Donchian (1960), Faith (2003) "Way of the Turtle"

```
Upper_t = max(High_{t-n}, ..., High_{t-1})
Lower_t = min(Low_{t-n}, ..., Low_{t-1})
Middle_t = (Upper_t + Lower_t) / 2

Entry Long:  Close_t > Upper_t
Entry Short: Close_t < Lower_t
Exit Long:   Close_t < Lower_exit (using shorter period)
```

### Opening Range Breakout (orb_breakout)
**Reference:** Crabel (1990) "Day Trading with Short Term Price Patterns"

```
OR_high = max(High) during first n bars
OR_low = min(Low) during first n bars
OR_range = OR_high - OR_low

Long Entry:  Price > OR_high + (stretch_factor × OR_range)
Short Entry: Price < OR_low - (stretch_factor × OR_range)
```

### VWAP (vwap)
**Reference:** Berkowitz et al. (1988), Almgren & Chriss (2000)

```
VWAP = Σ(Price_i × Volume_i) / Σ(Volume_i)

Z_score = (Price - VWAP) / σ_VWAP

Reversion Mode:
  Long:  Z_score < -threshold
  Short: Z_score > +threshold

Trend Mode:
  Long:  Z_score > +threshold
  Short: Z_score < -threshold
```

### RSI (rsi)
**Reference:** Wilder (1978) "New Concepts in Technical Trading Systems"

```
RSI = 100 - (100 / (1 + RS))
RS = Average Gain / Average Loss

Long:  RSI < oversold_threshold (default 30)
Short: RSI > overbought_threshold (default 70)
```

### MACD (macd)
**Reference:** Appel (1979)

```
MACD_line = EMA(close, fast_period) - EMA(close, slow_period)
Signal_line = EMA(MACD_line, signal_period)
Histogram = MACD_line - Signal_line

Long:  MACD_line crosses above Signal_line
Short: MACD_line crosses below Signal_line
```

### Bollinger Bands (bollinger)
**Reference:** Bollinger (2002) "Bollinger on Bollinger Bands"

```
Middle = SMA(close, period)
Upper = Middle + (std_dev × σ)
Lower = Middle - (std_dev × σ)

σ = √(Σ(close_i - Middle)² / period)

Reversion:
  Long:  Close < Lower
  Short: Close > Upper
```

### ATR Breakout (atr_breakout)
**Reference:** Wilder (1978) "New Concepts in Technical Trading Systems"

```
TR = max(High - Low, |High - Close_prev|, |Low - Close_prev|)
ATR = SMA(TR, period)

Upper_band = Close_prev + (multiplier × ATR)
Lower_band = Close_prev - (multiplier × ATR)

Long:  Close > Upper_band
Short: Close < Lower_band
```

### ADX Momentum (adx_momentum)
**Reference:** Wilder (1978) "New Concepts in Technical Trading Systems"

```
+DM = High_t - High_{t-1} (if positive and > -DM)
-DM = Low_{t-1} - Low_t (if positive and > +DM)

+DI = 100 × EMA(+DM) / ATR
-DI = 100 × EMA(-DM) / ATR

DX = 100 × |+DI - -DI| / (+DI + -DI)
ADX = EMA(DX, period)

Long:  ADX > threshold AND +DI > -DI
Short: ADX > threshold AND -DI > +DI
```

### Volatility Expansion (vol_expansion)
**Reference:** Bollinger (2002), Mandelbrot (1963)

```
BBW = (Upper - Lower) / Middle = 2 × std_dev × num_std / SMA

Percentile = rank of current BBW vs historical BBWs

Squeeze = BBW_percentile < squeeze_threshold

Long:  Close > Upper AND in_squeeze
Short: Close < Lower AND in_squeeze
```

### Adaptive Momentum (adaptive_momentum)
**Reference:** Kaufman (1995) "Smarter Trading"

```
Efficiency_Ratio = |Price_t - Price_{t-n}| / Σ|Price_i - Price_{i-1}|
                 = Direction / Volatility

Adaptive_period = min_period + (1 - ER) × (max_period - min_period)

Momentum = (Price_t / Price_{t-adaptive_period}) - 1

Long:  Momentum > threshold
Short: Momentum < -threshold
```

### Gap Fill (gap_fill)
**Reference:** Cooper et al. (2003) "Market States and Momentum"

```
Gap% = (Open_today - Close_yesterday) / Close_yesterday

Long:  Gap% < -min_gap_pct (gap down, expect fill up)
Short: Gap% > +min_gap_pct (gap up, expect fill down)
```

### VIX Reversion (vix_reversion)
**Reference:** Whaley (2000) "The Investor Fear Gauge"

```
Realized_Vol = √252 × std(daily_returns, period)

Z_score = (RV - mean(RV_history)) / std(RV_history)

Contrarian Logic:
  Long:  Z_score > +threshold (high fear = buy stocks)
  Short: Z_score < -threshold (low fear = sell stocks)
```

### Volume Breakout (volume_breakout)
**Reference:** Arms (1971) "Volume Cycles in the Stock Market"

```
Volume_Ratio = Current_Volume / SMA(Volume, period)

Long:  Price > Highest_High(period) AND Volume_Ratio > threshold
Short: Price < Lowest_Low(period) AND Volume_Ratio > threshold
```

### Volume Profile (volume_profile)
**Reference:** Steidlmayer (1986) "Markets and Market Logic"

```
POC = Price level with highest cumulative volume
VAH = Upper bound containing 70% of volume
VAL = Lower bound containing 70% of volume

Long:  Price < VAL - deviation%
Short: Price > VAH + deviation%
```

## Selection Blocks

### Sector Rotation (sector_rotation)
**Reference:** Stovall (1996), Faber (2010)

```
Momentum_sector = (Price_t / Price_{t-lookback}) - 1

Rank sectors by momentum
Select top_n sectors
```

### Multi-Factor (multi_factor)
**Reference:** Fama & French (1993), Asness et al. (2013)

```
Score = Σ(weight_i × Z_score_i)

Factors:
- Momentum: 12-1 month return
- Value: Low P/E, Low P/B
- Quality: High ROE, Low Debt
- Low Volatility: Low annualized vol

Select top_pct% by combined score
```

## Sizing Blocks

### Equal Weight (equal_weight)
```
Weight_i = 1 / N for all N selected assets
```

### Risk Parity (risk_parity)
**Reference:** Qian (2005)

```
Weight_i ∝ 1 / σ_i

Normalized: Weight_i = (1/σ_i) / Σ(1/σ_j)
```

### Volatility Targeting (vol_targeting)
**Reference:** Moskowitz et al. (2012)

```
Target_leverage = target_vol / realized_portfolio_vol

Weight_i = (target_vol / σ_i) × (1/N)
```

## Exit Blocks

### Stop Loss (stop_loss)
```
Exit Long:  Price < Entry_price × (1 - threshold_pct)
Exit Short: Price > Entry_price × (1 + threshold_pct)
```

### Take Profit (take_profit)
```
Exit Long:  Price > Entry_price × (1 + target_pct)
Exit Short: Price < Entry_price × (1 - target_pct)
```

### Trailing Stop (trailing_stop)
```
Peak = max(prices since entry)

Exit Long: Price < Peak × (1 - trailing_pct)
         AND gain >= activation_pct
```

### Time Exit (time_exit)
```
Exit: days_in_position >= max_days
```
