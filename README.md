## Introduction

This repository implements a professional, rules-based backtesting engine for the Brazilian market (B3) and documents the FuzzyFajuto strategy. The strategy is fully specified and instrumented for institutional auditability: indicator definitions, fuzzy scoring, position sizing under B3 board-lot rules, order placement (market + passive limits), bidirectional Buy↔Sell pairing, and daily end-of-day closing (MOC).

The ticker universe is sourced from `data/portfolio.csv` (first column = ticker). When you run without `--tickers`, the engine automatically reads this file as the single source of truth.

## Strategy Logic (FuzzyFajuto)

The strategy computes a daily FuzzyFajuto score per symbol using daily indicators and then applies fixed execution rules on the next trading day using intraday bars. Positions are always sized in B3 board-lot multiples (100 shares).

### Indicators and Scoring
- Daily Return vs Ibov daily return: +1 if outperforming, −1 if underperforming.
- Close vs EMAs(3, 5, 10, 15, 20): +0.25 if close > EMA(period), −0.25 if close < EMA(period) (applied per period).
- RSI: RSI > 65 → +0.25; RSI < 35 → −0.25.

The total FuzzyFajuto score is the sum of the above contributions. It is a continuous scalar where larger positive values favor long exposure and large negative values favor sells exposure.

## Position Sizing (B3 Board-Lot)

- Fixed total notional per symbol per session: 50,000 BRL.
- This is split into 4 equal tranches of 12,500 BRL.
- Shares per tranche are computed using the last daily close (close[T−1]):
  
  shares_raw = 12,500 ÷ close[T−1]
  
- Board-lot rounding (multiples of 100 shares):
  - If the last two digits of shares_raw are 00–49 → round down to nearest 100.
  - If the last two digits are 50–99 → round up to nearest 100.

Examples (≈ 50k total across 4 tranches; per-tranche quantities ×4):
- PETR4 close 30.17 → 12,500 / 30.17 ≈ 414 → 400 ×4.
- PETR3 close 32.60 → 12,500 / 32.60 ≈ 383 → 400 ×4.
- ITUB4 close 37.49 → ≈ 333 → 300 ×4.
- NVDA close 180.65 → ≈ 69 → 100 ×4.
- VALE3 close 53.32 → ≈ 234 → 200 ×4.
- GGBR4 close 16.23 → ≈ 770 → 800 ×4.

The same rounding policy is used consistently whenever quantities are calculated from notional ÷ price.

## Trading Rules (Entry & Exit)

- If FuzzyFajuto ≥ +1.50 → Generate Buy orders for tomorrow.
- If FuzzyFajuto ≤ −1.50 → Generate Sell orders for tomorrow.

For both Buy and Sell signals, four orders per side are emitted:
1. Market at open (first intraday bar of the day).
2. Passive limit at close[T−1] × (1 − 0.005) for Buy; (1 + 0.005) for Sell.
3. Passive limit at close[T−1] × (1 − 0.010) for Buy; (1 + 0.010) for Sell.
4. Passive limit at close[T−1] × (1 − 0.015) for Buy; (1 + 0.015) for Sell.

All open positions are closed at the auction call (end-of-day) via Market-on-Close (MOC) orders.

## Pair Matching Engine (Bidirectional)

The system enforces bidirectional pairing when `RISK_PAIR_MATCHING=True`:
- For each Sell: pair it with the Buy having the highest available FuzzyFajuto score.
- For each Buy: pair it with the Sell having the highest available FuzzyFajuto score.
- If counts differ: leftover Buys or Sells remain unpaired.
- Tie-breaking: sort by descending fuzzy score, then apply a deterministic secondary key (symbol lexical order).

Pairing is applied before order emission, and all four attempts (market, limit_alpha, limit_beta, limit_gamma) are sized and emitted per leg, with board-lot rounding.

## Order Placement Structure

- For each valid paired leg on a trading day:
  - P1: Market at open (always filled).
  - P2: Limit (alpha) at ±0.5% from close[T−1].
  - P3: Limit (beta) at ±1.0% from close[T−1].
  - P4: Limit (gamma) at ±1.5% from close[T−1].
- All quantities per attempt type adhere to the board-lot rounding described above.
- MOC: Positions are closed at the end of the session with Market-on-Close orders.

## Testing & Validation

### Unit Tests
- `tests/test_tranche_sizing.py`: validates round-lot enforcement and level monotonicity.
- `tests/test_pairing_logic.py`: bidirectional pairing scenarios and deterministic tie-breaking:
  - More Buys than Sells → leftover Buys.
  - More Sells than Buys → leftover Sells.
  - Equal counts → all paired.
  - Tie-breaking: equal fuzzy scores → deterministic symbol-based ordering.

### Quick Backtest Validation (10–15 trading days)
- Run with multi-frame processing and results enabled:
```
AUDIT_EXECUTIONS_ONLY=0 MULTIFRAME_MODE=1 \
python3 run_fuzzy_fajuto.py \
  --start-date 2025-07-15 --end-date 2025-08-01 \
  --save-results
```

Validate outputs:
- `results/unified_fills.csv` / `.json`: both legs must show P1 market, P2–P4 limits, and end-of-day MOC.
- `reports/portfolio_fuzzy_indicators.csv`: columns `notional_P1..P4` filled for paired symbols/dates.
- KPIs: `total_trades > 0` (round-trips recognized) and metrics populated.

## Examples

Sizing examples reproduced (per-tranche results ×4):
- PETR4 @ 30.17 → 12,500 / 30.17 ≈ 414 → 400 ×4 (≈ 50k total).
- PETR3 @ 32.60 → ≈ 383 → 400 ×4.
- ITUB4 @ 37.49 → ≈ 333 → 300 ×4.
- NVDA @ 180.65 → ≈ 69 → 100 ×4.
- VALE3 @ 53.32 → ≈ 234 → 200 ×4.
- GGBR4 @ 16.23 → ≈ 770 → 800 ×4.

Entry/Exit examples (BUY):
- Open: market at first bar.
- Limits: close[T−1] × (0.995, 0.990, 0.985).
- MOC: flatten at auction call.

Entry/Exit examples (SELL):
- Open: market at first bar.
- Limits: close[T−1] × (1.005, 1.010, 1.015).
- MOC: flatten at auction call.

## References

- B3 conventions: board-lot = 100 shares (no odd-lots in main book), tick size = 0.01 BRL.
- Strategy pairing flag: `RISK_PAIR_MATCHING=True` (bidirectional highest-fuzzy matching).
- Data & reports:
  - Universe: `data/portfolio.csv`.
  - Reports: `reports/portfolio_fuzzy_indicators.csv`, `results/unified_fills.*`.

