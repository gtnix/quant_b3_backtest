import numpy as np
import pandas as pd
import pytest
from datetime import date


@pytest.mark.core
@pytest.mark.parametrize(
    "stock_vs_ibov, ema_positions, rsi_val, expected",
    [
        # With five EMA periods (3,5,10,15,20), each contributes ±0.25 when aligned
        # Case 1: stock>ibov (+1), all EMAs below close (+1.25), RSI neutral (0) ⇒ 2.25
        (1, 2, 50.0, 2.25),
        # Case 2: stock<ibov (-1), all EMAs above close (-1.25), RSI neutral (0) ⇒ -2.25
        (-1, -2, 50.0, -2.25),
        # Case 3: stock==ibov (0), all EMAs below close (+1.25), RSI>65 (+0.25) ⇒ 1.50
        (0, 2, 70.0, 1.50),
        # Case 4: stock==ibov (0), all EMAs above close (-1.25), RSI<35 (-0.25) ⇒ -1.50
        (0, -2, 30.0, -1.50),
    ],
)
def test_fuzzy_score_edges(strategy_factory, stock_vs_ibov, ema_positions, rsi_val, expected):
    make = strategy_factory["make"]
    bar = strategy_factory["bar"]
    seed = strategy_factory["seed"]
    strat = make()
    sym = "PETR4"
    t = date(2025, 7, 10)
    dates = pd.bdate_range(end=t, periods=2)
    closes = np.array([10.0, 10.0])
    ibov = closes.copy()
    if stock_vs_ibov > 0:
        closes[-1] = 10.1
        ibov[-1] = 10.0
    elif stock_vs_ibov < 0:
        closes[-1] = 10.0
        ibov[-1] = 10.1
    ema_value = closes[-1]
    if ema_positions > 0:
        ema_value = closes[-1] - 0.01
    elif ema_positions < 0:
        ema_value = closes[-1] + 0.01
    seed(strat, sym, dates, closes, ibov_closes=ibov, ema_value=ema_value, rsi_value=rsi_val)
    strat._generate_signal(bar(sym, t, 20, closes[-1], closes[-1], closes[-1], closes[-1]))
    assert abs(getattr(strat, "_last_signal_strength", 0.0) - expected) < 1e-6

