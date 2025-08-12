import numpy as np
import pandas as pd
import pytest
from datetime import date


@pytest.mark.smoke
def test_smoke_score_and_constraints(strategy_factory):
    make = strategy_factory["make"]
    bar = strategy_factory["bar"]
    seed = strategy_factory["seed"]
    strat = make()
    sym = "PETR4"
    t = date(2025, 7, 10)
    dates = pd.bdate_range(end=t, periods=2)
    closes = np.array([10.0, 10.1])
    ibov = np.array([10.0, 10.0])
    seed(strat, sym, dates, closes, ibov_closes=ibov, ema_value=9.9, rsi_value=70.0)
    s = strat._generate_signal(bar(sym, t, 20, closes[-1], closes[-1], closes[-1], closes[-1]))
    assert s in (0, 1, -1)

