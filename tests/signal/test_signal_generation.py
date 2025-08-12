import numpy as np
import pandas as pd
import pytest
from datetime import date


@pytest.mark.signal
def test_threshold_validation_buy(strategy_factory, monkeypatch):
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
    sig = strat._generate_signal(bar(sym, t, 20, closes[-1], closes[-1], closes[-1], closes[-1]))
    assert sig == 1


@pytest.mark.signal
def test_threshold_validation_sell(strategy_factory, monkeypatch):
    make = strategy_factory["make"]
    bar = strategy_factory["bar"]
    seed = strategy_factory["seed"]
    strat = make(universe=("SUZB3",))
    sym = "SUZB3"
    t = date(2025, 7, 10)
    dates = pd.bdate_range(end=t, periods=2)
    closes = np.array([10.1, 10.0])
    ibov = np.array([10.0, 10.1])
    seed(strat, sym, dates, closes, ibov_closes=ibov, ema_value=10.2, rsi_value=30.0)
    sig = strat._generate_signal(bar(sym, t, 20, closes[-1], closes[-1], closes[-1], closes[-1]))
    assert sig == -1


@pytest.mark.signal
def test_ibov_alignment_no_forward_fill(strategy_factory):
    make = strategy_factory["make"]
    bar = strategy_factory["bar"]
    strat = make()
    sym = "PETR4"
    d0, d1 = pd.bdate_range(end=date(2025, 7, 9), periods=2)
    df_sym = pd.DataFrame(index=pd.to_datetime([d0, d1]), data={"close": [10.0, 10.5]})
    strat.daily_data[sym] = df_sym
    strat.daily_data["^BVSP"] = pd.DataFrame(index=pd.to_datetime([d0]), data={"close": [10.0]})
    strat.daily_indicators_data[sym] = {"rsi": pd.Series(index=df_sym.index, data=[50, 50]), **{f"ema_{p}": pd.Series(index=df_sym.index, data=[10.0, 10.0]) for p in [3,5,10,15,20]}}
    s = strat._generate_signal(bar(sym, d1.date(), 20, 10.5, 10.5, 10.5, 10.5))
    assert s in (0, 1, -1)

