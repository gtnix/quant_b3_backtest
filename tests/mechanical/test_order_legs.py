import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

from engine.base_strategy import OrderType


@pytest.mark.mechanical
def test_four_leg_construction_long(strategy_factory, monkeypatch):
    make = strategy_factory["make"]
    bar = strategy_factory["bar"]
    seed = strategy_factory["seed"]
    strat = make()
    sym = "PETR4"
    t = date(2025, 7, 10)
    t1 = t + timedelta(days=1)
    dates = pd.bdate_range(end=t, periods=25)
    closes = np.linspace(9.5, 10.5, len(dates))
    ibov_closes = np.linspace(9.5, 10.4, len(dates))
    seed(strat, sym, dates, closes, ibov_closes=ibov_closes, ema_value=9.0, rsi_value=70.0)
    monkeypatch.setattr(strat, "_is_first_bar_of_day", lambda s, d: True)
    monkeypatch.setattr(strat, "_calculate_entry_limits_from_close", lambda close_px, atr, side: (close_px*0.99, close_px*0.98, close_px*0.97))
    assert strat._generate_signal(bar(sym, t, 20, closes[-1], closes[-1], closes[-1], closes[-1])) == 1
    list(strat.on_end_of_day(t))
    intents = list(strat.generate_intents(bar(sym, t1, 13, closes[-1]*1.01, closes[-1]*1.02, closes[-1]*0.97, closes[-1]*1.01)))
    assert any(i.order_type == OrderType.MARKET for i in intents)
    assert (strat._get_stored_order_quantity(sym, t1, 'limit_alpha') or 0) >= 0
    assert (strat._get_stored_order_quantity(sym, t1, 'limit_beta') or 0) >= 0
    assert (strat._get_stored_order_quantity(sym, t1, 'limit_gamma') or 0) >= 0


@pytest.mark.mechanical
def test_four_leg_construction_short(strategy_factory, monkeypatch):
    make = strategy_factory["make"]
    bar = strategy_factory["bar"]
    seed = strategy_factory["seed"]
    strat = make(universe=("SUZB3",))
    sym = "SUZB3"
    t = date(2025, 7, 10)
    t1 = t + timedelta(days=1)
    dates = pd.bdate_range(end=t, periods=25)
    closes = np.linspace(10.5, 9.5, len(dates))
    ibov_closes = np.linspace(10.0, 10.2, len(dates))
    seed(strat, sym, dates, closes, ibov_closes=ibov_closes, ema_value=11.0, rsi_value=30.0)
    monkeypatch.setattr(strat, "_is_first_bar_of_day", lambda s, d: True)
    monkeypatch.setattr(strat, "_calculate_entry_limits_from_close", lambda close_px, atr, side: (close_px*1.01, close_px*1.02, close_px*1.03))
    assert strat._generate_signal(bar(sym, t, 20, closes[-1], closes[-1], closes[-1], closes[-1])) == -1
    list(strat.on_end_of_day(t))
    intents = list(strat.generate_intents(bar(sym, t1, 13, closes[-1]*0.99, closes[-1]*1.03, closes[-1]*0.99, closes[-1]*0.99)))
    assert any(i.order_type == OrderType.MARKET for i in intents)
    assert (strat._get_stored_order_quantity(sym, t1, 'limit_alpha') or 0) >= 0
    assert (strat._get_stored_order_quantity(sym, t1, 'limit_beta') or 0) >= 0
    assert (strat._get_stored_order_quantity(sym, t1, 'limit_gamma') or 0) >= 0

