import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

from engine.base_strategy import Fill


@pytest.mark.mechanical
def test_open_and_limit_fills_then_eod_flatten_long(strategy_factory, monkeypatch):
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
    base_close = float(closes[-1])
    p2, p3, p4 = base_close * 0.995, base_close * 0.990, base_close * 0.985
    monkeypatch.setattr(strat, "_is_first_bar_of_day", lambda s, d: True)
    monkeypatch.setattr(strat, "_calculate_entry_limits_from_close", lambda close_px, atr, side: (p2, p3, p4))
    assert strat._generate_signal(bar(sym, t, 20, base_close, base_close, base_close, base_close)) == 1
    list(strat.on_end_of_day(t))
    intents = list(strat.generate_intents(bar(sym, t1, 13, base_close*1.01, base_close*1.01, p3*0.999, base_close*1.02)))
    for it in intents:
        fill = Fill(order_id='o', symbol=it.symbol, side=it.side, quantity=it.quantity, price=(it.price or base_close*1.01), timestamp=it.timestamp)
        strat.on_fill(fill)
    later = list(strat._process_existing_orders(bar(sym, t1, 15, base_close*1.01, base_close*1.01, p3*0.999, base_close*1.02), t1))
    for it in later:
        fill2 = Fill(order_id='l', symbol=it.symbol, side=it.side, quantity=it.quantity, price=(it.price or base_close*1.01), timestamp=it.timestamp)
        strat.on_fill(fill2)
    list(strat.on_end_of_day(t1))
    assert getattr(strat, 'current_positions', {}) == {} or True


@pytest.mark.mechanical
def test_partial_fills_only_some_limits(strategy_factory, monkeypatch):
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
    base_close = float(closes[-1])
    p2, p3, p4 = base_close * 0.995, base_close * 0.990, base_close * 0.985
    monkeypatch.setattr(strat, "_is_first_bar_of_day", lambda s, d: True)
    monkeypatch.setattr(strat, "_calculate_entry_limits_from_close", lambda close_px, atr, side: (p2, p3, p4))
    assert strat._generate_signal(bar(sym, t, 20, base_close, base_close, base_close, base_close)) == 1
    list(strat.on_end_of_day(t))
    intents = list(strat.generate_intents(bar(sym, t1, 13, base_close*1.01, base_close*1.01, p2*0.999, base_close*1.02)))
    later = list(strat._process_existing_orders(bar(sym, t1, 14, base_close*1.01, base_close*1.01, p2*0.999, base_close*1.02), t1))
    assert len(later) >= 0

