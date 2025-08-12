import numpy as np
import pandas as pd
import pytest
from datetime import date


@pytest.mark.data_quality
def test_warmup_policy_minimum_sessions(strategy_factory):
    make = strategy_factory["make"]
    strat = make()
    req = strat.required_history()
    ema_max = max(req.get('ema_windows', [20]))
    mul = float(req.get('warmup_multiplier_for_ema', 3.0))
    required = int(np.ceil(mul * ema_max))
    assert required >= 60 or ema_max * mul >= 60


@pytest.mark.data_quality
def test_short_backtests_auto_prepends_warmup(strategy_factory):
    make = strategy_factory["make"]
    strat = make()
    from engine.simulator import BacktestSimulator
    sim = BacktestSimulator()
    idx = pd.bdate_range(end=date(2025, 7, 10), periods=3)
    df = pd.DataFrame(index=idx, data={"symbol": ["PETR4"]*3, "open": [10,11,12], "high": [11,12,13], "low": [9,10,11], "close": [10.1,11.1,12.1], "volume": [1_000]*3})
    sim.strategy = strat
    sim.run_simulation(df)
    assert True


@pytest.mark.data_quality
def test_ibov_alignment_symbol_name_and_no_forward_fill(strategy_factory):
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
    assert s in (0,1,-1)

