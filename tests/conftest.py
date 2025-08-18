import datetime as dt
from datetime import date, datetime, timedelta
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from engine.base_strategy import Bar, StrategyConfig, StrategyContext
from strategies.fuzzy_fajuto_strategy import FuzzyFajutoStrategy


def make_bar(symbol: str, d: date, h: int, o: float, hi: float, lo: float, c: float) -> Bar:
    ts = datetime.combine(d, dt.time(hour=h))
    return Bar(symbol=symbol, timestamp=ts, open=o, high=hi, low=lo, close=c, volume=100_000)


def _load_portfolio_symbols() -> list:
    candidates = [
        Path('data') / 'portfolio.csv',
        Path('portfolio.csv'),
    ]
    for p in candidates:
        try:
            if p.exists():
                df = pd.read_csv(p)
                col = 'symbol' if 'symbol' in df.columns else df.columns[0]
                syms = [str(s).strip().upper() for s in df[col].dropna() if str(s).strip()]
                if syms:
                    return list(dict.fromkeys(syms))
        except Exception:
            pass
    return []


def prepare_strategy(universe=None, atr: float = 2.0) -> FuzzyFajutoStrategy:
    if universe is None:
        syms = _load_portfolio_symbols()
        universe = tuple(syms[:1]) if syms else ("PETR4",)
    sc = StrategyConfig(universe=list(universe))
    dummy_portfolio = MagicMock()
    dummy_portfolio.get_portfolio_value.return_value = 1_000_000.0
    # Minimal stateful positions to allow MOC flatten
    positions = {}
    def _buy(sym, qty, price, ts):
        positions[sym] = positions.get(sym, 0) + int(qty)
        return True
    def _sell(sym, qty, price, ts):
        positions[sym] = positions.get(sym, 0) - int(qty)
        return True
    dummy_portfolio.buy.side_effect = _buy
    dummy_portfolio.sell.side_effect = _sell
    dummy_portfolio.positions = positions
    def _update_prices(map_prices, ts):
        return True
    dummy_portfolio.update_prices.side_effect = _update_prices
    dummy_logger = MagicMock()
    ctx = StrategyContext(
        data_portal=None,
        portfolio=dummy_portfolio,
        broker=None,
        market_rules=None,
        logger=dummy_logger,
        metadata={}
    )
    strat = FuzzyFajutoStrategy(sc, ctx)
    strat._universe_symbols = list(universe)
    # ATR removed from strategy; keep no-op field to avoid test coupling
    strat.current_atr_values = {}
    strat.daily_data = {}
    strat.daily_indicators_data = {}
    strat.daily_indicators_last_update = {}
    return strat


def seed_daily(
    strat: FuzzyFajutoStrategy,
    symbol: str,
    dates: pd.DatetimeIndex,
    closes: np.ndarray,
    ibov_closes: np.ndarray = None,
    ema_value: float = None,
    rsi_value: float = None,
) -> None:
    df = pd.DataFrame(index=pd.to_datetime(dates), data={
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": 100_000,
    })
    strat.daily_data[symbol] = df
    if ibov_closes is None:
        ibov_closes = closes.copy()
    strat.daily_data["^BVSP"] = pd.DataFrame(index=pd.to_datetime(dates), data={"close": ibov_closes})
    ind = {}
    if rsi_value is not None:
        ind["rsi"] = pd.Series(index=df.index, data=[rsi_value] * len(df))
    for p in [3, 5, 10, 15, 20]:
        ind[f"ema_{p}"] = pd.Series(index=df.index, data=[ema_value if ema_value is not None else closes[-1]] * len(df))
    strat.daily_indicators_data[symbol] = ind
    strat.daily_indicators_last_update[symbol] = dates[-1].date()


@pytest.fixture
def strategy_factory():
    return {
        "make": prepare_strategy,
        "bar": make_bar,
        "seed": seed_daily,
    }

from pathlib import Path


def pytest_addoption(parser):
    group = parser.getgroup("html_exec_validation")
    group.addoption(
        "--report-html",
        action="store",
        default=str(Path("reports") / "portfolio_execution_report.html"),
        help="Path to the generated backtest HTML report.",
    )
    group.addoption(
        "--report-csv",
        action="store",
        default=str(Path("reports") / "fuzzy_fajuto_execution_history.csv"),
        help="Optional path to execution history CSV fallback.",
    )
    group.addoption(
        "--tolerance",
        action="store",
        type=float,
        default=0.01,
        help="Absolute price tolerance in BRL (default 0.01).",
    )

    # BRAPI discovery/stress options
    disc = parser.getgroup("brapi_discovery")
    disc.addoption(
        "--limit-symbols",
        action="store_true",
        default=False,
        help="Enable BRAPI stress discovery tests (disabled by default)",
    )

