import os, sys
import pandas as pd
from datetime import datetime, date

# Ensure we import the same module path as the strategy (engine.*)
_THIS_DIR = os.path.dirname(__file__)
_ENGINE_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..', 'engine'))
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)

from engine.base_strategy import Bar, StrategyConfig, StrategyContext, OrderType, OrderSide
from quant_b3_backtest.strategies.fuzzy_fajuto_strategy import FuzzyFajutoStrategy


class DummyPortfolio:
    def __init__(self, initial_value: float = 25000.0):
        self._value = initial_value
        self.trade_history = []

    def get_portfolio_value(self) -> float:
        return self._value


class DummyDataPortal:
    def __init__(self):
        self._price = None

    def get_current_price(self, ticker: str):
        return self._price

    def set_current_price(self, ticker: str, price: float):
        self._price = price


def _build_daily_series(dates, highs, lows, closes):
    df = pd.DataFrame({
        'high': highs,
        'low': lows,
        'close': closes,
    }, index=pd.to_datetime(dates))
    return df


def test_slippage_positive_for_limit_buy_below_open():
    symbol = 'TEST3'

    # Daily data for ATR: construct 20 days so ATR(14) is defined
    # Use stable ranges to make ATR predictable
    days = pd.date_range('2025-01-01', periods=20, freq='D')
    highs = [10 + (i % 3) * 0.30 for i in range(20)]
    lows = [9.50 + (i % 3) * 0.30 for i in range(20)]
    closes = [9.80 + (i % 3) * 0.20 for i in range(20)]
    daily_df = _build_daily_series(days, highs, lows, closes)

    # Target trading day D = last day; ATR[D-1] must be used
    current_day = days[-1].date()

    # Create strategy with minimal config and dummy context
    cfg = StrategyConfig(universe=[symbol])
    portfolio = DummyPortfolio(25000.0)
    data_portal = DummyDataPortal()

    class DummyLogger:
        def info(self, *args, **kwargs):
            pass
        def debug(self, *args, **kwargs):
            pass
        def warning(self, *args, **kwargs):
            pass
        def error(self, *args, **kwargs):
            pass

    ctx = StrategyContext(
        data_portal=data_portal,
        portfolio=portfolio,
        broker=None,
        market_rules=None,
        logger=DummyLogger(),
        metadata={'complete_data': None}
    )

    strat = FuzzyFajutoStrategy(cfg, ctx)

    # Bypass external data loads: set ATR[D-1] directly and mark day as already recalculated
    strat.current_atr_values = {symbol: 0.50}
    strat.atr_calculation_dates = {symbol: current_day}
    strat.last_recalculation_dates = {symbol: current_day}

    # Build first hourly bar of day D: set Open[D] = 10.00
    open_price = 10.00
    first_bar_ts = datetime.combine(current_day, datetime.min.time().replace(hour=13))
    bar = Bar(symbol=symbol, timestamp=first_bar_ts, open=open_price, high=open_price, low=open_price, close=open_price, volume=1000)

    # Force a BUY signal by bypassing internals: monkeypatch simple method
    strat._generate_signal = lambda b: 1

    # Emit orders on first bar; capture intents and records
    intents = list(strat.generate_intents(bar))
    # Ensure at least alpha attempt considered
    assert isinstance(strat.daily_order_prices[symbol][current_day]['alpha_price'], float)

    # Compute expected slippage: Open - alpha_limit
    alpha_limit = strat.daily_order_prices[symbol][current_day]['alpha_price']
    expected_slippage = open_price - alpha_limit
    # For BUY, alpha_limit should be below open, so slippage > 0
    assert expected_slippage > 0

    # If first bar did not fill the limit, process another intraday bar that hits it
    hit_ts = datetime.combine(current_day, datetime.min.time().replace(hour=14))
    hit_bar = Bar(symbol=symbol, timestamp=hit_ts, open=open_price, high=open_price, low=alpha_limit, close=open_price, volume=1000)
    delayed_intents = list(strat.generate_intents(hit_bar))

    # Merge intents; at least one limit intent should exist after fill
    all_intents = intents + delayed_intents
    assert any(i.order_type == OrderType.LIMIT for i in all_intents)

    # Ensure execution history has slippage column with expected value (or very close)
    assert strat.execution_history, "execution history should not be empty"
    # Find the alpha attempt entry for the day
    alpha_records = [r for r in strat.execution_history if r.get('attempt_type') in ['limit_alpha', 'limit'] and r.get('symbol') == symbol and r.get('timestamp').date() == current_day]
    assert alpha_records, "no alpha attempt records found"
    # Use the last record (filled one)
    rec = alpha_records[-1]
    assert 'slippage' in rec
    assert rec['slippage'] == expected_slippage

