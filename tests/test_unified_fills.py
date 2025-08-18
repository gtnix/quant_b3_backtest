import pandas as pd
from datetime import date, datetime, timedelta

from engine.base_strategy import OrderSide
from engine.simulator import BacktestSimulator
from tests.conftest import prepare_strategy


def _make_day(symbol: str, day: date, open_px: float, highs: list[float], lows: list[float], closes: list[float]) -> pd.DataFrame:
    hours = [10, 11, 12, 16, 17, 18]  # include a late bar for MOC logic
    idx = [datetime(day.year, day.month, day.day, h, 0, 0) for h in hours[:len(highs)]]
    df = pd.DataFrame(
        {
            'symbol': [symbol] * len(idx),
            'open': [open_px] + [closes[i-1] for i in range(1, len(closes))],
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': [100000] * len(idx),
        },
        index=pd.DatetimeIndex(idx),
    )
    return df


def _run_with_schedule(symbol: str, d: date, side: OrderSide, base_close: float) -> pd.DataFrame:
    # Compute expected limits per README
    if side == OrderSide.BUY:
        p2 = round(base_close * 0.995, 2)
        p3 = round(base_close * 0.990, 2)
        p4 = round(base_close * 0.985, 2)
        highs = [base_close * 1.01, base_close * 1.01, base_close * 1.02, base_close * 1.03]
        lows = [base_close * 0.997, p2 - 0.01, p3 - 0.01, base_close * 0.98]
    else:
        p2 = round(base_close * 1.005, 2)
        p3 = round(base_close * 1.010, 2)
        p4 = round(base_close * 1.015, 2)
        highs = [base_close * 1.007, p2 + 0.01, p3 + 0.02, base_close * 1.03]
        lows = [base_close * 0.99, base_close * 0.99, base_close * 1.00, base_close * 1.01]

    closes = [base_close * 1.01, base_close * 1.00, base_close * 1.005, base_close * 1.002]
    df = _make_day(symbol, d, base_close * 1.01, highs, lows, closes)
    # Add one next-day bar to ensure end-of-day processing (MOC) is triggered
    next_bar = pd.DataFrame(
        {
            'symbol': [symbol],
            'open': [closes[-1]],
            'high': [closes[-1]],
            'low': [closes[-1]],
            'close': [closes[-1]],
            'volume': [100000],
        },
        index=pd.DatetimeIndex([datetime(d.year, d.month, d.day, 23, 59, 0) + timedelta(minutes=1)]),
    )
    df = pd.concat([df, next_bar]).sort_index()

    # Inject schedule for this D (execution day)
    sched = {
        d: {
            symbol: {
                'symbol': symbol,
                'side': side,
                'valid_for_date': d,
                'base_close_t': float(base_close),
                'limits_used': {'limit_level_2': float(p2), 'limit_level_3': float(p3), 'limit_level_4': float(p4)},
                'current_atr_t': 0.0,
                'fuzzy_score_t': 2.0,
            }
        }
    }

    strat = prepare_strategy(universe=(symbol,))
    strat._scheduled_day_trades = sched
    setattr(strat, "_is_first_bar_of_day", lambda s, d0: True)
    # Force empty intents so schedule fallback emits fills deterministically
    setattr(strat, 'handle_bar', lambda bar: [])

    sim = BacktestSimulator(strategy=strat, initial_capital=2_000_000.0, start_date=str(d), end_date=str(d + timedelta(days=1)), config_path="config/settings.yaml")
    sim.run_simulation(df)
    fills = sim.get_unified_fills_dataframe().copy()
    return fills, p2, p3


def test_unified_fills_buy_day_synthetic_round_lot_and_limits():
    symbol = "PETR4"; d = date(2025, 7, 15); base_close = 10.00
    fills, p2, p3 = _run_with_schedule(symbol, d, OrderSide.BUY, base_close)
    assert not fills.empty, "[Orders] Expected unified fills; none found"
    fday = fills.copy(); fday['day'] = pd.to_datetime(fday['timestamp']).dt.date
    fday = fday[(fday['day'] == d) & (fday['symbol'] == symbol)]
    assert (fday['order_type'] == 'MARKET').any(), "Expected P1 Market@Open fill"
    assert (fday.get('attempt_type') == 'moc').any(), "Expected MOC flatten"
    # Touch → fill for P2/P3
    assert (fday.get('attempt_name') == 'Limit Order Passive-1').any(), "P2 touched; expected fill missing"
    assert (fday.get('attempt_name') == 'Limit Order Passive-2').any(), "P3 touched; expected fill missing"
    # Board-lot enforcement
    assert (fday['quantity'] % 100 == 0).all(), "Quantities must be multiples of 100"
    # Price checks
    if 'price' in fday.columns:
        p2_filled = fday[(fday['attempt_name'] == 'Limit Order Passive-1')]['price']
        p3_filled = fday[(fday['attempt_name'] == 'Limit Order Passive-2')]['price']
        if not p2_filled.empty:
            assert abs(float(p2_filled.iloc[0]) - float(p2)) <= 0.01, "P2 price off tick"
        if not p3_filled.empty:
            assert abs(float(p3_filled.iloc[0]) - float(p3)) <= 0.01, "P3 price off tick"


def test_unified_fills_sell_day_synthetic_round_lot_and_limits():
    symbol = "VALE3"; d = date(2025, 7, 16); base_close = 20.00
    fills, p2, p3 = _run_with_schedule(symbol, d, OrderSide.SELL, base_close)
    assert not fills.empty, "[Orders] Expected unified fills; none found"
    fday = fills.copy(); fday['day'] = pd.to_datetime(fday['timestamp']).dt.date
    fday = fday[(fday['day'] == d) & (fday['symbol'] == symbol)]
    assert (fday['order_type'] == 'MARKET').any(), "Expected P1 Market@Open fill"
    assert (fday.get('attempt_type') == 'moc').any(), "Expected MOC flatten"
    # Touch → fill for P2/P3
    assert (fday.get('attempt_name') == 'Limit Order Passive-1').any(), "P2 touched; expected fill missing"
    assert (fday.get('attempt_name') == 'Limit Order Passive-2').any(), "P3 touched; expected fill missing"
    # Board-lot enforcement
    assert (fday['quantity'] % 100 == 0).all(), "Quantities must be multiples of 100"
    # Price checks
    if 'price' in fday.columns:
        p2_filled = fday[(fday['attempt_name'] == 'Limit Order Passive-1')]['price']
        p3_filled = fday[(fday['attempt_name'] == 'Limit Order Passive-2')]['price']
        if not p2_filled.empty:
            assert abs(float(p2_filled.iloc[0]) - float(p2)) <= 0.01, "P2 price off tick"
        if not p3_filled.empty:
            assert abs(float(p3_filled.iloc[0]) - float(p3)) <= 0.01, "P3 price off tick"


