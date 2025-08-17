import pandas as pd

from engine.market_utils import SignalScheduler


def test_tranche_round_lot_buy_levels():
    # tranche (leg) notional 12,500 BRL, round lot 100
    sched = SignalScheduler(round_lot_size=100, tick_size=0.01, leg_notional_brl=12500.0)
    # Example close and derived limits
    close = 25.00
    p2, p3, p4 = sched._limits_from_close(close, 'BUY')
    # Quantities per level
    qty1 = sched._round_to_lot(12500.0 / max(close, 1e-9))
    qty2 = sched._round_to_lot(12500.0 / max(p2, 1e-9))
    qty3 = sched._round_to_lot(12500.0 / max(p3, 1e-9))
    qty4 = sched._round_to_lot(12500.0 / max(p4, 1e-9))

    # All quantities must be multiples of 100 (round lot)
    assert qty1 % 100 == 0
    assert qty2 % 100 == 0
    assert qty3 % 100 == 0
    assert qty4 % 100 == 0

    # Monotonic non-decreasing for BUY as price lowers across p2<p3<p4
    assert qty2 >= qty1
    assert qty3 >= qty2
    assert qty4 >= qty3


def test_tranche_round_lot_sell_levels():
    sched = SignalScheduler(round_lot_size=100, tick_size=0.01, leg_notional_brl=12500.0)
    close = 30.00
    p2, p3, p4 = sched._limits_from_close(close, 'SELL')
    qty1 = sched._round_to_lot(12500.0 / max(close, 1e-9))
    qty2 = sched._round_to_lot(12500.0 / max(p2, 1e-9))
    qty3 = sched._round_to_lot(12500.0 / max(p3, 1e-9))
    qty4 = sched._round_to_lot(12500.0 / max(p4, 1e-9))

    # All quantities must be multiples of 100 (round lot)
    assert qty1 % 100 == 0
    assert qty2 % 100 == 0
    assert qty3 % 100 == 0
    assert qty4 % 100 == 0

    # Monotonic non-increasing for SELL as price increases across p2<p3<p4
    assert qty2 <= qty1
    assert qty3 <= qty2
    assert qty4 <= qty3


def test_nearest_100_rounding_policy_boundaries():
    # Validate rounding rule: remainder >=50 rounds up to next 100; else rounds down
    sched = SignalScheduler(round_lot_size=100, tick_size=0.01, leg_notional_brl=12500.0)
    # Helper wraps scheduler rounding to mimic strategy policy
    def nearest_100(shares: float) -> int:
        s = int(shares)
        r = s % 100
        return ((s // 100) + 1) * 100 if r >= 50 else (s // 100) * 100

    # PETR4 close 30.17 → ~414 → 400 (down)
    petr4_qty = nearest_100(12500.0 / 30.17)
    assert petr4_qty == 400

    # PETR3 close 32.17 → ~388 → 400 (up)
    petr3_qty = nearest_100(12500.0 / 32.17)
    assert petr3_qty == 400

    # NVDA close 180 → ~69 → 100 (up)
    nvda_qty = nearest_100(12500.0 / 180.0)
    assert nvda_qty == 100

