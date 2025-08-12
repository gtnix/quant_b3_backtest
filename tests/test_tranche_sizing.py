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

