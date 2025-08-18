import pandas as pd
import pytest

from engine.loader import load_portfolio_symbols
from engine.market_utils import IndicatorService, SignalScheduler, MarketDataRouter
from engine.base_strategy import OrderSide
from engine.simulator import BacktestSimulator
from tests.conftest import prepare_strategy


def _touched(side: OrderSide, intraday: pd.DataFrame, p: float) -> bool:
    if intraday.empty:
        return False
    if side == OrderSide.BUY:
        return 'low' in intraday.columns and float(intraday['low'].min()) <= float(p)
    return 'high' in intraday.columns and float(intraday['high'].max()) >= float(p)


def test_backtest_lifecycle_50bd_10symbols():
    universe = tuple(load_portfolio_symbols()[:10])
    assert len(universe) == 10, "[Data] Need 10 symbols from portfolio.csv"

    end_d = pd.Timestamp.today().normalize().date()
    start_d = (pd.bdate_range(end=end_d, periods=50)[0]).date()
    start, end = str(start_d), str(end_d)

    ind = IndicatorService()
    vectors = ind.compute_daily_vectors(list(universe), benchmark='^BVSP', start=start, end=end)
    assert vectors, "[Data] No daily vectors computed; check BRAPI token/cache"

    sched = SignalScheduler().build_schedule(vectors)
    assert sched, "[Signals] No BUY/SELL triggers in window; cannot validate lifecycle"

    router = MarketDataRouter()
    frames = []
    for d, syms in sched.items():
        for sym in syms.keys():
            if isinstance(sym, str) and sym.startswith('__'):
                continue
            df, _ = router.get_hourly_for_day(sym, pd.to_datetime(d))
            if df is not None and not df.empty:
                df = df.copy(); df['symbol'] = sym
                frames.append(df)
    combined = pd.concat(frames, axis=0).sort_index() if frames else pd.DataFrame()
    assert not combined.empty, "[Data] No hourly bars for scheduled symbol-days; lifecycle cannot be validated"

    strat = prepare_strategy(universe=universe)
    strat._scheduled_day_trades = sched
    setattr(strat, "_is_first_bar_of_day", lambda s, d: True)
    sim = BacktestSimulator(strategy=strat, start_date=start, end_date=end, config_path="config/settings.yaml")
    _ = sim.run_simulation(combined)
    fills = sim.get_unified_fills_dataframe().copy()
    assert not fills.empty, "[Orders] Expected executed orders; unified_fills is empty. Engine is broken."

    fills['day'] = pd.to_datetime(fills['timestamp']).dt.date
    sym_col = 'symbol' if 'symbol' in fills.columns else 'ticker'

    for d, syms in sched.items():
        for sym, rec in syms.items():
            if isinstance(sym, str) and sym.startswith('__'):
                continue
            f = fills[(fills[sym_col] == sym) & (fills['day'] == pd.to_datetime(d).date())]
            assert not f.empty, f"[Orders] No fills for {sym} on {d}."
            # P1 Market
            assert (f['order_type'] == 'MARKET').any(), f"[Orders] Expected Market@Open for {sym} T+1, not found"
            # MOC
            assert (f.get('attempt_type') == 'moc').any(), f"[Orders] MOC flatten missing for {sym} on {d}"
            # Touch → fill for P2/P3
            side = rec['side']
            p2 = rec['limits_used']['limit_level_2']
            p3 = rec['limits_used']['limit_level_3']
            intr = combined[(combined.index.date == pd.to_datetime(d).date()) & (combined['symbol'] == sym)]
            if _touched(side, intr, p2):
                assert (f.get('attempt_name') == 'Limit Order Passive-1').any(), f"[Orders] P2 touched but not filled for {sym} on {d}"
            if _touched(side, intr, p3):
                assert (f.get('attempt_name') == 'Limit Order Passive-2').any(), f"[Orders] P3 touched but not filled for {sym} on {d}"


