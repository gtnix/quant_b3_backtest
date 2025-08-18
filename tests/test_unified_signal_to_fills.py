import pandas as pd
import pytest

from engine.loader import load_portfolio_symbols
from engine.market_utils import IndicatorService, SignalScheduler, MarketDataRouter
from engine.simulator import BacktestSimulator
from engine.base_strategy import OrderSide
from tests.conftest import prepare_strategy


def _touched(side: OrderSide, intraday: pd.DataFrame, px: float) -> bool:
    if intraday.empty:
        return False
    if side == OrderSide.BUY:
        return 'low' in intraday.columns and float(intraday['low'].min()) <= float(px)
    return 'high' in intraday.columns and float(intraday['high'].max()) >= float(px)


def test_unified_signal_to_fills_50bd_10symbols():
    universe = tuple(load_portfolio_symbols()[:10])
    assert len(universe) == 10, "[Data] Need 10 symbols from portfolio.csv"

    # Determine latest cached hourly day across universe and clamp end date to it
    from engine.loader import DataLoader
    latest_days = []
    for sym in universe:
        dfc = DataLoader._load_best_intraday_cache(sym, 'data/brapi_cache')
        if dfc is None or dfc.empty:
            continue
        idx = pd.to_datetime(dfc.index)
        if getattr(idx, 'tz', None) is not None:
            idx = idx.tz_localize(None)
        latest_days.append(idx.date.max())
    assert latest_days, "[Data] No hourly cache available for any of the 10 symbols; cannot validate lifecycle"
    latest_common = min(latest_days)
    today = pd.Timestamp.today().normalize().date()
    end_d = min(today, latest_common)
    # Use previous business day for signal end so that D+1 has hourly bars within [.., end_d]
    end_sig_d = (pd.bdate_range(end=end_d, periods=2)[0]).date()
    start_d = (pd.bdate_range(end=end_sig_d, periods=50)[0]).date()
    start, end_sig = str(start_d), str(end_sig_d)

    # 1) Signals
    ind = IndicatorService()
    vectors = ind.compute_daily_vectors(list(universe), benchmark='^BVSP', start=start, end=end_sig)
    assert vectors, "[Data] No daily vectors computed; check BRAPI token/cache"

    # 2) Schedule
    sched = SignalScheduler().build_schedule(vectors)
    assert sched, "[Signals] No BUY/SELL triggers in 50-day window; cannot validate lifecycle"

    # 3) Align scheduled days with cached hourly availability
    router = MarketDataRouter()
    sliced_frames = []
    aligned_sched = {}
    for d, syms in sched.items():
        day_ok = False
        for sym, rec in syms.items():
            if isinstance(sym, str) and sym.startswith('__'):
                continue
            df, _ = router.get_hourly_for_day(sym, pd.to_datetime(d))
            if df is not None and not df.empty:
                df = df.copy(); df['symbol'] = sym
                sliced_frames.append(df)
                day_ok = True
        if day_ok:
            aligned_sched[d] = syms
    combined = pd.concat(sliced_frames, axis=0).sort_index() if sliced_frames else pd.DataFrame()
    assert aligned_sched and not combined.empty, "[Data] No hourly bars for any scheduled days; lifecycle cannot be validated"

    # 4) Simulate
    strat = prepare_strategy(universe=universe)
    strat._scheduled_day_trades = aligned_sched
    setattr(strat, "_is_first_bar_of_day", lambda s, d: True)
    sim = BacktestSimulator(strategy=strat, start_date=start, end_date=end, config_path="config/settings.yaml")
    _ = sim.run_simulation(combined)
    fills = sim.get_unified_fills_dataframe().copy()
    assert not fills.empty, "[Orders] No unified fills found — engine broken, stop CI."

    # 5) Validate per scheduled symbol/day
    fills['day'] = pd.to_datetime(fills['timestamp']).dt.date
    sym_col = 'symbol' if 'symbol' in fills.columns else 'ticker'

    for d, syms in aligned_sched.items():
        for sym, rec in syms.items():
            if isinstance(sym, str) and sym.startswith('__'):
                continue
            day = pd.to_datetime(d).date()
            f = fills[(fills[sym_col] == sym) & (fills['day'] == day)]
            assert not f.empty, f"[Orders] Expected fills for {sym} on {day}, but none found."
            # Market@Open mandatory
            assert (f['order_type'] == 'MARKET').any(), f"Expected Market@Open {'BUY' if rec['side']==OrderSide.BUY else 'SELL'} for {sym} on {day}, but not found."
            # MOC mandatory
            assert (f.get('attempt_type') == 'moc').any(), f"MOC flatten missing for {sym} on {day} — position leak."
            # P2/P3 touch→fill
            intr = combined[(combined.index.date == day) & (combined['symbol'] == sym)]
            side = rec['side']
            p2 = float(rec['limits_used']['limit_level_2']); p3 = float(rec['limits_used']['limit_level_3'])
            if _touched(side, intr, p2):
                assert (f.get('attempt_name') == 'Limit Order Passive-1').any(), f"P2 touched {p2:.2f} on intraday bars, expected fill missing for {sym} on {day}."
            if _touched(side, intr, p3):
                assert (f.get('attempt_name') == 'Limit Order Passive-2').any(), f"P3 touched {p3:.2f} on intraday bars, expected fill missing for {sym} on {day}."
            # P4 warn-only


