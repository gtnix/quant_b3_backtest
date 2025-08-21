"""
MOC (Market-on-Close) Flattening Validation Tests
=================================================

Tests to validate that all open positions are properly flattened at end-of-day
according to the strategy specification: "All open positions are flattened at 
the auction call (end-of-day) via Market-on-Close (MOC) orders."

This test enforces the critical invariant: NO OVERNIGHT POSITIONS ALLOWED.
"""

import pandas as pd
import pytest
from datetime import datetime, date, timedelta
from typing import Dict, Set, List, Tuple
from collections import defaultdict

from engine.loader import load_portfolio_symbols
from engine.market_utils import IndicatorService, SignalScheduler, MarketDataRouter
from engine.simulator import BacktestSimulator
from engine.base_strategy import OrderSide
from tests.conftest import prepare_strategy


class PositionTracker:
    """Track positions throughout the day to validate MOC flattening."""
    
    def __init__(self):
        self.positions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))  # date -> symbol -> position
        self.trading_days: Set[str] = set()
        self.symbols_with_activity: Dict[str, Set[str]] = defaultdict(set)  # date -> symbols
    
    def process_fill(self, timestamp: pd.Timestamp, symbol: str, side: str, quantity: int, order_type: str):
        """Process a fill and update position tracking."""
        trading_date = timestamp.date().isoformat()
        self.trading_days.add(trading_date)
        
        # Track symbols with activity on this date
        self.symbols_with_activity[trading_date].add(symbol)
        
        # Update position
        if side.upper() == 'BUY':
            self.positions[trading_date][symbol] += quantity
        elif side.upper() == 'SELL':
            self.positions[trading_date][symbol] -= quantity
    
    def get_eod_positions(self, trading_date: str) -> Dict[str, int]:
        """Get end-of-day positions for a specific date."""
        return dict(self.positions[trading_date])
    
    def get_overnight_positions(self) -> Dict[str, Dict[str, int]]:
        """Get all non-zero end-of-day positions (violations)."""
        violations = {}
        for trading_date in self.trading_days:
            eod_positions = self.get_eod_positions(trading_date)
            non_zero = {symbol: pos for symbol, pos in eod_positions.items() if pos != 0}
            if non_zero:
                violations[trading_date] = non_zero
        return violations
    
    def validate_moc_flattening(self) -> Tuple[bool, List[str]]:
        """Validate that all positions are flattened at EOD. Returns (is_valid, error_messages)."""
        overnight_positions = self.get_overnight_positions()
        
        if not overnight_positions:
            return True, []
        
        error_messages = []
        for trading_date, positions in overnight_positions.items():
            for symbol, position in positions.items():
                error_messages.append(
                    f"OVERNIGHT POSITION VIOLATION: {symbol} on {trading_date} has position {position:+,} (expected: 0)"
                )
        
        return False, error_messages


def test_moc_flattening_3month_validation():
    """
    CRITICAL TEST: Validate MOC flattening over 3 months of hourly data.
    
    This test enforces the strategy specification:
    "All open positions are flattened at the auction call (end-of-day) via Market-on-Close (MOC) orders."
    
    FAILURE CONDITIONS:
    - Any symbol has non-zero position at end of any trading day
    - MOC orders are missing for symbols with intraday positions
    - Position tracking shows overnight exposure
    """
    # Use first 10 symbols for focused validation
    universe = tuple(load_portfolio_symbols()[:10])
    assert len(universe) == 10, "[Data] Need 10 symbols from portfolio.csv for MOC validation"
    
    # Determine 3-month test window
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
    
    assert latest_days, "[Data] No hourly cache available for MOC validation"
    latest_common = min(latest_days)
    
    # 3-month window (approximately 65 business days)
    end_d = latest_common
    start_d = (pd.bdate_range(end=end_d, periods=65)[0]).date()
    start, end = str(start_d), str(end_d)
    
    print(f"[MOC-TEST] Validating MOC flattening from {start} to {end} ({universe})")
    
    # 1) Generate signals for the period
    ind = IndicatorService()
    vectors = ind.compute_daily_vectors(list(universe), benchmark='^BVSP', start=start, end=end)
    assert vectors, "[Data] No daily vectors computed for MOC validation"
    
    # 2) Build schedule
    sched = SignalScheduler().build_schedule(vectors)
    assert sched, "[Signals] No BUY/SELL triggers in 3-month window for MOC validation"
    
    # 3) Align with hourly data availability
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
                df = df.copy()
                df['symbol'] = sym
                sliced_frames.append(df)
                day_ok = True
        if day_ok:
            aligned_sched[d] = syms
    
    combined = pd.concat(sliced_frames, axis=0).sort_index() if sliced_frames else pd.DataFrame()
    assert aligned_sched and not combined.empty, "[Data] No hourly bars for MOC validation"
    
    print(f"[MOC-TEST] Testing {len(aligned_sched)} trading days with hourly data")
    
    # 4) Run simulation with position tracking
    strat = prepare_strategy(universe=universe)
    strat._scheduled_day_trades = aligned_sched
    setattr(strat, "_is_first_bar_of_day", lambda s, d: True)
    
    sim = BacktestSimulator(
        strategy=strat, 
        start_date=start, 
        end_date=end, 
        config_path="config/settings.yaml"
    )
    
    _ = sim.run_simulation(combined)
    fills_df = sim.get_unified_fills_dataframe().copy()
    assert not fills_df.empty, "[Orders] No unified fills found for MOC validation"
    
    # 5) Process fills and track positions
    position_tracker = PositionTracker()
    
    fills_df['day'] = pd.to_datetime(fills_df['timestamp']).dt.date
    sym_col = 'symbol' if 'symbol' in fills_df.columns else 'ticker'
    
    # Process each fill chronologically
    fills_sorted = fills_df.sort_values('timestamp')
    for _, fill in fills_sorted.iterrows():
        position_tracker.process_fill(
            timestamp=pd.to_datetime(fill['timestamp']),
            symbol=fill[sym_col],
            side=fill['side'],
            quantity=fill['quantity'],
            order_type=fill.get('order_type', 'UNKNOWN')
        )
    
    # 6) Validate MOC flattening invariant
    is_valid, error_messages = position_tracker.validate_moc_flattening()
    
    # 7) Additional validations
    moc_fills = fills_df[
        (fills_df.get('order_type') == 'MOC') | 
        (fills_df.get('attempt_type') == 'moc')
    ]
    
    trading_days_with_activity = len(position_tracker.trading_days)
    symbols_with_moc = len(moc_fills[sym_col].unique()) if not moc_fills.empty else 0
    
    print(f"[MOC-TEST] Trading days with activity: {trading_days_with_activity}")
    print(f"[MOC-TEST] Total fills: {len(fills_df):,}")
    print(f"[MOC-TEST] MOC fills: {len(moc_fills):,}")
    print(f"[MOC-TEST] Symbols with MOC: {symbols_with_moc}")
    
    # 8) Critical assertions - FAIL if ANY overnight positions exist
    if not is_valid:
        error_summary = "\n".join(error_messages[:10])  # Show first 10 violations
        if len(error_messages) > 10:
            error_summary += f"\n... and {len(error_messages) - 10} more violations"
        
        pytest.fail(
            f"MOC FLATTENING FAILURE: {len(error_messages)} overnight position violations detected!\n\n"
            f"STRATEGY SPECIFICATION VIOLATED: 'All open positions are flattened at the auction call (end-of-day) via Market-on-Close (MOC) orders.'\n\n"
            f"VIOLATIONS:\n{error_summary}\n\n"
            f"This indicates MOC orders are not properly closing all positions at end-of-day."
        )
    
    # 9) Validate MOC order presence for active symbols
    overnight_positions = position_tracker.get_overnight_positions()
    assert len(overnight_positions) == 0, f"Found {len(overnight_positions)} days with overnight positions"
    
    # 10) Validate that MOC fills exist for days with activity
    if trading_days_with_activity > 0:
        assert len(moc_fills) > 0, "Expected MOC fills for days with trading activity"
        assert symbols_with_moc > 0, "Expected MOC fills for symbols with positions"
    
    print(f"✅ [MOC-TEST] PASSED: All positions properly flattened across {trading_days_with_activity} trading days")


def test_moc_flattening_single_day_detailed():
    """
    Detailed single-day MOC validation test.
    
    Tests the position lifecycle within a single trading day:
    1. Positions can be opened during the day
    2. Positions must be closed by MOC at end-of-day
    3. Net position must be zero at end-of-day
    """
    universe = tuple(load_portfolio_symbols()[:3])  # Use 3 symbols for focused test
    
    # Get a single recent trading day with data
    from engine.loader import DataLoader
    test_symbol = universe[0]
    dfc = DataLoader._load_best_intraday_cache(test_symbol, 'data/brapi_cache')
    
    if dfc is None or dfc.empty:
        pytest.skip(f"No hourly data available for {test_symbol}")
    
    idx = pd.to_datetime(dfc.index)
    if getattr(idx, 'tz', None) is not None:
        idx = idx.tz_localize(None)
    
    # Use the most recent complete trading day
    latest_date = idx.date.max()
    test_date_str = str(latest_date)
    
    print(f"[MOC-DETAIL] Testing detailed MOC behavior for {test_date_str}")
    
    # Generate signals for this specific day
    ind = IndicatorService()
    vectors = ind.compute_daily_vectors(list(universe), benchmark='^BVSP', start=test_date_str, end=test_date_str)
    
    if not vectors:
        pytest.skip(f"No signals generated for {test_date_str}")
    
    sched = SignalScheduler().build_schedule(vectors)
    if not sched:
        pytest.skip(f"No schedule built for {test_date_str}")
    
    # Get hourly data for the test day
    router = MarketDataRouter()
    day_frames = []
    
    for sym in universe:
        df, _ = router.get_hourly_for_day(sym, pd.to_datetime(test_date_str))
        if df is not None and not df.empty:
            df = df.copy()
            df['symbol'] = sym
            day_frames.append(df)
    
    if not day_frames:
        pytest.skip(f"No hourly data available for {test_date_str}")
    
    combined = pd.concat(day_frames, axis=0).sort_index()
    
    # Run simulation for single day
    strat = prepare_strategy(universe=universe)
    strat._scheduled_day_trades = sched
    setattr(strat, "_is_first_bar_of_day", lambda s, d: True)
    
    sim = BacktestSimulator(
        strategy=strat,
        start_date=test_date_str,
        end_date=test_date_str,
        config_path="config/settings.yaml"
    )
    
    _ = sim.run_simulation(combined)
    fills_df = sim.get_unified_fills_dataframe().copy()
    
    if fills_df.empty:
        pytest.skip(f"No fills generated for {test_date_str}")
    
    # Track positions throughout the day
    position_tracker = PositionTracker()
    sym_col = 'symbol' if 'symbol' in fills_df.columns else 'ticker'
    
    fills_sorted = fills_df.sort_values('timestamp')
    intraday_positions = {}
    
    for _, fill in fills_sorted.iterrows():
        position_tracker.process_fill(
            timestamp=pd.to_datetime(fill['timestamp']),
            symbol=fill[sym_col],
            side=fill['side'],
            quantity=fill['quantity'],
            order_type=fill.get('order_type', 'UNKNOWN')
        )
        
        # Track maximum intraday position
        current_pos = position_tracker.get_eod_positions(test_date_str)
        for symbol, pos in current_pos.items():
            if symbol not in intraday_positions:
                intraday_positions[symbol] = {'max_long': 0, 'max_short': 0}
            intraday_positions[symbol]['max_long'] = max(intraday_positions[symbol]['max_long'], pos)
            intraday_positions[symbol]['max_short'] = min(intraday_positions[symbol]['max_short'], pos)
    
    # Validate end-of-day positions
    eod_positions = position_tracker.get_eod_positions(test_date_str)
    
    print(f"[MOC-DETAIL] Intraday position ranges: {intraday_positions}")
    print(f"[MOC-DETAIL] End-of-day positions: {eod_positions}")
    
    # Critical assertion: All EOD positions must be zero
    non_zero_positions = {sym: pos for sym, pos in eod_positions.items() if pos != 0}
    assert not non_zero_positions, f"End-of-day positions not flattened: {non_zero_positions}"
    
    # Validate that positions were actually opened during the day
    had_activity = any(
        info['max_long'] > 0 or info['max_short'] < 0 
        for info in intraday_positions.values()
    )
    
    if had_activity:
        # Ensure MOC fills exist
        moc_fills = fills_df[
            (fills_df.get('order_type') == 'MOC') | 
            (fills_df.get('attempt_type') == 'moc')
        ]
        assert len(moc_fills) > 0, "Expected MOC fills to close intraday positions"
        
        print(f"✅ [MOC-DETAIL] PASSED: {len(moc_fills)} MOC fills properly closed all intraday positions")
    else:
        print(f"ℹ️  [MOC-DETAIL] No intraday positions opened on {test_date_str}")


if __name__ == "__main__":
    # Run the tests directly
    test_moc_flattening_3month_validation()
    test_moc_flattening_single_day_detailed()
    print("All MOC flattening tests passed!")
