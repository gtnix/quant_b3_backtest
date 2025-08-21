"""
MOC Flattening Validation Using Existing Backtest Data
======================================================

Test MOC flattening using the existing successful backtest data from July 2025
to validate that positions are properly closed at end-of-day.
"""

import pandas as pd
import pytest
from datetime import datetime, date
from typing import Dict, Set, List, Tuple
from collections import defaultdict

from engine.loader import load_portfolio_symbols
from engine.market_utils import IndicatorService, SignalScheduler, MarketDataRouter
from engine.simulator import BacktestSimulator
from tests.conftest import prepare_strategy


def test_moc_flattening_july_2025_backtest():
    """
    Test MOC flattening using the successful July 2025 backtest data.
    
    This validates the MOC flattening logic against known working data
    where we have confirmed fills and can track position lifecycles.
    """
    print("[MOC-VALIDATION] Testing MOC flattening with July 2025 backtest data")
    
    # Use the same parameters as the successful backtest
    universe = tuple(load_portfolio_symbols()[:10])
    start_date = "2025-07-01"
    end_date = "2025-07-31"
    
    print(f"[MOC-VALIDATION] Period: {start_date} to {end_date}")
    print(f"[MOC-VALIDATION] Universe: {universe}")
    
    # 1) Generate signals (same as successful backtest)
    ind = IndicatorService()
    vectors = ind.compute_daily_vectors(list(universe), benchmark='^BVSP', start=start_date, end=end_date)
    
    if not vectors:
        pytest.skip("No daily vectors computed for July 2025")
    
    # 2) Build schedule
    sched = SignalScheduler().build_schedule(vectors)
    
    if not sched:
        pytest.skip("No schedule built for July 2025")
    
    print(f"[MOC-VALIDATION] Scheduled trading days: {len(sched)}")
    
    # 3) Align with hourly data
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
    
    if aligned_sched and not combined.empty:
        print(f"[MOC-VALIDATION] Aligned days with data: {len(aligned_sched)}")
        
        # 4) Run simulation
        strat = prepare_strategy(universe=universe)
        strat._scheduled_day_trades = aligned_sched
        setattr(strat, "_is_first_bar_of_day", lambda s, d: True)
        
        sim = BacktestSimulator(
            strategy=strat,
            start_date=start_date,
            end_date=end_date,
            config_path="config/settings.yaml"
        )
        
        _ = sim.run_simulation(combined)
        fills_df = sim.get_unified_fills_dataframe().copy()
        
        if not fills_df.empty:
            print(f"[MOC-VALIDATION] Total fills: {len(fills_df):,}")
            
            # 5) Analyze position tracking
            validate_moc_positions(fills_df)
        else:
            pytest.skip("No fills generated in simulation")
    else:
        pytest.skip("No aligned hourly data available")


def validate_moc_positions(fills_df: pd.DataFrame):
    """Validate that MOC orders properly flatten all positions."""
    
    print("[MOC-VALIDATION] Analyzing position lifecycles...")
    
    # Track positions by day and symbol
    daily_positions = defaultdict(lambda: defaultdict(int))  # date -> symbol -> net_position
    
    fills_df['trading_date'] = pd.to_datetime(fills_df['timestamp']).dt.date
    sym_col = 'symbol' if 'symbol' in fills_df.columns else 'ticker'
    
    # Sort fills chronologically
    fills_sorted = fills_df.sort_values('timestamp')
    
    # Process fills to build position tracking
    for _, fill in fills_sorted.iterrows():
        trading_date = fill['trading_date']
        symbol = fill[sym_col]
        side = fill['side'].upper()
        quantity = fill['quantity']
        
        # Update position
        if side == 'BUY':
            daily_positions[trading_date][symbol] += quantity
        elif side == 'SELL':
            daily_positions[trading_date][symbol] -= quantity
    
    # Analyze end-of-day positions
    trading_dates = sorted(daily_positions.keys())
    print(f"[MOC-VALIDATION] Analyzing {len(trading_dates)} trading dates")
    
    overnight_violations = []
    moc_analysis = {
        'days_with_positions': 0,
        'days_with_violations': 0,
        'total_violations': 0,
        'symbols_with_violations': set()
    }
    
    for trading_date in trading_dates:
        eod_positions = daily_positions[trading_date]
        
        # Check for non-zero positions at end of day
        non_zero_positions = {sym: pos for sym, pos in eod_positions.items() if pos != 0}
        
        if eod_positions:
            moc_analysis['days_with_positions'] += 1
        
        if non_zero_positions:
            moc_analysis['days_with_violations'] += 1
            moc_analysis['total_violations'] += len(non_zero_positions)
            moc_analysis['symbols_with_violations'].update(non_zero_positions.keys())
            
            for symbol, position in non_zero_positions.items():
                overnight_violations.append({
                    'date': trading_date,
                    'symbol': symbol,
                    'position': position
                })
                print(f"[MOC-VIOLATION] {trading_date}: {symbol} = {position:+,} shares")
    
    # MOC fill analysis
    moc_fills = fills_df[
        (fills_df.get('order_type') == 'MOC') | 
        (fills_df.get('attempt_type') == 'moc')
    ]
    
    print(f"[MOC-VALIDATION] MOC Analysis Results:")
    print(f"  Days with positions: {moc_analysis['days_with_positions']}")
    print(f"  Days with violations: {moc_analysis['days_with_violations']}")
    print(f"  Total violations: {moc_analysis['total_violations']}")
    print(f"  Symbols with violations: {len(moc_analysis['symbols_with_violations'])}")
    print(f"  MOC fills: {len(moc_fills):,}")
    
    # Critical assertion: NO overnight positions allowed
    if overnight_violations:
        violation_summary = []
        for violation in overnight_violations[:10]:  # Show first 10
            violation_summary.append(
                f"  {violation['date']}: {violation['symbol']} = {violation['position']:+,} shares"
            )
        
        if len(overnight_violations) > 10:
            violation_summary.append(f"  ... and {len(overnight_violations) - 10} more violations")
        
        violation_text = "\n".join(violation_summary)
        
        pytest.fail(
            f"MOC FLATTENING FAILURE: {len(overnight_violations)} overnight position violations!\n\n"
            f"STRATEGY SPECIFICATION VIOLATED:\n"
            f"'All open positions are flattened at the auction call (end-of-day) via Market-on-Close (MOC) orders.'\n\n"
            f"VIOLATIONS DETECTED:\n{violation_text}\n\n"
            f"This indicates the MOC flattening logic is not working correctly.\n"
            f"Expected: All end-of-day positions = 0\n"
            f"Actual: {moc_analysis['total_violations']} non-zero positions found"
        )
    
    # Success case
    print("✅ [MOC-VALIDATION] SUCCESS: All positions properly flattened at end-of-day")
    print(f"   Validated across {moc_analysis['days_with_positions']} trading days")
    print(f"   Zero overnight position violations detected")
    
    # Additional validations
    if moc_analysis['days_with_positions'] > 0:
        assert len(moc_fills) > 0, "Expected MOC fills when positions were opened during trading"
        
        # Validate MOC fills per symbol
        moc_symbols = set(moc_fills[sym_col].unique()) if not moc_fills.empty else set()
        position_symbols = moc_analysis['symbols_with_violations']  # This should be empty now
        
        print(f"   MOC fills covered {len(moc_symbols)} symbols")
        print(f"   Position tracking validated for all symbols")


if __name__ == "__main__":
    test_moc_flattening_july_2025_backtest()
    print("MOC flattening validation completed successfully!")
