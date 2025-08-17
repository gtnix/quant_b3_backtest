from __future__ import annotations

from datetime import datetime, date
from typing import List

import numpy as np
import pandas as pd

from engine.base_strategy import Bar


def calculate_calendar_days_for_trading_days(sim, required_trading_days: int, end_date: datetime) -> int:
    """Shim to preserve previous method. Uses dias_uteis when available."""
    if getattr(sim, 'DIAS_UTEIS_AVAILABLE', False):
        try:
            import dias_uteis  # type: ignore
            days = 0
            trading = 0
            # walk backwards until we count required_trading_days
            d = end_date.date()
            while trading < required_trading_days and days < required_trading_days * 4:
                if dias_uteis.is_du(d):
                    trading += 1
                days += 1
                d = d - pd.Timedelta(days=1).date()
            return max(days, required_trading_days)
        except Exception:
            pass
    # Fallback heuristic
    return int(required_trading_days * 1.5)


def get_warmup_bars(sim, data: pd.DataFrame, current_date: datetime, symbol: str) -> List[Bar]:
    """Extract warmup bars with the same logic as the original method.

    This function expects `sim` to be the BacktestSimulator instance to access
    configuration and strategy attributes.
    """
    try:
        warmup_bars_required = 20
        if hasattr(sim.strategy, 'warmup_bars'):
            warmup_bars_required = sim.strategy.warmup_bars

        atr_period = 14
        if hasattr(sim.strategy, 'atr_period'):
            atr_period = sim.strategy.atr_period

        if hasattr(sim.strategy, 'data_requirements'):
            intelligent_requirement = sim.strategy.data_requirements.get('total_minimum_requirement', warmup_bars_required)
            warmup_bars_required = max(warmup_bars_required, intelligent_requirement)

        min_intraday_bars_for_execution = max(60, warmup_bars_required)
        warmup_bars_required = max(warmup_bars_required, min_intraday_bars_for_execution)

        complete_data = data
        if hasattr(sim.strategy, 'context') and 'complete_data' in sim.strategy.context.metadata:
            complete_data = sim.strategy.context.metadata['complete_data']

        required_trading_days = atr_period + 15
        estimated_calendar_days = calculate_calendar_days_for_trading_days(sim, required_trading_days, current_date)

        if sim.start_date:
            requested_start_dt = sim.start_date
            warmup_start_dt = requested_start_dt - pd.Timedelta(days=estimated_calendar_days)
            slice_mask = (complete_data.index >= warmup_start_dt) & (complete_data.index < requested_start_dt)
            warmup_data = complete_data.loc[slice_mask]
        else:
            warmup_data = complete_data.iloc[:0]

        bars: List[Bar] = []
        if not warmup_data.empty:
            # If multi-asset, filter symbol
            if 'symbol' in warmup_data.columns:
                warmup_data = warmup_data[warmup_data['symbol'].astype(str).str.upper() == symbol.upper()]
            # Build one Bar per unique timestamp with the first row of that ts
            idx = warmup_data.index
            if len(idx) > 0:
                unique_ts = np.unique(idx.values)
                for ts in unique_ts[-warmup_bars_required:]:
                    ts = pd.to_datetime(ts)
                    row = warmup_data.loc[ts]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    bars.append(Bar(
                        symbol=symbol,
                        timestamp=ts.to_pydatetime(),
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=int(row.get('volume', 0)),
                    ))
        return bars
    except Exception:
        return []


