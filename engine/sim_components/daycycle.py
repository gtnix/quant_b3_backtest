from __future__ import annotations

from datetime import datetime, date
from typing import Any, Dict

import pandas as pd


def process_end_of_trading_day(sim, trading_day_date: date):
    """Preserve original end-of-day processing via simulator methods.

    Delegates to portfolio and strategy hooks according to the previous
    implementation to avoid behavior changes.
    """
    try:
        # Close day-trade positions or perform MOC logic if strategy requests
        if hasattr(sim.strategy, 'on_end_of_day'):
            sim.strategy.on_end_of_day(trading_day_date)
    finally:
        # Update portfolio value and daily returns tracking
        pv = sim.portfolio.get_portfolio_value()
        sim.daily_portfolio_values.append(pv)
        if len(sim.daily_portfolio_values) >= 2:
            prev = sim.daily_portfolio_values[-2]
            sim.daily_returns.append((pv - prev) / prev if prev else 0.0)
        else:
            sim.daily_returns.append(0.0)


def prepare_market_data(sim, data: pd.DataFrame, current_timestamp: datetime) -> Dict[str, Any]:
    """Return a minimal non-empty dict when any rows exist for the timestamp.

    Handles both single-asset (one row) and multi-asset (many rows) cases.
    """
    try:
        if current_timestamp not in data.index:
            return {}
        sel = data.loc[current_timestamp]
        # If multi-asset, sel is a DataFrame; if single, it's a Series
        if hasattr(sel, 'iloc') and hasattr(sel, 'columns'):
            # Take the first row to avoid heavy aggregation; simulator uses this only as a presence gate
            row = sel.iloc[0]
            return {
                'open': float(row.get('open', 0.0)),
                'high': float(row.get('high', 0.0)),
                'low': float(row.get('low', 0.0)),
                'close': float(row.get('close', 0.0)),
                'volume': int(row.get('volume', 0) or 0),
            }
        else:
            # Series path
            return {
                'open': float(sel.get('open', 0.0)),
                'high': float(sel.get('high', 0.0)),
                'low': float(sel.get('low', 0.0)),
                'close': float(sel.get('close', 0.0)),
                'volume': int(sel.get('volume', 0) or 0),
            }
    except Exception:
        return {}


def load_sgs_data_for_date(sim, current_date: datetime) -> Dict[str, float]:
    """Access preloaded SGS data; keep keys and types as before."""
    result: Dict[str, float] = {}
    try:
        if getattr(sim, 'all_sgs_data', None) and 11 in sim.all_sgs_data:
            df = sim.all_sgs_data[11]
            if df is not None and not df.empty:
                row = df[df['data'] <= current_date.strftime('%d/%m/%Y')].tail(1)
                if not row.empty:
                    result['selic'] = float(row['valor'].iloc[0])
    except Exception:
        pass
    return result


def load_ibov_data_for_date(sim, current_date: datetime) -> Dict[str, Any]:
    """Placeholder that mirrors original shape; rely on existing loaders upstream."""
    return {}


def calculate_selic_cdi_spread(sim, sgs_data: Dict[str, float]):
    try:
        selic = float(sgs_data.get('selic', 0.0))
        cdi = float(sgs_data.get('cdi', selic))
        return selic - cdi
    except Exception:
        return None


def classify_interest_rate_environment(sim, sgs_data: Dict[str, float]) -> str:
    try:
        selic = float(sgs_data.get('selic', 0.0))
        if selic >= 10.0:
            return 'high_rate'
        if selic >= 6.0:
            return 'moderate_rate'
        return 'low_rate'
    except Exception:
        return 'unknown'


def classify_inflation_environment(sim, sgs_data: Dict[str, float]) -> str:
    try:
        ipca = float(sgs_data.get('ipca', 0.0))
        if ipca >= 6.0:
            return 'high_inflation'
        if ipca >= 3.5:
            return 'moderate_inflation'
        return 'low_inflation'
    except Exception:
        return 'unknown'


