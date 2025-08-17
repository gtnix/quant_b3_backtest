"""
FuzzyFajuto reporting utilities.

Contains a single public function:

export_fuzzy_components_to_csv(symbols, start_date, end_date, cfg) -> str

This function uses prepare_fuzzy_data (daily-only indicators) to compute:
- fuzzy_score_raw = RS + sum(EMA signals) + RSI signal
- fuzzy_score = 0.50*RS + 0.30*sum(EMA signals) + 0.20*RSI signal
- qualified_signal: BUY if fuzzy_score_raw>=1.50 and inputs valid; SELL if <=-1.50

Outputs a CSV in reports/: fuzzy_components_YYYYMMDD-YYYYMMDD.csv
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from engine.market_utils import prepare_fuzzy_data


def export_fuzzy_components_to_csv(symbols: list[str], start_date: str, end_date: str, cfg: dict) -> str:
    """Generate razor-focused fuzzy components CSV per spec.

    Returns path to CSV written, or '' if no rows.
    """
    bm_symbol = ((cfg.get('benchmark', {}) or {}).get('symbol')) or ((cfg.get('brapi', {}) or {}).get('data', {}) or {}).get('ibov_symbol', '^BVSP')
    df_all = prepare_fuzzy_data(symbols, bm_symbol, start_date, end_date)
    if df_all.empty:
        return ''

    # Compute fuzzy scores and qualified label from prepared inputs
    ema_component = (
        df_all['ema_3_signal'].fillna(0.0) + df_all['ema_5_signal'].fillna(0.0) + df_all['ema_10_signal'].fillna(0.0)
        + df_all['ema_15_signal'].fillna(0.0) + df_all['ema_20_signal'].fillna(0.0)
    )
    df_all['fuzzy_score_raw'] = df_all['rs_component'].fillna(0.0) + ema_component + df_all['rsi_signal'].fillna(0.0)
    df_all['fuzzy_score'] = 0.50*df_all['rs_component'].fillna(0.0) + 0.30*ema_component + 0.20*df_all['rsi_signal'].fillna(0.0)

    # Qualified signal requires valid inputs
    valid_inputs = (~df_all['stock_return'].isna()) & (~df_all['ibov_return'].isna()) & (~df_all['atr_value'].isna())
    df_all['qualified_signal'] = ''
    df_all.loc[valid_inputs & (df_all['fuzzy_score_raw'] >= 1.50), 'qualified_signal'] = 'BUY'
    df_all.loc[valid_inputs & (df_all['fuzzy_score_raw'] <= -1.50), 'qualified_signal'] = 'SELL'

    # Order and rounding
    cols = [
        'date','symbol','close','fuzzy_score','fuzzy_score_raw','stock_return','ibov_return','rs_component',
        'ema_3_signal','ema_5_signal','ema_10_signal','ema_15_signal','ema_20_signal',
        'rsi_signal','atr_value','qualified_signal'
    ]
    for c in ['fuzzy_score']:
        df_all[c] = pd.to_numeric(df_all[c], errors='coerce').round(2)
    # Ensure numeric types and rounding
    df_all['close'] = pd.to_numeric(df_all.get('close'), errors='coerce').round(2)
    for c in ['stock_return','ibov_return','atr_value']:
        df_all[c] = pd.to_numeric(df_all[c], errors='coerce').round(4)
    for c in ['rs_component','ema_3_signal','ema_5_signal','ema_10_signal','ema_15_signal','ema_20_signal','rsi_signal']:
        df_all[c] = pd.to_numeric(df_all[c], errors='coerce').round(2)
    df_all = df_all[cols].sort_values(['date','symbol'])

    # Write CSV
    Path('reports').mkdir(exist_ok=True)
    out_path = Path('reports') / f"fuzzy_components_{start_date.replace('-','')}-{end_date.replace('-','')}.csv"
    df_all.to_csv(out_path, index=False)
    return str(out_path)


