"""Dynamic IBRA tickers loader.

For backward-compatibility, this module exposes IBRA_TICKERS, but it now reads
from `data/portfolio.csv` (single source of truth). If the file is missing,
IBRA_TICKERS is an empty list.
"""
from pathlib import Path
import pandas as pd

def _load_portfolio_symbols() -> list:
    candidates = [Path('data') / 'portfolio.csv', Path('portfolio.csv')]
    for p in candidates:
        try:
            if p.exists():
                df = pd.read_csv(p)
                col = 'symbol' if 'symbol' in df.columns else df.columns[0]
                syms = [str(s).strip().upper() for s in df[col].dropna() if str(s).strip()]
                return list(dict.fromkeys(syms))
        except Exception:
            pass
    return []

IBRA_TICKERS = _load_portfolio_symbols()
