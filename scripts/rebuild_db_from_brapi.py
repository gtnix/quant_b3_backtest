#!/usr/bin/env python3
"""
Rebuild local historical data cache from BRAPI for current portfolio tickers
using fixed range=5y and interval=1h (default intraday). Clears existing cached parquet (daily/hourly/intraday)
for those symbols before downloading.

Usage:
  PYTHONPATH=. python scripts/rebuild_db_from_brapi.py [--portfolio data/portfolio.csv] [--col symbol] [--cache-dir data/brapi_cache]
"""
import argparse
import os
from pathlib import Path
import yaml
import glob

from engine.brapi_provider import BrapiProvider


def read_token() -> str:
    token = ''
    sp = Path('config/secrets.yaml')
    if sp.exists():
        try:
            sec = yaml.safe_load(sp.read_text()) or {}
            token = ((sec.get('brapi') or {}).get('api_token')) or sec.get('BRAPI_API_TOKEN') or ''
        except Exception:
            token = ''
    if not token:
        token = os.environ.get('BRAPI_API_TOKEN', '')
    return token


essential_cols = ['symbol']

def load_portfolio_symbols(path: str, col: str) -> list:
    import pandas as pd
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Portfolio file not found: {path}")
    df = pd.read_csv(p)
    if col not in df.columns:
        col = df.columns[0]
    syms = [str(s).strip().upper() for s in df[col].dropna() if str(s).strip()]
    if not syms:
        raise SystemExit('No symbols in portfolio file')
    return list(dict.fromkeys(syms))


def clear_symbol_cache(cache_dir: Path, symbol: str) -> None:
    for sub in ['daily', 'hourly', 'intraday', 'max_range']:
        d = cache_dir / sub
        if not d.exists():
            continue
        for pf in glob.glob(str(d / f"{symbol}*")):
            try:
                Path(pf).unlink()
            except Exception:
                pass
        # Remove metadata if present
        for jf in glob.glob(str(d / f"{symbol}*_metadata.json")):
            try:
                Path(jf).unlink()
            except Exception:
                pass


def parse_args():
    p = argparse.ArgumentParser(description='Rebuild or sync local BRAPI cache with 5y/1h for portfolio tickers')
    p.add_argument('--portfolio', default='data/portfolio.csv', help='CSV with symbols (default: data/portfolio.csv)')
    p.add_argument('--col', default='symbol', help='Column name for tickers (default: symbol)')
    p.add_argument('--cache-dir', default='data/brapi_cache', help='Cache directory (default: data/brapi_cache)')
    p.add_argument('--mode', choices=['rebuild', 'sync'], default='sync', help='Operation mode: rebuild (clear then download) or sync (download missing only)')
    return p.parse_args()


def main():
    args = parse_args()
    token = read_token()
    bp = BrapiProvider(api_token=token, cache_dir=args.cache_dir, cache_ttl_hours=0)
    syms = load_portfolio_symbols(args.portfolio, args.col)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == 'rebuild':
        print(f"Rebuilding {len(syms)} symbols to 5y/1h ...")
        for i, sym in enumerate(syms, 1):
            print(f"[{i}/{len(syms)}] {sym} -> clearing cache")
            clear_symbol_cache(cache_dir, sym)
            # Force fetch via core path (defaults now 5y/1h)
            try:
                # Use internal generic fetch by calling get_ohlc with a dummy window; core uses fixed params now
                from datetime import datetime, timedelta
                end = datetime.utcnow()
                start = end - timedelta(days=5*365)
                df = bp.get_ohlc(sym, '1h', start, end)
                rows = 0 if df is None else len(df)
                print(f"    downloaded {rows} rows")
            except Exception as e:
                print(f"    ERROR: {e}")
    else:
        print(f"Syncing {len(syms)} symbols to 5y/1h ...")
        for i, sym in enumerate(syms, 1):
            try:
                res = bp.sync_brapi_history(sym)
                status = 'SKIP' if res.get('skipped') else 'SYNC'
                print(f"[{i}/{len(syms)}] {sym} [{status}] downloaded={res.get('downloaded_rows')} saved={res.get('saved_rows')} range={res.get('coverage_start')}→{res.get('coverage_end')}")
            except Exception as e:
                print(f"[{i}/{len(syms)}] {sym} [ERROR] {e}")

    print("Done.")


if __name__ == '__main__':
    main()
