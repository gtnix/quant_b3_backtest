#!/usr/bin/env python3
"""
Fetch maximum range BRAPI data across all supported intervals/timeframes for given tickers.

Outputs a table:
  Ticker | Interval | Earliest (Local/UTC) | Latest (Local/UTC) | Rows | Cache Path

Examples:
  PYTHONPATH=. python scripts/fetch_max_range.py --tickers VALE3,PETR4
  PYTHONPATH=. python scripts/fetch_max_range.py --file data/portfolio.csv --col symbol
"""

import argparse
import os
from pathlib import Path
import yaml

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


def parse_args():
    p = argparse.ArgumentParser(description='Fetch maximum range BRAPI dataset for B3 tickers')
    p.add_argument('--tickers', help='Comma-separated tickers (e.g., VALE3,PETR4)')
    p.add_argument('--file', default='data/portfolio.csv', help='CSV file with tickers (default: data/portfolio.csv)')
    p.add_argument('--col', default='symbol', help='Ticker column in CSV (default: symbol)')
    p.add_argument('--cache-dir', default='data/brapi_cache', help='Cache directory (default: data/brapi_cache)')
    return p.parse_args()


def load_tickers(args):
    syms = []
    if args.tickers:
        syms = [s.strip().upper() for s in args.tickers.split(',') if s.strip()]
    if not syms and args.file:
        import pandas as pd
        df = pd.read_csv(args.file)
        col = args.col if args.col in df.columns else df.columns[0]
        syms = [str(s).strip().upper() for s in df[col].dropna() if str(s).strip()]
    if not syms:
        raise SystemExit('No tickers provided. Use --tickers or ensure data/portfolio.csv exists.')
    return syms


def main():
    args = parse_args()
    syms = load_tickers(args)
    token = read_token()
    bp = BrapiProvider(api_token=token, cache_dir=args.cache_dir, cache_ttl_hours=0)

    print('Ticker | Interval | Earliest (Local)           | Earliest (UTC)         | Latest (Local)             | Latest (UTC)               | Rows | Cache Path')
    print('------ | -------- | --------------------------- | ----------------------- | --------------------------- | --------------------------- | ---- | ----------')
    for sym in syms:
        info = bp.get_max_range_data(sym)
        print(f"{sym} | {str(info.get('chosen_interval') or 'N/A'):<8} | "
              f"{str(info.get('earliest_local') or 'N/A'):<27} | {str(info.get('earliest_utc') or 'N/A'):<23} | "
              f"{str(info.get('latest_local') or 'N/A'):<27} | {str(info.get('latest_utc') or 'N/A'):<27} | "
              f"{int(info.get('rows') or 0):>4} | {info.get('cache_path') or 'N/A'}")


if __name__ == '__main__':
    main()


