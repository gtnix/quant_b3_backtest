#!/usr/bin/env python3
"""
Fetch earliest intraday dataset across all BRAPI intraday intervals for given tickers.

Outputs:
  - Ticker, chosen interval, earliest (local/UTC), rows downloaded, cache path

Examples:
  PYTHONPATH=. python scripts/fetch_earliest_intraday.py --tickers VALE3,PETR4
  PYTHONPATH=. python scripts/fetch_earliest_intraday.py --file data/portfolio.csv --col symbol
"""

import argparse
import os
from pathlib import Path
import yaml

from engine.brapi_provider import BrapiProvider


def read_token() -> str:
    token = ""
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
    p = argparse.ArgumentParser(description='Fetch earliest intraday data for tickers')
    p.add_argument('--tickers', help='Comma-separated tickers (e.g., VALE3,PETR4)')
    p.add_argument('--file', default='data/portfolio.csv', help='CSV file with tickers (default: data/portfolio.csv)')
    p.add_argument('--col', default='symbol', help='Ticker column in CSV (default: symbol)')
    p.add_argument('--cache-dir', default='data/brapi_cache', help='Cache directory (default: data/brapi_cache)')
    p.add_argument('--report-range', action='store_true', help='Scan cache and print chosen interval and date range for each ticker')
    p.add_argument('--range', dest='range_override', help='Optional BRAPI range (e.g., 6mo, 1y, max)')
    p.add_argument('--interval', dest='interval_override', help='Optional BRAPI interval (e.g., 1d, 1h, 30m). Requires --range')
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

    print('Ticker | Interval | Earliest (Local)           | Earliest (UTC)         | Rows | Cache Path')
    print('------ | -------- | --------------------------- | ----------------------- | ---- | ----------')
    for sym in syms:
        info = bp.get_earliest_intraday_data(sym, range_override=args.range_override, interval_override=args.interval_override)
        print(f"{sym} | {str(info.get('chosen_interval') or 'N/A'):<8} | "
              f"{str(info.get('earliest_local') or 'N/A'):<27} | {str(info.get('earliest_utc') or 'N/A'):<23} | "
              f"{int(info.get('rows') or 0):>4} | {info.get('cache_path') or 'N/A'}")

    # Optional range report based on cache metadata
    if args.report_range:
        from datetime import datetime as _dt
        import json as _json
        intraday_dir = Path(args.cache_dir) / 'intraday'
        if intraday_dir.exists():
            print('\nCache Range Report:')
            print('Ticker | Interval | Start (Local)             | End (Local)')
            print('------ | -------- | ------------------------- | -------------------------')
            for mf in sorted(intraday_dir.glob('*_metadata.json')):
                try:
                    md = _json.loads(mf.read_text())
                    sym = md.get('symbol','?')
                    interval = md.get('interval','?')
                    s = md.get('start')
                    e = md.get('end')
                    tz = pytz.timezone('America/Sao_Paulo')
                    s_loc = _dt.strptime(s, '%Y-%m-%d').replace(hour=10, minute=0).astimezone(tz).strftime('%Y-%m-%d %H:%M:%S %Z') if s else 'N/A'
                    e_loc = _dt.strptime(e, '%Y-%m-%d').replace(hour=17, minute=0).astimezone(tz).strftime('%Y-%m-%d %H:%M:%S %Z') if e else 'N/A'
                    print(f"{sym} | {interval:<8} | {s_loc:<25} | {e_loc}")
                except Exception:
                    continue


if __name__ == '__main__':
    main()


