#!/usr/bin/env python3
"""
Find earliest timestamps for given tickers and intervals.

Usage:
  python scripts/earliest_timestamps.py --file data/portfolio.csv --col symbol
  python scripts/earliest_timestamps.py --tickers VALE3,PETR4,ITUB4
"""
import argparse
import os
from pathlib import Path
import pandas as pd

from engine.brapi_provider import BrapiProvider


def parse_args():
    p = argparse.ArgumentParser(description='Earliest timestamps for tickers')
    p.add_argument("--tickers", help="Comma-separated list of tickers")
    p.add_argument("--file", default="data/portfolio.csv", help="CSV file with tickers (default: data/portfolio.csv)")
    p.add_argument("--col", default="symbol", help="Column name with tickers (default: symbol)")
    return p.parse_args()


def load_tickers(args) -> list:
    if args.tickers:
        return [s.strip().upper() for s in args.tickers.split(',') if s.strip()]
    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"portfolio.csv not found at {path}")
    df = pd.read_csv(path)
    col = args.col if args.col in df.columns else df.columns[0]
    syms = [str(s).strip().upper() for s in df[col].dropna() if str(s).strip()]
    syms = list(dict.fromkeys(syms))
    if not syms:
        raise SystemExit("portfolio.csv is empty or invalid")
    return syms


def main():
    args = parse_args()
    syms = load_tickers(args)
    token = os.environ.get('BRAPI_API_TOKEN','')
    bp = BrapiProvider(api_token=token, cache_dir='data/brapi_cache', cache_ttl_hours=0)
    for s in syms:
        for itv in ("1h","1d"):
            info = bp.get_earliest_timestamp(s, interval=itv)
            print(f"{s} {itv} earliest_utc={info.get('earliest_utc')} rows={info.get('rows')}")


if __name__ == "__main__":
    main()


