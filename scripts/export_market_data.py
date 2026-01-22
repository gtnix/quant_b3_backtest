#!/usr/bin/env python3
"""Export market data from Neon to CSV for backtesting.

Usage:
    python scripts/export_market_data.py --br      # Export BR only
    python scripts/export_market_data.py --us      # Export US only
    python scripts/export_market_data.py --all     # Export both (default)
"""

import argparse
import os
import psycopg2
import csv
from datetime import datetime, date, timedelta

DATABASE_URL = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")

# Date range: 5 years
END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=5*365)

def export_br(conn, output_file="data/market_data_ibov.csv"):
    """Export BR market data (IBOV components)."""
    cursor = conn.cursor()
    
    # Get IBOV components
    cursor.execute("""
        SELECT DISTINCT symbol FROM b3_index_composition 
        WHERE index_code = 'IBOV'
    """)
    symbols = [row[0] for row in cursor.fetchall()]
    
    if not symbols:
        print("  No IBOV symbols found, using fallback list")
        symbols = ['PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'ABEV3', 'B3SA3', 'WEGE3', 'RENT3', 'BBAS3']
    
    print(f"  BR: {len(symbols)} symbols")
    
    # Query OHLCV data
    query = """
        SELECT symbol, trading_date, open, high, low, close, adj_close, volume
        FROM ohlcv_daily
        WHERE symbol = ANY(%s)
        AND trading_date >= %s
        AND trading_date <= %s
        ORDER BY symbol, trading_date
    """
    
    cursor.execute(query, (symbols, START_DATE, END_DATE))
    rows = cursor.fetchall()
    
    print(f"  BR: {len(rows)} rows fetched")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume'])
        
        for row in rows:
            symbol, dt, open_p, high, low, close, adj_close, volume = row
            date_str = dt.strftime('%Y-%m-%d') if hasattr(dt, 'strftime') else str(dt)
            writer.writerow([symbol, date_str, open_p, high, low, close, adj_close, volume])
    
    print(f"  BR: Exported to {output_file}")
    cursor.close()
    return len(rows)


def export_us(conn, output_file="data/market_data_us.csv"):
    """Export US market data (S&P 500 components)."""
    cursor = conn.cursor()
    
    # Get S&P 500 components
    cursor.execute("""
        SELECT DISTINCT symbol FROM us_index_composition 
        WHERE index_code IN ('SPX', 'SP500', '^GSPC')
    """)
    symbols = [row[0] for row in cursor.fetchall()]
    
    if not symbols:
        print("  No SPX symbols found, using fallback list")
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ']
    
    print(f"  US: {len(symbols)} symbols")
    
    # Query OHLCV data
    query = """
        SELECT symbol, trading_date, open, high, low, close, adj_close, volume
        FROM ohlcv_us
        WHERE symbol = ANY(%s)
        AND trading_date >= %s
        AND trading_date <= %s
        ORDER BY symbol, trading_date
    """
    
    cursor.execute(query, (symbols, START_DATE, END_DATE))
    rows = cursor.fetchall()
    
    print(f"  US: {len(rows)} rows fetched")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume'])
        
        for row in rows:
            symbol, dt, open_p, high, low, close, adj_close, volume = row
            date_str = dt.strftime('%Y-%m-%d') if hasattr(dt, 'strftime') else str(dt)
            writer.writerow([symbol, date_str, open_p, high, low, close, adj_close, volume])
    
    print(f"  US: Exported to {output_file}")
    cursor.close()
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Export market data to CSV")
    parser.add_argument("--br", action="store_true", help="Export BR only")
    parser.add_argument("--us", action="store_true", help="Export US only")
    parser.add_argument("--all", action="store_true", help="Export both (default)")
    args = parser.parse_args()
    
    if not any([args.br, args.us, args.all]):
        args.all = True
    
    if not DATABASE_URL:
        print("ERROR: NEON_DATABASE_URL or DATABASE_URL not set")
        return 1
    
    print("="*50)
    print("  EXPORT MARKET DATA")
    print("="*50)
    print(f"  Date range: {START_DATE} to {END_DATE}")
    print("")
    
    conn = psycopg2.connect(DATABASE_URL)
    
    results = {}
    
    if args.all or args.br:
        results["BR"] = export_br(conn)
    
    if args.all or args.us:
        results["US"] = export_us(conn)
    
    conn.close()
    
    print("")
    print("="*50)
    print("  SUMMARY")
    print("="*50)
    for market, count in results.items():
        print(f"  {market}: {count:,} rows")
    
    return 0


if __name__ == '__main__':
    exit(main())
