#!/usr/bin/env python3
"""Export market data from Neon to CSV for backtesting."""

import os
import psycopg2
import csv
from datetime import datetime

# Database connection
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_HyU68iqJScrQ@ep-wild-cell-af18q8jx-pooler.c-2.us-west-2.aws.neon.tech/neondb?sslmode=require"
)

# Symbols to export (IBOV top stocks)
SYMBOLS = ['PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'ABEV3', 'B3SA3', 'WEGE3', 'RENT3', 'BBAS3']

# Date range
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'

OUTPUT_FILE = 'data/market_data_ibov.csv'

def main():
    print(f"Connecting to Neon database...")
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    # Query data
    query = """
        SELECT symbol, trading_date, open, high, low, close, adj_close, volume
        FROM ohlcv_daily
        WHERE symbol = ANY(%s)
        AND trading_date >= %s
        AND trading_date <= %s
        ORDER BY symbol, trading_date
    """
    
    print(f"Fetching data for {len(SYMBOLS)} symbols from {START_DATE} to {END_DATE}...")
    cursor.execute(query, (SYMBOLS, START_DATE, END_DATE))
    rows = cursor.fetchall()
    
    print(f"Retrieved {len(rows)} rows")
    
    # Write to CSV
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header matching CsvMarketDataProvider expected format
        writer.writerow(['symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume'])
        
        for row in rows:
            symbol, date, open_p, high, low, close, adj_close, volume = row
            # Format date as YYYY-MM-DD
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            writer.writerow([symbol, date_str, open_p, high, low, close, adj_close, volume])
    
    print(f"Exported to {OUTPUT_FILE}")
    
    cursor.close()
    conn.close()

if __name__ == '__main__':
    main()

