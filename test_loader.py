#!/usr/bin/env python3
"""
Test script for the DataLoader
Process all available tickers and show results
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from engine.loader import DataLoader
from datetime import datetime, timedelta

def main():
    # Initialize data loader
    loader = DataLoader()
    
    # Get available tickers
    available_tickers = loader.get_available_tickers()
    print(f"Found {len(available_tickers)} tickers: {available_tickers}")
    
    # Process data for the last 2 years
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
    
    print(f"\nProcessing data from {start_date} to {end_date}")
    print("=" * 60)
    
    results = {}
    
    for ticker in available_tickers:
        print(f"\nProcessing {ticker}...")
        
        data = loader.load_and_process(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            save_processed=True
        )
        
        if data is not None:
            results[ticker] = {
                'rows': len(data),
                'start_date': data.index.min(),
                'end_date': data.index.max(),
                'current_price': data['close'].iloc[-1],
                'avg_volume': data['volume'].mean(),
                'volatility': data['volatility_20d'].iloc[-1] * 100,
                'rsi': data['rsi'].iloc[-1]
            }
            
            print(f"  ✓ Success: {len(data)} rows")
            print(f"  Current price: R$ {data['close'].iloc[-1]:.2f}")
            print(f"  Volatility (20d): {data['volatility_20d'].iloc[-1]*100:.2f}%")
            print(f"  RSI: {data['rsi'].iloc[-1]:.2f}")
        else:
            print(f"  ✗ Failed to process {ticker}")
    
    # Summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    
    for ticker, stats in results.items():
        print(f"{ticker:6} | {stats['rows']:4} rows | R$ {stats['current_price']:6.2f} | "
              f"{stats['volatility']:5.2f}% | RSI: {stats['rsi']:5.2f}")
    
    print(f"\nTotal tickers processed: {len(results)}")
    print("All processed data saved to data/processed/")

if __name__ == "__main__":
    main() 