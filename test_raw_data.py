#!/usr/bin/env python3
"""
Test script to examine raw BCB data before validation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from market_data.bcb_daily_factor import BCBDailyFactorRetriever
from datetime import datetime

def test_raw_data():
    """Test to see raw BCB data before validation."""
    
    print("🔍 Testing Raw BCB Data (Before Validation)")
    print("=" * 50)
    
    # Initialize retriever
    retriever = BCBDailyFactorRetriever()
    
    # Test with a full year date range
    start_date = datetime(2024, 6, 27)
    end_date = datetime(2025, 6, 27)
    
    print(f"Fetching data from {start_date.date()} to {end_date.date()}")
    
    try:
        # Fetch raw data from API
        raw_data = retriever._fetch_from_bcb_api(start_date, end_date)
        print(f"✅ Raw API response: {len(raw_data)} records")
        
        # Transform to DataFrame
        df = retriever._transform_to_dataframe(raw_data)
        print(f"✅ Transformed DataFrame: {len(df)} records")
        
        # Show factor statistics
        print(f"\n📊 Factor Statistics (Before Validation):")
        print(f"   Min factor: {df['factor'].min():.8f}")
        print(f"   Max factor: {df['factor'].max():.8f}")
        print(f"   Mean factor: {df['factor'].mean():.8f}")
        print(f"   Std factor: {df['factor'].std():.8f}")
        
        # Show sample data
        print(f"\n📋 Sample Data (First 10 records):")
        print(df.head(10).to_string())
        
        # Check current validation bounds
        min_bound = retriever.config['market']['daily_factor']['validation']['min_factor']
        max_bound = retriever.config['market']['daily_factor']['validation']['max_factor']
        print(f"\n🔧 Current Validation Bounds: [{min_bound}, {max_bound}]")
        
        # Count records within bounds
        within_bounds = ((df['factor'] >= min_bound) & (df['factor'] <= max_bound)).sum()
        print(f"   Records within bounds: {within_bounds}/{len(df)} ({within_bounds/len(df)*100:.1f}%)")
        
        # Show some values outside bounds
        outside_bounds = df[~((df['factor'] >= min_bound) & (df['factor'] <= max_bound))]
        if not outside_bounds.empty:
            print(f"\n❌ Sample Values Outside Bounds:")
            print(outside_bounds.head(5).to_string())
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_raw_data() 