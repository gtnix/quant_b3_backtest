#!/usr/bin/env python3
"""
Test script for Enhanced FuzzyFajuto Strategy ATR calculation.

This script tests the ATR calculation functionality in the enhanced strategy
to ensure it's working correctly with the Alpha Vantage and YFinance data.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.portfolio import EnhancedPortfolio
from strategies.enhanced_fuzzy_fajuto_strategy import EnhancedFuzzyFajutoStrategy
from engine.loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_ibov_data():
    """Load IBOV data from CSV file."""
    try:
        ibov_file = Path("data/IBOV/IBOV_raw.csv")
        if not ibov_file.exists():
            return None
        
        ibov_data = pd.read_csv(ibov_file, index_col=0)
        ibov_data.index = pd.to_datetime(ibov_data.index, utc=True).tz_localize(None)
        ibov_data.index = ibov_data.index.normalize()
        
        return ibov_data
    except Exception as e:
        logger.error(f"Error loading IBOV data: {e}")
        return None


def test_atr_calculation():
    """Test ATR calculation in the enhanced strategy."""
    
    print("🧪 Testing Enhanced FuzzyFajuto Strategy ATR Calculation")
    print("=" * 60)
    
    try:
        # Initialize portfolio
        portfolio = EnhancedPortfolio(config_path="config/settings.yaml")
        
        # Initialize strategy
        strategy = EnhancedFuzzyFajutoStrategy(
            portfolio=portfolio,
            symbol="PETR4",
            risk_tolerance=0.02,
            config_path="config/settings.yaml",
            strategy_config_path="config/enhanced_strategy_config.yaml"
        )
        
        print(f"✅ Strategy initialized: {strategy}")
        print(f"📊 ATR Period: {strategy.parameters['atr_period']}")
        print(f"🔢 Alpha Factor: {strategy.parameters['alpha_factor']}")
        print(f"🔢 Beta Factor: {strategy.parameters['beta_factor']}")
        
        # Load test data
        data_loader = DataLoader()
        
        # Load PETR4 data
        petr4_data = data_loader.load_raw_data("PETR4")
        if petr4_data is None or petr4_data.empty:
            print("❌ Failed to load PETR4 data")
            return False
        
        print(f"✅ Loaded PETR4 data: {len(petr4_data)} records")
        print(f"📅 Date range: {petr4_data.index.min()} to {petr4_data.index.max()}")
        
        # Load IBOV data
        ibov_data = load_ibov_data()
        if ibov_data is None or ibov_data.empty:
            print("❌ Failed to load IBOV data")
            return False
        
        print(f"✅ Loaded IBOV data: {len(ibov_data)} records")
        
        # Test ATR calculation for a specific date
        test_date = datetime(2024, 1, 15)  # A Monday in 2024
        
        # Prepare market data
        market_data = {
            'price_data': petr4_data,
            'ibov_data': {
                'returns': ibov_data['close'].pct_change(),
                'data_points': len(ibov_data)
            },
            'timestamp': test_date,
            'current_day_data': {
                'open': 35.50,
                'high': 36.20,
                'low': 35.10,
                'close': 35.80,
                'volume': 1000000
            }
        }
        
        print(f"\n🔍 Testing ATR calculation for {test_date.date()}")
        
        # Test ATR calculation
        atr_value = strategy._calculate_atr_previous_day(market_data)
        
        if atr_value is not None:
            print(f"✅ ATR calculation successful: {atr_value:.4f}")
            
            # Test FuzzyFajuto score calculation
            fuzzy_score = strategy._calculate_fuzzy_score_for_day(market_data)
            
            if fuzzy_score is not None:
                print(f"✅ FuzzyFajuto score calculation successful: {fuzzy_score:.4f}")
                
                # Test signal generation
                signals = strategy.generate_signals(market_data)
                
                if signals:
                    print(f"✅ Signal generation successful: {len(signals)} signals")
                    for i, signal in enumerate(signals):
                        print(f"   Signal {i+1}: {signal.signal_type.value} at R$ {signal.price:.2f}")
                else:
                    print("ℹ️  No signals generated (score may be in neutral range)")
                
                # Test execution statistics
                stats = strategy.get_execution_statistics()
                if stats:
                    print(f"✅ Execution statistics available: {stats.get('total_days', 0)} days")
                else:
                    print("ℹ️  No execution statistics yet (first run)")
                
            else:
                print("❌ FuzzyFajuto score calculation failed")
                return False
                
        else:
            print("❌ ATR calculation failed")
            return False
        
        # Test with different dates
        print(f"\n🔍 Testing ATR calculation for multiple dates...")
        
        test_dates = [
            datetime(2024, 1, 15),
            datetime(2024, 2, 15),
            datetime(2024, 3, 15),
            datetime(2024, 4, 15),
            datetime(2024, 5, 15)
        ]
        
        atr_values = []
        for date in test_dates:
            market_data['timestamp'] = date
            market_data['current_day_data'] = {
                'open': 35.50,
                'high': 36.20,
                'low': 35.10,
                'close': 35.80,
                'volume': 1000000
            }
            
            atr = strategy._calculate_atr_previous_day(market_data)
            if atr is not None:
                atr_values.append(atr)
                print(f"   {date.date()}: ATR = {atr:.4f}")
            else:
                print(f"   {date.date()}: ATR calculation failed")
        
        if atr_values:
            print(f"\n📊 ATR Statistics:")
            print(f"   Mean ATR: {np.mean(atr_values):.4f}")
            print(f"   Min ATR: {np.min(atr_values):.4f}")
            print(f"   Max ATR: {np.max(atr_values):.4f}")
            print(f"   Std ATR: {np.std(atr_values):.4f}")
        
        print(f"\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        logger.exception("Test error details:")
        return False


def test_data_quality():
    """Test data quality for ATR calculation."""
    
    print("\n🔍 Testing Data Quality for ATR Calculation")
    print("=" * 50)
    
    try:
        data_loader = DataLoader()
        
        # Test PETR4 data quality
        petr4_data = data_loader.load_raw_data("PETR4")
        
        if petr4_data is None or petr4_data.empty:
            print("❌ Failed to load PETR4 data")
            return False
        
        print(f"📊 PETR4 Data Quality Check:")
        print(f"   Total records: {len(petr4_data)}")
        print(f"   Date range: {petr4_data.index.min()} to {petr4_data.index.max()}")
        print(f"   Missing values:")
        for col in ['open', 'high', 'low', 'close', 'volume']:
            missing = petr4_data[col].isna().sum()
            print(f"     {col}: {missing} ({missing/len(petr4_data)*100:.1f}%)")
        
        # Check for negative prices
        negative_prices = (petr4_data[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()
        print(f"   Records with non-positive prices: {negative_prices}")
        
        # Check for price consistency
        invalid_ohlc = (
            (petr4_data['high'] < petr4_data['low']) |
            (petr4_data['open'] > petr4_data['high']) |
            (petr4_data['close'] > petr4_data['high']) |
            (petr4_data['open'] < petr4_data['low']) |
            (petr4_data['close'] < petr4_data['low'])
        ).sum()
        print(f"   Records with invalid OHLC: {invalid_ohlc}")
        
        # Test IBOV data quality
        ibov_file = Path("data/IBOV/IBOV_raw.csv")
        if ibov_file.exists():
            ibov_data = pd.read_csv(ibov_file, index_col=0)
            ibov_data.index = pd.to_datetime(ibov_data.index, utc=True).tz_localize(None)
            ibov_data.index = ibov_data.index.normalize()
            
            print(f"\n📊 IBOV Data Quality Check:")
            print(f"   Total records: {len(ibov_data)}")
            print(f"   Date range: {ibov_data.index.min()} to {ibov_data.index.max()}")
            print(f"   Missing values:")
            for col in ['open', 'high', 'low', 'close']:
                missing = ibov_data[col].isna().sum()
                print(f"     {col}: {missing} ({missing/len(ibov_data)*100:.1f}%)")
        else:
            print(f"\n❌ IBOV data file not found: {ibov_file}")
            return False
        
        print(f"\n✅ Data quality check completed!")
        return True
        
    except Exception as e:
        print(f"❌ Data quality test failed: {e}")
        logger.exception("Data quality test error details:")
        return False


if __name__ == "__main__":
    print("🚀 Enhanced FuzzyFajuto Strategy ATR Test Suite")
    print("=" * 60)
    
    # Test data quality first
    data_ok = test_data_quality()
    
    if data_ok:
        # Test ATR calculation
        atr_ok = test_atr_calculation()
        
        if atr_ok:
            print("\n🎉 All tests passed! ATR calculation is working correctly.")
        else:
            print("\n❌ ATR calculation tests failed.")
            sys.exit(1)
    else:
        print("\n❌ Data quality tests failed.")
        sys.exit(1) 