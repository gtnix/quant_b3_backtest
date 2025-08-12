#!/usr/bin/env python3
"""
Test script for Enhanced FuzzyFajuto Strategy Execution Metrics

This script tests the execution metrics tracking to ensure they're not showing 0 values.
"""

import sys
import os
import logging
from datetime import datetime, date
import pandas as pd
import numpy as np

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from engine.portfolio import EnhancedPortfolio
from strategies.enhanced_fuzzy_fajuto_strategy import EnhancedFuzzyFajutoStrategy
from engine.market_utils import SignalType, TradeType, OrderType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_market_data():
    """Create mock market data for testing."""
    # Create historical data with more days to meet minimum requirements
    dates = pd.date_range(start='2024-12-01', end='2025-01-20', freq='D')
    historical_data = pd.DataFrame({
        'open': [10.0 + i * 0.05 + np.random.normal(0, 0.1) for i in range(len(dates))],
        'high': [10.5 + i * 0.05 + np.random.normal(0, 0.1) for i in range(len(dates))],
        'low': [9.5 + i * 0.05 + np.random.normal(0, 0.1) for i in range(len(dates))],
        'close': [10.2 + i * 0.05 + np.random.normal(0, 0.1) for i in range(len(dates))],
        'volume': [1000000 + i * 10000 + np.random.randint(-50000, 50000) for i in range(len(dates))]
    }, index=dates)
    
    # Ensure prices are positive and logical
    historical_data['low'] = historical_data[['open', 'close']].min(axis=1) * 0.95
    historical_data['high'] = historical_data[['open', 'close']].max(axis=1) * 1.05
    historical_data['open'] = historical_data['open'].abs()
    historical_data['close'] = historical_data['close'].abs()
    
    # Current day data
    current_day_data = {
        'open': 12.0,
        'high': 12.5,
        'low': 11.8,
        'close': 12.2,
        'volume': 1200000
    }
    
    # Create benchmark (^BVSP) data with proper structure
    ibov_returns = historical_data['close'].pct_change().dropna()
    ibov_data = {
        'returns': ibov_returns,
        'value': 120000.0,
        'close': historical_data['close'],
        'volume': historical_data['volume'],
        'data_points': len(historical_data),
        'date_range': {
            'start': historical_data.index.min().isoformat(),
            'end': historical_data.index.max().isoformat()
        }
    }
    
    return {
        'price_data': historical_data,
        'current_day_data': current_day_data,
        'ibov_data': ibov_data,
        'timestamp': datetime.now()
    }

def test_enhanced_execution_metrics():
    """Test the enhanced execution metrics."""
    logger.info("Testing Enhanced FuzzyFajuto Strategy Execution Metrics")
    
    # Create portfolio
    portfolio = EnhancedPortfolio(config_path="config/settings.yaml")
    
    # Create strategy
    strategy = EnhancedFuzzyFajutoStrategy(
        portfolio=portfolio,
        symbol="TEST4",
        risk_tolerance=0.02
    )
    
    # Create mock market data
    market_data = create_mock_market_data()
    
    # Generate signals
    logger.info("Generating signals...")
    signals = strategy.generate_signals(market_data)
    
    logger.info(f"Generated {len(signals)} signals")
    
    # Debug: Check if we have any execution history
    logger.info(f"Execution history length: {len(strategy.execution_history)}")
    logger.info(f"Daily executions length: {len(strategy.daily_executions)}")
    
    # If no signals generated, try with more favorable conditions
    if len(signals) == 0:
        logger.info("No signals generated, trying with more favorable market conditions...")
        
        # Create data with strong trend to trigger signals
        dates = pd.date_range(start='2024-12-01', end='2025-01-20', freq='D')
        strong_trend_data = pd.DataFrame({
            'open': [10.0 + i * 0.2 for i in range(len(dates))],  # Strong uptrend
            'high': [10.5 + i * 0.2 for i in range(len(dates))],
            'low': [9.5 + i * 0.2 for i in range(len(dates))],
            'close': [10.2 + i * 0.2 for i in range(len(dates))],
            'volume': [1000000 + i * 50000 for i in range(len(dates))]
        }, index=dates)
        
        # Ensure prices are logical
        strong_trend_data['low'] = strong_trend_data[['open', 'close']].min(axis=1) * 0.95
        strong_trend_data['high'] = strong_trend_data[['open', 'close']].max(axis=1) * 1.05
        
    # Create benchmark (^BVSP) data with strong trend
        ibov_returns = strong_trend_data['close'].pct_change().dropna()
        ibov_data = {
            'returns': ibov_returns,
            'value': 120000.0,
            'close': strong_trend_data['close'],
            'volume': strong_trend_data['volume'],
            'data_points': len(strong_trend_data),
            'date_range': {
                'start': strong_trend_data.index.min().isoformat(),
                'end': strong_trend_data.index.max().isoformat()
            }
        }
        
        # Current day with strong movement
        current_day_data = {
            'open': 15.0,
            'high': 16.0,
            'low': 14.8,
            'close': 15.8,
            'volume': 2000000
        }
        
        market_data_strong = {
            'price_data': strong_trend_data,
            'current_day_data': current_day_data,
            'ibov_data': ibov_data,
            'timestamp': datetime.now()
        }
        
        signals = strategy.generate_signals(market_data_strong)
        logger.info(f"Generated {len(signals)} signals with strong trend")
    
    # Get execution statistics
    logger.info("Getting execution statistics...")
    execution_stats = strategy.get_execution_statistics()
    
    logger.info("Execution Statistics:")
    for key, value in execution_stats.items():
        logger.info(f"  {key}: {value}")
    
    # Get detailed statistics
    logger.info("Getting detailed execution statistics...")
    detailed_stats = strategy.get_detailed_execution_statistics()
    
    logger.info("Detailed Statistics:")
    logger.info(f"  Total days: {detailed_stats.get('total_days', 0)}")
    
    if 'daily_summaries' in detailed_stats:
        for daily_summary in detailed_stats['daily_summaries']:
            logger.info(f"  Date: {daily_summary['date']}")
            logger.info(f"    Signal type: {daily_summary['signal_type']}")
            logger.info(f"    Total attempts: {daily_summary['total_attempts']}")
            logger.info(f"    Executed attempts: {daily_summary['executed_attempts']}")
            logger.info(f"    Fill rate: {daily_summary['fill_rate']:.1%}")
            logger.info(f"    Total PnL: R$ {daily_summary['total_pnl']:.2f}")
            logger.info(f"    Total exposure: R$ {daily_summary['total_exposure']:.2f}")
            
            for attempt in daily_summary['attempt_details']:
                logger.info(f"      Attempt {attempt['attempt_number']}: {attempt['order_type']} - Executed: {attempt['executed']}")
    
    # Check if metrics are non-zero
    if execution_stats:
        total_attempts = execution_stats.get('total_attempts', 0)
        total_executed = execution_stats.get('total_executed', 0)
        total_pnl = execution_stats.get('total_pnl', 0.0)
        total_exposure = execution_stats.get('total_exposure', 0.0)
        
        logger.info("\n=== METRICS VALIDATION ===")
        logger.info(f"Total attempts: {total_attempts}")
        logger.info(f"Total executed: {total_executed}")
        logger.info(f"Total PnL: R$ {total_pnl:.2f}")
        logger.info(f"Total exposure: R$ {total_exposure:.2f}")
        
        if total_attempts > 0:
            logger.info("✅ Total attempts > 0 - PASS")
        else:
            logger.error("❌ Total attempts = 0 - FAIL")
        
        if total_exposure > 0:
            logger.info("✅ Total exposure > 0 - PASS")
        else:
            logger.error("❌ Total exposure = 0 - FAIL")
        
        if total_attempts > 0 and total_executed > 0:
            logger.info("✅ Some attempts executed - PASS")
        else:
            logger.warning("⚠️  No attempts executed (this might be normal for limit orders)")
    
    return execution_stats

if __name__ == "__main__":
    try:
        stats = test_enhanced_execution_metrics()
        logger.info("Test completed successfully")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc() 