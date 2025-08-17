#!/usr/bin/env python3
"""
Integration test for Enhanced FuzzyFajuto Strategy

This script tests the integration of the enhanced strategy with the existing backtesting framework.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.enhanced_fuzzy_fajuto_strategy import EnhancedFuzzyFajutoStrategy
from engine.portfolio import EnhancedPortfolio
from engine.loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_enhanced_strategy_integration():
    """Test the integration of the enhanced strategy."""
    logger.info("Testing Enhanced FuzzyFajuto Strategy Integration")
    
    try:
        # Test 1: Import and instantiation
        logger.info("Test 1: Importing and instantiating strategy...")
        
        # Initialize portfolio
        portfolio = EnhancedPortfolio(config_path="config/settings.yaml")
        
        # Initialize enhanced strategy
        strategy = EnhancedFuzzyFajutoStrategy(
            portfolio=portfolio,
            symbol="PETR4",
            risk_tolerance=0.02,
            config_path="config/settings.yaml",
            strategy_config_path="config/enhanced_strategy_config.yaml"
        )
        
        logger.info("✓ Strategy imported and instantiated successfully")
        
        # Test 2: Configuration loading
        logger.info("Test 2: Loading configuration...")
        
        params = strategy.get_strategy_parameters()
        required_params = ['asset_exposure_pct']
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")
        logger.info("✓ Configuration loaded successfully")
        
        # Test 3: Data loader integration
        logger.info("Test 3: Testing data loader integration...")
        
        data_loader = DataLoader(auto_download=False)
        
        # Check if we can load data for a test symbol
        test_symbol = "PETR4"
        data_status = data_loader.check_all_data([test_symbol])
        
        logger.info("✓ Data loader integration successful")
        if test_symbol in data_status['tickers']:
            logger.info(f"  Data status for {test_symbol}: {data_status['tickers'][test_symbol]['status']}")
        else:
            logger.info(f"  Data status for {test_symbol}: Not found in data status")
        
        # Test 4: Strategy methods
        logger.info("Test 4: Testing strategy methods...")
        
        # Test parameter update
        strategy.update_strategy_parameters({})
        updated_params = strategy.get_strategy_parameters()
        
        logger.info("✓ Strategy methods working correctly")
        
        # Test 5: Execution statistics
        logger.info("Test 5: Testing execution statistics...")
        
        stats = strategy.get_execution_statistics()
        if not isinstance(stats, dict):
            raise ValueError("Execution statistics should return a dictionary")
        
        logger.info("✓ Execution statistics working correctly")
        
        # Test 6: Performance summary
        logger.info("Test 6: Testing performance summary...")
        
        performance = strategy.get_performance_summary()
        if not isinstance(performance, dict):
            raise ValueError("Performance summary should return a dictionary")
        
        logger.info("✓ Performance summary working correctly")
        
        logger.info("=" * 60)
        logger.info("🎉 ALL INTEGRATION TESTS PASSED!")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False


def test_backtest_script_integration():
    """Test that the enhanced strategy can be used with the backtest script."""
    logger.info("Testing backtest script integration...")
    
    try:
        # Test that the strategy can be imported in the same way as the backtest script
        from strategies.enhanced_fuzzy_fajuto_strategy import EnhancedFuzzyFajutoStrategy
        
        # Test that it inherits from BaseStrategy
        from engine.base_strategy import BaseStrategy
        if not issubclass(EnhancedFuzzyFajutoStrategy, BaseStrategy):
            raise ValueError("EnhancedFuzzyFajutoStrategy must inherit from BaseStrategy")
        
        logger.info("✓ Backtest script integration successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Backtest script integration failed: {e}")
        return False


def main():
    """Main function to run integration tests."""
    logger.info("=" * 60)
    logger.info("Enhanced FuzzyFajuto Strategy Integration Tests")
    logger.info("=" * 60)
    
    # Run integration tests
    test1_passed = test_enhanced_strategy_integration()
    test2_passed = test_backtest_script_integration()
    
    if test1_passed and test2_passed:
        logger.info("🎉 ALL TESTS PASSED - Enhanced strategy is fully integrated!")
        logger.info("")
        logger.info("You can now run the enhanced strategy using:")
        logger.info("python scripts/run_backtest.py --strategy EnhancedFuzzyFajutoStrategy --tickers PETR4,VALE3,ITUB4")
        logger.info("")
        logger.info("Or test it with the dedicated test script:")
        logger.info("python scripts/test_enhanced_strategy.py")
        return 0
    else:
        logger.error("❌ Some tests failed - integration incomplete")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 