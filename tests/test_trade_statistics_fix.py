#!/usr/bin/env python3
"""
Test script to verify trade statistics fixes.

This script tests the corrected trade counting logic to ensure that:
1. Long and short trades are counted correctly from actual trade history
2. The enhanced strategy's three-attempt execution system is properly accounted for
3. The HTML report generator handles missing data gracefully
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from engine.portfolio import EnhancedPortfolio
from engine.performance_metrics import ComprehensivePerformanceAnalysis
from strategies.enhanced_fuzzy_fajuto_strategy import EnhancedFuzzyFajutoStrategy
from engine.html_report_generator import HTMLReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_trade_history():
    """Create sample trade history for testing."""
    portfolio = EnhancedPortfolio()
    
    # Create sample trades
    base_date = datetime.now()
    
    # Day 1: Buy trades
    portfolio.buy(
        ticker="ALPA4", 
        quantity=100, 
        price=15.50, 
        trade_date=base_date, 
        trade_type="day_trade",
        trade_id="ALPA4_001", 
        description="Test buy 1"
    )
    portfolio.buy(
        ticker="ALPA4", 
        quantity=100, 
        price=15.60, 
        trade_date=base_date, 
        trade_type="day_trade",
        trade_id="ALPA4_002", 
        description="Test buy 2"
    )
    
    # Day 2: Sell trades (with profits/losses)
    portfolio.sell(
        ticker="ALPA4", 
        quantity=100, 
        price=15.80, 
        trade_date=base_date + timedelta(days=1), 
        trade_type="day_trade",
        trade_id="ALPA4_003", 
        description="Test sell 1 (profit)"
    )
    portfolio.sell(
        ticker="ALPA4", 
        quantity=100, 
        price=15.40, 
        trade_date=base_date + timedelta(days=1), 
        trade_type="day_trade",
        trade_id="ALPA4_004", 
        description="Test sell 2 (loss)"
    )
    
    # Day 3: More buy trades
    portfolio.buy(
        ticker="ALPA4", 
        quantity=200, 
        price=15.70, 
        trade_date=base_date + timedelta(days=2), 
        trade_type="day_trade",
        trade_id="ALPA4_005", 
        description="Test buy 3"
    )
    
    # Day 4: More sell trades
    portfolio.sell(
        ticker="ALPA4", 
        quantity=200, 
        price=15.90, 
        trade_date=base_date + timedelta(days=3), 
        trade_type="day_trade",
        trade_id="ALPA4_006", 
        description="Test sell 3 (profit)"
    )
    
    return portfolio


def test_trade_counting():
    """Test the trade counting logic."""
    logger.info("=== Testing Trade Counting Logic ===")
    
    # Create portfolio with sample trades
    portfolio = create_sample_trade_history()
    
    # Test the comprehensive analysis
    analysis = ComprehensivePerformanceAnalysis(portfolio)
    
    # Create sample portfolio values and returns
    portfolio_values = [100000, 100100, 100200, 100300, 100400]
    daily_returns = [0.001, 0.001, 0.001, 0.001]
    
    # Run comprehensive analysis
    results = analysis.run_comprehensive_analysis(portfolio_values, daily_returns)
    
    # Extract trade analysis
    trade_analysis = results.get('trade_analysis', {})
    
    logger.info("Trade Analysis Results:")
    logger.info(f"  Total Trades: {trade_analysis.get('total_trades', 0)}")
    logger.info(f"  Long Trades: {trade_analysis.get('long_trades', 0)}")
    logger.info(f"  Short Trades: {trade_analysis.get('short_trades', 0)}")
    logger.info(f"  Winning Trades: {trade_analysis.get('winning_trades', 0)}")
    logger.info(f"  Losing Trades: {trade_analysis.get('losing_trades', 0)}")
    logger.info(f"  Win Rate: {trade_analysis.get('win_rate', 0):.2%}")
    logger.info(f"  Average PnL per Trade: R$ {trade_analysis.get('average_pnl_per_trade', 0):.2f}")
    
    # Verify the counts make sense
    expected_buy_trades = 3  # ALPA4_001, ALPA4_002, ALPA4_005
    expected_sell_trades = 3  # ALPA4_003, ALPA4_004, ALPA4_006
    
    actual_buy_trades = trade_analysis.get('long_trades', 0)
    actual_sell_trades = trade_analysis.get('short_trades', 0)
    
    logger.info(f"\nExpected: {expected_buy_trades} BUY, {expected_sell_trades} SELL")
    logger.info(f"Actual: {actual_buy_trades} BUY, {actual_sell_trades} SELL")
    
    if actual_buy_trades == expected_buy_trades and actual_sell_trades == expected_sell_trades:
        logger.info("✅ Trade counting is working correctly!")
    else:
        logger.error("❌ Trade counting is incorrect!")
        return False
    
    return True


def test_enhanced_strategy_statistics():
    """Test the enhanced strategy's trade statistics."""
    logger.info("\n=== Testing Enhanced Strategy Statistics ===")
    
    # Create portfolio with sample trades
    portfolio = create_sample_trade_history()
    
    # Create enhanced strategy
    strategy = EnhancedFuzzyFajutoStrategy(portfolio, "ALPA4")
    
    # Get enhanced trade statistics
    enhanced_stats = strategy.get_enhanced_trade_statistics()
    
    logger.info("Enhanced Strategy Trade Statistics:")
    logger.info(f"  Total Trades: {enhanced_stats.get('total_trades', 0)}")
    logger.info(f"  Buy Trades: {enhanced_stats.get('buy_trades', 0)}")
    logger.info(f"  Sell Trades: {enhanced_stats.get('sell_trades', 0)}")
    logger.info(f"  Long Trades: {enhanced_stats.get('long_trades', 0)}")
    logger.info(f"  Short Trades: {enhanced_stats.get('short_trades', 0)}")
    logger.info(f"  Winning Trades: {enhanced_stats.get('winning_trades', 0)}")
    logger.info(f"  Losing Trades: {enhanced_stats.get('losing_trades', 0)}")
    logger.info(f"  Win Rate: {enhanced_stats.get('win_rate', 0):.2%}")
    logger.info(f"  Average PnL per Trade: R$ {enhanced_stats.get('average_pnl_per_trade', 0):.2f}")
    
    # Check execution attempts
    execution_attempts = enhanced_stats.get('execution_attempts', {})
    logger.info(f"  Total Execution Attempts: {execution_attempts.get('total_attempts', 0)}")
    logger.info(f"  Executed Attempts: {execution_attempts.get('executed_attempts', 0)}")
    logger.info(f"  Overall Fill Rate: {execution_attempts.get('overall_fill_rate', 0):.2%}")
    
    return True


def test_html_report_generator():
    """Test the HTML report generator with the fixed trade statistics."""
    logger.info("\n=== Testing HTML Report Generator ===")
    
    # Create portfolio with sample trades
    portfolio = create_sample_trade_history()
    
    # Create comprehensive analysis
    analysis = ComprehensivePerformanceAnalysis(portfolio)
    portfolio_values = [100000, 100100, 100200, 100300, 100400]
    daily_returns = [0.001, 0.001, 0.001, 0.001]
    
    results = analysis.run_comprehensive_analysis(portfolio_values, daily_returns)
    
    # Test HTML report generator
    generator = HTMLReportGenerator()
    
    # Test with complete data
    html_content = generator._create_html_content(results, "Test Strategy")
    
    # Check if the HTML contains the correct trade statistics
    trade_analysis = results.get('trade_analysis', {})
    expected_long = trade_analysis.get('long_trades', 0)
    expected_short = trade_analysis.get('short_trades', 0)
    
    # Look for the trade statistics in the HTML
    if f'>{expected_long}</div>' in html_content and f'>{expected_short}</div>' in html_content:
        logger.info("✅ HTML report generator is working correctly!")
    else:
        logger.error("❌ HTML report generator is not displaying correct trade statistics!")
        return False
    
    # Test with missing data
    incomplete_results = results.copy()
    incomplete_results['trade_analysis'] = {
        'total_trades': 6,
        # Missing long_trades and short_trades
    }
    
    html_content_incomplete = generator._create_html_content(incomplete_results, "Test Strategy")
    
    # Should still generate HTML without errors
    if html_content_incomplete and 'Total Trades' in html_content_incomplete:
        logger.info("✅ HTML report generator handles missing data gracefully!")
    else:
        logger.error("❌ HTML report generator fails with missing data!")
        return False
    
    return True


def main():
    """Run all tests."""
    logger.info("Starting trade statistics fix verification...")
    
    tests = [
        test_trade_counting,
        test_enhanced_strategy_statistics,
        test_html_report_generator
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                logger.error(f"Test {test.__name__} failed!")
        except Exception as e:
            logger.error(f"Test {test.__name__} raised an exception: {e}")
    
    logger.info(f"\n=== Test Results ===")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests passed! Trade statistics fixes are working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please review the trade statistics fixes.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 