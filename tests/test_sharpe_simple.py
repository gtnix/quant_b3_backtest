#!/usr/bin/env python3
"""
Simple test script to debug Sharpe ratio calculation directly.
"""

import sys
import os
import logging
import numpy as np
from datetime import datetime
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.performance_metrics import PerformanceMetrics, RiskAdjustedMetrics, ComprehensivePerformanceAnalysis
from engine.portfolio import EnhancedPortfolio

# Configure logging to see all debug information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/sharpe_simple_debug.log')
    ]
)

def main():
    """Test Sharpe ratio calculation directly."""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize portfolio
        portfolio = EnhancedPortfolio(config_path="config/settings.yaml")
        
        # Create sample data that matches the issue
        # Based on the notepad, we have:
        # - Terminal shows Sharpe = -1.7282
        # - JSON shows Sharpe = 0.0
        # - Total return: 2.30% over ~7 months
        
        # Create sample portfolio values and returns
        initial_capital = 100000.0
        final_value = 102300.0  # 2.30% return
        num_days = 150  # ~7 months of trading days
        
        # Create portfolio values with some variation
        portfolio_values = []
        daily_returns = []
        
        # Start with initial capital
        current_value = initial_capital
        portfolio_values.append(current_value)
        
        # Generate daily returns with some volatility
        np.random.seed(42)  # For reproducible results
        for i in range(num_days - 1):
            # Generate a daily return with some volatility
            daily_return = np.random.normal(0.00015, 0.02)  # Small positive mean, 2% volatility
            daily_returns.append(daily_return)
            
            # Update portfolio value
            current_value = current_value * (1 + daily_return)
            portfolio_values.append(current_value)
        
        # Ensure final value matches expected
        portfolio_values[-1] = final_value
        # Recalculate the last daily return
        daily_returns[-1] = (final_value - portfolio_values[-2]) / portfolio_values[-2]
        
        logger.info(f"Sample data created:")
        logger.info(f"  Initial capital: {initial_capital}")
        logger.info(f"  Final value: {final_value}")
        logger.info(f"  Total return: {(final_value - initial_capital) / initial_capital:.4f}")
        logger.info(f"  Number of days: {len(portfolio_values)}")
        logger.info(f"  Daily returns mean: {np.mean(daily_returns):.8f}")
        logger.info(f"  Daily returns std: {np.std(daily_returns):.8f}")
        
        # Test PerformanceMetrics Sharpe calculation
        logger.info("=" * 60)
        logger.info("TESTING PERFORMANCE_METRICS Sharpe calculation")
        logger.info("=" * 60)
        
        performance_metrics = PerformanceMetrics(portfolio)
        
        # Calculate risk metrics
        risk_metrics = performance_metrics.calculate_risk_metrics(daily_returns, datetime.now())
        logger.info(f"PerformanceMetrics Sharpe ratio: {risk_metrics.sharpe_ratio:.8f}")
        
        # Test RiskAdjustedMetrics Sharpe calculation
        logger.info("=" * 60)
        logger.info("TESTING RISK_ADJUSTED_METRICS Sharpe calculation")
        logger.info("=" * 60)
        
        risk_adjusted_metrics = RiskAdjustedMetrics(performance_metrics)
        risk_metrics_dict = risk_adjusted_metrics.calculate_all_risk_metrics(daily_returns, datetime.now())
        logger.info(f"RiskAdjustedMetrics Sharpe ratio: {risk_metrics_dict['sharpe_ratio']:.8f}")
        
        # Test ComprehensivePerformanceAnalysis
        logger.info("=" * 60)
        logger.info("TESTING COMPREHENSIVE_ANALYSIS Sharpe calculation")
        logger.info("=" * 60)
        
        comprehensive_analysis = ComprehensivePerformanceAnalysis(portfolio)
        analysis_results = comprehensive_analysis.run_comprehensive_analysis(
            portfolio_values=portfolio_values,
            daily_returns=daily_returns,
            start_date=datetime.now()
        )
        
        risk_analysis = analysis_results.get('risk_analysis', {})
        sharpe_in_analysis = risk_analysis.get('sharpe_ratio', 'NOT_FOUND')
        logger.info(f"ComprehensiveAnalysis Sharpe ratio: {sharpe_in_analysis}")
        
        # Test calculate_all_metrics
        logger.info("=" * 60)
        logger.info("TESTING calculate_all_metrics Sharpe calculation")
        logger.info("=" * 60)
        
        all_metrics = performance_metrics.calculate_all_metrics(
            portfolio_values=portfolio_values,
            daily_returns=daily_returns,
            start_date=datetime.now()
        )
        
        sharpe_in_all_metrics = all_metrics.get('sharpe_ratio', 'NOT_FOUND')
        logger.info(f"calculate_all_metrics Sharpe ratio: {sharpe_in_all_metrics}")
        
        # Compare all results
        logger.info("=" * 60)
        logger.info("COMPARISON OF ALL SHARPE RATIO CALCULATIONS")
        logger.info("=" * 60)
        logger.info(f"1. PerformanceMetrics.calculate_risk_metrics: {risk_metrics.sharpe_ratio:.8f}")
        logger.info(f"2. RiskAdjustedMetrics.calculate_all_risk_metrics: {risk_metrics_dict['sharpe_ratio']:.8f}")
        logger.info(f"3. ComprehensivePerformanceAnalysis: {sharpe_in_analysis}")
        logger.info(f"4. PerformanceMetrics.calculate_all_metrics: {sharpe_in_all_metrics}")
        
        # Check if there are discrepancies
        values = [
            risk_metrics.sharpe_ratio,
            risk_metrics_dict['sharpe_ratio'],
            sharpe_in_analysis if isinstance(sharpe_in_analysis, (int, float)) else None,
            sharpe_in_all_metrics if isinstance(sharpe_in_all_metrics, (int, float)) else None
        ]
        
        valid_values = [v for v in values if v is not None]
        if len(set(valid_values)) > 1:
            logger.error("DISCREPANCY DETECTED: Different Sharpe ratios calculated!")
            logger.error(f"Values: {valid_values}")
        else:
            logger.info("All Sharpe ratio calculations are consistent")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in test: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 