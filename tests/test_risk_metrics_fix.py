#!/usr/bin/env python3
"""
Focused test script to check if risk metrics calculation errors (2.1-2.4) are perfectly handled.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.performance_metrics import PerformanceMetrics, ComprehensivePerformanceAnalysis
from engine.portfolio import EnhancedPortfolio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/risk_metrics_test.log')
    ]
)

def create_test_data_with_issues():
    """
    Create test data that reproduces the issues mentioned in the notepad:
    - 2.1: Volatility shows 0.0 in JSON report
    - 2.2: Sharpe ratio discrepancy: terminal shows -1.7282, JSON shows 0.0
    - 2.3: Max drawdown discrepancy: terminal shows -3.63%, JSON shows 0.0
    - 2.4: Returns array calculation appears flawed
    """
    logger = logging.getLogger(__name__)
    
    # Create portfolio
    portfolio = EnhancedPortfolio(config_path="config/settings.yaml")
    
    # Create test scenario that matches the issues
    initial_capital = 100000.0
    num_days = 150
    
    # Create portfolio values with the issue described in 1.3: static 100,000 for first 29 entries
    portfolio_values = []
    daily_returns = []
    
    # First 29 days: static portfolio value (strategy not initialized yet)
    for i in range(29):
        portfolio_values.append(initial_capital)
        daily_returns.append(0.0)  # No returns during initialization
    
    # After day 29: strategy starts trading with varying returns
    current_value = initial_capital
    np.random.seed(42)  # For reproducible results
    
    for i in range(29, num_days):
        # Generate realistic daily returns with some volatility
        daily_return = np.random.normal(0.0005, 0.015)  # Small positive mean, 1.5% volatility
        daily_returns.append(daily_return)
        
        # Update portfolio value
        current_value = current_value * (1 + daily_return)
        portfolio_values.append(current_value)
    
    # Ensure final value gives us a reasonable total return
    final_value = 102300.0  # 2.30% total return
    portfolio_values[-1] = final_value
    # Recalculate the last daily return
    daily_returns[-1] = (final_value - portfolio_values[-2]) / portfolio_values[-2]
    
    logger.info(f"Test data created:")
    logger.info(f"  Initial capital: {initial_capital}")
    logger.info(f"  Final value: {final_value}")
    logger.info(f"  Total return: {(final_value - initial_capital) / initial_capital:.4f}")
    logger.info(f"  Number of days: {len(portfolio_values)}")
    logger.info(f"  Static days (no trading): 29")
    logger.info(f"  Trading days: {len(portfolio_values) - 29}")
    logger.info(f"  Daily returns mean: {np.mean(daily_returns):.8f}")
    logger.info(f"  Daily returns std: {np.std(daily_returns):.8f}")
    logger.info(f"  Non-zero returns: {np.count_nonzero(daily_returns)}")
    logger.info(f"  Zero returns: {np.sum(np.array(daily_returns) == 0)}")
    
    return portfolio, portfolio_values, daily_returns

def test_risk_metrics_calculation():
    """Test risk metrics calculation to check for errors 2.1-2.4."""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("TESTING RISK METRICS CALCULATION - CHECKING ERRORS 2.1-2.4")
    logger.info("=" * 80)
    
    # Create test data
    portfolio, portfolio_values, daily_returns = create_test_data_with_issues()
    
    # Test PerformanceMetrics risk calculation
    logger.info("\n" + "=" * 60)
    logger.info("TESTING PERFORMANCE_METRICS RISK CALCULATION")
    logger.info("=" * 60)
    
    performance_metrics = PerformanceMetrics(portfolio)
    
    # Calculate risk metrics
    risk_metrics = performance_metrics.calculate_risk_metrics(daily_returns, datetime.now())
    
    logger.info(f"Risk Metrics Results:")
    logger.info(f"  Volatility: {risk_metrics.volatility:.8f}")
    logger.info(f"  Sharpe Ratio: {risk_metrics.sharpe_ratio:.8f}")
    logger.info(f"  Max Drawdown: {risk_metrics.max_drawdown:.8f}")
    logger.info(f"  Sortino Ratio: {risk_metrics.sortino_ratio:.8f}")
    logger.info(f"  Calmar Ratio: {risk_metrics.calmar_ratio:.8f}")
    
    # Check for error 2.1: Volatility should not be 0.0
    if risk_metrics.volatility == 0.0:
        logger.error("❌ ERROR 2.1: Volatility is 0.0 - this should not happen with varying returns!")
    else:
        logger.info("✅ ERROR 2.1 FIXED: Volatility is not 0.0")
    
    # Check for error 2.2: Sharpe ratio should be reasonable
    if risk_metrics.sharpe_ratio == 0.0:
        logger.error("❌ ERROR 2.2: Sharpe ratio is 0.0 - this should not happen!")
    else:
        logger.info("✅ ERROR 2.2 FIXED: Sharpe ratio is not 0.0")
    
    # Check for error 2.3: Max drawdown should be reasonable
    if risk_metrics.max_drawdown == 0.0:
        logger.error("❌ ERROR 2.3: Max drawdown is 0.0 - this should not happen!")
    else:
        logger.info("✅ ERROR 2.3 FIXED: Max drawdown is not 0.0")
    
    # Test ComprehensivePerformanceAnalysis
    logger.info("\n" + "=" * 60)
    logger.info("TESTING COMPREHENSIVE_ANALYSIS RISK CALCULATION")
    logger.info("=" * 60)
    
    comprehensive_analysis = ComprehensivePerformanceAnalysis(portfolio)
    
    # Run comprehensive analysis
    analysis_results = comprehensive_analysis.run_comprehensive_analysis(
        portfolio_values=portfolio_values,
        daily_returns=daily_returns,
        start_date=datetime.now()
    )
    
    # Get risk analysis from results
    risk_analysis = analysis_results.get('risk_analysis', {})
    
    logger.info(f"Comprehensive Analysis Risk Metrics:")
    logger.info(f"  Volatility: {risk_analysis.get('volatility', 'NOT_FOUND')}")
    logger.info(f"  Sharpe Ratio: {risk_analysis.get('sharpe_ratio', 'NOT_FOUND')}")
    logger.info(f"  Max Drawdown: {risk_analysis.get('max_drawdown', 'NOT_FOUND')}")
    
    # Test JSON export
    logger.info("\n" + "=" * 60)
    logger.info("TESTING JSON EXPORT CONSISTENCY")
    logger.info("=" * 60)
    
    # Export to JSON
    json_report_path = "reports/test_risk_metrics_json_export.json"
    comprehensive_analysis.generate_performance_report(
        analysis_results=analysis_results,
        output_path=json_report_path
    )
    
    # Read back the JSON file
    with open(json_report_path, 'r') as f:
        exported_data = json.load(f)
    
    # Check risk metrics in JSON export
    exported_risk_analysis = exported_data.get('risk_analysis', {})
    
    logger.info(f"JSON Export Risk Metrics:")
    logger.info(f"  Volatility: {exported_risk_analysis.get('volatility', 'NOT_FOUND')}")
    logger.info(f"  Sharpe Ratio: {exported_risk_analysis.get('sharpe_ratio', 'NOT_FOUND')}")
    logger.info(f"  Max Drawdown: {exported_risk_analysis.get('max_drawdown', 'NOT_FOUND')}")
    
    # Compare terminal vs JSON
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON: TERMINAL vs JSON EXPORT")
    logger.info("=" * 60)
    
    terminal_volatility = risk_metrics.volatility
    terminal_sharpe = risk_metrics.sharpe_ratio
    terminal_max_dd = risk_metrics.max_drawdown
    
    json_volatility = exported_risk_analysis.get('volatility', 'NOT_FOUND')
    json_sharpe = exported_risk_analysis.get('sharpe_ratio', 'NOT_FOUND')
    json_max_dd = exported_risk_analysis.get('max_drawdown', 'NOT_FOUND')
    
    logger.info(f"Volatility comparison:")
    logger.info(f"  Terminal: {terminal_volatility:.8f}")
    logger.info(f"  JSON:     {json_volatility}")
    if terminal_volatility == json_volatility:
        logger.info("  ✅ CONSISTENT")
    else:
        logger.error("  ❌ INCONSISTENT")
    
    logger.info(f"Sharpe Ratio comparison:")
    logger.info(f"  Terminal: {terminal_sharpe:.8f}")
    logger.info(f"  JSON:     {json_sharpe}")
    if terminal_sharpe == json_sharpe:
        logger.info("  ✅ CONSISTENT")
    else:
        logger.error("  ❌ INCONSISTENT")
    
    logger.info(f"Max Drawdown comparison:")
    logger.info(f"  Terminal: {terminal_max_dd:.8f}")
    logger.info(f"  JSON:     {json_max_dd}")
    if terminal_max_dd == json_max_dd:
        logger.info("  ✅ CONSISTENT")
    else:
        logger.error("  ❌ INCONSISTENT")
    
    # Test returns array calculation (error 2.4)
    logger.info("\n" + "=" * 60)
    logger.info("TESTING RETURNS ARRAY CALCULATION (ERROR 2.4)")
    logger.info("=" * 60)
    
    # Calculate returns from portfolio values
    calculated_returns = []
    for i in range(1, len(portfolio_values)):
        if portfolio_values[i-1] > 0:
            daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
        else:
            daily_return = 0.0
        calculated_returns.append(daily_return)
    
    logger.info(f"Returns array analysis:")
    logger.info(f"  Original daily_returns length: {len(daily_returns)}")
    logger.info(f"  Calculated returns length: {len(calculated_returns)}")
    logger.info(f"  Original returns mean: {np.mean(daily_returns):.8f}")
    logger.info(f"  Calculated returns mean: {np.mean(calculated_returns):.8f}")
    logger.info(f"  Original returns std: {np.std(daily_returns):.8f}")
    logger.info(f"  Calculated returns std: {np.std(calculated_returns):.8f}")
    
    # Check if returns arrays are consistent
    if len(daily_returns) == len(calculated_returns):
        returns_diff = np.abs(np.array(daily_returns) - np.array(calculated_returns))
        max_diff = np.max(returns_diff)
        logger.info(f"  Maximum difference between arrays: {max_diff:.8f}")
        if max_diff < 1e-10:
            logger.info("  ✅ Returns arrays are consistent")
        else:
            logger.error("  ❌ Returns arrays are inconsistent")
    else:
        logger.error("  ❌ Returns arrays have different lengths")
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY - RISK METRICS ERRORS 2.1-2.4")
    logger.info("=" * 80)
    
    issues_found = []
    
    if risk_metrics.volatility == 0.0:
        issues_found.append("2.1: Volatility is 0.0")
    
    if risk_metrics.sharpe_ratio == 0.0:
        issues_found.append("2.2: Sharpe ratio is 0.0")
    
    if risk_metrics.max_drawdown == 0.0:
        issues_found.append("2.3: Max drawdown is 0.0")
    
    if terminal_volatility != json_volatility or terminal_sharpe != json_sharpe or terminal_max_dd != json_max_dd:
        issues_found.append("2.4: Terminal vs JSON inconsistency")
    
    if issues_found:
        logger.error(f"❌ ISSUES FOUND: {len(issues_found)}")
        for issue in issues_found:
            logger.error(f"  - {issue}")
    else:
        logger.info("✅ ALL RISK METRICS ERRORS 2.1-2.4 ARE FIXED!")
    
    return len(issues_found) == 0

def main():
    """Main test function."""
    logger = logging.getLogger(__name__)
    
    try:
        success = test_risk_metrics_calculation()
        
        if success:
            logger.info("\n🎉 SUCCESS: All risk metrics calculation errors are fixed!")
            return 0
        else:
            logger.error("\n❌ FAILURE: Some risk metrics calculation errors remain!")
            return 1
            
    except Exception as e:
        logger.error(f"Error during risk metrics test: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 