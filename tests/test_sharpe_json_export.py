#!/usr/bin/env python3
"""
Test script to debug Sharpe ratio JSON export specifically.
"""

import sys
import os
import logging
import numpy as np
import json
from datetime import datetime
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.performance_metrics import PerformanceMetrics, ComprehensivePerformanceAnalysis
from engine.portfolio import EnhancedPortfolio

# Configure logging to see all debug information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/sharpe_json_debug.log')
    ]
)

def main():
    """Test Sharpe ratio JSON export specifically."""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize portfolio
        portfolio = EnhancedPortfolio(config_path="config/settings.yaml")
        
        # Create sample data that matches the issue
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
        
        # Test ComprehensivePerformanceAnalysis with JSON export
        logger.info("=" * 60)
        logger.info("TESTING COMPREHENSIVE_ANALYSIS WITH JSON EXPORT")
        logger.info("=" * 60)
        
        comprehensive_analysis = ComprehensivePerformanceAnalysis(portfolio)
        
        # Run comprehensive analysis
        analysis_results = comprehensive_analysis.run_comprehensive_analysis(
            portfolio_values=portfolio_values,
            daily_returns=daily_returns,
            start_date=datetime.now()
        )
        
        # Check Sharpe ratio before JSON export
        risk_analysis = analysis_results.get('risk_analysis', {})
        sharpe_before_export = risk_analysis.get('sharpe_ratio', 'NOT_FOUND')
        logger.info(f"Sharpe ratio BEFORE JSON export: {sharpe_before_export}")
        
        # Export to JSON
        json_report_path = "reports/test_sharpe_json_export.json"
        comprehensive_analysis.generate_performance_report(
            analysis_results=analysis_results,
            output_path=json_report_path
        )
        
        # Read back the JSON file
        with open(json_report_path, 'r') as f:
            exported_data = json.load(f)
        
        # Check Sharpe ratio after JSON export
        exported_risk_analysis = exported_data.get('risk_analysis', {})
        sharpe_after_export = exported_risk_analysis.get('sharpe_ratio', 'NOT_FOUND')
        logger.info(f"Sharpe ratio AFTER JSON export: {sharpe_after_export}")
        
        # Compare
        logger.info("=" * 60)
        logger.info("COMPARISON")
        logger.info("=" * 60)
        logger.info(f"Before export: {sharpe_before_export}")
        logger.info(f"After export:  {sharpe_after_export}")
        
        if sharpe_before_export == sharpe_after_export:
            logger.info("✓ Sharpe ratio preserved during JSON export")
        else:
            logger.error("✗ Sharpe ratio changed during JSON export!")
        
        # Also test the simulator's export_results method
        logger.info("=" * 60)
        logger.info("TESTING SIMULATOR EXPORT RESULTS")
        logger.info("=" * 60)
        
        from engine.simulator import BacktestSimulator
        from strategies.enhanced_fuzzy_fajuto_strategy import EnhancedFuzzyFajutoStrategy
        
        # Create a mock simulator with our data
        strategy = EnhancedFuzzyFajutoStrategy(
            portfolio=portfolio,
            symbol="TEST",
            risk_tolerance=0.02,
            config_path="config/settings.yaml"
        )
        
        simulator = BacktestSimulator(
            strategy=strategy,
            initial_capital=initial_capital,
            start_date="2024-07-01",
            end_date="2024-12-31"
        )
        
        # Manually set the data
        simulator.daily_portfolio_values = portfolio_values
        simulator.daily_returns = daily_returns
        simulator.start_date = datetime.now()
        simulator.end_date = datetime.now()
        simulator.simulation_start_time = datetime.now()
        simulator.simulation_end_time = datetime.now()
        
        # Calculate performance metrics
        simulator._calculate_performance_metrics()
        
        # Check terminal output
        terminal_sharpe = simulator.performance_metrics.sharpe_ratio
        logger.info(f"Terminal Sharpe ratio: {terminal_sharpe}")
        
        # Export results
        simulator_export_path = "reports/test_simulator_export.json"
        simulator.export_results(simulator_export_path)
        
        # Read back the simulator export
        with open(simulator_export_path, 'r') as f:
            simulator_exported_data = json.load(f)
        
        # Check Sharpe ratio in simulator export
        simulator_performance = simulator_exported_data.get('performance_metrics', {})
        simulator_sharpe_export = simulator_performance.get('sharpe_ratio', 'NOT_FOUND')
        logger.info(f"Simulator export Sharpe ratio: {simulator_sharpe_export}")
        
        # Compare simulator results
        logger.info("=" * 60)
        logger.info("SIMULATOR COMPARISON")
        logger.info("=" * 60)
        logger.info(f"Terminal output: {terminal_sharpe}")
        logger.info(f"JSON export:    {simulator_sharpe_export}")
        
        if terminal_sharpe == simulator_sharpe_export:
            logger.info("✓ Simulator Sharpe ratio preserved during JSON export")
        else:
            logger.error("✗ Simulator Sharpe ratio changed during JSON export!")
        
        # Final summary
        logger.info("=" * 60)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"1. ComprehensiveAnalysis before export: {sharpe_before_export}")
        logger.info(f"2. ComprehensiveAnalysis after export:  {sharpe_after_export}")
        logger.info(f"3. Simulator terminal output:           {terminal_sharpe}")
        logger.info(f"4. Simulator JSON export:               {simulator_sharpe_export}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in test: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 