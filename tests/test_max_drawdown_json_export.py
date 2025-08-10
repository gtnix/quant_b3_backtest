#!/usr/bin/env python3
"""
Test script to debug max drawdown JSON export specifically.
This script creates a scenario with a known drawdown and verifies it's preserved in JSON export.
"""

import sys
import os
import json
import logging
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.portfolio import EnhancedPortfolio
from engine.performance_metrics import ComprehensivePerformanceAnalysis
from engine.simulator import BacktestSimulator
from strategies.enhanced_fuzzy_fajuto_strategy import EnhancedFuzzyFajutoStrategy

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/test_max_drawdown_json_export.log')
        ]
    )
    return logging.getLogger(__name__)

def create_downturn_scenario():
    """
    Create a portfolio scenario with a known drawdown.
    This simulates a downturn followed by recovery.
    """
    logger = logging.getLogger(__name__)
    
    # Create a scenario with a clear drawdown
    # Start with 100,000, drop to 96,370 (3.63% drawdown), then recover
    initial_capital = 100000.0
    
    # Create daily portfolio values that show a downturn
    # Days 1-50: Steady growth
    # Days 51-100: Sharp decline (creating drawdown)
    # Days 101-150: Recovery
    portfolio_values = []
    
    # Phase 1: Growth (days 1-50)
    current_value = initial_capital
    for i in range(50):
        daily_return = 0.001  # 0.1% daily growth
        current_value *= (1 + daily_return)
        portfolio_values.append(current_value)
    
    # Phase 2: Sharp decline (days 51-100) - creating the drawdown
    peak_value = current_value
    for i in range(50):
        daily_return = -0.002  # 0.2% daily decline
        current_value *= (1 + daily_return)
        portfolio_values.append(current_value)
    
    # Phase 3: Recovery (days 101-150)
    for i in range(50):
        daily_return = 0.0015  # 0.15% daily recovery
        current_value *= (1 + daily_return)
        portfolio_values.append(current_value)
    
    # Calculate daily returns
    daily_returns = []
    for i in range(1, len(portfolio_values)):
        daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
        daily_returns.append(daily_return)
    
    # Calculate expected max drawdown
    cumulative_returns = np.cumprod(1 + np.array(daily_returns))
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    expected_max_drawdown = np.min(drawdown)
    
    logger.info(f"Created downturn scenario:")
    logger.info(f"  Initial capital: R$ {initial_capital:,.2f}")
    logger.info(f"  Peak value: R$ {peak_value:,.2f}")
    logger.info(f"  Final value: R$ {current_value:,.2f}")
    logger.info(f"  Expected max drawdown: {expected_max_drawdown:.4f} ({expected_max_drawdown:.2%})")
    logger.info(f"  Total days: {len(portfolio_values)}")
    
    return portfolio_values, daily_returns, expected_max_drawdown

def main():
    """Test max drawdown JSON export specifically."""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("TESTING MAX DRAWDOWN JSON EXPORT")
    logger.info("=" * 60)
    
    try:
        # Create portfolio and downturn scenario
        portfolio = EnhancedPortfolio()
        portfolio_values, daily_returns, expected_max_drawdown = create_downturn_scenario()
        
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
        
        # Check max drawdown before JSON export
        risk_analysis = analysis_results.get('risk_analysis', {})
        max_drawdown_before_export = risk_analysis.get('max_drawdown', 'NOT_FOUND')
        logger.info(f"Max drawdown BEFORE JSON export: {max_drawdown_before_export}")
        logger.info(f"Expected max drawdown: {expected_max_drawdown}")
        
        # Export to JSON
        json_report_path = "reports/test_max_drawdown_json_export.json"
        comprehensive_analysis.generate_performance_report(
            analysis_results=analysis_results,
            output_path=json_report_path
        )
        
        # Read back the JSON file
        with open(json_report_path, 'r') as f:
            exported_data = json.load(f)
        
        # Check max drawdown after JSON export
        exported_risk_analysis = exported_data.get('risk_analysis', {})
        max_drawdown_after_export = exported_risk_analysis.get('max_drawdown', 'NOT_FOUND')
        logger.info(f"Max drawdown AFTER JSON export: {max_drawdown_after_export}")
        
        # Compare
        logger.info("=" * 60)
        logger.info("COMPARISON")
        logger.info("=" * 60)
        logger.info(f"Expected:           {expected_max_drawdown}")
        logger.info(f"Before export:      {max_drawdown_before_export}")
        logger.info(f"After export:       {max_drawdown_after_export}")
        
        # Check if values match
        if abs(max_drawdown_before_export - expected_max_drawdown) < 1e-6:
            logger.info("✓ Max drawdown calculation is correct")
        else:
            logger.error("✗ Max drawdown calculation is incorrect!")
        
        if max_drawdown_before_export == max_drawdown_after_export:
            logger.info("✓ Max drawdown preserved during JSON export")
        else:
            logger.error("✗ Max drawdown changed during JSON export!")
        
        # Also test the simulator's export_results method
        logger.info("=" * 60)
        logger.info("TESTING SIMULATOR EXPORT RESULTS")
        logger.info("=" * 60)
        
        # Create a mock simulator with our data
        strategy = EnhancedFuzzyFajutoStrategy(
            portfolio=portfolio,
            symbol="TEST",
            risk_tolerance=0.02,
            config_path="config/settings.yaml"
        )
        
        simulator = BacktestSimulator(
            strategy=strategy,
            initial_capital=100000.0,
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
        terminal_max_drawdown = simulator.performance_metrics.max_drawdown
        logger.info(f"Terminal max drawdown: {terminal_max_drawdown}")
        
        # Export results
        simulator_export_path = "reports/test_simulator_max_drawdown_export.json"
        simulator.export_results(simulator_export_path)
        
        # Read back the simulator export
        with open(simulator_export_path, 'r') as f:
            simulator_exported_data = json.load(f)
        
        # Check max drawdown in simulator export
        simulator_performance = simulator_exported_data.get('performance_metrics', {})
        simulator_max_drawdown_export = simulator_performance.get('max_drawdown', 'NOT_FOUND')
        logger.info(f"Simulator export max drawdown: {simulator_max_drawdown_export}")
        
        # Compare simulator results
        logger.info("=" * 60)
        logger.info("SIMULATOR COMPARISON")
        logger.info("=" * 60)
        logger.info(f"Expected:           {expected_max_drawdown}")
        logger.info(f"Terminal output:    {terminal_max_drawdown}")
        logger.info(f"JSON export:        {simulator_max_drawdown_export}")
        
        if abs(terminal_max_drawdown - expected_max_drawdown) < 1e-6:
            logger.info("✓ Simulator max drawdown calculation is correct")
        else:
            logger.error("✗ Simulator max drawdown calculation is incorrect!")
        
        if terminal_max_drawdown == simulator_max_drawdown_export:
            logger.info("✓ Simulator max drawdown preserved during JSON export")
        else:
            logger.error("✗ Simulator max drawdown changed during JSON export!")
        
        # Final summary
        logger.info("=" * 60)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"1. Expected max drawdown:              {expected_max_drawdown}")
        logger.info(f"2. Comprehensive analysis:             {max_drawdown_before_export}")
        logger.info(f"3. Comprehensive JSON export:          {max_drawdown_after_export}")
        logger.info(f"4. Simulator terminal:                 {terminal_max_drawdown}")
        logger.info(f"5. Simulator JSON export:              {simulator_max_drawdown_export}")
        
        # Check if all values are consistent
        all_values = [expected_max_drawdown, max_drawdown_before_export, max_drawdown_after_export, 
                     terminal_max_drawdown, simulator_max_drawdown_export]
        
        if all(abs(v - expected_max_drawdown) < 1e-6 for v in all_values if isinstance(v, (int, float))):
            logger.info("✓ All max drawdown values are consistent!")
            return 0
        else:
            logger.error("✗ Max drawdown values are inconsistent!")
            return 1
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 