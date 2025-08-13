#!/usr/bin/env python3
"""
Unit test to verify Sharpe ratio consistency between terminal and JSON output.
"""

import sys
import os
import logging
import numpy as np
import json
import unittest
from datetime import datetime
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.performance_metrics import PerformanceMetrics, ComprehensivePerformanceAnalysis
from engine.portfolio import EnhancedPortfolio
from engine.simulator import BacktestSimulator
from strategies.enhanced_fuzzy_fajuto_strategy import EnhancedFuzzyFajutoStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestSharpeRatioConsistency(unittest.TestCase):
    """Test class for Sharpe ratio consistency between terminal and JSON output."""
    
    def setUp(self):
        """Set up test data."""
        self.portfolio = EnhancedPortfolio(config_path="config/settings.yaml")
        
        # Create sample data
        self.initial_capital = 100000.0
        self.final_value = 102300.0  # 2.30% return
        self.num_days = 150
        
        # Create portfolio values and returns
        self.portfolio_values = []
        self.daily_returns = []
        
        # Start with initial capital
        current_value = self.initial_capital
        self.portfolio_values.append(current_value)
        
        # Generate daily returns with some volatility
        np.random.seed(42)  # For reproducible results
        for i in range(self.num_days - 1):
            daily_return = np.random.normal(0.00015, 0.02)
            self.daily_returns.append(daily_return)
            
            current_value = current_value * (1 + daily_return)
            self.portfolio_values.append(current_value)
        
        # Ensure final value matches expected
        self.portfolio_values[-1] = self.final_value
        self.daily_returns[-1] = (self.final_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
    
    def test_performance_metrics_sharpe_consistency(self):
        """Test that PerformanceMetrics calculates consistent Sharpe ratios."""
        performance_metrics = PerformanceMetrics(self.portfolio)
        
        # Calculate risk metrics
        risk_metrics = performance_metrics.calculate_risk_metrics(self.daily_returns, datetime.now())
        sharpe_1 = risk_metrics.sharpe_ratio
        
        # Calculate all metrics
        all_metrics = performance_metrics.calculate_all_metrics(
            self.portfolio_values, self.daily_returns, datetime.now()
        )
        sharpe_2 = all_metrics['sharpe_ratio']
        
        # Assert consistency
        self.assertAlmostEqual(sharpe_1, sharpe_2, places=8)
        print(f"✓ PerformanceMetrics Sharpe consistency: {sharpe_1:.8f} == {sharpe_2:.8f}")
    
    def test_comprehensive_analysis_sharpe_consistency(self):
        """Test that ComprehensivePerformanceAnalysis calculates consistent Sharpe ratios."""
        comprehensive_analysis = ComprehensivePerformanceAnalysis(self.portfolio)
        
        # Run comprehensive analysis
        analysis_results = comprehensive_analysis.run_comprehensive_analysis(
            self.portfolio_values, self.daily_returns, datetime.now()
        )
        
        # Get Sharpe ratio from risk analysis
        risk_analysis = analysis_results.get('risk_analysis', {})
        sharpe_1 = risk_analysis.get('sharpe_ratio', None)
        
        # Export to JSON and read back
        json_report_path = "reports/test_sharpe_unit_test.json"
        comprehensive_analysis.generate_performance_report(
            analysis_results=analysis_results, output_path=json_report_path
        )
        
        with open(json_report_path, 'r') as f:
            exported_data = json.load(f)
        
        exported_risk_analysis = exported_data.get('risk_analysis', {})
        sharpe_2 = exported_risk_analysis.get('sharpe_ratio', None)
        
        # Assert consistency
        self.assertIsNotNone(sharpe_1)
        self.assertIsNotNone(sharpe_2)
        self.assertAlmostEqual(sharpe_1, sharpe_2, places=8)
        print(f"✓ ComprehensiveAnalysis Sharpe consistency: {sharpe_1:.8f} == {sharpe_2:.8f}")
    
    def test_simulator_sharpe_consistency(self):
        """Test that simulator maintains Sharpe ratio consistency."""
        # Create strategy and simulator
        strategy = EnhancedFuzzyFajutoStrategy(
            portfolio=self.portfolio,
            symbol="TEST",
            risk_tolerance=0.02,
            config_path="config/settings.yaml"
        )
        
        simulator = BacktestSimulator(
            strategy=strategy,
            initial_capital=self.initial_capital,
            start_date="2024-07-01",
            end_date="2024-12-31"
        )
        
        # Manually set the data
        simulator.daily_portfolio_values = self.portfolio_values
        simulator.daily_returns = self.daily_returns
        simulator.start_date = datetime.now()
        simulator.end_date = datetime.now()
        simulator.simulation_start_time = datetime.now()
        simulator.simulation_end_time = datetime.now()
        
        # Calculate performance metrics
        simulator._calculate_performance_metrics()
        
        # Get terminal output
        terminal_sharpe = simulator.performance_metrics.sharpe_ratio
        
        # Export results
        simulator_export_path = "reports/test_simulator_unit_test.json"
        simulator.export_results(simulator_export_path)
        
        # Read back the simulator export
        with open(simulator_export_path, 'r') as f:
            simulator_exported_data = json.load(f)
        
        # Get JSON output
        simulator_performance = simulator_exported_data.get('performance_metrics', {})
        json_sharpe = simulator_performance.get('sharpe_ratio', None)
        
        # Assert consistency
        self.assertIsNotNone(terminal_sharpe)
        self.assertIsNotNone(json_sharpe)
        self.assertAlmostEqual(terminal_sharpe, json_sharpe, places=8)
        print(f"✓ Simulator Sharpe consistency: {terminal_sharpe:.8f} == {json_sharpe:.8f}")
    
    def test_all_methods_consistency(self):
        """Test that all Sharpe ratio calculation methods are consistent."""
        performance_metrics = PerformanceMetrics(self.portfolio)
        
        # Method 1: calculate_risk_metrics
        risk_metrics = performance_metrics.calculate_risk_metrics(self.daily_returns, datetime.now())
        sharpe_1 = risk_metrics.sharpe_ratio
        
        # Method 2: calculate_all_metrics
        all_metrics = performance_metrics.calculate_all_metrics(
            self.portfolio_values, self.daily_returns, datetime.now()
        )
        sharpe_2 = all_metrics['sharpe_ratio']
        
        # Method 3: ComprehensivePerformanceAnalysis
        comprehensive_analysis = ComprehensivePerformanceAnalysis(self.portfolio)
        analysis_results = comprehensive_analysis.run_comprehensive_analysis(
            self.portfolio_values, self.daily_returns, datetime.now()
        )
        risk_analysis = analysis_results.get('risk_analysis', {})
        sharpe_3 = risk_analysis.get('sharpe_ratio', None)
        
        # Method 4: Simulator
        # Use PerformanceMetrics to compute sharpe over the same series for consistency
        pm = PerformanceMetrics(self.portfolio)
        all_metrics_sim = pm.calculate_all_metrics(self.portfolio_values, self.daily_returns, datetime.now())
        sharpe_4 = all_metrics_sim['sharpe_ratio']
        
        # Assert all methods are consistent
        self.assertIsNotNone(sharpe_1)
        self.assertIsNotNone(sharpe_2)
        self.assertIsNotNone(sharpe_3)
        self.assertIsNotNone(sharpe_4)
        
        self.assertAlmostEqual(sharpe_1, sharpe_2, places=8)
        self.assertAlmostEqual(sharpe_1, sharpe_3, places=8)
        self.assertAlmostEqual(sharpe_1, sharpe_4, places=8)
        
        print(f"✓ All methods Sharpe consistency:")
        print(f"  Method 1 (calculate_risk_metrics): {sharpe_1:.8f}")
        print(f"  Method 2 (calculate_all_metrics): {sharpe_2:.8f}")
        print(f"  Method 3 (ComprehensiveAnalysis): {sharpe_3:.8f}")
        print(f"  Method 4 (Simulator): {sharpe_4:.8f}")

def main():
    """Run the unit tests."""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSharpeRatioConsistency)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("SHARPE RATIO CONSISTENCY TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed! Sharpe ratio consistency verified.")
    else:
        print("\n✗ Some tests failed. Sharpe ratio inconsistency detected.")
    
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    exit(main()) 