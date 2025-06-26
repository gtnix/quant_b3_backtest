"""
Test Suite for BacktestSimulator

Comprehensive tests for the BacktestSimulator class to ensure proper
integration with existing project modules and correct functionality.

Author: Your Name
Date: 2024
"""

import unittest
import tempfile
import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys

sys.path.append('..')

from engine.simulator import BacktestSimulator, SimulationResult, PerformanceMetrics
from engine.base_strategy import BaseStrategy, TradingSignal, SignalType, TradeType
from engine.portfolio import EnhancedPortfolio


class MockStrategy(BaseStrategy):
    """Mock strategy for testing purposes."""
    
    def generate_signals(self, market_data):
        """Generate mock trading signals."""
        signals = []
        
        # Simple buy signal on first day, sell signal on last day
        if len(market_data['price_data']) == 1:  # First day
            signals.append(TradingSignal(
                signal_type=SignalType.BUY,
                ticker=self.symbol,
                price=market_data['current_price'],
                quantity=100,
                timestamp=market_data['timestamp']
            ))
        elif len(market_data['price_data']) > 5:  # After 5 days
            signals.append(TradingSignal(
                signal_type=SignalType.SELL,
                ticker=self.symbol,
                price=market_data['current_price'],
                quantity=100,
                timestamp=market_data['timestamp']
            ))
        
        return signals
    
    def manage_risk(self, current_positions, market_data):
        """Mock risk management."""
        return {'action': 'hold'}
    
    def execute_trade(self, signal):
        """Mock trade execution."""
        return True


class TestBacktestSimulator(unittest.TestCase):
    """Comprehensive tests for BacktestSimulator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary config file
        self.config_data = {
            'market': {
                'trading_hours': {
                    'open': '10:00',
                    'close': '16:55',
                    'timezone': 'America/Sao_Paulo'
                },
                'costs': {
                    'brokerage_fee': 0.0,
                    'min_brokerage': 0.0,
                    'emolument': 0.00005,
                    'settlement_day_trade': 0.00018,
                    'settlement_swing_trade': 0.00025,
                    'iss_rate': 0.05
                }
            },
            'taxes': {
                'swing_trade': 0.15,
                'day_trade': 0.20,
                'exemption_limit': 20000,
                'irrf_swing_rate': 0.00005,
                'irrf_day_rate': 0.01
            },
            'portfolio': {
                'initial_cash': 100000,
                'max_positions': 10,
                'position_sizing': 'equal_weight'
            },
            'settlement': {
                'cycle_days': 2,
                'timezone': 'America/Sao_Paulo',
                'strict_mode': True,
                'holiday_calendar': 'b3'
            },
            'loss_carryforward': {
                'max_tracking_years': None,
                'global_loss_limit': None,
                'asset_specific_tracking': True,
                'temporal_management': True,
                'audit_trail_enabled': True,
                'auto_prune_old_losses': False,
                'partial_application': True,
                'performance_caching': True,
                'regulatory_compliance': {
                    'max_offset_percentage': 0.30,
                    'capital_gains_only': True,
                    'perpetual_carryforward': True,
                    'cvm_compliance': True,
                    'receita_federal_compliance': True
                }
            }
        }
        
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(self.config_data, self.temp_config)
        self.temp_config.close()
        
        # Create mock portfolio
        self.mock_portfolio = Mock(spec=EnhancedPortfolio)
        self.mock_portfolio.cash = 100000.0
        self.mock_portfolio.initial_cash = 100000.0
        self.mock_portfolio.total_value = 100000.0
        self.mock_portfolio.positions = {}
        self.mock_portfolio.trade_history = []
        self.mock_portfolio.total_trades = 0
        self.mock_portfolio.total_commission = 0.0
        self.mock_portfolio.total_taxes = 0.0
        self.mock_portfolio.get_portfolio_value.return_value = 100000.0
        self.mock_portfolio.buy.return_value = True
        self.mock_portfolio.sell.return_value = True
        self.mock_portfolio.update_prices.return_value = None
        
        # Create mock settlement manager
        self.mock_settlement_manager = Mock()
        self.mock_settlement_manager.get_available_cash.return_value = 100000.0
        self.mock_settlement_manager.process_settlements.return_value = None
        
        # Create mock strategy
        self.mock_strategy = MockStrategy(
            portfolio=self.mock_portfolio,
            symbol='VALE3',
            config_path=self.temp_config.name
        )
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'open': [50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0],
            'high': [51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0],
            'low': [49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0],
            'close': [50.5, 51.5, 52.5, 53.5, 54.5, 55.5, 56.5, 57.5, 58.5, 59.5],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000]
        }, index=pd.date_range('2024-01-01', periods=10, freq='D'))
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_config.name):
            os.unlink(self.temp_config.name)
    
    def test_initialization(self):
        """Test proper initialization of BacktestSimulator."""
        simulator = BacktestSimulator(
            strategy=self.mock_strategy,
            initial_capital=100000.0,
            start_date='2024-01-01',
            end_date='2024-01-10',
            config_path=self.temp_config.name
        )
        
        self.assertEqual(simulator.initial_capital, 100000.0)
        self.assertEqual(simulator.strategy, self.mock_strategy)
        self.assertIsNotNone(simulator.start_date)
        self.assertIsNotNone(simulator.end_date)
        self.assertEqual(simulator.start_date, datetime(2024, 1, 1))
        self.assertEqual(simulator.end_date, datetime(2024, 1, 10))
    
    def test_initialization_invalid_strategy(self):
        """Test initialization with invalid strategy."""
        with self.assertRaises(ValueError):
            BacktestSimulator(
                strategy="invalid_strategy",
                initial_capital=100000.0
            )
    
    def test_initialization_invalid_capital(self):
        """Test initialization with invalid capital."""
        with self.assertRaises(ValueError):
            BacktestSimulator(
                strategy=self.mock_strategy,
                initial_capital=-1000.0
            )
    
    def test_initialization_invalid_dates(self):
        """Test initialization with invalid date range."""
        with self.assertRaises(ValueError):
            BacktestSimulator(
                strategy=self.mock_strategy,
                initial_capital=100000.0,
                start_date='2024-01-10',
                end_date='2024-01-01'
            )
    
    def test_prepare_data(self):
        """Test data preparation functionality."""
        simulator = BacktestSimulator(
            strategy=self.mock_strategy,
            initial_capital=100000.0,
            config_path=self.temp_config.name
        )
        
        # Test with valid data
        prepared_data = simulator.prepare_data(self.sample_data)
        
        self.assertIsInstance(prepared_data, pd.DataFrame)
        self.assertEqual(len(prepared_data), len(self.sample_data))
        self.assertTrue(all(col in prepared_data.columns for col in ['open', 'high', 'low', 'close', 'volume']))
    
    def test_prepare_data_with_date_filtering(self):
        """Test data preparation with date filtering."""
        simulator = BacktestSimulator(
            strategy=self.mock_strategy,
            initial_capital=100000.0,
            start_date='2024-01-03',
            end_date='2024-01-07',
            config_path=self.temp_config.name
        )
        
        prepared_data = simulator.prepare_data(self.sample_data)
        
        self.assertIsInstance(prepared_data, pd.DataFrame)
        self.assertEqual(len(prepared_data), 5)  # 5 days from 2024-01-03 to 2024-01-07
        self.assertEqual(prepared_data.index.min().date(), datetime(2024, 1, 3).date())
        self.assertEqual(prepared_data.index.max().date(), datetime(2024, 1, 7).date())
    
    def test_prepare_data_invalid_input(self):
        """Test data preparation with invalid input."""
        simulator = BacktestSimulator(
            strategy=self.mock_strategy,
            initial_capital=100000.0,
            config_path=self.temp_config.name
        )
        
        # Test with empty DataFrame
        with self.assertRaises(ValueError):
            simulator.prepare_data(pd.DataFrame())
        
        # Test with missing columns
        invalid_data = pd.DataFrame({'open': [50.0], 'high': [51.0]})
        with self.assertRaises(ValueError):
            simulator.prepare_data(invalid_data)
        
        # Test with non-positive prices
        invalid_data = pd.DataFrame({
            'open': [50.0, -1.0],
            'high': [51.0, 52.0],
            'low': [49.0, 50.0],
            'close': [50.5, 51.5],
            'volume': [1000000, 1100000]
        })
        with self.assertRaises(ValueError):
            simulator.prepare_data(invalid_data)
    
    @patch('engine.simulator.EnhancedPortfolio')
    def test_run_simulation(self, mock_portfolio_class):
        """Test complete simulation run."""
        # Mock the portfolio class
        mock_portfolio_instance = Mock()
        mock_portfolio_instance.cash = 100000.0
        mock_portfolio_instance.initial_cash = 100000.0
        mock_portfolio_instance.total_value = 100000.0
        mock_portfolio_instance.positions = {}
        mock_portfolio_instance.trade_history = []
        mock_portfolio_instance.total_trades = 0
        mock_portfolio_instance.total_commission = 0.0
        mock_portfolio_instance.total_taxes = 0.0
        mock_portfolio_instance.get_portfolio_value.return_value = 100000.0
        mock_portfolio_instance.buy.return_value = True
        mock_portfolio_instance.sell.return_value = True
        mock_portfolio_instance.update_prices.return_value = None
        mock_portfolio_instance.settlement_manager = self.mock_settlement_manager
        
        mock_portfolio_class.return_value = mock_portfolio_instance
        
        simulator = BacktestSimulator(
            strategy=self.mock_strategy,
            initial_capital=100000.0,
            config_path=self.temp_config.name
        )
        
        # Mock the strategy's generate_signals method
        simulator.strategy.generate_signals = Mock(return_value=[])
        
        # Run simulation
        result = simulator.run_simulation(self.sample_data)
        
        # Verify result
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(result.initial_capital, 100000.0)
        self.assertEqual(result.total_trades, 0)  # No trades in mock
        self.assertIsInstance(result.daily_returns, list)
        self.assertIsInstance(result.portfolio_values, list)
        self.assertIsInstance(result.trade_log, list)
    
    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation."""
        simulator = BacktestSimulator(
            strategy=self.mock_strategy,
            initial_capital=100000.0,
            config_path=self.temp_config.name
        )
        
        # Set up test data
        simulator.daily_portfolio_values = [100000.0, 101000.0, 102000.0, 101500.0, 103000.0]
        simulator.daily_returns = [0.0, 0.01, 0.0099, -0.0049, 0.0148]
        simulator.portfolio.total_trades = 5
        simulator.portfolio.total_commission = 100.0
        simulator.portfolio.total_taxes = 50.0
        
        # Calculate metrics
        simulator._calculate_performance_metrics()
        
        # Verify metrics
        self.assertAlmostEqual(simulator.performance_metrics.total_return, 0.03, places=4)
        self.assertEqual(simulator.performance_metrics.total_trades, 5)
        self.assertEqual(simulator.performance_metrics.total_commission, 100.0)
        self.assertEqual(simulator.performance_metrics.total_taxes, 50.0)
        self.assertEqual(simulator.performance_metrics.final_portfolio_value, 103000.0)
    
    def test_get_performance_summary(self):
        """Test performance summary generation."""
        simulator = BacktestSimulator(
            strategy=self.mock_strategy,
            initial_capital=100000.0,
            config_path=self.temp_config.name
        )
        
        # Set up test metrics
        simulator.performance_metrics.total_return = 0.15
        simulator.performance_metrics.sharpe_ratio = 1.2
        simulator.performance_metrics.max_drawdown = 0.05
        simulator.performance_metrics.total_trades = 10
        
        summary = simulator.get_performance_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary['total_return'], 0.15)
        self.assertEqual(summary['sharpe_ratio'], 1.2)
        self.assertEqual(summary['max_drawdown'], 0.05)
        self.assertEqual(summary['total_trades'], 10)
    
    def test_export_results(self):
        """Test results export functionality."""
        simulator = BacktestSimulator(
            strategy=self.mock_strategy,
            initial_capital=100000.0,
            config_path=self.temp_config.name
        )
        
        # Set up test data
        simulator.daily_portfolio_values = [100000.0, 101000.0, 102000.0]
        simulator.daily_returns = [0.0, 0.01, 0.0099]
        simulator.trade_log = [{'date': '2024-01-01', 'action': 'BUY'}]
        simulator.simulation_start_time = datetime(2024, 1, 1)
        simulator.simulation_end_time = datetime(2024, 1, 3)
        
        # Create temporary file for export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_filepath = temp_file.name
        
        try:
            # Export results
            simulator.export_results(temp_filepath)
            
            # Verify file was created
            self.assertTrue(os.path.exists(temp_filepath))
            
            # Verify file content
            with open(temp_filepath, 'r') as f:
                import json
                data = json.load(f)
            
            self.assertIn('simulation_info', data)
            self.assertIn('performance_metrics', data)
            self.assertIn('daily_data', data)
            self.assertIn('trade_log', data)
            
        finally:
            # Clean up
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)
    
    def test_create_simulation_result(self):
        """Test simulation result creation."""
        simulator = BacktestSimulator(
            strategy=self.mock_strategy,
            initial_capital=100000.0,
            config_path=self.temp_config.name
        )
        
        # Set up test data
        simulator.daily_portfolio_values = [100000.0, 101000.0, 102000.0]
        simulator.daily_returns = [0.0, 0.01, 0.0099]
        simulator.trade_log = [{'date': '2024-01-01', 'action': 'BUY'}]
        simulator.simulation_start_time = datetime(2024, 1, 1)
        simulator.simulation_end_time = datetime(2024, 1, 3)
        simulator.performance_metrics.total_return = 0.02
        simulator.performance_metrics.sharpe_ratio = 1.0
        simulator.performance_metrics.max_drawdown = 0.01
        simulator.performance_metrics.total_trades = 5
        simulator.performance_metrics.initial_capital = 100000.0
        simulator.performance_metrics.final_portfolio_value = 102000.0
        
        result = simulator._create_simulation_result()
        
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(result.total_return, 0.02)
        self.assertEqual(result.sharpe_ratio, 1.0)
        self.assertEqual(result.max_drawdown, 0.01)
        self.assertEqual(result.total_trades, 5)
        self.assertEqual(result.initial_capital, 100000.0)
        self.assertEqual(len(result.daily_returns), 3)
        self.assertEqual(len(result.portfolio_values), 3)
        self.assertEqual(len(result.trade_log), 1)
    
    def test_prepare_market_data(self):
        """Test market data preparation for strategy."""
        simulator = BacktestSimulator(
            strategy=self.mock_strategy,
            initial_capital=100000.0,
            config_path=self.temp_config.name
        )
        
        current_date = datetime(2024, 1, 5)
        market_data = simulator._prepare_market_data(self.sample_data, current_date)
        
        self.assertIsInstance(market_data, dict)
        self.assertIn('price_data', market_data)
        self.assertIn('current_price', market_data)
        self.assertIn('current_volume', market_data)
        self.assertIn('timestamp', market_data)
        self.assertIn('market_conditions', market_data)
        
        # Verify data filtering
        self.assertEqual(len(market_data['price_data']), 5)  # Data up to 2024-01-05
        self.assertEqual(market_data['timestamp'], current_date)
    
    def test_execute_trade(self):
        """Test trade execution functionality."""
        simulator = BacktestSimulator(
            strategy=self.mock_strategy,
            initial_capital=100000.0,
            config_path=self.temp_config.name
        )
        
        # Mock strategy methods
        simulator.strategy.validate_market_data = Mock(return_value=True)
        simulator.strategy.check_brazilian_market_constraints = Mock(return_value=True)
        simulator.strategy.calculate_position_size = Mock(return_value=100)
        
        # Create test signal
        signal = TradingSignal(
            signal_type=SignalType.BUY,
            ticker='VALE3',
            price=50.0,
            quantity=100,
            timestamp=datetime(2024, 1, 1)
        )
        
        price_data = pd.Series({
            'open': 50.0,
            'high': 51.0,
            'low': 49.0,
            'close': 50.5,
            'volume': 1000000
        })
        
        # Execute trade
        simulator._execute_trade(signal, price_data)
        
        # Verify trade was executed
        self.assertEqual(len(simulator.trade_log), 1)
        trade_record = simulator.trade_log[0]
        self.assertEqual(trade_record['ticker'], 'VALE3')
        self.assertEqual(trade_record['signal_type'], 'buy')
        self.assertEqual(trade_record['quantity'], 100)
        self.assertEqual(trade_record['price'], 50.0)


class TestSimulationResult(unittest.TestCase):
    """Tests for SimulationResult dataclass."""
    
    def test_simulation_result_creation(self):
        """Test SimulationResult creation."""
        result = SimulationResult(
            total_return=0.15,
            sharpe_ratio=1.2,
            max_drawdown=0.05,
            win_loss_ratio=1.5,
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            final_portfolio_value=115000.0,
            initial_capital=100000.0,
            total_commission=500.0,
            total_taxes=250.0,
            daily_returns=[0.01, 0.02, -0.01],
            portfolio_values=[100000.0, 101000.0, 103020.0],
            trade_log=[{'date': '2024-01-01', 'action': 'BUY'}],
            simulation_duration=3600.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3)
        )
        
        self.assertEqual(result.total_return, 0.15)
        self.assertEqual(result.sharpe_ratio, 1.2)
        self.assertEqual(result.max_drawdown, 0.05)
        self.assertEqual(result.total_trades, 10)
        self.assertEqual(result.final_portfolio_value, 115000.0)
        self.assertEqual(len(result.daily_returns), 3)
        self.assertEqual(len(result.portfolio_values), 3)
        self.assertEqual(len(result.trade_log), 1)


class TestPerformanceMetrics(unittest.TestCase):
    """Tests for PerformanceMetrics dataclass."""
    
    def test_performance_metrics_defaults(self):
        """Test PerformanceMetrics default values."""
        metrics = PerformanceMetrics()
        
        self.assertEqual(metrics.total_return, 0.0)
        self.assertEqual(metrics.sharpe_ratio, 0.0)
        self.assertEqual(metrics.max_drawdown, 0.0)
        self.assertEqual(metrics.total_trades, 0)
        self.assertEqual(metrics.total_commission, 0.0)
        self.assertEqual(metrics.total_taxes, 0.0)
    
    def test_performance_metrics_custom_values(self):
        """Test PerformanceMetrics with custom values."""
        metrics = PerformanceMetrics(
            total_return=0.25,
            sharpe_ratio=1.5,
            max_drawdown=0.08,
            total_trades=20,
            winning_trades=12,
            losing_trades=8,
            total_commission=1000.0,
            total_taxes=500.0
        )
        
        self.assertEqual(metrics.total_return, 0.25)
        self.assertEqual(metrics.sharpe_ratio, 1.5)
        self.assertEqual(metrics.max_drawdown, 0.08)
        self.assertEqual(metrics.total_trades, 20)
        self.assertEqual(metrics.winning_trades, 12)
        self.assertEqual(metrics.losing_trades, 8)
        self.assertEqual(metrics.total_commission, 1000.0)
        self.assertEqual(metrics.total_taxes, 500.0)


def main():
    """Run the test suite."""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    main() 