"""
Comprehensive Test Suite for Performance Metrics Module

Tests all performance metrics functionality with Brazilian market compliance:
- Returns calculation with 252 trading days
- Risk-adjusted metrics validation
- Tax-aware calculations with Brazilian rules
- Integration with existing portfolio and loss manager
- Regulatory compliance validation

Author: Your Name
Date: 2024
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add the engine directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'engine'))

from performance_metrics import (
    PerformanceMetrics, 
    RiskAdjustedMetrics, 
    TaxAwareMetrics, 
    ComprehensivePerformanceAnalysis,
    ReturnsMetrics,
    RiskMetrics,
    TaxMetrics,
    TradeMetrics
)


class TestPerformanceMetrics(unittest.TestCase):
    """Test suite for PerformanceMetrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock portfolio
        self.mock_portfolio = Mock()
        self.mock_portfolio.initial_cash = 100000.0
        self.mock_portfolio.total_value = 110000.0
        self.mock_portfolio.total_taxes = 1500.0
        self.mock_portfolio.total_commission = 500.0
        self.mock_portfolio.trade_history = [
            {'pnl': 1000, 'trade_type': 'swing_trade', 'taxes': 150},
            {'pnl': -500, 'trade_type': 'swing_trade', 'taxes': 0},
            {'pnl': 2000, 'trade_type': 'day_trade', 'taxes': 400},
            {'pnl': -300, 'trade_type': 'day_trade', 'taxes': 0}
        ]
        
        # Create a mock loss manager
        self.mock_loss_manager = Mock()
        self.mock_loss_manager.get_total_loss_balance.return_value = 800.0
        self.mock_loss_manager.get_loss_summary.return_value = {
            'total_loss_balance': 800.0,
            'assets_with_losses': 2,
            'total_losses_recorded': 3
        }
        
        # Assign loss manager to portfolio
        self.mock_portfolio.loss_manager = self.mock_loss_manager
        
        # Mock configuration
        self.mock_config = {
            'market': {
                'trading_hours': {
                    'timezone': 'America/Sao_Paulo'
                }
            },
            'taxes': {
                'swing_trade': 0.15,
                'day_trade': 0.20,
                'exemption_limit': 20000,
                'irrf_swing_rate': 0.00005,
                'irrf_day_rate': 0.01
            }
        }
    
    @patch('performance_metrics.yaml.safe_load')
    @patch('builtins.open')
    def test_initialization(self, mock_open, mock_yaml_load):
        """Test PerformanceMetrics initialization."""
        mock_yaml_load.return_value = self.mock_config
        
        metrics = PerformanceMetrics(self.mock_portfolio)
        
        self.assertEqual(metrics.TRADING_DAYS_PER_YEAR, 252)
        self.assertEqual(metrics.RISK_FREE_RATE, 0.1175)
        self.assertIsInstance(metrics.returns_metrics, ReturnsMetrics)
        self.assertIsInstance(metrics.risk_metrics, RiskMetrics)
        self.assertIsInstance(metrics.tax_metrics, TaxMetrics)
        self.assertIsInstance(metrics.trade_metrics, TradeMetrics)
    
    def test_calculate_returns(self):
        """Test returns calculation with Brazilian market parameters."""
        # Create sample portfolio values (simulating 10 trading days)
        portfolio_values = [100000, 101000, 102500, 101800, 103200, 
                           104000, 103500, 105000, 104800, 106000]
        
        with patch('performance_metrics.yaml.safe_load') as mock_yaml_load:
            mock_yaml_load.return_value = self.mock_config
            
            metrics = PerformanceMetrics(self.mock_portfolio)
            returns_metrics = metrics.calculate_returns(
                portfolio_values, 
                datetime.now(), 
                datetime.now()
            )
        
        # Verify calculations
        expected_total_return = (106000 - 100000) / 100000  # 0.06
        self.assertAlmostEqual(returns_metrics.total_return, expected_total_return, places=4)
        self.assertEqual(returns_metrics.trading_days, 9)  # 9 daily returns
        self.assertEqual(len(returns_metrics.daily_returns), 9)
        self.assertEqual(len(returns_metrics.cumulative_returns), 10)
        
        # Verify annualized return calculation
        expected_annualized = ((106000 / 100000) ** (252 / 9)) - 1
        self.assertAlmostEqual(returns_metrics.annualized_return, expected_annualized, places=4)
    
    def test_calculate_risk_metrics(self):
        """Test risk-adjusted metrics calculation."""
        # Create sample daily returns
        daily_returns = [0.01, -0.005, 0.02, -0.01, 0.015, -0.008, 0.012, 0.005, -0.003]
        
        with patch('performance_metrics.yaml.safe_load') as mock_yaml_load:
            mock_yaml_load.return_value = self.mock_config
            
            metrics = PerformanceMetrics(self.mock_portfolio)
            # Set returns metrics first
            metrics.returns_metrics.annualized_return = 0.15
            risk_metrics = metrics.calculate_risk_metrics(daily_returns)
        
        # Verify risk metrics
        self.assertIsInstance(risk_metrics, RiskMetrics)
        self.assertGreater(risk_metrics.volatility, 0)
        self.assertLessEqual(risk_metrics.max_drawdown, 0)
        self.assertIsInstance(risk_metrics.sharpe_ratio, float)
        self.assertIsInstance(risk_metrics.sortino_ratio, float)
        self.assertIsInstance(risk_metrics.calmar_ratio, float)
    
    def test_calculate_tax_metrics(self):
        """Test Brazilian tax-specific metrics calculation."""
        with patch('performance_metrics.yaml.safe_load') as mock_yaml_load:
            mock_yaml_load.return_value = self.mock_config
            
            metrics = PerformanceMetrics(self.mock_portfolio)
            # Set returns metrics first
            metrics.returns_metrics.total_return = 0.10
            tax_metrics = metrics.calculate_tax_metrics()
        
        # Verify tax metrics
        self.assertEqual(tax_metrics.total_taxes_paid, 1500.0)
        self.assertEqual(tax_metrics.swing_trade_taxes, 150.0)  # Only swing trade with positive P&L
        self.assertEqual(tax_metrics.day_trade_taxes, 400.0)    # Only day trade with positive P&L
        
        # Verify tax efficiency calculation
        expected_tax_efficiency = (0.10 - (1500.0 / 100000.0)) / 0.10
        self.assertAlmostEqual(tax_metrics.tax_efficiency, expected_tax_efficiency, places=4)
    
    def test_calculate_trade_metrics(self):
        """Test trade-specific metrics calculation."""
        with patch('performance_metrics.yaml.safe_load') as mock_yaml_load:
            mock_yaml_load.return_value = self.mock_config
            
            metrics = PerformanceMetrics(self.mock_portfolio)
            trade_metrics = metrics.calculate_trade_metrics()
        
        # Verify trade metrics
        self.assertEqual(trade_metrics.total_trades, 4)
        self.assertEqual(trade_metrics.winning_trades, 2)
        self.assertEqual(trade_metrics.losing_trades, 2)
        self.assertEqual(trade_metrics.win_rate, 0.5)
        self.assertEqual(trade_metrics.total_commission, 500.0)
        
        # Verify profit factor
        total_wins = 1000 + 2000  # 3000
        total_losses = abs(-500 + -300)  # 800
        expected_profit_factor = 3000 / 800
        self.assertAlmostEqual(trade_metrics.profit_factor, expected_profit_factor, places=4)


class TestRiskAdjustedMetrics(unittest.TestCase):
    """Test suite for RiskAdjustedMetrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_portfolio = Mock()
        self.mock_portfolio.initial_cash = 100000.0
        
        self.mock_config = {
            'market': {
                'trading_hours': {
                    'timezone': 'America/Sao_Paulo'
                }
            },
            'taxes': {
                'swing_trade': 0.15,
                'day_trade': 0.20,
                'exemption_limit': 20000
            }
        }
    
    @patch('performance_metrics.yaml.safe_load')
    def test_calculate_all_risk_metrics(self, mock_yaml_load):
        """Test comprehensive risk metrics calculation."""
        mock_yaml_load.return_value = self.mock_config
        
        performance_metrics = PerformanceMetrics(self.mock_portfolio)
        risk_metrics = RiskAdjustedMetrics(performance_metrics)
        
        # Create sample daily returns
        daily_returns = [0.01, -0.005, 0.02, -0.01, 0.015, -0.008, 0.012, 0.005, -0.003]
        
        risk_metrics_dict = risk_metrics.calculate_all_risk_metrics(daily_returns)
        
        # Verify all risk metrics are present
        expected_metrics = [
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'information_ratio',
            'treynor_ratio', 'jensen_alpha', 'max_drawdown', 'volatility',
            'var_95', 'cvar_95', 'skewness', 'kurtosis', 'ulcer_index', 'gain_to_pain_ratio'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, risk_metrics_dict)
            self.assertIsInstance(risk_metrics_dict[metric], (int, float))
    
    def test_calculate_skewness(self):
        """Test skewness calculation."""
        mock_portfolio = Mock()
        mock_portfolio.initial_cash = 100000.0
        
        with patch('performance_metrics.yaml.safe_load') as mock_yaml_load:
            mock_yaml_load.return_value = self.mock_config
            
            performance_metrics = PerformanceMetrics(mock_portfolio)
            risk_metrics = RiskAdjustedMetrics(performance_metrics)
            
            # Test with symmetric returns
            symmetric_returns = np.array([0.01, -0.01, 0.02, -0.02, 0.005, -0.005])
            skewness = risk_metrics._calculate_skewness(symmetric_returns)
            self.assertIsInstance(skewness, float)
    
    def test_calculate_kurtosis(self):
        """Test kurtosis calculation."""
        mock_portfolio = Mock()
        mock_portfolio.initial_cash = 100000.0
        
        with patch('performance_metrics.yaml.safe_load') as mock_yaml_load:
            mock_yaml_load.return_value = self.mock_config
            
            performance_metrics = PerformanceMetrics(mock_portfolio)
            risk_metrics = RiskAdjustedMetrics(performance_metrics)
            
            # Test with normal-like returns
            returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015, -0.008])
            kurtosis = risk_metrics._calculate_kurtosis(returns)
            self.assertIsInstance(kurtosis, float)


class TestTaxAwareMetrics(unittest.TestCase):
    """Test suite for TaxAwareMetrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_portfolio = Mock()
        self.mock_portfolio.initial_cash = 100000.0
        self.mock_portfolio.total_taxes = 1500.0
        self.mock_portfolio.trade_history = []
        
        self.mock_loss_manager = Mock()
        self.mock_loss_manager.get_total_loss_balance.return_value = 800.0
        self.mock_loss_manager.get_loss_summary.return_value = {
            'total_loss_balance': 800.0,
            'assets_with_losses': 2,
            'total_losses_recorded': 3
        }
        
        self.mock_portfolio.loss_manager = self.mock_loss_manager
        
        self.mock_config = {
            'market': {
                'trading_hours': {
                    'timezone': 'America/Sao_Paulo'
                }
            },
            'taxes': {
                'swing_trade': 0.15,
                'day_trade': 0.20,
                'exemption_limit': 20000,
                'irrf_swing_rate': 0.00005,
                'irrf_day_rate': 0.01
            }
        }
    
    @patch('performance_metrics.yaml.safe_load')
    def test_calculate_tax_aware_returns(self, mock_yaml_load):
        """Test tax-aware returns calculation."""
        mock_yaml_load.return_value = self.mock_config
        
        performance_metrics = PerformanceMetrics(self.mock_portfolio)
        tax_metrics = TaxAwareMetrics(performance_metrics)
        
        # Create sample portfolio values
        portfolio_values = [100000, 105000, 110000]
        
        tax_aware_metrics = tax_metrics.calculate_tax_aware_returns(portfolio_values)
        
        # Verify tax-aware metrics
        expected_metrics = [
            'pre_tax_return', 'after_tax_return', 'tax_adjusted_annualized',
            'tax_efficiency', 'tax_drag', 'total_taxes_paid', 'effective_tax_rate',
            'loss_carryforward_balance', 'loss_utilization_rate',
            'swing_trade_taxes', 'day_trade_taxes', 'tax_exemption_utilized'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, tax_aware_metrics)
            self.assertIsInstance(tax_aware_metrics[metric], (int, float))
    
    @patch('performance_metrics.yaml.safe_load')
    def test_calculate_regulatory_compliance_metrics(self, mock_yaml_load):
        """Test regulatory compliance metrics calculation."""
        mock_yaml_load.return_value = self.mock_config
        
        performance_metrics = PerformanceMetrics(self.mock_portfolio)
        tax_metrics = TaxAwareMetrics(performance_metrics)
        
        compliance_metrics = tax_metrics.calculate_regulatory_compliance_metrics()
        
        # Verify compliance metrics
        self.assertEqual(compliance_metrics['regulatory_framework'], 'brazilian_2025')
        self.assertTrue(compliance_metrics['cvm_compliance'])
        self.assertTrue(compliance_metrics['receita_federal_compliance'])
        
        # Verify loss carryforward compliance
        loss_compliance = compliance_metrics['loss_carryforward_compliance']
        self.assertTrue(loss_compliance['perpetual_carryforward'])
        self.assertEqual(loss_compliance['max_offset_percentage'], 0.30)
        self.assertTrue(loss_compliance['capital_gains_only'])
        
        # Verify tax compliance
        tax_compliance = compliance_metrics['tax_compliance']
        self.assertEqual(tax_compliance['swing_trade_rate'], 0.15)
        self.assertEqual(tax_compliance['day_trade_rate'], 0.20)
        self.assertEqual(tax_compliance['exemption_limit'], 20000)


class TestComprehensivePerformanceAnalysis(unittest.TestCase):
    """Test suite for ComprehensivePerformanceAnalysis class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_portfolio = Mock()
        self.mock_portfolio.initial_cash = 100000.0
        self.mock_portfolio.total_value = 110000.0
        self.mock_portfolio.total_taxes = 1500.0
        self.mock_portfolio.total_commission = 500.0
        self.mock_portfolio.trade_history = [
            {'pnl': 1000, 'trade_type': 'swing_trade', 'taxes': 150},
            {'pnl': -500, 'trade_type': 'swing_trade', 'taxes': 0},
            {'pnl': 2000, 'trade_type': 'day_trade', 'taxes': 400},
            {'pnl': -300, 'trade_type': 'day_trade', 'taxes': 0}
        ]
        
        self.mock_loss_manager = Mock()
        self.mock_loss_manager.get_total_loss_balance.return_value = 800.0
        self.mock_loss_manager.get_loss_summary.return_value = {
            'total_loss_balance': 800.0,
            'assets_with_losses': 2,
            'total_losses_recorded': 3
        }
        
        self.mock_portfolio.loss_manager = self.mock_loss_manager
        self.mock_portfolio.get_portfolio_summary.return_value = {
            'total_value': 110000.0,
            'cash': 50000.0,
            'positions': {}
        }
        
        self.mock_config = {
            'market': {
                'trading_hours': {
                    'timezone': 'America/Sao_Paulo'
                }
            },
            'taxes': {
                'swing_trade': 0.15,
                'day_trade': 0.20,
                'exemption_limit': 20000,
                'irrf_swing_rate': 0.00005,
                'irrf_day_rate': 0.01
            }
        }
    
    @patch('performance_metrics.yaml.safe_load')
    def test_run_comprehensive_analysis(self, mock_yaml_load):
        """Test comprehensive performance analysis."""
        mock_yaml_load.return_value = self.mock_config
        
        analysis = ComprehensivePerformanceAnalysis(self.mock_portfolio)
        
        # Create sample data
        portfolio_values = [100000, 101000, 102500, 101800, 103200, 
                           104000, 103500, 105000, 104800, 106000]
        daily_returns = [0.01, 0.015, -0.007, 0.014, 0.008, -0.005, 0.014, -0.002, 0.011]
        
        results = analysis.run_comprehensive_analysis(portfolio_values, daily_returns)
        
        # Verify comprehensive analysis structure
        expected_sections = [
            'returns_analysis', 'risk_analysis', 'tax_analysis', 
            'trade_analysis', 'tax_aware_analysis', 'regulatory_compliance',
            'portfolio_summary', 'analysis_timestamp', 'market_parameters'
        ]
        
        for section in expected_sections:
            self.assertIn(section, results)
        
        # Verify market parameters
        market_params = results['market_parameters']
        self.assertEqual(market_params['trading_days_per_year'], 252)
        self.assertEqual(market_params['risk_free_rate'], 0.1175)
        self.assertEqual(market_params['market_timezone'], 'America/Sao_Paulo')
    
    @patch('performance_metrics.yaml.safe_load')
    @patch('builtins.open')
    def test_generate_performance_report(self, mock_open, mock_yaml_load):
        """Test performance report generation."""
        mock_yaml_load.return_value = self.mock_config
        
        analysis = ComprehensivePerformanceAnalysis(self.mock_portfolio)
        
        # Create sample analysis results
        analysis_results = {
            'returns_analysis': {
                'total_return': 0.06,
                'annualized_return': 0.15,
                'logarithmic_return': 0.058,
                'trading_days': 252
            },
            'risk_analysis': {
                'sharpe_ratio': 1.2,
                'sortino_ratio': 1.5,
                'calmar_ratio': 2.0,
                'max_drawdown': -0.05,
                'volatility': 0.12,
                'var_95': -0.02
            },
            'tax_analysis': {
                'total_taxes_paid': 1500.0,
                'tax_efficiency': 0.85,
                'effective_tax_rate': 0.15
            },
            'trade_analysis': {
                'total_trades': 100,
                'win_rate': 0.6,
                'profit_factor': 1.5,
                'average_win': 1000.0,
                'average_loss': -500.0
            },
            'regulatory_compliance': {
                'regulatory_framework': 'brazilian_2025',
                'cvm_compliance': True,
                'receita_federal_compliance': True
            },
            'analysis_timestamp': '2025-01-15T10:30:00'
        }
        
        # Test report generation
        analysis.generate_performance_report(analysis_results, 'test_report.txt')
        
        # Verify file was opened for writing
        mock_open.assert_called_with('test_report.txt', 'w')
    
    @patch('performance_metrics.yaml.safe_load')
    @patch('performance_metrics.plt')
    def test_plot_performance_charts(self, mock_plt, mock_yaml_load):
        """Test performance charts generation."""
        mock_yaml_load.return_value = self.mock_config
        
        analysis = ComprehensivePerformanceAnalysis(self.mock_portfolio)
        
        # Create sample data
        portfolio_values = [100000, 101000, 102500, 101800, 103200, 
                           104000, 103500, 105000, 104800, 106000]
        daily_returns = [0.01, 0.015, -0.007, 0.014, 0.008, -0.005, 0.014, -0.002, 0.011]
        
        # Test chart generation
        analysis.plot_performance_charts(portfolio_values, daily_returns, 'test_charts.png')
        
        # Verify matplotlib was called
        mock_plt.subplots.assert_called()
        mock_plt.savefig.assert_called_with('test_charts.png', dpi=300, bbox_inches='tight')
        mock_plt.close.assert_called()


class TestIntegrationWithExistingSystem(unittest.TestCase):
    """Test integration with existing quant_b3_backtest system components."""
    
    def setUp(self):
        """Set up test fixtures for integration testing."""
        self.mock_config = {
            'market': {
                'trading_hours': {
                    'timezone': 'America/Sao_Paulo'
                }
            },
            'taxes': {
                'swing_trade': 0.15,
                'day_trade': 0.20,
                'exemption_limit': 20000,
                'irrf_swing_rate': 0.00005,
                'irrf_day_rate': 0.01
            }
        }
    
    @patch('performance_metrics.yaml.safe_load')
    def test_integration_with_portfolio(self, mock_yaml_load):
        """Test integration with EnhancedPortfolio."""
        mock_yaml_load.return_value = self.mock_config
        
        # Create a realistic mock portfolio
        mock_portfolio = Mock()
        mock_portfolio.initial_cash = 100000.0
        mock_portfolio.total_value = 110000.0
        mock_portfolio.total_taxes = 1500.0
        mock_portfolio.total_commission = 500.0
        mock_portfolio.trade_history = [
            {
                'ticker': 'PETR4',
                'pnl': 1000,
                'trade_type': 'swing_trade',
                'taxes': 150,
                'entry_date': '2025-01-01',
                'exit_date': '2025-01-05'
            },
            {
                'ticker': 'VALE3',
                'pnl': -500,
                'trade_type': 'swing_trade',
                'taxes': 0,
                'entry_date': '2025-01-02',
                'exit_date': '2025-01-06'
            }
        ]
        
        # Create mock loss manager
        mock_loss_manager = Mock()
        mock_loss_manager.get_total_loss_balance.return_value = 800.0
        mock_loss_manager.get_loss_summary.return_value = {
            'total_loss_balance': 800.0,
            'assets_with_losses': 2,
            'total_losses_recorded': 3
        }
        mock_portfolio.loss_manager = mock_loss_manager
        
        # Test performance metrics integration
        performance_metrics = PerformanceMetrics(mock_portfolio)
        
        # Test returns calculation
        portfolio_values = [100000, 101000, 102500, 101800, 103200, 104000]
        returns_metrics = performance_metrics.calculate_returns(
            portfolio_values, datetime.now(), datetime.now()
        )
        
        self.assertIsInstance(returns_metrics, ReturnsMetrics)
        self.assertGreater(returns_metrics.total_return, 0)
        self.assertEqual(returns_metrics.trading_days, 5)
        
        # Test tax metrics integration
        tax_metrics = performance_metrics.calculate_tax_metrics()
        self.assertEqual(tax_metrics.total_taxes_paid, 1500.0)
        self.assertEqual(tax_metrics.swing_trade_taxes, 150.0)
        
        # Test trade metrics integration
        trade_metrics = performance_metrics.calculate_trade_metrics()
        self.assertEqual(trade_metrics.total_trades, 2)
        self.assertEqual(trade_metrics.winning_trades, 1)
        self.assertEqual(trade_metrics.losing_trades, 1)
    
    @patch('performance_metrics.yaml.safe_load')
    def test_brazilian_market_compliance(self, mock_yaml_load):
        """Test Brazilian market compliance features."""
        mock_yaml_load.return_value = self.mock_config
        
        mock_portfolio = Mock()
        mock_portfolio.initial_cash = 100000.0
        mock_portfolio.total_taxes = 1500.0
        mock_portfolio.trade_history = []
        
        mock_loss_manager = Mock()
        mock_loss_manager.get_total_loss_balance.return_value = 800.0
        mock_loss_manager.get_loss_summary.return_value = {
            'total_loss_balance': 800.0,
            'assets_with_losses': 2,
            'total_losses_recorded': 3
        }
        mock_portfolio.loss_manager = mock_loss_manager
        
        # Test Brazilian market constants
        performance_metrics = PerformanceMetrics(mock_portfolio)
        self.assertEqual(performance_metrics.TRADING_DAYS_PER_YEAR, 252)
        self.assertEqual(performance_metrics.RISK_FREE_RATE, 0.1175)
        
        # Test tax-aware metrics
        tax_metrics = TaxAwareMetrics(performance_metrics)
        compliance_metrics = tax_metrics.calculate_regulatory_compliance_metrics()
        
        # Verify Brazilian compliance
        self.assertEqual(compliance_metrics['regulatory_framework'], 'brazilian_2025')
        self.assertTrue(compliance_metrics['cvm_compliance'])
        self.assertTrue(compliance_metrics['receita_federal_compliance'])
        
        # Verify Brazilian tax rules
        loss_compliance = compliance_metrics['loss_carryforward_compliance']
        self.assertTrue(loss_compliance['perpetual_carryforward'])
        self.assertEqual(loss_compliance['max_offset_percentage'], 0.30)
        self.assertTrue(loss_compliance['capital_gains_only'])


def run_performance_metrics_demo():
    """Run a demonstration of the performance metrics module."""
    print("=" * 80)
    print("BRAZILIAN B3 QUANT BACKTEST - PERFORMANCE METRICS DEMONSTRATION")
    print("=" * 80)
    
    # Create sample data for demonstration
    portfolio_values = [100000, 101000, 102500, 101800, 103200, 104000, 
                       103500, 105000, 104800, 106000, 107500, 108000]
    daily_returns = [0.01, 0.015, -0.007, 0.014, 0.008, -0.005, 
                    0.014, -0.002, 0.011, 0.014, 0.005]
    
    # Create mock portfolio
    mock_portfolio = Mock()
    mock_portfolio.initial_cash = 100000.0
    mock_portfolio.total_value = 108000.0
    mock_portfolio.total_taxes = 1500.0
    mock_portfolio.total_commission = 500.0
    mock_portfolio.trade_history = [
        {'pnl': 1000, 'trade_type': 'swing_trade', 'taxes': 150},
        {'pnl': -500, 'trade_type': 'swing_trade', 'taxes': 0},
        {'pnl': 2000, 'trade_type': 'day_trade', 'taxes': 400},
        {'pnl': -300, 'trade_type': 'day_trade', 'taxes': 0}
    ]
    
    mock_loss_manager = Mock()
    mock_loss_manager.get_total_loss_balance.return_value = 800.0
    mock_loss_manager.get_loss_summary.return_value = {
        'total_loss_balance': 800.0,
        'assets_with_losses': 2,
        'total_losses_recorded': 3
    }
    mock_portfolio.loss_manager = mock_loss_manager
    mock_portfolio.get_portfolio_summary.return_value = {
        'total_value': 108000.0,
        'cash': 50000.0,
        'positions': {}
    }
    
    mock_config = {
        'market': {
            'trading_hours': {
                'timezone': 'America/Sao_Paulo'
            }
        },
        'taxes': {
            'swing_trade': 0.15,
            'day_trade': 0.20,
            'exemption_limit': 20000,
            'irrf_swing_rate': 0.00005,
            'irrf_day_rate': 0.01
        }
    }
    
    with patch('performance_metrics.yaml.safe_load') as mock_yaml_load:
        mock_yaml_load.return_value = mock_config
        
        # Demonstrate comprehensive analysis
        analysis = ComprehensivePerformanceAnalysis(mock_portfolio)
        results = analysis.run_comprehensive_analysis(portfolio_values, daily_returns)
        
        print("\nCOMPREHENSIVE PERFORMANCE ANALYSIS RESULTS:")
        print("-" * 50)
        
        # Returns Analysis
        returns = results['returns_analysis']
        print(f"Total Return: {returns['total_return']:.4f} ({returns['total_return']*100:.2f}%)")
        print(f"Annualized Return: {returns['annualized_return']:.4f} ({returns['annualized_return']*100:.2f}%)")
        print(f"Trading Days: {returns['trading_days']}")
        
        # Risk Analysis
        risk = results['risk_analysis']
        print(f"\nSharpe Ratio: {risk['sharpe_ratio']:.4f}")
        print(f"Sortino Ratio: {risk['sortino_ratio']:.4f}")
        print(f"Maximum Drawdown: {risk['max_drawdown']:.4f} ({risk['max_drawdown']*100:.2f}%)")
        print(f"Volatility: {risk['volatility']:.4f} ({risk['volatility']*100:.2f}%)")
        
        # Tax Analysis
        tax = results['tax_analysis']
        print(f"\nTotal Taxes Paid: R$ {tax['total_taxes_paid']:.2f}")
        print(f"Tax Efficiency: {tax['tax_efficiency']:.4f} ({tax['tax_efficiency']*100:.2f}%)")
        
        # Trade Analysis
        trade = results['trade_analysis']
        print(f"\nTotal Trades: {trade['total_trades']}")
        print(f"Win Rate: {trade['win_rate']:.4f} ({trade['win_rate']*100:.2f}%)")
        print(f"Profit Factor: {trade['profit_factor']:.4f}")
        
        # Regulatory Compliance
        compliance = results['regulatory_compliance']
        print(f"\nRegulatory Framework: {compliance['regulatory_framework']}")
        print(f"CVM Compliance: {compliance['cvm_compliance']}")
        print(f"Receita Federal Compliance: {compliance['receita_federal_compliance']}")
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)


if __name__ == "__main__":
    # Run the demonstration
    run_performance_metrics_demo()
    
    # Run unit tests
    print("\nRunning unit tests...")
    unittest.main(verbosity=2) 