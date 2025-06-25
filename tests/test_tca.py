"""
Comprehensive Test Suite for Transaction Cost Analysis (TCA) Module

This test suite covers:
- Transaction cost calculation accuracy
- Cost breakdown validation
- Configuration integration
- Error handling and validation
- Edge cases and boundary conditions
- Integration with portfolio manager

Author: Your Name
Date: 2024
"""

import unittest
import tempfile
import os
import yaml
from unittest.mock import patch, MagicMock
from decimal import Decimal

import sys
sys.path.append('..')

from engine.tca import TransactionCostAnalyzer, CostBreakdown


class TestTransactionCostAnalyzer(unittest.TestCase):
    """Comprehensive tests for Transaction Cost Analyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary config file for testing
        self.config_data = {
            'market': {
                'trading_hours': {
                    'open': '10:00',
                    'close': '16:55',
                    'timezone': 'America/Sao_Paulo'
                },
                'costs': {
                    'brokerage_fee': 0.0001,      # 0.01%
                    'min_brokerage': 5.00,        # R$ 5.00
                    'emolument': 0.00005,         # 0.005%
                    'settlement_day_trade': 0.00018,    # 0.018%
                    'settlement_swing_trade': 0.00025,  # 0.025%
                    'iss_rate': 0.05              # 5%
                }
            }
        }
        
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(self.config_data, self.temp_config)
        self.temp_config.close()
        
        # Initialize TCA with test config
        self.tca = TransactionCostAnalyzer(self.temp_config.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_config.name):
            os.unlink(self.temp_config.name)
    
    def test_initialization(self):
        """Test proper initialization of TCA."""
        self.assertIsNotNone(self.tca.config)
        self.assertIsNotNone(self.tca.cost_params)
        self.assertEqual(self.tca.cost_params['brokerage_fee'], 0.0001)
        self.assertEqual(self.tca.cost_params['min_brokerage'], 5.00)
        self.assertEqual(self.tca.cost_params['emolument'], 0.00005)
    
    def test_cost_parameter_validation(self):
        """Test cost parameter validation."""
        # Test valid parameters (should not raise)
        try:
            self.tca._validate_cost_parameters()
        except ValueError:
            self.fail("Valid cost parameters should not raise ValueError")
        
        # Test invalid brokerage fee
        original_brokerage = self.tca.cost_params['brokerage_fee']
        self.tca.cost_params['brokerage_fee'] = -0.001
        
        with self.assertRaises(ValueError):
            self.tca._validate_cost_parameters()
        
        # Restore original value
        self.tca.cost_params['brokerage_fee'] = original_brokerage
    
    def test_calculate_costs_swing_trade_buy(self):
        """Test cost calculation for swing trade buy order."""
        order_value = 10000.0  # R$ 10,000
        
        breakdown = self.tca.calculate_costs(
            order_value=order_value,
            is_buy=True,
            trade_type="swing_trade"
        )
        
        # Expected calculations:
        # Brokerage: max(10000 * 0.0001, 5.00) = max(1.00, 5.00) = 5.00
        # Emolument: 10000 * 0.00005 = 0.50
        # Settlement: 10000 * 0.00025 = 2.50
        # ISS: 5.00 * 0.05 = 0.25
        # Total: 5.00 + 0.50 + 2.50 + 0.25 = 8.25
        
        self.assertIsInstance(breakdown, CostBreakdown)
        self.assertEqual(breakdown.brokerage_fee, 5.00)
        self.assertTrue(breakdown.min_brokerage_applied)
        self.assertEqual(breakdown.emolument, 0.50)
        self.assertEqual(breakdown.settlement_fee, 2.50)
        self.assertEqual(breakdown.iss_fee, 0.25)
        self.assertEqual(breakdown.total_costs, 8.25)
        self.assertAlmostEqual(breakdown.cost_percentage, 0.0825, places=4)
    
    def test_calculate_costs_swing_trade_sell(self):
        """Test cost calculation for swing trade sell order."""
        order_value = 10000.0  # R$ 10,000
        
        breakdown = self.tca.calculate_costs(
            order_value=order_value,
            is_buy=False,
            trade_type="swing_trade"
        )
        
        # Costs should be the same for buy/sell (only direction differs)
        self.assertEqual(breakdown.brokerage_fee, 5.00)
        self.assertTrue(breakdown.min_brokerage_applied)
        self.assertEqual(breakdown.emolument, 0.50)
        self.assertEqual(breakdown.settlement_fee, 2.50)
        self.assertEqual(breakdown.iss_fee, 0.25)
        self.assertEqual(breakdown.total_costs, 8.25)
    
    def test_calculate_costs_day_trade(self):
        """Test cost calculation for day trade."""
        order_value = 10000.0  # R$ 10,000
        
        breakdown = self.tca.calculate_costs(
            order_value=order_value,
            is_buy=True,
            trade_type="day_trade"
        )
        
        # Expected calculations:
        # Brokerage: max(10000 * 0.0001, 5.00) = max(1.00, 5.00) = 5.00
        # Emolument: 10000 * 0.00005 = 0.50
        # Settlement: 10000 * 0.00018 = 1.80 (different from swing trade)
        # ISS: 5.00 * 0.05 = 0.25
        # Total: 5.00 + 0.50 + 1.80 + 0.25 = 7.55
        
        self.assertEqual(breakdown.brokerage_fee, 5.00)
        self.assertEqual(breakdown.emolument, 0.50)
        self.assertEqual(breakdown.settlement_fee, 1.80)  # Different from swing trade
        self.assertEqual(breakdown.iss_fee, 0.25)
        self.assertEqual(breakdown.total_costs, 7.55)
    
    def test_minimum_brokerage_enforcement(self):
        """Test minimum brokerage fee enforcement."""
        # Test with small order where brokerage would be below minimum
        small_order = 1000.0  # R$ 1,000
        
        breakdown = self.tca.calculate_costs(
            order_value=small_order,
            is_buy=True,
            trade_type="swing_trade"
        )
        
        # Brokerage should be minimum (5.00) not calculated value (0.10)
        self.assertEqual(breakdown.brokerage_fee, 5.00)
        self.assertTrue(breakdown.min_brokerage_applied)
        
        # Test with large order where brokerage exceeds minimum
        large_order = 100000.0  # R$ 100,000
        
        breakdown = self.tca.calculate_costs(
            order_value=large_order,
            is_buy=True,
            trade_type="swing_trade"
        )
        
        # Brokerage should be calculated value (10.00) not minimum
        expected_brokerage = large_order * 0.0001
        self.assertEqual(breakdown.brokerage_fee, expected_brokerage)
        self.assertFalse(breakdown.min_brokerage_applied)
    
    def test_input_validation(self):
        """Test input validation for calculate_costs method."""
        # Test invalid order value
        with self.assertRaises(ValueError):
            self.tca.calculate_costs(order_value=0, is_buy=True, trade_type="swing_trade")
        
        with self.assertRaises(ValueError):
            self.tca.calculate_costs(order_value=-1000, is_buy=True, trade_type="swing_trade")
        
        # Test invalid trade type
        with self.assertRaises(ValueError):
            self.tca.calculate_costs(order_value=1000, is_buy=True, trade_type="invalid_type")
    
    def test_get_cost_summary(self):
        """Test cost summary generation."""
        order_value = 10000.0
        
        summary = self.tca.get_cost_summary(
            order_value=order_value,
            is_buy=True,
            trade_type="swing_trade"
        )
        
        self.assertIn('order_value', summary)
        self.assertIn('is_buy', summary)
        self.assertIn('trade_type', summary)
        self.assertIn('costs', summary)
        self.assertIn('net_amount', summary)
        self.assertIn('calculation_timestamp', summary)
        
        self.assertEqual(summary['order_value'], order_value)
        self.assertTrue(summary['is_buy'])
        self.assertEqual(summary['trade_type'], 'swing_trade')
        self.assertEqual(summary['costs']['total_costs'], 8.25)
        
        # Net amount for buy should be order_value + costs
        self.assertEqual(summary['net_amount'], order_value + 8.25)
    
    def test_compare_costs(self):
        """Test cost comparison across trade types."""
        order_value = 10000.0
        
        comparison = self.tca.compare_costs(order_value)
        
        self.assertIn('day_trade', comparison)
        self.assertIn('swing_trade', comparison)
        
        # Check structure
        for trade_type in comparison:
            self.assertIn('buy', comparison[trade_type])
            self.assertIn('sell', comparison[trade_type])
            self.assertIn('round_trip', comparison[trade_type])
        
        # Day trade should have lower settlement fees
        day_trade_round_trip = comparison['day_trade']['round_trip']['total_costs']
        swing_trade_round_trip = comparison['swing_trade']['round_trip']['total_costs']
        
        # Day trade should be cheaper due to lower settlement fees
        self.assertLess(day_trade_round_trip, swing_trade_round_trip)
    
    def test_get_cost_parameters(self):
        """Test retrieval of cost parameters."""
        params = self.tca.get_cost_parameters()
        
        self.assertIsInstance(params, dict)
        self.assertIn('brokerage_fee', params)
        self.assertIn('min_brokerage', params)
        self.assertIn('emolument', params)
        self.assertIn('settlement_day_trade', params)
        self.assertIn('settlement_swing_trade', params)
        self.assertIn('iss_rate', params)
        
        # Should be a copy, not reference
        self.assertIsNot(params, self.tca.cost_params)
    
    def test_update_cost_parameters(self):
        """Test dynamic cost parameter updates."""
        original_brokerage = self.tca.cost_params['brokerage_fee']
        
        # Update with valid parameters
        new_params = {'brokerage_fee': 0.0002}  # 0.02%
        self.tca.update_cost_parameters(new_params)
        
        self.assertEqual(self.tca.cost_params['brokerage_fee'], 0.0002)
        
        # Test with invalid parameters (should restore original)
        invalid_params = {'brokerage_fee': -0.001}
        
        with self.assertRaises(ValueError):
            self.tca.update_cost_parameters(invalid_params)
        
        # Should restore original value
        self.assertEqual(self.tca.cost_params['brokerage_fee'], 0.0002)
    
    def test_config_file_not_found(self):
        """Test handling of missing configuration file."""
        with self.assertRaises(FileNotFoundError):
            TransactionCostAnalyzer("nonexistent_config.yaml")
    
    def test_invalid_config_structure(self):
        """Test handling of invalid configuration structure."""
        # Create config without market.costs section
        invalid_config = {
            'market': {
                'trading_hours': {
                    'open': '10:00',
                    'close': '16:55'
                }
                # Missing 'costs' section
            }
        }
        
        temp_invalid_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(invalid_config, temp_invalid_config)
        temp_invalid_config.close()
        
        try:
            with self.assertRaises(ValueError):
                TransactionCostAnalyzer(temp_invalid_config.name)
        finally:
            if os.path.exists(temp_invalid_config.name):
                os.unlink(temp_invalid_config.name)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test very small order
        tiny_order = 0.01
        breakdown = self.tca.calculate_costs(tiny_order, True, "swing_trade")
        
        # Should still apply minimum brokerage
        self.assertEqual(breakdown.brokerage_fee, 5.00)
        self.assertTrue(breakdown.min_brokerage_applied)
        
        # Test very large order
        huge_order = 10000000.0  # R$ 10 million
        breakdown = self.tca.calculate_costs(huge_order, True, "swing_trade")
        
        # Should use calculated brokerage, not minimum
        expected_brokerage = huge_order * 0.0001
        self.assertEqual(breakdown.brokerage_fee, expected_brokerage)
        self.assertFalse(breakdown.min_brokerage_applied)
    
    def test_cost_breakdown_dataclass(self):
        """Test CostBreakdown dataclass functionality."""
        breakdown = CostBreakdown(
            brokerage_fee=5.00,
            min_brokerage_applied=True,
            emolument=0.50,
            settlement_fee=2.50,
            iss_fee=0.25,
            total_costs=8.25,
            cost_percentage=0.0825
        )
        
        self.assertEqual(breakdown.brokerage_fee, 5.00)
        self.assertTrue(breakdown.min_brokerage_applied)
        self.assertEqual(breakdown.emolument, 0.50)
        self.assertEqual(breakdown.settlement_fee, 2.50)
        self.assertEqual(breakdown.iss_fee, 0.25)
        self.assertEqual(breakdown.total_costs, 8.25)
        self.assertEqual(breakdown.cost_percentage, 0.0825)


class TestTCAIntegration(unittest.TestCase):
    """Integration tests for TCA with portfolio manager."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Create test config
        self.config_data = {
            'market': {
                'trading_hours': {
                    'open': '10:00',
                    'close': '16:55',
                    'timezone': 'America/Sao_Paulo'
                },
                'costs': {
                    'brokerage_fee': 0.0001,
                    'min_brokerage': 5.00,
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
                'holiday_calendar': 'b3',
                'max_retry_attempts': 3,
                'auto_process_settlements': True
            },
            'loss_carryforward': {
                'max_tracking_years': 5,
                'global_loss_limit': None,
                'asset_specific_tracking': True,
                'temporal_management': True,
                'audit_trail_enabled': True,
                'auto_prune_old_losses': True,
                'partial_application': True,
                'performance_caching': True
            }
        }
        
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(self.config_data, self.temp_config)
        self.temp_config.close()
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        if os.path.exists(self.temp_config.name):
            os.unlink(self.temp_config.name)
    
    def test_tca_portfolio_integration(self):
        """Test TCA integration with portfolio manager."""
        from engine.portfolio import EnhancedPortfolio
        
        # Initialize portfolio with TCA
        portfolio = EnhancedPortfolio(self.temp_config.name)
        
        # Verify TCA is initialized
        self.assertIsNotNone(portfolio.tca)
        self.assertIsInstance(portfolio.tca, TransactionCostAnalyzer)
        
        # Test buy order with TCA
        from datetime import datetime
        trade_date = datetime.now()
        
        success = portfolio.buy(
            ticker="VALE3",
            quantity=100,
            price=50.0,
            trade_date=trade_date,
            trade_type="swing_trade"
        )
        
        self.assertTrue(success)
        
        # Verify costs were calculated using TCA
        trade_record = portfolio.trade_history[-1]
        self.assertIn('costs', trade_record)
        self.assertIn('min_brokerage_applied', trade_record['costs'])
        self.assertIn('cost_percentage', trade_record['costs'])
        
        # Verify cost breakdown matches TCA expectations
        expected_brokerage = max(100 * 50.0 * 0.0001, 5.00)
        self.assertEqual(trade_record['costs']['brokerage_fee'], expected_brokerage)


def run_performance_benchmarks():
    """Run performance benchmarks for TCA module."""
    import time
    
    print("=== TCA Performance Benchmarks ===")
    
    # Initialize TCA
    config_data = {
        'market': {
            'costs': {
                'brokerage_fee': 0.0001,
                'min_brokerage': 5.00,
                'emolument': 0.00005,
                'settlement_day_trade': 0.00018,
                'settlement_swing_trade': 0.00025,
                'iss_rate': 0.05
            }
        }
    }
    
    temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(config_data, temp_config)
    temp_config.close()
    
    try:
        tca = TransactionCostAnalyzer(temp_config.name)
        
        # Benchmark cost calculations
        iterations = 10000
        order_values = [1000.0, 10000.0, 100000.0, 1000000.0]
        
        for order_value in order_values:
            start_time = time.time()
            
            for _ in range(iterations):
                tca.calculate_costs(order_value, True, "swing_trade")
            
            end_time = time.time()
            elapsed = end_time - start_time
            rate = iterations / elapsed
            
            print(f"Order Value R$ {order_value:,.0f}: {rate:.0f} calculations/second")
        
    finally:
        if os.path.exists(temp_config.name):
            os.unlink(temp_config.name)


if __name__ == '__main__':
    # Run unit tests
    unittest.main(verbosity=2)
    
    # Run performance benchmarks
    run_performance_benchmarks() 