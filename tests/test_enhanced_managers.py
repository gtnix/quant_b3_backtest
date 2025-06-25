"""
Comprehensive Test Suite for Enhanced Loss Carryforward and Settlement Managers

This test suite covers:
- Advanced loss carryforward functionality with temporal management
- Sophisticated T+2 settlement tracking with business day handling
- Comprehensive error handling and validation
- Performance optimization and caching
- Regulatory compliance features
- Edge cases and boundary conditions

Author: Your Name
Date: 2024
"""

import unittest
import tempfile
import os
import json
from datetime import datetime, timedelta, date
from unittest.mock import patch, MagicMock

import sys
sys.path.append('..')

from engine.loss_manager import EnhancedLossCarryforwardManager, LossRecord, LossApplication
from engine.settlement_manager import AdvancedSettlementManager, SettlementItem, TradeType


class TestEnhancedLossCarryforwardManager(unittest.TestCase):
    """Comprehensive tests for Enhanced Loss Carryforward Manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loss_manager = EnhancedLossCarryforwardManager(max_tracking_years=3)
        self.base_date = datetime.now()
    
    def test_initialization(self):
        """Test proper initialization of the loss manager."""
        self.assertEqual(self.loss_manager.max_tracking_years, 3)
        self.assertEqual(self.loss_manager.global_loss_balance, 0.0)
        self.assertEqual(len(self.loss_manager.asset_losses), 0)
        self.assertEqual(len(self.loss_manager.loss_history), 0)
    
    def test_record_loss(self):
        """Test recording a loss with comprehensive tracking."""
        loss_amount = 1000.0
        trade_date = self.base_date
        
        taxable_profit = self.loss_manager.record_trade_result(
            ticker="VALE3",
            trade_profit=-loss_amount,
            trade_date=trade_date,
            trade_type="swing_trade",
            trade_id="VALE3_001",
            description="Test loss"
        )
        
        # Should return 0 taxable profit for losses
        self.assertEqual(taxable_profit, 0.0)
        
        # Check asset-specific loss tracking
        asset_loss = self.loss_manager.get_asset_loss_balance("VALE3")
        self.assertEqual(asset_loss, loss_amount)
        
        # Check global loss balance
        total_loss = self.loss_manager.get_total_loss_balance()
        self.assertEqual(total_loss, loss_amount)
        
        # Check loss history
        self.assertEqual(len(self.loss_manager.loss_history), 1)
        loss_record = self.loss_manager.loss_history[0]
        self.assertEqual(loss_record.amount, loss_amount)
        self.assertEqual(loss_record.asset, "VALE3")
        self.assertEqual(loss_record.trade_type, "swing_trade")
    
    def test_apply_loss_carryforward(self):
        """Test applying loss carryforward against profits."""
        # First, record a loss
        self.loss_manager.record_trade_result(
            ticker="VALE3",
            trade_profit=-1000.0,
            trade_date=self.base_date,
            trade_type="swing_trade"
        )
        
        # Then, apply against a profit
        profit_amount = 1500.0
        taxable_profit = self.loss_manager.calculate_taxable_amount(
            current_gain=profit_amount,
            current_date=self.base_date + timedelta(days=1),
            ticker="VALE3"
        )
        
        # Should apply loss and return remaining taxable amount
        expected_taxable = profit_amount - 1000.0
        self.assertEqual(taxable_profit, expected_taxable)
        
        # Check that loss was applied
        remaining_loss = self.loss_manager.get_asset_loss_balance("VALE3")
        self.assertEqual(remaining_loss, 0.0)
    
    def test_partial_loss_application(self):
        """Test partial application of losses."""
        # Record multiple losses
        self.loss_manager.record_trade_result("VALE3", -500.0, self.base_date, "swing_trade")
        self.loss_manager.record_trade_result("VALE3", -300.0, self.base_date + timedelta(days=1), "swing_trade")
        
        # Apply against smaller profit
        profit_amount = 600.0
        taxable_profit = self.loss_manager.calculate_taxable_amount(
            current_gain=profit_amount,
            current_date=self.base_date + timedelta(days=2),
            ticker="VALE3"
        )
        
        # Should apply 600 out of 800 total loss
        self.assertEqual(taxable_profit, 0.0)
        
        # Check remaining loss
        remaining_loss = self.loss_manager.get_asset_loss_balance("VALE3")
        self.assertEqual(remaining_loss, 200.0)  # 800 - 600
    
    def test_temporal_loss_management(self):
        """Test temporal loss management and pruning."""
        # Create loss manager with short tracking period
        short_tracking_manager = EnhancedLossCarryforwardManager(max_tracking_years=1)
        
        # Record loss
        short_tracking_manager.record_trade_result(
            ticker="VALE3",
            trade_profit=-1000.0,
            trade_date=self.base_date,
            trade_type="swing_trade"
        )
        
        # Simulate time passing (more than max_tracking_years)
        future_date = self.base_date + timedelta(days=400)  # More than 1 year
        
        # Record another trade to trigger pruning
        short_tracking_manager.record_trade_result(
            ticker="PETR4",
            trade_profit=-500.0,
            trade_date=future_date,
            trade_type="swing_trade"
        )
        
        # Old loss should be pruned
        remaining_loss = short_tracking_manager.get_asset_loss_balance("VALE3")
        self.assertEqual(remaining_loss, 0.0)
        
        # New loss should remain
        new_loss = short_tracking_manager.get_asset_loss_balance("PETR4")
        self.assertEqual(new_loss, 500.0)
    
    def test_global_loss_application(self):
        """Test global loss application when asset-specific losses are exhausted."""
        # Record losses for different assets
        self.loss_manager.record_trade_result("VALE3", -500.0, self.base_date, "swing_trade")
        self.loss_manager.record_trade_result("PETR4", -300.0, self.base_date + timedelta(days=1), "swing_trade")
        
        # Apply against profit for a different asset
        profit_amount = 1000.0
        taxable_profit = self.loss_manager.calculate_taxable_amount(
            current_gain=profit_amount,
            current_date=self.base_date + timedelta(days=2),
            ticker="ITUB4"  # Different asset
        )
        
        # Should apply global losses (800 total)
        expected_taxable = profit_amount - 800.0
        self.assertEqual(taxable_profit, expected_taxable)
        
        # Check global loss balance is reduced
        global_loss = self.loss_manager.get_total_loss_balance()
        self.assertEqual(global_loss, 0.0)
    
    def test_error_handling(self):
        """Test comprehensive error handling."""
        # Test invalid ticker
        with self.assertRaises(ValueError):
            self.loss_manager.record_trade_result("", -100.0, self.base_date, "swing_trade")
        
        # Test invalid profit amount
        with self.assertRaises(ValueError):
            self.loss_manager.record_trade_result("VALE3", "invalid", self.base_date, "swing_trade")
        
        # Test invalid date
        with self.assertRaises(ValueError):
            self.loss_manager.record_trade_result("VALE3", -100.0, "invalid_date", "swing_trade")
    
    def test_audit_trail_export(self):
        """Test audit trail export functionality."""
        # Record some losses
        self.loss_manager.record_trade_result("VALE3", -1000.0, self.base_date, "swing_trade")
        self.loss_manager.record_trade_result("PETR4", -500.0, self.base_date + timedelta(days=1), "day_trade")
        
        # Export audit trail
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            audit_file = f.name
        
        try:
            self.loss_manager.export_audit_trail(audit_file)
            
            # Verify file was created and contains expected data
            self.assertTrue(os.path.exists(audit_file))
            
            with open(audit_file, 'r') as f:
                audit_data = json.load(f)
            
            self.assertIn('loss_records', audit_data)
            self.assertIn('application_history', audit_data)
            self.assertIn('summary', audit_data)
            
            # Check loss records
            loss_records = audit_data['loss_records']
            self.assertEqual(len(loss_records), 2)
            
            # Check summary
            summary = audit_data['summary']
            self.assertEqual(summary['total_cumulative_loss'], 1500.0)
            self.assertEqual(summary['total_losses_recorded'], 2)
            
        finally:
            # Clean up
            if os.path.exists(audit_file):
                os.unlink(audit_file)
    
    def test_performance_caching(self):
        """Test performance optimization with caching."""
        # Record multiple losses for the same asset
        for i in range(10):
            self.loss_manager.record_trade_result(
                f"ASSET{i}",
                -100.0,
                self.base_date + timedelta(days=i),
                "swing_trade"
            )
        
        # Multiple calls should use cache
        start_time = datetime.now()
        for _ in range(100):
            self.loss_manager.get_asset_loss_balance("ASSET5")
        end_time = datetime.now()
        
        # Should be fast due to caching
        duration = (end_time - start_time).total_seconds()
        self.assertLess(duration, 1.0)  # Should complete in less than 1 second
    
    def test_loss_summary(self):
        """Test comprehensive loss summary generation."""
        # Record losses
        self.loss_manager.record_trade_result("VALE3", -1000.0, self.base_date, "swing_trade")
        self.loss_manager.record_trade_result("PETR4", -500.0, self.base_date + timedelta(days=1), "day_trade")
        
        summary = self.loss_manager.get_loss_summary()
        
        self.assertEqual(summary['total_cumulative_loss'], 1500.0)
        self.assertEqual(summary['total_losses_recorded'], 2)
        self.assertEqual(summary['assets_with_losses'], 2)
        self.assertEqual(summary['max_tracking_years'], 3)
        
        # Check asset-specific losses
        asset_losses = summary['asset_losses']
        self.assertEqual(asset_losses['VALE3'], 1000.0)
        self.assertEqual(asset_losses['PETR4'], 500.0)


class TestAdvancedSettlementManager(unittest.TestCase):
    """Comprehensive tests for Advanced Settlement Manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.settlement_manager = AdvancedSettlementManager(
            initial_cash=10000.0,
            settlement_days=2,
            market_timezone='America/Sao_Paulo'
        )
        self.base_date = datetime.now()
    
    def test_initialization(self):
        """Test proper initialization of the settlement manager."""
        self.assertEqual(self.settlement_manager.settlement_days, 2)
        self.assertEqual(self.settlement_manager.settled_cash, 10000.0)
        self.assertEqual(self.settlement_manager.total_cash, 10000.0)
        self.assertEqual(len(self.settlement_manager.settlement_queue), 0)
    
    def test_schedule_buy_trade(self):
        """Test scheduling a buy trade with comprehensive validation."""
        trade_amount = 5000.0
        trade_date = self.base_date
        
        self.settlement_manager.schedule_trade(
            trade_date=trade_date,
            amount=trade_amount,
            trade_type='BUY',
            ticker="VALE3",
            trade_id="VALE3_001",
            description="Buy VALE3"
        )
        
        # Check queue
        self.assertEqual(len(self.settlement_manager.settlement_queue), 1)
        
        # Check total cash (should be reduced immediately)
        self.assertEqual(self.settlement_manager.total_cash, 10000.0 - trade_amount)
        
        # Check settled cash (should remain unchanged until settlement)
        self.assertEqual(self.settlement_manager.settled_cash, 10000.0)
    
    def test_schedule_sell_trade(self):
        """Test scheduling a sell trade with comprehensive validation."""
        trade_amount = 3000.0
        trade_date = self.base_date
        
        self.settlement_manager.schedule_trade(
            trade_date=trade_date,
            amount=trade_amount,
            trade_type='SELL',
            ticker="PETR4",
            trade_id="PETR4_001",
            description="Sell PETR4"
        )
        
        # Check queue
        self.assertEqual(len(self.settlement_manager.settlement_queue), 1)
        
        # Check total cash (should be increased immediately)
        self.assertEqual(self.settlement_manager.total_cash, 10000.0 + trade_amount)
        
        # Check settled cash (should remain unchanged until settlement)
        self.assertEqual(self.settlement_manager.settled_cash, 10000.0)
    
    def test_business_day_calculation(self):
        """Test business day calculation with holiday handling."""
        # Test weekend handling
        friday = date(2024, 1, 5)  # Friday
        monday = date(2024, 1, 8)  # Monday
        
        # T+2 from Friday should be Tuesday (skip weekend)
        settlement_date = self.settlement_manager._calculate_t2_settlement_date(friday)
        expected_date = date(2024, 1, 9)  # Tuesday
        self.assertEqual(settlement_date, expected_date)
        
        # T+2 from Monday should be Wednesday
        settlement_date = self.settlement_manager._calculate_t2_settlement_date(monday)
        expected_date = date(2024, 1, 10)  # Wednesday
        self.assertEqual(settlement_date, expected_date)
    
    def test_settlement_processing(self):
        """Test settlement processing with comprehensive validation."""
        # Schedule trades
        trade_date = self.base_date
        settlement_date = self.settlement_manager._calculate_t2_settlement_date(trade_date.date())
        
        self.settlement_manager.schedule_trade(
            trade_date=trade_date,
            amount=5000.0,
            trade_type='BUY',
            ticker="VALE3"
        )
        
        self.settlement_manager.schedule_trade(
            trade_date=trade_date,
            amount=3000.0,
            trade_type='SELL',
            ticker="PETR4"
        )
        
        # Process settlements before due date (should not process)
        before_settlement = settlement_date - timedelta(days=1)
        newly_available = self.settlement_manager.process_settlements(before_settlement)
        self.assertEqual(newly_available, 0.0)
        self.assertEqual(len(self.settlement_manager.settlement_queue), 2)
        
        # Process settlements on due date
        newly_available = self.settlement_manager.process_settlements(settlement_date)
        self.assertEqual(len(self.settlement_manager.settlement_queue), 0)
        
        # Check settled cash
        expected_settled_cash = 10000.0 - 5000.0 + 3000.0  # Initial - buy + sell
        self.assertEqual(self.settlement_manager.settled_cash, expected_settled_cash)
    
    def test_available_cash_calculation(self):
        """Test available cash calculation with strict and non-strict modes."""
        # Schedule a buy trade
        self.settlement_manager.schedule_trade(
            trade_date=self.base_date,
            amount=5000.0,
            trade_type='BUY',
            ticker="VALE3"
        )
        
        # Test strict mode (only settled cash)
        available_strict = self.settlement_manager.get_available_cash(
            self.base_date.date(), strict_mode=True
        )
        self.assertEqual(available_strict, 10000.0)  # Only settled cash
        
        # Test non-strict mode (including unsettled)
        available_non_strict = self.settlement_manager.get_available_cash(
            self.base_date.date(), strict_mode=False
        )
        self.assertEqual(available_non_strict, 5000.0)  # Total cash after buy
    
    def test_error_handling(self):
        """Test comprehensive error handling."""
        # Test invalid trade date
        with self.assertRaises(ValueError):
            self.settlement_manager.schedule_trade(
                trade_date="invalid_date",
                amount=1000.0,
                trade_type='BUY'
            )
        
        # Test invalid amount
        with self.assertRaises(ValueError):
            self.settlement_manager.schedule_trade(
                trade_date=self.base_date,
                amount=-1000.0,  # Negative amount
                trade_type='BUY'
            )
        
        # Test invalid trade type
        with self.assertRaises(ValueError):
            self.settlement_manager.schedule_trade(
                trade_date=self.base_date,
                amount=1000.0,
                trade_type='INVALID'
            )
    
    def test_failed_settlements(self):
        """Test handling of failed settlements."""
        # Schedule a trade
        self.settlement_manager.schedule_trade(
            trade_date=self.base_date,
            amount=5000.0,
            trade_type='BUY',
            ticker="VALE3"
        )
        
        # Mock a settlement failure
        with patch.object(self.settlement_manager, '_calculate_t2_settlement_date') as mock_calc:
            mock_calc.side_effect = Exception("Settlement calculation failed")
            
            # This should handle the error gracefully
            with self.assertRaises(Exception):
                self.settlement_manager.schedule_trade(
                    trade_date=self.base_date + timedelta(days=1),
                    amount=1000.0,
                    trade_type='SELL',
                    ticker="PETR4"
                )
    
    def test_audit_trail_export(self):
        """Test audit trail export functionality."""
        # Schedule some trades
        self.settlement_manager.schedule_trade(
            trade_date=self.base_date,
            amount=5000.0,
            trade_type='BUY',
            ticker="VALE3"
        )
        
        self.settlement_manager.schedule_trade(
            trade_date=self.base_date + timedelta(days=1),
            amount=3000.0,
            trade_type='SELL',
            ticker="PETR4"
        )
        
        # Export audit trail
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            audit_file = f.name
        
        try:
            self.settlement_manager.export_audit_trail(audit_file)
            
            # Verify file was created and contains expected data
            self.assertTrue(os.path.exists(audit_file))
            
            with open(audit_file, 'r') as f:
                audit_data = json.load(f)
            
            self.assertIn('settlement_history', audit_data)
            self.assertIn('failed_settlements', audit_data)
            self.assertIn('pending_settlements', audit_data)
            self.assertIn('summary', audit_data)
            
            # Check pending settlements
            pending_settlements = audit_data['pending_settlements']
            self.assertEqual(len(pending_settlements), 2)
            
            # Check summary
            summary = audit_data['summary']
            self.assertEqual(summary['pending_settlements'], 2)
            self.assertEqual(summary['settlement_days'], 2)
            
        finally:
            # Clean up
            if os.path.exists(audit_file):
                os.unlink(audit_file)
    
    def test_settlement_summary(self):
        """Test comprehensive settlement summary generation."""
        # Schedule trades
        self.settlement_manager.schedule_trade(
            trade_date=self.base_date,
            amount=5000.0,
            trade_type='BUY',
            ticker="VALE3"
        )
        
        self.settlement_manager.schedule_trade(
            trade_date=self.base_date + timedelta(days=1),
            amount=3000.0,
            trade_type='SELL',
            ticker="PETR4"
        )
        
        summary = self.settlement_manager.get_settlement_summary()
        
        self.assertEqual(summary['settled_cash'], 10000.0)
        self.assertEqual(summary['total_cash'], 10000.0 - 5000.0 + 3000.0)
        self.assertEqual(summary['pending_settlements'], 2)
        self.assertEqual(summary['pending_buys'], 5000.0)
        self.assertEqual(summary['pending_sells'], 3000.0)
        self.assertEqual(summary['settlement_days'], 2)
    
    def test_performance_optimization(self):
        """Test performance optimization with caching."""
        # Test business day cache
        test_date = date(2024, 1, 5)  # Friday
        
        # First call should calculate
        start_time = datetime.now()
        is_business_day = self.settlement_manager._is_business_day(test_date)
        first_call_time = (datetime.now() - start_time).total_seconds()
        
        # Second call should use cache
        start_time = datetime.now()
        is_business_day_cached = self.settlement_manager._is_business_day(test_date)
        second_call_time = (datetime.now() - start_time).total_seconds()
        
        # Cached call should be faster
        self.assertLess(second_call_time, first_call_time)
        self.assertEqual(is_business_day, is_business_day_cached)


class TestIntegration(unittest.TestCase):
    """Integration tests for both managers working together."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loss_manager = EnhancedLossCarryforwardManager(max_tracking_years=3)
        self.settlement_manager = AdvancedSettlementManager(initial_cash=10000.0)
        self.base_date = datetime.now()
    
    def test_complete_trade_cycle(self):
        """Test complete trade cycle with both managers."""
        # Buy trade
        self.settlement_manager.schedule_trade(
            trade_date=self.base_date,
            amount=5000.0,
            trade_type='BUY',
            ticker="VALE3"
        )
        
        # Sell trade at loss
        self.loss_manager.record_trade_result(
            ticker="VALE3",
            trade_profit=-500.0,
            trade_date=self.base_date + timedelta(days=1),
            trade_type="swing_trade"
        )
        
        # Schedule sell settlement
        self.settlement_manager.schedule_trade(
            trade_date=self.base_date + timedelta(days=1),
            amount=4500.0,  # 5000 - 500 loss
            trade_type='SELL',
            ticker="VALE3"
        )
        
        # Process settlements
        settlement_date = self.settlement_manager._calculate_t2_settlement_date(
            (self.base_date + timedelta(days=1)).date()
        )
        self.settlement_manager.process_settlements(settlement_date)
        
        # Verify results
        loss_balance = self.loss_manager.get_total_loss_balance()
        self.assertEqual(loss_balance, 500.0)
        
        settled_cash = self.settlement_manager.settled_cash
        self.assertEqual(settled_cash, 10000.0 - 5000.0 + 4500.0)
    
    def test_loss_carryforward_with_settlement(self):
        """Test loss carryforward applied to new trade with settlement."""
        # Record initial loss
        self.loss_manager.record_trade_result(
            ticker="VALE3",
            trade_profit=-1000.0,
            trade_date=self.base_date,
            trade_type="swing_trade"
        )
        
        # Buy new position
        self.settlement_manager.schedule_trade(
            trade_date=self.base_date + timedelta(days=1),
            amount=3000.0,
            trade_type='BUY',
            ticker="PETR4"
        )
        
        # Sell at profit (should apply loss carryforward)
        profit_amount = 1500.0
        taxable_profit = self.loss_manager.calculate_taxable_amount(
            current_gain=profit_amount,
            current_date=self.base_date + timedelta(days=2),
            ticker="PETR4"
        )
        
        # Should apply loss carryforward
        expected_taxable = profit_amount - 1000.0
        self.assertEqual(taxable_profit, expected_taxable)
        
        # Schedule sell settlement
        net_proceeds = 3000.0 + profit_amount - 500.0  # Buy amount + profit - loss carryforward
        self.settlement_manager.schedule_trade(
            trade_date=self.base_date + timedelta(days=2),
            amount=net_proceeds,
            trade_type='SELL',
            ticker="PETR4"
        )
        
        # Verify final state
        remaining_loss = self.loss_manager.get_total_loss_balance()
        self.assertEqual(remaining_loss, 0.0)  # All loss applied


def run_performance_benchmarks():
    """Run performance benchmarks for the enhanced managers."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARKS")
    print("="*60)
    
    # Loss Manager Performance
    print("\n--- Loss Manager Performance ---")
    loss_manager = EnhancedLossCarryforwardManager(max_tracking_years=5)
    
    start_time = datetime.now()
    for i in range(1000):
        loss_manager.record_trade_result(
            ticker=f"ASSET{i % 10}",
            trade_profit=-100.0,
            trade_date=datetime.now() + timedelta(days=i),
            trade_type="swing_trade"
        )
    end_time = datetime.now()
    
    loss_duration = (end_time - start_time).total_seconds()
    print(f"Recorded 1000 losses in {loss_duration:.3f} seconds")
    print(f"Average: {loss_duration/1000*1000:.2f} ms per loss")
    
    # Settlement Manager Performance
    print("\n--- Settlement Manager Performance ---")
    settlement_manager = AdvancedSettlementManager(initial_cash=100000.0)
    
    start_time = datetime.now()
    for i in range(1000):
        settlement_manager.schedule_trade(
            trade_date=datetime.now() + timedelta(days=i),
            amount=1000.0,
            trade_type='BUY' if i % 2 == 0 else 'SELL',
            ticker=f"ASSET{i % 10}"
        )
    end_time = datetime.now()
    
    settlement_duration = (end_time - start_time).total_seconds()
    print(f"Scheduled 1000 trades in {settlement_duration:.3f} seconds")
    print(f"Average: {settlement_duration/1000*1000:.2f} ms per trade")
    
    print("="*60)


if __name__ == '__main__':
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Run performance benchmarks
    run_performance_benchmarks() 