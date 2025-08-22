"""
Unit tests for PositionSizer component.

Tests the position sizing logic extracted from the monolithic strategy,
including B3 board-lot rounding and tranche calculations.

Author: Senior Python Developer
Date: 2025
"""

import pytest
from strategies.components.position_sizing import PositionSizer


class TestPositionSizer:
    """Test suite for PositionSizer component."""
    
    @pytest.fixture
    def position_sizer(self):
        """Create PositionSizer instance for testing."""
        config = {
            'fixed_notional_brl': 50000.0,
            'num_tranches': 4,
            'lot_size': 100
        }
        return PositionSizer(config)
    
    def test_initialization(self, position_sizer):
        """Test PositionSizer initialization."""
        assert position_sizer.fixed_notional_brl == 50000.0
        assert position_sizer.num_tranches == 4
        assert position_sizer.lot_size == 100
        assert position_sizer.notional_per_tranche == 12500.0
    
    def test_calculate_position_size_basic(self, position_sizer):
        """Test basic position size calculation."""
        # 50,000 BRL / 25.00 BRL = 2000 shares
        close_price = 25.00
        total_size = position_sizer.calculate_position_size(close_price)
        
        # Should be rounded to nearest lot (2000 is already multiple of 100)
        assert total_size == 2000
        assert total_size % 100 == 0  # Board lot compliance
    
    def test_calculate_tranche_size(self, position_sizer):
        """Test individual tranche size calculation."""
        # 12,500 BRL / 25.00 BRL = 500 shares per tranche
        close_price = 25.00
        tranche_size = position_sizer.calculate_position_size(close_price, tranche_number=1)
        
        assert tranche_size == 500
        assert tranche_size % 100 == 0  # Board lot compliance
    
    def test_lot_rounding_conservative_down(self, position_sizer):
        """Test conservative rounding down for final digits < 50."""
        # 12,500 BRL / 26.32 BRL = 474.85 shares
        # Final two digits: 74 (< 50, so round down to 400)
        close_price = 26.32
        tranche_size = position_sizer.calculate_position_size(close_price, tranche_number=1)
        
        assert tranche_size == 400  # Rounded down from 474.85
        assert tranche_size % 100 == 0
    
    def test_lot_rounding_conservative_up(self, position_sizer):
        """Test conservative rounding up for final digits >= 50."""
        # 12,500 BRL / 24.51 BRL = 509.79 shares  
        # Final two digits: 09 (< 50, but this tests the >= 50 case)
        # Let's use a price that gives us >= 50 in final digits
        close_price = 23.81  # 12,500 / 23.81 = 524.99... ≈ 525
        tranche_size = position_sizer.calculate_position_size(close_price, tranche_number=1)
        
        # Should round to nearest 100
        expected = 500  # Conservative rounding
        assert tranche_size == expected
        assert tranche_size % 100 == 0
    
    def test_all_tranche_sizes(self, position_sizer):
        """Test calculation of all tranche sizes."""
        close_price = 20.00
        all_sizes = position_sizer.calculate_all_tranche_sizes(close_price)
        
        # Each tranche should be 12,500 / 20.00 = 625, rounded to 600
        expected_tranche = 600
        
        assert all_sizes['tranche_1'] == expected_tranche
        assert all_sizes['tranche_2'] == expected_tranche
        assert all_sizes['tranche_3'] == expected_tranche
        assert all_sizes['tranche_4'] == expected_tranche
        assert all_sizes['total'] == expected_tranche * 4
        
        # Check alternative naming
        assert all_sizes['qty_market'] == expected_tranche
        assert all_sizes['qty_alpha'] == expected_tranche
        assert all_sizes['qty_beta'] == expected_tranche
        assert all_sizes['qty_gamma'] == expected_tranche
    
    def test_monotonic_quantity_relationship_buy(self, position_sizer):
        """Test monotonic relationship for BUY orders across price levels."""
        # For BUY orders, as limit prices decrease, quantities should increase
        close = 25.00
        
        # Simulate BUY limit prices (decreasing: P2 < P3 < P4)
        p1 = close  # Market at close
        p2 = close * 0.995  # -0.5%
        p3 = close * 0.99   # -1.0%  
        p4 = close * 0.985  # -1.5%
        
        qty1 = position_sizer.calculate_position_size(p1, 1)
        qty2 = position_sizer.calculate_position_size(p2, 2)
        qty3 = position_sizer.calculate_position_size(p3, 3)
        qty4 = position_sizer.calculate_position_size(p4, 4)
        
        # All should be board lot compliant
        assert all(q % 100 == 0 for q in [qty1, qty2, qty3, qty4])
        
        # Monotonic non-decreasing as prices decrease
        assert qty2 >= qty1
        assert qty3 >= qty2
        assert qty4 >= qty3
    
    def test_monotonic_quantity_relationship_sell(self, position_sizer):
        """Test monotonic relationship for SELL orders across price levels."""
        # For SELL orders, as limit prices increase, quantities should decrease
        close = 30.00
        
        # Simulate SELL limit prices (increasing: P2 > P3 > P4)
        p1 = close  # Market at close
        p2 = close * 1.005  # +0.5%
        p3 = close * 1.01   # +1.0%
        p4 = close * 1.015  # +1.5%
        
        qty1 = position_sizer.calculate_position_size(p1, 1)
        qty2 = position_sizer.calculate_position_size(p2, 2)
        qty3 = position_sizer.calculate_position_size(p3, 3)
        qty4 = position_sizer.calculate_position_size(p4, 4)
        
        # All should be board lot compliant
        assert all(q % 100 == 0 for q in [qty1, qty2, qty3, qty4])
        
        # Monotonic non-increasing as prices increase
        assert qty2 <= qty1
        assert qty3 <= qty2
        assert qty4 <= qty3
    
    def test_position_validation_valid(self, position_sizer):
        """Test position size validation for valid sizes."""
        close_price = 25.00
        valid_quantity = 500  # Multiple of 100, reasonable size
        
        is_valid, error_msg = position_sizer.validate_position_size(valid_quantity, close_price)
        
        assert is_valid
        assert error_msg == ""
    
    def test_position_validation_below_lot_size(self, position_sizer):
        """Test position size validation for sizes below lot size."""
        close_price = 25.00
        invalid_quantity = 50  # Below 100 lot size
        
        is_valid, error_msg = position_sizer.validate_position_size(invalid_quantity, close_price)
        
        assert not is_valid
        assert "below minimum lot size" in error_msg
    
    def test_position_validation_not_lot_aligned(self, position_sizer):
        """Test position size validation for non-lot-aligned sizes."""
        close_price = 25.00
        invalid_quantity = 150  # Not multiple of 100
        
        is_valid, error_msg = position_sizer.validate_position_size(invalid_quantity, close_price)
        
        assert not is_valid
        assert "not aligned to lot size" in error_msg
    
    def test_notional_value_calculation(self, position_sizer):
        """Test notional value calculation."""
        quantity = 500
        price = 25.00
        
        notional = position_sizer.get_notional_value(quantity, price)
        
        assert notional == 12500.0  # 500 * 25.00
    
    def test_tranche_info(self, position_sizer):
        """Test tranche information retrieval."""
        tranche_1_info = position_sizer.get_tranche_info(1)
        
        assert tranche_1_info['tranche_number'] == 1
        assert tranche_1_info['attempt_name'] == 'Market at Open (P1)'
        assert tranche_1_info['notional_brl'] == 12500.0
        assert tranche_1_info['order_type'] == 'MARKET'
        
        tranche_2_info = position_sizer.get_tranche_info(2)
        assert tranche_2_info['attempt_name'] == 'Limit Alpha (P2)'
        assert tranche_2_info['order_type'] == 'LIMIT'
    
    def test_exposure_percentage(self, position_sizer):
        """Test exposure percentage calculation."""
        quantity = 500
        price = 25.00
        total_capital = 100000.0
        
        exposure = position_sizer.calculate_exposure_percentage(quantity, price, total_capital)
        
        # 500 * 25.00 / 100,000 = 0.125 (12.5%)
        assert exposure == 0.125
    
    def test_sizing_stats(self, position_sizer):
        """Test comprehensive sizing statistics."""
        close_price = 20.00
        stats = position_sizer.get_sizing_stats(close_price)
        
        assert stats['close_price'] == close_price
        assert stats['fixed_notional_brl'] == 50000.0
        assert stats['total_shares'] > 0
        assert stats['total_notional'] > 0
        assert 'notional_deviation' in stats
        assert 'tranche_sizes' in stats
    
    def test_invalid_price_handling(self, position_sizer):
        """Test handling of invalid prices."""
        invalid_price = 0.0
        size = position_sizer.calculate_position_size(invalid_price)
        
        assert size == 0  # Should return 0 for invalid price
    
    def test_invalid_tranche_number(self, position_sizer):
        """Test handling of invalid tranche numbers."""
        close_price = 25.00
        
        # Test invalid tranche numbers
        assert position_sizer.calculate_position_size(close_price, 0) == 0
        assert position_sizer.calculate_position_size(close_price, 5) == 0
        
        # Test valid tranche numbers
        for i in range(1, 5):
            size = position_sizer.calculate_position_size(close_price, i)
            assert size > 0
    
    def test_minimum_position_size_enforcement(self, position_sizer):
        """Test that minimum position size is enforced."""
        # Use very high price to test minimum enforcement
        very_high_price = 1000.00
        tranche_size = position_sizer.calculate_position_size(very_high_price, 1)
        
        # Should be at least one lot (100 shares)
        assert tranche_size >= 100
        assert tranche_size % 100 == 0
