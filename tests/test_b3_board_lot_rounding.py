"""
Comprehensive unit tests for B3 board-lot rounding logic.

This module tests the round_b3_board_lot helper function and calculate_tranche_quantity
function to ensure they comply with the B3 board-lot rounding rules and produce
the exact results specified in the requirements.

The acceptance criteria examples from the requirements document are used as ground truth.
"""

import pytest
from engine.utils import round_b3_board_lot, calculate_tranche_quantity


class TestRoundB3BoardLot:
    """Test the round_b3_board_lot function with acceptance criteria examples."""

    def test_acceptance_criteria_examples(self):
        """Test all acceptance criteria examples exactly as specified."""
        # Acceptance criteria test cases
        assert round_b3_board_lot(414.35) == 400
        assert round_b3_board_lot(383.43) == 400
        assert round_b3_board_lot(69.19) == 100
        assert round_b3_board_lot(770.17) == 800
        assert round_b3_board_lot(450) == 500
        assert round_b3_board_lot(449.99) == 400

    def test_boundary_conditions(self):
        """Test boundary conditions for the 50-remainder rule."""
        # Exactly 50 remainder should round up
        assert round_b3_board_lot(150.0) == 200
        assert round_b3_board_lot(250.0) == 300
        assert round_b3_board_lot(350.0) == 400
        
        # 49 remainder should round down
        assert round_b3_board_lot(149.0) == 100
        assert round_b3_board_lot(249.0) == 200
        assert round_b3_board_lot(349.0) == 300
        
        # 51 remainder should round up
        assert round_b3_board_lot(151.0) == 200
        assert round_b3_board_lot(251.0) == 300
        assert round_b3_board_lot(351.0) == 400

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Zero and negative values
        assert round_b3_board_lot(0) == 0
        assert round_b3_board_lot(-1) == 0
        assert round_b3_board_lot(-100.5) == 0
        
        # Very small positive values
        assert round_b3_board_lot(0.1) == 0
        assert round_b3_board_lot(1.0) == 0
        assert round_b3_board_lot(49.9) == 0
        assert round_b3_board_lot(50.0) == 100
        
        # Exact multiples of 100
        assert round_b3_board_lot(100.0) == 100
        assert round_b3_board_lot(200.0) == 200
        assert round_b3_board_lot(300.0) == 300

    def test_large_numbers(self):
        """Test with large numbers to ensure no overflow issues."""
        assert round_b3_board_lot(1000.0) == 1000
        assert round_b3_board_lot(1049.9) == 1000
        assert round_b3_board_lot(1050.0) == 1100
        assert round_b3_board_lot(9999.9) == 10000  # 9999 % 100 = 99, which is >= 50, so rounds up
        assert round_b3_board_lot(10050.0) == 10100

    def test_integer_inputs(self):
        """Test that integer inputs work correctly."""
        assert round_b3_board_lot(414) == 400
        assert round_b3_board_lot(383) == 400
        assert round_b3_board_lot(69) == 100
        assert round_b3_board_lot(770) == 800


class TestCalculateTrancheQuantity:
    """Test the calculate_tranche_quantity function with real examples."""

    def test_requirements_examples(self):
        """Test examples from the requirements document (B3_Board_Lot_Rounding_Examples)."""
        # PETR4 close 30.17 → 12,500 / 30.17 ≈ 414.35 → rounds to 400 shares
        assert calculate_tranche_quantity(12500.0, 30.17) == 400
        
        # PETR3 close 32.60 → 12,500 / 32.60 ≈ 383.43 → rounds to 400 shares  
        assert calculate_tranche_quantity(12500.0, 32.60) == 400
        
        # ITUB4 close 37.49 → 12,500 / 37.49 ≈ 333.42 → rounds to 300 shares
        assert calculate_tranche_quantity(12500.0, 37.49) == 300
        
        # NVDA close 180.65 → 12,500 / 180.65 ≈ 69.19 → rounds to 100 shares
        assert calculate_tranche_quantity(12500.0, 180.65) == 100
        
        # VALE3 close 53.32 → 12,500 / 53.32 ≈ 234.43 → rounds to 200 shares
        assert calculate_tranche_quantity(12500.0, 53.32) == 200
        
        # GGBR4 close 16.23 → 12,500 / 16.23 ≈ 770.17 → rounds to 800 shares
        assert calculate_tranche_quantity(12500.0, 16.23) == 800

    def test_different_notional_amounts(self):
        """Test with different notional amounts."""
        # Test with 25,000 BRL notional (double the standard tranche)
        # 25000/30.17 ≈ 828.7 → rounds to 800
        assert calculate_tranche_quantity(25000.0, 30.17) == 800
        
        # Test with 6,250 BRL notional (half the standard tranche)
        # 6250/30.17 ≈ 207.2 → rounds to 200
        assert calculate_tranche_quantity(6250.0, 30.17) == 200
        
        # Test with 50,000 BRL notional (full allocation)
        # 50000/30.17 ≈ 1657.4 → rounds to 1700 (due to B3 board-lot rounding)
        assert calculate_tranche_quantity(50000.0, 30.17) == 1700

    def test_edge_cases(self):
        """Test edge cases for calculate_tranche_quantity."""
        # Zero or negative prices
        assert calculate_tranche_quantity(12500.0, 0) == 0
        assert calculate_tranche_quantity(12500.0, -1.0) == 0
        
        # Zero notional
        assert calculate_tranche_quantity(0, 30.17) == 0
        assert calculate_tranche_quantity(-1000.0, 30.17) == 0
        
        # Very high prices (small quantities)
        assert calculate_tranche_quantity(12500.0, 1000.0) == 0   # 12.5 shares → rounds to 0
        assert calculate_tranche_quantity(12500.0, 500.0) == 0    # 25 shares → rounds to 0
        assert calculate_tranche_quantity(12500.0, 250.0) == 100  # 50 shares → rounds to 100
        
        # Very low prices (large quantities)
        assert calculate_tranche_quantity(12500.0, 1.0) == 12500  # 12,500 shares → rounds to 12,500
        assert calculate_tranche_quantity(12500.0, 0.50) == 25000 # 25,000 shares → rounds to 25,000

    def test_floating_point_precision(self):
        """Test floating point precision handling."""
        # Test with prices that might cause floating point precision issues
        price = 30.17
        raw_shares = 12500.0 / price
        expected_result = round_b3_board_lot(raw_shares)
        
        assert calculate_tranche_quantity(12500.0, price) == expected_result
        
        # Test with recurring decimals
        assert calculate_tranche_quantity(12500.0, 3.33) == 3800  # ~3753.75 → rounds to 3800


class TestIntegration:
    """Integration tests to ensure the functions work together correctly."""

    def test_consistency_between_functions(self):
        """Test that calculate_tranche_quantity is consistent with round_b3_board_lot."""
        test_cases = [
            (12500.0, 30.17),
            (12500.0, 32.60),
            (12500.0, 37.49),
            (12500.0, 180.65),
            (12500.0, 53.32),
            (12500.0, 16.23),
        ]
        
        for notional, price in test_cases:
            raw_shares = notional / price
            expected_quantity = round_b3_board_lot(raw_shares)
            actual_quantity = calculate_tranche_quantity(notional, price)
            
            assert actual_quantity == expected_quantity, (
                f"Inconsistency for notional={notional}, price={price}: "
                f"expected {expected_quantity}, got {actual_quantity}"
            )

    def test_total_portfolio_allocation(self):
        """Test that 4 tranches behave correctly with B3 board-lot rounding."""
        # Test specific cases where we know the expected behavior
        test_cases = [
            (30.17, 1600, 48272.0),  # PETR4: 4 * 400 * 30.17
            (32.60, 1600, 52160.0),  # PETR3: 4 * 400 * 32.60  
            (37.49, 1200, 44988.0),  # ITUB4: 4 * 300 * 37.49
            (53.32, 800, 42656.0),   # VALE3: 4 * 200 * 53.32
            (16.23, 3200, 51936.0),  # GGBR4: 4 * 800 * 16.23
        ]
        
        for price, expected_total_qty, expected_notional in test_cases:
            # Calculate quantity for one tranche
            tranche_qty = calculate_tranche_quantity(12500.0, price)
            
            # Calculate total for 4 tranches
            total_qty = tranche_qty * 4
            total_notional = total_qty * price
            
            # Verify expected quantities and notional values
            assert total_qty == expected_total_qty, (
                f"Expected {expected_total_qty} total shares for price {price}, got {total_qty}"
            )
            assert abs(total_notional - expected_notional) < 0.01, (
                f"Expected ~{expected_notional} notional for price {price}, got {total_notional}"
            )
        
        # Test extreme high price case (NVDA) separately
        # With very high prices, board-lot rounding can cause large notional deviations
        nvda_qty = calculate_tranche_quantity(12500.0, 180.65)
        assert nvda_qty == 100  # 69.19 rounds to 100
        nvda_total_notional = nvda_qty * 4 * 180.65
        assert nvda_total_notional == 72260.0  # This is expected - no constraint violated


if __name__ == "__main__":
    pytest.main([__file__])
