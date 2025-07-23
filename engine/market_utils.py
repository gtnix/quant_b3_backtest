"""
Brazilian Market Utilities - Essential Functions Only

This module provides only the essential utilities for Brazilian market (B3) compliance:
- Price tick normalization (R$ 0.01 increments)
- Lot size validation (round lots vs odd lots)
- Basic order validation

Author: Your Name
Date: 2024
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logger = logging.getLogger(__name__)


class TradeType(Enum):
    """Enumeration for Brazilian market trade types."""
    DAY_TRADE = "day_trade"
    SWING_TRADE = "swing_trade"
    AUTO = "auto"


class SignalType(Enum):
    """Enumeration for trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class OrderType(Enum):
    """Order types for Brazilian market."""
    MARKET = "market"
    LIMIT = "limit"


class LotType(Enum):
    """Lot types for Brazilian market."""
    ROUND_LOT = "round_lot"      # Multiples of 100 shares
    ODD_LOT = "odd_lot"          # Less than 100 or non-multiples of 100


@dataclass
class OrderValidation:
    """Result of order validation."""
    is_valid: bool
    normalized_price: float
    normalized_quantity: int
    lot_type: LotType
    is_fractional: bool
    validation_messages: list[str]
    original_price: float
    original_quantity: int


class BrazilianMarketUtils:
    """
    Brazilian market utilities for price ticks and lot sizes.
    
    Brazilian market conventions:
    - Price ticks: R$ 0.01 (two decimal places)
    - Round lots: Multiples of 100 shares
    - Odd lots: Less than 100 shares or non-multiples of 100
    """
    
    # Brazilian market constants
    TICK_SIZE = 0.01
    ROUND_LOT_SIZE = 100
    MIN_QUANTITY = 1
    
    def __init__(self, tick_size: float = None, round_lot_size: int = None):
        """
        Initialize market utilities.
        
        Args:
            tick_size: Price tick size (default: 0.01 for Brazilian market)
            round_lot_size: Round lot size (default: 100 for Brazilian market)
        """
        self.tick_size = tick_size if tick_size is not None else self.TICK_SIZE
        self.round_lot_size = round_lot_size if round_lot_size is not None else self.ROUND_LOT_SIZE
        
        # Validate constants
        if self.tick_size <= 0:
            raise ValueError("Tick size must be positive")
        if self.round_lot_size <= 0:
            raise ValueError("Round lot size must be positive")
    
    def normalize_price_tick(self, price: float) -> float:
        """
        Normalize price to the nearest tick size.
        
        Brazilian market: All prices must be rounded to the nearest R$ 0.01.
        
        Args:
            price: Original price
            
        Returns:
            Normalized price rounded to nearest tick
            
        Examples:
            >>> utils = BrazilianMarketUtils()
            >>> utils.normalize_price_tick(12.3456)
            12.35
            >>> utils.normalize_price_tick(12.344)
            12.34
        """
        if price <= 0:
            raise ValueError("Price must be positive")
        
        # Round to nearest tick using proper rounding
        normalized = round(price / self.tick_size) * self.tick_size
        
        # Ensure we have exactly 2 decimal places
        return round(normalized, 2)
    
    def validate_lot_size(self, quantity: int) -> Tuple[bool, LotType, bool]:
        """
        Validate and classify lot size.
        
        Args:
            quantity: Number of shares
            
        Returns:
            Tuple of (is_valid, lot_type, is_fractional)
            
        Examples:   
            >>> utils = BrazilianMarketUtils()
            >>> utils.validate_lot_size(100)
            (True, <LotType.ROUND_LOT: 'round_lot'>, False)
            >>> utils.validate_lot_size(150)
            (True, <LotType.ODD_LOT: 'odd_lot'>, True)
            >>> utils.validate_lot_size(50)
            (True, <LotType.ODD_LOT: 'odd_lot'>, True)
        """
        if quantity < self.MIN_QUANTITY:
            return False, LotType.ODD_LOT, True
        
        if quantity % self.round_lot_size == 0:
            return True, LotType.ROUND_LOT, False
        else:
            return True, LotType.ODD_LOT, True
    
    def normalize_quantity(self, quantity: int) -> int:
        """
        Normalize quantity to valid lot size.
        
        For round lots: ensures quantity is multiple of 100
        For odd lots: returns quantity as-is (no normalization)
        
        Args:
            quantity: Original quantity
            
        Returns:
            Normalized quantity
        """
        if quantity < self.MIN_QUANTITY:
            return 0
        
        # Check if this is a round lot (multiple of round_lot_size)
        if quantity % self.round_lot_size == 0:
            # It's already a round lot, no normalization needed
            return quantity
        else:
            # It's an odd lot, no normalization
            return quantity
    
    def validate_order(self, price: float, quantity: int, 
                      order_type: OrderType = OrderType.MARKET,
                      allow_fractional: bool = True) -> OrderValidation:
        """
        Comprehensive order validation for Brazilian market.
        
        Args:
            price: Order price
            quantity: Order quantity
            order_type: Type of order
            allow_fractional: Whether to allow fractional (odd) lots
            
        Returns:
            OrderValidation object with validation results
        """
        validation_messages = []
        original_price = price
        original_quantity = quantity
        
        # Validate price
        if price <= 0:
            validation_messages.append("Price must be positive")
            return OrderValidation(
                is_valid=False,
                normalized_price=0.0,
                normalized_quantity=0,
                lot_type=LotType.ODD_LOT,
                is_fractional=True,
                validation_messages=validation_messages,
                original_price=original_price,
                original_quantity=original_quantity
            )
        
        # Normalize price
        normalized_price = self.normalize_price_tick(price)
        if abs(normalized_price - price) > 1e-6:
            validation_messages.append(f"Price {price} normalized to {normalized_price}")
        
        # Validate quantity
        if quantity < self.MIN_QUANTITY:
            validation_messages.append(f"Quantity {quantity} is below minimum {self.MIN_QUANTITY}")
            return OrderValidation(
                is_valid=False,
                normalized_price=normalized_price,
                normalized_quantity=0,
                lot_type=LotType.ODD_LOT,
                is_fractional=True,
                validation_messages=validation_messages,
                original_price=original_price,
                original_quantity=original_quantity
            )
        
        # Check lot size
        is_valid_lot, lot_type, is_fractional = self.validate_lot_size(quantity)
        
        if not is_valid_lot:
            validation_messages.append(f"Invalid lot size: {quantity}")
            return OrderValidation(
                is_valid=False,
                normalized_price=normalized_price,
                normalized_quantity=0,
                lot_type=lot_type,
                is_fractional=is_fractional,
                validation_messages=validation_messages,
                original_price=original_price,
                original_quantity=original_quantity
            )
        
        # Check fractional lot restrictions
        if is_fractional and not allow_fractional:
            validation_messages.append(f"Fractional lots not allowed: {quantity}")
            return OrderValidation(
                is_valid=False,
                normalized_price=normalized_price,
                normalized_quantity=quantity,
                lot_type=lot_type,
                is_fractional=is_fractional,
                validation_messages=validation_messages,
                original_price=original_price,
                original_quantity=original_quantity
            )
        
        # Normalize quantity if needed (only for round lots)
        normalized_quantity = quantity
        if lot_type == LotType.ROUND_LOT:
            # For round lots, ensure it's a multiple of round_lot_size
            normalized_quantity = (quantity // self.round_lot_size) * self.round_lot_size
            if normalized_quantity != quantity:
                validation_messages.append(f"Quantity {quantity} normalized to {normalized_quantity}")
        # For odd lots, keep original quantity (no normalization)
        
        # Add lot type information
        if lot_type == LotType.ROUND_LOT:
            validation_messages.append(f"Round lot order: {quantity} shares")
        else:
            validation_messages.append(f"Odd lot order: {quantity} shares")
        
        return OrderValidation(
            is_valid=True,
            normalized_price=normalized_price,
            normalized_quantity=normalized_quantity,
            lot_type=lot_type,
            is_fractional=is_fractional,
            validation_messages=validation_messages,
            original_price=original_price,
            original_quantity=original_quantity
        )
    
    def calculate_order_value(self, price: float, quantity: int) -> float:
        """
        Calculate order value with normalized price.
        
        Args:
            price: Order price
            quantity: Order quantity
            
        Returns:
            Order value in BRL
        """
        normalized_price = self.normalize_price_tick(price)
        return normalized_price * quantity
    
    def get_market_info(self) -> Dict[str, Any]:
        """
        Get Brazilian market information.
        
        Returns:
            Dictionary with market constants and rules
        """
        return {
            'tick_size': self.tick_size,
            'round_lot_size': self.round_lot_size,
            'min_quantity': self.MIN_QUANTITY,
            'price_precision': 2,
            'currency': 'BRL',
            'market': 'B3',
            'description': 'Brazilian market conventions for price ticks and lot sizes'
        }


# Convenience functions for easy access
def create_market_utils(config: Dict[str, Any] = None) -> BrazilianMarketUtils:
    """
    Create BrazilianMarketUtils instance with optional configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        BrazilianMarketUtils instance
    """
    if config is None:
        return BrazilianMarketUtils()
    
    market_config = config.get('market', {})
    return BrazilianMarketUtils(
        tick_size=market_config.get('tick_size', 0.01),
        round_lot_size=market_config.get('round_lot_size', 100)
    )


# Example usage and testing
if __name__ == "__main__":
    # Test market utilities
    utils = BrazilianMarketUtils()
    
    print("=== Brazilian Market Utils Test ===")
    
    # Test price normalization
    test_prices = [12.3456, 12.344, 15.0, 0.99]
    for price in test_prices:
        normalized = utils.normalize_price_tick(price)
        print(f"Price {price} -> {normalized}")
    
    # Test lot validation
    test_quantities = [100, 150, 50, 200, 75]
    for qty in test_quantities:
        is_valid, lot_type, is_fractional = utils.validate_lot_size(qty)
        print(f"Quantity {qty}: valid={is_valid}, type={lot_type.value}, fractional={is_fractional}")
    
    # Test order validation
    validation = utils.validate_order(price=12.3456, quantity=150)
    print(f"\nOrder validation: {validation}")
    
    print("\nMarket utilities module ready for use!") 