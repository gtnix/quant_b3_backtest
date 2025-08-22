"""
Position Sizing Component for FuzzyFajuto Strategy

Handles position sizing calculations, lot rounding, and B3 market constraints.
Extracted from the monolithic strategy class for better maintainability.

Author: Senior Python Developer
Date: 2025
"""

import logging
import math
from typing import Dict, Any, Optional, Tuple
from engine.logging_config import get_logger

logger = get_logger(__name__)


class PositionSizer:
    """
    Handles position sizing calculations for the FuzzyFajuto strategy.
    
    Responsibilities:
    - Fixed notional position sizing (50,000 BRL per symbol)
    - B3 board-lot rounding (100 shares minimum)
    - Tranche-based sizing (4 equal tranches)
    - Position size validation and constraints
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize position sizer with configuration."""
        self.config = config
        
        # Fixed configuration per strategy specification
        self.fixed_notional_brl = 50000.0  # Fixed 50k BRL per symbol per session
        self.num_tranches = 4              # 4 equal tranches per position
        self.lot_size = 100                # B3 board-lot size (100 shares)
        
        # Derived values
        self.notional_per_tranche = self.fixed_notional_brl / self.num_tranches  # 12,500 BRL per tranche
        
        logger.debug(f"PositionSizer initialized: {self.fixed_notional_brl} BRL total, "
                    f"{self.notional_per_tranche} BRL per tranche, {self.lot_size} share lots")
    
    def calculate_position_size(self, close_price: float, 
                              tranche_number: Optional[int] = None) -> int:
        """
        Calculate position size based on fixed notional and price.
        
        Args:
            close_price: Previous close price for sizing
            tranche_number: Optional tranche number (1-4), if None returns total size
            
        Returns:
            Position size in shares (rounded to lot size)
        """
        try:
            if close_price <= 0:
                logger.warning(f"Invalid close price for position sizing: {close_price}")
                return 0
            
            # Determine notional amount
            if tranche_number is not None:
                if not 1 <= tranche_number <= self.num_tranches:
                    logger.warning(f"Invalid tranche number: {tranche_number}")
                    return 0
                notional = self.notional_per_tranche
            else:
                notional = self.fixed_notional_brl
            
            # Calculate raw quantity
            raw_quantity = notional / close_price
            
            # Round to B3 board-lot size
            rounded_quantity = self._round_to_lot_size(raw_quantity)
            
            logger.debug(f"Position sizing: {notional} BRL @ {close_price:.2f} = "
                        f"{raw_quantity:.2f} shares -> {rounded_quantity} shares (lot-rounded)")
            
            return rounded_quantity
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def _round_to_lot_size(self, raw_quantity: float) -> int:
        """
        Round quantity to B3 board-lot size (100 shares).
        
        Uses conservative rounding: final two digits <50 → round down, ≥50 → round up.
        
        Args:
            raw_quantity: Raw quantity in shares
            
        Returns:
            Rounded quantity in board lots
        """
        try:
            if raw_quantity <= 0:
                return 0
            
            # Get the final two digits for rounding decision
            final_two_digits = int((raw_quantity * 100) % 10000) % 100
            
            # Conservative rounding logic per specification
            if final_two_digits < 50:
                # Round down to nearest 100
                rounded = math.floor(raw_quantity / self.lot_size) * self.lot_size
            else:
                # Round up to nearest 100
                rounded = math.ceil(raw_quantity / self.lot_size) * self.lot_size
            
            # Ensure minimum position size
            if rounded < self.lot_size:
                rounded = self.lot_size
            
            logger.debug(f"Lot rounding: {raw_quantity:.2f} -> {rounded} "
                        f"(final digits: {final_two_digits})")
            
            return int(rounded)
            
        except Exception as e:
            logger.error(f"Error in lot rounding: {e}")
            return self.lot_size  # Fallback to minimum lot size
    
    def calculate_all_tranche_sizes(self, close_price: float) -> Dict[str, int]:
        """
        Calculate position sizes for all tranches.
        
        Args:
            close_price: Previous close price for sizing
            
        Returns:
            Dictionary with tranche sizes
        """
        try:
            sizes = {}
            total_size = 0
            
            # Calculate each tranche
            for i in range(1, self.num_tranches + 1):
                tranche_size = self.calculate_position_size(close_price, i)
                sizes[f'tranche_{i}'] = tranche_size
                sizes[f'qty_tranche_{i}'] = tranche_size  # Alternative naming
                total_size += tranche_size
            
            # Add total and per-attempt quantities
            sizes['total'] = total_size
            sizes['qty_market'] = tranche_size  # P1 Market
            sizes['qty_alpha'] = tranche_size   # P2 Limit Alpha  
            sizes['qty_beta'] = tranche_size    # P3 Limit Beta
            sizes['qty_gamma'] = tranche_size   # P4 Limit Gamma
            
            logger.debug(f"All tranche sizes @ {close_price:.2f}: {sizes}")
            return sizes
            
        except Exception as e:
            logger.error(f"Error calculating all tranche sizes: {e}")
            return self._get_fallback_sizes()
    
    def _get_fallback_sizes(self) -> Dict[str, int]:
        """Get fallback sizes in case of calculation errors."""
        fallback_size = self.lot_size
        return {
            'tranche_1': fallback_size,
            'tranche_2': fallback_size, 
            'tranche_3': fallback_size,
            'tranche_4': fallback_size,
            'total': fallback_size * 4,
            'qty_market': fallback_size,
            'qty_alpha': fallback_size,
            'qty_beta': fallback_size,
            'qty_gamma': fallback_size,
            'qty_tranche_1': fallback_size,
            'qty_tranche_2': fallback_size,
            'qty_tranche_3': fallback_size,
            'qty_tranche_4': fallback_size
        }
    
    def validate_position_size(self, quantity: int, close_price: float) -> Tuple[bool, str]:
        """
        Validate a position size against strategy constraints.
        
        Args:
            quantity: Position size in shares
            close_price: Price for validation
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check minimum lot size
            if quantity < self.lot_size:
                return False, f"Position size {quantity} below minimum lot size {self.lot_size}"
            
            # Check lot size alignment
            if quantity % self.lot_size != 0:
                return False, f"Position size {quantity} not aligned to lot size {self.lot_size}"
            
            # Check maximum reasonable size (10x normal)
            max_reasonable_size = self.calculate_position_size(close_price) * 10
            if quantity > max_reasonable_size:
                return False, f"Position size {quantity} exceeds reasonable maximum {max_reasonable_size}"
            
            # Check notional value reasonableness
            notional_value = quantity * close_price
            max_notional = self.fixed_notional_brl * 2  # Allow 2x for flexibility
            if notional_value > max_notional:
                return False, f"Notional value {notional_value:.2f} exceeds maximum {max_notional:.2f}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def get_notional_value(self, quantity: int, price: float) -> float:
        """
        Calculate notional value of a position.
        
        Args:
            quantity: Position size in shares
            price: Price per share
            
        Returns:
            Notional value in BRL
        """
        return quantity * price
    
    def get_tranche_info(self, tranche_number: int) -> Dict[str, Any]:
        """
        Get information about a specific tranche.
        
        Args:
            tranche_number: Tranche number (1-4)
            
        Returns:
            Dictionary with tranche information
        """
        if not 1 <= tranche_number <= self.num_tranches:
            return {}
        
        attempt_names = {
            1: 'Market at Open (P1)',
            2: 'Limit Alpha (P2)', 
            3: 'Limit Beta (P3)',
            4: 'Limit Gamma (P4)'
        }
        
        return {
            'tranche_number': tranche_number,
            'attempt_name': attempt_names.get(tranche_number, f'Tranche {tranche_number}'),
            'notional_brl': self.notional_per_tranche,
            'order_type': 'MARKET' if tranche_number == 1 else 'LIMIT'
        }
    
    def calculate_exposure_percentage(self, quantity: int, price: float, 
                                   total_capital: float) -> float:
        """
        Calculate position exposure as percentage of total capital.
        
        Args:
            quantity: Position size in shares
            price: Price per share
            total_capital: Total available capital
            
        Returns:
            Exposure percentage (0.0 to 1.0)
        """
        try:
            if total_capital <= 0:
                return 0.0
            
            notional_value = self.get_notional_value(quantity, price)
            return notional_value / total_capital
            
        except Exception as e:
            logger.error(f"Error calculating exposure percentage: {e}")
            return 0.0
    
    def get_sizing_stats(self, close_price: float) -> Dict[str, Any]:
        """
        Get comprehensive sizing statistics for a given price.
        
        Args:
            close_price: Close price for calculations
            
        Returns:
            Dictionary with sizing statistics
        """
        try:
            # Calculate all sizes
            tranche_sizes = self.calculate_all_tranche_sizes(close_price)
            
            # Calculate statistics
            total_notional = self.get_notional_value(tranche_sizes['total'], close_price)
            
            return {
                'close_price': close_price,
                'fixed_notional_brl': self.fixed_notional_brl,
                'notional_per_tranche': self.notional_per_tranche,
                'lot_size': self.lot_size,
                'num_tranches': self.num_tranches,
                'tranche_sizes': tranche_sizes,
                'total_shares': tranche_sizes['total'],
                'total_notional': total_notional,
                'notional_deviation': abs(total_notional - self.fixed_notional_brl),
                'notional_deviation_pct': abs(total_notional - self.fixed_notional_brl) / self.fixed_notional_brl
            }
            
        except Exception as e:
            logger.error(f"Error calculating sizing stats: {e}")
            return {}
    
    def get_config(self) -> Dict[str, Any]:
        """Get position sizer configuration."""
        return {
            'fixed_notional_brl': self.fixed_notional_brl,
            'num_tranches': self.num_tranches,
            'notional_per_tranche': self.notional_per_tranche,
            'lot_size': self.lot_size
        }
