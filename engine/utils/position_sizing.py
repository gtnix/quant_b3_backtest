"""Position sizing utilities for Brazilian market compliance.

This module provides utilities for calculating position sizes according to 
Brazilian market B3 board-lot rules and fixed notional allocation strategies.
"""

from typing import Union


def round_b3_board_lot(shares_raw: Union[float, int]) -> int:
    """
    Round raw share quantity to comply with B3 board-lot rules.
    
    B3 board-lot rules require positions to be in multiples of 100 shares.
    The rounding logic is:
    - If the last two digits are 00-49: round DOWN to nearest 100
    - If the last two digits are 50-99: round UP to nearest 100
    
    Args:
        shares_raw: Raw number of shares (can be float or int)
        
    Returns:
        Integer number of shares rounded to nearest 100-share board-lot
        
    Examples:
        >>> round_b3_board_lot(414.35)
        400
        >>> round_b3_board_lot(383.43)  
        400
        >>> round_b3_board_lot(333.42)
        300
        >>> round_b3_board_lot(69.19)
        100
        >>> round_b3_board_lot(234.43)
        200
        >>> round_b3_board_lot(770.17)
        800
        >>> round_b3_board_lot(450.0)
        500
        >>> round_b3_board_lot(449.99)
        400
    """
    if shares_raw <= 0:
        return 0
    
    shares_int = int(shares_raw)
    remainder = shares_int % 100
    
    if remainder >= 50:
        # Round up to next 100
        return ((shares_int // 100) + 1) * 100
    else:
        # Round down to current 100
        return (shares_int // 100) * 100


def calculate_tranche_quantity(notional_brl: float, price: float) -> int:
    """
    Calculate position quantity for a single tranche using fixed notional allocation.
    
    Args:
        notional_brl: Fixed notional amount in BRL for this tranche
        price: Stock price to use for calculation (typically close[T-1])
        
    Returns:
        Integer number of shares rounded to B3 board-lot compliance
        
    Examples:
        >>> calculate_tranche_quantity(12500.0, 30.17)  # PETR4
        400
        >>> calculate_tranche_quantity(12500.0, 32.60)  # PETR3  
        400
        >>> calculate_tranche_quantity(12500.0, 37.49)  # ITUB4
        300
        >>> calculate_tranche_quantity(12500.0, 180.65) # NVDA
        100
        >>> calculate_tranche_quantity(12500.0, 53.32)  # VALE3
        200
        >>> calculate_tranche_quantity(12500.0, 16.23)  # GGBR4
        800
    """
    if price <= 0:
        return 0
        
    shares_raw = notional_brl / price
    return round_b3_board_lot(shares_raw)
