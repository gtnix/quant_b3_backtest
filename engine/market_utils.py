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
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
import pandas as pd
import numpy as np
from collections import defaultdict

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
        
        # Round to nearest tick using HALF_UP semantics (Brazilian market convention)
        try:
            from decimal import Decimal, ROUND_HALF_UP
            tick = Decimal(str(self.tick_size))
            value = Decimal(str(price)) / tick
            # Round the number of ticks using HALF_UP, then scale back
            ticks_rounded = value.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
            normalized = ticks_rounded * tick
            # Ensure exactly 2 decimal places
            return float(normalized.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        except Exception:
            # Fallback: float-based half-up approximation
            import math
            normalized = math.floor((price / self.tick_size) + 0.5) * self.tick_size
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
    
    # Quiet test output for performance runs
    
    # Test price normalization
    test_prices = [12.3456, 12.344, 15.0, 0.99]
    for price in test_prices:
        _ = utils.normalize_price_tick(price)
    
    # Test lot validation
    test_quantities = [100, 150, 50, 200, 75]
    for qty in test_quantities:
        _ = utils.validate_lot_size(qty)
    
    # Test order validation
    _ = utils.validate_order(price=12.3456, quantity=150)


# DailyAggregator removed - using Brapi.dev daily data directly


class DailyTechnicalIndicators:
    """
    Calculate daily technical indicators (ATR, EMA, RSI) from Brapi.dev daily OHLCV data.
    
    All indicators are calculated using daily bars from Brapi.dev API,
    with periods expressed in trading days.
    """
    
    def __init__(self):
        """Initialize the technical indicators calculator."""
        self.calculated_indicators = {}
    
    def calculate_atr(self, daily_data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate daily ATR (Average True Range) using daily bars with enhanced validation.
        
        Args:
            daily_data: Daily OHLCV DataFrame
            period: ATR period in days (default: 14)
            
        Returns:
            Series with ATR values indexed by date
        """
        # Enhanced validation
        if len(daily_data) < period + 1:
            logger.warning(f"Insufficient data for ATR calculation: {len(daily_data)} < {period + 1}")
            return pd.Series(dtype=float, index=daily_data.index)
        
        # Validate OHLC data
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in daily_data.columns:
                logger.error(f"Missing required column {col} for ATR calculation")
                return pd.Series(dtype=float, index=daily_data.index)
        
        # Check for NaN values and handle them
        for col in required_cols:
            if daily_data[col].isna().any():
                logger.warning(f"NaN values found in {col}, filling with forward fill")
                daily_data[col] = daily_data[col].ffill().bfill()
                
                # If still have NaNs, return empty series
                if daily_data[col].isna().any():
                    logger.error(f"Unable to fill NaN values in {col}")
                    return pd.Series(dtype=float, index=daily_data.index)
        
        logger.debug(f"Calculating ATR({period}) on {len(daily_data)} daily bars")
        
        # Calculate True Range for each day
        high = daily_data['high']
        low = daily_data['low']
        close = daily_data['close']
        prev_close = close.shift(1)
        
        # Validate price relationships
        if not (low <= high).all():
            logger.error("Invalid price data: low > high found")
            return pd.Series(dtype=float, index=daily_data.index)
        
        # Calculate True Range with error handling
        try:
            # TR = max(H-L, |H-C_prev|, |L-C_prev|)
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Validate true range
            if true_range.isna().any():
                logger.warning("NaN values in true range calculation")
                true_range = true_range.ffill().bfill()
                
                # If still have NaNs, return empty series
                if true_range.isna().any():
                    logger.error("Unable to fill NaN values in true range")
                    return pd.Series(dtype=float, index=daily_data.index)
            
            # Check for extreme true range values
            if true_range.max() > 1000:  # More than R$ 1000 true range is suspicious
                logger.warning(f"Unusually high true range detected: {true_range.max()}")
            
            # ATR = Simple Moving Average of True Range
            atr = true_range.rolling(window=period, min_periods=period).mean()
            
            # Final validation
            if atr.isna().all():
                logger.error("All ATR values are NaN")
                return pd.Series(dtype=float, index=daily_data.index)
            
            # Check for extreme ATR values
            if atr.max() > 500:  # More than R$ 500 ATR is suspicious
                logger.warning(f"Unusually high ATR detected: {atr.max()}")
            
            logger.debug(f"ATR calculation complete: {atr.count()} valid values")
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series(dtype=float, index=daily_data.index)
    
    def calculate_ema(self, daily_data: pd.DataFrame, period: int = 10, column: str = 'close') -> pd.Series:
        """
        Calculate daily EMA (Exponential Moving Average) using daily bars.
        
        Args:
            daily_data: Daily OHLCV DataFrame
            period: EMA period in days (default: 10)
            column: Column to calculate EMA on (default: 'close')
            
        Returns:
            Series with EMA values indexed by date
        """
        if len(daily_data) < period:
            logger.warning(f"Insufficient data for EMA calculation: {len(daily_data)} < {period}")
            return pd.Series(dtype=float, index=daily_data.index)
        
        logger.debug(f"Calculating EMA({period}) on {column} for {len(daily_data)} daily bars")
        
        # Calculate EMA using pandas ewm
        ema = daily_data[column].ewm(span=period, adjust=False).mean()
        
        logger.debug(f"EMA calculation complete: {ema.count()} valid values")
        return ema
    
    def calculate_rsi(self, daily_data: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
        """
        Calculate daily RSI (Relative Strength Index) using daily bars.
        
        Args:
            daily_data: Daily OHLCV DataFrame
            period: RSI period in days (default: 14)
            column: Column to calculate RSI on (default: 'close')
            
        Returns:
            Series with RSI values indexed by date
        """
        if len(daily_data) < period + 1:
            logger.warning(f"Insufficient data for RSI calculation: {len(daily_data)} < {period + 1}")
            return pd.Series(dtype=float, index=daily_data.index)
        
        logger.debug(f"Calculating RSI({period}) on {column} for {len(daily_data)} daily bars")
        
        # Calculate daily returns
        prices = daily_data[column]
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate exponential moving averages of gains and losses
        avg_gains = gains.ewm(span=period, adjust=False).mean()
        avg_losses = losses.ewm(span=period, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        logger.debug(f"RSI calculation complete: {rsi.count()} valid values")
        return rsi
    
    def calculate_sma(self, daily_data: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
        """
        Calculate daily SMA (Simple Moving Average) using daily bars.
        
        Args:
            daily_data: Daily OHLCV DataFrame
            period: SMA period in days (default: 20)
            column: Column to calculate SMA on (default: 'close')
            
        Returns:
            Series with SMA values indexed by date
        """
        if len(daily_data) < period:
            logger.warning(f"Insufficient data for SMA calculation: {len(daily_data)} < {period}")
            return pd.Series(dtype=float, index=daily_data.index)
        
        logger.debug(f"Calculating SMA({period}) on {column} for {len(daily_data)} daily bars")
        
        # Calculate SMA using pandas rolling
        sma = daily_data[column].rolling(window=period, min_periods=period).mean()
        
        logger.debug(f"SMA calculation complete: {sma.count()} valid values")
        return sma
    
    def calculate_all_indicators(self, daily_data: pd.DataFrame, 
                               atr_period: int = 14,
                               ema_periods: List[int] = [10, 20],
                               rsi_period: int = 14,
                               sma_periods: List[int] = [20, 50]) -> Dict[str, pd.Series]:
        """
        Calculate all daily technical indicators at once.
        
        Args:
            daily_data: Daily OHLCV DataFrame
            atr_period: ATR period in days
            ema_periods: List of EMA periods in days
            rsi_period: RSI period in days
            sma_periods: List of SMA periods in days
            
        Returns:
            Dictionary with all calculated indicators
        """
        logger.info(f"Calculating all daily indicators for {len(daily_data)} bars")
        
        indicators = {}
        
        # Calculate ATR
        indicators['atr'] = self.calculate_atr(daily_data, atr_period)
        
        # Calculate EMAs
        for period in ema_periods:
            indicators[f'ema_{period}'] = self.calculate_ema(daily_data, period)
        
        # Calculate RSI
        indicators['rsi'] = self.calculate_rsi(daily_data, rsi_period)
        
        # Calculate SMAs
        for period in sma_periods:
            indicators[f'sma_{period}'] = self.calculate_sma(daily_data, period)
        
        # Store for future reference
        self.calculated_indicators.update(indicators)
        
        logger.info(f"Calculated {len(indicators)} indicators")
        return indicators
    
    def get_indicator_summary(self) -> Dict[str, Any]:
        """
        Get summary of calculated indicators.
        
        Returns:
            Dictionary with indicator statistics
        """
        if not self.calculated_indicators:
            return {'status': 'no_indicators', 'indicators': []}
        
        summary = {}
        for name, series in self.calculated_indicators.items():
            summary[name] = {
                'count': series.count(),
                'mean': float(series.mean()) if series.count() > 0 else None,
                'std': float(series.std()) if series.count() > 0 else None,
                'min': float(series.min()) if series.count() > 0 else None,
                'max': float(series.max()) if series.count() > 0 else None,
                'last_value': float(series.iloc[-1]) if series.count() > 0 else None
            }
        
        return {
            'status': 'calculated',
            'indicator_count': len(self.calculated_indicators),
            'indicators': summary
        }


class DataValidationComparator:
    """
    Compare and validate data from different sources for educational insights.
    
    Features:
    - Compare external daily vs local intraday aggregation
    - Statistical analysis of differences
    - Data quality assessment
    - Educational insights for data source selection
    """
    
    def __init__(self):
        """Initialize the data validation comparator."""
        self.comparison_results = {}
        # DailyAggregator removed - using Brapi.dev daily data directly
        self.indicators_calculator = DailyTechnicalIndicators()
    
    def compare_data_sources(self, symbol: str, reference_daily: pd.DataFrame, 
                           local_intraday: pd.DataFrame, date_range: Tuple[str, str] = None) -> Dict[str, Any]:
        """
        Compare external daily data vs local intraday aggregation.
        
        Args:
            symbol: Trading symbol
            reference_daily: Daily data from external provider
            local_intraday: Intraday data for aggregation
            date_range: Optional (start_date, end_date) tuple to filter comparison
            
        Returns:
            Dictionary with detailed comparison results
        """
        logger.info(f"🧪 Starting data validation comparison for {symbol}")
        
        try:
            # Filter data to comparison range if specified
            if date_range:
                start_date, end_date = date_range
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                
                if not reference_daily.empty:
                    mask_yf = (reference_daily.index >= start_dt) & (reference_daily.index <= end_dt)
                    reference_daily = reference_daily.loc[mask_yf]
                
                if not local_intraday.empty:
                    mask_local = (local_intraday.index >= start_dt) & (local_intraday.index <= end_dt)
                    local_intraday = local_intraday.loc[mask_local]
            
            # DailyAggregator removed - this comparison is no longer valid
            # Using Brapi.dev daily data directly instead of aggregation
            local_daily = pd.DataFrame()  # Empty DataFrame to indicate no local aggregation
            
            if reference_daily.empty or local_daily.empty:
                return {
                    'comparison_possible': False,
                    'reason': 'Insufficient data for comparison',
                    'reference_bars': len(reference_daily),
                    'local_bars': len(local_daily),
                    'recommendations': ['Ensure both data sources have sufficient data for comparison']
                }
            
            # Align data by common dates
            aligned_data = self._align_datasets(reference_daily, local_daily)
            
            if aligned_data['common_dates'] == 0:
                return {
                    'comparison_possible': False,
                    'reason': 'No overlapping dates between data sources',
                    'reference_range': f"{reference_daily.index.min().date()} to {reference_daily.index.max().date()}",
                    'local_range': f"{local_daily.index.min().date()} to {local_daily.index.max().date()}",
                    'recommendations': ['Check data date ranges for overlap']
                }
            
            # Perform OHLCV comparison
            ohlcv_comparison = self._compare_ohlcv_data(aligned_data['reference'], aligned_data['local'])
            
            # Perform technical indicators comparison
            indicators_comparison = self._compare_technical_indicators(
                aligned_data['reference'], aligned_data['local'], symbol
            )
            
            # Generate overall assessment
            overall_assessment = self._generate_overall_assessment(
                ohlcv_comparison, indicators_comparison, aligned_data
            )
            
            comparison_result = {
                'comparison_possible': True,
                'symbol': symbol,
                'comparison_period': {
                'start_date': aligned_data['reference'].index.min().strftime('%Y-%m-%d'),
                'end_date': aligned_data['reference'].index.max().strftime('%Y-%m-%d'),
                    'common_trading_days': aligned_data['common_dates']
                },
                'ohlcv_comparison': ohlcv_comparison,
                'indicators_comparison': indicators_comparison,
                'overall_assessment': overall_assessment,
                'data_quality_insights': self._generate_data_quality_insights(
                    ohlcv_comparison, indicators_comparison
                ),
                'recommendations': self._generate_recommendations(
                    ohlcv_comparison, indicators_comparison, overall_assessment
                )
            }
            
            # Store for future reference
            self.comparison_results[symbol] = comparison_result
            
            logger.info(f"✅ Data validation comparison completed for {symbol}")
            return comparison_result
            
        except Exception as e:
            logger.error(f"Error in data validation comparison for {symbol}: {e}")
            return {
                'comparison_possible': False,
                'reason': f'Comparison failed: {str(e)}',
                'recommendations': ['Check data quality and format consistency']
            }
    
    def _align_datasets(self, reference_data: pd.DataFrame, local_data: pd.DataFrame) -> Dict[str, Any]:
        """Align datasets by common trading dates."""
        # Find common dates
        yf_dates = set(reference_data.index.date)
        local_dates = set(local_data.index.date)
        common_dates = yf_dates.intersection(local_dates)
        
        if not common_dates:
            return {
                'reference': pd.DataFrame(),
                'local': pd.DataFrame(),
                'common_dates': 0
            }
        
        # Filter to common dates
        yf_mask = reference_data.index.date.isin(common_dates)
        local_mask = local_data.index.date.isin(common_dates)
        
        aligned_yf = reference_data.loc[yf_mask].sort_index()
        aligned_local = local_data.loc[local_mask].sort_index()
        
        return {
            'reference': aligned_yf,
            'local': aligned_local,
            'common_dates': len(common_dates)
        }
    
    def _compare_ohlcv_data(self, reference_daily: pd.DataFrame, local_data: pd.DataFrame) -> Dict[str, Any]:
        """Compare OHLCV data between sources."""
        comparison = {}
        
        # Align by exact dates for point-to-point comparison
        aligned_index = reference_daily.index.intersection(local_data.index)
        
        if len(aligned_index) == 0:
            return {'error': 'No aligned data points for OHLCV comparison'}
        
        yf_aligned = reference_daily.loc[aligned_index]
        local_aligned = local_data.loc[aligned_index]
        
        # Compare each OHLCV component
        for column in ['open', 'high', 'low', 'close', 'volume']:
            if column in yf_aligned.columns and column in local_aligned.columns:
                yf_values = yf_aligned[column]
                local_values = local_aligned[column]
                
                # Calculate differences
                absolute_diff = (yf_values - local_values).abs()
                percentage_diff = (absolute_diff / yf_values.abs() * 100).fillna(0)
                
                # Calculate correlation
                correlation = yf_values.corr(local_values) if len(yf_values) > 1 else 1.0
                
                comparison[column] = {
                    'correlation': float(correlation) if not pd.isna(correlation) else 0.0,
                    'mean_absolute_diff': float(absolute_diff.mean()),
                    'mean_percentage_diff': float(percentage_diff.mean()),
                    'max_percentage_diff': float(percentage_diff.max()),
                    'points_compared': len(yf_values),
                    'similar_points': int((percentage_diff < 1.0).sum()),  # Within 1%
                    'similarity_ratio': float((percentage_diff < 1.0).mean())
                }
        
        return comparison
    
    def _compare_technical_indicators(self, reference_daily: pd.DataFrame, 
                                    local_data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Compare technical indicators calculated from both data sources."""
        try:
            # Calculate indicators for both datasets
            yf_indicators = self.indicators_calculator.calculate_all_indicators(
                reference_daily, atr_period=14, ema_periods=[10, 20], rsi_period=14
            )
            
            local_indicators = self.indicators_calculator.calculate_all_indicators(
                local_data, atr_period=14, ema_periods=[10, 20], rsi_period=14
            )
            
            indicators_comparison = {}
            
            # Compare each indicator
            for indicator_name in yf_indicators.keys():
                if indicator_name in local_indicators:
                    yf_values = yf_indicators[indicator_name].dropna()
                    local_values = local_indicators[indicator_name].dropna()
                    
                    # Align by common dates
                    common_index = yf_values.index.intersection(local_values.index)
                    
                    if len(common_index) > 1:
                        yf_aligned = yf_values.loc[common_index]
                        local_aligned = local_values.loc[common_index]
                        
                        # Calculate comparison metrics
                        correlation = yf_aligned.corr(local_aligned)
                        absolute_diff = (yf_aligned - local_aligned).abs()
                        percentage_diff = (absolute_diff / yf_aligned.abs() * 100).fillna(0)
                        
                        indicators_comparison[indicator_name] = {
                            'correlation': float(correlation) if not pd.isna(correlation) else 0.0,
                            'mean_absolute_diff': float(absolute_diff.mean()),
                            'mean_percentage_diff': float(percentage_diff.mean()),
                            'points_compared': len(yf_aligned),
                            'data_quality': 'excellent' if correlation > 0.95 else 'good' if correlation > 0.85 else 'fair'
                        }
            
            return indicators_comparison
            
        except Exception as e:
            logger.error(f"Error comparing technical indicators: {e}")
            return {'error': f'Indicators comparison failed: {str(e)}'}
    
    def _generate_overall_assessment(self, ohlcv_comparison: Dict, 
                                   indicators_comparison: Dict, aligned_data: Dict) -> Dict[str, Any]:
        """Generate overall assessment of data source comparison."""
        
        # Calculate overall correlation from OHLCV
        ohlcv_correlations = []
        for component, stats in ohlcv_comparison.items():
            if isinstance(stats, dict) and 'correlation' in stats:
                ohlcv_correlations.append(stats['correlation'])
        
        avg_ohlcv_correlation = sum(ohlcv_correlations) / len(ohlcv_correlations) if ohlcv_correlations else 0
        
        # Calculate overall correlation from indicators
        indicator_correlations = []
        for indicator, stats in indicators_comparison.items():
            if isinstance(stats, dict) and 'correlation' in stats:
                indicator_correlations.append(stats['correlation'])
        
        avg_indicator_correlation = sum(indicator_correlations) / len(indicator_correlations) if indicator_correlations else 0
        
        # Overall assessment
        overall_correlation = (avg_ohlcv_correlation + avg_indicator_correlation) / 2
        
        if overall_correlation > 0.95:
            quality_rating = "excellent"
            confidence = "high"
        elif overall_correlation > 0.85:
            quality_rating = "good"
            confidence = "medium-high"
        elif overall_correlation > 0.70:
            quality_rating = "fair"
            confidence = "medium"
        else:
            quality_rating = "poor"
            confidence = "low"
        
        return {
            'overall_correlation': overall_correlation,
            'quality_rating': quality_rating,
            'confidence_level': confidence,
            'data_consistency': 'high' if overall_correlation > 0.90 else 'medium' if overall_correlation > 0.75 else 'low',
            'trading_days_compared': aligned_data['common_dates']
        }
    
    def _generate_data_quality_insights(self, ohlcv_comparison: Dict, 
                                      indicators_comparison: Dict) -> List[str]:
        """Generate insights about data quality."""
        insights = []
        
        # OHLCV insights
        if 'close' in ohlcv_comparison:
            close_stats = ohlcv_comparison['close']
            if close_stats.get('correlation', 0) > 0.95:
                insights.append("✅ Close prices show excellent agreement between sources")
            elif close_stats.get('mean_percentage_diff', 0) > 2:
                insights.append("⚠️ Close prices show significant differences - investigate data sources")
        
        if 'volume' in ohlcv_comparison:
            volume_stats = ohlcv_comparison['volume']
            if volume_stats.get('correlation', 0) < 0.80:
                insights.append("📊 Volume data differs significantly - expected for different data providers")
        
        # Indicators insights
        if 'atr' in indicators_comparison:
            atr_stats = indicators_comparison['atr']
            if atr_stats.get('correlation', 0) > 0.90:
                insights.append("✅ ATR calculations are highly consistent across sources")
        
        if 'rsi' in indicators_comparison:
            rsi_stats = indicators_comparison['rsi']
            if rsi_stats.get('correlation', 0) > 0.95:
                insights.append("✅ RSI calculations show excellent consistency")
        
        if not insights:
            insights.append("📊 Basic comparison completed - review detailed metrics for insights")
        
        return insights
    
    def _generate_recommendations(self, ohlcv_comparison: Dict, 
                                indicators_comparison: Dict, overall_assessment: Dict) -> List[str]:
        """Generate recommendations based on comparison results."""
        recommendations = []
        
        overall_correlation = overall_assessment.get('overall_correlation', 0)
        
        if overall_correlation > 0.95:
            recommendations.append("✅ Data sources show excellent agreement - either can be used confidently")
            recommendations.append("💡 Consider using BRAPI daily for indicators and local data for execution precision")
        elif overall_correlation > 0.85:
            recommendations.append("👍 Data sources show good agreement with minor differences")
            recommendations.append("🔍 Monitor differences in critical periods or volatile markets")
        elif overall_correlation > 0.70:
            recommendations.append("⚠️ Data sources show moderate differences - investigate causes")
            recommendations.append("📊 Consider data quality improvements or alternative sources")
        else:
            recommendations.append("❌ Significant differences detected - data validation required")
            recommendations.append("🔧 Review data processing pipelines and source reliability")
        
        # Volume-specific recommendations
        if 'volume' in ohlcv_comparison:
            volume_corr = ohlcv_comparison['volume'].get('correlation', 0)
            if volume_corr < 0.70:
                recommendations.append("📈 Volume differences are normal between data providers")
        
        return recommendations
    
    def generate_comparison_report(self, symbol: str) -> None:
        """Generate and print a comprehensive comparison report."""
        if symbol not in self.comparison_results:
            print(f"❌ No comparison results available for {symbol}")
            return
        
        result = self.comparison_results[symbol]
        
        if not result.get('comparison_possible', False):
            print(f"\n❌ Data Comparison Not Possible for {symbol}")
            print(f"Reason: {result.get('reason', 'Unknown')}")
            return
        
        print(f"\n🧪 DATA VALIDATION REPORT: {symbol}")
        print("=" * 60)


# =============================
# Multi-frame services (no new files)
# =============================

class IndicatorService:
    """
    Daily-only indicator orchestration and FuzzyFajuto score calculation.
    Operates independently from the simulator; does not fetch or require hourly data.
    """
    def __init__(self, api_token_env: str = 'BRAPI_API_TOKEN', cache_dir: str = 'data/brapi_cache'):
        self.api_token_env = api_token_env
        self.cache_dir = cache_dir
        self.ind_calc = DailyTechnicalIndicators()

    def _fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        from engine.brapi_provider import BrapiProvider
        import os as _os
        bp = BrapiProvider(api_token=_os.environ.get(self.api_token_env, ''), cache_dir=self.cache_dir)
        df = bp.get_daily_data(symbol, start, end)
        if df is None:
            return pd.DataFrame()
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
        return df

    def compute_daily_vectors(self, symbols: list[str], benchmark: str, start: str, end: str,
                               ema_periods: list[int] = [3,5,10,15,20], rsi_period: int = 10,
                               atr_period: int = 14, warmup_min_sessions: int = 60,
                               buffer_sessions: int = 5) -> dict[str, pd.DataFrame]:
        # Extend daily window for warmup
        start_dt = pd.to_datetime(start)
        warmup_days = max(3 * max(ema_periods), rsi_period + 1, atr_period + 1, warmup_min_sessions) + buffer_sessions
        start_warm = (start_dt - pd.tseries.offsets.BDay(warmup_days)).date().strftime('%Y-%m-%d')
        # Helper: aggregate intraday to daily OHLCV if needed
        def _to_daily_ohlcv(df_in: pd.DataFrame) -> pd.DataFrame:
            if df_in is None or df_in.empty:
                return pd.DataFrame()
            df = df_in.copy()
            try:
                idx = pd.to_datetime(df.index)
            except Exception:
                return pd.DataFrame()
            # Normalize to date; aggregate OHLCV
            dates = idx.tz_localize(None).normalize() if getattr(idx, 'tz', None) is not None else idx.normalize()
            df = df.copy()
            df.index = dates
            agg_map = {}
            for c in ('open','high','low','close','volume'):
                if c in df.columns:
                    if c == 'open':
                        agg_map[c] = 'first'
                    elif c == 'high':
                        agg_map[c] = 'max'
                    elif c == 'low':
                        agg_map[c] = 'min'
                    elif c == 'close':
                        agg_map[c] = 'last'
                    elif c == 'volume':
                        agg_map[c] = 'sum'
            if not agg_map:
                return pd.DataFrame()
            daily = df.groupby(df.index).agg(agg_map)
            daily = daily.sort_index()
            return daily

        # Fetch benchmark and aggregate to daily
        ibov_raw = self._fetch_daily(benchmark, start_warm, end)
        ibov = _to_daily_ohlcv(ibov_raw)
        ibov_available = (ibov is not None and not ibov.empty)
        if ibov_available:
            ibov = ibov.sort_index()
            ibov['ret'] = ibov['close'].pct_change()
        results: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                df_raw = self._fetch_daily(sym, start_warm, end)
                if df_raw is None or df_raw.empty:
                    continue
                # Aggregate to daily OHLCV
                df = _to_daily_ohlcv(df_raw)
                if df is None or df.empty:
                    continue
                # Align dates strictly (intersection) with benchmark if available; otherwise use symbol dates
                common = df.index.intersection(ibov.index) if ibov_available else df.index
                if len(common) == 0:
                    continue
                df = df.loc[common]
                # Indicators
                ind = {
                    'atr': self.ind_calc.calculate_atr(df, period=atr_period),
                    'rsi': self.ind_calc.calculate_rsi(df, period=rsi_period)
                }
                for p in ema_periods:
                    ind[f'ema_{p}'] = self.ind_calc.calculate_ema(df, period=p)
                # Vectorize FuzzyFajuto score on aligned days
                sym_ret = df['close'].pct_change()
                if ibov_available:
                    ibov_al = ibov.loc[common]
                    ibov_ret = ibov_al['ret']
                else:
                    # Fallback: no benchmark available; treat ret_vs_ibov as sign of symbol return only
                    ibov_ret = pd.Series(0.0, index=common)
                # Components
                ret_vs_ibov_term = (sym_ret > ibov_ret).astype(int) - (sym_ret < ibov_ret).astype(int)
                ema_sum = pd.Series(0.0, index=common)
                for p in ema_periods:
                    key = f'ema_{p}'
                    s = ind[key]
                    # Align to index; if missing values, keep at 0 contribution
                    s_al = s.reindex(common)
                    ema_sum = ema_sum + ((df['close'].reindex(common) > s_al).astype(float) - (df['close'].reindex(common) < s_al).astype(float)) * 0.25
                rsi_term = pd.Series(0.0, index=pd.Index(common))
                rsi_series = ind.get('rsi')
                if rsi_series is not None and not rsi_series.empty:
                    rsi_al = rsi_series.reindex(common)
                    try:
                        mask_hi = (rsi_al.astype(float) > 65)
                        mask_lo = (rsi_al.astype(float) < 35)
                        mask_hi = mask_hi.reindex(rsi_term.index).fillna(False)
                        mask_lo = mask_lo.reindex(rsi_term.index).fillna(False)
                        rsi_term.loc[mask_hi] = 0.25
                        rsi_term.loc[mask_lo] = -0.25
                    except Exception:
                        pass
                score = ret_vs_ibov_term.astype(float) + ema_sum.fillna(0.0) + rsi_term.fillna(0.0)
                # Slice to requested [start, end] using daily dates (tz-naive)
                try:
                    start_d = pd.to_datetime(start).tz_localize(None).normalize()
                except Exception:
                    start_d = pd.to_datetime(start).normalize()
                try:
                    end_d = pd.to_datetime(end).tz_localize(None).normalize()
                except Exception:
                    end_d = pd.to_datetime(end).normalize()
                # Build output on the common aligned index; filter to [start,end] at the end
                out = pd.DataFrame(index=common)
                out['close'] = df['close'].reindex(common)
                out['ibov_return'] = ibov_ret.reindex(common)
                out['symbol_return'] = sym_ret.reindex(common)
                for k, s in ind.items():
                    out[k] = s.reindex(common)
                out['fuzzy_score'] = score.reindex(common)
                # Final window slice
                mask = (out.index >= start_d) & (out.index <= end_d)
                out = out.loc[mask]
                if not out.empty:
                    results[sym] = out
            except Exception as e:
                try:
                    print(f"[compute_daily_vectors] error for {sym}: {e}")
                except Exception:
                    pass
                continue
        return results


class SignalScheduler:
    """
    Consumes daily vectors and creates T+1 four-leg schedules per symbol-date.
    """
    def __init__(self, round_lot_size: int = 100, tick_size: float = 0.01,
                 leg_notional_brl: float = 10000.0,
                 buy_threshold: float = 1.50, sell_threshold: float = -1.50):
        self.round_lot_size = int(round_lot_size)
        self.tick_size = float(tick_size)
        self.leg_notional_brl = float(leg_notional_brl)
        self.buy_th = float(buy_threshold)
        self.sell_th = float(sell_threshold)

    def _round_to_tick(self, price: float) -> float:
        return round(round(price / self.tick_size) * self.tick_size, 2)

    def _round_to_lot(self, qty: float) -> int:
        q = int(qty // self.round_lot_size) * self.round_lot_size
        return max(q, 0)

    def _limits_from_close(self, close_price: float, side: str) -> tuple[float, float, float]:
        step1, step2, step3 = 0.005, 0.010, 0.015
        if side == 'BUY':
            p2 = max(close_price * (1.0 - step1), 0.01)
            p3 = max(close_price * (1.0 - step2), 0.01)
            p4 = max(close_price * (1.0 - step3), 0.01)
        else:
            p2 = close_price * (1.0 + step1)
            p3 = close_price * (1.0 + step2)
            p4 = close_price * (1.0 + step3)
        return (self._round_to_tick(p2), self._round_to_tick(p3), self._round_to_tick(p4))

    def build_schedule(self, vectors: dict[str, pd.DataFrame]) -> dict:
        """
        Returns: schedule dict keyed by execution date D = T+1
        schedule[D][symbol] = { side, valid_for_date, base_close_t, limits_used:{...}, current_atr_t, fuzzy_score_t }
        """
        schedule: dict = {}
        for sym, df in vectors.items():
            if df is None or df.empty or 'fuzzy_score' not in df.columns:
                continue
            for ts, row in df.iterrows():
                try:
                    fs = float(row.get('fuzzy_score', 0.0))
                    if fs >= self.buy_th:
                        side = 'BUY'
                    elif fs <= self.sell_th:
                        side = 'SELL'
                    else:
                        continue
                    close_t = float(row.get('close', 0.0) or 0.0)
                    atr_t = float(row.get('atr', 0.0) or 0.0)
                    p2, p3, p4 = self._limits_from_close(close_t, side)
                    # Precompute quantities per leg (rounded to lot); market qty sized at open≈close for schedule
                    qty1 = self._round_to_lot(self.leg_notional_brl / max(close_t, 1e-9))
                    qty2 = self._round_to_lot(self.leg_notional_brl / max(p2, 1e-9))
                    qty3 = self._round_to_lot(self.leg_notional_brl / max(p3, 1e-9))
                    qty4 = self._round_to_lot(self.leg_notional_brl / max(p4, 1e-9))
                    # Reject fully unfeasible zero-qty signals
                    if max(qty1, qty2, qty3, qty4) == 0:
                        continue
                    d = (pd.to_datetime(ts) + pd.Timedelta(days=1)).date()
                    day_store = schedule.setdefault(d, {})
                    day_store[sym] = {
                        'symbol': sym,
                        'side': OrderSide.BUY if side == 'BUY' else OrderSide.SELL,
                        'valid_for_date': d,
                        'base_close_t': close_t,
                        'limits_used': {'limit_level_2': p2, 'limit_level_3': p3, 'limit_level_4': p4},
                        'current_atr_t': atr_t,
                        'fuzzy_score_t': float(abs(fs))
                    }
                except Exception:
                    continue
        return schedule


class DataRequirements:
    """
    Deterministic computation of data ranges needed for a run.
    """
    @staticmethod
    def compute_daily_warmup(start: str, ema_max: int, rsi: int, atr: int, floor_sessions: int = 60, buffer_sessions: int = 5) -> dict:
        start_dt = pd.to_datetime(start)
        warmup_days = max(3 * int(ema_max), int(rsi) + 1, int(atr) + 1, int(floor_sessions)) + int(buffer_sessions)
        warmup_start = (start_dt - pd.tseries.offsets.BDay(warmup_days)).date().strftime('%Y-%m-%d')
        return {
            'warmup_days': int(warmup_days),
            'warmup_start': warmup_start,
        }

    @staticmethod
    def list_execution_days(schedule: dict) -> dict[str, list[pd.Timestamp]]:
        sym_to_days: dict[str, list[pd.Timestamp]] = {}
        for d, syms in (schedule or {}).items():
            for sym in syms.keys():
                # Skip non-symbol metadata keys
                if isinstance(sym, str) and sym.startswith('__'):
                    continue
                sym_to_days.setdefault(sym, []).append(pd.to_datetime(d))
        for sym, lst in sym_to_days.items():
            sym_to_days[sym] = sorted(list({pd.to_datetime(x).normalize() for x in lst}))
        return sym_to_days


class MarketDataRouter:
    """
    Centralized data access with local-first policy and graceful degradation.
    """
    def __init__(self, api_token_env: str = 'BRAPI_API_TOKEN', cache_dir: str = 'data/brapi_cache'):
        from engine.brapi_provider import BrapiProvider
        import os as _os
        self._bp = BrapiProvider(api_token=_os.environ.get(api_token_env, ''), cache_dir=cache_dir)

    def get_daily(self, symbol: str, start: str, end: str, min_sessions: int = 60, auto_extend_days: int = 120) -> tuple[pd.DataFrame, dict]:
        meta = {'degraded_warmup': False, 'source': 'brapi_daily', 'attempts': []}
        def _fetch(_start: str) -> pd.DataFrame:
            try:
                df = self._bp.get_daily_data(symbol, _start, end)
                if df is None:
                    return pd.DataFrame()
                df.index = pd.to_datetime(df.index)
                return df.sort_index()
            except Exception as e:
                meta['attempts'].append({'source': 'brapi_daily', 'error': str(e)})
                return pd.DataFrame()
        df = _fetch(start)
        if len(df) < min_sessions:
            # Auto-extend lookback using business days
            start_dt = pd.to_datetime(start)
            for extra in (30, 60, 90, auto_extend_days):
                ext_start = (start_dt - pd.tseries.offsets.BDay(int(extra))).date().strftime('%Y-%m-%d')
                dfx = _fetch(ext_start)
                if len(dfx) >= min_sessions:
                    df = dfx
                    break
            if len(df) < min_sessions:
                meta['degraded_warmup'] = True
        return df, meta

    def get_hourly_for_day(self, symbol: str, day: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
        meta = {'source': 'brapi_hourly', 'date': str(pd.to_datetime(day).date())}
        try:
            # Prefer local cache first to avoid tz/window mismatches
            from engine.loader import DataLoader
            df_cache = DataLoader._load_best_intraday_cache(symbol, 'data/brapi_cache')
            if df_cache is not None and not df_cache.empty:
                df_local = df_cache.copy()
                try:
                    idx = pd.to_datetime(df_local.index)
                except Exception:
                    idx = df_local.index
                # Treat tz-naive as session-local; only convert when tz-aware
                if getattr(idx, 'tz', None) is not None:
                    try:
                        idx = idx.tz_convert('America/Sao_Paulo').tz_localize(None)
                    except Exception:
                        idx = idx.tz_localize(None)
                df_local.index = idx
                mask = (df_local.index.date == pd.to_datetime(day).date())
                sliced = df_local.loc[mask]
                if sliced is not None and not sliced.empty:
                    return sliced.sort_index(), meta
            # Fallback to hourly/ cache if intraday not present
            try:
                from pathlib import Path as _Path
                import pandas as _pd
                hourly_dir = _Path('data/brapi_cache/hourly')
                f = next((p for p in hourly_dir.glob(f"{symbol}_*.parquet")), None)
                if f and f.exists():
                    dfh = _pd.read_parquet(f)
                    idxh = _pd.to_datetime(dfh.index)
                    if getattr(idxh, 'tz', None) is not None:
                        try:
                            idxh = idxh.tz_convert('America/Sao_Paulo').tz_localize(None)
                        except Exception:
                            idxh = idxh.tz_localize(None)
                    dfh.index = idxh
                    maskh = (dfh.index.date == pd.to_datetime(day).date())
                    slicedh = dfh.loc[maskh]
                    if slicedh is not None and not slicedh.empty:
                        return slicedh.sort_index(), meta
            except Exception:
                pass
            # Fallback to provider
            start = pd.to_datetime(day).strftime('%Y-%m-%d')
            end = (pd.to_datetime(day) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            df = self._bp.get_hourly_data(symbol, start, end)
            if df is None:
                return pd.DataFrame(), meta
            df = df.copy()
            idx = pd.to_datetime(df.index)
            if getattr(idx, 'tz', None) is not None:
                idx = idx.tz_localize(None)
            df.index = idx
            mask = (df.index.date == pd.to_datetime(day).date())
            df = df.loc[mask]
            return df.sort_index(), meta
        except Exception as e:
            meta['error'] = str(e)
            return pd.DataFrame(), meta
        
        # Period info
        period = result['comparison_period']
        print(f"📅 Period: {period['start_date']} to {period['end_date']} ({period['common_trading_days']} days)")
        
        # Overall assessment
        assessment = result['overall_assessment']
        print(f"\n🎯 OVERALL ASSESSMENT:")
        print(f"  Correlation: {assessment['overall_correlation']:.3f}")
        print(f"  Quality Rating: {assessment['quality_rating']}")
        print(f"  Confidence: {assessment['confidence_level']}")
        
        # OHLCV comparison
        print(f"\n📊 OHLCV COMPARISON:")
        ohlcv = result['ohlcv_comparison']
        for component, stats in ohlcv.items():
            if isinstance(stats, dict):
                print(f"  {component.upper()}: corr={stats['correlation']:.3f}, "
                      f"avg_diff={stats['mean_percentage_diff']:.2f}%")
        
        # Technical indicators
        print(f"\n🧮 TECHNICAL INDICATORS:")
        indicators = result['indicators_comparison']
        for indicator, stats in indicators.items():
            if isinstance(stats, dict):
                print(f"  {indicator.upper()}: corr={stats['correlation']:.3f}, "
                      f"quality={stats['data_quality']}")
        
        # Insights
        print(f"\n💡 INSIGHTS:")
        for insight in result['data_quality_insights']:
            print(f"  {insight}")
        
        # Recommendations
        print(f"\n📋 RECOMMENDATIONS:")
        for rec in result['recommendations']:
            print(f"  {rec}")
        
        print("=" * 60) 


# =============================
# FuzzyFajuto data preparation
# =============================
def prepare_fuzzy_data(symbols: list[str], benchmark: str, start_date, end_date) -> pd.DataFrame:
    """
    Prepare aligned daily OHLC-derived inputs for FuzzyFajuto.

    Fetch window: max(90 calendar days, 60 business days) prior to start_date through end_date.
    Outputs one row per (date, symbol) including:
      stock_return, ibov_return, rs_component,
      ema_3/5/10/15/20 signals (+0.25/-0.25/0), rsi_signal (+0.25/-0.25/0), atr_value.

    Notes:
    - No forward-fill of returns; alignment on common trading days only, then reindex to calendar
    - Indicators computed from daily bars via generic DailyTechnicalIndicators
    - Caller can safely slice to simulation window; lookback ensures EMA/ATR warm-up.
    """
    from pandas.tseries.offsets import BDay
    ind = IndicatorService()
    ind_calc = DailyTechnicalIndicators()

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    lookback_90c = (start_dt - pd.Timedelta(days=90)).date().strftime('%Y-%m-%d')
    lookback_60b = (start_dt - BDay(60)).date().strftime('%Y-%m-%d')
    start_warm = min(lookback_90c, lookback_60b)

    # Benchmark daily
    ibov = ind._fetch_daily(benchmark, start_warm, end_dt.strftime('%Y-%m-%d'))
    if ibov is None:
        ibov = pd.DataFrame()
    try:
        ibov.index = pd.to_datetime(ibov.index).tz_localize(None).normalize()
    except Exception:
        ibov.index = pd.to_datetime(ibov.index).normalize()
    ibov = ibov.sort_index()
    ibov['ibov_return'] = ibov['close'].pct_change()

    cal = pd.date_range(start=start_dt, end=end_dt, freq='D')
    out_frames: list[pd.DataFrame] = []
    for sym in symbols:
        try:
            df = ind._fetch_daily(sym, start_warm, end_dt.strftime('%Y-%m-%d'))
            if df is None or df.empty:
                continue
            try:
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            except Exception:
                df.index = pd.to_datetime(df.index).normalize()
            df = df.sort_index()
            # Indicators (daily)
            atr = ind_calc.calculate_atr(df, period=14)
            ema3 = df['close'].ewm(span=3, adjust=False).mean()
            ema5 = df['close'].ewm(span=5, adjust=False).mean()
            ema10 = df['close'].ewm(span=10, adjust=False).mean()
            ema15 = df['close'].ewm(span=15, adjust=False).mean()
            ema20 = df['close'].ewm(span=20, adjust=False).mean()
            delta = df['close'].diff()
            up = delta.clip(lower=0).rolling(10).mean()
            down = (-delta.clip(upper=0)).rolling(10).mean()
            rs = up / (down.replace(0, pd.NA))
            rsi = 100 - (100 / (1 + rs))
            stock_ret = df['close'].pct_change()

            common = df.index.intersection(ibov.index)
            if len(common) == 0:
                continue
            base = pd.DataFrame(index=common)
            base['close'] = df['close'].reindex(common)
            base['stock_return'] = stock_ret.reindex(common)
            base['ibov_return'] = ibov['ibov_return'].reindex(common)
            rs_cmp = pd.Series(0.0, index=common)
            rs_cmp[base['stock_return'] > base['ibov_return']] = 1.0
            rs_cmp[base['stock_return'] < base['ibov_return']] = -1.0
            base['rs_component'] = rs_cmp

            def em_sig(ema_series: pd.Series) -> pd.Series:
                sig = pd.Series(0.0, index=common)
                ema_al = ema_series.reindex(common)
                close_al = df['close'].reindex(common)
                sig[close_al > ema_al] = 0.25
                sig[close_al < ema_al] = -0.25
                return sig

            base['ema_3_signal'] = em_sig(ema3)
            base['ema_5_signal'] = em_sig(ema5)
            base['ema_10_signal'] = em_sig(ema10)
            base['ema_15_signal'] = em_sig(ema15)
            base['ema_20_signal'] = em_sig(ema20)

            rsi_al = rsi.reindex(common)
            rsi_sig = pd.Series(0.0, index=common)
            rsi_sig[rsi_al > 65] = 0.25
            rsi_sig[rsi_al < 35] = -0.25
            base['rsi_signal'] = rsi_sig
            base['atr_value'] = atr.reindex(common)

            # Align to calendar range and forward-fill close to avoid NaNs on non-intersection days
            base = base.reindex(cal)
            try:
                base['close'] = base['close'].ffill()
            except Exception:
                pass
            base.index.name = 'date'
            base = base.reset_index()
            base['symbol'] = sym
            out_frames.append(base)
        except Exception:
            continue
    if not out_frames:
        cols = ['date','symbol','stock_return','ibov_return','rs_component','ema_3_signal','ema_5_signal','ema_10_signal','ema_15_signal','ema_20_signal','rsi_signal','atr_value']
        return pd.DataFrame(columns=cols)
    return pd.concat(out_frames, ignore_index=True).sort_values(['date','symbol'])