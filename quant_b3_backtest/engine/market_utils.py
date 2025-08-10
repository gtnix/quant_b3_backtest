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
    - Compare yfinance daily vs local intraday aggregation
    - Statistical analysis of differences
    - Data quality assessment
    - Educational insights for data source selection
    """
    
    def __init__(self):
        """Initialize the data validation comparator."""
        self.comparison_results = {}
        # DailyAggregator removed - using Brapi.dev daily data directly
        self.indicators_calculator = DailyTechnicalIndicators()
    
    def compare_data_sources(self, symbol: str, yfinance_daily: pd.DataFrame, 
                           local_intraday: pd.DataFrame, date_range: Tuple[str, str] = None) -> Dict[str, Any]:
        """
        Compare yfinance daily data vs local intraday aggregation.
        
        Args:
            symbol: Trading symbol
            yfinance_daily: Daily data from yfinance
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
                
                if not yfinance_daily.empty:
                    mask_yf = (yfinance_daily.index >= start_dt) & (yfinance_daily.index <= end_dt)
                    yfinance_daily = yfinance_daily.loc[mask_yf]
                
                if not local_intraday.empty:
                    mask_local = (local_intraday.index >= start_dt) & (local_intraday.index <= end_dt)
                    local_intraday = local_intraday.loc[mask_local]
            
            # DailyAggregator removed - this comparison is no longer valid
            # Using Brapi.dev daily data directly instead of aggregation
            local_daily = pd.DataFrame()  # Empty DataFrame to indicate no local aggregation
            
            if yfinance_daily.empty or local_daily.empty:
                return {
                    'comparison_possible': False,
                    'reason': 'Insufficient data for comparison',
                    'yfinance_bars': len(yfinance_daily),
                    'local_bars': len(local_daily),
                    'recommendations': ['Ensure both data sources have sufficient data for comparison']
                }
            
            # Align data by common dates
            aligned_data = self._align_datasets(yfinance_daily, local_daily)
            
            if aligned_data['common_dates'] == 0:
                return {
                    'comparison_possible': False,
                    'reason': 'No overlapping dates between data sources',
                    'yfinance_range': f"{yfinance_daily.index.min().date()} to {yfinance_daily.index.max().date()}",
                    'local_range': f"{local_daily.index.min().date()} to {local_daily.index.max().date()}",
                    'recommendations': ['Check data date ranges for overlap']
                }
            
            # Perform OHLCV comparison
            ohlcv_comparison = self._compare_ohlcv_data(aligned_data['yfinance'], aligned_data['local'])
            
            # Perform technical indicators comparison
            indicators_comparison = self._compare_technical_indicators(
                aligned_data['yfinance'], aligned_data['local'], symbol
            )
            
            # Generate overall assessment
            overall_assessment = self._generate_overall_assessment(
                ohlcv_comparison, indicators_comparison, aligned_data
            )
            
            comparison_result = {
                'comparison_possible': True,
                'symbol': symbol,
                'comparison_period': {
                    'start_date': aligned_data['yfinance'].index.min().strftime('%Y-%m-%d'),
                    'end_date': aligned_data['yfinance'].index.max().strftime('%Y-%m-%d'),
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
    
    def _align_datasets(self, yfinance_data: pd.DataFrame, local_data: pd.DataFrame) -> Dict[str, Any]:
        """Align datasets by common trading dates."""
        # Find common dates
        yf_dates = set(yfinance_data.index.date)
        local_dates = set(local_data.index.date)
        common_dates = yf_dates.intersection(local_dates)
        
        if not common_dates:
            return {
                'yfinance': pd.DataFrame(),
                'local': pd.DataFrame(),
                'common_dates': 0
            }
        
        # Filter to common dates
        yf_mask = yfinance_data.index.date.isin(common_dates)
        local_mask = local_data.index.date.isin(common_dates)
        
        aligned_yf = yfinance_data.loc[yf_mask].sort_index()
        aligned_local = local_data.loc[local_mask].sort_index()
        
        return {
            'yfinance': aligned_yf,
            'local': aligned_local,
            'common_dates': len(common_dates)
        }
    
    def _compare_ohlcv_data(self, yfinance_data: pd.DataFrame, local_data: pd.DataFrame) -> Dict[str, Any]:
        """Compare OHLCV data between sources."""
        comparison = {}
        
        # Align by exact dates for point-to-point comparison
        aligned_index = yfinance_data.index.intersection(local_data.index)
        
        if len(aligned_index) == 0:
            return {'error': 'No aligned data points for OHLCV comparison'}
        
        yf_aligned = yfinance_data.loc[aligned_index]
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
    
    def _compare_technical_indicators(self, yfinance_data: pd.DataFrame, 
                                    local_data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Compare technical indicators calculated from both data sources."""
        try:
            # Calculate indicators for both datasets
            yf_indicators = self.indicators_calculator.calculate_all_indicators(
                yfinance_data, atr_period=14, ema_periods=[10, 20], rsi_period=14
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
            recommendations.append("💡 Consider using yfinance for indicators and local data for execution precision")
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