"""
Fuzzy Indicator Calculation Component

Handles all indicator calculations, validation, and fuzzy scoring logic.
Extracted from the monolithic strategy class for better maintainability.

Author: Senior Python Developer
Date: 2025
"""

import logging
import pandas as pd
import numpy as np
from datetime import date
from typing import Dict, Any, Optional, Tuple
from engine.market_utils import DailyTechnicalIndicators
from engine.logging_config import get_logger

logger = get_logger(__name__)


class IndicatorCalculator:
    """
    Handles all indicator calculations for the FuzzyFajuto strategy.
    
    Responsibilities:
    - EMA calculation and validation
    - RSI calculation
    - Fuzzy score computation
    - Indicator validation and quality checks
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize indicator calculator with configuration."""
        self.config = config
        
        # Indicator configuration
        self.ema_periods = [3, 5, 10, 15, 20]
        self.rsi_period = 14
        self.fuzzy_threshold = config.get('fuzzy_threshold', 1.5)
        
        # Quality tracking
        self.ema_nonconverged_reported_dates: Dict[str, set] = {}
        
        logger.debug(f"IndicatorCalculator initialized with fuzzy_threshold={self.fuzzy_threshold}")
    
    def calculate_indicators_guaranteed(self, symbol: str, current_date: date, 
                                     daily_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate all required indicators with guaranteed output.
        
        Args:
            symbol: Stock symbol
            current_date: Current date
            daily_data: Daily OHLC data
            
        Returns:
            Dictionary containing all calculated indicators
        """
        try:
            if daily_data.empty:
                logger.warning(f"No daily data available for {symbol} indicator calculation")
                return self._get_empty_indicators()
            
            # Ensure we have enough data
            min_required_rows = max(self.ema_periods) + self.rsi_period + 5
            if len(daily_data) < min_required_rows:
                logger.warning(f"Insufficient data for {symbol}: {len(daily_data)} < {min_required_rows} rows")
                return self._get_empty_indicators()
            
            # Calculate technical indicators
            indicators = DailyTechnicalIndicators.compute_daily_vectors(
                daily_data, 
                ema_periods=self.ema_periods,
                rsi_period=self.rsi_period
            )
            
            # Validate indicators
            if not self._validate_indicators(indicators):
                logger.warning(f"Indicator validation failed for {symbol}")
                return self._get_empty_indicators()
            
            # Check for convergence issues
            self._check_ema_convergence(symbol, current_date, indicators)
            
            logger.debug(f"Successfully calculated indicators for {symbol}")
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return self._get_empty_indicators()
    
    def _get_empty_indicators(self) -> Dict[str, pd.Series]:
        """Get empty indicator dictionary for fallback."""
        return {
            'ema_3': pd.Series(dtype=float),
            'ema_5': pd.Series(dtype=float),
            'ema_10': pd.Series(dtype=float),
            'ema_15': pd.Series(dtype=float),
            'ema_20': pd.Series(dtype=float),
            'rsi': pd.Series(dtype=float),
            'stock_return': pd.Series(dtype=float),
            'ibov_return': pd.Series(dtype=float)
        }
    
    def _validate_indicators(self, indicators: Dict[str, pd.Series]) -> bool:
        """
        Validate calculated indicators for quality and completeness.
        
        Args:
            indicators: Dictionary of calculated indicators
            
        Returns:
            True if indicators are valid
        """
        try:
            required_indicators = ['ema_3', 'ema_5', 'ema_10', 'ema_15', 'ema_20', 'rsi']
            
            # Check all required indicators exist
            for indicator in required_indicators:
                if indicator not in indicators:
                    logger.warning(f"Missing required indicator: {indicator}")
                    return False
                
                series = indicators[indicator]
                if series.empty:
                    logger.warning(f"Empty indicator series: {indicator}")
                    return False
                
                # Check for excessive NaN values
                if self._has_excessive_nan_gaps(series):
                    logger.warning(f"Excessive NaN gaps in {indicator}")
                    return False
            
            # Validate EMA ordering (shorter periods should be more responsive)
            last_close = indicators.get('close', pd.Series()).iloc[-1] if 'close' in indicators else None
            if last_close is not None:
                ema_values = {
                    period: indicators[f'ema_{period}'].iloc[-1] 
                    for period in self.ema_periods
                    if not pd.isna(indicators[f'ema_{period}'].iloc[-1])
                }
                
                if len(ema_values) >= 2:
                    # EMAs should be reasonably close to current price
                    for period, ema_val in ema_values.items():
                        price_deviation = abs(ema_val - last_close) / last_close
                        if price_deviation > 0.5:  # 50% deviation threshold
                            logger.warning(f"EMA_{period} deviates significantly from price: {price_deviation:.1%}")
                            return False
            
            # Validate RSI range
            rsi_series = indicators['rsi']
            if not rsi_series.empty:
                rsi_min, rsi_max = rsi_series.min(), rsi_series.max()
                if rsi_min < 0 or rsi_max > 100:
                    logger.warning(f"RSI values outside valid range: {rsi_min:.2f} - {rsi_max:.2f}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating indicators: {e}")
            return False
    
    def _has_excessive_nan_gaps(self, series: pd.Series, max_gap_ratio: float = 0.3) -> bool:
        """
        Check if a series has excessive NaN gaps.
        
        Args:
            series: Series to check
            max_gap_ratio: Maximum allowed ratio of NaN values
            
        Returns:
            True if series has excessive NaN gaps
        """
        if series.empty:
            return True
        
        nan_count = series.isna().sum()
        total_count = len(series)
        nan_ratio = nan_count / total_count
        
        return nan_ratio > max_gap_ratio
    
    def _check_ema_convergence(self, symbol: str, current_date: date, 
                             indicators: Dict[str, pd.Series]):
        """
        Check EMA convergence and log warnings for non-converged indicators.
        
        Args:
            symbol: Stock symbol
            current_date: Current date
            indicators: Calculated indicators
        """
        try:
            # Initialize tracking for this symbol
            if symbol not in self.ema_nonconverged_reported_dates:
                self.ema_nonconverged_reported_dates[symbol] = set()
            
            # Check each EMA for convergence
            for period in self.ema_periods:
                ema_key = f'ema_{period}'
                if ema_key not in indicators:
                    continue
                
                ema_series = indicators[ema_key]
                if ema_series.empty:
                    continue
                
                # Check if EMA has converged (no NaN in recent values)
                recent_values = ema_series.tail(5)  # Check last 5 values
                if recent_values.isna().any():
                    # Only log once per symbol per date
                    date_key = f"{current_date}_{period}"
                    if date_key not in self.ema_nonconverged_reported_dates[symbol]:
                        logger.warning(f"EMA_{period} not fully converged for {symbol} on {current_date}")
                        self.ema_nonconverged_reported_dates[symbol].add(date_key)
                        
        except Exception as e:
            logger.error(f"Error checking EMA convergence for {symbol}: {e}")
    
    def calculate_fuzzy_score(self, symbol: str, current_date: date, 
                            indicators: Dict[str, pd.Series], 
                            ibov_data: Optional[pd.DataFrame] = None) -> Tuple[float, Dict[str, float]]:
        """
        Calculate fuzzy score based on indicators and benchmark comparison.
        
        Args:
            symbol: Stock symbol
            current_date: Current date
            indicators: Calculated technical indicators
            ibov_data: Optional Ibovespa benchmark data
            
        Returns:
            Tuple of (fuzzy_score, component_scores)
        """
        try:
            # Initialize component scores
            components = {
                'ema_signal': 0.0,
                'rsi_signal': 0.0,
                'relative_strength': 0.0
            }
            
            # Get latest values
            if not indicators or all(series.empty for series in indicators.values()):
                logger.warning(f"No valid indicators for fuzzy score calculation: {symbol}")
                return 0.0, components
            
            # EMA trend signal (shorter vs longer EMAs)
            ema_signal = self._calculate_ema_signal(indicators)
            components['ema_signal'] = ema_signal
            
            # RSI momentum signal
            rsi_signal = self._calculate_rsi_signal(indicators)
            components['rsi_signal'] = rsi_signal
            
            # Relative strength vs benchmark
            rs_signal = self._calculate_relative_strength_signal(indicators, ibov_data)
            components['relative_strength'] = rs_signal
            
            # Combine signals with weights
            fuzzy_score = (
                ema_signal * 0.4 +      # 40% weight on trend
                rsi_signal * 0.3 +      # 30% weight on momentum  
                rs_signal * 0.3         # 30% weight on relative strength
            )
            
            logger.debug(f"Fuzzy score for {symbol}: {fuzzy_score:.3f} "
                        f"(EMA: {ema_signal:.3f}, RSI: {rsi_signal:.3f}, RS: {rs_signal:.3f})")
            
            return fuzzy_score, components
            
        except Exception as e:
            logger.error(f"Error calculating fuzzy score for {symbol}: {e}")
            return 0.0, components
    
    def _calculate_ema_signal(self, indicators: Dict[str, pd.Series]) -> float:
        """
        Calculate EMA trend signal based on EMA relationships.
        
        Args:
            indicators: Technical indicators
            
        Returns:
            EMA signal strength (-3 to +3)
        """
        try:
            # Get latest EMA values
            ema_values = {}
            for period in self.ema_periods:
                ema_key = f'ema_{period}'
                if ema_key in indicators and not indicators[ema_key].empty:
                    latest_val = indicators[ema_key].iloc[-1]
                    if not pd.isna(latest_val):
                        ema_values[period] = latest_val
            
            if len(ema_values) < 3:
                return 0.0
            
            # Calculate signal based on EMA ordering
            signal = 0.0
            
            # Short-term vs medium-term trend
            if 3 in ema_values and 10 in ema_values:
                if ema_values[3] > ema_values[10]:
                    signal += 1.0
                else:
                    signal -= 1.0
            
            # Medium-term vs long-term trend
            if 10 in ema_values and 20 in ema_values:
                if ema_values[10] > ema_values[20]:
                    signal += 1.0
                else:
                    signal -= 1.0
            
            # Overall trend strength
            if 5 in ema_values and 15 in ema_values:
                trend_strength = (ema_values[5] - ema_values[15]) / ema_values[15]
                signal += np.clip(trend_strength * 5, -1.0, 1.0)  # Scale and clip
            
            return np.clip(signal, -3.0, 3.0)
            
        except Exception as e:
            logger.error(f"Error calculating EMA signal: {e}")
            return 0.0
    
    def _calculate_rsi_signal(self, indicators: Dict[str, pd.Series]) -> float:
        """
        Calculate RSI momentum signal.
        
        Args:
            indicators: Technical indicators
            
        Returns:
            RSI signal strength (-2 to +2)
        """
        try:
            if 'rsi' not in indicators or indicators['rsi'].empty:
                return 0.0
            
            rsi_latest = indicators['rsi'].iloc[-1]
            if pd.isna(rsi_latest):
                return 0.0
            
            # RSI signal based on overbought/oversold levels
            if rsi_latest >= 70:
                return -2.0  # Overbought, bearish signal
            elif rsi_latest <= 30:
                return 2.0   # Oversold, bullish signal
            elif rsi_latest >= 60:
                return -1.0  # Moderately overbought
            elif rsi_latest <= 40:
                return 1.0   # Moderately oversold
            else:
                return 0.0   # Neutral zone
                
        except Exception as e:
            logger.error(f"Error calculating RSI signal: {e}")
            return 0.0
    
    def _calculate_relative_strength_signal(self, indicators: Dict[str, pd.Series], 
                                          ibov_data: Optional[pd.DataFrame]) -> float:
        """
        Calculate relative strength signal vs benchmark.
        
        Args:
            indicators: Technical indicators
            ibov_data: Benchmark data
            
        Returns:
            Relative strength signal (-2 to +2)
        """
        try:
            # Check if we have stock return data
            if 'stock_return' not in indicators or indicators['stock_return'].empty:
                return 0.0
            
            stock_return = indicators['stock_return'].iloc[-1]
            if pd.isna(stock_return):
                return 0.0
            
            # If no benchmark data, use absolute return
            if ibov_data is None or ibov_data.empty:
                if stock_return > 0.02:  # > 2% gain
                    return 1.0
                elif stock_return < -0.02:  # < -2% loss
                    return -1.0
                else:
                    return 0.0
            
            # Calculate benchmark return
            if 'ibov_return' in indicators and not indicators['ibov_return'].empty:
                ibov_return = indicators['ibov_return'].iloc[-1]
                if not pd.isna(ibov_return):
                    relative_performance = stock_return - ibov_return
                    
                    # Scale relative performance to signal
                    if relative_performance > 0.01:  # Outperforming by >1%
                        return 2.0
                    elif relative_performance > 0.005:  # Outperforming by >0.5%
                        return 1.0
                    elif relative_performance < -0.01:  # Underperforming by >1%
                        return -2.0
                    elif relative_performance < -0.005:  # Underperforming by >0.5%
                        return -1.0
                    else:
                        return 0.0
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating relative strength signal: {e}")
            return 0.0
    
    def is_buy_signal(self, fuzzy_score: float) -> bool:
        """Check if fuzzy score indicates a buy signal."""
        return fuzzy_score >= self.fuzzy_threshold
    
    def is_sell_signal(self, fuzzy_score: float) -> bool:
        """Check if fuzzy score indicates a sell signal."""
        return fuzzy_score <= -self.fuzzy_threshold
    
    def is_hold_signal(self, fuzzy_score: float) -> bool:
        """Check if fuzzy score indicates a hold signal."""
        return abs(fuzzy_score) < self.fuzzy_threshold
    
    def get_signal_type(self, fuzzy_score: float) -> str:
        """Get signal type string based on fuzzy score."""
        if self.is_buy_signal(fuzzy_score):
            return 'BUY'
        elif self.is_sell_signal(fuzzy_score):
            return 'SELL'
        else:
            return 'HOLD'
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indicator calculator statistics."""
        return {
            'ema_periods': self.ema_periods,
            'rsi_period': self.rsi_period,
            'fuzzy_threshold': self.fuzzy_threshold,
            'symbols_with_convergence_issues': len(self.ema_nonconverged_reported_dates)
        }
