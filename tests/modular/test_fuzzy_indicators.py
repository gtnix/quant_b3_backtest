"""
Unit tests for IndicatorCalculator component.

Tests the fuzzy indicator calculation logic extracted from the monolithic strategy,
including EMA, RSI, and fuzzy scoring logic.

Author: Senior Python Developer
Date: 2025
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date
from strategies.components.fuzzy_indicators import IndicatorCalculator


class TestIndicatorCalculator:
    """Test suite for IndicatorCalculator component."""
    
    @pytest.fixture
    def indicator_calculator(self):
        """Create IndicatorCalculator instance for testing."""
        config = {
            'fuzzy_threshold': 1.5,
            'ema_periods': [3, 5, 10, 15, 20],
            'rsi_period': 14
        }
        return IndicatorCalculator(config)
    
    @pytest.fixture
    def sample_daily_data(self):
        """Create sample daily OHLC data for testing."""
        dates = pd.date_range('2025-01-01', periods=60, freq='D')
        np.random.seed(42)  # For reproducible tests
        
        # Generate realistic price data
        base_price = 25.0
        returns = np.random.normal(0, 0.02, 60)  # 2% daily volatility
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = []
        for i, (date_val, price) in enumerate(zip(dates, prices)):
            # Create OHLC with some intraday variation
            open_price = price * (1 + np.random.normal(0, 0.005))
            high_price = price * (1 + abs(np.random.normal(0, 0.01)))
            low_price = price * (1 - abs(np.random.normal(0, 0.01)))
            close_price = price
            volume = np.random.randint(10000, 100000)
            
            data.append({
                'date': date_val,
                'open': max(0.01, open_price),
                'high': max(high_price, open_price, close_price),
                'low': min(low_price, open_price, close_price),
                'close': max(0.01, close_price),
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def test_initialization(self, indicator_calculator):
        """Test IndicatorCalculator initialization."""
        assert indicator_calculator.fuzzy_threshold == 1.5
        assert indicator_calculator.ema_periods == [3, 5, 10, 15, 20]
        assert indicator_calculator.rsi_period == 14
    
    def test_calculate_indicators_guaranteed_valid_data(self, indicator_calculator, sample_daily_data):
        """Test indicator calculation with valid data."""
        symbol = "TEST3"
        current_date = date(2025, 2, 1)
        
        indicators = indicator_calculator.calculate_indicators_guaranteed(
            symbol, current_date, sample_daily_data
        )
        
        # Check all required indicators are present
        required_indicators = ['ema_3', 'ema_5', 'ema_10', 'ema_15', 'ema_20', 'rsi']
        for indicator in required_indicators:
            assert indicator in indicators
            assert not indicators[indicator].empty
            assert not indicators[indicator].iloc[-1] == 0  # Should have meaningful values
    
    def test_calculate_indicators_insufficient_data(self, indicator_calculator):
        """Test indicator calculation with insufficient data."""
        symbol = "TEST3"
        current_date = date(2025, 1, 15)
        
        # Create minimal data (less than required)
        minimal_data = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=5),
            'open': [25.0] * 5,
            'high': [25.5] * 5,
            'low': [24.5] * 5,
            'close': [25.0] * 5,
            'volume': [10000] * 5
        })
        
        indicators = indicator_calculator.calculate_indicators_guaranteed(
            symbol, current_date, minimal_data
        )
        
        # Should return empty indicators for insufficient data
        for indicator in indicators.values():
            assert indicator.empty
    
    def test_calculate_indicators_empty_data(self, indicator_calculator):
        """Test indicator calculation with empty data."""
        symbol = "TEST3"
        current_date = date(2025, 1, 15)
        empty_data = pd.DataFrame()
        
        indicators = indicator_calculator.calculate_indicators_guaranteed(
            symbol, current_date, empty_data
        )
        
        # Should return empty indicators
        for indicator in indicators.values():
            assert indicator.empty
    
    def test_indicator_validation(self, indicator_calculator, sample_daily_data):
        """Test indicator validation logic."""
        # Calculate indicators first
        indicators = indicator_calculator.calculate_indicators_guaranteed(
            "TEST3", date(2025, 2, 1), sample_daily_data
        )
        
        # Test validation
        is_valid = indicator_calculator._validate_indicators(indicators)
        assert is_valid
        
        # Test with invalid indicators (missing required indicator)
        invalid_indicators = {k: v for k, v in indicators.items() if k != 'ema_3'}
        is_valid = indicator_calculator._validate_indicators(invalid_indicators)
        assert not is_valid
    
    def test_ema_signal_calculation(self, indicator_calculator):
        """Test EMA signal calculation."""
        # Create mock indicators with known EMA values
        indicators = {
            'ema_3': pd.Series([25.0, 25.2, 25.5]),
            'ema_5': pd.Series([24.8, 25.0, 25.2]),
            'ema_10': pd.Series([24.5, 24.7, 24.9]),
            'ema_15': pd.Series([24.2, 24.4, 24.6]),
            'ema_20': pd.Series([24.0, 24.2, 24.4])
        }
        
        ema_signal = indicator_calculator._calculate_ema_signal(indicators)
        
        # Should be positive since shorter EMAs > longer EMAs (uptrend)
        assert ema_signal > 0
        assert -3.0 <= ema_signal <= 3.0  # Within expected range
    
    def test_rsi_signal_calculation(self, indicator_calculator):
        """Test RSI signal calculation."""
        # Test overbought condition (RSI > 70)
        indicators_overbought = {'rsi': pd.Series([75.0])}
        rsi_signal = indicator_calculator._calculate_rsi_signal(indicators_overbought)
        assert rsi_signal == -2.0  # Bearish signal
        
        # Test oversold condition (RSI < 30)
        indicators_oversold = {'rsi': pd.Series([25.0])}
        rsi_signal = indicator_calculator._calculate_rsi_signal(indicators_oversold)
        assert rsi_signal == 2.0  # Bullish signal
        
        # Test neutral condition
        indicators_neutral = {'rsi': pd.Series([50.0])}
        rsi_signal = indicator_calculator._calculate_rsi_signal(indicators_neutral)
        assert rsi_signal == 0.0  # Neutral
    
    def test_relative_strength_signal(self, indicator_calculator):
        """Test relative strength signal calculation."""
        # Test outperforming stock
        indicators_outperform = {
            'stock_return': pd.Series([0.05]),  # 5% gain
            'ibov_return': pd.Series([0.02])    # 2% gain
        }
        
        ibov_data = pd.DataFrame({'close': [100, 102]})  # Mock benchmark data
        rs_signal = indicator_calculator._calculate_relative_strength_signal(
            indicators_outperform, ibov_data
        )
        assert rs_signal > 0  # Should be positive for outperformance
        
        # Test underperforming stock
        indicators_underperform = {
            'stock_return': pd.Series([-0.03]),  # -3% loss
            'ibov_return': pd.Series([0.01])     # 1% gain
        }
        rs_signal = indicator_calculator._calculate_relative_strength_signal(
            indicators_underperform, ibov_data
        )
        assert rs_signal < 0  # Should be negative for underperformance
    
    def test_fuzzy_score_calculation(self, indicator_calculator, sample_daily_data):
        """Test complete fuzzy score calculation."""
        symbol = "TEST3"
        current_date = date(2025, 2, 1)
        
        # Calculate indicators
        indicators = indicator_calculator.calculate_indicators_guaranteed(
            symbol, current_date, sample_daily_data
        )
        
        # Calculate fuzzy score
        fuzzy_score, components = indicator_calculator.calculate_fuzzy_score(
            symbol, current_date, indicators
        )
        
        # Check score is within reasonable bounds
        assert isinstance(fuzzy_score, float)
        assert -10.0 <= fuzzy_score <= 10.0  # Reasonable bounds
        
        # Check components exist
        assert 'ema_signal' in components
        assert 'rsi_signal' in components
        assert 'relative_strength' in components
    
    def test_signal_type_determination(self, indicator_calculator):
        """Test signal type determination from fuzzy scores."""
        # Test BUY signal
        assert indicator_calculator.is_buy_signal(2.0)
        assert indicator_calculator.get_signal_type(2.0) == 'BUY'
        
        # Test SELL signal
        assert indicator_calculator.is_sell_signal(-2.0)
        assert indicator_calculator.get_signal_type(-2.0) == 'SELL'
        
        # Test HOLD signal
        assert indicator_calculator.is_hold_signal(0.5)
        assert indicator_calculator.get_signal_type(0.5) == 'HOLD'
        
        # Test threshold boundaries
        assert indicator_calculator.is_buy_signal(1.5)  # Exactly at threshold
        assert indicator_calculator.is_sell_signal(-1.5)  # Exactly at threshold
        assert indicator_calculator.is_hold_signal(1.49)  # Just below threshold
    
    def test_has_excessive_nan_gaps(self, indicator_calculator):
        """Test NaN gap detection."""
        # Series with acceptable NaN ratio
        good_series = pd.Series([1, 2, np.nan, 4, 5])
        assert not indicator_calculator._has_excessive_nan_gaps(good_series)
        
        # Series with excessive NaN ratio
        bad_series = pd.Series([1, np.nan, np.nan, np.nan, np.nan])
        assert indicator_calculator._has_excessive_nan_gaps(bad_series)
        
        # Empty series
        empty_series = pd.Series(dtype=float)
        assert indicator_calculator._has_excessive_nan_gaps(empty_series)
    
    def test_ema_convergence_tracking(self, indicator_calculator, sample_daily_data):
        """Test EMA convergence issue tracking."""
        symbol = "TEST3"
        current_date = date(2025, 2, 1)
        
        # Create indicators with some NaN values to trigger convergence warning
        indicators = {
            'ema_3': pd.Series([25.0, 25.2, np.nan, 25.5, 25.7]),
            'ema_5': pd.Series([24.8, 25.0, 25.2, 25.3, 25.4]),
            'ema_10': pd.Series([24.5, 24.7, 24.9, 25.0, 25.1]),
            'ema_15': pd.Series([24.2, 24.4, 24.6, 24.7, 24.8]),
            'ema_20': pd.Series([24.0, 24.2, 24.4, 24.5, 24.6]),
            'rsi': pd.Series([45, 47, 49, 51, 53])
        }
        
        # This should track convergence issues internally
        indicator_calculator._check_ema_convergence(symbol, current_date, indicators)
        
        # Verify tracking structure exists
        assert symbol in indicator_calculator.ema_nonconverged_reported_dates
    
    def test_get_stats(self, indicator_calculator):
        """Test statistics retrieval."""
        stats = indicator_calculator.get_stats()
        
        assert 'ema_periods' in stats
        assert 'rsi_period' in stats
        assert 'fuzzy_threshold' in stats
        assert 'symbols_with_convergence_issues' in stats
        
        assert stats['ema_periods'] == [3, 5, 10, 15, 20]
        assert stats['rsi_period'] == 14
        assert stats['fuzzy_threshold'] == 1.5
    
    def test_error_handling_in_calculations(self, indicator_calculator):
        """Test error handling in various calculation methods."""
        # Test with malformed indicators
        bad_indicators = {'invalid': 'not_a_series'}
        
        # Should not raise exceptions, should return safe defaults
        ema_signal = indicator_calculator._calculate_ema_signal(bad_indicators)
        assert ema_signal == 0.0
        
        rsi_signal = indicator_calculator._calculate_rsi_signal(bad_indicators)
        assert rsi_signal == 0.0
        
        rs_signal = indicator_calculator._calculate_relative_strength_signal(bad_indicators, None)
        assert rs_signal == 0.0
