"""
Unit tests for DataManager component.

Tests the data management logic extracted from the monolithic strategy,
including BRAPI provider management, data validation, and caching.

Author: Senior Python Developer
Date: 2025
"""

import pytest
import pandas as pd
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch
from strategies.components.data_management import DataManager


class TestDataManager:
    """Test suite for DataManager component."""
    
    @pytest.fixture
    def mock_context(self):
        """Create mock context for testing."""
        context = Mock()
        context.logger = Mock()
        return context
    
    @pytest.fixture
    def data_manager_config(self):
        """Create configuration for DataManager."""
        return {
            'brapi_token': 'test_token_123',
            'brapi_cache_dir': 'test_cache',
            'data_quality_threshold': 0.8
        }
    
    @pytest.fixture
    def data_manager(self, data_manager_config, mock_context):
        """Create DataManager instance for testing."""
        with patch('strategies.components.data_management.BrapiProvider'):
            return DataManager(data_manager_config, mock_context)
    
    @pytest.fixture
    def sample_daily_data(self):
        """Create sample daily data for testing."""
        dates = pd.date_range('2025-01-01', periods=30, freq='D')
        data = []
        
        for i, date_val in enumerate(dates):
            price = 25.0 + i * 0.1  # Trending price
            data.append({
                'date': date_val,
                'open': price * 0.99,
                'high': price * 1.02,
                'low': price * 0.98,
                'close': price,
                'volume': 10000 + i * 100
            })
        
        return pd.DataFrame(data)
    
    def test_initialization_with_token(self, data_manager_config, mock_context):
        """Test DataManager initialization with valid BRAPI token."""
        with patch('strategies.components.data_management.BrapiProvider') as mock_brapi:
            dm = DataManager(data_manager_config, mock_context)
            
            assert dm.config == data_manager_config
            assert dm.context == mock_context
            assert isinstance(dm.daily_data_cache, dict)
            assert isinstance(dm.last_daily_update, dict)
            mock_brapi.assert_called_once()
    
    def test_initialization_without_token(self, mock_context):
        """Test DataManager initialization without BRAPI token."""
        config_no_token = {'other_setting': 'value'}
        
        with patch('strategies.components.data_management.BrapiProvider'):
            dm = DataManager(config_no_token, mock_context)
            assert dm.brapi_provider is None
    
    def test_validate_and_clean_daily_data_valid(self, data_manager, sample_daily_data):
        """Test data validation and cleaning with valid data."""
        cleaned_data = data_manager._validate_and_clean_daily_data(sample_daily_data, "TEST3")
        
        assert len(cleaned_data) == len(sample_daily_data)
        assert all(cleaned_data['open'] > 0)
        assert all(cleaned_data['high'] >= cleaned_data['low'])
        assert all(cleaned_data['high'] >= cleaned_data['open'])
        assert all(cleaned_data['high'] >= cleaned_data['close'])
    
    def test_validate_and_clean_daily_data_invalid(self, data_manager):
        """Test data validation and cleaning with invalid data."""
        invalid_data = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=5),
            'open': [25.0, 0.0, -5.0, 30.0, 35.0],  # Invalid: zero and negative
            'high': [26.0, 25.0, 30.0, 32.0, 37.0],
            'low': [24.0, 23.0, 28.0, 29.0, 34.0],
            'close': [25.5, 24.5, 29.5, 31.0, 36.0],
            'volume': [10000, 12000, 8000, 15000, 11000]
        })
        
        cleaned_data = data_manager._validate_and_clean_daily_data(invalid_data, "TEST3")
        
        # Should remove rows with invalid prices
        assert len(cleaned_data) < len(invalid_data)
        assert all(cleaned_data['open'] > 0)
        assert all(cleaned_data['high'] > 0)
        assert all(cleaned_data['low'] > 0)
        assert all(cleaned_data['close'] > 0)
    
    def test_validate_and_clean_empty_data(self, data_manager):
        """Test data validation with empty DataFrame."""
        empty_data = pd.DataFrame()
        cleaned_data = data_manager._validate_and_clean_daily_data(empty_data, "TEST3")
        
        assert cleaned_data.empty
    
    def test_is_new_trading_day(self, data_manager):
        """Test new trading day detection."""
        symbol = "TEST3"
        today = date.today()
        
        # First call should be new trading day
        assert data_manager.is_new_trading_day(symbol, today)
        
        # Second call same day should not be new
        assert not data_manager.is_new_trading_day(symbol, today)
        
        # Next day should be new trading day
        tomorrow = today + timedelta(days=1)
        assert data_manager.is_new_trading_day(symbol, tomorrow)
    
    def test_should_update_daily_data_new_day(self, data_manager):
        """Test data update decision for new trading day."""
        symbol = "TEST3"
        today = date.today()
        
        # Should update on new trading day
        assert data_manager.should_update_daily_data(symbol, today)
    
    def test_should_update_daily_data_cached(self, data_manager, sample_daily_data):
        """Test data update decision with cached data."""
        symbol = "TEST3"
        today = date.today()
        
        # Populate cache
        cache_key = f"{symbol}_{today}"
        sample_daily_data._cache_time = datetime.now()
        data_manager.daily_data_cache[cache_key] = sample_daily_data
        data_manager.last_daily_update[symbol] = today
        
        # Should not update if recently cached
        assert not data_manager.should_update_daily_data(symbol, today)
    
    def test_get_daily_data_for_date_cached(self, data_manager, sample_daily_data):
        """Test getting daily data from cache."""
        symbol = "TEST3"
        target_date = date(2025, 1, 15)
        
        # Populate cache
        cache_key = f"{symbol}_{target_date}"
        data_manager.daily_data_cache[cache_key] = sample_daily_data
        
        result = data_manager.get_daily_data_for_date(symbol, target_date)
        
        assert not result.empty
        assert len(result) == len(sample_daily_data)
    
    def test_get_daily_data_for_date_no_provider(self, data_manager_config, mock_context):
        """Test getting daily data without BRAPI provider."""
        # Create DataManager without BRAPI provider
        dm = DataManager({}, mock_context)  # No token
        
        symbol = "TEST3"
        target_date = date(2025, 1, 15)
        
        result = dm.get_daily_data_for_date(symbol, target_date)
        
        assert result.empty
    
    @patch('strategies.components.data_management.BrapiProvider')
    def test_fetch_new_daily_data_success(self, mock_brapi_class, data_manager_config, mock_context, sample_daily_data):
        """Test successful fetching of new daily data."""
        # Setup mock BRAPI provider
        mock_brapi_instance = Mock()
        mock_brapi_instance.get_daily_data.return_value = sample_daily_data
        mock_brapi_class.return_value = mock_brapi_instance
        
        dm = DataManager(data_manager_config, mock_context)
        
        symbol = "TEST3"
        from_date = date(2025, 1, 1)
        to_date = date(2025, 1, 30)
        
        result = dm.fetch_new_daily_data(symbol, from_date, to_date)
        
        assert not result.empty
        mock_brapi_instance.get_daily_data.assert_called_once()
    
    @patch('strategies.components.data_management.BrapiProvider')
    def test_fetch_new_daily_data_empty_response(self, mock_brapi_class, data_manager_config, mock_context):
        """Test fetching daily data with empty response."""
        # Setup mock BRAPI provider to return empty data
        mock_brapi_instance = Mock()
        mock_brapi_instance.get_daily_data.return_value = pd.DataFrame()
        mock_brapi_class.return_value = mock_brapi_instance
        
        dm = DataManager(data_manager_config, mock_context)
        
        symbol = "TEST3"
        from_date = date(2025, 1, 1)
        to_date = date(2025, 1, 30)
        
        result = dm.fetch_new_daily_data(symbol, from_date, to_date)
        
        assert result.empty
    
    def test_refresh_daily_data_not_needed(self, data_manager, sample_daily_data):
        """Test refresh when data update is not needed."""
        symbol = "TEST3"
        current_date = date.today()
        
        # Setup cache to indicate no update needed
        cache_key = f"{symbol}_{current_date}"
        sample_daily_data._cache_time = datetime.now()
        data_manager.daily_data_cache[cache_key] = sample_daily_data
        data_manager.last_daily_update[symbol] = current_date
        
        result = data_manager.refresh_daily_data(symbol, current_date)
        
        assert result is True  # Should return True for "already fresh" data
    
    def test_get_daily_data_up_to_date(self, data_manager, sample_daily_data):
        """Test getting daily data up to a specific date."""
        symbol = "TEST3"
        end_date = date(2025, 1, 30)
        
        # Mock successful refresh
        with patch.object(data_manager, 'refresh_daily_data', return_value=True):
            with patch.object(data_manager, 'get_daily_data_for_date', return_value=sample_daily_data):
                result = data_manager.get_daily_data_up_to_date(symbol, end_date)
                
                assert not result.empty
                assert len(result) == len(sample_daily_data)
    
    def test_clear_cache_specific_symbol(self, data_manager, sample_daily_data):
        """Test clearing cache for specific symbol."""
        # Populate cache with multiple symbols
        data_manager.daily_data_cache["TEST3_2025-01-01"] = sample_daily_data
        data_manager.daily_data_cache["TEST4_2025-01-01"] = sample_daily_data
        data_manager.daily_data_cache["TEST3_2025-01-02"] = sample_daily_data
        
        # Clear cache for TEST3 only
        data_manager.clear_cache("TEST3")
        
        # TEST3 entries should be removed, TEST4 should remain
        remaining_keys = list(data_manager.daily_data_cache.keys())
        assert all("TEST3" not in key for key in remaining_keys)
        assert any("TEST4" in key for key in remaining_keys)
    
    def test_clear_cache_all_symbols(self, data_manager, sample_daily_data):
        """Test clearing cache for all symbols."""
        # Populate cache
        data_manager.daily_data_cache["TEST3_2025-01-01"] = sample_daily_data
        data_manager.daily_data_cache["TEST4_2025-01-01"] = sample_daily_data
        data_manager.last_daily_update["TEST3"] = date.today()
        
        # Clear all cache
        data_manager.clear_cache()
        
        assert len(data_manager.daily_data_cache) == 0
        assert len(data_manager.last_daily_update) == 0
    
    def test_get_cache_stats(self, data_manager, sample_daily_data):
        """Test cache statistics retrieval."""
        # Populate cache with test data
        data_manager.daily_data_cache["TEST3_2025-01-01"] = sample_daily_data
        data_manager.daily_data_cache["TEST3_2025-01-02"] = sample_daily_data
        data_manager.daily_data_cache["TEST4_2025-01-01"] = sample_daily_data
        data_manager.last_daily_update["TEST3"] = date.today()
        data_manager.last_daily_update["TEST4"] = date.today()
        
        stats = data_manager.get_cache_stats()
        
        assert stats['cached_symbols'] == 2  # TEST3 and TEST4
        assert stats['total_cache_entries'] == 3
        assert stats['symbols_with_updates'] == 2
        assert 'brapi_provider_available' in stats
    
    def test_log_data_quality_report_with_data(self, data_manager, sample_daily_data):
        """Test data quality report generation with valid data."""
        symbol = "TEST3"
        
        with patch.object(data_manager, 'get_daily_data_up_to_date', return_value=sample_daily_data):
            # Should not raise exception
            data_manager.log_data_quality_report(symbol)
            
            # Verify logger was called (quality report logged)
            assert data_manager.context.logger.info.called
    
    def test_log_data_quality_report_no_data(self, data_manager):
        """Test data quality report generation with no data."""
        symbol = "TEST3"
        
        with patch.object(data_manager, 'get_daily_data_up_to_date', return_value=pd.DataFrame()):
            data_manager.log_data_quality_report(symbol)
            
            # Should log warning about no data
            data_manager.context.logger.warning.assert_called()
    
    def test_error_handling_in_data_operations(self, data_manager):
        """Test error handling in various data operations."""
        symbol = "TEST3"
        target_date = date(2025, 1, 15)
        
        # Test error handling in get_daily_data_for_date
        with patch.object(data_manager, 'brapi_provider', None):
            result = data_manager.get_daily_data_for_date(symbol, target_date)
            assert result.empty
        
        # Test error handling in data quality report
        with patch.object(data_manager, 'get_daily_data_up_to_date', side_effect=Exception("Test error")):
            # Should not raise exception
            data_manager.log_data_quality_report(symbol)
