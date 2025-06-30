"""
Unit tests for SGS Data Loader module.

This module tests:
- Data fetching from Banco Central SGS API
- LOCF normalization functionality
- Data quality validation
- Error handling and edge cases
- Integration with the backtesting framework

Author: quant_b3_backtest team
Date: 2025
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
import sys
import os
import pandas_market_calendars as mcal

# Add the engine directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'engine'))

from sgs_data_loader import SGSDataLoader


class TestSGSDataLoader(unittest.TestCase):
    """Test cases for SGSDataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.data_path = os.path.join(self.test_dir, "data", "sgs")
        
        # Create test configuration
        self.config_path = os.path.join(self.test_dir, "test_config.yaml")
        self._create_test_config()
        
        # Initialize loader with test paths
        self.loader = SGSDataLoader(
            data_path=self.data_path,
            config_path=self.config_path
        )
        
        # Test dates
        self.start_date = "01/01/2024"
        self.end_date = "31/12/2024"
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def _create_test_config(self):
        """Create a test configuration file."""
        config_content = """
market:
  trading_hours:
    open: "10:00"
    close: "17:00"
  timezone: "America/Sao_Paulo"
  selic_rate: 0.15
  trading_days_per_year: 252

costs:
  brokerage_fee: 0.0
  emolument: 0.00005
  settlement_swing_trade: 0.00025

taxes:
  swing_trade: 0.15
  day_trade: 0.20
  exemption_limit: 20000

portfolio:
  initial_cash: 100000
  max_positions: 10
  position_sizing: "equal_weight"

sgs:
  series:
    11: "Selic Interest Rate"
    12: "CDI Interest Rate"
    433: "IPCA Inflation Index"
  api:
    base_url: "http://api.bcb.gov.br/dados/serie/bcdata.sgs"
    timeout: 30
    max_retries: 3
    user_agent: "quant_b3_backtest/1.0"
  processing:
    cache_enabled: true
    save_processed: true
    data_path: "data/sgs"
    normalization_method: "LOCF"
  validation:
    enable_quality_checks: true
    interest_rate_range: [0, 100]
    inflation_range: [-50, 100]
    min_data_points: 10
"""
        with open(self.config_path, 'w') as f:
            f.write(config_content)
    
    def test_initialization(self):
        """Test SGSDataLoader initialization."""
        self.assertIsNotNone(self.loader)
        self.assertEqual(self.loader.base_url, "http://api.bcb.gov.br/dados/serie/bcdata.sgs")
        self.assertIsInstance(self.loader.SGS_SERIES, dict)
        self.assertEqual(len(self.loader.SGS_SERIES), 3)
    
    def test_sgs_series_definitions(self):
        """Test that all required SGS series are defined."""
        expected_series = {11, 12, 433}
        self.assertEqual(set(self.loader.SGS_SERIES.keys()), expected_series)
        
        # Check series names
        self.assertEqual(self.loader.SGS_SERIES[11], "Selic Interest Rate")
        self.assertEqual(self.loader.SGS_SERIES[12], "CDI Interest Rate")
        self.assertEqual(self.loader.SGS_SERIES[433], "IPCA Inflation Index")
    
    def test_invalid_series_id(self):
        """Test handling of invalid series ID."""
        result = self.loader.fetch_series_data(999, self.start_date, self.end_date)
        self.assertIsNone(result)
    
    def test_normalize_data_locf_empty_dataframe(self):
        """Test LOCF normalization with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = self.loader.normalize_data_locf(empty_df, self.start_date, self.end_date)
        self.assertTrue(result.empty)
    
    def test_normalize_data_locf_none_dataframe(self):
        """Test LOCF normalization with None DataFrame."""
        result = self.loader.normalize_data_locf(None, self.start_date, self.end_date)
        self.assertTrue(result.empty)
    
    def test_normalize_data_locf_with_sample_data(self):
        """Test LOCF normalization with sample data."""
        # Create sample data with gaps
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        values = [1.0, 2.0, np.nan, 4.0, np.nan, np.nan, 7.0, 8.0, 9.0, 10.0]
        
        sample_df = pd.DataFrame({
            'valor': values
        }, index=dates)
        
        # Apply LOCF normalization
        result = self.loader.normalize_data_locf(sample_df, "01/01/2024", "10/01/2024")
        
        # Get B3 trading days for the period
        b3 = mcal.get_calendar('BVMF')
        schedule = b3.schedule(start_date='2024-01-01', end_date='2024-01-10')
        b3_days = schedule.index
        
        # Check that all NaN values are filled
        self.assertFalse(result['valor'].isna().any())
        
        # Check that result index matches B3 trading days
        self.assertTrue(result.index.equals(b3_days))
        
        # Check that LOCF was applied correctly for a few B3 trading days
        # Find the first three B3 trading days after 2024-01-01
        b3_days_list = list(b3_days)
        if len(b3_days_list) >= 4:
            self.assertEqual(result.loc[b3_days_list[1], 'valor'], 2.0)  # Should be filled with previous value
            self.assertEqual(result.loc[b3_days_list[2], 'valor'], 4.0)  # Should be filled with previous value
            self.assertEqual(result.loc[b3_days_list[3], 'valor'], 4.0)  # Should be filled with previous value
    
    def test_validate_data_quality_empty_dataframe(self):
        """Test data quality validation with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = self.loader.validate_data_quality(empty_df, 11)
        self.assertFalse(result)
    
    def test_validate_data_quality_none_dataframe(self):
        """Test data quality validation with None DataFrame."""
        result = self.loader.validate_data_quality(None, 11)
        self.assertFalse(result)
    
    def test_validate_data_quality_interest_rates(self):
        """Test data quality validation for interest rate series."""
        # Create sample interest rate data
        dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='D')
        values = [10.5, 10.75, 11.0, 10.25, 10.5]  # Reasonable interest rates
        
        sample_df = pd.DataFrame({
            'valor': values
        }, index=dates)
        
        result = self.loader.validate_data_quality(sample_df, 11)  # Selic
        self.assertTrue(result)
    
    def test_validate_data_quality_extreme_interest_rates(self):
        """Test data quality validation with extreme interest rate values."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='D')
        values = [10.5, 150.0, 11.0, -5.0, 10.5]  # Extreme values
        
        sample_df = pd.DataFrame({
            'valor': values
        }, index=dates)
        
        result = self.loader.validate_data_quality(sample_df, 11)  # Selic
        self.assertTrue(result)  # Should still pass but log warnings
    
    def test_validate_data_quality_ipca(self):
        """Test data quality validation for IPCA series."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='D')
        values = [0.5, 0.75, 1.0, 0.25, 0.5]  # Reasonable inflation rates
        
        sample_df = pd.DataFrame({
            'valor': values
        }, index=dates)
        
        result = self.loader.validate_data_quality(sample_df, 433)  # IPCA
        self.assertTrue(result)
    
    def test_save_and_load_processed_data(self):
        """Test saving and loading processed data."""
        # Create sample data
        dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='D')
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        sample_df = pd.DataFrame({
            'valor': values
        }, index=dates)
        
        # Save data
        save_result = self.loader.save_processed_data(sample_df, 11, self.start_date, self.end_date)
        self.assertTrue(save_result)
        
        # Load data
        loaded_df = self.loader.load_processed_data(11, self.start_date, self.end_date)
        self.assertIsNotNone(loaded_df)
        self.assertTrue(sample_df.equals(loaded_df))
    
    def test_load_nonexistent_processed_data(self):
        """Test loading non-existent processed data."""
        result = self.loader.load_processed_data(999, self.start_date, self.end_date)
        self.assertIsNone(result)
    
    def test_get_available_processed_files(self):
        """Test getting list of available processed files."""
        # Create some test files
        test_files = [
            "sgs_11_01012024_31122024.csv",
            "sgs_12_01012024_31122024.csv"
        ]
        
        for filename in test_files:
            filepath = Path(self.data_path) / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.touch()
        
        available_files = self.loader.get_available_processed_files()
        self.assertEqual(len(available_files), 2)
        self.assertIn("sgs_11_01012024_31122024.csv", available_files)
        self.assertIn("sgs_12_01012024_31122024.csv", available_files)
    
    def test_config_loading(self):
        """Test configuration loading."""
        config = self.loader._load_config(self.config_path)
        self.assertIsInstance(config, dict)
        self.assertIn('market', config)
        self.assertIn('costs', config)
        self.assertIn('taxes', config)
        self.assertIn('portfolio', config)
    
    def test_config_loading_invalid_path(self):
        """Test configuration loading with invalid path."""
        config = self.loader._load_config("nonexistent_config.yaml")
        self.assertEqual(config, {})
    
    def test_date_format_handling(self):
        """Test handling of different date formats."""
        # Test with different date formats
        dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='D')
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        sample_df = pd.DataFrame({
            'valor': values
        }, index=dates)
        
        # Test normalization with different date formats
        result1 = self.loader.normalize_data_locf(sample_df, "01/01/2024", "05/01/2024")
        result2 = self.loader.normalize_data_locf(sample_df, "1/1/2024", "5/1/2024")
        
        self.assertTrue(result1.equals(result2))
    
    def test_business_day_range(self):
        """Test that LOCF normalization uses B3 trading days."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        values = [1.0] * len(dates)
        
        sample_df = pd.DataFrame({
            'valor': values
        }, index=dates)
        
        result = self.loader.normalize_data_locf(sample_df, "01/01/2024", "10/01/2024")
        
        # Get B3 trading days for the period
        b3 = mcal.get_calendar('BVMF')
        schedule = b3.schedule(start_date='2024-01-01', end_date='2024-01-10')
        b3_days = schedule.index
        
        # Check that result index matches B3 trading days
        self.assertTrue(result.index.equals(b3_days))
        
        # Verify that all dates in result are B3 trading days
        for date in result.index:
            self.assertIn(date, b3_days)
        
        # Check that the result has the expected number of B3 trading days
        self.assertEqual(len(result), len(b3_days))


class TestSGSDataLoaderIntegration(unittest.TestCase):
    """Integration tests for SGSDataLoader with actual API calls."""
    
    def setUp(self):
        """Set up test fixtures for integration tests."""
        self.test_dir = tempfile.mkdtemp()
        self.data_path = os.path.join(self.test_dir, "data", "sgs")
        
        # Create test configuration
        self.config_path = os.path.join(self.test_dir, "test_config.yaml")
        self._create_test_config()
        
        # Initialize loader
        self.loader = SGSDataLoader(
            data_path=self.data_path,
            config_path=self.config_path
        )
        
        # Use a shorter date range for testing
        self.start_date = "01/01/2024"
        self.end_date = "31/01/2024"
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def _create_test_config(self):
        """Create a test configuration file."""
        config_content = """
market:
  trading_hours:
    open: "10:00"
    close: "17:00"
  timezone: "America/Sao_Paulo"
  selic_rate: 0.15
  trading_days_per_year: 252

costs:
  brokerage_fee: 0.0
  emolument: 0.00005
  settlement_swing_trade: 0.00025

taxes:
  swing_trade: 0.15
  day_trade: 0.20
  exemption_limit: 20000

portfolio:
  initial_cash: 100000
  max_positions: 10
  position_sizing: "equal_weight"

sgs:
  series:
    11: "Selic Interest Rate"
    12: "CDI Interest Rate"
    433: "IPCA Inflation Index"
  api:
    base_url: "http://api.bcb.gov.br/dados/serie/bcdata.sgs"
    timeout: 30
    max_retries: 3
    user_agent: "quant_b3_backtest/1.0"
  processing:
    cache_enabled: true
    save_processed: true
    data_path: "data/sgs"
    normalization_method: "LOCF"
  validation:
    enable_quality_checks: true
    interest_rate_range: [0, 100]
    inflation_range: [-50, 100]
    min_data_points: 10
"""
        with open(self.config_path, 'w') as f:
            f.write(config_content)
    
    @unittest.skip("Skip actual API calls in unit tests")
    def test_fetch_series_data_integration(self):
        """Test actual API calls to Banco Central SGS."""
        # This test is skipped by default to avoid making actual API calls
        # during unit testing. Uncomment the skip decorator to run integration tests.
        
        for series_id in [11, 12]:  # Test with interest rate series
            result = self.loader.fetch_series_data(series_id, self.start_date, self.end_date)
            
            if result is not None:
                self.assertIsInstance(result, pd.DataFrame)
                self.assertGreater(len(result), 0)
                self.assertIn('valor', result.columns)
                self.assertTrue(result.index.is_monotonic_increasing)
    
    @unittest.skip("Skip actual API calls in unit tests")
    def test_get_series_data_integration(self):
        """Test complete data retrieval and processing pipeline."""
        # This test is skipped by default to avoid making actual API calls
        # during unit testing. Uncomment the skip decorator to run integration tests.
        
        result = self.loader.get_series_data(11, self.start_date, self.end_date)
        
        if result is not None:
            self.assertIsInstance(result, pd.DataFrame)
            self.assertGreater(len(result), 0)
            self.assertIn('valor', result.columns)
            self.assertFalse(result['valor'].isna().any())  # Should have no NaN values after LOCF


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add unit tests
    test_suite.addTest(unittest.makeSuite(TestSGSDataLoader))
    
    # Add integration tests (optional)
    # test_suite.addTest(unittest.makeSuite(TestSGSDataLoaderIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 