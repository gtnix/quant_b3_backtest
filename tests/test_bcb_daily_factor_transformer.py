"""
Unit tests for BCB Daily Factor Transformer.

This module contains comprehensive tests for the BCBDailyFactorTransformer class,
covering transformation methods, validation strategies, outlier detection, and reporting.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import shutil
import yaml
import warnings

# Add the parent directory to the path to import the module
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from market_data.bcb_daily_factor_transformer import (
    BCBDailyFactorTransformer,
    TransformationMethod,
    ValidationMethod,
    ValidationConfig,
    TransformationConfig,
    ValidationReport
)


class TestBCBDailyFactorTransformer(unittest.TestCase):
    """Test cases for BCBDailyFactorTransformer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for tests
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_settings.yaml')
        
        # Create test configuration with advanced settings
        self.test_config = {
            'market': {
                'trading_days_per_year': 252,
                'daily_factor': {
                    'source': 'bcb_official_api',
                    'transformation': {
                        'method': 'multiplicative',
                        'annualization_factor': 252.0,
                        'compound_frequency': 1
                    },
                    'validation': {
                        'strategies': [
                            {
                                'method': 'iqr',
                                'multiplier': 1.5
                            },
                            {
                                'method': 'zscore',
                                'threshold': 3.0
                            },
                            {
                                'method': 'rolling_window',
                                'window_size': 30
                            }
                        ],
                        'bounds': {
                            'min_factor': 0.95,
                            'max_factor': 1.05
                        }
                    },
                    'cache': {
                        'directory': os.path.join(self.temp_dir, 'cache'),
                        'format': 'parquet'
                    }
                }
            }
        }
        
        # Write test configuration
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
        
        # Sample raw rates data
        dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
        self.sample_raw_rates = pd.DataFrame({
            'valor': [0.1, 0.15, 0.12, 0.18, 0.14, 0.16, 0.13, 0.17, 0.11, 0.19]
        }, index=dates)
        
        # Sample transformed data
        self.sample_transformed_data = pd.DataFrame({
            'daily_factor': [1.001, 1.0015, 1.0012, 1.0018, 1.0014, 1.0016, 1.0013, 1.0017, 1.0011, 1.0019],
            'return': [0.001, 0.0015, 0.0012, 0.0018, 0.0014, 0.0016, 0.0013, 0.0017, 0.0011, 0.0019]
        }, index=dates)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initialization of BCBDailyFactorTransformer."""
        transformer = BCBDailyFactorTransformer(self.config_path)
        
        self.assertIsNotNone(transformer.config)
        self.assertIsNotNone(transformer.logger)
        self.assertIsNotNone(transformer.transformation_config)
        self.assertIsNotNone(transformer.validation_configs)
        self.assertEqual(len(transformer.validation_configs), 3)
    
    def test_load_transformation_config(self):
        """Test transformation configuration loading."""
        transformer = BCBDailyFactorTransformer(self.config_path)
        
        config = transformer.transformation_config
        self.assertEqual(config.method, TransformationMethod.MULTIPLICATIVE)
        self.assertEqual(config.annualization_factor, 252.0)
        self.assertEqual(config.compound_frequency, 1)
    
    def test_load_validation_configs(self):
        """Test validation configuration loading."""
        transformer = BCBDailyFactorTransformer(self.config_path)
        
        configs = transformer.validation_configs
        self.assertEqual(len(configs), 3)
        
        # Check IQR config
        iqr_config = next(c for c in configs if c.method == ValidationMethod.IQR)
        self.assertEqual(iqr_config.multiplier, 1.5)
        
        # Check Z-score config
        zscore_config = next(c for c in configs if c.method == ValidationMethod.ZSCORE)
        self.assertEqual(zscore_config.threshold, 3.0)
        
        # Check rolling window config
        rolling_config = next(c for c in configs if c.method == ValidationMethod.ROLLING_WINDOW)
        self.assertEqual(rolling_config.window_size, 30)
    
    def test_transform_rates_multiplicative(self):
        """Test multiplicative transformation method."""
        transformer = BCBDailyFactorTransformer(self.config_path)
        
        result = transformer.transform_rates(self.sample_raw_rates)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('daily_factor', result.columns)
        self.assertIn('return', result.columns)
        
        # Check transformation logic
        expected_factors = 1.0 + (self.sample_raw_rates['valor'] / 100.0)
        pd.testing.assert_series_equal(
            result['daily_factor'], 
            expected_factors, 
            check_names=False
        )
    
    def test_transform_rates_exponential(self):
        """Test exponential transformation method."""
        # Update config for exponential method
        self.test_config['market']['daily_factor']['transformation']['method'] = 'exponential'
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
        
        transformer = BCBDailyFactorTransformer(self.config_path)
        result = transformer.transform_rates(self.sample_raw_rates)
        
        # Check exponential transformation
        expected_factors = np.exp(self.sample_raw_rates['valor'] / 100.0)
        pd.testing.assert_series_equal(
            result['daily_factor'], 
            expected_factors, 
            check_names=False
        )
    
    def test_transform_rates_compound(self):
        """Test compound transformation method."""
        # Update config for compound method
        self.test_config['market']['daily_factor']['transformation']['method'] = 'compound'
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
        
        transformer = BCBDailyFactorTransformer(self.config_path)
        result = transformer.transform_rates(self.sample_raw_rates)
        
        # Check compound transformation
        annual_rates = self.sample_raw_rates['valor'] / 100.0
        daily_rates = (1 + annual_rates) ** (1 / 252.0) - 1
        expected_factors = 1.0 + daily_rates
        
        pd.testing.assert_series_equal(
            result['daily_factor'], 
            expected_factors, 
            check_names=False
        )
    
    def test_transform_rates_empty_dataframe(self):
        """Test transformation with empty DataFrame."""
        transformer = BCBDailyFactorTransformer(self.config_path)
        
        empty_df = pd.DataFrame()
        result = transformer.transform_rates(empty_df)
        
        self.assertTrue(result.empty)
    
    def test_transform_rates_invalid_column(self):
        """Test transformation with missing required column."""
        transformer = BCBDailyFactorTransformer(self.config_path)
        
        invalid_df = pd.DataFrame({'wrong_column': [1, 2, 3]})
        
        with self.assertRaises(ValueError):
            transformer.transform_rates(invalid_df)
    
    def test_transform_rates_invalid_values(self):
        """Test transformation with invalid values."""
        transformer = BCBDailyFactorTransformer(self.config_path)
        
        # Create data with negative values
        invalid_data = pd.DataFrame({
            'valor': [0.1, -0.2, 0.15, np.nan, 0.12]
        }, index=pd.date_range('2024-01-01', '2024-01-05', freq='D'))
        
        result = transformer.transform_rates(invalid_data)
        
        # Should remove invalid values
        self.assertEqual(len(result), 3)  # Only valid values remain
    
    def test_validate_factors_iqr(self):
        """Test IQR outlier detection."""
        transformer = BCBDailyFactorTransformer(self.config_path)
        
        # Create data with outliers
        dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
        test_data = pd.DataFrame({
            'daily_factor': [1.001, 1.002, 0.90, 1.003, 1.10, 1.004, 1.005, 1.006, 1.007, 1.008]
        }, index=dates)
        
        validated_df = transformer.validate_factors(test_data)
        
        # Should remove outliers (0.90 and 1.10)
        self.assertLess(len(validated_df), len(test_data))
        self.assertTrue(all(0.95 <= factor <= 1.05 for factor in validated_df['daily_factor']))
    
    def test_validate_factors_zscore(self):
        """Test Z-score outlier detection."""
        # Create config with only Z-score validation
        self.test_config['market']['daily_factor']['validation']['strategies'] = [
            {'method': 'zscore', 'threshold': 2.0}
        ]
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
        
        transformer = BCBDailyFactorTransformer(self.config_path)
        
        # Create data with extreme values
        dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
        test_data = pd.DataFrame({
            'daily_factor': [1.001, 1.002, 1.003, 1.004, 1.005, 1.006, 1.007, 1.008, 1.009, 2.0]  # 2.0 is extreme
        }, index=dates)
        
        validated_df = transformer.validate_factors(test_data)
        
        # Should remove the extreme value
        self.assertEqual(len(validated_df), 9)
        self.assertNotIn(2.0, validated_df['daily_factor'].values)
    
    def test_validate_factors_rolling_window(self):
        """Test rolling window validation."""
        # Create config with only rolling window validation
        self.test_config['market']['daily_factor']['validation']['strategies'] = [
            {'method': 'rolling_window', 'window_size': 5}
        ]
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
        
        transformer = BCBDailyFactorTransformer(self.config_path)
        
        # Create data with local outliers that will be detected by rolling window
        dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
        test_data = pd.DataFrame({
            'daily_factor': [1.001, 1.002, 1.003, 1.004, 1.005, 1.006, 1.007, 1.008, 1.009, 2.0]  # 2.0 is extreme outlier
        }, index=dates)
        
        validated_df = transformer.validate_factors(test_data)
        
        # Should remove the extreme outlier
        self.assertLess(len(validated_df), len(test_data))
        self.assertNotIn(2.0, validated_df['daily_factor'].values)
    
    def test_validate_factors_bounds(self):
        """Test bounds validation."""
        # Create config with only bounds validation
        self.test_config['market']['daily_factor']['validation']['strategies'] = []
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
        
        transformer = BCBDailyFactorTransformer(self.config_path)
        
        # Create data with out-of-bounds values
        dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
        test_data = pd.DataFrame({
            'daily_factor': [1.001, 1.002, 0.90, 1.003, 1.10, 1.004, 1.005, 1.006, 1.007, 1.008]
        }, index=dates)
        
        validated_df = transformer.validate_factors(test_data)
        
        # Should remove out-of-bounds values
        self.assertEqual(len(validated_df), 8)
        self.assertTrue(all(0.95 <= factor <= 1.05 for factor in validated_df['daily_factor']))
    
    def test_validate_factors_with_nan(self):
        """Test validation with NaN values."""
        transformer = BCBDailyFactorTransformer(self.config_path)
        
        # Create data with NaN values
        dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')
        test_data = pd.DataFrame({
            'daily_factor': [1.001, np.nan, 1.003, 1.004, 1.005]
        }, index=dates)
        
        validated_df = transformer.validate_factors(test_data)
        
        # Should interpolate NaN and keep all records
        self.assertEqual(len(validated_df), 5)
        self.assertFalse(validated_df['daily_factor'].isna().any())
    
    def test_detect_outliers_iqr(self):
        """Test IQR outlier detection method."""
        transformer = BCBDailyFactorTransformer(self.config_path)
        
        # Create data with known outliers
        test_data = pd.DataFrame({
            'daily_factor': [1.001, 1.002, 0.90, 1.003, 1.10, 1.004, 1.005]
        })
        
        result = transformer._detect_outliers_iqr(test_data, 'daily_factor', 1.5)
        
        # Should remove outliers
        self.assertLess(len(result), len(test_data))
        self.assertNotIn(0.90, result['daily_factor'].values)
        self.assertNotIn(1.10, result['daily_factor'].values)
    
    def test_detect_outliers_zscore(self):
        """Test Z-score outlier detection method."""
        transformer = BCBDailyFactorTransformer(self.config_path)
        
        # Create data with extreme values
        test_data = pd.DataFrame({
            'daily_factor': [1.001, 1.002, 1.003, 1.004, 1.005, 2.0]  # 2.0 is extreme
        })
        
        result = transformer._detect_outliers_zscore(test_data, 'daily_factor', 2.0)
        
        # Should remove extreme value
        self.assertEqual(len(result), 5)
        self.assertNotIn(2.0, result['daily_factor'].values)
    
    def test_validate_rolling_window(self):
        """Test rolling window validation method."""
        transformer = BCBDailyFactorTransformer(self.config_path)
        
        # Create data with local outliers that will be detected
        dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
        test_data = pd.DataFrame({
            'daily_factor': [1.001, 1.002, 1.003, 1.004, 1.005, 1.006, 1.007, 1.008, 1.009, 2.0]  # 2.0 is extreme outlier
        }, index=dates)
        
        result = transformer._validate_rolling_window(test_data, 'daily_factor', 5)
        
        # Should remove the extreme outlier
        self.assertLess(len(result), len(test_data))
        self.assertNotIn(2.0, result['daily_factor'].values)
    
    def test_validate_bounds(self):
        """Test bounds validation method."""
        transformer = BCBDailyFactorTransformer(self.config_path)
        
        # Create data with out-of-bounds values
        test_data = pd.DataFrame({
            'daily_factor': [1.001, 1.002, 0.90, 1.003, 1.10, 1.004, 1.005]
        })
        
        result = transformer._validate_bounds(test_data, 'daily_factor', 0.95, 1.05)
        
        # Should remove out-of-bounds values
        self.assertEqual(len(result), 5)
        self.assertTrue(all(0.95 <= factor <= 1.05 for factor in result['daily_factor']))
    
    def test_generate_validation_report(self):
        """Test validation report generation."""
        transformer = BCBDailyFactorTransformer(self.config_path)
        
        # Create sample data
        dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
        original_data = pd.DataFrame({
            'daily_factor': [1.001, 1.002, 0.90, 1.003, 1.10, 1.004, 1.005, 1.006, 1.007, 1.008]
        }, index=dates)
        
        validated_data = pd.DataFrame({
            'daily_factor': [1.001, 1.002, 1.003, 1.004, 1.005, 1.006, 1.007, 1.008]
        }, index=dates[:8])
        
        report = transformer.generate_validation_report(original_data, validated_data)
        
        self.assertIsInstance(report, ValidationReport)
        self.assertEqual(report.original_count, 10)
        self.assertEqual(report.validated_count, 8)
        self.assertEqual(report.retention_rate, 0.8)
        self.assertEqual(report.outliers_removed, 2)
        self.assertIn('iqr', report.validation_methods_applied)
        self.assertIn('zscore', report.validation_methods_applied)
    
    def test_transform_and_validate_pipeline(self):
        """Test complete transformation and validation pipeline."""
        transformer = BCBDailyFactorTransformer(self.config_path)
        
        # Run complete pipeline
        validated_df, report = transformer.transform_and_validate(self.sample_raw_rates)
        
        self.assertIsInstance(validated_df, pd.DataFrame)
        self.assertIsInstance(report, ValidationReport)
        self.assertIn('daily_factor', validated_df.columns)
        self.assertIn('return', validated_df.columns)
        self.assertGreater(len(validated_df), 0)
    
    def test_save_validation_report(self):
        """Test saving validation report to file."""
        transformer = BCBDailyFactorTransformer(self.config_path)
        
        # Create sample report
        report = ValidationReport(
            original_count=10,
            validated_count=8,
            retention_rate=0.8,
            outliers_removed=2
        )
        
        output_path = os.path.join(self.temp_dir, 'test_report.yaml')
        transformer.save_validation_report(report, output_path)
        
        # Check if file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Check if file contains valid YAML
        with open(output_path, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        self.assertEqual(saved_data['original_count'], 10)
        self.assertEqual(saved_data['validated_count'], 8)
        self.assertEqual(saved_data['retention_rate'], 0.8)
    
    def test_get_performance_metrics(self):
        """Test performance metrics retrieval."""
        transformer = BCBDailyFactorTransformer(self.config_path)
        
        # Add some metrics
        transformer.performance_metrics['processing_time'] = 1.5
        transformer.performance_metrics['memory_usage'] = 1024.0
        
        metrics = transformer.get_performance_metrics()
        
        self.assertIn('processing_time', metrics)
        self.assertIn('memory_usage', metrics)
        self.assertEqual(metrics['processing_time'], 1.5)
    
    def test_reset_performance_metrics(self):
        """Test performance metrics reset."""
        transformer = BCBDailyFactorTransformer(self.config_path)
        
        # Add some metrics
        transformer.performance_metrics['test_metric'] = 1.0
        
        # Reset metrics
        transformer.reset_performance_metrics()
        
        self.assertEqual(len(transformer.performance_metrics), 0)
    
    def test_invalid_transformation_method(self):
        """Test handling of invalid transformation method."""
        # Create config with invalid method
        self.test_config['market']['daily_factor']['transformation']['method'] = 'invalid_method'
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
        
        transformer = BCBDailyFactorTransformer(self.config_path)
        
        # Should default to multiplicative
        self.assertEqual(transformer.transformation_config.method, TransformationMethod.MULTIPLICATIVE)
    
    def test_invalid_validation_method(self):
        """Test handling of invalid validation method."""
        # Create config with invalid validation method
        self.test_config['market']['daily_factor']['validation']['strategies'] = [
            {'method': 'invalid_method', 'multiplier': 1.5}
        ]
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
        
        transformer = BCBDailyFactorTransformer(self.config_path)
        
        # Should skip invalid method and use default bounds
        self.assertEqual(len(transformer.validation_configs), 1)
        self.assertEqual(transformer.validation_configs[0].method, ValidationMethod.BOUNDS)


if __name__ == '__main__':
    unittest.main() 