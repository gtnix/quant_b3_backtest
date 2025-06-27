"""
BCB Daily Factor Transformer Module

This module provides advanced data transformation and validation capabilities for
daily risk-free rate data from the Banco Central do Brasil (BCB) API.

The module includes:
- Multiple transformation strategies (multiplicative, exponential, compound)
- Advanced statistical validation methods (IQR, Z-score, rolling window)
- Comprehensive outlier detection and handling
- Flexible configuration management
- Detailed validation reporting
- Performance optimization features

"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, Union
import yaml
from pathlib import Path
import warnings
from scipy import stats
from dataclasses import dataclass, field
from enum import Enum


class TransformationMethod(Enum):
    """Enumeration of available transformation methods."""
    MULTIPLICATIVE = "multiplicative"
    EXPONENTIAL = "exponential"
    COMPOUND = "compound"


class ValidationMethod(Enum):
    """Enumeration of available validation methods."""
    IQR = "iqr"
    ZSCORE = "zscore"
    ROLLING_WINDOW = "rolling_window"
    BOUNDS = "bounds"


@dataclass
class ValidationConfig:
    """Configuration for validation methods."""
    method: ValidationMethod
    multiplier: float = 1.5  # For IQR method
    threshold: float = 3.0   # For Z-score method
    window_size: int = 30    # For rolling window method
    min_factor: float = 0.95 # For bounds method
    max_factor: float = 1.05 # For bounds method


@dataclass
class TransformationConfig:
    """Configuration for transformation methods."""
    method: TransformationMethod = TransformationMethod.MULTIPLICATIVE
    annualization_factor: float = 252.0  # Trading days per year
    compound_frequency: int = 1  # For compound method


@dataclass
class ValidationReport:
    """Comprehensive validation reporting structure."""
    original_count: int = 0
    validated_count: int = 0
    retention_rate: float = 0.0
    outliers_removed: int = 0
    nan_values_handled: int = 0
    validation_methods_applied: List[str] = field(default_factory=list)
    outlier_details: Dict[str, Any] = field(default_factory=dict)
    transformation_insights: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class BCBDailyFactorTransformer:
    """
    Comprehensive Data Transformation and Validation System
    
    Key Responsibilities:
    - Transform raw BCB daily rates to usable factors
    - Implement multiple transformation strategies
    - Provide statistically robust validation
    - Support flexible configuration
    - Generate comprehensive validation reports
    """
    
    def __init__(
        self, 
        config_path: str = 'config/settings.yaml',
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Transformer with Flexible Configuration
        
        Configuration Considerations:
        - Transformation method selection
        - Validation strategy
        - Outlier detection parameters
        
        Args:
            config_path (str): Path to configuration file
            logger (Optional[logging.Logger]): Logger instance
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = logger or self._setup_logger()
        
        # Initialize configurations
        self.transformation_config = self._load_transformation_config()
        self.validation_configs = self._load_validation_configs()
        
        # Performance tracking
        self.performance_metrics = {}
        
        self.logger.info("BCBDailyFactorTransformer initialized successfully")
    
    def _load_config(self) -> dict:
        """
        Load configuration from settings.yaml file.
        
        Returns:
            dict: Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            # Ensure market.daily_factor.transformation section exists
            if 'market' not in config:
                config['market'] = {}
            if 'daily_factor' not in config['market']:
                config['market']['daily_factor'] = {}
            if 'transformation' not in config['market']['daily_factor']:
                config['market']['daily_factor']['transformation'] = {}
            
            return config
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing configuration file: {e}")
    
    def _setup_logger(self) -> logging.Logger:
        """
        Set up logging configuration for both console and file output.
        
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, 'bcb_transformer.log'),
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _load_transformation_config(self) -> TransformationConfig:
        """
        Load transformation configuration from settings.
        
        Returns:
            TransformationConfig: Transformation configuration
        """
        transform_config = self.config['market']['daily_factor']['transformation']
        
        method_str = transform_config.get('method', 'multiplicative')
        try:
            method = TransformationMethod(method_str)
        except ValueError:
            self.logger.warning(f"Invalid transformation method: {method_str}. Using multiplicative.")
            method = TransformationMethod.MULTIPLICATIVE
        
        return TransformationConfig(
            method=method,
            annualization_factor=transform_config.get('annualization_factor', 252.0),
            compound_frequency=transform_config.get('compound_frequency', 1)
        )
    
    def _load_validation_configs(self) -> List[ValidationConfig]:
        """
        Load validation configurations from settings.
        
        Returns:
            List[ValidationConfig]: List of validation configurations
        """
        validation_config = self.config['market']['daily_factor'].get('validation', {})
        strategies = validation_config.get('strategies', [])
        
        configs = []
        for strategy in strategies:
            method_str = strategy.get('method', 'iqr')
            try:
                method = ValidationMethod(method_str)
            except ValueError:
                self.logger.warning(f"Invalid validation method: {method_str}. Skipping.")
                continue
            
            config = ValidationConfig(
                method=method,
                multiplier=strategy.get('multiplier', 1.5),
                threshold=strategy.get('threshold', 3.0),
                window_size=strategy.get('window_size', 30),
                min_factor=strategy.get('min_factor', 0.95),
                max_factor=strategy.get('max_factor', 1.05)
            )
            configs.append(config)
        
        # Add default bounds validation if no strategies specified
        if not configs:
            bounds_config = validation_config.get('bounds', {})
            configs.append(ValidationConfig(
                method=ValidationMethod.BOUNDS,
                min_factor=bounds_config.get('min_factor', 0.95),
                max_factor=bounds_config.get('max_factor', 1.05)
            ))
        
        return configs
    
    def transform_rates(
        self, 
        raw_rates: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Core Transformation Method
        
        Transformation Strategies:
        1. Multiplicative Factor: 1 + daily_rate
        2. Exponential Transformation
        3. Compound Rate Conversion
        
        Args:
            raw_rates (pd.DataFrame): Raw daily rates from BCB API
        
        Returns:
            pd.DataFrame: Transformed daily factors
        """
        self.logger.info(f"Transforming rates using method: {self.transformation_config.method.value}")
        
        if raw_rates.empty:
            self.logger.warning("Empty DataFrame provided for transformation")
            return raw_rates
        
        # Ensure we have the required columns
        if 'valor' not in raw_rates.columns:
            raise ValueError("Raw rates DataFrame must contain 'valor' column")
        
        # Clone DataFrame to prevent side effects
        df = raw_rates.copy()
        
        # Convert valor column to numeric, handling errors
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        
        # Remove rows with invalid values
        invalid_mask = df['valor'].isna() | (df['valor'] < 0)
        if invalid_mask.any():
            self.logger.warning(f"Removing {invalid_mask.sum()} rows with invalid values")
            df = df[~invalid_mask]
        
        if df.empty:
            raise ValueError("No valid data remaining after cleaning")
        
        # Apply transformation based on configured method
        if self.transformation_config.method == TransformationMethod.MULTIPLICATIVE:
            df['daily_factor'] = 1.0 + (df['valor'] / 100.0)  # Convert percentage to decimal
        
        elif self.transformation_config.method == TransformationMethod.EXPONENTIAL:
            # Exponential transformation: exp(daily_rate)
            daily_rate = df['valor'] / 100.0
            df['daily_factor'] = np.exp(daily_rate)
        
        elif self.transformation_config.method == TransformationMethod.COMPOUND:
            # Compound transformation with configurable frequency
            annual_rate = df['valor'] / 100.0
            daily_rate = (1 + annual_rate) ** (1 / self.transformation_config.annualization_factor) - 1
            df['daily_factor'] = 1.0 + daily_rate
        
        # Add return column for convenience
        df['return'] = df['daily_factor'] - 1.0
        
        # Log transformation insights
        self.logger.info(
            f"Transformation complete:\n"
            f"- Method: {self.transformation_config.method.value}\n"
            f"- Records processed: {len(df)}\n"
            f"- Factor range: [{df['daily_factor'].min():.6f}, {df['daily_factor'].max():.6f}]\n"
            f"- Mean factor: {df['daily_factor'].mean():.6f}"
        )
        
        return df
    
    def validate_factors(
        self, 
        transformed_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Advanced Factor Validation
        
        Validation Approaches:
        - Statistical Outlier Detection
        - Multiple Validation Strategies
        - Configurable Bounds
        
        Validation Methods:
        1. Interquartile Range (IQR)
        2. Z-Score Method
        3. Rolling Window Validation
        4. Bounds Validation
        
        Args:
            transformed_df (pd.DataFrame): Transformed daily factors
        
        Returns:
            pd.DataFrame: Validated factors
        """
        self.logger.info("Starting advanced factor validation")
        
        if transformed_df.empty:
            self.logger.warning("Empty DataFrame provided for validation")
            return transformed_df
        
        # Clone DataFrame to prevent side effects
        df = transformed_df.copy()
        original_count = len(df)
        
        # Ensure we have the required column
        if 'daily_factor' not in df.columns:
            raise ValueError("Transformed DataFrame must contain 'daily_factor' column")
        
        # Sort by date to ensure chronological order
        if df.index.dtype == 'datetime64[ns]':
            df = df.sort_index()
        
        # Handle NaN values first
        nan_count = df['daily_factor'].isna().sum()
        if nan_count > 0:
            self.logger.info(f"Handling {nan_count} NaN values")
            df['daily_factor'] = df['daily_factor'].interpolate(method='time', limit=5)
            df = df.dropna()
        
        # Apply each validation method in sequence
        for config in self.validation_configs:
            method_name = config.method.value
            self.logger.info(f"Applying validation method: {method_name}")
            
            if config.method == ValidationMethod.IQR:
                df = self._detect_outliers_iqr(df, 'daily_factor', config.multiplier)
            
            elif config.method == ValidationMethod.ZSCORE:
                df = self._detect_outliers_zscore(df, 'daily_factor', config.threshold)
            
            elif config.method == ValidationMethod.ROLLING_WINDOW:
                df = self._validate_rolling_window(df, 'daily_factor', config.window_size)
            
            elif config.method == ValidationMethod.BOUNDS:
                df = self._validate_bounds(df, 'daily_factor', config.min_factor, config.max_factor)
        
        # Final validation: ensure no extreme values remain
        final_count = len(df)
        retention_rate = final_count / original_count if original_count > 0 else 0
        
        self.logger.info(
            f"Validation complete:\n"
            f"- Original records: {original_count}\n"
            f"- Validated records: {final_count}\n"
            f"- Retention rate: {retention_rate:.2%}\n"
            f"- Records removed: {original_count - final_count}"
        )
        
        return df
    
    def _detect_outliers_iqr(
        self, 
        df: pd.DataFrame, 
        column: str = 'daily_factor',
        multiplier: float = 1.5
    ) -> pd.DataFrame:
        """
        Interquartile Range (IQR) Outlier Detection
        
        Robust Statistical Method:
        - Calculates Q1, Q3, and IQR
        - Identifies outliers beyond multiplier * IQR
        
        Args:
            df (pd.DataFrame): Input DataFrame
            column (str): Column to analyze
            multiplier (float): IQR multiplier for outlier detection
        
        Returns:
            pd.DataFrame: DataFrame with outliers removed
        """
        if df.empty or column not in df.columns:
            return df
        
        # Calculate quartiles
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # Identify outliers
        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            self.logger.info(
                f"IQR outlier detection:\n"
                f"- Q1: {Q1:.6f}, Q3: {Q3:.6f}, IQR: {IQR:.6f}\n"
                f"- Bounds: [{lower_bound:.6f}, {upper_bound:.6f}]\n"
                f"- Outliers removed: {outlier_count}"
            )
        
        return df[~outlier_mask]
    
    def _detect_outliers_zscore(
        self, 
        df: pd.DataFrame, 
        column: str = 'daily_factor', 
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Z-Score Outlier Detection
        
        Statistical Approach:
        - Calculates standard deviations from mean
        - Removes extreme values
        
        Args:
            df (pd.DataFrame): Input DataFrame
            column (str): Column to analyze
            threshold (float): Z-score threshold
        
        Returns:
            pd.DataFrame: DataFrame with outliers removed
        """
        if df.empty or column not in df.columns:
            return df
        
        # Calculate z-scores
        mean_val = df[column].mean()
        std_val = df[column].std()
        
        if std_val == 0:
            self.logger.warning("Standard deviation is zero, skipping z-score validation")
            return df
        
        z_scores = np.abs((df[column] - mean_val) / std_val)
        
        # Identify outliers
        outlier_mask = z_scores > threshold
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            self.logger.info(
                f"Z-score outlier detection:\n"
                f"- Mean: {mean_val:.6f}, Std: {std_val:.6f}\n"
                f"- Threshold: {threshold}\n"
                f"- Outliers removed: {outlier_count}"
            )
        
        return df[~outlier_mask]
    
    def _validate_rolling_window(
        self,
        df: pd.DataFrame,
        column: str = 'daily_factor',
        window_size: int = 30
    ) -> pd.DataFrame:
        """
        Rolling Window Validation
        
        Dynamic validation using rolling statistics:
        - Calculates rolling mean and standard deviation
        - Identifies values outside rolling bounds
        
        Args:
            df (pd.DataFrame): Input DataFrame
            column (str): Column to analyze
            window_size (int): Rolling window size
        
        Returns:
            pd.DataFrame: DataFrame with outliers removed
        """
        if df.empty or column not in df.columns or len(df) < window_size:
            return df
        
        # Calculate rolling statistics
        rolling_mean = df[column].rolling(window=window_size, center=True).mean()
        rolling_std = df[column].rolling(window=window_size, center=True).std()
        
        # Define dynamic bounds (3 standard deviations)
        upper_bound = rolling_mean + 3 * rolling_std
        lower_bound = rolling_mean - 3 * rolling_std
        
        # Identify outliers
        outlier_mask = (df[column] > upper_bound) | (df[column] < lower_bound)
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            self.logger.info(
                f"Rolling window validation:\n"
                f"- Window size: {window_size}\n"
                f"- Outliers removed: {outlier_count}"
            )
        
        return df[~outlier_mask]
    
    def _validate_bounds(
        self,
        df: pd.DataFrame,
        column: str = 'daily_factor',
        min_factor: float = 0.95,
        max_factor: float = 1.05
    ) -> pd.DataFrame:
        """
        Bounds Validation
        
        Simple bounds checking:
        - Removes values outside specified range
        
        Args:
            df (pd.DataFrame): Input DataFrame
            column (str): Column to analyze
            min_factor (float): Minimum allowed factor
            max_factor (float): Maximum allowed factor
        
        Returns:
            pd.DataFrame: DataFrame with outliers removed
        """
        if df.empty or column not in df.columns:
            return df
        
        # Identify outliers
        outlier_mask = (df[column] < min_factor) | (df[column] > max_factor)
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            self.logger.info(
                f"Bounds validation:\n"
                f"- Bounds: [{min_factor}, {max_factor}]\n"
                f"- Outliers removed: {outlier_count}"
            )
        
        return df[~outlier_mask]
    
    def generate_validation_report(
        self, 
        original_df: pd.DataFrame, 
        validated_df: pd.DataFrame
    ) -> ValidationReport:
        """
        Comprehensive Validation Reporting
        
        Report Components:
        - Data retention statistics
        - Outlier information
        - Transformation insights
        - Performance metrics
        
        Args:
            original_df (pd.DataFrame): Original input data
            validated_df (pd.DataFrame): Validated data
        
        Returns:
            ValidationReport: Comprehensive validation report
        """
        report = ValidationReport()
        
        # Basic statistics
        report.original_count = len(original_df)
        report.validated_count = len(validated_df)
        report.retention_rate = report.validated_count / report.original_count if report.original_count > 0 else 0
        report.outliers_removed = report.original_count - report.validated_count
        
        # Validation methods applied
        report.validation_methods_applied = [config.method.value for config in self.validation_configs]
        
        # Outlier details
        if 'daily_factor' in original_df.columns and 'daily_factor' in validated_df.columns:
            original_factors = original_df['daily_factor']
            validated_factors = validated_df['daily_factor']
            
            report.outlier_details = {
                'original_stats': {
                    'mean': float(original_factors.mean()),
                    'std': float(original_factors.std()),
                    'min': float(original_factors.min()),
                    'max': float(original_factors.max()),
                    'q25': float(original_factors.quantile(0.25)),
                    'q75': float(original_factors.quantile(0.75))
                },
                'validated_stats': {
                    'mean': float(validated_factors.mean()),
                    'std': float(validated_factors.std()),
                    'min': float(validated_factors.min()),
                    'max': float(validated_factors.max()),
                    'q25': float(validated_factors.quantile(0.25)),
                    'q75': float(validated_factors.quantile(0.75))
                }
            }
        
        # Transformation insights
        report.transformation_insights = {
            'method': self.transformation_config.method.value,
            'annualization_factor': self.transformation_config.annualization_factor,
            'compound_frequency': self.transformation_config.compound_frequency
        }
        
        # Performance metrics
        report.performance_metrics = self.performance_metrics.copy()
        
        # Generate warnings
        if report.retention_rate < 0.8:
            report.warnings.append(f"Low retention rate: {report.retention_rate:.2%}")
        
        if report.outliers_removed > report.original_count * 0.1:
            report.warnings.append(f"High outlier removal: {report.outliers_removed} records")
        
        return report
    
    def transform_and_validate(
        self,
        raw_rates: pd.DataFrame
    ) -> Tuple[pd.DataFrame, ValidationReport]:
        """
        Complete transformation and validation pipeline.
        
        Args:
            raw_rates (pd.DataFrame): Raw daily rates from BCB API
        
        Returns:
            Tuple[pd.DataFrame, ValidationReport]: Transformed data and validation report
        """
        self.logger.info("Starting complete transformation and validation pipeline")
        
        # Transform rates
        transformed_df = self.transform_rates(raw_rates)
        
        # Validate factors
        validated_df = self.validate_factors(transformed_df)
        
        # Generate report
        report = self.generate_validation_report(transformed_df, validated_df)
        
        self.logger.info(
            f"Pipeline complete:\n"
            f"- Original records: {report.original_count}\n"
            f"- Final records: {report.validated_count}\n"
            f"- Retention rate: {report.retention_rate:.2%}"
        )
        
        return validated_df, report
    
    def save_validation_report(
        self,
        report: ValidationReport,
        output_path: str = "reports/validation_report.yaml"
    ) -> None:
        """
        Save validation report to file.
        
        Args:
            report (ValidationReport): Validation report to save
            output_path (str): Output file path
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert report to dictionary
        report_dict = {
            'timestamp': datetime.now().isoformat(),
            'original_count': report.original_count,
            'validated_count': report.validated_count,
            'retention_rate': report.retention_rate,
            'outliers_removed': report.outliers_removed,
            'nan_values_handled': report.nan_values_handled,
            'validation_methods_applied': report.validation_methods_applied,
            'outlier_details': report.outlier_details,
            'transformation_insights': report.transformation_insights,
            'performance_metrics': report.performance_metrics,
            'warnings': report.warnings,
            'errors': report.errors
        }
        
        # Save to YAML file
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(report_dict, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Validation report saved to: {output_path}")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics from the transformer.
        
        Returns:
            Dict[str, float]: Performance metrics
        """
        return self.performance_metrics.copy()
    
    def reset_performance_metrics(self) -> None:
        """Reset performance metrics."""
        self.performance_metrics.clear()
        self.logger.info("Performance metrics reset") 