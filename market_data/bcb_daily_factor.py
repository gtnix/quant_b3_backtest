"""
BCB Daily Factor Retriever Module

This module provides functionality to retrieve historical daily factor data
from Banco Central do Brasil (BCB) API with B3 business days consideration.

The module includes:
- Data fetching from BCB API (series code 12)
- Data validation and cleaning
- Business day alignment using B3 calendar
- Risk-free rate curve calculation
- Local caching in Parquet format
"""

import os
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
import yaml
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import pandas_market_calendars as mcal


class BCBDailyFactorRetriever:
    """
    Retrieves and processes historical daily factor data from Banco Central do Brasil.
    
    This class fetches daily factor data from BCB API, validates it, aligns it to
    B3 business days, and provides methods for risk-free rate calculations.
    
    Attributes:
        config (dict): Configuration loaded from settings.yaml
        cache_dir (str): Directory for caching data
        logger (logging.Logger): Logger instance for the class
        b3_calendar: B3 market calendar instance
    """
    
    def __init__(self, config_path: str = "config/settings.yaml", cache_dir: str = "data/daily_factors/"):
        """
        Initialize the BCB Daily Factor Retriever.
        
        Args:
            config_path (str): Path to the configuration file
            cache_dir (str): Directory for caching data
        """
        self.config_path = config_path
        self.cache_dir = cache_dir
        self.config = self._load_config()
        self.logger = self._setup_logger()
        self.b3_calendar = mcal.get_calendar('B3')
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.logger.info("BCBDailyFactorRetriever initialized successfully")
    
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
            
            # Set default values if not present
            if 'market' not in config:
                config['market'] = {}
            if 'daily_factor' not in config['market']:
                config['market']['daily_factor'] = {}
            
            daily_factor_config = config['market']['daily_factor']
            daily_factor_config.setdefault('source', 'bcb_official_api')
            daily_factor_config.setdefault('validation', {})
            daily_factor_config['validation'].setdefault('min_factor', 0.95)
            daily_factor_config['validation'].setdefault('max_factor', 1.05)
            daily_factor_config.setdefault('cache', {})
            daily_factor_config['cache'].setdefault('directory', self.cache_dir)
            daily_factor_config['cache'].setdefault('format', 'parquet')
            
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
            os.path.join(log_dir, 'bcb_daily_factor.log'),
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def fetch_daily_factors(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch daily factor data from BCB API and process it.
        
        Args:
            start_date (datetime): Start date for data retrieval
            end_date (datetime): End date for data retrieval
            
        Returns:
            pd.DataFrame: Processed daily factors DataFrame with columns:
                - date: datetime index
                - factor: daily factor values
                - return: daily returns
                
        Raises:
            requests.RequestException: If API request fails
            ValueError: If data validation fails
        """
        self.logger.info(f"Fetching daily factors from {start_date} to {end_date}")
        
        try:
            # Fetch raw data from BCB API
            raw_data = self._fetch_from_bcb_api(start_date, end_date)
            
            # Transform to DataFrame
            df = self._transform_to_dataframe(raw_data)
            
            # Validate and clean data
            df = self._validate_factors(df)
            
            # Align to business days
            df = self._align_to_business_days(df)
            
            # Cache the processed data
            self._cache_factors(df)
            
            self.logger.info(f"Successfully processed {len(df)} daily factor records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching daily factors: {e}")
            raise
    
    def _fetch_from_bcb_api(self, start_date: datetime, end_date: datetime) -> dict:
        """
        Fetch raw data from BCB API.
        
        Args:
            start_date (datetime): Start date for data retrieval
            end_date (datetime): End date for data retrieval
            
        Returns:
            dict: Raw JSON response from BCB API
            
        Raises:
            requests.RequestException: If API request fails
        """
        # BCB API endpoint for daily factor (series code 12)
        base_url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
        
        # Format dates for API
        start_str = start_date.strftime("%d/%m/%Y")
        end_str = end_date.strftime("%d/%m/%Y")
        
        params = {
            'dataInicial': start_str,
            'dataFinal': end_str,
            'formato': 'json'
        }
        
        self.logger.info(f"Making API request to BCB: {base_url}")
        self.logger.debug(f"Parameters: {params}")
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            self.logger.info(f"Successfully retrieved {len(data)} records from BCB API")
            
            return data
            
        except requests.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise
    
    def _transform_to_dataframe(self, raw_data: dict) -> pd.DataFrame:
        """
        Transform raw API response to DataFrame.
        
        Args:
            raw_data (dict): Raw JSON response from BCB API
            
        Returns:
            pd.DataFrame: DataFrame with date and valor columns
            
        Raises:
            ValueError: If data transformation fails
        """
        try:
            if not raw_data:
                raise ValueError("Empty response from BCB API")
            
            # Extract data from response
            records = []
            for item in raw_data:
                try:
                    date = pd.to_datetime(item['data'], format='%d/%m/%Y')
                    valor = float(item['valor'])
                    records.append({'date': date, 'factor': valor})
                except (KeyError, ValueError) as e:
                    self.logger.warning(f"Skipping invalid record: {item}, error: {e}")
                    continue
            
            if not records:
                raise ValueError("No valid records found in API response")
            
            df = pd.DataFrame(records)
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            self.logger.info(f"Transformed {len(df)} valid records to DataFrame")
            return df
            
        except Exception as e:
            self.logger.error(f"Error transforming data to DataFrame: {e}")
            raise ValueError(f"Data transformation failed: {e}")
    
    def _validate_factors(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean daily factor data.
        
        Validation Workflow:
        1. Interpolate NaN values FIRST (preserves time series continuity)
        2. Apply outlier filtering AFTER interpolation
        3. Ensure data continuity and quality
        
        Args:
            factors_df (pd.DataFrame): Raw factors DataFrame
            
        Returns:
            pd.DataFrame: Validated and cleaned DataFrame
        """
        self.logger.info("Validating and cleaning daily factors")
        
        # Clone DataFrame to prevent side effects
        df = factors_df.copy()
        original_count = len(df)
        
        # Get validation bounds from config
        min_factor = self.config['market']['daily_factor']['validation']['min_factor']
        max_factor = self.config['market']['daily_factor']['validation']['max_factor']
        
        # Step 1: Sort by date to ensure chronological order
        df = df.sort_index()
        
        # Step 2: Comprehensive NaN Handling - INTERPOLATE FIRST
        nan_count_before = df['factor'].isna().sum()
        if nan_count_before > 0:
            self.logger.info(f"Interpolating {nan_count_before} missing values")
            
            # Get interpolation settings from config
            interpolation_config = self.config['market']['daily_factor'].get('interpolation', {})
            method = interpolation_config.get('method', 'time')
            limit_direction = interpolation_config.get('limit_direction', 'both')
            limit = interpolation_config.get('max_consecutive_interpolations', 5)
            
            df['factor'] = df['factor'].interpolate(
                method=method,  # Time-based interpolation preferred for financial data
                limit_direction=limit_direction,  # Interpolate in both directions
                limit=limit  # Limit consecutive interpolations to prevent over-interpolation
            )
        
        # Step 3: Apply outlier filtering AFTER interpolation
        initial_count = len(df)
        valid_mask = (df['factor'] >= min_factor) & (df['factor'] <= max_factor)
        df = df[valid_mask]
        
        removed_count = initial_count - len(df)
        if removed_count > 0:
            self.logger.warning(f"Removed {removed_count} records outside factor bounds [{min_factor}, {max_factor}]")
        
        # Step 4: Remove any remaining NaN values (should not occur after interpolation)
        df = df.dropna()
        
        # Logging validation insights
        final_count = len(df)
        retention_rate = final_count / original_count if original_count > 0 else 0
        
        self.logger.info(
            f"Validation complete:\n"
            f"- Original Records: {original_count}\n"
            f"- NaN Values Before: {nan_count_before}\n"
            f"- Validated Records: {final_count}\n"
            f"- Retention Rate: {retention_rate:.2%}"
        )
        
        return df
    
    def _align_to_business_days(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align factors DataFrame to B3 business days.
        
        Args:
            factors_df (pd.DataFrame): Factors DataFrame with datetime index
            
        Returns:
            pd.DataFrame: DataFrame aligned to B3 business days
        """
        self.logger.info("Aligning data to B3 business days")
        
        if factors_df.empty:
            self.logger.warning("Empty DataFrame provided for business day alignment")
            return factors_df
        
        # Get date range
        start_date = factors_df.index.min()
        end_date = factors_df.index.max()
        
        # Create B3 business day calendar
        business_days = self.b3_calendar.valid_days(start_date=start_date, end_date=end_date)
        
        # Convert business_days to timezone-naive if needed
        if business_days.tz is not None:
            business_days = business_days.tz_localize(None)
        
        # Ensure factors_df index is also timezone-naive
        if factors_df.index.tz is not None:
            factors_df = factors_df.copy()
            factors_df.index = factors_df.index.tz_localize(None)
        
        # Reindex to business days and forward-fill missing values
        factors_df = factors_df.reindex(business_days, method='ffill')
        
        # Remove any remaining NaN values (should not occur with forward-fill)
        factors_df = factors_df.dropna()
        
        self.logger.info(f"Aligned to {len(factors_df)} B3 business days")
        return factors_df
    
    def _cache_factors(self, factors_df: pd.DataFrame) -> None:
        """
        Cache the processed factors DataFrame to local storage.
        
        Args:
            factors_df (pd.DataFrame): Processed factors DataFrame
        """
        try:
            cache_format = self.config['market']['daily_factor']['cache']['format']
            
            if cache_format.lower() == 'parquet':
                cache_file = os.path.join(self.cache_dir, 'daily_factors.parquet')
                factors_df.to_parquet(cache_file, engine='pyarrow')
                self.logger.info(f"Cached factors to {cache_file}")
            else:
                self.logger.warning(f"Unsupported cache format: {cache_format}")
                
        except Exception as e:
            self.logger.error(f"Error caching factors: {e}")
    
    def calculate_risk_free_curve(self, factors_df: pd.DataFrame) -> pd.Series:
        """
        Calculate annualized risk-free rate curve from daily factors.
        
        Args:
            factors_df (pd.DataFrame): Daily factors DataFrame
            
        Returns:
            pd.Series: Annualized risk-free rates with datetime index
        """
        self.logger.info("Calculating risk-free rate curve")
        
        if factors_df.empty:
            self.logger.warning("Empty DataFrame provided for risk-free curve calculation")
            return pd.Series(dtype=float)
        
        try:
            # Calculate daily returns from factors
            daily_returns = factors_df['factor'] - 1.0
            
            # Calculate cumulative return
            cumulative_return = (1 + daily_returns).cumprod()
            
            # Calculate annualized rate using 252 trading days
            trading_days_per_year = self.config['market']['trading_days_per_year']
            
            # Calculate annualized rate for each point
            annualized_rates = []
            dates = []
            
            for i, (date, cum_return) in enumerate(cumulative_return.items()):
                if i == 0:
                    # First day: use the daily return annualized
                    annualized_rate = (1 + daily_returns.iloc[i]) ** trading_days_per_year - 1
                else:
                    # Calculate annualized rate based on cumulative return and days elapsed
                    days_elapsed = i + 1
                    annualized_rate = (cum_return ** (trading_days_per_year / days_elapsed)) - 1
                
                annualized_rates.append(annualized_rate)
                dates.append(date)
            
            risk_free_curve = pd.Series(annualized_rates, index=dates)
            
            # Validate the curve
            if risk_free_curve.isna().any():
                self.logger.warning("Risk-free curve contains NaN values")
            
            self.logger.info(f"Calculated risk-free curve with {len(risk_free_curve)} points")
            return risk_free_curve
            
        except Exception as e:
            self.logger.error(f"Error calculating risk-free curve: {e}")
            raise
    
    def load_cached_factors(self) -> Optional[pd.DataFrame]:
        """
        Load cached factors from local storage.
        
        Returns:
            Optional[pd.DataFrame]: Cached factors DataFrame or None if not found
        """
        try:
            cache_format = self.config['market']['daily_factor']['cache']['format']
            
            if cache_format.lower() == 'parquet':
                cache_file = os.path.join(self.cache_dir, 'daily_factors.parquet')
                
                if os.path.exists(cache_file):
                    factors_df = pd.read_parquet(cache_file)
                    self.logger.info(f"Loaded cached factors from {cache_file}")
                    return factors_df
                else:
                    self.logger.info("No cached factors found")
                    return None
            else:
                self.logger.warning(f"Unsupported cache format: {cache_format}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading cached factors: {e}")
            return None
    
    def get_risk_free_rate_for_date(self, target_date: datetime, 
                                   lookback_days: int = 30) -> Optional[float]:
        """
        Get risk-free rate for a specific date using recent historical data.
        
        Args:
            target_date (datetime): Target date for risk-free rate
            lookback_days (int): Number of days to look back for calculation
            
        Returns:
            Optional[float]: Risk-free rate for the target date or None if not available
        """
        try:
            # Calculate date range
            start_date = target_date - timedelta(days=lookback_days)
            end_date = target_date
            
            # Fetch or load data
            factors_df = self.load_cached_factors()
            if factors_df is None or factors_df.empty:
                factors_df = self.fetch_daily_factors(start_date, end_date)
            
            if factors_df.empty:
                return None
            
            # Filter to relevant date range
            mask = (factors_df.index >= start_date) & (factors_df.index <= target_date)
            relevant_data = factors_df[mask]
            
            if relevant_data.empty:
                return None
            
            # Calculate risk-free curve
            risk_free_curve = self.calculate_risk_free_curve(relevant_data)
            
            # Get the rate for the target date (or closest available)
            if target_date in risk_free_curve.index:
                return risk_free_curve[target_date]
            else:
                # Find the closest available date
                available_dates = risk_free_curve.index
                closest_date = min(available_dates, key=lambda x: abs((x - target_date).days))
                return risk_free_curve[closest_date]
                
        except Exception as e:
            self.logger.error(f"Error getting risk-free rate for {target_date}: {e}")
            return None 