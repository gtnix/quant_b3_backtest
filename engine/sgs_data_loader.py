"""
Banco Central SGS Historical Data Loader with LOCF Normalization

This module provides functionality to:
- Retrieve historical data from Banco Central SGS API
- Apply Last Observation Carried Forward (LOCF) normalization
- Integrate with the quant_b3_backtest framework
- Handle data quality and error scenarios

SGS Series Supported:
- 8: Bovespa Total Volume
- 11: Selic Interest Rate
- 12: CDI Interest Rate
- 433: IPCA Inflation Index
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, List, Union
import json
from datetime import datetime, timedelta
import yaml
import pandas_market_calendars as mcal

# Configure logging
logger = logging.getLogger(__name__)

# Utility function for SELIC daily factor conversion
def get_daily_factor(valor):
    """
    Convert Banco Central SGS 'valor' to a daily compounding factor.
    Handles:
      - 'valor' as daily factor (e.g., 1.00050788)
      - 'valor' as daily rate in decimal (e.g., 0.00050788)
      - 'valor' as daily rate in percent (e.g., 0.050788)
    Returns:
      - daily_factor (float): value to use for compounding (e.g., 1.00050788)
    """
    if valor > 1.0:
        return valor
    elif valor > 0.01:
        return 1 + (valor / 100)
    else:
        return 1 + valor

class SGSDataLoader:
    """
    A class to load and process Banco Central SGS historical data.
    
    This class handles:
    - Fetching data from Banco Central SGS API
    - Applying LOCF normalization for missing dates
    - Data quality validation and error handling
    - Integration with the backtesting framework
    """
    
    def __init__(self, data_path: str = None, config_path: str = "config/settings.yaml"):
        """
        Initialize the SGS Data Loader.
        
        Args:
            data_path (str): Path to SGS data directory (overrides config)
            config_path (str): Path to configuration file
        """
        # Load configuration first
        self.config = self._load_config(config_path)
        
        # Get SGS configuration with defaults
        sgs_config = self.config.get('sgs', {})
        
        # Set data path (config takes precedence, then parameter, then default)
        self.data_path = Path(data_path or sgs_config.get('processing', {}).get('data_path', 'data/sgs'))
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Get series configuration
        self.sgs_series = sgs_config.get('series', {
            11: "Selic Interest Rate",
            12: "CDI Interest Rate", 
            433: "IPCA Inflation Index"
        })
        
        # API configuration
        api_config = sgs_config.get('api', {})
        self.base_url = api_config.get('base_url', "http://api.bcb.gov.br/dados/serie/bcdata.sgs")
        self.timeout = api_config.get('timeout', 30)
        self.max_retries = api_config.get('max_retries', 3)
        self.user_agent = api_config.get('user_agent', 'quant_b3_backtest/1.0')
        
        # Processing configuration
        processing_config = sgs_config.get('processing', {})
        self.cache_enabled = processing_config.get('cache_enabled', True)
        self.save_processed = processing_config.get('save_processed', True)
        self.normalization_method = processing_config.get('normalization_method', 'LOCF')
        
        # Validation configuration
        validation_config = sgs_config.get('validation', {})
        self.enable_quality_checks = validation_config.get('enable_quality_checks', True)
        self.interest_rate_range = validation_config.get('interest_rate_range', [0, 100])
        self.inflation_range = validation_config.get('inflation_range', [-50, 100])
        self.min_data_points = validation_config.get('min_data_points', 10)
        
        # Initialize session
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
        
        logger.info(f"SGSDataLoader initialized with data_path: {self.data_path}")
        logger.info(f"Configured series: {list(self.sgs_series.keys())}")
    
    @property
    def SGS_SERIES(self):
        """
        Property to maintain backward compatibility with existing code.
        Returns the configured SGS series.
        """
        return self.sgs_series
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            Dict: Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return {}
    
    def fetch_series_data(self, series_id: int, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a given SGS series between start_date and end_date.
        
        Args:
            series_id (int): SGS series identifier
            start_date (str): Start date in format 'dd/mm/yyyy'
            end_date (str): End date in format 'dd/mm/yyyy'
        
        Returns:
            Optional[pd.DataFrame]: DataFrame with dates and values, or None if error
        """
        if series_id not in self.SGS_SERIES:
            logger.error(f"Invalid series_id: {series_id}. Valid series: {list(self.SGS_SERIES.keys())}")
            return None
        
        url = f"{self.base_url}.{series_id}/dados"
        params = {
            "formato": "json",
            "dataInicial": start_date,
            "dataFinal": end_date
        }
        
        try:
            logger.info(f"Fetching SGS series {series_id} ({self.SGS_SERIES[series_id]}) from {start_date} to {end_date}")
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                logger.warning(f"No data returned for series {series_id}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Convert date column to datetime
            df['data'] = pd.to_datetime(df['data'], dayfirst=True)
            df.set_index('data', inplace=True)
            
            # Convert value column to numeric
            df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
            
            # Remove rows with NaN values
            df = df.dropna()
            
            logger.info(f"Successfully fetched {len(df)} data points for series {series_id}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error fetching series {series_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching series {series_id}: {e}")
            return None
    
    def normalize_data_locf(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Normalize the DataFrame by applying Last Observation Carried Forward (LOCF)
        to fill missing dates, using the B3 (BVMF) trading calendar.
        
        Args:
            df (pd.DataFrame): Original DataFrame with date index
            start_date (str): Start date for normalization (dd/mm/yyyy)
            end_date (str): End date for normalization (dd/mm/yyyy)
        
        Returns:
            pd.DataFrame: DataFrame reindexed with B3 trading days and LOCF applied
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided for normalization")
            return pd.DataFrame()
        
        # Convert dates to datetime
        start_dt = pd.to_datetime(start_date, dayfirst=True)
        end_dt = pd.to_datetime(end_date, dayfirst=True)
        
        # Get B3 (BVMF) trading calendar
        b3 = mcal.get_calendar('BVMF')
        schedule = b3.schedule(start_date=start_dt, end_date=end_dt)
        date_range = schedule.index  # This is a DatetimeIndex of valid trading days
        
        # Reindex the DataFrame to include every B3 trading day in the range
        df_normalized = df.reindex(date_range)
        
        # Apply forward fill (LOCF)
        df_normalized['valor'] = df_normalized['valor'].ffill()
        
        # Apply backward fill for any remaining NaN values at the beginning
        df_normalized['valor'] = df_normalized['valor'].bfill()
        
        logger.info(f"Applied LOCF normalization (B3 calendar). Original: {len(df)} rows, Normalized: {len(df_normalized)} rows")
        logger.info(f"Filled {len(df_normalized) - len(df)} missing B3 trading days with LOCF")
        
        return df_normalized
    
    def validate_data_quality(self, df: pd.DataFrame, series_id: int) -> bool:
        """
        Validate data quality for the given series using configurable settings.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            series_id (int): SGS series identifier
        
        Returns:
            bool: True if data quality is acceptable
        """
        if not self.enable_quality_checks:
            return True
            
        if df is None or df.empty:
            logger.error("DataFrame is None or empty")
            return False
        
        # Check for missing values
        missing_count = df['valor'].isna().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values in series {series_id}")
        
        # Check for extreme values based on series type and configurable ranges
        if series_id in [11, 12]:  # Interest rates
            extreme_values = df[
                (df['valor'] < self.interest_rate_range[0]) | 
                (df['valor'] > self.interest_rate_range[1])
            ]
            if len(extreme_values) > 0:
                logger.warning(f"Found {len(extreme_values)} extreme interest rate values in series {series_id}")
        
        elif series_id == 433:  # IPCA
            extreme_values = df[
                (df['valor'] < self.inflation_range[0]) | 
                (df['valor'] > self.inflation_range[1])
            ]
            if len(extreme_values) > 0:
                logger.warning(f"Found {len(extreme_values)} extreme IPCA values in series {series_id}")
        
        # Check for data consistency
        if len(df) < self.min_data_points:
            logger.warning(f"Very few data points ({len(df)}) for series {series_id}")
        
        return True
    
    def save_processed_data(self, df: pd.DataFrame, series_id: int, start_date: str, end_date: str) -> bool:
        """
        Save processed data to CSV file.
        
        Args:
            df (pd.DataFrame): Processed DataFrame
            series_id (int): SGS series identifier
            start_date (str): Start date
            end_date (str): End date
        
        Returns:
            bool: True if saved successfully
        """
        try:
            # Create filename
            start_str = start_date.replace('/', '')
            end_str = end_date.replace('/', '')
            filename = f"sgs_{series_id}_{start_str}_{end_str}.csv"
            filepath = self.data_path / filename
            
            # Save to CSV
            df.to_csv(filepath, float_format='%.8f')
            
            # Save metadata
            metadata = {
                'series_id': series_id,
                'series_name': self.SGS_SERIES[series_id],
                'start_date': start_date,
                'end_date': end_date,
                'data_points': len(df),
                'processing_date': datetime.now().isoformat(),
                'normalization_method': self.normalization_method
            }
            
            metadata_file = filepath.with_suffix('.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved processed data to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            return False
    
    def load_processed_data(self, series_id: int, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Load previously processed data if available.
        
        Args:
            series_id (int): SGS series identifier
            start_date (str): Start date
            end_date (str): End date
        
        Returns:
            Optional[pd.DataFrame]: Loaded DataFrame or None if not found
        """
        try:
            start_str = start_date.replace('/', '')
            end_str = end_date.replace('/', '')
            filename = f"sgs_{series_id}_{start_str}_{end_str}.csv"
            filepath = self.data_path / filename
            
            if filepath.exists():
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                logger.info(f"Loaded cached data for series {series_id}")
                return df
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            return None
    
    def get_series_data(self, series_id: int, start_date: str, end_date: str, 
                       use_cache: bool = None, save_processed: bool = None) -> Optional[pd.DataFrame]:
        """
        Main method to get SGS series data with caching and processing.
        
        Args:
            series_id (int): SGS series identifier
            start_date (str): Start date in format 'dd/mm/yyyy'
            end_date (str): End date in format 'dd/mm/yyyy'
            use_cache (bool): Whether to use cached data if available (overrides config)
            save_processed (bool): Whether to save processed data (overrides config)
        
        Returns:
            Optional[pd.DataFrame]: Processed DataFrame with LOCF normalization
        """
        # Use config values if not explicitly provided
        use_cache = self.cache_enabled if use_cache is None else use_cache
        save_processed = self.save_processed if save_processed is None else save_processed
        
        # Try to load from cache first
        if use_cache:
            cached_data = self.load_processed_data(series_id, start_date, end_date)
            if cached_data is not None:
                return cached_data
        
        # Fetch fresh data
        raw_data = self.fetch_series_data(series_id, start_date, end_date)
        if raw_data is None:
            return None
        
        # Apply LOCF normalization
        normalized_data = self.normalize_data_locf(raw_data, start_date, end_date)
        
        # Validate data quality
        if not self.validate_data_quality(normalized_data, series_id):
            logger.warning(f"Data quality validation failed for series {series_id}")
        
        # Save processed data
        if save_processed:
            self.save_processed_data(normalized_data, series_id, start_date, end_date)
        
        # --- Add daily_factor column for SELIC (series 11) ---
        if series_id == 11 and normalized_data is not None and not normalized_data.empty:
            normalized_data['daily_factor'] = normalized_data['valor'].apply(get_daily_factor)
            logger.info("Added 'daily_factor' column to SELIC (series 11) DataFrame.")

        return normalized_data
    
    def get_all_series_data(self, start_date: str, end_date: str, 
                           use_cache: bool = True, save_processed: bool = True) -> Dict[int, pd.DataFrame]:
        """
        Get data for all supported SGS series.
        
        Args:
            start_date (str): Start date in format 'dd/mm/yyyy'
            end_date (str): End date in format 'dd/mm/yyyy'
            use_cache (bool): Whether to use cached data if available
            save_processed (bool): Whether to save processed data
        
        Returns:
            Dict[int, pd.DataFrame]: Dictionary mapping series_id to DataFrame
        """
        results = {}
        
        for series_id in self.SGS_SERIES.keys():
            logger.info(f"Processing series {series_id}: {self.SGS_SERIES[series_id]}")
            
            data = self.get_series_data(series_id, start_date, end_date, use_cache, save_processed)
            if data is not None:
                results[series_id] = data
            else:
                logger.error(f"Failed to get data for series {series_id}")
        
        logger.info(f"Successfully processed {len(results)} out of {len(self.SGS_SERIES)} series")
        return results
    
    def get_available_processed_files(self) -> List[str]:
        """
        Get list of available processed data files.
        
        Returns:
            List[str]: List of available filenames
        """
        try:
            files = [f.name for f in self.data_path.glob("sgs_*.csv")]
            return sorted(files)
        except Exception as e:
            logger.error(f"Error listing processed files: {e}")
            return []


def main():
    """
    Example usage and testing of the SGS Data Loader.
    """
    # Initialize the loader
    loader = SGSDataLoader()
    
    # Define date range (example: last year)
    end_date = datetime.now().strftime('%d/%m/%Y')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%d/%m/%Y')
    
    print(f"Fetching SGS data from {start_date} to {end_date}")
    
    # Get data for all series
    all_data = loader.get_all_series_data(start_date, end_date)
    
    # Display results
    for series_id, df in all_data.items():
        series_name = loader.SGS_SERIES[series_id]
        print(f"\nSeries {series_id}: {series_name}")
        print(f"Data points: {len(df)}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Value range: {df['valor'].min():.4f} to {df['valor'].max():.4f}")
        print(f"Sample data:")
        print(df.head())


if __name__ == "__main__":
    main() 