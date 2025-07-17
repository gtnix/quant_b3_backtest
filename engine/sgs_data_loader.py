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
from typing import Optional, Dict, List, Union, Any
import json
from datetime import datetime, timedelta
import yaml
import pandas_market_calendars as mcal

# Configure logging
logger = logging.getLogger(__name__)

# Custom Exception Classes for SELIC Data Handling
class SELICDataError(Exception):
    """Base exception for SELIC data issues."""
    pass

class SELICDataUnavailableError(SELICDataError):
    """Raised when SELIC data is completely unavailable."""
    pass

class SELICDataInsufficientError(SELICDataError):
    """Raised when SELIC data coverage is insufficient."""
    pass

class SELICDataQualityError(SELICDataError):
    """Raised when SELIC data quality is below required thresholds."""
    pass

class SELICDataValidationError(SELICDataError):
    """Raised when SELIC data validation fails."""
    pass

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
    
    def validate_selic_data_coverage(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Validate SELIC data coverage for the specified period.
        
        Args:
            start_date (str): Start date in format 'dd/mm/yyyy'
            end_date (str): End date in format 'dd/mm/yyyy'
            
        Returns:
            Dict[str, Any]: Detailed coverage analysis
        """
        try:
            # Get strict mode configuration
            strict_config = self.config.get('sgs', {}).get('strict_mode', {})
            quality_config = self.config.get('sgs', {}).get('quality_thresholds', {})
            
            # Check if strict mode is enabled
            if not strict_config.get('enabled', False):
                return {'strict_mode_enabled': False, 'coverage_percentage': 100.0}
            
            # Fetch SELIC data (series 11)
            selic_data = self.get_series_data(11, start_date, end_date, use_cache=True, save_processed=False)
            
            if selic_data is None or selic_data.empty:
                raise SELICDataUnavailableError(f"No SELIC data available for period {start_date} to {end_date}")
            
            # Calculate coverage metrics
            start_dt = pd.to_datetime(start_date, dayfirst=True)
            end_dt = pd.to_datetime(end_date, dayfirst=True)
            
            # Get B3 trading calendar for the period
            b3 = mcal.get_calendar('BVMF')
            schedule = b3.schedule(start_date=start_dt, end_date=end_dt)
            total_trading_days = len(schedule)
            
            # Count days with SELIC data
            days_with_data = len(selic_data.dropna())
            coverage_percentage = (days_with_data / total_trading_days) * 100 if total_trading_days > 0 else 0
            
            # Check minimum coverage requirement
            min_coverage = strict_config.get('minimum_coverage_percentage', 95.0)
            if coverage_percentage < min_coverage:
                raise SELICDataInsufficientError(
                    f"SELIC data coverage ({coverage_percentage:.1f}%) below minimum requirement ({min_coverage}%)"
                )
            
            # Check minimum data points requirement
            min_data_points = quality_config.get('minimum_data_points', 100)
            if days_with_data < min_data_points:
                raise SELICDataInsufficientError(
                    f"Insufficient SELIC data points ({days_with_data}) below minimum requirement ({min_data_points})"
                )
            
            # Check for large gaps
            max_gap_days = quality_config.get('maximum_gap_days', 5)
            gaps = self._find_data_gaps(selic_data, max_gap_days)
            
            # Check rate validity
            rate_range = quality_config.get('rate_validity_range', [0.001, 100.0])
            invalid_rates = self._check_rate_validity(selic_data, rate_range)
            
            # Calculate quality score
            quality_score = self._calculate_selic_quality_score(selic_data, gaps, invalid_rates)
            
            return {
                'strict_mode_enabled': True,
                'coverage_percentage': coverage_percentage,
                'total_trading_days': total_trading_days,
                'days_with_data': days_with_data,
                'data_gaps': gaps,
                'invalid_rates': invalid_rates,
                'quality_score': quality_score,
                'meets_requirements': True
            }
            
        except (SELICDataUnavailableError, SELICDataInsufficientError) as e:
            # Re-raise these specific exceptions
            raise
        except Exception as e:
            logger.error(f"Error validating SELIC data coverage: {e}")
            raise SELICDataValidationError(f"Failed to validate SELIC data coverage: {e}")
    
    def get_selic_data_quality_score(self, data: pd.DataFrame) -> float:
        """
        Calculate SELIC data quality score (0-100%).
        
        Args:
            data (pd.DataFrame): SELIC data DataFrame
            
        Returns:
            float: Quality score from 0 to 100
        """
        if data is None or data.empty:
            return 0.0
        
        try:
            score = 0.0
            max_score = 100.0
            
            # Completeness (40% of score)
            completeness = 1.0 - (data['valor'].isna().sum() / len(data))
            score += completeness * 40
            
            # Consistency (30% of score)
            # Check for reasonable rate changes (not more than 50% in one day)
            if len(data) > 1:
                rate_changes = data['valor'].pct_change().abs()
                reasonable_changes = (rate_changes <= 0.5).sum()
                consistency = reasonable_changes / (len(data) - 1)
                score += consistency * 30
            else:
                score += 30  # Single data point gets full consistency score
            
            # Reasonableness (30% of score)
            # Check if rates are within reasonable bounds (0.1% to 100%)
            reasonable_rates = ((data['valor'] >= 0.001) & (data['valor'] <= 100.0)).sum()
            reasonableness = reasonable_rates / len(data)
            score += reasonableness * 30
            
            return min(score, max_score)
            
        except Exception as e:
            logger.error(f"Error calculating SELIC quality score: {e}")
            return 0.0
    
    def _find_data_gaps(self, data: pd.DataFrame, max_gap_days: int) -> List[Dict[str, Any]]:
        """
        Find gaps in SELIC data that exceed the maximum allowed gap.
        
        Args:
            data (pd.DataFrame): SELIC data DataFrame
            max_gap_days (int): Maximum allowed gap in days
            
        Returns:
            List[Dict[str, Any]]: List of gap information
        """
        gaps = []
        
        if len(data) < 2:
            return gaps
        
        # Sort by date
        data_sorted = data.sort_index()
        
        # Find gaps
        for i in range(1, len(data_sorted)):
            current_date = data_sorted.index[i]
            previous_date = data_sorted.index[i-1]
            
            # Calculate business days between dates
            business_days = len(pd.bdate_range(previous_date, current_date)) - 1
            
            if business_days > max_gap_days:
                gaps.append({
                    'start_date': previous_date,
                    'end_date': current_date,
                    'gap_days': business_days,
                    'max_allowed': max_gap_days
                })
        
        return gaps
    
    def _check_rate_validity(self, data: pd.DataFrame, rate_range: List[float]) -> List[Dict[str, Any]]:
        """
        Check if SELIC rates are within valid range.
        
        Args:
            data (pd.DataFrame): SELIC data DataFrame
            rate_range (List[float]): Valid rate range [min, max]
            
        Returns:
            List[Dict[str, Any]]: List of invalid rate information
        """
        invalid_rates = []
        
        if len(data) == 0:
            return invalid_rates
        
        min_rate, max_rate = rate_range
        
        # Find rates outside valid range
        invalid_mask = (data['valor'] < min_rate) | (data['valor'] > max_rate)
        invalid_data = data[invalid_mask]
        
        for date, row in invalid_data.iterrows():
            invalid_rates.append({
                'date': date,
                'rate': row['valor'],
                'min_allowed': min_rate,
                'max_allowed': max_rate
            })
        
        return invalid_rates
    
    def _calculate_selic_quality_score(self, data: pd.DataFrame, gaps: List[Dict], invalid_rates: List[Dict]) -> float:
        """
        Calculate overall SELIC data quality score.
        
        Args:
            data (pd.DataFrame): SELIC data DataFrame
            gaps (List[Dict]): List of data gaps
            invalid_rates (List[Dict]): List of invalid rates
            
        Returns:
            float: Quality score from 0 to 100
        """
        if data is None or data.empty:
            return 0.0
        
        try:
            base_score = self.get_selic_data_quality_score(data)
            
            # Penalize for gaps
            gap_penalty = len(gaps) * 5  # 5 points per gap
            
            # Penalize for invalid rates
            rate_penalty = len(invalid_rates) * 10  # 10 points per invalid rate
            
            # Calculate final score
            final_score = max(0.0, base_score - gap_penalty - rate_penalty)
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating final SELIC quality score: {e}")
            return 0.0
    
    def generate_selic_data_report(self, start_date: str, end_date: str) -> str:
        """
        Generate detailed report about SELIC data availability.
        
        Args:
            start_date (str): Start date in format 'dd/mm/yyyy'
            end_date (str): End date in format 'dd/mm/yyyy'
            
        Returns:
            str: Detailed report
        """
        try:
            coverage_info = self.validate_selic_data_coverage(start_date, end_date)
            
            report = f"""
SELIC Data Analysis Report
==========================
Period: {start_date} to {end_date}
Strict Mode: {'Enabled' if coverage_info.get('strict_mode_enabled', False) else 'Disabled'}

Data Coverage: {coverage_info.get('coverage_percentage', 0):.1f}%
Total Trading Days: {coverage_info.get('total_trading_days', 0)}
Days with Data: {coverage_info.get('days_with_data', 0)}
Quality Score: {coverage_info.get('quality_score', 0):.1f}%

Data Gaps: {len(coverage_info.get('data_gaps', []))}
Invalid Rates: {len(coverage_info.get('invalid_rates', []))}

Requirements Met: {'Yes' if coverage_info.get('meets_requirements', False) else 'No'}
"""
            
            # Add gap details
            gaps = coverage_info.get('data_gaps', [])
            if gaps:
                report += "\nData Gaps:\n"
                for gap in gaps:
                    report += f"  - {gap['start_date'].strftime('%Y-%m-%d')} to {gap['end_date'].strftime('%Y-%m-%d')} ({gap['gap_days']} days)\n"
            
            # Add invalid rate details
            invalid_rates = coverage_info.get('invalid_rates', [])
            if invalid_rates:
                report += "\nInvalid Rates:\n"
                for rate in invalid_rates[:5]:  # Show first 5
                    report += f"  - {rate['date'].strftime('%Y-%m-%d')}: {rate['rate']:.6f}\n"
                if len(invalid_rates) > 5:
                    report += f"  ... and {len(invalid_rates) - 5} more\n"
            
            return report
            
        except Exception as e:
            return f"Error generating SELIC data report: {e}"
    
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