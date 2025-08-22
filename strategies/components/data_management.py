"""
Data Management Component for FuzzyFajuto Strategy

Handles data fetching, caching, validation, and daily data management.
Extracted from the monolithic strategy class for better maintainability.

Author: Senior Python Developer
Date: 2025
"""

import logging
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List
from engine.brapi_provider import BrapiProvider
from engine.logging_config import get_logger

logger = get_logger(__name__)


class DataManager:
    """
    Manages all data operations for the FuzzyFajuto strategy.
    
    Responsibilities:
    - Daily data fetching and caching
    - Data validation and quality checks
    - BRAPI provider management
    - Data freshness and update logic
    """
    
    def __init__(self, config: Dict[str, Any], context):
        """Initialize data manager with configuration and context."""
        self.config = config
        self.context = context
        self.brapi_provider = None
        
        # Data caches and state
        self.daily_data_cache: Dict[str, pd.DataFrame] = {}
        self.last_daily_update: Dict[str, date] = {}
        self.insufficient_daily_reported_dates: Dict[str, set] = {}
        
        # Initialize BRAPI provider
        self._initialize_brapi_provider()
    
    def _initialize_brapi_provider(self):
        """Initialize BRAPI provider for daily data fetching."""
        try:
            brapi_token = self.config.get('brapi_token')
            if not brapi_token:
                logger.warning("No BRAPI token found in configuration")
                return
            
            self.brapi_provider = BrapiProvider(
                token=brapi_token,
                cache_dir=self.config.get('brapi_cache_dir', 'data/brapi_cache')
            )
            logger.info("BRAPI provider initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize BRAPI provider: {e}")
            self.brapi_provider = None
    
    def get_daily_data_for_date(self, symbol: str, target_date: date) -> pd.DataFrame:
        """
        Get daily data for a specific symbol and date.
        
        Args:
            symbol: Stock symbol
            target_date: Target date for data
            
        Returns:
            DataFrame with daily data or empty DataFrame if unavailable
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{target_date}"
            if cache_key in self.daily_data_cache:
                return self.daily_data_cache[cache_key]
            
            if not self.brapi_provider:
                logger.warning(f"No BRAPI provider available for {symbol} on {target_date}")
                return pd.DataFrame()
            
            # Calculate date range (need history for indicators)
            start_date = target_date - timedelta(days=60)  # 60 days for indicators
            
            # Fetch data from BRAPI
            daily_df = self.brapi_provider.get_daily_data(
                symbol=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=target_date.strftime('%Y-%m-%d')
            )
            
            if daily_df.empty:
                logger.warning(f"No daily data available for {symbol} up to {target_date}")
                return pd.DataFrame()
            
            # Validate and clean data
            daily_df = self._validate_and_clean_daily_data(daily_df, symbol)
            
            # Cache the result
            self.daily_data_cache[cache_key] = daily_df
            
            logger.debug(f"Fetched daily data for {symbol}: {len(daily_df)} rows up to {target_date}")
            return daily_df
            
        except Exception as e:
            logger.error(f"Failed to get daily data for {symbol} on {target_date}: {e}")
            return pd.DataFrame()
    
    def _validate_and_clean_daily_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate and clean daily data.
        
        Args:
            df: Raw daily data DataFrame
            symbol: Stock symbol for logging
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        original_len = len(df)
        
        # Remove rows with missing OHLC data
        required_cols = ['open', 'high', 'low', 'close']
        df = df.dropna(subset=required_cols)
        
        # Remove rows with zero or negative prices
        for col in required_cols:
            df = df[df[col] > 0]
        
        # Validate OHLC relationships
        df = df[
            (df['high'] >= df['low']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])
        ]
        
        # Sort by date
        if 'date' in df.columns:
            df = df.sort_values('date')
        
        cleaned_len = len(df)
        if cleaned_len < original_len:
            logger.info(f"Cleaned daily data for {symbol}: {original_len} -> {cleaned_len} rows")
        
        return df
    
    def is_new_trading_day(self, symbol: str, current_date: date) -> bool:
        """
        Check if this is a new trading day that requires data updates.
        
        Args:
            symbol: Stock symbol
            current_date: Current trading date
            
        Returns:
            True if this is a new trading day
        """
        last_update = self.last_daily_update.get(symbol)
        if last_update is None or last_update < current_date:
            self.last_daily_update[symbol] = current_date
            return True
        return False
    
    def should_update_daily_data(self, symbol: str, current_date: date) -> bool:
        """
        Determine if daily data should be updated for the given symbol and date.
        
        Args:
            symbol: Stock symbol
            current_date: Current date
            
        Returns:
            True if data should be updated
        """
        # Always update on new trading day
        if self.is_new_trading_day(symbol, current_date):
            return True
        
        # Check if we have recent data
        cache_key = f"{symbol}_{current_date}"
        if cache_key not in self.daily_data_cache:
            return True
        
        cached_data = self.daily_data_cache[cache_key]
        if cached_data.empty:
            return True
        
        # Check data freshness (update if older than 1 hour)
        cache_time = getattr(cached_data, '_cache_time', None)
        if cache_time and (datetime.now() - cache_time).seconds > 3600:
            return True
        
        return False
    
    def fetch_new_daily_data(self, symbol: str, from_date: date, to_date: date) -> pd.DataFrame:
        """
        Fetch new daily data for a date range.
        
        Args:
            symbol: Stock symbol
            from_date: Start date
            to_date: End date
            
        Returns:
            DataFrame with new daily data
        """
        try:
            if not self.brapi_provider:
                logger.warning(f"No BRAPI provider available for {symbol}")
                return pd.DataFrame()
            
            # Fetch data from BRAPI
            daily_df = self.brapi_provider.get_daily_data(
                symbol=symbol,
                start_date=from_date.strftime('%Y-%m-%d'),
                end_date=to_date.strftime('%Y-%m-%d')
            )
            
            if daily_df.empty:
                return pd.DataFrame()
            
            # Validate and clean
            daily_df = self._validate_and_clean_daily_data(daily_df, symbol)
            
            # Add cache timestamp
            daily_df._cache_time = datetime.now()
            
            logger.debug(f"Fetched new daily data for {symbol}: {len(daily_df)} rows")
            return daily_df
            
        except Exception as e:
            logger.error(f"Failed to fetch new daily data for {symbol}: {e}")
            return pd.DataFrame()
    
    def refresh_daily_data(self, symbol: str, current_date: date) -> bool:
        """
        Refresh daily data for a symbol if needed.
        
        Args:
            symbol: Stock symbol
            current_date: Current date
            
        Returns:
            True if data was successfully refreshed
        """
        try:
            if not self.should_update_daily_data(symbol, current_date):
                return True
            
            # Fetch new data
            start_date = current_date - timedelta(days=60)
            new_data = self.fetch_new_daily_data(symbol, start_date, current_date)
            
            if new_data.empty:
                logger.warning(f"Failed to refresh daily data for {symbol}")
                return False
            
            # Update cache
            cache_key = f"{symbol}_{current_date}"
            self.daily_data_cache[cache_key] = new_data
            
            logger.debug(f"Successfully refreshed daily data for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error refreshing daily data for {symbol}: {e}")
            return False
    
    def get_daily_data_up_to_date(self, symbol: str, end_date: date) -> pd.DataFrame:
        """
        Get daily data up to a specific date with automatic refresh if needed.
        
        Args:
            symbol: Stock symbol
            end_date: End date for data
            
        Returns:
            DataFrame with daily data
        """
        # Try to refresh data first
        if self.refresh_daily_data(symbol, end_date):
            return self.get_daily_data_for_date(symbol, end_date)
        
        # Fallback to cached data
        cache_key = f"{symbol}_{end_date}"
        return self.daily_data_cache.get(cache_key, pd.DataFrame())
    
    def log_data_quality_report(self, symbol: str):
        """
        Log data quality report for a symbol.
        
        Args:
            symbol: Stock symbol to report on
        """
        try:
            # Get recent data for analysis
            current_date = date.today()
            recent_data = self.get_daily_data_up_to_date(symbol, current_date)
            
            if recent_data.empty:
                logger.warning(f"Data Quality Report for {symbol}: NO DATA AVAILABLE")
                return
            
            # Calculate quality metrics
            total_rows = len(recent_data)
            missing_volume = recent_data['volume'].isna().sum() if 'volume' in recent_data.columns else 0
            zero_volume_days = (recent_data['volume'] == 0).sum() if 'volume' in recent_data.columns else 0
            
            # Date range
            if 'date' in recent_data.columns:
                date_range = f"{recent_data['date'].min()} to {recent_data['date'].max()}"
            else:
                date_range = "Unknown"
            
            # Price statistics
            if 'close' in recent_data.columns:
                price_stats = {
                    'min': recent_data['close'].min(),
                    'max': recent_data['close'].max(),
                    'mean': recent_data['close'].mean()
                }
            else:
                price_stats = {'min': 0, 'max': 0, 'mean': 0}
            
            logger.info(f"Data Quality Report for {symbol}:")
            logger.info(f"  Date Range: {date_range}")
            logger.info(f"  Total Rows: {total_rows}")
            logger.info(f"  Missing Volume: {missing_volume}")
            logger.info(f"  Zero Volume Days: {zero_volume_days}")
            logger.info(f"  Price Range: {price_stats['min']:.2f} - {price_stats['max']:.2f} (avg: {price_stats['mean']:.2f})")
            
        except Exception as e:
            logger.error(f"Failed to generate data quality report for {symbol}: {e}")
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear data cache for a symbol or all symbols.
        
        Args:
            symbol: Optional symbol to clear, or None for all
        """
        if symbol:
            keys_to_remove = [k for k in self.daily_data_cache.keys() if k.startswith(symbol)]
            for key in keys_to_remove:
                del self.daily_data_cache[key]
            logger.debug(f"Cleared cache for {symbol}")
        else:
            self.daily_data_cache.clear()
            self.last_daily_update.clear()
            logger.debug("Cleared all data cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cached_symbols': len(set(k.split('_')[0] for k in self.daily_data_cache.keys())),
            'total_cache_entries': len(self.daily_data_cache),
            'symbols_with_updates': len(self.last_daily_update),
            'brapi_provider_available': self.brapi_provider is not None
        }
