"""
Load and preprocess B3 market data
Handle corporate actions, liquidity filters, and data quality

This module provides functionality to:
- Load raw CSV data from Alpha Vantage downloads
- Apply B3-specific processing and filters
- Calculate technical indicators and features
- Handle missing data and data quality issues
- Prepare data for backtesting strategies
- Automatically download missing data when needed
- Handle SGS and IBOV data downloads

"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, List, Any
import json
from datetime import datetime, timedelta
import sys
import os

# Add scripts directory to path for imports
scripts_path = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_path))

# Import download modules
try:
    from download_data import EnhancedB3DataDownloader, DownloadResult
    from download_ibov_yahoo import YahooIBOVDownloader
    from sgs_data_loader import SGSDataLoader
    DOWNLOAD_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Download modules not available: {e}")
    DOWNLOAD_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class DataLoader:
    """
    A class to load and process B3 market data for backtesting.
    
    This class handles:
    - Loading raw data from CSV files
    - Applying B3-specific filters and processing
    - Calculating technical indicators
    - Handling data quality issues
    - Preparing data for strategy backtesting
    - Automatically downloading missing data
    - Handling SGS and IBOV data
    """
    
    def __init__(self, raw_path: str = "data/raw", processed_path: str = "data/processed", 
                 auto_download: bool = True, config_path: str = "config/secrets.yaml"):
        """
        Initialize the DataLoader.
        
        Args:
            raw_path (str): Path to raw data directory
            processed_path (str): Path to processed data directory
            auto_download (bool): Whether to automatically download missing data
            config_path (str): Path to configuration file for API keys
        """
        self.raw_path = Path(raw_path)
        self.processed_path = Path(processed_path)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        # Auto-download configuration
        self.auto_download = auto_download and DOWNLOAD_AVAILABLE
        self.config_path = config_path
        
        # Initialize downloaders if available
        self.stock_downloader = None
        self.ibov_downloader = None
        self.sgs_downloader = None
        
        if self.auto_download:
            try:
                self.stock_downloader = EnhancedB3DataDownloader(config_path)
                self.ibov_downloader = YahooIBOVDownloader()
                self.sgs_downloader = SGSDataLoader()
                logger.info("Auto-download functionality initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize downloaders: {e}")
                self.auto_download = False
        
        # B3-specific configuration
        self.min_volume_brl = 1_000_000  # Minimum daily volume in BRL
        self.min_price = 1.0  # Minimum price to avoid penny stocks
        self.max_price_change = 0.20  # Maximum daily price change (20%)
        
        logger.info(f"DataLoader initialized with raw_path: {self.raw_path}")
        logger.info(f"DataLoader initialized with processed_path: {self.processed_path}")
        logger.info(f"Auto-download enabled: {self.auto_download}")
    
    def check_sgs_data(self) -> Dict[str, Any]:
        """
        Check for missing SGS data.
        
        Returns:
            Dict[str, Any]: Dictionary with SGS data status
        """
        if not self.auto_download or not self.sgs_downloader:
            return {
                'has_data': False,
                'missing_series': [],
                'available_series': [],
                'needs_download': False
            }
        
        try:
            # Check which SGS series files exist
            sgs_path = Path("data/sgs")
            available_series = []
            missing_series = []
            
            # Check for each SGS series (8, 11, 12, 433)
            for series_id in [8, 11, 12, 433]:
                # Check for processed files with the naming convention from SGSDataLoader
                series_files = list(sgs_path.glob(f"sgs_{series_id}_*.csv"))
                if series_files:
                    available_series.append(series_id)
                else:
                    missing_series.append(series_id)
            
            return {
                'has_data': len(available_series) > 0,
                'missing_series': missing_series,
                'available_series': available_series,
                'needs_download': len(missing_series) > 0
            }
            
        except Exception as e:
            logger.error(f"Error checking SGS data: {e}")
            return {
                'has_data': False,
                'missing_series': [8, 11, 12, 433],
                'available_series': [],
                'needs_download': True
            }
    
    def check_ibov_data(self) -> Dict[str, Any]:
        """
        Check for missing IBOV data.
        
        Returns:
            Dict[str, Any]: Dictionary with IBOV data status
        """
        if not self.auto_download or not self.ibov_downloader:
            return {
                'has_data': False,
                'missing_dates': [],
                'data_range': None,
                'needs_download': False
            }
        
        try:
            # Check IBOV data file (using the Yahoo Finance downloader's file structure)
            ibov_file = Path("data/IBOV/raw/IBOV_raw.csv")
            
            if not ibov_file.exists():
                return {
                    'has_data': False,
                    'missing_dates': [],
                    'data_range': None,
                    'needs_download': True
                }
            
            # Load existing data
            data = pd.read_csv(ibov_file, index_col=0, parse_dates=True)
            if data.empty:
                return {
                    'has_data': False,
                    'missing_dates': [],
                    'data_range': None,
                    'needs_download': True
                }
            
            # Get data range
            data_start = data.index.min()
            data_end = data.index.max()
            today = pd.Timestamp.now().normalize()
            
            # Check if data is up to date
            if data_end >= today:
                return {
                    'has_data': True,
                    'missing_dates': [],
                    'data_range': {
                        'start': data_start.isoformat(),
                        'end': data_end.isoformat()
                    },
                    'needs_download': False,
                    'is_up_to_date': True
                }
            
            # Check for missing dates from last available date to today
            missing_dates = []
            current_date = data_end + pd.Timedelta(days=1)
            
            while current_date <= today:
                if current_date not in data.index:
                    missing_dates.append(current_date)
                current_date += pd.Timedelta(days=1)
            
            # Use dias_uteis to get Brazilian business days
            try:
                from dias_uteis import range_du
                
                # Convert dates to datetime.date for dias_uteis
                start_date_du = (data_end + pd.Timedelta(days=1)).date()
                end_date_du = today.date()
                
                # Get business days between last available date and today
                business_days = range_du(start_date_du, end_date_du)
                business_days_list = [pd.Timestamp(date) for date in business_days]
                
                # Filter missing dates to only include business days
                missing_business_days = [date for date in missing_dates if date in business_days_list]
                
                logger.info(f"IBOV: Using dias_uteis: {len(business_days_list)} business days, {len(missing_business_days)} missing business days")
                
            except ImportError:
                # Fallback to pandas business day range if dias_uteis not available
                business_days = pd.bdate_range(start=data_end + pd.Timedelta(days=1), end=today)
                missing_business_days = [date for date in missing_dates if date in business_days]
                logger.warning("IBOV: dias_uteis not available, using pandas business day range")
            
            return {
                'has_data': True,
                'missing_dates': missing_business_days,
                'data_range': {
                    'start': data_start.isoformat(),
                    'end': data_end.isoformat()
                },
                'needs_download': len(missing_business_days) > 0,
                'missing_count': len(missing_business_days),
                'is_up_to_date': len(missing_business_days) == 0
            }
            
        except Exception as e:
            logger.error(f"Error checking IBOV data: {e}")
            return {
                'has_data': False,
                'missing_dates': [],
                'data_range': None,
                'needs_download': True
            }
    
    def _download_sgs_data(self) -> bool:
        """
        Download missing SGS data.
        
        Returns:
            bool: True if download was successful, False otherwise
        """
        if not self.auto_download or not self.sgs_downloader:
            logger.warning("SGS auto-download is disabled or SGS downloader not available")
            return False
        
        try:
            logger.info("Downloading missing SGS data...")
            
            # Get default date range (last year to today)
            end_date = datetime.now().strftime("%d/%m/%Y")
            start_date = (datetime.now() - timedelta(days=365)).strftime("%d/%m/%Y")
            
            # Download all series using the SGSDataLoader's method
            result = self.sgs_downloader.get_all_series_data(
                start_date=start_date,
                end_date=end_date,
                use_cache=False,
                save_processed=True
            )
            
            if result:
                logger.info(f"Successfully downloaded SGS data for {len(result)} series")
                return True
            else:
                logger.error("Failed to download SGS data")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading SGS data: {e}")
            return False
    
    def _download_ibov_data(self) -> bool:
        """
        Download missing IBOV data.
        
        Returns:
            bool: True if download was successful, False otherwise
        """
        if not self.auto_download or not self.ibov_downloader:
            logger.warning("IBOV auto-download is disabled or IBOV downloader not available")
            return False
        
        try:
            logger.info("Downloading missing IBOV data...")
            
            # Download recent data (last 30 days) using the YahooIBOVDownloader's method
            result = self.ibov_downloader.get_recent_data(days=30)
            
            if result.success:
                logger.info(f"Successfully downloaded IBOV data: {result.data_points} data points")
                return True
            else:
                logger.error(f"Failed to download IBOV data: {result.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading IBOV data: {e}")
            return False
    
    def check_missing_data(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Check for missing data for given tickers.
        
        Args:
            tickers (List[str]): List of ticker symbols to check
            
        Returns:
            Dict[str, Any]: Dictionary with missing data status
        """
        missing_tickers = []
        tickers_with_gaps = []
        total_missing_days = 0
        
        for ticker in tickers:
            ticker_file = self.raw_path / f"{ticker}_raw.csv"
            
            if not ticker_file.exists():
                missing_tickers.append(ticker)
                continue
            
            # Check for data gaps
            try:
                data = pd.read_csv(ticker_file, index_col=0, parse_dates=True)
                if data.empty:
                    missing_tickers.append(ticker)
                    continue
                
                # Check for recent missing dates
                data_end = data.index.max()
                today = pd.Timestamp.now().normalize()
                
                if data_end < today:
                    # Calculate missing business days
                    try:
                        from dias_uteis import range_du
                        start_date_du = (data_end + pd.Timedelta(days=1)).date()
                        end_date_du = today.date()
                        business_days = range_du(start_date_du, end_date_du)
                        missing_days = len(business_days)
                    except ImportError:
                        # Fallback to pandas business day range
                        business_days = pd.bdate_range(start=data_end + pd.Timedelta(days=1), end=today)
                        missing_days = len(business_days)
                    
                    if missing_days > 0:
                        tickers_with_gaps.append({
                            'ticker': ticker,
                            'last_date': data_end.isoformat(),
                            'missing_days': missing_days
                        })
                        total_missing_days += missing_days
                        
            except Exception as e:
                logger.error(f"Error checking data for {ticker}: {e}")
                missing_tickers.append(ticker)
        
        return {
            'missing_tickers': missing_tickers,
            'tickers_with_gaps': tickers_with_gaps,
            'summary': {
                'missing_tickers_count': len(missing_tickers),
                'tickers_with_gaps_count': len(tickers_with_gaps),
                'total_missing_days': total_missing_days
            }
        }
    
    def download_missing_data_batch(self, tickers: List[str], show_progress: bool = True) -> Dict[str, List[str]]:
        """
        Download missing data for a batch of tickers.
        
        Args:
            tickers (List[str]): List of ticker symbols to download
            show_progress (bool): Whether to show progress messages
            
        Returns:
            Dict[str, List[str]]: Dictionary with successful and failed tickers
        """
        if not self.auto_download or not self.stock_downloader:
            logger.warning("Auto-download is disabled or stock downloader not available")
            return {'success': [], 'failed': tickers}
        
        successful = []
        failed = []
        
        if show_progress:
            logger.info(f"Downloading data for {len(tickers)} tickers...")
        
        for i, ticker in enumerate(tickers, 1):
            if show_progress:
                logger.info(f"Downloading {ticker} ({i}/{len(tickers)})...")
            
            try:
                result = self.stock_downloader.download_ticker_data(ticker)
                if result.success:
                    successful.append(ticker)
                    if show_progress:
                        logger.info(f"✓ {ticker}: {result.data_points} data points")
                else:
                    failed.append(ticker)
                    if show_progress:
                        logger.error(f"✗ {ticker}: {result.error_message}")
                        
            except Exception as e:
                failed.append(ticker)
                if show_progress:
                    logger.error(f"✗ {ticker}: {e}")
        
        if show_progress:
            logger.info(f"Download completed: {len(successful)} successful, {len(failed)} failed")
        
        return {'success': successful, 'failed': failed}
    
    def check_all_data(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Check for missing data across all sources (tickers, SGS, IBOV).
        
        Args:
            tickers (List[str]): List of ticker symbols to check
            
        Returns:
            Dict[str, Any]: Dictionary with comprehensive data status
        """
        # Check ticker data
        ticker_status = self.check_missing_data(tickers)
        
        # Check SGS data
        sgs_status = self.check_sgs_data()
        
        # Check IBOV data
        ibov_status = self.check_ibov_data()
        
        return {
            'tickers': ticker_status,
            'sgs': sgs_status,
            'ibov': ibov_status,
            'summary': {
                'total_missing_tickers': ticker_status['summary']['missing_tickers_count'],
                'total_tickers_with_gaps': ticker_status['summary']['tickers_with_gaps_count'],
                'total_missing_days': ticker_status['summary']['total_missing_days'],
                'sgs_needs_download': sgs_status['needs_download'],
                'ibov_needs_download': ibov_status['needs_download'],
                'any_missing_data': (
                    ticker_status['summary']['missing_tickers_count'] > 0 or
                    ticker_status['summary']['tickers_with_gaps_count'] > 0 or
                    sgs_status['needs_download'] or
                    ibov_status['needs_download']
                )
            }
        }
    
    def download_all_missing_data(self, tickers: List[str], show_progress: bool = True) -> Dict[str, Any]:
        """
        Download all missing data (tickers, SGS, IBOV).
        
        Args:
            tickers (List[str]): List of ticker symbols to download
            show_progress (bool): Whether to show progress messages
            
        Returns:
            Dict[str, Any]: Dictionary with download results
        """
        if not self.auto_download:
            logger.warning("Auto-download is disabled")
            return {'success': False, 'message': 'Auto-download disabled'}
        
        results = {
            'tickers': {'success': [], 'failed': []},
            'sgs': {'success': False, 'message': ''},
            'ibov': {'success': False, 'message': ''}
        }
        
        if show_progress:
            logger.info("Starting comprehensive data download...")
        
        # Download missing tickers
        if tickers:
            ticker_results = self.download_missing_data_batch(tickers, show_progress)
            results['tickers'] = ticker_results
        
        # Download SGS data if needed
        sgs_status = self.check_sgs_data()
        if sgs_status['needs_download']:
            if show_progress:
                logger.info("Downloading SGS data...")
            sgs_success = self._download_sgs_data()
            results['sgs'] = {
                'success': sgs_success,
                'message': 'SGS data downloaded successfully' if sgs_success else 'Failed to download SGS data'
            }
        
        # Download IBOV data if needed
        ibov_status = self.check_ibov_data()
        if ibov_status['needs_download']:
            if show_progress:
                logger.info("Downloading IBOV data...")
            ibov_success = self._download_ibov_data()
            results['ibov'] = {
                'success': ibov_success,
                'message': 'IBOV data downloaded successfully' if ibov_success else 'Failed to download IBOV data'
            }
        
        if show_progress:
            logger.info("Comprehensive data download completed")
        
        return results
    
    def load_raw_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load raw data for a ticker, attempting to download if missing.
        
        Args:
            ticker (str): Ticker symbol
            
        Returns:
            Optional[pd.DataFrame]: Raw data or None if not available
        """
        ticker_file = self.raw_path / f"{ticker}_raw.csv"
        
        if not ticker_file.exists():
            if self.auto_download and self.stock_downloader:
                logger.info(f"Data not found for {ticker}, attempting to download...")
                result = self.stock_downloader.download_ticker_data(ticker)
                if not result.success:
                    logger.error(f"Failed to download data for {ticker}: {result.error_message}")
                    return None
            else:
                logger.error(f"Data not found for {ticker} and auto-download is disabled")
                return None
        
        try:
            data = pd.read_csv(ticker_file, index_col=0, parse_dates=True)
            return data
        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {e}")
            return None 