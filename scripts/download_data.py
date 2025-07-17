"""
Enhanced Data Downloader for B3 (Brazilian Stock Exchange) using Alpha Vantage API.

This module provides comprehensive functionality to download and manage stock data from B3
using the Alpha Vantage API. It includes robust error handling, retry logic, data consolidation,
and proper data organization.

Key Features:
- Rate-limited API calls (5 calls/minute for free tier)
- Comprehensive error handling with retry mechanisms
- Flexible ticker retrieval with fallback mechanisms
- Data consolidation into long-format CSV
- Detailed logging and progress tracking
- Configuration management with validation
- Adjusted data only (includes corporate action adjustments)

Author: Enhanced Implementation
Date: 2024
"""

import requests
import pandas as pd
import json
import time
import logging
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import os
import sys
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add import for IBRA tickers
try:
    from .ibra_tickers import IBRA_TICKERS
except ImportError:
    # Fallback for when running as script
    import sys
    sys.path.append('.')
    from ibra_tickers import IBRA_TICKERS

# Configure comprehensive logging
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up comprehensive logging with both file and console handlers.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"data_download_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

# Data classes for type safety
@dataclass
class DownloadResult:
    """Result of a data download operation."""
    ticker: str
    success: bool
    data_points: int
    date_range: Optional[Dict[str, str]]
    error_message: Optional[str] = None
    download_time: Optional[float] = None

@dataclass
class DownloadStats:
    """Statistics for download operations."""
    total_requests: int = 0
    successful_downloads: int = 0
    failed_downloads: int = 0
    total_data_points: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_downloads / self.total_requests) * 100
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate total duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

class ConfigurationManager:
    """
    Manages configuration loading and validation.
    
    This class handles:
    - Loading API keys from secrets.yaml
    - Validating configuration parameters
    - Providing default values
    - Error handling for missing configurations
    """
    
    def __init__(self, config_path: str = "config/secrets.yaml"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path (str): Path to the secrets configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_configuration()
    
    def _load_configuration(self) -> Dict:
        """
        Load and validate configuration from secrets.yaml.
        
        Returns:
            Dict: Validated configuration dictionary
            
        Raises:
            FileNotFoundError: If secrets.yaml file doesn't exist
            KeyError: If required configuration keys are missing
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                "Please copy config/secrets.yaml.example to config/secrets.yaml "
                "and add your Alpha Vantage API key."
            )
        
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {self.config_path}: {e}")
        
        # Validate Alpha Vantage configuration
        if 'alpha_vantage' not in config:
            raise KeyError("alpha_vantage configuration missing in secrets.yaml")
        
        alpha_vantage_config = config['alpha_vantage']
        
        if 'api_key' not in alpha_vantage_config:
            raise KeyError("alpha_vantage.api_key missing in secrets.yaml")
        
        if not alpha_vantage_config['api_key'] or alpha_vantage_config['api_key'] == "YOUR_ALPHA_VANTAGE_API_KEY_HERE":
            raise ValueError("Please provide a valid Alpha Vantage API key in secrets.yaml")
        
        # Set default values
        alpha_vantage_config.setdefault('base_url', "https://www.alphavantage.co/query")
        alpha_vantage_config.setdefault('data_type', 'adjusted')  # Default to adjusted data
        alpha_vantage_config.setdefault('rate_limit_calls_per_minute', 5)
        alpha_vantage_config.setdefault('max_retries', 3)
        alpha_vantage_config.setdefault('timeout_seconds', 30)
        
        # Validate data_type setting
        if alpha_vantage_config['data_type'] != 'adjusted':
            raise ValueError(f"Invalid data_type '{alpha_vantage_config['data_type']}'. Only 'adjusted' is supported.")
        
        return config
    
    def get_alpha_vantage_config(self) -> Dict:
        """
        Get Alpha Vantage configuration.
        
        Returns:
            Dict: Alpha Vantage configuration dictionary
        """
        return self.config['alpha_vantage']

class EnhancedB3DataDownloader:
    """
    Enhanced B3 data downloader with comprehensive features.
    
    This class handles:
    - Rate-limited API calls with retry logic
    - Comprehensive error handling
    - Data consolidation and formatting
    - Progress tracking and statistics
    - Adjusted data only (includes corporate action adjustments)
    - Uses IBRA_TICKERS for Brazilian stock symbols
    """
    
    def __init__(self, config_path: str = "config/secrets.yaml"):
        """
        Initialize the enhanced B3 data downloader.
        
        Args:
            config_path (str): Path to configuration file
        """
        # Setup logging
        self.logger = setup_logging()
        
        # Load configuration
        self.config_manager = ConfigurationManager(config_path)
        self.alpha_vantage_config = self.config_manager.get_alpha_vantage_config()
        
        # Initialize API parameters
        self.api_key = self.alpha_vantage_config['api_key']
        self.base_url = self.alpha_vantage_config['base_url']
        self.data_type = self.alpha_vantage_config['data_type']
        self.calls_per_minute = self.alpha_vantage_config['rate_limit_calls_per_minute']
        self.max_retries = self.alpha_vantage_config['max_retries']
        self.timeout = self.alpha_vantage_config['timeout_seconds']
        
        # Setup directories
        self.raw_data_dir = Path("data/raw")
        self.processed_data_dir = Path("data/processed")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting
        self.last_call_time = 0
        self.rate_limit_lock = threading.Lock()
        
        # Statistics tracking
        self.stats = DownloadStats()
        
        self.logger.info(f"Enhanced B3 Data Downloader initialized successfully")
        self.logger.info(f"Data type configured: {self.data_type} (includes corporate action adjustments)")
    
    def _rate_limit(self):
        """Implement thread-safe rate limiting."""
        with self.rate_limit_lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            min_interval = 60 / self.calls_per_minute
            
            if time_since_last_call < min_interval:
                sleep_time = min_interval - time_since_last_call
                self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
            
            self.last_call_time = time.time()
    
    def _make_request_with_retry(self, params: Dict) -> Optional[Dict]:
        """
        Make an API request with exponential backoff retry logic.
        
        Args:
            params (Dict): API parameters
            
        Returns:
            Optional[Dict]: API response or None if all retries failed
        """
        for attempt in range(self.max_retries + 1):
            try:
                self._rate_limit()
                
                response = requests.get(
                    self.base_url, 
                    params=params, 
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Check for API error messages
                if "Error Message" in data:
                    self.logger.error(f"API Error: {data['Error Message']}")
                    return None
                
                if "Note" in data:
                    self.logger.warning(f"API Note: {data['Note']}")
                    # Handle rate limiting
                    if "premium" in data["Note"].lower() or "limit" in data["Note"].lower():
                        wait_time = 60 * (2 ** attempt)
                        self.logger.info(f"Rate limited, waiting {wait_time} seconds before retry")
                        time.sleep(wait_time)
                        continue
                
                return data
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"All retry attempts failed for params: {params}")
                    return None
        
        return None
    
    def download_ticker_data(self, ticker: str) -> DownloadResult:
        """
        Download daily adjusted data for a single ticker.
        
        Args:
            ticker (str): Stock ticker symbol (without .SA suffix)
            
        Returns:
            DownloadResult: Result of the download operation with adjusted data
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Downloading data for {ticker}")
            
            # Add .SA suffix for Alpha Vantage
            full_symbol = f"{ticker}.SA"
            
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": full_symbol,
                "outputsize": "full",
                "apikey": self.api_key
            }
            
            data = self._make_request_with_retry(params)
            if not data:
                return DownloadResult(
                    ticker=ticker,
                    success=False,
                    data_points=0,
                    date_range=None,
                    error_message="Failed to retrieve data from API",
                    download_time=time.time() - start_time
                )
            
            # Extract and process time series data
            time_series_key = "Time Series (Daily)"
            if time_series_key not in data:
                return DownloadResult(
                    ticker=ticker,
                    success=False,
                    data_points=0,
                    date_range=None,
                    error_message="No time series data found in API response",
                    download_time=time.time() - start_time
                )
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Rename columns - TIME_SERIES_DAILY_ADJUSTED format
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. adjusted close': 'adjusted_close',
                '6. volume': 'volume',
                '7. dividend amount': 'dividend_amount',
                '8. split coefficient': 'split_coefficient'
            }
            
            # Apply mapping for columns that exist
            existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
            df = df.rename(columns=existing_columns)
            
            # Log which columns were found and mapped
            if existing_columns:
                self.logger.info(f"Column mapping for {ticker}: {existing_columns}")
            
            # Log any unmapped columns
            unmapped_columns = [col for col in df.columns if col not in existing_columns.values()]
            if unmapped_columns:
                self.logger.warning(f"Unmapped columns for {ticker}: {unmapped_columns}")
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Create DateTimeIndex
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Filter to last 15 years
            start_date = datetime.now() - timedelta(days=15*365)
            df = df[df.index >= start_date]
            
            # Add ticker column for consolidation
            df['ticker'] = ticker
            
            # Save individual ticker data
            self._save_ticker_data(ticker, df)
            
            download_time = time.time() - start_time
            
            return DownloadResult(
                ticker=ticker,
                success=True,
                data_points=len(df),
                date_range={
                    "start": df.index.min().isoformat() if len(df) > 0 else None,
                    "end": df.index.max().isoformat() if len(df) > 0 else None
                },
                download_time=download_time
            )
            
        except Exception as e:
            self.logger.error(f"Error downloading data for {ticker}: {e}")
            return DownloadResult(
                ticker=ticker,
                success=False,
                data_points=0,
                date_range=None,
                error_message=str(e),
                download_time=time.time() - start_time
            )
    
    def _save_ticker_data(self, ticker: str, df: pd.DataFrame):
        """
        Save individual ticker data to files.
        
        Args:
            ticker (str): Stock ticker symbol
            df (pd.DataFrame): Data to save
        """
        try:
            # Save CSV data
            csv_path = self.raw_data_dir / f"{ticker}_raw.csv"
            df.to_csv(csv_path)
            
            # Save enhanced metadata
            metadata = {
                "version": "1.3",
                "ticker": ticker,
                "currency": "BRL",
                "data_type": "adjusted",
                "price_type": {
                    "open": "raw",
                    "high": "raw",
                    "low": "raw",
                    "close": "raw",
                    "adjusted_close": "adjusted",
                    "dividend_amount": "raw",
                    "split_coefficient": "adjustment_factor"
                },
                "data_sources": "Alpha Vantage TIME_SERIES_DAILY_ADJUSTED",
                "download_date": datetime.now().isoformat(),
                "data_points": len(df),
                "date_range": {
                    "start": df.index.min().isoformat() if len(df) > 0 else None,
                    "end": df.index.max().isoformat() if len(df) > 0 else None
                },
                "columns": list(df.columns),
                "note": "OHLC prices are raw BRL values from B3. adjusted_close incorporates corporate actions."
            }
            
            meta_path = self.raw_data_dir / f"{ticker}_meta.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.debug(f"Saved data for {ticker}: {len(df)} rows")
            
        except Exception as e:
            self.logger.error(f"Error saving data for {ticker}: {e}")
    
    def download_multiple_tickers(self, tickers: List[str], max_workers: int = 1) -> List[DownloadResult]:
        """
        Download data for multiple tickers with optional parallel processing.
        
        Args:
            tickers (List[str]): List of ticker symbols to download
            max_workers (int): Number of parallel workers (1 for sequential)
            
        Returns:
            List[DownloadResult]: Results for all download operations
        """
        self.stats.start_time = datetime.now()
        self.stats.total_requests = len(tickers)
        
        self.logger.info(f"Starting adjusted data download for {len(tickers)} tickers")
        
        results = []
        
        if max_workers == 1:
            # Sequential processing
            for i, ticker in enumerate(tickers, 1):
                self.logger.info(f"Processing {ticker} ({i}/{len(tickers)})")
                result = self.download_ticker_data(ticker)
                results.append(result)
                
                # Update statistics
                if result.success:
                    self.stats.successful_downloads += 1
                    self.stats.total_data_points += result.data_points
                else:
                    self.stats.failed_downloads += 1
                
                # Progress update
                if i % 10 == 0:
                    self.logger.info(f"Progress: {i}/{len(tickers)} tickers processed")
        else:
            # Parallel processing (use with caution due to rate limits)
            self.logger.warning(f"Parallel processing with {max_workers} workers may hit rate limits")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_ticker = {
                    executor.submit(self.download_ticker_data, ticker): ticker 
                    for ticker in tickers
                }
                
                for i, future in enumerate(as_completed(future_to_ticker), 1):
                    ticker = future_to_ticker[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Update statistics
                        if result.success:
                            self.stats.successful_downloads += 1
                            self.stats.total_data_points += result.data_points
                        else:
                            self.stats.failed_downloads += 1
                        
                        # Progress update
                        if i % 10 == 0:
                            self.logger.info(f"Progress: {i}/{len(tickers)} tickers processed")
                            
                    except Exception as e:
                        self.logger.error(f"Error processing {ticker}: {e}")
                        self.stats.failed_downloads += 1
                        results.append(DownloadResult(
                            ticker=ticker,
                            success=False,
                            data_points=0,
                            date_range=None,
                            error_message=str(e)
                        ))
        
        self.stats.end_time = datetime.now()
        
        # Log final statistics
        self._log_download_statistics()
        
        return results
    
    def consolidate_data(self, output_filename: str = "b3_consolidated_data.csv") -> Optional[str]:
        """
        Consolidate all downloaded data into a single long-format CSV.
        
        Args:
            output_filename (str): Name of the output consolidated file
            
        Returns:
            Optional[str]: Path to the consolidated file or None if failed
        """
        try:
            self.logger.info("Starting data consolidation...")
            
            # Find all raw data files
            raw_files = list(self.raw_data_dir.glob("*_raw.csv"))
            
            if not raw_files:
                self.logger.warning("No raw data files found for consolidation")
                return None
            
            # Load and combine all data
            all_data = []
            
            for file_path in raw_files:
                try:
                    ticker = file_path.stem.replace("_raw", "")
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    
                    # Ensure ticker column exists
                    if 'ticker' not in df.columns:
                        df['ticker'] = ticker
                    
                    # Reset index to make date a column
                    df = df.reset_index()
                    df = df.rename(columns={'index': 'date'})
                    
                    all_data.append(df)
                    self.logger.debug(f"Loaded data for {ticker}: {len(df)} rows")
                    
                except Exception as e:
                    self.logger.error(f"Error loading {file_path}: {e}")
                    continue
            
            if not all_data:
                self.logger.error("No valid data files found for consolidation")
                return None
            
            # Combine all data
            consolidated_df = pd.concat(all_data, ignore_index=True)
            
            # Ensure proper column order
            expected_columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume']
            for col in expected_columns:
                if col not in consolidated_df.columns:
                    self.logger.warning(f"Missing column {col} in consolidated data")
            
            # Reorder columns
            available_columns = [col for col in expected_columns if col in consolidated_df.columns]
            consolidated_df = consolidated_df[available_columns]
            
            # Sort by date and ticker
            consolidated_df = consolidated_df.sort_values(['date', 'ticker'])
            
            # Save consolidated data
            output_path = self.processed_data_dir / output_filename
            consolidated_df.to_csv(output_path, index=False)
            
            self.logger.info(f"Data consolidation completed: {len(consolidated_df)} total rows")
            self.logger.info(f"Consolidated file saved to: {output_path}")
            
            # Save consolidation metadata
            consolidation_meta = {
                "consolidation_date": datetime.now().isoformat(),
                "total_rows": len(consolidated_df),
                "unique_tickers": consolidated_df['ticker'].nunique(),
                "date_range": {
                    "start": consolidated_df['date'].min().isoformat(),
                    "end": consolidated_df['date'].max().isoformat()
                },
                "columns": list(consolidated_df.columns),
                "source_files": len(raw_files)
            }
            
            meta_path = self.processed_data_dir / "consolidation_metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(consolidation_meta, f, indent=2)
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error during data consolidation: {e}")
            return None
    
    def _log_download_statistics(self):
        """Log comprehensive download statistics."""
        self.logger.info("=" * 60)
        self.logger.info("DOWNLOAD STATISTICS")
        self.logger.info("=" * 60)
        self.logger.info(f"Total Requests: {self.stats.total_requests}")
        self.logger.info(f"Successful Downloads: {self.stats.successful_downloads}")
        self.logger.info(f"Failed Downloads: {self.stats.failed_downloads}")
        self.logger.info(f"Success Rate: {self.stats.success_rate:.2f}%")
        self.logger.info(f"Total Data Points: {self.stats.total_data_points:,}")
        
        if self.stats.duration:
            self.logger.info(f"Total Duration: {self.stats.duration:.2f} seconds")
            if self.stats.successful_downloads > 0:
                avg_time = self.stats.duration / self.stats.successful_downloads
                self.logger.info(f"Average Time per Download: {avg_time:.2f} seconds")
        
        self.logger.info("=" * 60)
    
    def run_complete_download(self, 
                            tickers: Optional[List[str]] = None,
                            max_tickers: Optional[int] = None,
                            consolidate: bool = True,
                            output_filename: str = "b3_consolidated_data.csv") -> bool:
        """
        Run a complete download process with all features.
        
        Args:
            tickers (Optional[List[str]]): Specific tickers to download (defaults to IBRA_TICKERS)
            max_tickers (Optional[int]): Maximum number of tickers to download
            consolidate (bool): Whether to consolidate data into single CSV
            output_filename (str): Name of the consolidated output file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.info("Starting complete B3 data download process")
            
            # Get tickers
            if tickers is None:
                # Use IBRA_TICKERS as the default source
                tickers = IBRA_TICKERS.copy()
            
            if not tickers:
                self.logger.error("No tickers available for download")
                return False
            
            # Limit tickers if specified
            if max_tickers:
                tickers = tickers[:max_tickers]
                self.logger.info(f"Limited to {max_tickers} tickers for testing")
            
            self.logger.info(f"Downloading data for {len(tickers)} tickers")
            
            # Download data
            results = self.download_multiple_tickers(tickers)
            
            # Consolidate data if requested
            if consolidate:
                consolidated_path = self.consolidate_data(output_filename)
                if consolidated_path:
                    self.logger.info(f"Complete download process successful. Consolidated file: {consolidated_path}")
                    return True
                else:
                    self.logger.error("Data consolidation failed")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Complete download process failed: {e}")
            return False

def main():
    """
    Main function to download all B3 stocks from Alpha Vantage.
    """
    try:
        # Initialize the enhanced downloader
        downloader = EnhancedB3DataDownloader()
        
        print("🚀 B3 Adjusted Stocks Data Downloader")
        print("=" * 50)
        print("Downloading all IBRA stocks (adjusted data) for the last 15 years...")
        print("Note: This may take several hours due to API rate limits (5 calls/minute)")
        print()
        
        # Use IBRA tickers
        tickers = IBRA_TICKERS
        
        # Run complete download process for all IBRA stocks
        success = downloader.run_complete_download(
            tickers=tickers,
            max_tickers=None,  # No limit - download all available
            consolidate=True,
            output_filename="b3_adjusted_stocks_data.csv"
        )
        
        if success:
            print("✅ Download process completed successfully!")
            print(f"📁 Check the following directories for results:")
            print(f"   - Raw data: {downloader.raw_data_dir}")
            print(f"   - Processed data: {downloader.processed_data_dir}")
            print(f"   - Logs: logs/")
        else:
            print("❌ Download process failed. Check logs for details.")
        
    except FileNotFoundError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\n🔧 To fix this error:")
        print("1. Copy config/secrets.yaml.example to config/secrets.yaml")
        print("2. Add your Alpha Vantage API key to the secrets.yaml file")
        print("3. Get a free API key at: https://www.alphavantage.co/support/#api-key")
        
    except KeyError as e:
        print(f"\n❌ Missing Configuration: {e}")
        print("\n🔧 Please check your secrets.yaml file and ensure all required keys are present.")
        
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\n🔧 Please provide a valid Alpha Vantage API key in your secrets.yaml file.")
        
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        print("\n🔧 Check the logs for detailed error information.")

if __name__ == "__main__":
    main()
