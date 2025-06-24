"""
Data Downloader for B3 (Brazilian Stock Exchange) using Alpha Vantage API.

This module provides functionality to download and manage stock data from B3
using the Alpha Vantage API. It includes robust error handling, retry logic,
and proper data organization.

Author: Your Name
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
from typing import List, Dict, Optional, Tuple
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_secrets() -> Dict:
    """
    Load API keys and secrets from secrets.yaml file.
    
    Returns:
        Dict: Dictionary containing API configurations
        
    Raises:
        FileNotFoundError: If secrets.yaml file doesn't exist
        KeyError: If required API keys are missing
    """
    secrets_path = Path("config/secrets.yaml")
    
    if not secrets_path.exists():
        raise FileNotFoundError(
            "secrets.yaml file not found! Please copy secrets.yaml.example to "
            "secrets.yaml and add your API keys."
        )
    
    with open(secrets_path, 'r') as file:
        secrets = yaml.safe_load(file)
    
    # Validate required keys
    if 'alpha_vantage' not in secrets:
        raise KeyError("alpha_vantage configuration missing in secrets.yaml")
    
    if 'api_key' not in secrets['alpha_vantage']:
        raise KeyError("alpha_vantage.api_key missing in secrets.yaml")
    
    return secrets


class B3DataDownloader:
    """
    A class to download B3 stock data using Alpha Vantage API.
    
    This class handles:
    - Fetching the list of all B3 symbols from Alpha Vantage
    - Downloading daily adjusted data for each symbol
    - Saving data to CSV files and metadata to JSON files
    - Robust error handling with retry logic
    """
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        Initialize the B3DataDownloader.
        
        Args:
            api_key (str): Your Alpha Vantage API key (optional if using secrets.yaml)
            base_url (str): Base URL for Alpha Vantage API (optional if using secrets.yaml)
        """
        # Load secrets if API key not provided
        if api_key is None or base_url is None:
            secrets = load_secrets()
            self.api_key = api_key or secrets['alpha_vantage']['api_key']
            self.base_url = base_url or secrets['alpha_vantage'].get('base_url', "https://www.alphavantage.co/query")
        else:
            self.api_key = api_key
            self.base_url = base_url
        
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting: Alpha Vantage free tier allows 5 calls per minute
        self.calls_per_minute = 5
        self.last_call_time = 0
        
    def _rate_limit(self):
        """Implement rate limiting to respect API limits."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        min_interval = 60 / self.calls_per_minute  # seconds between calls
        
        if time_since_last_call < min_interval:
            sleep_time = min_interval - time_since_last_call
            logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def _make_request_with_retry(self, params: Dict, max_retries: int = 3) -> Optional[Dict]:
        """
        Make an API request with exponential backoff retry logic.
        
        Args:
            params (Dict): API parameters
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            Optional[Dict]: API response or None if all retries failed
        """
        for attempt in range(max_retries + 1):
            try:
                self._rate_limit()
                
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                # Check for API error messages
                if "Error Message" in data:
                    logger.error(f"API Error: {data['Error Message']}")
                    return None
                
                if "Note" in data:
                    logger.warning(f"API Note: {data['Note']}")
                    # If rate limited, wait longer and retry
                    if "premium" in data["Note"].lower() or "limit" in data["Note"].lower():
                        wait_time = 60 * (2 ** attempt)  # Exponential backoff
                        logger.info(f"Rate limited, waiting {wait_time} seconds before retry")
                        time.sleep(wait_time)
                        continue
                
                return data
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All retry attempts failed for params: {params}")
                    return None
        
        return None
    
    def get_b3_symbols(self) -> List[str]:
        """
        Fetch all B3 symbols from Alpha Vantage LISTING_STATUS endpoint.
        
        Returns:
            List[str]: List of B3 symbols without .SA suffix
        """
        logger.info("Fetching B3 symbols from Alpha Vantage...")
        
        params = {
            "function": "LISTING_STATUS",
            "apikey": self.api_key
        }
        
        data = self._make_request_with_retry(params)
        if not data:
            logger.error("Failed to fetch listing status data")
            return []
        
        # Parse CSV data
        try:
            # The LISTING_STATUS endpoint returns CSV data
            csv_text = requests.get(self.base_url, params=params).text
            df = pd.read_csv(pd.StringIO(csv_text))
            
            # Filter for B3 exchange (BVMF)
            b3_symbols = df[df['exchange'] == 'BVMF']['symbol'].tolist()
            
            # Remove .SA suffix
            b3_symbols_clean = [symbol.replace('.SA', '') for symbol in b3_symbols]
            
            logger.info(f"Found {len(b3_symbols_clean)} B3 symbols")
            return b3_symbols_clean
            
        except Exception as e:
            logger.error(f"Error parsing listing status data: {e}")
            return []
    
    def download_daily_data(self, symbol: str) -> Optional[Tuple[pd.DataFrame, Dict]]:
        """
        Download daily adjusted data for a given symbol.
        
        Args:
            symbol (str): Stock symbol (without .SA suffix)
            
        Returns:
            Optional[Tuple[pd.DataFrame, Dict]]: DataFrame with data and metadata, or None if failed
        """
        logger.info(f"Downloading daily data for {symbol}")
        
        # Add .SA suffix for Alpha Vantage
        full_symbol = f"{symbol}.SA"
        
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": full_symbol,
            "outputsize": "full",
            "apikey": self.api_key
        }
        
        data = self._make_request_with_retry(params)
        if not data:
            logger.error(f"Failed to download data for {symbol}")
            return None
        
        try:
            # Extract time series data
            time_series_key = "Time Series (Daily)"
            if time_series_key not in data:
                logger.error(f"No time series data found for {symbol}")
                return None
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Rename columns
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. adjusted close': 'adjusted_close',
                '6. volume': 'volume'
            }
            df = df.rename(columns=column_mapping)
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Create DateTimeIndex
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()  # Ensure ascending order
            
            # Limit to last 15 years
            start_date = datetime.now() - timedelta(days=15*365)
            df = df[df.index >= start_date]
            
            # Create metadata
            metadata = {
                "ticker": symbol,
                "name": symbol,  # We don't have company names from this endpoint
                "download_date": datetime.now().isoformat(),
                "data_points": len(df),
                "date_range": {
                    "start": df.index.min().isoformat() if len(df) > 0 else None,
                    "end": df.index.max().isoformat() if len(df) > 0 else None
                }
            }
            
            logger.info(f"Successfully downloaded {len(df)} data points for {symbol}")
            return df, metadata
            
        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {e}")
            return None
    
    def save_data(self, symbol: str, df: pd.DataFrame, metadata: Dict):
        """
        Save data and metadata to files.
        
        Args:
            symbol (str): Stock symbol
            df (pd.DataFrame): Data to save
            metadata (Dict): Metadata to save
        """
        try:
            # Save CSV data
            csv_path = self.data_dir / f"{symbol}_raw.csv"
            df.to_csv(csv_path)
            logger.info(f"Saved data to {csv_path}")
            
            # Save metadata
            meta_path = self.data_dir / f"{symbol}_meta.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata to {meta_path}")
            
        except Exception as e:
            logger.error(f"Error saving data for {symbol}: {e}")
    
    def download_all_b3_data(self, symbols: Optional[List[str]] = None, max_symbols: Optional[int] = None):
        """
        Download data for all B3 symbols or a subset.
        
        Args:
            symbols (Optional[List[str]]): Specific symbols to download. If None, fetches all B3 symbols
            max_symbols (Optional[int]): Maximum number of symbols to download (for testing)
        """
        if symbols is None:
            symbols = self.get_b3_symbols()
        
        if max_symbols:
            symbols = symbols[:max_symbols]
            logger.info(f"Limiting download to {max_symbols} symbols for testing")
        
        successful_downloads = 0
        failed_downloads = 0
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"Processing {symbol} ({i}/{len(symbols)})")
            
            result = self.download_daily_data(symbol)
            if result:
                df, metadata = result
                self.save_data(symbol, df, metadata)
                successful_downloads += 1
            else:
                failed_downloads += 1
            
            # Progress update every 10 symbols
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(symbols)} symbols processed")
        
        logger.info(f"Download completed. Successful: {successful_downloads}, Failed: {failed_downloads}")


def main():
    """Example usage of the B3DataDownloader class."""
    
    try:
        # Initialize downloader (will load API key from secrets.yaml)
        downloader = B3DataDownloader()
        
        # Example 1: Download data for a single symbol
        logger.info("=== Example 1: Download single symbol ===")
        result = downloader.download_daily_data("VALE3")
        if result:
            df, metadata = result
            downloader.save_data("VALE3", df, metadata)
            print(f"Downloaded {len(df)} data points for VALE3")
            print(f"Date range: {metadata['date_range']}")
        
        # Example 2: Download data for multiple symbols (limited for testing)
        logger.info("=== Example 2: Download multiple symbols ===")
        test_symbols = ["VALE3", "PETR4", "ITUB4"]
        downloader.download_all_b3_data(symbols=test_symbols)
        
        # Example 3: Download all B3 symbols (uncomment to run)
        # logger.info("=== Example 3: Download all B3 symbols ===")
        # downloader.download_all_b3_data(max_symbols=10)  # Limit to 10 for testing
        
    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        print("\nTo fix this error:")
        print("1. Copy config/secrets.yaml.example to config/secrets.yaml")
        print("2. Add your Alpha Vantage API key to the secrets.yaml file")
    except KeyError as e:
        logger.error(f"Missing configuration: {e}")
        print("\nPlease check your secrets.yaml file and ensure all required keys are present.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main() 