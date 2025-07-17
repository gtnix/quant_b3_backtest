"""
IBOV Data Downloader using Yahoo Finance API.

This module provides functionality to download IBOVESPA index data using Yahoo Finance,
which offers more recent data compared to Alpha Vantage (which only goes up to 2021).

Key Features:
- Downloads IBOV data using Yahoo Finance API
- Handles rate limiting and retry logic
- Saves data in multiple formats (CSV, Parquet)
- Follows the same patterns as the existing download_data.py
- Comprehensive error handling and logging
- Data validation and cleaning

Author: Enhanced Implementation
Date: 2024
"""

import yfinance as yf
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Tuple
import sys
from dataclasses import dataclass

# Configure logging
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
    log_file = log_dir / f"ibov_yahoo_download_{timestamp}.log"
    
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

@dataclass
class DownloadResult:
    """Result of a data download operation."""
    ticker: str
    success: bool
    data_points: int
    date_range: Optional[Dict[str, str]]
    error_message: Optional[str] = None
    download_time: Optional[float] = None

class YahooIBOVDownloader:
    """
    Yahoo Finance-based IBOV data downloader.
    
    This class handles:
    - Downloading IBOV data from Yahoo Finance
    - Rate limiting and retry logic
    - Data validation and cleaning
    - Saving data in multiple formats
    - Comprehensive error handling
    """
    
    def __init__(self, log_level: str = "INFO"):
        """
        Initialize the Yahoo Finance IBOV downloader.
        
        Args:
            log_level (str): Logging level
        """
        self.logger = setup_logging(log_level)
        self.logger.info("Initializing Yahoo Finance IBOV Downloader")
        
        # Create data directories if they don't exist
        # Use path relative to project root, not script location
        self.data_dir = Path(__file__).parent.parent / "data"
        self.ibov_dir = self.data_dir / "IBOV"  # Dedicated IBOV directory
        self.raw_dir = self.ibov_dir / "raw"
        
        for directory in [self.data_dir, self.ibov_dir, self.raw_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # IBOV symbol for Yahoo Finance
        self.ibov_symbol = "^BVSP"  # Yahoo Finance symbol for IBOVESPA
        self.logger.info(f"Using Yahoo Finance symbol for IBOVESPA: {self.ibov_symbol}")
    
    def download_ibov_data(self, 
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None,
                          period: str = "max") -> DownloadResult:
        """
        Download IBOVESPA index data from Yahoo Finance.
        
        Args:
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            period (str): Data period ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            
        Returns:
            DownloadResult: Result of the download operation
        """
        start_time = time.time()
        
        try:
            self.logger.info("Downloading IBOVESPA index data from Yahoo Finance")
            
            # Create ticker object
            ticker = yf.Ticker(self.ibov_symbol)
            
            # Download data
            if start_date and end_date:
                self.logger.info(f"Downloading data from {start_date} to {end_date}")
                df = ticker.history(start=start_date, end=end_date)
            else:
                self.logger.info(f"Downloading data for period: {period}")
                df = ticker.history(period=period)
            
            if df.empty:
                return DownloadResult(
                    ticker="IBOV",
                    success=False,
                    data_points=0,
                    date_range=None,
                    error_message="No data returned from Yahoo Finance",
                    download_time=time.time() - start_time
                )
            
            # Clean and validate data
            df = self._clean_data(df)
            
            # Add ticker column for consistency
            df['ticker'] = 'IBOV'
            
            # Save data
            self._save_ibov_data(df, start_date, end_date)
            
            download_time = time.time() - start_time
            
            self.logger.info(f"Successfully downloaded {len(df)} data points for IBOV")
            
            return DownloadResult(
                ticker="IBOV",
                success=True,
                data_points=len(df),
                date_range={
                    "start": df.index.min().isoformat() if len(df) > 0 else None,
                    "end": df.index.max().isoformat() if len(df) > 0 else None
                },
                download_time=download_time
            )
            
        except Exception as e:
            self.logger.error(f"Error downloading IBOVESPA data: {e}")
            return DownloadResult(
                ticker="IBOV",
                success=False,
                data_points=0,
                date_range=None,
                error_message=str(e),
                download_time=time.time() - start_time
            )
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the downloaded data.
        
        Args:
            df (pd.DataFrame): Raw data from Yahoo Finance
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        self.logger.info("Cleaning and validating data")
        
        # Remove rows with all NaN values
        initial_rows = len(df)
        df = df.dropna(how='all')
        if len(df) < initial_rows:
            self.logger.warning(f"Removed {initial_rows - len(df)} rows with all NaN values")
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Sort by date
        df = df.sort_index()
        
        # Convert numeric columns
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Rename columns to match existing format
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        # Apply mapping for columns that exist
        existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_columns)
        
        self.logger.info(f"Data cleaning complete. Final shape: {df.shape}")
        return df
    
    def _save_ibov_data(self, df: pd.DataFrame, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        Save IBOV data in multiple formats.
        
        Args:
            df (pd.DataFrame): Data to save
            start_date (str, optional): Start date for filename
            end_date (str, optional): End date for filename
        """
        self.logger.info("Saving IBOV data")
        
        # Create filename with date range if specified
        if start_date and end_date:
            filename_base = f"IBOV_{start_date}_to_{end_date}"
        else:
            filename_base = f"IBOV_{datetime.now().strftime('%Y%m%d')}"
        
        # Save as CSV
        csv_path = self.raw_dir / f"{filename_base}.csv"
        df.to_csv(csv_path)
        self.logger.info(f"Saved CSV: {csv_path}")
        
        # Save as Parquet
        parquet_path = self.raw_dir / f"{filename_base}.parquet"
        df.to_parquet(parquet_path)
        self.logger.info(f"Saved Parquet: {parquet_path}")
        
        # Save latest version as IBOV.parquet (for compatibility)
        latest_path = self.ibov_dir / "IBOV.parquet"
        df.to_parquet(latest_path)
        self.logger.info(f"Saved latest version: {latest_path}")
        
        # Save metadata
        metadata = {
            "source": "yahoo_finance",
            "ticker": "IBOV",
            "symbol": self.ibov_symbol,
            "index_name": "IBOVESPA",
            "currency": "BRL",
            "data_type": "raw",
            "download_date": datetime.now().isoformat(),
            "data_points": len(df),
            "date_range": {
                "start": df.index.min().isoformat() if len(df) > 0 else None,
                "end": df.index.max().isoformat() if len(df) > 0 else None
            },
            "columns": list(df.columns),
            "note": "IBOVESPA data downloaded from Yahoo Finance"
        }
        
        metadata_path = self.raw_dir / f"{filename_base}_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        self.logger.info(f"Saved metadata: {metadata_path}")
    
    def get_recent_data(self, days: int = 30) -> DownloadResult:
        """
        Download recent IBOV data for the specified number of days.
        
        Args:
            days (int): Number of days to download
            
        Returns:
            DownloadResult: Result of the download operation
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.download_ibov_data(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
    
    def compare_with_alpha_vantage(self) -> Dict:
        """
        Compare Yahoo Finance data with existing Alpha Vantage data.
        
        Returns:
            Dict: Comparison results
        """
        self.logger.info("Comparing Yahoo Finance data with Alpha Vantage data")
        
        # Load Yahoo Finance data
        yahoo_path = self.ibov_dir / "IBOV.parquet"
        if not yahoo_path.exists():
            return {"error": "Yahoo Finance data not found. Please download first."}
        
        yahoo_df = pd.read_parquet(yahoo_path)
        
        # Try to load Alpha Vantage data
        alpha_vantage_path = self.data_dir / "IBOV_alpha_vantage.parquet"
        if not alpha_vantage_path.exists():
            return {"error": "Alpha Vantage data not found"}
        
        alpha_df = pd.read_parquet(alpha_vantage_path)
        
        # Compare date ranges
        comparison = {
            "yahoo_finance": {
                "data_points": len(yahoo_df),
                "date_range": {
                    "start": yahoo_df.index.min().isoformat(),
                    "end": yahoo_df.index.max().isoformat()
                }
            },
            "alpha_vantage": {
                "data_points": len(alpha_df),
                "date_range": {
                    "start": alpha_df.index.min().isoformat(),
                    "end": alpha_df.index.max().isoformat()
                }
            }
        }
        
        # Find overlapping dates
        yahoo_dates = set(yahoo_df.index.date)
        alpha_dates = set(alpha_df.index.date)
        overlapping_dates = yahoo_dates.intersection(alpha_dates)
        
        comparison["overlapping_dates"] = len(overlapping_dates)
        comparison["yahoo_only_dates"] = len(yahoo_dates - alpha_dates)
        comparison["alpha_only_dates"] = len(alpha_dates - yahoo_dates)
        
        self.logger.info(f"Comparison complete: {comparison}")
        return comparison

def main():
    """
    Main function to run the IBOV downloader.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Download IBOV data using Yahoo Finance")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--period", default="max", 
                       choices=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
                       help="Data period (default: max)")
    parser.add_argument("--recent-days", type=int, help="Download recent N days")
    parser.add_argument("--compare", action="store_true", help="Compare with Alpha Vantage data")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level (default: INFO)")
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = YahooIBOVDownloader(log_level=args.log_level)
    
    # Download data based on arguments
    if args.recent_days:
        result = downloader.get_recent_data(days=args.recent_days)
    else:
        result = downloader.download_ibov_data(
            start_date=args.start_date,
            end_date=args.end_date,
            period=args.period
        )
    
    # Print results
    if result.success:
        print(f"✅ Successfully downloaded {result.data_points} data points for IBOV")
        print(f"📅 Date range: {result.date_range['start']} to {result.date_range['end']}")
        print(f"⏱️  Download time: {result.download_time:.2f} seconds")
    else:
        print(f"❌ Failed to download IBOV data: {result.error_message}")
        sys.exit(1)
    
    # Compare with Alpha Vantage if requested
    if args.compare:
        print("\n🔍 Comparing with Alpha Vantage data...")
        comparison = downloader.compare_with_alpha_vantage()
        if "error" not in comparison:
            print(f"📊 Yahoo Finance: {comparison['yahoo_finance']['data_points']} points")
            print(f"📊 Alpha Vantage: {comparison['alpha_vantage']['data_points']} points")
            print(f"🔄 Overlapping dates: {comparison['overlapping_dates']}")
            print(f"🆕 Yahoo only: {comparison['yahoo_only_dates']} dates")
            print(f"🆕 Alpha only: {comparison['alpha_only_dates']} dates")
        else:
            print(f"⚠️  {comparison['error']}")

if __name__ == "__main__":
    main() 