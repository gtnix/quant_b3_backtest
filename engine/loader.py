"""
Load and preprocess B3 market data
Handle corporate actions, liquidity filters, and data quality

This module provides functionality to:
- Load raw CSV data from Alpha Vantage downloads
- Apply B3-specific processing and filters
- Calculate technical indicators and features
- Handle missing data and data quality issues
- Prepare data for backtesting strategies

"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, List
import json
from datetime import datetime, timedelta

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
    """
    
    def __init__(self, raw_path: str = "data/raw", processed_path: str = "data/processed"):
        """
        Initialize the DataLoader.
        
        Args:
            raw_path (str): Path to raw data directory
            processed_path (str): Path to processed data directory
        """
        self.raw_path = Path(raw_path)
        self.processed_path = Path(processed_path)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        # B3-specific configuration
        self.min_volume_brl = 1_000_000  # Minimum daily volume in BRL
        self.min_price = 1.0  # Minimum price to avoid penny stocks
        self.max_price_change = 0.20  # Maximum daily price change (20%)
        
        logger.info(f"DataLoader initialized with raw_path: {self.raw_path}")
        logger.info(f"DataLoader initialized with processed_path: {self.processed_path}")
    
    def load_raw_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load raw data for a given ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Optional[pd.DataFrame]: Raw data or None if file not found
        """
        csv_file = self.raw_path / f"{ticker}_raw.csv"
        
        if not csv_file.exists():
            logger.error(f"Raw data file not found: {csv_file}")
            return None
        
        try:
            # Load CSV data
            data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            
            # Ensure proper column names
            expected_columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume']
            if not all(col in data.columns for col in expected_columns):
                logger.error(f"Missing expected columns in {ticker}. Expected: {expected_columns}")
                return None
            
            logger.info(f"Loaded raw data for {ticker}: {len(data)} rows")
            return data
            
        except Exception as e:
            logger.error(f"Error loading raw data for {ticker}: {e}")
            return None
    
    def load_metadata(self, ticker: str) -> Optional[Dict]:
        """
        Load metadata for a given ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Optional[Dict]: Metadata or None if file not found
        """
        meta_file = self.raw_path / f"{ticker}_meta.json"
        
        if not meta_file.exists():
            logger.warning(f"Metadata file not found: {meta_file}")
            return None
        
        try:
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            logger.error(f"Error loading metadata for {ticker}: {e}")
            return None
    
    def apply_data_quality_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data quality filters to remove problematic data points.
        
        Args:
            data (pd.DataFrame): Raw price data
            
        Returns:
            pd.DataFrame: Filtered data
        """
        initial_rows = len(data)
        
        # Remove rows with missing values
        data = data.dropna()
        logger.info(f"Removed {initial_rows - len(data)} rows with missing values")
        
        # Remove rows with zero or negative prices
        data = data[
            (data['open'] > 0) & 
            (data['high'] > 0) & 
            (data['low'] > 0) & 
            (data['close'] > 0) & 
            (data['adjusted_close'] > 0)
        ]
        
        # Remove rows with zero volume
        data = data[data['volume'] > 0]
        
        # Remove extreme price changes (likely data errors)
        data['price_change'] = data['close'].pct_change().abs()
        data = data[data['price_change'] <= self.max_price_change]
        data = data.drop('price_change', axis=1)
        
        # Remove very low-priced stocks (penny stocks)
        data = data[data['close'] >= self.min_price]
        
        logger.info(f"Applied quality filters. Remaining rows: {len(data)}")
        return data
    
    def apply_liquidity_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply liquidity filters to remove low-volume days.
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Data with liquidity filters applied
        """
        initial_rows = len(data)
        
        # Calculate daily volume in BRL
        data['volume_brl'] = data['volume'] * data['close']
        
        # Remove low liquidity days
        data = data[data['volume_brl'] >= self.min_volume_brl]
        
        # Remove the temporary column
        data = data.drop('volume_brl', axis=1)
        
        logger.info(f"Applied liquidity filters. Remaining rows: {len(data)} (removed {initial_rows - len(data)})")
        return data
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for trading strategies.
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Data with technical indicators
        """
        # Price-based indicators
        data['returns'] = data['adjusted_close'].pct_change()
        data['log_returns'] = np.log(data['adjusted_close'] / data['adjusted_close'].shift(1))
        
        # Moving averages
        data['sma_5'] = data['close'].rolling(window=5).mean()
        data['sma_10'] = data['close'].rolling(window=10).mean()
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        data['sma_200'] = data['close'].rolling(window=200).mean()
        
        # Exponential moving averages
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()
        
        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Volatility indicators
        data['volatility_5d'] = data['returns'].rolling(window=5).std()
        data['volatility_20d'] = data['returns'].rolling(window=20).std()
        data['volatility_60d'] = data['returns'].rolling(window=60).std()
        
        # Volume indicators
        data['volume_sma_20'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma_20']
        
        # Price momentum
        data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
        data['momentum_20d'] = data['close'] / data['close'].shift(20) - 1
        data['momentum_60d'] = data['close'] / data['close'].shift(60) - 1
        
        # Support and resistance levels (simplified)
        data['resistance_20d'] = data['high'].rolling(window=20).max()
        data['support_20d'] = data['low'].rolling(window=20).min()
        
        logger.info(f"Calculated {len([col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'adjusted_close', 'volume']])} technical indicators")
        return data
    
    def handle_corporate_actions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle corporate actions like splits and dividends.
        Note: Alpha Vantage data is already adjusted for splits.
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Data with corporate actions handled
        """
        # Alpha Vantage TIME_SERIES_DAILY_ADJUSTED already handles splits
        # We just need to ensure we're using adjusted_close for calculations
        
        # Create a price ratio to detect potential splits
        data['price_ratio'] = data['close'] / data['adjusted_close']
        
        # Flag potential splits (price ratio significantly different from 1)
        potential_splits = data[abs(data['price_ratio'] - 1) > 0.1]
        if len(potential_splits) > 0:
            logger.info(f"Detected {len(potential_splits)} potential corporate actions")
        
        # Remove the temporary column
        data = data.drop('price_ratio', axis=1)
        
        return data
    
    def load_and_process(self, ticker: str, start_date: Optional[str] = None, 
                        end_date: Optional[str] = None, save_processed: bool = True) -> Optional[pd.DataFrame]:
        """
        Load raw data and apply comprehensive processing.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (Optional[str]): Start date in 'YYYY-MM-DD' format
            end_date (Optional[str]): End date in 'YYYY-MM-DD' format
            save_processed (bool): Whether to save processed data
            
        Returns:
            Optional[pd.DataFrame]: Processed data or None if failed
        """
        logger.info(f"Processing data for {ticker}")
        
        # Load raw data
        data = self.load_raw_data(ticker)
        if data is None:
            return None
        
        # Load metadata
        metadata = self.load_metadata(ticker)
        
        # Apply date filters
        if start_date:
            start_dt = pd.to_datetime(start_date)
            data = data[data.index >= start_dt]
            logger.info(f"Filtered data from {start_date}")
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            data = data[data.index <= end_dt]
            logger.info(f"Filtered data until {end_date}")
        
        # Apply processing steps
        data = self.apply_data_quality_filters(data)
        data = self.handle_corporate_actions(data)
        data = self.apply_liquidity_filters(data)
        data = self.calculate_technical_indicators(data)
        
        # Remove rows with NaN values (from technical indicators)
        initial_rows = len(data)
        data = data.dropna()
        logger.info(f"Removed {initial_rows - len(data)} rows with NaN values from technical indicators")
        
        # Save processed data if requested
        if save_processed and len(data) > 0:
            processed_file = self.processed_path / f"{ticker}_processed.csv"
            data.to_csv(processed_file)
            
            # Save processing metadata
            processing_meta = {
                "ticker": ticker,
                "processing_date": datetime.now().isoformat(),
                "original_metadata": metadata,
                "processing_stats": {
                    "final_rows": len(data),
                    "date_range": {
                        "start": data.index.min().isoformat(),
                        "end": data.index.max().isoformat()
                    },
                    "filters_applied": [
                        "data_quality",
                        "liquidity",
                        "technical_indicators"
                    ]
                }
            }
            
            meta_file = self.processed_path / f"{ticker}_processing_meta.json"
            with open(meta_file, 'w') as f:
                json.dump(processing_meta, f, indent=2)
            
            logger.info(f"Saved processed data to {processed_file}")
        
        logger.info(f"Successfully processed {ticker}: {len(data)} rows")
        return data
    
    def get_available_tickers(self) -> List[str]:
        """
        Get list of available tickers with raw data.
        
        Returns:
            List[str]: List of available ticker symbols
        """
        tickers = []
        for file in self.raw_path.glob("*_raw.csv"):
            ticker = file.stem.replace("_raw", "")
            tickers.append(ticker)
        
        return sorted(tickers)
    
    def get_processed_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load already processed data if available.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Optional[pd.DataFrame]: Processed data or None if not found
        """
        processed_file = self.processed_path / f"{ticker}_processed.csv"
        
        if processed_file.exists():
            try:
                data = pd.read_csv(processed_file, index_col=0, parse_dates=True)
                logger.info(f"Loaded processed data for {ticker}: {len(data)} rows")
                return data
            except Exception as e:
                logger.error(f"Error loading processed data for {ticker}: {e}")
                return None
        
        return None


def main():
    """Example usage of the DataLoader class."""
    
    # Initialize data loader
    loader = DataLoader()
    
    # Get available tickers
    available_tickers = loader.get_available_tickers()
    print(f"Available tickers: {available_tickers}")
    
    # Process data for a single ticker
    if available_tickers:
        ticker = available_tickers[0]
        print(f"\nProcessing {ticker}...")
        
        # Process data for the last 2 years
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
        
        data = loader.load_and_process(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            save_processed=True
        )
        
        if data is not None:
            print(f"Successfully processed {ticker}")
            print(f"Data shape: {data.shape}")
            print(f"Date range: {data.index.min()} to {data.index.max()}")
            print(f"Columns: {list(data.columns)}")
            
            # Show some statistics
            print(f"\nPrice statistics:")
            print(f"Current price: {data['close'].iloc[-1]:.2f}")
            print(f"Average volume: {data['volume'].mean():,.0f}")
            print(f"Volatility (20d): {data['volatility_20d'].iloc[-1]*100:.2f}%")
            print(f"RSI: {data['rsi'].iloc[-1]:.2f}")


if __name__ == "__main__":
    main() 