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
            
            # Handle column name variations and additional columns
            column_mapping = {}
            
            # Map potential column name variations
            for col in data.columns:
                col_lower = col.lower().strip()
                
                # Handle numbered prefixes for main price columns
                if col_lower.startswith('1.') or col_lower.startswith('1 '):
                    column_mapping[col] = 'open'
                elif col_lower.startswith('2.') or col_lower.startswith('2 '):
                    column_mapping[col] = 'high'
                elif col_lower.startswith('3.') or col_lower.startswith('3 '):
                    column_mapping[col] = 'low'
                elif col_lower.startswith('4.') or col_lower.startswith('4 '):
                    column_mapping[col] = 'close'
                elif col_lower.startswith('5.') or col_lower.startswith('5 '):
                    column_mapping[col] = 'adjusted_close'
                elif col_lower.startswith('6.') or col_lower.startswith('6 '):
                    column_mapping[col] = 'volume'
                # Handle numbered prefixes for additional columns
                elif col_lower.startswith('7.') or col_lower.startswith('7 '):
                    column_mapping[col] = 'dividend_amount'
                elif col_lower.startswith('8.') or col_lower.startswith('8 '):
                    column_mapping[col] = 'split_coefficient'
                elif col_lower == 'ticker':
                    column_mapping[col] = 'ticker'
                # Keep standard columns as they are
                elif col_lower in ['open', 'high', 'low', 'close', 'adjusted_close', 'volume']:
                    column_mapping[col] = col_lower
            
            # Apply column mapping
            if column_mapping:
                data = data.rename(columns=column_mapping)
                logger.info(f"Applied column mapping for {ticker}: {column_mapping}")
            
            # Ensure required columns exist (be more flexible for index data)
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns in {ticker}: {missing_columns}")
                logger.error(f"Available columns: {list(data.columns)}")
                return None
            
            # Handle adjusted_close based on ticker type
            if 'adjusted_close' not in data.columns:
                # For index data (like IBOV), use close as adjusted_close
                if ticker.upper() in ['IBOV', 'IBOV.SA', 'IBOVESPA', '^BVSP']:
                    logger.info(f"Using close as adjusted_close for index {ticker}")
                    data['adjusted_close'] = data['close']
                else:
                    # For stocks, they should have adjusted_close
                    logger.error(f"Stock {ticker} missing adjusted_close column - this is required for stocks")
                    logger.error(f"Available columns: {list(data.columns)}")
                    return None
            
            # Keep only required columns plus any additional useful columns
            columns_to_keep = required_columns.copy()

            # Always keep adjusted_close if present
            if 'adjusted_close' in data.columns and 'adjusted_close' not in columns_to_keep:
                columns_to_keep.append('adjusted_close')
            
            # Add dividend and split information if available
            if 'dividend_amount' in data.columns:
                columns_to_keep.append('dividend_amount')
                logger.info(f"Found dividend_amount column for {ticker}")
            
            if 'split_coefficient' in data.columns:
                columns_to_keep.append('split_coefficient')
                logger.info(f"Found split_coefficient column for {ticker}")
            
            # Remove ticker column if it exists (we already know the ticker)
            if 'ticker' in columns_to_keep:
                columns_to_keep.remove('ticker')
            
            # Select only the columns we want to keep
            data = data[columns_to_keep]
            
            # Convert numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Convert dividend and split columns if they exist
            if 'dividend_amount' in data.columns:
                data['dividend_amount'] = pd.to_numeric(data['dividend_amount'], errors='coerce')
            
            if 'split_coefficient' in data.columns:
                data['split_coefficient'] = pd.to_numeric(data['split_coefficient'], errors='coerce')
            
            logger.info(f"Loaded raw data for {ticker}: {len(data)} rows with columns: {list(data.columns)}")
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
        data['ema_50'] = data['close'].ewm(span=50).mean()
        
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
        Enhanced to use actual dividend and split data when available.
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Data with corporate actions handled
        """
        # Track corporate actions for reporting
        corporate_actions = {
            'dividends': 0,
            'splits': 0,
            'total_actions': 0
        }
        
        # Handle dividends if dividend_amount column exists
        if 'dividend_amount' in data.columns:
            dividend_days = data[data['dividend_amount'] > 0]
            if len(dividend_days) > 0:
                corporate_actions['dividends'] = len(dividend_days)
                logger.info(f"Found {len(dividend_days)} dividend payments")
                
                # Calculate dividend yield for dividend days
                data['dividend_yield'] = 0.0
                data.loc[data['dividend_amount'] > 0, 'dividend_yield'] = (
                    data.loc[data['dividend_amount'] > 0, 'dividend_amount'] / 
                    data.loc[data['dividend_amount'] > 0, 'close']
                )
        
        # Handle splits if split_coefficient column exists
        if 'split_coefficient' in data.columns:
            split_days = data[data['split_coefficient'] != 1.0]
            if len(split_days) > 0:
                corporate_actions['splits'] = len(split_days)
                logger.info(f"Found {len(split_days)} stock splits")
                
                # Log split details
                for date, row in split_days.iterrows():
                    logger.info(f"Split on {date}: {row['split_coefficient']:.3f}")
        
        # Create a price ratio to detect potential splits (fallback method)
        data['price_ratio'] = data['close'] / data['adjusted_close']
        
        # Flag potential splits (price ratio significantly different from 1)
        potential_splits = data[abs(data['price_ratio'] - 1) > 0.1]
        if len(potential_splits) > 0 and 'split_coefficient' not in data.columns:
            logger.info(f"Detected {len(potential_splits)} potential corporate actions via price ratio")
        
        # Remove the temporary price_ratio column
        data = data.drop('price_ratio', axis=1)
        
        # Calculate total corporate actions
        corporate_actions['total_actions'] = corporate_actions['dividends'] + corporate_actions['splits']
        
        if corporate_actions['total_actions'] > 0:
            logger.info(f"Corporate actions summary: {corporate_actions}")
        
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
        
        # Remove rows with NaN values in essential columns only
        initial_rows = len(data)
        essential_columns = ['close', 'volume', 'ema_12', 'ema_26', 'ema_50', 'rsi']
        available_essential_columns = [col for col in essential_columns if col in data.columns]
        
        if available_essential_columns:
            data = data.dropna(subset=available_essential_columns)
            logger.info(f"Removed {initial_rows - len(data)} rows with NaN values in essential columns: {available_essential_columns}")
        else:
            logger.warning(f"No essential columns found for NaN filtering")
        
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