"""Abstract base class for data providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
import pandas as pd


class ProviderError(Exception):
    """Base exception for provider errors."""
    pass


class RateLimitError(ProviderError):
    """Rate limit exceeded."""
    pass


class NotFoundError(ProviderError):
    """Symbol not found."""
    pass


@dataclass
class FetchResult:
    """Result of a fetch operation."""
    symbol: str
    df: pd.DataFrame
    provider: str
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    rows_fetched: int = 0
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    has_dividends: bool = False
    has_splits: bool = False
    error: Optional[str] = None
    
    def __post_init__(self):
        if not self.df.empty:
            self.rows_fetched = len(self.df)
            if 'date' in self.df.columns:
                self.start_date = pd.to_datetime(self.df['date'].min()).date()
                self.end_date = pd.to_datetime(self.df['date'].max()).date()


@dataclass
class ProviderCapabilities:
    """Capabilities of a data provider."""
    name: str
    supports_ohlcv: bool = True
    supports_dividends: bool = False
    supports_splits: bool = False
    supports_adjusted: bool = True
    max_history_years: int = 20
    rate_limit_per_minute: Optional[int] = None
    requires_api_key: bool = False


class Provider(ABC):
    """Abstract base class for market data providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass
    
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities."""
        pass
    
    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        adjusted: bool = True
    ) -> FetchResult:
        """Fetch OHLCV data for a symbol.
        
        Args:
            symbol: Ticker symbol (e.g., 'AAPL')
            start: Start date
            end: End date
            adjusted: Use adjusted prices (default True)
            
        Returns:
            FetchResult with DataFrame containing columns:
            date, open, high, low, close, volume
        """
        pass
    
    @abstractmethod
    def fetch_actions(
        self,
        symbol: str,
        start: date,
        end: date
    ) -> pd.DataFrame:
        """Fetch corporate actions (dividends, splits).
        
        Returns:
            DataFrame with columns: date, type, value
        """
        pass
    
    @abstractmethod
    def healthcheck(self) -> bool:
        """Check if provider is available.
        
        Returns:
            True if provider is healthy
        """
        pass
    
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame to standard schema.
        
        Standard schema: date, open, high, low, close, volume
        """
        if df.empty:
            return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        
        df = df.copy()
        
        # Ensure column names are lowercase
        df.columns = df.columns.str.lower()
        
        # Rename common variants
        rename_map = {
            'adj close': 'close',
            'adj_close': 'close',
            'adjusted_close': 'close',
        }
        df = df.rename(columns=rename_map)
        
        # Select only required columns
        required = ['date', 'open', 'high', 'low', 'close', 'volume']
        available = [c for c in required if c in df.columns]
        df = df[available]
        
        # Ensure date is string format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        # Round prices to 4 decimals
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = df[col].round(4)
        
        # Volume as integer
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0).astype(int)
        
        return df.reset_index(drop=True)













