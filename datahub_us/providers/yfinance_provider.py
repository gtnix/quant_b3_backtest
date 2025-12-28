"""yfinance provider implementation."""

import time
import logging
from datetime import date, datetime
from typing import Optional, List
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance is required. Install with: pip install yfinance")

from .base import (
    Provider, ProviderCapabilities, FetchResult,
    ProviderError, RateLimitError, NotFoundError
)

logger = logging.getLogger(__name__)


class YFinanceProvider(Provider):
    """Yahoo Finance data provider using yfinance library."""
    
    def __init__(
        self,
        delay_between_requests: float = 0.3,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ):
        self.delay = delay_between_requests
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._last_request_time: Optional[float] = None
        self._request_count = 0
    
    @property
    def name(self) -> str:
        return "yfinance"
    
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.name,
            supports_ohlcv=True,
            supports_dividends=True,
            supports_splits=True,
            supports_adjusted=True,
            max_history_years=30,
            rate_limit_per_minute=None,  # No hard limit, but be respectful
            requires_api_key=False,
        )
    
    def _rate_limit(self):
        """Apply rate limiting between requests."""
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
        self._last_request_time = time.time()
        self._request_count += 1
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait = self.backoff_factor ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait}s")
                    time.sleep(wait)
        raise ProviderError(f"All {self.max_retries} attempts failed: {last_error}")
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        adjusted: bool = True
    ) -> FetchResult:
        """Fetch OHLCV data from Yahoo Finance."""
        logger.info(f"Fetching {symbol} from {start} to {end}")
        
        def _fetch():
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start.isoformat(),
                end=end.isoformat(),
                auto_adjust=adjusted,
            )
            return df
        
        try:
            df = self._retry_with_backoff(_fetch)
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            return FetchResult(
                symbol=symbol,
                df=pd.DataFrame(),
                provider=self.name,
                error=str(e)
            )
        
        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return FetchResult(
                symbol=symbol,
                df=pd.DataFrame(),
                provider=self.name,
                error="No data available"
            )
        
        # Convert index to date column
        df = df.reset_index()
        df = df.rename(columns={'Date': 'date'})
        
        # Normalize to standard schema
        df = self.normalize(df)
        
        logger.info(f"Fetched {len(df)} rows for {symbol}")
        
        return FetchResult(
            symbol=symbol,
            df=df,
            provider=self.name,
        )
    
    def fetch_actions(
        self,
        symbol: str,
        start: date,
        end: date
    ) -> pd.DataFrame:
        """Fetch dividends and splits."""
        logger.info(f"Fetching actions for {symbol}")
        
        def _fetch():
            ticker = yf.Ticker(symbol)
            dividends = ticker.dividends
            splits = ticker.splits
            return dividends, splits
        
        try:
            dividends, splits = self._retry_with_backoff(_fetch)
        except Exception as e:
            logger.error(f"Failed to fetch actions for {symbol}: {e}")
            return pd.DataFrame(columns=['date', 'type', 'value'])
        
        actions = []
        
        # Process dividends
        if not dividends.empty:
            for dt, value in dividends.items():
                dt_date = pd.to_datetime(dt).date()
                if start <= dt_date <= end:
                    actions.append({
                        'date': dt_date.isoformat(),
                        'type': 'dividend',
                        'value': float(value)
                    })
        
        # Process splits
        if not splits.empty:
            for dt, value in splits.items():
                dt_date = pd.to_datetime(dt).date()
                if start <= dt_date <= end:
                    actions.append({
                        'date': dt_date.isoformat(),
                        'type': 'split',
                        'value': float(value)
                    })
        
        return pd.DataFrame(actions)
    
    def healthcheck(self) -> bool:
        """Check if yfinance is working."""
        try:
            ticker = yf.Ticker("AAPL")
            info = ticker.fast_info
            return info is not None
        except Exception as e:
            logger.error(f"Healthcheck failed: {e}")
            return False
    
    def fetch_universe(self, index: str = "sp500") -> List[str]:
        """Fetch list of symbols from an index.
        
        Args:
            index: Index name ('sp500', 'nasdaq100', 'dow')
            
        Returns:
            List of ticker symbols
        """
        # Top 100 US stocks by market cap (hardcoded for reliability)
        top100 = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "TSLA",
            "UNH", "XOM", "JPM", "JNJ", "V", "PG", "MA", "HD", "CVX", "MRK",
            "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "MCD", "WMT", "CSCO",
            "TMO", "ABT", "ACN", "CRM", "NKE", "DHR", "TXN", "NEE", "PM",
            "LIN", "UNP", "VZ", "CMCSA", "HON", "RTX", "LOW", "INTC", "ORCL",
            "AMD", "QCOM", "IBM", "CAT", "BA", "GE", "AMGN", "SBUX", "GS",
            "BLK", "SPGI", "AXP", "DE", "ISRG", "MDT", "GILD", "ADI", "REGN",
            "BKNG", "VRTX", "SYK", "MDLZ", "CI", "MMC", "CB", "SCHW", "ZTS",
            "LRCX", "MO", "DUK", "SO", "PLD", "TJX", "CME", "APD", "ITW",
            "PNC", "USB", "KLAC", "TGT", "CL", "NSC", "WM", "ICE", "EQIX",
            "SHW", "AON", "MCK", "EMR", "FDX", "PCAR", "GM", "F"
        ]
        
        if index == "sp500":
            return top100
        elif index == "sample":
            return top100[:10]
        else:
            return top100
    
    @property
    def request_count(self) -> int:
        """Total requests made."""
        return self._request_count












