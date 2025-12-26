"""Provider router with retry logic."""

import logging
from datetime import date
from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .providers import Provider, FetchResult, ProviderError
from .providers.yfinance_provider import YFinanceProvider

logger = logging.getLogger(__name__)


@dataclass
class RouterStats:
    """Statistics for router operations."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_rows_fetched: int = 0
    errors: List[dict] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests * 100
    
    @property
    def duration_seconds(self) -> float:
        return (datetime.utcnow() - self.start_time).total_seconds()


class DataRouter:
    """Routes data requests to providers with retry logic."""
    
    def __init__(self, provider: Optional[Provider] = None):
        """Initialize router with yfinance provider.
        
        Args:
            provider: Custom provider (default: YFinanceProvider)
        """
        self.provider = provider or YFinanceProvider()
        self.stats = RouterStats()
        logger.info(f"Router initialized with provider: {self.provider.name}")
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        adjusted: bool = True
    ) -> FetchResult:
        """Fetch OHLCV data for a symbol.
        
        Args:
            symbol: Ticker symbol
            start: Start date
            end: End date
            adjusted: Use adjusted prices
            
        Returns:
            FetchResult with data or error
        """
        self.stats.total_requests += 1
        
        logger.info(f"Fetching {symbol} ({start} to {end})")
        
        try:
            result = self.provider.fetch_ohlcv(symbol, start, end, adjusted)
            
            if result.error:
                self.stats.failed_requests += 1
                self.stats.errors.append({
                    "symbol": symbol,
                    "error": result.error,
                    "timestamp": datetime.utcnow().isoformat()
                })
                logger.warning(f"Failed to fetch {symbol}: {result.error}")
            else:
                self.stats.successful_requests += 1
                self.stats.total_rows_fetched += result.rows_fetched
                logger.info(f"Successfully fetched {result.rows_fetched} rows for {symbol}")
            
            return result
            
        except Exception as e:
            self.stats.failed_requests += 1
            self.stats.errors.append({
                "symbol": symbol,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            logger.error(f"Error fetching {symbol}: {e}")
            
            return FetchResult(
                symbol=symbol,
                df=__import__('pandas').DataFrame(),
                provider=self.provider.name,
                error=str(e)
            )
    
    def fetch_batch(
        self,
        symbols: List[str],
        start: date,
        end: date,
        adjusted: bool = True,
        on_progress: Optional[callable] = None
    ) -> List[FetchResult]:
        """Fetch data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            start: Start date
            end: End date
            adjusted: Use adjusted prices
            on_progress: Callback for progress updates (current, total, symbol)
            
        Returns:
            List of FetchResults
        """
        results = []
        total = len(symbols)
        
        logger.info(f"Starting batch fetch for {total} symbols")
        
        for i, symbol in enumerate(symbols, 1):
            result = self.fetch_ohlcv(symbol, start, end, adjusted)
            results.append(result)
            
            if on_progress:
                on_progress(i, total, symbol)
            
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{total} symbols ({self.stats.success_rate:.1f}% success)")
        
        logger.info(
            f"Batch complete: {self.stats.successful_requests}/{self.stats.total_requests} "
            f"successful, {self.stats.total_rows_fetched} total rows"
        )
        
        return results
    
    def healthcheck(self) -> bool:
        """Check if provider is healthy."""
        return self.provider.healthcheck()
    
    def get_stats(self) -> RouterStats:
        """Get current statistics."""
        return self.stats
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = RouterStats()







