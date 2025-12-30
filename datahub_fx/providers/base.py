"""Base class for FX rate providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import List, Optional


@dataclass
class FxRecord:
    """A single FX rate observation."""
    
    date: date
    rate: Decimal
    source: str
    pair: str  # e.g., "USD/BRL"
    
    def to_csv_row(self) -> str:
        """Format as CSV row."""
        return f"{self.date},{self.rate},{self.source}"
    
    @classmethod
    def from_csv_row(cls, row: str, pair: str) -> Optional["FxRecord"]:
        """Parse from CSV row."""
        parts = row.strip().split(",")
        if len(parts) < 2:
            return None
        
        try:
            dt = date.fromisoformat(parts[0])
            rate = Decimal(parts[1])
            source = parts[2] if len(parts) > 2 else "unknown"
            return cls(date=dt, rate=rate, source=source, pair=pair)
        except (ValueError, IndexError):
            return None


class FxProvider(ABC):
    """Abstract base class for FX data providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'BCB', 'FRED')."""
        pass
    
    @property
    @abstractmethod
    def supported_pairs(self) -> List[str]:
        """List of supported currency pairs."""
        pass
    
    @abstractmethod
    def fetch(
        self,
        pair: str,
        start_date: date,
        end_date: Optional[date] = None,
    ) -> List[FxRecord]:
        """
        Fetch FX rates for a currency pair.
        
        Args:
            pair: Currency pair (e.g., "USD/BRL")
            start_date: Start of date range
            end_date: End of date range (default: today)
        
        Returns:
            List of FxRecord ordered by date
        
        Raises:
            ValueError: If pair not supported
            ConnectionError: If API request fails
        """
        pass
    
    def fetch_latest(self, pair: str) -> Optional[FxRecord]:
        """Fetch the most recent rate for a pair."""
        today = date.today()
        # Look back 10 days to handle weekends/holidays
        from datetime import timedelta
        start = today - timedelta(days=10)
        
        records = self.fetch(pair, start, today)
        return records[-1] if records else None









