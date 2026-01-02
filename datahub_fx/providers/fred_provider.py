"""FRED (Federal Reserve Economic Data) FX rate provider.

Fetches FX rates from the St. Louis Fed's FRED API.
Works without API key for low volume, but key recommended for production.

API Documentation:
    https://fred.stlouisfed.org/docs/api/fred/

Series:
    - DEXUSEU: US Dollar to Euro Spot Exchange Rate
    - DEXBZUS: Brazil / U.S. Foreign Exchange Rate (backup for USD/BRL)
"""

import logging
import os
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional

import requests

from .base import FxProvider, FxRecord

logger = logging.getLogger(__name__)


class FREDProvider(FxProvider):
    """FRED API provider for international FX rates."""
    
    # FRED series codes for each pair
    # Note: FRED quotes as "foreign currency per USD" so we may need to invert
    SERIES_MAP: Dict[str, dict] = {
        "EUR/USD": {"series": "DEXUSEU", "invert": True},   # FRED gives USD/EUR
        "USD/BRL": {"series": "DEXBZUS", "invert": False},  # FRED gives BRL/USD (direct)
    }
    
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
    
    @property
    def name(self) -> str:
        return "FRED"
    
    @property
    def supported_pairs(self) -> List[str]:
        return list(self.SERIES_MAP.keys())
    
    def fetch(
        self,
        pair: str,
        start_date: date,
        end_date: Optional[date] = None,
    ) -> List[FxRecord]:
        """Fetch rates from FRED API."""
        if pair not in self.SERIES_MAP:
            raise ValueError(f"Unsupported pair: {pair}. Supported: {self.supported_pairs}")
        
        config = self.SERIES_MAP[pair]
        series_id = config["series"]
        invert = config["invert"]
        end_date = end_date or date.today()
        
        params = {
            "series_id": series_id,
            "observation_start": start_date.isoformat(),
            "observation_end": end_date.isoformat(),
            "file_type": "json",
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        logger.info(f"Fetching {pair} from FRED: {start_date} to {end_date}")
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(self.BASE_URL, params=params, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                break
            except requests.RequestException as e:
                logger.warning(f"FRED request attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise ConnectionError(f"Failed to fetch from FRED after {self.max_retries} attempts: {e}")
        
        records = []
        observations = data.get("observations", [])
        
        for item in observations:
            try:
                value_str = item.get("value", ".")
                if value_str == "." or not value_str:
                    # FRED uses "." for missing data
                    continue
                
                dt = date.fromisoformat(item["date"])
                rate = Decimal(value_str)
                
                # Invert if necessary (e.g., USD/EUR -> EUR/USD)
                if invert and rate != 0:
                    rate = Decimal("1") / rate
                
                records.append(FxRecord(
                    date=dt,
                    rate=rate,
                    source="FRED",
                    pair=pair,
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse FRED record: {item}, error: {e}")
                continue
        
        logger.info(f"Fetched {len(records)} {pair} rates from FRED")
        return sorted(records, key=lambda r: r.date)



















