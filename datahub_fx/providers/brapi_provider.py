"""Brapi FX rate provider.

Fetches FX rates from Brapi API - same API used for B3 stocks.
More reliable than BCB for real-time rates.

API Documentation:
    https://brapi.dev/docs/moedas
    
Endpoint:
    GET https://brapi.dev/api/v2/currency?currency=USD-BRL,EUR-BRL
"""

import logging
import os
import time
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional

import requests

from .base import FxProvider, FxRecord

logger = logging.getLogger(__name__)


class BrapiProvider(FxProvider):
    """Brapi API provider for FX rates."""
    
    # Brapi currency pair format: MOEDA_ORIGEM-MOEDA_DESTINO
    PAIRS_MAP: Dict[str, str] = {
        "USD/BRL": "USD-BRL",
        "EUR/BRL": "EUR-BRL",
        "EUR/USD": "EUR-USD",
        "GBP/BRL": "GBP-BRL",
        "JPY/BRL": "JPY-BRL",
        "CAD/BRL": "CAD-BRL",
        "AUD/BRL": "AUD-BRL",
        "CHF/BRL": "CHF-BRL",
    }
    
    BASE_URL = "https://brapi.dev/api/v2/currency"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 5,
    ):
        self.api_key = api_key or os.environ.get("BRAPI_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
    
    @property
    def name(self) -> str:
        return "BRAPI"
    
    @property
    def supported_pairs(self) -> List[str]:
        return list(self.PAIRS_MAP.keys())
    
    def fetch(
        self,
        pair: str,
        start_date: date = None,
        end_date: Optional[date] = None,
    ) -> List[FxRecord]:
        """Fetch current rate from Brapi API.
        
        Note: Brapi only provides current rates, not historical.
        For historical data, BCB should be used.
        """
        if pair not in self.PAIRS_MAP:
            raise ValueError(f"Unsupported pair: {pair}. Supported: {self.supported_pairs}")
        
        brapi_pair = self.PAIRS_MAP[pair]
        
        params = {"currency": brapi_pair}
        headers = {}
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        logger.info(f"Fetching {pair} from Brapi")
        
        data = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    self.BASE_URL,
                    params=params,
                    headers=headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                break
            except requests.RequestException as e:
                wait_time = 2 ** attempt
                logger.warning(f"Brapi request attempt {attempt + 1} failed: {e}, retrying in {wait_time}s")
                if attempt == self.max_retries - 1:
                    raise ConnectionError(f"Failed to fetch from Brapi after {self.max_retries} attempts: {e}")
                time.sleep(wait_time)
        
        if data is None:
            return []
        
        records = []
        currencies = data.get("currency", [])
        
        for item in currencies:
            try:
                # Use bid price as the rate
                bid_price = item.get("bidPrice")
                if not bid_price:
                    continue
                
                rate = Decimal(str(bid_price))
                
                # Use today's date since Brapi only provides current rates
                dt = date.today()
                
                records.append(FxRecord(
                    date=dt,
                    rate=rate,
                    source="BRAPI",
                    pair=pair,
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse Brapi record: {item}, error: {e}")
                continue
        
        logger.info(f"Fetched {len(records)} {pair} rates from Brapi")
        return records
    
    def fetch_multiple(self, pairs: List[str]) -> List[FxRecord]:
        """Fetch multiple pairs in a single request."""
        brapi_pairs = []
        pair_map = {}
        
        for pair in pairs:
            if pair in self.PAIRS_MAP:
                brapi_pair = self.PAIRS_MAP[pair]
                brapi_pairs.append(brapi_pair)
                pair_map[brapi_pair.replace("-", "")] = pair
        
        if not brapi_pairs:
            return []
        
        params = {"currency": ",".join(brapi_pairs)}
        headers = {}
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        logger.info(f"Fetching {len(brapi_pairs)} pairs from Brapi")
        
        data = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    self.BASE_URL,
                    params=params,
                    headers=headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                break
            except requests.RequestException as e:
                wait_time = 2 ** attempt
                logger.warning(f"Brapi request attempt {attempt + 1} failed: {e}, retrying in {wait_time}s")
                if attempt == self.max_retries - 1:
                    raise ConnectionError(f"Failed to fetch from Brapi after {self.max_retries} attempts: {e}")
                time.sleep(wait_time)
        
        if data is None:
            return []
        
        records = []
        currencies = data.get("currency", [])
        
        for item in currencies:
            try:
                from_curr = item.get("fromCurrency", "")
                to_curr = item.get("toCurrency", "")
                key = f"{from_curr}{to_curr}"
                
                original_pair = pair_map.get(key)
                if not original_pair:
                    continue
                
                bid_price = item.get("bidPrice")
                if not bid_price:
                    continue
                
                rate = Decimal(str(bid_price))
                dt = date.today()
                
                records.append(FxRecord(
                    date=dt,
                    rate=rate,
                    source="BRAPI",
                    pair=original_pair,
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse Brapi record: {item}, error: {e}")
                continue
        
        logger.info(f"Fetched {len(records)} rates from Brapi")
        return records




