"""BCB (Banco Central do Brasil) FX rate provider.

Fetches official PTAX rates from BCB's public API.
No authentication required.

API Documentation:
    https://dadosabertos.bcb.gov.br/dataset/taxas-de-cambio-todos-os-boletins-diarios
    
Series:
    - 1: USD/BRL (PTAX venda)
    - 21619: EUR/BRL (venda)
"""

import logging
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional

import requests

from .base import FxProvider, FxRecord

logger = logging.getLogger(__name__)


class BCBProvider(FxProvider):
    """BCB API provider for Brazilian FX rates."""
    
    # BCB series codes for each pair
    SERIES_MAP: Dict[str, int] = {
        "USD/BRL": 1,       # PTAX USD venda
        "EUR/BRL": 21619,   # EUR venda
    }
    
    BASE_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs"
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
    
    @property
    def name(self) -> str:
        return "BCB"
    
    @property
    def supported_pairs(self) -> List[str]:
        return list(self.SERIES_MAP.keys())
    
    def fetch(
        self,
        pair: str,
        start_date: date,
        end_date: Optional[date] = None,
    ) -> List[FxRecord]:
        """Fetch rates from BCB API."""
        if pair not in self.SERIES_MAP:
            raise ValueError(f"Unsupported pair: {pair}. Supported: {self.supported_pairs}")
        
        series_id = self.SERIES_MAP[pair]
        end_date = end_date or date.today()
        
        # BCB API date format: DD/MM/YYYY
        start_str = start_date.strftime("%d/%m/%Y")
        end_str = end_date.strftime("%d/%m/%Y")
        
        url = f"{self.BASE_URL}.{series_id}/dados"
        params = {
            "formato": "json",
            "dataInicial": start_str,
            "dataFinal": end_str,
        }
        
        logger.info(f"Fetching {pair} from BCB: {start_date} to {end_date}")
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                break
            except requests.RequestException as e:
                logger.warning(f"BCB request attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise ConnectionError(f"Failed to fetch from BCB after {self.max_retries} attempts: {e}")
        
        records = []
        for item in data:
            try:
                # BCB returns date as "DD/MM/YYYY"
                dt = date(
                    int(item["data"][6:10]),   # year
                    int(item["data"][3:5]),    # month
                    int(item["data"][0:2]),    # day
                )
                rate = Decimal(str(item["valor"]))
                records.append(FxRecord(
                    date=dt,
                    rate=rate,
                    source="BCB",
                    pair=pair,
                ))
            except (KeyError, ValueError, IndexError) as e:
                logger.warning(f"Failed to parse BCB record: {item}, error: {e}")
                continue
        
        logger.info(f"Fetched {len(records)} {pair} rates from BCB")
        return sorted(records, key=lambda r: r.date)





