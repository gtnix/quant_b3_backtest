"""US Indices Scraper - Coleta composição de índices americanos."""
import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional
import httpx
import pandas as pd

logger = logging.getLogger(__name__)

# Hardcoded NASDAQ-100 (Dec 2024)
NASDAQ100_SYMBOLS = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN",
    "AMZN", "ANSS", "APP", "ARM", "ASML", "AVGO", "AXON", "AZN", "BIIB", "BKNG",
    "BKR", "CCEP", "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD",
    "CSCO", "CSGP", "CSX", "CTAS", "CTSH", "DASH", "DDOG", "DLTR", "DXCM", "EA",
    "EXC", "FANG", "FAST", "FTNT", "GEHC", "GFS", "GILD", "GOOG", "GOOGL", "HON",
    "IDXX", "ILMN", "INTC", "INTU", "ISRG", "KDP", "KHC", "KLAC", "LIN", "LRCX",
    "LULU", "MAR", "MCHP", "MDB", "MDLZ", "MELI", "META", "MNST", "MRNA", "MRVL",
    "MSFT", "MU", "NFLX", "NVDA", "NXPI", "ODFL", "ON", "ORLY", "PANW", "PAYX",
    "PCAR", "PDD", "PEP", "PYPL", "QCOM", "REGN", "ROP", "ROST", "SBUX", "SMCI",
    "SNPS", "TEAM", "TMUS", "TSLA", "TTD", "TTWO", "TXN", "VRSK", "VRTX", "WBD",
    "WDAY", "XEL", "ZS",
]

# Hardcoded Dow Jones 30 (Dec 2024)
DOW30_SYMBOLS = [
    "AMGN", "AMZN", "AAPL", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
]

# Russell 2000 - Too many, will scrape from source
# S&P 500 - Already in universe.py


@dataclass
class USIndexComponent:
    """Componente de um índice americano."""
    symbol: str
    company_name: str
    sector: Optional[str] = None
    weight: Optional[float] = None


@dataclass
class USIndexData:
    """Dados completos de um índice americano."""
    index_code: str
    index_name: str
    date: date
    total_components: int
    components: list[USIndexComponent]


def _fetch_table_from_wikipedia(url: str, table_index: int = 0) -> pd.DataFrame:
    """Fetch table from Wikipedia page."""
    from io import StringIO
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    }
    response = httpx.get(url, headers=headers, timeout=30.0)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    return tables[table_index]


def fetch_sp500() -> USIndexData:
    """Fetch S&P 500 components from Wikipedia."""
    logger.info("Fetching S&P 500 from Wikipedia...")
    try:
        df = _fetch_table_from_wikipedia(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
        components = []
        for _, row in df.iterrows():
            sym = str(row['Symbol']).replace('.', '-')
            components.append(USIndexComponent(
                symbol=sym,
                company_name=str(row.get('Security', '')),
                sector=str(row.get('GICS Sector', ''))
            ))
        logger.info(f"✓ S&P 500: {len(components)} components")
        return USIndexData(
            index_code="SPX",
            index_name="S&P 500",
            date=date.today(),
            total_components=len(components),
            components=components
        )
    except Exception as e:
        logger.error(f"Failed to fetch S&P 500: {e}")
        raise


def fetch_nasdaq100() -> USIndexData:
    """Fetch NASDAQ-100 components from Wikipedia."""
    logger.info("Fetching NASDAQ-100 from Wikipedia...")
    try:
        df = _fetch_table_from_wikipedia(
            "https://en.wikipedia.org/wiki/Nasdaq-100", table_index=4
        )
        components = []
        for _, row in df.iterrows():
            sym = str(row.get('Ticker', row.get('Symbol', ''))).strip()
            if sym and len(sym) <= 5:
                components.append(USIndexComponent(
                    symbol=sym.replace('.', '-'),
                    company_name=str(row.get('Company', '')),
                    sector=str(row.get('GICS Sector', ''))
                ))
        
        if len(components) < 90:
            # Fallback to hardcoded
            logger.warning("Using hardcoded NASDAQ-100 list")
            components = [USIndexComponent(s, s) for s in NASDAQ100_SYMBOLS]
        
        logger.info(f"✓ NASDAQ-100: {len(components)} components")
        return USIndexData(
            index_code="NDX",
            index_name="NASDAQ-100",
            date=date.today(),
            total_components=len(components),
            components=components
        )
    except Exception as e:
        logger.warning(f"Wikipedia failed, using hardcoded: {e}")
        components = [USIndexComponent(s, s) for s in NASDAQ100_SYMBOLS]
        return USIndexData(
            index_code="NDX",
            index_name="NASDAQ-100",
            date=date.today(),
            total_components=len(components),
            components=components
        )


def fetch_dow30() -> USIndexData:
    """Fetch Dow Jones 30 components."""
    logger.info("Fetching Dow Jones 30...")
    try:
        df = _fetch_table_from_wikipedia(
            "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", table_index=1
        )
        components = []
        for _, row in df.iterrows():
            sym = str(row.get('Symbol', row.get('Ticker', ''))).strip()
            if sym and len(sym) <= 5 and sym.isalpha():
                components.append(USIndexComponent(
                    symbol=sym,
                    company_name=str(row.get('Company', '')),
                ))
        
        if len(components) < 25:
            logger.warning("Using hardcoded Dow 30 list")
            components = [USIndexComponent(s, s) for s in DOW30_SYMBOLS]
        
        logger.info(f"✓ Dow 30: {len(components)} components")
        return USIndexData(
            index_code="DJI",
            index_name="Dow Jones Industrial Average",
            date=date.today(),
            total_components=len(components),
            components=components
        )
    except Exception as e:
        logger.warning(f"Wikipedia failed, using hardcoded: {e}")
        components = [USIndexComponent(s, s) for s in DOW30_SYMBOLS]
        return USIndexData(
            index_code="DJI",
            index_name="Dow Jones Industrial Average",
            date=date.today(),
            total_components=len(components),
            components=components
        )


# Index fetchers registry
US_INDEX_FETCHERS = {
    "SPX": ("S&P 500", fetch_sp500),
    "NDX": ("NASDAQ-100", fetch_nasdaq100),
    "DJI": ("Dow Jones 30", fetch_dow30),
}


def fetch_index(index_code: str) -> Optional[USIndexData]:
    """Fetch a specific US index."""
    if index_code not in US_INDEX_FETCHERS:
        logger.error(f"Unknown index: {index_code}")
        return None
    
    _, fetcher = US_INDEX_FETCHERS[index_code]
    return fetcher()


def fetch_all_indices() -> dict[str, USIndexData]:
    """Fetch all US indices."""
    results = {}
    for code in US_INDEX_FETCHERS:
        try:
            data = fetch_index(code)
            if data:
                results[code] = data
        except Exception as e:
            logger.error(f"Failed to fetch {code}: {e}")
    return results

