"""S&P 500 universe management."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import pandas as pd

from .config import CACHE_DIR

logger = logging.getLogger(__name__)

SP500_CACHE_FILE = CACHE_DIR / "sp500_symbols.json"
CACHE_TTL_DAYS = 7  # Refresh weekly


# Hardcoded S&P 500 symbols (updated Dec 2024) - fallback if Wikipedia fails
SP500_SYMBOLS = [
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM",
    "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM",
    "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP",
    "AMT", "AMZN", "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH", "APTV",
    "ARE", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP", "AZO", "BA",
    "BAC", "BALL", "BAX", "BBWI", "BBY", "BDX", "BEN", "BF-B", "BG", "BIIB",
    "BIO", "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK-B", "BRO",
    "BSX", "BWA", "BX", "BXP", "C", "CAG", "CAH", "CARR", "CAT", "CB",
    "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CEG", "CF", "CFG",
    "CHD", "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMCSA", "CME", "CMG",
    "CMI", "CMS", "CNC", "CNP", "COF", "COO", "COP", "COR", "COST", "CPAY",
    "CPB", "CPRT", "CPT", "CRL", "CRM", "CSCO", "CSGP", "CSX", "CTAS", "CTLT",
    "CTRA", "CTSH", "CTVA", "CVS", "CVX", "CZR", "D", "DAL", "DAY", "DD",
    "DE", "DECK", "DFS", "DG", "DGX", "DHI", "DHR", "DIS", "DLR", "DLTR",
    "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM",
    "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL", "ELV", "EMN",
    "EMR", "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ES", "ESS", "ETN",
    "ETR", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR", "F", "FANG", "FAST",
    "FCX", "FDS", "FDX", "FE", "FFIV", "FI", "FICO", "FIS", "FITB", "FMC",
    "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD", "GDDY", "GE", "GEHC",
    "GEN", "GEV", "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOG", "GOOGL",
    "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS", "HBAN", "HCA", "HD",
    "HES", "HIG", "HII", "HLT", "HOLX", "HON", "HPE", "HPQ", "HRL", "HSIC",
    "HST", "HSY", "HUBB", "HUM", "HWM", "IBM", "ICE", "IDXX", "IEX", "IFF",
    "ILMN", "INCY", "INTC", "INTU", "INVH", "IP", "IPG", "IQV", "IR", "IRM",
    "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JBL", "JCI", "JKHY", "JNJ",
    "JNPR", "JPM", "K", "KDP", "KEY", "KEYS", "KHC", "KIM", "KKR", "KLAC",
    "KMB", "KMI", "KMX", "KO", "KR", "KVUE", "L", "LDOS", "LEN", "LH",
    "LHX", "LIN", "LKQ", "LLY", "LMT", "LNT", "LOW", "LRCX", "LULU", "LUV",
    "LVS", "LW", "LYB", "LYV", "MA", "MAA", "MAR", "MAS", "MCD", "MCHP",
    "MCK", "MCO", "MDLZ", "MDT", "MET", "META", "MGM", "MHK", "MKC", "MKTX",
    "MLM", "MMC", "MMM", "MNST", "MO", "MOH", "MOS", "MPC", "MPWR", "MRK",
    "MRNA", "MRO", "MS", "MSCI", "MSFT", "MSI", "MTB", "MTCH", "MTD", "MU",
    "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE", "NOC", "NOW",
    "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWS", "NWSA", "NXPI",
    "O", "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY", "PANW",
    "PARA", "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PEP", "PFE", "PFG", "PG",
    "PGR", "PH", "PHM", "PKG", "PLD", "PM", "PNC", "PNR", "PNW", "PODD",
    "POOL", "PPG", "PPL", "PRU", "PSA", "PSX", "PTC", "PWR", "PYPL", "QCOM",
    "QRVO", "RCL", "REG", "REGN", "RF", "RJF", "RL", "RMD", "ROK", "ROL",
    "ROP", "ROST", "RSG", "RTX", "RVTY", "SBAC", "SBUX", "SCHW", "SHW", "SJM",
    "SLB", "SMCI", "SNA", "SNPS", "SO", "SOLV", "SPG", "SPGI", "SRE", "STE",
    "STLD", "STT", "STX", "STZ", "SWK", "SWKS", "SYF", "SYK", "SYY", "T",
    "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TFX", "TGT", "TJX",
    "TMO", "TMUS", "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN",
    "TT", "TTWO", "TXN", "TXT", "TYL", "UAL", "UBER", "UDR", "UHS", "ULTA",
    "UNH", "UNP", "UPS", "URI", "USB", "V", "VICI", "VLO", "VLTO", "VMC",
    "VRSK", "VRSN", "VRTX", "VST", "VTR", "VTRS", "VZ", "WAB", "WAT", "WBA",
    "WBD", "WDC", "WEC", "WELL", "WFC", "WM", "WMB", "WMT", "WRB", "WRK",
    "WST", "WTW", "WY", "WYNN", "XEL", "XOM", "XYL", "YUM", "ZBH", "ZBRA", "ZTS"
]


def fetch_sp500_from_wikipedia() -> List[str]:
    """Fetch S&P 500 symbols from Wikipedia with proper headers.
    
    Returns:
        List of ticker symbols
    """
    import httpx
    
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = httpx.get(url, headers=headers, timeout=30.0)
        response.raise_for_status()
        
        tables = pd.read_html(response.text)
        df = tables[0]  # First table is the S&P 500 list
        
        # Column is 'Symbol'
        symbols = df['Symbol'].tolist()
        
        # Clean up symbols (some have . instead of -)
        symbols = [s.replace('.', '-') for s in symbols]
        
        logger.info(f"Fetched {len(symbols)} S&P 500 symbols from Wikipedia")
        return sorted(symbols)
        
    except Exception as e:
        logger.warning(f"Failed to fetch from Wikipedia: {e}. Using hardcoded list.")
        return SP500_SYMBOLS


def get_sp500_symbols(force_refresh: bool = False) -> List[str]:
    """Get S&P 500 symbols with caching.
    
    Args:
        force_refresh: Force refresh from Wikipedia
        
    Returns:
        List of ticker symbols
    """
    # Check cache
    if not force_refresh and SP500_CACHE_FILE.exists():
        try:
            with open(SP500_CACHE_FILE) as f:
                data = json.load(f)
            
            cached_at = datetime.fromisoformat(data['cached_at'])
            if datetime.utcnow() - cached_at < timedelta(days=CACHE_TTL_DAYS):
                logger.debug(f"Using cached S&P 500 list ({len(data['symbols'])} symbols)")
                return data['symbols']
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
    
    # Try Wikipedia first, fallback to hardcoded
    try:
        symbols = fetch_sp500_from_wikipedia()
    except Exception:
        logger.info("Using hardcoded S&P 500 list")
        symbols = SP500_SYMBOLS
    
    # Save cache
    try:
        SP500_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SP500_CACHE_FILE, 'w') as f:
            json.dump({
                'symbols': symbols,
                'count': len(symbols),
                'cached_at': datetime.utcnow().isoformat(),
                'source': 'wikipedia' if len(symbols) > len(SP500_SYMBOLS) else 'hardcoded',
            }, f, indent=2)
        logger.debug(f"Cached {len(symbols)} symbols to {SP500_CACHE_FILE}")
    except Exception as e:
        logger.warning(f"Cache write error: {e}")
    
    return symbols


def get_sample_symbols(n: int = 10) -> List[str]:
    """Get a sample of popular US stocks for testing.
    
    Args:
        n: Number of symbols
        
    Returns:
        List of ticker symbols
    """
    # Top US stocks by market cap
    top_stocks = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", 
        "META", "BRK-B", "TSLA", "UNH", "XOM",
        "JPM", "JNJ", "V", "PG", "MA",
        "HD", "CVX", "MRK", "ABBV", "LLY",
    ]
    return top_stocks[:n]


def get_universe(universe: str = "sp500") -> List[str]:
    """Get universe of symbols.
    
    Args:
        universe: Universe name ('sp500', 'sample', 'sample20')
        
    Returns:
        List of ticker symbols
    """
    if universe == "sp500":
        return get_sp500_symbols()
    elif universe == "sample":
        return get_sample_symbols(10)
    elif universe == "sample20":
        return get_sample_symbols(20)
    else:
        raise ValueError(f"Unknown universe: {universe}")
