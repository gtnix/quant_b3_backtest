"""S&P 500 universe management."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import pandas as pd

from .config import CACHE_DIR
from .symbols import SP500_SYMBOLS, TOP_20_US

logger = logging.getLogger(__name__)

SP500_CACHE_FILE = CACHE_DIR / "sp500_symbols.json"
CACHE_TTL_DAYS = 7  # Refresh weekly


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
    return TOP_20_US[:n]


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
