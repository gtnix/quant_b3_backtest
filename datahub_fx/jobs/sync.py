"""Full synchronization job - fetches all historical data."""

import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from ..config import FxConfig, DEFAULT_CONFIG
from ..providers import BCBProvider, FREDProvider, FxProvider
from ..storage import CsvFxStorage

logger = logging.getLogger(__name__)


# Default pairs to sync
DEFAULT_PAIRS = ["USD/BRL", "EUR/BRL", "EUR/USD"]


def get_provider_for_pair(pair: str) -> Optional[FxProvider]:
    """Get the appropriate provider for a currency pair."""
    bcb = BCBProvider()
    fred = FREDProvider()
    
    if pair in bcb.supported_pairs:
        return bcb
    elif pair in fred.supported_pairs:
        return fred
    else:
        return None


def sync_pair(
    pair: str,
    config: FxConfig = DEFAULT_CONFIG,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> Dict:
    """
    Sync a single currency pair from inception.
    
    Returns:
        Dict with sync results
    """
    start_date = start_date or config.start_date
    end_date = end_date or date.today()
    
    provider = get_provider_for_pair(pair)
    if not provider:
        logger.error(f"No provider found for pair: {pair}")
        return {"pair": pair, "status": "error", "error": "No provider available"}
    
    storage = CsvFxStorage(config.cache_dir)
    
    try:
        logger.info(f"Syncing {pair} from {provider.name}: {start_date} to {end_date}")
        records = provider.fetch(pair, start_date, end_date)
        
        if records:
            count = storage.save(pair, records)
            logger.info(f"Synced {count} records for {pair}")
            return {
                "pair": pair,
                "status": "success",
                "records": count,
                "first_date": records[0].date.isoformat(),
                "last_date": records[-1].date.isoformat(),
                "source": provider.name,
            }
        else:
            return {"pair": pair, "status": "empty", "records": 0}
            
    except Exception as e:
        logger.exception(f"Failed to sync {pair}")
        return {"pair": pair, "status": "error", "error": str(e)}


def sync_all(
    pairs: Optional[List[str]] = None,
    config: FxConfig = DEFAULT_CONFIG,
) -> Dict[str, Dict]:
    """
    Sync all currency pairs.
    
    Args:
        pairs: List of pairs to sync (default: DEFAULT_PAIRS)
        config: Configuration to use
    
    Returns:
        Dict mapping pair to sync results
    """
    pairs = pairs or DEFAULT_PAIRS
    results = {}
    
    for pair in pairs:
        results[pair] = sync_pair(pair, config)
    
    # Summary
    success = sum(1 for r in results.values() if r.get("status") == "success")
    total = len(pairs)
    logger.info(f"Sync complete: {success}/{total} pairs successful")
    
    return results









