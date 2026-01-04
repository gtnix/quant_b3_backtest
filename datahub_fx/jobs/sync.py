"""Full synchronization job - fetches all historical data."""

import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from ..config import FxConfig, DEFAULT_CONFIG
from ..providers import BCBProvider, FREDProvider, BrapiProvider, FxProvider
from ..storage import CsvFxStorage
from ..db import get_connection, ensure_table_exists, upsert_rates

logger = logging.getLogger(__name__)


# Default pairs to sync
DEFAULT_PAIRS = ["USD/BRL", "EUR/BRL", "EUR/USD"]


def get_provider_for_pair(pair: str, prefer_brapi: bool = True) -> Optional[FxProvider]:
    """Get the appropriate provider for a currency pair.
    
    Priority: Brapi (current rates) > BCB (historical BRL) > FRED (historical USD)
    """
    brapi = BrapiProvider()
    bcb = BCBProvider()
    fred = FREDProvider()
    
    # Brapi is preferred for current rates (more reliable)
    if prefer_brapi and pair in brapi.supported_pairs:
        return brapi
    elif pair in bcb.supported_pairs:
        return bcb
    elif pair in fred.supported_pairs:
        return fred
    elif pair in brapi.supported_pairs:
        return brapi
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
            # Save to CSV cache
            count = storage.save(pair, records)
            logger.info(f"Synced {count} records for {pair} to CSV")
            
            # Persist to Neon DB
            db_records = [(r.pair, r.date, r.rate, r.source) for r in records]
            with get_connection() as conn:
                ensure_table_exists(conn)
                db_count = upsert_rates(conn, db_records)
                logger.info(f"Synced {db_count} records for {pair} to Neon DB")
            
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

























