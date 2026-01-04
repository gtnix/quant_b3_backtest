"""Incremental update job - fetches only new data."""

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional

from ..config import FxConfig, DEFAULT_CONFIG
from ..storage import CsvFxStorage
from ..db import get_connection, ensure_table_exists, upsert_rates, get_latest_date as db_get_latest_date
from .sync import get_provider_for_pair, DEFAULT_PAIRS

logger = logging.getLogger(__name__)


def update_pair(
    pair: str,
    config: FxConfig = DEFAULT_CONFIG,
) -> Dict:
    """
    Update a single currency pair with new data.
    
    Fetches data from the last known date to today.
    
    Returns:
        Dict with update results
    """
    storage = CsvFxStorage(config.cache_dir)
    provider = get_provider_for_pair(pair)
    
    if not provider:
        logger.error(f"No provider found for pair: {pair}")
        return {"pair": pair, "status": "error", "error": "No provider available"}
    
    # Get last known date from Neon DB first, fallback to CSV
    try:
        with get_connection() as conn:
            ensure_table_exists(conn)
            last_date = db_get_latest_date(conn, pair)
    except Exception as e:
        logger.warning(f"Could not get date from DB: {e}, using CSV")
        last_date = storage.get_latest_date(pair)
    
    if last_date is None:
        # No existing data, do full sync
        from .sync import sync_pair
        logger.info(f"No existing data for {pair}, performing full sync")
        return sync_pair(pair, config)
    
    # Fetch from day after last known date
    start_date = last_date + timedelta(days=1)
    end_date = date.today()
    
    if start_date > end_date:
        logger.info(f"{pair} is already up to date")
        return {"pair": pair, "status": "up_to_date", "last_date": last_date.isoformat()}
    
    try:
        logger.info(f"Updating {pair} from {provider.name}: {start_date} to {end_date}")
        records = provider.fetch(pair, start_date, end_date)
        
        if records:
            # Save to CSV cache
            added = storage.append(pair, records)
            new_last = storage.get_latest_date(pair)
            logger.info(f"Added {added} new records for {pair} to CSV")
            
            # Persist to Neon DB
            db_records = [(r.pair, r.date, r.rate, r.source) for r in records]
            with get_connection() as conn:
                db_count = upsert_rates(conn, db_records)
                logger.info(f"Added {db_count} records for {pair} to Neon DB")
            
            return {
                "pair": pair,
                "status": "updated",
                "added": added,
                "last_date": new_last.isoformat() if new_last else None,
                "source": provider.name,
            }
        else:
            return {
                "pair": pair,
                "status": "no_new_data",
                "last_date": last_date.isoformat(),
            }
            
    except Exception as e:
        logger.exception(f"Failed to update {pair}")
        return {"pair": pair, "status": "error", "error": str(e)}


def update_all(
    pairs: Optional[List[str]] = None,
    config: FxConfig = DEFAULT_CONFIG,
) -> Dict[str, Dict]:
    """
    Update all currency pairs with new data.
    
    Args:
        pairs: List of pairs to update (default: all stored + DEFAULT_PAIRS)
        config: Configuration to use
    
    Returns:
        Dict mapping pair to update results
    """
    storage = CsvFxStorage(config.cache_dir)
    
    # Use stored pairs + defaults
    if pairs is None:
        stored = set(storage.list_pairs())
        pairs = list(stored.union(DEFAULT_PAIRS))
    
    results = {}
    
    for pair in pairs:
        results[pair] = update_pair(pair, config)
    
    # Summary
    updated = sum(1 for r in results.values() if r.get("status") in ("updated", "up_to_date"))
    added = sum(r.get("added", 0) for r in results.values())
    logger.info(f"Update complete: {updated}/{len(pairs)} pairs ok, {added} new records")
    
    return results

























