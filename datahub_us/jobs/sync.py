"""Sync job for downloading data from Neon to local cache."""

import asyncio
import logging
from typing import List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

from ..db import Database
from ..storage import CSVStorage
from ..config import OHLCV_DIR

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Result of sync operation."""
    symbols_total: int
    symbols_synced: int
    total_rows: int
    start_time: datetime
    end_time: datetime = field(default_factory=datetime.utcnow)
    output_dir: str = ""
    
    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()


async def sync_cache_async(
    symbols: Optional[List[str]] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> SyncResult:
    """Sync data from Neon database to local CSV cache.
    
    This is used before running backtests to ensure local cache
    is up to date with the database.
    
    Args:
        symbols: List of symbols. If None, syncs all in DB.
        on_progress: Progress callback
        
    Returns:
        SyncResult with statistics
    """
    start_time = datetime.utcnow()
    
    # Connect to database
    db = await Database.connect()
    
    try:
        # Get symbols from DB if not provided
        if symbols is None:
            symbols = await db.get_symbols()
        
        if not symbols:
            logger.warning("No symbols in database to sync")
            await db.close()
            return SyncResult(
                symbols_total=0,
                symbols_synced=0,
                total_rows=0,
                start_time=start_time,
                output_dir=str(OHLCV_DIR),
            )
        
        logger.info(f"Sync starting: {len(symbols)} symbols → {OHLCV_DIR}")
        
        # Export all to CSV
        result = await db.export_all_to_csv(
            symbols=symbols,
            on_progress=on_progress,
        )
        
        # Update storage metadata
        storage = CSVStorage()
        storage.update_metadata()
        
    finally:
        await db.close()
    
    end_time = datetime.utcnow()
    
    sync_result = SyncResult(
        symbols_total=len(symbols),
        symbols_synced=result['symbols_exported'],
        total_rows=result['total_rows'],
        start_time=start_time,
        end_time=end_time,
        output_dir=result['output_dir'],
    )
    
    logger.info(
        f"Sync complete: {sync_result.symbols_synced}/{sync_result.symbols_total} symbols, "
        f"{sync_result.total_rows} rows, {sync_result.duration_seconds:.1f}s"
    )
    
    return sync_result


def sync_cache(
    symbols: Optional[List[str]] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> SyncResult:
    """Sync wrapper for sync_cache_async."""
    return asyncio.get_event_loop().run_until_complete(
        sync_cache_async(symbols=symbols, on_progress=on_progress)
    )















