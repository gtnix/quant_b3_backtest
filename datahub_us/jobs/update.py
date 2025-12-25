"""Incremental daily update job with Neon persistence."""

import asyncio
import logging
from datetime import date, timedelta
from typing import List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

from ..providers.yfinance_provider import YFinanceProvider
from ..router import DataRouter
from ..storage import CSVStorage
from ..qa import DataValidator
from ..db import Database

logger = logging.getLogger(__name__)


@dataclass
class UpdateResult:
    """Result of update operation."""
    symbols_total: int
    symbols_updated: int
    symbols_skipped: int
    symbols_failed: int
    new_rows: int
    rows_to_db: int
    start_time: datetime
    end_time: datetime = field(default_factory=datetime.utcnow)
    errors: List[dict] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()


async def update_daily_async(
    symbols: Optional[List[str]] = None,
    force_days: int = 0,
    validate: bool = True,
    export_csv: bool = True,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> UpdateResult:
    """Update data incrementally since last known date.
    
    Flow: Check DB last date → yfinance → Neon DB → export CSV
    
    Args:
        symbols: List of symbols. If None, updates all in DB.
        force_days: Force re-fetch last N days
        validate: Run QA validation
        export_csv: Export to CSV after update
        on_progress: Progress callback
        
    Returns:
        UpdateResult with statistics
    """
    start_time = datetime.utcnow()
    
    # Initialize components
    provider = YFinanceProvider()
    router = DataRouter(provider)
    storage = CSVStorage()
    validator = DataValidator() if validate else None
    
    # Connect to database
    db = await Database.connect()
    
    # Get symbols from DB if not provided
    if symbols is None:
        symbols = await db.get_symbols()
        if not symbols:
            logger.warning("No symbols in database. Run bootstrap first.")
            await db.close()
            return UpdateResult(
                symbols_total=0,
                symbols_updated=0,
                symbols_skipped=0,
                symbols_failed=0,
                new_rows=0,
                rows_to_db=0,
                start_time=start_time
            )
    
    today = date.today()
    
    logger.info(f"Update starting: {len(symbols)} symbols")
    
    updated_count = 0
    skipped_count = 0
    failed_count = 0
    total_new_rows = 0
    rows_to_db = 0
    errors = []
    
    try:
        for i, symbol in enumerate(symbols, 1):
            try:
                # Get last date from DB
                last_date = await db.get_last_date(symbol)
                
                if last_date is None:
                    logger.info(f"[{i}/{len(symbols)}] {symbol}: No existing data, skipping")
                    skipped_count += 1
                    continue
                
                # Calculate fetch range
                fetch_start = last_date - timedelta(days=force_days) if force_days else last_date + timedelta(days=1)
                
                if fetch_start >= today:
                    logger.debug(f"{symbol}: Already up to date")
                    skipped_count += 1
                    continue
                
                # Fetch new data
                result = router.fetch_ohlcv(symbol, fetch_start, today)
                
                if result.error:
                    failed_count += 1
                    errors.append({"symbol": symbol, "error": result.error})
                    logger.warning(f"[{i}/{len(symbols)}] {symbol}: FAILED - {result.error}")
                    continue
                
                if result.df.empty:
                    skipped_count += 1
                    logger.debug(f"{symbol}: No new data")
                    continue
                
                # Validate if enabled
                if validator:
                    validation = validator.validate(symbol, result.df)
                    if not validation.is_valid:
                        logger.warning(f"{symbol}: {validation.error_count} validation errors")
                
                # Persist to Neon
                db_rows = await db.upsert_batch(symbol, result.df)
                rows_to_db += db_rows
                total_new_rows += result.rows_fetched
                
                # Export to CSV
                if export_csv:
                    await db.export_symbol_to_csv(symbol)
                
                updated_count += 1
                
                logger.info(f"[{i}/{len(symbols)}] {symbol}: +{result.rows_fetched} rows")
                
                if on_progress:
                    on_progress(i, len(symbols), symbol)
                    
            except Exception as e:
                failed_count += 1
                errors.append({"symbol": symbol, "error": str(e)})
                logger.error(f"[{i}/{len(symbols)}] {symbol}: ERROR - {e}")
        
        # Update storage metadata
        storage.update_metadata()
        
    finally:
        await db.close()
    
    end_time = datetime.utcnow()
    
    result = UpdateResult(
        symbols_total=len(symbols),
        symbols_updated=updated_count,
        symbols_skipped=skipped_count,
        symbols_failed=failed_count,
        new_rows=total_new_rows,
        rows_to_db=rows_to_db,
        start_time=start_time,
        end_time=end_time,
        errors=errors
    )
    
    logger.info(
        f"Update complete: {result.symbols_updated} updated, "
        f"{result.symbols_skipped} skipped, {result.symbols_failed} failed, "
        f"+{result.rows_to_db} rows to DB, {result.duration_seconds:.1f}s"
    )
    
    return result


def update_daily(
    symbols: Optional[List[str]] = None,
    force_days: int = 0,
    validate: bool = True,
    export_csv: bool = True,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> UpdateResult:
    """Sync wrapper for update_daily_async."""
    return asyncio.get_event_loop().run_until_complete(
        update_daily_async(
            symbols=symbols,
            force_days=force_days,
            validate=validate,
            export_csv=export_csv,
            on_progress=on_progress,
        )
    )
