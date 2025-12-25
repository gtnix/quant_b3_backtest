"""Bootstrap job for initial 20-year data download with Neon persistence."""

import asyncio
import logging
from datetime import date
from typing import List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

from ..providers.yfinance_provider import YFinanceProvider
from ..router import DataRouter
from ..storage import CSVStorage
from ..qa import DataValidator
from ..db import Database
from ..universe import get_universe, get_sample_symbols
from ..config import BOOTSTRAP_START, BOOTSTRAP_END

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    """Result of bootstrap operation."""
    symbols_total: int
    symbols_success: int
    symbols_failed: int
    total_rows: int
    rows_to_db: int
    start_time: datetime
    end_time: datetime = field(default_factory=datetime.utcnow)
    errors: List[dict] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        if self.symbols_total == 0:
            return 0.0
        return self.symbols_success / self.symbols_total * 100


async def bootstrap_20y_async(
    symbols: Optional[List[str]] = None,
    universe: str = "sp500",
    start: Optional[date] = None,
    end: Optional[date] = None,
    validate: bool = True,
    export_csv: bool = True,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> BootstrapResult:
    """Bootstrap 20 years of historical data with Neon persistence.
    
    Flow: yfinance → validate → Neon DB → export CSV
    
    Args:
        symbols: List of symbols. If None, uses universe.
        universe: Universe name ('sp500', 'sample')
        start: Start date (default: 2005-01-01)
        end: End date (default: today)
        validate: Run QA validation
        export_csv: Export to CSV after DB insert
        on_progress: Progress callback
        
    Returns:
        BootstrapResult with statistics
    """
    start_time = datetime.utcnow()
    
    # Initialize components
    provider = YFinanceProvider()
    router = DataRouter(provider)
    storage = CSVStorage()
    validator = DataValidator() if validate else None
    
    # Connect to database
    db = await Database.connect()
    await db.ensure_schema()
    
    # Get symbols
    if symbols is None:
        symbols = get_universe(universe)
    
    start_date = start or BOOTSTRAP_START
    end_date = end or BOOTSTRAP_END
    
    logger.info(f"Bootstrap starting: {len(symbols)} symbols, {start_date} to {end_date}")
    
    success_count = 0
    failed_count = 0
    total_rows = 0
    rows_to_db = 0
    errors = []
    
    # Get existing symbols to skip
    existing_symbols = set(await db.get_symbols())
    symbols_to_fetch = [s for s in symbols if s not in existing_symbols]
    
    if len(symbols_to_fetch) < len(symbols):
        logger.info(f"Skipping {len(symbols) - len(symbols_to_fetch)} existing symbols, fetching {len(symbols_to_fetch)} new")
    
    try:
        for i, symbol in enumerate(symbols_to_fetch, 1):
            try:
                # Fetch data from yfinance
                result = router.fetch_ohlcv(symbol, start_date, end_date)
                
                if result.error:
                    failed_count += 1
                    errors.append({"symbol": symbol, "error": result.error})
                    logger.warning(f"[{i}/{len(symbols_to_fetch)}] {symbol}: FAILED - {result.error}")
                    continue
                
                if result.df.empty:
                    failed_count += 1
                    errors.append({"symbol": symbol, "error": "No data"})
                    logger.warning(f"[{i}/{len(symbols_to_fetch)}] {symbol}: No data")
                    continue
                
                # Validate if enabled
                if validator:
                    validation = validator.validate(symbol, result.df)
                    if not validation.is_valid:
                        logger.warning(f"{symbol}: {validation.error_count} validation errors")
                
                # Persist to Neon
                db_rows = await db.upsert_batch(symbol, result.df)
                rows_to_db += db_rows
                
                # Export to CSV
                if export_csv:
                    storage.write(symbol, result.df)
                
                total_rows += result.rows_fetched
                success_count += 1
                
                logger.info(f"[{i}/{len(symbols_to_fetch)}] {symbol}: OK ({result.rows_fetched} rows → DB)")
                
                if on_progress:
                    on_progress(i, len(symbols_to_fetch), symbol)
                    
            except Exception as e:
                failed_count += 1
                errors.append({"symbol": symbol, "error": str(e)})
                logger.error(f"[{i}/{len(symbols_to_fetch)}] {symbol}: ERROR - {e}")
        
        # Update storage metadata
        storage.update_metadata()
        
    finally:
        await db.close()
    
    end_time = datetime.utcnow()
    
    result = BootstrapResult(
        symbols_total=len(symbols),
        symbols_success=success_count,
        symbols_failed=failed_count,
        total_rows=total_rows,
        rows_to_db=rows_to_db,
        start_time=start_time,
        end_time=end_time,
        errors=errors
    )
    
    logger.info(
        f"Bootstrap complete: {result.symbols_success}/{result.symbols_total} success "
        f"({result.success_rate:.1f}%), {result.rows_to_db} rows to DB, "
        f"{result.duration_seconds:.1f}s"
    )
    
    return result


def bootstrap_20y(
    symbols: Optional[List[str]] = None,
    universe: str = "sp500",
    start: Optional[date] = None,
    end: Optional[date] = None,
    validate: bool = True,
    export_csv: bool = True,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> BootstrapResult:
    """Sync wrapper for bootstrap_20y_async."""
    return asyncio.get_event_loop().run_until_complete(
        bootstrap_20y_async(
            symbols=symbols,
            universe=universe,
            start=start,
            end=end,
            validate=validate,
            export_csv=export_csv,
            on_progress=on_progress,
        )
    )
