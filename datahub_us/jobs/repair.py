"""Gap repair job for filling missing data."""

import logging
from datetime import date, timedelta
from typing import List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

from ..providers.yfinance_provider import YFinanceProvider
from ..router import DataRouter
from ..storage import CSVStorage
from ..qa import DataValidator

logger = logging.getLogger(__name__)


@dataclass
class GapInfo:
    """Information about a detected gap."""
    symbol: str
    start: str
    end: str
    days: int


@dataclass
class RepairResult:
    """Result of repair operation."""
    symbols_scanned: int
    gaps_detected: int
    gaps_repaired: int
    gaps_failed: int
    rows_added: int
    start_time: datetime
    end_time: datetime = field(default_factory=datetime.utcnow)
    gaps: List[GapInfo] = field(default_factory=list)
    errors: List[dict] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()


def repair_gaps(
    symbols: Optional[List[str]] = None,
    max_gap_days: int = 5,
    dry_run: bool = False,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> RepairResult:
    """Detect and repair gaps in stored data.
    
    This job scans all stored data for gaps larger than max_gap_days
    and attempts to fill them by fetching missing data.
    
    Args:
        symbols: List of symbols to scan. If None, scans all stored symbols.
        max_gap_days: Maximum allowed gap before it's flagged (default: 5)
        dry_run: If True, only detect gaps without repairing
        on_progress: Callback for progress updates
        
    Returns:
        RepairResult with statistics
    """
    start_time = datetime.utcnow()
    
    # Initialize components
    provider = YFinanceProvider()
    router = DataRouter(provider)
    storage = CSVStorage()
    
    # Get symbols
    if symbols is None:
        symbols = storage.list_symbols()
    
    logger.info(f"Gap repair starting: {len(symbols)} symbols, max_gap={max_gap_days} days")
    if dry_run:
        logger.info("DRY RUN - no data will be modified")
    
    all_gaps = []
    repaired_count = 0
    failed_count = 0
    total_rows_added = 0
    errors = []
    
    for i, symbol in enumerate(symbols, 1):
        try:
            # Detect gaps
            gaps = storage.detect_gaps(symbol, max_gap_days)
            
            if not gaps:
                logger.debug(f"{symbol}: No gaps detected")
                continue
            
            logger.info(f"[{i}/{len(symbols)}] {symbol}: {len(gaps)} gaps detected")
            
            for gap in gaps:
                gap_info = GapInfo(
                    symbol=symbol,
                    start=gap['start'],
                    end=gap['end'],
                    days=gap['days']
                )
                all_gaps.append(gap_info)
                
                if dry_run:
                    logger.info(f"  Gap: {gap['start']} to {gap['end']} ({gap['days']} days)")
                    continue
                
                # Try to repair
                try:
                    gap_start = date.fromisoformat(gap['start']) + timedelta(days=1)
                    gap_end = date.fromisoformat(gap['end']) - timedelta(days=1)
                    
                    if gap_start >= gap_end:
                        continue
                    
                    result = router.fetch_ohlcv(symbol, gap_start, gap_end)
                    
                    if result.error or result.df.empty:
                        logger.warning(f"  Gap {gap['start']}-{gap['end']}: No data available")
                        failed_count += 1
                        continue
                    
                    new_rows = storage.write(symbol, result.df, overwrite=False)
                    total_rows_added += new_rows
                    repaired_count += 1
                    
                    logger.info(f"  Gap {gap['start']}-{gap['end']}: REPAIRED (+{new_rows} rows)")
                    
                except Exception as e:
                    failed_count += 1
                    logger.error(f"  Gap {gap['start']}-{gap['end']}: ERROR - {e}")
            
            if on_progress:
                on_progress(i, len(symbols), symbol)
                
        except Exception as e:
            errors.append({"symbol": symbol, "error": str(e)})
            logger.error(f"[{i}/{len(symbols)}] {symbol}: ERROR - {e}")
    
    # Update metadata if not dry run
    if not dry_run:
        storage.update_metadata()
    
    end_time = datetime.utcnow()
    
    result = RepairResult(
        symbols_scanned=len(symbols),
        gaps_detected=len(all_gaps),
        gaps_repaired=repaired_count,
        gaps_failed=failed_count,
        rows_added=total_rows_added,
        start_time=start_time,
        end_time=end_time,
        gaps=all_gaps,
        errors=errors
    )
    
    logger.info(
        f"Gap repair complete: {result.gaps_detected} gaps detected, "
        f"{result.gaps_repaired} repaired, {result.gaps_failed} failed, "
        f"+{result.rows_added} rows, {result.duration_seconds:.1f}s"
    )
    
    return result





































