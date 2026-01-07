"""Intraday data sync via Brapi PRO API."""
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional
import httpx
from psycopg2.extras import execute_values

from .db import get_connection

logger = logging.getLogger(__name__)

BRAPI_BASE_URL = "https://brapi.dev"
BRAPI_TOKEN = os.getenv("BRAPI_API_KEY", "")
BATCH_SIZE = 20  # Brapi max tickers per request


@dataclass
class SyncResult:
    symbols_total: int
    symbols_success: int
    symbols_failed: int
    bars_inserted: int
    duration_secs: float
    errors: list


def get_symbols_from_indices(conn) -> list[str]:
    """Get unique symbols from all B3 indices."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT symbol FROM b3_index_composition ORDER BY symbol
        """)
        return [row[0] for row in cur.fetchall()]


def _parse_brapi_response(data: dict) -> list[tuple]:
    """Parse Brapi response into (symbol, timestamp, open, high, low, close, volume) tuples."""
    bars = []
    for result in data.get("results", []):
        symbol = result.get("symbol", "")
        for bar in result.get("historicalDataPrice", []):
            ts = bar.get("date")
            if ts:
                timestamp = datetime.fromtimestamp(ts)
                bars.append((
                    symbol,
                    timestamp,
                    bar.get("open"),
                    bar.get("high"),
                    bar.get("low"),
                    bar.get("close"),
                    bar.get("volume", 0)
                ))
    return bars


def fetch_batch(
    symbols: list[str],
    interval: str = "30m",
    range_param: str = "5d",
    timeout: float = 60.0
) -> tuple[list[tuple], list[str]]:
    """Fetch intraday data for a batch of symbols.
    
    Returns:
        (bars, failed_symbols)
    """
    if not BRAPI_TOKEN:
        raise ValueError("BRAPI_API_KEY not set")
    
    tickers = ",".join(symbols)
    url = f"{BRAPI_BASE_URL}/api/quote/{tickers}"
    params = {
        "range": range_param,
        "interval": interval,
        "token": BRAPI_TOKEN
    }
    
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.error(f"Batch failed: {e}")
        return [], symbols
    
    bars = _parse_brapi_response(data)
    
    # Check which symbols returned data
    returned_symbols = {r.get("symbol") for r in data.get("results", [])}
    failed = [s for s in symbols if s not in returned_symbols]
    
    return bars, failed


def ensure_intraday_table(conn):
    """Create intraday table if not exists."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_intraday_br (
                symbol VARCHAR(20) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                interval VARCHAR(10) NOT NULL,
                open NUMERIC(12,4),
                high NUMERIC(12,4),
                low NUMERIC(12,4),
                close NUMERIC(12,4),
                volume BIGINT,
                ingested_at TIMESTAMPTZ DEFAULT now(),
                PRIMARY KEY (symbol, timestamp, interval)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_intraday_br_sym ON ohlcv_intraday_br(symbol)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_intraday_br_ts ON ohlcv_intraday_br(timestamp DESC)")
        conn.commit()


def upsert_intraday_bars(conn, bars: list[tuple], interval: str) -> int:
    """Insert/update intraday bars."""
    if not bars:
        return 0
    
    values = [(b[0], b[1], interval, b[2], b[3], b[4], b[5], b[6]) for b in bars]
    
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO ohlcv_intraday_br 
            (symbol, timestamp, interval, open, high, low, close, volume)
            VALUES %s
            ON CONFLICT (symbol, timestamp, interval) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                ingested_at = now()
        """, values)
        conn.commit()
    
    return len(values)


def sync_intraday(
    symbols: list[str] = None,
    interval: str = "30m",
    range_param: str = "5d",
    on_progress: Optional[Callable[[int, int, str, int], None]] = None
) -> SyncResult:
    """Sync intraday data for all symbols.
    
    Args:
        symbols: List of symbols (default: all from indices)
        interval: Candle interval (default: 30m)
        range_param: Date range (default: 5d)
        on_progress: Callback(current, total, current_symbols, bars_so_far)
    
    Returns:
        SyncResult with stats
    """
    import time
    start = time.time()
    
    with get_connection() as conn:
        ensure_intraday_table(conn)
        
        if symbols is None:
            symbols = get_symbols_from_indices(conn)
        
        total = len(symbols)
        logger.info(f"Starting intraday sync: {total} symbols, interval={interval}, range={range_param}")
        
        all_bars = 0
        failed_symbols = []
        processed = 0
        
        # Process in batches
        for i in range(0, total, BATCH_SIZE):
            batch = symbols[i:i + BATCH_SIZE]
            batch_str = ",".join(batch[:3]) + ("..." if len(batch) > 3 else "")
            
            bars, failed = fetch_batch(batch, interval, range_param)
            failed_symbols.extend(failed)
            
            if bars:
                inserted = upsert_intraday_bars(conn, bars, interval)
                all_bars += inserted
            
            processed += len(batch)
            
            if on_progress:
                on_progress(processed, total, batch_str, all_bars)
            
            logger.info(f"[{processed}/{total}] {batch_str} | +{len(bars)} bars | Total: {all_bars}")
    
    duration = time.time() - start
    
    return SyncResult(
        symbols_total=total,
        symbols_success=total - len(failed_symbols),
        symbols_failed=len(failed_symbols),
        bars_inserted=all_bars,
        duration_secs=duration,
        errors=[{"symbol": s, "error": "no data"} for s in failed_symbols]
    )


def sync_daily(
    symbols: list[str] = None,
    range_param: str = "1mo",
    on_progress: Optional[Callable[[int, int, str, int], None]] = None
) -> SyncResult:
    """Sync daily OHLCV data for all symbols.
    
    Uses existing ohlcv_daily table.
    """
    import time
    start = time.time()
    
    with get_connection() as conn:
        if symbols is None:
            symbols = get_symbols_from_indices(conn)
        
        total = len(symbols)
        logger.info(f"Starting daily sync: {total} symbols, range={range_param}")
        
        all_bars = 0
        failed_symbols = []
        processed = 0
        
        for i in range(0, total, BATCH_SIZE):
            batch = symbols[i:i + BATCH_SIZE]
            batch_str = ",".join(batch[:3]) + ("..." if len(batch) > 3 else "")
            
            bars, failed = fetch_batch(batch, interval="1d", range_param=range_param)
            failed_symbols.extend(failed)
            
            if bars:
                # Insert into ohlcv_daily
                values = [(b[0], b[1].date(), b[2], b[3], b[4], b[5], b[6]) for b in bars]
                with conn.cursor() as cur:
                    execute_values(cur, """
                        INSERT INTO ohlcv_daily 
                        (symbol, trading_date, open, high, low, close, volume)
                        VALUES %s
                        ON CONFLICT (symbol, trading_date) DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume,
                            ingested_at = now()
                    """, values)
                    conn.commit()
                all_bars += len(values)
            
            processed += len(batch)
            
            if on_progress:
                on_progress(processed, total, batch_str, all_bars)
            
            logger.info(f"[{processed}/{total}] {batch_str} | +{len(bars)} bars | Total: {all_bars}")
    
    duration = time.time() - start
    
    return SyncResult(
        symbols_total=total,
        symbols_success=total - len(failed_symbols),
        symbols_failed=len(failed_symbols),
        bars_inserted=all_bars,
        duration_secs=duration,
        errors=[{"symbol": s, "error": "no data"} for s in failed_symbols]
    )








































