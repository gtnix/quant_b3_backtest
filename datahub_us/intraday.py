"""Intraday data sync for US stocks via yfinance."""
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional
import yfinance as yf
import pandas as pd
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DATABASE_URL = os.getenv("DATABASE_URL", "")


@dataclass
class SyncResult:
    symbols_total: int
    symbols_success: int
    symbols_failed: int
    bars_inserted: int
    duration_secs: float
    errors: list


def get_connection():
    """Get database connection."""
    import psycopg2
    return psycopg2.connect(DATABASE_URL)


def get_symbols_from_indices(conn) -> list[str]:
    """Get unique symbols from all US indices."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT symbol FROM us_index_composition ORDER BY symbol
        """)
        return [row[0] for row in cur.fetchall()]


def ensure_intraday_table(conn):
    """Create US intraday table if not exists."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_intraday_us (
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
        cur.execute("CREATE INDEX IF NOT EXISTS idx_intraday_us_sym ON ohlcv_intraday_us(symbol)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_intraday_us_ts ON ohlcv_intraday_us(timestamp DESC)")
        conn.commit()


def ensure_daily_table(conn):
    """Create US daily OHLCV table if not exists."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_daily_us (
                symbol VARCHAR(20) NOT NULL,
                trading_date DATE NOT NULL,
                open NUMERIC(12,4),
                high NUMERIC(12,4),
                low NUMERIC(12,4),
                close NUMERIC(12,4),
                volume BIGINT,
                ingested_at TIMESTAMPTZ DEFAULT now(),
                PRIMARY KEY (symbol, trading_date)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_daily_us_sym ON ohlcv_daily_us(symbol)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_daily_us_dt ON ohlcv_daily_us(trading_date DESC)")
        conn.commit()


def fetch_intraday_yfinance(symbol: str, interval: str = "30m", period: str = "5d") -> pd.DataFrame:
    """Fetch intraday data for a single symbol using yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        
        # Rename datetime column
        if 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'timestamp'})
        elif 'date' in df.columns:
            df = df.rename(columns={'date': 'timestamp'})
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        logger.warning(f"{symbol}: {e}")
        return pd.DataFrame()


def upsert_intraday_bars(conn, symbol: str, df: pd.DataFrame, interval: str) -> int:
    """Insert/update intraday bars for a symbol."""
    if df.empty:
        return 0
    
    values = []
    for _, row in df.iterrows():
        ts = row['timestamp']
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        values.append((
            symbol, ts, interval,
            float(row['open']) if pd.notna(row['open']) else None,
            float(row['high']) if pd.notna(row['high']) else None,
            float(row['low']) if pd.notna(row['low']) else None,
            float(row['close']) if pd.notna(row['close']) else None,
            int(row['volume']) if pd.notna(row['volume']) else 0
        ))
    
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO ohlcv_intraday_us 
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


def sync_intraday_us(
    symbols: list[str] = None,
    interval: str = "30m",
    period: str = "5d",
    on_progress: Optional[Callable[[int, int, str, int], None]] = None
) -> SyncResult:
    """Sync intraday data for US stocks.
    
    Args:
        symbols: List of symbols (default: all from indices)
        interval: Candle interval (default: 30m)
        period: Date range (default: 5d)
        on_progress: Callback(current, total, symbol, bars_so_far)
    """
    import time
    start = time.time()
    
    conn = get_connection()
    try:
        ensure_intraday_table(conn)
        
        if symbols is None:
            symbols = get_symbols_from_indices(conn)
        
        total = len(symbols)
        logger.info(f"Starting US intraday sync: {total} symbols, interval={interval}, period={period}")
        
        all_bars = 0
        failed_symbols = []
        
        for i, symbol in enumerate(symbols, 1):
            df = fetch_intraday_yfinance(symbol, interval, period)
            
            if df.empty:
                failed_symbols.append(symbol)
            else:
                inserted = upsert_intraday_bars(conn, symbol, df, interval)
                all_bars += inserted
            
            if on_progress:
                on_progress(i, total, symbol, all_bars)
            
            if i % 50 == 0:
                logger.info(f"[{i}/{total}] {symbol} | Total: {all_bars:,} bars")
    finally:
        conn.close()
    
    duration = time.time() - start
    
    return SyncResult(
        symbols_total=total,
        symbols_success=total - len(failed_symbols),
        symbols_failed=len(failed_symbols),
        bars_inserted=all_bars,
        duration_secs=duration,
        errors=[{"symbol": s, "error": "no data"} for s in failed_symbols]
    )


def sync_daily_us(
    symbols: list[str] = None,
    period: str = "1mo",
    on_progress: Optional[Callable[[int, int, str, int], None]] = None
) -> SyncResult:
    """Sync daily OHLCV data for US stocks."""
    import time
    start = time.time()
    
    conn = get_connection()
    try:
        ensure_daily_table(conn)
        
        if symbols is None:
            symbols = get_symbols_from_indices(conn)
        
        total = len(symbols)
        logger.info(f"Starting US daily sync: {total} symbols, period={period}")
        
        all_bars = 0
        failed_symbols = []
        
        for i, symbol in enumerate(symbols, 1):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval="1d")
                
                if df.empty:
                    failed_symbols.append(symbol)
                    continue
                
                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]
                
                values = []
                for _, row in df.iterrows():
                    dt = row.get('date', row.get('datetime'))
                    if isinstance(dt, pd.Timestamp):
                        dt = dt.date()
                    values.append((
                        symbol, dt,
                        float(row['open']) if pd.notna(row['open']) else None,
                        float(row['high']) if pd.notna(row['high']) else None,
                        float(row['low']) if pd.notna(row['low']) else None,
                        float(row['close']) if pd.notna(row['close']) else None,
                        int(row['volume']) if pd.notna(row['volume']) else 0
                    ))
                
                with conn.cursor() as cur:
                    execute_values(cur, """
                        INSERT INTO ohlcv_daily_us 
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
                
            except Exception as e:
                logger.warning(f"{symbol}: {e}")
                failed_symbols.append(symbol)
            
            if on_progress:
                on_progress(i, total, symbol, all_bars)
            
            if i % 50 == 0:
                logger.info(f"[{i}/{total}] {symbol} | Total: {all_bars:,} bars")
    finally:
        conn.close()
    
    duration = time.time() - start
    
    return SyncResult(
        symbols_total=total,
        symbols_success=total - len(failed_symbols),
        symbols_failed=len(failed_symbols),
        bars_inserted=all_bars,
        duration_secs=duration,
        errors=[{"symbol": s, "error": "no data"} for s in failed_symbols]
    )

