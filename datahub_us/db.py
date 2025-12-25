"""Database layer for Neon Postgres - US Market Data."""

import os
import asyncio
import logging
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Tuple, Callable
import pandas as pd

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import asyncpg
except ImportError:
    raise ImportError("asyncpg is required. Install with: pip install asyncpg")

from .config import OHLCV_DIR, CSV_COLUMNS

logger = logging.getLogger(__name__)


class DbError(Exception):
    """Database error."""
    pass


class Database:
    """Async database connection for Neon Postgres."""
    
    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool
    
    @classmethod
    async def connect(cls) -> "Database":
        """Connect to Neon database using DATABASE_URL."""
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise DbError("DATABASE_URL not set")
        
        # asyncpg needs postgresql:// not postgres://
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        
        try:
            pool = await asyncpg.create_pool(
                database_url,
                min_size=1,
                max_size=5,
                ssl="require",
            )
            logger.info("Connected to Neon database")
            return cls(pool)
        except Exception as e:
            raise DbError(f"Connection failed: {e}")
    
    async def close(self):
        """Close database connection."""
        await self._pool.close()
        logger.info("Database connection closed")
    
    async def ensure_schema(self):
        """Create tables if they don't exist."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_daily_us (
                    symbol TEXT NOT NULL,
                    trading_date DATE NOT NULL,
                    open DECIMAL(12,4),
                    high DECIMAL(12,4),
                    low DECIMAL(12,4),
                    close DECIMAL(12,4),
                    volume BIGINT,
                    ingested_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (symbol, trading_date)
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlcv_us_date 
                ON ohlcv_daily_us(trading_date)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlcv_us_symbol 
                ON ohlcv_daily_us(symbol)
            """)
            logger.info("Schema ensured: ohlcv_daily_us")
    
    async def upsert_batch(self, symbol: str, df: pd.DataFrame) -> int:
        """Upsert OHLCV data for a symbol.
        
        Args:
            symbol: Ticker symbol
            df: DataFrame with columns: date, open, high, low, close, volume
            
        Returns:
            Number of rows upserted
        """
        if df.empty:
            return 0
        
        # Prepare data
        rows = []
        for _, row in df.iterrows():
            try:
                trading_date = pd.to_datetime(row['date']).date()
                rows.append((
                    symbol,
                    trading_date,
                    float(row['open']) if pd.notna(row['open']) else None,
                    float(row['high']) if pd.notna(row['high']) else None,
                    float(row['low']) if pd.notna(row['low']) else None,
                    float(row['close']) if pd.notna(row['close']) else None,
                    int(row['volume']) if pd.notna(row['volume']) else 0,
                ))
            except Exception as e:
                logger.warning(f"Skipping row for {symbol}: {e}")
                continue
        
        if not rows:
            return 0
        
        async with self._pool.acquire() as conn:
            # Use executemany with ON CONFLICT
            result = await conn.executemany("""
                INSERT INTO ohlcv_daily_us (symbol, trading_date, open, high, low, close, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (symbol, trading_date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    ingested_at = NOW()
            """, rows)
        
        logger.debug(f"Upserted {len(rows)} rows for {symbol}")
        return len(rows)
    
    async def get_last_date(self, symbol: str) -> Optional[date]:
        """Get last trading date for a symbol."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT MAX(trading_date) FROM ohlcv_daily_us WHERE symbol = $1
            """, symbol)
            return row[0] if row and row[0] else None
    
    async def get_date_range(self, symbol: str) -> Tuple[Optional[date], Optional[date]]:
        """Get date range for a symbol."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT MIN(trading_date), MAX(trading_date) 
                FROM ohlcv_daily_us WHERE symbol = $1
            """, symbol)
            if row:
                return row[0], row[1]
            return None, None
    
    async def get_symbols(self) -> List[str]:
        """Get all symbols in database."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT DISTINCT symbol FROM ohlcv_daily_us ORDER BY symbol
            """)
            return [row['symbol'] for row in rows]
    
    async def get_stats(self) -> dict:
        """Get database statistics."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT 
                    COUNT(DISTINCT symbol) as symbols,
                    COUNT(*) as total_bars,
                    MIN(trading_date) as start_date,
                    MAX(trading_date) as end_date
                FROM ohlcv_daily_us
            """)
            return {
                "symbols": row['symbols'] or 0,
                "total_bars": row['total_bars'] or 0,
                "start_date": row['start_date'].isoformat() if row['start_date'] else None,
                "end_date": row['end_date'].isoformat() if row['end_date'] else None,
            }
    
    async def fetch_ohlcv(
        self, 
        symbol: str, 
        start: Optional[date] = None,
        end: Optional[date] = None
    ) -> pd.DataFrame:
        """Fetch OHLCV data from database.
        
        Args:
            symbol: Ticker symbol
            start: Start date (optional)
            end: End date (optional)
            
        Returns:
            DataFrame with OHLCV data
        """
        async with self._pool.acquire() as conn:
            query = """
                SELECT trading_date as date, open, high, low, close, volume
                FROM ohlcv_daily_us
                WHERE symbol = $1
            """
            params = [symbol]
            
            if start:
                query += " AND trading_date >= $2"
                params.append(start)
            if end:
                idx = len(params) + 1
                query += f" AND trading_date <= ${idx}"
                params.append(end)
            
            query += " ORDER BY trading_date"
            
            rows = await conn.fetch(query, *params)
        
        if not rows:
            return pd.DataFrame(columns=CSV_COLUMNS)
        
        data = []
        for row in rows:
            data.append({
                'date': row['date'].isoformat(),
                'open': float(row['open']) if row['open'] else None,
                'high': float(row['high']) if row['high'] else None,
                'low': float(row['low']) if row['low'] else None,
                'close': float(row['close']) if row['close'] else None,
                'volume': int(row['volume']) if row['volume'] else 0,
            })
        
        return pd.DataFrame(data)
    
    async def export_symbol_to_csv(self, symbol: str, output_dir: Optional[Path] = None) -> int:
        """Export a symbol's data to CSV file.
        
        Args:
            symbol: Ticker symbol
            output_dir: Output directory (default: cache/us/ohlcv/)
            
        Returns:
            Number of rows exported
        """
        output_dir = output_dir or OHLCV_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = await self.fetch_ohlcv(symbol)
        
        if df.empty:
            return 0
        
        file_path = output_dir / f"{symbol}.csv"
        df.to_csv(file_path, index=False)
        
        logger.debug(f"Exported {len(df)} rows for {symbol} to {file_path}")
        return len(df)
    
    async def export_all_to_csv(
        self, 
        symbols: Optional[List[str]] = None,
        output_dir: Optional[Path] = None,
        on_progress: Optional[callable] = None
    ) -> dict:
        """Export all symbols to CSV files.
        
        Args:
            symbols: List of symbols (default: all in database)
            output_dir: Output directory
            on_progress: Progress callback (current, total, symbol)
            
        Returns:
            Dictionary with export statistics
        """
        output_dir = output_dir or OHLCV_DIR
        
        if symbols is None:
            symbols = await self.get_symbols()
        
        total_rows = 0
        exported = 0
        
        for i, symbol in enumerate(symbols, 1):
            rows = await self.export_symbol_to_csv(symbol, output_dir)
            total_rows += rows
            if rows > 0:
                exported += 1
            
            if on_progress:
                on_progress(i, len(symbols), symbol)
        
        logger.info(f"Exported {exported} symbols, {total_rows} total rows")
        
        return {
            "symbols_exported": exported,
            "total_rows": total_rows,
            "output_dir": str(output_dir),
        }
    
    async def delete_symbol(self, symbol: str) -> int:
        """Delete all data for a symbol."""
        async with self._pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM ohlcv_daily_us WHERE symbol = $1
            """, symbol)
            count = int(result.split()[-1])
            logger.info(f"Deleted {count} rows for {symbol}")
            return count

    # ========================================================================
    # Dividends Methods
    # ========================================================================

    async def upsert_dividend(
        self,
        symbol: str,
        ex_date: date,
        rate: float,
        dividend_type: str = "DIVIDEND",
        payment_date: Optional[date] = None,
    ) -> bool:
        """Upsert a single dividend entry.
        
        Args:
            symbol: Ticker symbol
            ex_date: Ex-dividend date
            rate: Dividend amount per share
            dividend_type: Type (DIVIDEND for US)
            payment_date: Payment date (optional)
            
        Returns:
            True if inserted/updated successfully
        """
        try:
            async with self._pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO dividends_history (symbol, ex_date, payment_date, rate, dividend_type)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (symbol, payment_date, dividend_type, rate) DO NOTHING
                """, symbol, ex_date, payment_date or ex_date, rate, dividend_type)
            return True
        except Exception as e:
            logger.warning(f"Failed to upsert dividend for {symbol}: {e}")
            return False

    async def upsert_dividends_batch(
        self,
        symbol: str,
        dividends: List[Tuple[date, float]],
        dividend_type: str = "DIVIDEND",
    ) -> int:
        """Upsert multiple dividends for a symbol.
        
        Args:
            symbol: Ticker symbol
            dividends: List of (ex_date, rate) tuples
            dividend_type: Type (DIVIDEND for US)
            
        Returns:
            Number of dividends inserted
        """
        if not dividends:
            return 0
        
        inserted = 0
        async with self._pool.acquire() as conn:
            for ex_date, rate in dividends:
                try:
                    await conn.execute("""
                        INSERT INTO dividends_history (symbol, ex_date, payment_date, rate, dividend_type)
                        VALUES ($1, $2, $2, $3, $4)
                        ON CONFLICT (symbol, payment_date, dividend_type, rate) DO NOTHING
                    """, symbol, ex_date, rate, dividend_type)
                    inserted += 1
                except Exception as e:
                    logger.warning(f"Failed to insert dividend {symbol} {ex_date}: {e}")
        
        if inserted > 0:
            logger.debug(f"Inserted {inserted} dividends for {symbol}")
        return inserted

    async def get_dividend_count(self, symbol: str) -> int:
        """Get count of dividends for a symbol."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT COUNT(*) FROM dividends_history WHERE symbol = $1
            """, symbol)
            return row[0] if row else 0

    async def get_us_symbols_with_dividends(self) -> List[str]:
        """Get list of US symbols that have dividend data."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT DISTINCT d.symbol 
                FROM dividends_history d
                INNER JOIN instruments i ON d.symbol = i.symbol
                WHERE i.currency = 'USD'
                ORDER BY d.symbol
            """)
            return [row['symbol'] for row in rows]

    async def get_synced_dividend_symbols(self) -> set:
        """Get set of symbols that already have dividend data (for skip logic)."""
        symbols = await self.get_us_symbols_with_dividends()
        return set(symbols)


def run_async(coro):
    """Run async coroutine in sync context."""
    return asyncio.get_event_loop().run_until_complete(coro)

