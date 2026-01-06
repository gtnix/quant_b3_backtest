"""Database operations for FX rates."""
import logging
import os
from contextlib import contextmanager
from datetime import date
from decimal import Decimal
from typing import Generator, List, Optional, Tuple

import psycopg2
from psycopg2.extras import execute_values

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")


@contextmanager
def get_connection() -> Generator:
    """Context manager for database connection."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


def ensure_table_exists(conn) -> None:
    """Create fx_rates table if not exists."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fx_rates (
                pair VARCHAR(10) NOT NULL,
                rate_date DATE NOT NULL,
                rate NUMERIC(18,8) NOT NULL,
                source VARCHAR(10) NOT NULL,
                ingested_at TIMESTAMPTZ DEFAULT now(),
                PRIMARY KEY (pair, rate_date)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_fx_rates_pair ON fx_rates(pair)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_fx_rates_date ON fx_rates(rate_date DESC)")
        conn.commit()


def upsert_rates(conn, records: List[Tuple[str, date, Decimal, str]]) -> int:
    """Insert or update FX rates. Returns count inserted."""
    if not records:
        return 0
    
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO fx_rates (pair, rate_date, rate, source)
            VALUES %s
            ON CONFLICT (pair, rate_date) DO UPDATE SET
                rate = EXCLUDED.rate,
                source = EXCLUDED.source,
                ingested_at = now()
        """, records)
        conn.commit()
    
    return len(records)


def get_latest_date(conn, pair: str) -> Optional[date]:
    """Get the most recent date for a pair."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT MAX(rate_date) FROM fx_rates WHERE pair = %s
        """, (pair,))
        row = cur.fetchone()
        return row[0] if row and row[0] else None


def get_all_rates(conn, pair: str) -> List[Tuple[date, Decimal, str]]:
    """Get all rates for a pair."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT rate_date, rate, source FROM fx_rates
            WHERE pair = %s ORDER BY rate_date
        """, (pair,))
        return cur.fetchall()


def get_stats(conn) -> dict:
    """Get statistics for all pairs."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT pair, COUNT(*), MIN(rate_date), MAX(rate_date)
            FROM fx_rates GROUP BY pair ORDER BY pair
        """)
        rows = cur.fetchall()
        return {
            row[0]: {"count": row[1], "min_date": row[2], "max_date": row[3]}
            for row in rows
        }



