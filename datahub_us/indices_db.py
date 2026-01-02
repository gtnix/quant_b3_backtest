"""Database operations for US indices."""
import logging
from contextlib import contextmanager
from typing import Generator
import psycopg2
from psycopg2.extras import execute_values
import os

from .indices import USIndexData, US_INDEX_FETCHERS

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@contextmanager
def get_connection() -> Generator:
    """Context manager for database connection."""
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    try:
        yield conn
    finally:
        conn.close()


def ensure_tables_exist(conn) -> None:
    """Create US indices tables if they don't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS us_indices (
                index_code VARCHAR(20) PRIMARY KEY,
                index_name VARCHAR(100),
                description TEXT,
                total_components INTEGER,
                last_updated_at TIMESTAMPTZ DEFAULT now(),
                created_at TIMESTAMPTZ DEFAULT now()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS us_index_composition (
                id SERIAL PRIMARY KEY,
                index_code VARCHAR(20) REFERENCES us_indices(index_code),
                symbol VARCHAR(20) NOT NULL,
                company_name VARCHAR(100),
                sector VARCHAR(50),
                weight NUMERIC(8,4),
                as_of_date DATE NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now(),
                UNIQUE(index_code, symbol, as_of_date)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_us_comp_idx ON us_index_composition(index_code)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_us_comp_sym ON us_index_composition(symbol)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_us_comp_date ON us_index_composition(as_of_date)")
        conn.commit()


def upsert_index(conn, data: USIndexData) -> int:
    """Insert or update US index and its composition."""
    with conn.cursor() as cur:
        # Upsert index metadata
        cur.execute("""
            INSERT INTO us_indices (index_code, index_name, total_components)
            VALUES (%s, %s, %s)
            ON CONFLICT (index_code) DO UPDATE SET
                total_components = EXCLUDED.total_components,
                last_updated_at = now()
        """, (data.index_code, data.index_name, data.total_components))
        
        # Ensure instruments exist (for US symbols)
        for comp in data.components:
            cur.execute("""
                INSERT INTO instruments (symbol, name, short_name, type, currency)
                VALUES (%s, %s, %s, 'stock', 'USD')
                ON CONFLICT (symbol) DO UPDATE SET updated_at = now()
            """, (comp.symbol, comp.company_name, comp.company_name))
        
        # Delete old composition for this date
        cur.execute(
            "DELETE FROM us_index_composition WHERE index_code = %s AND as_of_date = %s",
            (data.index_code, data.date)
        )
        
        # Insert new composition
        values = [
            (data.index_code, c.symbol, c.company_name, c.sector, c.weight, data.date)
            for c in data.components
        ]
        execute_values(cur, """
            INSERT INTO us_index_composition 
            (index_code, symbol, company_name, sector, weight, as_of_date)
            VALUES %s
        """, values)
        
        conn.commit()
        return len(data.components)


def get_index_symbols(conn, index_code: str, as_of_date=None) -> list[str]:
    """Return list of symbols for an index."""
    with conn.cursor() as cur:
        if as_of_date:
            cur.execute("""
                SELECT symbol FROM us_index_composition
                WHERE index_code = %s AND as_of_date = %s
                ORDER BY symbol
            """, (index_code, as_of_date))
        else:
            cur.execute("""
                SELECT symbol FROM us_index_composition
                WHERE index_code = %s AND as_of_date = (
                    SELECT MAX(as_of_date) FROM us_index_composition WHERE index_code = %s
                )
                ORDER BY symbol
            """, (index_code, index_code))
        
        return [row[0] for row in cur.fetchall()]


def get_all_indices(conn) -> list[dict]:
    """Get all US indices with their stats."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT index_code, index_name, total_components, last_updated_at
            FROM us_indices ORDER BY index_code
        """)
        return [
            {"code": r[0], "name": r[1], "components": r[2], "updated": r[3]}
            for r in cur.fetchall()
        ]






























