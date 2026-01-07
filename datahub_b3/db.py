"""Database operations for B3 indices."""
import logging
from contextlib import contextmanager
from typing import Generator
import psycopg2
from psycopg2.extras import execute_values

from .config import DATABASE_URL
from .scraper import IndexData

logger = logging.getLogger(__name__)


@contextmanager
def get_connection() -> Generator:
    """Context manager for database connection."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


def ensure_tables_exist(conn) -> None:
    """Cria tabelas se não existirem."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS b3_indices (
                index_code VARCHAR(20) PRIMARY KEY,
                index_name VARCHAR(100),
                description TEXT,
                total_components INTEGER,
                last_updated_at TIMESTAMPTZ DEFAULT now(),
                reductor NUMERIC,
                theoretical_qty BIGINT,
                created_at TIMESTAMPTZ DEFAULT now()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS b3_index_composition (
                id SERIAL PRIMARY KEY,
                index_code VARCHAR(20) REFERENCES b3_indices(index_code),
                symbol VARCHAR(20) NOT NULL,
                company_name VARCHAR(100),
                stock_type VARCHAR(30),
                participation_pct NUMERIC(8,4),
                theoretical_qty BIGINT,
                as_of_date DATE NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now(),
                UNIQUE(index_code, symbol, as_of_date)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_b3_comp_idx ON b3_index_composition(index_code)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_b3_comp_sym ON b3_index_composition(symbol)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_b3_comp_date ON b3_index_composition(as_of_date)")
        conn.commit()


def upsert_index(conn, data: IndexData) -> int:
    """Insere ou atualiza índice e sua composição. Retorna quantidade inserida."""
    from .config import B3_INDICES
    
    meta = B3_INDICES.get(data.index_code, {})
    
    with conn.cursor() as cur:
        # Upsert index metadata
        cur.execute("""
            INSERT INTO b3_indices (index_code, index_name, description, total_components, reductor, theoretical_qty)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (index_code) DO UPDATE SET
                total_components = EXCLUDED.total_components,
                reductor = EXCLUDED.reductor,
                theoretical_qty = EXCLUDED.theoretical_qty,
                last_updated_at = now()
        """, (
            data.index_code,
            meta.get("name", data.index_code),
            meta.get("desc", ""),
            data.total_components,
            data.reductor,
            data.theoretical_qty
        ))
        
        # Ensure instruments exist
        for comp in data.components:
            cur.execute("""
                INSERT INTO instruments (symbol, name, short_name, type, currency)
                VALUES (%s, %s, %s, 'stock', 'BRL')
                ON CONFLICT (symbol) DO UPDATE SET updated_at = now()
            """, (comp.symbol, comp.company_name, comp.company_name))
        
        # Delete old composition for this date
        cur.execute(
            "DELETE FROM b3_index_composition WHERE index_code = %s AND as_of_date = %s",
            (data.index_code, data.date)
        )
        
        # Insert new composition
        values = [
            (data.index_code, c.symbol, c.company_name, c.stock_type, c.participation_pct, c.theoretical_qty, data.date)
            for c in data.components
        ]
        execute_values(cur, """
            INSERT INTO b3_index_composition 
            (index_code, symbol, company_name, stock_type, participation_pct, theoretical_qty, as_of_date)
            VALUES %s
        """, values)
        
        conn.commit()
        return len(data.components)


def get_index_symbols(conn, index_code: str, as_of_date=None) -> list[str]:
    """Retorna lista de símbolos de um índice."""
    with conn.cursor() as cur:
        if as_of_date:
            cur.execute("""
                SELECT symbol FROM b3_index_composition
                WHERE index_code = %s AND as_of_date = %s
                ORDER BY participation_pct DESC
            """, (index_code, as_of_date))
        else:
            cur.execute("""
                SELECT symbol FROM b3_index_composition
                WHERE index_code = %s AND as_of_date = (
                    SELECT MAX(as_of_date) FROM b3_index_composition WHERE index_code = %s
                )
                ORDER BY participation_pct DESC
            """, (index_code, index_code))
        
        return [row[0] for row in cur.fetchall()]








































