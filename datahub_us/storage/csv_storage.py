"""CSV storage compatible with BR system format."""

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd

from ..config import OHLCV_DIR, METADATA_FILE, CSV_COLUMNS

logger = logging.getLogger(__name__)


class CSVStorage:
    """CSV storage for OHLCV data, compatible with BR system format.
    
    Storage format:
        cache/us/ohlcv/{SYMBOL}.csv
        
    CSV schema:
        date,open,high,low,close,volume
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or OHLCV_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.base_dir.parent / "metadata.json"
    
    def _get_path(self, symbol: str) -> Path:
        """Get file path for a symbol."""
        return self.base_dir / f"{symbol}.csv"
    
    def exists(self, symbol: str) -> bool:
        """Check if data exists for a symbol."""
        return self._get_path(symbol).exists()
    
    def read(self, symbol: str) -> pd.DataFrame:
        """Read data for a symbol.
        
        Returns:
            DataFrame with OHLCV data or empty DataFrame if not found
        """
        path = self._get_path(symbol)
        
        if not path.exists():
            logger.debug(f"No data found for {symbol}")
            return pd.DataFrame(columns=CSV_COLUMNS)
        
        try:
            df = pd.read_csv(path)
            logger.debug(f"Read {len(df)} rows for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Failed to read {symbol}: {e}")
            return pd.DataFrame(columns=CSV_COLUMNS)
    
    def write(self, symbol: str, df: pd.DataFrame, overwrite: bool = False) -> int:
        """Write data for a symbol.
        
        Args:
            symbol: Ticker symbol
            df: DataFrame with OHLCV data
            overwrite: If True, replace existing data. If False, merge (idempotent)
            
        Returns:
            Number of new rows written
        """
        if df.empty:
            logger.debug(f"Empty DataFrame for {symbol}, skipping write")
            return 0
        
        path = self._get_path(symbol)
        
        # Ensure correct columns
        df = df[CSV_COLUMNS].copy()
        
        if overwrite or not path.exists():
            # Full write
            df.to_csv(path, index=False)
            logger.info(f"Wrote {len(df)} rows for {symbol}")
            return len(df)
        
        # Merge with existing data (idempotent upsert)
        existing = self.read(symbol)
        
        if existing.empty:
            df.to_csv(path, index=False)
            return len(df)
        
        # Combine and deduplicate by date
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date'], keep='last')
        combined = combined.sort_values('date').reset_index(drop=True)
        
        new_rows = len(combined) - len(existing)
        
        combined.to_csv(path, index=False)
        logger.info(f"Merged {symbol}: {len(existing)} -> {len(combined)} rows (+{new_rows})")
        
        return new_rows
    
    def get_date_range(self, symbol: str) -> tuple[Optional[date], Optional[date]]:
        """Get date range for a symbol.
        
        Returns:
            (start_date, end_date) or (None, None) if no data
        """
        df = self.read(symbol)
        
        if df.empty:
            return None, None
        
        dates = pd.to_datetime(df['date'])
        return dates.min().date(), dates.max().date()
    
    def get_last_date(self, symbol: str) -> Optional[date]:
        """Get last date for a symbol."""
        _, last = self.get_date_range(symbol)
        return last
    
    def list_symbols(self) -> List[str]:
        """List all stored symbols."""
        return [p.stem for p in self.base_dir.glob("*.csv")]
    
    def get_stats(self) -> Dict:
        """Get storage statistics."""
        symbols = self.list_symbols()
        total_rows = 0
        earliest = None
        latest = None
        
        for symbol in symbols:
            df = self.read(symbol)
            total_rows += len(df)
            
            if not df.empty:
                dates = pd.to_datetime(df['date'])
                sym_min, sym_max = dates.min().date(), dates.max().date()
                
                if earliest is None or sym_min < earliest:
                    earliest = sym_min
                if latest is None or sym_max > latest:
                    latest = sym_max
        
        return {
            "symbols_count": len(symbols),
            "total_bars": total_rows,
            "start_date": earliest.isoformat() if earliest else None,
            "end_date": latest.isoformat() if latest else None,
        }
    
    def delete(self, symbol: str) -> bool:
        """Delete data for a symbol."""
        path = self._get_path(symbol)
        if path.exists():
            path.unlink()
            logger.info(f"Deleted {symbol}")
            return True
        return False
    
    def update_metadata(self):
        """Update metadata file with current stats."""
        stats = self.get_stats()
        stats["exported_at"] = datetime.utcnow().isoformat() + "Z"
        stats["source"] = "yfinance"
        
        with open(self.metadata_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Updated metadata: {stats}")
    
    def detect_gaps(self, symbol: str, max_gap_days: int = 5) -> List[dict]:
        """Detect gaps in data for a symbol."""
        df = self.read(symbol)
        
        if len(df) < 2:
            return []
        
        df = df.sort_values('date').reset_index(drop=True)
        dates = pd.to_datetime(df['date'])
        
        gaps = []
        for i in range(1, len(dates)):
            prev = dates.iloc[i-1]
            curr = dates.iloc[i]
            delta = (curr - prev).days
            
            if delta > max_gap_days:
                gaps.append({
                    "start": prev.strftime("%Y-%m-%d"),
                    "end": curr.strftime("%Y-%m-%d"),
                    "days": delta
                })
        
        return gaps





























