"""CSV storage backend for FX rates.

Storage format:
    cache/fx/{pair}.csv (e.g., USD_BRL.csv)
    
CSV format:
    date,rate,source
    2024-01-02,4.8521,BCB
    2024-01-03,4.8934,BCB
"""

import logging
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..providers.base import FxRecord

logger = logging.getLogger(__name__)


class CsvFxStorage:
    """CSV-based storage for FX rates."""
    
    CSV_HEADER = "date,rate,source"
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _pair_to_filename(self, pair: str) -> str:
        """Convert pair (USD/BRL) to filename (USD_BRL.csv)."""
        return pair.replace("/", "_") + ".csv"
    
    def _filename_to_pair(self, filename: str) -> str:
        """Convert filename (USD_BRL.csv) to pair (USD/BRL)."""
        return filename.replace(".csv", "").replace("_", "/")
    
    def get_path(self, pair: str) -> Path:
        """Get file path for a currency pair."""
        return self.cache_dir / self._pair_to_filename(pair)
    
    def load(self, pair: str) -> List[FxRecord]:
        """Load all rates for a pair from CSV."""
        path = self.get_path(pair)
        
        if not path.exists():
            logger.debug(f"No data file for {pair}")
            return []
        
        records = []
        with open(path, "r") as f:
            # Skip header
            next(f, None)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = FxRecord.from_csv_row(line, pair)
                if record:
                    records.append(record)
        
        logger.info(f"Loaded {len(records)} records for {pair}")
        return sorted(records, key=lambda r: r.date)
    
    def save(self, pair: str, records: List[FxRecord]) -> int:
        """Save rates to CSV (overwrites existing)."""
        if not records:
            return 0
        
        path = self.get_path(pair)
        records = sorted(records, key=lambda r: r.date)
        
        with open(path, "w") as f:
            f.write(self.CSV_HEADER + "\n")
            for record in records:
                f.write(record.to_csv_row() + "\n")
        
        logger.info(f"Saved {len(records)} records for {pair} to {path}")
        return len(records)
    
    def append(self, pair: str, new_records: List[FxRecord]) -> int:
        """Append new records, avoiding duplicates."""
        existing = self.load(pair)
        existing_dates = {r.date for r in existing}
        
        added = 0
        for record in new_records:
            if record.date not in existing_dates:
                existing.append(record)
                existing_dates.add(record.date)
                added += 1
        
        if added > 0:
            self.save(pair, existing)
        
        return added
    
    def get_date_range(self, pair: str) -> Optional[Tuple[date, date]]:
        """Get the date range of stored data."""
        records = self.load(pair)
        if not records:
            return None
        return (records[0].date, records[-1].date)
    
    def get_latest_date(self, pair: str) -> Optional[date]:
        """Get the most recent date in storage."""
        range_info = self.get_date_range(pair)
        return range_info[1] if range_info else None
    
    def list_pairs(self) -> List[str]:
        """List all pairs with stored data."""
        pairs = []
        for file in self.cache_dir.glob("*.csv"):
            if file.stem:  # Ignore empty stems
                pairs.append(self._filename_to_pair(file.name))
        return pairs
    
    def get_status(self) -> Dict[str, dict]:
        """Get status for all stored pairs."""
        status = {}
        for pair in self.list_pairs():
            records = self.load(pair)
            if records:
                status[pair] = {
                    "count": len(records),
                    "first_date": records[0].date.isoformat(),
                    "last_date": records[-1].date.isoformat(),
                    "sources": list(set(r.source for r in records)),
                }
        return status































