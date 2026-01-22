#!/usr/bin/env python3
"""
Data Sync Check - Verifica se os dados estão sincronizados antes do mining.
Uso: python scripts/check_data_sync.py [--market BR|US|all] [--fail-on-stale]
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

MARKET_FILES = {
    "BR": "market_data_ibov.csv",
    "US": "market_data_us.csv",
}

# Tolerância: dados devem ter no máximo N dias de atraso
MAX_STALE_DAYS = 3


def check_file_freshness(filepath: Path, market: str) -> dict:
    """Verifica se o arquivo de dados está fresco."""
    result = {
        "market": market,
        "file": str(filepath),
        "exists": False,
        "rows": 0,
        "last_date": None,
        "stale_days": None,
        "status": "UNKNOWN",
    }
    
    if not filepath.exists():
        result["status"] = "MISSING"
        return result
    
    result["exists"] = True
    
    try:
        import pandas as pd
        df = pd.read_csv(filepath)
        result["rows"] = len(df)
        
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            last_date = df["date"].max()
            result["last_date"] = last_date.strftime("%Y-%m-%d")
            
            # Calcular dias de atraso (considera fins de semana)
            today = datetime.now()
            stale_days = (today - last_date).days
            result["stale_days"] = stale_days
            
            if stale_days <= MAX_STALE_DAYS:
                result["status"] = "OK"
            elif stale_days <= 7:
                result["status"] = "STALE"
            else:
                result["status"] = "OUTDATED"
        else:
            result["status"] = "NO_DATE_COLUMN"
            
    except Exception as e:
        result["status"] = f"ERROR: {e}"
    
    return result


def check_data_sync(markets: list, fail_on_stale: bool = False) -> bool:
    """Verifica sincronia de dados para os mercados especificados."""
    print("=" * 60)
    print("DATA SYNC CHECK")
    print("=" * 60)
    
    all_ok = True
    results = []
    
    for market in markets:
        if market not in MARKET_FILES:
            print(f"[WARN] Unknown market: {market}")
            continue
            
        filepath = DATA_DIR / MARKET_FILES[market]
        result = check_file_freshness(filepath, market)
        results.append(result)
        
        status_icon = {
            "OK": "✓",
            "STALE": "⚠",
            "OUTDATED": "✗",
            "MISSING": "✗",
        }.get(result["status"], "?")
        
        print(f"\n[{market}] {status_icon} {result['status']}")
        print(f"  File: {result['file']}")
        
        if result["exists"]:
            print(f"  Rows: {result['rows']:,}")
            if result["last_date"]:
                print(f"  Last date: {result['last_date']}")
                print(f"  Stale days: {result['stale_days']}")
        
        if result["status"] in ("OUTDATED", "MISSING"):
            all_ok = False
        elif result["status"] == "STALE" and fail_on_stale:
            all_ok = False
    
    print("\n" + "=" * 60)
    
    if all_ok:
        print("✓ DATA SYNC OK - Ready for mining")
    else:
        print("✗ DATA SYNC FAILED - Run data sync before mining")
        print("\nTo sync data:")
        print("  BR: python -m datahub_b3 sync --days 30")
        print("  US: python datahub_us/router.py sync --universe sample")
    
    print("=" * 60)
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Check data sync status")
    parser.add_argument(
        "--market", 
        choices=["BR", "US", "all"], 
        default="all",
        help="Market to check (default: all)"
    )
    parser.add_argument(
        "--fail-on-stale",
        action="store_true",
        help="Fail if data is stale (>3 days old)"
    )
    
    args = parser.parse_args()
    
    markets = ["BR", "US"] if args.market == "all" else [args.market]
    
    if not check_data_sync(markets, args.fail_on_stale):
        sys.exit(1)


if __name__ == "__main__":
    main()
