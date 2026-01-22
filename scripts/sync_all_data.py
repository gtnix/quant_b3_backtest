#!/usr/bin/env python3
"""Unified data sync to Neon database and local CSV cache.

This script synchronizes all market data from various sources:
- B3: Brazilian market indices (IBOV, etc.) via B3 API
- US: S&P 500 stocks via yfinance
- FX: Currency rates via BCB/FRED/Brapi

Usage:
    python scripts/sync_all_data.py           # Sync all
    python scripts/sync_all_data.py --b3      # Only B3
    python scripts/sync_all_data.py --us      # Only US
    python scripts/sync_all_data.py --fx      # Only FX
    python scripts/sync_all_data.py --export  # Export Neon to CSV
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def sync_b3():
    """Sync B3 indices composition to Neon."""
    try:
        from datahub_b3.scraper import fetch_index
        from datahub_b3.db import get_connection, ensure_tables_exist, upsert_index
        from datahub_b3.config import B3_INDICES
    except ImportError as e:
        logger.error(f"Cannot import datahub_b3: {e}")
        return {"status": "error", "error": str(e)}
    
    logger.info("=== B3 Indices Sync ===")
    results = {}
    
    with get_connection() as conn:
        ensure_tables_exist(conn)
        
        for index_code in B3_INDICES.keys():
            try:
                data = fetch_index(index_code)
                if data:
                    count = upsert_index(conn, data)
                    results[index_code] = {"status": "success", "components": count}
                    logger.info(f"  {index_code}: {count} components")
                else:
                    results[index_code] = {"status": "empty"}
            except Exception as e:
                results[index_code] = {"status": "error", "error": str(e)}
                logger.error(f"  {index_code}: {e}")
    
    success = sum(1 for r in results.values() if r.get("status") == "success")
    logger.info(f"B3 sync complete: {success}/{len(results)} indices")
    return results


async def sync_us_async(universe: str = "sample"):
    """Sync US OHLCV data to Neon."""
    try:
        from datahub_us.db import Database
        from datahub_us.providers.yfinance_provider import YFinanceProvider
        from datahub_us.universe import get_universe
    except ImportError as e:
        logger.error(f"Cannot import datahub_us: {e}")
        return {"status": "error", "error": str(e)}
    
    logger.info(f"=== US Market Sync ({universe}) ===")
    
    provider = YFinanceProvider(delay_between_requests=0.5)
    symbols = get_universe(universe)
    
    # Connect to database
    db = await Database.connect()
    await db.ensure_schema()
    
    # Date range: 5 years of data
    end_date = date.today()
    start_date = end_date - timedelta(days=5*365)
    
    results = {"synced": 0, "failed": 0, "total_rows": 0}
    
    try:
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"  [{i}/{len(symbols)}] {symbol}...")
            
            result = provider.fetch_ohlcv(symbol, start_date, end_date)
            if result.error:
                results["failed"] += 1
                continue
            
            rows = await db.upsert_batch(symbol, result.df)
            results["synced"] += 1
            results["total_rows"] += rows
            
            # Fetch and store dividends
            actions_df = provider.fetch_actions(symbol, start_date, end_date)
            if not actions_df.empty:
                dividends = [
                    (date.fromisoformat(row['date']), row['value'])
                    for _, row in actions_df.iterrows()
                    if row['type'] == 'dividend'
                ]
                if dividends:
                    await db.upsert_dividends_batch(symbol, dividends)
    finally:
        await db.close()
    
    logger.info(f"US sync complete: {results['synced']} symbols, {results['total_rows']} rows")
    return results


def sync_us(universe: str = "sample"):
    """Sync wrapper for sync_us_async."""
    return asyncio.get_event_loop().run_until_complete(sync_us_async(universe))


def sync_fx():
    """Sync FX rates to Neon."""
    try:
        from datahub_fx.jobs.sync import sync_all as fx_sync_all
    except ImportError as e:
        logger.error(f"Cannot import datahub_fx: {e}")
        return {"status": "error", "error": str(e)}
    
    logger.info("=== FX Rates Sync ===")
    return fx_sync_all()


async def export_to_csv_async():
    """Export Neon data to local CSV cache."""
    try:
        from datahub_us.db import Database
        from datahub_us.config import OHLCV_DIR
    except ImportError as e:
        logger.error(f"Cannot import datahub_us: {e}")
        return {"status": "error", "error": str(e)}
    
    logger.info("=== Export Neon → CSV ===")
    
    db = await Database.connect()
    
    try:
        stats = await db.get_stats()
        logger.info(f"  Database: {stats['symbols']} symbols, {stats['total_bars']} bars")
        
        result = await db.export_all_to_csv()
        logger.info(f"  Exported to {result['output_dir']}")
        return result
    finally:
        await db.close()


def export_to_csv():
    """Export wrapper."""
    return asyncio.get_event_loop().run_until_complete(export_to_csv_async())


def main():
    parser = argparse.ArgumentParser(description="Unified data sync")
    parser.add_argument("--b3", action="store_true", help="Sync B3 indices")
    parser.add_argument("--us", action="store_true", help="Sync US stocks")
    parser.add_argument("--fx", action="store_true", help="Sync FX rates")
    parser.add_argument("--export", action="store_true", help="Export Neon to CSV")
    parser.add_argument("--universe", default="sample", help="US universe: sample, sp500")
    parser.add_argument("--all", action="store_true", help="Sync everything")
    
    args = parser.parse_args()
    
    # If no specific option, default to --all
    if not any([args.b3, args.us, args.fx, args.export, args.all]):
        args.all = True
    
    results = {}
    
    if args.all or args.b3:
        results["b3"] = sync_b3()
    
    if args.all or args.us:
        results["us"] = sync_us(args.universe)
    
    if args.all or args.fx:
        results["fx"] = sync_fx()
    
    if args.all or args.export:
        results["export"] = export_to_csv()
    
    # Summary
    print("\n" + "="*50)
    print("SYNC SUMMARY")
    print("="*50)
    for key, value in results.items():
        status = value.get("status", "complete") if isinstance(value, dict) else "complete"
        print(f"  {key}: {status}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
