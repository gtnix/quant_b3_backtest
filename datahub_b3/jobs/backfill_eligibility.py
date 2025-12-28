#!/usr/bin/env python3
"""
V2 Eligibility Backfill Job.

Populates listing_date/delisting_date in provider_universe from cache/universe.csv.
This provides V2 eligibility data derived from V1 min_date/max_date.

Usage:
    python -m datahub_b3.jobs.backfill_eligibility
    
    # Or with explicit paths
    python -m datahub_b3.jobs.backfill_eligibility --csv-path cache/universe.csv
"""

import asyncio
import csv
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import asyncpg

from datahub_b3.config import Settings


async def backfill_from_csv(
    db: asyncpg.Pool,
    csv_path: Path,
    source: str = "DATA_DERIVED",
) -> dict:
    """
    Populate listing_date/delisting_date from universe.csv.
    
    Args:
        db: Database connection pool
        csv_path: Path to cache/universe.csv
        source: Eligibility source tag (DATA_DERIVED, PROVIDER_API, MANUAL)
    
    Returns:
        dict with counts: updated, skipped, errors
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    stats = {"updated": 0, "skipped": 0, "errors": 0, "total": 0}
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            stats["total"] += 1
            symbol = row.get("symbol", "").strip().upper()
            min_date_str = row.get("min_date", "").strip()
            max_date_str = row.get("max_date", "").strip()
            
            if not symbol or not min_date_str or not max_date_str:
                stats["skipped"] += 1
                continue
            
            try:
                listing_date = datetime.strptime(min_date_str, "%Y-%m-%d").date()
                max_date = datetime.strptime(max_date_str, "%Y-%m-%d").date()
                
                # Check ticker status to determine if it should have delisting_date
                status_row = await db.fetchrow(
                    "SELECT status FROM provider_universe WHERE ticker = $1",
                    symbol,
                )
                
                if status_row is None:
                    stats["skipped"] += 1
                    continue
                
                status = status_row["status"]
                
                # Only set delisting_date if ticker is INACTIVE
                delisting_date: Optional[date] = None
                if status == "INACTIVE":
                    delisting_date = max_date
                
                # Update only if listing_date is NULL (don't overwrite existing data)
                result = await db.execute(
                    """
                    UPDATE provider_universe 
                    SET listing_date = COALESCE(listing_date, $2),
                        delisting_date = COALESCE(delisting_date, $3),
                        eligibility_source = COALESCE(eligibility_source, $4)
                    WHERE ticker = $1 AND listing_date IS NULL
                    """,
                    symbol,
                    listing_date,
                    delisting_date,
                    source,
                )
                
                # Check if row was updated
                if "UPDATE 1" in result:
                    stats["updated"] += 1
                else:
                    stats["skipped"] += 1
                    
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                stats["errors"] += 1
    
    return stats


async def get_eligibility_stats(db: asyncpg.Pool) -> dict:
    """Get current eligibility data coverage statistics."""
    row = await db.fetchrow(
        """
        SELECT 
            COUNT(*) FILTER (WHERE listing_date IS NOT NULL) as with_listing,
            COUNT(*) FILTER (WHERE delisting_date IS NOT NULL) as with_delisting,
            COUNT(*) FILTER (WHERE eligibility_source = 'DATA_DERIVED') as data_derived,
            COUNT(*) FILTER (WHERE eligibility_source = 'PROVIDER_API') as provider_api,
            COUNT(*) FILTER (WHERE eligibility_source = 'MANUAL') as manual,
            COUNT(*) as total
        FROM provider_universe
        """
    )
    
    return {
        "with_listing_date": row["with_listing"],
        "with_delisting_date": row["with_delisting"],
        "data_derived": row["data_derived"],
        "provider_api": row["provider_api"],
        "manual": row["manual"],
        "total": row["total"],
    }


async def main():
    """Run the backfill job."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Backfill eligibility dates from CSV")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("cache/universe.csv"),
        help="Path to universe.csv",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making changes",
    )
    args = parser.parse_args()
    
    settings = Settings()
    
    print(f"Connecting to database...")
    db = await asyncpg.create_pool(settings.database_url)
    
    try:
        # Show stats before
        print("\n=== Before Backfill ===")
        stats_before = await get_eligibility_stats(db)
        for k, v in stats_before.items():
            print(f"  {k}: {v}")
        
        if args.dry_run:
            print("\n[DRY RUN] Would backfill from:", args.csv_path)
            print("[DRY RUN] No changes made.")
            return
        
        # Run backfill
        print(f"\nBackfilling from: {args.csv_path}")
        result = await backfill_from_csv(db, args.csv_path)
        
        print("\n=== Backfill Result ===")
        for k, v in result.items():
            print(f"  {k}: {v}")
        
        # Show stats after
        print("\n=== After Backfill ===")
        stats_after = await get_eligibility_stats(db)
        for k, v in stats_after.items():
            print(f"  {k}: {v}")
        
        # Summary
        print("\n=== Summary ===")
        print(f"  New listing_date entries: {stats_after['with_listing_date'] - stats_before['with_listing_date']}")
        print(f"  Coverage: {stats_after['with_listing_date']}/{stats_after['total']} ({100*stats_after['with_listing_date']/max(stats_after['total'],1):.1f}%)")
        
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())

