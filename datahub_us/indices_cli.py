#!/usr/bin/env python3
"""CLI for US Indices sync."""
import argparse
import logging
import sys

from .indices import fetch_index, fetch_all_indices, US_INDEX_FETCHERS
from .indices_db import get_connection, ensure_tables_exist, upsert_index, get_index_symbols, get_all_indices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def cmd_sync(args):
    """Sync US indices to Neon."""
    indices = args.indices if args.indices else list(US_INDEX_FETCHERS.keys())
    
    logger.info(f"Syncing {len(indices)} US indices: {', '.join(indices)}")
    
    with get_connection() as conn:
        ensure_tables_exist(conn)
        
        for code in indices:
            try:
                data = fetch_index(code)
                if data:
                    count = upsert_index(conn, data)
                    logger.info(f"✓ {code}: {count} components saved (date: {data.date})")
                else:
                    logger.warning(f"✗ {code}: no data available")
            except Exception as e:
                logger.error(f"✗ {code}: {e}")
    
    logger.info("Sync complete!")


def cmd_list(args):
    """List available US indices."""
    print("\nUS Indices available:\n")
    for code, (name, _) in US_INDEX_FETCHERS.items():
        print(f"  {code:10} - {name}")
    print()


def cmd_show(args):
    """Show composition of a US index."""
    with get_connection() as conn:
        symbols = get_index_symbols(conn, args.index)
        
        if not symbols:
            print(f"No data for {args.index}")
            return
        
        print(f"\n{args.index} - {len(symbols)} components:\n")
        for i, sym in enumerate(symbols, 1):
            print(f"  {i:3}. {sym}")
        print()


def cmd_status(args):
    """Show status of all US indices in database."""
    with get_connection() as conn:
        indices = get_all_indices(conn)
        
        if not indices:
            print("No US indices in database yet.")
            return
        
        print("\nUS Indices in database:\n")
        print(f"{'Code':<10} {'Name':<30} {'Components':>12} {'Last Updated'}")
        print("-" * 70)
        for idx in indices:
            updated = idx['updated'].strftime("%Y-%m-%d %H:%M") if idx['updated'] else "N/A"
            print(f"{idx['code']:<10} {idx['name']:<30} {idx['components']:>12} {updated}")
        print()


def main():
    parser = argparse.ArgumentParser(description="US Indices Sync")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # sync
    p_sync = subparsers.add_parser("sync", help="Sync US indices to Neon")
    p_sync.add_argument("indices", nargs="*", help="Index codes (default: all)")
    p_sync.set_defaults(func=cmd_sync)
    
    # list
    p_list = subparsers.add_parser("list", help="List available indices")
    p_list.set_defaults(func=cmd_list)
    
    # show
    p_show = subparsers.add_parser("show", help="Show index composition")
    p_show.add_argument("index", help="Index code")
    p_show.set_defaults(func=cmd_show)
    
    # status
    p_status = subparsers.add_parser("status", help="Show database status")
    p_status.set_defaults(func=cmd_status)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()













