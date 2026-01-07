"""Command-line interface for FX data pipeline."""

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

from .config import FxConfig
from .jobs import sync_all, sync_pair, update_all, update_pair
from .storage import CsvFxStorage


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_sync(args):
    """Handle sync command."""
    config = FxConfig(cache_dir=Path(args.cache_dir))
    
    if args.pair:
        result = sync_pair(args.pair, config)
        print(json.dumps(result, indent=2))
    else:
        results = sync_all(config=config)
        print(json.dumps(results, indent=2))


def cmd_update(args):
    """Handle update command."""
    config = FxConfig(cache_dir=Path(args.cache_dir))
    
    if args.pair:
        result = update_pair(args.pair, config)
        print(json.dumps(result, indent=2))
    else:
        results = update_all(config=config)
        print(json.dumps(results, indent=2))


def cmd_status(args):
    """Handle status command."""
    config = FxConfig(cache_dir=Path(args.cache_dir))
    storage = CsvFxStorage(config.cache_dir)
    
    status = storage.get_status()
    
    if not status:
        print("No FX data found. Run 'sync' to fetch data.")
        return
    
    print(f"\nFX Data Status ({config.cache_dir})")
    print("=" * 60)
    
    for pair, info in sorted(status.items()):
        print(f"\n{pair}:")
        print(f"  Records: {info['count']:,}")
        print(f"  Range:   {info['first_date']} to {info['last_date']}")
        print(f"  Sources: {', '.join(info['sources'])}")
    
    print()


def cmd_show(args):
    """Handle show command - display rates for a pair."""
    config = FxConfig(cache_dir=Path(args.cache_dir))
    storage = CsvFxStorage(config.cache_dir)
    
    records = storage.load(args.pair)
    
    if not records:
        print(f"No data for {args.pair}")
        return
    
    # Apply limit
    if args.tail:
        records = records[-args.tail:]
    elif args.head:
        records = records[:args.head]
    
    print(f"\n{args.pair} Rates")
    print("-" * 40)
    for record in records:
        print(f"{record.date}  {record.rate:>12.6f}  {record.source}")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="FX Rate Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--cache-dir",
        default="cache/fx",
        help="Cache directory for FX data (default: cache/fx)",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Sync command
    sync_parser = subparsers.add_parser(
        "sync",
        help="Full sync from inception (overwrites existing data)",
    )
    sync_parser.add_argument(
        "--pair",
        help="Specific pair to sync (default: all)",
    )
    sync_parser.set_defaults(func=cmd_sync)
    
    # Update command
    update_parser = subparsers.add_parser(
        "update",
        help="Incremental update (fetch new data only)",
    )
    update_parser.add_argument(
        "--pair",
        help="Specific pair to update (default: all)",
    )
    update_parser.set_defaults(func=cmd_update)
    
    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show data status",
    )
    status_parser.set_defaults(func=cmd_status)
    
    # Show command
    show_parser = subparsers.add_parser(
        "show",
        help="Show rates for a pair",
    )
    show_parser.add_argument(
        "pair",
        help="Currency pair to show (e.g., USD/BRL)",
    )
    show_parser.add_argument(
        "--head",
        type=int,
        help="Show first N records",
    )
    show_parser.add_argument(
        "--tail",
        type=int,
        default=20,
        help="Show last N records (default: 20)",
    )
    show_parser.set_defaults(func=cmd_show)
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
    except Exception as e:
        logging.exception("Command failed")
        sys.exit(1)


if __name__ == "__main__":
    main()






























