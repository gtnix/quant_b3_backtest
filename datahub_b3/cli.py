#!/usr/bin/env python3
"""CLI para DataHub B3."""
import argparse
import logging
import sys

from .config import B3_INDICES
from .scraper import fetch_index, fetch_all_indices
from .db import get_connection, ensure_tables_exist, upsert_index, get_index_symbols
from .intraday import sync_intraday, sync_daily, get_symbols_from_indices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def cmd_sync(args):
    """Sincroniza índices da B3 para o Neon."""
    indices = args.indices if args.indices else list(B3_INDICES.keys())
    
    logger.info(f"Syncing {len(indices)} indices: {', '.join(indices)}")
    
    with get_connection() as conn:
        ensure_tables_exist(conn)
        
        for code in indices:
            data = fetch_index(code)
            if data:
                count = upsert_index(conn, data)
                logger.info(f"✓ {code}: {count} components saved (date: {data.date})")
            else:
                logger.warning(f"✗ {code}: no data available")
    
    logger.info("Sync complete!")


def cmd_list(args):
    """Lista índices disponíveis."""
    print("\nÍndices B3 disponíveis:\n")
    for code, meta in B3_INDICES.items():
        print(f"  {code:10} - {meta['name']}")
    print()


def cmd_show(args):
    """Mostra composição de um índice."""
    with get_connection() as conn:
        symbols = get_index_symbols(conn, args.index)
        
        if not symbols:
            print(f"Nenhum dado para {args.index}")
            return
        
        print(f"\n{args.index} - {len(symbols)} componentes:\n")
        for i, sym in enumerate(symbols, 1):
            print(f"  {i:3}. {sym}")
        print()


def _progress_bar(current: int, total: int, symbols: str, bars: int):
    """Print progress bar."""
    pct = current / total if total > 0 else 0
    filled = int(pct * 30)
    bar = "=" * filled + ">" + " " * (29 - filled)
    sys.stdout.write(f"\r[{bar}] {pct*100:5.1f}% | {symbols:20} | {current}/{total} | {bars:,} bars")
    sys.stdout.flush()
    if current >= total:
        print()


def cmd_intraday_sync(args):
    """Sincroniza dados intraday (30m) de todos os símbolos dos índices."""
    logger.info("Starting intraday sync...")
    
    result = sync_intraday(
        interval=args.interval,
        range_param=args.range,
        on_progress=_progress_bar
    )
    
    print(f"\n✓ Intraday sync complete!")
    print(f"  Symbols: {result.symbols_success}/{result.symbols_total}")
    print(f"  Bars inserted: {result.bars_inserted:,}")
    print(f"  Duration: {result.duration_secs:.1f}s")
    
    if result.errors:
        print(f"  Failed: {len(result.errors)} symbols")


def cmd_daily_sync(args):
    """Sincroniza dados diários de todos os símbolos dos índices."""
    logger.info("Starting daily sync...")
    
    result = sync_daily(
        range_param=args.range,
        on_progress=_progress_bar
    )
    
    print(f"\n✓ Daily sync complete!")
    print(f"  Symbols: {result.symbols_success}/{result.symbols_total}")
    print(f"  Bars inserted: {result.bars_inserted:,}")
    print(f"  Duration: {result.duration_secs:.1f}s")
    
    if result.errors:
        print(f"  Failed: {len(result.errors)} symbols")


def cmd_full_sync(args):
    """Sincroniza índices + dados diários + intraday."""
    # 1. Sync indices
    logger.info("Step 1/3: Syncing B3 indices...")
    cmd_sync(args)
    
    # 2. Daily data
    logger.info("\nStep 2/3: Syncing daily OHLCV...")
    result_daily = sync_daily(range_param="1mo", on_progress=_progress_bar)
    print(f"  Daily: {result_daily.bars_inserted:,} bars")
    
    # 3. Intraday data
    logger.info("\nStep 3/3: Syncing intraday 30m...")
    result_intraday = sync_intraday(interval="30m", range_param="5d", on_progress=_progress_bar)
    print(f"  Intraday: {result_intraday.bars_inserted:,} bars")
    
    print("\n✓ Full sync complete!")


def main():
    parser = argparse.ArgumentParser(description="DataHub B3 - Índices brasileiros")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # sync
    p_sync = subparsers.add_parser("sync", help="Sincroniza índices da B3")
    p_sync.add_argument("indices", nargs="*", help="Códigos dos índices (default: todos)")
    p_sync.set_defaults(func=cmd_sync)
    
    # list
    p_list = subparsers.add_parser("list", help="Lista índices disponíveis")
    p_list.set_defaults(func=cmd_list)
    
    # show
    p_show = subparsers.add_parser("show", help="Mostra composição de um índice")
    p_show.add_argument("index", help="Código do índice")
    p_show.set_defaults(func=cmd_show)
    
    # intraday-sync
    p_intraday = subparsers.add_parser("intraday-sync", help="Sync intraday 30m data")
    p_intraday.add_argument("--interval", default="30m", help="Candle interval (default: 30m)")
    p_intraday.add_argument("--range", default="5d", help="Date range (default: 5d)")
    p_intraday.set_defaults(func=cmd_intraday_sync)
    
    # daily-sync
    p_daily = subparsers.add_parser("daily-sync", help="Sync daily OHLCV data")
    p_daily.add_argument("--range", default="1mo", help="Date range (default: 1mo)")
    p_daily.set_defaults(func=cmd_daily_sync)
    
    # full-sync
    p_full = subparsers.add_parser("full-sync", help="Sync indices + daily + intraday")
    p_full.add_argument("indices", nargs="*", help="Códigos dos índices (default: todos)")
    p_full.set_defaults(func=cmd_full_sync)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

