#!/usr/bin/env python3
"""DataHub US CLI - US Market Data Pipeline with Neon Persistence.

Usage:
    python -m datahub_us.cli bootstrap --sample    # Test with 10 symbols
    python -m datahub_us.cli bootstrap             # Full S&P 500
    python -m datahub_us.cli update
    python -m datahub_us.cli sync                  # Neon → CSV cache
    python -m datahub_us.cli db-status
    python -m datahub_us.cli status
"""

import asyncio
import logging
import sys
from datetime import date
from typing import Optional

try:
    import typer
except ImportError:
    print("typer is required. Install with: pip install typer")
    sys.exit(1)

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .providers.yfinance_provider import YFinanceProvider
from .storage import CSVStorage
from .jobs import bootstrap_20y, update_daily, repair_gaps, sync_cache
from .db import Database
from .universe import get_universe, get_sample_symbols
from .reports.generator import ReportGenerator
from .config import BOOTSTRAP_START, BOOTSTRAP_END
from .indices import fetch_index, US_INDEX_FETCHERS
from .indices_db import get_connection as get_idx_conn, ensure_tables_exist as ensure_idx_tables, upsert_index, get_index_symbols, get_all_indices
from .intraday import sync_intraday_us, sync_daily_us, sync_aggregate_us, AGGREGATE_INTERVALS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="DataHub US - US Market Data Pipeline with Neon DB")
console = Console()


@app.command()
def bootstrap(
    sample: bool = typer.Option(False, "--sample", "-s", help="Use sample of 10 symbols"),
    sample20: bool = typer.Option(False, "--sample20", help="Use sample of 20 symbols"),
    symbols: Optional[str] = typer.Option(None, "--symbols", help="Comma-separated symbols"),
    start: Optional[str] = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)"),
    end: Optional[str] = typer.Option(None, "--end", help="End date (YYYY-MM-DD)"),
    no_validate: bool = typer.Option(False, "--no-validate", help="Skip validation"),
    no_csv: bool = typer.Option(False, "--no-csv", help="Skip CSV export"),
):
    """Bootstrap 20 years of historical data (yfinance → Neon → CSV)."""
    console.print("\n[bold blue]DataHub US - Bootstrap (Neon Persistence)[/bold blue]\n")
    
    # Determine universe
    universe = "sp500"
    symbol_list = None
    
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        console.print(f"Symbols: {', '.join(symbol_list[:10])}{'...' if len(symbol_list) > 10 else ''}")
    elif sample:
        universe = "sample"
        symbol_list = get_sample_symbols(10)
        console.print(f"Using sample: {', '.join(symbol_list)}")
    elif sample20:
        universe = "sample20"
        symbol_list = get_sample_symbols(20)
        console.print(f"Using sample20: {', '.join(symbol_list)}")
    else:
        symbol_list = get_universe("sp500")
        console.print(f"Using full S&P 500: {len(symbol_list)} symbols")
    
    # Parse dates
    start_date = date.fromisoformat(start) if start else BOOTSTRAP_START
    end_date = date.fromisoformat(end) if end else BOOTSTRAP_END
    
    console.print(f"Date range: {start_date} to {end_date}")
    console.print(f"Persist to: Neon PostgreSQL + CSV cache")
    console.print("")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading...", total=len(symbol_list))
        
        def on_progress(current, total, symbol):
            progress.update(task, completed=current, description=f"[cyan]{symbol}[/cyan]")
        
        result = bootstrap_20y(
            symbols=symbol_list,
            start=start_date,
            end=end_date,
            validate=not no_validate,
            export_csv=not no_csv,
            on_progress=on_progress,
        )
    
    # Print summary
    console.print("\n[bold green]Bootstrap Complete[/bold green]\n")
    
    table = Table(title="Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Symbols Total", str(result.symbols_total))
    table.add_row("Symbols Success", f"[green]{result.symbols_success}[/green]")
    table.add_row("Symbols Failed", f"[red]{result.symbols_failed}[/red]" if result.symbols_failed else "0")
    table.add_row("Total Rows Fetched", f"{result.total_rows:,}")
    table.add_row("Rows to Neon DB", f"[blue]{result.rows_to_db:,}[/blue]")
    table.add_row("Success Rate", f"{result.success_rate:.1f}%")
    table.add_row("Duration", f"{result.duration_seconds:.1f}s")
    
    console.print(table)
    
    if result.errors:
        console.print(f"\n[yellow]Errors ({len(result.errors)}):[/yellow]")
        for err in result.errors[:5]:
            console.print(f"  - {err['symbol']}: {err['error']}")
        if len(result.errors) > 5:
            console.print(f"  ... and {len(result.errors) - 5} more")


@app.command()
def update(
    symbols: Optional[str] = typer.Option(None, "--symbols", help="Comma-separated symbols"),
    force_days: int = typer.Option(0, "--force-days", help="Force re-fetch last N days"),
    no_validate: bool = typer.Option(False, "--no-validate", help="Skip validation"),
    no_csv: bool = typer.Option(False, "--no-csv", help="Skip CSV export"),
):
    """Update data incrementally since last date (yfinance → Neon → CSV)."""
    console.print("\n[bold blue]DataHub US - Update[/bold blue]\n")
    
    symbol_list = None
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Updating...", total=100)
        
        def on_progress(current, total, symbol):
            progress.update(task, completed=current, total=total, description=f"[cyan]{symbol}[/cyan]")
        
        result = update_daily(
            symbols=symbol_list,
            force_days=force_days,
            validate=not no_validate,
            export_csv=not no_csv,
            on_progress=on_progress,
        )
    
    console.print("\n[bold green]Update Complete[/bold green]\n")
    
    table = Table(title="Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Symbols Total", str(result.symbols_total))
    table.add_row("Updated", f"[green]{result.symbols_updated}[/green]")
    table.add_row("Skipped", str(result.symbols_skipped))
    table.add_row("Failed", f"[red]{result.symbols_failed}[/red]" if result.symbols_failed else "0")
    table.add_row("New Rows", f"+{result.new_rows:,}")
    table.add_row("Rows to Neon DB", f"[blue]+{result.rows_to_db:,}[/blue]")
    table.add_row("Duration", f"{result.duration_seconds:.1f}s")
    
    console.print(table)


@app.command()
def sync(
    symbols: Optional[str] = typer.Option(None, "--symbols", help="Comma-separated symbols"),
):
    """Sync data from Neon database to local CSV cache."""
    console.print("\n[bold blue]DataHub US - Sync (Neon → CSV)[/bold blue]\n")
    
    symbol_list = None
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Syncing...", total=100)
        
        def on_progress(current, total, symbol):
            progress.update(task, completed=current, total=total, description=f"[cyan]{symbol}[/cyan]")
        
        result = sync_cache(symbols=symbol_list, on_progress=on_progress)
    
    console.print("\n[bold green]Sync Complete[/bold green]\n")
    
    table = Table(title="Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Symbols", str(result.symbols_synced))
    table.add_row("Total Rows", f"{result.total_rows:,}")
    table.add_row("Output Dir", result.output_dir)
    table.add_row("Duration", f"{result.duration_seconds:.1f}s")
    
    console.print(table)


@app.command("db-status")
def db_status():
    """Show Neon database status and statistics."""
    console.print("\n[bold blue]DataHub US - Database Status[/bold blue]\n")
    
    async def _get_stats():
        db = await Database.connect()
        try:
            stats = await db.get_stats()
            symbols = await db.get_symbols()
            return stats, symbols
        finally:
            await db.close()
    
    try:
        stats, symbols = asyncio.get_event_loop().run_until_complete(_get_stats())
    except Exception as e:
        console.print(f"[red]Database connection failed: {e}[/red]")
        return
    
    table = Table(title="Neon Database Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Symbols", str(stats['symbols']))
    table.add_row("Total Bars", f"{stats['total_bars']:,}")
    table.add_row("Start Date", stats['start_date'] or "N/A")
    table.add_row("End Date", stats['end_date'] or "N/A")
    
    console.print(table)
    
    if symbols:
        console.print(f"\n[bold]Symbols ({len(symbols)}):[/bold]")
        # Show first 20
        for i in range(0, min(20, len(symbols)), 5):
            row = ", ".join(symbols[i:i+5])
            console.print(f"  {row}")
        if len(symbols) > 20:
            console.print(f"  ... and {len(symbols) - 20} more")


@app.command()
def repair(
    symbols: Optional[str] = typer.Option(None, "--symbols", help="Comma-separated symbols"),
    max_gap_days: int = typer.Option(5, "--max-gap", help="Max gap days threshold"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only detect gaps, don't repair"),
):
    """Detect and repair gaps in stored data."""
    console.print("\n[bold blue]DataHub US - Gap Repair[/bold blue]\n")
    
    if dry_run:
        console.print("[yellow]DRY RUN - no data will be modified[/yellow]\n")
    
    symbol_list = None
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
    
    storage = CSVStorage()
    stored_symbols = storage.list_symbols()
    
    if not stored_symbols:
        console.print("[yellow]No symbols in storage. Run bootstrap first.[/yellow]")
        return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        total = len(symbol_list) if symbol_list else len(stored_symbols)
        task = progress.add_task("Scanning...", total=total)
        
        def on_progress(current, total, symbol):
            progress.update(task, completed=current, description=f"[cyan]{symbol}[/cyan]")
        
        result = repair_gaps(
            symbols=symbol_list,
            max_gap_days=max_gap_days,
            dry_run=dry_run,
            on_progress=on_progress,
        )
    
    console.print("\n[bold green]Repair Complete[/bold green]\n")
    
    table = Table(title="Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Symbols Scanned", str(result.symbols_scanned))
    table.add_row("Gaps Detected", f"[yellow]{result.gaps_detected}[/yellow]" if result.gaps_detected else "0")
    table.add_row("Gaps Repaired", f"[green]{result.gaps_repaired}[/green]" if not dry_run else "N/A")
    table.add_row("Gaps Failed", f"[red]{result.gaps_failed}[/red]" if result.gaps_failed else "0")
    table.add_row("Rows Added", f"+{result.rows_added:,}" if not dry_run else "N/A")
    table.add_row("Duration", f"{result.duration_seconds:.1f}s")
    
    console.print(table)


@app.command()
def status():
    """Show local storage status and statistics."""
    console.print("\n[bold blue]DataHub US - Local Storage Status[/bold blue]\n")
    
    storage = CSVStorage()
    stats = storage.get_stats()
    
    table = Table(title="Local Cache Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Symbols", str(stats['symbols_count']))
    table.add_row("Total Bars", f"{stats['total_bars']:,}")
    table.add_row("Start Date", stats['start_date'] or "N/A")
    table.add_row("End Date", stats['end_date'] or "N/A")
    
    console.print(table)
    
    # Provider health
    console.print("\n[bold]Provider Health[/bold]")
    provider = YFinanceProvider()
    healthy = provider.healthcheck()
    
    if healthy:
        console.print("  yfinance: [green]OK[/green]")
    else:
        console.print("  yfinance: [red]FAILED[/red]")


@app.command()
def report():
    """Generate status report."""
    console.print("\n[bold blue]DataHub US - Report Generation[/bold blue]\n")
    
    generator = ReportGenerator()
    report_path = generator.generate()
    
    console.print(f"[green]Report generated:[/green] {report_path}")


@app.command("indices-sync")
def indices_sync(
    indices: Optional[str] = typer.Option(None, "--indices", "-i", help="Comma-separated index codes (SPX, NDX, DJI)"),
):
    """Sync US indices composition to Neon database."""
    console.print("\n[bold blue]DataHub US - Indices Sync[/bold blue]\n")
    
    index_list = [i.strip().upper() for i in indices.split(",")] if indices else list(US_INDEX_FETCHERS.keys())
    
    console.print(f"Syncing indices: {', '.join(index_list)}")
    
    with get_idx_conn() as conn:
        ensure_idx_tables(conn)
        
        for code in index_list:
            try:
                data = fetch_index(code)
                if data:
                    count = upsert_index(conn, data)
                    console.print(f"  [green]✓[/green] {code}: {count} components saved")
                else:
                    console.print(f"  [red]✗[/red] {code}: no data available")
            except Exception as e:
                console.print(f"  [red]✗[/red] {code}: {e}")
    
    console.print("\n[bold green]Indices sync complete![/bold green]")


@app.command("indices-list")
def indices_list():
    """List available US indices."""
    console.print("\n[bold blue]US Indices Available[/bold blue]\n")
    
    table = Table()
    table.add_column("Code", style="cyan")
    table.add_column("Name", style="white")
    
    for code, (name, _) in US_INDEX_FETCHERS.items():
        table.add_row(code, name)
    
    console.print(table)


@app.command("indices-show")
def indices_show(
    index: str = typer.Argument(..., help="Index code (SPX, NDX, DJI)"),
):
    """Show composition of a US index."""
    with get_idx_conn() as conn:
        symbols = get_index_symbols(conn, index.upper())
    
    if not symbols:
        console.print(f"[yellow]No data for {index}. Run 'indices-sync' first.[/yellow]")
        return
    
    console.print(f"\n[bold]{index.upper()} - {len(symbols)} components[/bold]\n")
    
    # Print in columns
    for i in range(0, len(symbols), 5):
        row = "  ".join(f"{s:6}" for s in symbols[i:i+5])
        console.print(f"  {row}")


@app.command("intraday-sync")
def intraday_sync(
    interval: str = typer.Option("30m", "--interval", "-i", help="Candle interval (e.g., 30m, 1h)"),
    period: str = typer.Option("5d", "--period", "-p", help="Date range (e.g., 5d, 1mo)"),
):
    """Sync intraday OHLCV data for all US index stocks."""
    console.print("\n[bold blue]DataHub US - Intraday Sync[/bold blue]\n")
    console.print(f"Interval: {interval} | Period: {period}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Syncing intraday...", total=100)
        
        def on_progress(current, total, symbol, bars):
            progress.update(task, completed=current, total=total, description=f"[cyan]{symbol}[/cyan]")
        
        result = sync_intraday_us(
            symbols=None,  # All from indices
            interval=interval,
            period=period,
            on_progress=on_progress,
        )
    
    console.print("\n[bold green]Intraday Sync Complete[/bold green]\n")
    
    table = Table(title="Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Symbols Total", str(result.symbols_total))
    table.add_row("Success", f"[green]{result.symbols_success}[/green]")
    table.add_row("Failed", f"[red]{result.symbols_failed}[/red]" if result.symbols_failed else "0")
    table.add_row("Bars Inserted", f"{result.bars_inserted:,}")
    table.add_row("Duration", f"{result.duration_secs:.1f}s")
    
    console.print(table)


@app.command("daily-sync")
def daily_sync(
    period: str = typer.Option("1mo", "--period", "-p", help="Date range (e.g., 1mo, 3mo, 1y)"),
):
    """Sync daily OHLCV data for all US index stocks."""
    console.print("\n[bold blue]DataHub US - Daily Sync[/bold blue]\n")
    console.print(f"Period: {period}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Syncing daily...", total=100)
        
        def on_progress(current, total, symbol, bars):
            progress.update(task, completed=current, total=total, description=f"[cyan]{symbol}[/cyan]")
        
        result = sync_daily_us(
            symbols=None,
            period=period,
            on_progress=on_progress,
        )
    
    console.print("\n[bold green]Daily Sync Complete[/bold green]\n")
    
    table = Table(title="Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Symbols Total", str(result.symbols_total))
    table.add_row("Success", f"[green]{result.symbols_success}[/green]")
    table.add_row("Failed", f"[red]{result.symbols_failed}[/red]" if result.symbols_failed else "0")
    table.add_row("Bars Inserted", f"{result.bars_inserted:,}")
    table.add_row("Duration", f"{result.duration_secs:.1f}s")
    
    console.print(table)


@app.command("aggregate")
def aggregate(
    interval: Optional[str] = typer.Option(None, "--interval", "-i", help="Single interval (default: all 30m,1h,1d)"),
    period: str = typer.Option("1mo", "--period", "-p", help="Date range (e.g., 5d, 1mo)"),
):
    """Aggregate intraday data for multiple intervals (30m, 1h, 1d)."""
    intervals = [interval] if interval else AGGREGATE_INTERVALS
    
    console.print("\n[bold blue]DataHub US - Aggregate Sync[/bold blue]\n")
    console.print(f"Intervals: {', '.join(intervals)} | Period: {period}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Aggregating...", total=100)
        
        def on_progress(intv, current, total, symbol, bars):
            progress.update(task, completed=current, total=total, description=f"[cyan]{intv}[/cyan] {symbol}")
        
        results = sync_aggregate_us(
            symbols=None,
            intervals=intervals,
            period=period,
            on_progress=on_progress,
        )
    
    console.print("\n[bold green]Aggregate Sync Complete[/bold green]\n")
    
    table = Table(title="Results by Interval")
    table.add_column("Interval", style="cyan")
    table.add_column("Success", style="green")
    table.add_column("Failed", style="red")
    table.add_column("Bars", style="white")
    table.add_column("Duration", style="dim")
    
    total_bars = 0
    total_duration = 0
    for intv, r in results.items():
        table.add_row(
            intv,
            str(r.symbols_success),
            str(r.symbols_failed) if r.symbols_failed else "-",
            f"{r.bars_inserted:,}",
            f"{r.duration_secs:.1f}s"
        )
        total_bars += r.bars_inserted
        total_duration += r.duration_secs
    
    table.add_row("", "", "", "", "", end_section=True)
    table.add_row("[bold]TOTAL[/bold]", "", "", f"[bold]{total_bars:,}[/bold]", f"[bold]{total_duration:.1f}s[/bold]")
    
    console.print(table)


@app.command("dividends-sync")
def dividends_sync(
    symbols: Optional[str] = typer.Option(None, "--symbols", "-s", help="Comma-separated symbols"),
    start_year: int = typer.Option(2010, "--start-year", help="Start year for dividend history"),
    skip_existing: bool = typer.Option(True, "--skip-existing/--force-all", help="Skip symbols already synced"),
):
    """Sync dividend history for US stocks from yfinance."""
    console.print("\n[bold blue]DataHub US - Dividends Sync[/bold blue]\n")
    
    async def _sync_dividends():
        db = await Database.connect()
        provider = YFinanceProvider()
        
        try:
            # Get symbols from database or argument
            if symbols:
                symbol_list = [s.strip().upper() for s in symbols.split(",")]
            else:
                symbol_list = await db.get_symbols()
            
            total_symbols = len(symbol_list)
            
            # Skip already synced symbols
            skipped = 0
            if skip_existing:
                synced = await db.get_synced_dividend_symbols()
                skipped = len([s for s in symbol_list if s in synced])
                symbol_list = [s for s in symbol_list if s not in synced]
                if skipped > 0:
                    console.print(f"[yellow]Skipping {skipped} already synced symbols[/yellow]")
            
            console.print(f"Processing {len(symbol_list)} symbols (of {total_symbols} total)")
            console.print(f"Date range: {start_year}-01-01 to today\n")
            
            from datetime import date
            start_date = date(start_year, 1, 1)
            end_date = date.today()
            
            total_dividends = 0
            symbols_with_divs = 0
            errors = []
            
            for i, symbol in enumerate(symbol_list, 1):
                try:
                    # Fetch dividends from yfinance
                    actions_df = provider.fetch_actions(symbol, start_date, end_date)
                    
                    if actions_df.empty:
                        continue
                    
                    # Filter only dividends
                    div_rows = actions_df[actions_df['type'] == 'dividend']
                    
                    if div_rows.empty:
                        continue
                    
                    # Convert to list of tuples
                    dividends = []
                    for _, row in div_rows.iterrows():
                        ex_date = date.fromisoformat(row['date'])
                        rate = float(row['value'])
                        dividends.append((ex_date, rate))
                    
                    # Insert to database
                    inserted = await db.upsert_dividends_batch(symbol, dividends, "DIVIDEND")
                    
                    if inserted > 0:
                        total_dividends += inserted
                        symbols_with_divs += 1
                        console.print(f"  [{i}/{len(symbol_list)}] {symbol}: +{inserted} dividends")
                    
                except Exception as e:
                    errors.append({"symbol": symbol, "error": str(e)})
                    if len(errors) <= 3:
                        console.print(f"  [{i}/{len(symbol_list)}] {symbol}: [red]ERROR[/red] {e}")
            
            return {
                "symbols_total": len(symbol_list),
                "symbols_with_dividends": symbols_with_divs,
                "total_dividends": total_dividends,
                "errors": errors,
            }
        finally:
            await db.close()
    
    result = asyncio.get_event_loop().run_until_complete(_sync_dividends())
    
    console.print("\n[bold green]Dividends Sync Complete[/bold green]\n")
    
    table = Table(title="Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Symbols Processed", str(result["symbols_total"]))
    table.add_row("Symbols with Dividends", f"[green]{result['symbols_with_dividends']}[/green]")
    table.add_row("Total Dividends Inserted", f"[blue]{result['total_dividends']:,}[/blue]")
    table.add_row("Errors", f"[red]{len(result['errors'])}[/red]" if result["errors"] else "0")
    
    console.print(table)


@app.command()
def healthcheck():
    """Check provider and database connectivity."""
    console.print("\n[bold blue]DataHub US - Healthcheck[/bold blue]\n")
    
    # Provider
    console.print("Testing yfinance...")
    provider = YFinanceProvider()
    provider_ok = provider.healthcheck()
    
    if provider_ok:
        console.print("  yfinance: [green]OK[/green]")
    else:
        console.print("  yfinance: [red]FAILED[/red]")
    
    # Database
    console.print("\nTesting Neon database...")
    
    async def _test_db():
        try:
            db = await Database.connect()
            await db.ensure_schema()
            await db.close()
            return True
        except Exception as e:
            logger.error(f"DB error: {e}")
            return False
    
    db_ok = asyncio.get_event_loop().run_until_complete(_test_db())
    
    if db_ok:
        console.print("  Neon DB: [green]OK[/green]")
    else:
        console.print("  Neon DB: [red]FAILED[/red]")
    
    # Summary
    all_ok = provider_ok and db_ok
    console.print(f"\n[bold]Overall: {'[green]HEALTHY[/green]' if all_ok else '[red]UNHEALTHY[/red]'}[/bold]")


def main():
    app()


if __name__ == "__main__":
    main()
