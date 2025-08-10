#!/usr/bin/env python3
"""
Run FuzzyFajuto Strategy with Profile Support

Enhanced version that supports different parameter profiles and result tracking.
Use --profile to specify different parameter sets.

Examples:
    python run_fuzzy_fajuto.py --ticker ALPA4 --profile default
    python run_fuzzy_fajuto.py --ticker PETR4 --profile conservative --start-date 2025-06-01 --end-date 2025-07-31
    python run_fuzzy_fajuto.py --ticker VALE3 --start-date 2024-01-01 --end-date 2024-12-31
    python run_fuzzy_fajuto.py --ticker BBAS3 --config-file my_custom.yaml --start-date 2025-04-01
"""

import sys
import os
import logging
import argparse
from datetime import datetime, date
from pathlib import Path
import pandas as pd # Added missing import for pandas
import json
import threading
import queue
import atexit
import time

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    # Quiet: avoid noisy prints in performance runs
except ImportError:
    pass
except Exception as e:
    pass

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Add engine directory to path
engine_dir = os.path.join(current_dir, 'engine')
sys.path.insert(0, engine_dir)

# Add strategies directory to path
strategies_dir = os.path.join(current_dir, 'strategies')
sys.path.insert(0, strategies_dir)

# Setup logging (raise to ERROR to reduce I/O during performance runs)
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Force critical logs only across our modules for this run
logging.getLogger().setLevel(logging.ERROR)
for noisy in (
    'FuzzyFajutoStrategy',
    'engine.simulator',
    'engine.performance_metrics',
    'engine.portfolio',
    'engine.loader',
    'engine.result_manager',
    'engine.sgs_data_loader',
    'engine.brapi_provider',
    'engine.market_utils',
    'engine.loss_manager',
):
    try:
        logging.getLogger(noisy).setLevel(logging.ERROR)
    except Exception:
        pass

# Enable pandas Copy-on-Write to reduce unnecessary copies (no logic change)
try:
    pd.options.mode.copy_on_write = True
except Exception:
    pass

class AsyncJsonlLogger:
    """Minimal async JSONL writer with batching and background thread."""
    def __init__(self, base_dir: Path, batch_size: int = 256, flush_ms: int = 200):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = int(batch_size)
        self.flush_ms = int(flush_ms)
        self._q: 'queue.Queue[tuple[str,str]]' = queue.Queue(maxsize=10000)
        self._buffers: dict[str, list[str]] = {}
        self._files: dict[str, any] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="JsonlWriter", daemon=True)
        self._thread.start()

    def emit(self, stream: str, event: dict):
        try:
            if 'type' not in event:
                event['type'] = stream
            line = json.dumps(event, ensure_ascii=True, separators=(",", ":")) + "\n"
            self._q.put_nowait((stream, line))
        except Exception:
            pass

    def _run(self):
        last = time.time()
        while not self._stop.is_set() or not self._q.empty():
            try:
                stream, line = self._q.get(timeout=0.05)
                buf = self._buffers.setdefault(stream, [])
                buf.append(line)
                if len(buf) >= self.batch_size:
                    self._flush(stream)
            except queue.Empty:
                pass
            now = time.time()
            if (now - last) * 1000.0 >= self.flush_ms:
                for s in list(self._buffers.keys()):
                    if self._buffers[s]:
                        self._flush(s)
                last = now
        for s in list(self._buffers.keys()):
            if self._buffers[s]:
                self._flush(s)
        for f in self._files.values():
            try:
                f.flush(); f.close()
            except Exception:
                pass

    def _flush(self, stream: str):
        try:
            fh = self._files.get(stream)
            if fh is None:
                path = self.base_dir / f"{stream}.jsonl"
                fh = open(path, 'a', encoding='utf-8')
                self._files[stream] = fh
            buf = self._buffers.get(stream, [])
            if buf:
                fh.writelines(buf)
                fh.flush()
                self._buffers[stream] = []
        except Exception:
            self._buffers[stream] = []

    def shutdown(self):
        try:
            self._stop.set()
            self._thread.join(timeout=3.0)
        except Exception:
            pass


def generate_execution_table(strategy_summary: dict) -> str:
    """
    Generate a clean and modern execution summary table for terminal display.
    
    Args:
        strategy_summary: Dictionary containing execution summary data
        
    Returns:
        Formatted table string with consistent styling
    """
    if not strategy_summary:
        return "No execution data available."
    
    # Prepare data rows
    rows = []
    
    # Add daily executions (all order types - all execute once per day)
    daily_executions = strategy_summary.get('daily_executions', {})
    for attempt_type, metrics in daily_executions.items():
        rows.append({
            'type': attempt_type,
            'category': 'Daily',
            'attempts': metrics['attempts'],
            'successful': metrics['successful'],
            'failed': metrics['failed'],
            'fill_rate': metrics['fill_rate']
        })
    
    if not rows:
        return "No execution data available."
    
    # Calculate totals
    total_attempts = sum(row['attempts'] for row in rows)
    total_successful = sum(row['successful'] for row in rows)
    total_failed = sum(row['failed'] for row in rows)
    overall_fill_rate = total_successful / total_attempts if total_attempts > 0 else 0
    
    # Generate clean table
    table_lines = []
    
    # Header
    table_lines.append("=" * 80)
    table_lines.append("EXECUTION PERFORMANCE SUMMARY")
    table_lines.append("=" * 80)
    table_lines.append(f"{'Execution Type':<15} {'Category':<10} {'Attempts':>10} {'Successful':>10} {'Failed':>10} {'Fill Rate':>10}")
    table_lines.append("-" * 80)
    
    # Data rows
    for row in rows:
        # Create simple progress bar for fill rate
        fill_rate_pct = row['fill_rate'] * 100
        progress_bars = int(fill_rate_pct / 10)  # 10% per bar
        progress_bar = "[" + "█" * progress_bars + " " * (10 - progress_bars) + "]"
        
        table_lines.append(
            f"{row['type']:<15} {row['category']:<10} {row['attempts']:>10} {row['successful']:>10} "
            f"{row['failed']:>10} {row['fill_rate']:>9.1%} {progress_bar}"
        )
    
    # Totals row
    table_lines.append("-" * 80)
    
    # Create progress bar for overall fill rate
    overall_pct = overall_fill_rate * 100
    overall_bars = int(overall_pct / 10)
    overall_progress_bar = "[" + "█" * overall_bars + " " * (10 - overall_bars) + "]"
    
    table_lines.append(
        f"{'TOTAL':<15} {'ALL':<10} {total_attempts:>10} {total_successful:>10} "
        f"{total_failed:>10} {overall_fill_rate:>9.1%} {overall_progress_bar}"
    )
    
    # Footer
    table_lines.append("=" * 80)
    
    return "\n".join(table_lines)

def _load_portfolio_symbols() -> list:
    """Load symbols from portfolio.csv in project root or data/.

    Returns uppercase symbols list or empty list if not found/invalid.
    """
    candidates = [
        Path(__file__).parent / 'portfolio.csv',
        Path(__file__).parent / 'data' / 'portfolio.csv',
    ]
    for path in candidates:
        try:
            if path.exists():
                df = pd.read_csv(path)
                if 'symbol' in df.columns:
                    symbols = [str(s).strip().upper() for s in df['symbol'].dropna().tolist()]
                    symbols = [s for s in symbols if len(s) > 0]
                    return list(dict.fromkeys(symbols))  # de-duplicate preserving order
        except Exception as e:
            logger.warning(f"Failed to load portfolio from {path}: {e}")
    return []


def main():
    """Main function to run FuzzyFajuto strategy with profile support."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run FuzzyFajuto Strategy with Profile Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s --ticker ALPA4 --profile default
  %(prog)s --ticker PETR4 --profile conservative --start-date 2025-06-01 --end-date 2025-07-31
  %(prog)s --ticker VALE3 --start-date 2024-01-01 --end-date 2024-12-31
  %(prog)s --ticker BBAS3 --config-file my_custom_config.yaml --start-date 2025-04-01"""
    )
    
    parser.add_argument('--ticker', default='ALPA4', help='Ticker symbol (default: ALPA4)')
    parser.add_argument('--tickers', help='Comma-separated list of tickers for multi-asset backtest (e.g., BBAS3,PETR4)')
    parser.add_argument('--profile', default='default', help='Strategy profile (default: default)')
    parser.add_argument('--config-file', help='Custom config file (overrides profile)')
    parser.add_argument('--start-date', default=None, help='Start date (default from config)')
    parser.add_argument('--end-date', default=None, help='End date (default from config)')
    parser.add_argument('--save-results', action='store_true', default=None, help='Save results for comparison (default from config)')
    parser.add_argument('--show-comparison', action='store_true', help='Show comparison with previous runs')
    
    args = parser.parse_args()
    
    try:
        # Silence verbose data-source prints during multi-symbol runs
        try:
            import os as _os
            _os.environ.setdefault('DISABLE_DATA_SOURCE_REPORT', '1')
        except Exception:
            pass

        # Import required modules
        from engine.loader import DataLoader, HybridDataManager
        from engine.simulator import BacktestSimulator
        from engine.portfolio import EnhancedPortfolio
        from engine.base_strategy import StrategyConfig, StrategyContext
        from engine.market_utils import BrazilianMarketUtils
        from engine.result_manager import ResultManager
        from strategies import create_fuzzy_fajuto
        
        # High-level start
        logger.info("Starting FuzzyFajuto Strategy backtest")
        t0_total = time.perf_counter()
        # Initialize run-scoped event logger and meta
        from datetime import datetime as _dt
        run_ts = _dt.utcnow().strftime('%Y%m%d_%H%M%S')
        short_hash = abs(hash((str(args.profile), str(args.config_file), str(args.start_date), str(args.end_date)))) % 10_000_000
        run_id = f"{run_ts}_{short_hash}"
        base_log_dir = Path(os.environ.get('LOG_DIR', 'logs')) / run_id
        ev = AsyncJsonlLogger(base_log_dir, int(os.environ.get('LOG_BATCH_SIZE','256')), int(os.environ.get('LOG_FLUSH_MS','200')))
        try:
            import engine as _engine_pkg
            _engine_pkg.event_logger = ev
        except Exception:
            pass
        atexit.register(lambda: ev.shutdown())
        try:
            with open(base_log_dir / 'meta.json', 'w', encoding='utf-8') as _mf:
                json.dump({'run_id':run_id,'start_ts':int(time.time()),'args':vars(args)}, _mf, ensure_ascii=True, separators=(",", ":"))
        except Exception:
            pass
        
        # Load defaults from config
        import yaml as _yaml
        _cfg = {}
        try:
            with open('config/settings.yaml', 'r') as _f:
                _cfg = _yaml.safe_load(_f) or {}
        except Exception as _e:
            logger.warning(f"Could not load config/settings.yaml for defaults: {_e}")

        default_start = (_cfg.get('backtest', {}) or {}).get('default_start_date', '2025-01-01')
        default_end = (_cfg.get('backtest', {}) or {}).get('default_end_date', '2025-08-07')
        default_save = (_cfg.get('backtest', {}) or {}).get('save_results', True)

        # Resolve dates/flags from CLI or defaults
        start_date = args.start_date or default_start
        end_date = args.end_date or default_end
        args.save_results = default_save if args.save_results is None else args.save_results

        # Determine tickers universe
        # Enforce: always read from portfolio.csv only
        csv_symbols = _load_portfolio_symbols()
        if not csv_symbols:
            raise SystemExit("portfolio.csv não encontrado ou sem coluna 'symbol'. Coloque sua lista em data/portfolio.csv.")
        tickers = csv_symbols
        logger.info(f"Loaded portfolio from CSV: {','.join(tickers)}")

        # Prefilter symbols by hourly coverage to avoid mid-run cancellation
        # Uses the same thresholds as loader._assess_execution_coverage
        try:
            from engine.brapi_provider import BrapiProvider as _BrapiProvider
            import os as _os
            from datetime import datetime as _dt
            import pandas as _pd

            _bp = _BrapiProvider(api_token=_os.environ.get('BRAPI_API_TOKEN', ''))
            _start_dt = _dt.strptime(start_date, '%Y-%m-%d')
            _end_dt = _dt.strptime(end_date, '%Y-%m-%d')
            _total_calendar_days = (_end_dt - _start_dt).days + 1
            _expected_trading_days = max(_total_calendar_days * 5 // 7, 1)

            _kept = []
            _dropped = []
            for _sym in tickers:
                try:
                    _dfh = _bp.get_ohlc(_sym, '1h', _start_dt, _end_dt)
                    if _dfh is None or _dfh.empty:
                        _dropped.append((_sym, 'empty'))
                        continue
                    _unique_days = len(set(_dfh.index.date))
                    _coverage = _unique_days / _expected_trading_days
                    _bars_per_day = len(_dfh) / max(_unique_days, 1)
                    _sufficient = (_coverage >= 0.8 and _bars_per_day >= 4 and len(_dfh) >= 50)
                    if _sufficient:
                        _kept.append(_sym)
                    else:
                        _dropped.append((_sym, f"cov={_coverage:.1%}, bpd={_bars_per_day:.2f}, bars={len(_dfh)}"))
                except Exception as _e:
                    _dropped.append((_sym, f"error:{_e}"))

            if _kept:
                logger.info(f"Prefilter coverage: kept {len(_kept)}/{len(tickers)} symbols")
                if len(_dropped) > 0:
                    logger.info("Dropped (insufficient hourly coverage): " + ",".join([d[0] for d in _dropped[:20]]) + ("..." if len(_dropped) > 20 else ""))
                tickers = _kept
            else:
                logger.warning("No symbols met hourly coverage threshold; proceeding with original list (may cancel mid-run).")
        except Exception as _e:
            logger.warning(f"Coverage prefilter skipped due to error: {_e}")
        # start_date and end_date already resolved above
        profile = args.profile
        config_file = args.config_file
        
        logger.info(f"Tickers: {','.join(tickers)}")
        logger.info(f"Profile: {profile}")
        logger.info(f"Config file: {config_file or 'profile-based'}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Initialize shared managers
        logger.info("🚀 Initializing Hybrid Data Management System...")
        hybrid_data_manager = HybridDataManager(config_path="config/settings.yaml")
        data_loader = DataLoader(auto_download=True)
        
        # Check local data availability (for transparency)
        logger.info("Checking local data availability...")
        data_status = data_loader.check_all_data(tickers)
        
        # Suppress verbose data status table in performance mode; compact log only
        try:
            ticker_status = data_status['tickers']['summary']
            sgs_has = data_status['sgs']['has_data']
            ibov_has = data_status['ibov']['has_data']
            logger.info(
                "Local data: missing=%s, gaps=%s, missing_days=%s, SGS=%s, IBOV=%s",
                ticker_status.get('missing_tickers_count'),
                ticker_status.get('tickers_with_gaps_count'),
                ticker_status.get('total_missing_days'),
                sgs_has,
                ibov_has,
            )
        except Exception:
            pass
        
        # Per-date, multi-symbol execution: build combined data once, then run a single simulator
        logger.info("🔄 Building combined execution dataset for all symbols...")
        t0_data = time.perf_counter()
        # Progress indicator for simulation stage as well
        def _progress_sim(stage: str, i: int, total: int):
            pct = int(i * 100 / max(total, 1))
            bar = "#" * (pct // 4) + "-" * (25 - (pct // 4))
            sys.stdout.write(f"\r[{stage}] [{bar}] {pct}%")
            sys.stdout.flush()
        from concurrent.futures import ThreadPoolExecutor, as_completed
        frames = []
        failures = []
        def _load_exec(sym: str):
            try:
                res = hybrid_data_manager.initialize_backtest_data(
                    symbol=sym,
                    start_date=start_date,
                    end_date=end_date,
                    local_loader=data_loader
                )
                exec_df = res.get('execution_data')
                if exec_df is None or exec_df.empty:
                    logger.error(f"No execution data for {sym}")
                    return None
                df = exec_df.copy()
                if getattr(df.index, 'tz', None) is not None:
                    df.index = df.index.tz_localize(None)
                df['symbol'] = sym
                return df, res
            except Exception as e:
                logger.error(f"Error loading data for {sym}: {e}")
                return None

        # Lightweight progress indicator (single-line) for batch loading
        total_syms = len(tickers)
        loaded_count = 0
        def _print_progress(done: int, total: int, phase: str = "loading"):
            pct = int(done * 100 / max(total, 1))
            bar = "#" * (pct // 4) + "-" * (25 - (pct // 4))
            sys.stdout.write(f"\r[{phase}] [{bar}] {pct}%  symbols {done}/{total}")
            sys.stdout.flush()

        with ThreadPoolExecutor(max_workers=min(8, max(1, total_syms))) as ex:
            futs = {ex.submit(_load_exec, sym): sym for sym in tickers}
            for fut in as_completed(futs):
                out = fut.result()
                sym = futs[fut]
                if out is None:
                    failures.append(sym)
                else:
                    df, res = out
                    frames.append(df)
                loaded_count += 1
                _print_progress(loaded_count, total_syms, phase="data")
        sys.stdout.write("\n")
        data_load_secs = time.perf_counter() - t0_data

        if not frames:
            raise SystemExit("No execution data available for any symbols")

        # Use categorical dtype for symbol to reduce memory and speed groupbys
        combined = pd.concat(frames, axis=0, ignore_index=False)
        try:
            if 'symbol' in combined.columns:
                combined['symbol'] = combined['symbol'].astype('category')
        except Exception:
            pass
        combined = combined.sort_index(kind='mergesort')

        # Warmup-aware start adjustment (global)
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        try:
            has_pre_start = (combined.index < start_dt).any()
        except Exception:
            has_pre_start = False
        if not has_pre_start and len(combined) > 174:
            new_start_ts = combined.index[174]
            start_dt = datetime(new_start_ts.year, new_start_ts.month, new_start_ts.day)
            logger.warning(f"Adjusted global start date to {start_dt.date()} to satisfy warmup (174 bars)")

        mask = (combined.index >= start_dt) & (combined.index <= end_dt)
        filtered_data = combined.loc[mask]
        if filtered_data.empty:
            raise SystemExit(f"No combined data available in date range {start_date} to {end_date}")

        # Initialize single portfolio and strategy for the full universe
        portfolio = EnhancedPortfolio(config_path="config/settings.yaml")
        strategy_config = StrategyConfig(
            universe=tickers,
            warmup_bars=30,
            risk_tolerance=0.02,
            max_position_size=0.10,
            max_daily_loss=0.02,
            stop_loss_pct=0.05,
            take_profit_pct=0.10
        )
        market_utils = BrazilianMarketUtils()
        strategy_context = StrategyContext(
            data_portal=data_loader,
            portfolio=portfolio,
            broker=None,
            market_rules=market_utils,
            logger=logging.getLogger("FuzzyFajutoStrategy"),
            metadata={
                'strategy_config_path': "config/profiles/fuzzy_fajuto_default.yaml",
                'complete_data': combined,
                'hybrid_data_result': {'execution_data': combined},
            }
        )
        strategy_context.hybrid_data_manager = hybrid_data_manager

        strategy = create_fuzzy_fajuto(
            cfg=strategy_config,
            ctx=strategy_context,
            profile=profile,
            config_file=config_file
        )

        simulator = BacktestSimulator(
            strategy=strategy,
            start_date=start_dt.strftime('%Y-%m-%d'),
            end_date=end_date,
            config_path="config/settings.yaml"
        )
        # Run simulation with timing
        t0_sim = time.perf_counter()
        # Compact single-line indicator before sim
        sys.stdout.write(f"[simulate] days={len(set(filtered_data.index.date))} bars={len(filtered_data)} symbols={len(tickers)}...\r")
        sys.stdout.flush()
        results = simulator.run_simulation(filtered_data)
        sim_secs = time.perf_counter() - t0_sim
        sys.stdout.write("\n")

        # Optionally save portfolio-level result
        if args.save_results:
            try:
                result_manager = ResultManager()
                _run_id = result_manager.save_result(
                    strategy_name="fuzzy_fajuto",
                    profile=profile,
                    ticker=','.join(tickers),
                    start_date=start_date,
                    end_date=end_date,
                    results=results,
                    strategy_summary=strategy.get_performance_summary(),
                    config_file=config_file
                )
                print(f"\nResults saved for portfolio with ID: {_run_id}")
            except Exception:
                pass

        # Render single HTML report (no CSV files)
        try:
            # Ensure output directory and path are defined up-front
            Path('reports').mkdir(parents=True, exist_ok=True)
            html_path = Path('reports') / 'portfolio_execution_report.html'

            # Get execution history from strategy (prefer in-memory DF; fallback to list)
            aggregated_exec_history = []
            try:
                df_exec_all = getattr(strategy, 'latest_execution_history_df', None)
                if df_exec_all is None or (hasattr(df_exec_all, 'empty') and df_exec_all.empty):
                    # Fallback to raw list if present
                    raw_hist = getattr(strategy, 'execution_history', None)
                    if raw_hist:
                        try:
                            tmp_df = pd.DataFrame([rec for rec in raw_hist if rec.get('filled') is True])
                            if not tmp_df.empty:
                                df_exec_all = tmp_df
                        except Exception:
                            pass
                if df_exec_all is not None and not df_exec_all.empty:
                    if 'symbol' not in df_exec_all.columns:
                        df_exec_all = df_exec_all.copy()
                        df_exec_all['symbol'] = df_exec_all.get('symbol', pd.Series([''] * len(df_exec_all)))
                    aggregated_exec_history.append(df_exec_all)
            except Exception as e:
                logger.warning(f"Failed to collect execution history: {e}")

            if aggregated_exec_history:
                try:
                    merged_exec = pd.concat(aggregated_exec_history, axis=0, ignore_index=True)
                    # Ensure timestamp is datetime and add date column for consolidation
                    if 'timestamp' in merged_exec.columns:
                        try:
                            merged_exec['timestamp'] = pd.to_datetime(merged_exec['timestamp'])
                        except Exception:
                            pass
                        merged_exec['date'] = merged_exec['timestamp'].dt.date if hasattr(merged_exec['timestamp'], 'dt') else merged_exec['timestamp']
                    else:
                        logger.warning("Execution history missing 'timestamp' column; consolidation may be limited")
                        merged_exec['date'] = None

                    # Build a professional daily consolidated view (all symbols aggregated per date)
                    try:
                        exec_df = merged_exec.copy()
                        # Compute basic daily aggregates
                        if 'filled' in exec_df.columns:
                            filled_series = exec_df['filled'].astype(str).str.lower().isin(['true', '1', 'yes']).astype(int) if exec_df['filled'].dtype == object else exec_df['filled'].astype(int)
                        else:
                            filled_series = pd.Series([0] * len(exec_df))
                        exec_df['_filled_int'] = filled_series
                        exec_df['_is_market'] = (exec_df.get('attempt_type', pd.Series([''] * len(exec_df))) == 'market').astype(int) * exec_df['_filled_int']
                        exec_df['_is_limit'] = (exec_df.get('attempt_type', pd.Series([''] * len(exec_df))).str.startswith('limit')).astype(int) * exec_df['_filled_int']
                        exec_df['_is_moc'] = (exec_df.get('attempt_type', pd.Series([''] * len(exec_df))) == 'moc').astype(int) * exec_df['_filled_int']

                        # Normalize side/quantity
                        side_series = exec_df.get('side', pd.Series([''] * len(exec_df))).astype(str).str.upper()
                        qty_series = pd.to_numeric(exec_df.get('quantity', pd.Series([0] * len(exec_df))), errors='coerce').fillna(0)
                        price_series = pd.to_numeric(exec_df.get('execution_price', pd.Series([0.0] * len(exec_df))), errors='coerce').fillna(0.0)
                        slippage_series = pd.to_numeric(exec_df.get('slippage', pd.Series([0.0] * len(exec_df))), errors='coerce')

                        # Buy/Sell masks for filled trades
                        filled_mask = exec_df['_filled_int'] == 1
                        buy_mask = (side_series == 'BUY') & filled_mask
                        sell_mask = (side_series == 'SELL') & filled_mask

                        # Notionals and VWAP components
                        buy_notional = (qty_series.clip(lower=0) * price_series).where(buy_mask, 0.0)
                        sell_notional = (qty_series.abs() * price_series).where(sell_mask, 0.0)
                        buy_qty = qty_series.where(buy_mask, 0).clip(lower=0)
                        sell_qty = qty_series.where(sell_mask, 0).abs()

                        exec_df['_buy_notional'] = buy_notional
                        exec_df['_sell_notional'] = sell_notional
                        exec_df['_buy_qty'] = buy_qty
                        exec_df['_sell_qty'] = sell_qty
                        exec_df['_turnover'] = (qty_series.abs() * price_series).where(filled_mask, 0.0)

                        # Compact fills per date: "SYM: BUY 100@25.28 | SELL 100@25.38; PETR4: ..."
                        def _fills_compact(sub):
                            try:
                                parts = []
                                for sym, df_sym in sub[sub['_filled_int'] == 1].groupby('symbol'):
                                    rows = []
                                    for _, r in df_sym.iterrows():
                                        try:
                                            rows.append(f"{str(r.get('side','')).upper()} {int(abs(r.get('quantity',0)))}@{float(r.get('execution_price',0.0)):.2f}")
                                        except Exception:
                                            continue
                                    if rows:
                                        parts.append(f"{sym}: " + ", ".join(rows))
                                return "; ".join(parts)
                            except Exception:
                                return ""

                        grouped = exec_df.groupby('date').agg(
                            symbols=('symbol', lambda s: ','.join(sorted(set(map(str, s))))),
                            attempts=('symbol', 'size'),
                            filled_count=('_filled_int', 'sum'),
                            market_fills=('_is_market', 'sum'),
                            limit_fills=('_is_limit', 'sum'),
                            moc_fills=('_is_moc', 'sum'),
                            net_quantity=('quantity', 'sum'),
                            total_pnl=('pnl', 'sum'),
                            turnover_value=('_turnover', 'sum'),
                            qty_bought=('_buy_qty', 'sum'),
                            qty_sold=('_sell_qty', 'sum'),
                            notional_bought=('_buy_notional', 'sum'),
                            notional_sold=('_sell_notional', 'sum'),
                            avg_slippage_bps=('slippage', 'mean'),
                            median_slippage_bps=('slippage', 'median'),
                            fills_compact=('symbol', lambda _: _fills_compact(exec_df.loc[_.
                                index] if hasattr(_, 'index') else exec_df))
                        ).reset_index()
                        grouped = grouped.sort_values('date')

                        # Compute portfolio VWAPs (guard zero division)
                        grouped['portfolio_vwap_buy'] = grouped.apply(lambda r: (r['notional_bought']/r['qty_bought']) if r['qty_bought'] > 0 else None, axis=1)
                        grouped['portfolio_vwap_sell'] = grouped.apply(lambda r: (r['notional_sold']/r['qty_sold']) if r['qty_sold'] > 0 else None, axis=1)
                        grouped['fill_rate'] = grouped.apply(lambda r: (r['filled_count']/r['attempts']) if r['attempts'] > 0 else 0.0, axis=1)

                        # Render single HTML report similar to provided tester output
                        def fmt_money(x):
                            """Format numeric to pt-BR style (1.234,56). Fallback without changing business logic."""
                            try:
                                val = float(x)
                            except Exception:
                                return str(x)
                            try:
                                # Prefer locale when available; fallback to deterministic replace
                                import locale as _locale
                                _prev = _locale.setlocale(_locale.LC_NUMERIC)
                                applied = False
                                for _loc in ('pt_BR.UTF-8', 'pt_BR.utf8', 'pt_BR'):
                                    try:
                                        _locale.setlocale(_locale.LC_NUMERIC, _loc)
                                        applied = True
                                        break
                                    except Exception:
                                        continue
                                if applied:
                                    s = _locale.format_string('%.2f', val, grouping=True)
                                    _locale.setlocale(_locale.LC_NUMERIC, _prev)
                                    return s
                            except Exception:
                                pass
                            # Fallback (robust for negatives and grouping)
                            s = f"{val:,.2f}"
                            return s.replace(',', 'X').replace('.', ',').replace('X', '.')
                        def _html_escape(s):
                            import html as _html
                            return _html.escape(str(s))
                        
                        def _fills_compact(sub):
                            """Build compact per-symbol fill string using same money formatting."""
                            try:
                                parts = []
                                for sym, df_sym in sub[sub.get('_filled_int', 0) == 1].groupby('symbol'):
                                    rows = []
                                    for _, r in df_sym.iterrows():
                                        try:
                                            side = str(r.get('side', '')).upper()
                                            qty = int(abs(r.get('quantity', 0)))
                                            price_txt = fmt_money(r.get('execution_price', 0.0))
                                            rows.append(f"{side} {qty}@{price_txt}")
                                        except Exception:
                                            continue
                                    if rows:
                                        parts.append(f"{sym}: " + ", ".join(rows))
                                return "; ".join(parts)
                            except Exception:
                                return ""
                        # Summary stats
                        total_pnl = grouped['total_pnl'].sum() if 'total_pnl' in grouped.columns else 0.0
                        total_trades = int(exec_df['_filled_int'].sum()) if '_filled_int' in exec_df.columns else 0
                        unique_symbols = ','.join(sorted(set(exec_df.get('symbol', pd.Series(dtype=str)).astype(str))))
                        period_text = f"{min(exec_df['date'])} - {max(exec_df['date'])}" if 'date' in exec_df.columns and len(exec_df) else ""
                        # Build enriched data payload for advanced HTML report (UI/UX only; no engine changes)
                        import json as _json
                        # Ensure timestamp parsed
                        try:
                            exec_df['timestamp_dt'] = pd.to_datetime(exec_df['timestamp'], utc=True, errors='coerce')
                        except Exception:
                            exec_df['timestamp_dt'] = pd.NaT
                        # Derive date and hour
                        if 'date' not in exec_df.columns:
                            exec_df['date'] = exec_df['timestamp_dt'].dt.strftime('%Y-%m-%d')
                        exec_df['_hour'] = exec_df['timestamp_dt'].dt.hour.fillna(-1).astype(int)
                        # Side sign for PnL math
                        side_series_u = side_series  # already uppercase
                        exec_df['_side_sign'] = side_series_u.map({'BUY': 1, 'SELL': -1}).fillna(0).astype(int)
                        # Map 13:00 open per symbol/date
                        try:
                            open13_map = (
                                exec_df[exec_df['_hour'] == 13]
                                .groupby(['symbol','date'])['open']
                                .first()
                                .to_dict()
                            )
                        except Exception:
                            open13_map = {}
                        # Map MOC (20:00) close per symbol/date from MOC rows
                        try:
                            moc_close_map = {}
                            df_moc = exec_df[(exec_df['_hour'] == 20) | (exec_df.get('attempt_type','') == 'moc') | (exec_df.get('order_type','').astype(str).str.upper() == 'MOC')]
                            for (sym, d), sub in df_moc.groupby(['symbol','date']):
                                # Prefer close if present else execution_price
                                val = None
                                try:
                                    v = float(sub.iloc[0].get('close'))
                                    if v > 0:
                                        val = v
                                except Exception:
                                    val = None
                                if val is None:
                                    try:
                                        v = float(sub.iloc[0].get('execution_price'))
                                        if v > 0:
                                            val = v
                                    except Exception:
                                        val = None
                                if val is not None:
                                    moc_close_map[(sym, d)] = val
                        except Exception:
                            moc_close_map = {}
                        # Compute derived fields per row (UI-only)
                        def _safe_float(x, default_val=None):
                            try:
                                return float(x)
                            except Exception:
                                return default_val
                        derived_rows = []
                        for _, rr in exec_df.iterrows():
                            try:
                                sym = str(rr.get('symbol',''))
                                d = str(rr.get('date',''))
                                ts = rr.get('timestamp_dt')
                                ts_iso = ts.isoformat() if pd.notna(ts) else str(rr.get('timestamp',''))
                                side_u = str(rr.get('side','')).upper()
                                qty = int(abs(_safe_float(rr.get('quantity',0), 0) or 0))
                                price = _safe_float(rr.get('execution_price',0.0), 0.0) or 0.0
                                o = _safe_float(rr.get('open', None), None)
                                h = _safe_float(rr.get('high', None), None)
                                l = _safe_float(rr.get('low', None), None)
                                c = _safe_float(rr.get('close', None), None)
                                slp = _safe_float(rr.get('slippage', None), None)
                                ord_type = str(rr.get('order_type','')).upper()
                                attempt = str(rr.get('attempt_name', rr.get('attempt_type','')))
                                filled = int(rr.get('_filled_int', 0)) == 1
                                # Distances
                                open13 = open13_map.get((sym, d))
                                dist_from_open = abs(price - open13) if (open13 is not None and price is not None) else None
                                # Bar index (13:00 -> 0)
                                hr = int(rr.get('_hour', -1))
                                bar_index = (hr - 13) if (hr >= 13 and hr <= 20) else None
                                # PnL allocated (if not present)
                                pnl_existing = rr.get('pnl', None)
                                if pnl_existing is None or (isinstance(pnl_existing, float) and pd.isna(pnl_existing)):
                                    moc_close = moc_close_map.get((sym, d))
                                    side_sign = int(rr.get('_side_sign', 0))
                                    pnl_alloc = ((moc_close - price) * qty * side_sign) if (moc_close is not None and price is not None) else None
                                else:
                                    try:
                                        pnl_alloc = float(pnl_existing)
                                    except Exception:
                                        pnl_alloc = None
                                derived_rows.append({
                                    'symbol': sym,
                                    'timestamp': ts_iso,
                                    'date': d,
                                    'side': side_u,
                                    'order_type': ord_type,
                                    'attempt': attempt,
                                    'filled': bool(filled),
                                    'quantity': qty,
                                    'execution_price': price,
                                    'open': o, 'high': h, 'low': l, 'close': c,
                                    'slippage': slp,
                                    'bar_index': bar_index,
                                    'distance_from_open': dist_from_open,
                                    'pnl': pnl_alloc
                                })
                            except Exception:
                                continue
                        # Daily neutrality aggregates from executed (non-MOC)
                        daily_neutral = {}
                        for r in derived_rows:
                            if not r['filled']:
                                continue
                            if r['order_type'] == 'MOC':
                                continue
                            key = r['date']
                            dn = daily_neutral.setdefault(key, {'date': key, 'buy_notional': 0.0, 'sell_notional': 0.0, 'trades': 0})
                            notional = (r['execution_price'] or 0.0) * (r['quantity'] or 0)
                            if r['side'] == 'BUY':
                                dn['buy_notional'] += notional
                            elif r['side'] == 'SELL':
                                dn['sell_notional'] += notional
                            dn['trades'] += 1
                        for dn in daily_neutral.values():
                            total = dn['buy_notional'] + dn['sell_notional']
                            net = dn['buy_notional'] - dn['sell_notional']
                            dn['net_exposure'] = net
                            dn['neutrality_dev_abs'] = abs(net)
                            dn['neutrality_dev_pct'] = (abs(net) / total) if total > 0 else 0.0
                        # Symbol stats
                        symbol_stats = {}
                        for r in derived_rows:
                            if not r['filled']:
                                continue
                            ss = symbol_stats.setdefault(r['symbol'], {'symbol': r['symbol'], 'trades': 0, 'qty': 0, 'notional': 0.0, 'pnl': 0.0, 'wins': 0, 'slippages': [], 'limit_attempts': 0, 'limit_fills': 0})
                            ss['trades'] += 1
                            ss['qty'] += int(r['quantity'] or 0)
                            ss['notional'] += (r['execution_price'] or 0.0) * (r['quantity'] or 0)
                            if r['pnl'] is not None:
                                ss['pnl'] += r['pnl']
                                if r['pnl'] > 0:
                                    ss['wins'] += 1
                            if r['slippage'] is not None and r['order_type'] == 'MARKET':
                                ss['slippages'].append(r['slippage'])
                            if r['order_type'] == 'LIMIT':
                                ss['limit_attempts'] += 1
                                if r['filled']:
                                    ss['limit_fills'] += 1
                        # Daily PnL series
                        daily_pnl = {}
                        for r in derived_rows:
                            if r['order_type'] == 'MOC':
                                continue
                            if r['pnl'] is None:
                                continue
                            daily_pnl[r['date']] = daily_pnl.get(r['date'], 0.0) + r['pnl']
                        daily_pnl_series = sorted(([d, v] for d, v in daily_pnl.items()), key=lambda x: x[0])
                        # Symbols list and date bounds
                        symbols_list = sorted(list({r['symbol'] for r in derived_rows if r['symbol']}))
                        date_list = sorted(list({r['date'] for r in derived_rows if r['date']}))
                        date_min = date_list[0] if date_list else ''
                        date_max = date_list[-1] if date_list else ''
                        # Optional fuzzy_by_date ingestion if present
                        fuzzy_rows = []
                        try:
                            import os as _os
                            _fz = Path('reports') / 'fuzzy_by_date.csv'
                            if _fz.exists():
                                _fz_df = pd.read_csv(_fz)
                                # Normalize types
                                if 'date' in _fz_df.columns:
                                    _fz_df['date'] = _fz_df['date'].astype(str)
                                fuzzy_rows = _fz_df.to_dict(orient='records')
                        except Exception:
                            fuzzy_rows = []

                        report_payload = {
                            'rows': derived_rows,
                            'dailyNeutral': list(daily_neutral.values()),
                            'symbolStats': list(symbol_stats.values()),
                            'dailyPnl': daily_pnl_series,
                            'symbols': symbols_list,
                            'dateMin': date_min,
                            'dateMax': date_max,
                             'fuzzyByDate': fuzzy_rows,
                            'kpis': {
                                'totalPnl': float(sum(r['pnl'] for r in derived_rows if r['pnl'] is not None)),
                                'totalTrades': int(sum(1 for r in derived_rows if r['filled'])),
                            }
                        }
                        REPORT_JSON = _json.dumps(report_payload, ensure_ascii=False)
                        # Server-side render of fuzzy table for graceful no-JS viewing
                        def _fmt_brl(x):
                            try:
                                return f"{float(x):,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
                            except Exception:
                                return ''
                        fuzzy_rows_sorted = sorted(fuzzy_rows, key=lambda r: (str(r.get('date','')), str(r.get('symbol',''))))
                        fuzzy_table_html = "\n".join(
                            [
                                f"<tr align='right'>"
                                f"<td>{_html_escape(str(r.get('date','')))}</td>"
                                f"<td>{_html_escape(str(r.get('symbol','')))}</td>"
                                f"<td>{_html_escape(str(r.get('side','')))}</td>"
                                f"<td>{(float(r.get('fuzzy_score',0.0))):.4f}</td>"
                                f"<td>{'Yes' if r.get('eligible') else 'No'}</td>"
                                f"<td>{_html_escape(str(r.get('reason_if_not','') or ''))}</td>"
                                f"<td>{_fmt_brl(r.get('exposure_cap_brl',0.0))}</td>"
                                f"<td>{_fmt_brl(r.get('notional_P1',0.0))}</td>"
                                f"<td>{_fmt_brl(r.get('notional_P2',0.0))}</td>"
                                f"<td>{_fmt_brl(r.get('notional_P3',0.0))}</td>"
                                f"<td>{_fmt_brl(r.get('notional_P4',0.0))}</td>"
                                f"</tr>"
                                for r in fuzzy_rows_sorted
                            ]
                        )
                        # Build HTML
                        rows_daily = []
                        for _, r in grouped.iterrows():
                            # Derivations for presentation (no business logic changes)
                            fill_rate = r.get('fill_rate', 0.0)
                            try:
                                fill_rate_pct = f"{(float(fill_rate) * 100):.1f}%"
                            except Exception:
                                fill_rate_pct = "0.0%"
                            vwap_buy_txt = fmt_money(r['portfolio_vwap_buy']) if r.get('portfolio_vwap_buy') is not None else ''
                            vwap_sell_txt = fmt_money(r['portfolio_vwap_sell']) if r.get('portfolio_vwap_sell') is not None else ''
                            avg_slip = r.get('avg_slippage_bps')
                            med_slip = r.get('median_slippage_bps')
                            slip_txt = ''
                            try:
                                import math as _math
                                _avg_is_nan = _math.isnan(avg_slip) if isinstance(avg_slip, float) else False
                                _med_is_nan = _math.isnan(med_slip) if isinstance(med_slip, float) else False
                                if not _avg_is_nan and not _med_is_nan:
                                    slip_txt = f"{float(avg_slip):.0f}/{float(med_slip):.0f}"
                                elif not _avg_is_nan:
                                    slip_txt = f"{float(avg_slip):.0f}/-"
                                elif not _med_is_nan:
                                    slip_txt = f"-/{float(med_slip):.0f}"
                            except Exception:
                                slip_txt = ''

                            rows_daily.append(
                                f"<tr align='right'><td>{_html_escape(r['date'])}</td>"
                                f"<td>{_html_escape(r.get('symbols',''))}</td>"
                                f"<td>{int(r.get('attempts',0))}</td>"
                                f"<td>{int(r.get('filled_count',0))}</td>"
                                f"<td>{fill_rate_pct}</td>"
                                f"<td>{vwap_buy_txt}</td>"
                                f"<td>{vwap_sell_txt}</td>"
                                f"<td>{_html_escape(slip_txt)}</td>"
                                f"<td>{fmt_money(r.get('total_pnl',0.0))}</td>"
                                f"<td>{_html_escape(r.get('fills_compact',''))}</td></tr>"
                            )
                        daily_table = "\n".join(rows_daily)
                        # Orders table (attempt-level)
                        orders_rows = []
                        if len(exec_df):
                            for _, rr in exec_df.sort_values(['timestamp']).iterrows():
                                try:
                                    ts = pd.to_datetime(rr.get('timestamp')).strftime('%Y.%m.%d %H:%M:%S') if rr.get('timestamp') else ''
                                except Exception:
                                    ts = _html_escape(rr.get('timestamp',''))
                                orders_rows.append(
                                    f"<tr bgcolor='#FFFFFF' align=right>"
                                    f"<td>{ts}</td>"
                                    f"<td>{_html_escape(rr.get('symbol',''))}</td>"
                                    f"<td>{_html_escape(rr.get('side',''))}</td>"
                                    f"<td>{int(abs(rr.get('quantity',0)))}</td>"
                                    f"<td>{fmt_money(rr.get('execution_price',0.0))}</td>"
                                    f"<td>{fmt_money(rr.get('pnl',0.0))}</td>"
                                    f"<td>{_html_escape(rr.get('attempt_type',''))}</td>"
                                    f"</tr>"
                                )
                        orders_table = "\n".join(orders_rows)
                        placeholder_tpl = """
<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'>
  <title>Relatório de Execução de Portfólio</title>
  <style>
    :root { --bg:#ffffff; --fg:#000000; --muted:#555; --card:#f6f8fa; --accent:#0b5fff; }
    [data-theme="dark"] { --bg:#0d1117; --fg:#c9d1d9; --muted:#8b949e; --card:#161b22; --accent:#58a6ff; }
    html, body {{ height:100%; }}
    body {{ font-family: Tahoma, Arial, sans-serif; margin: 8px; background:var(--bg); color:var(--fg); }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #2d333b; padding: 6px; font-size: 10pt; }}
    th {{ background: #243b53; color:#fff; text-align: center; }}
    .section-title {{ font-size: 12pt; font-weight: bold; text-align: left; margin: 12px 0; }}
    .flex {{ display:flex; gap:12px; flex-wrap:wrap; align-items:flex-end; }}
    .card {{ background: var(--card); padding:10px; border-radius:6px; border:1px solid #2d333b; min-width:180px; }}
    .kpi {{ font-size:12px; color:var(--muted); }}
    .kpi strong {{ font-size:18px; color:var(--fg); }}
    .controls label {{ font-size:10pt; display:block; margin-bottom:4px; }}
    .controls input, .controls select {{ padding:6px; border-radius:4px; border:1px solid #2d333b; background:var(--bg); color:var(--fg); }}
    .btn {{ padding:6px 10px; background:var(--accent); color:#fff; border:none; border-radius:4px; cursor:pointer; }}
    .btn.secondary {{ background:#6c757d; }}
    .muted {{ color:var(--muted); font-size:10pt; }}
    tbody.zebra tr:nth-child(odd) {{ background-color:#f9fbfd; }}
    [data-theme="dark"] tbody.zebra tr:nth-child(odd) {{ background-color:#0f141a; }}
    thead.sticky th {{ position:sticky; top:0; z-index:2; }}
  </style>
  <meta name='generator' content='quant_b3_backtest'>
  <meta name='report-class' content='A'>
  <meta name='period' content='%%PERIOD%%'>
  <meta name='symbols' content='%%SYMBOLS%%'>
  <meta name='total-trades' content='%%TOTAL_TRADES%%'>
  <meta name='total-pnl' content='%%TOTAL_PNL%%'>
  <meta name='currency' content='BRL'>
  <meta name='locale' content='pt-BR'>
  <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
</head>
<body data-theme="light">
  <div>
    <div class='flex' style='align-items:center; justify-content:space-between;'>
      <div>
        <div style='font: 16pt Tahoma'><b>Relatório de Execução de Portfólio</b></div>
        <div class='muted'>Período: %%PERIOD%%</div>
      </div>
      <div class='flex'>
        <button class='btn' id='downloadCsvBtn'>Download CSV (filtrado)</button>
        <button class='btn secondary' id='themeBtn'>Tema: Light</button>
      </div>
    </div>
    <div class='controls card' style='margin-top:10px;'>
      <div class='flex'>
        <div>
          <label>Data inicial</label>
          <input type='date' id='dateStart'>
        </div>
        <div>
          <label>Data final</label>
          <input type='date' id='dateEnd'>
        </div>
        <div>
          <label>Símbolos</label>
          <select id='symbolsSelect' multiple size='4' style='min-width:220px;'></select>
        </div>
        <div>
          <label>Lados</label>
          <div><input type='checkbox' id='sideBuy' checked> BUY</div>
          <div><input type='checkbox' id='sideSell' checked> SELL</div>
        </div>
        <div>
          <label>Tipos</label>
          <div><input type='checkbox' id='typeMarket' checked> MARKET</div>
          <div><input type='checkbox' id='typeLimit' checked> LIMIT</div>
          <div><input type='checkbox' id='typeMoc' checked> MOC</div>
        </div>
        <div style='align-self:flex-end;'>
          <button class='btn' id='applyBtn'>Aplicar filtros</button>
          <button class='btn secondary' id='resetBtn'>Reset</button>
        </div>
      </div>
    </div>
    <div class='flex' style='margin-top:10px;'>
      <div class='card kpi'><div>Total PnL</div><strong id='kpiTotalPnl'>-</strong></div>
      <div class='card kpi'><div>Win rate</div><strong id='kpiWinRate'>-</strong></div>
      <div class='card kpi'><div>LIMIT Fill Ratio</div><strong id='kpiLimitFill'>-</strong></div>
      <div class='card kpi'><div>MARKET Slippage (avg)</div><strong id='kpiMktSlip'>-</strong></div>
      <div class='card kpi'><div>Neutrality Dev (avg)</div><strong id='kpiNeutralDev'>-</strong></div>
    </div>
    <div class='section-title'>Gráficos</div>
    <div id='chartDailyPnl' style='height:260px;'></div>
    <div id='chartCumPnl' style='height:260px; margin-top:8px;'></div>
    <div id='chartSlippage' style='height:260px; margin-top:8px;'></div>
    <div class='muted' id='noPlotly' style='display:none;'>Sem Plotly (CDN bloqueado). KPIs e tabelas continuam disponíveis.</div>

    <div class='section-title'>Consolidação Diária</div>
    <table cellspacing='1' cellpadding='3' border='0'>
      <tr align='center'>
        <th>Data</th><th>Símbolos</th><th>Ofertas</th><th>Fills</th><th>Fill Rate</th><th>VWAP Buy</th><th>VWAP Sell</th><th>Slippage (bps, avg/med)</th><th>PNL</th><th>Fills (compact)</th>
      </tr>
      %%DAILY_TABLE%%
    </table>
    <br>
    <div class='section-title'>Ordens (Detalhe)</div>
    <table cellspacing='1' cellpadding='3' border='0'>
      <tr align='center'>
        <th>Horário</th><th>Ativo</th><th>Direção</th><th>Quantidade</th><th>Preço</th><th>PNL (alloc.)</th><th>Tipo</th>
      </tr>
      %%ORDERS_TABLE%%
    </table>
    <div style='font-size:9pt;color:#555;text-align:left;margin-top:4px;'>PNL (alloc.): valores alocados por fill para auditoria; totais diários permanecem inalterados.</div>
    <br>
    <div class='section-title'>Fuzzy por Data e Ativo</div>
    <table cellspacing='1' cellpadding='3' border='0'>
      <thead class='sticky'>
        <tr align='center'>
          <th>Data</th><th>Ativo</th><th>Lado</th><th>Fuzzy</th><th>Elegível</th><th>Motivo (se não)</th><th>Capex (BRL)</th><th>P1</th><th>P2</th><th>P3</th><th>P4</th>
        </tr>
      </thead>
      <tbody id='fuzzyTableBody' class='zebra'>
        %%FUZZY_TABLE%%
      </tbody>
    </table>
  </div>
  <script>
    const REPORT_DATA = %%REPORT_JSON%%;
    const $ = (id) => document.getElementById(id);
    function brl(x) {{ if(x==null||isNaN(x)) return '-'; return (x).toLocaleString('pt-BR', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }}); }}
    function pct(x) {{ if(x==null||isNaN(x)) return '-'; return (x*100).toFixed(1)+'%'; }}
    function initControls() {{
      const sel = $('symbolsSelect');
      REPORT_DATA.symbols.forEach(s=>{{ const o=document.createElement('option'); o.value=s; o.textContent=s; sel.appendChild(o); }});
      $('dateStart').value = REPORT_DATA.dateMin || '';
      $('dateEnd').value = REPORT_DATA.dateMax || '';
      $('applyBtn').onclick = applyFilters;
      $('resetBtn').onclick = ()=>{{
        $('dateStart').value = REPORT_DATA.dateMin||'';
        $('dateEnd').value = REPORT_DATA.dateMax||'';
        Array.from(sel.options).forEach(o=>o.selected=false);
        ['sideBuy','sideSell','typeMarket','typeLimit','typeMoc'].forEach(id=>$(id).checked=true);
        applyFilters();
      }};
      $('themeBtn').onclick=()=>{{ const b=document.body; const dark=b.getAttribute('data-theme')==='dark'; b.setAttribute('data-theme', dark?'light':'dark'); $('themeBtn').textContent='Tema: '+(dark?'Light':'Dark'); }}
      $('downloadCsvBtn').onclick=()=>{{ const rows=currentFilteredRows; const headers=Object.keys(rows[0]||{{}}); const csv=[headers.join(',')].concat(rows.map(r=>headers.map(h=>JSON.stringify(r[h]??'')).join(','))).join('\n'); const blob=new Blob([csv], {{type:'text/csv'}}); const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='executions_filtered.csv'; a.click(); }}
    }}
    function fitDate(d) {{ return d>=($('dateStart').value||'') && d<=($('dateEnd').value||''); }}
    function fitSymbol(s) {{ const opts=Array.from($('symbolsSelect').selectedOptions).map(o=>o.value); return opts.length===0 || opts.includes(s); }}
    function fitSide(x) {{ return (x==='BUY' && $('sideBuy').checked) || (x==='SELL' && $('sideSell').checked); }}
    function fitType(t) {{ return (t==='MARKET' && $('typeMarket').checked) || (t==='LIMIT' && $('typeLimit').checked) || (t==='MOC' && $('typeMoc').checked); }}
    let currentFilteredRows = [];
    function applyFilters() {{
      const rows = REPORT_DATA.rows.filter(r=> r.filled && fitDate(r.date) && fitSymbol(r.symbol) && fitSide(r.side) && fitType(r.order_type));
      currentFilteredRows = rows;
      // KPIs
      const totalPnl = rows.reduce((a,r)=> a + (r.pnl||0), 0);
      const wins = rows.filter(r=> (r.pnl||0) > 0).length;
      const total = rows.length;
      const winRate = total? (wins/total):0;
      const limitRows = rows.filter(r=> r.order_type==='LIMIT');
      const limitFillRatio = (limitRows.length && REPORT_DATA.rows.filter(r=> r.order_type==='LIMIT').length)? (limitRows.length / REPORT_DATA.rows.filter(r=> r.order_type==='LIMIT').length): (limitRows.length?1:0);
      const mktRows = rows.filter(r=> r.order_type==='MARKET' && r.slippage!=null);
      const mktSlipAvg = mktRows.length? (mktRows.reduce((a,r)=>a+r.slippage,0)/mktRows.length):0;
      // Neutrality metrics on filtered days
      const dn = REPORT_DATA.dailyNeutral.filter(x=> fitDate(x.date));
      const meanDev = dn.length? dn.reduce((a,x)=>a + (x.neutrality_dev_pct||0),0)/dn.length : 0;
      $('kpiTotalPnl').textContent = 'R$ '+brl(totalPnl);
      $('kpiWinRate').textContent = (winRate*100).toFixed(1)+'%';
      $('kpiLimitFill').textContent = (limitFillRatio*100).toFixed(1)+'%';
      $('kpiMktSlip').textContent = mktSlipAvg.toFixed(1)+' bps';
      $('kpiNeutralDev').textContent = pct(meanDev);
      // Charts if Plotly available
      const hasPlotly = typeof Plotly !== 'undefined';
      document.getElementById('noPlotly').style.display = hasPlotly? 'none':'block';
      if (hasPlotly) {{
        // Daily PnL (filtered aggregation)
        const byDay = new Map();
        rows.forEach(r=> {{ if(r.order_type==='MOC') return; byDay.set(r.date, (byDay.get(r.date)||0)+(r.pnl||0)); }});
        const days = Array.from(byDay.keys()).sort();
        const vals = days.map(d=> byDay.get(d));
        Plotly.newPlot('chartDailyPnl', [{{ x: days, y: vals, type:'bar', marker:{{color: vals.map(v=> v>=0?'#2ca02c':'#d62728')}} }}], {{ margin:{{t:20,l:40,r:10,b:40}}, paper_bgcolor:'transparent', plot_bgcolor:'transparent', xaxis:{{tickangle:-45}} }});
        // Cumulative PnL
        let cum=0; const cumVals = vals.map(v=> (cum+=v));
        Plotly.newPlot('chartCumPnl', [{{ x: days, y: cumVals, type:'scatter', mode:'lines', line:{{color:'#0b5fff'}} }}], {{ margin:{{t:20,l:40,r:10,b:40}}, paper_bgcolor:'transparent', plot_bgcolor:'transparent' }});
        // Slippage histogram (MARKET)
        const slips = rows.filter(r=> r.order_type==='MARKET' && r.slippage!=null).map(r=> r.slippage);
        Plotly.newPlot('chartSlippage', [{{ x: slips, type:'histogram', marker:{{color:'#9467bd'}} }}], {{ margin:{{t:20,l:40,r:10,b:40}}, paper_bgcolor:'transparent', plot_bgcolor:'transparent', xaxis:{{title:'bps'}} }});
      }}

      // Render fuzzy table (no filters except date/symbol/side)
      const fz = (REPORT_DATA.fuzzyByDate||[]).filter(r=> fitDate(r.date) && fitSymbol(r.symbol) && fitSide(r.side));
      const fzRows = fz.map(r=>`<tr align='right'><td>${r.date||''}</td><td>${r.symbol||''}</td><td>${r.side||''}</td><td>${(r.fuzzy_score??'').toFixed ? Number(r.fuzzy_score).toFixed(4) : (r.fuzzy_score||'')}</td><td>${r.eligible? 'Yes':'No'}</td><td>${r.reason_if_not||''}</td><td>${brl(r.exposure_cap_brl||0)}</td><td>${brl(r.notional_P1||0)}</td><td>${brl(r.notional_P2||0)}</td><td>${brl(r.notional_P3||0)}</td><td>${brl(r.notional_P4||0)}</td></tr>`).join('\n');
      const body = document.getElementById('fuzzyTableBody');
      if (body) body.innerHTML = fzRows || '';
    }}
    document.addEventListener('DOMContentLoaded', ()=>{ initControls(); applyFilters(); });
  </script>
</body>
</html>
"""
                        template = (
                            placeholder_tpl
                            .replace('%%PERIOD%%', _html_escape(period_text))
                            .replace('%%SYMBOLS%%', _html_escape(unique_symbols))
                            .replace('%%TOTAL_TRADES%%', str(total_trades))
                            .replace('%%TOTAL_PNL%%', fmt_money(total_pnl))
                            .replace('%%DAILY_TABLE%%', daily_table)
                            .replace('%%ORDERS_TABLE%%', orders_table)
                            .replace('%%FUZZY_TABLE%%', fuzzy_table_html)
                            .replace('%%REPORT_JSON%%', REPORT_JSON)
                        )
                        t0_report = time.perf_counter()
                        with open(html_path, 'w', encoding='utf-8') as f:
                            f.write(template)
                        print(f"HTML report saved to: {html_path}")
                        report_secs = time.perf_counter() - t0_report

                        # Post-generation validation: run execution-vs-BRAPI test
                        try:
                            import os as _os
                            import subprocess as _sp
                            # Allow disabling via env var if needed
                            if _os.getenv('RUN_EXEC_VALIDATION', '1').lower() in ('1', 'true', 'yes'):
                                validation_args = [
                                    'pytest',
                                    'tests/test_validate_html_executions_against_brapi.py',
                                    '--report-html', str(html_path),
                                    '--tolerance', _os.getenv('VALIDATION_TOLERANCE', '0.01'),
                                    '-q'
                                ]
                                print("Running execution validation test (html vs BRAPI)...")
                                _sp.run(validation_args, check=False)
                        except Exception as _e:
                            logger.warning(f"Execution validation step skipped: {_e}")
                    except Exception as e:
                        logger.warning(f"Failed to build daily consolidated execution history: {e}")
                except Exception as e:
                    logger.warning(f"Failed to save aggregated execution history: {e}")
            else:
                # No executions available: write minimal stub so a file exists
                try:
                    fallback_period = f"{start_date} - {end_date}"
                    fallback_symbols = ','.join(tickers)
                    stub_tpl = """
<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'>
  <title>Relatório de Execução de Portfólio</title>
  <meta name='generator' content='quant_b3_backtest'>
  <meta name='report-class' content='A'>
  <meta name='period' content='%%PERIOD%%'>
  <meta name='symbols' content='%%SYMBOLS%%'>
  <meta name='total-trades' content='0'>
  <meta name='total-pnl' content='0,00'>
  <meta name='currency' content='BRL'>
  <meta name='locale' content='pt-BR'>
  <style>
    body { font-family: Tahoma, Arial, sans-serif; margin: 16px; }
    .box { border: 1px solid #ccc; padding: 14px; border-radius: 6px; background: #f8f9fa; }
  </style>
  </head>
  <body>
    <h2>Relatório de Execução de Portfólio</h2>
    <div class='box'>
      <div><strong>Período:</strong> %%PERIOD%%</div>
      <div><strong>Ativos:</strong> %%SYMBOLS%%</div>
      <div style='margin-top:10px;'>Nenhuma execução foi gerada neste intervalo. O relatório foi criado sem dados.</div>
    </div>
  </body>
</html>
"""
                    stub_html = (
                        stub_tpl
                        .replace('%%PERIOD%%', str(fallback_period))
                        .replace('%%SYMBOLS%%', str(fallback_symbols))
                    )
                    with open(html_path, 'w', encoding='utf-8') as _f:
                        _f.write(stub_html)
                    print(f"HTML report saved to: {html_path} (no executions)")
                except Exception as _e:
                    logger.warning(f"Failed to write stub HTML report: {_e}")
        except Exception as e:
            logger.error(f"Failed to render HTML report: {e}")

        if failures:
            logger.warning(f"Tickers with failures (skipped): {','.join(failures)}")
        
        # Final compact timing summary
        total_secs = time.perf_counter() - t0_total
        try:
            print(f"Completed in {total_secs:.2f}s (data {data_load_secs:.2f}s, sim {sim_secs:.2f}s, report {report_secs if 'report_secs' in locals() else 0.0:.2f}s)")
        except Exception:
            print(f"Completed in {total_secs:.2f}s")
        logger.info("Backtest batch completed.")
        return 0
        
        # Pre-simulation data validation
        logger.info("Validating data quality...")
        
        # Check data completeness
        total_bars = len(ticker_data)
        unique_dates = len(set(ticker_data.index.date))
        avg_bars_per_day = total_bars / unique_dates if unique_dates > 0 else 0
        
        logger.info(f"Data validation for universe {','.join(tickers)}:")
        logger.info(f"  - Total intraday bars: {total_bars}")
        logger.info(f"  - Unique trading days: {unique_dates}")
        logger.info(f"  - Average bars per day: {avg_bars_per_day:.1f}")
        logger.info(f"  - Date range: {ticker_data.index.min().date()} to {ticker_data.index.max().date()}")
        
        # Check for expected data structure (7 bars per day for Brazilian market)
        if avg_bars_per_day < 6.0:
            logger.warning(f"Low average bars per day ({avg_bars_per_day:.1f}) - expected ~7 for Brazilian market")
            logger.warning("This may indicate data quality issues or partial trading days")
        
        # Check data gaps
        date_range = pd.date_range(start=ticker_data.index.min().date(), 
                                  end=ticker_data.index.max().date(), 
                                  freq='D')
        missing_dates = []
        for date in date_range:
            if date.weekday() < 5:  # Weekdays only
                day_data = ticker_data[ticker_data.index.date == date.date()]
                if len(day_data) == 0:
                    missing_dates.append(date.date())
        
        if missing_dates:
            logger.warning(f"Found {len(missing_dates)} missing business days: {missing_dates[:5]}{'...' if len(missing_dates) > 5 else ''}")
        
        # Filter data for the specified date range
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        mask = (ticker_data.index >= start_dt) & (ticker_data.index <= end_dt)
        filtered_data = ticker_data.loc[mask]
        
        if filtered_data.empty:
            logger.error(f"No data available for {ticker} in date range {start_date} to {end_date}")
            return 1
        
        # Validate filtered data
        filtered_bars = len(filtered_data)
        filtered_dates = len(set(filtered_data.index.date))
        filtered_avg_bars = filtered_bars / filtered_dates if filtered_dates > 0 else 0
        
        logger.info(f"Filtered data for simulation period:")
        logger.info(f"  - Simulation bars: {filtered_bars}")
        logger.info(f"  - Simulation trading days: {filtered_dates}")
        logger.info(f"  - Average bars per day: {filtered_avg_bars:.1f}")
        
        # Check if we have enough data for warmup
        warmup_required = 30  # Default warmup requirement
        if filtered_bars < warmup_required:
            logger.warning(f"Limited simulation data ({filtered_bars} bars) - warmup may use historical data")
            logger.info("Strategy will use complete historical data for warmup period")
        
        logger.info(f"Loaded {len(filtered_data)} data points for universe {','.join(tickers)}")
        
        # Initialize portfolio
        portfolio = EnhancedPortfolio(config_path="config/settings.yaml")
        
        # Create strategy configuration
        strategy_config = StrategyConfig(
            universe=tickers,
            warmup_bars=30,
            risk_tolerance=0.02,
            max_position_size=0.10,
            max_daily_loss=0.02,
            stop_loss_pct=0.05,
            take_profit_pct=0.10
        )
        
        # Create enhanced strategy context with hybrid data management
        market_utils = BrazilianMarketUtils()
        strategy_context = StrategyContext(
            data_portal=data_loader,
            portfolio=portfolio,
            broker=None,
            market_rules=market_utils,
            logger=logging.getLogger("FuzzyFajutoStrategy"),
            metadata={
                'strategy_config_path': "config/profiles/fuzzy_fajuto_default.yaml",
                'complete_data': ticker_data,  # Combined complete data for all symbols
                # Minimal compatible hybrid metadata for downstream consumers
                'hybrid_data_result': {'execution_data': ticker_data},
                'data_sources_used': (last_hybrid_meta.get('data_sources', {}) if last_hybrid_meta else {})
            }
        )
        
        # Add hybrid data manager to context for smart warmup access
        strategy_context.hybrid_data_manager = hybrid_data_manager
        
        # Create strategy with profile or custom config
        strategy = create_fuzzy_fajuto(
            cfg=strategy_config,
            ctx=strategy_context,
            profile=profile,
            config_file=config_file
        )
        
        # Initialize simulator
        simulator = BacktestSimulator(
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            config_path="config/settings.yaml"
        )
        
        # Run backtest
        logger.info("Running backtest...")
        results = simulator.run_simulation(filtered_data)
        
        # Initialize result manager
        result_manager = ResultManager() if args.save_results else None
        
        import os as _os
        if not (_os.getenv('AUDIT_EXECUTIONS_ONLY', '1').lower() in ('1', 'true', 'yes')):
            # Print results
            print("\n" + "="*80)
            print("BACKTEST RESULTS")
            print("="*80)
            print(f"Strategy: FuzzyFajutoStrategy    Profile: {profile}    Config: {config_file or 'profile-based'}")
            print(f"Universe: {','.join(tickers)}         Period: {start_date} to {end_date}")
            print("="*80)
            print(f"{'Metric':<25} {'Value':<25} {'Metric':<25} {'Value':<25}")
            print("-" * 80)
            print(f"{'Total Return':<25} {results.total_return:>24.2%} {'Sharpe Ratio':<25} {results.sharpe_ratio:>24.2f}")
            print(f"{'Max Drawdown':<25} {results.max_drawdown:>24.2%} {'Win/Loss Ratio':<25} {results.win_loss_ratio:>24.2f}")
            print(f"{'Total Trades':<25} {results.total_trades:>24d} {'Winning Trades':<25} {results.winning_trades:>24d}")
            win_rate = (results.winning_trades/results.total_trades*100) if results.total_trades > 0 else 0.0
            print(f"{'Losing Trades':<25} {results.losing_trades:>24d} {'Win Rate':<25} {win_rate:>24.1f}%")
            print(f"{'Initial Capital':<25} R$ {results.initial_capital:>21,.2f} {'Final Portfolio Value':<25} R$ {results.final_portfolio_value:>21,.2f}")
            print(f"{'Total Commission':<25} R$ {results.total_commission:>21,.2f} {'Total Taxes':<25} R$ {results.total_taxes:>21,.2f}")
            print("="*80)
        
        # Get strategy performance summary
        strategy_summary = strategy.get_performance_summary()
        if strategy_summary:
            print("\n" + "="*80)
            print("STRATEGY PERFORMANCE SUMMARY")
            print("="*80)
            
            # Generate and display clean execution table
            table_str = generate_execution_table(strategy_summary)
            print(table_str)
        
        if not (_os.getenv('AUDIT_EXECUTIONS_ONLY', '1').lower() in ('1', 'true', 'yes')):
            # Show hybrid data system performance
            print("\n" + "="*80)
            print("HYBRID DATA SYSTEM PERFORMANCE")
            print("="*80)
            
            cache_status = hybrid_data_manager.brapi_provider.get_cache_status()
            print(f"{'Category':<25} {'Metric':<25} {'Value':<30}")
            print("-" * 80)
            
            # Cache Performance
            print(f"{'Cache Performance':<25} {'Cache Hits':<25} {cache_status['performance']['hits']:<30}")
            print(f"{'':<25} {'API Fetches':<25} {cache_status['performance']['fetches']:<30}")
            print(f"{'':<25} {'Hit Ratio':<25} {cache_status['performance']['hit_ratio']:<30}")
            print(f"{'':<25} {'Total Load Time':<25} {cache_status['performance']['total_load_time']:<30}")
            
            # Storage
            print(f"{'Storage':<25} {'Daily Symbols Cached':<25} {cache_status['storage']['daily_symbols']:<30}")
            print(f"{'':<25} {'Total Cache Size':<25} {cache_status['storage']['total_size_mb']} MB{'':<25}")
            
            # Data sources summary
            data_sources = hybrid_data_result.get('data_sources', {})
            indicators_info = data_sources.get('indicators', {})
            execution_info = data_sources.get('execution', {})
            
            print(f"{'Data Sources':<25} {'Technical Indicators':<25} {indicators_info.get('source', 'Unknown')} ({indicators_info.get('bars_count', 0)} bars){'':<10}")
            print(f"{'':<25} {'Execution Data':<25} {execution_info.get('source', 'Unknown')} ({execution_info.get('bars_count', 0)} bars){'':<10}")
            print(f"{'':<25} {'Execution Coverage':<25} {execution_info.get('coverage', 'Unknown'):<30}")
            print(f"{'':<25} {'Execution Accuracy':<25} {execution_info.get('accuracy', 'Unknown'):<30}")
            
            overall_info = data_sources.get('overall', {})
            print(f"{'Data Quality':<25} {'Confidence Level':<25} {overall_info.get('confidence', 'Unknown'):<30}")
            print(f"{'':<25} {'Limitations':<25} {overall_info.get('limitations', 'None'):<30}")
            
            print("="*80)
        
        # Unreachable with batch return above; kept for structure
        return 0
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 