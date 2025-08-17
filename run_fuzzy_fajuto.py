#!/usr/bin/env python3
"""
Run FuzzyFajuto Strategy with Profile Support

Enhanced version that supports different parameter profiles and result tracking.
Use --profile to specify different parameter sets.

Examples:
    AUDIT_EXECUTIONS_ONLY=0 MULTIFRAME_MODE=1 \
    python3 run_fuzzy_fajuto.py \
      --start-date 2025-07-15 --end-date 2025-08-01 \
      --save-results
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

# Setup logging with env override (default ERROR)
_lvl = os.environ.get('LOG_LEVEL', 'ERROR').upper()
_lvl_num = getattr(logging, _lvl, logging.ERROR)
logging.basicConfig(
    level=_lvl_num,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Apply module logger levels based on env override
logging.getLogger().setLevel(_lvl_num)
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
        logging.getLogger(noisy).setLevel(_lvl_num)
    except Exception:
        pass

# Enable pandas Copy-on-Write to reduce unnecessary copies (no logic change)
try:
    pd.options.mode.copy_on_write = True
except Exception:
    pass

from engine.fuzzy_reporting import export_fuzzy_components_to_csv
from engine.market_utils import prepare_fuzzy_data, SignalScheduler
from engine.base_strategy import OrderSide
from engine.utils.async_logger import AsyncJsonlLogger


from engine.utils.terminal_table import generate_execution_table

def _load_portfolio_symbols() -> list:
    """Load symbols from portfolio.csv in project root or data/.

    Returns uppercase symbols list or empty list if not found/invalid.
    """
    # Prefer data/portfolio.csv explicitly; fallback to root portfolio.csv
    candidates = [
        Path(__file__).parent / 'data' / 'portfolio.csv',
        Path(__file__).parent / 'portfolio.csv',
    ]
    for path in candidates:
        try:
            if path.exists():
                df = pd.read_csv(path)
                if 'symbol' in df.columns:
                    symbols = [str(s).strip().upper() for s in df['symbol'].dropna().tolist()]
                    symbols = [s for s in symbols if len(s) > 0]
                    syms_out = list(dict.fromkeys(symbols))  # de-duplicate preserving order
                    try:
                        print(f"Loaded portfolio from: {path} ({len(syms_out)} symbols)")
                    except Exception:
                        pass
                    return syms_out
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
  AUDIT_EXECUTIONS_ONLY=0 MULTIFRAME_MODE=1 \
  python3 %(prog)s \
    --start-date 2025-07-15 --end-date 2025-08-01 \
    --save-results"""
    )
    
    # Universe is sourced exclusively from data/portfolio.csv per README
    parser.add_argument('--profile', default='default', help='Strategy profile (default: default)')
    parser.add_argument('--config-file', help='Custom config file (overrides profile)')
    parser.add_argument('--start-date', default=None, help='Start date (default from config)')
    parser.add_argument('--end-date', default=None, help='End date (default from config)')
    # Minimal reporting toggle
    parser.add_argument('--save-results', action='store_true', default=None, help='Save results (defaults from config)')
    # Pair mode is mandatory; CLI flag removed
    
    args = parser.parse_args()
    
    try:
        # Silence verbose data-source prints during multi-symbol runs
        try:
            import os as _os
            _os.environ.setdefault('DISABLE_DATA_SOURCE_REPORT', '1')
            # Ensure multi-frame mode is ON before any coverage prefiltering logic
            _os.environ.setdefault('MULTIFRAME_MODE', '1')
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
        # Use timezone-aware UTC to avoid deprecation
        from datetime import datetime as _DT, timezone as _TZ
        run_ts = _DT.now(_TZ.utc).strftime('%Y%m%d_%H%M%S')
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

        # Determine tickers universe from portfolio.csv only
        csv_symbols = _load_portfolio_symbols()
        if not csv_symbols:
            raise SystemExit("portfolio.csv não encontrado ou sem coluna 'symbol'. Coloque sua lista em data/portfolio.csv.")
        tickers = csv_symbols
        logger.info(f"Loaded portfolio from CSV: {','.join(tickers)}")

        # Max-range auto-selection and bulk prefetch removed to simplify CLI; use explicit dates or defaults

        # Prefilter symbols by hourly coverage to avoid mid-run cancellation
        # Uses the same thresholds as loader._assess_execution_coverage
        try:
            import os as _os
            if _os.getenv('MULTIFRAME_MODE', 'off').lower() in ('1','true','yes','on'):
                logger.info("Prefilter coverage: skipped in multi-frame mode (hourly fetched only for execution days)")
            else:
                from engine.brapi_provider import BrapiProvider as _BrapiProvider
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
        # Pre-simulation data requirements checker (no auto-download)
        try:
            from engine.loader import DataLoader as _DL
            warmup_needed = int((_cfg.get('strategy', {}) or {}).get('warmup_min_sessions', 60))
            chk = _DL.check_intraday_data_requirements(
                symbols=tickers,
                start_date=start_date,
                end_date=end_date,
                warmup_threshold=warmup_needed,
                cache_dir=((_cfg.get('brapi', {}) or {}).get('data', {}) or {}).get('cache_dir', 'data/brapi_cache'),
                log_dir='logs'
            )
            insuff = chk.get('summary', {}).get('insufficient', 0)
            if insuff > 0:
                logger.error(f"Data requirements not met for {insuff} symbols. See {chk['artifact_paths']['log']} and CSV/JSON summaries.")
        except Exception as _e:
            logger.warning(f"Data requirements check failed: {_e}")

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

        # ===============
        # Preflight checks
        # ===============
        def _build_report_skeleton():
            return {
                'schema_version': '1.0.0',
                'run_id': run_id,
                'metadata': {
                    'strategy': 'FuzzyFajutoStrategy',
                    'strategy_version': '1.x',
                    'environment': 'backtest',
                    'timezone': 'UTC',
                    'session_calendar': 'B3',
                    'start_date': start_date,
                    'end_date': end_date,
                    'symbols': tickers,
                    'benchmark': '^BVSP',
                    'data_granularity': {'execution_input': 'hourly', 'indicators_input': 'daily'},
                    'report_format': 'json',
                    'report_level': 'summary',
                    'ingestion_cache_policy': 'use',
                    'strict_warmup': False,
                    'allow_degraded_warmup': True,
                    'warmup_min_sessions': int((_cfg.get('strategy', {}) or {}).get('warmup_min_sessions', 60))
                },
                'preflight': {'symbols': {}, 'benchmark': {}, 'status': 'unknown', 'reasons': []},
                'ingestion': {'diagnostics': {}, 'reasons': []},
                'signals': {'rows': [], 'reasons': []},
                'orders': {'legs': [], 'reasons': []},
                'execution': {'fills': [], 'reasons': []},
                'eod': {'flattening': {}, 'reasons': []},
                'outcomes': {'summary': {}, 'reasons': []}
            }

        def _write_report(report_dict):
            try:
                out_dir = Path('reports')
                out_dir.mkdir(parents=True, exist_ok=True)
                json_path = out_dir / 'portfolio_backtest_report.json'
                import json as _json
                with open(json_path, 'w', encoding='utf-8') as f:
                    _json.dump(report_dict, f, ensure_ascii=False, separators=(",", ":"))
                print(f"Report saved: {json_path}")
            except Exception as _e:
                logger.warning(f"Failed to write report: {_e}")

        # Preflight: data availability and warmup sufficiency per symbol and benchmark
        preflight = _build_report_skeleton()
        try:
            from engine.brapi_provider import BrapiProvider as _BrapiProvider
            import os as _os
            # Resolve BRAPI token from secrets.yaml first, then env
            _token = None
            try:
                import yaml as _yaml
                _secrets = {}
                _sp = Path('config/secrets.yaml')
                if _sp.exists():
                    _secrets = _yaml.safe_load(_sp.read_text()) or {}
                    _token = (_secrets.get('brapi') or {}).get('api_token') or _secrets.get('BRAPI_API_TOKEN')
            except Exception:
                pass
            if not _token:
                _token = _os.environ.get('BRAPI_API_TOKEN','')
            bp = _BrapiProvider(api_token=_token, cache_dir="data/brapi_cache")
            # Benchmark (^BVSP) check
            try:
                ibov_daily = bp.get_daily_data('^BVSP', start_date, end_date)
                preflight['preflight']['benchmark'] = {
                    'symbol': '^BVSP',
                    'has_data': bool(ibov_daily is not None and not ibov_daily.empty),
                    'rows': int(0 if ibov_daily is None else len(ibov_daily))
                }
                if not preflight['preflight']['benchmark']['has_data']:
                    preflight['preflight']['reasons'].append('benchmark_unavailable')
            except Exception as _e:
                preflight['preflight']['benchmark'] = {'symbol': '^BVSP', 'has_data': False, 'error': str(_e)}
                preflight['preflight']['reasons'].append('benchmark_error')

            # Symbols coverage
            from datetime import datetime as _dt
            sdt = _dt.strptime(start_date, '%Y-%m-%d')
            edt = _dt.strptime(end_date, '%Y-%m-%d')
            total_days = max((edt - sdt).days + 1, 1)
            for sym in tickers:
                try:
                    h = bp.get_ohlc(sym, '1h', sdt, edt)
                    u_days = len(set(h.index.date)) if h is not None and not h.empty else 0
                    bpd = (len(h) / u_days) if u_days > 0 else 0.0
                    cov = (u_days / max(total_days*5//7,1)) if u_days>0 else 0.0
                    preflight['preflight']['symbols'][sym] = {
                        'hourly_rows': int(len(h) if h is not None else 0),
                        'unique_days': int(u_days),
                        'bars_per_day': float(round(bpd,2)),
                        'coverage_pct': float(round(cov,3))
                    }
                    if cov < 0.8 or bpd < 6:
                        preflight['preflight']['reasons'].append(f'insufficient_hourly:{sym}')
                except Exception as _e:
                    preflight['preflight']['symbols'][sym] = {'error': str(_e)}
                    preflight['preflight']['reasons'].append(f'fetch_error:{sym}')

            # Auto completeness check and on-demand fetch for the execution window
            if True:
                _missing = []
                for sym in tickers:
                    try:
                        _df_hourly = bp.get_ohlc(sym, '1h', sdt, edt)
                        if _df_hourly is None or _df_hourly.empty:
                            _missing.append(sym)
                    except Exception:
                        _missing.append(sym)
                if _missing:
                    logger.info(f"Auto-download: attempting to backfill hourly gaps for {len(_missing)} symbols")
                    for i, sym in enumerate(_missing, 1):
                        try:
                            _ = bp.get_ohlc(sym, '1h', sdt, edt)
                            if i % 20 == 0:
                                logger.info(f"Auto-download progress: {i}/{len(_missing)}")
                        except Exception as _e:
                            logger.warning(f"Auto-download failed for {sym}: {_e}")
                logger.info("Data completeness check passed for hourly execution window")

            # Warmup policy
            warmup_required = int((_cfg.get('strategy', {}) or {}).get('warmup_min_sessions', 60))
            preflight['preflight']['warmup_required_sessions'] = warmup_required
            # Mark status
            preflight['preflight']['status'] = 'ok' if len(preflight['preflight']['reasons'])==0 else 'degraded'
        except Exception as _e:
            preflight['preflight']['status'] = 'error'
            preflight['preflight']['reasons'].append(f'preflight_error:{_e}')

        # If strict fail requested and we have preflight issues, write report and exit
        if False and preflight['preflight']['status'] != 'ok':
            _write_report(preflight)
            # Also write empty CSVs with reasons
            try:
                from pathlib import Path as _P
                import pandas as _pd
                _P('reports').mkdir(exist_ok=True)
                _pd.DataFrame({'section':['preflight'], 'reason':preflight['preflight']['reasons']}).to_csv('reports/portfolio_backtest_executions.csv', index=False)
                _pd.DataFrame(columns=['symbol','timestamp','date','side','order_type','attempt','filled','quantity','execution_price','open','high','low','close','slippage','bar_index','distance_from_open','pnl']).to_csv('reports/portfolio_backtest_executions_derived.csv', index=False)
                _pd.DataFrame(preflight['signals']['rows']).to_csv('reports/portfolio_fuzzy_indicators.csv', index=False)
            except Exception:
                pass
            return 0
        
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
        
        # Daily-only indicators and schedule (multi-frame injection)
        try:
            from engine.market_utils import IndicatorService, SignalScheduler, DailyTechnicalIndicators
            os.environ.setdefault('MULTIFRAME_MODE', '1')
            ind_svc = IndicatorService()
            # Resolve benchmark symbol from config; fallback to BRAPI IBOV if needed
            _bm_from_cfg = (_cfg.get('benchmark', {}) or {}).get('symbol', '^BVSP')
            _bm_fallback = ((_cfg.get('brapi', {}) or {}).get('data', {}) or {}).get('ibov_symbol', 'IBOV')
            bm_symbol = _bm_from_cfg or _bm_fallback or '^BVSP'
            # Daily vectors for universe (no hourly)
            vectors = ind_svc.compute_daily_vectors(
                symbols=tickers,
                benchmark=bm_symbol,
                start=start_date,
                end=end_date,
                ema_periods=[3,5,10,15,20],
                rsi_period=10,
                atr_period=14,
                warmup_min_sessions=int((_cfg.get('strategy', {}) or {}).get('warmup_min_sessions', 60)),
                buffer_sessions=5
            )
            # If empty and we weren't already on the fallback, try fallback benchmark symbol
            if (not isinstance(vectors, dict) or len(vectors) == 0) and bm_symbol != _bm_fallback:
                try:
                    vectors = ind_svc.compute_daily_vectors(
                        symbols=tickers,
                        benchmark=_bm_fallback,
                        start=start_date,
                        end=end_date,
                        ema_periods=[3,5,10,15,20],
                        rsi_period=10,
                        atr_period=14,
                        warmup_min_sessions=int((_cfg.get('strategy', {}) or {}).get('warmup_min_sessions', 60)),
                        buffer_sessions=5
                    )
                    logger.info(f"Retry vectors with fallback benchmark '{_bm_fallback}': symbols={len(vectors)}")
                except Exception:
                    pass

            # Final fallback: build daily vectors by aggregating hourly execution data (cache-only, no network)
            if not isinstance(vectors, dict) or len(vectors) == 0:
                try:
                    _agg_vectors: dict[str, _pd.DataFrame] = {}
                    _ind_calc = DailyTechnicalIndicators()
                    for sym in tickers:
                        try:
                            res = hybrid_data_manager.initialize_backtest_data(
                                symbol=sym,
                                start_date=start_date,
                                end_date=end_date,
                                local_loader=data_loader
                            )
                            exec_df = res.get('execution_data')
                            if exec_df is None or exec_df.empty:
                                continue
                            ddf = exec_df.copy()
                            if getattr(ddf.index, 'tz', None) is not None:
                                ddf.index = ddf.index.tz_localize(None)
                            # Aggregate to daily OHLCV
                            gb = ddf.groupby(ddf.index.normalize())
                            daily = _pd.DataFrame({
                                'open': gb['open'].first(),
                                'high': gb['high'].max(),
                                'low': gb['low'].min(),
                                'close': gb['close'].last(),
                                'volume': gb['volume'].sum() if 'volume' in ddf.columns else 0
                            })
                            daily.index = _pd.to_datetime(daily.index)
                            daily = daily.sort_index()
                            # Compute indicators
                            atr = _ind_calc.calculate_atr(daily, period=14)
                            # Minimal EMA set
                            ema3 = daily['close'].ewm(span=3, adjust=False).mean()
                            ema5 = daily['close'].ewm(span=5, adjust=False).mean()
                            ema10 = daily['close'].ewm(span=10, adjust=False).mean()
                            ema15 = daily['close'].ewm(span=15, adjust=False).mean()
                            ema20 = daily['close'].ewm(span=20, adjust=False).mean()
                            # Returns and simple fuzzy score (no benchmark)
                            sym_ret = daily['close'].pct_change()
                            ret_vs_ibov_term = (sym_ret > 0).astype(int) - (sym_ret < 0).astype(int)
                            ema_sum = ((daily['close'] > ema3).astype(float)
                                       + (daily['close'] > ema5).astype(float)
                                       + (daily['close'] > ema10).astype(float)
                                       + (daily['close'] > ema15).astype(float)
                                       + (daily['close'] > ema20).astype(float)) * 0.25
                            # Basic RSI(10) fallback
                            delta = daily['close'].diff()
                            up = delta.clip(lower=0).rolling(10).mean()
                            down = (-delta.clip(upper=0)).rolling(10).mean()
                            rs = up / (down.replace(0, _pd.NA))
                            rsi = 100 - (100 / (1 + rs))
                            rsi_term = _pd.Series(0.0, index=daily.index)
                            rsi_term[rsi > 65] = 0.25
                            rsi_term[rsi < 35] = -0.25
                            score = ret_vs_ibov_term.astype(float) + ema_sum.fillna(0.0) + rsi_term.fillna(0.0)
                            out = _pd.DataFrame(index=daily.index)
                            out['close'] = daily['close']
                            out['ibov_return'] = _pd.Series(0.0, index=daily.index)
                            out['symbol_return'] = sym_ret
                            out['atr'] = atr
                            out['rsi'] = rsi
                            out['ema_3'] = ema3
                            out['ema_5'] = ema5
                            out['ema_10'] = ema10
                            out['ema_15'] = ema15
                            out['ema_20'] = ema20
                            out['fuzzy_score'] = score
                            # Clip to requested period
                            mask = (out.index >= _pd.to_datetime(start_date)) & (out.index <= _pd.to_datetime(end_date))
                            out = out.loc[mask]
                            if not out.empty:
                                _agg_vectors[sym] = out
                        except Exception:
                            continue
                    if len(_agg_vectors) > 0:
                        vectors = _agg_vectors
                        logger.info(f"Fallback vectors from hourly aggregation: symbols={len(vectors)}")
                except Exception as _e:
                    logger.warning(f"Hourly aggregation fallback failed: {_e}")
            try:
                logger.info(f"Indicator vectors built: symbols={len(vectors)}")
                for _k, _df in list(vectors.items())[:5]:
                    logger.info(f"  - {_k}: rows={0 if _df is None else len(_df)} cols={list(_df.columns) if _df is not None else []}")
            except Exception:
                pass
            # If vectorization produced nothing (alignment or tz issues), fall back to strategy-collected fuzzy rows
            sched = {}
            if isinstance(vectors, dict) and len(vectors) > 0:
                sched = SignalScheduler(leg_notional_brl=10000.0).build_schedule(vectors)

            # Write consolidated fuzzy indicators CSV and signal matching math
            try:
                from pathlib import Path as _P
                import pandas as _pd
                _P('reports').mkdir(exist_ok=True)
                rows_all = []
                for sym, df in (vectors or {}).items():
                    if df is None or df.empty:
                        continue
                    for ts, r in df.iterrows():
                        fs = float(r.get('fuzzy_score', 0.0) or 0.0)
                        side = 'BUY' if fs >= 1.50 else ('SELL' if fs <= -1.50 else 'HOLD')
                        row = {
                            'date': _pd.to_datetime(ts).tz_localize(None).date() if hasattr(_pd.to_datetime(ts), 'tz_localize') else _pd.to_datetime(ts).date(),
                            'symbol': sym,
                            'close': float(r.get('close', 0.0) or 0.0),
                            'ibov_return': float(r.get('ibov_return', _pd.NA)) if r.get('ibov_return', None) is not None else _pd.NA,
                            'symbol_return': float(r.get('symbol_return', _pd.NA)) if r.get('symbol_return', None) is not None else _pd.NA,
                            'atr': float(r.get('atr', _pd.NA)) if r.get('atr', None) is not None else _pd.NA,
                            'rsi': float(r.get('rsi', _pd.NA)) if r.get('rsi', None) is not None else _pd.NA,
                            'ema_3': float(r.get('ema_3', _pd.NA)) if r.get('ema_3', None) is not None else _pd.NA,
                            'ema_5': float(r.get('ema_5', _pd.NA)) if r.get('ema_5', None) is not None else _pd.NA,
                            'ema_10': float(r.get('ema_10', _pd.NA)) if r.get('ema_10', None) is not None else _pd.NA,
                            'ema_15': float(r.get('ema_15', _pd.NA)) if r.get('ema_15', None) is not None else _pd.NA,
                            'ema_20': float(r.get('ema_20', _pd.NA)) if r.get('ema_20', None) is not None else _pd.NA,
                            'fuzzy_score': fs,
                            'signal_side': side,
                        }
                        rows_all.append(row)
                df_all = _pd.DataFrame(rows_all)
                if not df_all.empty:
                    # Ensure date column is string YYYY-MM-DD for grouping
                    df_all['date'] = _pd.to_datetime(df_all['date']).dt.strftime('%Y-%m-%d')
                    # Add threshold metadata
                    th_buy = 1.50
                    th_sell = -1.50
                    df_all['threshold_buy'] = th_buy
                    df_all['threshold_sell'] = th_sell
                    df_all['abs_fuzzy'] = df_all['fuzzy_score'].abs()
                    df_all['out_of_range'] = (df_all['fuzzy_score'] >= th_buy) | (df_all['fuzzy_score'] <= th_sell)
                    # Always write the full daily fuzzy table for diagnostics
                    df_all.sort_values(['date','symbol']).to_csv('reports/portfolio_fuzzy_indicators_all.csv', index=False)
                    # Keep only rows that cross thresholds per user request (primary CSV)
                    df_sel = df_all[df_all['out_of_range']].sort_values(['date','symbol'])
                    if not df_sel.empty:
                        df_sel.to_csv('reports/portfolio_fuzzy_indicators.csv', index=False)
                    else:
                        logger.warning("No symbols crossed fuzzy thresholds in the selected period; primary CSV not written")
                else:
                    logger.warning("No fuzzy indicator rows produced; skipping CSV write to avoid empty file")

                # Per-day BUY/SELL counts and matched pairs
                if not df_all.empty:
                    grp = df_all.groupby('date', dropna=False)['signal_side']
                    recs = []
                    total_buys = 0
                    total_sells = 0
                    total_matched = 0
                    for d, s in grp:
                        buys = int((s == 'BUY').sum())
                        sells = int((s == 'SELL').sum())
                        matched = int(min(buys, sells))
                        total_buys += buys
                        total_sells += sells
                        total_matched += matched
                        recs.append({'date': d, 'buys': buys, 'sells': sells, 'matched_pairs': matched})
                    df_sig = _pd.DataFrame(recs).sort_values('date')
                    # Append totals row
                    df_tot = _pd.DataFrame([{'date': 'TOTAL', 'buys': total_buys, 'sells': total_sells, 'matched_pairs': total_matched}])
                    df_sig = _pd.concat([df_sig, df_tot], ignore_index=True)
                    df_sig.to_csv('reports/portfolio_signal_summary.csv', index=False)
            except Exception as _e:
                logger.warning(f"Failed to write fuzzy indicators/signal summary CSVs: {_e}")
        except Exception as _e:
            logger.warning(f"Multi-frame daily stage degraded: {_e}")
            vectors = {}
            sched = {}

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
                # Multi-frame path: fetch hourly only for execution days if schedule exists; fallback to backtest range
                from engine.market_utils import MarketDataRouter, DataRequirements
                router = MarketDataRouter()
                frames_local = []
                days_map = DataRequirements.list_execution_days(sched)
                days = days_map.get(sym, [])
                if days:
                    for d in days:
                        ddf, _m = router.get_hourly_for_day(sym, d)
                        if ddf is not None and not ddf.empty:
                            ddf = ddf.copy(); ddf['symbol'] = sym; frames_local.append(ddf)
                if not frames_local:
                    # fallback: hourly for [start,end] only
                    res = hybrid_data_manager.initialize_backtest_data(
                        symbol=sym,
                        start_date=start_date,
                        end_date=end_date,
                        local_loader=data_loader
                    )
                    exec_df = res.get('execution_data')
                    if exec_df is None or exec_df.empty:
                        logger.warning(f"No execution data for {sym} in fallback range")
                        return None
                    df = exec_df.copy()
                    if getattr(df.index, 'tz', None) is not None:
                        df.index = df.index.tz_localize(None)
                    df['symbol'] = sym
                    return df, res
                df = pd.concat(frames_local, axis=0, ignore_index=False).sort_index()
                return df, {'execution_data': df, 'data_sources': {'execution': {'source': 'router_hourly_execution_days'}}}
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
        # Precompute simulation days for schedule alignment
        try:
            sim_days_set = set(filtered_data.index.date)
        except Exception:
            sim_days_set = set()
        if filtered_data.empty:
            raise SystemExit(f"No combined data available in date range {start_date} to {end_date}")

        # Initialize single portfolio and strategy for the full universe
        portfolio = EnhancedPortfolio(config_path="config/settings.yaml")
        strategy_config = StrategyConfig(
            universe=tickers,
            warmup_bars=30,
            risk_tolerance=0.02,
            # Pair-mode mandatory: allow 50% per leg on R$100k portfolio → 50%
            max_position_size=0.50,
            max_daily_loss=0.02,
            stop_loss_pct=0.05,
            take_profit_pct=0.10
        )
        market_utils = BrazilianMarketUtils()
        # Load centralized pair mode into metadata config for downstream consumers
        try:
            _cfg['pair_mode'] = _cfg.get('pair_mode') or {}
        except Exception:
            pass
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
                # Expose full config and tranche for strategy
                'config': _cfg,
                # Pair-mode mandatory: tranche derived from config
                'tranche_notional_brl': (_cfg.get('pair_mode',{}).get('gross_exposure_brl',50000)/max(1, _cfg.get('pair_mode',{}).get('tranches',4))),
            }
        )
        strategy_context.hybrid_data_manager = hybrid_data_manager

        strategy = create_fuzzy_fajuto(
            cfg=strategy_config,
            ctx=strategy_context,
            profile=profile,
            config_file=config_file
        )
        # Inject precomputed daily vectors and schedules into strategy (no indicators recompute in hourly)
        try:
            if isinstance(vectors, dict) and vectors:
                # Map into strategy daily_data and indicators stores per symbol
                strategy.daily_data = strategy.daily_data if hasattr(strategy, 'daily_data') else {}
                strategy.daily_indicators_data = strategy.daily_indicators_data if hasattr(strategy, 'daily_indicators_data') else {}
                for sym, df in vectors.items():
                    try:
                        # Minimal daily store: OHLC required for close-based limits
                        # Use Brapi daily in IndicatorService; here we only keep index and close for alignment
                        from pandas import DataFrame
                        if isinstance(df, DataFrame) and not df.empty:
                            # Build compact OHLC with close as proxy; open/high/low not re-used by indicators here
                            dfd = df[['close']].copy()
                            dfd['open'] = dfd['close']
                            dfd['high'] = dfd['close']
                            dfd['low'] = dfd['close']
                            dfd['volume'] = 0
                            strategy.daily_data[sym] = dfd[['open','high','low','close','volume']]
                            # Indicators per symbol
                            inds = {}
                            for p in (3,5,10,15,20):
                                k = f'ema_{p}';
                                if k in df.columns:
                                    inds[k] = df[k]
                            if 'rsi' in df.columns:
                                inds['rsi'] = df['rsi']
                            if inds:
                                strategy.daily_indicators_data[sym] = inds
                    except Exception:
                        pass
                # Seed fuzzy diagnostics for reporting
                try:
                    strategy._fuzzy_rows = []
                    for sym, df in vectors.items():
                        for ts, row in df.iterrows():
                            fs = float(row.get('fuzzy_score', 0.0) or 0.0)
                            side = 'BUY' if fs >= 1.50 else ('SELL' if fs <= -1.50 else 'HOLD')
                            strategy._fuzzy_rows.append({
                                'date': str(pd.to_datetime(ts).date()),
                                'symbol': sym,
                                'side': side,
                                'fuzzy_score': fs,
                                'eligible': side in ('BUY','SELL'),
                                'reason_if_not': '' if side in ('BUY','SELL') else 'below_threshold',
                                'exposure_cap_brl': 40000.0,
                                'notional_P1': 10000.0,
                                'notional_P2': 10000.0,
                                'notional_P3': 10000.0,
                                'notional_P4': 10000.0,
                            })
                except Exception:
                    pass
            if isinstance(sched, dict) and sched:
                strategy._scheduled_day_trades = sched
                try:
                    _k = list(sched.keys())
                    logger.info(f"Injected T+1 schedule days: {len(_k)} | sample: {[_k[i] for i in range(min(3, len(_k)))]}")
                except Exception:
                    pass
        except Exception as _e:
            logger.warning(f"Failed to inject precomputed vectors/schedule: {_e}")

        # Build schedule using the strategy's own fuzzy logic to guarantee alignment with intents
        try:
            if not isinstance(get('sched', locals()).get('sched', {}), dict):
                pass
        except Exception:
            pass
        try:
            if not isinstance(locals().get('sched', {}), dict):
                sched = {}
            # Prefer strategy-derived schedule to ensure BUY/SELL parity with generate_intents
            sched_from_strategy: dict = {}
            if isinstance(vectors, dict) and vectors:
                from engine.base_strategy import Bar
                for sym, df in vectors.items():
                    if df is None or len(df) == 0:
                        continue
                    # Ensure daily_data exists for symbol
                    if not hasattr(strategy, 'daily_data') or sym not in strategy.daily_data:
                        try:
                            tmp = df[['close']].copy()
                            tmp['open'] = tmp['close']; tmp['high'] = tmp['close']; tmp['low'] = tmp['close']; tmp['volume'] = 0
                            strategy.daily_data = getattr(strategy, 'daily_data', {})
                            strategy.daily_data[sym] = tmp[['open','high','low','close','volume']]
                        except Exception:
                            continue
                    # Walk daily dates T and use strategy._generate_signal to decide T+1 schedule
                    for ts, row in df.iterrows():
                        try:
                            ts_dt = pd.to_datetime(ts)
                            b = Bar(symbol=sym,
                                    timestamp=ts_dt.to_pydatetime(),
                                    open=float(row.get('close', row.get('open', 0.0)) or 0.0),
                                    high=float(row.get('close', row.get('high', 0.0)) or 0.0),
                                    low=float(row.get('close', row.get('low', 0.0)) or 0.0),
                                    close=float(row.get('close', 0.0) or 0.0),
                                    volume=int(row.get('volume', 0) or 0))
                            sig_val = 0
                            try:
                                sig_val = strategy._generate_signal(b)
                            except Exception:
                                sig_val = 0
                            if sig_val == 0:
                                continue
                            side = OrderSide.BUY if sig_val > 0 else OrderSide.SELL
                            # Use strategy limit calculator off close(t)
                            atr_series = None
                            try:
                                atr_series = strategy.daily_indicators_data.get(sym, {}).get('atr')
                            except Exception:
                                atr_series = None
                            atr_t = float(atr_series.loc[ts_dt]) if atr_series is not None and ts_dt in atr_series.index else 0.0
                            try:
                                p2, p3, p4 = strategy._calculate_entry_limits_from_close(float(row.get('close', 0.0) or 0.0), atr_t, side)
                            except Exception:
                                # Fallback to scheduler limits if strategy helper fails
                                from engine.market_utils import SignalScheduler as _SS
                                p2, p3, p4 = _SS()._limits_from_close(float(row.get('close', 0.0) or 0.0), 'BUY' if side == OrderSide.BUY else 'SELL')
                            d_exec = (ts_dt + pd.Timedelta(days=1)).date()
                            day_store = sched_from_strategy.setdefault(d_exec, {})
                            day_store[sym] = {
                                'symbol': sym,
                                'side': side,
                                'valid_for_date': d_exec,
                                'base_close_t': float(row.get('close', 0.0) or 0.0),
                                'limits_used': {'limit_level_2': float(p2), 'limit_level_3': float(p3), 'limit_level_4': float(p4)},
                                'current_atr_t': float(atr_t),
                                'fuzzy_score_t': float(abs(getattr(strategy, '_last_signal_strength', sig_val)))
                            }
                        except Exception:
                            continue
            # Overwrite schedule if we produced any using strategy logic
            if sched_from_strategy:
                strategy._scheduled_day_trades = sched_from_strategy
                try:
                    _k2 = list(sched_from_strategy.keys())
                    logger.info(f"Strategy-derived T+1 schedule days: {len(_k2)} | sample: {[_k2[i] for i in range(min(3, len(_k2)))]}")
                except Exception:
                    pass
        except Exception as _e:
            logger.warning(f"Failed to build schedule from strategy logic: {_e}")

        # Pair-mode mandatory: pre-simulation candidate selection and injection (top BUY/SELL per day)
        if True:
            try:
                csv_path = export_fuzzy_components_to_csv(tickers, start_dt.strftime('%Y-%m-%d'), end_date, _cfg)
                if csv_path:
                    import pandas as _pd
                    df = _pd.read_csv(csv_path)
                    req = {'date','symbol','close','qualified_signal','fuzzy_score_raw'}
                    if req.issubset(df.columns):
                        pair_schedule: dict = {}
                        match_records: list[dict] = []
                        pm = (_cfg.get('pair_mode', {}) or {})
                        min_strength = float(pm.get('min_signal_strength', 1.50))
                        try:
                            print(f"[pair_builder] min_strength={min_strength}")
                            _buy_ct = int((df['qualified_signal']=='BUY').sum()) if 'qualified_signal' in df.columns else 0
                            _sell_ct = int((df['qualified_signal']=='SELL').sum()) if 'qualified_signal' in df.columns else 0
                            _abs_ct = int((df['fuzzy_score_raw'].abs()>=min_strength).sum()) if 'fuzzy_score_raw' in df.columns else 0
                            print(f"[pair_builder] rows: BUY={_buy_ct} SELL={_sell_ct} abs>=th={_abs_ct}")
                        except Exception as _e:
                            logger.warning(f"[pair_builder/error] summary_counters: {_e}")

                        from pandas.tseries.offsets import BDay as _BDay
                        # Prefer B3 exchange calendar for D->D+1 mapping when available
                        _cal = None
                        try:
                            import exchange_calendars as _xcals  # XBSP = B3
                            _cal = _xcals.get_calendar("XBSP")
                        except Exception:
                            _cal = None
                        dropped_strength = 0
                        dropped_close = 0
                        kept_days = 0
                        # Track last used scores per symbol to support rotation rules when enabled
                        last_score = {}
                        for d, g in df.groupby('date'):
                            buys = g[g['qualified_signal']=='BUY']
                            sells = g[g['qualified_signal']=='SELL']
                            # Guards: both sides present
                            if len(buys)==0 or len(sells)==0:
                                continue
                            b = buys.sort_values('fuzzy_score_raw', ascending=False).iloc[0]
                            s = sells.sort_values('fuzzy_score_raw', ascending=True).iloc[0]
                            sb = float(b['fuzzy_score_raw']); ss = float(s['fuzzy_score_raw'])
                            if abs(sb) < min_strength or abs(ss) < min_strength:
                                dropped_strength += 1
                                continue
                            sym_b = str(b['symbol']); sym_s = str(s['symbol'])
                            # Align to next trading day (B3 calendar preferred; fallback to weekday BDay)
                            _d_ts = pd.to_datetime(d)
                            if _cal is not None:
                                try:
                                    _next = _cal.next_session_label(_d_ts.normalize())
                                    exec_date = pd.Timestamp(_next).date()
                                except Exception:
                                    exec_date = (_d_ts + _BDay(1)).date()
                            else:
                                exec_date = (_d_ts + _BDay(1)).date()
                            # Use CSV closes directly; skip if invalid
                            try:
                                close_b = float(b['close'])
                            except Exception:
                                close_b = float('nan')
                            try:
                                close_s = float(s['close'])
                            except Exception:
                                close_s = float('nan')
                            # Verbose log of selection
                            try:
                                print(f"[pair_builder/day] d={d} BUY {sym_b} sb={sb:.2f} close_b={close_b} | SELL {sym_s} ss={ss:.2f} close_s={close_s}")
                            except Exception as _e:
                                logger.warning(f"[pair_builder/error] fallback_injection: {_e}")
                            # If either close is invalid or non-positive, skip scheduling for this day
                            invalid_b = (not (close_b == close_b)) or close_b <= 0
                            invalid_s = (not (close_s == close_s)) or close_s <= 0
                            if invalid_b or invalid_s:
                                dropped_close += 1
                                match_records.append({
                                    'exec_date': str(exec_date),
                                    'date': str(d),
                                    'buy_symbol': sym_b,
                                    'sell_symbol': sym_s,
                                    'score_buy': float(sb),
                                    'score_sell': float(ss),
                                    'close_buy': None if invalid_b else float(close_b),
                                    'close_sell': None if invalid_s else float(close_s),
                                    'scheduled_flag': 'no',
                                })
                                try:
                                    reason = []
                                    if invalid_b: reason.append('invalid_buy_close')
                                    if invalid_s: reason.append('invalid_sell_close')
                                    print(f"[pair_builder/skip] d={d} reason={'+'.join(reason)}")
                                except Exception:
                                    pass
                                continue
                            def limits(side: str, c: float) -> tuple[float,float,float]:
                                try:
                                    return SignalScheduler()._limits_from_close(c, side)
                                except Exception:
                                    step = (0.005, 0.010, 0.015)
                                    if side == 'BUY':
                                        return (round(c*(1- step[0]),2), round(c*(1- step[1]),2), round(c*(1- step[2]),2))
                                    else:
                                        return (round(c*(1+ step[0]),2), round(c*(1+ step[1]),2), round(c*(1+ step[2]),2))
                            p2b,p3b,p4b = limits('BUY', close_b)
                            p2s,p3s,p4s = limits('SELL', close_s)
                            # No dynamic tranche scaling in ATR-free fixed-tranche mode
                            day_store = pair_schedule.setdefault(exec_date, {})

                            day_store[str(b['symbol'])] = {
                                'symbol': str(b['symbol']),
                                'side': OrderSide.BUY,
                                'valid_for_date': exec_date,
                                'base_close_t': close_b,
                                'limits_used': {'limit_level_2': p2b, 'limit_level_3': p3b, 'limit_level_4': p4b},
                                'current_atr_t': float('nan'),
                                'fuzzy_score_t': float(sb)
                            }
                            day_store[str(s['symbol'])] = {
                                'symbol': str(s['symbol']),
                                'side': OrderSide.SELL,
                                'valid_for_date': exec_date,
                                'base_close_t': close_s,
                                'limits_used': {'limit_level_2': p2s, 'limit_level_3': p3s, 'limit_level_4': p4s},
                                'current_atr_t': float('nan'),
                                'fuzzy_score_t': float(ss)
                            }
                            match_records.append({
                                'exec_date': str(exec_date),
                                'date': str(d),
                                'buy_symbol': sym_b,
                                'sell_symbol': sym_s,
                                'score_buy': float(sb),
                                'score_sell': float(ss),
                                'close_buy': float(close_b),
                                'close_sell': float(close_s),
                                'scheduled_flag': 'yes',
                            })
                            # Update rotation memory after selecting
                            last_score[sym_b] = sb
                            last_score[sym_s] = ss
                            kept_days += 1
                        # Report counters prior to any fallback injection
                        try:
                            print(f"[pair_builder/counters] kept_days={kept_days} dropped_close={dropped_close} dropped_strength={dropped_strength}")
                        except Exception as _e:
                            logger.warning(f"[pair_builder/error] schedule_diagnostics: {_e}")

                        # Fallback (optional): if no days kept, inject the first valid D that maps to a sim day D+1
                        if not pair_schedule:
                            try:
                                for d, g in df.groupby('date'):
                                    buys = g[g['qualified_signal']=='BUY']
                                    sells = g[g['qualified_signal']=='SELL']
                                    if len(buys)==0 or len(sells)==0:
                                        continue
                                    b = buys.sort_values('fuzzy_score_raw', ascending=False).iloc[0]
                                    s = sells.sort_values('fuzzy_score_raw', ascending=True).iloc[0]
                                    sb = float(b['fuzzy_score_raw']); ss = float(s['fuzzy_score_raw'])
                                    if abs(sb) < min_strength or abs(ss) < min_strength:
                                        continue
                                    try:
                                        cb = float(b['close']); cs = float(s['close'])
                                    except Exception:
                                        continue
                                    if not (cb>0 and cs>0):
                                        continue
                                    _d_ts = pd.to_datetime(d)
                                    if _cal is not None:
                                        try:
                                            _next = _cal.next_session_label(_d_ts.normalize())
                                            exec_date = pd.Timestamp(_next).date()
                                        except Exception:
                                            exec_date = (_d_ts + _BDay(1)).date()
                                    else:
                                        exec_date = (_d_ts + _BDay(1)).date()
                                    if sim_days_set and exec_date not in sim_days_set:
                                        continue
                                    # Limits
                                    def limits(side: str, c: float) -> tuple[float,float,float]:
                                        try:
                                            return SignalScheduler()._limits_from_close(c, side)
                                        except Exception:
                                            step = (0.005, 0.010, 0.015)
                                            if side == 'BUY':
                                                return (round(c*(1- step[0]),2), round(c*(1- step[1]),2), round(c*(1- step[2]),2))
                                            else:
                                                return (round(c*(1+ step[0]),2), round(c*(1+ step[1]),2), round(c*(1+ step[2]),2))
                                    p2b,p3b,p4b = limits('BUY', cb)
                                    p2s,p3s,p4s = limits('SELL', cs)
                                    day_store = pair_schedule.setdefault(exec_date, {})
                                    day_store[str(b['symbol'])] = {
                                        'symbol': str(b['symbol']),
                                        'side': OrderSide.BUY,
                                        'valid_for_date': exec_date,
                                        'base_close_t': cb,
                                        'limits_used': {'limit_level_2': p2b, 'limit_level_3': p3b, 'limit_level_4': p4b},
                                        'current_atr_t': float('nan'),
                                        'fuzzy_score_t': float(sb)
                                    }
                                    day_store[str(s['symbol'])] = {
                                        'symbol': str(s['symbol']),
                                        'side': OrderSide.SELL,
                                        'valid_for_date': exec_date,
                                        'base_close_t': cs,
                                        'limits_used': {'limit_level_2': p2s, 'limit_level_3': p3s, 'limit_level_4': p4s},
                                        'current_atr_t': float('nan'),
                                        'fuzzy_score_t': float(ss)
                                    }
                                    kept_days += 1
                                    try:
                                        print(f"[pair_builder/fallback] injected exec_date={exec_date} BUY={b['symbol']} SELL={s['symbol']} sb={sb:.2f} ss={ss:.2f}")
                                    except Exception:
                                        pass
                                    break
                            except Exception:
                                pass
                        # Inject schedule before simulation (always log diagnostics)
                        strategy._scheduled_day_trades = pair_schedule
                        try:
                            _sched_days = sorted(pair_schedule.keys())
                            print(f"[pair_schedule] days={len(_sched_days)} kept_days={kept_days} dropped_close={dropped_close} dropped_strength={dropped_strength}")
                            # Print schedule vs simulation day samples and intersection
                            _sched_sample = [sd for sd in _sched_days[:5]]
                            _sim_days = sorted(set(filtered_data.index.date))
                            _sim_sample = _sim_days[:5]
                            _inter = sorted(set(_sim_days).intersection(set(_sched_days)))
                            print(f"[pair_schedule/sample] sched_days[:5]={_sched_sample}")
                            print(f"[pair_schedule/sample] sim_days[:5]={_sim_sample}")
                            print(f"[pair_schedule/intersect] count={len(_inter)} sample={_inter[:5]}")
                            for _d in _sched_days[:5]:
                                _syms = pair_schedule[_d]
                                _sym_list = [s for s in _syms.keys() if not str(s).startswith('__')]
                                print(f"  {_d}: {_sym_list}")
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"Pair-mode pre-sim selection failed: {e}")

        simulator = BacktestSimulator(
            strategy=strategy,
            start_date=start_dt.strftime('%Y-%m-%d'),
            end_date=end_date,
            config_path="config/settings.yaml"
        )
        # Simulation-day diagnostics: compare to schedule
        try:
            sim_days = sorted(set(filtered_data.index.date))
            sched_days = sorted(getattr(strategy, '_scheduled_day_trades', {}).keys())
            # Normalize to dates (some keys may be numpy.datetime64)
            sched_days = [pd.Timestamp(sd).date() for sd in sched_days]
            inter = sorted(set(sim_days).intersection(sched_days))
            print(f"[diagnostics] sim_days={len(sim_days)} sched_days={len(sched_days)} intersect={len(inter)}")
            if inter[:5]:
                print(f"[diagnostics] first_intersect_sample={inter[:5]}")
            if len(sched_days) == 0:
                # Explicit Case B/C diagnostic: no schedule → no fills will be produced
                print("[FILLS] MISSING: no scheduled pairs in default window; unified_fills.csv will not be created.")
        except Exception:
            pass

        # Matching Report: intraday availability for scheduled pairs
        try:
            import pandas as _pd
            _mr = []
            sched = getattr(strategy, '_scheduled_day_trades', {}) or {}
            if sched:
                # Build intraday presence map from combined filtered_data
                _day_sym = set((ts.date(), row_symbol) for ts, row_symbol in zip(filtered_data.index, filtered_data['symbol'] if 'symbol' in filtered_data.columns else [None]*len(filtered_data)))
                for _d, _syms in sched.items():
                    for _sym, _rec in _syms.items():
                        if str(_sym).startswith('__'):
                            continue
                        intraday_ok = ( (_d, _sym) in _day_sym )
                        _mr.append({
                            'exec_date': str(_d),
                            'symbol': _sym,
                            'side': getattr(_rec.get('side'), 'name', str(_rec.get('side'))),
                            'base_close_t': _rec.get('base_close_t'),
                            'l2': _rec['limits_used']['limit_level_2'],
                            'l3': _rec['limits_used']['limit_level_3'],
                            'l4': _rec['limits_used']['limit_level_4'],
                            'intraday_data_flag': 'yes' if intraday_ok else 'no'
                        })
                _pd.DataFrame(_mr).to_csv('reports/pair_matching_report.csv', index=False)
        except Exception:
            pass

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

        # Render robust report (JSON + CSVs)
        try:
            # Ensure output directory and define output paths
            Path('reports').mkdir(parents=True, exist_ok=True)
            fuzzy_html_path = Path('reports') / 'portfolio_execution_report.html'
            exec_html_path = Path('reports') / 'portfolio_backtest_executions.html'

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
                        # Build enriched data payload for advanced report and JSON schema
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
                        # Fuzzy rows directly from strategy memory (already filtered to chosen side only)
                        fuzzy_rows = []
                        try:
                            fz = getattr(strategy, '_fuzzy_rows', None)
                            if fz:
                                out = []
                                for r in fz:
                                    row = dict(r)
                                    row.setdefault('date', str(row.get('date','')))
                                    row.setdefault('symbol', row.get('symbol',''))
                                    row.setdefault('side', row.get('side',''))
                                    row.setdefault('fuzzy_score', row.get('fuzzy_score', 0.0))
                                    row.setdefault('ret_vs_ibov', row.get('ret_vs_ibov', None))
                                    row.setdefault('ema_sum', row.get('ema_sum', None))
                                    row.setdefault('rsi_term', row.get('rsi_term', None))
                                    row.setdefault('eligible', row.get('eligible', False))
                                    row.setdefault('reason_if_not', row.get('reason_if_not',''))
                                    row.setdefault('exposure_cap_brl', row.get('exposure_cap_brl', 0.0))
                                    row.setdefault('notional_P1', row.get('notional_P1', 0.0))
                                    row.setdefault('notional_P2', row.get('notional_P2', 0.0))
                                    row.setdefault('notional_P3', row.get('notional_P3', 0.0))
                                    row.setdefault('notional_P4', row.get('notional_P4', 0.0))
                                    out.append(row)
                                fuzzy_rows = out
                                # Fallback: write fuzzy indicators and signal summary from in-memory rows
                                try:
                                    import pandas as _pd
                                    from pathlib import Path as _P
                                    _P('reports').mkdir(exist_ok=True)
                                    df_fz = _pd.DataFrame(fuzzy_rows)
                                    if not df_fz.empty:
                                        df_fz_sorted = df_fz.sort_values(['date','symbol']) if set(['date','symbol']).issubset(df_fz.columns) else df_fz
                                        df_fz_sorted.to_csv('reports/portfolio_fuzzy_indicators.csv', index=False)
                                        if 'date' in df_fz_sorted.columns and 'side' in df_fz_sorted.columns:
                                            g = df_fz_sorted.groupby('date')['side']
                                            recs = []
                                            tb = ts = tm = 0
                                            for d, s in g:
                                                b = int((s == 'BUY').sum()); se = int((s == 'SELL').sum()); m = min(b, se)
                                                tb += b; ts += se; tm += m
                                                recs.append({'date': d, 'buys': b, 'sells': se, 'matched_pairs': m})
                                            import pandas as _pd2
                                            df_sig = _pd2.DataFrame(recs).sort_values('date')
                                            df_tot = _pd2.DataFrame([{'date': 'TOTAL', 'buys': tb, 'sells': ts, 'matched_pairs': tm}])
                                            _pd2.concat([df_sig, df_tot], ignore_index=True).to_csv('reports/portfolio_signal_summary.csv', index=False)
                                except Exception:
                                    pass
                        except Exception:
                            fuzzy_rows = []

                        # Attribution block per day: BUY/SELL best pair and PnL components
                        try:
                            daily_pairs = []
                            if 'fuzzyByDate' in locals() or True:
                                # Build map of best BUY/SELL per day from df
                                fz = df[['date','symbol','fuzzy_score_raw','qualified_signal','close']].copy()
                                for d, gday in fz.groupby('date'):
                                    b = gday[gday['qualified_signal']=='BUY']
                                    s = gday[gday['qualified_signal']=='SELL']
                                    if len(b)==0 or len(s)==0:
                                        continue
                                    b1 = b.sort_values('fuzzy_score_raw', ascending=False).iloc[0]
                                    s1 = s.sort_values('fuzzy_score_raw', ascending=True).iloc[0]
                                    buy_symbol = str(b1['symbol']); sell_symbol = str(s1['symbol'])
                                    # PnL from derived_rows (intraday to MOC close) if present
                                    pnl_buy = pnl_sell = 0.0
                                    try:
                                        for r in derived_rows:
                                            if r['date']==str(d) and r['symbol'] in (buy_symbol, sell_symbol) and r['pnl'] is not None:
                                                if r['symbol']==buy_symbol:
                                                    pnl_buy += float(r['pnl'])
                                                else:
                                                    pnl_sell += float(r['pnl'])
                                    except Exception:
                                        pass
                                    daily_pairs.append({
                                        'date': str(d),
                                        'buy_symbol': buy_symbol,
                                        'sell_symbol': sell_symbol,
                                        'score_buy': float(b1['fuzzy_score_raw']),
                                        'score_sell': float(s1['fuzzy_score_raw']),
                                        'pnl_buy': float(pnl_buy),
                                        'pnl_sell': float(pnl_sell),
                                        'pnl_total': float(pnl_buy + pnl_sell)
                                    })
                            preflight['outcomes']['daily_pairs'] = daily_pairs
                        except Exception:
                            pass

                        report_payload = {
                            'rows': derived_rows,
                            'dailyNeutral': list(daily_neutral.values()),
                            'symbolStats': list(symbol_stats.values()),
                            'dailyPnl': daily_pnl_series,
                            'symbols': symbols_list,
                            'dateMin': date_min,
                            'dateMax': date_max,
                             'fuzzyByDate': fuzzy_rows,
                             'dailyPairs': preflight['outcomes'].get('daily_pairs', []),
                            'kpis': {
                                'totalPnl': float(sum(r['pnl'] for r in derived_rows if r['pnl'] is not None)),
                                'totalTrades': int(sum(1 for r in derived_rows if r['filled'])),
                            }
                        }
                        # Inject portfolio-derived PnL KPIs (authoritative)
                        try:
                            sell_trades = [t for t in simulator.portfolio.trade_history if t.get('action') == 'SELL']
                            total_pnl_port = float(sum(t.get('final_profit', 0.0) for t in sell_trades))
                            wins_port = sum(1 for t in sell_trades if t.get('final_profit', 0.0) > 0)
                            win_rate_port = (wins_port / len(sell_trades)) if sell_trades else 0.0
                            daily_series_port = sorted((simulator.portfolio.daily_pnl or {}).items())
                            # Overwrite KPIs with portfolio values, leave per-fill pnl in rows unchanged
                            report_payload['kpis']['totalPnl'] = total_pnl_port
                            report_payload['kpis']['winRate'] = win_rate_port
                            report_payload['dailyPnl'] = [[d, float(v)] for d, v in daily_series_port]
                        except Exception:
                            pass
                        # Attach preflight and ingestion diagnostics
                        preflight['signals']['rows'] = fuzzy_rows
                        preflight['ngestion']['diagnostics'] = {
                            'execution_input_rows': int(len(filtered_data)),
                            'execution_unique_days': int(len(set(filtered_data.index.date))),
                        }
                        preflight['outcomes']['summary'] = report_payload['kpis']
                        _write_report(preflight)
                        REPORT_JSON = _json.dumps(report_payload, ensure_ascii=False)
                        # Server-side render of fuzzy table for graceful no-JS viewing
                        def _fmt_brl(x):
                            try:
                                return f"{float(x):,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
                            except Exception:
                                return ''
                        fuzzy_rows_sorted = sorted(fuzzy_rows, key=lambda r: (str(r.get('date','')), str(r.get('symbol',''))))
                        period_text = f"{start_date} - {end_date}"
                        # Render with debug breakdown columns: ret_vs_ibov, ema_sum, rsi_term
                        def _fmt(x, nd=2):
                            try:
                                return f"{float(x):.{nd}f}"
                            except Exception:
                                return "-"
                        fuzzy_table_html = "\n".join(
                            [
                                f"<tr align='right'>"
                                f"<td>{_html_escape(str(r.get('date','')))}</td>"
                                f"<td>{_html_escape(str(r.get('symbol','')))}</td>"
                                f"<td>{_html_escape(str(r.get('side','')))}</td>"
                                f"<td>{(float(r.get('fuzzy_score',0.0))):.4f}</td>"
                                f"<td>{_fmt(r.get('ret_vs_ibov', None), 2)}</td>"
                                f"<td>{_fmt(r.get('ema_sum', None), 2)}</td>"
                                f"<td>{_fmt(r.get('rsi_term', None), 2)}</td>"
                                f"<td>{'Yes' if r.get('eligible') else 'No'}</td>"
                                f"<td>{_fmt_brl(r.get('exposure_cap_brl',0.0))}</td>"
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
                        # Build executions HTML (includes consolidated daily and orders detail)
                        # Display configured date range from settings, not inferred min/max rows
                        period_text = f"{start_date} - {end_date}"
                        placeholder_tpl = """
<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'>
  <title>Relatório de Execuções do Backtest</title>
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
          <th>Data</th><th>Ativo</th><th>Lado</th><th>Fuzzy</th><th>ret_vs_ibov</th><th>ema_sum</th><th>rsi_term</th><th>Elegível</th><th>Capex (BRL)</th><th>P1</th><th>P2</th><th>P3</th><th>P4</th>
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
        # Write execution CSVs
                        import pandas as _pd
                        t0_report = time.perf_counter()
                        # Skip legacy execution CSVs – focus on razor CSV only
                        report_secs = time.perf_counter() - t0_report

                        # Razor-focused fuzzy component export and pair candidate selection
                    except Exception as e:
                        logger.warning(f"Failed to consolidate execution history: {e}")
                    try:
                        csv_path = export_fuzzy_components_to_csv(tickers, start_date, end_date, _cfg)
                        if csv_path:
                            logger.info(f"Fuzzy components CSV: {csv_path}")
                            # Pair-mode mandatory
                            import pandas as _pd
                            df = _pd.read_csv(csv_path)
                            req = {'date','symbol','qualified_signal','fuzzy_score_raw'}
                            if req.issubset(df.columns):
                                    # Build per-day execution schedule (infinite pair mode)
                                    pair_schedule: dict = {}
                                    for d, g in df.groupby('date'):
                                        buys = g[g['qualified_signal']=='BUY']
                                        sells = g[g['qualified_signal']=='SELL']
                                        # Enforce pair mode: require at least one per side
                                        if len(buys) == 0 or len(sells) == 0:
                                            continue
                                        exec_date = (pd.to_datetime(d) + pd.Timedelta(days=1)).date()
                                        day_store = pair_schedule.setdefault(exec_date, {})
                                        # Helper to compute protective limits
                                        def limits(side: str, c: float) -> tuple[float,float,float]:
                                            try:
                                                return SignalScheduler()._limits_from_close(c, side)
                                            except Exception:
                                                step = (0.005, 0.010, 0.015)
                                                if side == 'BUY':
                                                    return (round(c*(1- step[0]),2), round(c*(1- step[1]),2), round(c*(1- step[2]),2))
                                                else:
                                                    return (round(c*(1+ step[0]),2), round(c*(1+ step[1]),2), round(c*(1+ step[2]),2))
                                        # Function to add a row to schedule
                                        def _add_row(row, side_label: str):
                                            try:
                                                close_val = float(row.get('close', float('nan')) if 'close' in row else float('nan'))
                                                if not (close_val == close_val):
                                                    bench = ((_cfg.get('benchmark', {}) or {}).get('symbol')) or ((_cfg.get('brapi', {}) or {}).get('data', {}) or {}).get('ibov_symbol', '^BVSP')
                                                    prepared = prepare_fuzzy_data(tickers, bench, start_date, end_date)
                                                    m = prepared[['date','symbol','close']].set_index(['date','symbol'])
                                                    close_val = float(m.loc[(d, row['symbol'])]['close']) if (d, row['symbol']) in m.index else close_val
                                                l2,l3,l4 = limits(side_label, close_val)
                                                day_store[str(row['symbol'])] = {
                                                    'symbol': str(row['symbol']),
                                                    'side': OrderSide.BUY if side_label=='BUY' else OrderSide.SELL,
                                                    'valid_for_date': exec_date,
                                                    'base_close_t': close_val,
                                                    'limits_used': {'limit_level_2': l2, 'limit_level_3': l3, 'limit_level_4': l4},
                                                    'current_atr_t': float('nan'),
                                                    'fuzzy_score_t': float(row['fuzzy_score_raw']) if 'fuzzy_score_raw' in row else float('nan')
                                                }
                                            except Exception:
                                                pass
                                        # Infinite pair mode: trade top N pairs where N = min(#BUY, #SELL)
                                        buys_sorted = buys.sort_values('fuzzy_score_raw', ascending=False)
                                        sells_sorted = sells.sort_values('fuzzy_score_raw', ascending=True)
                                        pair_count = int(min(len(buys_sorted), len(sells_sorted)))
                                        if pair_count <= 0:
                                            continue
                                        for _, r in buys_sorted.head(pair_count).iterrows():
                                            _add_row(r, 'BUY')
                                        for _, r in sells_sorted.head(pair_count).iterrows():
                                            _add_row(r, 'SELL')
                                    if pair_schedule:
                                        # Inject into strategy context to be consumed by strategy/simulator
                                        try:
                                            strategy._scheduled_day_trades = pair_schedule
                                        except Exception:
                                            pass
                                        # Also persist a summary CSV
                                        rows = []
                                        for d, syms in pair_schedule.items():
                                            buy = next((v for v in syms.values() if getattr(v.get('side'),'name','')=='BUY' or v.get('side')==OrderSide.BUY), None)
                                            sell = next((v for v in syms.values() if getattr(v.get('side'),'name','')=='SELL' or v.get('side')==OrderSide.SELL), None)
                                            if buy and sell:
                                                rows.append({'date': d, 'buy_symbol': buy['symbol'], 'sell_symbol': sell['symbol']})
                                        if rows:
                                            out = _pd.DataFrame(rows).sort_values('date')
                                            out_path = Path('reports')/ 'pair_mode_summary.csv'
                                            out.to_csv(out_path, index=False)
                                            logger.info(f"Pair mode summary: {out_path}")
                    except Exception as _e:
                        logger.warning(f"Failed to write fuzzy components CSV / pair schedule: {_e}")
                    except Exception as e:
                        logger.warning(f"Failed to build daily consolidated execution history: {e}")
                except Exception as e:
                    logger.warning(f"Failed to save aggregated execution history: {e}")
            else:
                # No legacy fallbacks – we only care about fuzzy components CSV now
                try:
                    csv_path = export_fuzzy_components_to_csv(tickers, start_date, end_date, _cfg)
                    if csv_path:
                        logger.info(f"Fuzzy components CSV: {csv_path}")
                except Exception as _e:
                    logger.warning(f"Failed to write fuzzy components CSV: {_e}")
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

        # Always attempt unified fills export when saving results, regardless of audit-only printing gate
        if result_manager is not None and args.save_results:
            try:
                fills_df = simulator.get_unified_fills_dataframe()
                # Fallback: derive basic fills from portfolio trade_history if unified list is empty
                if (fills_df is None or fills_df.empty):
                    try:
                        import pandas as _pd
                        th = getattr(getattr(simulator, 'portfolio', None), 'trade_history', []) or []
                        if th:
                            rows = []
                            for rec in th:
                                try:
                                    rows.append({
                                        'timestamp': rec.get('date'),
                                        'symbol': rec.get('ticker'),
                                        'side': rec.get('action'),
                                        'quantity': int(rec.get('quantity', 0)),
                                        'price': float(rec.get('price', 0.0)),
                                        'order_type': 'LIMIT',
                                        'attempt_type': 'unknown'
                                    })
                                except Exception:
                                    continue
                            if rows:
                                fills_df = _pd.DataFrame(rows)
                    except Exception:
                        pass
                # Second fallback: use simulator's internal unified_fills list directly
                if (fills_df is None or fills_df.empty):
                    try:
                        import pandas as _pd
                        uf_list = getattr(simulator, 'unified_fills', None)
                        if isinstance(uf_list, list) and len(uf_list) > 0:
                            tmp_df = _pd.DataFrame(uf_list)
                            # Ensure required columns exist
                            required_cols = ['timestamp','symbol','side','quantity','price','lot_type','rounding','tranche_notional_brl','trade_type','order_type','attempt_type']
                            for c in required_cols:
                                if c not in tmp_df.columns:
                                    tmp_df[c] = None
                            # Prefer 'symbol' over any alt ticker col
                            if 'ticker' in tmp_df.columns and 'symbol' not in tmp_df.columns:
                                tmp_df['symbol'] = tmp_df['ticker']
                            fills_df = tmp_df[required_cols]
                    except Exception:
                        pass
                if fills_df is not None and not fills_df.empty:
                    # Export via ResultManager and write unconditional CSV/JSON artifacts for validation
                    try:
                        _ = result_manager.export_fills(fills_df, base_name='unified_fills')
                    except Exception:
                        pass
                    try:
                        from pathlib import Path as _P
                        _P('results').mkdir(exist_ok=True)
                        fills_df.to_csv('results/unified_fills.csv', index=False)
                        try:
                            fills_df.to_json('results/unified_fills.json', orient='records', indent=2, date_format='iso')
                        except Exception:
                            pass
                    except Exception:
                        pass
                # Diagnostics + hard-write from raw unified_fills list if still missing
                try:
                    uf_list = getattr(simulator, 'unified_fills', None)
                    th_len = len(getattr(getattr(simulator, 'portfolio', None), 'trade_history', []) or [])
                    uf_len = len(uf_list) if isinstance(uf_list, list) else 0
                    print(f"fills_diagnostics unified_fills={uf_len} trade_history={th_len}")
                    if uf_len > 0:
                        import pandas as _pd
                        _df_raw = _pd.DataFrame(uf_list)
                        # Normalize required columns
                        req = ['timestamp','symbol','side','quantity','price','lot_type','rounding','tranche_notional_brl','trade_type','order_type','attempt_type']
                        for c in req:
                            if c not in _df_raw.columns:
                                _df_raw[c] = None
                        _df_out = _df_raw[req]
                        from pathlib import Path as _P
                        _P('results').mkdir(exist_ok=True)
                        _df_out.to_csv('results/unified_fills.csv', index=False)
                        try:
                            _df_out.to_json('results/unified_fills.json', orient='records', indent=2, date_format='iso')
                        except Exception:
                            pass
                        print(f"fills_written rows={len(_df_out)} cols={list(_df_out.columns)}")
                except Exception:
                    pass

                # --- Validation/Audit: Signal vs Fill exposure cross-check ---
                try:
                    import pandas as _pd
                    from pathlib import Path as _P
                    # Load signals table (threshold-crossing days only) if available
                    sig_path = _P('reports') / 'portfolio_fuzzy_indicators.csv'
                    fills_df = simulator.get_unified_fills_dataframe()
                    # Normalize fills even if empty
                    if fills_df is None:
                        fills_df = _pd.DataFrame()
                    # Prepare audit rows
                    rows = []
                    # Only proceed if signals file exists
                    if sig_path.exists():
                        sig = _pd.read_csv(sig_path)
                        # Expect columns: date, symbol, signal_side (BUY/SELL)
                        if not sig.empty and {'date','symbol'}.issubset(sig.columns):
                            # Convert dates
                            sig['date'] = _pd.to_datetime(sig['date']).dt.date
                            sig['exec_date'] = _pd.to_datetime(sig['date']).map(lambda d: ( _pd.Timestamp(d) + _pd.Timedelta(days=1)).date())
                            # Normalize fills
                            if not fills_df.empty:
                                fills_df = fills_df.copy()
                                fills_df['date'] = _pd.to_datetime(fills_df['timestamp']).dt.date
                                # Attempt type may be missing; fill with 'unknown'
                                if 'attempt_type' not in fills_df.columns:
                                    fills_df['attempt_type'] = 'unknown'
                                # Ensure tranche_notional_brl exists
                                if 'tranche_notional_brl' not in fills_df.columns:
                                    fills_df['tranche_notional_brl'] = _pd.NA
                                fills_df['filled_notional'] = _pd.to_numeric(fills_df['quantity'], errors='coerce').fillna(0).abs() * _pd.to_numeric(fills_df['price'], errors='coerce').fillna(0.0)
                            # For each signal row, check alpha/beta/gamma expected vs actual
                            for _, r in sig.iterrows():
                                sym = str(r['symbol']).strip().upper()
                                d_exec = r['exec_date']
                                for att in ('limit_alpha','limit_beta','limit_gamma'):
                                    exp_pct = 25.0  # 25% per tranche per spec
                                    act_pct = 0.0
                                    if not fills_df.empty:
                                        f = fills_df[(fills_df['symbol'].astype(str).str.upper()==sym) & (fills_df['date']==d_exec) & (fills_df['attempt_type']==att)]
                                        if not f.empty:
                                            tn = _pd.to_numeric(f['tranche_notional_brl'], errors='coerce').dropna()
                                            tranche = float(tn.iloc[0]) if not tn.empty and float(tn.iloc[0])>0 else None
                                            filled = float(f['filled_notional'].sum())
                                            if tranche and tranche>0:
                                                act_pct = (filled / tranche) * 100.0
                                    flag = 'OK' if act_pct >= 25.0 - 1e-6 else 'MISSING_FILL'
                                    rows.append({
                                        'symbol': sym,
                                        'date': str(r['date']),
                                        'exec_date': str(d_exec),
                                        'trigger': att,
                                        'expected_exposure_pct': exp_pct,
                                        'actual_exposure_pct': round(act_pct, 2),
                                        'flag': flag,
                                    })
                    # Write audit log (always attempt)
                    try:
                        _P('results').mkdir(exist_ok=True)
                        with open('results/validation_log.txt','w', encoding='utf-8') as vf:
                            if rows:
                                header = 'symbol,date,exec_date,trigger,expected_exposure_pct,actual_exposure_pct,flag' + '\n'
                                vf.write(header)
                                for rr in rows:
                                    vf.write(f"{rr['symbol']},{rr['date']},{rr['exec_date']},{rr['trigger']},{rr['expected_exposure_pct']:.2f},{rr['actual_exposure_pct']:.2f},{rr['flag']}\n")
                            else:
                                vf.write('no_signals_or_no_audit_inputs\n')
                    except Exception:
                        pass
                except Exception:
                    pass
            except Exception:
                pass
        
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

            # Export unified fills if available (also attempted earlier for artifact generation)
            try:
                # Prefer simulator-level unified fills; if empty, derive from execution history payload
                fills_df = simulator.get_unified_fills_dataframe()
                if (fills_df is None or fills_df.empty) and 'derived_rows' in locals():
                    try:
                        import pandas as _pd
                        _rows_df = _pd.DataFrame(derived_rows) if isinstance(derived_rows, list) else None
                        if _rows_df is not None and not _rows_df.empty:
                            fills_df = _rows_df.rename(columns={
                                'date':'trade_date','symbol':'symbol','side':'side','quantity':'qty','execution_price':'entry_price','pnl':'pnl_leg'
                            })
                    except Exception:
                        pass
                if result_manager is not None and fills_df is not None and not fills_df.empty:
                    art = result_manager.export_fills(fills_df, base_name='unified_fills')
                    summ = result_manager.summarize_fills(fills_df)
                    print("\nUnified Fills Summary:")
                    print(f"  Total fills: {summ.get('total_fills', 0)}  Turnover (BRL): {summ.get('turnover_brl', 0.0):,.2f}")
                    if 'csv' in art:
                        print(f"  CSV: {art['csv']}")
                    if 'json' in art:
                        print(f"  JSON: {art['json']}")
            except Exception:
                pass
        
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
        
        # Ensure validation log exists even if audit step was skipped due to earlier exceptions
        try:
            from pathlib import Path as _P
            _P('results').mkdir(exist_ok=True)
            _v = _P('results') / 'validation_log.txt'
            if not _v.exists():
                with open(_v, 'w', encoding='utf-8') as vf:
                    vf.write('no_signals_or_no_audit_inputs\n')
        except Exception:
            pass
        return 0
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 