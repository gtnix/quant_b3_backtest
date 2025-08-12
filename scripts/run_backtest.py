#!/usr/bin/env python3
"""
Run backtest with automatic data download and comprehensive data checking.

This script:
1. Checks for missing data across all sources (tickers, SGS, IBOV)
2. Automatically downloads missing data if available
3. Runs the backtest with complete data
4. Provides detailed reporting on data status

Usage:
    python scripts/run_backtest.py [--strategy STRATEGY] [--start-date START_DATE] [--end-date END_DATE] [--tickers TICKERS] [--no-download]

Examples:
    # Run backtest with automatic data download
    python scripts/run_backtest.py --strategy momentum --tickers PETR4,VALE3,ITUB4
    
    # Run backtest without downloading missing data
    python scripts/run_backtest.py --strategy momentum --tickers PETR4,VALE3,ITUB4 --no-download
    
    # Run backtest with custom date range
    python scripts/run_backtest.py --strategy momentum --start-date 2023-01-01 --end-date 2023-12-31 --tickers PETR4,VALE3,ITUB4
"""

import argparse
import sys
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Hierarchical test runner")
    parser.add_argument("--suite", choices=["all", "core", "signal", "mechanical", "smoke", "data"], default="smoke")
    args, extra = parser.parse_known_args()

    base = ["tests/core", "tests/signal", "tests/mechanical", "tests/data", "tests/smoke"]
    mapping = {
        "all": base,
        "core": ["tests/core", "-m", "core"],
        "signal": ["tests/signal", "-m", "signal"],
        "mechanical": ["tests/mechanical", "-m", "mechanical"],
        "data": ["tests/data", "-m", "data_quality"],
        "smoke": ["tests/smoke", "-m", "smoke or core"],
    }
    cmd = [sys.executable, "-m", "pytest"] + mapping[args.suite] + extra
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
import sys
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add the engine directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'engine'))

from loader import DataLoader
from simulator import BacktestSimulator
from base_strategy import BaseStrategy


def setup_logging(level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run backtest with automatic data download',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_backtest.py --strategy momentum --tickers PETR4,VALE3,ITUB4
  python scripts/run_backtest.py --strategy momentum --tickers PETR4,VALE3,ITUB4 --no-download
  python scripts/run_backtest.py --strategy momentum --start-date 2023-01-01 --end-date 2023-12-31 --tickers PETR4,VALE3,ITUB4
        """
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        help='Strategy class name to use for backtesting'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date in format YYYY-MM-DD (default: 1 year ago)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date in format YYYY-MM-DD (default: today)'
    )
    
    parser.add_argument(
        '--tickers',
        type=str,
        required=False,
        help='Comma-separated list of ticker symbols (ignored if portfolio.csv is present)'
    )
    
    parser.add_argument(
        '--no-download',
        action='store_true',
        help='Disable automatic data download'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def get_default_dates():
    """Get default start and end dates."""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)
    return start_date, end_date


def print_data_status(data_status):
    """Print comprehensive data status."""
    print("\n" + "="*60)
    print("COMPREHENSIVE DATA STATUS")
    print("="*60)
    
    # Ticker data status
    ticker_status = data_status['tickers']
    print(f"\n📊 TICKER DATA:")
    print(f"   Missing tickers: {ticker_status['summary']['missing_tickers_count']}")
    print(f"   Tickers with gaps: {ticker_status['summary']['tickers_with_gaps_count']}")
    print(f"   Total missing days: {ticker_status['summary']['total_missing_days']}")
    
    if ticker_status['missing_tickers']:
        print(f"   Missing tickers: {', '.join(ticker_status['missing_tickers'])}")
    
    if ticker_status['tickers_with_gaps']:
        print("   Tickers with recent gaps:")
        for gap in ticker_status['tickers_with_gaps'][:5]:  # Show first 5
            print(f"     {gap['ticker']}: {gap['missing_days']} days since {gap['last_date']}")
        if len(ticker_status['tickers_with_gaps']) > 5:
            print(f"     ... and {len(ticker_status['tickers_with_gaps']) - 5} more")
    
    # SGS data status
    sgs_status = data_status['sgs']
    print(f"\n🏦 SGS DATA:")
    print(f"   Has data: {sgs_status['has_data']}")
    print(f"   Needs download: {sgs_status['needs_download']}")
    if sgs_status['available_series']:
        print(f"   Available series: {sgs_status['available_series']}")
    if sgs_status['missing_series']:
        print(f"   Missing series: {sgs_status['missing_series']}")
    
    # IBOV data status
    ibov_status = data_status['ibov']
    print(f"\n📈 Benchmark (^BVSP) DATA:")
    print(f"   Has data: {ibov_status['has_data']}")
    print(f"   Needs download: {ibov_status['needs_download']}")
    if ibov_status['data_range']:
        print(f"   Data range: {ibov_status['data_range']['start']} to {ibov_status['data_range']['end']}")
    if ibov_status.get('is_up_to_date'):
        print(f"   Status: Up to date")
    elif ibov_status.get('missing_count'):
        print(f"   Missing days: {ibov_status['missing_count']}")
    
    # Summary
    summary = data_status['summary']
    print(f"\n📋 SUMMARY:")
    print(f"   Any missing data: {summary['any_missing_data']}")
    print(f"   Total missing tickers: {summary['total_missing_tickers']}")
    print(f"   Total tickers with gaps: {summary['total_tickers_with_gaps']}")
    print(f"   Total missing days: {summary['total_missing_days']}")
    print(f"   SGS needs download: {summary['sgs_needs_download']}")
    print(f"   IBOV needs download: {summary['ibov_needs_download']}")
    
    print("="*60)


def estimate_download_time(data_status):
    """Estimate download time based on missing data."""
    total_time = 0
    
    # Ticker downloads (assuming 2 seconds per ticker)
    missing_tickers = data_status['tickers']['summary']['missing_tickers_count']
    tickers_with_gaps = data_status['tickers']['summary']['tickers_with_gaps_count']
    total_tickers = missing_tickers + tickers_with_gaps
    total_time += total_tickers * 2
    
    # SGS download (assuming 30 seconds)
    if data_status['sgs']['needs_download']:
        total_time += 30
    
    # IBOV download (assuming 10 seconds)
    if data_status['ibov']['needs_download']:
        total_time += 10
    
    return total_time


def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting backtest with comprehensive data checking")
    
    # Parse tickers – prefer portfolio.csv if available
    def _load_portfolio_symbols() -> list:
        from pathlib import Path as _Path
        import pandas as _pd
        candidates = [
            _Path(__file__).parent.parent / 'portfolio.csv',
            _Path(__file__).parent.parent / 'data' / 'portfolio.csv',
        ]
        for _p in candidates:
            try:
                if _p.exists():
                    _df = _pd.read_csv(_p)
                    if 'symbol' in _df.columns:
                        _syms = [str(s).strip().upper() for s in _df['symbol'].dropna().tolist()]
                        return list(dict.fromkeys([s for s in _syms if s]))
            except Exception:
                pass
        return []

    csv_syms = _load_portfolio_symbols()
    if csv_syms:
        tickers = csv_syms
    elif args.tickers:
        tickers = [ticker.strip().upper() for ticker in args.tickers.split(',') if ticker.strip()]
    else:
        raise SystemExit("portfolio.csv não encontrado e --tickers não informado.")
    logger.info(f"Tickers to process: {tickers}")
    
    # Get dates
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    else:
        start_date, end_date = get_default_dates()
    
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Initialize data loader
    auto_download = not args.no_download
    data_loader = DataLoader(auto_download=auto_download)
    
    # Check comprehensive data status
    logger.info("Checking comprehensive data status...")
    data_status = data_loader.check_all_data(tickers)
    
    # Print data status
    print_data_status(data_status)
    
    # Check if any data is missing
    if data_status['summary']['any_missing_data']:
        if not auto_download:
            logger.error("Missing data detected but auto-download is disabled. Use --no-download to skip this check.")
            return 1
        
        # Estimate download time
        estimated_time = estimate_download_time(data_status)
        print(f"\n⏱️  Estimated download time: {estimated_time} seconds")
        
        # Automatically download missing data
        logger.info("Automatically downloading missing data")
        
        # Download all missing data
        logger.info("Starting comprehensive data download...")
        download_results = data_loader.download_all_missing_data(tickers)
        
        # Print download results
        print("\n" + "="*60)
        print("DOWNLOAD RESULTS")
        print("="*60)
        
        # Ticker results
        ticker_results = download_results['tickers']
        print(f"\n📊 TICKER DOWNLOADS:")
        print(f"   Successful: {len(ticker_results['success'])}")
        print(f"   Failed: {len(ticker_results['failed'])}")
        if ticker_results['success']:
            print(f"   Successful tickers: {', '.join(ticker_results['success'])}")
        if ticker_results['failed']:
            print(f"   Failed tickers: {', '.join(ticker_results['failed'])}")
        
        # SGS results
        sgs_results = download_results['sgs']
        print(f"\n🏦 SGS DOWNLOAD:")
        print(f"   Success: {sgs_results['success']}")
        if sgs_results['message']:
            print(f"   Message: {sgs_results['message']}")
        
        # IBOV results
        ibov_results = download_results['ibov']
        print(f"\n📈 IBOV DOWNLOAD:")
        print(f"   Success: {ibov_results['success']}")
        if ibov_results['message']:
            print(f"   Message: {ibov_results['message']}")
        
        print("="*60)
        
        # Check if all downloads were successful
        all_successful = (
            len(ticker_results['failed']) == 0 and
            sgs_results['success'] and
            ibov_results['success']
        )
        
        if not all_successful:
            logger.warning("Some downloads failed. Backtest may proceed with incomplete data.")
    
    # Run backtest
    logger.info("Starting backtest...")
    
    try:
        # Import strategy class from strategies module
        if args.strategy == 'FuzzyFajutoStrategy':
            from strategies.fuzzy_fajuto_strategy import FuzzyFajutoStrategy
            strategy_class = FuzzyFajutoStrategy
        elif args.strategy == 'EnhancedFuzzyFajutoStrategy':
            from strategies.enhanced_fuzzy_fajuto_strategy import EnhancedFuzzyFajutoStrategy
            strategy_class = EnhancedFuzzyFajutoStrategy
        else:
            strategy_module = __import__(f"strategies.{args.strategy}", fromlist=[args.strategy])
            strategy_class = getattr(strategy_module, args.strategy)
        
        # Initialize portfolio
        from engine.portfolio import EnhancedPortfolio
        portfolio = EnhancedPortfolio(config_path="config/settings.yaml")
        
        # Initialize strategy with new BaseStrategy interface
        if args.strategy in ['EnhancedFuzzyFajutoStrategy', 'FuzzyFajutoStrategy']:
            # Create StrategyConfig for enhanced strategy
            from engine.base_strategy import StrategyConfig, StrategyContext
            from engine.market_utils import BrazilianMarketUtils
            
            # Create strategy configuration
            strategy_config = StrategyConfig(
                universe=tickers,
                warmup_bars=30,  # Enhanced strategy needs more warmup
                risk_tolerance=0.02,
                max_position_size=0.10,
                max_daily_loss=0.02,
                stop_loss_pct=0.05,
                take_profit_pct=0.10
            )
            
            # Debug: Check the created config
            logger.info(f"Created strategy_config type: {type(strategy_config)}")
            logger.info(f"Strategy_config class: {strategy_config.__class__}")
            logger.info(f"Strategy_config module: {strategy_config.__class__.__module__}")
            logger.info(f"StrategyConfig class: {StrategyConfig}")
            logger.info(f"StrategyConfig module: {StrategyConfig.__module__}")
            logger.info(f"Is instance check: {isinstance(strategy_config, StrategyConfig)}")
            
            # Debug: Check the created config
            logger.info(f"Created strategy_config type: {type(strategy_config)}")
            logger.info(f"Strategy_config universe: {strategy_config.universe}")
            
            # Create strategy context
            market_utils = BrazilianMarketUtils()
            strategy_context = StrategyContext(
                data_portal=data_loader,  # Use data loader as data portal
                portfolio=portfolio,
                broker=None,  # Will be handled by simulator
                market_rules=market_utils,
                logger=logging.getLogger(args.strategy),
                metadata={
                    'strategy_config_path': "config/enhanced_strategy_config.yaml"
                }
            )
            
            strategy = strategy_class(
                cfg=strategy_config,
                ctx=strategy_context
            )
        else:
            # For legacy strategies, use old interface
            strategy = strategy_class(
                portfolio=portfolio,
                symbol=tickers[0],  # Use first ticker as primary symbol
                risk_tolerance=0.02,
                config_path="config/settings.yaml"
            )
        
        # Initialize simulator
        simulator = BacktestSimulator(
            strategy=strategy,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            config_path="config/settings.yaml"
        )
        
        # Run backtest for each ticker
        all_results = {}
        for ticker in tickers:
            logger.info(f"Running backtest for {ticker}...")
            try:
                # Load data for this ticker
                ticker_data = data_loader.load_raw_data(ticker)
                if ticker_data is not None and not ticker_data.empty:
                    # Filter data for the specified date range
                    mask = (ticker_data.index >= pd.Timestamp(start_date)) & (ticker_data.index <= pd.Timestamp(end_date))
                    filtered_data = ticker_data.loc[mask]
                    
                    if not filtered_data.empty:
                        results = simulator.run_simulation(filtered_data)
                        all_results[ticker] = results
                    else:
                        logger.warning(f"No data available for {ticker} in date range {start_date} to {end_date}")
                else:
                    logger.warning(f"No data available for {ticker}")
            except Exception as e:
                logger.error(f"Error running backtest for {ticker}: {e}")
        
        # Aggregate results
        if all_results:
            # Calculate aggregate metrics using proper portfolio combination
            # Combine all portfolio values and daily returns
            all_portfolio_values = []
            all_daily_returns = []
            
            for result in all_results.values():
                all_portfolio_values.extend(result.portfolio_values)
                all_daily_returns.extend(result.daily_returns)
            
            # Calculate aggregate total return from combined portfolio values
            if len(all_portfolio_values) >= 2:
                initial_value = all_portfolio_values[0]
                final_value = all_portfolio_values[-1]
                aggregate_total_return = (final_value - initial_value) / initial_value if initial_value > 0 else 0.0
                
                # Calculate proper annualized return using compound growth formula
                trading_days = len(all_daily_returns)
                if trading_days > 0:
                    aggregate_annualized_return = ((final_value / initial_value) ** (252 / trading_days)) - 1
                else:
                    aggregate_annualized_return = 0.0
            else:
                aggregate_total_return = 0.0
                aggregate_annualized_return = 0.0
            
            # Calculate mean of individual metrics for comparison
            sharpe_ratio = np.mean([r.sharpe_ratio for r in all_results.values()])
            max_drawdown = np.mean([r.max_drawdown for r in all_results.values()])
            
            results = {
                'total_return': aggregate_total_return,
                'annualized_return': aggregate_annualized_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'individual_results': all_results
            }
        else:
            results = {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'individual_results': {}
            }
        
        # Print results
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Strategy: {args.strategy}")
        print(f"Tickers: {', '.join(tickers)}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Total Return: {results.get('total_return', 'N/A'):.2%}")
        print(f"Annualized Return: {results.get('annualized_return', 'N/A'):.2%}")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 'N/A'):.2f}")
        print(f"Max Drawdown: {results.get('max_drawdown', 'N/A'):.2%}")
        print("="*60)
        
        logger.info("Backtest completed successfully")
        
        # Generate comprehensive performance reports
        try:
            from engine.performance_metrics import ComprehensivePerformanceAnalysis
            comprehensive_analysis = ComprehensivePerformanceAnalysis(portfolio, strategy=strategy)
            
            # Generate individual ticker reports
            for ticker, result in all_results.items():
                analysis_results = comprehensive_analysis.run_comprehensive_analysis(
                    portfolio_values=result.portfolio_values,
                    daily_returns=result.daily_returns,
                    start_date=start_date
                )
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Generate JSON report for individual ticker
                json_report_path = f"reports/{ticker}_performance_report_{timestamp}.json"
                comprehensive_analysis.generate_performance_report(
                    analysis_results=analysis_results,
                    output_path=json_report_path
                )
                
                # Generate HTML report for individual ticker
                html_report_path = comprehensive_analysis.generate_html_report(
                    portfolio_values=result.portfolio_values,
                    daily_returns=result.daily_returns,
                    start_date=start_date,
                    strategy_name=f"FuzzyFajuto Strategy - {ticker}"
                )
                
                logger.info(f"Generated reports for {ticker}: {json_report_path}, {html_report_path}")
            
            # Generate aggregate report
            if all_results:
                # Combine all portfolio values and returns
                all_portfolio_values = []
                all_daily_returns = []
                
                for result in all_results.values():
                    all_portfolio_values.extend(result.portfolio_values)
                    all_daily_returns.extend(result.daily_returns)
                
                # Run comprehensive analysis for aggregate data
                aggregate_analysis = comprehensive_analysis.run_comprehensive_analysis(
                    portfolio_values=all_portfolio_values,
                    daily_returns=all_daily_returns,
                    start_date=start_date
                )
                
                # Generate aggregate JSON report
                aggregate_json_path = f"reports/aggregate_performance_report_{timestamp}.json"
                comprehensive_analysis.generate_performance_report(
                    analysis_results=aggregate_analysis,
                    output_path=aggregate_json_path
                )
                
                # Generate aggregate HTML report
                aggregate_html_path = comprehensive_analysis.generate_html_report(
                    portfolio_values=all_portfolio_values,
                    daily_returns=all_daily_returns,
                    start_date=start_date,
                    strategy_name="FuzzyFajuto Strategy - Aggregate Results"
                )
                
                logger.info(f"Generated aggregate reports: {aggregate_json_path}, {aggregate_html_path}")
                
        except Exception as e:
            logger.warning(f"Failed to generate comprehensive reports: {e}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 