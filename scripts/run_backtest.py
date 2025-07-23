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
import os
import logging
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
        required=True,
        help='Comma-separated list of ticker symbols'
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
    print(f"\n📈 IBOV DATA:")
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
    
    # Parse tickers
    tickers = [ticker.strip() for ticker in args.tickers.split(',')]
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
        
        # Ask for confirmation
        response = input("\nDo you want to download missing data? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            logger.info("Download cancelled by user")
            return 1
        
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
        # Import strategy class
        strategy_module = __import__(f"strategies.{args.strategy}", fromlist=[args.strategy])
        strategy_class = getattr(strategy_module, args.strategy)
        
        # Initialize strategy
        strategy = strategy_class()
        
        # Initialize simulator
        simulator = BacktestSimulator(
            data_loader=data_loader,
            strategy=strategy,
            start_date=start_date,
            end_date=end_date
        )
        
        # Run backtest
        results = simulator.run_backtest(tickers)
        
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
        return 0
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 