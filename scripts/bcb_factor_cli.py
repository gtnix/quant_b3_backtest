#!/usr/bin/env python3
"""
BCB Daily Factor Command Line Interface

This script provides a command-line interface for retrieving and managing
daily factor data from Banco Central do Brasil (BCB).

Usage:
    python bcb_factor_cli.py fetch --start-date 2024-01-01 --end-date 2024-12-31
    python bcb_factor_cli.py info
    python bcb_factor_cli.py clear-cache
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add the parent directory to the path to import the module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from market_data.bcb_daily_factor import BCBDailyFactorRetriever


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def fetch_command(args):
    """Handle the fetch command."""
    try:
        # Initialize retriever
        retriever = BCBDailyFactorRetriever()
        
        # Parse dates
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)
        
        if start_date >= end_date:
            print("Error: Start date must be before end date")
            sys.exit(1)
        
        print(f"Fetching daily factors from {start_date.date()} to {end_date.date()}")
        
        # Fetch data
        factors_df = retriever.fetch_daily_factors(start_date, end_date)
        
        if factors_df.empty:
            print("No data retrieved for the specified date range")
            return
        
        # Display summary
        print(f"\nRetrieved {len(factors_df)} daily factor records")
        print(f"Date range: {factors_df.index.min().date()} to {factors_df.index.max().date()}")
        print(f"Factor range: {factors_df['factor'].min():.6f} to {factors_df['factor'].max():.6f}")
        
        # Calculate and display risk-free curve summary
        risk_free_curve = retriever.calculate_risk_free_curve(factors_df)
        print(f"Risk-free rate range: {risk_free_curve.min():.4%} to {risk_free_curve.max():.4%}")
        
        # Save to file if requested
        if args.output:
            output_path = args.output
            if output_path.endswith('.csv'):
                factors_df.to_csv(output_path)
            elif output_path.endswith('.parquet'):
                factors_df.to_parquet(output_path)
            else:
                factors_df.to_csv(output_path + '.csv')
            
            print(f"Data saved to {output_path}")
        
        # Display sample data
        if args.verbose:
            print("\nSample data:")
            print(factors_df.head(10).to_string())
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def info_command(args):
    """Handle the info command."""
    try:
        retriever = BCBDailyFactorRetriever()
        
        # Load cached data
        cached_data = retriever.load_cached_factors()
        
        if cached_data is None or cached_data.empty:
            print("No cached data found")
            return
        
        print(f"Cached data summary:")
        print(f"Records: {len(cached_data)}")
        print(f"Date range: {cached_data.index.min().date()} to {cached_data.index.max().date()}")
        print(f"Factor range: {cached_data['factor'].min():.6f} to {cached_data['factor'].max():.6f}")
        print(f"Mean factor: {cached_data['factor'].mean():.6f}")
        print(f"Standard deviation: {cached_data['factor'].std():.6f}")
        
        # Calculate risk-free curve
        risk_free_curve = retriever.calculate_risk_free_curve(cached_data)
        print(f"\nRisk-free curve summary:")
        print(f"Rate range: {risk_free_curve.min():.4%} to {risk_free_curve.max():.4%}")
        print(f"Mean rate: {risk_free_curve.mean():.4%}")
        print(f"Latest rate: {risk_free_curve.iloc[-1]:.4%}")
        
        if args.verbose:
            print("\nRecent data:")
            print(cached_data.tail(10).to_string())
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def clear_cache_command(args):
    """Handle the clear-cache command."""
    try:
        retriever = BCBDailyFactorRetriever()
        
        cache_file = os.path.join(retriever.cache_dir, 'daily_factors.parquet')
        
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"Cache cleared: {cache_file}")
        else:
            print("No cache file found")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def get_rate_command(args):
    """Handle the get-rate command."""
    try:
        retriever = BCBDailyFactorRetriever()
        
        # Parse target date
        target_date = parse_date(args.date)
        
        # Get risk-free rate
        rate = retriever.get_risk_free_rate_for_date(
            target_date, 
            lookback_days=args.lookback_days
        )
        
        if rate is None:
            print(f"No risk-free rate available for {target_date.date()}")
            sys.exit(1)
        
        print(f"Risk-free rate for {target_date.date()}: {rate:.4%}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="BCB Daily Factor Data Retriever CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bcb_factor_cli.py fetch --start-date 2024-01-01 --end-date 2024-12-31
  python bcb_factor_cli.py fetch --start-date 2024-01-01 --end-date 2024-12-31 --output data.csv
  python bcb_factor_cli.py info --verbose
  python bcb_factor_cli.py get-rate --date 2024-06-15
  python bcb_factor_cli.py clear-cache
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Fetch command
    fetch_parser = subparsers.add_parser('fetch', help='Fetch daily factors from BCB API')
    fetch_parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    fetch_parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    fetch_parser.add_argument('--output', help='Output file path (CSV or Parquet)')
    fetch_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    fetch_parser.set_defaults(func=fetch_command)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show information about cached data')
    info_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    info_parser.set_defaults(func=info_command)
    
    # Clear cache command
    clear_parser = subparsers.add_parser('clear-cache', help='Clear cached data')
    clear_parser.set_defaults(func=clear_cache_command)
    
    # Get rate command
    rate_parser = subparsers.add_parser('get-rate', help='Get risk-free rate for specific date')
    rate_parser.add_argument('--date', required=True, help='Target date (YYYY-MM-DD)')
    rate_parser.add_argument('--lookback-days', type=int, default=30, 
                           help='Number of days to look back (default: 30)')
    rate_parser.set_defaults(func=get_rate_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    args.func(args)


if __name__ == '__main__':
    main() 