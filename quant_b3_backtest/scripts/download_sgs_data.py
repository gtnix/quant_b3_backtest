#!/usr/bin/env python3
"""
Download and process Banco Central SGS historical data.

This script demonstrates how to use the SGSDataLoader to:
- Fetch historical data for all supported SGS series
- Apply LOCF normalization
- Save processed data for use in backtesting

Usage:
    python scripts/download_sgs_data.py [--start-date DD/MM/YYYY] [--end-date DD/MM/YYYY] [--series-id SERIES_ID]

Examples:
    # Download all series for the last year
    python scripts/download_sgs_data.py
    
    # Download specific series for custom date range
    python scripts/download_sgs_data.py --series-id 11 --start-date 01/01/2023 --end-date 31/12/2023
    
    # Download all series for custom date range
    python scripts/download_sgs_data.py --start-date 01/01/2023 --end-date 31/12/2023
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
import logging

# Add the engine directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'engine'))

from sgs_data_loader import SGSDataLoader


def setup_logging(level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/sgs_download.log')
        ]
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Download and process Banco Central SGS historical data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_sgs_data.py
  python scripts/download_sgs_data.py --series-id 11 --start-date 01/01/2023 --end-date 31/12/2023
  python scripts/download_sgs_data.py --start-date 01/01/2023 --end-date 31/12/2023
        """
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date in format DD/MM/YYYY (default: 1 year ago)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date in format DD/MM/YYYY (default: today)'
    )
    
    parser.add_argument(
        '--series-id',
        type=int,
        choices=[8, 11, 12, 433],
        help='Specific SGS series ID to download (default: all series)'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Force fresh download (ignore cached data)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save processed data to files'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def validate_date_format(date_str):
    """Validate date format DD/MM/YYYY."""
    try:
        datetime.strptime(date_str, '%d/%m/%Y')
        return True
    except ValueError:
        return False


def get_default_dates():
    """Get default start and end dates (last year to today)."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    return start_date.strftime('%d/%m/%Y'), end_date.strftime('%d/%m/%Y')


def print_series_info(loader):
    """Print information about available SGS series."""
    print("\n" + "="*60)
    print("BANCO CENTRAL SGS SERIES")
    print("="*60)
    print(f"{'ID':<5} {'Series Name':<25} {'Description'}")
    print("-"*60)
    
    for series_id, name in loader.SGS_SERIES.items():
        if series_id == 8:
            desc = "Total trading volume on B3"
        elif series_id == 11:
            desc = "Brazilian benchmark interest rate"
        elif series_id == 12:
            desc = "Interbank deposit rate"
        elif series_id == 433:
            desc = "Consumer price index"
        else:
            desc = "Unknown"
        
        print(f"{series_id:<5} {name:<25} {desc}")
    
    print("="*60)


def download_single_series(loader, series_id, start_date, end_date, use_cache, save_processed):
    """Download and process a single SGS series."""
    print(f"\nProcessing Series {series_id}: {loader.SGS_SERIES[series_id]}")
    print(f"Date range: {start_date} to {end_date}")
    
    try:
        # Get data
        data = loader.get_series_data(
            series_id=series_id,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache,
            save_processed=save_processed
        )
        
        if data is not None:
            print(f"✓ Successfully processed {len(data)} data points")
            print(f"  Date range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
            print(f"  Value range: {data['valor'].min():.4f} to {data['valor'].max():.4f}")
            print(f"  Missing values: {data['valor'].isna().sum()}")
            
            # Show sample data
            print(f"  Sample data:")
            print(data.head().to_string())
            
            return True
        else:
            print(f"✗ Failed to process series {series_id}")
            return False
            
    except Exception as e:
        print(f"✗ Error processing series {series_id}: {e}")
        return False


def download_all_series(loader, start_date, end_date, use_cache, save_processed):
    """Download and process all SGS series."""
    print(f"\nDownloading all SGS series from {start_date} to {end_date}")
    
    results = loader.get_all_series_data(
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache,
        save_processed=save_processed
    )
    
    print(f"\nDownload Summary:")
    print(f"{'Series ID':<10} {'Series Name':<20} {'Status':<10} {'Data Points':<12}")
    print("-" * 55)
    
    success_count = 0
    for series_id in loader.SGS_SERIES.keys():
        if series_id in results:
            data = results[series_id]
            print(f"{series_id:<10} {loader.SGS_SERIES[series_id]:<20} {'✓ Success':<10} {len(data):<12}")
            success_count += 1
        else:
            print(f"{series_id:<10} {loader.SGS_SERIES[series_id]:<20} {'✗ Failed':<10} {'N/A':<12}")
    
    print("-" * 55)
    print(f"Successfully processed {success_count} out of {len(loader.SGS_SERIES)} series")
    
    return results


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Get dates
    if args.start_date and args.end_date:
        if not validate_date_format(args.start_date) or not validate_date_format(args.end_date):
            print("Error: Invalid date format. Use DD/MM/YYYY")
            sys.exit(1)
        start_date = args.start_date
        end_date = args.end_date
    else:
        start_date, end_date = get_default_dates()
    
    # Initialize loader
    try:
        loader = SGSDataLoader()
    except Exception as e:
        print(f"Error initializing SGS Data Loader: {e}")
        sys.exit(1)
    
    # Print series information
    print_series_info(loader)
    
    # Determine cache and save settings
    use_cache = not args.no_cache
    save_processed = not args.no_save
    
    print(f"\nSettings:")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Use cache: {use_cache}")
    print(f"  Save processed: {save_processed}")
    
    # Download data
    if args.series_id:
        # Download specific series
        success = download_single_series(
            loader, args.series_id, start_date, end_date, use_cache, save_processed
        )
        if not success:
            sys.exit(1)
    else:
        # Download all series
        results = download_all_series(
            loader, start_date, end_date, use_cache, save_processed
        )
        
        if not results:
            print("No data was successfully downloaded")
            sys.exit(1)
    
    # Show available processed files
    if save_processed:
        print(f"\nAvailable processed files:")
        files = loader.get_available_processed_files()
        if files:
            for file in files:
                print(f"  {file}")
        else:
            print("  No processed files found")
    
    print(f"\n✓ SGS data download completed successfully!")
    print(f"  Data saved to: {loader.data_path}")


if __name__ == "__main__":
    main() 