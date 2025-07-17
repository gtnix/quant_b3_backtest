#!/usr/bin/env python3
"""
Main CLI script for running backtests with HTML report generation

This script provides a command-line interface to run backtests with interactive
HTML/PDF reports using Plotly.

Usage:
    python scripts/run_backtest.py --strategy mean_reversion --tickers PETR4,VALE3 --start-date 2023-01-01 --end-date 2023-12-31
    python scripts/run_backtest.py --strategy mean_reversion --export-pdf --tickers PETR4,VALE3 --start-date 2023-01-01 --end-date 2023-12-31
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.simulator import BacktestSimulator
from engine.loader import DataLoader
from engine.portfolio import EnhancedPortfolio
from engine.tca import TransactionCostAnalyzer
from engine.base_strategy import BaseStrategy
from reports.data_prep import ReportData
from reports.html_renderer import HTMLRenderer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backtest_cli.log')
    ]
)

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run backtests with HTML report generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic HTML report
  python scripts/run_backtest.py --strategy mean_reversion --tickers PETR4,VALE3 --start-date 2023-01-01 --end-date 2023-12-31
  
  # With PDF export
  python scripts/run_backtest.py --strategy mean_reversion --export-pdf --tickers PETR4,VALE3 --start-date 2023-01-01 --end-date 2023-12-31
  
  # With custom output directory
  python scripts/run_backtest.py --strategy mean_reversion --output-dir custom_reports --tickers PETR4,VALE3 --start-date 2023-01-01 --end-date 2023-12-31
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        help='Strategy name to use for backtesting'
    )
    
    parser.add_argument(
        '--tickers',
        type=str,
        required=True,
        help='Comma-separated list of ticker symbols (e.g., PETR4,VALE3,ITUB4)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        required=True,
        help='Start date in YYYY-MM-DD format'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        required=True,
        help='End date in YYYY-MM-DD format'
    )
    
    # Optional arguments
    parser.add_argument(
        '--export-pdf',
        action='store_true',
        help='Export HTML report to PDF'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports',
        help='Directory to save reports (default: reports)'
    )
    
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=100000.0,
        help='Initial capital in BRL (default: 100000.0)'
    )
    
    parser.add_argument(
        '--config-path',
        type=str,
        default='config/settings.yaml',
        help='Path to configuration file (default: config/settings.yaml)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    # Validate dates
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
            
    except ValueError as e:
        print(f"Error: Invalid date format. {e}")
        print("Use YYYY-MM-DD format (e.g., 2023-01-01)")
        sys.exit(1)
    
    # Validate tickers
    if not args.tickers:
        print("Error: At least one ticker must be specified")
        sys.exit(1)
    
    # Validate initial capital
    if args.initial_capital <= 0:
        print("Error: Initial capital must be positive")
        sys.exit(1)
    
    # Validate config file
    if not Path(args.config_path).exists():
        print(f"Error: Configuration file not found: {args.config_path}")
        sys.exit(1)


def load_strategy(strategy_name: str) -> BaseStrategy:
    """
    Load strategy class by name.
    
    Args:
        strategy_name: Name of the strategy to load
        
    Returns:
        BaseStrategy instance
    """
    try:
        # Import strategy module
        strategy_module = __import__(f'strategies.{strategy_name}', fromlist=[strategy_name])
        strategy_class = getattr(strategy_module, strategy_name.title().replace('_', ''))
        
        # Create strategy instance
        strategy = strategy_class()
        
        logger.info(f"Loaded strategy: {strategy_name}")
        return strategy
        
    except (ImportError, AttributeError) as e:
        logger.error(f"Could not load strategy '{strategy_name}': {e}")
        print(f"Error: Strategy '{strategy_name}' not found or invalid")
        print("Available strategies:")
        # List available strategies
        strategies_dir = Path("strategies")
        if strategies_dir.exists():
            for strategy_file in strategies_dir.glob("*.py"):
                if strategy_file.name != "__init__.py":
                    print(f"  - {strategy_file.stem}")
        sys.exit(1)


def run_backtest(args) -> ReportData:
    """
    Run the backtest simulation.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        ReportData instance with simulation results
    """
    logger.info("Starting backtest simulation...")
    
    try:
        # Load strategy
        strategy = load_strategy(args.strategy)
        
        # Parse tickers
        tickers = [ticker.strip() for ticker in args.tickers.split(',')]
        
        # Create simulator
        simulator = BacktestSimulator(
            strategy=strategy,
            initial_capital=args.initial_capital,
            start_date=args.start_date,
            end_date=args.end_date,
            config_path=args.config_path
        )
        
        # Load data
        data_loader = DataLoader()
        
        # Load and process data for each ticker
        all_data = []
        for ticker in tickers:
            ticker_data = data_loader.load_and_process(
                ticker=ticker,
                start_date=args.start_date,
                end_date=args.end_date,
                save_processed=True
            )
            if ticker_data is not None:
                ticker_data['ticker'] = ticker
                all_data.append(ticker_data)
        
        if not all_data:
            print("Error: No data found for the specified tickers and date range")
            sys.exit(1)
        
        # Combine all ticker data
        data = pd.concat(all_data, axis=0)
        
        # Ensure we have a proper DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            # If the index is not a DatetimeIndex, try to convert it
            data.index = pd.to_datetime(data.index)
        
        if data.empty:
            print("Error: No data found for the specified tickers and date range")
            sys.exit(1)
        
        # Run simulation
        logger.info(f"Running simulation for {len(tickers)} tickers from {args.start_date} to {args.end_date}")
        result = simulator.run_simulation(data)
        
        # Convert to ReportData
        report_data = ReportData.from_simulation_result(result)
        
        logger.info("Backtest simulation completed successfully")
        return report_data
        
    except Exception as e:
        logger.error(f"Error during backtest simulation: {e}")
        print(f"Error: {e}")
        sys.exit(1)


def generate_report(args, report_data: ReportData):
    """
    Generate HTML report.
    
    Args:
        args: Parsed command line arguments
        report_data: ReportData instance with simulation results
    """
    logger.info("Generating HTML report...")
    
    try:
        # HTML renderer
        renderer = HTMLRenderer(output_dir=args.output_dir)
        
        # Generate HTML report
        html_path = Path(args.output_dir) / f"backtest_report_{report_data.strategy_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        renderer.render_full_report(report_data, str(html_path))
        
        print(f"\n✅ HTML report generated: {html_path}")
        
        # Export to PDF if requested
        if args.export_pdf:
            pdf_path = Path(html_path).with_suffix('.pdf')
            renderer.export_to_pdf(report_data, str(pdf_path))
            print(f"✅ PDF report generated: {pdf_path}")
            
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        print(f"Error generating report: {e}")
        sys.exit(1)


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Validate arguments
    validate_arguments(args)
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Print minimal header
    print("=" * 60)
    print("QUANT B3 BACKTEST - HTML REPORTING")
    print("=" * 60)
    print(f"Strategy: {args.strategy}")
    print(f"Tickers: {args.tickers}")
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Initial Capital: R$ {args.initial_capital:,.2f}")
    print("=" * 60)
    
    # Run backtest
    report_data = run_backtest(args)
    
    # Generate report
    generate_report(args, report_data)
    
    # Print minimal summary
    print("\n" + "=" * 60)
    print("BACKTEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Total Return: {report_data.total_return * 100:.2f}%")
    print(f"Sharpe Ratio: {report_data.sharpe_ratio:.4f}")
    print(f"Max Drawdown: {report_data.max_drawdown * 100:.2f}%")
    print(f"Total Trades: {report_data.total_trades}")
    print(f"Win Rate: {report_data.winning_trades / report_data.total_trades * 100:.1f}%" if report_data.total_trades > 0 else "Win Rate: N/A")
    
    # Print benchmark metrics if available
    if report_data.benchmark_return != 0.0 or report_data.excess_return != 0.0:
        print("\n" + "-" * 40)
        print("BENCHMARK ANALYSIS")
        print("-" * 40)
        print(f"Benchmark Return: {report_data.benchmark_return * 100:.2f}%")
        print(f"Excess Return: {report_data.excess_return * 100:.2f}%")
        print(f"Information Ratio: {report_data.information_ratio:.4f}")
        print(f"Beta: {report_data.beta:.4f}")
        print(f"Alpha: {report_data.alpha:.6f}")
        print(f"Tracking Error: {report_data.tracking_error:.4f}")
        print(f"Rolling Correlation: {report_data.rolling_correlation:.4f}")
        print(f"Benchmark Sharpe: {report_data.benchmark_sharpe:.4f}")
        print(f"Benchmark Max Drawdown: {report_data.benchmark_max_drawdown * 100:.2f}%")
    
    print("=" * 60)


if __name__ == "__main__":
    main() 