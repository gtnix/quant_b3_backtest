#!/usr/bin/env python3
"""
Profile Manager Utility

Simple utility to help manage strategy profiles and view results.
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.result_manager import ResultManager


def list_results(strategy_name=None, ticker=None, limit=10):
    """List recent results."""
    manager = ResultManager()
    results = manager.get_results(strategy_name=strategy_name, ticker=ticker, limit=limit)
    
    if results.empty:
        print("No results found.")
        return
    
    print(f"\nRecent Results (limit: {limit}):")
    print("-" * 80)
    print(f"{'Profile':<15} {'Ticker':<8} {'Return':<10} {'Sharpe':<8} {'MaxDD':<10} {'Trades':<8} {'Date':<12}")
    print("-" * 80)
    
    for _, row in results.iterrows():
        print(f"{row['profile']:<15} {row['ticker']:<8} {row['total_return']:>8.2%} "
              f"{row['sharpe_ratio']:>6.2f} {row['max_drawdown']:>8.2%} "
              f"{row['total_trades']:>6.0f} {row['timestamp'][:8]}")


def compare_profiles(strategy_name, ticker=None):
    """Compare profiles for a strategy."""
    manager = ResultManager()
    comparison = manager.compare_profiles(strategy_name, ticker=ticker)
    
    if comparison.empty:
        print(f"No results found for strategy: {strategy_name}")
        return
    
    print(f"\nProfile Comparison - {strategy_name}")
    if ticker:
        print(f"Ticker: {ticker}")
    
    manager.print_summary(comparison)


def show_detailed_result(run_id):
    """Show detailed result for a specific run."""
    manager = ResultManager()
    detailed = manager.get_detailed_result(run_id)
    
    if not detailed:
        print(f"Result not found: {run_id}")
        return
    
    print(f"\nDetailed Result: {run_id}")
    print("-" * 60)
    print(f"Strategy: {detailed['strategy_name']}")
    print(f"Profile: {detailed['profile']}")
    print(f"Ticker: {detailed['ticker']}")
    print(f"Period: {detailed['start_date']} to {detailed['end_date']}")
    print(f"Config File: {detailed.get('config_file', 'default')}")
    print(f"Timestamp: {detailed['timestamp']}")
    
    results = detailed['results']
    print(f"\nPerformance:")
    print(f"  Total Return: {results['total_return']:.2%}")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Win Rate: {results['winning_trades']/max(results['total_trades'], 1):.2%}")
    print(f"  Final Value: R$ {results['final_portfolio_value']:,.2f}")
    print(f"  Total Commission: R$ {results['total_commission']:,.2f}")
    print(f"  Total Taxes: R$ {results['total_taxes']:,.2f}")
    
    if detailed.get('strategy_summary'):
        summary = detailed['strategy_summary']
        print(f"\nStrategy Summary:")
        print(f"  Total Executions: {summary.get('total_executions', 0)}")
        print(f"  Execution History: {summary.get('execution_history_count', 0)}")
        
        if summary.get('fill_rates'):
            print(f"  Fill Rates:")
            for attempt_type, rate in summary['fill_rates'].items():
                print(f"    {attempt_type}: {rate:.2%}")


def main():
    parser = argparse.ArgumentParser(description='Profile Manager Utility')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List recent results')
    list_parser.add_argument('--strategy', help='Filter by strategy name')
    list_parser.add_argument('--ticker', help='Filter by ticker')
    list_parser.add_argument('--limit', type=int, default=10, help='Limit number of results')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare profiles')
    compare_parser.add_argument('strategy', help='Strategy name (e.g., fuzzy_fajuto)')
    compare_parser.add_argument('--ticker', help='Filter by ticker')
    
    # Detail command
    detail_parser = subparsers.add_parser('detail', help='Show detailed result')
    detail_parser.add_argument('run_id', help='Run ID to show details for')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_results(args.strategy, args.ticker, args.limit)
    elif args.command == 'compare':
        compare_profiles(args.strategy, args.ticker)
    elif args.command == 'detail':
        show_detailed_result(args.run_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()