#!/usr/bin/env python3
"""
Simple test script for Enhanced FuzzyFajuto Strategy with 3 tickers.
This script bypasses the complex SGS data fetching to focus on strategy performance.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from engine.portfolio import EnhancedPortfolio
from engine.simulator import BacktestSimulator
from strategies.enhanced_fuzzy_fajuto_strategy import EnhancedFuzzyFajutoStrategy
from engine.loader import DataLoader

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main function to run the enhanced strategy test."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Configuration
    tickers = ['PETR4', 'VALE3', 'ITUB4']
    start_date = '2025-01-01'
    end_date = '2025-07-24'
    initial_capital = 100000.0
    
    logger.info(f"Starting Enhanced FuzzyFajuto Strategy test")
    logger.info(f"Tickers: {tickers}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Initial Capital: R$ {initial_capital:,.2f}")
    
    try:
        # Initialize data loader
        data_loader = DataLoader(auto_download=False)  # Don't download, use existing data
        
        # Initialize portfolio
        portfolio = EnhancedPortfolio(config_path="config/settings.yaml")
        
        # Initialize strategy with first ticker
        strategy = EnhancedFuzzyFajutoStrategy(
            portfolio=portfolio,
            symbol=tickers[0],
            risk_tolerance=0.02,
            config_path="config/settings.yaml",
            strategy_config_path="config/enhanced_strategy_config.yaml"
        )
        
        # Initialize simulator
        simulator = BacktestSimulator(
            strategy=strategy,
            initial_capital=initial_capital,
            start_date=start_date,
            end_date=end_date,
            config_path="config/settings.yaml"
        )
        
        # Run backtest for each ticker
        all_results = {}
        
        for ticker in tickers:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing ticker: {ticker}")
            logger.info(f"{'='*60}")
            
            try:
                # Load data for this ticker
                ticker_data = data_loader.load_raw_data(ticker)
                
                if ticker_data is None or ticker_data.empty:
                    logger.warning(f"No data available for {ticker}")
                    continue
                
                # Filter data for the specified date range
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                mask = (ticker_data.index >= start_dt) & (ticker_data.index <= end_dt)
                filtered_data = ticker_data.loc[mask]
                
                if filtered_data.empty:
                    logger.warning(f"No data available for {ticker} in date range {start_date} to {end_date}")
                    continue
                
                logger.info(f"Data loaded for {ticker}: {len(filtered_data)} trading days")
                logger.info(f"Date range: {filtered_data.index.min().date()} to {filtered_data.index.max().date()}")
                
                # Update strategy symbol for this ticker
                strategy.symbol = ticker
                
                # Run simulation
                logger.info(f"Running simulation for {ticker}...")
                results = simulator.run_simulation(filtered_data)
                all_results[ticker] = results
                
                # Print individual results
                logger.info(f"\nResults for {ticker}:")
                logger.info(f"  Total Return: {results.total_return:.2%}")
                logger.info(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
                logger.info(f"  Max Drawdown: {results.max_drawdown:.2%}")
                logger.info(f"  Total Trades: {results.total_trades}")
                logger.info(f"  Win Rate: {results.winning_trades/len(results.trade_log)*100:.1f}%" if results.trade_log else "N/A")
                logger.info(f"  Final Portfolio Value: R$ {results.final_portfolio_value:,.2f}")
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                continue
        
        # Print aggregate results
        if all_results:
            logger.info(f"\n{'='*60}")
            logger.info("AGGREGATE RESULTS")
            logger.info(f"{'='*60}")
            
            total_return = np.mean([r.total_return for r in all_results.values()])
            sharpe_ratio = np.mean([r.sharpe_ratio for r in all_results.values()])
            max_drawdown = np.mean([r.max_drawdown for r in all_results.values()])
            total_trades = sum([r.total_trades for r in all_results.values()])
            
            logger.info(f"Average Total Return: {total_return:.2%}")
            logger.info(f"Average Sharpe Ratio: {sharpe_ratio:.2f}")
            logger.info(f"Average Max Drawdown: {max_drawdown:.2%}")
            logger.info(f"Total Trades Across All Tickers: {total_trades}")
            
            # Enhanced strategy specific metrics
            if hasattr(strategy, 'get_execution_statistics'):
                execution_stats = strategy.get_execution_statistics()
                if execution_stats:
                    logger.info(f"\nEnhanced Execution Statistics:")
                    logger.info(f"  Overall Fill Rate: {execution_stats.get('overall_fill_rate', 0):.1%}")
                    logger.info(f"  Total Days: {execution_stats.get('total_days', 0)}")
                    logger.info(f"  Total Attempts: {execution_stats.get('total_attempts', 0)}")
                    logger.info(f"  Total Executed: {execution_stats.get('total_executed', 0)}")
                    logger.info(f"  Total PnL: R$ {execution_stats.get('total_pnl', 0):,.2f}")
                    logger.info(f"  ROI: {execution_stats.get('roi', 0):.2%}")
            
            # Performance summary
            if hasattr(strategy, 'get_performance_summary'):
                perf_summary = strategy.get_performance_summary()
                if 'fuzzy_fajuto_metrics' in perf_summary:
                    fuzzy_metrics = perf_summary['fuzzy_fajuto_metrics']
                    logger.info(f"\nFuzzyFajuto Metrics:")
                    logger.info(f"  Signal Frequency: {fuzzy_metrics.get('signal_frequency', 0):.1%}")
                    logger.info(f"  Buy Signal Ratio: {fuzzy_metrics.get('buy_signal_ratio', 0):.1%}")
                    logger.info(f"  Sell Signal Ratio: {fuzzy_metrics.get('sell_signal_ratio', 0):.1%}")
        
        else:
            logger.error("No successful results to report")
            return 1
        
        logger.info(f"\n{'='*60}")
        logger.info("TEST COMPLETED SUCCESSFULLY")
        logger.info(f"{'='*60}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 