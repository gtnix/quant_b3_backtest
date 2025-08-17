#!/usr/bin/env python3
"""
Test script for Enhanced FuzzyFajuto Strategy with Three-Attempt Execution System

This script demonstrates how to use the enhanced strategy with:
- Four execution attempts per asset per day
- Percent-based limit order pricing (0.5%, 1.0%, 1.5% from close[T-1])
- Proper Brazilian market simulation
- End-of-day position closure
- Comprehensive execution tracking

Author: Quantitative Trading Specialist
Date: 2025
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import yaml

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.portfolio import EnhancedPortfolio
from strategies.enhanced_fuzzy_fajuto_strategy import EnhancedFuzzyFajutoStrategy
from engine.loader import DataLoader
from engine.simulator import BacktestSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_strategy_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def load_enhanced_config():
    """Load enhanced strategy configuration."""
    config_path = "config/enhanced_strategy_config.yaml"
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Loaded enhanced strategy configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading enhanced config: {e}")
        return None


def create_sample_market_data(symbol: str, start_date: str, end_date: str) -> dict:
    """
    Create sample market data for testing.
    
    Args:
        symbol: Stock symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        Dictionary with market data
    """
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Filter for weekdays only (Brazilian market)
    dates = dates[dates.weekday < 5]
    
    # Create sample OHLC data
    np.random.seed(42)  # For reproducible results
    
    # Start with a base price
    base_price = 50.0
    
    # Generate price data with some trend and volatility
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 0.01))  # Ensure positive prices
    
    # Create OHLC data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Add some intraday volatility
        volatility = price * 0.02  # 2% intraday volatility
        
        open_price = price * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, price) + np.random.uniform(0, volatility)
        low_price = min(open_price, price) - np.random.uniform(0, volatility)
        close_price = price
        
        volume = np.random.randint(100000, 1000000)
        
        data.append({
            'date': date,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    # Create IBOV data (simplified)
    ibov_data = {
        'returns': pd.Series(np.random.normal(0.0005, 0.015, len(df)), index=df.index),
        'prices': pd.Series(np.cumprod(1 + np.random.normal(0.0005, 0.015, len(df))), index=df.index)
    }
    
    return {
        'price_data': df,
        'ibov_data': ibov_data,
        'symbol': symbol
    }


def test_enhanced_strategy():
    """Test the enhanced FuzzyFajuto strategy."""
    logger.info("Starting Enhanced FuzzyFajuto Strategy Test")
    
    # Load configuration
    config = load_enhanced_config()
    if not config:
        logger.error("Failed to load configuration")
        return
    
    # Create sample market data
    symbol = "PETR4"
    start_date = "2024-01-01"
    end_date = "2024-03-31"
    
    logger.info(f"Creating sample market data for {symbol} from {start_date} to {end_date}")
    market_data = create_sample_market_data(symbol, start_date, end_date)
    
    # Initialize portfolio
    initial_cash = 100000  # R$ 100,000
    portfolio = EnhancedPortfolio(config_path="config/settings.yaml")
    
    # Initialize enhanced strategy
    strategy = EnhancedFuzzyFajutoStrategy(
        portfolio=portfolio,
        symbol=symbol,
        risk_tolerance=0.02,
        config_path="config/settings.yaml"
    )
    
    # Update strategy parameters from config
    if 'enhanced_fuzzy_fajuto' in config:
        strategy.update_strategy_parameters(config['enhanced_fuzzy_fajuto'])
    
    logger.info(f"Strategy initialized with parameters: {strategy.get_strategy_parameters()}")
    
    # Test strategy on each trading day
    price_data = market_data['price_data']
    trading_days = price_data.index
    
    logger.info(f"Testing strategy on {len(trading_days)} trading days")
    
    for i, trading_day in enumerate(trading_days):
        if i < 20:  # Skip first 20 days to allow for indicator calculation
            continue
        
        logger.info(f"Processing trading day: {trading_day.date()}")
        
        # Get current day's OHLC data
        current_day_data = price_data.loc[trading_day]
        
        # Create market data for this day
        day_market_data = {
            'price_data': price_data.loc[:trading_day],  # Data up to current day
            'ibov_data': market_data['ibov_data'],
            'current_day_data': {
                'open': current_day_data['open'],
                'high': current_day_data['high'],
                'low': current_day_data['low'],
                'close': current_day_data['close'],
                'volume': current_day_data['volume']
            },
            'timestamp': trading_day
        }
        
        # Generate signals
        signals = strategy.generate_signals(day_market_data)
        
        if signals:
            logger.info(f"Generated {len(signals)} signals for {trading_day.date()}")
            
            # Execute signals
            for signal in signals:
                success = strategy.execute_trade(signal)
                if success:
                    logger.info(f"Successfully executed signal: {signal}")
                else:
                    logger.warning(f"Failed to execute signal: {signal}")
        else:
            logger.debug(f"No signals generated for {trading_day.date()}")
        
        # Manage risk
        current_positions = portfolio.get_position_summary()
        risk_decisions = strategy.manage_risk(current_positions, day_market_data)
        
        if risk_decisions['risk_level'] == 'high':
            logger.warning(f"High risk level detected on {trading_day.date()}")
    
    # Get performance summary
    performance = strategy.get_performance_summary()
    logger.info("Strategy Performance Summary:")
    logger.info(f"Total trades: {len(portfolio.trade_history)}")
    logger.info(f"Final portfolio value: R$ {portfolio.total_value:.2f}")
    logger.info(f"Total return: {((portfolio.total_value / initial_cash) - 1) * 100:.2f}%")
    
    # Get execution statistics
    execution_stats = strategy.get_execution_statistics()
    if execution_stats:
        logger.info("Execution Statistics:")
        logger.info(f"Total days: {execution_stats['total_days']}")
        logger.info(f"Total attempts: {execution_stats['total_attempts']}")
        logger.info(f"Total executed: {execution_stats['total_executed']}")
        logger.info(f"Overall fill rate: {execution_stats['overall_fill_rate']:.1%}")
        logger.info(f"Fill rates by type: {execution_stats['fill_rates_by_type']}")
        logger.info(f"Total PnL: R$ {execution_stats['total_pnl']:.2f}")
        logger.info(f"ROI: {execution_stats['roi']:.2%}")
    
    # Save detailed execution report
    save_execution_report(strategy, execution_stats)


def save_execution_report(strategy, execution_stats):
    """Save detailed execution report."""
    try:
        report_data = []
        
        for summary in strategy.execution_history:
            for attempt in summary.attempts:
                report_data.append({
                    'date': summary.date,
                    'symbol': summary.symbol,
                    'attempt_number': attempt.attempt_number,
                    'execution_type': attempt.metadata.get('execution_type'),
                    'order_type': attempt.order_type.value,
                    'price': attempt.price,
                    'quantity': attempt.quantity,
                    'executed': attempt.executed,
                    'execution_price': attempt.execution_price,
                    'fill_quantity': attempt.fill_quantity,
                    'pnl': attempt.pnl,
                    'open_price': summary.open_price,
                    'close_price': summary.close_price,
                    'high_price': summary.high_price,
                    'low_price': summary.low_price,
                    # 'atr_prev': summary.atr_prev,  # removed
                    'fuzzy_score': summary.metadata.get('fuzzy_score')
                })
        
        df = pd.DataFrame(report_data)
        report_file = f"reports/enhanced_strategy_execution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(report_file, index=False)
        logger.info(f"Execution report saved to {report_file}")
        
        # Save summary statistics
        summary_file = f"reports/enhanced_strategy_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(summary_file, 'w') as f:
            json.dump(execution_stats, f, indent=2, default=str)
        logger.info(f"Summary statistics saved to {summary_file}")
        
    except Exception as e:
        logger.error(f"Error saving execution report: {e}")


def main():
    """Main function to run the enhanced strategy test."""
    logger.info("=" * 60)
    logger.info("Enhanced FuzzyFajuto Strategy Test")
    logger.info("Four-Attempt Execution System with Percent-Based Limits")
    logger.info("=" * 60)
    
    try:
        test_enhanced_strategy()
        logger.info("Enhanced strategy test completed successfully")
    except Exception as e:
        logger.error(f"Error during enhanced strategy test: {e}")
        raise
    
    logger.info("=" * 60)
    logger.info("Test completed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main() 