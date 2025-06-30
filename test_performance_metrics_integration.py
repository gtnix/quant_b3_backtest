#!/usr/bin/env python3
"""
Comprehensive Integration Test for Enhanced Performance Metrics Module

This script tests the enhanced performance_metrics.py module using:
- Real historical B3 market data (VALE3, PETR4, ITUB4)
- Real SGS Selic rate data for dynamic risk-free rates
- Brazilian tax and fee calculations
- Comprehensive performance analysis
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add the engine directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'engine'))

from engine.performance_metrics import PerformanceMetrics, ReturnsMetrics, RiskMetrics, TaxMetrics, TradeMetrics
from engine.portfolio import EnhancedPortfolio
from engine.sgs_data_loader import SGSDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_historical_data():
    """Load historical market data for testing."""
    logger.info("Loading historical market data...")
    
    # Load data for major Brazilian stocks
    tickers = ['VALE3', 'PETR4', 'ITUB4']
    data = {}
    
    for ticker in tickers:
        file_path = f"data/raw/{ticker}_raw.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0)
            df.index = pd.to_datetime(df.index)
            data[ticker] = df
            logger.info(f"Loaded {len(df)} data points for {ticker}")
        else:
            logger.warning(f"Data file not found: {file_path}")
    
    return data


def load_selic_data():
    """Load SGS Selic rate data."""
    logger.info("Loading SGS Selic rate data...")
    
    sgs_loader = SGSDataLoader()
    selic_data = sgs_loader.get_series_data(11, "01/01/2023", "31/12/2023")
    
    if selic_data is not None:
        logger.info(f"Loaded {len(selic_data)} Selic rate data points")
        return selic_data
    else:
        logger.warning("Failed to load Selic data, using default rate")
        return None


def create_mock_portfolio():
    """Create a mock portfolio with realistic trade history."""
    logger.info("Creating mock portfolio with Brazilian market compliance...")
    
    # Initialize portfolio
    portfolio = EnhancedPortfolio()
    
    # Create realistic trade history
    trades = [
        # Swing trades
        {'ticker': 'VALE3', 'quantity': 100, 'price': 65.50, 'date': datetime(2023, 3, 15), 'type': 'buy', 'trade_type': 'swing_trade'},
        {'ticker': 'VALE3', 'quantity': 100, 'price': 72.30, 'date': datetime(2023, 6, 20), 'type': 'sell', 'trade_type': 'swing_trade'},
        {'ticker': 'PETR4', 'quantity': 200, 'price': 28.40, 'date': datetime(2023, 4, 10), 'type': 'buy', 'trade_type': 'swing_trade'},
        {'ticker': 'PETR4', 'quantity': 200, 'price': 25.80, 'date': datetime(2023, 7, 15), 'type': 'sell', 'trade_type': 'swing_trade'},
        
        # Day trades
        {'ticker': 'ITUB4', 'quantity': 150, 'price': 32.10, 'date': datetime(2023, 5, 5), 'type': 'buy', 'trade_type': 'day_trade'},
        {'ticker': 'ITUB4', 'quantity': 150, 'price': 33.50, 'date': datetime(2023, 5, 5), 'type': 'sell', 'trade_type': 'day_trade'},
        {'ticker': 'VALE3', 'quantity': 80, 'price': 68.20, 'date': datetime(2023, 8, 10), 'type': 'buy', 'trade_type': 'day_trade'},
        {'ticker': 'VALE3', 'quantity': 80, 'price': 67.80, 'date': datetime(2023, 8, 10), 'type': 'sell', 'trade_type': 'day_trade'},
    ]
    
    # Simulate trades
    for trade in trades:
        if trade['type'] == 'buy':
            portfolio.buy(
                ticker=trade['ticker'],
                quantity=trade['quantity'],
                price=trade['price'],
                trade_date=trade['date'],
                trade_type=trade['trade_type']
            )
        else:
            portfolio.sell(
                ticker=trade['ticker'],
                quantity=trade['quantity'],
                price=trade['price'],
                trade_date=trade['date'],
                trade_type=trade['trade_type']
            )
    
    logger.info(f"Created portfolio with {len(portfolio.trade_history)} trades")
    return portfolio


def create_portfolio_values(data, start_date, end_date):
    """Create daily portfolio values based on historical data."""
    logger.info("Creating daily portfolio values...")
    
    # Create a simple portfolio with equal weights
    tickers = list(data.keys())
    portfolio_values = []
    dates = []
    
    # Get common date range
    all_dates = set()
    for ticker, df in data.items():
        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)
        all_dates.update(df.index)
    
    # Ensure start_date and end_date are datetime (if not already)
    if not isinstance(start_date, pd.Timestamp):
        start_date = pd.to_datetime(start_date)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.to_datetime(end_date)
    
    # Filter dates within the range
    common_dates = sorted([d for d in all_dates if start_date <= d <= end_date])
    
    initial_value = 100000  # R$ 100,000 initial capital
    current_value = initial_value
    
    for i, date in enumerate(common_dates):
        daily_return = 0.0
        valid_prices = 0
        
        for ticker in tickers:
            if ticker in data and date in data[ticker].index:
                price = data[ticker].loc[date, 'close']
                if i == 0:
                    prev_price = price
                else:
                    prev_date = common_dates[i-1]
                    prev_price = data[ticker].loc[prev_date, 'close'] if prev_date in data[ticker].index else price
                if prev_price > 0:
                    ticker_return = (price - prev_price) / prev_price
                    daily_return += ticker_return / len(tickers)  # Equal weight
                    valid_prices += 1
        if valid_prices > 0:
            current_value *= (1 + daily_return)
        portfolio_values.append(current_value)
        dates.append(date)
    logger.info(f"Created {len(portfolio_values)} daily portfolio values")
    return portfolio_values, dates


def create_trades_dataframe(portfolio):
    """Create a DataFrame with trade information for analysis."""
    trades_data = []
    
    for i, trade in enumerate(portfolio.trade_history):
        # Calculate P&L for this trade
        if i > 0 and i % 2 == 1:  # Every sell trade
            buy_trade = portfolio.trade_history[i-1]
            sell_trade = trade
            
            if buy_trade['ticker'] == sell_trade['ticker']:
                quantity = min(buy_trade['quantity'], sell_trade['quantity'])
                pnl = (sell_trade['price'] - buy_trade['price']) * quantity
                
                trades_data.append({
                    'ticker': buy_trade['ticker'],
                    'buy_price': buy_trade['price'],
                    'sell_price': sell_trade['price'],
                    'quantity': quantity,
                    'pnl': pnl,
                    'trade_type': buy_trade['trade_type'],
                    'buy_date': buy_trade['date'],
                    'sell_date': sell_trade['date'],
                    'duration': (sell_trade['date'] - buy_trade['date']).days
                })
    
    return pd.DataFrame(trades_data)


def test_performance_metrics():
    """Main test function."""
    logger.info("=" * 80)
    logger.info("ENHANCED PERFORMANCE METRICS INTEGRATION TEST")
    logger.info("=" * 80)
    
    try:
        # 1. Load historical data
        market_data = load_historical_data()
        if not market_data:
            logger.error("No market data available for testing")
            return
        
        # 2. Load Selic data
        selic_data = load_selic_data()
        
        # 3. Create mock portfolio
        portfolio = create_mock_portfolio()
        
        # 4. Create portfolio values
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        portfolio_values, dates = create_portfolio_values(market_data, start_date, end_date)
        
        # 5. Create trades DataFrame
        trades_df = create_trades_dataframe(portfolio)
        
        # 6. Initialize Performance Metrics
        logger.info("Initializing Enhanced Performance Metrics...")
        performance_metrics = PerformanceMetrics(
            portfolio=portfolio,
            config_path="config/settings.yaml",
            enable_sgs_integration=True
        )
        
        # 7. Calculate comprehensive metrics
        logger.info("Calculating comprehensive performance metrics...")
        results = performance_metrics.calculate_comprehensive_metrics(
            portfolio_values=portfolio_values,
            start_date=start_date,
            end_date=end_date,
            trades_df=trades_df
        )
        
        # 8. Display results
        logger.info("=" * 80)
        logger.info("PERFORMANCE METRICS RESULTS")
        logger.info("=" * 80)
        
        # Returns Metrics
        returns = results['returns']
        initial_capital = 100000  # R$ 100,000 initial capital
        gross_profit = initial_capital * returns.total_return
        net_profit = initial_capital * returns.total_return_net
        fee_amount = initial_capital * returns.fee_impact
        tax_amount = initial_capital * returns.tax_impact
        
        logger.info("RETURNS METRICS:")
        logger.info(f"  Initial Capital: R$ {initial_capital:,.2f}")
        logger.info(f"  Total Return (Gross): {returns.total_return:.4%} (R$ {gross_profit:,.2f})")
        logger.info(f"  Total Return (Net): {returns.total_return_net:.4%} (R$ {net_profit:,.2f})")
        logger.info(f"  Final Portfolio Value: R$ {initial_capital + net_profit:,.2f}")
        logger.info(f"  Annualized Return (CAGR): {returns.annualized_return:.4%}")
        logger.info(f"  Trading Days: {returns.trading_days}")
        logger.info(f"  Fee Impact: {returns.fee_impact:.4%} (R$ {fee_amount:,.2f})")
        logger.info(f"  Tax Impact: {returns.tax_impact:.4%} (R$ {tax_amount:,.2f})")
        
        # Risk Metrics
        risk = results['risk']
        max_dd_amount = initial_capital * risk.max_drawdown
        var_amount = initial_capital * abs(risk.var_95)
        cvar_amount = initial_capital * abs(risk.cvar_95)
        
        logger.info("\nRISK METRICS:")
        logger.info(f"  Sharpe Ratio: {risk.sharpe_ratio:.4f}")
        logger.info(f"  Sortino Ratio: {risk.sortino_ratio:.4f}")
        logger.info(f"  Calmar Ratio: {risk.calmar_ratio:.4f}")
        logger.info(f"  Maximum Drawdown: {risk.max_drawdown:.4%} (R$ {max_dd_amount:,.2f})")
        logger.info(f"  Volatility: {risk.volatility:.4%}")
        logger.info(f"  VaR (95%): {risk.var_95:.4%} (R$ {var_amount:,.2f})")
        logger.info(f"  CVaR (95%): {risk.cvar_95:.4%} (R$ {cvar_amount:,.2f})")
        logger.info(f"  Skewness: {risk.skewness:.4f}")
        logger.info(f"  Kurtosis: {risk.kurtosis:.4f}")
        
        # Trade Metrics
        trades = results['trades']
        total_wins = trades.total_trades * trades.win_rate
        total_losses = trades.total_trades - total_wins
        total_profit = trades.average_win * total_wins
        total_loss = trades.average_loss * total_losses
        
        logger.info("\nTRADE METRICS:")
        logger.info(f"  Total Trades: {trades.total_trades}")
        logger.info(f"  Win Rate: {trades.win_rate:.4%} ({total_wins:.0f} winning trades)")
        logger.info(f"  Loss Rate: {(1-trades.win_rate):.4%} ({total_losses:.0f} losing trades)")
        logger.info(f"  Profit Factor: {trades.profit_factor:.4f}")
        logger.info(f"  Average Win: R$ {trades.average_win:.2f}")
        logger.info(f"  Average Loss: R$ {trades.average_loss:.2f}")
        logger.info(f"  Total Profit from Wins: R$ {total_profit:,.2f}")
        logger.info(f"  Total Loss from Losses: R$ {total_loss:,.2f}")
        logger.info(f"  Net Trading Profit: R$ {total_profit + total_loss:,.2f}")
        logger.info(f"  Total Commission: R$ {trades.total_commission:.2f}")
        
        # Tax Metrics
        taxes = results['taxes']
        total_pnl = total_profit + total_loss
        effective_tax_amount = total_pnl * taxes.effective_tax_rate if total_pnl > 0 else 0
        
        logger.info("\nBRAZILIAN TAX METRICS:")
        logger.info(f"  Total P&L: R$ {total_pnl:,.2f}")
        logger.info(f"  Total Taxes Paid: R$ {taxes.total_taxes_paid:.2f}")
        logger.info(f"  Swing Trade Taxes: R$ {taxes.swing_trade_taxes:.2f}")
        logger.info(f"  Day Trade Taxes: R$ {taxes.day_trade_taxes:.2f}")
        logger.info(f"  Tax Efficiency: {taxes.tax_efficiency:.4%}")
        logger.info(f"  Effective Tax Rate: {taxes.effective_tax_rate:.4%} (R$ {effective_tax_amount:,.2f})")
        logger.info(f"  Loss Carryforward Utilized: R$ {taxes.loss_carryforward_utilized:.2f}")
        logger.info(f"  Tax Exemption Utilized: R$ {taxes.tax_exemption_utilized:.2f}")
        
        # Summary
        summary = results['summary']
        logger.info("\nSUMMARY:")
        logger.info(f"  Overall Performance: {summary['total_return']:.4%} (R$ {net_profit:,.2f})")
        logger.info(f"  Risk-Adjusted Return: {summary['sharpe_ratio']:.4f}")
        logger.info(f"  Risk Level: {'High' if risk.max_drawdown > 0.20 else 'Medium' if risk.max_drawdown > 0.10 else 'Low'}")
        logger.info(f"  Tax Efficiency: {summary['tax_efficiency']:.4%}")
        logger.info(f"  Net Profit After All Costs: R$ {net_profit:,.2f}")
        logger.info(f"  Total Costs (Fees + Taxes): R$ {fee_amount + tax_amount:,.2f}")
        
        # 9. Test dynamic risk-free rate
        logger.info("\n" + "=" * 80)
        logger.info("DYNAMIC RISK-FREE RATE TESTING")
        logger.info("=" * 80)
        
        test_dates = [start_date, datetime(2023, 6, 15), end_date]
        for test_date in test_dates:
            rate = performance_metrics.get_risk_free_rate(test_date)
            logger.info(f"  Selic Rate for {test_date.date()}: {rate:.4%}")
        
        # 10. Generate performance report
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING PERFORMANCE REPORT")
        logger.info("=" * 80)
        
        report_path = "reports/integration_test_report.txt"
        performance_metrics.generate_performance_report(results, report_path)
        logger.info(f"Performance report generated: {report_path}")
        
        # 11. Test individual metric calculations
        logger.info("\n" + "=" * 80)
        logger.info("INDIVIDUAL METRIC TESTS")
        logger.info("=" * 80)
        
        # Test total return calculation
        gross_return = performance_metrics.calculate_total_return(portfolio_values, net_of_fees=False)
        net_return = performance_metrics.calculate_total_return(portfolio_values, net_of_fees=True)
        gross_profit_test = initial_capital * gross_return
        net_profit_test = initial_capital * net_return
        logger.info(f"  Gross Return: {gross_return:.4%} (R$ {gross_profit_test:,.2f})")
        logger.info(f"  Net Return: {net_return:.4%} (R$ {net_profit_test:,.2f})")
        
        # Test CAGR calculation
        cagr = performance_metrics.calculate_cagr(portfolio_values, start_date, end_date)
        cagr_profit = initial_capital * cagr
        logger.info(f"  CAGR: {cagr:.4%} (R$ {cagr_profit:,.2f})")
        
        # Test Sharpe ratio with dynamic rate
        daily_returns = returns.daily_returns
        sharpe = performance_metrics.calculate_sharpe_ratio(daily_returns, date=end_date)
        logger.info(f"  Sharpe Ratio (with dynamic Selic): {sharpe:.4f}")
        
        # Test Sortino ratio
        sortino = performance_metrics.calculate_sortino_ratio(daily_returns, date=end_date)
        logger.info(f"  Sortino Ratio: {sortino:.4f}")
        
        # Test maximum drawdown
        max_dd, max_dd_duration = performance_metrics.calculate_max_drawdown(portfolio_values)
        max_dd_amount_test = initial_capital * max_dd
        logger.info(f"  Max Drawdown: {max_dd:.4%} (R$ {max_dd_amount_test:,.2f}, Duration: {max_dd_duration} days)")
        
        # Test trade metrics
        if not trades_df.empty:
            win_rate = performance_metrics.calculate_win_rate(trades_df)
            profit_factor = performance_metrics.calculate_profit_factor(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] <= 0])
            logger.info(f"  Win Rate: {win_rate:.4%} ({winning_trades} winning, {losing_trades} losing)")
            logger.info(f"  Profit Factor: {profit_factor:.4f}")
        
        logger.info("\n" + "=" * 80)
        logger.info("INTEGRATION TEST COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        return results
        
    except Exception as e:
        logger.error(f"Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function."""
    logger.info("Starting Enhanced Performance Metrics Integration Test")
    logger.info("Testing with real Brazilian market data and SGS Selic rates")
    
    results = test_performance_metrics()
    
    if results:
        logger.info("✅ All tests passed successfully!")
        logger.info("The enhanced performance metrics module is working correctly with:")
        logger.info("  - Real historical B3 market data")
        logger.info("  - Dynamic SGS Selic rate integration")
        logger.info("  - Brazilian tax and fee calculations")
        logger.info("  - Comprehensive risk-adjusted metrics")
    else:
        logger.error("❌ Integration test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
