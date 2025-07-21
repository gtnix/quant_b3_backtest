# Developer Guide

### **Project Overview**

You're building a professional-grade backtesting engine for Brazilian stocks (B3). This system will simulate trading strategies with realistic market conditions, including Brazilian-specific costs and regulations.

### **Phase 1: Foundation & Data Pipeline (Week 1-2)**

### **Step 1: Project Setup**

```bash
# Create the project structure
mkdir quant_backtest
cd quant_backtest
mkdir -p data/{raw,processed} engine strategies reports scripts config
touch README.md requirements.txt

```

### **Step 2: Core Dependencies**

Create `requirements.txt`:

```
pandas==2.0.3
numpy==1.24.3
alpha_vantage==2.3.1
pyyaml==6.0
matplotlib==3.7.1
plotly==5.15.0
scipy==1.11.1
joblib==1.3.1

```

### **Step 3: Configuration System**

Create `config/settings.yaml`:

```yaml
# B3 Market Configuration
market:
  trading_hours:
    open: "10:00"
    close: "17:00"
  timezone: "America/Sao_Paulo"

# Transaction Costs (B3 specific)
costs:
  brokerage_fee: 0.0001  # 0.01% typical Brazilian broker
  emolumentos: 0.00005   # B3 fee
  liquidacao: 0.0000275  # Settlement fee
  iss: 0.05              # 5% ISS on brokerage (São Paulo)
  min_brokerage: 5.00    # Minimum broker fee in BRL

# Tax Configuration
taxes:
  swing_trade: 0.15      # 15% on profits
  day_trade: 0.20        # 20% on day trade profits
  exemption_limit: 20000 # Monthly exemption for swing trade

# Portfolio Settings
portfolio:
  initial_cash: 100000   # BRL
  max_positions: 10
  position_sizing: "equal_weight"

```

### **Phase 2: Data Management (Week 2-3)**

### **Step 4: Data Downloader**

Create `scripts/download_data.py`:

```python
"""
Download B3 stock data with proper suffix handling
Uses Alpha Vantage API for B3 market data
"""
import pandas as pd
from pathlib import Path
import yaml
import requests
import time

class B3DataDownloader:
    def __init__(self, api_key):
        self.api_key = api_key
        self.data_path = Path("data/raw")
        self.data_path.mkdir(exist_ok=True)
        self.base_url = "https://www.alphavantage.co/query"

    def download_stock(self, ticker, start_date, end_date):
        """
        Download B3 stock data using Alpha Vantage
        """
        try:
            # Alpha Vantage API call
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': f"{ticker}.SAO",  # B3 suffix for Alpha Vantage
                'apikey': self.api_key,
                'outputsize': 'full'
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if 'Time Series (Daily)' not in data:
                print(f"Error downloading {ticker}: {data.get('Note', 'API limit exceeded')}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
            df.index = pd.to_datetime(df.index)
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])

            # Filter date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]

            # Save with metadata
            filename = self.data_path / f"{ticker}_raw.csv"
            df.to_csv(filename)

            return df

        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            return None

```

### **Step 5: Data Processor**

Create `engine/loader.py`:

```python
"""
Load and preprocess B3 market data
Handle corporate actions, liquidity filters, and data quality
"""
import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    def __init__(self):
        self.raw_path = Path("data/raw")
        self.processed_path = Path("data/processed")
        self.processed_path.mkdir(exist_ok=True)

    def load_and_process(self, ticker, start_date=None, end_date=None):
        """
        Load raw data and apply B3-specific processing
        """
        # Load raw data
        raw_file = self.raw_path / f"{ticker}_raw.csv"
        data = pd.read_csv(raw_file, index_col='Date', parse_dates=True)

        # Filter dates
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]

        # Calculate additional features
        data['Returns'] = data['Adj Close'].pct_change()
        data['Volume_BRL'] = data['Volume'] * data['Close']
        data['Volatility_20D'] = data['Returns'].rolling(20).std()
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()

        # Remove low liquidity days (volume < 1M BRL)
        data = data[data['Volume_BRL'] > 1_000_000]

        # Handle missing data
        data = data.dropna()

        return data

```

### **Phase 3: Backtesting Engine Core (Week 3-4)**

### **Step 6: Portfolio Management**

Create `engine/portfolio.py`:

```python
"""
Portfolio tracker with B3-specific features
Handles positions, cash, and Brazilian tax implications
"""
import pandas as pd
from datetime import datetime
from collections import defaultdict

class Portfolio:
    def __init__(self, initial_cash, tax_config):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}  # {ticker: {'shares': int, 'avg_price': float, 'purchase_date': datetime}}
        self.tax_config = tax_config
        self.trades = []
        self.daily_values = []

    def buy(self, ticker, shares, price, date, costs):
        """Execute buy order with B3 costs"""
        total_cost = (shares * price) + costs

        if total_cost > self.cash:
            return False, "Insufficient funds"

        self.cash -= total_cost

        if ticker in self.positions:
            # Update average price
            current = self.positions[ticker]
            total_shares = current['shares'] + shares
            avg_price = ((current['shares'] * current['avg_price']) +
                        (shares * price)) / total_shares

            self.positions[ticker] = {
                'shares': total_shares,
                'avg_price': avg_price,
                'purchase_date': current['purchase_date']
            }
        else:
            self.positions[ticker] = {
                'shares': shares,
                'avg_price': price,
                'purchase_date': date
            }

        self.trades.append({
            'date': date,
            'ticker': ticker,
            'action': 'BUY',
            'shares': shares,
            'price': price,
            'costs': costs
        })

        return True, "Order executed"

    def sell(self, ticker, shares, price, date, costs):
        """Execute sell order with tax calculation"""
        if ticker not in self.positions:
            return False, "No position"

        position = self.positions[ticker]
        if shares > position['shares']:
            return False, "Insufficient shares"

        # Calculate gross profit
        gross_profit = shares * (price - position['avg_price'])

        # Calculate tax
        is_daytrade = (date - position['purchase_date']).days == 0
        tax_rate = self.tax_config['day_trade'] if is_daytrade else self.tax_config['swing_trade']

        tax_owed = max(0, gross_profit * tax_rate)
        net_proceeds = (shares * price) - costs - tax_owed

        self.cash += net_proceeds

        # Update position
        position['shares'] -= shares
        if position['shares'] == 0:
            del self.positions[ticker]

        self.trades.append({
            'date': date,
            'ticker': ticker,
            'action': 'SELL',
            'shares': shares,
            'price': price,
            'costs': costs,
            'tax': tax_owed,
            'profit': gross_profit
        })

        return True, "Order executed"

    def get_portfolio_value(self, prices):
        """Calculate total portfolio value"""
        positions_value = sum(
            self.positions[ticker]['shares'] * prices.get(ticker, 0)
            for ticker in self.positions
        )
        return self.cash + positions_value

```

### **Step 7: Transaction Cost Analysis**

Create `engine/tca.py`:

```python
"""
B3-specific transaction cost modeling
Includes all fees, taxes, and market impact
"""
class TransactionCostAnalyzer:
    def __init__(self, config):
        self.config = config['costs']

    def calculate_costs(self, order_value, is_buy=True):
        """
        Calculate total transaction costs for B3
        """
        # Percentage-based costs
        brokerage = max(
            order_value * self.config['brokerage_fee'],
            self.config['min_brokerage']
        )

        emolumentos = order_value * self.config['emolumentos']
        liquidacao = order_value * self.config['liquidacao']

        # ISS on brokerage (São Paulo tax)
        iss = brokerage * self.config['iss']

        total_costs = brokerage + emolumentos + liquidacao + iss

        return {
            'total': total_costs,
            'brokerage': brokerage,
            'emolumentos': emolumentos,
            'liquidacao': liquidacao,
            'iss': iss
        }

    def estimate_market_impact(self, order_size, avg_volume, volatility):
        """
        Estimate market impact for large orders
        Based on square-root model
        """
        participation_rate = order_size / avg_volume

        # Simplified impact model
        temporary_impact = 0.1 * volatility * np.sqrt(participation_rate)
        permanent_impact = 0.5 * temporary_impact

        return temporary_impact + permanent_impact

```

### **Phase 4: Strategy Framework (Week 4-5)**

### **Step 8: Base Strategy Class**

Create `engine/strategy.py`:

```python
"""
Base class for all trading strategies
Define the interface that all strategies must implement
"""
from abc import ABC, abstractmethod
import pandas as pd

class Strategy(ABC):
    def __init__(self, parameters=None):
        self.parameters = parameters or {}
        self.signals = pd.DataFrame()

    @abstractmethod
    def generate_signals(self, data):
        """
        Generate trading signals
        Return DataFrame with columns: ['signal', 'position_size']
        signal: 1 (buy), 0 (hold), -1 (sell)
        """
        pass

    @abstractmethod
    def calculate_position_size(self, signal, current_price, portfolio_value):
        """
        Determine how many shares to trade
        """
        pass

    def set_parameters(self, **kwargs):
        """Update strategy parameters"""
        self.parameters.update(kwargs)

```

### **Step 9: Example Strategy - Mean Reversion**

Create `strategies/mean_reversion.py`:

```python
"""
Mean reversion strategy for B3 stocks
Buy when price < SMA - 2*std, Sell when price > SMA + 2*std
"""
import pandas as pd
import numpy as np
from engine.strategy import Strategy

class MeanReversionStrategy(Strategy):
    def __init__(self, lookback=20, entry_std=2, exit_std=1):
        super().__init__()
        self.parameters = {
            'lookback': lookback,
            'entry_std': entry_std,
            'exit_std': exit_std
        }

    def generate_signals(self, data):
        """
        Generate mean reversion signals
        """
        # Calculate bands
        sma = data['Close'].rolling(self.parameters['lookback']).mean()
        std = data['Close'].rolling(self.parameters['lookback']).std()

        upper_band = sma + (self.parameters['exit_std'] * std)
        lower_band = sma - (self.parameters['entry_std'] * std)

        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        signals['sma'] = sma
        signals['upper'] = upper_band
        signals['lower'] = lower_band

        # Entry signals (buy when oversold)
        signals['buy'] = (data['Close'] < lower_band).astype(int)

        # Exit signals (sell when overbought)
        signals['sell'] = (data['Close'] > upper_band).astype(int)

        # Combined signal
        signals['signal'] = signals['buy'] - signals['sell']

        return signals

    def calculate_position_size(self, signal, current_price, portfolio_value):
        """
        Equal weight position sizing
        """
        if signal == 0:
            return 0

        # Use 10% of portfolio per position
        position_value = portfolio_value * 0.1
        shares = int(position_value / current_price)

        return shares

```

### **Phase 5: Backtesting Simulator (Week 5-6)**

### **Step 10: Main Simulator**

Create `engine/simulator.py`:

```python
"""
Main backtesting engine
Coordinates data, strategy, portfolio, and execution
"""
import pandas as pd
import numpy as np
from datetime import datetime

class Backtester:
    def __init__(self, data_loader, portfolio, strategy, tca, config):
        self.data_loader = data_loader
        self.portfolio = portfolio
        self.strategy = strategy
        self.tca = tca
        self.config = config

    def run(self, tickers, start_date, end_date):
        """
        Run backtest for multiple tickers
        """
        results = {
            'portfolio_values': [],
            'trades': [],
            'positions': []
        }

        # Get all data
        all_data = {}
        for ticker in tickers:
            data = self.data_loader.load_and_process(ticker, start_date, end_date)
            if data is not None:
                all_data[ticker] = data

        # Get common dates
        common_dates = None
        for ticker, data in all_data.items():
            if common_dates is None:
                common_dates = set(data.index)
            else:
                common_dates = common_dates.intersection(set(data.index))

        common_dates = sorted(list(common_dates))

        # Main simulation loop
        for date in common_dates:
            daily_prices = {}

            # Process each ticker
            for ticker in tickers:
                if ticker not in all_data:
                    continue

                data = all_data[ticker]
                if date not in data.index:
                    continue

                daily_prices[ticker] = data.loc[date, 'Close']

                # Generate signals
                historical_data = data.loc[:date]
                signals = self.strategy.generate_signals(historical_data)

                if len(signals) > 0:
                    current_signal = signals.iloc[-1]['signal']

                    if current_signal != 0:
                        current_price = daily_prices[ticker]
                        portfolio_value = self.portfolio.get_portfolio_value(daily_prices)

                        # Calculate position size
                        shares = self.strategy.calculate_position_size(
                            current_signal,
                            current_price,
                            portfolio_value
                        )

                        if shares > 0:
                            # Calculate costs
                            order_value = shares * current_price
                            costs = self.tca.calculate_costs(order_value)

                            # Execute trade
                            if current_signal > 0:  # Buy
                                success, msg = self.portfolio.buy(
                                    ticker, shares, current_price, date, costs['total']
                                )
                            else:  # Sell
                                success, msg = self.portfolio.sell(
                                    ticker, shares, current_price, date, costs['total']
                                )

            # Record daily portfolio value
            portfolio_value = self.portfolio.get_portfolio_value(daily_prices)
            results['portfolio_values'].append({
                'date': date,
                'value': portfolio_value,
                'cash': self.portfolio.cash,
                'positions_value': portfolio_value - self.portfolio.cash
            })

        return results

```

### **Phase 6: Performance Analytics (Week 6-7)**

### **Step 11: Performance Metrics**

Create `reports/metrics.py`:

```python
"""
Calculate performance metrics specific to B3 and Brazilian market
"""
import pandas as pd
import numpy as np

class PerformanceAnalyzer:
    def __init__(self, risk_free_rate=0.1175):  # Current SELIC rate
        self.risk_free_rate = risk_free_rate

    def calculate_metrics(self, portfolio_values, benchmark_data=None):
        """
        Calculate comprehensive performance metrics
        """
        # Convert to series
        values = pd.Series(
            [v['value'] for v in portfolio_values],
            index=[v['date'] for v in portfolio_values]
        )

        # Returns
        returns = values.pct_change().dropna()

        metrics = {
            # Basic metrics
            'total_return': (values.iloc[-1] / values.iloc[0] - 1) * 100,
            'annualized_return': self._annualized_return(returns),
            'volatility': returns.std() * np.sqrt(252) * 100,

            # Risk metrics
            'sharpe_ratio': self._sharpe_ratio(returns),
            'sortino_ratio': self._sortino_ratio(returns),
            'max_drawdown': self._max_drawdown(values) * 100,
            'var_95': np.percentile(returns, 5) * 100,

            # Trade metrics
            'win_rate': self._calculate_win_rate(portfolio_values),
            'avg_win_loss_ratio': self._win_loss_ratio(portfolio_values),

            # Brazilian specific
            'real_return': self._real_return(values),  # Adjusted for inflation
            'tax_efficiency': self._tax_efficiency(portfolio_values)
        }

        return metrics

    def _annualized_return(self, returns):
        """Calculate annualized return"""
        days = len(returns)
        total_return = (1 + returns).prod() - 1
        return (1 + total_return) ** (252 / days) - 1

    def _sharpe_ratio(self, returns):
        """Sharpe ratio using SELIC as risk-free rate"""
        excess_returns = returns - self.risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def _max_drawdown(self, values):
        """Calculate maximum drawdown"""
        cumulative = (1 + values.pct_change()).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

```

### **Step 12: Visualization**

Create `reports/plots.py`:

```python
"""
Generate professional trading reports and visualizations
"""
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ReportGenerator:
    def __init__(self, results, metrics):
        self.results = results
        self.metrics = metrics

    def generate_full_report(self, save_path="reports/backtest_report.html"):
        """
        Generate comprehensive HTML report
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Portfolio Value', 'Daily Returns',
                          'Drawdown', 'Trade Distribution',
                          'Monthly Returns', 'Risk Metrics'),
            vertical_spacing=0.1
        )

        # Portfolio value
        dates = [v['date'] for v in self.results['portfolio_values']]
        values = [v['value'] for v in self.results['portfolio_values']]

        fig.add_trace(
            go.Scatter(x=dates, y=values, name='Portfolio Value'),
            row=1, col=1
        )

        # Add more visualizations...

        # Save report
        fig.write_html(save_path)

    def plot_trades_on_chart(self, ticker_data, trades):
        """
        Visualize entry/exit points on price chart
        """
        plt.figure(figsize=(15, 8))

        # Plot price
        plt.plot(ticker_data.index, ticker_data['Close'],
                label='Close Price', alpha=0.7)

        # Plot moving averages
        plt.plot(ticker_data.index, ticker_data['SMA_20'],
                label='SMA 20', alpha=0.5)
        plt.plot(ticker_data.index, ticker_data['SMA_50'],
                label='SMA 50', alpha=0.5)

        # Mark trades
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']

        for trade in buy_trades:
            plt.scatter(trade['date'], trade['price'],
                       color='green', s=100, marker='^', label='Buy')

        for trade in sell_trades:
            plt.scatter(trade['date'], trade['price'],
                       color='red', s=100, marker='v', label='Sell')

        plt.legend()
        plt.title(f'Trading Strategy Execution')
        plt.show()

```

### **Phase 7: Main Execution Script (Week 7)**

### **Step 13: Main Runner**

Create `scripts/run_backtest.py`:

```python
"""
Main script to run backtests
This is what your developer will use to test strategies
"""
import yaml
import pandas as pd
from datetime import datetime
from pathlib import Path

# Import all components
from engine.loader import DataLoader
from engine.portfolio import Portfolio
from engine.tca import TransactionCostAnalyzer
from engine.simulator import Backtester
from strategies.mean_reversion import MeanReversionStrategy
from reports.metrics import PerformanceAnalyzer
from reports.plots import ReportGenerator

def main():
    # Load configuration
    with open('config/settings.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize components
    data_loader = DataLoader()
    portfolio = Portfolio(
        initial_cash=config['portfolio']['initial_cash'],
        tax_config=config['taxes']
    )
    tca = TransactionCostAnalyzer(config)
    strategy = MeanReversionStrategy(lookback=20, entry_std=2, exit_std=1)

    # Create backtester
    backtester = Backtester(
        data_loader=data_loader,
        portfolio=portfolio,
        strategy=strategy,
        tca=tca,
        config=config
    )

    # Define test parameters
    tickers = ['PETR4', 'VALE3', 'ITUB4', 'BBDC4']  # Major B3 stocks
    start_date = '2022-01-01'
    end_date = '2023-12-31'

    # Run backtest
    print("Starting backtest...")
    results = backtester.run(tickers, start_date, end_date)

    # Analyze performance
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.calculate_metrics(results['portfolio_values'])

    # Print results
    print("\\n=== Backtest Results ===")
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")

    # Generate report
    report_gen = ReportGenerator(results, metrics)
    report_gen.generate_full_report()
    print("\\nReport saved to reports/backtest_report.html")

if __name__ == "__main__":
    main()

```

### **Phase 8: Testing & Documentation (Week 8)**

### **Step 14: Create Tests**

Create `tests/test_portfolio.py`:

```python
"""
Unit tests for portfolio management
"""
import unittest
from datetime import datetime
from engine.portfolio import Portfolio

class TestPortfolio(unittest.TestCase):
    def setUp(self):
        self.portfolio = Portfolio(
            initial_cash=100000,
            tax_config={'swing_trade': 0.15, 'day_trade': 0.20}
        )

    def test_buy_order(self):
        success, msg = self.portfolio.buy(
            'PETR4', 100, 30.0, datetime.now(), 10.0
        )
        self.assertTrue(success)
        self.assertEqual(self.portfolio.cash, 96990.0)  # 100000 - (100*30 + 10)

    def test_insufficient_funds(self):
        success, msg = self.portfolio.buy(
            'PETR4', 10000, 30.0, datetime.now(), 10.0
        )
        self.assertFalse(success)
        self.assertEqual(msg, "Insufficient funds")

```

### **Step 15: Documentation**

Create comprehensive `README.md`:

```markdown
# B3 Quantitative Backtesting System

## Overview
Professional backtesting system for Brazilian stock market (B3) with realistic cost modeling and tax calculations.

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Download data: `python scripts/download_data.py`
3. Run backtest: `python scripts/run_backtest.py`

## Features
- Accurate B3 transaction costs (brokerage, emolumentos, ISS)
- Brazilian tax calculations (15% swing trade, 20% day trade)
- Multiple strategy support
- Professional performance metrics
- HTML reports with interactive charts

## Project Structure
[Include the directory structure here]

## Adding New Strategies
1. Create new file in `strategies/`
2. Inherit from `Strategy` base class
3. Implement `generate_signals()` and `calculate_position_size()`
4. Update `run_backtest.py` to use your strategy

## Configuration
Edit `config/settings.yaml` to adjust:
- Initial capital
- Transaction costs
- Tax rates
- Position sizing rules

## Important Notes for B3
- All tickers need ".SAO" suffix for Alpha Vantage
- Market hours: 10:00 - 17:00 São Paulo time
- Minimum lot size: 100 shares
- Settlement: T+2

```

### **Key Instructions for Your Developer:**

1. **Start Simple**: First get data downloading working, then portfolio tracking, then add strategies
2. **Test Everything**: Each component should be tested independently before integration
3. **Use Version Control**: Commit after each working component
4. **Ask Questions**: Trading concepts like Sharpe ratio, drawdown, etc. - ask if unclear
5. **Validate Against Real Brokers**: Compare calculated costs with real Brazilian broker statements
6. **Performance First**: Use vectorized operations in pandas, avoid loops in backtesting logic
7. **Document as You Go**: Add docstrings and comments explaining financial logic

### **Common Pitfalls to Avoid:**

- Don't forget the `.SA` suffix for B3 tickers
- Account for Brazilian market holidays
- Handle stock splits and dividends properly
- Consider liquidity - some B3 stocks have low volume
- Remember tax calculations affect net returns significantly

This system will give you professional-grade backtesting capabilities specifically tailored for the Brazilian market. The modular design allows easy expansion and strategy development.