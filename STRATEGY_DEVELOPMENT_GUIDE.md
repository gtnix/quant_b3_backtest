# Strategy Development Guide for Brazilian B3 Backtesting System

## Overview

This guide will help you understand the Brazilian stock market backtesting system and create your own trading strategy. The system is designed for the B3 (Brazilian Stock Exchange) with comprehensive Brazilian market compliance, including taxes, settlement rules, and transaction costs.

## System Architecture

### Core Components

1. **BaseStrategy** (`engine/base_strategy.py`) - Abstract base class for all strategies
2. **EnhancedPortfolio** (`engine/portfolio.py`) - Manages positions, cash, and Brazilian market compliance
3. **DataLoader** (`engine/loader.py`) - Loads and processes B3 market data
4. **BacktestSimulator** (`engine/simulator.py`) - Runs backtests with full Brazilian market simulation
5. **Transaction Cost Analysis** (`engine/tca.py`) - Calculates all Brazilian trading costs
6. **Loss Manager** (`engine/loss_manager.py`) - Handles Brazilian tax loss carryforward rules
7. **Settlement Manager** (`engine/settlement_manager.py`) - Manages T+2 settlement cycles

### Key Brazilian Market Features

- **T+2 Settlement**: Trades settle 2 business days after execution
- **Tax Rates**: 15% for swing trades, 20% for day trades
- **Tax Exemption**: R$ 20,000/month for swing trades
- **Transaction Costs**: B3 fees, brokerage, ISS tax
- **Loss Carryforward**: Perpetual loss tracking per asset and globally

## Creating Your Strategy

### Step 1: Create Your Strategy File

Create a new file in the `strategies/` directory:

```python
# strategies/my_strategy.py

from engine.base_strategy import BaseStrategy, TradingSignal, SignalType, TradeType
import pandas as pd
import numpy as np
from typing import Dict, Any, List

class MyStrategy(BaseStrategy):
    """
    Your custom trading strategy for Brazilian B3 market.
    
    This is a template - implement your own logic!
    """
    
    def __init__(self, portfolio, symbol, risk_tolerance=0.02, config_path="config/settings.yaml"):
        super().__init__(portfolio, symbol, risk_tolerance, config_path, "MyStrategy")
        
        # Add your strategy-specific parameters
        self.lookback_period = 20
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
    def generate_signals(self, market_data: Dict[str, Any]) -> List[TradingSignal]:
        """
        Generate trading signals based on market data.
        
        This is where you implement your trading logic!
        
        Args:
            market_data: Dictionary containing:
                - 'price_data': DataFrame with OHLCV data
                - 'technical_indicators': Pre-calculated indicators
                - 'sgs_data': Brazilian economic data (SELIC, CDI, IPCA)
                - 'interest_rate_env': Current interest rate environment
                - 'inflation_env': Current inflation environment
                
        Returns:
            List of TradingSignal objects
        """
        signals = []
        
        # Extract price data
        price_data = market_data.get('price_data', pd.DataFrame())
        if price_data.empty:
            return signals
            
        # Get the latest data point
        current_data = price_data.iloc[-1]
        current_price = current_data['close']
        
        # Example: Simple moving average crossover strategy
        if len(price_data) >= self.lookback_period:
            # Calculate moving averages
            short_ma = price_data['close'].rolling(window=5).mean().iloc[-1]
            long_ma = price_data['close'].rolling(window=self.lookback_period).mean().iloc[-1]
            
            # Generate signals based on MA crossover
            if short_ma > long_ma and short_ma > current_price * 1.01:  # 1% above current price
                # Buy signal
                signal = TradingSignal(
                    signal_type=SignalType.BUY,
                    ticker=self.symbol,
                    price=current_price,
                    quantity=100,  # Will be adjusted by position sizing
                    confidence=0.7,
                    trade_type=TradeType.SWING_TRADE,
                    metadata={'strategy': 'MA_Crossover', 'short_ma': short_ma, 'long_ma': long_ma}
                )
                signals.append(signal)
                
            elif short_ma < long_ma and short_ma < current_price * 0.99:  # 1% below current price
                # Sell signal
                signal = TradingSignal(
                    signal_type=SignalType.SELL,
                    ticker=self.symbol,
                    price=current_price,
                    quantity=100,  # Will be adjusted by position sizing
                    confidence=0.7,
                    trade_type=TradeType.SWING_TRADE,
                    metadata={'strategy': 'MA_Crossover', 'short_ma': short_ma, 'long_ma': long_ma}
                )
                signals.append(signal)
        
        return signals
    
    def manage_risk(self, current_positions: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement your risk management logic.
        
        Args:
            current_positions: Current portfolio positions
            market_data: Current market data
            
        Returns:
            Risk management decisions
        """
        risk_decisions = {
            'stop_loss_triggered': False,
            'take_profit_triggered': False,
            'position_adjustments': []
        }
        
        # Example: Check stop loss and take profit levels
        for ticker, position in current_positions.items():
            if ticker == self.symbol:
                current_price = market_data.get('price_data', pd.DataFrame()).iloc[-1]['close']
                
                # Stop loss check (5% loss)
                if position['unrealized_pnl_pct'] < -self.risk_metrics.stop_loss_pct:
                    risk_decisions['stop_loss_triggered'] = True
                    risk_decisions['position_adjustments'].append({
                        'ticker': ticker,
                        'action': 'SELL',
                        'reason': 'Stop Loss',
                        'quantity': position['quantity']
                    })
                
                # Take profit check (10% gain)
                elif position['unrealized_pnl_pct'] > self.risk_metrics.take_profit_pct:
                    risk_decisions['take_profit_triggered'] = True
                    risk_decisions['position_adjustments'].append({
                        'ticker': ticker,
                        'action': 'SELL',
                        'reason': 'Take Profit',
                        'quantity': position['quantity']
                    })
        
        return risk_decisions
    
    def execute_trade(self, signal: TradingSignal) -> bool:
        """
        Execute a trade based on the signal.
        
        Args:
            signal: TradingSignal object
            
        Returns:
            True if trade executed successfully, False otherwise
        """
        try:
            # Get current market price
            current_price = signal.price
            
            # Calculate position size based on available cash and risk metrics
            available_cash = self.portfolio.cash
            position_size = self.calculate_position_size(signal, available_cash)
            
            # Check Brazilian market constraints
            if not self.check_brazilian_market_constraints(signal):
                logger.warning(f"Trade rejected due to Brazilian market constraints: {signal}")
                return False
            
            # Execute the trade
            if signal.signal_type == SignalType.BUY:
                success = self.portfolio.buy(
                    ticker=signal.ticker,
                    quantity=position_size,
                    price=current_price,
                    trade_date=signal.timestamp,
                    trade_type=signal.trade_type.value,
                    description=f"{self.strategy_name}: {signal.metadata.get('strategy', 'Unknown')}"
                )
            elif signal.signal_type == SignalType.SELL:
                success = self.portfolio.sell(
                    ticker=signal.ticker,
                    quantity=position_size,
                    price=current_price,
                    trade_date=signal.timestamp,
                    trade_type=signal.trade_type.value,
                    description=f"{self.strategy_name}: {signal.metadata.get('strategy', 'Unknown')}"
                )
            else:
                logger.warning(f"Unknown signal type: {signal.signal_type}")
                return False
            
            if success:
                logger.info(f"Trade executed successfully: {signal}")
                self.log_trade({
                    'signal': signal,
                    'position_size': position_size,
                    'execution_price': current_price
                })
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
```

### Step 2: Understanding Market Data Structure

The `market_data` dictionary contains:

```python
market_data = {
    'price_data': pd.DataFrame({
        'open': [...],
        'high': [...], 
        'low': [...],
        'close': [...],
        'volume': [...],
        'adjusted_close': [...]
    }),
    'technical_indicators': {
        'rsi': [...],
        'sma_20': [...],
        'ema_12': [...],
        'bollinger_upper': [...],
        'bollinger_lower': [...],
        # ... more indicators
    },
    'sgs_data': {
        'selic_rate': 15.0,  # Current SELIC rate
        'cdi_rate': 14.8,    # Current CDI rate
        'ipca_inflation': 3.2  # Current IPCA inflation
    },
    'interest_rate_env': 'high',  # 'low', 'medium', 'high'
    'inflation_env': 'controlled',  # 'low', 'controlled', 'high'
    'current_date': datetime.now()
}
```

### Step 3: Brazilian Market Considerations

#### Trading Types
- **Swing Trade**: Buy and hold for multiple days (15% tax rate)
- **Day Trade**: Buy and sell same day (20% tax rate)

#### Tax Rules
- R$ 20,000/month exemption for swing trades
- Loss carryforward is perpetual and per-asset
- IRRF withholding on sales (credit against final tax)

#### Settlement Rules
- T+2 settlement cycle (2 business days)
- Cash must be available for settlement
- Position tracking includes settlement delays

### Step 4: Running Your Strategy

```python
# Example: Run your strategy

from strategies.my_strategy import MyStrategy
from engine.portfolio import EnhancedPortfolio
from engine.simulator import BacktestSimulator
from engine.loader import DataLoader

# 1. Load data
data_loader = DataLoader()
data = data_loader.load_and_process("VALE3", start_date="2024-01-01", end_date="2024-12-31")

# 2. Create portfolio
portfolio = EnhancedPortfolio("config/settings.yaml")

# 3. Create strategy
strategy = MyStrategy(portfolio, "VALE3", risk_tolerance=0.02)

# 4. Run backtest
simulator = BacktestSimulator(
    strategy=strategy,
    initial_capital=100000.0,
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# 5. Execute simulation
results = simulator.run_simulation(data)

# 6. Get results
print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
print(f"Total Trades: {results.total_trades}")
```

## Strategy Development Tips

### 1. Start Simple
- Begin with basic technical indicators (SMA, RSI, MACD)
- Test with a single stock before expanding
- Use swing trades initially (simpler tax treatment)

### 2. Brazilian Market Specifics
- Consider interest rate environment (SELIC/CDI)
- Account for inflation impact (IPCA)
- Understand T+2 settlement timing
- Plan for Brazilian holidays and market hours

### 3. Risk Management
- Set position size limits (max 10% per position)
- Implement stop losses and take profits
- Monitor day trade exposure (max 30%)
- Track drawdown limits

### 4. Performance Metrics
The system calculates:
- Total return and annualized return
- Sharpe ratio and maximum drawdown
- Win/loss ratio and profit factor
- Transaction costs and taxes
- Benchmark comparison (IBOV)

### 5. Testing Best Practices
- Use out-of-sample testing
- Test across different market conditions
- Validate with Brazilian market data
- Check tax compliance and settlement rules

## Common Strategy Patterns

### 1. Mean Reversion
```python
# Example: RSI-based mean reversion
if rsi < 30:  # Oversold
    return BUY signal
elif rsi > 70:  # Overbought
    return SELL signal
```

### 2. Trend Following
```python
# Example: Moving average crossover
if short_ma > long_ma:
    return BUY signal
elif short_ma < long_ma:
    return SELL signal
```

### 3. Momentum
```python
# Example: Price momentum
if price > price_20_days_ago * 1.05:  # 5% momentum
    return BUY signal
```

### 4. Fundamental Integration
```python
# Example: Interest rate environment
if market_data['interest_rate_env'] == 'low':
    # More aggressive in low rate environment
    confidence *= 1.2
```

## Configuration

The system uses `config/settings.yaml` for all market parameters. Key settings:

```yaml
market:
  trading_hours:
    open: "10:00"
    close: "16:55"
    timezone: "America/Sao_Paulo"
  costs:
    emolument: 0.00005
    settlement_swing_trade: 0.00025
    brokerage_fee: 0.0

taxes:
  swing_trade: 0.15
  day_trade: 0.20
  swing_exemption_limit: 20000

portfolio:
  initial_cash: 100000
  max_positions: 10
```

## Next Steps

1. **Study the BaseStrategy class** - Understand all available methods
2. **Review existing tests** - See how strategies are tested
3. **Start with a simple strategy** - Implement basic moving average crossover
4. **Test thoroughly** - Use different time periods and market conditions
5. **Optimize gradually** - Add complexity step by step
6. **Monitor performance** - Track all metrics and compliance

## Support Files

- `config/settings.yaml` - Market configuration
- `engine/base_strategy.py` - Strategy interface
- `engine/portfolio.py` - Portfolio management
- `engine/simulator.py` - Backtesting engine
- `tests/` - Test examples and validation

## Important Notes

- **Never commit API keys** - Use `config/secrets.yaml.example` as template
- **Brazilian market hours** - 10:00-16:55 BRT (with 5-minute closing auction)
- **Tax compliance** - System handles Brazilian tax rules automatically
- **Settlement timing** - T+2 affects cash flow and position tracking
- **Data quality** - System includes liquidity and quality filters

Good luck building your strategy! The system is designed to handle all Brazilian market complexities so you can focus on your trading logic. 