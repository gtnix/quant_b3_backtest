# Base Strategy Module Documentation

## Overview

The `BaseStrategy` module provides an abstract base class for implementing trading strategies in the Brazilian stock market (B3) with comprehensive integration to the existing backtesting engine components.

**File Location**: `quant_backtest/engine/base_strategy.py`

## Key Features

- **Abstract Interface**: Provides abstract methods for strategy implementation
- **Brazilian Market Integration**: Full integration with EnhancedPortfolio and TCA
- **Market-Specific Considerations**: T+2 settlement, tax rules, and market constraints
- **Risk Management**: Built-in risk metrics and position sizing
- **SGS Data Integration**: Access to interest rates and inflation data
- **Comprehensive Logging**: Detailed logging and error handling
- **Type Hints**: Full type annotation support

## Classes

### TradeType

Enumeration for Brazilian market trade types.

```python
class TradeType(Enum):
    DAY_TRADE = "day_trade"
    SWING_TRADE = "swing_trade"
    AUTO = "auto"
```

### SignalType

Enumeration for trading signal types.

```python
class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
```

### TradingSignal

A dataclass representing comprehensive trading signals with Brazilian market metadata.

```python
@dataclass
class TradingSignal:
    signal_type: SignalType
    ticker: str
    price: float
    quantity: int
    confidence: float = 1.0
    trade_type: TradeType = TradeType.SWING_TRADE
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Attributes**:
- `signal_type`: Type of signal (buy, sell, hold)
- `ticker`: Stock ticker symbol
- `price`: Suggested execution price
- `quantity`: Suggested quantity
- `confidence`: Signal confidence (0.0 to 1.0)
- `trade_type`: Brazilian trade type (day_trade, swing_trade, or auto)
- `timestamp`: Signal generation timestamp
- `metadata`: Additional signal metadata

### RiskMetrics

A dataclass for risk management metrics.

```python
@dataclass
class RiskMetrics:
    max_position_size: float = 0.1  # 10% of portfolio
    max_daily_loss: float = 0.02    # 2% daily loss limit
    max_drawdown: float = 0.15      # 15% maximum drawdown
    stop_loss_pct: float = 0.05     # 5% stop loss
    take_profit_pct: float = 0.10   # 10% take profit
    max_day_trade_exposure: float = 0.3  # 30% day trade exposure
```

### BaseStrategy

The abstract base class for Brazilian market trading strategies.

#### Constructor

```python
def __init__(self, portfolio: EnhancedPortfolio, symbol: str, 
             risk_tolerance: float = 0.02, config_path: str = "config/settings.yaml",
             strategy_name: Optional[str] = None)
```

**Parameters**:
- `portfolio` (EnhancedPortfolio): Portfolio instance for position management
- `symbol` (str): Primary trading symbol (B3 ticker)
- `risk_tolerance` (float): Risk tolerance level (0.0 to 1.0)
- `config_path` (str): Path to configuration file
- `strategy_name` (Optional[str]): Optional strategy name for logging

**Initialized Components**:
- **Portfolio**: EnhancedPortfolio instance
- **TCA**: TransactionCostAnalyzer for cost calculation
- **Data Loader**: DataLoader for market data
- **Risk Metrics**: Risk management parameters
- **Configuration**: Market and tax configuration

## Abstract Methods

### `generate_signals(market_data: Dict[str, Any]) -> List[TradingSignal]`

**Abstract method** that must be implemented by all strategies.

**Parameters**:
- `market_data` (Dict[str, Any]): Market data for current period

**Returns**:
- `List[TradingSignal]`: List of trading signals

**Example Implementation**:
```python
def generate_signals(self, market_data: Dict[str, Any]) -> List[TradingSignal]:
    signals = []
    
    # Example: Simple moving average crossover
    if 'sma_20' in market_data and 'sma_50' in market_data:
        current_price = market_data['close']
        sma_20 = market_data['sma_20']
        sma_50 = market_data['sma_50']
        
        if sma_20 > sma_50 and current_price > sma_20:
            # Buy signal
            signal = TradingSignal(
                signal_type=SignalType.BUY,
                ticker=self.symbol,
                price=current_price,
                quantity=self.calculate_position_size(current_price),
                confidence=0.8,
                trade_type=TradeType.SWING_TRADE
            )
            signals.append(signal)
    
    return signals
```

### `manage_risk(current_positions: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]`

**Abstract method** for risk management implementation.

**Parameters**:
- `current_positions` (Dict[str, Any]): Current portfolio positions
- `market_data` (Dict[str, Any]): Current market data

**Returns**:
- `Dict[str, Any]`: Risk management decisions

**Example Implementation**:
```python
def manage_risk(self, current_positions: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
    risk_decisions = {}
    
    # Example: Stop loss management
    for ticker, position in current_positions.items():
        current_price = market_data.get('close', 0)
        avg_price = position.get('avg_price', 0)
        
        if avg_price > 0:
            loss_pct = (current_price - avg_price) / avg_price
            
            if loss_pct < -self.risk_metrics.stop_loss_pct:
                risk_decisions[ticker] = {
                    'action': 'stop_loss',
                    'reason': f'Stop loss triggered: {loss_pct:.2%}',
                    'price': current_price
                }
    
    return risk_decisions
```

### `execute_trade(signal: TradingSignal) -> bool`

**Abstract method** for trade execution implementation.

**Parameters**:
- `signal` (TradingSignal): Trading signal to execute

**Returns**:
- `bool`: True if trade executed successfully

**Example Implementation**:
```python
def execute_trade(self, signal: TradingSignal) -> bool:
    try:
        if signal.signal_type == SignalType.BUY:
            success = self.portfolio.buy(
                ticker=signal.ticker,
                quantity=signal.quantity,
                price=signal.price,
                trade_date=signal.timestamp,
                trade_type=signal.trade_type.value
            )
        elif signal.signal_type == SignalType.SELL:
            success = self.portfolio.sell(
                ticker=signal.ticker,
                quantity=signal.quantity,
                price=signal.price,
                trade_date=signal.timestamp,
                trade_type=signal.trade_type.value
            )
        else:
            success = True  # HOLD signal
        
        if success:
            self.log_trade({
                'signal': signal,
                'execution_time': datetime.now(),
                'success': success
            })
        
        return success
        
    except Exception as e:
        logger.error(f"Trade execution failed: {e}")
        return False
```

## Core Methods

### Market Data Validation

#### `validate_market_data(market_data: Dict[str, Any]) -> bool`

Validates market data for strategy requirements.

**Parameters**:
- `market_data` (Dict[str, Any]): Market data to validate

**Validation Checks**:
- Required price data (open, high, low, close)
- Volume data availability
- Technical indicators if required
- Data quality checks

**Returns**:
- `bool`: True if data is valid

### SGS Data Integration

#### `get_sgs_data(market_data: Dict[str, Any]) -> Dict[str, float]`

Retrieves SGS data for economic context.

**Parameters**:
- `market_data` (Dict[str, Any]): Market data dictionary

**Returns**:
- `Dict[str, float]`: SGS data including SELIC, CDI, IPCA

#### `get_interest_rate_environment(market_data: Dict[str, Any]) -> str`

Classifies current interest rate environment.

**Returns**:
- `str`: 'high', 'moderate', or 'low' interest rate environment

#### `get_inflation_environment(market_data: Dict[str, Any]) -> str`

Classifies current inflation environment.

**Returns**:
- `str`: 'high', 'moderate', or 'low' inflation environment

### Position Sizing

#### `calculate_position_size(signal: TradingSignal, available_cash: float) -> int`

Calculates position size based on risk management rules.

**Parameters**:
- `signal` (TradingSignal): Trading signal
- `available_cash` (float): Available cash for trading

**Position Sizing Logic**:
- Risk-based position sizing
- Maximum position size limits
- Day trade exposure limits
- Cash availability constraints

**Returns**:
- `int`: Calculated position size

### Brazilian Market Constraints

#### `check_brazilian_market_constraints(signal: TradingSignal) -> bool`

Validates signal against Brazilian market constraints.

**Constraints Checked**:
- Trading hours compliance
- Minimum volume requirements
- Price change limits
- Position limits
- Day trade restrictions

**Returns**:
- `bool`: True if signal meets all constraints

### Day Trade Management

#### `_calculate_day_trade_exposure() -> float`

Calculates current day trade exposure.

**Returns**:
- `float`: Current day trade exposure as percentage of portfolio

## Configuration Integration

### Market Configuration
```python
# Access market configuration
trading_hours = self.config['market']['trading_hours']
selic_rate = self.config['market']['selic_rate']
```

### Tax Configuration
```python
# Access tax configuration
swing_tax_rate = self.config['taxes']['swing_trade']
day_tax_rate = self.config['taxes']['day_trade']
exemption_limit = self.config['taxes']['swing_exemption_limit']
```

### Settlement Configuration
```python
# Access settlement configuration
settlement_days = self.config['settlement']['cycle_days']
strict_mode = self.config['settlement']['strict_mode']
```

## Logging and Monitoring

### Signal Logging

#### `log_signal(signal: TradingSignal) -> None`

Logs trading signal for audit trail.

**Parameters**:
- `signal` (TradingSignal): Signal to log

### Trade Logging

#### `log_trade(trade_result: Dict[str, Any]) -> None`

Logs trade execution results.

**Parameters**:
- `trade_result` (Dict[str, Any]): Trade execution result

## Performance Tracking

### Performance Summary

#### `get_performance_summary() -> Dict[str, Any]`

Returns strategy performance summary.

**Returns**:
- `Dict[str, Any]`: Performance metrics including:
  - Total return
  - Sharpe ratio
  - Win rate
  - Maximum drawdown
  - Number of trades

### Strategy Reset

#### `reset_strategy() -> None`

Resets strategy state for new backtest.

**Reset Actions**:
- Clear signal history
- Clear trade history
- Reset performance metrics
- Clear cached data

## Usage Examples

### Basic Strategy Implementation
```python
from engine.base_strategy import BaseStrategy, TradingSignal, SignalType, TradeType
from engine.portfolio import EnhancedPortfolio

class MyStrategy(BaseStrategy):
    def generate_signals(self, market_data: Dict[str, Any]) -> List[TradingSignal]:
        signals = []
        
        # Simple strategy logic
        if market_data['close'] > market_data['sma_20']:
            signal = TradingSignal(
                signal_type=SignalType.BUY,
                ticker=self.symbol,
                price=market_data['close'],
                quantity=100,
                confidence=0.7,
                trade_type=TradeType.SWING_TRADE
            )
            signals.append(signal)
        
        return signals
    
    def manage_risk(self, current_positions: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        # Risk management logic
        return {}
    
    def execute_trade(self, signal: TradingSignal) -> bool:
        # Trade execution logic
        return True

# Usage
portfolio = EnhancedPortfolio()
strategy = MyStrategy(portfolio, "PETR4")
```

### Advanced Strategy with SGS Data
```python
class AdvancedStrategy(BaseStrategy):
    def generate_signals(self, market_data: Dict[str, Any]) -> List[TradingSignal]:
        signals = []
        
        # Get economic context
        sgs_data = self.get_sgs_data(market_data)
        interest_env = self.get_interest_rate_environment(market_data)
        
        # Adjust strategy based on economic environment
        if interest_env == 'high':
            # More conservative in high interest rate environment
            confidence = 0.6
        else:
            confidence = 0.8
        
        # Strategy logic with economic context
        if market_data['rsi_14'] < 30 and sgs_data.get('selic', 0) < 0.12:
            signal = TradingSignal(
                signal_type=SignalType.BUY,
                ticker=self.symbol,
                price=market_data['close'],
                quantity=self.calculate_position_size(market_data['close']),
                confidence=confidence,
                trade_type=TradeType.SWING_TRADE,
                metadata={'interest_env': interest_env}
            )
            signals.append(signal)
        
        return signals
```

## Best Practices

1. **Implement All Abstract Methods**: Ensure all required methods are implemented
2. **Validate Market Data**: Always validate data before processing
3. **Use Risk Management**: Implement proper risk management controls
4. **Handle Brazilian Constraints**: Consider T+2 settlement and tax rules
5. **Log Everything**: Maintain comprehensive audit trails
6. **Test Thoroughly**: Test strategies with historical data
7. **Monitor Performance**: Track strategy performance metrics
8. **Use Type Hints**: Maintain code quality with type annotations

## Testing

### Strategy Testing
```python
# Test strategy implementation
def test_strategy():
    portfolio = EnhancedPortfolio()
    strategy = MyStrategy(portfolio, "PETR4")
    
    # Test signal generation
    market_data = {
        'close': 25.50,
        'sma_20': 25.00,
        'volume': 1000000
    }
    
    signals = strategy.generate_signals(market_data)
    assert len(signals) > 0
    assert signals[0].signal_type == SignalType.BUY
```

## Dependencies

- **abc**: Abstract base classes
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **pytz**: Timezone handling
- **yaml**: Configuration loading
- **logging**: Logging and debugging
- **datetime**: Date and time operations 