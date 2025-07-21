# Brazilian Market Conventions Implementation

This document describes the implementation of Brazilian market (B3) conventions in the quantitative backtesting system, including price tick normalization and lot size validation.

## Overview

The system now enforces Brazilian market conventions to ensure realistic simulation of trading on B3:

- **Price Tick Normalization**: All prices are rounded to the nearest R$ 0.01
- **Lot Size Validation**: Orders are classified as round lots (multiples of 100) or odd lots
- **Order Routing**: Round lots go to main book, odd lots to fractional book
- **Automatic Validation**: All orders are validated against market conventions

## Implementation Details

### 1. Market Utilities Module (`engine/base_strategy.py`)

#### Core Classes
- `BrazilianMarketUtils`: Main utility class for market conventions
- `OrderType`: Enumeration for order types (MARKET, LIMIT, STOP, STOP_LIMIT)
- `LotType`: Enumeration for lot types (ROUND_LOT, ODD_LOT, FRACTIONAL)
- `OrderValidation`: Data class for validation results

#### Key Methods

##### Price Normalization
```python
def normalize_price_tick(self, price: float) -> float:
    """
    Normalize price to the nearest tick size (R$ 0.01).
    
    Examples:
        normalize_price_tick(12.3456) -> 12.35
        normalize_price_tick(12.344) -> 12.34
    """
```

##### Lot Size Validation
```python
def validate_lot_size(self, quantity: int) -> Tuple[bool, LotType, bool]:
    """
    Validate and classify lot size.
    
    Returns:
        (is_valid, lot_type, is_fractional)
    
    Examples:
        validate_lot_size(100) -> (True, ROUND_LOT, False)
        validate_lot_size(150) -> (True, ODD_LOT, True)
    """
```

##### Order Validation
```python
def validate_order(self, price: float, quantity: int, 
                  order_type: OrderType = OrderType.MARKET,
                  allow_fractional: bool = True) -> OrderValidation:
    """
    Comprehensive order validation for Brazilian market.
    
    Returns OrderValidation object with:
    - is_valid: Whether order meets all constraints
    - normalized_price: Price rounded to nearest tick
    - normalized_quantity: Quantity (no normalization for odd lots)
    - lot_type: ROUND_LOT or ODD_LOT
    - is_fractional: Whether order is fractional
    - validation_messages: List of validation messages
    - original_price/quantity: Original values for audit
    """
```

### 2. Configuration (`config/settings.yaml`)

Added market convention settings:

```yaml
market:
  # Price and lot size conventions
  tick_size: 0.01              # Minimum price increment (R$ 0.01)
  round_lot_size: 100          # Standard lot size (multiples of 100 shares)
  min_quantity: 1              # Minimum order quantity
  allow_fractional_lots: true  # Allow odd lot orders (fractional book)
  enforce_price_ticks: true    # Enforce price tick normalization
  enforce_lot_sizes: true      # Enforce lot size validation
```

### 3. Portfolio Integration (`engine/portfolio.py`)

#### Enhanced Buy/Sell Methods
- **Automatic Validation**: All orders are validated using market utilities
- **Price Normalization**: Prices are automatically normalized to valid ticks
- **Lot Classification**: Orders are classified as round lot or odd lot
- **Trade History**: Original and normalized values are tracked for audit

#### Trade Record Enhancement
```python
trade_record = {
    'date': trade_date,
    'ticker': ticker,
    'action': 'BUY',
    'quantity': normalized_quantity,
    'price': normalized_price,
    'value': trade_value,
    'costs': costs,
    'trade_type': resolved_trade_type,
    'trade_id': trade_id,
    'description': description,
    'lot_type': validation.lot_type.value,        # NEW
    'is_fractional': validation.is_fractional,    # NEW
    'original_quantity': quantity,                # NEW
    'original_price': price                       # NEW
}
```

### 4. Strategy Integration (`engine/base_strategy.py`)

#### Enhanced Constraint Validation
```python
def check_brazilian_market_constraints(self, signal: TradingSignal) -> bool:
    """
    Check Brazilian market-specific constraints including price ticks and lot sizes.
    
    This enhanced implementation validates:
    - Trading hours
    - Day trade exposure limits
    - Price tick normalization (R$ 0.01)
    - Lot size validation (round lots = multiples of 100)
    - Fractional lot handling
    """
```

#### Automatic Signal Normalization
- Signal prices are automatically normalized during constraint validation
- Signal quantities are validated but not normalized (odd lots remain odd lots)
- Normalized values are used in subsequent trade execution

## Usage Examples

### Basic Market Utilities

```python
from engine.market_utils import BrazilianMarketUtils

utils = BrazilianMarketUtils()

# Price normalization
price = utils.normalize_price_tick(12.3456)  # Returns 12.35

# Lot validation
is_valid, lot_type, is_fractional = utils.validate_lot_size(150)
# Returns: (True, LotType.ODD_LOT, True)

# Order validation
validation = utils.validate_order(
    price=12.3456,
    quantity=150,
    allow_fractional=True
)
print(f"Valid: {validation.is_valid}")
print(f"Normalized price: {validation.normalized_price}")
print(f"Lot type: {validation.lot_type}")
```

### Portfolio Usage

```python
from engine.portfolio import EnhancedPortfolio

portfolio = EnhancedPortfolio("config/settings.yaml")

# Buy with automatic price normalization
success = portfolio.buy(
    ticker="PETR4",
    quantity=100,
    price=12.3456,  # Automatically normalized to 12.35
    trade_date=datetime.now(),
    trade_type="swing_trade"
)

# Check trade history
trade = portfolio.trade_history[0]
print(f"Original price: {trade['original_price']}")  # 12.3456
print(f"Normalized price: {trade['price']}")         # 12.35
print(f"Lot type: {trade['lot_type']}")              # round_lot
```

### Strategy Usage

```python
from engine.base_strategy import BaseStrategy, TradingSignal, SignalType

class MyStrategy(BaseStrategy):
    def generate_signals(self, market_data):
        signal = TradingSignal(
            signal_type=SignalType.BUY,
            ticker="PETR4",
            price=12.3456,  # Will be normalized automatically
            quantity=150,    # Will be classified as odd lot
            confidence=1.0,
            trade_type=TradeType.SWING_TRADE
        )
        return [signal]
    
    def execute_trade(self, signal):
        # Market constraints are automatically validated
        constraints_ok = self.check_brazilian_market_constraints(signal)
        if constraints_ok:
            # Signal price is already normalized
            return self.portfolio.buy(
                ticker=signal.ticker,
                quantity=signal.quantity,
                price=signal.price,  # Already normalized
                trade_date=signal.timestamp,
                trade_type=signal.trade_type.value
            )
        return False
```

## Testing

### Unit Tests (`tests/test_market_utils.py`)
- **Price Normalization**: Tests for various price scenarios
- **Lot Size Validation**: Tests for round lots, odd lots, and edge cases
- **Order Validation**: Tests for comprehensive order validation
- **Order Routing**: Tests for main vs fractional book routing
- **Edge Cases**: Tests for boundary conditions and custom configurations

### Integration Tests (`tests/test_market_integration.py`)
- **Portfolio Integration**: Tests buy/sell with price normalization
- **Strategy Integration**: Tests constraint validation in strategies
- **Configuration**: Tests market utils configuration from settings
- **Complete Workflow**: Tests end-to-end trading workflow

### Test Coverage
- 25 unit tests for market utilities
- 9 integration tests for portfolio and strategy integration
- 100% test coverage for core functionality
- Edge case testing for boundary conditions

## Configuration Options

### Enable/Disable Features
```yaml
market:
  allow_fractional_lots: true   # Allow odd lot orders
  enforce_price_ticks: true     # Enforce price normalization
  enforce_lot_sizes: true       # Enforce lot size validation
```

### Custom Market Rules
```yaml
market:
  tick_size: 0.05              # Custom tick size (e.g., for options)
  round_lot_size: 50           # Custom lot size
  min_quantity: 1              # Minimum order quantity
```

## Benefits

### Realistic Simulation
- Enforces actual B3 market rules
- Prevents unrealistic price/quantity combinations
- Provides accurate order routing simulation

### Compliance
- Ensures backtest results reflect real market constraints
- Prevents over-optimization on invalid orders
- Maintains audit trail of original vs normalized values

### Flexibility
- Configurable market rules
- Optional fractional lot restrictions
- Custom tick sizes and lot sizes

### Integration
- Seamless integration with existing portfolio and strategy classes
- Automatic validation without code changes
- Comprehensive logging and audit trails

## Future Enhancements

### Potential Improvements
1. **Dynamic Tick Sizes**: Support for different tick sizes based on price ranges
2. **Market Hours Validation**: Integration with actual B3 trading hours
3. **Order Book Simulation**: More sophisticated order book matching
4. **Market Impact**: Modeling of order impact on market prices
5. **Regulatory Compliance**: Additional B3 regulatory requirements

### Extensibility
- Modular design allows easy addition of new market rules
- Configuration-driven approach for different markets
- Plugin architecture for custom validation rules

## Conclusion

The Brazilian market conventions implementation provides a robust foundation for realistic backtesting on B3. The system automatically enforces market rules while maintaining flexibility and providing comprehensive audit trails. The integration with existing portfolio and strategy classes ensures seamless adoption without requiring changes to existing code. 