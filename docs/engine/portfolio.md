# Portfolio Manager Module Documentation

## Overview

The `EnhancedPortfolio` module provides comprehensive portfolio management for Brazilian market backtesting with full compliance to Brazilian tax regulations, T+2 settlement tracking, and advanced loss carryforward management.

**File Location**: `quant_backtest/engine/portfolio.py`

## Key Features

- **Enhanced Portfolio Management**: Comprehensive position tracking with unrealized P&L
- **Brazilian Tax Compliance**: Full compliance with individual taxpayer rules (2025)
- **T+2 Settlement Tracking**: Accurate settlement date calculation with business day handling
- **Loss Carryforward Management**: Per-asset and global loss tracking with 100% offset capability
- **Transaction Cost Analysis**: Integration with TCA module for accurate cost calculation
- **Day Trade Detection**: Automatic detection of day trades vs swing trades
- **Comprehensive Audit Trails**: Detailed logging for regulatory compliance

## Classes

### Position

A dataclass representing a portfolio position with comprehensive metadata.

```python
@dataclass
class Position:
    ticker: str
    quantity: int
    avg_price: float
    current_price: float
    last_update: datetime
    trade_type: str = "swing_trade"  # 'day_trade' or 'swing_trade'
    position_id: Optional[str] = None
    description: str = ""
```

**Properties**:
- `market_value`: Current market value of position
- `unrealized_pnl`: Unrealized profit/loss
- `unrealized_pnl_pct`: Unrealized profit/loss percentage

### EnhancedPortfolio

The main portfolio management class with Brazilian market compliance.

#### Constructor

```python
def __init__(self, config_path: str = "config/settings.yaml")
```

**Parameters**:
- `config_path` (str): Path to configuration file

**Initialized Components**:
- **Loss Manager**: `EnhancedLossCarryforwardManager` for tax loss tracking
- **Settlement Manager**: `AdvancedSettlementManager` for T+2 settlement
- **TCA**: `TransactionCostAnalyzer` for transaction costs
- **Portfolio State**: Positions, cash, trade history, performance tracking

## Core Methods

### Trade Execution

#### `buy(ticker: str, quantity: int, price: float, trade_date: datetime, trade_type: str = "swing_trade", trade_id: Optional[str] = None, description: str = "") -> bool`

Executes a buy order with comprehensive validation and cost calculation.

**Parameters**:
- `ticker` (str): Stock ticker symbol
- `quantity` (int): Number of shares to buy
- `price` (float): Price per share
- `trade_date` (datetime): Date of trade
- `trade_type` (str): 'day_trade', 'swing_trade', or 'auto'
- `trade_id` (Optional[str]): Unique trade identifier
- `description` (str): Trade description

**Process**:
1. Validate inputs
2. Calculate transaction costs
3. Check available cash (including unsettled amounts)
4. Update position or create new position
5. Record trade in history
6. Schedule settlement

**Returns**:
- `bool`: True if trade executed successfully

**Example**:
```python
portfolio = EnhancedPortfolio()
success = portfolio.buy(
    ticker="PETR4",
    quantity=100,
    price=25.50,
    trade_date=datetime.now(),
    trade_type="swing_trade"
)
```

#### `sell(ticker: str, quantity: int, price: float, trade_date: datetime, trade_type: str = "swing_trade", trade_id: Optional[str] = None, description: str = "") -> bool`

Executes a sell order with profit/loss calculation and tax handling.

**Parameters**:
- `ticker` (str): Stock ticker symbol
- `quantity` (int): Number of shares to sell
- `price` (float): Price per share
- `trade_date` (datetime): Date of trade
- `trade_type` (str): 'day_trade', 'swing_trade', or 'auto'
- `trade_id` (Optional[str]): Unique trade identifier
- `description` (str): Trade description

**Process**:
1. Validate inputs and position availability
2. Detect day trade vs swing trade
3. Calculate transaction costs
4. Calculate profit/loss
5. Apply loss carryforward
6. Calculate taxes
7. Update position
8. Record trade and schedule settlement

**Returns**:
- `bool`: True if trade executed successfully

### Portfolio Management

#### `update_prices(price_updates: Dict[str, float], update_date: datetime) -> None`

Updates current prices for all positions.

**Parameters**:
- `price_updates` (Dict[str, float]): Dictionary of ticker: price updates
- `update_date` (datetime): Date of price update

#### `get_portfolio_summary() -> Dict`

Returns comprehensive portfolio summary.

**Returns**:
- `Dict`: Portfolio summary including:
  - Total value
  - Cash balance
  - Unrealized P&L
  - Number of positions
  - Performance metrics

#### `get_position_summary() -> List[Dict]`

Returns summary of all positions.

**Returns**:
- `List[Dict]`: List of position summaries

### Tax Management

#### `calculate_monthly_tax_liability(month_ref: date, trade_type: str = None) -> Dict[str, float]`

Calculates monthly tax liability for Brazilian tax compliance.

**Parameters**:
- `month_ref` (date): Reference month for tax calculation
- `trade_type` (str): Optional filter for trade type

**Returns**:
- `Dict[str, float]`: Tax liability breakdown including:
  - Swing trade taxes
  - Day trade taxes
  - Total liability
  - Exemption utilized

### Settlement Management

#### `process_settlements(current_date: date) -> None`

Processes due settlements for the current date.

**Parameters**:
- `current_date` (date): Current date for settlement processing

### Audit and Reporting

#### `export_audit_trails(base_path: str = "audit_trails") -> None`

Exports comprehensive audit trails for regulatory compliance.

**Parameters**:
- `base_path` (str): Base path for audit trail files

**Exported Files**:
- Trade history
- Loss carryforward records
- Settlement records
- Tax calculations

## Brazilian Market Compliance

### Tax Rules (2025)

#### Individual Taxpayer Compliance
- **Swing Trade Tax**: 15% on monthly net profit
- **Day Trade Tax**: 20% on monthly net profit
- **Exemption**: R$ 20,000/month for swing trades
- **Loss Offset**: 100% loss offset capability (no 30% limit)
- **Perpetual Carryforward**: Losses carried forward indefinitely

#### Trade Type Detection
- **Day Trade**: Buy and sell of same asset on same trading day
- **Swing Trade**: All other trades
- **Automatic Detection**: System automatically detects trade type

### Settlement Rules

#### T+2 Settlement
- **Standard Cycle**: 2 business days after trade
- **Business Day Calculation**: Uses Brazilian business day calendar
- **Cash Flow Impact**: Realistic cash flow modeling

### Transaction Costs

#### B3 Fee Structure
- **Emolument**: 0.005% negotiation fee
- **Settlement**: 0.018% (day trade) / 0.025% (swing trade)
- **Brokerage**: Configurable (default: 0% for Modal)
- **ISS**: 5% on brokerage fees

## Performance Tracking

### Metrics Tracked
- **Total Trades**: Number of completed trades
- **Winning/Losing Trades**: Trade success rate
- **Total Commission**: Cumulative transaction costs
- **Total Taxes**: Cumulative tax payments
- **Daily P&L**: Daily profit/loss tracking

### Position Metrics
- **Market Value**: Current position value
- **Unrealized P&L**: Unrealized profit/loss
- **Average Price**: Weighted average purchase price
- **Position Duration**: Time since position opened

## Error Handling

### Input Validation
- **Ticker Validation**: Ensures valid ticker symbols
- **Quantity Validation**: Positive integer quantities
- **Price Validation**: Positive numeric prices
- **Date Validation**: Valid datetime objects
- **Trade Type Validation**: Valid trade type strings

### Business Logic Validation
- **Cash Availability**: Checks available cash for purchases
- **Position Availability**: Validates position existence for sales
- **Settlement Constraints**: Enforces settlement rules
- **Tax Compliance**: Validates tax calculations

## Integration with Other Modules

### Loss Manager Integration
- **Loss Recording**: Records losses for carryforward
- **Loss Application**: Applies losses against profits
- **Audit Trail**: Maintains loss application history

### Settlement Manager Integration
- **Settlement Scheduling**: Schedules trades for settlement
- **Cash Flow Management**: Manages cash flow with settlement delays
- **Business Day Handling**: Uses Brazilian business day calendar

### TCA Integration
- **Cost Calculation**: Calculates comprehensive transaction costs
- **Cost Breakdown**: Provides detailed cost components
- **Cost Optimization**: Optimizes costs based on trade type

## Usage Examples

### Basic Portfolio Operations
```python
from engine.portfolio import EnhancedPortfolio
from datetime import datetime

# Initialize portfolio
portfolio = EnhancedPortfolio()

# Buy shares
portfolio.buy("PETR4", 100, 25.50, datetime.now())

# Update prices
portfolio.update_prices({"PETR4": 26.00}, datetime.now())

# Sell shares
portfolio.sell("PETR4", 50, 26.00, datetime.now())

# Get summary
summary = portfolio.get_portfolio_summary()
print(f"Total Value: R$ {summary['total_value']:,.2f}")
```

### Tax Management
```python
from datetime import date

# Calculate monthly tax liability
tax_liability = portfolio.calculate_monthly_tax_liability(
    month_ref=date(2023, 12, 1)
)
print(f"Total Tax: R$ {tax_liability['total_liability']:,.2f}")
```

### Settlement Processing
```python
from datetime import date

# Process settlements for current date
portfolio.process_settlements(date.today())

# Get available cash
available_cash = portfolio.settlement_manager.get_available_cash(date.today())
print(f"Available Cash: R$ {available_cash:,.2f}")
```

### Audit Trail Export
```python
# Export audit trails
portfolio.export_audit_trails("audit_trails_2023")
```

## Configuration

The portfolio uses configuration from `config/settings.yaml`:

```yaml
portfolio:
  initial_cash: 100000     # Starting capital
  max_positions: 10        # Maximum positions
  position_sizing: "equal_weight"  # Position sizing method

taxes:
  swing_trade: 0.15        # 15% swing trade tax
  day_trade: 0.20          # 20% day trade tax
  swing_exemption_limit: 20000  # R$ 20,000 exemption
  max_loss_offset_percentage: 1.0  # 100% loss offset

settlement:
  cycle_days: 2            # T+2 settlement
  timezone: "America/Sao_Paulo"
  strict_mode: true        # Enforce settlement rules
```

## Testing

The module includes comprehensive tests:

```bash
# Run portfolio tests
python -m unittest tests.test_portfolio
```

## Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **pytz**: Timezone handling
- **yaml**: Configuration loading
- **datetime**: Date and time operations
- **logging**: Logging and debugging

## Best Practices

1. **Always validate inputs**: Use input validation before trade execution
2. **Monitor cash flow**: Track available cash including unsettled amounts
3. **Handle day trades**: Be aware of day trade vs swing trade implications
4. **Maintain audit trails**: Export audit trails for compliance
5. **Update prices regularly**: Keep position prices current for accurate P&L
6. **Process settlements**: Regularly process due settlements
7. **Monitor tax liability**: Track tax obligations for compliance 