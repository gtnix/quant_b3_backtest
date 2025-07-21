# Transaction Cost Analyzer (TCA) Module Documentation

## Overview

The `TransactionCostAnalyzer` module provides comprehensive transaction cost calculation for Brazilian stock market (B3) operations, including brokerage fees, B3 emolument fees, settlement fees, and ISS taxes with full compliance to Brazilian market regulations.

**File Location**: `quant_backtest/engine/tca.py`

## Key Features

- **Complete B3 Fee Structure**: All mandatory B3 fees and taxes
- **Brokerage Fee Calculation**: Configurable brokerage with minimum charge enforcement
- **B3 Emolument Fees**: Negotiation fees as per B3 regulations
- **Settlement Fees**: Different rates for day trade vs swing trade
- **ISS Tax Calculation**: Service tax on brokerage fees
- **Cost Breakdown**: Detailed cost component analysis
- **Cost Comparison**: Compare costs across different trade types
- **Configuration Management**: Flexible cost parameter configuration

## Classes

### CostBreakdown

A dataclass representing detailed breakdown of transaction costs.

```python
@dataclass
class CostBreakdown:
    brokerage_fee: float
    min_brokerage_applied: bool
    emolument: float
    settlement_fee: float
    iss_fee: float
    total_costs: float
    cost_percentage: float
```

**Attributes**:
- `brokerage_fee`: Brokerage commission
- `min_brokerage_applied`: Whether minimum brokerage was applied
- `emolument`: B3 negotiation fee
- `settlement_fee`: Settlement fee (day trade or swing trade)
- `iss_fee`: ISS tax on brokerage fee
- `total_costs`: Sum of all costs
- `cost_percentage`: Total costs as percentage of order value

### TransactionCostAnalyzer

The main transaction cost analysis class with Brazilian market compliance.

#### Constructor

```python
def __init__(self, config_path: str = "config/settings.yaml")
```

**Parameters**:
- `config_path` (str): Path to configuration file

**Initialization Process**:
1. Load configuration from YAML file
2. Extract cost parameters from market.costs section
3. Validate cost parameters for logical consistency
4. Set up logging and error handling

## Core Methods

### Cost Calculation

#### `calculate_costs(order_value: float, is_buy: bool = True, trade_type: str = "swing_trade") -> CostBreakdown`

Calculates comprehensive transaction costs for Brazilian market operations.

**Parameters**:
- `order_value` (float): Total order value in BRL
- `is_buy` (bool): True for buy orders, False for sell orders
- `trade_type` (str): 'day_trade' or 'swing_trade'

**Cost Components**:
1. **Brokerage Fee**: Order value × brokerage rate (with minimum enforcement)
2. **B3 Emolument**: Order value × emolument rate
3. **Settlement Fee**: Order value × settlement rate (varies by trade type)
4. **ISS Tax**: Brokerage fee × ISS rate

**Returns**:
- `CostBreakdown`: Detailed cost breakdown object

**Example**:
```python
tca = TransactionCostAnalyzer()
costs = tca.calculate_costs(
    order_value=10000.0,
    is_buy=True,
    trade_type="swing_trade"
)
print(f"Total costs: R$ {costs.total_costs:,.2f}")
print(f"Cost percentage: {costs.cost_percentage:.4f}%")
```

### Cost Analysis

#### `get_cost_summary(order_value: float, is_buy: bool = True, trade_type: str = "swing_trade") -> Dict`

Returns a summary of transaction costs in dictionary format.

**Parameters**:
- `order_value` (float): Total order value in BRL
- `is_buy` (bool): True for buy orders, False for sell orders
- `trade_type` (str): 'day_trade' or 'swing_trade'

**Returns**:
- `Dict`: Cost summary including all components and percentages

#### `compare_costs(order_value: float, trade_types: list = None) -> Dict`

Compares transaction costs across different trade types.

**Parameters**:
- `order_value` (float): Total order value in BRL
- `trade_types` (list): List of trade types to compare (default: ['day_trade', 'swing_trade'])

**Returns**:
- `Dict`: Comparison of costs across trade types

**Example**:
```python
tca = TransactionCostAnalyzer()
comparison = tca.compare_costs(10000.0)
for trade_type, costs in comparison.items():
    print(f"{trade_type}: R$ {costs['total_costs']:,.2f}")
```

### Configuration Management

#### `get_cost_parameters() -> Dict`

Returns current cost parameters from configuration.

**Returns**:
- `Dict`: Current cost parameters

#### `update_cost_parameters(new_params: Dict) -> None`

Updates cost parameters with new values.

**Parameters**:
- `new_params` (Dict): New cost parameters

**Validation**:
- Ensures all parameters are non-negative
- Validates ISS rate range (0-5%)
- Updates internal configuration

## Brazilian Market Fee Structure

### B3 Mandatory Fees

#### Emolument (Negotiation Fee)
- **Rate**: 0.005% (0.00005)
- **Description**: B3 negotiation fee on all trades
- **Calculation**: Order value × 0.00005

#### Settlement Fees
- **Day Trade**: 0.018% (0.00018)
- **Swing Trade**: 0.025% (0.00025)
- **Description**: B3 settlement fee
- **Calculation**: Order value × settlement rate

### Brokerage Fees

#### Modal Brokerage (Default Configuration)
- **Rate**: 0.0% (zero brokerage)
- **Minimum**: R$ 0.00
- **Description**: Electronic-only brokerage via web/app

#### Configurable Brokerage
- **Rate**: Configurable via settings
- **Minimum**: Configurable minimum charge
- **Description**: Can be adjusted for different brokers

### ISS Tax (Service Tax)

#### Tax Rate
- **Rate**: 5% (0.05) on brokerage fees
- **Description**: Municipal service tax
- **Calculation**: Brokerage fee × 0.05
- **Note**: Currently 0 when brokerage = 0

## Cost Calculation Examples

### Example 1: Swing Trade Purchase
```python
# R$ 10,000 swing trade purchase
costs = tca.calculate_costs(10000.0, is_buy=True, trade_type="swing_trade")

# Breakdown:
# Brokerage: R$ 0.00 (Modal zero brokerage)
# Emolument: R$ 0.50 (10000 × 0.00005)
# Settlement: R$ 2.50 (10000 × 0.00025)
# ISS: R$ 0.00 (0 × 0.05)
# Total: R$ 3.00
# Percentage: 0.0300%
```

### Example 2: Day Trade Sale
```python
# R$ 5,000 day trade sale
costs = tca.calculate_costs(5000.0, is_buy=False, trade_type="day_trade")

# Breakdown:
# Brokerage: R$ 0.00 (Modal zero brokerage)
# Emolument: R$ 0.25 (5000 × 0.00005)
# Settlement: R$ 0.90 (5000 × 0.00018)
# ISS: R$ 0.00 (0 × 0.05)
# Total: R$ 1.15
# Percentage: 0.0230%
```

### Example 3: Traditional Brokerage
```python
# With traditional brokerage (0.5% + R$ 5 minimum)
# R$ 1,000 swing trade purchase
costs = tca.calculate_costs(1000.0, is_buy=True, trade_type="swing_trade")

# Breakdown:
# Brokerage: R$ 5.00 (minimum applied)
# Emolument: R$ 0.05 (1000 × 0.00005)
# Settlement: R$ 0.25 (1000 × 0.00025)
# ISS: R$ 0.25 (5 × 0.05)
# Total: R$ 5.55
# Percentage: 0.5550%
```

## Configuration

The TCA module uses configuration from `config/settings.yaml`:

```yaml
market:
  costs:
    # B3 mandatory fees
    emolument: 0.00005                 # 0.005% negotiation fee
    settlement_day_trade: 0.00018      # 0.018% settlement (day trade)
    settlement_swing_trade: 0.00025    # 0.025% settlement (swing trade)
    
    # Brokerage fees
    brokerage_fee: 0.0                 # 0% brokerage (Modal)
    min_brokerage: 0.0                 # R$ 0 minimum
    
    # ISS tax
    iss_rate: 0.05                     # 5% ISS on brokerage
```

## Error Handling

### Input Validation
- **Order Value**: Must be positive
- **Trade Type**: Must be 'day_trade' or 'swing_trade'
- **Configuration**: Validates cost parameters on initialization

### Configuration Validation
- **Non-negative Rates**: All fee rates must be non-negative
- **ISS Rate Range**: ISS rate must be between 0 and 5%
- **Logical Consistency**: Validates parameter relationships

## Integration with Other Modules

### Portfolio Manager Integration
- **Cost Calculation**: Provides costs for trade execution
- **Cost Tracking**: Tracks cumulative transaction costs
- **Performance Impact**: Includes costs in performance calculations

### Settlement Manager Integration
- **Settlement Costs**: Provides settlement fee calculations
- **Cash Flow Impact**: Accounts for costs in cash flow modeling

### Performance Metrics Integration
- **Cost Analysis**: Includes costs in performance metrics
- **Cost Efficiency**: Calculates cost efficiency ratios

## Usage Examples

### Basic Cost Calculation
```python
from engine.tca import TransactionCostAnalyzer

# Initialize TCA
tca = TransactionCostAnalyzer()

# Calculate costs for swing trade
costs = tca.calculate_costs(10000.0, is_buy=True, trade_type="swing_trade")
print(f"Total costs: R$ {costs.total_costs:,.2f}")
print(f"Cost breakdown: {costs}")
```

### Cost Comparison
```python
# Compare day trade vs swing trade costs
comparison = tca.compare_costs(5000.0)
for trade_type, costs in comparison.items():
    print(f"{trade_type}: R$ {costs['total_costs']:,.2f} ({costs['cost_percentage']:.4f}%)")
```

### Cost Summary
```python
# Get detailed cost summary
summary = tca.get_cost_summary(15000.0, is_buy=False, trade_type="day_trade")
print(f"Brokerage: R$ {summary['brokerage_fee']:,.2f}")
print(f"Emolument: R$ {summary['emolument']:,.2f}")
print(f"Settlement: R$ {summary['settlement_fee']:,.2f}")
print(f"ISS: R$ {summary['iss_fee']:,.2f}")
print(f"Total: R$ {summary['total_costs']:,.2f}")
```

### Custom Configuration
```python
# Update cost parameters for different broker
tca.update_cost_parameters({
    'brokerage_fee': 0.005,  # 0.5%
    'min_brokerage': 5.0     # R$ 5 minimum
})

# Calculate costs with new parameters
costs = tca.calculate_costs(1000.0, is_buy=True, trade_type="swing_trade")
print(f"New total costs: R$ {costs.total_costs:,.2f}")
```

## Performance Considerations

### Caching
- **Parameter Caching**: Cost parameters cached after validation
- **Calculation Efficiency**: Optimized mathematical operations
- **Memory Usage**: Minimal memory footprint

### Scalability
- **Batch Processing**: Support for multiple cost calculations
- **Parallel Processing**: Thread-safe operations
- **Large Orders**: Efficient handling of large order values

## Testing

The module includes comprehensive tests:

```bash
# Run TCA tests
python -m unittest tests.test_tca
```

## Dependencies

- **yaml**: Configuration loading
- **logging**: Logging and debugging
- **dataclasses**: Data structure definition
- **typing**: Type hints and validation

## Best Practices

1. **Always validate inputs**: Check order values and trade types
2. **Use appropriate trade types**: Distinguish between day trade and swing trade
3. **Monitor cost impact**: Track costs relative to order values
4. **Update configurations**: Keep cost parameters current
5. **Handle edge cases**: Consider minimum brokerage scenarios
6. **Document cost structure**: Maintain clear cost documentation
7. **Test configurations**: Validate cost parameters regularly 