# Backtest Simulator Module Documentation

## Overview

The `BacktestSimulator` module is the main simulation engine for Brazilian market backtesting. It integrates all system components to provide comprehensive backtesting capabilities with full Brazilian market compliance, performance metrics calculation, and detailed reporting.

**File Location**: `quant_backtest/engine/simulator.py`

## Key Features

- **Strategy-Agnostic Design**: Works with any strategy implementing BaseStrategy
- **Comprehensive Performance Tracking**: Detailed metrics and analysis
- **Brazilian Market Compliance**: Full compliance with tax and settlement rules
- **SGS Data Integration**: Dynamic interest rate and inflation data
- **Transaction Cost Analysis**: Accurate cost calculation and tracking
- **Settlement Management**: T+2 settlement with business day handling
- **Loss Carryforward**: Advanced loss tracking and application
- **Detailed Logging**: Comprehensive audit trails and error handling

## Classes

### SimulationResult

A dataclass representing comprehensive simulation results.

```python
@dataclass
class SimulationResult:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_loss_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    final_portfolio_value: float
    initial_capital: float
    total_commission: float
    total_taxes: float
    daily_returns: List[float]
    portfolio_values: List[float]
    trade_log: List[Dict[str, Any]]
    simulation_duration: float
    start_date: datetime
    end_date: datetime
    # Benchmark metrics (optional)
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    information_ratio: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    tracking_error: float = 0.0
    rolling_correlation: float = 0.0
    benchmark_sharpe: float = 0.0
    benchmark_max_drawdown: float = 0.0
    benchmark_win_rate: float = 0.0
```

### BacktestSimulator

The main simulation engine class.

#### Constructor

```python
def __init__(self, strategy: BaseStrategy, initial_capital: float = 100000.0, 
             start_date: Optional[str] = None, end_date: Optional[str] = None,
             config_path: str = "config/settings.yaml")
```

**Parameters**:
- `strategy` (BaseStrategy): Strategy instance to simulate
- `initial_capital` (float): Starting capital in BRL
- `start_date` (Optional[str]): Simulation start date (YYYY-MM-DD)
- `end_date` (Optional[str]): Simulation end date (YYYY-MM-DD)
- `config_path` (str): Path to configuration file

**Initialization Process**:
1. Validate strategy implementation
2. Parse and validate dates
3. Initialize portfolio with strategy
4. Set up performance metrics
5. Configure logging

## Core Methods

### Data Preparation

#### `prepare_data(data: pd.DataFrame) -> pd.DataFrame`

Prepares market data for simulation with validation and preprocessing.

**Parameters**:
- `data` (pd.DataFrame): Raw market data

**Process**:
1. Validate data structure and quality
2. Apply Brazilian market constraints
3. Add technical indicators if needed
4. Handle missing data
5. Sort by date

**Returns**:
- `pd.DataFrame`: Prepared data for simulation

### Simulation Execution

#### `run_simulation(data: pd.DataFrame) -> SimulationResult`

Executes the complete backtest simulation.

**Parameters**:
- `data` (pd.DataFrame): Prepared market data

**Simulation Process**:
1. Initialize simulation state
2. Iterate through each trading day
3. Prepare market data for current date
4. Load SGS data (interest rates, inflation)
5. Generate trading signals
6. Execute trades with validation
7. Update portfolio and settlements
8. Calculate daily performance
9. Generate final results

**Returns**:
- `SimulationResult`: Comprehensive simulation results

**Example**:
```python
from engine.simulator import BacktestSimulator
from engine.base_strategy import BaseStrategy

# Initialize simulator
simulator = BacktestSimulator(
    strategy=my_strategy,
    initial_capital=100000.0,
    start_date="2023-01-01",
    end_date="2023-12-31"
)

# Run simulation
result = simulator.run_simulation(market_data)
print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")
```

### Market Data Preparation

#### `_prepare_market_data(data: pd.DataFrame, current_date: datetime) -> Dict[str, Any]`

Prepares market data for the current simulation date.

**Parameters**:
- `data` (pd.DataFrame): Full market dataset
- `current_date` (datetime): Current simulation date

**Returns**:
- `Dict[str, Any]`: Market data for current date including:
  - Price data
  - Technical indicators
  - Volume data
  - Market conditions

### SGS Data Integration

#### `_load_sgs_data_for_date(current_date: datetime) -> Dict[str, float]`

Loads Banco Central SGS data for the current date.

**Parameters**:
- `current_date` (datetime): Current simulation date

**SGS Series**:
- **Series 11**: SELIC interest rate
- **Series 12**: CDI interest rate
- **Series 433**: IPCA inflation index

**Returns**:
- `Dict[str, float]`: SGS data for current date

#### `_calculate_selic_cdi_spread(sgs_data: Dict[str, float]) -> Optional[float]`

Calculates SELIC-CDI spread for market analysis.

**Parameters**:
- `sgs_data` (Dict[str, float]): SGS data dictionary

**Returns**:
- `Optional[float]`: SELIC-CDI spread or None if data unavailable

#### `_classify_interest_rate_environment(sgs_data: Dict[str, float]) -> str`

Classifies current interest rate environment.

**Classifications**:
- **High**: SELIC > 12%
- **Moderate**: 8% ≤ SELIC ≤ 12%
- **Low**: SELIC < 8%

**Returns**:
- `str`: Interest rate environment classification

#### `_classify_inflation_environment(sgs_data: Dict[str, float]) -> str`

Classifies current inflation environment.

**Classifications**:
- **High**: IPCA > 6%
- **Moderate**: 3% ≤ IPCA ≤ 6%
- **Low**: IPCA < 3%

**Returns**:
- `str`: Inflation environment classification

### Trade Execution

#### `_execute_trade(signal: TradingSignal, price_data: pd.Series) -> None`

Executes a trading signal with comprehensive validation.

**Parameters**:
- `signal` (TradingSignal): Trading signal to execute
- `price_data` (pd.Series): Current price data

**Execution Process**:
1. Validate signal parameters
2. Check Brazilian market constraints
3. Calculate position size
4. Execute trade through portfolio
5. Record trade in simulation log
6. Update performance metrics

### Performance Calculation

#### `_calculate_performance_metrics() -> None`

Calculates comprehensive performance metrics.

**Metrics Calculated**:
- **Returns**: Total, annualized, and daily returns
- **Risk Metrics**: Sharpe ratio, max drawdown, volatility
- **Trade Metrics**: Win rate, profit factor, average trade
- **Cost Metrics**: Total commission, taxes, net profit
- **Benchmark Metrics**: Alpha, beta, information ratio

#### `_create_simulation_result() -> SimulationResult`

Creates comprehensive simulation result object.

**Returns**:
- `SimulationResult`: Complete simulation results

### Reporting and Analysis

#### `get_performance_summary() -> Dict[str, Any]`

Returns comprehensive performance summary.

**Returns**:
- `Dict[str, Any]`: Performance summary including all metrics

#### `export_results(filepath: str) -> None`

Exports simulation results to file.

**Parameters**:
- `filepath` (str): Path to export file

**Exported Data**:
- Performance metrics
- Trade log
- Daily returns
- Portfolio values
- Configuration used

#### `get_summary_data() -> Dict[str, Any]`

Returns summary data for reporting.

**Returns**:
- `Dict[str, Any]`: Summary data for external reporting

## Brazilian Market Integration

### Tax Compliance
- **Swing Trade Tax**: 15% on monthly net profit
- **Day Trade Tax**: 20% on monthly net profit
- **Exemption**: R$ 20,000/month for swing trades
- **Loss Carryforward**: 100% offset capability

### Settlement Rules
- **T+2 Settlement**: Standard Brazilian settlement cycle
- **Business Day Handling**: Uses Brazilian business day calendar
- **Cash Flow Modeling**: Realistic cash flow with settlement delays

### Market Constraints
- **Trading Hours**: 10:00-16:55 (continuous session)
- **Minimum Volume**: R$ 1,000,000 daily volume
- **Price Limits**: Maximum 20% daily price change
- **Position Limits**: Configurable maximum positions

## SGS Data Integration

### Interest Rate Analysis
- **SELIC Rate**: Brazilian benchmark interest rate
- **CDI Rate**: Interbank deposit rate
- **Spread Analysis**: SELIC-CDI spread for market conditions
- **Environment Classification**: High/Moderate/Low rate environments

### Inflation Analysis
- **IPCA Index**: Consumer price index
- **Inflation Environment**: High/Moderate/Low inflation periods
- **Economic Context**: Inflation impact on strategy performance

## Error Handling

### Strategy Validation
- **Required Methods**: Validates strategy implements all required methods
- **Method Signatures**: Checks method parameter compatibility
- **Return Types**: Validates method return types

### Data Validation
- **Data Quality**: Checks for missing or invalid data
- **Date Range**: Validates simulation date range
- **Market Data**: Ensures required columns and data types

### Simulation Errors
- **Trade Execution**: Handles failed trade executions
- **Settlement Errors**: Manages settlement failures
- **Data Errors**: Handles missing SGS or market data

## Performance Optimization

### Caching
- **SGS Data**: Caches SGS data to avoid repeated API calls
- **Business Days**: Caches business day calculations
- **Technical Indicators**: Caches calculated indicators

### Memory Management
- **Data Streaming**: Processes data in chunks for large datasets
- **Garbage Collection**: Manages memory usage during simulation
- **Efficient Data Structures**: Uses optimized data structures

## Usage Examples

### Basic Simulation
```python
from engine.simulator import BacktestSimulator
from engine.loader import DataLoader

# Load data
loader = DataLoader()
data = loader.load_and_process("PETR4", "2023-01-01", "2023-12-31")

# Create simulator
simulator = BacktestSimulator(
    strategy=my_strategy,
    initial_capital=100000.0,
    start_date="2023-01-01",
    end_date="2023-12-31"
)

# Run simulation
result = simulator.run_simulation(data)
```

### Performance Analysis
```python
# Get performance summary
summary = simulator.get_performance_summary()
print(f"Total Return: {summary['total_return']:.2%}")
print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {summary['max_drawdown']:.2%}")
print(f"Win Rate: {summary['win_rate']:.2%}")
```

### Export Results
```python
# Export results to file
simulator.export_results("backtest_results.json")

# Get summary data for external reporting
summary_data = simulator.get_summary_data()
```

## Configuration

The simulator uses configuration from `config/settings.yaml`:

```yaml
market:
  trading_hours:
    open: "10:00"
    close: "16:55"
    timezone: "America/Sao_Paulo"
  selic_rate: 0.15
  trading_days_per_year: 252

benchmark:
  enabled: true
  symbol: "IBOV"
  auto_load: true
  required: true

sgs:
  strict_mode:
    enabled: true
    require_selic_data: true
    minimum_coverage_percentage: 95.0
```

## Testing

The module includes comprehensive tests:

```bash
# Run simulator tests
python -m unittest tests.test_simulator
```

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **pytz**: Timezone handling
- **yaml**: Configuration loading
- **logging**: Logging and debugging
- **datetime**: Date and time operations

## Best Practices

1. **Validate strategy implementation**: Ensure strategy implements all required methods
2. **Check data quality**: Validate market data before simulation
3. **Monitor SGS data**: Ensure SGS data availability for accurate simulation
4. **Handle errors gracefully**: Implement proper error handling for simulation failures
5. **Optimize performance**: Use caching and efficient data structures
6. **Document results**: Export and document simulation results
7. **Test thoroughly**: Run comprehensive tests before production use 