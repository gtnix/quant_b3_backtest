# Data Loader Module Documentation

## Overview

The `DataLoader` module is responsible for loading, preprocessing, and managing B3 market data for backtesting. It provides comprehensive data quality management, technical indicator calculation, and corporate action handling.

**File Location**: `quant_backtest/engine/loader.py`

## Key Features

- **Raw Data Loading**: Load CSV data from Alpha Vantage downloads
- **Data Quality Filters**: Remove problematic data points and outliers
- **Liquidity Filters**: Filter based on volume and price criteria
- **Technical Indicators**: Calculate common technical indicators
- **Corporate Actions**: Handle dividends and stock splits
- **Data Validation**: Comprehensive input validation and error handling

## Class: DataLoader

### Constructor

```python
def __init__(self, raw_path: str = "data/raw", processed_path: str = "data/processed")
```

**Parameters**:
- `raw_path` (str): Path to raw data directory (default: "data/raw")
- `processed_path` (str): Path to processed data directory (default: "data/processed")

**B3-Specific Configuration**:
- Minimum daily volume: R$ 1,000,000
- Minimum price: R$ 1.00 (avoid penny stocks)
- Maximum daily price change: 20%

### Core Methods

#### `load_raw_data(ticker: str) -> Optional[pd.DataFrame]`

Loads raw data for a given ticker from CSV files.

**Parameters**:
- `ticker` (str): Stock ticker symbol

**Returns**:
- `Optional[pd.DataFrame]`: Raw data or None if file not found

**Features**:
- Handles column name variations from Alpha Vantage
- Maps numbered prefixes (e.g., "7. dividend amount" → "dividend_amount")
- Validates required columns (open, high, low, close, adjusted_close, volume)
- Converts numeric columns with error handling

**Example**:
```python
loader = DataLoader()
data = loader.load_raw_data("PETR4")
if data is not None:
    print(f"Loaded {len(data)} rows for PETR4")
```

#### `load_metadata(ticker: str) -> Optional[Dict]`

Loads metadata for a given ticker from JSON files.

**Parameters**:
- `ticker` (str): Stock ticker symbol

**Returns**:
- `Optional[Dict]`: Metadata or None if file not found

#### `apply_data_quality_filters(data: pd.DataFrame) -> pd.DataFrame`

Applies data quality filters to remove problematic data points.

**Filters Applied**:
- Remove rows with missing values
- Remove rows with zero or negative prices
- Remove rows with zero volume

**Returns**:
- `pd.DataFrame`: Filtered data

#### `apply_liquidity_filters(data: pd.DataFrame) -> pd.DataFrame`

Applies liquidity filters based on B3 market criteria.

**Filters Applied**:
- Minimum daily volume: R$ 1,000,000
- Minimum price: R$ 1.00
- Maximum daily price change: 20%

**Returns**:
- `pd.DataFrame`: Filtered data

#### `calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame`

Calculates common technical indicators for strategy development.

**Indicators Calculated**:
- **SMA (Simple Moving Average)**: 20, 50, 200-day periods
- **EMA (Exponential Moving Average)**: 12, 26-day periods
- **RSI (Relative Strength Index)**: 14-day period
- **MACD**: 12, 26, 9-day periods
- **Bollinger Bands**: 20-day period, 2 standard deviations
- **ATR (Average True Range)**: 14-day period
- **Volume SMA**: 20-day period

**Returns**:
- `pd.DataFrame`: Data with technical indicators added

#### `handle_corporate_actions(data: pd.DataFrame) -> pd.DataFrame`

Handles corporate actions like dividends and stock splits.

**Features**:
- Dividend adjustment for price data
- Split coefficient application
- Forward-fill missing corporate action data

**Returns**:
- `pd.DataFrame`: Data with corporate actions applied

#### `load_and_process(ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None, save_processed: bool = True) -> Optional[pd.DataFrame]`

Complete data loading and processing pipeline.

**Parameters**:
- `ticker` (str): Stock ticker symbol
- `start_date` (Optional[str]): Start date filter (YYYY-MM-DD)
- `end_date` (Optional[str]): End date filter (YYYY-MM-DD)
- `save_processed` (bool): Save processed data to file

**Processing Steps**:
1. Load raw data
2. Apply data quality filters
3. Apply liquidity filters
4. Calculate technical indicators
5. Handle corporate actions
6. Save processed data (if enabled)

**Returns**:
- `Optional[pd.DataFrame]`: Processed data or None if error

**Example**:
```python
loader = DataLoader()
processed_data = loader.load_and_process(
    ticker="PETR4",
    start_date="2023-01-01",
    end_date="2023-12-31",
    save_processed=True
)
```

#### `get_available_tickers() -> List[str]`

Returns list of available tickers with processed data.

**Returns**:
- `List[str]`: List of available ticker symbols

#### `get_processed_data(ticker: str) -> Optional[pd.DataFrame]`

Loads previously processed data for a ticker.

**Parameters**:
- `ticker` (str): Stock ticker symbol

**Returns**:
- `Optional[pd.DataFrame]`: Processed data or None if not found

## Data Quality Standards

### Required Columns
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `adjusted_close`: Adjusted closing price
- `volume`: Trading volume

### Optional Columns
- `dividend_amount`: Dividend amount
- `split_coefficient`: Stock split coefficient

### Data Validation Rules
- All prices must be positive
- Volume must be positive
- No missing values in required columns
- Maximum daily price change: 20%
- Minimum daily volume: R$ 1,000,000

## Technical Indicators

### Moving Averages
- **SMA**: Simple moving average for trend identification
- **EMA**: Exponential moving average for responsive signals

### Momentum Indicators
- **RSI**: Relative Strength Index for overbought/oversold conditions
- **MACD**: Moving Average Convergence Divergence for trend changes

### Volatility Indicators
- **Bollinger Bands**: Price volatility and potential reversal points
- **ATR**: Average True Range for volatility measurement

### Volume Indicators
- **Volume SMA**: Average volume for liquidity assessment

## Corporate Actions Handling

### Dividends
- Dividend amounts are preserved in the data
- Price adjustments are handled automatically
- Dividend yield can be calculated from dividend amounts

### Stock Splits
- Split coefficients are applied to historical data
- Ensures price continuity across split events
- Maintains accurate historical price levels

## Error Handling

The module includes comprehensive error handling:

- **File Not Found**: Graceful handling of missing data files
- **Invalid Data**: Validation of data quality and format
- **API Errors**: Handling of Alpha Vantage API issues
- **Memory Issues**: Efficient data processing for large datasets

## Performance Considerations

- **Caching**: Processed data is cached to avoid reprocessing
- **Lazy Loading**: Data is loaded only when needed
- **Memory Optimization**: Efficient DataFrame operations
- **Parallel Processing**: Support for concurrent data loading

## Usage Examples

### Basic Data Loading
```python
from engine.loader import DataLoader

# Initialize loader
loader = DataLoader()

# Load and process data
data = loader.load_and_process("PETR4")
print(f"Loaded {len(data)} rows of data")
```

### Custom Date Range
```python
# Load data for specific period
data = loader.load_and_process(
    ticker="VALE3",
    start_date="2023-01-01",
    end_date="2023-12-31"
)
```

### Check Available Data
```python
# Get list of available tickers
tickers = loader.get_available_tickers()
print(f"Available tickers: {tickers}")
```

### Load Processed Data
```python
# Load previously processed data
data = loader.get_processed_data("ITUB4")
if data is not None:
    print(f"RSI values: {data['rsi_14'].tail()}")
```

## Integration with Other Modules

The DataLoader integrates with:

- **Portfolio Manager**: Provides processed data for position management
- **Base Strategy**: Supplies technical indicators for signal generation
- **Backtest Simulator**: Feeds data into simulation engine
- **Performance Metrics**: Provides data for performance calculation

## Configuration

The module uses the following configuration parameters:

```yaml
# Data quality thresholds
min_volume_brl: 1000000  # Minimum daily volume
min_price: 1.0          # Minimum price
max_price_change: 0.20  # Maximum daily price change

# Technical indicator parameters
sma_periods: [20, 50, 200]
ema_periods: [12, 26]
rsi_period: 14
macd_fast: 12
macd_slow: 26
macd_signal: 9
bollinger_period: 20
bollinger_std: 2
atr_period: 14
```

## Testing

The module includes comprehensive tests:

```bash
# Run loader tests
python -m unittest tests.test_loader
```

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **pathlib**: File path handling
- **logging**: Logging and debugging
- **json**: Metadata file handling
- **datetime**: Date and time operations

## Best Practices

1. **Always validate data**: Check for missing or invalid data before processing
2. **Use processed data**: Prefer processed data over raw data for consistency
3. **Handle errors gracefully**: Implement proper error handling for missing files
4. **Monitor data quality**: Regularly check data quality metrics
5. **Cache processed data**: Save processed data to avoid reprocessing
6. **Use appropriate date ranges**: Load only necessary data to improve performance 