# Data Downloader Script Documentation

## Overview

The `download_data.py` script provides comprehensive functionality to download and manage Brazilian stock market (B3) data using the Alpha Vantage API. It includes robust error handling, retry logic, data consolidation, and proper data organization.

**Note**: This script downloads stock data only. IBOV index data is downloaded using Yahoo Finance via `download_ibov_yahoo.py`.

**File Location**: `quant_backtest/scripts/download_data.py`

## Key Features

- **Alpha Vantage Integration**: Automated data download with rate limiting
- **Rate-Limited API Calls**: 5 calls/minute for free tier compliance
- **Comprehensive Error Handling**: Retry mechanisms and fallback strategies
- **Flexible Ticker Management**: IBRA tickers with fallback mechanisms
- **Data Consolidation**: Long-format CSV with metadata
- **Progress Tracking**: Detailed logging and statistics
- **Configuration Management**: Secure API key management
- **Adjusted Data Only**: Includes corporate action adjustments
- **Stock Data Only**: Downloads individual stock data (not indices)

## Classes

### DownloadResult

A dataclass representing the result of a data download operation.

```python
@dataclass
class DownloadResult:
    ticker: str
    success: bool
    data_points: int
    date_range: Optional[Dict[str, str]]
    error_message: Optional[str] = None
    download_time: Optional[float] = None
```

### DownloadStats

A dataclass for tracking download operation statistics.

```python
@dataclass
class DownloadStats:
    total_requests: int = 0
    successful_downloads: int = 0
    failed_downloads: int = 0
    total_data_points: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
```

**Properties**:
- `success_rate`: Calculate success rate as percentage
- `duration`: Calculate total duration in seconds

### ConfigurationManager

Manages configuration loading and validation.

#### Constructor

```python
def __init__(self, config_path: str = "config/secrets.yaml")
```

**Parameters**:
- `config_path` (str): Path to the secrets configuration file

**Features**:
- Loads API keys from secrets.yaml
- Validates configuration parameters
- Provides default values
- Error handling for missing configurations

### EnhancedB3DataDownloader

The main data downloader class with comprehensive features.

#### Constructor

```python
def __init__(self, config_path: str = "config/secrets.yaml")
```

**Parameters**:
- `config_path` (str): Path to configuration file

**Initialized Components**:
- **Configuration Manager**: API key and settings management
- **Session Management**: HTTP session with retry logic
- **Rate Limiting**: 5 calls/minute compliance
- **Statistics Tracking**: Download operation statistics

## Core Methods

### Rate Limiting and Retry Logic

#### `_rate_limit()`

Implements rate limiting for Alpha Vantage API compliance.

**Rate Limit**: 5 calls per minute for free tier
**Implementation**: Thread-safe rate limiting with sleep intervals

#### `_make_request_with_retry(params: Dict) -> Optional[Dict]`

Makes API requests with comprehensive retry logic.

**Parameters**:
- `params` (Dict): API request parameters

**Retry Logic**:
- Maximum 3 retry attempts
- Exponential backoff between retries
- Error handling for various HTTP status codes
- Timeout handling (30 seconds)

**Returns**:
- `Optional[Dict]`: API response or None if failed

### Individual Ticker Download

#### `download_ticker_data(ticker: str) -> DownloadResult`

Downloads data for a single ticker with comprehensive error handling.

**Parameters**:
- `ticker` (str): Stock ticker symbol

**Process**:
1. Validate ticker symbol
2. Apply rate limiting
3. Make API request with retry logic
4. Parse and validate response
5. Save data to CSV and metadata to JSON
6. Return download result

**Returns**:
- `DownloadResult`: Download operation result

**Example**:
```python
downloader = EnhancedB3DataDownloader()
result = downloader.download_ticker_data("PETR4")
if result.success:
    print(f"Downloaded {result.data_points} data points for {result.ticker}")
else:
    print(f"Failed to download {result.ticker}: {result.error_message}")
```

### Data Saving

#### `_save_ticker_data(ticker: str, df: pd.DataFrame)`

Saves ticker data to CSV and metadata to JSON files.

**Parameters**:
- `ticker` (str): Stock ticker symbol
- `df` (pd.DataFrame): Data to save

**Saved Files**:
- `{ticker}_raw.csv`: Raw price data
- `{ticker}_meta.json`: Metadata including symbol info



### Multiple Ticker Download

#### `download_multiple_tickers(tickers: List[str], max_workers: int = 1) -> List[DownloadResult]`

Downloads data for multiple tickers with optional parallel processing.

**Parameters**:
- `tickers` (List[str]): List of ticker symbols
- `max_workers` (int): Maximum parallel workers (default: 1 for rate limiting)

**Returns**:
- `List[DownloadResult]`: List of download results

**Example**:
```python
downloader = EnhancedB3DataDownloader()
tickers = ["PETR4", "VALE3", "ITUB4"]
results = downloader.download_multiple_tickers(tickers)
for result in results:
    if result.success:
        print(f"✓ {result.ticker}: {result.data_points} points")
    else:
        print(f"✗ {result.ticker}: {result.error_message}")
```

### Data Consolidation

#### `consolidate_data(output_filename: str = "b3_consolidated_data.csv") -> Optional[str]`

Consolidates all downloaded data into a single long-format CSV file.

**Parameters**:
- `output_filename` (str): Output filename

**Consolidation Process**:
1. Load all available ticker data
2. Add ticker column to each dataset
3. Concatenate all datasets
4. Sort by date and ticker
5. Save consolidated file

**Returns**:
- `Optional[str]`: Path to consolidated file or None if failed

### Statistics and Reporting

#### `_log_download_statistics()`

Logs comprehensive download statistics.

**Statistics Logged**:
- Total requests made
- Success/failure counts
- Success rate percentage
- Total data points downloaded
- Duration of download operation

#### `run_complete_download(tickers: Optional[List[str]] = None, max_tickers: Optional[int] = None, consolidate: bool = True, output_filename: str = "b3_consolidated_data.csv") -> bool`

Runs a complete download operation with all features.

**Parameters**:
- `tickers` (Optional[List[str]]): List of tickers to download
- `max_tickers` (Optional[int]): Maximum number of tickers to download
- `consolidate` (bool): Whether to consolidate data
- `output_filename` (str): Consolidated output filename

**Process**:
1. Initialize download statistics
2. Download individual ticker data
3. Download IBOV data
4. Consolidate data if requested
5. Log comprehensive statistics
6. Return success status

**Returns**:
- `bool`: True if operation completed successfully

## Alpha Vantage API Integration

### API Configuration
- **Base URL**: https://www.alphavantage.co/query
- **Data Type**: Adjusted data only (includes corporate actions)
- **Rate Limit**: 5 calls per minute (free tier)
- **Timeout**: 30 seconds per request

### Supported Data
- **Daily Adjusted**: OHLCV with corporate action adjustments
- **Metadata**: Symbol information and data details
- **Error Handling**: Comprehensive error response handling

### API Parameters
```python
params = {
    "function": "TIME_SERIES_DAILY_ADJUSTED",
    "symbol": ticker,
    "outputsize": "full",
    "datatype": "json",
    "apikey": api_key
}
```

## Data Structure

### Raw Data Format
```csv
timestamp,open,high,low,close,adjusted_close,volume,dividend_amount,split_coefficient
2023-01-01,25.50,25.75,25.25,25.60,25.60,1000000,0.0,1.0
```

### Metadata Format
```json
{
    "Information": "Daily Prices (open, high, low, close) and Volumes",
    "Symbol": "PETR4.SAO",
    "Last Refreshed": "2023-12-31",
    "Output Size": "Full size",
    "Time Zone": "US/Eastern"
}
```

## Error Handling

### API Errors
- **Rate Limit Exceeded**: Automatic retry with backoff
- **Invalid API Key**: Clear error message and exit
- **Symbol Not Found**: Graceful handling with error logging
- **Network Errors**: Retry logic with exponential backoff

### Data Validation
- **Empty Responses**: Validation of API response content
- **Invalid Data**: Checking for required columns and data types
- **Date Range**: Validation of date ranges in responses

### File System Errors
- **Directory Creation**: Automatic creation of data directories
- **File Write Errors**: Error handling for file system issues
- **Permission Errors**: Clear error messages for permission issues

## Configuration

### Secrets Configuration (`config/secrets.yaml`)
```yaml
alpha_vantage:
  api_key: "YOUR_ALPHA_VANTAGE_API_KEY"
  base_url: "https://www.alphavantage.co/query"
  data_type: "adjusted"
  rate_limit_calls_per_minute: 5
  max_retries: 3
  timeout_seconds: 30
```

### Required Setup
1. Copy `config/secrets.yaml.example` to `config/secrets.yaml`
2. Add your Alpha Vantage API key
3. Ensure data directories exist

## Usage Examples

### Basic Download
```python
from scripts.download_data import EnhancedB3DataDownloader

# Initialize downloader
downloader = EnhancedB3DataDownloader()

# Download single ticker
result = downloader.download_ticker_data("PETR4")
print(f"Success: {result.success}, Points: {result.data_points}")
```

### Multiple Ticker Download
```python
# Download multiple tickers
tickers = ["PETR4", "VALE3", "ITUB4"]
results = downloader.download_multiple_tickers(tickers)

# Check results
for result in results:
    print(f"{result.ticker}: {'✓' if result.success else '✗'}")
```

### Complete Download Operation
```python
# Run complete download with consolidation
success = downloader.run_complete_download(
    tickers=["PETR4", "VALE3"],
    consolidate=True,
    output_filename="my_consolidated_data.csv"
)
print(f"Download completed: {success}")
```

### IBOV Download
```python
# Download IBOV data
ibov_result = downloader.download_ibov_data()
if ibov_result.success:
    print(f"IBOV downloaded: {ibov_result.data_points} points")
```

## Command Line Usage

### Basic Usage
```bash
# Download data for test symbols
python scripts/download_data.py

# Download specific tickers
python scripts/download_data.py --tickers PETR4,VALE3,ITUB4

# Download with consolidation
python scripts/download_data.py --consolidate
```

### Advanced Options
```bash
# Download with custom output
python scripts/download_data.py --output consolidated_b3_data.csv

# Download with parallel processing (use with caution due to rate limits)
python scripts/download_data.py --max-workers 2

# Download with progress tracking
python scripts/download_data.py --verbose
```

## Performance Considerations

### Rate Limiting
- **Free Tier**: 5 calls per minute maximum
- **Premium Tier**: Higher limits available
- **Compliance**: Automatic rate limiting to avoid API blocks

### Memory Management
- **Streaming**: Processes data in chunks for large datasets
- **Garbage Collection**: Manages memory usage during downloads
- **Efficient Storage**: Optimized CSV and JSON storage

### Parallel Processing
- **Default**: Single-threaded for rate limit compliance
- **Optional**: Multi-threading for premium API keys
- **Caution**: Parallel processing may exceed rate limits

## Testing

The script includes comprehensive error handling and validation:

```bash
# Test with small dataset
python scripts/download_data.py --tickers PETR4 --test

# Validate downloaded data
python -c "import pandas as pd; df = pd.read_csv('data/raw/PETR4_raw.csv'); print(df.head())"
```

## Dependencies

- **requests**: HTTP API calls
- **pandas**: Data manipulation and CSV handling
- **yaml**: Configuration file parsing
- **concurrent.futures**: Parallel processing support
- **logging**: Comprehensive logging
- **datetime**: Date and time operations

## Data Organization

### Directory Structure
The system organizes data in the following structure:

```
data/
├── IBOV/                   # IBOV index data (from Yahoo Finance)
│   ├── IBOV.parquet        # Main IBOV data file
│   ├── IBOV_raw.csv        # CSV version
│   └── raw/                # Raw timestamped files
│       ├── IBOV_YYYYMMDD.csv
│       ├── IBOV_YYYYMMDD.parquet
│       └── IBOV_YYYYMMDD_metadata.json
├── raw/                    # Raw CSV files with timestamps (stocks)
│   ├── TICKER_YYYYMMDD.csv
│   └── TICKER_YYYYMMDD_metadata.json
├── processed/              # Processed data files
└── consolidated/           # Consolidated data files
```

**Note**: IBOV data is stored separately from stock data for better organization and to reflect its different data source (Yahoo Finance vs Alpha Vantage).

## Best Practices

1. **API Key Security**: Never commit API keys to version control
2. **Rate Limit Compliance**: Respect Alpha Vantage rate limits
3. **Error Handling**: Always check download results
4. **Data Validation**: Validate downloaded data quality
5. **Backup Strategy**: Keep backups of important data
6. **Monitoring**: Monitor download success rates
7. **Documentation**: Document any custom configurations 