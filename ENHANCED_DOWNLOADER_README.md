# Enhanced B3 Data Downloader

## Overview

This enhanced data downloader provides comprehensive functionality to download and manage B3 (Brazilian Stock Exchange) stock data using the Alpha Vantage API. It's designed to be both powerful and beginner-friendly, with robust error handling, detailed logging, and flexible configuration options.

## Key Features

### 🚀 Core Functionality
- **Rate-limited API calls** (5 calls/minute for free tier)
- **Comprehensive error handling** with exponential backoff retry logic
- **Flexible ticker retrieval** with API and fallback mechanisms
- **Data consolidation** into long-format CSV files
- **Detailed logging** and progress tracking
- **Configuration management** with validation

### 📊 Data Processing
- Downloads daily adjusted prices for B3 tickers
- Uses full historical data (`outputsize="full"`)
- Filters to last 15 calendar years
- Handles B3 ticker naming conventions (.SA suffix)
- Creates consolidated long-format CSV with columns:
  - `date`
  - `ticker`
  - `open`
  - `high`
  - `low`
  - `close`
  - `adjusted_close`
  - `volume`

### 🔧 Technical Features
- **Type hints** throughout the codebase
- **Configurable parameters** via YAML configuration
- **Retry mechanisms** for failed downloads
- **Thread-safe rate limiting**
- **Comprehensive statistics** tracking
- **Modular architecture** with separate classes for different responsibilities

## Installation & Setup

### 1. Prerequisites
```bash
pip install requests pandas pyyaml
```

### 2. Configuration Setup
1. Copy the example configuration:
   ```bash
   cp config/secrets.yaml.example config/secrets.yaml
   ```

2. Edit `config/secrets.yaml` and add your Alpha Vantage API key:
   ```yaml
   alpha_vantage:
     api_key: "YOUR_ACTUAL_API_KEY_HERE"
     base_url: "https://www.alphavantage.co/query"
     rate_limit_calls_per_minute: 5
     max_retries: 3
     timeout_seconds: 30
   ```

### 3. Get Alpha Vantage API Key
1. Visit [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Sign up for a free account
3. Get your API key
4. Add it to `config/secrets.yaml`

## Usage Examples

### Basic Usage

```python
from scripts.download_data import EnhancedB3DataDownloader

# Initialize the downloader
downloader = EnhancedB3DataDownloader()

# Download specific tickers
test_tickers = ["VALE3", "PETR4", "ITUB4"]
results = downloader.download_multiple_tickers(test_tickers)

# Run complete download process
success = downloader.run_complete_download(
    max_tickers=10,  # Limit for testing
    consolidate=True,
    output_filename="b3_data.csv"
)
```

### Advanced Usage

```python
# Custom configuration
downloader = EnhancedB3DataDownloader("path/to/custom/secrets.yaml")

# Download with specific options
success = downloader.run_complete_download(
    tickers=["VALE3", "PETR4"],  # Specific tickers
    use_api_tickers=False,       # Use curated list only
    max_tickers=None,            # No limit
    consolidate=True,
    output_filename="custom_b3_data.csv"
)

# Manual consolidation
consolidated_path = downloader.consolidate_data("my_consolidated_data.csv")
```

### Command Line Usage

```bash
# Run the script directly
python scripts/download_data.py
```

## Architecture

### Class Structure

#### `ConfigurationManager`
- **Purpose**: Manages API configuration and validation
- **Location**: Lines 108-175 in `download_data.py`
- **Key Methods**:
  - `_load_configuration()`: Loads and validates YAML config
  - `get_alpha_vantage_config()`: Returns validated API config

#### `B3TickerManager`
- **Purpose**: Handles ticker retrieval with fallback mechanisms
- **Location**: Lines 177-250 in `download_data.py`
- **Key Methods**:
  - `get_tickers_from_api()`: Fetches tickers from Alpha Vantage
  - `get_tickers()`: Returns tickers with fallback to curated list

#### `EnhancedB3DataDownloader`
- **Purpose**: Main downloader class with comprehensive features
- **Location**: Lines 252-650 in `download_data.py`
- **Key Methods**:
  - `download_ticker_data()`: Downloads data for single ticker
  - `download_multiple_tickers()`: Downloads data for multiple tickers
  - `consolidate_data()`: Consolidates data into single CSV
  - `run_complete_download()`: Runs complete download process

### Data Flow

1. **Configuration Loading** → `ConfigurationManager`
2. **Ticker Retrieval** → `B3TickerManager`
3. **Data Download** → `EnhancedB3DataDownloader.download_ticker_data()`
4. **Rate Limiting** → `_rate_limit()` method
5. **Error Handling** → `_make_request_with_retry()` method
6. **Data Processing** → Column renaming, date filtering, numeric conversion
7. **File Saving** → Individual CSV and JSON metadata files
8. **Data Consolidation** → Single long-format CSV file

## Output Structure

### Directory Structure
```
quant_backtest/
├── data/
│   ├── raw/                    # Individual ticker files
│   │   ├── VALE3_raw.csv
│   │   ├── VALE3_meta.json
│   │   ├── PETR4_raw.csv
│   │   └── PETR4_meta.json
│   └── processed/              # Consolidated files
│       ├── b3_consolidated_data.csv
│       └── consolidation_metadata.json
├── logs/                       # Detailed log files
│   └── data_download_YYYYMMDD_HHMMSS.log
└── config/
    └── secrets.yaml            # API configuration
```

### File Formats

#### Individual Ticker CSV (`data/raw/TICKER_raw.csv`)
```csv
date,open,high,low,close,adjusted_close,volume,ticker
2023-01-03,67.50,68.20,67.10,67.85,67.85,1234567,VALE3
2023-01-04,67.85,68.50,67.30,68.20,68.20,1456789,VALE3
...
```

#### Consolidated CSV (`data/processed/b3_consolidated_data.csv`)
```csv
date,ticker,open,high,low,close,adjusted_close,volume
2023-01-03,VALE3,67.50,68.20,67.10,67.85,67.85,1234567
2023-01-03,PETR4,25.30,25.80,25.20,25.65,25.65,2345678
2023-01-04,VALE3,67.85,68.50,67.30,68.20,68.20,1456789
2023-01-04,PETR4,25.65,26.10,25.60,25.95,25.95,3456789
...
```

#### Metadata JSON (`data/raw/TICKER_meta.json`)
```json
{
  "ticker": "VALE3",
  "download_date": "2024-01-15T10:30:00",
  "data_points": 2500,
  "date_range": {
    "start": "2015-01-02T00:00:00",
    "end": "2024-01-15T00:00:00"
  },
  "columns": ["open", "high", "low", "close", "adjusted_close", "volume", "ticker"]
}
```

## Error Handling

### Comprehensive Error Management
- **API Errors**: Handles Alpha Vantage API error messages
- **Rate Limiting**: Automatic retry with exponential backoff
- **Network Issues**: Retry logic for connection failures
- **Data Validation**: Checks for missing or malformed data
- **File System**: Handles file creation and writing errors

### Error Recovery Strategies
1. **Exponential Backoff**: Waits longer between retries
2. **Fallback Mechanisms**: Uses curated ticker list if API fails
3. **Graceful Degradation**: Continues processing other tickers if one fails
4. **Detailed Logging**: Comprehensive error reporting for debugging

## Performance Optimization

### Rate Limiting
- **Thread-safe implementation** using locks
- **Configurable limits** (default: 5 calls/minute)
- **Automatic sleep** between API calls

### Parallel Processing
- **Optional parallel downloads** (use with caution due to rate limits)
- **ThreadPoolExecutor** for concurrent processing
- **Configurable worker count**

### Memory Management
- **Streaming data processing** for large datasets
- **Efficient DataFrame operations**
- **Automatic cleanup** of temporary data

## Monitoring & Logging

### Log Levels
- **DEBUG**: Detailed API calls and data processing
- **INFO**: Progress updates and statistics
- **WARNING**: Rate limiting and fallback usage
- **ERROR**: Failed downloads and configuration issues

### Statistics Tracking
- **Download success rate**
- **Total data points downloaded**
- **Processing time per ticker**
- **Overall duration and performance metrics**

### Log File Location
```
logs/data_download_YYYYMMDD_HHMMSS.log
```

## Configuration Options

### Alpha Vantage Configuration
```yaml
alpha_vantage:
  api_key: "YOUR_API_KEY"
  base_url: "https://www.alphavantage.co/query"
  rate_limit_calls_per_minute: 5    # API rate limit
  max_retries: 3                    # Retry attempts
  timeout_seconds: 30               # Request timeout
```

### Download Options
- **`tickers`**: Specific ticker list (optional)
- **`use_api_tickers`**: Whether to fetch from API (default: True)
- **`max_tickers`**: Limit number of tickers (optional)
- **`consolidate`**: Create consolidated CSV (default: True)
- **`output_filename`**: Name of consolidated file

## Best Practices

### For Beginners
1. **Start Small**: Use `max_tickers=5` for testing
2. **Check Logs**: Monitor log files for issues
3. **Validate Configuration**: Ensure API key is correct
4. **Use Fallback**: Enable fallback to curated ticker list

### For Advanced Users
1. **Custom Configuration**: Create custom secrets.yaml files
2. **Parallel Processing**: Use multiple workers for faster downloads
3. **Batch Processing**: Download tickers in batches
4. **Custom Consolidation**: Modify consolidation logic as needed

### Performance Tips
1. **Respect Rate Limits**: Don't exceed API limits
2. **Use Appropriate Timeouts**: Adjust timeout values based on network
3. **Monitor Memory Usage**: For large datasets, process in chunks
4. **Regular Cleanup**: Remove old log files periodically

## Troubleshooting

### Common Issues

#### Configuration Errors
```
❌ Configuration Error: Configuration file not found
```
**Solution**: Copy `config/secrets.yaml.example` to `config/secrets.yaml`

#### API Key Issues
```
❌ Configuration Error: Please provide a valid Alpha Vantage API key
```
**Solution**: Get a free API key from Alpha Vantage and add it to secrets.yaml

#### Rate Limiting
```
API Note: Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute
```
**Solution**: The system automatically handles rate limiting, just wait for retry

#### Network Issues
```
Request failed (attempt 1/4): Connection timeout
```
**Solution**: Check internet connection and increase timeout in configuration

### Debug Mode
Enable debug logging for detailed information:
```python
logger = setup_logging("DEBUG")
```

## Contributing

### Code Structure
- **Type hints**: All functions include type annotations
- **Docstrings**: Comprehensive documentation for all methods
- **Error handling**: Robust error handling throughout
- **Modular design**: Separate classes for different responsibilities

### Adding Features
1. **New data sources**: Extend `B3TickerManager` for additional sources
2. **Custom processing**: Modify data processing in `download_ticker_data()`
3. **Additional formats**: Extend `consolidate_data()` for new output formats
4. **Enhanced logging**: Add new log levels or output formats

## License

This implementation is part of the quant_b3_backtest project and follows the same licensing terms.

## Support

For issues and questions:
1. Check the log files for detailed error information
2. Review the configuration settings
3. Test with a small number of tickers first
4. Ensure your Alpha Vantage API key is valid and active

---

**Note**: This enhanced downloader is designed to be production-ready while remaining accessible to beginners. It includes comprehensive error handling, detailed logging, and flexible configuration options to handle various use cases and environments. 