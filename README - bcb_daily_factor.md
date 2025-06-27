# Market Data Module

This module provides functionality for retrieving and processing market data from various sources, with a focus on Brazilian market data including daily factors from Banco Central do Brasil (BCB).

## BCB Daily Factor Retriever

The `BCBDailyFactorRetriever` class provides a robust system for retrieving historical daily factor data from the Banco Central do Brasil API with B3 business days consideration.

### Features

- **Official BCB API Integration**: Fetches data directly from BCB's official API (series code 12)
- **B3 Business Day Alignment**: Aligns data to actual B3 trading calendar using `pandas_market_calendars`
- **Data Validation**: Removes outliers and interpolates missing values
- **Risk-Free Rate Calculation**: Converts daily factors to annualized risk-free rates
- **Local Caching**: Caches processed data in Parquet format for performance
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Error Handling**: Robust error handling with graceful degradation

### Installation

Ensure you have the required dependencies:

```bash
pip install -r requirements.txt
```

The module requires:
- `pandas>=2.1.0`
- `numpy>=1.24.0`
- `requests>=2.31.0`
- `pyarrow>=14.0.0`
- `pandas-market-calendars>=2.0.0`
- `pyyaml>=6.0`

### Configuration

The module uses the configuration from `config/settings.yaml`. The relevant section is:

```yaml
market:
  daily_factor:
    source: 'bcb_official_api'
    validation:
      min_factor: 0.95
      max_factor: 1.05
    cache:
      directory: 'data/daily_factors/'
      format: 'parquet'
```

### Usage

#### Basic Usage

```python
from market_data.bcb_daily_factor import BCBDailyFactorRetriever
from datetime import datetime

# Initialize retriever
retriever = BCBDailyFactorRetriever()

# Fetch daily factors
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)
factors_df = retriever.fetch_daily_factors(start_date, end_date)

# Calculate risk-free curve
risk_free_curve = retriever.calculate_risk_free_curve(factors_df)

# Get risk-free rate for specific date
rate = retriever.get_risk_free_rate_for_date(datetime(2024, 6, 15))
```

#### Command Line Interface

The module includes a CLI for easy data management:

```bash
# Fetch data
python scripts/bcb_factor_cli.py fetch --start-date 2024-01-01 --end-date 2024-12-31

# Show cached data info
python scripts/bcb_factor_cli.py info --verbose

# Get risk-free rate for specific date
python scripts/bcb_factor_cli.py get-rate --date 2024-06-15

# Clear cache
python scripts/bcb_factor_cli.py clear-cache
```

### API Reference

#### BCBDailyFactorRetriever

##### Constructor

```python
BCBDailyFactorRetriever(config_path="config/settings.yaml", cache_dir="data/daily_factors/")
```

- `config_path`: Path to configuration file
- `cache_dir`: Directory for caching data

##### Methods

###### `fetch_daily_factors(start_date, end_date)`

Fetches and processes daily factors from BCB API.

- `start_date`: Start date for data retrieval
- `end_date`: End date for data retrieval
- Returns: DataFrame with processed daily factors

###### `calculate_risk_free_curve(factors_df)`

Calculates annualized risk-free rate curve from daily factors.

- `factors_df`: DataFrame with daily factors
- Returns: Series with annualized risk-free rates

###### `get_risk_free_rate_for_date(target_date, lookback_days=30)`

Gets risk-free rate for a specific date.

- `target_date`: Target date for rate calculation
- `lookback_days`: Number of days to look back (default: 30)
- Returns: Risk-free rate as float or None if not available

###### `load_cached_factors()`

Loads cached factors from local storage.

- Returns: DataFrame with cached factors or None if not found

### Data Format

#### Input Data (BCB API Response)

The BCB API returns data in the following format:

```json
[
  {"data": "01/01/2024", "valor": "1.001"},
  {"data": "02/01/2024", "valor": "1.002"},
  ...
]
```

#### Output Data (Processed DataFrame)

The processed DataFrame has the following structure:

```python
# Index: datetime
# Columns: factor (float)
2024-01-01    1.001
2024-01-02    1.002
2024-01-03    1.003
...
```

#### Risk-Free Curve (Series)

The risk-free curve is a pandas Series with:

```python
# Index: datetime
# Values: annualized risk-free rates (float)
2024-01-01    0.0250
2024-01-02    0.0251
2024-01-03    0.0252
...
```

### Business Day Consideration

The module uses the B3 trading calendar to ensure that:

1. **Data Alignment**: All data is aligned to actual B3 business days
2. **Forward-Filling**: Missing trading days are filled with the last available value
3. **Accurate Annualization**: Risk-free rate calculations use 252 trading days per year

### Data Validation

The module performs several validation steps in the correct order for financial time series:

1. **Chronological Ordering**: Data is sorted by date
2. **NaN Interpolation**: Missing values are interpolated using time-based interpolation (configurable)
3. **Outlier Removal**: Factors outside the configured bounds (default: 0.95-1.05) are removed
4. **Business Day Alignment**: Data is reindexed to B3 business days

**Key Validation Strategy**: NaN values are interpolated BEFORE outlier filtering to preserve time series continuity and maximize data retention, following quantitative finance best practices.

### Caching

Processed data is cached in Parquet format for performance:

- **Location**: `data/daily_factors/daily_factors.parquet`
- **Format**: Parquet (compressed, fast I/O)
- **Automatic**: Caching happens automatically after data processing
- **Manual**: Use CLI commands to manage cache

### Error Handling

The module includes comprehensive error handling:

- **API Errors**: Network timeouts, HTTP errors, and malformed responses
- **Data Errors**: Invalid data formats, missing required fields
- **Configuration Errors**: Missing or malformed configuration files
- **File System Errors**: Permission issues, disk space problems

All errors are logged with detailed information for debugging.

### Testing

Run the test suite:

```bash
cd quant_backtest
python -m pytest tests/test_bcb_daily_factor.py -v
```

The test suite covers:

- API integration (with mocking)
- Data validation and transformation
- Business day alignment
- Risk-free curve calculation
- Caching functionality
- Error handling

### Performance Considerations

- **Caching**: Use cached data when possible to avoid API calls
- **Date Ranges**: Request only necessary date ranges to minimize data transfer
- **Batch Processing**: For large date ranges, consider processing in batches
- **Memory Usage**: Large datasets are processed efficiently with pandas

### Troubleshooting

#### Common Issues

1. **API Timeout**: Increase timeout in requests or check network connectivity
2. **Invalid Data**: Check BCB API status and data format
3. **Cache Issues**: Clear cache and re-fetch data
4. **Configuration Errors**: Verify `settings.yaml` format and paths

#### Logging

Enable debug logging for detailed information:

```python
import logging
logging.getLogger('market_data.bcb_daily_factor').setLevel(logging.DEBUG)
```

### Contributing

When contributing to this module:

1. Follow the existing code style and documentation standards
2. Add comprehensive tests for new functionality
3. Update this README for any new features
4. Ensure all tests pass before submitting changes

### License

This module is part of the quant_b3_backtest project and follows the same license terms. 