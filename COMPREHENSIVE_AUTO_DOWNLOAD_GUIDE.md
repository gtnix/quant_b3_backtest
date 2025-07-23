# Comprehensive Auto-Download Guide

## Overview

The backtest system now includes comprehensive automatic data downloading functionality that covers all required data sources:

- **📊 Ticker Data**: Individual stock data from Alpha Vantage API
- **🏦 SGS Data**: Brazilian Central Bank data (Selic, CDI, IPCA)
- **📈 IBOV Data**: IBOVESPA index data from Yahoo Finance

## Key Features

### 1. Automatic Data Detection
- Checks for missing ticker files
- Detects recent data gaps (missing business days)
- Validates SGS series availability
- Monitors IBOV data freshness

### 2. Smart Download Logic
- Only downloads missing data (not entire datasets)
- Uses Brazilian business days (`dias_uteis`) for gap detection
- Respects API rate limits
- Handles holidays and weekends properly

### 3. Comprehensive Status Reporting
- Detailed breakdown of missing data by source
- Time estimates for downloads
- Success/failure tracking
- Progress indicators

## Usage

### Basic Usage

```bash
# Check data status for specific tickers
python scripts/test_comprehensive_data.py --tickers PETR4,VALE3,ITUB4

# Check and download missing data
python scripts/test_comprehensive_data.py --tickers PETR4,VALE3,ITUB4 --download

# Check all IBRA tickers
python scripts/test_comprehensive_data.py --tickers all --download
```

### Backtest Integration

```bash
# Run backtest with automatic data download
python scripts/run_backtest.py --strategy momentum --tickers PETR4,VALE3,ITUB4

# Run backtest without downloading (check only)
python scripts/run_backtest.py --strategy momentum --tickers PETR4,VALE3,ITUB4 --no-download
```

## Data Sources

### 1. Ticker Data (Alpha Vantage)
- **Source**: Alpha Vantage API
- **Rate Limit**: 5 calls/minute (free tier)
- **Data Type**: Adjusted prices (includes corporate actions)
- **Storage**: `data/raw/{TICKER}_raw.csv`
- **Gap Detection**: Uses `dias_uteis` for Brazilian business days

### 2. SGS Data (Banco Central)
- **Source**: Banco Central SGS API
- **Series**: 
  - 11: Selic Interest Rate
  - 12: CDI Interest Rate  
  - 433: IPCA Inflation Index
- **Processing**: LOCF normalization for missing dates
- **Storage**: `data/sgs/sgs_{SERIES_ID}_*.csv`

### 3. IBOV Data (Yahoo Finance)
- **Source**: Yahoo Finance API
- **Symbol**: ^BVSP (IBOVESPA)
- **Advantage**: More recent data than Alpha Vantage
- **Storage**: `data/IBOV/raw/IBOV_*.csv`

## Configuration

### API Keys
Create `config/secrets.yaml` with your API keys:

```yaml
alpha_vantage:
  api_key: YOUR_ALPHA_VANTAGE_API_KEY
  base_url: https://www.alphavantage.co/query
  data_type: adjusted
  rate_limit_calls_per_minute: 5
  max_retries: 3
  timeout_seconds: 30
```

### SGS Configuration
SGS data uses the existing `config/settings.yaml` configuration.

## Implementation Details

### DataLoader Class Enhancements

The `DataLoader` class now includes:

```python
# Check all data sources
data_status = data_loader.check_all_data(tickers)

# Download all missing data
download_results = data_loader.download_all_missing_data(tickers)

# Individual source checking
sgs_status = data_loader.check_sgs_data()
ibov_status = data_loader.check_ibov_data()
ticker_status = data_loader.check_missing_data(tickers)
```

### Gap Detection Logic

```python
# Uses dias_uteis for Brazilian business days
from dias_uteis import range_du

start_date_du = (last_available_date + 1).date()
end_date_du = today.date()
business_days = range_du(start_date_du, end_date_du)
missing_business_days = len(business_days)
```

### Download Methods

```python
# Ticker downloads
result = stock_downloader.download_ticker_data(ticker)

# SGS downloads  
result = sgs_downloader.get_all_series_data(start_date, end_date)

# IBOV downloads
result = ibov_downloader.get_recent_data(days=30)
```

## Status Reporting

### Comprehensive Status Output

```
============================================================
COMPREHENSIVE DATA STATUS
============================================================

📊 TICKER DATA:
   Missing tickers: 0
   Tickers with gaps: 2
   Total missing days: 20
   Tickers with recent gaps:
     PETR4: 9 days since 2025-07-09T00:00:00
     VALE3: 11 days since 2025-07-07T00:00:00

🏦 SGS DATA:
   Has data: False
   Needs download: True
   Missing series: [8, 11, 12, 433]

📈 IBOV DATA:
   Has data: False
   Needs download: True

📋 SUMMARY:
   Any missing data: True
   Total missing tickers: 0
   Total tickers with gaps: 2
   Total missing days: 20
   SGS needs download: True
   IBOV needs download: True
============================================================
```

### Download Results

```
============================================================
DOWNLOAD RESULTS
============================================================

📊 TICKER DOWNLOADS:
   Successful: 2
   Failed: 0
   Successful tickers: PETR4, VALE3

🏦 SGS DOWNLOAD:
   Success: True
   Message: SGS data downloaded successfully

📈 IBOV DOWNLOAD:
   Success: True
   Message: IBOV data downloaded successfully
============================================================
```

## Error Handling

### Graceful Degradation
- If downloaders fail to initialize, auto-download is disabled
- Missing data warnings are shown but don't stop execution
- Partial downloads are reported with success/failure counts

### Rate Limiting
- Alpha Vantage: 5 calls/minute (configurable)
- SGS API: Built-in rate limiting
- Yahoo Finance: No rate limits

### Retry Logic
- Exponential backoff for failed requests
- Configurable retry attempts
- Timeout handling

## Performance Considerations

### Download Time Estimates
- Ticker downloads: ~2 seconds per ticker
- SGS downloads: ~30 seconds total
- IBOV downloads: ~10 seconds

### Storage Optimization
- Only downloads missing data, not entire datasets
- Efficient gap detection using business day calendars
- Compressed storage formats (Parquet) for IBOV data

## Best Practices

### 1. Regular Data Updates
```bash
# Check data freshness weekly
python scripts/test_comprehensive_data.py --tickers all
```

### 2. Batch Processing
```bash
# Download data for multiple tickers efficiently
python scripts/test_comprehensive_data.py --tickers PETR4,VALE3,ITUB4,BBDC4 --download
```

### 3. Monitoring
- Check logs in `logs/` directory
- Monitor API usage for rate limits
- Verify data quality after downloads

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure `config/secrets.yaml` exists with valid API key
   - Check API key permissions and rate limits

2. **Import Errors**
   - Verify all required packages are installed
   - Check Python path configuration

3. **Rate Limiting**
   - Reduce concurrent downloads
   - Increase delays between requests
   - Consider upgrading to paid API tier

4. **Data Quality Issues**
   - Check for missing dates in downloaded data
   - Verify business day calculations
   - Review log files for errors

### Debug Mode
```bash
# Enable verbose logging
python scripts/test_comprehensive_data.py --tickers PETR4 --download --verbose
```

## Future Enhancements

### Planned Features
- Parallel downloads for faster processing
- Data validation and quality checks
- Automatic data cleanup and maintenance
- Integration with additional data sources

### Configuration Options
- Customizable download schedules
- Advanced rate limiting controls
- Data retention policies
- Quality threshold settings

## Conclusion

The comprehensive auto-download functionality ensures that your backtest system always has the most up-to-date data from all required sources. The system is designed to be efficient, reliable, and user-friendly, with detailed reporting and graceful error handling.

By following this guide, you can maintain a robust data pipeline for your Brazilian stock market backtesting strategies. 