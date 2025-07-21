# Quant B3 Backtest System - Complete Documentation

## Overview

The Quant B3 Backtest System is a comprehensive backtesting engine designed specifically for the Brazilian stock market (B3). It provides advanced features for data management, portfolio simulation, transaction cost analysis, tax compliance, and performance evaluation with full Brazilian market regulations compliance.

## System Architecture

The system follows a modular architecture with the following key components:

```
quant_backtest/
├── engine/           # Core backtesting engine modules
├── scripts/          # Data download and execution scripts
├── config/           # Configuration files
├── strategies/       # Trading strategy implementations
├── data/            # Data storage (raw and processed)
├── reports/         # Generated reports
└── tests/           # Test suites
```

## Core Modules Documentation

### Engine Modules (Core Backtesting Engine)

1. **[Data Loader](engine/loader.md)** - Data loading, preprocessing, and quality management
2. **[Portfolio Manager](engine/portfolio.md)** - Advanced portfolio management with Brazilian compliance
3. **[Transaction Cost Analyzer](engine/tca.md)** - Comprehensive transaction cost calculation
4. **[Loss Carryforward Manager](engine/loss_manager.md)** - Brazilian tax loss tracking and management
5. **[Settlement Manager](engine/settlement_manager.md)** - T+2 settlement tracking and cash flow management
6. **[Base Strategy](engine/base_strategy.md)** - Abstract base class for trading strategies
7. **[Backtest Simulator](engine/simulator.md)** - Main simulation engine
8. **[Performance Metrics](engine/performance_metrics.md)** - Comprehensive performance analysis
9. **[SGS Data Loader](engine/sgs_data_loader.md)** - Banco Central data integration

### Scripts (Data Management & Execution)

1. **[Data Downloader](scripts/download_data.md)** - Alpha Vantage API stock data download
2. **[Backtest Runner](scripts/run_backtest.md)** - CLI for running backtests
3. **[SGS Data Downloader](scripts/download_sgs_data.md)** - Banco Central data download
4. **[IBRA Tickers](scripts/ibra_tickers.md)** - Brazilian stock ticker management

### Configuration

1. **[Settings Configuration](config/settings.md)** - Market parameters and system configuration
2. **[Secrets Management](config/secrets.md)** - API keys and sensitive data management

## Key Features

### Brazilian Market Compliance
- **T+2 Settlement**: Accurate settlement date calculation using Brazilian business days
- **Tax Compliance**: Full compliance with Brazilian individual taxpayer rules (2025)
- **Transaction Costs**: Complete B3 fee structure including emolument, settlement, and ISS
- **Loss Carryforward**: Per-asset and global loss tracking with 100% offset capability

### Advanced Portfolio Management
- **Enhanced Portfolio**: Comprehensive position tracking with unrealized P&L
- **Risk Management**: Built-in risk metrics and position sizing
- **Cash Flow Simulation**: Realistic cash flow modeling with settlement delays

### Data Management
- **Alpha Vantage Integration**: Automated stock data download with rate limiting
- **Yahoo Finance Integration**: IBOV index data download (free, no rate limits)
- **Data Quality**: Comprehensive data validation and filtering
- **SGS Integration**: Banco Central interest rate and inflation data
- **Corporate Actions**: Automatic handling of dividends and splits

### Performance Analysis
- **Comprehensive Metrics**: Sharpe ratio, drawdown, win rate, and more
- **Tax-Aware Returns**: After-tax performance calculation
- **Benchmark Analysis**: IBOV comparison and relative performance
- **Risk-Adjusted Metrics**: Advanced risk metrics with Brazilian market parameters

## Getting Started

### Prerequisites
- Python 3.8+
- Alpha Vantage API key (for stock data)
- Required Python packages (see `requirements.txt`)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd quant_backtest

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp config/secrets.yaml.example config/secrets.yaml
# Edit config/secrets.yaml with your Alpha Vantage API key (for stock data)
```

### Basic Usage
```bash
# Download data
python scripts/download_data.py

# Run a backtest
python scripts/run_backtest.py --strategy your_strategy --tickers PETR4,VALE3 --start-date 2023-01-01 --end-date 2023-12-31
```

## Configuration Guide

The system uses YAML configuration files for all parameters:

- **`config/settings.yaml`**: Market parameters, tax rates, portfolio settings
- **`config/secrets.yaml`**: API keys and sensitive data (not committed to Git)

## Testing

The system includes comprehensive test suites:

```bash
# Run all tests
python -m unittest discover tests

# Run specific test modules
python -m unittest tests.test_tca
python -m unittest tests.test_enhanced_managers
python -m unittest tests.test_simulator
```

## Security Features

- **API Key Protection**: Secrets file excluded from Git
- **Data Privacy**: Raw data files excluded from version control
- **Configuration Validation**: Comprehensive input validation
- **Error Handling**: Robust error handling and logging

## Compliance & Regulations

The system is designed for compliance with:
- **Brazilian Tax Law**: IN RFB 1.585/2015 compliance
- **B3 Trading Rules**: Accurate fee structure and settlement cycles
- **CVM Regulations**: Brazilian Securities Commission compliance
- **Individual Taxpayer Rules**: 2025 Brazilian tax regulations

## Support & Contributing

For issues and questions:
1. Check the documentation
2. Review the code comments
3. Create an issue on GitHub

## License

This project is for educational and research purposes only. It is not financial advice. Always consult with qualified financial professionals before making investment decisions.

## Disclaimer

This software is for educational and research purposes only. It is not financial advice. Always consult with qualified financial professionals before making investment decisions. 