# Brazilian Stock Market Backtesting Engine

A comprehensive backtesting engine for Brazilian stocks (B3) with support for real-time data fetching, portfolio management, advanced transaction cost analysis, and strategy testing.

## Features

- **Data Management**: Download and process Brazilian stock data using Alpha Vantage API
- **Portfolio Management**: Automated Brazilian market fees, taxes, and exemptions
- **Transaction Cost Analysis (TCA)**: Accurate, modular calculation of all trading costs (brokerage, emolument, settlement, ISS)
- **Advanced Loss Carryforward**: Full compliance with Brazilian tax law, including indefinite and per-asset loss tracking
- **Advanced Settlement Manager**: T+2 settlement queue, business day handling, and cash flow simulation
- **Strategy Testing**: Framework for implementing and testing trading strategies
- **Configuration**: Flexible settings for market hours, fees, portfolio, compliance, and performance
- **Security**: Secure API key management for safe GitHub uploads
- **Comprehensive Testing**: Extensive test suites for all advanced modules

## Project Structure

```
quant_backtest/
├── config/
│   ├── settings.yaml          # Market configuration (safe to share)
│   ├── secrets.yaml.example   # Template for API keys (safe to share)
│   └── secrets.yaml           # Your actual API keys (NOT shared)
├── data/
│   ├── raw/                   # Downloaded raw data (NOT shared)
│   └── processed/             # Processed data (NOT shared)
├── engine/
│   ├── loader.py              # Data loading utilities
│   ├── portfolio.py           # Portfolio management (uses advanced managers)
│   ├── tca.py                 # Transaction Cost Analysis (TCA) module
│   ├── loss_manager.py        # Enhanced Loss Carryforward Manager
│   └── settlement_manager.py  # Advanced Settlement Manager (T+2)
├── scripts/
│   └── download_data.py       # Data downloader
├── strategies/                # Trading strategies
├── reports/                   # Backtest reports (NOT shared)
├── tests/                     # Comprehensive test suites
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Advanced Features

- **Transaction Cost Analysis (TCA)**: Modular, accurate calculation of all trading costs, including brokerage, emolument, settlement, and ISS. Fully configurable and tested.
- **Enhanced Loss Carryforward Manager**: Tracks losses per asset and globally, supports indefinite carryforward, and provides audit trails for compliance.
- **Advanced Settlement Manager**: Models T+2 settlement with business day handling, cash flow simulation, and robust error handling.
- **Comprehensive Testing**: Extensive unit and integration tests for TCA, loss carryforward, and settlement logic.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd quant_backtest
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

**IMPORTANT**: Never commit your actual API keys to GitHub!

1. Copy the example secrets file:
   ```bash
   cp config/secrets.yaml.example config/secrets.yaml
   ```

2. Edit `config/secrets.yaml` and add your Alpha Vantage API key:
   ```yaml
   alpha_vantage:
     api_key: "YOUR_ACTUAL_API_KEY_HERE"
     base_url: "https://www.alphavantage.co/query"
   ```

### 5. Get Alpha Vantage API Key

1. Visit [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Sign up for a free account
3. Get your API key
4. Add it to `config/secrets.yaml`

## Usage

### Download Stock Data

```bash
python scripts/download_data.py
```

This will:
- Download data for test symbols (VALE3, PETR4, ITUB4)
- Save data to `data/raw/` directory
- Create metadata files for each symbol

### Run Backtests

```bash
# Example: Run a simple backtest
python -c "
from engine.portfolio import EnhancedPortfolio
portfolio = EnhancedPortfolio('config/settings.yaml')
# Add your backtest logic here
"
```

## Security Features

### What's Protected

The following files and directories are automatically excluded from Git:

- **API Keys**: `config/secrets.yaml` (contains your actual keys)
- **Data Files**: All CSV, JSON, and data files
- **Logs**: All log files
- **Virtual Environment**: `venv/` directory
- **Reports**: Generated reports and outputs

### What's Safe to Share

- **Code**: All Python scripts and modules
- **Configuration**: `config/settings.yaml` (market settings only)
- **Examples**: `config/secrets.yaml.example` (template without real keys)
- **Documentation**: README and other documentation files

## Configuration

### Market & Advanced Settings (`config/settings.yaml`)

Configure Brazilian market parameters and advanced features:

```yaml
market:
  trading_hours:
    open: "10:00"
    close: "16:55"
    timezone: "America/Sao_Paulo"
  costs:
    emolument: 0.00005                 # B3 negotiation fee
    settlement_day_trade: 0.00018      # Day-trade settlement
    settlement_swing_trade: 0.00025    # Swing-trade settlement
    brokerage_fee: 0.0                 # Modal brokerage (zero)
    min_brokerage: 0.0                 # Minimum brokerage
    iss_rate: 0.05                     # ISS tax rate

taxes:
  swing_trade: 0.15        # 15% capital gains
  day_trade: 0.20          # 20% capital gains
  exemption_limit: 20000   # Monthly tax-free limit
  irrf_swing_rate: 0.00005 # IRRF withholding (swing)
  irrf_day_rate: 0.01      # IRRF withholding (day trade)

portfolio:
  initial_cash: 100000     # Starting capital
  max_positions: 10
  position_sizing: "equal_weight"

settlement:
  cycle_days: 2                    # T+2 settlement cycle
  timezone: "America/Sao_Paulo"    # Market timezone
  strict_mode: true                # Enforce settlement rules
  holiday_calendar: "b3"           # Use B3 holiday calendar

loss_carryforward:
  asset_specific_tracking: true    # Enable per-asset loss tracking
  audit_trail_enabled: true        # Enable audit trail
```

## Advanced Testing

Run the comprehensive test suites for TCA, loss carryforward, and settlement:

```bash
python -m unittest discover tests
```

- `tests/test_tca.py`: Transaction Cost Analysis tests
- `tests/test_enhanced_managers.py`: Loss carryforward and settlement manager tests

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational purposes. Please ensure compliance with Alpha Vantage's terms of service and Brazilian financial regulations.

## Support

For issues and questions:
1. Check the documentation
2. Review the code comments
3. Create an issue on GitHub

## Disclaimer

This software is for educational and research purposes only. It is not financial advice. Always consult with qualified financial professionals before making investment decisions.
