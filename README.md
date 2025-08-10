# Brazilian Stock Market Backtesting Engine

A comprehensive backtesting engine for Brazilian stocks (B3) with support for real-time data fetching, portfolio management, advanced transaction cost analysis, and strategy testing.

## Features

- **Data Management**: Download and process Brazilian stock data using Alpha Vantage API
- **Portfolio Management**: Automated Brazilian market fees, taxes, and exemptions
- **Transaction Cost Analysis (TCA)**: Accurate, modular calculation of all trading costs (brokerage, emolument, settlement, ISS)
- **Advanced Loss Carryforward**: Full compliance with Brazilian tax law, including indefinite and per-asset loss tracking
- **Advanced Settlement Manager**: T+2 settlement queue, business day handling, and cash flow simulation
- **Backtest Simulator**: Comprehensive backtesting with performance metrics and compliance (BacktestSimulator)
- **Strategy Testing**: Framework for implementing and testing trading strategies
- **HTML Reporting**: Interactive Plotly dashboards with PDF export capabilities
- **Configuration**: Flexible settings for market hours, fees, portfolio, compliance, and performance
- **Security**: Secure API key management for safe GitHub uploads
- **Comprehensive Testing**: Extensive test suites for all advanced modules

## Brazilian Market Conventions

The backtesting engine enforces Brazilian market (B3) conventions to ensure realistic simulation:

### Price Tick Normalization
- **Tick Size**: R$ 0.01 (minimum price increment)
- **Normalization**: All prices are automatically rounded to the nearest centavo
- **Example**: R$ 12.3456 → R$ 12.35, R$ 12.344 → R$ 12.34

### Lot Size Validation
- **Round Lots**: Multiples of 100 shares (100, 200, 300, etc.)
- **Odd Lots**: Non-multiples of 100 (50, 150, 250, etc.)
- **Order Routing**: Round lots go to main book, odd lots to fractional book
- **Configuration**: Can disable fractional lots via `allow_fractional_lots: false`

### Order Validation
- **Automatic Validation**: All orders are validated against market conventions
- **Price Normalization**: Prices are automatically normalized to valid ticks
- **Lot Classification**: Orders are classified as round lot or odd lot
- **Trade History**: Original and normalized values are tracked for audit

### Configuration
```yaml
market:
  tick_size: 0.01              # Price tick size
  round_lot_size: 100          # Standard lot size
  min_quantity: 1              # Minimum order quantity
  allow_fractional_lots: true  # Allow odd lot orders
  enforce_price_ticks: true    # Enforce price normalization
  enforce_lot_sizes: true      # Enforce lot size validation
```

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
│   ├── base_strategy.py  # Base strategy with Brazilian market utilities
│   ├── tca.py                 # Transaction Cost Analysis (TCA) module
│   ├── loss_manager.py        # Enhanced Loss Carryforward Manager
│   ├── settlement_manager.py  # Advanced Settlement Manager (T+2)
│   ├── base_strategy.py       # Abstract base class for strategies
│   └── simulator.py           # Backtest simulator
├── scripts/
│   └── download_data.py       # Data downloader
├── strategies/                # User trading strategies (currently empty)
├── reports/                   # Backtest reports (NOT shared)
├── tests/                     # Comprehensive test suites
│   ├── test_market_utils.py   # Market utilities tests
│   ├── test_market_integration.py # Integration tests
│   ├── test_tca.py            # Transaction Cost Analysis tests
│   ├── test_enhanced_managers.py # Loss carryforward and settlement manager tests
│   └── test_simulator.py      # Backtest simulator tests
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Advanced Features

- **Brazilian Market Conventions**: Enforces B3 market rules including price tick normalization (R$ 0.01), lot size validation (round lots = multiples of 100), and order routing to main/fractional books.
- **Transaction Cost Analysis (TCA)**: Modular, accurate calculation of all trading costs, including brokerage, emolument, settlement, and ISS. Fully configurable and tested.
- **Enhanced Loss Carryforward Manager**: Tracks losses per asset and globally, supports indefinite carryforward, and provides audit trails for compliance.
- **Advanced Settlement Manager**: Models T+2 settlement with business day handling, cash flow simulation, and robust error handling.
- **Backtest Simulator**: Comprehensive backtesting with performance metrics and compliance (BacktestSimulator in `engine/simulator.py`).
- **Extensible Strategy Framework**: Implement your own trading strategies by subclassing the `BaseStrategy` class in `engine/base_strategy.py` and placing your strategy files in the `strategies/` directory.
- **Comprehensive Testing**: Extensive unit and integration tests for TCA, loss carryforward, settlement logic, and simulation.

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

## Usage Examples

### Market Utilities Usage

```python
from engine.base_strategy import BrazilianMarketUtils

# Initialize market utilities
utils = BrazilianMarketUtils()

# Price normalization
normalized_price = utils.normalize_price_tick(12.3456)  # Returns 12.35

# Lot size validation
is_valid, lot_type, is_fractional = utils.validate_lot_size(150)
# Returns: (True, LotType.ODD_LOT, True)

# Order validation
validation = utils.validate_order(
    price=12.3456,
    quantity=150,
    allow_fractional=True
)
# Returns OrderValidation with normalized values and lot classification
```

### Portfolio with Market Conventions

```python
from engine.portfolio import EnhancedPortfolio

# Initialize portfolio (automatically uses market conventions)
portfolio = EnhancedPortfolio("config/settings.yaml")

# Buy with price normalization
success = portfolio.buy(
    ticker="PETR4",
    quantity=100,
    price=12.3456,  # Automatically normalized to 12.35
    trade_date=datetime.now(),
    trade_type="swing_trade"
)

# Trade history includes original and normalized values
trade = portfolio.trade_history[0]
print(f"Original price: {trade['original_price']}")  # 12.3456
print(f"Normalized price: {trade['price']}")         # 12.35
print(f"Lot type: {trade['lot_type']}")              # round_lot
```

### Strategy with Market Constraints

```python
from engine.base_strategy import BaseStrategy, TradingSignal, SignalType

class MyStrategy(BaseStrategy):
    def generate_signals(self, market_data):
        # Your signal generation logic
        signal = TradingSignal(
            signal_type=SignalType.BUY,
            ticker="PETR4",
            price=12.3456,  # Will be normalized automatically
            quantity=150,    # Will be classified as odd lot
            confidence=1.0,
            trade_type=TradeType.SWING_TRADE
        )
        return [signal]
    
    def execute_trade(self, signal):
        # Market constraints are automatically validated
        constraints_ok = self.check_brazilian_market_constraints(signal)
        if constraints_ok:
            # Signal price and quantity are automatically normalized
            return self.portfolio.buy(
                ticker=signal.ticker,
                quantity=signal.quantity,  # Already normalized
                price=signal.price,        # Already normalized
                trade_date=signal.timestamp,
                trade_type=signal.trade_type.value
            )
        return False
```
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

### Implement Your Own Strategies

To create a custom trading strategy, subclass the `BaseStrategy` class from `engine/base_strategy.py` and place your strategy file in the `strategies/` directory. The `strategies/` directory is currently empty and intended for user strategies.

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

Run the comprehensive test suites for TCA, loss carryforward, settlement, and simulation:

```bash
python -m unittest discover tests
```

- `tests/test_tca.py`: Transaction Cost Analysis tests
- `tests/test_enhanced_managers.py`: Loss carryforward and settlement manager tests
- `tests/test_simulator.py`: Backtest simulator tests

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
