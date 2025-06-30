# Performance Metrics Module for Brazilian B3 Quant Backtest System

## 📊 Overview

The Performance Metrics Module provides comprehensive, tax-aware performance analysis for Brazilian stock market backtesting with full regulatory compliance. This module integrates seamlessly with the existing `quant_b3_backtest` system to deliver professional-grade performance evaluation.

## 🎯 Key Features

### Core Performance Metrics
- **Returns Calculation**: Total, annualized, and logarithmic returns using 252 trading days (Brazilian market standard)
- **Risk-Adjusted Metrics**: Sharpe, Sortino, Calmar ratios with Brazilian SELIC rate integration
- **Tax-Aware Analysis**: Brazilian tax rules (2025) with loss carryforward integration
- **Trade Analysis**: Win rate, profit factor, average win/loss calculations
- **Regulatory Compliance**: CVM and Receita Federal compliance reporting

### Brazilian Market Compliance (2025)
- ✅ **252 Trading Days**: Brazilian market standard calculation (configurable)
- ✅ **SELIC Rate**: 15.0% risk-free rate integration (configurable in settings.yaml)
- ✅ **Tax Rules**: 15% swing trade, 20% day trade rates (configurable)
- ✅ **Loss Carryforward**: 30% maximum offset, perpetual carryforward
- ✅ **R$20,000 Exemption**: Monthly swing trade exemption (configurable)
- ✅ **Regulatory Compliance**: Brazilian Capital Markets 2025

## 🏗️ Architecture

### Module Structure
```
engine/performance_metrics.py
├── PerformanceMetrics (Base class)
├── RiskAdjustedMetrics (Risk analysis)
├── TaxAwareMetrics (Brazilian tax compliance)
└── ComprehensivePerformanceAnalysis (Complete analysis)
```

### Integration Points
- **EnhancedPortfolio**: Portfolio state and trade history
- **EnhancedLossCarryforwardManager**: Loss tracking and offset
- **TransactionCostAnalyzer**: Cost calculations
- **Configuration**: `settings.yaml` for market parameters

## 📋 Installation & Setup

### Prerequisites
```bash
# Ensure you have the required dependencies
pip install -r requirements.txt
```

### Dependencies
The module requires the following packages (already included in `requirements.txt`):
- `numpy>=1.24.0`
- `pandas>=2.1.0`
- `matplotlib>=3.7.0`
- `pyyaml>=6.0`
- `pytz` (for timezone handling)

## 🚀 Quick Start

### Basic Usage
```python
from engine.performance_metrics import PerformanceMetrics, ComprehensivePerformanceAnalysis
from engine.portfolio import EnhancedPortfolio

# Initialize portfolio
portfolio = EnhancedPortfolio("config/settings.yaml")

# Create performance metrics
performance_metrics = PerformanceMetrics(portfolio)

# Calculate returns
portfolio_values = [100000, 101000, 102500, 103200, 104000]
daily_returns = [0.01, 0.015, 0.007, 0.008]
returns_metrics = performance_metrics.calculate_returns(
    portfolio_values, datetime.now(), datetime.now()
)

print(f"Total Return: {returns_metrics.total_return:.4f}")
print(f"Annualized Return: {returns_metrics.annualized_return:.4f}")
```

### Comprehensive Analysis
```python
# Run complete analysis
analysis = ComprehensivePerformanceAnalysis(portfolio)
results = analysis.run_comprehensive_analysis(portfolio_values, daily_returns)

# Generate reports
analysis.generate_performance_report(results, "reports/performance_report.txt")
analysis.plot_performance_charts(portfolio_values, daily_returns, "reports/charts.png")
```

## 📊 Available Metrics

### Returns Metrics (`ReturnsMetrics`)
```python
@dataclass
class ReturnsMetrics:
    total_return: float          # Total return over period
    annualized_return: float     # Annualized return (252 days)
    logarithmic_return: float    # Logarithmic return
    daily_returns: List[float]   # Daily return series
    cumulative_returns: List[float]  # Cumulative return series
    trading_days: int            # Number of trading days
```

### Risk Metrics (`RiskMetrics`)
```python
@dataclass
class RiskMetrics:
    sharpe_ratio: float          # Sharpe ratio (excess return / volatility)
    sortino_ratio: float         # Sortino ratio (excess return / downside deviation)
    calmar_ratio: float          # Calmar ratio (annualized return / max drawdown)
    max_drawdown: float          # Maximum drawdown
    volatility: float            # Annualized volatility
    var_95: float               # Value at Risk (95% confidence)
    cvar_95: float              # Conditional Value at Risk (95% confidence)
```

### Tax Metrics (`TaxMetrics`)
```python
@dataclass
class TaxMetrics:
    total_taxes_paid: float      # Total taxes paid
    swing_trade_taxes: float     # Swing trade taxes (15%)
    day_trade_taxes: float       # Day trade taxes (20%)
    tax_efficiency: float        # After-tax / pre-tax return ratio
    loss_carryforward_utilized: float  # Loss carryforward used
    effective_tax_rate: float    # Effective tax rate
    tax_exemption_utilized: float # R$20,000 exemption used
```

### Trade Metrics (`TradeMetrics`)
```python
@dataclass
class TradeMetrics:
    total_trades: int            # Total number of trades
    winning_trades: int          # Number of winning trades
    losing_trades: int           # Number of losing trades
    win_rate: float             # Win rate percentage
    profit_factor: float        # Profit factor (gross profit / gross loss)
    average_win: float          # Average winning trade
    average_loss: float         # Average losing trade
    total_commission: float     # Total commission paid
```

## 🔧 Advanced Usage

### Risk-Adjusted Metrics
```python
from engine.performance_metrics import RiskAdjustedMetrics

# Calculate advanced risk metrics
risk_metrics = RiskAdjustedMetrics(performance_metrics)
all_risk_metrics = risk_metrics.calculate_all_risk_metrics(daily_returns)

# Available metrics:
# - sharpe_ratio, sortino_ratio, calmar_ratio
# - information_ratio, treynor_ratio, jensen_alpha
# - max_drawdown, volatility, var_95, cvar_95
# - skewness, kurtosis, ulcer_index, gain_to_pain_ratio
```

### Tax-Aware Analysis
```python
from engine.performance_metrics import TaxAwareMetrics

# Calculate tax-aware returns
tax_metrics = TaxAwareMetrics(performance_metrics)
tax_aware_returns = tax_metrics.calculate_tax_aware_returns(portfolio_values)

# Get regulatory compliance
compliance_metrics = tax_metrics.calculate_regulatory_compliance_metrics()
```

### Custom Configuration
```yaml
# config/settings.yaml
market:
  trading_hours:
    timezone: "America/Sao_Paulo"
  selic_rate: 0.15         # Current Brazilian SELIC rate (15.0% - 2025)
  trading_days_per_year: 252  # Brazilian market standard

taxes:
  swing_trade: 0.15        # 15% swing trade tax
  day_trade: 0.20          # 20% day trade tax
  exemption_limit: 20000   # R$20,000 monthly exemption
  irrf_swing_rate: 0.00005 # IRRF withholding (swing)
  irrf_day_rate: 0.01      # IRRF withholding (day trade)
```

## 📈 Performance Visualization

### Generate Charts
```python
# Create performance charts
analysis.plot_performance_charts(
    portfolio_values, 
    daily_returns, 
    "reports/performance_charts.png"
)
```

### Generate Reports
```python
# Create comprehensive report
analysis.generate_performance_report(
    results, 
    "reports/performance_report.txt"
)
```

## 🧪 Testing

### Run Tests
```bash
# Run all performance metrics tests
python -m pytest tests/test_performance_metrics.py -v

# Run specific test class
python -m pytest tests/test_performance_metrics.py::TestPerformanceMetrics -v

# Run with coverage
python -m pytest tests/test_performance_metrics.py --cov=engine.performance_metrics
```

### Demo Script
```bash
# Run demonstration
python scripts/demo_performance_metrics.py
```

## 🔍 Integration Examples

### With Backtest Simulator
```python
from engine.simulator import BacktestSimulator
from engine.performance_metrics import ComprehensivePerformanceAnalysis

# Run backtest
simulator = BacktestSimulator(strategy, initial_capital=100000)
simulation_result = simulator.run_simulation(data)

# Analyze performance
analysis = ComprehensivePerformanceAnalysis(simulator.portfolio)
results = analysis.run_comprehensive_analysis(
    simulation_result.portfolio_values,
    simulation_result.daily_returns
)
```

### With Portfolio Manager
```python
from engine.portfolio import EnhancedPortfolio
from engine.performance_metrics import PerformanceMetrics

# Track portfolio performance
portfolio = EnhancedPortfolio("config/settings.yaml")

# After running trades, analyze performance
performance_metrics = PerformanceMetrics(portfolio)
returns = performance_metrics.calculate_returns(portfolio_values, start_date, end_date)
risk = performance_metrics.calculate_risk_metrics(daily_returns)
taxes = performance_metrics.calculate_tax_metrics()
trades = performance_metrics.calculate_trade_metrics()
```

## 📋 Brazilian Regulatory Compliance

### Tax Rules (2025)
- **Swing Trade Tax**: 15% on profits
- **Day Trade Tax**: 20% on profits
- **Monthly Exemption**: R$20,000 for swing trades
- **Loss Carryforward**: Maximum 30% offset against capital gains
- **Perpetual Carryforward**: No time limit for loss carryforward

### Regulatory Framework
- **Brazilian Securities Commission (CVM)**: Current 2025 regulations
- **Brazilian Tax Authority (Receita Federal)**: Current 2025 regulations
- **PwC Tax Summaries 2025**: Confirmed regulatory requirements

### Compliance Features
- ✅ Automatic tax calculation with Brazilian rates
- ✅ Loss carryforward integration with 30% limit
- ✅ Regulatory compliance reporting
- ✅ Audit trail generation
- ✅ CVM and Receita Federal compliance validation

## 🚨 Error Handling

### Common Issues
```python
# Configuration file not found
try:
    metrics = PerformanceMetrics(portfolio, "config/settings.yaml")
except FileNotFoundError:
    print("Configuration file not found")

# Insufficient data
if len(portfolio_values) < 2:
    print("Insufficient portfolio values for analysis")

# Invalid trade data
if not portfolio.trade_history:
    print("No trade history available")
```

### Validation
The module includes comprehensive input validation:
- Portfolio values must be positive
- Daily returns must be numeric
- Configuration must contain required sections
- Trade history must have valid structure

## 📊 Performance Optimization

### Caching
- LRU caching for asset loss balance calculations
- Performance optimization for large datasets
- Memory-efficient data structures

### Computational Efficiency
- **O(1)**: Loss carryforward calculation
- **O(n)**: Asset-specific loss tracking
- **O(log n)**: Cached balance calculations

## 🔧 Customization

### Extending Metrics
```python
class CustomPerformanceMetrics(PerformanceMetrics):
    def calculate_custom_metric(self, data):
        """Calculate custom performance metric."""
        # Your custom calculation
        return custom_result
```

### Adding New Risk Metrics
```python
class CustomRiskMetrics(RiskAdjustedMetrics):
    def calculate_custom_risk_metric(self, returns):
        """Calculate custom risk metric."""
        # Your custom risk calculation
        return custom_risk_result
```

## 📚 API Reference

### PerformanceMetrics Class
```python
class PerformanceMetrics:
    def __init__(self, portfolio: EnhancedPortfolio, config_path: str = "config/settings.yaml")
    def calculate_returns(self, portfolio_values: List[float], start_date: datetime, end_date: datetime) -> ReturnsMetrics
    def calculate_risk_metrics(self, daily_returns: List[float]) -> RiskMetrics
    def calculate_tax_metrics(self) -> TaxMetrics
    def calculate_trade_metrics(self) -> TradeMetrics
```

### ComprehensivePerformanceAnalysis Class
```python
class ComprehensivePerformanceAnalysis:
    def __init__(self, portfolio: EnhancedPortfolio, config_path: str = "config/settings.yaml")
    def run_comprehensive_analysis(self, portfolio_values: List[float], daily_returns: List[float]) -> Dict[str, Any]
    def generate_performance_report(self, analysis_results: Dict[str, Any], output_path: str) -> None
    def plot_performance_charts(self, portfolio_values: List[float], daily_returns: List[float], output_path: str) -> None
```

## 🤝 Contributing

### Development Guidelines
1. Follow Brazilian market compliance requirements
2. Maintain regulatory compliance validation
3. Add comprehensive tests for new features
4. Update documentation for API changes
5. Ensure backward compatibility

### Testing Requirements
- Unit tests for all new functionality
- Integration tests with existing components
- Brazilian regulatory compliance validation
- Performance benchmarks for optimization

## 📄 License

This module is part of the `quant_b3_backtest`