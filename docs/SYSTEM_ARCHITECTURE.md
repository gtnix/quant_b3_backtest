# Quant B3 Backtest System - System Architecture

## Overview

The Quant B3 Backtest System is a comprehensive backtesting engine designed specifically for the Brazilian stock market (B3) with full compliance to Brazilian tax regulations, settlement rules, and market constraints. This document provides a detailed overview of the system architecture and how all components work together.

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUANT B3 BACKTEST SYSTEM                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   SCRIPTS   │    │    ENGINE   │    │    CONFIG   │         │
│  │             │    │             │    │             │         │
│  │ • download_ │    │ • loader    │    │ • settings  │         │
│  │   data.py   │    │ • portfolio │    │ • secrets   │         │
│  │ • run_back  │    │ • tca       │    │             │         │
│  │   test.py   │    │ • loss_mgr  │    │             │         │
│  │ • download_ │    │ • settlement│    │             │         │
│  │   sgs.py    │    │ • simulator │    │             │         │
│  │             │    │ • base_str  │    │             │         │
│  │             │    │ • perf_mtr  │    │             │         │
│  │             │    │ • sgs_ldr   │    │             │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             │                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                     DATA FLOW                               │ │
│  │                                                             │ │
│  │  Alpha Vantage API (Stocks) → Data Loader → Portfolio → Simulator   │ │
│  │  Yahoo Finance API (IBOV) → Data Loader → Portfolio → Simulator   │ │
│  │       ↓              ↓           ↓           ↓             │ │
│  │  SGS Data → SGS Loader → Performance Metrics → Reports     │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Management Layer

#### Alpha Vantage Integration (`scripts/download_data.py`)
- **Purpose**: Downloads Brazilian stock market data (stocks only)
- **Features**: Rate limiting, error handling, data consolidation
- **Output**: Raw CSV files with OHLCV data and metadata

#### Yahoo Finance Integration (`scripts/download_ibov_yahoo.py`)
- **Purpose**: Downloads IBOV index data
- **Features**: Free access, no rate limits, comprehensive historical data
- **Output**: IBOV data files with metadata

#### SGS Data Integration (`engine/sgs_data_loader.py`)
- **Purpose**: Retrieves Banco Central economic data
- **Data**: SELIC rates, CDI rates, IPCA inflation
- **Features**: LOCF normalization, data quality validation

#### Data Processing (`engine/loader.py`)
- **Purpose**: Processes and validates market data
- **Features**: Technical indicators, corporate actions, quality filters
- **Output**: Processed data ready for backtesting

### 2. Portfolio Management Layer

#### Enhanced Portfolio (`engine/portfolio.py`)
- **Purpose**: Manages portfolio positions and cash flow
- **Features**: 
  - Position tracking with unrealized P&L
  - T+2 settlement management
  - Brazilian tax compliance
  - Day trade detection
  - Comprehensive audit trails

#### Transaction Cost Analysis (`engine/tca.py`)
- **Purpose**: Calculates all transaction costs
- **Costs**: B3 fees, brokerage, settlement, ISS taxes
- **Features**: Cost breakdown, comparison across trade types

#### Loss Carryforward Manager (`engine/loss_manager.py`)
- **Purpose**: Tracks and applies tax losses
- **Features**:
  - Per-asset and global loss tracking
  - 100% loss offset capability
  - Monthly exemption handling
  - Comprehensive audit trails

#### Settlement Manager (`engine/settlement_manager.py`)
- **Purpose**: Manages T+2 settlement cycle
- **Features**:
  - Business day calculations using dias_uteis
  - Cash flow simulation
  - Settlement queue management
  - Error handling and retry logic

### 3. Strategy Framework

#### Base Strategy (`engine/base_strategy.py`)
- **Purpose**: Abstract base class for trading strategies
- **Features**:
  - Abstract methods for signal generation
  - Risk management integration
  - Brazilian market constraints
  - SGS data integration
  - Comprehensive logging

### 4. Simulation Engine

#### Backtest Simulator (`engine/simulator.py`)
- **Purpose**: Main simulation engine
- **Features**:
  - Strategy-agnostic design
  - Comprehensive performance tracking
  - SGS data integration
  - Detailed reporting and analysis

#### Performance Metrics (`engine/performance_metrics.py`)
- **Purpose**: Calculates comprehensive performance metrics
- **Metrics**:
  - Returns (total, annualized, daily)
  - Risk metrics (Sharpe, drawdown, volatility)
  - Tax-aware metrics
  - Benchmark analysis
  - Trade metrics

### 5. Configuration Management

#### Settings (`config/settings.yaml`)
- **Purpose**: System-wide configuration
- **Sections**:
  - Market parameters and trading rules
  - Brazilian tax configuration
  - Portfolio settings
  - Settlement configuration
  - Loss carryforward settings
  - Performance and monitoring
  - SGS data configuration

#### Secrets (`config/secrets.yaml`)
- **Purpose**: API keys and sensitive data
- **Security**: Excluded from version control
- **Content**: Alpha Vantage API keys (for stock data)

## Data Flow Architecture

### 1. Data Ingestion Flow

```
Alpha Vantage API (Stocks) → EnhancedB3DataDownloader → Raw CSV Files
Yahoo Finance API (IBOV) → YahooIBOVDownloader → IBOV Data Files
                                                    ↓
SGS API → SGSDataLoader → Processed SGS Data → Interest Rate Data
                                                    ↓
DataLoader → Processed Market Data → Technical Indicators
```

### 2. Backtesting Flow

```
Processed Data → BacktestSimulator → Strategy.generate_signals()
                                           ↓
Trading Signals → Portfolio.buy/sell() → Transaction Cost Analysis
                                           ↓
Settlement Manager → T+2 Settlement → Cash Flow Updates
                                           ↓
Loss Manager → Tax Loss Tracking → Tax Calculations
                                           ↓
Performance Metrics → Comprehensive Analysis → Reports
```

### 3. Portfolio Management Flow

```
Trade Signal → Portfolio Validation → TCA Cost Calculation
                                           ↓
Cash Availability Check → Position Update → Settlement Scheduling
                                           ↓
Loss Recording → Tax Calculation → Performance Update
```

## Brazilian Market Compliance

### Tax Compliance (2025)
- **Individual Taxpayer Rules**: IN RFB 1.585/2015 compliance
- **Swing Trade Tax**: 15% on monthly net profit
- **Day Trade Tax**: 20% on monthly net profit
- **Exemption**: R$ 20,000/month for swing trades
- **Loss Offset**: 100% capability (no 30% limit)
- **Perpetual Carryforward**: Losses carried forward indefinitely

### Settlement Compliance
- **T+2 Settlement**: Standard Brazilian settlement cycle
- **Business Day Calculation**: Uses dias_uteis library
- **Cash Flow Modeling**: Realistic settlement delays
- **Holiday Calendar**: B3 holiday calendar integration

### Market Constraints
- **Trading Hours**: 10:00-16:55 (continuous session)
- **Minimum Volume**: R$ 1,000,000 daily volume
- **Price Limits**: Maximum 20% daily price change
- **Position Limits**: Configurable maximum positions

## Integration Patterns

### 1. Dependency Injection
- **Portfolio**: Injected into strategies and simulator
- **TCA**: Injected into portfolio for cost calculation
- **SGS Loader**: Injected into performance metrics
- **Configuration**: Injected into all components

### 2. Event-Driven Architecture
- **Trade Events**: Trigger settlement scheduling
- **Settlement Events**: Trigger cash flow updates
- **Loss Events**: Trigger tax calculations
- **Performance Events**: Trigger metric updates

### 3. Observer Pattern
- **Portfolio Changes**: Notify performance metrics
- **Trade Execution**: Notify audit trail
- **Settlement Processing**: Notify cash flow manager
- **Loss Application**: Notify tax manager

## Error Handling Architecture

### 1. Graceful Degradation
- **SGS Data Unavailable**: Fallback to static rates
- **API Failures**: Retry with exponential backoff
- **Data Quality Issues**: Filter and continue
- **Configuration Errors**: Use defaults and log warnings

### 2. Comprehensive Logging
- **Trade Execution**: Detailed trade logs
- **Settlement Processing**: Settlement audit trails
- **Loss Application**: Loss tracking logs
- **Performance Calculation**: Metric calculation logs

### 3. Validation Layers
- **Input Validation**: Validate all inputs before processing
- **Business Logic Validation**: Validate business rules
- **Data Quality Validation**: Validate data integrity
- **Configuration Validation**: Validate configuration parameters

## Performance Optimization

### 1. Caching Strategy
- **SGS Data**: Cache interest rate data
- **Business Days**: Cache business day calculations
- **Technical Indicators**: Cache calculated indicators
- **Configuration**: Cache configuration parameters

### 2. Memory Management
- **Lazy Loading**: Load data only when needed
- **Streaming**: Process large datasets in chunks
- **Garbage Collection**: Manage memory usage
- **Efficient Data Structures**: Use optimized data structures

### 3. Parallel Processing
- **Data Download**: Parallel ticker downloads (with rate limiting)
- **Technical Indicators**: Parallel indicator calculation
- **Performance Metrics**: Parallel metric calculation
- **Report Generation**: Parallel report generation

## Security Architecture

### 1. API Key Management
- **Secrets File**: Separate file for sensitive data
- **Environment Variables**: Override with environment variables
- **Git Exclusion**: Secrets excluded from version control
- **Access Control**: Restrict access to secrets file

### 2. Data Privacy
- **Raw Data**: Excluded from version control
- **Processed Data**: Excluded from version control
- **Audit Trails**: Secure audit trail storage
- **Reports**: Secure report storage

### 3. Input Validation
- **API Inputs**: Validate all API inputs
- **Configuration**: Validate configuration parameters
- **Market Data**: Validate data quality
- **User Inputs**: Validate user inputs

## Testing Architecture

### 1. Unit Testing
- **Module Tests**: Test individual modules
- **Component Tests**: Test component interactions
- **Integration Tests**: Test system integration
- **Performance Tests**: Test performance characteristics

### 2. Test Coverage
- **Code Coverage**: Comprehensive code coverage
- **Business Logic**: Test all business rules
- **Error Handling**: Test error scenarios
- **Edge Cases**: Test edge cases and boundary conditions

### 3. Test Data
- **Mock Data**: Use mock data for testing
- **Historical Data**: Use historical data for backtesting
- **Synthetic Data**: Generate synthetic data for edge cases
- **Real Data**: Use real data for integration testing

## Deployment Architecture

### 1. Development Environment
- **Local Setup**: Local development environment
- **Virtual Environment**: Isolated Python environment
- **Configuration**: Development-specific configuration
- **Logging**: Debug-level logging

### 2. Production Environment
- **Configuration**: Production-specific configuration
- **Logging**: Production-level logging
- **Monitoring**: Performance monitoring
- **Backup**: Data backup and recovery

### 3. CI/CD Pipeline
- **Automated Testing**: Run tests on every commit
- **Code Quality**: Check code quality and style
- **Documentation**: Generate and update documentation
- **Deployment**: Automated deployment process

## Monitoring and Observability

### 1. Logging Strategy
- **Structured Logging**: JSON-formatted logs
- **Log Levels**: DEBUG, INFO, WARNING, ERROR
- **Log Rotation**: Automatic log rotation
- **Log Aggregation**: Centralized log aggregation

### 2. Metrics Collection
- **Performance Metrics**: System performance metrics
- **Business Metrics**: Business-specific metrics
- **Error Metrics**: Error rates and types
- **Usage Metrics**: System usage patterns

### 3. Alerting
- **Error Alerts**: Alert on errors and failures
- **Performance Alerts**: Alert on performance issues
- **Business Alerts**: Alert on business rule violations
- **System Alerts**: Alert on system issues

## Future Enhancements

### 1. Real-Time Processing
- **Real-Time Data**: Real-time market data integration
- **Real-Time Trading**: Real-time trading capabilities
- **Real-Time Monitoring**: Real-time performance monitoring
- **Real-Time Alerts**: Real-time alerting system

### 2. Machine Learning Integration
- **ML Models**: Machine learning model integration
- **Feature Engineering**: Automated feature engineering
- **Model Training**: Automated model training
- **Model Deployment**: Automated model deployment

### 3. Cloud Integration
- **Cloud Storage**: Cloud-based data storage
- **Cloud Computing**: Cloud-based computation
- **Cloud APIs**: Cloud-based API integration
- **Cloud Monitoring**: Cloud-based monitoring

## Conclusion

The Quant B3 Backtest System provides a comprehensive, compliant, and extensible framework for Brazilian market backtesting. The modular architecture ensures maintainability, while the comprehensive integration ensures accuracy and compliance with Brazilian market regulations.

The system is designed to be:
- **Compliant**: Full compliance with Brazilian tax and market regulations
- **Accurate**: Precise transaction cost and tax calculations
- **Extensible**: Easy to add new strategies and features
- **Maintainable**: Clear separation of concerns and comprehensive documentation
- **Testable**: Comprehensive testing framework
- **Performant**: Optimized for speed and memory usage
- **Secure**: Secure handling of sensitive data 