"""
Engine package for Brazilian Market Backtesting Framework.

This package provides the core engine components for backtesting trading strategies
in the Brazilian stock market (B3).

Key Components:
- BaseStrategy: Abstract base class for all trading strategies
- Portfolio: Enhanced portfolio management with Brazilian tax compliance
- Simulator: Backtesting simulator with comprehensive performance tracking
- Market Utils: Brazilian market utilities for price ticks and lot sizes
- TCA: Transaction Cost Analysis module
- Settlement Manager: T+2 settlement tracking
- Loss Manager: Enhanced loss carryforward management

Author: Quantitative Trading Specialist
Date: 2024
"""

# Core strategy interface
from .base_strategy import (
    BaseStrategy,
    StrategyConfig,
    StrategyContext,
    Bar,
    Fill,
    OrderIntent,
    OrderType,
    OrderSide,
    DataPortal,
    BrokerSimulation
)

# Market utilities
from .market_utils import (
    BrazilianMarketUtils,
    TradeType,
    SignalType,
    LotType,
    OrderValidation
)

# Portfolio management
from .portfolio import (
    EnhancedPortfolio,
    Position,
    ClassifiedTrade,
    AssetType
)

# Performance and analysis
from .performance_metrics import (
    PerformanceMetrics,
    ComprehensivePerformanceAnalysis
)

# Transaction cost analysis
from .tca import TransactionCostAnalyzer

# Loss management
from .loss_manager import EnhancedLossCarryforwardManager

# Data loading
from .loader import DataLoader
from .sgs_data_loader import (
    SELICDataError,
    SELICDataUnavailableError,
    SELICDataInsufficientError,
    SELICDataQualityError,
    SELICDataValidationError,
    get_daily_factor
)

# Simulator
from .simulator import BacktestSimulator, SimulationResult, SimulationMetrics


__all__ = [
    # Core strategy interface
    'BaseStrategy',
    'StrategyConfig', 
    'StrategyContext',
    'Bar',
    'Fill',
    'OrderIntent',
    'OrderType',
    'OrderSide',
    
    # Market utilities
    'BrazilianMarketUtils',
    'TradeType',
    'SignalType',
    'LotType',
    'OrderValidation',
    
    # Portfolio management
    'EnhancedPortfolio',
    'Position',
    'ClassifiedTrade',
    'AssetType',
    
    # Performance and analysis
    'PerformanceMetrics',
    'ComprehensivePerformanceAnalysis',
    
    # Transaction cost analysis
    'TransactionCostAnalyzer',
    
    # Loss management
    'EnhancedLossCarryforwardManager',
    
    # Data loading
    'DataLoader',
    'SELICDataError',
    'SELICDataUnavailableError',
    'SELICDataInsufficientError',
    'SELICDataQualityError',
    'SELICDataValidationError',
    'get_daily_factor',
    
    # Simulator
    'BacktestSimulator',
    'SimulationResult',
    'SimulationMetrics',
    
]

__version__ = "1.0.0" 

# Optional run-scoped async event logger (injected by runner). Modules may emit to it if present.
event_logger = None