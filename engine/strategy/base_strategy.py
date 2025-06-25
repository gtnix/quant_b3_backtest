"""
Abstract Base Strategy for Brazilian Market Backtesting

This module provides an abstract base class for implementing trading strategies
in the Brazilian stock market (B3) with comprehensive integration to the existing
backtesting engine components.

Features:
- Abstract interface for strategy implementation
- Integration with EnhancedPortfolio for position management
- Transaction Cost Analysis (TCA) integration
- Brazilian market-specific considerations (T+2 settlement, tax rules)
- Comprehensive logging and error handling
- Type hints and documentation

Compliance: Brazilian market regulations, B3 trading rules

Author: Your Name
Date: 2024
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import logging
import yaml
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

from ..portfolio import EnhancedPortfolio
from ..tca import TransactionCostAnalyzer
from ..loader import DataLoader

# Configure logging
logger = logging.getLogger(__name__)


class TradeType(Enum):
    """Enumeration for Brazilian market trade types."""
    DAY_TRADE = "day_trade"
    SWING_TRADE = "swing_trade"


class SignalType(Enum):
    """Enumeration for trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TradingSignal:
    """
    Comprehensive trading signal with Brazilian market metadata.
    
    Attributes:
        signal_type: Type of signal (buy, sell, hold)
        ticker: Stock ticker symbol
        price: Suggested execution price
        quantity: Suggested quantity
        confidence: Signal confidence (0.0 to 1.0)
        trade_type: Brazilian trade type (day_trade or swing_trade)
        timestamp: Signal generation timestamp
        metadata: Additional signal metadata
    """
    signal_type: SignalType
    ticker: str
    price: float
    quantity: int
    confidence: float = 1.0
    trade_type: TradeType = TradeType.SWING_TRADE
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate signal parameters."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        if self.price <= 0:
            raise ValueError("Price must be positive")
        
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")


@dataclass
class RiskMetrics:
    """
    Risk management metrics for Brazilian market strategies.
    
    Attributes:
        max_position_size: Maximum position size as percentage of portfolio
        max_daily_loss: Maximum daily loss limit
        max_drawdown: Maximum drawdown limit
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        max_day_trade_exposure: Maximum day trade exposure
    """
    max_position_size: float = 0.1  # 10% of portfolio
    max_daily_loss: float = 0.02    # 2% daily loss limit
    max_drawdown: float = 0.15      # 15% maximum drawdown
    stop_loss_pct: float = 0.05     # 5% stop loss
    take_profit_pct: float = 0.10   # 10% take profit
    max_day_trade_exposure: float = 0.3  # 30% day trade exposure


class BaseStrategy(ABC):
    """
    Abstract base class for Brazilian market trading strategies.
    
    This class provides a comprehensive foundation for implementing trading
    strategies in the Brazilian stock market with full integration to the
    existing backtesting engine components.
    
    Key Features:
    - Abstract methods for signal generation, risk management, and trade execution
    - Integration with EnhancedPortfolio for position management
    - Transaction Cost Analysis (TCA) integration
    - Brazilian market-specific considerations
    - Comprehensive logging and error handling
    - Parameter management and validation
    """
    
    def __init__(
        self,
        portfolio: EnhancedPortfolio,
        symbol: str,
        risk_tolerance: float = 0.02,
        config_path: str = "config/settings.yaml",
        strategy_name: Optional[str] = None
    ):
        """
        Initialize the base strategy.
        
        Args:
            portfolio: EnhancedPortfolio instance for position management
            symbol: Primary trading symbol (B3 ticker)
            risk_tolerance: Risk tolerance level (0.0 to 1.0)
            config_path: Path to configuration file
            strategy_name: Optional strategy name for logging
        """
        # Validate inputs
        if not isinstance(portfolio, EnhancedPortfolio):
            raise TypeError("Portfolio must be an EnhancedPortfolio instance")
        
        if not 0.0 <= risk_tolerance <= 1.0:
            raise ValueError("Risk tolerance must be between 0.0 and 1.0")
        
        # Core components
        self.portfolio = portfolio
        self.symbol = symbol.upper()
        self.risk_tolerance = risk_tolerance
        self.strategy_name = strategy_name or self.__class__.__name__
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.tca = TransactionCostAnalyzer(config_path)
        self.data_loader = DataLoader()
        
        # Strategy state
        self.risk_metrics = self._initialize_risk_metrics()
        self.parameters: Dict[str, Any] = {}
        self.signals_history: List[TradingSignal] = []
        self.trades_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # Brazilian market specific
        self.trading_hours = self.config['market']['trading_hours']
        self.tax_config = self.config['taxes']
        self.settlement_config = self.config['settlement']
        
        logger.info(f"Strategy '{self.strategy_name}' initialized for {self.symbol}")
        logger.info(f"Risk tolerance: {self.risk_tolerance:.2%}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration with error handling.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file not found
            yaml.YAMLError: If config file is invalid
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Validate required sections
            required_sections = ['market', 'taxes', 'portfolio', 'settlement']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Configuration missing required section: {section}")
            
            logger.info(f"Configuration loaded from {config_path}")
            return config
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML configuration: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def _initialize_risk_metrics(self) -> RiskMetrics:
        """
        Initialize risk metrics based on risk tolerance.
        
        Returns:
            RiskMetrics instance with calibrated parameters
        """
        # Scale risk metrics based on risk tolerance
        base_metrics = RiskMetrics()
        
        # Adjust metrics based on risk tolerance
        risk_multiplier = 1.0 + (self.risk_tolerance - 0.5) * 2  # 0.0 to 2.0 range
        
        return RiskMetrics(
            max_position_size=base_metrics.max_position_size * risk_multiplier,
            max_daily_loss=base_metrics.max_daily_loss * risk_multiplier,
            max_drawdown=base_metrics.max_drawdown * risk_multiplier,
            stop_loss_pct=base_metrics.stop_loss_pct * risk_multiplier,
            take_profit_pct=base_metrics.take_profit_pct * risk_multiplier,
            max_day_trade_exposure=base_metrics.max_day_trade_exposure * risk_multiplier
        )
    
    @abstractmethod
    def generate_signals(self, market_data: Dict[str, Any]) -> List[TradingSignal]:
        """
        Generate trading signals based on market data.
        
        This is the core method that concrete strategies must implement.
        It should analyze market data and return a list of trading signals.
        
        Args:
            market_data: Dictionary containing market data for analysis
                Expected keys:
                - 'price_data': DataFrame with OHLCV data
                - 'technical_indicators': Dict of calculated indicators
                - 'market_conditions': Dict of market state information
                - 'timestamp': Current timestamp
                
        Returns:
            List of TradingSignal objects
            
        Raises:
            NotImplementedError: Must be implemented by concrete strategies
        """
        pass
    
    @abstractmethod
    def manage_risk(self, current_positions: Dict[str, Any], 
                   market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement risk management specific to Brazilian market regulations.
        
        This method should handle:
        - Position size limits
        - Stop loss and take profit logic
        - Day trade exposure limits
        - Brazilian market-specific risk rules
        
        Args:
            current_positions: Current portfolio positions
            market_data: Current market data
            
        Returns:
            Dictionary containing risk management decisions
            
        Raises:
            NotImplementedError: Must be implemented by concrete strategies
        """
        pass
    
    @abstractmethod
    def execute_trade(self, signal: TradingSignal) -> bool:
        """
        Execute trades considering T+2 settlement and transaction costs.
        
        This method should:
        - Validate signal parameters
        - Check portfolio constraints
        - Calculate transaction costs
        - Execute the trade through portfolio
        - Handle Brazilian market specifics
        
        Args:
            signal: TradingSignal to execute
            
        Returns:
            True if trade executed successfully, False otherwise
            
        Raises:
            NotImplementedError: Must be implemented by concrete strategies
        """
        pass
    
    def update_parameters(self, new_parameters: Dict[str, Any]) -> None:
        """
        Update strategy parameters dynamically.
        
        Args:
            new_parameters: Dictionary of new parameter values
        """
        # Validate parameters
        for param_name, param_value in new_parameters.items():
            if not isinstance(param_name, str):
                raise ValueError("Parameter names must be strings")
            
            # Store parameter
            self.parameters[param_name] = param_value
            
            logger.info(f"Updated parameter '{param_name}': {param_value}")
        
        # Recalculate risk metrics if risk-related parameters changed
        risk_params = ['risk_tolerance', 'max_position_size', 'stop_loss_pct']
        if any(param in new_parameters for param in risk_params):
            self.risk_metrics = self._initialize_risk_metrics()
            logger.info("Risk metrics recalculated due to parameter update")
    
    def get_parameter(self, param_name: str, default: Any = None) -> Any:
        """
        Get strategy parameter value.
        
        Args:
            param_name: Parameter name
            default: Default value if parameter not found
            
        Returns:
            Parameter value or default
        """
        return self.parameters.get(param_name, default)
    
    def validate_market_data(self, market_data: Dict[str, Any]) -> bool:
        """
        Validate market data structure and completeness.
        
        Args:
            market_data: Market data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        required_keys = ['price_data', 'timestamp']
        
        # Check required keys
        if not all(key in market_data for key in required_keys):
            logger.error(f"Market data missing required keys: {required_keys}")
            return False
        
        # Validate price data
        price_data = market_data.get('price_data')
        if not isinstance(price_data, pd.DataFrame):
            logger.error("Price data must be a pandas DataFrame")
            return False
        
        # Check for minimum required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in price_data.columns for col in required_columns):
            logger.error(f"Price data missing required columns: {required_columns}")
            return False
        
        # Validate timestamp
        timestamp = market_data.get('timestamp')
        if not isinstance(timestamp, datetime):
            logger.error("Timestamp must be a datetime object")
            return False
        
        return True
    
    def calculate_position_size(self, signal: TradingSignal, 
                              available_cash: float) -> int:
        """
        Calculate optimal position size based on risk metrics.
        
        Args:
            signal: Trading signal
            available_cash: Available cash for trading
            
        Returns:
            Calculated position size (quantity)
        """
        # Get current portfolio value
        portfolio_value = self.portfolio.total_value
        
        # Calculate maximum position value
        max_position_value = portfolio_value * self.risk_metrics.max_position_size
        
        # Calculate position value based on signal
        position_value = min(
            signal.price * signal.quantity,
            max_position_value,
            available_cash
        )
        
        # Calculate quantity
        quantity = int(position_value / signal.price)
        
        # Ensure minimum quantity
        if quantity < 1:
            quantity = 0
        
        logger.debug(f"Calculated position size: {quantity} shares (value: R$ {position_value:,.2f})")
        return quantity
    
    def check_brazilian_market_constraints(self, signal: TradingSignal) -> bool:
        """
        Check Brazilian market-specific constraints.
        
        Args:
            signal: Trading signal to validate
            
        Returns:
            True if constraints are satisfied, False otherwise
        """
        # Check trading hours (simplified - in real implementation, check actual market hours)
        current_time = signal.timestamp.time()
        market_open = datetime.strptime(self.trading_hours['open'], "%H:%M").time()
        market_close = datetime.strptime(self.trading_hours['close'], "%H:%M").time()
        
        if not (market_open <= current_time <= market_close):
            logger.warning(f"Trade outside market hours: {current_time}")
            return False
        
        # Check day trade constraints
        if signal.trade_type == TradeType.DAY_TRADE:
            # Check day trade exposure limits
            current_day_trade_exposure = self._calculate_day_trade_exposure()
            if current_day_trade_exposure > self.risk_metrics.max_day_trade_exposure:
                logger.warning(f"Day trade exposure limit exceeded: {current_day_trade_exposure:.2%}")
                return False
        
        return True
    
    def _calculate_day_trade_exposure(self) -> float:
        """
        Calculate current day trade exposure as percentage of portfolio.
        
        Returns:
            Day trade exposure percentage
        """
        # This is a simplified calculation
        # In a real implementation, you would track actual day trade positions
        portfolio_value = self.portfolio.total_value
        if portfolio_value == 0:
            return 0.0
        
        # For now, return a conservative estimate
        return 0.0
    
    def log_signal(self, signal: TradingSignal) -> None:
        """
        Log trading signal with comprehensive metadata.
        
        Args:
            signal: Trading signal to log
        """
        log_entry = {
            'timestamp': signal.timestamp,
            'strategy': self.strategy_name,
            'symbol': signal.ticker,
            'signal_type': signal.signal_type.value,
            'price': signal.price,
            'quantity': signal.quantity,
            'confidence': signal.confidence,
            'trade_type': signal.trade_type.value,
            'metadata': signal.metadata
        }
        
        self.signals_history.append(signal)
        logger.info(f"Signal generated: {log_entry}")
    
    def log_trade(self, trade_result: Dict[str, Any]) -> None:
        """
        Log trade execution result.
        
        Args:
            trade_result: Trade execution result dictionary
        """
        self.trades_history.append(trade_result)
        logger.info(f"Trade executed: {trade_result}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get strategy performance summary.
        
        Returns:
            Dictionary containing performance metrics
        """
        total_signals = len(self.signals_history)
        executed_trades = len(self.trades_history)
        
        # Calculate basic metrics
        performance = {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'total_signals': total_signals,
            'executed_trades': executed_trades,
            'execution_rate': executed_trades / total_signals if total_signals > 0 else 0,
            'risk_tolerance': self.risk_tolerance,
            'risk_metrics': {
                'max_position_size': self.risk_metrics.max_position_size,
                'max_daily_loss': self.risk_metrics.max_daily_loss,
                'stop_loss_pct': self.risk_metrics.stop_loss_pct,
                'take_profit_pct': self.risk_metrics.take_profit_pct
            },
            'parameters': self.parameters.copy()
        }
        
        return performance
    
    def reset_strategy(self) -> None:
        """
        Reset strategy state (signals and trades history).
        """
        self.signals_history.clear()
        self.trades_history.clear()
        self.performance_metrics.clear()
        logger.info(f"Strategy '{self.strategy_name}' state reset")
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"{self.strategy_name}(symbol={self.symbol}, risk_tolerance={self.risk_tolerance:.2%})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the strategy."""
        return (f"{self.__class__.__name__}("
                f"portfolio={self.portfolio}, "
                f"symbol='{self.symbol}', "
                f"risk_tolerance={self.risk_tolerance}, "
                f"strategy_name='{self.strategy_name}')") 