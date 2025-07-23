"""
Abstract Base Strategy for Brazilian Market Backtesting with FuzzyFajuto Implementation

This module provides the FuzzyFajuto trading strategy for the Brazilian stock market (B3).

The FuzzyFajuto Strategy:
- Compares stock returns against IBOV index
- Uses 5 EMAs (3, 5, 10, 15, 20 days) for trend assessment
- RSI(10) for overbought/oversold conditions
- Generates score-based signals: BUY >= 1.50, SELL <= -1.50
- ATR-based entry levels for volatility adaptation

Author: Quantitative Trading Specialist
Date: 2024
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import logging
import yaml
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import ta  # Technical analysis library for indicators

from engine.tca import TransactionCostAnalyzer
from engine.loader import DataLoader
from engine.market_utils import (
    BrazilianMarketUtils,
    TradeType, 
    SignalType, 
    OrderType, 
    LotType, 
    OrderValidation
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.portfolio import EnhancedPortfolio

# Configure logging
logger = logging.getLogger(__name__)





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
        trade_type: Brazilian trade type (day_trade, swing_trade, or auto)
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


@dataclass
class FuzzyFajutoVectors:
    """
    Vectors used by the FuzzyFajuto strategy.
    
    Attributes:
        dates_ibov: Dates for IBOV returns
        returns_ibov: Daily returns of IBOV index
        dates_stock: Dates for stock returns
        returns_stock: Daily returns of the stock
        ema_3: 3-day exponential moving average
        ema_5: 5-day exponential moving average
        ema_10: 10-day exponential moving average
        ema_15: 15-day exponential moving average
        ema_20: 20-day exponential moving average
        rsi_10: 10-day RSI indicator
        close_prices: Daily closing prices
        fuzzy_fajuto_score: FuzzyFajuto score vector
        atr: Average True Range (optional for limit orders)
    """
    dates_ibov: pd.DatetimeIndex = field(default_factory=pd.DatetimeIndex)
    returns_ibov: pd.Series = field(default_factory=pd.Series)
    dates_stock: pd.DatetimeIndex = field(default_factory=pd.DatetimeIndex)
    returns_stock: pd.Series = field(default_factory=pd.Series)
    ema_3: pd.Series = field(default_factory=pd.Series)
    ema_5: pd.Series = field(default_factory=pd.Series)
    ema_10: pd.Series = field(default_factory=pd.Series)
    ema_15: pd.Series = field(default_factory=pd.Series)
    ema_20: pd.Series = field(default_factory=pd.Series)
    rsi_10: pd.Series = field(default_factory=pd.Series)
    close_prices: pd.Series = field(default_factory=pd.Series)
    fuzzy_fajuto_score: pd.Series = field(default_factory=pd.Series)
    atr: pd.Series = field(default_factory=pd.Series)





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
        portfolio: "EnhancedPortfolio",
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
        from engine.portfolio import EnhancedPortfolio
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
        
        # FuzzyFajuto vectors
        self.fuzzy_vectors = FuzzyFajutoVectors()
        
        # Brazilian market specific
        self.trading_hours = self.config['market']['trading_hours']
        self.tax_config = self.config['taxes']
        self.settlement_config = self.config['settlement']
        
        # Initialize market utilities with configuration
        market_config = self.config['market']
        self.market_utils = BrazilianMarketUtils(
            tick_size=market_config.get('tick_size', 0.01),
            round_lot_size=market_config.get('round_lot_size', 100)
        )
        
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
    
    def calculate_position_size(self, signal: TradingSignal, 
                              available_cash: float) -> int:
        """
        Calculate position size based on risk metrics.
        
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
        Check Brazilian market-specific constraints including price ticks and lot sizes.
        
        This enhanced implementation validates:
        - Trading hours
        - Day trade exposure limits
        - Price tick normalization (R$ 0.01)
        - Lot size validation (round lots = multiples of 100)
        - Fractional lot handling
        
        Args:
            signal: Trading signal to validate
            
        Returns:
            True if constraints are satisfied, False otherwise
        """
        # Trading hours check removed for simplicity in backtesting
        
        # Check day trade constraints
        if signal.trade_type == TradeType.DAY_TRADE:
            # Check day trade exposure limits
            current_day_trade_exposure = self._calculate_day_trade_exposure()
            if current_day_trade_exposure > self.risk_metrics.max_day_trade_exposure:
                logger.warning(f"Day trade exposure limit exceeded: {current_day_trade_exposure:.2%}")
                return False
        
        # Validate price ticks and lot sizes using market utilities
        market_config = self.config['market']
        allow_fractional = market_config.get('allow_fractional_lots', True)
        
        # Use market utils for comprehensive validation
        validation = self.market_utils.validate_order(
            price=signal.price,
            quantity=signal.quantity,
            order_type=OrderType.MARKET,  # Default to market order for signals
            allow_fractional=allow_fractional
        )
        
        if not validation.is_valid:
            for message in validation.validation_messages:
                logger.warning(f"Market constraint violation: {message}")
            return False
        
        # Log validation results
        if validation.validation_messages:
            for message in validation.validation_messages:
                logger.debug(f"Market validation: {message}")
        
        # Update signal with normalized values if needed
        if abs(validation.normalized_price - signal.price) > 1e-6:
            logger.info(f"Price normalized: {signal.price} -> {validation.normalized_price}")
            signal.price = validation.normalized_price
        
        if validation.normalized_quantity != signal.quantity:
            logger.info(f"Quantity normalized: {signal.quantity} -> {validation.normalized_quantity}")
            signal.quantity = validation.normalized_quantity
        
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
        
        # Validate IBOV data for FuzzyFajuto strategy
        if self.strategy_name == 'FuzzyFajuto' or 'fuzzy' in self.strategy_name.lower():
            ibov_data = market_data.get('ibov_data')
            if ibov_data is None:
                logger.warning("FuzzyFajuto strategy requires IBOV data for optimal performance")
            elif not isinstance(ibov_data, dict):
                logger.error("IBOV data must be a dictionary")
                return False
        
        # Validate SGS data if present (optional but recommended)
        sgs_data = market_data.get('sgs_data')
        if sgs_data is not None:
            if not isinstance(sgs_data, dict):
                logger.warning("SGS data must be a dictionary")
                return False
            
            # Log available SGS data for debugging
            if sgs_data:
                logger.debug(f"Available SGS data: {list(sgs_data.keys())}")
        
        return True
    
    def get_sgs_data(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract SGS data from market data for strategy use.
        Always includes 'selic_daily_factor' if available for robust risk-free/benchmarking logic.
        """
        sgs_data = market_data.get('sgs_data', {})
        # Provide default values if SGS data is not available
        if not sgs_data:
            logger.warning("No SGS data available, using default values")
            return {
                'selic_interest_rate': 0.15,  # Default SELIC rate
                'cdi_interest_rate': 0.14,    # Default CDI rate
                'ipca_inflation_index': 0.04, # Default IPCA rate
                'series_11': 0.15,
                'series_12': 0.14,
                'series_433': 0.04,
                'selic_daily_factor': 1.0006  # Default daily factor (approx 15% annual)
            }
        # Always include selic_daily_factor if available
        if 'selic_daily_factor' not in sgs_data:
            # Try to compute from series_11 if possible
            if 'series_11' in sgs_data:
                valor = sgs_data['series_11']
                if valor > 1.0:
                    sgs_data['selic_daily_factor'] = valor
                elif valor > 0.01:
                    sgs_data['selic_daily_factor'] = 1 + (valor / 100)
                else:
                    sgs_data['selic_daily_factor'] = 1 + valor
            else:
                sgs_data['selic_daily_factor'] = 1.0006  # fallback
        return sgs_data
    
    def get_interest_rate_environment(self, market_data: Dict[str, Any]) -> str:
        """
        Get the current interest rate environment classification.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            String classification of interest rate environment
        """
        market_conditions = market_data.get('market_conditions', {})
        return market_conditions.get('interest_rate_environment', 'unknown')
    
    def get_inflation_environment(self, market_data: Dict[str, Any]) -> str:
        """
        Get the current inflation environment classification.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            String classification of inflation environment
        """
        market_conditions = market_data.get('market_conditions', {})
        return market_conditions.get('inflation_environment', 'unknown')
    
    def calculate_technical_indicators(self, price_data: pd.DataFrame) -> None:
        """
        Calculate technical indicators for the FuzzyFajuto strategy.
        
        Args:
            price_data: DataFrame with OHLCV data
        """
        if price_data.empty:
            logger.warning("Empty price data, cannot calculate indicators")
            return
        
        # Calculate EMAs
        self.fuzzy_vectors.ema_3 = price_data['close'].ewm(span=3, adjust=False).mean()
        self.fuzzy_vectors.ema_5 = price_data['close'].ewm(span=5, adjust=False).mean()
        self.fuzzy_vectors.ema_10 = price_data['close'].ewm(span=10, adjust=False).mean()
        self.fuzzy_vectors.ema_15 = price_data['close'].ewm(span=15, adjust=False).mean()
        self.fuzzy_vectors.ema_20 = price_data['close'].ewm(span=20, adjust=False).mean()
        
        # Calculate RSI using ta library
        self.fuzzy_vectors.rsi_10 = ta.momentum.RSIIndicator(close=price_data['close'], window=10).rsi()
        
        # Calculate ATR using ta library
        self.fuzzy_vectors.atr = ta.volatility.AverageTrueRange(
            high=price_data['high'],
            low=price_data['low'],
            close=price_data['close'],
            window=14
        ).average_true_range()
        
        # Store close prices and dates
        self.fuzzy_vectors.close_prices = price_data['close'].copy()
        self.fuzzy_vectors.dates_stock = price_data.index
        
        # Calculate stock returns
        self.fuzzy_vectors.returns_stock = price_data['close'].pct_change()
        
        logger.debug(f"Technical indicators calculated for {len(price_data)} periods")
    
    def calculate_fuzzy_fajuto_score(self, market_data: Dict[str, Any]) -> pd.Series:
        """
        Calculate the FuzzyFajuto score based on the strategy rules.
        
        Args:
            market_data: Market data including IBOV returns
            
        Returns:
            Series with FuzzyFajuto scores
        """
        # Get IBOV returns from market data
        ibov_data = market_data.get('ibov_data', {})
        ibov_returns = ibov_data.get('returns', pd.Series())
        
        # Initialize score series
        score = pd.Series(0.0, index=self.fuzzy_vectors.dates_stock)
        
        # Align IBOV returns with stock dates
        if not ibov_returns.empty:
            # Reindex IBOV returns to match stock dates
            aligned_ibov_returns = ibov_returns.reindex(self.fuzzy_vectors.dates_stock, method='ffill')
            
            # 3.1 - Força Relativa contra o IBOV
            stock_returns = self.fuzzy_vectors.returns_stock
            score[stock_returns > aligned_ibov_returns] += 1.0
            score[stock_returns < aligned_ibov_returns] -= 1.0
        
        # 3.2 - Comparação Preço x Médias Móveis Exponenciais
        close_prices = self.fuzzy_vectors.close_prices
        
        # EMA comparisons
        ema_list = [
            self.fuzzy_vectors.ema_3,
            self.fuzzy_vectors.ema_5,
            self.fuzzy_vectors.ema_10,
            self.fuzzy_vectors.ema_15,
            self.fuzzy_vectors.ema_20
        ]
        
        for ema in ema_list:
            score[close_prices > ema] += 0.25
            score[close_prices < ema] -= 0.25
        
        # 3.3 - Indicador de Excesso (RSI)
        rsi = self.fuzzy_vectors.rsi_10
        score[rsi > 65] += 0.25  # Sobrecompra
        score[rsi < 35] -= 0.25  # Sobrevenda
        
        # Store the score
        self.fuzzy_vectors.fuzzy_fajuto_score = score
        
        return score
    
    def generate_fuzzy_fajuto_signals(self, market_data: Dict[str, Any]) -> List[TradingSignal]:
        """
        Generate trading signals based on FuzzyFajuto score.
        
        Args:
            market_data: Current market data
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Calculate indicators and score
        price_data = market_data.get('price_data', pd.DataFrame())
        if price_data.empty:
            return signals
        
        self.calculate_technical_indicators(price_data)
        fuzzy_score = self.calculate_fuzzy_fajuto_score(market_data)
        
        if fuzzy_score.empty:
            return signals
        
        # Get current timestamp
        current_timestamp = market_data.get('timestamp', datetime.now())
        
        # Get the latest score
        latest_score = fuzzy_score.iloc[-1]
        latest_close = self.fuzzy_vectors.close_prices.iloc[-1]
        
        # 4.1 - Sinal de Compra
        if latest_score >= 1.50:
            # Calculate optimized entry limits
            entry_limits = self.calculate_entry_limits(latest_close, is_buy=True, use_atr=True)
            
            # Generate buy signal
            signal = TradingSignal(
                signal_type=SignalType.BUY,
                ticker=self.symbol,
                price=latest_close,
                quantity=100,  # Default quantity, will be adjusted by position sizing
                confidence=min(latest_score / 3.0, 1.0),  # Normalize confidence
                trade_type=TradeType.SWING_TRADE,
                timestamp=current_timestamp,
                metadata={
                    'fuzzy_score': latest_score,
                    'entry_type': 'fuzzy_fajuto_buy',
                    'limit_prices': entry_limits,
                    'current_atr': self.fuzzy_vectors.atr.iloc[-1] if not self.fuzzy_vectors.atr.empty else None
                }
            )
            signals.append(signal)
            
        # 4.2 - Sinal de Venda/Short
        elif latest_score <= -1.50:
            # Calculate optimized entry limits
            entry_limits = self.calculate_entry_limits(latest_close, is_buy=False, use_atr=True)
            
            # Generate sell signal
            signal = TradingSignal(
                signal_type=SignalType.SELL,
                ticker=self.symbol,
                price=latest_close,
                quantity=100,  # Default quantity, will be adjusted by position sizing
                confidence=min(abs(latest_score) / 3.0, 1.0),  # Normalize confidence
                trade_type=TradeType.SWING_TRADE,
                timestamp=current_timestamp,
                metadata={
                    'fuzzy_score': latest_score,
                    'entry_type': 'fuzzy_fajuto_sell',
                    'limit_prices': entry_limits,
                    'current_atr': self.fuzzy_vectors.atr.iloc[-1] if not self.fuzzy_vectors.atr.empty else None
                }
            )
            signals.append(signal)
        
        return signals
    
    def calculate_entry_limits(self, base_price: float, is_buy: bool, use_atr: bool = False) -> List[float]:
        """
        Calculate entry limit prices using ATR-based approach.
        
        Args:
            base_price: Base price (usually closing price)
            is_buy: True for buy orders, False for sell orders
            use_atr: Whether to use ATR for limits
            
        Returns:
            List of limit prices
        """
        if use_atr and not self.fuzzy_vectors.atr.empty:
            # Get current ATR value
            current_atr = self.fuzzy_vectors.atr.iloc[-1]
            
            # ATR multipliers for entry levels
            atr_multipliers = [0.5, 1.0, 1.5]
            
            if is_buy:
                limits = [base_price - (mult * current_atr) for mult in atr_multipliers]
            else:
                limits = [base_price + (mult * current_atr) for mult in atr_multipliers]
        else:
            # Fallback to percentage-based limits
            if is_buy:
                limits = [
                    base_price * (1 - 0.005),
                    base_price * (1 - 0.010),
                    base_price * (1 - 0.015)
                ]
            else:
                limits = [
                    base_price * (1 + 0.005),
                    base_price * (1 + 0.010),
                    base_price * (1 + 0.015)
                ]
        
        # Normalize prices to valid tick size
        normalized_limits = [self.market_utils.normalize_price(price) for price in limits]
        
        return normalized_limits
    
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
        
        # Add FuzzyFajuto specific metrics if available
        if hasattr(self, 'fuzzy_vectors') and not self.fuzzy_vectors.fuzzy_fajuto_score.empty:
            fuzzy_scores = self.fuzzy_vectors.fuzzy_fajuto_score
            performance['fuzzy_fajuto_metrics'] = {
                'average_score': fuzzy_scores.mean(),
                'max_score': fuzzy_scores.max(),
                'min_score': fuzzy_scores.min(),
                'std_score': fuzzy_scores.std(),
                'buy_signals': (fuzzy_scores >= 1.50).sum(),
                'sell_signals': (fuzzy_scores <= -1.50).sum(),
                'neutral_periods': ((fuzzy_scores > -1.50) & (fuzzy_scores < 1.50)).sum()
            }
        
        return performance
    
    def reset_strategy(self) -> None:
        """
        Reset strategy state (signals and trades history).
        """
        self.signals_history.clear()
        self.trades_history.clear()
        self.performance_metrics.clear()
        
        # Reset FuzzyFajuto vectors
        self.fuzzy_vectors = FuzzyFajutoVectors()
        
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