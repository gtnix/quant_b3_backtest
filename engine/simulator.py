"""
Backtest Simulator for Brazilian Stock Market

A sophisticated backtesting simulator that integrates with the existing
quant_backtest engine components to provide comprehensive simulation
capabilities for Brazilian financial markets.

Features:
- Strategy-agnostic design with BaseStrategy integration
- Comprehensive performance tracking and metrics calculation
- Transaction cost analysis integration
- Settlement and loss carryforward management
- Detailed logging and error handling
- Brazilian market compliance

Author: Your Name
Date: 2024
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, date
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml
import sys

# Import dias_uteis for Brazilian business day calculations
try:
    import dias_uteis
    DIAS_UTEIS_AVAILABLE = True
except ImportError:
    dias_uteis = None
    DIAS_UTEIS_AVAILABLE = False

from engine.portfolio import EnhancedPortfolio
from engine.base_strategy import BaseStrategy, OrderIntent, Bar, OrderSide, OrderType, StrategyContext, DataPortal, BrokerSimulation
from engine.market_utils import SignalType, TradeType
from engine.performance_metrics import PerformanceMetrics, ComprehensivePerformanceAnalysis
from engine.tca import TransactionCostAnalyzer
from engine.sgs_data_loader import SELICDataError, SELICDataUnavailableError, SELICDataInsufficientError, SELICDataQualityError, SELICDataValidationError, get_daily_factor

# Configure logging
logger = logging.getLogger(__name__)

# Reduce noisy info logs for utility availability
if DIAS_UTEIS_AVAILABLE:
    logger.debug("dias_uteis library loaded successfully for warm-up business day calculations")
else:
    logger.debug("dias_uteis library not available, using calendar day counting for warm-up")


class SimulationDataPortal(DataPortal):
    """
    Concrete implementation of DataPortal for backtesting simulation.
    
    Provides strategy access to market data during simulation.
    """
    
    def __init__(self, data_loader=None):
        """Initialize with data loader."""
        self.data_loader = data_loader
        self._current_data = None
        self._historical_data = {}
    
    def set_current_data(self, data: pd.DataFrame):
        """Set current market data for the simulation."""
        self._current_data = data
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol."""
        if self._current_data is not None and 'close' in self._current_data.columns:
            return float(self._current_data['close'].iloc[-1])
        return None
    
    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Get historical market data for a symbol."""
        if self.data_loader:
            # Use data loader to get historical data
            try:
                return self.data_loader.load_raw_data(symbol)
            except Exception as e:
                logger.error(f"Error loading historical data for {symbol}: {e}")
                return None
        
        # Return cached data if available
        return self._historical_data.get(symbol)
    
    def get_market_data(self, symbol: str, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Get market data for a specific symbol and timestamp."""
        if self._current_data is not None:
            # Find closest timestamp in current data
            try:
                if timestamp in self._current_data.index:
                    row = self._current_data.loc[timestamp]
                    return {
                        'open': float(row.get('open', 0)),
                        'high': float(row.get('high', 0)),
                        'low': float(row.get('low', 0)),
                        'close': float(row.get('close', 0)),
                        'volume': int(row.get('volume', 0)),
                        'timestamp': timestamp
                    }
            except Exception as e:
                logger.debug(f"Error getting market data for {symbol} at {timestamp}: {e}")
        
        return None


class SimulationBroker(BrokerSimulation):
    """
    Concrete implementation of BrokerSimulation for backtesting.
    
    Provides order execution simulation during backtesting.
    """
    
    def __init__(self, portfolio):
        """Initialize with portfolio reference."""
        self.portfolio = portfolio
        self._order_counter = 0
        self._pending_orders = {}
    
    def submit_order(self, intent: OrderIntent) -> str:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        """Submit an order intent and return order ID."""
        self._order_counter += 1
        order_id = f"sim_order_{self._order_counter}"
        
        # Store pending order for later execution
        self._pending_orders[order_id] = intent
        
        logger.debug(f"Order submitted: {order_id} - {intent.side.value} {intent.quantity} {intent.symbol}")
        return order_id
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current position information for a symbol."""
        if symbol in self.portfolio.positions:
            pos = self.portfolio.positions[symbol]
            return {
                'symbol': symbol,
                'quantity': pos.quantity,
                'avg_price': pos.avg_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl
            }
        return None
    
    def get_cash_balance(self) -> float:
        """Get current cash balance."""
        return getattr(self.portfolio, 'cash', 0.0)
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        return self.portfolio.get_portfolio_value()
    
    def execute_pending_orders(self, current_prices: Dict[str, float]) -> List[str]:
        """Execute pending orders with current market prices."""
        executed_orders = []
        
        for order_id, intent in list(self._pending_orders.items()):
            try:
                # Use current market price for execution
                execution_price = current_prices.get(intent.symbol, intent.price)
                if execution_price:
                    # Execute the order through portfolio
                    if intent.side == OrderSide.BUY:
                        success = self.portfolio.buy(
                            ticker=intent.symbol,
                            quantity=intent.quantity,
                            price=execution_price,
                            trade_date=intent.timestamp,
                            trade_id=order_id
                        )
                    else:  # SELL
                        success = self.portfolio.sell(
                            ticker=intent.symbol,
                            quantity=intent.quantity,
                            price=execution_price,
                            trade_date=intent.timestamp,
                            trade_id=order_id
                        )
                    
                    if success:
                        executed_orders.append(order_id)
                        del self._pending_orders[order_id]
                        logger.debug(f"Order executed: {order_id} at price {execution_price}")
                    else:
                        logger.warning(f"Order execution failed: {order_id}")
                        
            except Exception as e:
                logger.error(f"Error executing order {order_id}: {e}")
        
        return executed_orders


@dataclass
class SimulationResult:
    """Comprehensive simulation results with performance metrics."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_loss_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    final_portfolio_value: float
    initial_capital: float
    total_commission: float
    total_taxes: float
    daily_returns: List[float]
    portfolio_values: List[float]
    trade_log: List[Dict[str, Any]]
    simulation_duration: float
    start_date: datetime
    end_date: datetime
    # Benchmark metrics (optional)
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    information_ratio: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    tracking_error: float = 0.0
    rolling_correlation: float = 0.0
    benchmark_sharpe: float = 0.0
    benchmark_max_drawdown: float = 0.0
    benchmark_win_rate: float = 0.0


@dataclass
class SimulationMetrics:
    """Detailed simulation metrics for analysis."""
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_loss_ratio: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    total_commission: float = 0.0
    total_taxes: float = 0.0
    net_profit: float = 0.0
    final_portfolio_value: float = 0.0
    initial_capital: float = 0.0


class BacktestSimulator:
    """
    Sophisticated backtesting simulator for Brazilian stock market.
    
    This simulator integrates with all existing engine components to provide
    comprehensive backtesting capabilities with full Brazilian market compliance.
    """
    
    def __init__(
        self, 
        strategy: BaseStrategy, 
        initial_capital: float = 100000.0, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        config_path: str = "config/settings.yaml"
    ):
        """
        Initialize the backtest simulator.
        
        Args:
            strategy: BaseStrategy instance to simulate
            initial_capital: Starting capital in BRL
            start_date: Simulation start date (YYYY-MM-DD format)
            end_date: Simulation end date (YYYY-MM-DD format)
            config_path: Path to configuration file
        """
        # Validate inputs
        if not isinstance(strategy, BaseStrategy):
            raise ValueError("Strategy must be a BaseStrategy instance")
        
        # Validate required strategy methods - only check abstract methods
        required_methods = [
            'generate_intents'
        ]
        
        missing_methods = []
        for method in required_methods:
            if not hasattr(strategy, method):
                missing_methods.append(method)
        
        if missing_methods:
            raise ValueError(f"Strategy missing required abstract methods: {missing_methods}. "
                           f"Ensure BaseStrategy is properly implemented with all abstract methods.")
        
        # Validate optional methods that should be available
        optional_methods = [
            'on_end_of_day',
            'on_start',
            'on_warmup',
            'on_fill',
            'on_end'
        ]
        
        missing_optional = []
        for method in optional_methods:
            if not hasattr(strategy, method):
                missing_optional.append(method)
        
        if missing_optional:
            logger.warning(f"Strategy missing optional methods: {missing_optional}. "
                          f"These methods are recommended for full functionality.")
        
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        
        # Parse dates
        self.start_date = None
        self.end_date = None
        
        if start_date:
            try:
                self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("Start date must be in YYYY-MM-DD format")
        
        if end_date:
            try:
                self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("End date must be in YYYY-MM-DD format")
        
        if self.start_date and self.end_date and self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")
        
        # Initialize components
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.config_path = config_path
        
        # Load configuration
        self.config = self._load_config()
        
        # Create portfolio with initial capital
        self.portfolio = EnhancedPortfolio(config_path)
        self.portfolio.cash = initial_capital
        self.portfolio.initial_cash = initial_capital
        self.portfolio.total_value = initial_capital
        
        # Create market utilities
        from engine.market_utils import BrazilianMarketUtils
        self.market_utils = BrazilianMarketUtils()
        
        # Create data portal with data loader capability
        from engine.loader import DataLoader
        data_loader = DataLoader(auto_download=False)  # Use existing data
        self.data_portal = SimulationDataPortal(data_loader)
        
        # Create broker simulation
        self.broker = SimulationBroker(self.portfolio)
        
        # Create strategy logger
        strategy_logger = logging.getLogger(f"{strategy.__class__.__name__}")
        
        # Initialize tracking
        self.daily_portfolio_values: List[float] = []
        self.daily_returns: List[float] = []
        self.trade_log: List[Dict[str, Any]] = []
        self.simulation_start_time: Optional[datetime] = None
        self.simulation_end_time: Optional[datetime] = None
        
        # Performance metrics (includes benchmark analysis)
        self.performance_metrics = PerformanceMetrics(self.portfolio, config_path)
        
        # Simulation-specific metrics
        self.simulation_metrics = SimulationMetrics()
        
        # Track days to skip due to missing open bar policy
        self._skipped_trading_days = set()

        # Performance caches (pure-Python speedups; do not alter behavior)
        # Mapping of timestamp -> (start_index, end_index_exclusive) for fast per-timestamp slicing
        self._ts_bounds_map: Dict[datetime, Tuple[int, int]] = {}
        # Mapping of timestamp -> first index position for fast historical slicing up to ts (exclusive)
        self._ts_pos_map: Dict[datetime, int] = {}
        # Cached, normalized benchmark (^BVSP) dataframe (loaded once)
        self._ibov_df_cached = None

        # Unified fills capture (authoritative, simulator-level)
        # Each entry will contain: timestamp, symbol, side, quantity, price, lot_type,
        # original vs normalized values (rounding deltas), trade_type, attempt metadata,
        # and tranche sizing metadata from centralized config/strategy context.
        self.unified_fills: list[dict] = []
        self.unified_fills_df: Optional[pd.DataFrame] = None
        
        # Initialize SGS data for the entire backtest period
        self.selic_data = None
        self.all_sgs_data = {}
        self._initialize_sgs_data()
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"BacktestSimulator initialized with R$ {initial_capital:,.2f} initial capital")
        logger.info(f"Strategy: {self.strategy.name}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration with error handling."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def _setup_logging(self) -> None:
        """Setup simulation-specific logging."""
        # Reduce I/O overhead: only add a file handler once per process
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"backtest_simulator_{timestamp}.log"
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        logger.info(f"Simulation logging initialized: {log_file}")
    
    def _initialize_sgs_data(self):
        """
        Initialize all SGS data for the entire backtest period.
        Preloads SELIC, CDI, and IPCA data to eliminate per-bar API calls.
        """
        try:
            # Only initialize if we have date range
            if not self.start_date or not self.end_date:
                logger.warning("No date range specified, SGS data will be loaded per date")
                return
            
            # Initialize SGS loader
            from engine.sgs_data_loader import SGSDataLoader
            self.sgs_loader = SGSDataLoader()
            
            # Load configuration for strict mode
            config = self._load_config()
            strict_config = config.get('sgs', {}).get('strict_mode', {})
            
            # Format dates for SGS API
            start_str = self.start_date.strftime('%d/%m/%Y')
            end_str = self.end_date.strftime('%d/%m/%Y')
            
            logger.info(f"Preloading all SGS data for simulation period {start_str} to {end_str}")
            
            # Preload all SGS data (SELIC, CDI, IPCA)
            self.all_sgs_data = self.sgs_loader.preload_simulation_data(
                start_date=start_str,
                end_date=end_str
            )
            
            # Extract SELIC data for backward compatibility
            if 11 in self.all_sgs_data:
                selic_data = self.all_sgs_data[11]
                if selic_data is not None and not selic_data.empty:
                    self.selic_data = selic_data
                    
                    # Set SELIC data in performance metrics
                    self.performance_metrics.set_selic_data(selic_data)
                    
                    logger.info(f"SELIC data ready: {len(selic_data)} data points")
                    logger.info(f"SELIC rate range: {selic_data['valor'].min():.4f}% to {selic_data['valor'].max():.4f}%")
                else:
                    if strict_config.get('fail_on_missing_data', False):
                        raise RuntimeError("No SELIC data available for the backtest period")
                    else:
                        logger.warning("No SELIC data available, will use static rate")
            else:
                logger.warning("SELIC data not found in preloaded SGS data")
                    
        except Exception as e:
            config = self._load_config()
            strict_config = config.get('sgs', {}).get('strict_mode', {})
            # Allow override via selic.strict_mode: false
            try:
                selic_override = (config.get('selic', {}) or {}).get('strict_mode', None)
            except Exception:
                selic_override = None
            if selic_override is False:
                logger.warning(f"SELIC strict-mode override active; proceeding despite SGS init error: {e}")
                return
            if strict_config.get('fail_on_missing_data', False):
                logger.error(f"Critical SGS data initialization error: {e}")
                raise RuntimeError(f"Backtest cannot proceed due to SGS data issues: {e}")
            else:
                logger.warning(f"SGS data initialization failed: {e}. Will load per date.")
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and prepare input data for simulation.
        
        Args:
            data: Raw market data DataFrame
            
        Returns:
            Prepared DataFrame with date filtering and validation
        """
        logger.info("Preparing data for simulation...")
        
        # Validate input
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        if data.empty:
            raise ValueError("Data DataFrame is empty")
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data = data.set_index('date')
            else:
                data.index = pd.to_datetime(data.index)
        
        # Sort by date
        data = data.sort_index()
        
        # Filter by date range if specified
        if self.start_date:
            data = data[data.index >= self.start_date]
            logger.info(f"Filtered data from {self.start_date.date()}")
        
        if self.end_date:
            data = data[data.index <= self.end_date]
            logger.info(f"Filtered data until {self.end_date.date()}")
        
        # Remove rows with missing values
        initial_rows = len(data)
        data = data.dropna()
        removed_rows = initial_rows - len(data)
        
        if removed_rows > 0:
            logger.warning(f"Removed {removed_rows} rows with missing values")
        
        # Validate data quality
        if data.empty:
            raise ValueError("No valid data remaining after filtering")
        
        # Check for reasonable price values
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if (data[col] <= 0).any():
                raise ValueError(f"Found non-positive values in {col} column")
        
        # Check for reasonable volume values
        if (data['volume'] < 0).any():
            raise ValueError("Found negative values in volume column")
        
        logger.info(f"Data preparation complete: {len(data)} rows, "
                   f"date range: {data.index.min().date()} to {data.index.max().date()}")
        
        return data
    
    def run_simulation(self, data: pd.DataFrame) -> SimulationResult:
        """
        Run the complete backtest simulation.
        
        Args:
            data: Prepared market data DataFrame
            
        Returns:
            SimulationResult with comprehensive performance metrics
        """
        logger.info("Starting backtest simulation...")
        # Determine and log extended warmup based on strategy requirements
        try:
            req = self.strategy.required_history()
            ema_max = max(req.get('ema_windows', [20]))
            mul = float(req.get('warmup_multiplier_for_ema', 3.0))
            buf = int(req.get('buffer_sessions', 5))
            cal_buf = int(req.get('calendar_buffer_sessions', 3))
            required_warmup_sessions = max(
                int(math.ceil(mul * ema_max)),
                int(req.get('rsi_window', 10)) + buf,
                int(req.get('atr_window', 14)) + buf,
                int(req.get('rel_strength_return_window', 5)) + buf
            )
            # Compute extended_start by approximating business days
            first_dt = pd.to_datetime(data.index.min()).date()
            # Heuristic: 2x sessions in calendar days, then backtest loader ensures slicing
            extended_start = pd.to_datetime(first_dt) - pd.Timedelta(days=(required_warmup_sessions + cal_buf) * 2)
            logger.info(f"Warmup requirements: sessions={required_warmup_sessions}, calendar_buffer={cal_buf}")
            logger.info(f"Extended start (approx): {extended_start.date()} | Backtest start: {first_dt}")
            # Allow strategy to prewarm indicators on extended window
            try:
                self.strategy.prewarm_indicators(start_d=extended_start.date(), end_d=pd.to_datetime(data.index.max()).date())
            except Exception as _e:
                logger.debug(f"prewarm_indicators skipped: {_e}")
        except Exception as _e:
            logger.debug(f"required_history not available or failed: {_e}")
        
        # Validate data
        if data.empty:
            raise ValueError("Cannot run simulation with empty data")
        
        # Record simulation start time
        self.simulation_start_time = datetime.now()
        
        try:
            # Detect universe; support combined data with 'symbol' column
            universe: List[str] = []
            if 'symbol' in data.columns:
                try:
                    universe = sorted(list({str(s).upper() for s in data['symbol'].unique()}))
                except Exception:
                    universe = []
            if not universe and hasattr(self.strategy, 'config') and hasattr(self.strategy.config, 'universe'):
                universe = list(self.strategy.config.universe or [])
            if not universe:
                universe = ['UNKNOWN']
            logger.info(f"Using universe: {universe}")
            
            # Create strategy context with complete data
            strategy_context = self._create_strategy_context(data)
            
            # Ensure strategy has the context with complete data
            if hasattr(self.strategy, 'context'):
                # Preserve existing complete_data if it's larger than simulation data
                existing_complete_data = self.strategy.context.metadata.get('complete_data')
                if existing_complete_data is not None and len(existing_complete_data) > len(data):
                    logger.info(f"Preserving existing complete_data with {len(existing_complete_data)} records (simulation has {len(data)})")
                    # Keep the larger complete dataset and update other metadata
                    self.strategy.context.metadata.update({
                        'simulation_start': self.start_date,
                        'simulation_end': self.end_date,
                        'initial_capital': self.initial_capital
                    })
                else:
                    # Update with simulation data if it's larger or no existing data
                    self.strategy.context.metadata['complete_data'] = data.copy()
                    logger.debug(f"Updated existing strategy context with {len(data)} records")
                
                # Ensure hybrid_data_result is preserved for technical indicators
                if 'hybrid_data_result' not in self.strategy.context.metadata:
                    logger.warning("No hybrid_data_result found in strategy context - technical indicators may fail")
            else:
                # Assign the new context to the strategy
                self.strategy.context = strategy_context
                logger.info(f"Assigned new strategy context with {len(data)} records")
            
            # Initialize tracking
            self.daily_portfolio_values = []
            self.daily_returns = []
            self.trade_log = []
            
            # Reset portfolio to initial state
            self.portfolio.cash = self.initial_capital
            self.portfolio.positions = {}
            self.portfolio.total_value = self.initial_capital
            self.portfolio.trade_history = []
            self.portfolio.total_trades = 0
            self.portfolio.total_commission = 0.0
            self.portfolio.total_taxes = 0.0
            self.portfolio.total_operations = 0
            self.portfolio.buy_operations = 0
            self.portfolio.sell_operations = 0
            
            # Reset strategy if method exists
            if hasattr(self.strategy, 'reset_strategy'):
                self.strategy.reset_strategy()
            elif hasattr(self.strategy, 'clear_state'):
                self.strategy.clear_state()
            else:
                logger.warning("Strategy does not have reset_strategy or clear_state method")
            
            # Call strategy on_start if available
            if hasattr(self.strategy, 'on_start'):
                try:
                    self.strategy.on_start(data.index[0])
                    logger.info("Strategy on_start called")
                except Exception as e:
                    logger.error(f"Error in strategy on_start: {e}")
            
            # Record initial portfolio value BEFORE processing any days
            # This ensures the first entry in portfolio_values is the actual initial capital
            initial_portfolio_value = self.portfolio.get_portfolio_value()
            self.daily_portfolio_values.append(initial_portfolio_value)
            self.daily_returns.append(0.0)  # No return on day 0
            
            logger.info(f"Initial portfolio value recorded: R$ {initial_portfolio_value:,.2f}")
            
            # Track strategy warm-up period with both calendar and trading days
            strategy_active = False
            warmup_completed = False  # Track if warmup has been called
            warm_up_calendar_days = 0  # Total calendar days in warm-up
            warm_up_trading_days = 0   # Total trading days in warm-up
            
            # Precompute daily open info and simple progress planning
            # Precompute per-day index slices to avoid repeated boolean masking
            # Build an ordered mapping from date -> slice indices once
            idx = data.index
            dates_arr = idx.date
            unique_dates = np.unique(dates_arr)
            total_days = int(len(unique_dates)) if unique_dates is not None else 0
            # Map date to (start_pos, end_pos_exclusive)
            date_boundaries: Dict[date, tuple] = {}
            if total_days > 0:
                # Compute boundaries in a single pass
                last_date = dates_arr[0]
                start_pos = 0
                for i in range(1, len(idx)):
                    d = dates_arr[i]
                    if d != last_date:
                        date_boundaries[last_date] = (start_pos, i)
                        start_pos = i
                        last_date = d
                date_boundaries[last_date] = (start_pos, len(idx))
            processed_days = 0

            # Process each hourly bar (intraday processing)
            current_trading_day = None
            
            # Precompute fast per-timestamp index ranges (avoid repeated boolean masks)
            try:
                ts_bounds: Dict[datetime, Tuple[int, int]] = {}
                ts_pos: Dict[datetime, int] = {}
                if len(idx) > 0:
                    last_ts = idx[0]
                    start_pos_ts = 0
                    for i in range(1, len(idx)):
                        ts_i = idx[i]
                        if ts_i != last_ts:
                            ts_bounds[last_ts] = (start_pos_ts, i)
                            ts_pos[last_ts] = start_pos_ts
                            start_pos_ts = i
                            last_ts = ts_i
                    ts_bounds[last_ts] = (start_pos_ts, len(idx))
                    ts_pos[last_ts] = start_pos_ts
                # Expose caches for downstream helpers
                self._ts_bounds_map = ts_bounds
                self._ts_pos_map = ts_pos
            except Exception:
                # Fallback to empty (helpers will use slower path)
                self._ts_bounds_map = {}
                self._ts_pos_map = {}

            # Iterate by unique timestamps using precomputed bounds
            for timestamp, (ts_start, ts_end) in self._ts_bounds_map.items():
                current_date = timestamp.date()
                current_hour = timestamp.hour
                
                # If this trading day was marked to be skipped, continue
                if current_date in self._skipped_trading_days:
                    continue
                
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Processing bar: %s (Hour: %d)", timestamp, current_hour)
                
                # If we are starting a new trading day, process the previous day's EOD BEFORE any new intents
                if current_trading_day is not None and current_date != current_trading_day:
                    # Process previous trading day's end-of-day activities (MOC must be the last op of the day)
                    self._process_end_of_trading_day(current_trading_day)
                
                # Detect new trading day and handle missing open bar policy (initialize per-day metadata) BEFORE intents
                is_new_trading_day = (current_trading_day is None) or (current_date != current_trading_day)
                if is_new_trading_day:
                    try:
                        # Determine expected open timestamp in UTC hours (BRAPI UTC):
                        # B3 continuous session: 13:00 UTC open, 20:00 UTC close
                        expected_open_hour_utc = 13
                        # Use precomputed boundaries to slice quickly
                        start_end = date_boundaries.get(current_date)
                        if start_end is not None:
                            s, e = start_end
                            day_index = idx[s:e]
                        else:
                            day_index = idx[(dates_arr == current_date)]  # fallback
                        first_bar_ts = day_index[0] if len(day_index) > 0 else None
                        if len(day_index) > 0:
                            hours = day_index.hour
                            # Find first position where hour==expected_open_hour_utc
                            pos = int(np.argmax(hours == expected_open_hour_utc)) if (hours == expected_open_hour_utc).any() else -1
                            expected_open_ts = (day_index[pos] if pos >= 0 else None)
                        else:
                            expected_open_ts = None
                        missing_open = (expected_open_ts is None) or (first_bar_ts is not None and first_bar_ts != expected_open_ts)
                        behavior = (
                            self.config.get('strategy', {})
                                       .get('execution', {})
                                       .get('missing_open_bar_behavior', 'skip_day')
                        )
                        # Expose per-day metadata into strategy context for CSV enrichment
                        try:
                            if hasattr(self.strategy, 'context') and hasattr(self.strategy.context, 'metadata'):
                                day_meta = self.strategy.context.metadata.get('day_metadata', {})
                                if not isinstance(day_meta, dict):
                                    day_meta = {}
                                day_meta[current_date.isoformat()] = {
                                    'expected_open': expected_open_ts,
                                    'first_bar': first_bar_ts,
                                    'missing_open_bar': bool(missing_open),
                                    'policy': behavior
                                }
                                self.strategy.context.metadata['day_metadata'] = day_meta
                        except Exception:
                            pass
                        
                        if missing_open:
                            if behavior == 'skip_day':
                                logger.warning(f"Missing expected open bar for {current_date}. Policy=skip_day → skipping all trading for the day.")
                                # Mark entire day to be skipped
                                self._skipped_trading_days.add(current_date)
                            elif behavior == 'use_first_available':
                                logger.warning(f"Missing expected open bar for {current_date}. Policy=use_first_available → using first available bar {first_bar_ts}.")
                            elif behavior == 'alert_only':
                                logger.warning(f"Missing expected open bar for {current_date}. Policy=alert_only → proceeding normally.")
                    except Exception as e:
                        logger.error(f"Error evaluating missing open bar policy for {current_date}: {e}")

                    # Update simple progress bar once per new trading day
                    processed_days += 1
                    if total_days > 0:
                        pct = processed_days / total_days
                        bar_len = 30
                        filled = int(bar_len * pct)
                        bar = '[' + ('#' * filled) + ('-' * (bar_len - filled)) + ']'
                        try:
                            sys.stdout.write(f"\rSimulating {processed_days}/{total_days} days {bar} {pct:6.2%}")
                            sys.stdout.flush()
                        except Exception:
                            pass
                
                # Update current trading day after EOD processing and day-init
                current_trading_day = current_date
                
                # Single-asset path is handled later using fast iloc; skip here
                
                # Prepare market data for strategy
                market_data = self._prepare_market_data(data, timestamp)
                
                if not market_data:
                    logger.warning(f"Skipping bar {timestamp} due to missing market data")
                    continue
                
                # Prepare current slice (may contain multiple symbols) using fast iloc
                try:
                    current_slice = data.iloc[ts_start:ts_end]
                except Exception:
                    # Fallback to original mask-based selection
                    current_slice = data.loc[[timestamp]]
                self.data_portal.set_current_data(current_slice)
                
                if 'symbol' in current_slice.columns:
                    # Multi-asset: process each symbol row at this timestamp
                    price_updates = {}
                    # itertuples is faster than iterrows and avoids Series allocation per row
                    for row in current_slice.itertuples(index=False):
                        sym = str(getattr(row, 'symbol')).upper()
                        close_v = float(getattr(row, 'close'))
                        price_updates[sym] = close_v
                        bar = Bar(
                            symbol=sym,
                            timestamp=timestamp,
                            open=float(getattr(row, 'open')),
                            high=float(getattr(row, 'high')),
                            low=float(getattr(row, 'low')),
                            close=close_v,
                            volume=int(getattr(row, 'volume')) if hasattr(row, 'volume') else 0
                        )
                        # Handle warmup once globally; per-symbol warmup uses same bars
                        if not warmup_completed and hasattr(self.strategy, 'on_warmup'):
                            try:
                                complete_data = data
                                if hasattr(self.strategy, 'context') and 'complete_data' in self.strategy.context.metadata:
                                    complete_data = self.strategy.context.metadata['complete_data']
                                warmup_bars = self._get_warmup_bars(complete_data, timestamp, sym)
                                if warmup_bars:
                                    self.strategy.on_warmup(sym, warmup_bars)
                                    warmup_completed = True
                                    logger.info(f"Strategy warmup completed with {len(warmup_bars)} bars")
                                else:
                                    # Multi-frame: allow proceeding without intraday warmup
                                    mf_mode = os.getenv('MULTIFRAME_MODE', 'off').lower() in ('1','true','yes','on')
                                    if mf_mode:
                                        logger.warning("Multi-frame: no intraday warmup bars; proceeding without on_warmup (multi-asset path)")
                                        warmup_completed = True
                            except Exception as e:
                                logger.error(f"Error in strategy on_warmup: {e}")
                                warmup_completed = True
                        # Generate intents for this symbol
                        if hasattr(self.strategy, 'handle_bar'):
                            intents = list(self.strategy.handle_bar(bar))
                        else:
                            intents = list(self.strategy.generate_intents(bar))
                        if not strategy_active and warmup_completed:
                            strategy_active = True
                            logger.info(f"Strategy became active on {timestamp} after warmup completion")
                        if strategy_active and len(intents) > 0:
                            logger.info(f"First intents generated: {len(intents)} intents at {timestamp}")
                        for intent in intents:
                            if intent.side in [OrderSide.BUY, OrderSide.SELL]:
                                intent.timestamp = timestamp
                                # Mapping compatible with price_data expectations
                                bar_map = {
                                    'open': float(getattr(row, 'open')),
                                    'high': float(getattr(row, 'high')),
                                    'low': float(getattr(row, 'low')),
                                    'close': float(getattr(row, 'close')),
                                    'volume': int(getattr(row, 'volume')) if hasattr(row, 'volume') else 0
                                }
                                self._execute_trade(intent, bar_map)
                    if price_updates:
                        self.portfolio.update_prices(price_updates, timestamp)
                    # Move to next timestamp after processing all symbols
                    continue
                else:
                    # Legacy single-symbol path
                    primary_symbol = universe[0]
                    # Get first row for this timestamp via iloc
                    try:
                        row0 = data.iloc[ts_start]
                    except Exception:
                        row0 = data.loc[timestamp]
                    self.portfolio.update_prices({primary_symbol: row0['close']}, timestamp)
                    bar = Bar(
                        symbol=primary_symbol,
                        timestamp=timestamp,
                        open=row0['open'],
                        high=row0['high'],
                        low=row0['low'],
                        close=row0['close'],
                        volume=row0['volume']
                    )
                
                # Call strategy on_warmup only once at the beginning
                if not warmup_completed and hasattr(self.strategy, 'on_warmup'):
                    try:
                        # Get historical bars for warmup using complete data
                        complete_data = data
                        if hasattr(self.strategy, 'context') and 'complete_data' in self.strategy.context.metadata:
                            complete_data = self.strategy.context.metadata['complete_data']
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug("Using complete data for warmup: %d records", len(complete_data))
                        
                        warmup_bars = self._get_warmup_bars(complete_data, timestamp, primary_symbol)
                        if warmup_bars:  # Check if list is not empty
                            self.strategy.on_warmup(primary_symbol, warmup_bars)
                            warmup_completed = True  # Mark warmup as completed
                            logger.info(f"Strategy warmup completed with {len(warmup_bars)} bars")
                        else:
                            # Multi-frame: allow proceeding without warmup
                            mf_mode = os.getenv('MULTIFRAME_MODE', 'off').lower() in ('1','true','yes','on')
                            if mf_mode:
                                logger.warning(f"Multi-frame: no intraday warmup bars at {timestamp}; proceeding without on_warmup")
                                warmup_completed = True
                            else:
                                logger.warning(f"Insufficient warmup data for {timestamp} - using available data")
                    except Exception as e:
                        logger.error(f"Error in strategy on_warmup: {e}")
                        warmup_completed = True  # Prevent infinite retry
                
                # Generate trading intents using BaseStrategy interface
                # Use handle_bar which includes sizing and risk checks
                if hasattr(self.strategy, 'handle_bar'):
                    # New BaseStrategy interface
                    intents = list(self.strategy.handle_bar(bar))
                else:
                    # Legacy strategy interface
                    intents = list(self.strategy.generate_intents(bar))
                
                # Track strategy activation based on warmup completion
                if not strategy_active and warmup_completed:
                    strategy_active = True
                    logger.info(f"Strategy became active on {timestamp} after warmup completion")
                
                # Log when first intents are generated
                if strategy_active and len(intents) > 0:
                    logger.info(f"First intents generated: {len(intents)} intents at {timestamp}")
                
                # Execute entry trades for each intent
                for intent in intents:
                    if intent.side in [OrderSide.BUY, OrderSide.SELL]:
                        # Override intent timestamp to use exact bar timestamp
                        intent.timestamp = timestamp
                        # Pass the current row (Series) as price_data
                        self._execute_trade(intent, row0)
                
                
                # Get current portfolio value and daily return for logging
                portfolio_value = self.portfolio.get_portfolio_value()
                daily_return = 0.0  # Default for warm-up days
                
                # Log daily summary with warm-up status
                if not strategy_active:
                    warm_up_calendar_days += 1
                    # Count trading days using dias_uteis if available
                    if DIAS_UTEIS_AVAILABLE and dias_uteis.is_du(current_date):
                        warm_up_trading_days += 1
                    
                    portfolio_str = f"R$ {portfolio_value:,.2f}" if portfolio_value is not None else "N/A"
                    return_str = f"{daily_return:.4f}" if daily_return is not None else "N/A"
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "Date: %s, Portfolio Value: %s, Daily Return: %s, Warm-up Calendar Day %d, Trading Day %d (No signals)",
                            current_date, portfolio_str, return_str, warm_up_calendar_days, warm_up_trading_days
                        )
                else:
                    portfolio_str = f"R$ {portfolio_value:,.2f}" if portfolio_value is not None else "N/A"
                    return_str = f"{daily_return:.4f}" if daily_return is not None else "N/A"
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "Date: %s, Portfolio Value: %s, Daily Return: %s, Active Trading",
                            current_date, portfolio_str, return_str
                        )
            
            # Process the final trading day's end-of-day activities
            if current_trading_day is not None:
                self._process_end_of_trading_day(current_trading_day)
            
            # Clear progress line
            try:
                sys.stdout.write("\n")
                sys.stdout.flush()
            except Exception:
                pass

            # Call strategy on_end if available
            if hasattr(self.strategy, 'on_end'):
                try:
                    self.strategy.on_end(data.index[-1])
                    logger.info("Strategy on_end called")
                except Exception as e:
                    logger.error(f"Error in strategy on_end: {e}")
            
            # Record simulation end time
            self.simulation_end_time = datetime.now()
            
            # Log warm-up period summary with both calendar and trading days
            if not strategy_active:
                logger.warning(f"Strategy never became active during simulation - all {len(data)} days were warm-up")
            else:
                if DIAS_UTEIS_AVAILABLE:
                    logger.info(f"Strategy warm-up period: {warm_up_trading_days} trading days ({warm_up_calendar_days} calendar days)")
                    logger.info(f"Active trading period: {len(data) - warm_up_calendar_days} calendar days")
                    logger.info(f"Strategy activation ratio: {(len(data) - warm_up_calendar_days) / len(data):.1%}")
                else:
                    logger.info(f"Strategy warm-up period: {warm_up_calendar_days} calendar days")
                    logger.info(f"Active trading period: {len(data) - warm_up_calendar_days} calendar days")
                    logger.info(f"Strategy activation ratio: {(len(data) - warm_up_calendar_days) / len(data):.1%}")
            
            # Calculate all performance metrics (including benchmark)
            self._calculate_performance_metrics()

            # Build unified fills DataFrame for downstream reporting
            try:
                self.unified_fills_df = pd.DataFrame(self.unified_fills) if self.unified_fills else pd.DataFrame()
            except Exception:
                self.unified_fills_df = pd.DataFrame()
            
            # Create simulation result
            result = self._create_simulation_result()
            
            logger.info("Backtest simulation completed successfully")
            if result.final_portfolio_value is not None:
                logger.info(f"Final portfolio value: R$ {result.final_portfolio_value:,.2f}")
            else:
                logger.info("Final portfolio value: N/A")
            if result.total_return is not None:
                logger.info(f"Total return: {result.total_return:.4f}")
            else:
                logger.info("Total return: N/A")
            if result.sharpe_ratio is not None:
                logger.info(f"Sharpe ratio: {result.sharpe_ratio:.4f}")
            else:
                logger.info("Sharpe ratio: N/A")
            if result.max_drawdown is not None:
                logger.info(f"Max drawdown: {result.max_drawdown:.4f}")
            else:
                logger.info("Max drawdown: N/A")
            
            return result
            
        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            raise
    
    def _get_warmup_bars(self, data: pd.DataFrame, current_date: datetime, symbol: str) -> List[Bar]:
        """
        Get warmup bars for strategy initialization with enhanced validation.
        
        This method ensures sufficient intraday bars are provided to create the required
        number of daily bars for technical indicator calculations (ATR, RSI, EMA).
        
        Args:
            data: Complete market data
            current_date: Current date being processed
            symbol: Trading symbol
            
        Returns:
            List of Bar objects for warmup
        """
        try:
            # Get warmup requirement from strategy (this is intraday bars)
            warmup_bars_required = 20  # Default intraday bars
            if hasattr(self.strategy, 'warmup_bars'):
                warmup_bars_required = self.strategy.warmup_bars
            
            # Calculate intelligent requirement for technical indicators
            # Strategy now uses Brapi.dev daily data directly for indicators
            atr_period = 14  # Default ATR period
            if hasattr(self.strategy, 'atr_period'):
                atr_period = self.strategy.atr_period
            
            # Get strategy's intelligent data requirements if available
            if hasattr(self.strategy, 'data_requirements'):
                intelligent_requirement = self.strategy.data_requirements.get('total_minimum_requirement', warmup_bars_required)
                logger.info(f"Using strategy's intelligent data requirement: {intelligent_requirement} bars")
                warmup_bars_required = max(warmup_bars_required, intelligent_requirement)
            
            # For execution simulation, we need enough intraday bars for realistic backtesting.
            # ATR is computed from DAILY data (not hourly), so we decouple execution warmup from ATR period.
            # Require at least a reasonable number of hourly bars to model fills; default to strategy warmup.
            min_intraday_bars_for_execution = max(60, warmup_bars_required)
            
            # Use the maximum of strategy requirement and calculated execution requirement
            warmup_bars_required = max(warmup_bars_required, min_intraday_bars_for_execution)
            
            logger.info(f"Strategy warmup calculation for {symbol}:")
            logger.info(f"  - Base warmup bars: {warmup_bars_required}")
            logger.info(f"  - ATR period needed: {atr_period} daily bars (from Brapi.dev)")
            logger.info(f"  - Minimum intraday bars for execution simulation: {min_intraday_bars_for_execution} (ATR uses daily data)")
            logger.info(f"  - Final warmup requirement: {warmup_bars_required} intraday bars")
            
            # Always use complete data from strategy context - this contains full historical intraday data
            complete_data = data
            if hasattr(self.strategy, 'context') and 'complete_data' in self.strategy.context.metadata:
                complete_data = self.strategy.context.metadata['complete_data']
                logger.info(f"Using complete historical data: {len(complete_data)} intraday records")
                logger.info(f"Complete data range: {complete_data.index.min()} to {complete_data.index.max()}")
                
                # Log data availability for warmup
                simulation_start = complete_data.index.min()
                if current_date < simulation_start:
                    logger.warning(f"Current date {current_date} is before simulation data start {simulation_start}")
                    logger.info("This may cause warmup warnings but is normal for historical data access")
            else:
                logger.warning(f"No complete_data in strategy context, using provided data: {len(data)} records")
            
            # FIXED APPROACH: Use available historical data for warmup regardless of current date
            # The strategy needs sufficient data for technical indicators, not data "before" current date
            historical_data = complete_data
            
            logger.info(f"Available historical data: {len(historical_data)} bars from {historical_data.index.min()} to {historical_data.index.max()}")
            
            # Calculate required trading days for proper indicator initialization
            required_trading_days = atr_period + 15  # 14 + 15 = 29 trading days minimum
            
            # Use dias_uteis for precise calendar day calculation if available
            if DIAS_UTEIS_AVAILABLE:
                try:
                    # Calculate exact calendar days needed for required trading days
                    estimated_calendar_days = self._calculate_calendar_days_for_trading_days(required_trading_days, current_date)
                    logger.info(f"Using dias_uteis: {required_trading_days} trading days requires ~{estimated_calendar_days} calendar days")
                except Exception as e:
                    logger.warning(f"Error calculating calendar days with dias_uteis: {e}, using fallback")
                    estimated_calendar_days = int(required_trading_days * 1.5)  # Fallback to heuristic
            else:
                # Fallback to heuristic calculation
                estimated_calendar_days = int(required_trading_days * 1.5)  # Account for weekends/holidays
                logger.info(f"Using heuristic: {required_trading_days} trading days requires ~{estimated_calendar_days} calendar days")
            
            # FIXED: Get warmup data from BEFORE the requested start date to avoid look-ahead bias
            # This ensures technical indicators are initialized with historical data only
            if self.start_date:
                # Filter data to only include bars before the simulation start date
                requested_start_dt = self.start_date
                mask = historical_data.index < requested_start_dt
                available_warmup_data = historical_data[mask]
                
                logger.info(f"🔧 Warmup data filtering:")
                logger.info(f"   - Requested simulation start: {requested_start_dt}")
                logger.info(f"   - Available warmup data: {len(available_warmup_data)} bars")
                logger.info(f"   - Warmup data range: {available_warmup_data.index.min() if not available_warmup_data.empty else 'N/A'} to {available_warmup_data.index.max() if not available_warmup_data.empty else 'N/A'}")
                
                if len(available_warmup_data) >= warmup_bars_required:
                    # Take the most recent bars from available warmup data
                    warmup_data = available_warmup_data.tail(warmup_bars_required)
                    logger.info(f"✅ Using {len(warmup_data)} bars from before simulation start date")
                else:
                    # Fallback: use all available warmup data if insufficient
                    warmup_data = available_warmup_data
                    logger.warning(f"⚠️ Insufficient warmup data before start date: {len(available_warmup_data)} < {warmup_bars_required} required")
                    logger.warning(f"   Using all available warmup data: {len(warmup_data)} bars")
            else:
                # Fallback for cases where start_date is not set
                logger.warning("⚠️ No start_date available, using legacy warmup logic")
                warmup_data = historical_data.tail(warmup_bars_required)
            
            # If we need more data for proper warmup, extend the range (but still respect start_date)
            if len(warmup_data) < warmup_bars_required and self.start_date:
                # Try to get more data while still respecting the start_date boundary
                available_warmup_data = historical_data[historical_data.index < self.start_date]
                if len(available_warmup_data) > len(warmup_data):
                    extended_requirement = min(len(available_warmup_data), warmup_bars_required * 2)
                    warmup_data = available_warmup_data.tail(extended_requirement)
                    logger.info(f"📈 Extended warmup data to {len(warmup_data)} bars (still before start date)")
            elif len(warmup_data) < warmup_bars_required and not self.start_date:
                # Legacy fallback
                extended_requirement = min(len(historical_data), warmup_bars_required * 2)
                warmup_data = historical_data.tail(extended_requirement)
            
            actual_bars = len(warmup_data)
            unique_dates = len(set(warmup_data.index.date)) if not warmup_data.empty else 0
            
            logger.info(f"Extracted warmup data: {actual_bars} bars spanning {unique_dates} unique trading days")
            logger.info(f"Date range: {warmup_data.index.min() if not warmup_data.empty else 'N/A'} to {warmup_data.index.max() if not warmup_data.empty else 'N/A'}")
            
            # CRITICAL VALIDATION: Ensure warmup data is before simulation start date
            if self.start_date and not warmup_data.empty:
                warmup_end = warmup_data.index.max()
                if warmup_end >= self.start_date:
                    logger.error(f"❌ CRITICAL ERROR: Warmup data ends after simulation start!")
                    logger.error(f"   - Warmup end: {warmup_end}")
                    logger.error(f"   - Simulation start: {self.start_date}")
                    logger.error(f"   - This creates look-ahead bias and invalidates the backtest!")
                    logger.error(f"   - Please check the warmup logic implementation")
                    return []
                else:
                    logger.info(f"✅ Warmup data validation passed: {warmup_end} < {self.start_date}")
            elif not warmup_data.empty:
                logger.warning("⚠️ Cannot validate warmup data position (no start_date available)")
            
            # Validate we have sufficient unique trading days for execution simulation
            if unique_dates < required_trading_days:
                logger.warning(f"Limited unique trading days: {unique_dates} < {required_trading_days} required")
                logger.warning("ATR is computed from daily data; limited days may reduce stability but will not block execution")
                
                # FIXED: Try to get more data while respecting start_date boundary
                if self.start_date:
                    # Only use data before the simulation start date
                    available_warmup_data = historical_data[historical_data.index < self.start_date]
                    if len(available_warmup_data) > actual_bars:
                        extended_warmup = available_warmup_data.tail(min(len(available_warmup_data), warmup_bars_required * 2))
                        extended_unique_dates = len(set(extended_warmup.index.date))
                        if extended_unique_dates > unique_dates:
                            warmup_data = extended_warmup
                            actual_bars = len(warmup_data)
                            unique_dates = extended_unique_dates
                            logger.info(f"📈 Extended warmup data: {actual_bars} bars spanning {unique_dates} unique days (before start date)")
                else:
                    # Legacy fallback (only if no start_date available)
                    if len(historical_data) > actual_bars:
                        extended_warmup = historical_data.tail(min(len(historical_data), warmup_bars_required * 2))
                        extended_unique_dates = len(set(extended_warmup.index.date))
                        if extended_unique_dates > unique_dates:
                            warmup_data = extended_warmup
                            actual_bars = len(warmup_data)
                            unique_dates = extended_unique_dates
                            logger.warning(f"⚠️ Extended warmup data: {actual_bars} bars spanning {unique_dates} unique days (legacy mode)")
            
            # Final validation with better messaging
            mf_mode = os.getenv('MULTIFRAME_MODE', 'off').lower() in ('1','true','yes','on')
            if actual_bars < min_intraday_bars_for_execution:
                if mf_mode:
                    # In multi-frame, execution can proceed with fewer hourly bars; indicators are computed from daily elsewhere
                    logger.warning(f"Multi-frame mode: proceeding with {actual_bars} hourly warmup bars (<{min_intraday_bars_for_execution})")
                else:
                    logger.error("CRITICAL: Insufficient warmup data for execution simulation!")
                    logger.error(f"  - Available intraday bars: {actual_bars}")
                    logger.error(f"  - Required intraday bars for execution modeling: {min_intraday_bars_for_execution}")
                    logger.error("  - ATR is computed from daily data; this shortfall only impacts execution realism")
                    logger.error("  - Consider extending historical intraday range or lowering warmup bars")
                    # Do not block; proceed with available data to allow order emission and ATR usage
                    # return []
            
            # Validate general warmup requirement
            if actual_bars < warmup_bars_required:
                logger.warning(f"Using available warmup data: {actual_bars} bars (requested: {warmup_bars_required})")
                # In multi-frame, indicators are daily-only; allow proceeding with fewer hourly bars
                if mf_mode:
                    logger.warning(f"Multi-frame mode: continuing despite hourly warmup shortfall ({actual_bars} < {warmup_bars_required})")
                else:
                    # Still proceed if we have enough for execution simulation
                    if actual_bars >= min_intraday_bars_for_execution:
                        logger.info(f"Sufficient for execution simulation, proceeding with {actual_bars} warmup bars")
                    else:
                        logger.error(f"Critical: Insufficient data for reliable execution simulation")
                        return []
            
            # Create Bar objects from intraday data
            bars = []
            for date, row in warmup_data.iterrows():
                bar = Bar(
                    symbol=symbol,
                    timestamp=date,
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume']
                )
                bars.append(bar)
            
            # Log warmup data summary
            unique_dates = warmup_data.index.date
            estimated_trading_days = len(set(unique_dates))
            
            logger.info(f"✓ Warmup bars prepared for {symbol}:")
            logger.info(f"  - Intraday bars provided: {len(bars)}")
            logger.info(f"  - Trading days covered: {estimated_trading_days}")
            logger.info(f"  - ATR period requirement: {atr_period} daily bars (from Brapi.dev)")
            if len(bars) > 0:
                logger.info(f"  - Warmup period: {bars[0].timestamp} to {bars[-1].timestamp}")
            else:
                logger.warning("  - No intraday warmup bars available")
            if self.start_date:
                logger.info(f"  - Simulation start: {self.start_date}")
                if len(bars) > 0:
                    logger.info(f"  - Look-ahead bias: {'NO' if bars[-1].timestamp < self.start_date else 'YES (CRITICAL ERROR)'}")
                else:
                    logger.info("  - Look-ahead bias: NO (no intraday warmup)")
            
            if estimated_trading_days >= atr_period:
                logger.info(f"✓ Sufficient trading days for execution simulation")
            else:
                logger.warning(f"⚠ May have insufficient trading days: {estimated_trading_days} < {atr_period}")
            
            return bars
            
        except Exception as e:
            logger.error(f"Error getting warmup bars for {symbol}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _calculate_calendar_days_for_trading_days(self, required_trading_days: int, end_date: datetime) -> int:
        """
        Calculate exact calendar days needed to get required trading days using dias_uteis.
        
        Args:
            required_trading_days: Number of trading days needed
            end_date: End date for calculation
            
        Returns:
            int: Number of calendar days needed
        """
        if not DIAS_UTEIS_AVAILABLE:
            return int(required_trading_days * 1.5)  # Fallback to heuristic
        
        try:
            calendar_days = 0
            trading_days_found = 0
            current_date = end_date.date()
            
            # Go backwards in time until we find enough trading days
            while trading_days_found < required_trading_days:
                if dias_uteis.is_du(current_date):
                    trading_days_found += 1
                calendar_days += 1
                current_date -= timedelta(days=1)
            
            return calendar_days
            
        except Exception as e:
            logger.error(f"Error calculating calendar days with dias_uteis: {e}")
            return int(required_trading_days * 1.5)  # Fallback to heuristic
    
    def _process_end_of_trading_day(self, trading_day_date: date):
        """
        Process end-of-day activities: strategy end-of-day calls, settlements, portfolio valuation.
        """
        logger.debug(f"Processing end of trading day: {trading_day_date}")
        
        # 1) Generate and execute MOC orders based on actual portfolio positions (Option B)
        try:
            positions_snapshot = dict(self.portfolio.positions) if hasattr(self.portfolio, 'positions') else {}
            for ticker, position in list(positions_snapshot.items()):
                qty = getattr(position, 'quantity', 0)
                if qty == 0:
                    continue
                side = OrderSide.SELL if qty > 0 else OrderSide.BUY
                # Use exact market close timestamp (20:00 UTC final bar for B3)
                close_ts = datetime.combine(trading_day_date, datetime.min.time().replace(hour=20))

                # Determine MOC price from the 20:00 UTC bar (fallback to last bar of day, then current price)
                moc_close_price = None
                moc_bar_series = None
                try:
                    import pandas as pd
                    complete_data = None
                    if hasattr(self.strategy, 'context') and hasattr(self.strategy.context, 'metadata'):
                        complete_data = self.strategy.context.metadata.get('complete_data')
                    df = complete_data
                    if df is not None:
                        # Optional symbol filter if multi-symbol frame
                        if 'symbol' in df.columns or 'ticker' in df.columns:
                            sym_col = 'symbol' if 'symbol' in df.columns else 'ticker'
                            df = df[df[sym_col] == ticker]
                        # Normalize tz-aware index to naive for comparison if needed
                        if getattr(df.index, 'tz', None) is not None:
                            df = df.copy()
                            df.index = df.index.tz_localize(None)
                        # First, try the 20:00 exact bar
                        if close_ts in df.index:
                            row = df.loc[close_ts]
                            if isinstance(row, pd.DataFrame):
                                row = row.iloc[0]
                            moc_bar_series = row
                            moc_close_price = float(row.get('close')) if 'close' in row else None
                        # If not present, use the last available bar on that date
                        if moc_close_price is None:
                            same_day = df[(df.index.date == trading_day_date)]
                            if len(same_day) > 0:
                                last_row = same_day.iloc[-1]
                                moc_bar_series = last_row
                                moc_close_price = float(last_row.get('close')) if 'close' in last_row else None
                except Exception:
                    moc_close_price = None
                    moc_bar_series = None
                moc_intent = OrderIntent(
                    symbol=ticker,
                    side=side,
                    quantity=abs(qty),
                    order_type=OrderType.MARKET,
                    timestamp=close_ts,
                    metadata={'order_type': 'MOC', 'attempt_type': 'moc', 'attempt_name': 'MOC', 'reason': 'End of day position close', 'original_position': qty}
                )
                # Create bar data from the identified close bar; if no close found, skip MOC to avoid unrealistic fills
                import pandas as pd
                current_price_local = None
                if moc_close_price is None:
                    logger.warning(f"No official close available for {ticker} on {trading_day_date}; skipping MOC to avoid price bias")
                    continue
                else:
                    # Build dummy bar using the found bar's OHLC if available
                    if moc_bar_series is not None:
                        dummy_bar_data = pd.Series({
                            'open': float(moc_bar_series.get('open', moc_close_price)),
                            'high': float(moc_bar_series.get('high', moc_close_price)),
                            'low': float(moc_bar_series.get('low', moc_close_price)),
                            'close': moc_close_price,
                            'volume': int(moc_bar_series.get('volume', 0)) if 'volume' in moc_bar_series else 0
                        })
                    else:
                        dummy_bar_data = pd.Series({'open': moc_close_price, 'high': moc_close_price, 'low': moc_close_price, 'close': moc_close_price, 'volume': 0})
                # Execute
                self._execute_trade(moc_intent, dummy_bar_data)
                price_str = f"R$ {moc_intent.price:.2f}" if moc_intent.price is not None else "market"
                logger.info(f"End-of-day closure executed: {moc_intent.side.value} {moc_intent.quantity} {moc_intent.symbol} @ {price_str}")
                # Synchronize strategy state via on_fill for consistency
                if hasattr(self.strategy, 'on_fill'):
                    try:
                        from engine.base_strategy import Fill
                        # Determine a safe fill price for on_fill strictly from close
                        fill_price = moc_close_price if moc_close_price is not None else moc_intent.price
                        fill = Fill(
                            order_id=f"sim_{len(self.trade_log)}",
                            symbol=moc_intent.symbol,
                            side=moc_intent.side,
                            quantity=moc_intent.quantity,
                            price=fill_price,
                            timestamp=moc_intent.timestamp,
                            metadata={'attempt_name': 'MOC', 'attempt_type': 'moc'}
                        )
                        self.strategy.on_fill(fill)
                    except Exception as e:
                        logger.error(f"Error in strategy on_fill for MOC: {e}")
        except Exception as e:
            logger.error(f"Error executing MOC from portfolio positions: {e}")
        
        # 2) Allow strategy to perform end-of-day tasks (e.g., housekeeping, saving history)
        try:
            if hasattr(self.strategy, 'on_end_of_day'):
                list(self.strategy.on_end_of_day(trading_day_date))
        except Exception as e:
            logger.error(f"Error in strategy on_end_of_day: {e}")
        
        # 3) T+0 model: no end-of-day settlement processing
        
        # Record daily portfolio value at end of day
        # Mark-to-market all positions using official close of the trading day to avoid EOD spikes
        try:
            price_updates = {}
            if hasattr(self.strategy, 'context') and hasattr(self.strategy.context, 'metadata'):
                complete_data = self.strategy.context.metadata.get('complete_data')
            else:
                complete_data = None
            if complete_data is not None and hasattr(self.portfolio, 'positions'):
                df = complete_data
                # Normalize tz-aware index if needed
                if getattr(df.index, 'tz', None) is not None:
                    df = df.copy()
                    df.index = df.index.tz_localize(None)
                for sym in list(self.portfolio.positions.keys()):
                    sdf = df[df['symbol'] == sym] if 'symbol' in df.columns else (df[df['ticker'] == sym] if 'ticker' in df.columns else df)
                    same_day = sdf[(sdf.index.date == trading_day_date)]
                    if len(same_day) > 0:
                        price_updates[sym] = float(same_day.iloc[-1].get('close'))
            if price_updates:
                from datetime import datetime as _dt
                self.portfolio.update_prices(price_updates, update_date=_dt.combine(trading_day_date, _dt.min.time()))
        except Exception as _e:
            logger.warning(f"EOD mark-to-market update failed: {_e}")
        portfolio_value = self.portfolio.get_portfolio_value()
        self.daily_portfolio_values.append(portfolio_value)
        
        # Calculate daily return with defensive checks
        if len(self.daily_portfolio_values) > 1 and portfolio_value is not None:
            prev_value = self.daily_portfolio_values[-2]
            if prev_value is not None and prev_value != 0:
                daily_return = (portfolio_value / prev_value) - 1
            else:
                daily_return = 0.0
        else:
            daily_return = 0.0
        
        self.daily_returns.append(daily_return)
        
        # Log end of day summary
        portfolio_str = f"R$ {portfolio_value:,.2f}" if portfolio_value is not None else "N/A"
        return_str = f"{daily_return:.4f} ({daily_return*100:.2f}%)" if daily_return is not None else "N/A"
        logger.info(f"End of day {trading_day_date}: Portfolio Value = {portfolio_str}, Daily Return = {return_str}")
    
    def _create_strategy_context(self, data: pd.DataFrame) -> StrategyContext:
        """
        Create strategy context with all necessary components.
        
        Args:
            data: Market data for the simulation
            
        Returns:
            StrategyContext object
        """
        # Preserve existing metadata if strategy already has it
        existing_metadata = {}
        if hasattr(self.strategy, 'context') and hasattr(self.strategy.context, 'metadata'):
            existing_metadata = self.strategy.context.metadata.copy()
        
        # Preserve existing complete_data if strategy already has it, otherwise use simulation data
        existing_complete_data = existing_metadata.get('complete_data')
        complete_data_for_context = existing_complete_data if existing_complete_data is not None else data
        
        logger.debug(f"Creating strategy context with complete_data: {len(complete_data_for_context)} records")
        
        # Merge existing metadata with simulation metadata, preserving hybrid data result
        metadata = {
            'complete_data': complete_data_for_context,  # Preserve existing complete data or use simulation data
            'simulation_start': self.start_date,
            'simulation_end': self.end_date,
            'initial_capital': self.initial_capital
        }
        
        # Preserve critical hybrid data result for technical indicators
        if 'hybrid_data_result' in existing_metadata:
            metadata['hybrid_data_result'] = existing_metadata['hybrid_data_result']
            logger.info("Preserved hybrid_data_result in strategy context for technical indicators")
        
        # Preserve other important metadata
        for key in ['strategy_config_path', 'data_sources_used', 'hybrid_data_manager']:
            if key in existing_metadata:
                metadata[key] = existing_metadata[key]
        
        # Create context with preserved metadata
        context = StrategyContext(
            data_portal=self.data_portal,
            portfolio=self.portfolio,
            broker=self.broker,
            market_rules=self.market_utils,
            logger=logging.getLogger(f"{self.strategy.__class__.__name__}"),
            metadata=metadata
        )
        
        return context
    
    def _prepare_market_data(self, data: pd.DataFrame, current_timestamp: datetime) -> Dict[str, Any]:
        """
        Prepare market data for strategy signal generation with hourly bar support.
        
        Args:
            data: Complete market data DataFrame
            current_timestamp: Current trading timestamp (hourly)
            
        Returns:
            Dictionary containing market data for strategy
        """
        # Get current hour's bar data using precomputed bounds for speed
        try:
            s_e = self._ts_bounds_map.get(current_timestamp)
            if s_e is None:
                # Fallback to mask
                current_bar_data = data[data.index == current_timestamp]
                if current_bar_data.empty:
                    logger.warning(f"No data found for current timestamp: {current_timestamp}")
                    return {}
                current_bar_ohlcv = current_bar_data.iloc[0]
            else:
                s, e = s_e
                # Use first row in this timestamp block
                current_bar_ohlcv = data.iloc[s]
        except Exception:
            current_bar_data = data[data.index == current_timestamp]
            if current_bar_data.empty:
                logger.warning(f"No data found for current timestamp: {current_timestamp}")
                return {}
            current_bar_ohlcv = current_bar_data.iloc[0]
        
        # Get historical data up to previous bar (to prevent look-ahead bias)
        # Historical data up to previous bar via iloc slicing
        try:
            pos = self._ts_pos_map.get(current_timestamp)
            if pos is None:
                historical_data = data[data.index < current_timestamp].copy()
            else:
                historical_data = data.iloc[:pos]
        except Exception:
            historical_data = data[data.index < current_timestamp].copy()
        
        # Calculate technical indicators using historical data only
        if len(historical_data) >= 20:
            historical_data['sma_20'] = historical_data['close'].rolling(window=20).mean()
            historical_data['sma_50'] = historical_data['close'].rolling(window=50).mean()
            historical_data['volume_sma_20'] = historical_data['volume'].rolling(window=20).mean()
        
        # Load SGS data for current date (use date only, not hour)
        current_date = current_timestamp.date()
        sgs_data = self._load_sgs_data_for_date(current_timestamp)
        
        # Load benchmark (^BVSP) data for current date (use date only, not hour)  
        ibov_data = self._load_ibov_data_for_date(current_timestamp)
        
        # Monitor benchmark data health
        if not ibov_data:
            logger.error(f"Benchmark (^BVSP) data loading failed for {current_date} - strategy performance will be degraded")
        else:
            logger.debug(f"Benchmark (^BVSP) data loaded successfully for {current_date}: {ibov_data['data_points']} data points")
        
        # Calculate SELIC-CDI spread for historical analysis
        selic_cdi_spread = self._calculate_selic_cdi_spread(sgs_data)
        
        # Prepare current bar's OHLCV data for strategy
        current_bar_data_dict = {
            'open': current_bar_ohlcv['open'],
            'high': current_bar_ohlcv['high'],
            'low': current_bar_ohlcv['low'],
            'close': current_bar_ohlcv['close'],
            'volume': current_bar_ohlcv['volume']
        }
        
        # Prepare market data dictionary
        market_data = {
            'price_data': historical_data,  # Historical data only (no look-ahead bias)
            'current_bar_data': current_bar_data_dict,  # Current bar's OHLCV (renamed from current_day_data)
            'current_day_data': current_bar_data_dict,  # Keep for backward compatibility
            'current_price': current_bar_ohlcv['close'],
            'current_volume': current_bar_ohlcv['volume'],
            'timestamp': current_timestamp,  # Full timestamp including hour
            'current_date': current_date,    # Date only for day-level data 
            'sgs_data': sgs_data,  # Add SGS data
            'ibov_data': ibov_data,  # Add IBOV data
            'selic_cdi_spread': selic_cdi_spread,  # Add spread for historical analysis
            'market_conditions': {
                'trend': 'up' if len(historical_data) >= 2 and 
                         historical_data['close'].iloc[-1] > historical_data['close'].iloc[-2] else 'down',
                'volatility': historical_data['close'].pct_change().std() if len(historical_data) > 1 else 0.0,
                'interest_rate_environment': self._classify_interest_rate_environment(sgs_data),
                'inflation_environment': self._classify_inflation_environment(sgs_data)
            }
        }
        
        return market_data
    
    def _load_sgs_data_for_date(self, current_date: datetime) -> Dict[str, float]:
        """
        Load SGS data (interest rates, inflation) for the given date with strict validation.
        Returns a dictionary with keys like 'selic_daily_factor', 'cdi_interest_rate', etc.
        Uses pre-loaded SELIC data when available for better performance.
        """
        try:
            # Initialize SGS loader if not already done
            if not hasattr(self, 'sgs_loader'):
                from engine.sgs_data_loader import SGSDataLoader
                self.sgs_loader = SGSDataLoader()
            
            # Load configuration for strict mode
            config = self._load_config()
            strict_config = config.get('sgs', {}).get('strict_mode', {})
            
            sgs_data = {}
            
            # Use pre-loaded SELIC data if available
            if self.selic_data is not None and not self.selic_data.empty:
                current_date_pd = pd.to_datetime(current_date.date())
                if current_date_pd in self.selic_data.index:
                    row = self.selic_data.loc[current_date_pd]
                    sgs_data['selic_daily_factor'] = row['daily_factor']
                    sgs_data['series_11'] = row['valor']
                    sgs_data['selic_interest_rate'] = row['valor']
                    logger.debug(f"Using pre-loaded SELIC data for {current_date.date()}: {row['valor']:.4f}%")
                else:
                    # Find the closest available date (LOCF)
                    available_dates = self.selic_data.index[self.selic_data.index <= current_date_pd]
                    if len(available_dates) > 0:
                        closest_date = available_dates[-1]
                        row = self.selic_data.loc[closest_date]
                        sgs_data['selic_daily_factor'] = row['daily_factor']
                        sgs_data['series_11'] = row['valor']
                        sgs_data['selic_interest_rate'] = row['valor']
                        logger.debug(f"Using LOCF SELIC data for {current_date.date()} (from {closest_date.date()}): {row['valor']:.4f}%")
                    else:
                        logger.warning(f"No SELIC data available for {current_date.date()}")
            
            # Load other SGS series data (CDI, IPCA) from preloaded data
            for series_id in [12, 433]:  # CDI and IPCA
                if series_id in self.all_sgs_data:
                    try:
                        series_data = self.all_sgs_data[series_id]
                        if series_data is not None and not series_data.empty:
                            current_date_pd = pd.to_datetime(current_date.date())
                            if current_date_pd in series_data.index:
                                row = series_data.loc[current_date_pd]
                                logger.debug(f"Using preloaded {self.sgs_loader.SGS_SERIES[series_id]} data for {current_date.date()}")
                            else:
                                # Find the closest available date (LOCF)
                                available_dates = series_data.index[series_data.index <= current_date_pd]
                                if len(available_dates) > 0:
                                    closest_date = available_dates[-1]
                                    row = series_data.loc[closest_date]
                                    logger.debug(f"Using LOCF {self.sgs_loader.SGS_SERIES[series_id]} data for {current_date.date()} (from {closest_date.date()})")
                                else:
                                    row = None
                                    logger.warning(f"No {self.sgs_loader.SGS_SERIES[series_id]} data available for {current_date.date()}")
                            
                            if row is not None:
                                sgs_data[f'series_{series_id}'] = row['valor']
                                sgs_data[self.sgs_loader.SGS_SERIES[series_id].lower().replace(' ', '_')] = row['valor']
                    except Exception as e:
                        logger.warning(f"Failed to load preloaded SGS series {series_id}: {e}")
                        continue
                else:
                    logger.debug(f"SGS series {series_id} not found in preloaded data, will fallback to per-date loading")
            
            # In strict mode, ensure we have SELIC data
            if strict_config.get('enabled', False) and 'selic_daily_factor' not in sgs_data:
                if strict_config.get('fail_on_missing_data', False):
                    raise SELICDataUnavailableError(f"No SELIC data available for {current_date.date()}")
                else:
                    logger.warning(f"No SELIC data available for {current_date.date()}, using fallback")
            
            return sgs_data
        except (SELICDataUnavailableError, SELICDataInsufficientError, SELICDataQualityError, SELICDataValidationError) as e:
            strict_config = config.get('sgs', {}).get('strict_mode', {})
            # Allow override via selic.strict_mode: false
            try:
                selic_override = (config.get('selic', {}) or {}).get('strict_mode', None)
            except Exception:
                selic_override = None
            if selic_override is False:
                logger.warning(f"SELIC strict-mode override active; skipping SELIC for {current_date.date()}: {e}")
                return {}
            if strict_config.get('fail_on_missing_data', False):
                logger.error(f"Critical SELIC data issue: {e}")
                raise RuntimeError(f"Backtest cannot proceed due to SELIC data issues: {e}")
            else:
                logger.error(f"SELIC data issue (non-strict mode): {e}")
                return {}
        except Exception as e:
            logger.error(f"Error loading SGS data: {e}")
            return {}
    
    def _load_ibov_data_for_date(self, current_date: datetime) -> Dict[str, Any]:
        """
        Load IBOV data for the given date.
        Returns a dictionary with IBOV data including returns.
        """
        try:
            # Load IBOV data once and cache (avoid per-bar CSV read)
            ibov_file = Path("data/IBOV/IBOV_raw.csv")
            if self._ibov_df_cached is None:
                # Load benchmark from BRAPI daily if CSV not present
                ibov_df = None
                if ibov_file.exists():
                    try:
                        ibov_df = pd.read_csv(ibov_file, index_col=0)
                    except Exception:
                        ibov_df = None
                if ibov_df is None or ibov_df.empty:
                    try:
                        from .brapi_provider import BrapiProvider
                        end_dt = current_date
                        start_dt = end_dt - timedelta(days=365)
                        # Defensive local import to avoid NameError in alternate import contexts
                        try:
                            import os as _os
                            token = _os.getenv('BRAPI_API_TOKEN', '')
                        except Exception:
                            token = ''
                        brapi = BrapiProvider(api_token=token, cache_dir="data/brapi_cache")
                        # BRAPI index ticker mapping: IBOV -> ^BVSP
                        ibov_df = brapi.get_daily_data('^BVSP', start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d'))
                    except Exception as _e:
                        logger.error(f"Failed to load IBOV via BRAPI: {_e}")
                        return {}
                if ibov_df is None or ibov_df.empty:
                    logger.error("IBOV data unavailable")
                    return {}
                # Normalize time index once
                ibov_df.index = pd.to_datetime(ibov_df.index, utc=True).tz_localize(None)
                ibov_df.index = ibov_df.index.normalize()
                self._ibov_df_cached = ibov_df
            else:
                ibov_df = self._ibov_df_cached
            
            # Validate required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in ibov_df.columns]
            if missing_columns:
                logger.error(f"IBOV data missing required columns: {missing_columns}")
                return {}
            
            
            # Ensure current_date is timezone-naive
            if current_date.tzinfo is not None:
                current_date = current_date.replace(tzinfo=None)
            
            # Filter data up to current date
            ibov_data_filtered = ibov_df[ibov_df.index <= current_date]
            
            if ibov_data_filtered.empty:
                logger.error(f"No IBOV data available for date {current_date.date()}")
                logger.debug(f"Available IBOV date range: {ibov_df.index.min()} to {ibov_df.index.max()}")
                return {}
            
            # Calculate returns
            ibov_returns = ibov_data_filtered['close'].pct_change()
            
            # Validate returns data
            if ibov_returns.empty or ibov_returns.isna().all():
                logger.error("IBOV returns calculation failed - all values are NaN")
                return {}
            
            # Get current IBOV value
            current_ibov_value = ibov_data_filtered['close'].iloc[-1]
            
            # Prepare IBOV data dictionary
            ibov_data = {
                'value': current_ibov_value,
                'returns': ibov_returns,
                'close': ibov_data_filtered['close'],
                'volume': ibov_data_filtered['volume'],
                'data_points': len(ibov_data_filtered),
                'date_range': {
                    'start': ibov_data_filtered.index.min().isoformat(),
                    'end': ibov_data_filtered.index.max().isoformat()
                }
            }
            
            logger.debug(f"Loaded IBOV data for {current_date.date()}: value={current_ibov_value:.2f}, data_points={len(ibov_data_filtered)}")
            return ibov_data
            
        except Exception as e:
            logger.error(f"Error loading IBOV data for {current_date.date()}: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return {}
    
    def _calculate_selic_cdi_spread(self, sgs_data: Dict[str, float]) -> Optional[float]:
        """
        Calculate the SELIC-CDI spread for historical analysis.
        
        This spread was historically significant (≈ 0,06 p.p. ao ano) but has
        converged to zero since 2019 due to infrastructure integration.
        
        Args:
            sgs_data: Dictionary with SGS data
            
        Returns:
            Spread value (SELIC - CDI) or None if data unavailable
        """
        selic_rate = sgs_data.get('selic_interest_rate')
        cdi_rate = sgs_data.get('cdi_interest_rate')
        
        if selic_rate is not None and cdi_rate is not None:
            return selic_rate - cdi_rate
        
        return None
    
    def _classify_interest_rate_environment(self, sgs_data: Dict[str, float]) -> str:
        """
        Classify the current interest rate environment.
        
        Note: SELIC (series 11) and CDI (series 12) have evolved over time:
        - 2015-2017: Real differences (CDI slightly higher than SELIC)
        - 2018: Gradual convergence due to infrastructure integration
        - 2019+: Perfect convergence (Banco Central fills series 12 with SELIC)
        
        For 15-year historical analysis, maintaining both series is important
        to preserve historical spread data and enable spread-based strategies.
        
        Args:
            sgs_data: Dictionary with SGS data
            
        Returns:
            String classification of interest rate environment
        """
        selic_rate = sgs_data.get('selic_interest_rate')
        cdi_rate = sgs_data.get('cdi_interest_rate')
        
        # Use SELIC as primary indicator (official rate)
        # CDI is maintained for historical spread analysis
        primary_rate = selic_rate if selic_rate is not None else cdi_rate
        
        if primary_rate is None:
            return 'unknown'
        
        # Classify based on current Brazilian market conditions
        if primary_rate >= 12.0:
            return 'high_rates'  # High interest rate environment
        elif primary_rate >= 8.0:
            return 'moderate_rates'  # Moderate interest rate environment
        elif primary_rate >= 4.0:
            return 'low_rates'  # Low interest rate environment
        else:
            return 'very_low_rates'  # Very low interest rate environment
    
    def _classify_inflation_environment(self, sgs_data: Dict[str, float]) -> str:
        """
        Classify the current inflation environment.
        
        Args:
            sgs_data: Dictionary with SGS data
            
        Returns:
            String classification of inflation environment
        """
        ipca_rate = sgs_data.get('ipca_inflation_index')
        
        if ipca_rate is None:
            return 'unknown'
        
        # Classify based on Brazilian inflation targets
        if ipca_rate >= 6.0:
            return 'high_inflation'  # High inflation environment
        elif ipca_rate >= 4.5:
            return 'above_target'  # Above target inflation
        elif ipca_rate >= 2.5:
            return 'target_range'  # Within target range
        elif ipca_rate >= 1.5:
            return 'below_target'  # Below target inflation
        else:
            return 'very_low_inflation'  # Very low inflation environment
    
    def _execute_trade(self, signal: OrderIntent, price_data: pd.Series) -> None:
        """
        Execute individual trade based on signal.
        
        Args:
            signal: Trading signal to execute
            price_data: Current day's price data
        """
        try:
            # High-signal debug: summarize incoming order
            try:
                logger.info(
                    f"EXECUTE intent: sym={getattr(signal,'symbol',None)} side={getattr(signal,'side',None)} "
                    f"type={getattr(signal,'order_type',None)} qty={getattr(signal,'quantity',None)} "
                    f"meta.entry_leg={getattr(getattr(signal,'metadata',{}),'get',lambda k:None)('entry_leg')} "
                    f"meta.attempt={getattr(getattr(signal,'metadata',{}),'get',lambda k:None)('attempt_type')}"
                )
            except Exception:
                pass
            # Skip redundant market data validation for strategies that validate during signal generation
            # (e.g., FuzzyFajuto strategy already validates market data including IBOV data in generate_signals)
            skip_validation = (
                hasattr(self.strategy, 'strategy_name') and 
                (self.strategy.strategy_name == 'FuzzyFajuto' or 'fuzzy' in self.strategy.strategy_name.lower())
            )
            
            # Validate market data (optional method) - skip for strategies that already validate
            if hasattr(self.strategy, 'validate_market_data') and not skip_validation:
                try:
                    market_data_valid = self.strategy.validate_market_data({
                        'price_data': pd.DataFrame([price_data]),
                        'timestamp': signal.timestamp
                    })
                    if not market_data_valid:
                        logger.warning(f"Invalid market data for signal: {signal}")
                        return
                except Exception as e:
                    logger.error(f"Error in validate_market_data: {str(e)}")
                    return
            elif hasattr(self.strategy, 'validate_market_data') and skip_validation:
                logger.debug("Skipping redundant market data validation for strategy that validates during signal generation")
            else:
                logger.debug("Strategy does not implement validate_market_data - skipping validation")
            
            # Check Brazilian market constraints (now available by default)
            try:
                constraints_ok = self.strategy.check_brazilian_market_constraints(signal)
                if not constraints_ok:
                    logger.warning(f"Signal violates Brazilian market constraints: {signal}")
                    return
            except Exception as e:
                logger.error(f"Error in check_brazilian_market_constraints: {str(e)}")
                return
            
            # T+0 model: get current available cash directly from portfolio
            available_cash = getattr(self.portfolio, 'cash', 0.0)
            
            # Use strategy's position sizing (from signal)
            quantity = signal.quantity
            
            # Optional: Apply risk management if strategy provides it
            if hasattr(self.strategy, 'calculate_position_size'):
                try:
                    risk_adjusted_quantity = self.strategy.calculate_position_size(signal, available_cash)
                    # Use the smaller of strategy's choice or risk-adjusted
                    quantity = min(quantity, risk_adjusted_quantity)
                    logger.debug(f"Strategy wanted {signal.quantity}, risk allows {risk_adjusted_quantity}, using {quantity}")
                except Exception as e:
                    logger.warning(f"Error in calculate_position_size: {str(e)}, using strategy's quantity")
            
            # Final validation (single guard)
            if quantity <= 0:
                logger.warning(
                    f"quantity==0 after sizing: sym={getattr(signal,'symbol',None)} side={getattr(signal,'side',None)} "
                    f"order_type={getattr(signal,'order_type',None)} requested={getattr(signal,'quantity',None)} available_cash={available_cash}"
                )
                return
            
            # Execute trade based on signal type
            # Map OrderIntent attributes to simulator expectations
            ticker = getattr(signal, 'ticker', signal.symbol)  # Use ticker if available, fallback to symbol
            signal_type = getattr(signal, 'signal_type', signal.side)  # Use signal_type if available, fallback to side
            trade_type = getattr(signal, 'trade_type', signal.order_type.value)  # Use trade_type if available, fallback to order_type
            price = signal.price if signal.price is not None else 0.0  # Handle None price
            
            # Normalize trade type to expected format
            if isinstance(trade_type, str):
                if trade_type.lower() in ['market', 'limit']:
                    trade_type = 'day_trade'  # Default to day trade for market/limit orders
                elif trade_type.lower() not in ['day_trade', 'swing_trade', 'auto']:
                    trade_type = 'day_trade'  # Default to day trade for unknown types
            else:
                trade_type = 'day_trade'  # Default to day trade for non-string types
            
            # Handle market orders (ensure fill uses OPEN for first-bar market)
            if signal.order_type == OrderType.MARKET or price == 0.0:
                # Prefer exact execution price provided by strategy for first-bar market orders
                execution_price_from_metadata = None
                try:
                    if hasattr(signal, 'metadata') and isinstance(signal.metadata, dict):
                        attempt_type = signal.metadata.get('attempt_type')
                        emission_type = signal.metadata.get('emission_type')
                        if (attempt_type == 'market') or (emission_type == 'first_bar'):
                            execution_price_from_metadata = signal.metadata.get('execution_price')
                except Exception:
                    execution_price_from_metadata = None
                
                if execution_price_from_metadata is not None and execution_price_from_metadata > 0:
                    price = execution_price_from_metadata
                else:
                    # For first bar of the day, use the bar OPEN instead of CLOSE to avoid slippage
                    # Keep current behavior for non-first-bar market orders (e.g., MOC dummy bars)
                    is_first_bar_market = False
                    try:
                        if hasattr(signal, 'metadata') and isinstance(signal.metadata, dict):
                            attempt_type = signal.metadata.get('attempt_type')
                            emission_type = signal.metadata.get('emission_type')
                            is_first_bar_market = (attempt_type == 'market') or (emission_type == 'first_bar')
                    except Exception:
                        is_first_bar_market = False
                    
                    if is_first_bar_market:
                        # Use OPEN to honor market-at-open semantics
                        price = price_data.get('open', 0.0) if price_data is not None else 0.0
                    else:
                        # Default to CLOSE for other market orders (e.g., MOC)
                        price = price_data.get('close', 0.0) if price_data is not None else 0.0
                
                if price <= 0.0:
                    logger.warning(
                        f"Cannot determine market price: will skip. sym={ticker} open={price_data.get('open',None)} close={price_data.get('close',None)}"
                    )
                    return
            
            if signal_type == OrderSide.BUY or signal_type == SignalType.BUY:
                success = self.portfolio.buy(
                    ticker=ticker,
                    quantity=quantity,
                    price=price,
                    trade_date=signal.timestamp,
                    trade_type=trade_type if isinstance(trade_type, str) else trade_type.value,
                    description=f"Strategy signal: {signal_type.value if hasattr(signal_type, 'value') else str(signal_type)}"
                )
                
                if success:
                    if price is not None and price > 0:
                        price_str = f"R$ {price:.2f}"
                    else:
                        price_str = "market"
                    logger.info(f"Buy executed: {quantity} {ticker} @ {price_str}")
                    # Append unified fill row (BUY)
                    try:
                        self._append_unified_fill(
                            timestamp=signal.timestamp,
                            symbol=ticker,
                            side='BUY',
                            quantity=quantity,
                            price=price,
                            metadata=getattr(signal, 'metadata', {})
                        )
                    except Exception:
                        pass
                else:
                    if price is not None and price > 0:
                        price_str = f"R$ {price:.2f}"
                    else:
                        price_str = "market"
                    logger.warning(f"Buy failed: {quantity} {ticker} @ {price_str}")
            
            elif signal_type == OrderSide.SELL or signal_type == SignalType.SELL:
                # Check if we have position to sell
                if ticker not in self.portfolio.positions:
                    logger.warning(f"No position in {ticker} to sell")
                    return
                
                position = self.portfolio.positions[ticker]
                sell_quantity = min(quantity, position.quantity)
                
                if sell_quantity <= 0:
                    logger.warning(f"No shares available to sell in {ticker}")
                    return
                
                success = self.portfolio.sell(
                    ticker=ticker,
                    quantity=sell_quantity,
                    price=price,
                    trade_date=signal.timestamp,
                    trade_type=trade_type if isinstance(trade_type, str) else trade_type.value,
                    description=f"Strategy signal: {signal_type.value if hasattr(signal_type, 'value') else str(signal_type)}"
                )
                
                if success:
                    if price is not None and price > 0:
                        price_str = f"R$ {price:.2f}"
                    else:
                        price_str = "market"
                    logger.info(f"Sell executed: {sell_quantity} {ticker} @ {price_str}")
                    # Append unified fill row (SELL)
                    try:
                        self._append_unified_fill(
                            timestamp=signal.timestamp,
                            symbol=ticker,
                            side='SELL',
                            quantity=sell_quantity,
                            price=price,
                            metadata=getattr(signal, 'metadata', {})
                        )
                    except Exception:
                        pass
                else:
                    if price is not None and price > 0:
                        price_str = f"R$ {price:.2f}"
                    else:
                        price_str = "market"
                    logger.warning(f"Sell failed: {sell_quantity} {ticker} @ {price_str}")

            # Notify strategy on successful fills (authoritative, with executed price and metadata)
            try:
                if hasattr(self.strategy, 'on_fill'):
                    from engine.base_strategy import Fill
                    if (signal_type == OrderSide.BUY or signal_type == SignalType.BUY) and success:
                        fill = Fill(
                            order_id=f"sim_{len(self.trade_log)}",
                            symbol=ticker,
                            side=OrderSide.BUY,
                            quantity=quantity,
                            price=price,
                            timestamp=signal.timestamp,
                            metadata=getattr(signal, 'metadata', {})
                        )
                        self.strategy.on_fill(fill)
                    elif (signal_type == OrderSide.SELL or signal_type == SignalType.SELL) and success:
                        fill = Fill(
                            order_id=f"sim_{len(self.trade_log)}",
                            symbol=ticker,
                            side=OrderSide.SELL,
                            quantity=sell_quantity,
                            price=price,
                            timestamp=signal.timestamp,
                            metadata=getattr(signal, 'metadata', {})
                        )
                        self.strategy.on_fill(fill)
            except Exception as e:
                logger.error(f"Error in strategy on_fill notification: {str(e)}")
            
            # Record trade in log
            confidence = getattr(signal, 'confidence', 1.0)  # Default confidence if not available
            trade_record = {
                'date': signal.timestamp,
                'ticker': ticker,
                'signal_type': signal_type.value if hasattr(signal_type, 'value') else str(signal_type),
                'quantity': quantity,
                'price': price,
                'trade_type': trade_type,  # Already normalized above
                'confidence': confidence,
                'portfolio_value': self.portfolio.get_portfolio_value()
            }
            
            self.trade_log.append(trade_record)
            
        except Exception as e:
            logger.error(f"Error executing trade for signal {signal}: {str(e)}")

    def _append_unified_fill(self, timestamp: datetime, symbol: str, side: str,
                              quantity: int, price: float, metadata: Dict[str, Any]) -> None:
        """Append a normalized unified fill row capturing execution details.

        Columns: timestamp, symbol, side, quantity, price, lot_type, rounding,
        tranche_notional_brl, trade_type, order_type, attempt_type, attempt_name.
        """
        try:
            # Lot typing and rounding delta (100-share round-lot)
            lot_multiple = 100
            is_round = (int(quantity) % lot_multiple == 0)
            lot_type = 'round_lot' if is_round else 'odd_lot'
            rounding_delta = int(quantity) % lot_multiple

            # Determine order_type/attempts
            order_type_txt = None
            attempt_type = None
            attempt_name = None
            emission_type = None
            if isinstance(metadata, dict):
                order_type_txt = str(metadata.get('order_type') or metadata.get('OrderType') or '').upper() or None
                attempt_type = metadata.get('attempt_type')
                attempt_name = metadata.get('attempt_name')
                emission_type = metadata.get('emission_type')
            # Trade type (day/swing) best-effort
            trade_type_txt = 'day_trade'
            if isinstance(metadata, dict) and isinstance(metadata.get('trade_type'), str):
                trade_type_txt = metadata.get('trade_type')

            # Tranche notional from centralized config/strategy context
            tranche_notional = self._get_tranche_notional_brl()

            row = {
                'timestamp': timestamp,
                'ticker': symbol,
                'symbol': symbol,
                'side': side.upper(),
                'quantity': int(quantity),
                'price': float(price),
                'lot_type': lot_type,
                'rounding': int(rounding_delta),
                'tranche_notional_brl': float(tranche_notional),
                'trade_type': trade_type_txt,
                'order_type': (order_type_txt or ('MOC' if (attempt_type == 'moc') else ('MARKET' if emission_type in ('first_bar', 'open_market') else 'LIMIT'))),
                'attempt_type': attempt_type,
                'attempt_name': attempt_name,
            }
            self.unified_fills.append(row)
        except Exception as _:
            # Do not let reporting impact execution
            pass

    def _get_tranche_notional_brl(self) -> float:
        """Resolve tranche_notional_brl from strategy context/config with safe defaults."""
        try:
            # Strategy context metadata first
            if hasattr(self.strategy, 'context') and hasattr(self.strategy.context, 'metadata'):
                md = self.strategy.context.metadata
                if isinstance(md, dict):
                    t = md.get('tranche_notional_brl')
                    if t is not None:
                        return float(t)
                    cfg = md.get('config') or {}
                    pair_cfg = (cfg.get('pair_mode') or {}) if isinstance(cfg, dict) else {}
                    gross = float(pair_cfg.get('gross_exposure_brl', 50000.0))
                    tranches = int(pair_cfg.get('tranches', 4) or 4)
                    if tranches > 0:
                        return float(gross / tranches)
        except Exception:
            pass
        try:
            # Fallback to simulator config
            pair_cfg = (self.config.get('pair_mode') or {}) if hasattr(self, 'config') else {}
            gross = float(pair_cfg.get('gross_exposure_brl', 50000.0))
            tranches = int(pair_cfg.get('tranches', 4) or 4)
            if tranches > 0:
                return float(gross / tranches)
        except Exception:
            pass
        return 10000.0
    
    def _calculate_performance_metrics(self) -> None:
        """Calculate comprehensive performance metrics including benchmark analysis."""
        if not self.daily_portfolio_values:
            logger.warning("No portfolio values available for performance calculation")
            return
        
        # Use the integrated performance metrics module
        all_metrics = self.performance_metrics.calculate_all_metrics(
            portfolio_values=self.daily_portfolio_values,
            daily_returns=self.daily_returns,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Update the legacy performance metrics for backward compatibility with safe defaults
        self.performance_metrics.total_return = all_metrics.get('total_return', 0.0)
        self.performance_metrics.annualized_return = all_metrics.get('annualized_return', 0.0)
        self.performance_metrics.sharpe_ratio = all_metrics.get('sharpe_ratio', 0.0)
        self.performance_metrics.max_drawdown = all_metrics.get('max_drawdown', 0.0)
        self.performance_metrics.total_trades = all_metrics.get('total_trades', 0)
        self.performance_metrics.winning_trades = all_metrics.get('winning_trades', 0)
        self.performance_metrics.losing_trades = all_metrics.get('losing_trades', 0)
        self.performance_metrics.total_commission = all_metrics.get('total_commission', 0.0)
        self.performance_metrics.total_taxes = self.portfolio.total_taxes
        self.performance_metrics.final_portfolio_value = self.daily_portfolio_values[-1]
        self.performance_metrics.initial_capital = self.daily_portfolio_values[0]
        self.performance_metrics.net_profit = self.daily_portfolio_values[-1] - self.daily_portfolio_values[0]
        
        # Add logging to track Sharpe ratio update in simulator
        sharpe_ratio = all_metrics.get('sharpe_ratio')
        if sharpe_ratio is not None:
            sharpe_str = f"{sharpe_ratio:.8f}"
        else:
            sharpe_str = "N/A"
        logger.info(f"SIMULATOR Updated performance_metrics.sharpe_ratio to: {sharpe_str}")
        logger.info(f"SIMULATOR Sharpe ratio from all_metrics: {sharpe_str}")
        
        # Store benchmark metrics for simulation result
        self.benchmark_metrics = all_metrics
        
        # Generate comprehensive performance reports
        try:
            comprehensive_analysis = ComprehensivePerformanceAnalysis(self.portfolio, strategy=self.strategy)
            
            # Run comprehensive analysis first
            analysis_results = comprehensive_analysis.run_comprehensive_analysis(
                portfolio_values=self.daily_portfolio_values,
                daily_returns=self.daily_returns,
                start_date=self.start_date
            )
            
            # Generate timestamp for unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_name = self.strategy.__class__.__name__.lower()
            
            # Generate JSON report
            import os as _os
            if not (_os.getenv('AUDIT_EXECUTIONS_ONLY', '1').lower() in ('1', 'true', 'yes')):
                json_report_path = f"reports/{strategy_name}_performance_report_{timestamp}.json"
                comprehensive_analysis.generate_performance_report(
                    analysis_results=analysis_results,
                    output_path=json_report_path
                )
                logger.info(f"Comprehensive performance reports generated: {json_report_path}")

                # Generate HTML report
                html_content = comprehensive_analysis.generate_html_report(
                    portfolio_values=self.daily_portfolio_values,
                    daily_returns=self.daily_returns,
                    start_date=self.start_date,
                    strategy_name=self.strategy.__class__.__name__
                )

                html_report_path = f"reports/{strategy_name}_performance_report_{timestamp}.html"
                with open(html_report_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                logger.info(f"HTML performance report generated: {html_report_path}")

                # Generate performance charts
                charts_path = f"reports/{strategy_name}_performance_charts_{timestamp}.png"
                comprehensive_analysis.plot_performance_charts(
                portfolio_values=self.daily_portfolio_values,
                daily_returns=self.daily_returns,
                output_path=charts_path
                )
                logger.info(f"Performance charts generated: {charts_path}")
            
        except Exception as e:
            logger.warning(f"Failed to generate comprehensive reports: {e}")
        
        logger.info("All performance metrics (including benchmark) calculated successfully")
    
    def _create_simulation_result(self) -> SimulationResult:
        """
        Create comprehensive simulation result.
        
        Returns:
            SimulationResult with all performance data
        """
        if not self.simulation_start_time or not self.simulation_end_time:
            raise RuntimeError("Simulation start/end times not recorded")
        
        simulation_duration = (
            self.simulation_end_time - self.simulation_start_time
        ).total_seconds()
        
        # Prepare benchmark metrics from integrated performance metrics
        if self.benchmark_metrics is not None:
            benchmark_return = self.benchmark_metrics.get('benchmark_return', 0.0)
            excess_return = self.benchmark_metrics.get('excess_return', 0.0)
            information_ratio = self.benchmark_metrics.get('information_ratio', 0.0)
            beta = self.benchmark_metrics.get('beta', 0.0)
            alpha = self.benchmark_metrics.get('alpha', 0.0)
            tracking_error = self.benchmark_metrics.get('tracking_error', 0.0)
            rolling_correlation = self.benchmark_metrics.get('rolling_correlation', 0.0)
            benchmark_sharpe = self.benchmark_metrics.get('benchmark_sharpe', 0.0)
            benchmark_max_drawdown = self.benchmark_metrics.get('benchmark_max_drawdown', 0.0)
            benchmark_win_rate = self.benchmark_metrics.get('benchmark_win_rate', 0.0)
        else:
            logger.warning("No benchmark metrics available, using default values")
            benchmark_return = 0.0
            excess_return = 0.0
            information_ratio = 0.0
            beta = 0.0
            alpha = 0.0
            tracking_error = 0.0
            rolling_correlation = 0.0
            benchmark_sharpe = 0.0
            benchmark_max_drawdown = 0.0
            benchmark_win_rate = 0.0
        
        # Add defensive checks for performance metrics to prevent None values
        total_return = self.performance_metrics.total_return if self.performance_metrics.total_return is not None else 0.0
        annualized_return = self.performance_metrics.annualized_return if self.performance_metrics.annualized_return is not None else 0.0
        sharpe_ratio = self.performance_metrics.sharpe_ratio if self.performance_metrics.sharpe_ratio is not None else 0.0
        max_drawdown_value = self.performance_metrics.max_drawdown if self.performance_metrics.max_drawdown is not None else 0.0
        win_loss_ratio = self.performance_metrics.win_loss_ratio if self.performance_metrics.win_loss_ratio is not None else 0.0
        profit_factor = self.performance_metrics.profit_factor if self.performance_metrics.profit_factor is not None else 0.0
        total_trades = self.performance_metrics.total_trades if self.performance_metrics.total_trades is not None else 0
        winning_trades = self.performance_metrics.winning_trades if self.performance_metrics.winning_trades is not None else 0
        losing_trades = self.performance_metrics.losing_trades if self.performance_metrics.losing_trades is not None else 0
        avg_win = self.performance_metrics.avg_win if self.performance_metrics.avg_win is not None else 0.0
        avg_loss = self.performance_metrics.avg_loss if self.performance_metrics.avg_loss is not None else 0.0
        largest_win = self.performance_metrics.largest_win if self.performance_metrics.largest_win is not None else 0.0
        largest_loss = self.performance_metrics.largest_loss if self.performance_metrics.largest_loss is not None else 0.0
        total_commission = self.performance_metrics.total_commission if self.performance_metrics.total_commission is not None else 0.0
        total_taxes = self.performance_metrics.total_taxes if self.performance_metrics.total_taxes is not None else 0.0
        net_profit = self.performance_metrics.net_profit if self.performance_metrics.net_profit is not None else 0.0
        final_portfolio_value = self.performance_metrics.final_portfolio_value if self.performance_metrics.final_portfolio_value is not None else 0.0
        initial_capital = self.performance_metrics.initial_capital if self.performance_metrics.initial_capital is not None else 0.0
        
        logger.info(f"PERFORMANCE_SUMMARY Max drawdown from performance_metrics: {max_drawdown_value}")
        logger.info(f"PERFORMANCE_SUMMARY Max drawdown type: {type(max_drawdown_value)}")
        
        summary = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown_value,
            'win_loss_ratio': win_loss_ratio,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'total_commission': total_commission,
            'total_taxes': total_taxes,
            'net_profit': net_profit,
            'final_portfolio_value': final_portfolio_value,
            'initial_capital': initial_capital
        }
        
        # Add benchmark metrics from integrated performance metrics
        if self.benchmark_metrics is not None:
            summary.update({
                'benchmark_return': self.benchmark_metrics.get('benchmark_return', 0.0),
                'excess_return': self.benchmark_metrics.get('excess_return', 0.0),
                'information_ratio': self.benchmark_metrics.get('information_ratio', 0.0),
                'beta': self.benchmark_metrics.get('beta', 0.0),
                'alpha': self.benchmark_metrics.get('alpha', 0.0),
                'tracking_error': self.benchmark_metrics.get('tracking_error', 0.0),
                'rolling_correlation': self.benchmark_metrics.get('rolling_correlation', 0.0),
                'benchmark_sharpe': self.benchmark_metrics.get('benchmark_sharpe', 0.0),
                'benchmark_max_drawdown': self.benchmark_metrics.get('benchmark_max_drawdown', 0.0),
                'benchmark_win_rate': self.benchmark_metrics.get('benchmark_win_rate', 0.0),
                'benchmark_symbol': self.benchmark_metrics.get('benchmark_symbol', 'IBOV')
            })
        
        return SimulationResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown_value,
            win_loss_ratio=win_loss_ratio,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            final_portfolio_value=final_portfolio_value,
            initial_capital=initial_capital,
            total_commission=total_commission,
            total_taxes=total_taxes,
            daily_returns=self.daily_returns.copy(),
            portfolio_values=self.daily_portfolio_values.copy(),
            trade_log=self.trade_log.copy(),
            simulation_duration=simulation_duration,
            start_date=self.simulation_start_time,
            end_date=self.simulation_end_time,
            # Benchmark metrics
            benchmark_return=benchmark_return,
            excess_return=excess_return,
            information_ratio=information_ratio,
            beta=beta,
            alpha=alpha,
            tracking_error=tracking_error,
            rolling_correlation=rolling_correlation,
            benchmark_sharpe=benchmark_sharpe,
            benchmark_max_drawdown=benchmark_max_drawdown,
            benchmark_win_rate=benchmark_win_rate
        )

    def get_unified_fills_dataframe(self) -> pd.DataFrame:
        """Return unified fills as a DataFrame (cached)."""
        if self.unified_fills_df is None:
            try:
                self.unified_fills_df = pd.DataFrame(self.unified_fills) if self.unified_fills else pd.DataFrame(
                    columns=['timestamp','symbol','side','quantity','price','lot_type','rounding','tranche_notional_brl','trade_type','order_type','attempt_type','attempt_name']
                )
            except Exception:
                self.unified_fills_df = pd.DataFrame()
        return self.unified_fills_df
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Returns:
            Dictionary containing all performance metrics
        """
        # Add defensive checks for performance metrics to prevent None values
        total_return = self.performance_metrics.total_return if self.performance_metrics.total_return is not None else 0.0
        annualized_return = self.performance_metrics.annualized_return if self.performance_metrics.annualized_return is not None else 0.0
        sharpe_ratio = self.performance_metrics.sharpe_ratio if self.performance_metrics.sharpe_ratio is not None else 0.0
        max_drawdown_value = self.performance_metrics.max_drawdown if self.performance_metrics.max_drawdown is not None else 0.0
        win_loss_ratio = self.performance_metrics.win_loss_ratio if self.performance_metrics.win_loss_ratio is not None else 0.0
        profit_factor = self.performance_metrics.profit_factor if self.performance_metrics.profit_factor is not None else 0.0
        total_trades = self.performance_metrics.total_trades if self.performance_metrics.total_trades is not None else 0
        winning_trades = self.performance_metrics.winning_trades if self.performance_metrics.winning_trades is not None else 0
        losing_trades = self.performance_metrics.losing_trades if self.performance_metrics.losing_trades is not None else 0
        avg_win = self.performance_metrics.avg_win if self.performance_metrics.avg_win is not None else 0.0
        avg_loss = self.performance_metrics.avg_loss if self.performance_metrics.avg_loss is not None else 0.0
        largest_win = self.performance_metrics.largest_win if self.performance_metrics.largest_win is not None else 0.0
        largest_loss = self.performance_metrics.largest_loss if self.performance_metrics.largest_loss is not None else 0.0
        total_commission = self.performance_metrics.total_commission if self.performance_metrics.total_commission is not None else 0.0
        total_taxes = self.performance_metrics.total_taxes if self.performance_metrics.total_taxes is not None else 0.0
        net_profit = self.performance_metrics.net_profit if self.performance_metrics.net_profit is not None else 0.0
        final_portfolio_value = self.performance_metrics.final_portfolio_value if self.performance_metrics.final_portfolio_value is not None else 0.0
        initial_capital = self.performance_metrics.initial_capital if self.performance_metrics.initial_capital is not None else 0.0
        
        logger.info(f"PERFORMANCE_SUMMARY Max drawdown from performance_metrics: {max_drawdown_value}")
        logger.info(f"PERFORMANCE_SUMMARY Max drawdown type: {type(max_drawdown_value)}")
        
        summary = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown_value,
            'win_loss_ratio': win_loss_ratio,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'total_commission': total_commission,
            'total_taxes': total_taxes,
            'net_profit': net_profit,
            'final_portfolio_value': final_portfolio_value,
            'initial_capital': initial_capital
        }
        
        # Add benchmark metrics from integrated performance metrics
        if self.benchmark_metrics is not None:
            summary.update({
                'benchmark_return': self.benchmark_metrics.get('benchmark_return', 0.0),
                'excess_return': self.benchmark_metrics.get('excess_return', 0.0),
                'information_ratio': self.benchmark_metrics.get('information_ratio', 0.0),
                'beta': self.benchmark_metrics.get('beta', 0.0),
                'alpha': self.benchmark_metrics.get('alpha', 0.0),
                'tracking_error': self.benchmark_metrics.get('tracking_error', 0.0),
                'rolling_correlation': self.benchmark_metrics.get('rolling_correlation', 0.0),
                'benchmark_sharpe': self.benchmark_metrics.get('benchmark_sharpe', 0.0),
                'benchmark_max_drawdown': self.benchmark_metrics.get('benchmark_max_drawdown', 0.0),
                'benchmark_win_rate': self.benchmark_metrics.get('benchmark_win_rate', 0.0),
                'benchmark_symbol': self.benchmark_metrics.get('benchmark_symbol', 'IBOV')
            })
        
        return summary
    
    def export_results(self, filepath: str) -> None:
        """
        Export simulation results to JSON file.
        
        Args:
            filepath: Path to export file
        """
        try:
            # Add debug logging for max_drawdown before export
            performance_summary = self.get_performance_summary()
            max_drawdown_before_export = performance_summary.get('max_drawdown', 'NOT_FOUND')
            logger.info(f"SIMULATOR_EXPORT Max drawdown before JSON export: {max_drawdown_before_export}")
            logger.info(f"SIMULATOR_EXPORT Performance metrics object max_drawdown: {self.performance_metrics.max_drawdown}")
            
            results = {
                'simulation_info': {
                    'strategy_name': self.strategy.name,
                    'initial_capital': self.initial_capital,
                    'start_date': self.start_date.isoformat() if self.start_date else None,
                    'end_date': self.end_date.isoformat() if self.end_date else None,
                    'simulation_duration_seconds': (
                        self.simulation_end_time - self.simulation_start_time
                    ).total_seconds() if self.simulation_start_time and self.simulation_end_time else None
                },
                'performance_metrics': performance_summary,
                'daily_data': {
                    'dates': [d.isoformat() for d in pd.date_range(
                        start=self.start_date or pd.Timestamp.min,
                        end=self.end_date or pd.Timestamp.max,
                        periods=len(self.daily_portfolio_values)
                    )],
                    'portfolio_values': self.daily_portfolio_values,
                    'daily_returns': self.daily_returns
                },
                'trade_log': self.trade_log
            }
            
            import os as _os
            if not (_os.getenv('AUDIT_EXECUTIONS_ONLY', '1').lower() in ('1', 'true', 'yes')):
                with open(filepath, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info(f"Simulation results exported to: {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            raise
    
    def get_summary_data(self) -> Dict[str, Any]:
        """Get simulation summary data for HTML reports."""
        # Add defensive checks for performance metrics to prevent None values
        final_portfolio_value = self.performance_metrics.final_portfolio_value if self.performance_metrics.final_portfolio_value is not None else 0.0
        net_profit = self.performance_metrics.net_profit if self.performance_metrics.net_profit is not None else 0.0
        total_return = self.performance_metrics.total_return if self.performance_metrics.total_return is not None else 0.0
        annualized_return = self.performance_metrics.annualized_return if self.performance_metrics.annualized_return is not None else 0.0
        sharpe_ratio = self.performance_metrics.sharpe_ratio if self.performance_metrics.sharpe_ratio is not None else 0.0
        max_drawdown = self.performance_metrics.max_drawdown if self.performance_metrics.max_drawdown is not None else 0.0
        win_loss_ratio = self.performance_metrics.win_loss_ratio if self.performance_metrics.win_loss_ratio is not None else 0.0
        profit_factor = self.performance_metrics.profit_factor if self.performance_metrics.profit_factor is not None else 0.0
        total_trades = self.performance_metrics.total_trades if self.performance_metrics.total_trades is not None else 0
        winning_trades = self.performance_metrics.winning_trades if self.performance_metrics.winning_trades is not None else 0
        losing_trades = self.performance_metrics.losing_trades if self.performance_metrics.losing_trades is not None else 0
        avg_win = self.performance_metrics.avg_win if self.performance_metrics.avg_win is not None else 0.0
        avg_loss = self.performance_metrics.avg_loss if self.performance_metrics.avg_loss is not None else 0.0
        largest_win = self.performance_metrics.largest_win if self.performance_metrics.largest_win is not None else 0.0
        largest_loss = self.performance_metrics.largest_loss if self.performance_metrics.largest_loss is not None else 0.0
        total_commission = self.performance_metrics.total_commission if self.performance_metrics.total_commission is not None else 0.0
        total_taxes = self.performance_metrics.total_taxes if self.performance_metrics.total_taxes is not None else 0.0
        
        return {
            'strategy_name': self.strategy.name,
            'initial_capital': self.initial_capital,
            'final_portfolio_value': final_portfolio_value,
            'net_profit': net_profit,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_loss_ratio': win_loss_ratio,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'total_commission': total_commission,
            'total_taxes': total_taxes,
            'simulation_duration': (
                self.simulation_end_time - self.simulation_start_time
            ).total_seconds() if self.simulation_start_time and self.simulation_end_time else None
        }


def main():
    """Example usage of BacktestSimulator."""
# Quiet import-time prints


if __name__ == "__main__":
    main() 