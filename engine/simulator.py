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
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml

from engine.portfolio import EnhancedPortfolio
from engine.base_strategy import BaseStrategy, TradingSignal, SignalType, TradeType
from engine.performance_metrics import PerformanceMetrics
from engine.tca import TransactionCostAnalyzer
from engine.sgs_data_loader import SELICDataError, SELICDataUnavailableError, SELICDataInsufficientError, SELICDataQualityError, SELICDataValidationError

# Configure logging
logger = logging.getLogger(__name__)


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
            'generate_signals',
            'manage_risk',
            'execute_trade'
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
            'validate_market_data',
            'reset_strategy'
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
        
        # Use strategy's existing portfolio or create new one if needed
        if hasattr(self.strategy, 'portfolio') and self.strategy.portfolio is not None:
            self.portfolio = self.strategy.portfolio
            # Update portfolio with initial capital
            self.portfolio.cash = initial_capital
            self.portfolio.initial_cash = initial_capital
            self.portfolio.total_value = initial_capital
        else:
            # Create portfolio with initial capital
            self.portfolio = EnhancedPortfolio(config_path)
            self.portfolio.cash = initial_capital
            self.portfolio.initial_cash = initial_capital
            self.portfolio.total_value = initial_capital
            # Assign portfolio to strategy
            self.strategy.portfolio = self.portfolio
        
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
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"BacktestSimulator initialized with R$ {initial_capital:,.2f} initial capital")
        logger.info(f"Strategy: {self.strategy.strategy_name}")
    
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
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"backtest_simulator_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.info(f"Simulation logging initialized: {log_file}")
    
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
        
        # Validate data
        if data.empty:
            raise ValueError("Cannot run simulation with empty data")
        
        # Record simulation start time
        self.simulation_start_time = datetime.now()
        
        try:
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
            
            # Reset strategy if method exists
            if hasattr(self.strategy, 'reset_strategy'):
                self.strategy.reset_strategy()
            else:
                logger.warning("Strategy does not have reset_strategy method")
            
            # Process each trading day
            for date, daily_data in data.iterrows():
                logger.debug(f"Processing date: {date.date()}")
                
                # Update portfolio prices
                price_updates = {self.strategy.symbol: daily_data['close']}
                self.portfolio.update_prices(price_updates, date)
                
                # Prepare market data for strategy
                market_data = self._prepare_market_data(data, date)
                
                # Generate trading signals
                signals = self.strategy.generate_signals(market_data)
                
                # Execute trades for each signal
                for signal in signals:
                    if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                        self._execute_trade(signal, daily_data)
                
                # Process settlements with defensive check
                try:
                    if hasattr(self.portfolio, 'settlement_manager') and self.portfolio.settlement_manager is not None:
                        self.portfolio.settlement_manager.process_settlements(date.date())
                    else:
                        logger.warning(f"Settlement manager not available for date {date.date()}")
                except Exception as e:
                    logger.error(f"Error processing settlements for date {date.date()}: {str(e)}")
                
                # Record daily portfolio value
                portfolio_value = self.portfolio.get_portfolio_value()
                self.daily_portfolio_values.append(portfolio_value)
                
                # Calculate daily return
                if len(self.daily_portfolio_values) > 1:
                    daily_return = (portfolio_value / self.daily_portfolio_values[-2]) - 1
                else:
                    daily_return = 0.0
                
                self.daily_returns.append(daily_return)
                
                # Log daily summary
                logger.debug(f"Date: {date.date()}, Portfolio Value: R$ {portfolio_value:,.2f}, "
                           f"Daily Return: {daily_return:.4f}")
            
            # Record simulation end time
            self.simulation_end_time = datetime.now()
            
            # Calculate all performance metrics (including benchmark)
            self._calculate_performance_metrics()
            
            # Create simulation result
            result = self._create_simulation_result()
            
            logger.info("Backtest simulation completed successfully")
            logger.info(f"Final portfolio value: R$ {result.final_portfolio_value:,.2f}")
            logger.info(f"Total return: {result.total_return:.4f}")
            logger.info(f"Sharpe ratio: {result.sharpe_ratio:.4f}")
            logger.info(f"Max drawdown: {result.max_drawdown:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            raise RuntimeError(f"Simulation failed: {str(e)}")
    
    def _prepare_market_data(self, data: pd.DataFrame, current_date: datetime) -> Dict[str, Any]:
        """
        Prepare market data for strategy signal generation.
        
        Args:
            data: Complete market data DataFrame
            current_date: Current trading date
            
        Returns:
            Dictionary containing market data for strategy
        """
        # Get historical data up to current date
        historical_data = data[data.index <= current_date].copy()
        
        # Calculate technical indicators (simplified)
        if len(historical_data) >= 20:
            historical_data['sma_20'] = historical_data['close'].rolling(window=20).mean()
            historical_data['sma_50'] = historical_data['close'].rolling(window=50).mean()
            historical_data['volume_sma_20'] = historical_data['volume'].rolling(window=20).mean()
        
        # Get current day's data
        current_data = historical_data.iloc[-1] if not historical_data.empty else None
        
        # Load SGS data for current date
        sgs_data = self._load_sgs_data_for_date(current_date)
        
        # Calculate SELIC-CDI spread for historical analysis
        selic_cdi_spread = self._calculate_selic_cdi_spread(sgs_data)
        
        # Prepare market data dictionary
        market_data = {
            'price_data': historical_data,
            'current_price': current_data['close'] if current_data is not None else 0.0,
            'current_volume': current_data['volume'] if current_data is not None else 0.0,
            'timestamp': current_date,
            'sgs_data': sgs_data,  # Add SGS data
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
        """
        try:
            # Initialize SGS loader if not already done
            if not hasattr(self, 'sgs_loader'):
                from engine.sgs_data_loader import SGSDataLoader
                self.sgs_loader = SGSDataLoader()
            
            # Load configuration for strict mode
            config = self._load_config()
            strict_config = config.get('sgs', {}).get('strict_mode', {})
            
            end_date = current_date.strftime('%d/%m/%Y')
            start_date = (current_date - timedelta(days=30)).strftime('%d/%m/%Y')
            sgs_data = {}
            
            # Special handling for SELIC (series 11) in strict mode
            if strict_config.get('enabled', False):
                try:
                    # Validate SELIC data coverage for the period
                    coverage_info = self.sgs_loader.validate_selic_data_coverage(start_date, end_date)
                    
                    if not coverage_info.get('meets_requirements', False):
                        raise SELICDataInsufficientError(
                            f"SELIC data coverage ({coverage_info.get('coverage_percentage', 0):.1f}%) "
                            f"does not meet strict mode requirements for date {current_date.date()}"
                        )
                    
                    logger.debug(f"SELIC data validation passed for {current_date.date()}: "
                               f"{coverage_info.get('coverage_percentage', 0):.1f}% coverage")
                except (SELICDataUnavailableError, SELICDataInsufficientError, SELICDataQualityError, SELICDataValidationError) as e:
                    if strict_config.get('fail_on_missing_data', False):
                        logger.error(f"Critical SELIC data issue for {current_date.date()}: {e}")
                        raise RuntimeError(f"Backtest cannot proceed due to SELIC data issues: {e}")
                    else:
                        logger.warning(f"SELIC data issue (non-strict mode) for {current_date.date()}: {e}")
            
            # Load all SGS series data
            for series_id in self.sgs_loader.SGS_SERIES.keys():
                try:
                    series_data = self.sgs_loader.get_series_data(
                        series_id=series_id,
                        start_date=start_date,
                        end_date=end_date,
                        use_cache=True,
                        save_processed=False
                    )
                    if series_data is not None and not series_data.empty:
                        current_date_pd = pd.to_datetime(current_date.date())
                        if current_date_pd in series_data.index:
                            row = series_data.loc[current_date_pd]
                        else:
                            available_dates = series_data.index[series_data.index <= current_date_pd]
                            if len(available_dates) > 0:
                                closest_date = available_dates[-1]
                                row = series_data.loc[closest_date]
                            else:
                                row = None
                        if row is not None:
                            if series_id == 11 and 'daily_factor' in row:
                                sgs_data['selic_daily_factor'] = row['daily_factor']
                            sgs_data[f'series_{series_id}'] = row['valor']
                            sgs_data[self.sgs_loader.SGS_SERIES[series_id].lower().replace(' ', '_')] = row['valor']
                except Exception as e:
                    logger.warning(f"Failed to load SGS series {series_id}: {e}")
                    continue
            
            # In strict mode, ensure we have SELIC data
            if strict_config.get('enabled', False) and 'selic_daily_factor' not in sgs_data:
                if strict_config.get('fail_on_missing_data', False):
                    raise SELICDataUnavailableError(f"No SELIC data available for {current_date.date()}")
                else:
                    logger.warning(f"No SELIC data available for {current_date.date()}, using fallback")
            
            return sgs_data
        except (SELICDataUnavailableError, SELICDataInsufficientError, SELICDataQualityError, SELICDataValidationError) as e:
            strict_config = config.get('sgs', {}).get('strict_mode', {})
            if strict_config.get('fail_on_missing_data', False):
                logger.error(f"Critical SELIC data issue: {e}")
                raise RuntimeError(f"Backtest cannot proceed due to SELIC data issues: {e}")
            else:
                logger.error(f"SELIC data issue (non-strict mode): {e}")
                return {}
        except Exception as e:
            logger.error(f"Error loading SGS data: {e}")
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
    
    def _execute_trade(self, signal: TradingSignal, price_data: pd.Series) -> None:
        """
        Execute individual trade based on signal.
        
        Args:
            signal: Trading signal to execute
            price_data: Current day's price data
        """
        try:
            # Validate market data (optional method)
            if hasattr(self.strategy, 'validate_market_data'):
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
            
            # Get available cash with defensive check
            try:
                if not hasattr(self.portfolio, 'settlement_manager'):
                    logger.error("Portfolio missing settlement_manager")
                    return
                
                available_cash = self.portfolio.settlement_manager.get_available_cash(
                    signal.timestamp.date()
                )
            except Exception as e:
                logger.error(f"Error getting available cash: {str(e)}")
                return
            
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
            
            # Final validation
            if quantity <= 0:
                logger.debug(f"Insufficient quantity for signal: {signal}")
                return
            
            if quantity <= 0:
                logger.debug(f"Insufficient cash or invalid position size for signal: {signal}")
                return
            
            # Execute trade based on signal type
            if signal.signal_type == SignalType.BUY:
                success = self.portfolio.buy(
                    ticker=signal.ticker,
                    quantity=quantity,
                    price=signal.price,
                    trade_date=signal.timestamp,
                    trade_type=signal.trade_type.value,
                    description=f"Strategy signal: {signal.signal_type.value}"
                )
                
                if success:
                    logger.info(f"Buy executed: {quantity} {signal.ticker} @ R$ {signal.price:.2f}")
                else:
                    logger.warning(f"Buy failed: {quantity} {signal.ticker} @ R$ {signal.price:.2f}")
            
            elif signal.signal_type == SignalType.SELL:
                # Check if we have position to sell
                if signal.ticker not in self.portfolio.positions:
                    logger.warning(f"No position in {signal.ticker} to sell")
                    return
                
                position = self.portfolio.positions[signal.ticker]
                sell_quantity = min(quantity, position.quantity)
                
                if sell_quantity <= 0:
                    logger.warning(f"No shares available to sell in {signal.ticker}")
                    return
                
                success = self.portfolio.sell(
                    ticker=signal.ticker,
                    quantity=sell_quantity,
                    price=signal.price,
                    trade_date=signal.timestamp,
                    trade_type=signal.trade_type.value,
                    description=f"Strategy signal: {signal.signal_type.value}"
                )
                
                if success:
                    logger.info(f"Sell executed: {sell_quantity} {signal.ticker} @ R$ {signal.price:.2f}")
                else:
                    logger.warning(f"Sell failed: {sell_quantity} {signal.ticker} @ R$ {signal.price:.2f}")
            
            # Record trade in log
            trade_record = {
                'date': signal.timestamp,
                'ticker': signal.ticker,
                'signal_type': signal.signal_type.value,
                'quantity': quantity,
                'price': signal.price,
                'trade_type': signal.trade_type.value,
                'confidence': signal.confidence,
                'portfolio_value': self.portfolio.get_portfolio_value()
            }
            
            self.trade_log.append(trade_record)
            
        except Exception as e:
            logger.error(f"Error executing trade for signal {signal}: {str(e)}")
    
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
        
        # Update the legacy performance metrics for backward compatibility
        self.performance_metrics.total_return = all_metrics['total_return']
        self.performance_metrics.annualized_return = all_metrics['annualized_return']
        self.performance_metrics.sharpe_ratio = all_metrics['sharpe_ratio']
        self.performance_metrics.max_drawdown = all_metrics['max_drawdown']
        self.performance_metrics.total_trades = all_metrics['total_trades']
        self.performance_metrics.winning_trades = all_metrics['winning_trades']
        self.performance_metrics.losing_trades = all_metrics['losing_trades']
        self.performance_metrics.total_commission = all_metrics['total_commission']
        self.performance_metrics.total_taxes = self.portfolio.total_taxes
        self.performance_metrics.final_portfolio_value = self.daily_portfolio_values[-1]
        self.performance_metrics.initial_capital = self.daily_portfolio_values[0]
        self.performance_metrics.net_profit = self.daily_portfolio_values[-1] - self.daily_portfolio_values[0]
        
        # Store benchmark metrics for simulation result
        self.benchmark_metrics = all_metrics
        
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
        
        return SimulationResult(
            total_return=self.performance_metrics.total_return,
            sharpe_ratio=self.performance_metrics.sharpe_ratio,
            max_drawdown=self.performance_metrics.max_drawdown,
            win_loss_ratio=self.performance_metrics.win_loss_ratio,
            total_trades=self.performance_metrics.total_trades,
            winning_trades=self.performance_metrics.winning_trades,
            losing_trades=self.performance_metrics.losing_trades,
            final_portfolio_value=self.performance_metrics.final_portfolio_value,
            initial_capital=self.performance_metrics.initial_capital,
            total_commission=self.performance_metrics.total_commission,
            total_taxes=self.performance_metrics.total_taxes,
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
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Returns:
            Dictionary containing all performance metrics
        """
        summary = {
            'total_return': self.performance_metrics.total_return,
            'annualized_return': self.performance_metrics.annualized_return,
            'sharpe_ratio': self.performance_metrics.sharpe_ratio,
            'max_drawdown': self.performance_metrics.max_drawdown,
            'win_loss_ratio': self.performance_metrics.win_loss_ratio,
            'profit_factor': self.performance_metrics.profit_factor,
            'total_trades': self.performance_metrics.total_trades,
            'winning_trades': self.performance_metrics.winning_trades,
            'losing_trades': self.performance_metrics.losing_trades,
            'avg_win': self.performance_metrics.avg_win,
            'avg_loss': self.performance_metrics.avg_loss,
            'largest_win': self.performance_metrics.largest_win,
            'largest_loss': self.performance_metrics.largest_loss,
            'total_commission': self.performance_metrics.total_commission,
            'total_taxes': self.performance_metrics.total_taxes,
            'net_profit': self.performance_metrics.net_profit,
            'final_portfolio_value': self.performance_metrics.final_portfolio_value,
            'initial_capital': self.performance_metrics.initial_capital
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
            results = {
                'simulation_info': {
                    'strategy_name': self.strategy.strategy_name,
                    'initial_capital': self.initial_capital,
                    'start_date': self.start_date.isoformat() if self.start_date else None,
                    'end_date': self.end_date.isoformat() if self.end_date else None,
                    'simulation_duration_seconds': (
                        self.simulation_end_time - self.simulation_start_time
                    ).total_seconds() if self.simulation_start_time and self.simulation_end_time else None
                },
                'performance_metrics': self.get_performance_summary(),
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
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Simulation results exported to: {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            raise
    
    def get_summary_data(self) -> Dict[str, Any]:
        """Get simulation summary data for HTML reports."""
        return {
            'strategy_name': self.strategy.strategy_name,
            'initial_capital': self.initial_capital,
            'final_portfolio_value': self.performance_metrics.final_portfolio_value,
            'net_profit': self.performance_metrics.net_profit,
            'total_return': self.performance_metrics.total_return,
            'annualized_return': self.performance_metrics.annualized_return,
            'sharpe_ratio': self.performance_metrics.sharpe_ratio,
            'max_drawdown': self.performance_metrics.max_drawdown,
            'win_loss_ratio': self.performance_metrics.win_loss_ratio,
            'profit_factor': self.performance_metrics.profit_factor,
            'total_trades': self.performance_metrics.total_trades,
            'winning_trades': self.performance_metrics.winning_trades,
            'losing_trades': self.performance_metrics.losing_trades,
            'avg_win': self.performance_metrics.avg_win,
            'avg_loss': self.performance_metrics.avg_loss,
            'largest_win': self.performance_metrics.largest_win,
            'largest_loss': self.performance_metrics.largest_loss,
            'total_commission': self.performance_metrics.total_commission,
            'total_taxes': self.performance_metrics.total_taxes,
            'simulation_duration': (
                self.simulation_end_time - self.simulation_start_time
            ).total_seconds() if self.simulation_start_time and self.simulation_end_time else None
        }


def main():
    """Example usage of BacktestSimulator."""
    print("BacktestSimulator module loaded successfully")
    print("Use with a concrete BaseStrategy implementation for backtesting")


if __name__ == "__main__":
    main() 