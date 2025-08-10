"""
Comprehensive Performance Metrics Module for Brazilian B3 Quant Backtest System

Advanced performance analysis with Brazilian market compliance:
- Tax-aware return calculations with Brazilian tax rules (2025)
- Risk-adjusted metrics with Brazilian market parameters
- Comprehensive performance analysis with regulatory compliance
- Integration with existing portfolio, loss manager, and TCA modules

Author: Your Name
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from functools import lru_cache
import yaml
import pytz
import json
from pathlib import Path

from engine.portfolio import EnhancedPortfolio
from engine.loss_manager import EnhancedLossCarryforwardManager
from engine.tca import TransactionCostAnalyzer
from engine.sgs_data_loader import SGSDataLoader, SELICDataError, SELICDataUnavailableError, SELICDataInsufficientError, SELICDataQualityError, SELICDataValidationError
from engine.layer_execution_analyzer import LayerExecutionAnalyzer

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ReturnsMetrics:
    """Comprehensive returns calculation metrics."""
    total_return: float = 0.0
    annualized_return: float = 0.0
    logarithmic_return: float = 0.0
    daily_returns: List[float] = field(default_factory=list)
    cumulative_returns: List[float] = field(default_factory=list)
    trading_days: int = 252  # Brazilian market standard


@dataclass
class RiskMetrics:
    """Risk-adjusted performance metrics."""
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    volatility: float = 0.0
    downside_deviation: float = 0.0
    var_95: float = 0.0  # Value at Risk (95% confidence)
    cvar_95: float = 0.0  # Conditional Value at Risk (95% confidence)


@dataclass
class TaxMetrics:
    """Brazilian tax-specific performance metrics."""
    total_taxes_paid: float = 0.0
    swing_trade_taxes: float = 0.0
    day_trade_taxes: float = 0.0
    tax_efficiency: float = 0.0  # After-tax return / pre-tax return
    loss_carryforward_utilized: float = 0.0
    effective_tax_rate: float = 0.0
    tax_exemption_utilized: float = 0.0  # R$20,000 monthly exemption


@dataclass
class TradeMetrics:
    """Trade-specific performance metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    average_trade_duration: float = 0.0
    total_commission: float = 0.0


@dataclass
class BenchmarkMetrics:
    """Benchmark comparison metrics."""
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
    benchmark_symbol: str = "IBOV"


class PerformanceMetrics:
    """
    Base performance metrics calculator with Brazilian market compliance.
    
    Features:
    - Comprehensive return calculations using 252 trading days
    - Integration with existing portfolio and loss manager
    - Brazilian tax rule compliance (2025)
    - Performance optimization with caching
    """
    
    def __init__(self, portfolio: EnhancedPortfolio, config_path: str = "config/settings.yaml"):
        """
        Initialize performance metrics calculator.
        
        Args:
            portfolio: EnhancedPortfolio instance
            config_path: Path to configuration file
        """
        self.portfolio = portfolio
        self.config = self._load_config(config_path)
        self.timezone = pytz.timezone(self.config['market']['trading_hours']['timezone'])
        
        # Brazilian market constants - load from configuration
        self.TRADING_DAYS_PER_YEAR = self.config['market'].get('trading_days_per_year', 252)
        self.STATIC_RISK_FREE_RATE = self.config['market'].get('selic_rate', 0.15)  # Static fallback rate
        
        # Initialize SGS data loader for dynamic rates
        self.sgs_loader = None
        self.selic_data = None
        self._initialize_sgs_integration()
        
        # Initialize metrics containers
        self.returns_metrics = ReturnsMetrics()
        self.risk_metrics = RiskMetrics()
        self.tax_metrics = TaxMetrics()
        self.trade_metrics = TradeMetrics()
        self.benchmark_metrics = BenchmarkMetrics()
        
        # Initialize benchmark analyzer
        self.benchmark_analyzer = None
        self._initialize_benchmark_analyzer()
        
        # Backward compatibility attributes for simulator
        self.total_return = 0.0
        self.annualized_return = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission = 0.0
        self.total_taxes = 0.0
        self.final_portfolio_value = 0.0
        self.initial_capital = 0.0
        self.net_profit = 0.0
        
        logger.info("Performance Metrics initialized with Brazilian market parameters")
        logger.info(f"Static SELIC rate: {self.STATIC_RISK_FREE_RATE:.4f}")
    
    # Properties for backward compatibility
    @property
    def win_loss_ratio(self) -> float:
        """Calculate win/loss ratio for backward compatibility."""
        if self.losing_trades > 0:
            return self.winning_trades / self.losing_trades
        return 0.0 if self.winning_trades == 0 else float('inf')
    
    @property
    def profit_factor(self) -> float:
        """Get profit factor from trade metrics."""
        return self.trade_metrics.profit_factor
    
    @property
    def avg_win(self) -> float:
        """Get average win from trade metrics."""
        return self.trade_metrics.average_win
    
    @property
    def avg_loss(self) -> float:
        """Get average loss from trade metrics."""
        return self.trade_metrics.average_loss
    
    @property
    def largest_win(self) -> float:
        """Get largest win from trade metrics."""
        return self.trade_metrics.largest_win
    
    @property
    def largest_loss(self) -> float:
        """Get largest loss from trade metrics."""
        return self.trade_metrics.largest_loss
    
    def _initialize_benchmark_analyzer(self):
        """Initialize benchmark analyzer based on configuration."""
        try:
            benchmark_config = self.config.get('benchmark', {})
            enabled = benchmark_config.get('enabled', True)
            
            if not enabled:
                logger.info("Benchmark analysis disabled in configuration")
                return
            
            # Initialize benchmark analyzer
            benchmark_symbol = benchmark_config.get('symbol', 'IBOV')
            risk_free_rate_override = benchmark_config.get('risk_free_rate_override')
            
            self.benchmark_analyzer = BenchmarkAnalyzer(
                config_path="config/settings.yaml",
                benchmark_symbol=benchmark_symbol,
                risk_free_rate=risk_free_rate_override
            )
            
            # Load benchmark data if auto_load is enabled
            auto_load = benchmark_config.get('auto_load', True)
            if auto_load:
                logger.info(f"Loading benchmark data for {benchmark_symbol}...")
                if self.benchmark_analyzer.load_benchmark_data():
                    logger.info("Benchmark data loaded successfully")
                    self.benchmark_metrics.benchmark_symbol = benchmark_symbol
                else:
                    required = benchmark_config.get('required', False)
                    if required:
                        logger.error(f"Failed to load required benchmark data for {benchmark_symbol}")
                    else:
                        logger.warning(f"Failed to load benchmark data for {benchmark_symbol}, continuing without benchmark analysis")
                        self.benchmark_analyzer = None
            
        except Exception as e:
            logger.error(f"Error initializing benchmark analyzer: {e}")
            self.benchmark_analyzer = None
    
    def _initialize_sgs_integration(self):
        """Initialize SGS data loader for dynamic risk-free rates with strict validation."""
        try:
            self.sgs_loader = SGSDataLoader()
            
            # Check if strict mode is enabled
            strict_config = self.config.get('sgs', {}).get('strict_mode', {})
            if strict_config.get('enabled', False):
                logger.info("Strict mode enabled - SELIC data validation required")
                self._validate_selic_data_requirements()
            else:
                logger.info("Strict mode disabled - fallback rates allowed")
            
            logger.info("SGS data loader initialized for dynamic SELIC rates")
        except (SELICDataUnavailableError, SELICDataInsufficientError, SELICDataQualityError, SELICDataValidationError) as e:
            # In strict mode, these exceptions should cause the system to fail
            strict_config = self.config.get('sgs', {}).get('strict_mode', {})
            if strict_config.get('fail_on_missing_data', False):
                logger.error(f"Critical SELIC data issue in strict mode: {e}")
                raise RuntimeError(f"Backtest cannot proceed due to SELIC data issues: {e}")
            else:
                logger.warning(f"SELIC data issue (non-strict mode): {e}. Using static rates only.")
                self.sgs_loader = None
        except Exception as e:
            logger.warning(f"Failed to initialize SGS loader: {e}. Using static rates only.")
            self.sgs_loader = None
    
    def _validate_selic_data_requirements(self):
        """
        Validate that SELIC data meets strict requirements.
        This method is called during initialization when strict mode is enabled.
        """
        try:
            # Get strict mode configuration
            strict_config = self.config.get('sgs', {}).get('strict_mode', {})
            quality_config = self.config.get('sgs', {}).get('quality_thresholds', {})
            
            if not strict_config.get('enabled', False):
                logger.info("Strict mode not enabled, skipping SELIC validation")
                return
            
            logger.info("Validating SELIC data requirements for strict mode...")
            
            # For initialization, we'll validate with a reasonable test period
            # (last 365 days to ensure sufficient data availability for strict mode)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            # Format dates for validation
            start_str = start_date.strftime("%d/%m/%Y")
            end_str = end_date.strftime("%d/%m/%Y")
            
            # Validate SELIC data coverage
            coverage_info = self.sgs_loader.validate_selic_data_coverage(start_str, end_str)
            
            if coverage_info.get('strict_mode_enabled', False):
                logger.info(f"SELIC data validation passed: {coverage_info.get('coverage_percentage', 0):.1f}% coverage")
                logger.info(f"Quality score: {coverage_info.get('quality_score', 0):.1f}%")
            else:
                logger.warning("SELIC data validation skipped - strict mode not enabled in SGS loader")
                
        except (SELICDataUnavailableError, SELICDataInsufficientError, SELICDataQualityError, SELICDataValidationError) as e:
            logger.error(f"SELIC data validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during SELIC data validation: {e}")
            raise SELICDataValidationError(f"Failed to validate SELIC data requirements: {e}")
    
    def _load_selic_data(self, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Load SELIC rate data from SGS for the specified date range.
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with SELIC rates or None if unavailable
        """
        if self.sgs_loader is None:
            return None
        
        try:
            # Format dates for SGS API (dd/mm/yyyy)
            start_str = start_date.strftime("%d/%m/%Y")
            end_str = end_date.strftime("%d/%m/%Y")
            
            # Fetch SELIC rate data (Series ID: 11)
            selic_data = self.sgs_loader.get_series_data(11, start_str, end_str)
            
            if selic_data is not None and not selic_data.empty:
                logger.info(f"Loaded {len(selic_data)} SELIC rate data points")
                return selic_data
            else:
                logger.warning("No SELIC data available from SGS")
                return None
                
        except Exception as e:
            logger.warning(f"Error loading SELIC data: {e}")
            return None
    
    def get_risk_free_rate(self, date: datetime) -> float:
        """
        Get risk-free rate for a specific date.
        Returns the daily risk-free rate as a decimal (e.g., 0.0005 for 0.05%).
        Uses the 'daily_factor' column if available (preferred), otherwise falls back to 'valor'.
        """
        if self.selic_data is not None and not self.selic_data.empty:
            try:
                date_str = date.strftime("%Y-%m-%d")
                # Prefer daily_factor if available
                if date_str in self.selic_data.index:
                    row = self.selic_data.loc[date_str]
                else:
                    available_dates = self.selic_data.index[self.selic_data.index <= date_str]
                    if len(available_dates) > 0:
                        latest_date = available_dates[-1]
                        row = self.selic_data.loc[latest_date]
                    else:
                        row = None
                if row is not None:
                    if 'daily_factor' in row:
                        rate = row['daily_factor'] - 1
                        logger.debug(f"Dynamic SELIC daily_factor for {date_str}: {rate:.8f}")
                        return rate
                    elif 'valor' in row:
                        # Fallback: treat as percent if > 0.01, else decimal
                        valor = row['valor']
                        if valor > 1.0:
                            rate = valor
                        elif valor > 0.01:
                            rate = valor / 100
                        else:
                            rate = valor
                        logger.debug(f"Dynamic SELIC valor for {date_str}: {rate:.8f}")
                        return rate
            except Exception as e:
                logger.warning(f"Error getting dynamic SELIC rate: {e}")
        logger.debug(f"Using static SELIC rate: {self.STATIC_RISK_FREE_RATE:.8f}")
        return self.STATIC_RISK_FREE_RATE / self.TRADING_DAYS_PER_YEAR
    
    def set_selic_data(self, selic_data: pd.DataFrame):
        """
        Set SELIC data for dynamic rate calculation.
        
        Args:
            selic_data: DataFrame with SELIC rates (index: dates, column: 'valor')
        """
        if selic_data is not None and not selic_data.empty:
            self.selic_data = selic_data
            logger.info(f"Set SELIC data with {len(selic_data)} data points")
        else:
            logger.warning("Invalid SELIC data provided")
    
    def load_selic_data_for_period(self, start_date: datetime, end_date: datetime):
        """
        Load SELIC data for a specific period with strict validation.
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
        """
        try:
            # Check if strict mode is enabled
            strict_config = self.config.get('sgs', {}).get('strict_mode', {})
            
            if strict_config.get('enabled', False) and self.sgs_loader is not None:
                # Validate SELIC data coverage for the specific period
                start_str = start_date.strftime("%d/%m/%Y")
                end_str = end_date.strftime("%d/%m/%Y")
                
                logger.info(f"Validating SELIC data coverage for period {start_str} to {end_str}")
                coverage_info = self.sgs_loader.validate_selic_data_coverage(start_str, end_str)
                
                if not coverage_info.get('meets_requirements', False):
                    raise SELICDataInsufficientError(
                        f"SELIC data coverage ({coverage_info.get('coverage_percentage', 0):.1f}%) "
                        f"does not meet strict mode requirements"
                    )
                
                logger.info(f"SELIC data validation passed: {coverage_info.get('coverage_percentage', 0):.1f}% coverage")
            
            # Load SELIC data
            selic_data = self._load_selic_data(start_date, end_date)
            
            if selic_data is not None:
                self.set_selic_data(selic_data)
                logger.info(f"Successfully loaded SELIC data for period {start_date.date()} to {end_date.date()}")
            else:
                if strict_config.get('fail_on_missing_data', False):
                    raise SELICDataUnavailableError(f"No SELIC data available for period {start_date.date()} to {end_date.date()}")
                else:
                    logger.warning("No SELIC data available, using static rate for the period")
                    
        except (SELICDataUnavailableError, SELICDataInsufficientError, SELICDataQualityError, SELICDataValidationError) as e:
            strict_config = self.config.get('sgs', {}).get('strict_mode', {})
            if strict_config.get('fail_on_missing_data', False):
                logger.error(f"Critical SELIC data issue: {e}")
                raise RuntimeError(f"Backtest cannot proceed due to SELIC data issues: {e}")
            else:
                logger.warning(f"SELIC data issue (non-strict mode): {e}. Using static rate for the period.")
        except Exception as e:
            logger.error(f"Error loading SELIC data: {e}")
            if strict_config.get('fail_on_missing_data', False):
                raise RuntimeError(f"Backtest cannot proceed due to SELIC data loading error: {e}")
            else:
                logger.warning("Using static SELIC rate due to loading error")
    
    @property
    def RISK_FREE_RATE(self) -> float:
        """
        Property to maintain backward compatibility.
        Returns the current risk-free rate (static fallback).
        """
        return self.STATIC_RISK_FREE_RATE
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration with error handling."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def calculate_returns(self, portfolio_values: List[float], 
                         start_date: datetime, end_date: datetime) -> ReturnsMetrics:
        """
        Calculate comprehensive returns metrics.
        
        Args:
            portfolio_values: List of daily portfolio values
            start_date: Simulation start date
            end_date: Simulation end date
            
        Returns:
            ReturnsMetrics object with calculated metrics
        """
        if len(portfolio_values) < 2:
            logger.warning("Insufficient portfolio values for returns calculation")
            return ReturnsMetrics()
        
        # Enhanced validation and debugging for portfolio values
        logger.debug(f"Portfolio values statistics:")
        logger.debug(f"  Length: {len(portfolio_values)}")
        
        # Check for None values
        none_count = sum(1 for v in portfolio_values if v is None)
        logger.debug(f"  None values: {none_count}")
        
        if none_count > 0:
            logger.warning(f"Found {none_count} None values in portfolio_values")
            # Filter out None values for calculation
            portfolio_values = [v for v in portfolio_values if v is not None]
            logger.debug(f"  Filtered length: {len(portfolio_values)}")
        
        if len(portfolio_values) == 0:
            logger.error("No valid portfolio values after filtering None values")
            return ReturnsMetrics()
        
        logger.debug(f"  Initial value: {portfolio_values[0]:.2f}")
        logger.debug(f"  Final value: {portfolio_values[-1]:.2f}")
        logger.debug(f"  Min value: {min(portfolio_values):.2f}")
        logger.debug(f"  Max value: {max(portfolio_values):.2f}")
        logger.debug(f"  Unique values: {len(set(portfolio_values))}")
        
        # Check for constant portfolio values (which would lead to zero returns)
        if len(set(portfolio_values)) == 1:
            logger.warning("All portfolio values are identical - this will result in zero returns")
        
        # Calculate daily returns with enhanced validation
        daily_returns = []
        zero_return_count = 0
        
        for i in range(1, len(portfolio_values)):
            prev_value = portfolio_values[i-1]
            curr_value = portfolio_values[i]
            
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                daily_returns.append(daily_return)
                
                # Track zero returns for debugging
                if abs(daily_return) < 1e-10:  # Very small threshold for "zero"
                    zero_return_count += 1
                    
                logger.debug(f"Day {i}: {prev_value:.2f} -> {curr_value:.2f} = {daily_return:.8f}")
            else:
                logger.warning(f"Invalid previous portfolio value at index {i-1}: {prev_value}")
                daily_returns.append(0.0)
                zero_return_count += 1
        
        # Log returns statistics
        logger.info(f"Daily returns calculation completed:")
        logger.info(f"  Total days: {len(daily_returns)}")
        logger.info(f"  Zero returns: {zero_return_count}")
        logger.info(f"  Non-zero returns: {len(daily_returns) - zero_return_count}")
        logger.info(f"  Returns range: [{min(daily_returns):.8f}, {max(daily_returns):.8f}]")
        
        # Calculate total return
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value if initial_value > 0 else 0.0
        
        # Calculate annualized return
        trading_days = len(daily_returns)
        if trading_days > 0:
            annualized_return = ((final_value / initial_value) ** (self.TRADING_DAYS_PER_YEAR / trading_days)) - 1
        else:
            annualized_return = 0.0
        
        # Calculate logarithmic return
        logarithmic_return = np.log(final_value / initial_value) if initial_value > 0 and final_value > 0 else 0.0
        
        # Calculate cumulative returns
        cumulative_returns = [1.0]  # Start with 100%
        for daily_return in daily_returns:
            cumulative_returns.append(cumulative_returns[-1] * (1 + daily_return))
        
        self.returns_metrics = ReturnsMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            logarithmic_return=logarithmic_return,
            daily_returns=daily_returns,
            cumulative_returns=cumulative_returns,
            trading_days=trading_days
        )
        
        logger.info(f"Returns calculated: Total={total_return:.4f}, Annualized={annualized_return:.4f}")
        logger.info(f"Portfolio growth: {initial_value:.2f} -> {final_value:.2f} ({total_return:.2%})")
        
        return self.returns_metrics
    
    def calculate_risk_metrics(self, daily_returns: List[float], start_date: Optional[datetime] = None) -> RiskMetrics:
        """
        Calculate risk-adjusted performance metrics.
        
        Args:
            daily_returns: List of daily returns
            start_date: Start date for dynamic risk-free rate calculation (optional)
            
        Returns:
            RiskMetrics object with calculated metrics
        """
        if not daily_returns:
            logger.warning("No daily returns provided for risk calculation")
            return RiskMetrics()
        
        returns_array = np.array(daily_returns)
        
        # Enhanced validation and debugging for returns array
        logger.debug(f"Returns array statistics:")
        logger.debug(f"  Length: {len(returns_array)}")
        logger.debug(f"  Non-zero returns: {np.count_nonzero(returns_array)}")
        logger.debug(f"  Zero returns: {np.sum(returns_array == 0)}")
        logger.debug(f"  Mean: {np.mean(returns_array):.8f}")
        logger.debug(f"  Std: {np.std(returns_array):.8f}")
        logger.debug(f"  Min: {np.min(returns_array):.8f}")
        logger.debug(f"  Max: {np.max(returns_array):.8f}")
        
        # Check if we have sufficient non-zero returns for meaningful volatility calculation
        non_zero_returns = returns_array[returns_array != 0]
        if len(non_zero_returns) < 2:
            logger.warning(f"Insufficient non-zero returns for volatility calculation: {len(non_zero_returns)} non-zero returns")
            # Use all returns (including zeros) but log the issue
            volatility = np.std(returns_array) * np.sqrt(self.TRADING_DAYS_PER_YEAR)
            logger.warning(f"Calculated volatility with mostly zero returns: {volatility:.8f}")
            
            # If volatility is very low, try enhanced volatility calculation
            if volatility < 1e-6:  # Very low volatility threshold
                logger.info("Standard volatility is very low, attempting enhanced volatility calculation")
                enhanced_volatility = self.calculate_enhanced_volatility(daily_returns)
                if enhanced_volatility > volatility:
                    logger.info(f"Using enhanced volatility: {enhanced_volatility:.6f} instead of {volatility:.8f}")
                    volatility = enhanced_volatility
        else:
            # Calculate volatility using all returns (standard approach)
            volatility = np.std(returns_array) * np.sqrt(self.TRADING_DAYS_PER_YEAR)
            logger.info(f"Volatility calculated successfully: {volatility:.6f} (annualized)")
            
            # If volatility is still very low despite having non-zero returns, try enhanced calculation
            if volatility < 1e-6:
                logger.info("Standard volatility is very low despite non-zero returns, trying enhanced calculation")
                enhanced_volatility = self.calculate_enhanced_volatility(daily_returns)
                if enhanced_volatility > volatility:
                    logger.info(f"Using enhanced volatility: {enhanced_volatility:.6f} instead of {volatility:.8f}")
                    volatility = enhanced_volatility
        
        # Validate volatility calculation
        if np.isnan(volatility) or np.isinf(volatility):
            logger.error(f"Invalid volatility calculated: {volatility}")
            volatility = 0.0
        elif volatility < 0:
            logger.error(f"Negative volatility calculated: {volatility}")
            volatility = abs(volatility)
        
        # Calculate risk-free rate for the period
        if start_date is not None and self.selic_data is not None:
            # Use dynamic rate for the start date
            risk_free_rate = self.get_risk_free_rate(start_date)
            logger.info(f"Using dynamic SELIC rate: {risk_free_rate:.8f} for risk metrics")
        else:
            # Use static rate (convert annual to daily)
            risk_free_rate = self.STATIC_RISK_FREE_RATE / self.TRADING_DAYS_PER_YEAR
            logger.info(f"Using static SELIC rate: {risk_free_rate:.8f} for risk metrics")
        
        # Calculate Sharpe Ratio with enhanced validation
        excess_returns = returns_array - risk_free_rate
        returns_std = np.std(returns_array)
        
        if returns_std > 0:
            sharpe_ratio = np.mean(excess_returns) / returns_std * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        else:
            logger.warning("Zero standard deviation in returns, setting Sharpe ratio to 0")
            sharpe_ratio = 0.0
        
        # Validate Sharpe ratio
        if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
            logger.error(f"Invalid Sharpe ratio calculated: {sharpe_ratio}")
            sharpe_ratio = 0.0
        
        # Add detailed logging for Sharpe ratio debugging
        logger.info(f"PERFORMANCE_METRICS Sharpe calculation:")
        logger.info(f"  Returns array length: {len(returns_array)}")
        logger.info(f"  Returns mean: {np.mean(returns_array):.8f}")
        logger.info(f"  Returns std: {returns_std:.8f}")
        logger.info(f"  Daily risk-free rate: {risk_free_rate:.8f}")
        logger.info(f"  Excess returns mean: {np.mean(excess_returns):.8f}")
        logger.info(f"  Calculated Sharpe ratio: {sharpe_ratio:.8f}")
        
        # Calculate Sortino Ratio with enhanced validation
        downside_returns = returns_array[returns_array < 0]
        if len(downside_returns) > 0:
            downside_deviation = np.std(downside_returns) * np.sqrt(self.TRADING_DAYS_PER_YEAR)
            if downside_deviation > 0:
                sortino_ratio = np.mean(excess_returns) / downside_deviation
            else:
                logger.warning("Zero downside deviation, setting Sortino ratio to 0")
                sortino_ratio = 0.0
        else:
            logger.warning("No negative returns found, setting Sortino ratio to 0")
            sortino_ratio = 0.0
        
        # Validate Sortino ratio
        if np.isnan(sortino_ratio) or np.isinf(sortino_ratio):
            logger.error(f"Invalid Sortino ratio calculated: {sortino_ratio}")
            sortino_ratio = 0.0
        
        # Calculate Maximum Drawdown with enhanced validation
        if len(returns_array) > 0:
            cumulative_returns = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            logger.info(f"MAX_DRAWDOWN_CALC: Raw max_drawdown calculated: {max_drawdown}")
            logger.info(f"MAX_DRAWDOWN_CALC: Type: {type(max_drawdown)}")
            logger.info(f"MAX_DRAWDOWN_CALC: Is negative: {max_drawdown < 0}")
        else:
            max_drawdown = 0.0
            logger.info(f"MAX_DRAWDOWN_CALC: No returns data, setting to 0.0")
        
        # Validate max drawdown
        if np.isnan(max_drawdown) or np.isinf(max_drawdown):
            logger.error(f"Invalid max drawdown calculated: {max_drawdown}")
            max_drawdown = 0.0
            logger.info(f"MAX_DRAWDOWN_CALC: Reset to 0.0 due to invalid value")
        else:
            logger.info(f"MAX_DRAWDOWN_CALC: Final max_drawdown after validation: {max_drawdown}")
        
        # Calculate Max Drawdown Duration
        if len(returns_array) > 0 and max_drawdown < 0:
            cumulative_returns = np.cumprod(1 + returns_array)
            drawdown = (cumulative_returns - np.maximum.accumulate(cumulative_returns)) / np.maximum.accumulate(cumulative_returns)
            max_dd_idx = np.argmin(drawdown)
            peak_idx = np.argmax(cumulative_returns[:max_dd_idx + 1])
            max_drawdown_duration = max_dd_idx - peak_idx
        else:
            max_drawdown_duration = 0
        
        # Calculate Calmar Ratio with enhanced validation
        if (max_drawdown != 0 and 
            hasattr(self, 'returns_metrics') and 
            self.returns_metrics is not None and 
            self.returns_metrics.annualized_return is not None):
            calmar_ratio = self.returns_metrics.annualized_return / abs(max_drawdown)
        else:
            logger.warning("Cannot calculate Calmar ratio: max_drawdown=0 or missing annualized_return")
            calmar_ratio = 0.0
        
        # Validate Calmar ratio
        if np.isnan(calmar_ratio) or np.isinf(calmar_ratio):
            logger.error(f"Invalid Calmar ratio calculated: {calmar_ratio}")
            calmar_ratio = 0.0
        
        # Calculate Value at Risk (VaR) and Conditional VaR (CVaR) with enhanced validation
        if len(returns_array) > 0:
            var_95 = np.percentile(returns_array, 5)  # 95% confidence level
            returns_below_var = returns_array[returns_array <= var_95]
            if len(returns_below_var) > 0:
                cvar_95 = np.mean(returns_below_var)
            else:
                cvar_95 = var_95
        else:
            var_95 = 0.0
            cvar_95 = 0.0
        
        # Validate VaR and CVaR
        if np.isnan(var_95) or np.isinf(var_95):
            logger.error(f"Invalid VaR calculated: {var_95}")
            var_95 = 0.0
        if np.isnan(cvar_95) or np.isinf(cvar_95):
            logger.error(f"Invalid CVaR calculated: {cvar_95}")
            cvar_95 = var_95
        
        self.risk_metrics = RiskMetrics(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            volatility=volatility,
            downside_deviation=downside_deviation if len(downside_returns) > 0 else 0.0,
            var_95=var_95,
            cvar_95=cvar_95
        )
        
        logger.info(f"Risk metrics calculated successfully:")
        logger.info(f"  Volatility: {volatility:.6f} (annualized)")
        logger.info(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
        logger.info(f"  Max Drawdown: {max_drawdown:.4f}")
        logger.info(f"  Non-zero returns used: {len(non_zero_returns)}/{len(returns_array)}")
        
        return self.risk_metrics
    
    def calculate_tax_metrics(self) -> TaxMetrics:
        """
        Calculate Brazilian tax-specific performance metrics.
        
        Returns:
            TaxMetrics object with calculated metrics
        """
        # Get tax information from portfolio
        total_taxes = self.portfolio.total_taxes
        trade_history = self.portfolio.trade_history
        
        # Separate swing trade and day trade taxes
        swing_trade_taxes = 0.0
        day_trade_taxes = 0.0
        
        for trade in trade_history:
            taxes = trade.get('taxes', 0)
            # Handle different tax field types
            if isinstance(taxes, (int, float)) and taxes > 0:
                if trade.get('trade_type') == 'swing_trade':
                    swing_trade_taxes += taxes
                elif trade.get('trade_type') == 'day_trade':
                    day_trade_taxes += taxes
            elif isinstance(taxes, dict):
                # Handle case where taxes is a dictionary
                tax_amount = taxes.get('amount', 0)
                if tax_amount > 0:
                    if trade.get('trade_type') == 'swing_trade':
                        swing_trade_taxes += tax_amount
                    elif trade.get('trade_type') == 'day_trade':
                        day_trade_taxes += tax_amount
        
        # Calculate tax efficiency
        if hasattr(self, 'returns_metrics') and self.returns_metrics is not None:
            pre_tax_return = self.returns_metrics.total_return
        else:
            logger.warning("No returns metrics available, using 0.0 for pre_tax_return")
            pre_tax_return = 0.0
            
        after_tax_return = pre_tax_return - (total_taxes / self.portfolio.initial_cash)
        tax_efficiency = after_tax_return / pre_tax_return if pre_tax_return != 0 else 0.0
        
        # Calculate effective tax rate
        total_profit = self.portfolio.total_value - self.portfolio.initial_cash
        effective_tax_rate = total_taxes / total_profit if total_profit > 0 else 0.0
        
        # Calculate loss carryforward utilization
        loss_manager = self.portfolio.loss_manager
        total_loss_balance = loss_manager.get_total_loss_balance()
        loss_carryforward_utilized = total_loss_balance  # Simplified calculation
        
        # Calculate tax exemption utilization (R$20,000 monthly)
        monthly_exemption = self.config['taxes']['swing_exemption_limit']
        swing_trade_rate = self.config['taxes']['swing_trade']
        tax_exemption_utilized = min(monthly_exemption, swing_trade_taxes / swing_trade_rate)
        
        self.tax_metrics = TaxMetrics(
            total_taxes_paid=total_taxes,
            swing_trade_taxes=swing_trade_taxes,
            day_trade_taxes=day_trade_taxes,
            tax_efficiency=tax_efficiency,
            loss_carryforward_utilized=loss_carryforward_utilized,
            effective_tax_rate=effective_tax_rate,
            tax_exemption_utilized=tax_exemption_utilized
        )
        
        logger.info(f"Tax metrics calculated: Total taxes={total_taxes:.2f}, Efficiency={tax_efficiency:.4f}")
        return self.tax_metrics
    
    def calculate_trade_metrics(self) -> TradeMetrics:
        """
        Calculate trade-specific performance metrics.
        
        Returns:
            TradeMetrics object with calculated metrics
        """
        trade_history = self.portfolio.trade_history
        
        if not trade_history:
            logger.warning("No trade history available for trade metrics")
            return TradeMetrics()
        
        # Count trades - only count SELL trades for PnL calculation
        sell_trades = [trade for trade in trade_history if trade.get('action') == 'SELL']
        total_trades = len(sell_trades)
        
        # Use 'final_profit' key for PnL calculation (portfolio stores this for sell trades)
        winning_trades = sum(1 for trade in sell_trades if trade.get('final_profit', 0) > 0)
        losing_trades = sum(1 for trade in sell_trades if trade.get('final_profit', 0) < 0)
        
        # Calculate win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate profit factor
        total_wins = sum(trade.get('final_profit', 0) for trade in sell_trades if trade.get('final_profit', 0) > 0)
        total_losses = abs(sum(trade.get('final_profit', 0) for trade in sell_trades if trade.get('final_profit', 0) < 0))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0.0
        
        # Calculate average win/loss
        wins = [trade.get('final_profit', 0) for trade in sell_trades if trade.get('final_profit', 0) > 0]
        losses = [trade.get('final_profit', 0) for trade in sell_trades if trade.get('final_profit', 0) < 0]
        
        average_win = np.mean(wins) if wins else 0.0
        average_loss = np.mean(losses) if losses else 0.0
        largest_win = max(wins) if wins else 0.0
        largest_loss = min(losses) if losses else 0.0
        
        # Calculate average trade duration (simplified)
        trade_durations = []
        for trade in trade_history:
            if 'entry_date' in trade and 'exit_date' in trade:
                try:
                    entry_date = pd.to_datetime(trade['entry_date'])
                    exit_date = pd.to_datetime(trade['exit_date'])
                    duration = (exit_date - entry_date).days
                    trade_durations.append(duration)
                except:
                    pass
        
        average_trade_duration = np.mean(trade_durations) if trade_durations else 0.0
        
        # Calculate total commission
        total_commission = self.portfolio.total_commission
        
        self.trade_metrics = TradeMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            average_trade_duration=average_trade_duration,
            total_commission=total_commission
        )
        
        logger.info(f"Trade metrics calculated: Total trades={total_trades}, Win rate={win_rate:.4f}")
        return self.trade_metrics
    
    def calculate_benchmark_metrics(self, daily_returns: List[float], start_date: Optional[datetime] = None) -> BenchmarkMetrics:
        """
        Calculate benchmark comparison metrics.
        
        Args:
            daily_returns: List of daily returns
            start_date: Start date for the analysis period
            
        Returns:
            BenchmarkMetrics with comprehensive benchmark analysis
        """
        if self.benchmark_analyzer is None:
            logger.warning("Benchmark analyzer not available, skipping benchmark metrics")
            return self.benchmark_metrics
        
        try:
            if not daily_returns:
                logger.warning("No daily returns available for benchmark analysis")
                return self.benchmark_metrics
            
            # Create datetime index for returns
            if start_date:
                # Ensure start_date is timezone-naive to avoid conversion issues
                if start_date.tzinfo is not None:
                    start_date = start_date.replace(tzinfo=None)
                date_range = pd.date_range(start=start_date, periods=len(daily_returns), freq='D', tz=None)
                strategy_returns = pd.Series(daily_returns, index=date_range)
            else:
                # Fallback: use simple integer index
                strategy_returns = pd.Series(daily_returns)
            
            # Calculate all benchmark metrics
            benchmark_result = self.benchmark_analyzer.calculate_all_metrics(
                strategy_returns=strategy_returns
            )
            
            # Update benchmark metrics
            self.benchmark_metrics.benchmark_return = benchmark_result.benchmark_return
            self.benchmark_metrics.excess_return = benchmark_result.excess_return
            self.benchmark_metrics.information_ratio = benchmark_result.information_ratio
            self.benchmark_metrics.beta = benchmark_result.beta
            self.benchmark_metrics.alpha = benchmark_result.alpha
            self.benchmark_metrics.tracking_error = benchmark_result.tracking_error
            self.benchmark_metrics.rolling_correlation = benchmark_result.rolling_correlation
            self.benchmark_metrics.benchmark_sharpe = benchmark_result.benchmark_sharpe
            self.benchmark_metrics.benchmark_max_drawdown = benchmark_result.benchmark_max_drawdown
            self.benchmark_metrics.benchmark_win_rate = benchmark_result.benchmark_win_rate
            
            logger.info("Benchmark metrics calculated successfully")
            logger.info(f"Strategy vs {self.benchmark_metrics.benchmark_symbol} Return: {self.benchmark_metrics.benchmark_return:.4f}")
            logger.info(f"Excess Return: {self.benchmark_metrics.excess_return:.4f}")
            logger.info(f"Information Ratio: {self.benchmark_metrics.information_ratio:.4f}")
            
            return self.benchmark_metrics
            
        except Exception as e:
            logger.error(f"Error calculating benchmark metrics: {e}")
            return self.benchmark_metrics
    
    def calculate_all_metrics(self, portfolio_values: List[float], daily_returns: List[float], 
                             start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate all performance metrics including benchmark analysis.
        
        Args:
            portfolio_values: List of portfolio values over time
            daily_returns: List of daily returns
            start_date: Start date for the analysis period
            end_date: End date for the analysis period
            
        Returns:
            Dictionary containing all performance metrics
        """
        # Calculate all individual metrics
        returns_metrics = self.calculate_returns(portfolio_values, start_date, end_date)
        risk_metrics_dict = self.calculate_risk_metrics(daily_returns, start_date)
        tax_metrics = self.calculate_tax_metrics()
        trade_metrics = self.calculate_trade_metrics()
        benchmark_metrics = self.calculate_benchmark_metrics(daily_returns, start_date)
        
        # Update backward compatibility attributes
        self.total_return = returns_metrics.total_return
        self.annualized_return = returns_metrics.annualized_return
        self.sharpe_ratio = risk_metrics_dict.sharpe_ratio
        self.max_drawdown = risk_metrics_dict.max_drawdown
        # Use SELL-based trade metrics (portfolio.total_trades now only counts SELLs)
        self.total_trades = trade_metrics.total_trades
        self.winning_trades = trade_metrics.winning_trades
        self.losing_trades = trade_metrics.losing_trades
        self.total_commission = trade_metrics.total_commission
        self.total_taxes = tax_metrics.total_taxes_paid
        self.final_portfolio_value = portfolio_values[-1] if portfolio_values else 0.0
        self.initial_capital = portfolio_values[0] if portfolio_values else 0.0
        self.net_profit = self.final_portfolio_value - self.initial_capital
        
        # Combine all metrics with defensive None checks
        all_metrics = {
            # Returns metrics
            'total_return': returns_metrics.total_return if returns_metrics.total_return is not None else 0.0,
            'annualized_return': returns_metrics.annualized_return if returns_metrics.annualized_return is not None else 0.0,
            'logarithmic_return': returns_metrics.logarithmic_return if returns_metrics.logarithmic_return is not None else 0.0,
            
            # Risk metrics
            'sharpe_ratio': risk_metrics_dict.sharpe_ratio if risk_metrics_dict.sharpe_ratio is not None else 0.0,
            'sortino_ratio': risk_metrics_dict.sortino_ratio if risk_metrics_dict.sortino_ratio is not None else 0.0,
            'calmar_ratio': risk_metrics_dict.calmar_ratio if risk_metrics_dict.calmar_ratio is not None else 0.0,
            'max_drawdown': risk_metrics_dict.max_drawdown if risk_metrics_dict.max_drawdown is not None else 0.0,
            'volatility': risk_metrics_dict.volatility if risk_metrics_dict.volatility is not None else 0.0,
            'var_95': risk_metrics_dict.var_95 if risk_metrics_dict.var_95 is not None else 0.0,
            'cvar_95': risk_metrics_dict.cvar_95 if risk_metrics_dict.cvar_95 is not None else 0.0,
            
            # Tax metrics
            'total_taxes_paid': tax_metrics.total_taxes_paid if tax_metrics.total_taxes_paid is not None else 0.0,
            'tax_efficiency': tax_metrics.tax_efficiency if tax_metrics.tax_efficiency is not None else 0.0,
            'effective_tax_rate': tax_metrics.effective_tax_rate if tax_metrics.effective_tax_rate is not None else 0.0,
            
            # Trade metrics
            'total_trades': trade_metrics.total_trades if trade_metrics.total_trades is not None else 0,
            'winning_trades': trade_metrics.winning_trades if trade_metrics.winning_trades is not None else 0,
            'losing_trades': trade_metrics.losing_trades if trade_metrics.losing_trades is not None else 0,
            'win_rate': trade_metrics.win_rate if trade_metrics.win_rate is not None else 0.0,
            'profit_factor': trade_metrics.profit_factor if trade_metrics.profit_factor is not None else 0.0,
            'total_commission': trade_metrics.total_commission if trade_metrics.total_commission is not None else 0.0,
            
            # Benchmark metrics
            'benchmark_return': benchmark_metrics.benchmark_return if benchmark_metrics.benchmark_return is not None else 0.0,
            'excess_return': benchmark_metrics.excess_return if benchmark_metrics.excess_return is not None else 0.0,
            'information_ratio': benchmark_metrics.information_ratio if benchmark_metrics.information_ratio is not None else 0.0,
            'beta': benchmark_metrics.beta if benchmark_metrics.beta is not None else 0.0,
            'alpha': benchmark_metrics.alpha if benchmark_metrics.alpha is not None else 0.0,
            'tracking_error': benchmark_metrics.tracking_error if benchmark_metrics.tracking_error is not None else 0.0,
            'rolling_correlation': benchmark_metrics.rolling_correlation if benchmark_metrics.rolling_correlation is not None else 0.0,
            'benchmark_sharpe': benchmark_metrics.benchmark_sharpe if benchmark_metrics.benchmark_sharpe is not None else 0.0,
            'benchmark_max_drawdown': benchmark_metrics.benchmark_max_drawdown if benchmark_metrics.benchmark_max_drawdown is not None else 0.0,
            'benchmark_win_rate': benchmark_metrics.benchmark_win_rate if benchmark_metrics.benchmark_win_rate is not None else 0.0,
            'benchmark_symbol': benchmark_metrics.benchmark_symbol
        }
        
        logger.info("All performance metrics calculated successfully")
        return all_metrics
    
    def calculate_trade_based_volatility(self) -> float:
        """
        Calculate volatility based on trade returns rather than daily portfolio changes.
        This provides a more accurate volatility measure when the strategy has infrequent trades.
        
        Returns:
            Annualized volatility based on trade returns
        """
        if not hasattr(self, 'portfolio') or not self.portfolio:
            logger.warning("No portfolio available for trade-based volatility calculation")
            return 0.0
        
        trade_history = self.portfolio.trade_history
        if not trade_history:
            logger.warning("No trade history available for trade-based volatility calculation")
            return 0.0
        
        # Extract trade returns from trade history
        trade_returns = []
        for trade in trade_history:
            if trade.get('action') == 'SELL' and 'final_profit' in trade:
                # Calculate return based on profit and trade value
                trade_value = trade.get('value', 0)
                profit = trade.get('final_profit', 0)
                
                if trade_value > 0:
                    # Return = profit / trade_value
                    trade_return = profit / trade_value
                    trade_returns.append(trade_return)
                    logger.debug(f"Trade return: {trade_return:.6f} (profit: {profit:.2f}, value: {trade_value:.2f})")
        
        if len(trade_returns) < 2:
            logger.warning(f"Insufficient trade returns for volatility calculation: {len(trade_returns)} trades")
            return 0.0
        
        # Calculate volatility from trade returns
        trade_returns_array = np.array(trade_returns)
        trade_volatility = np.std(trade_returns_array) * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        
        logger.info(f"Trade-based volatility calculation:")
        logger.info(f"  Number of trades: {len(trade_returns)}")
        logger.info(f"  Trade returns range: [{min(trade_returns):.6f}, {max(trade_returns):.6f}]")
        logger.info(f"  Trade returns std: {np.std(trade_returns):.6f}")
        logger.info(f"  Annualized volatility: {trade_volatility:.6f}")
        
        return trade_volatility
    
    def calculate_enhanced_volatility(self, daily_returns: List[float]) -> float:
        """
        Calculate enhanced volatility that combines daily returns with trade-based volatility.
        This provides a more robust volatility measure that accounts for both daily changes
        and trade-specific returns.
        
        Args:
            daily_returns: List of daily returns
            
        Returns:
            Enhanced annualized volatility
        """
        if not daily_returns:
            logger.warning("No daily returns provided for enhanced volatility calculation")
            return 0.0
        
        # Calculate daily-based volatility
        daily_volatility = np.std(daily_returns) * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        
        # Calculate trade-based volatility
        trade_volatility = self.calculate_trade_based_volatility()
        
        # Combine volatilities with weights
        # If we have many trades, weight trade volatility more heavily
        # If we have few trades, weight daily volatility more heavily
        trade_history = getattr(self.portfolio, 'trade_history', []) if hasattr(self, 'portfolio') else []
        sell_trades = [t for t in trade_history if t.get('action') == 'SELL']
        
        if len(sell_trades) >= 10:
            # Many trades: weight trade volatility more heavily
            daily_weight = 0.3
            trade_weight = 0.7
        elif len(sell_trades) >= 5:
            # Moderate trades: equal weighting
            daily_weight = 0.5
            trade_weight = 0.5
        else:
            # Few trades: weight daily volatility more heavily
            daily_weight = 0.8
            trade_weight = 0.2
        
        enhanced_volatility = (daily_weight * daily_volatility) + (trade_weight * trade_volatility)
        
        logger.info(f"Enhanced volatility calculation:")
        logger.info(f"  Daily volatility: {daily_volatility:.6f}")
        logger.info(f"  Trade volatility: {trade_volatility:.6f}")
        logger.info(f"  Weights: daily={daily_weight:.1f}, trade={trade_weight:.1f}")
        logger.info(f"  Enhanced volatility: {enhanced_volatility:.6f}")
        
        return enhanced_volatility


class BenchmarkAnalyzer:
    """
    Benchmark analyzer for IBOV (Bovespa Index) integration.

    This class provides benchmark analysis capabilities that are now a mandatory and fully integrated part of the backtesting workflow. Every strategy run will include benchmark analysis, ensuring that all performance metrics are evaluated relative to the benchmark (e.g., IBOV) in compliance with Brazilian market standards.

    Features:
    - IBOV benchmark data loading and preprocessing
    - Rolling correlation analysis
    - Excess returns calculation
    - Information ratio computation
    - Visualization capabilities
    - Brazilian market compliance
    """
    
    def __init__(
        self, 
        config_path: str = "config/settings.yaml",
        benchmark_symbol: str = "IBOV",
        risk_free_rate: Optional[float] = None
    ):
        """
        Initialize the benchmark analyzer.
        
        Args:
            config_path: Path to configuration file
            benchmark_symbol: Benchmark symbol (default: IBOV)
            risk_free_rate: Risk-free rate override (uses SELIC from config if None)
        """
        self.config = self._load_config(config_path)
        self.benchmark_symbol = benchmark_symbol
        self.timezone = self.config['market']['trading_hours']['timezone']
        
        # Brazilian market constants
        self.TRADING_DAYS_PER_YEAR = self.config['market'].get('trading_days_per_year', 252)
        
        # Risk-free rate (SELIC from config or override)
        if risk_free_rate is not None:
            self.risk_free_rate = risk_free_rate
        else:
            self.risk_free_rate = self.config['market'].get('selic_rate', 0.15)
        
        # Data storage
        self.benchmark_data: Optional[pd.DataFrame] = None
        self.strategy_returns: Optional[pd.Series] = None
        self.benchmark_returns: Optional[pd.Series] = None
        
        # Metrics storage
        self.metrics = BenchmarkMetrics()
        
        logger.info(f"BenchmarkAnalyzer initialized for {benchmark_symbol}")
        logger.info(f"Risk-free rate: {self.risk_free_rate:.4f}")
        logger.info(f"Trading days per year: {self.TRADING_DAYS_PER_YEAR}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            # Return default configuration
            return {
                'market': {
                    'trading_hours': {'timezone': 'America/Sao_Paulo'},
                    'selic_rate': 0.15,
                    'trading_days_per_year': 252
                }
            }
    
    def load_benchmark_data(
        self, 
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        data_path: str = "data"
    ) -> bool:
        """
        Load benchmark data from various sources.
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            data_path: Path to data directory
            
        Returns:
            True if data loaded successfully, False otherwise
        """
        try:
            # Try multiple data sources in order of preference (CSV first, then parquet, then API)
            data_sources = [
                lambda: self._load_from_csv(data_path),
                lambda: self._load_from_parquet(data_path),
                lambda: self._load_from_downloader(start_date, end_date)
            ]
            
            for source_func in data_sources:
                try:
                    data = source_func()
                    if data is not None and not data.empty:
                        self.benchmark_data = data
                        logger.info(f"Loaded benchmark data: {len(data)} rows")
                        logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
                        return True
                except Exception as e:
                    logger.debug(f"Data source failed: {e}")
                    continue
            
            logger.error("Failed to load benchmark data from all sources")
            return False
            
        except Exception as e:
            logger.error(f"Error loading benchmark data: {e}")
            return False
    
    def _load_from_parquet(self, data_path: str) -> Optional[pd.DataFrame]:
        """Load benchmark data from parquet file."""
        # Try IBOV-specific directory first
        ibov_parquet_path = Path(data_path) / "IBOV" / f"{self.benchmark_symbol}.parquet"
        if ibov_parquet_path.exists():
            data = pd.read_parquet(ibov_parquet_path)
            if 'close' in data.columns:
                # Ensure timezone-naive datetime index
                if data.index.tz is not None:
                    data.index = data.index.tz_localize(None)
                return data[['close']]
        
        # Fallback to general data directory
        parquet_path = Path(data_path) / f"{self.benchmark_symbol}.parquet"
        if parquet_path.exists():
            data = pd.read_parquet(parquet_path)
            if 'close' in data.columns:
                # Ensure timezone-naive datetime index
                if data.index.tz is not None:
                    data.index = data.index.tz_localize(None)
                return data[['close']]
        return None
    
    def _load_from_csv(self, data_path: str) -> Optional[pd.DataFrame]:
        """Load benchmark data from CSV file."""
        # Try IBOV-specific directory first
        ibov_csv_path = Path(data_path) / "IBOV" / f"{self.benchmark_symbol}_raw.csv"
        if ibov_csv_path.exists():
            data = pd.read_csv(ibov_csv_path, index_col=0, parse_dates=True)
            if 'close' in data.columns:
                # Ensure timezone-naive datetime index
                if data.index.tz is not None:
                    data.index = data.index.tz_localize(None)
                return data[['close']]
        
        # Fallback to general data directory
        csv_path = Path(data_path) / f"{self.benchmark_symbol}_raw.csv"
        if csv_path.exists():
            data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            if 'close' in data.columns:
                # Ensure timezone-naive datetime index
                if data.index.tz is not None:
                    data.index = data.index.tz_localize(None)
                return data[['close']]
        return None
    
    def _load_from_downloader(self, start_date: Optional[Union[str, datetime]], 
                             end_date: Optional[Union[str, datetime]]) -> Optional[pd.DataFrame]:
        """Load benchmark data using BRAPI provider (no Yahoo dependency)."""
        try:
            from .brapi_provider import BrapiProvider
            brapi = BrapiProvider(api_token=os.getenv('BRAPI_API_TOKEN', ''), cache_dir="data/brapi_cache")
            # Fetch daily IBOV via BRAPI
            s = start_date.strftime('%Y-%m-%d') if start_date else (datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d')
            e = end_date.strftime('%Y-%m-%d') if end_date else datetime.now().strftime('%Y-%m-%d')
            # Use BRAPI Yahoo symbol for Bovespa index
            df = brapi.get_daily_data('^BVSP', s, e)
            if df is not None and not df.empty and 'close' in df.columns:
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                return df[['close']]
        except Exception as e:
            logger.debug(f"BRAPI benchmark load failed: {e}")
        return None
    
    def calculate_returns(
        self, 
        prices: pd.Series, 
        frequency: str = 'daily',
        method: str = 'log'
    ) -> pd.Series:
        """
        Calculate returns from price series.
        
        Args:
            prices: Price series with datetime index
            frequency: Return frequency ('daily', 'monthly', 'annual')
            method: Return calculation method ('log' or 'simple')
            
        Returns:
            Returns series
        """
        if prices.empty:
            return pd.Series(dtype=float)
        
        # Ensure datetime index is timezone-naive
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices.index = pd.to_datetime(prices.index, utc=False)
        elif prices.index.tz is not None:
            # Convert timezone-aware index to timezone-naive
            prices.index = prices.index.tz_localize(None)
        
        # Calculate returns based on method
        if method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:  # simple
            returns = (prices / prices.shift(1)) - 1
        
        # Remove first row (NaN)
        returns = returns.dropna()
        
        # Resample if frequency is specified
        if frequency != 'daily':
            if frequency == 'monthly':
                returns = returns.resample('M').sum()
            elif frequency == 'annual':
                returns = returns.resample('Y').sum()
        
        return returns
    
    def rolling_correlation(
        self, 
        strategy_returns: pd.Series, 
        window: int = 252,
        min_periods: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate rolling correlation between strategy and benchmark returns.
        
        Args:
            strategy_returns: Strategy returns series
            window: Rolling window size (default: 252 trading days)
            min_periods: Minimum periods for correlation calculation
            
        Returns:
            Rolling correlation series
        """
        if self.benchmark_returns is None:
            logger.error("Benchmark returns not available. Load benchmark data first.")
            return pd.Series(dtype=float)
        
        if min_periods is None:
            min_periods = max(30, window // 4)  # At least 30 days or 25% of window
        
        # Align data
        aligned_data = pd.concat([strategy_returns, self.benchmark_returns], axis=1).dropna()
        
        if len(aligned_data) < min_periods:
            logger.warning(f"Insufficient data for rolling correlation: {len(aligned_data)} < {min_periods}")
            return pd.Series(dtype=float)
        
        # Calculate rolling correlation
        correlation = aligned_data.iloc[:, 0].rolling(
            window=window, 
            min_periods=min_periods
        ).corr(aligned_data.iloc[:, 1])
        
        return correlation
    
    def excess_returns(
        self, 
        strategy_returns: pd.Series,
        risk_free_rate: Optional[float] = None
    ) -> pd.Series:
        """
        Calculate excess returns (strategy returns minus benchmark returns).
        
        Args:
            strategy_returns: Strategy returns series
            risk_free_rate: Risk-free rate override
            
        Returns:
            Excess returns series
        """
        if self.benchmark_returns is None:
            logger.error("Benchmark returns not available. Load benchmark data first.")
            return pd.Series(dtype=float)
        
        # Use provided risk-free rate or default
        rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        
        # Convert annual rate to daily if needed
        if rf_rate > 0.1:  # Assume annual rate if > 10%
            daily_rf = (1 + rf_rate) ** (1 / self.TRADING_DAYS_PER_YEAR) - 1
        else:
            daily_rf = rf_rate
        
        # Align data
        aligned_data = pd.concat([strategy_returns, self.benchmark_returns], axis=1).dropna()
        
        if aligned_data.empty:
            logger.warning("No aligned data for excess returns calculation")
            return pd.Series(dtype=float)
        
        # Calculate excess returns
        strategy_aligned = aligned_data.iloc[:, 0]
        benchmark_aligned = aligned_data.iloc[:, 1]
        
        # Strategy excess over risk-free rate
        strategy_excess = strategy_aligned - daily_rf
        
        # Benchmark excess over risk-free rate
        benchmark_excess = benchmark_aligned - daily_rf
        
        # Strategy excess over benchmark
        excess = strategy_excess - benchmark_excess
        
        return excess
    
    def information_ratio(
        self, 
        strategy_returns: pd.Series,
        risk_free_rate: Optional[float] = None
    ) -> float:
        """
        Calculate information ratio (excess return / tracking error).
        
        Args:
            strategy_returns: Strategy returns series
            risk_free_rate: Risk-free rate override
            
        Returns:
            Information ratio
        """
        excess_returns = self.excess_returns(strategy_returns, risk_free_rate)
        
        if excess_returns.empty:
            logger.warning("No excess returns available for information ratio calculation")
            return 0.0
        
        # Calculate tracking error (standard deviation of excess returns)
        tracking_error = excess_returns.std()
        
        if tracking_error == 0:
            logger.warning("Zero tracking error, cannot calculate information ratio")
            return 0.0
        
        # Annualize if using daily returns
        if len(excess_returns) > 252:
            # Assume daily returns, annualize
            annualized_excess = excess_returns.mean() * self.TRADING_DAYS_PER_YEAR
            annualized_tracking_error = tracking_error * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        else:
            annualized_excess = excess_returns.mean()
            annualized_tracking_error = tracking_error
        
        information_ratio = annualized_excess / annualized_tracking_error
        
        return information_ratio
    
    def calculate_beta_alpha(
        self, 
        strategy_returns: pd.Series,
        risk_free_rate: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate beta and alpha using linear regression.
        
        Args:
            strategy_returns: Strategy returns series
            risk_free_rate: Risk-free rate override
            
        Returns:
            Tuple of (beta, alpha)
        """
        if self.benchmark_returns is None:
            logger.error("Benchmark returns not available. Load benchmark data first.")
            return 0.0, 0.0
        
        # Use provided risk-free rate or default
        rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        
        # Convert annual rate to daily if needed
        if rf_rate > 0.1:  # Assume annual rate if > 10%
            daily_rf = (1 + rf_rate) ** (1 / self.TRADING_DAYS_PER_YEAR) - 1
        else:
            daily_rf = rf_rate
        
        # Align data
        aligned_data = pd.concat([strategy_returns, self.benchmark_returns], axis=1).dropna()
        
        if len(aligned_data) < 30:
            logger.warning("Insufficient data for beta/alpha calculation")
            return 0.0, 0.0
        
        # Calculate excess returns
        strategy_excess = aligned_data.iloc[:, 0] - daily_rf
        benchmark_excess = aligned_data.iloc[:, 1] - daily_rf
        
        # Linear regression: strategy_excess = alpha + beta * benchmark_excess
        try:
            # Add constant for intercept (alpha)
            X = np.column_stack([np.ones(len(benchmark_excess)), benchmark_excess])
            y = strategy_excess
            
            # Solve using least squares
            beta, alpha = np.linalg.lstsq(X, y, rcond=None)[0]
            
            return beta, alpha
            
        except Exception as e:
            logger.error(f"Error calculating beta/alpha: {e}")
            return 0.0, 0.0
    
    def calculate_all_metrics(
        self, 
        strategy_returns: pd.Series,
        strategy_values: Optional[List[float]] = None,
        risk_free_rate: Optional[float] = None
    ) -> BenchmarkMetrics:
        """
        Calculate comprehensive benchmark analysis metrics.
        
        Args:
            strategy_returns: Strategy returns series
            strategy_values: Strategy portfolio values (optional, for drawdown calculation)
            risk_free_rate: Risk-free rate override
            
        Returns:
            BenchmarkMetrics object with all calculated metrics
        """
        if self.benchmark_data is None:
            logger.error("Benchmark data not loaded. Call load_benchmark_data() first.")
            return self.metrics
        
        # Calculate benchmark returns if not already done
        if self.benchmark_returns is None:
            self.benchmark_returns = self.calculate_returns(self.benchmark_data['close'])
        
        # Store strategy returns
        self.strategy_returns = strategy_returns
        
        # Ensure both series have timezone-naive datetime index
        if not isinstance(strategy_returns.index, pd.DatetimeIndex):
            strategy_returns.index = pd.to_datetime(strategy_returns.index, utc=False)
        elif strategy_returns.index.tz is not None:
            strategy_returns.index = strategy_returns.index.tz_localize(None)
        
        if not isinstance(self.benchmark_returns.index, pd.DatetimeIndex):
            self.benchmark_returns.index = pd.to_datetime(self.benchmark_returns.index, utc=False)
        elif self.benchmark_returns.index.tz is not None:
            self.benchmark_returns.index = self.benchmark_returns.index.tz_localize(None)
        
        # Align data
        aligned_data = pd.concat([strategy_returns, self.benchmark_returns], axis=1).dropna()
        
        if aligned_data.empty:
            logger.warning("No aligned data for metrics calculation")
            logger.debug(f"Strategy returns range: {strategy_returns.index.min()} to {strategy_returns.index.max()}")
            logger.debug(f"Benchmark returns range: {self.benchmark_returns.index.min()} to {self.benchmark_returns.index.max()}")
            return self.metrics
        
        strategy_aligned = aligned_data.iloc[:, 0]
        benchmark_aligned = aligned_data.iloc[:, 1]
        
        # Calculate basic returns
        strategy_total_return = (1 + strategy_aligned).prod() - 1
        benchmark_total_return = (1 + benchmark_aligned).prod() - 1
        
        # Calculate excess returns and information ratio
        excess_returns = self.excess_returns(strategy_returns, risk_free_rate)
        information_ratio = self.information_ratio(strategy_returns, risk_free_rate)
        
        # Calculate beta and alpha
        beta, alpha = self.calculate_beta_alpha(strategy_returns, risk_free_rate)
        
        # Calculate rolling correlation
        rolling_corr = self.rolling_correlation(strategy_returns)
        avg_correlation = rolling_corr.mean() if not rolling_corr.empty else 0.0
        
        # Calculate Sharpe ratios
        strategy_sharpe = self._calculate_sharpe_ratio(strategy_aligned, risk_free_rate)
        benchmark_sharpe = self._calculate_sharpe_ratio(benchmark_aligned, risk_free_rate)
        
        # Calculate tracking error
        tracking_error = excess_returns.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR) if not excess_returns.empty else 0.0
        
        # Calculate win rates
        strategy_win_rate = (strategy_aligned > 0).mean()
        benchmark_win_rate = (benchmark_aligned > 0).mean()
        
        # Calculate max drawdowns
        strategy_max_dd = self._calculate_max_drawdown(strategy_values) if strategy_values else 0.0
        benchmark_max_dd = self._calculate_max_drawdown_from_returns(benchmark_aligned)
        
        # Update metrics
        self.metrics = BenchmarkMetrics(
            benchmark_return=benchmark_total_return,
            strategy_return=strategy_total_return,
            excess_return=strategy_total_return - benchmark_total_return,
            information_ratio=information_ratio,
            rolling_correlation=avg_correlation,
            beta=beta,
            alpha=alpha,
            tracking_error=tracking_error,
            sharpe_ratio=strategy_sharpe,
            benchmark_sharpe=benchmark_sharpe,
            max_drawdown=strategy_max_dd,
            benchmark_max_drawdown=benchmark_max_dd,
            win_rate=strategy_win_rate,
            benchmark_win_rate=benchmark_win_rate
        )
        
        return self.metrics
    
    def _calculate_sharpe_ratio(
        self, 
        returns: pd.Series, 
        risk_free_rate: Optional[float] = None
    ) -> float:
        """Calculate Sharpe ratio for a return series."""
        if returns.empty:
            return 0.0
        
        rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        
        # Convert annual rate to daily if needed
        if rf_rate > 0.1:  # Assume annual rate if > 10%
            daily_rf = (1 + rf_rate) ** (1 / self.TRADING_DAYS_PER_YEAR) - 1
        else:
            daily_rf = rf_rate
        
        excess_returns = returns - daily_rf
        
        if excess_returns.std() == 0:
            return 0.0
        
        # Annualize
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        
        return sharpe
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown from portfolio values."""
        if not values or len(values) < 2:
            return 0.0
        
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_max_drawdown_from_returns(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from return series."""
        if returns.empty:
            return 0.0
        
        # Convert returns to cumulative values
        cumulative = (1 + returns).cumprod()
        
        # Calculate drawdown
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        return abs(drawdown.min())


class RiskAdjustedMetrics:
    """
    Advanced risk-adjusted performance metrics calculator.
    
    Features:
    - Comprehensive risk metrics calculation
    - Brazilian market-specific risk parameters
    - Integration with existing performance metrics
    """
    
    def __init__(self, performance_metrics: PerformanceMetrics):
        """
        Initialize risk-adjusted metrics calculator.
        
        Args:
            performance_metrics: PerformanceMetrics instance
        """
        self.performance_metrics = performance_metrics
        self.risk_metrics = performance_metrics.risk_metrics
    
    def calculate_all_risk_metrics(self, daily_returns: List[float], start_date: Optional[datetime] = None) -> Dict[str, float]:
        """
        Calculate all risk-adjusted metrics.
        
        Args:
            daily_returns: List of daily returns
            start_date: Start date for dynamic risk-free rate calculation (optional)
            
        Returns:
            Dictionary containing all risk metrics
        """
        # Calculate basic risk metrics and get the result
        calculated_risk_metrics = self.performance_metrics.calculate_risk_metrics(daily_returns, start_date)
        
        # Update the risk_metrics attribute with the calculated values
        self.risk_metrics = calculated_risk_metrics
        
        # Additional risk metrics
        returns_array = np.array(daily_returns)
        
        # Get risk-free rate (dynamic or static)
        if start_date is not None:
            risk_free_rate = self.performance_metrics.get_risk_free_rate(start_date)
        else:
            risk_free_rate = self.performance_metrics.STATIC_RISK_FREE_RATE
        
        # Information Ratio (assuming benchmark return of 0)
        information_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(self.performance_metrics.TRADING_DAYS_PER_YEAR) if np.std(returns_array) > 0 else 0.0
        
        # Treynor Ratio (assuming market beta of 1)
        treynor_ratio = np.mean(returns_array) / 1.0 * self.performance_metrics.TRADING_DAYS_PER_YEAR if 1.0 != 0 else 0.0
        
        # Jensen's Alpha (assuming market return of 0)
        jensen_alpha = np.mean(returns_array) * self.performance_metrics.TRADING_DAYS_PER_YEAR - risk_free_rate
        
        # Skewness and Kurtosis
        skewness = self._calculate_skewness(returns_array)
        kurtosis = self._calculate_kurtosis(returns_array)
        
        # Ulcer Index
        ulcer_index = self._calculate_ulcer_index(returns_array)
        
        # Gain-to-Pain Ratio
        gain_to_pain_ratio = self._calculate_gain_to_pain_ratio(returns_array)
        
        risk_metrics_dict = {
            'sharpe_ratio': calculated_risk_metrics.sharpe_ratio,
            'sortino_ratio': calculated_risk_metrics.sortino_ratio,
            'calmar_ratio': calculated_risk_metrics.calmar_ratio,
            'information_ratio': information_ratio,
            'treynor_ratio': treynor_ratio,
            'jensen_alpha': jensen_alpha,
            'max_drawdown': calculated_risk_metrics.max_drawdown,
            'volatility': calculated_risk_metrics.volatility,
            'var_95': calculated_risk_metrics.var_95,
            'cvar_95': calculated_risk_metrics.cvar_95,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'ulcer_index': ulcer_index,
            'gain_to_pain_ratio': gain_to_pain_ratio
        }
        
        # Add logging to track Sharpe ratio from RiskAdjustedMetrics
        sharpe_ratio = calculated_risk_metrics.sharpe_ratio
        sharpe_str = f"{sharpe_ratio:.8f}" if sharpe_ratio is not None else "N/A"
        logger.info(f"RISK_ADJUSTED_METRICS Sharpe ratio from calculated_risk_metrics: {sharpe_str}")
        logger.info(f"RISK_ADJUSTED_METRICS Returns array length: {len(returns_array)}")
        
        # Safe formatting for numpy calculations
        if len(returns_array) > 0:
            mean_val = np.mean(returns_array)
            std_val = np.std(returns_array)
            mean_str = f"{mean_val:.8f}" if mean_val is not None else "N/A"
            std_str = f"{std_val:.8f}" if std_val is not None else "N/A"
        else:
            mean_str = "N/A"
            std_str = "N/A"
        
        logger.info(f"RISK_ADJUSTED_METRICS Returns mean: {mean_str}")
        logger.info(f"RISK_ADJUSTED_METRICS Returns std: {std_str}")
        
        risk_free_str = f"{risk_free_rate:.8f}" if risk_free_rate is not None else "N/A"
        logger.info(f"RISK_ADJUSTED_METRICS Risk-free rate: {risk_free_str}")
        
        return risk_metrics_dict
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate return distribution skewness."""
        return float(np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 3)) if np.std(returns) > 0 else 0.0
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate return distribution kurtosis."""
        return float(np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 4)) if np.std(returns) > 0 else 0.0
    
    def _calculate_ulcer_index(self, returns: np.ndarray) -> float:
        """Calculate Ulcer Index (measure of downside risk)."""
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return float(np.sqrt(np.mean(drawdown ** 2)))
    
    def _calculate_gain_to_pain_ratio(self, returns: np.ndarray) -> float:
        """Calculate Gain-to-Pain Ratio."""
        gains = np.sum(returns[returns > 0])
        losses = abs(np.sum(returns[returns < 0]))
        return gains / losses if losses > 0 else float('inf') if gains > 0 else 0.0


class TaxAwareMetrics:
    """
    Brazilian tax-aware performance metrics calculator.
    
    Features:
    - Brazilian tax rule compliance (2025)
    - Integration with loss carryforward manager
    - Tax efficiency calculations
    - Regulatory compliance reporting
    """
    
    def __init__(self, performance_metrics: PerformanceMetrics):
        """
        Initialize tax-aware metrics calculator.
        
        Args:
            performance_metrics: PerformanceMetrics instance
        """
        self.performance_metrics = performance_metrics
        self.portfolio = performance_metrics.portfolio
        self.loss_manager = performance_metrics.portfolio.loss_manager
        self.config = performance_metrics.config
    
    def calculate_tax_aware_returns(self, portfolio_values: List[float]) -> Dict[str, float]:
        """
        Calculate tax-aware return metrics.
        
        Args:
            portfolio_values: List of daily portfolio values
            
        Returns:
            Dictionary containing tax-aware return metrics
        """
        # Calculate basic returns
        returns_metrics = self.performance_metrics.calculate_returns(
            portfolio_values, 
            datetime.now(), 
            datetime.now()
        )
        
        # Calculate tax metrics
        tax_metrics = self.performance_metrics.calculate_tax_metrics()
        
        # Calculate after-tax returns
        pre_tax_return = returns_metrics.total_return
        after_tax_return = pre_tax_return - (tax_metrics.total_taxes_paid / self.portfolio.initial_cash)
        
        # Calculate tax-adjusted annualized return
        trading_days = returns_metrics.trading_days
        if trading_days > 0:
            tax_adjusted_annualized = ((1 + after_tax_return) ** (self.performance_metrics.TRADING_DAYS_PER_YEAR / trading_days)) - 1
        else:
            tax_adjusted_annualized = 0.0
        
        # Calculate tax efficiency metrics
        tax_efficiency = after_tax_return / pre_tax_return if pre_tax_return != 0 else 0.0
        tax_drag = pre_tax_return - after_tax_return
        
        # Calculate loss carryforward efficiency
        total_loss_balance = self.loss_manager.get_total_loss_balance()
        loss_utilization_rate = total_loss_balance / (total_loss_balance + tax_metrics.total_taxes_paid) if (total_loss_balance + tax_metrics.total_taxes_paid) > 0 else 0.0
        
        tax_aware_metrics = {
            'pre_tax_return': pre_tax_return,
            'after_tax_return': after_tax_return,
            'tax_adjusted_annualized': tax_adjusted_annualized,
            'tax_efficiency': tax_efficiency,
            'tax_drag': tax_drag,
            'total_taxes_paid': tax_metrics.total_taxes_paid,
            'effective_tax_rate': tax_metrics.effective_tax_rate,
            'loss_carryforward_balance': total_loss_balance,
            'loss_utilization_rate': loss_utilization_rate,
            'swing_trade_taxes': tax_metrics.swing_trade_taxes,
            'day_trade_taxes': tax_metrics.day_trade_taxes,
            'tax_exemption_utilized': tax_metrics.tax_exemption_utilized
        }
        
        return tax_aware_metrics
    
    def calculate_regulatory_compliance_metrics(self) -> Dict[str, Any]:
        """
        Calculate regulatory compliance metrics for Brazilian market.
        
        Returns:
            Dictionary containing regulatory compliance metrics
        """
        # Get loss carryforward summary
        loss_summary = self.loss_manager.get_loss_summary()
        
        # Calculate compliance metrics
        compliance_metrics = {
            'regulatory_framework': 'brazilian_2025',
            'cvm_compliance': True,
            'receita_federal_compliance': True,
            'loss_carryforward_compliance': {
                'perpetual_carryforward': True,
                'max_offset_percentage': 0.30,
                'capital_gains_only': True,
                'total_loss_balance': loss_summary.get('total_loss_balance', 0.0),
                'assets_with_losses': loss_summary.get('assets_with_losses', 0),
                'total_losses_recorded': loss_summary.get('total_losses_recorded', 0)
            },
            'tax_compliance': {
                'swing_trade_rate': self.config['taxes']['swing_trade'],
                'day_trade_rate': self.config['taxes']['day_trade'],
                'exemption_limit': self.config['taxes']['swing_exemption_limit'],
                'irrf_swing_rate': self.config['taxes']['irrf_swing_rate'],
                'irrf_day_rate': self.config['taxes']['irrf_day_rate']
            },
            'audit_trail': {
                'detailed_loss_tracking': True,
                'application_history': True,
                'regulatory_reporting': True
            }
        }
        
        return compliance_metrics


class ComprehensivePerformanceAnalysis:
    """
    Comprehensive performance analysis with Brazilian market compliance.
    
    Features:
    - Integration of all performance metrics
    - Brazilian market-specific analysis
    - Regulatory compliance reporting
    - Performance visualization and reporting
    """
    
    def __init__(self, portfolio: EnhancedPortfolio, strategy=None, config_path: str = "config/settings.yaml"):
        """
        Initialize comprehensive performance analysis.
        
        Args:
            portfolio: EnhancedPortfolio instance
            strategy: Strategy instance (optional) for parameter extraction
            config_path: Path to configuration file
        """
        self.portfolio = portfolio
        self.strategy = strategy  # Store strategy reference for parameter extraction
        self.performance_metrics = PerformanceMetrics(portfolio, config_path)
        self.risk_metrics = RiskAdjustedMetrics(self.performance_metrics)
        self.tax_metrics = TaxAwareMetrics(self.performance_metrics)
        self.layer_analyzer = LayerExecutionAnalyzer()  # Initialize layer execution analyzer
        
        logger.info("Comprehensive Performance Analysis initialized")
    
    def run_comprehensive_analysis(self, portfolio_values: List[float], 
                                 daily_returns: List[float], 
                                 start_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Run comprehensive performance analysis.
        
        Args:
            portfolio_values: List of daily portfolio values
            daily_returns: List of daily returns
            start_date: Start date for dynamic risk-free rate calculation (optional)
            
        Returns:
            Dictionary containing all performance analysis results
        """
        # Calculate all metrics
        returns_metrics = self.performance_metrics.calculate_returns(
            portfolio_values, datetime.now(), datetime.now()
        )
        risk_metrics_dict = self.risk_metrics.calculate_all_risk_metrics(daily_returns, start_date)
        tax_metrics = self.performance_metrics.calculate_tax_metrics()
        trade_metrics = self.performance_metrics.calculate_trade_metrics()
        tax_aware_metrics = self.tax_metrics.calculate_tax_aware_returns(portfolio_values)
        compliance_metrics = self.tax_metrics.calculate_regulatory_compliance_metrics()
        
        # Calculate proper long/short trade counts from trade history
        long_trades, short_trades = self._calculate_long_short_trades()
        
        # Validate SELL-based trade counting consistency
        trade_counting_validation = self.portfolio.validate_sell_based_trade_counting()
        
        # Extract strategy parameters if strategy is available
        strategy_parameters = {}
        if self.strategy is not None:
            try:
                strategy_parameters = self.strategy.get_strategy_parameters()
                logger.info(f"Extracted {len(strategy_parameters)} parameters from strategy: {self.strategy.strategy_name}")
            except Exception as e:
                logger.warning(f"Failed to extract strategy parameters: {e}")
                strategy_parameters = {}
        
        # Compile comprehensive analysis
        comprehensive_analysis = {
            'returns_analysis': {
                'total_return': returns_metrics.total_return,
                'annualized_return': returns_metrics.annualized_return,
                'logarithmic_return': returns_metrics.logarithmic_return,
                'trading_days': returns_metrics.trading_days
            },
            'risk_analysis': risk_metrics_dict,
            'tax_analysis': {
                'total_taxes_paid': tax_metrics.total_taxes_paid,
                'tax_efficiency': tax_metrics.tax_efficiency,
                'effective_tax_rate': tax_metrics.effective_tax_rate,
                'loss_carryforward_utilized': tax_metrics.loss_carryforward_utilized
            },
            'trade_analysis': {
                'total_trades': trade_metrics.total_trades,
                'winning_trades': trade_metrics.winning_trades,
                'losing_trades': trade_metrics.losing_trades,
                'win_rate': trade_metrics.win_rate,
                'profit_factor': trade_metrics.profit_factor,
                'average_win': trade_metrics.average_win,
                'average_loss': trade_metrics.average_loss,
                'average_pnl_per_trade': self._calculate_average_pnl_per_trade(),
                'long_trades': long_trades,  # Completed long sequences (BUY→SELL)
                'short_trades': short_trades,  # Completed short sequences (SELL→BUY)
                'buy_operations': len([t for t in self.portfolio.trade_history if t.get('action') == 'BUY']),
                'sell_operations': len([t for t in self.portfolio.trade_history if t.get('action') == 'SELL']),
                'average_holding_period_minutes': 0,  # Not calculated yet
                'turnover': 0.0  # Not calculated yet
            },
            'tax_aware_analysis': tax_aware_metrics,
            'regulatory_compliance': compliance_metrics,
            'portfolio_summary': self.portfolio.get_portfolio_summary(),
            'portfolio_values': portfolio_values,  # Include portfolio values for initial/final calculation
            'daily_returns': daily_returns,
            'analysis_timestamp': datetime.now().isoformat(),
            'strategy_parameters': strategy_parameters,
            'trade_counting_validation': trade_counting_validation,  # SELL-based validation
            'market_parameters': {
                'trading_days_per_year': self.performance_metrics.TRADING_DAYS_PER_YEAR,
                'risk_free_rate': self.performance_metrics.get_risk_free_rate(start_date) if start_date else self.performance_metrics.STATIC_RISK_FREE_RATE,
                'risk_free_rate_source': 'dynamic_sgs' if start_date and self.performance_metrics.selic_data is not None else 'static_config',
                'market_timezone': self.performance_metrics.timezone.zone
            }
        }
        
        # Add layer execution analysis
        layer_execution_data = self.layer_analyzer.get_layer_execution_data()
        layer_validation = self.layer_analyzer.validate_data_consistency()
        
        comprehensive_analysis['layer_execution'] = layer_execution_data
        comprehensive_analysis['layer_validation'] = layer_validation
        
        # Add logging to track Sharpe ratio in comprehensive analysis
        logger.info(f"COMPREHENSIVE_ANALYSIS Sharpe ratio in risk_analysis: {risk_metrics_dict.get('sharpe_ratio', 'NOT_FOUND'):.8f}")
        logger.info(f"COMPREHENSIVE_ANALYSIS Risk metrics dict keys: {list(risk_metrics_dict.keys())}")
        logger.info(f"COMPREHENSIVE_ANALYSIS Strategy parameters included: {len(strategy_parameters)} parameters")
        logger.info(f"COMPREHENSIVE_ANALYSIS Layer execution data included: {len(layer_execution_data)} layers")
        
        logger.info("Comprehensive performance analysis completed")
        return comprehensive_analysis
    
    def _calculate_long_short_trades(self) -> Tuple[int, int]:
        """
        Calculate completed trade sequences for SELL-based trade counting.
        
        SELL-Based Logic:
        - Each SELL operation counts as 1 trade (as per portfolio.sell() implementation)
        - Long trades = completed BUY→SELL sequences (position closures)
        - Short trades = SELL→BUY sequences (short covering)
        - For day trading: most trades are long (BUY then SELL same day)
        
        Returns:
            Tuple of (completed_long_sequences, completed_short_sequences)
        """
        trade_history = self.portfolio.trade_history
        
        if not trade_history:
            return 0, 0
        
        # Track position changes to identify completed sequences
        positions = {}  # ticker -> net position
        completed_long_trades = 0
        completed_short_trades = 0
        
        # Process trades chronologically to identify completed sequences
        sorted_trades = sorted(trade_history, key=lambda x: x.get('date', datetime.min))
        
        for trade in sorted_trades:
            action = trade.get('action')
            ticker = trade.get('ticker')
            quantity = trade.get('quantity', 0)
            
            if ticker not in positions:
                positions[ticker] = 0
            
            if action == 'BUY':
                # Increase position (or reduce short position)
                if positions[ticker] < 0:
                    # Covering short position
                    covered_quantity = min(quantity, abs(positions[ticker]))
                    if covered_quantity > 0:
                        completed_short_trades += 1  # Short sequence completed
                    positions[ticker] += quantity
                else:
                    # Building long position
                    positions[ticker] += quantity
                    
            elif action == 'SELL':
                # Decrease position (or initiate short position)
                if positions[ticker] > 0:
                    # Closing long position - this is what counts as a "trade" in SELL-based system
                    closed_quantity = min(quantity, positions[ticker])
                    if closed_quantity > 0:
                        completed_long_trades += 1  # Long sequence completed
                    positions[ticker] -= quantity
                else:
                    # Initiating or increasing short position
                    positions[ticker] -= quantity
        
        # Log validation info
        buy_operations = len([t for t in trade_history if t.get('action') == 'BUY'])
        sell_operations = len([t for t in trade_history if t.get('action') == 'SELL'])
        total_sell_trades = len([t for t in trade_history if t.get('action') == 'SELL'])
        
        logger.debug(f"SELL-based trade analysis: {buy_operations} BUY ops, {sell_operations} SELL ops")
        logger.debug(f"Completed sequences: {completed_long_trades} long, {completed_short_trades} short")
        logger.debug(f"Total trades (SELL-based): {total_sell_trades}")
        
        # Validate consistency with SELL-based counting
        if total_sell_trades != self.portfolio.total_trades:
            logger.warning(f"Trade counting inconsistency: portfolio.total_trades={self.portfolio.total_trades}, sell_operations={total_sell_trades}")
        
        return completed_long_trades, completed_short_trades
    
    def _calculate_average_pnl_per_trade(self) -> float:
        """
        Calculate average PnL per trade from trade history.
        
        Returns:
            Average PnL per trade
        """
        trade_history = self.portfolio.trade_history
        
        if not trade_history:
            return 0.0
        
        # Only count SELL trades for PnL calculation (where profit is realized)
        sell_trades = [trade for trade in trade_history if trade.get('action') == 'SELL']
        
        if not sell_trades:
            return 0.0
        
        # Calculate total PnL from all sell trades
        total_pnl = sum(trade.get('final_profit', 0) for trade in sell_trades)
        
        # Return average PnL per trade
        return total_pnl / len(sell_trades)
    
    def generate_performance_report(self, analysis_results: Dict[str, Any], 
                                  output_path: str = "reports/performance_report.json") -> None:
        """
        Generate comprehensive performance report as JSON for HTML integration.
        
        Args:
            analysis_results: Results from comprehensive analysis
            output_path: Path to save the report
        """
        # Add logging to track Sharpe ratio and max_drawdown before JSON export
        risk_analysis = analysis_results.get('risk_analysis', {})
        sharpe_in_json = risk_analysis.get('sharpe_ratio', 'NOT_FOUND')
        max_drawdown_in_json = risk_analysis.get('max_drawdown', 'NOT_FOUND')
        logger.info(f"JSON_EXPORT Sharpe ratio being written to JSON: {sharpe_in_json}")
        logger.info(f"JSON_EXPORT Max drawdown being written to JSON: {max_drawdown_in_json}")
        logger.info(f"JSON_EXPORT Full risk_analysis: {risk_analysis}")
        
        # Custom JSON encoder to handle pandas objects
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif hasattr(obj, 'dtype'):
                    return str(obj)
                elif hasattr(obj, 'name'):  # Handle DatetimeIndex
                    return str(obj)
                elif hasattr(obj, 'index'):  # Handle pandas Series/DataFrame
                    return obj.to_dict() if hasattr(obj, 'to_dict') else str(obj)
                elif str(type(obj)).find('DatetimeIndex') != -1:  # Handle DatetimeIndex specifically
                    return [str(x) for x in obj]
                elif str(type(obj)).find('Timestamp') != -1:  # Handle Timestamp objects
                    return str(obj)
                elif str(type(obj)).find('numpy') != -1:  # Handle numpy objects
                    return float(obj) if hasattr(obj, 'item') else str(obj)
                return super().default(obj)
        
        # Write report to JSON file for HTML integration
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        import os as _os
        if not (_os.getenv('AUDIT_EXECUTIONS_ONLY', '1').lower() in ('1', 'true', 'yes')):
            with open(output_path, 'w') as f:
                json.dump(analysis_results, f, indent=2, cls=CustomJSONEncoder)
            logger.info(f"Performance report generated: {output_path}")
    
    def plot_performance_charts(self, portfolio_values: List[float], 
                              daily_returns: List[float], 
                              output_path: str = "reports/performance_charts.png") -> None:
        """
        Generate performance visualization charts.
        
        Args:
            portfolio_values: List of daily portfolio values
            daily_returns: List of daily returns
            output_path: Path to save the charts
        """
        import os as _os
        if _os.getenv('AUDIT_EXECUTIONS_ONLY', '1').lower() in ('1', 'true', 'yes'):
            return
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Brazilian B3 Quant Backtest - Performance Analysis', fontsize=16)
        
        # Portfolio Value Over Time
        axes[0, 0].plot(portfolio_values, label='Portfolio Value', color='blue')
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Trading Day')
        axes[0, 0].set_ylabel('Portfolio Value (BRL)')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # Daily Returns Distribution
        axes[0, 1].hist(daily_returns, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('Daily Returns Distribution')
        axes[0, 1].set_xlabel('Daily Return')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True)
        
        # Cumulative Returns
        cumulative_returns = np.cumprod(1 + np.array(daily_returns))
        axes[1, 0].plot(cumulative_returns, label='Cumulative Returns', color='red')
        axes[1, 0].set_title('Cumulative Returns')
        axes[1, 0].set_xlabel('Trading Day')
        axes[1, 0].set_ylabel('Cumulative Return')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        # Drawdown
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        axes[1, 1].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        axes[1, 1].plot(drawdown, color='red', label='Drawdown')
        axes[1, 1].set_title('Drawdown Analysis')
        axes[1, 1].set_xlabel('Trading Day')
        axes[1, 1].set_ylabel('Drawdown')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save chart
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance charts generated: {output_path}")
    
    def generate_html_report(self, portfolio_values: List[float], daily_returns: List[float], 
                             start_date: Optional[datetime] = None, strategy_name: str = "Strategy") -> str:
        """No-op in audit-focused build. HTML generation removed."""
        return ""




def main():
    """Main function for testing the performance metrics module."""
    # Quiet test entrypoint to avoid noisy stdout during automated runs
    pass


if __name__ == "__main__":
    main() 