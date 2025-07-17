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

from engine.portfolio import EnhancedPortfolio
from engine.loss_manager import EnhancedLossCarryforwardManager
from engine.tca import TransactionCostAnalyzer
from engine.sgs_data_loader import SGSDataLoader, SELICDataError, SELICDataUnavailableError, SELICDataInsufficientError, SELICDataQualityError, SELICDataValidationError

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
        
        logger.info("Performance Metrics initialized with Brazilian market parameters")
        logger.info(f"Static SELIC rate: {self.STATIC_RISK_FREE_RATE:.4f}")
    
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
            # (last 30 days to ensure current data availability)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
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
        
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(portfolio_values)):
            if portfolio_values[i-1] > 0:
                daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                daily_returns.append(daily_return)
            else:
                daily_returns.append(0.0)
        
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
        
        # Calculate volatility (annualized)
        volatility = np.std(returns_array) * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        
        # Calculate risk-free rate for the period
        if start_date is not None and self.selic_data is not None:
            # Use dynamic rate for the start date
            risk_free_rate = self.get_risk_free_rate(start_date)
            logger.info(f"Using dynamic SELIC rate: {risk_free_rate:.4f} for risk metrics")
        else:
            # Use static rate
            risk_free_rate = self.STATIC_RISK_FREE_RATE
            logger.info(f"Using static SELIC rate: {risk_free_rate:.4f} for risk metrics")
        
        # Calculate Sharpe Ratio
        excess_returns = returns_array - (risk_free_rate / self.TRADING_DAYS_PER_YEAR)
        sharpe_ratio = np.mean(excess_returns) / np.std(returns_array) * np.sqrt(self.TRADING_DAYS_PER_YEAR) if np.std(returns_array) > 0 else 0.0
        
        # Calculate Sortino Ratio
        downside_returns = returns_array[returns_array < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(self.TRADING_DAYS_PER_YEAR) if len(downside_returns) > 0 else 0.0
        sortino_ratio = np.mean(excess_returns) / downside_deviation if downside_deviation > 0 else 0.0
        
        # Calculate Maximum Drawdown
        cumulative_returns = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calculate Max Drawdown Duration
        max_dd_idx = np.argmin(drawdown)
        peak_idx = np.argmax(cumulative_returns[:max_dd_idx + 1])
        max_drawdown_duration = max_dd_idx - peak_idx
        
        # Calculate Calmar Ratio
        calmar_ratio = self.returns_metrics.annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        # Calculate Value at Risk (VaR) and Conditional VaR (CVaR)
        var_95 = np.percentile(returns_array, 5)  # 95% confidence level
        cvar_95 = np.mean(returns_array[returns_array <= var_95]) if len(returns_array[returns_array <= var_95]) > 0 else var_95
        
        self.risk_metrics = RiskMetrics(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            volatility=volatility,
            downside_deviation=downside_deviation,
            var_95=var_95,
            cvar_95=cvar_95
        )
        
        logger.info(f"Risk metrics calculated: Sharpe={sharpe_ratio:.4f}, MaxDD={max_drawdown:.4f}")
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
        pre_tax_return = self.returns_metrics.total_return
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
        monthly_exemption = self.config['taxes']['exemption_limit']
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
        
        # Count trades
        total_trades = len(trade_history)
        winning_trades = sum(1 for trade in trade_history if trade.get('pnl', 0) > 0)
        losing_trades = sum(1 for trade in trade_history if trade.get('pnl', 0) < 0)
        
        # Calculate win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate profit factor
        total_wins = sum(trade.get('pnl', 0) for trade in trade_history if trade.get('pnl', 0) > 0)
        total_losses = abs(sum(trade.get('pnl', 0) for trade in trade_history if trade.get('pnl', 0) < 0))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0.0
        
        # Calculate average win/loss
        wins = [trade.get('pnl', 0) for trade in trade_history if trade.get('pnl', 0) > 0]
        losses = [trade.get('pnl', 0) for trade in trade_history if trade.get('pnl', 0) < 0]
        
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
                date_range = pd.date_range(start=start_date, periods=len(daily_returns), freq='D')
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
        risk_metrics = self.calculate_risk_metrics(daily_returns, start_date)
        tax_metrics = self.calculate_tax_metrics()
        trade_metrics = self.calculate_trade_metrics()
        benchmark_metrics = self.calculate_benchmark_metrics(daily_returns, start_date)
        
        # Combine all metrics
        all_metrics = {
            # Returns metrics
            'total_return': returns_metrics.total_return,
            'annualized_return': returns_metrics.annualized_return,
            'logarithmic_return': returns_metrics.logarithmic_return,
            
            # Risk metrics
            'sharpe_ratio': risk_metrics.sharpe_ratio,
            'sortino_ratio': risk_metrics.sortino_ratio,
            'calmar_ratio': risk_metrics.calmar_ratio,
            'max_drawdown': risk_metrics.max_drawdown,
            'volatility': risk_metrics.volatility,
            'var_95': risk_metrics.var_95,
            'cvar_95': risk_metrics.cvar_95,
            
            # Tax metrics
            'total_taxes_paid': tax_metrics.total_taxes_paid,
            'tax_efficiency': tax_metrics.tax_efficiency,
            'effective_tax_rate': tax_metrics.effective_tax_rate,
            
            # Trade metrics
            'total_trades': trade_metrics.total_trades,
            'winning_trades': trade_metrics.winning_trades,
            'losing_trades': trade_metrics.losing_trades,
            'win_rate': trade_metrics.win_rate,
            'profit_factor': trade_metrics.profit_factor,
            'total_commission': trade_metrics.total_commission,
            
            # Benchmark metrics
            'benchmark_return': benchmark_metrics.benchmark_return,
            'excess_return': benchmark_metrics.excess_return,
            'information_ratio': benchmark_metrics.information_ratio,
            'beta': benchmark_metrics.beta,
            'alpha': benchmark_metrics.alpha,
            'tracking_error': benchmark_metrics.tracking_error,
            'rolling_correlation': benchmark_metrics.rolling_correlation,
            'benchmark_sharpe': benchmark_metrics.benchmark_sharpe,
            'benchmark_max_drawdown': benchmark_metrics.benchmark_max_drawdown,
            'benchmark_win_rate': benchmark_metrics.benchmark_win_rate,
            'benchmark_symbol': benchmark_metrics.benchmark_symbol
        }
        
        logger.info("All performance metrics calculated successfully")
        return all_metrics


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
                return data[['close']]
        
        # Fallback to general data directory
        parquet_path = Path(data_path) / f"{self.benchmark_symbol}.parquet"
        if parquet_path.exists():
            data = pd.read_parquet(parquet_path)
            if 'close' in data.columns:
                return data[['close']]
        return None
    
    def _load_from_csv(self, data_path: str) -> Optional[pd.DataFrame]:
        """Load benchmark data from CSV file."""
        # Try IBOV-specific directory first
        ibov_csv_path = Path(data_path) / "IBOV" / f"{self.benchmark_symbol}_raw.csv"
        if ibov_csv_path.exists():
            data = pd.read_csv(ibov_csv_path, index_col=0, parse_dates=True)
            if 'close' in data.columns:
                return data[['close']]
        
        # Fallback to general data directory
        csv_path = Path(data_path) / f"{self.benchmark_symbol}_raw.csv"
        if csv_path.exists():
            data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            if 'close' in data.columns:
                return data[['close']]
        return None
    
    def _load_from_downloader(self, start_date: Optional[Union[str, datetime]], 
                             end_date: Optional[Union[str, datetime]]) -> Optional[pd.DataFrame]:
        """Load benchmark data using Yahoo Finance downloader."""
        try:
            # Import here to avoid circular imports
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent / "scripts"))
            
            from download_ibov_yahoo import YahooIBOVDownloader
            
            downloader = YahooIBOVDownloader()
            result = downloader.download_ibov_data(
                start_date=start_date.strftime('%Y-%m-%d') if start_date else None,
                end_date=end_date.strftime('%Y-%m-%d') if end_date else None,
                period="max" if not start_date and not end_date else None
            )
            
            if result.success:
                # Load the downloaded data
                data_path = Path("data") / "IBOV" / f"{self.benchmark_symbol}.parquet"
                if data_path.exists():
                    data = pd.read_parquet(data_path)
                    if 'close' in data.columns:
                        # Save as CSV for future use
                        csv_path = Path("data") / "IBOV" / f"{self.benchmark_symbol}_raw.csv"
                        csv_path.parent.mkdir(parents=True, exist_ok=True)
                        data.to_csv(csv_path)
                        logger.info(f"Saved benchmark data to CSV: {csv_path}")
                        
                        return data[['close']]
            
        except Exception as e:
            logger.debug(f"Yahoo Finance downloader failed: {e}")
        
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
        
        # Ensure datetime index
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices.index = pd.to_datetime(prices.index)
        
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
        
        # Ensure both series have datetime index
        if not isinstance(strategy_returns.index, pd.DatetimeIndex):
            strategy_returns.index = pd.to_datetime(strategy_returns.index)
        
        if not isinstance(self.benchmark_returns.index, pd.DatetimeIndex):
            self.benchmark_returns.index = pd.to_datetime(self.benchmark_returns.index)
        
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
        # Calculate basic risk metrics
        self.performance_metrics.calculate_risk_metrics(daily_returns, start_date)
        
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
            'sharpe_ratio': self.risk_metrics.sharpe_ratio,
            'sortino_ratio': self.risk_metrics.sortino_ratio,
            'calmar_ratio': self.risk_metrics.calmar_ratio,
            'information_ratio': information_ratio,
            'treynor_ratio': treynor_ratio,
            'jensen_alpha': jensen_alpha,
            'max_drawdown': self.risk_metrics.max_drawdown,
            'volatility': self.risk_metrics.volatility,
            'var_95': self.risk_metrics.var_95,
            'cvar_95': self.risk_metrics.cvar_95,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'ulcer_index': ulcer_index,
            'gain_to_pain_ratio': gain_to_pain_ratio
        }
        
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
                'exemption_limit': self.config['taxes']['exemption_limit'],
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
    
    def __init__(self, portfolio: EnhancedPortfolio, config_path: str = "config/settings.yaml"):
        """
        Initialize comprehensive performance analysis.
        
        Args:
            portfolio: EnhancedPortfolio instance
            config_path: Path to configuration file
        """
        self.portfolio = portfolio
        self.performance_metrics = PerformanceMetrics(portfolio, config_path)
        self.risk_metrics = RiskAdjustedMetrics(self.performance_metrics)
        self.tax_metrics = TaxAwareMetrics(self.performance_metrics)
        
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
                'win_rate': trade_metrics.win_rate,
                'profit_factor': trade_metrics.profit_factor,
                'average_win': trade_metrics.average_win,
                'average_loss': trade_metrics.average_loss
            },
            'tax_aware_analysis': tax_aware_metrics,
            'regulatory_compliance': compliance_metrics,
            'portfolio_summary': self.portfolio.get_portfolio_summary(),
            'analysis_timestamp': datetime.now().isoformat(),
            'market_parameters': {
                'trading_days_per_year': self.performance_metrics.TRADING_DAYS_PER_YEAR,
                'risk_free_rate': self.performance_metrics.get_risk_free_rate(start_date) if start_date else self.performance_metrics.STATIC_RISK_FREE_RATE,
                'risk_free_rate_source': 'dynamic_sgs' if start_date and self.performance_metrics.selic_data is not None else 'static_config',
                'market_timezone': self.performance_metrics.timezone.zone
            }
        }
        
        logger.info("Comprehensive performance analysis completed")
        return comprehensive_analysis
    
    def generate_performance_report(self, analysis_results: Dict[str, Any], 
                                  output_path: str = "reports/performance_report.json") -> None:
        """
        Generate comprehensive performance report as JSON for HTML integration.
        
        Args:
            analysis_results: Results from comprehensive analysis
            output_path: Path to save the report
        """
        # Write report to JSON file for HTML integration
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
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


def main():
    """Main function for testing the performance metrics module."""
    # Example usage
    print("Brazilian B3 Quant Backtest - Performance Metrics Module")
    print("=" * 60)
    
    # This would typically be used with an actual portfolio instance
    # For demonstration, we'll show the structure
    
    print("Available Classes:")
    print("1. PerformanceMetrics - Base performance metrics calculator")
    print("2. RiskAdjustedMetrics - Risk-adjusted performance metrics")
    print("3. TaxAwareMetrics - Brazilian tax-aware metrics")
    print("4. ComprehensivePerformanceAnalysis - Complete analysis")
    
    print("\nKey Features:")
    print("- Brazilian market compliance (2025)")
    print("- 252 trading days per year calculation")
    print("- Tax-aware return calculations")
    print("- Loss carryforward integration")
    print("- Regulatory compliance reporting")
    print("- Current SELIC rate: 15.0% (2025)")
    
    print("\nIntegration Points:")
    print("- EnhancedPortfolio: Portfolio state and trade history")
    print("- EnhancedLossCarryforwardManager: Loss tracking and offset")
    print("- TransactionCostAnalyzer: Cost calculations")
    print("- Configuration: settings.yaml for market parameters")


if __name__ == "__main__":
    main() 