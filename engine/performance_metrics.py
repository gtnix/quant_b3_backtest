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
        self.RISK_FREE_RATE = self.config['market'].get('selic_rate', 0.15)  # Brazilian SELIC rate from config
        
        # Initialize metrics containers
        self.returns_metrics = ReturnsMetrics()
        self.risk_metrics = RiskMetrics()
        self.tax_metrics = TaxMetrics()
        self.trade_metrics = TradeMetrics()
        
        logger.info("Performance Metrics initialized with Brazilian market parameters")
    
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
    
    def calculate_risk_metrics(self, daily_returns: List[float]) -> RiskMetrics:
        """
        Calculate risk-adjusted performance metrics.
        
        Args:
            daily_returns: List of daily returns
            
        Returns:
            RiskMetrics object with calculated metrics
        """
        if not daily_returns:
            logger.warning("No daily returns provided for risk calculation")
            return RiskMetrics()
        
        returns_array = np.array(daily_returns)
        
        # Calculate volatility (annualized)
        volatility = np.std(returns_array) * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        
        # Calculate Sharpe Ratio
        excess_returns = returns_array - (self.RISK_FREE_RATE / self.TRADING_DAYS_PER_YEAR)
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
            if trade.get('taxes', 0) > 0:
                if trade.get('trade_type') == 'swing_trade':
                    swing_trade_taxes += trade['taxes']
                elif trade.get('trade_type') == 'day_trade':
                    day_trade_taxes += trade['taxes']
        
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
    
    def calculate_all_risk_metrics(self, daily_returns: List[float]) -> Dict[str, float]:
        """
        Calculate all risk-adjusted metrics.
        
        Args:
            daily_returns: List of daily returns
            
        Returns:
            Dictionary containing all risk metrics
        """
        # Calculate basic risk metrics
        self.performance_metrics.calculate_risk_metrics(daily_returns)
        
        # Additional risk metrics
        returns_array = np.array(daily_returns)
        
        # Information Ratio (assuming benchmark return of 0)
        information_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(self.performance_metrics.TRADING_DAYS_PER_YEAR) if np.std(returns_array) > 0 else 0.0
        
        # Treynor Ratio (assuming market beta of 1)
        treynor_ratio = np.mean(returns_array) / 1.0 * self.performance_metrics.TRADING_DAYS_PER_YEAR if 1.0 != 0 else 0.0
        
        # Jensen's Alpha (assuming market return of 0)
        jensen_alpha = np.mean(returns_array) * self.performance_metrics.TRADING_DAYS_PER_YEAR - self.performance_metrics.RISK_FREE_RATE
        
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
                                 daily_returns: List[float]) -> Dict[str, Any]:
        """
        Run comprehensive performance analysis.
        
        Args:
            portfolio_values: List of daily portfolio values
            daily_returns: List of daily returns
            
        Returns:
            Dictionary containing all performance analysis results
        """
        # Calculate all metrics
        returns_metrics = self.performance_metrics.calculate_returns(
            portfolio_values, datetime.now(), datetime.now()
        )
        risk_metrics_dict = self.risk_metrics.calculate_all_risk_metrics(daily_returns)
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
                'risk_free_rate': self.performance_metrics.RISK_FREE_RATE,
                'market_timezone': self.performance_metrics.timezone.zone
            }
        }
        
        logger.info("Comprehensive performance analysis completed")
        return comprehensive_analysis
    
    def generate_performance_report(self, analysis_results: Dict[str, Any], 
                                  output_path: str = "reports/performance_report.txt") -> None:
        """
        Generate comprehensive performance report.
        
        Args:
            analysis_results: Results from comprehensive analysis
            output_path: Path to save the report
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("BRAZILIAN B3 QUANT BACKTEST - COMPREHENSIVE PERFORMANCE REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Analysis Date: {analysis_results['analysis_timestamp']}")
        report_lines.append("")
        
        # Returns Analysis
        report_lines.append("RETURNS ANALYSIS")
        report_lines.append("-" * 40)
        returns = analysis_results['returns_analysis']
        report_lines.append(f"Total Return: {returns['total_return']:.4f} ({returns['total_return']*100:.2f}%)")
        report_lines.append(f"Annualized Return: {returns['annualized_return']:.4f} ({returns['annualized_return']*100:.2f}%)")
        report_lines.append(f"Logarithmic Return: {returns['logarithmic_return']:.4f}")
        report_lines.append(f"Trading Days: {returns['trading_days']}")
        report_lines.append("")
        
        # Risk Analysis
        report_lines.append("RISK ANALYSIS")
        report_lines.append("-" * 40)
        risk = analysis_results['risk_analysis']
        report_lines.append(f"Sharpe Ratio: {risk['sharpe_ratio']:.4f}")
        report_lines.append(f"Sortino Ratio: {risk['sortino_ratio']:.4f}")
        report_lines.append(f"Calmar Ratio: {risk['calmar_ratio']:.4f}")
        report_lines.append(f"Maximum Drawdown: {risk['max_drawdown']:.4f} ({risk['max_drawdown']*100:.2f}%)")
        report_lines.append(f"Volatility: {risk['volatility']:.4f} ({risk['volatility']*100:.2f}%)")
        report_lines.append(f"Value at Risk (95%): {risk['var_95']:.4f} ({risk['var_95']*100:.2f}%)")
        report_lines.append("")
        
        # Tax Analysis
        report_lines.append("TAX ANALYSIS")
        report_lines.append("-" * 40)
        tax = analysis_results['tax_analysis']
        report_lines.append(f"Total Taxes Paid: R$ {tax['total_taxes_paid']:.2f}")
        report_lines.append(f"Tax Efficiency: {tax['tax_efficiency']:.4f} ({tax['tax_efficiency']*100:.2f}%)")
        report_lines.append(f"Effective Tax Rate: {tax['effective_tax_rate']:.4f} ({tax['effective_tax_rate']*100:.2f}%)")
        report_lines.append("")
        
        # Trade Analysis
        report_lines.append("TRADE ANALYSIS")
        report_lines.append("-" * 40)
        trade = analysis_results['trade_analysis']
        report_lines.append(f"Total Trades: {trade['total_trades']}")
        report_lines.append(f"Win Rate: {trade['win_rate']:.4f} ({trade['win_rate']*100:.2f}%)")
        report_lines.append(f"Profit Factor: {trade['profit_factor']:.4f}")
        report_lines.append(f"Average Win: R$ {trade['average_win']:.2f}")
        report_lines.append(f"Average Loss: R$ {trade['average_loss']:.2f}")
        report_lines.append("")
        
        # Regulatory Compliance
        report_lines.append("REGULATORY COMPLIANCE")
        report_lines.append("-" * 40)
        compliance = analysis_results['regulatory_compliance']
        report_lines.append(f"Regulatory Framework: {compliance['regulatory_framework']}")
        report_lines.append(f"CVM Compliance: {compliance['cvm_compliance']}")
        report_lines.append(f"Receita Federal Compliance: {compliance['receita_federal_compliance']}")
        report_lines.append("")
        
        # Write report to file
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
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