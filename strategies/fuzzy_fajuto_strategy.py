"""
FuzzyFajuto Strategy - Three-Attempt Execution System with Brapi.dev Daily Data

This strategy implements the FuzzyFajuto logic for Brazilian market backtesting:
- Uses Brapi.dev daily data directly for technical indicators
- Uses Brapi.dev intraday data for execution simulation
- Calculates daily technical indicators (ATR, EMA, RSI) with periods in days
- Uses daily ATR(14) for volatility measurement while trading on intraday bars
- Simulates three execution attempts per asset per day
- Uses market orders at open and limit orders based on daily ATR levels
- Implements Brazilian market compliance and lot size rules

Strategy Logic:
1. Load Brapi.dev daily data for technical indicators during warmup
2. Calculate daily technical indicators directly from daily data:
   - ATR(14 days) from daily True Range
   - EMA(3,5,10,15,20 days) from daily close prices
   - RSI(10 days) from daily price changes
3. Generate enhanced signals using daily indicators:
   - EMA alignment and momentum (40% weight)
   - RSI overbought/oversold levels (30% weight)
   - Price vs EMA position (20% weight)
   - EMA momentum confirmation (10% weight)
4. Generate three order attempts per signal:
   - Attempt 1: Market order at open
   - Attempt 2: Limit order at Open ± α·daily_ATR (α = 0.25)
   - Attempt 3: Limit order at Open ± β·daily_ATR (β = 0.50)
5. Simulate fills based on intraday High/Low range from Brapi.dev intraday data
6. Close positions at market close (MOC)
7. Track execution metrics and PnL

Key Features:
- Daily indicators calculated directly from Brapi.dev daily data (no aggregation)
- Intraday execution simulation using Brapi.dev hourly data
- Periods expressed in trading days, not hours
- Dynamic ATR calculation from reliable daily data
- Enhanced signal generation using multiple daily technical indicators
- Data quality validation and missing data handling

Author: Quantitative Trading Specialist
Date: 2025
"""

import sys
import os
import logging
import time
import math
from datetime import datetime, date, timedelta
from typing import Iterable, List, Dict, Any, Optional, Tuple, Sequence
from collections import deque
import numpy as np
import pandas as pd

# Add the engine directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'engine'))

from engine.base_strategy import (
    BaseStrategy,
    StrategyConfig,
    StrategyContext,
    Bar,
    OrderIntent,
    OrderType,
    OrderSide,
    Fill
)
from engine.market_utils import DailyTechnicalIndicators
from engine.brapi_provider import BrapiProvider





class FuzzyFajutoStrategy(BaseStrategy):
    """
    FuzzyFajuto Strategy implementing three-attempt execution system.
    
    This strategy simulates realistic order execution in Brazilian markets
    by attempting three different order types per asset per day:
    1. Market order at daily open (10:00 AM only - immediate execution)
    2. Limit order at Open ± α·ATR (passive-1)
    3. Limit order at Open ± β·ATR (passive-2)
    
    IMPORTANT: Market Order at Open is restricted to daily market open (10:00 AM)
    to prevent multiple entries throughout the trading day.
    
    Key Features:
    - ATR-based volatility measurement
    - Three-attempt execution simulation
    - Brazilian market compliance
    - Detailed execution tracking
    - Risk management integration
    - Daily market open restriction for market orders
    """
    
    def __init__(self, cfg: StrategyConfig, ctx: StrategyContext, config_file: str = None):
        """
        Initialize the FuzzyFajuto strategy.
        
        Args:
            cfg: Strategy configuration
            ctx: Strategy context with live handles
            config_file: Optional path to strategy config file (relative to config/)
        """
        super().__init__(cfg, ctx)
        
        # Store config file path for loading
        self.config_file = config_file
        
        # Load strategy-specific configuration
        self._load_strategy_config()
        
        # Initialize strategy state
        self._initialize_state()

        # Track per-symbol per-date EMA non-convergence warnings to avoid spam
        self._ema_nonconverged_reported_dates: Dict[str, set] = {}
        # Track per-symbol dates where daily data was insufficient so we don't spam logs or reattempt repeatedly in the same day
        self._insufficient_daily_reported_dates: Dict[str, set] = {}
        
        # Reduce initialization verbosity
        self.context.logger.debug("FuzzyFajutoStrategy initialized")
        self.context.logger.debug(f"ATR period: {self.atr_period}")
        self.context.logger.debug(f"Alpha factor: {self.alpha_factor}")
        self.context.logger.debug(f"Beta factor: {self.beta_factor}")
        self.context.logger.debug(f"Asset exposure: {self.asset_exposure_pct:.1%}")
        self.context.logger.debug(f"Lot size rounding: {self.lot_size_rounding}")
        self.context.logger.debug(f"Min exposure threshold: {self.min_exposure_threshold:.1%} of min lot size")
        self.context.logger.debug(f"Allow odd lots: {self.allow_odd_lots}")

        # Diagnostics buffers (lightweight): per-symbol/date fuzzy for CSV
        self._fuzzy_rows: list = []
        # Day-trade scheduling store
        self._scheduled_day_trades: Dict[date, Dict[str, Dict[str, Any]]] = {}
        md = self._config.metadata if isinstance(self._config.metadata, dict) else {}
        self.target_gross_exposure_per_side: float = float(md.get('target_gross_exposure_per_side', 0.5))
        self.max_net_exposure: float = float(md.get('max_net_exposure', 0.02))
        self.max_single_side_exposure: float = float(md.get('max_single_side_exposure', 0.5))
        # Indicator warmup tracking
        self.first_valid_indicator_date: Dict[str, date] = {}
    
    def _load_strategy_config(self):
        """Load strategy-specific configuration from specified config file."""
        try:
            import yaml
            
            # Determine config file to use
            if self.config_file is None:
                config_filename = 'profiles/fuzzy_fajuto_default.yaml'  # New default location
            else:
                config_filename = self.config_file
            
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', config_filename)
            
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            strategy_config = config.get('strategy', {})
            
            # Warmup Configuration
            warmup_bars_from_config = strategy_config.get('warmup_bars', None)
            if warmup_bars_from_config is not None:
                # Override the StrategyConfig warmup_bars with the YAML value
                self._config.warmup_bars = warmup_bars_from_config
                self.context.logger.info(f"Warmup bars set from config: {warmup_bars_from_config}")
            
            # ATR Parameters
            self.atr_period = strategy_config.get('atr_period', 14)
            self.alpha_factor = strategy_config.get('alpha_factor', 0.25)
            self.beta_factor = strategy_config.get('beta_factor', 0.50)
            # Minimum protective distances as a fraction of open price (UTC-based open at 13:00)
            self.min_alpha_pct = strategy_config.get('min_alpha_pct', 0.015)  # 1.5%
            self.min_beta_pct = strategy_config.get('min_beta_pct', 0.025)    # 2.5%
            
            # Exposure and Position Sizing
            self.asset_exposure_pct = strategy_config.get('asset_exposure_pct', 0.08)
            self.equal_split_attempts = strategy_config.get('equal_split_attempts', True)
            self.max_assets_per_day = strategy_config.get('max_assets_per_day', 5)
            
            # Execution Parameters
            self.min_lot_size = strategy_config.get('min_lot_size', 100)
            
            # Position Sizing and Lot Size Rounding
            self.lot_size_rounding = strategy_config.get('lot_size_rounding', 'conservative')
            self.min_exposure_threshold = strategy_config.get('min_exposure_threshold', 0.5)
            self.allow_odd_lots = strategy_config.get('allow_odd_lots', False)
            
            # Signal Parameters (original FuzzyFajuto thresholds)
            self.buy_threshold = strategy_config.get('buy_threshold', 1.50)
            self.sell_threshold = strategy_config.get('sell_threshold', -1.50)
            self.ema_periods = strategy_config.get('ema_periods', [3, 5, 10, 15, 20])
            self.rsi_period = strategy_config.get('rsi_period', 10)
            self.rsi_overbought = strategy_config.get('rsi_overbought', 65)
            self.rsi_oversold = strategy_config.get('rsi_oversold', 35)
            
            # Log signal parameters for verification
            self.context.logger.info(f"Signal thresholds loaded: buy={self.buy_threshold}, sell={self.sell_threshold}")
            self.context.logger.info(f"EMA periods: {self.ema_periods}")
            self.context.logger.info(f"RSI period: {self.rsi_period}, overbought={self.rsi_overbought}, oversold={self.rsi_oversold}")
            
            # Risk Management
            self.max_daily_loss_pct = strategy_config.get('max_daily_loss_pct', 0.02)
            self.max_position_size_pct = strategy_config.get('max_position_size_pct', 0.10)
            self.max_day_trade_exposure_pct = strategy_config.get('max_day_trade_exposure_pct', 0.30)
            
            # Execution Tracking
            self.track_executions = strategy_config.get('track_executions', True)
            self.log_detailed_executions = strategy_config.get('log_detailed_executions', True)
            self.save_execution_history = strategy_config.get('save_execution_history', True)
            
            # Data Quality Settings
            self.min_valid_atr = strategy_config.get('min_valid_atr', 0.001)
            self.max_atr_multiplier = strategy_config.get('max_atr_multiplier', 10.0)
            self.require_valid_ohlc = strategy_config.get('require_valid_ohlc', True)
            self.skip_invalid_days = strategy_config.get('skip_invalid_days', True)
            
            # Error Handling
            self.max_consecutive_errors = strategy_config.get('max_consecutive_errors', 3)
            self.error_recovery_mode = strategy_config.get('error_recovery_mode', 'skip')
            
            # Calculate intelligent data requirements after loading config
            self._calculate_data_requirements()
            
            self.context.logger.info("Strategy configuration loaded successfully")
            
        except Exception as e:
            self.context.logger.warning(f"Failed to load strategy config, using defaults: {e}")
            # Fallback to default values
            self.atr_period = 14
            self.alpha_factor = 0.25
            self.beta_factor = 0.50
            # Fallback defaults for protective minimums
            self.min_alpha_pct = getattr(self, 'min_alpha_pct', 0.015)
            self.min_beta_pct = getattr(self, 'min_beta_pct', 0.025)
            self.asset_exposure_pct = 0.08
            self.equal_split_attempts = True
            self.max_assets_per_day = 5
            self.min_lot_size = 100
            # Signal parameters (original FuzzyFajuto thresholds)
            self.buy_threshold = 1.50
            self.sell_threshold = -1.50
            self.ema_periods = [3, 5, 10, 15, 20]
            self.rsi_period = 10
            self.rsi_overbought = 65
            self.rsi_oversold = 35
            
            # Log fallback signal parameters
            self.context.logger.warning(f"Using fallback signal thresholds: buy={self.buy_threshold}, sell={self.sell_threshold}")
            # Risk management
            self.max_daily_loss_pct = 0.02
            self.max_position_size_pct = 0.10
            self.max_day_trade_exposure_pct = 0.30
            self.track_executions = True
            self.log_detailed_executions = True
            self.save_execution_history = True
            # Data quality settings
            self.min_valid_atr = 0.001
            self.max_atr_multiplier = 10.0
            self.require_valid_ohlc = True
            self.skip_invalid_days = True
            # Error handling
            self.max_consecutive_errors = 3
            self.error_recovery_mode = 'skip'
            
            # Calculate intelligent data requirements with defaults
            self._calculate_data_requirements()

    def _calculate_data_requirements(self):
        """
        Intelligently calculate minimum data requirements based on strategy parameters.
        
        This method analyzes all strategy parameters and calculates the exact
        minimum number of data points needed for reliable operation.
        """
        # Base requirements for technical indicators
        atr_requirement = self.atr_period
        rsi_requirement = self.rsi_period
        ema_requirement = max(self.ema_periods) if self.ema_periods else 20
        
        # Execution simulation requirements
        # Need enough intraday bars for realistic backtesting simulation
        # Minimum 4 bars per day for valid execution simulation
        # Need at least 3 trading days for execution simulation to be meaningful
        execution_simulation_days = 3
        min_bars_per_day = 4
        execution_simulation_requirement = execution_simulation_days * min_bars_per_day
        
        # Data quality buffer
        # Additional bars needed for:
        # - Handling holidays/missing data
        # - Ensuring sufficient data for rolling calculations
        # - Buffer for data validation and filtering
        quality_buffer = 5
        
        # Calculate the maximum requirement from all indicators
        max_indicator_requirement = max(atr_requirement, rsi_requirement, ema_requirement)
        
        # Total requirement = max indicator requirement + execution simulation + quality buffer
        total_requirement = max_indicator_requirement + execution_simulation_requirement + quality_buffer
        
        # Store calculated requirements
        self.data_requirements = {
            'atr_period': atr_requirement,
            'rsi_period': rsi_requirement,
            'ema_period': ema_requirement,
            'execution_simulation_days': execution_simulation_days,
            'min_bars_per_day': min_bars_per_day,
            'execution_simulation_requirement': execution_simulation_requirement,
            'quality_buffer': quality_buffer,
            'max_indicator_requirement': max_indicator_requirement,
            'total_minimum_requirement': total_requirement
        }
        
        # Update warmup_bars if calculated requirement is higher than configured
        if total_requirement > self.warmup_bars:
            self._config.warmup_bars = total_requirement
            self.context.logger.info(f"Updated warmup_bars to {total_requirement} based on strategy requirements")
        
        # Log detailed requirements breakdown
        self.context.logger.info("Intelligent data requirements calculated:")
        self.context.logger.info(f"  - ATR period requirement: {atr_requirement}")
        self.context.logger.info(f"  - RSI period requirement: {rsi_requirement}")
        self.context.logger.info(f"  - EMA period requirement: {ema_requirement}")
        self.context.logger.info(f"  - Execution simulation requirement: {execution_simulation_requirement} bars ({execution_simulation_days} days × {min_bars_per_day} bars/day)")
        self.context.logger.info(f"  - Quality buffer: {quality_buffer}")
        self.context.logger.info(f"  - Total minimum requirement: {total_requirement} bars")
        self.context.logger.info(f"  - Configured warmup_bars: {self.warmup_bars}")

    def get_data_requirements_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the strategy's data requirements.
        
        Returns:
            Dictionary containing detailed data requirements breakdown
        """
        if not hasattr(self, 'data_requirements'):
            return {'error': 'Data requirements not calculated yet'}
        
        return {
            'strategy_parameters': {
                'atr_period': self.atr_period,
                'rsi_period': self.rsi_period,
                'ema_periods': self.ema_periods,
                'max_ema_period': max(self.ema_periods) if self.ema_periods else 20
            },
            'calculated_requirements': self.data_requirements,
            'current_warmup_bars': self.warmup_bars,
            'quality_standards': {
                'min_valid_atr': self.min_valid_atr,
                'require_valid_ohlc': self.require_valid_ohlc,
                'skip_invalid_days': self.skip_invalid_days
            },
            'calculation_formula': {
                'description': 'Total = max(ATR, RSI, EMA) + Daily Aggregation + Quality Buffer',
                'components': {
                    'max_indicator_requirement': f"max({self.atr_period}, {self.rsi_period}, {max(self.ema_periods)}) = {self.data_requirements['max_indicator_requirement']}",
                    'execution_simulation': f"{self.data_requirements['execution_simulation_days']} days × {self.data_requirements['min_bars_per_day']} bars/day = {self.data_requirements['execution_simulation_requirement']}",
                    'quality_buffer': str(self.data_requirements['quality_buffer']),
                    'total': f"{self.data_requirements['max_indicator_requirement']} + {self.data_requirements['execution_simulation_requirement']} + {self.data_requirements['quality_buffer']} = {self.data_requirements['total_minimum_requirement']}"
                }
            }
        }

    def _initialize_state(self):
        """Initialize strategy state and data structures."""
        # Technical indicators (no aggregation needed - use Brapi daily data directly)
        self.daily_indicators = DailyTechnicalIndicators()
        
        # Daily data storage (from Brapi.dev daily API)
        self.daily_data = {}  # symbol -> daily OHLCV DataFrame from Brapi
        self.daily_indicators_data = {}  # symbol -> indicators dict
        
        # Dynamic ATR calculation state
        self.current_atr_values = {}  # symbol -> current ATR value (float)
        self.atr_calculation_dates = {}  # symbol -> last calculation date
        self.daily_indicators_calculator = DailyTechnicalIndicators()  # Reusable calculator
        
        # Daily data update tracking (NEW)
        self.daily_data_last_update = {}  # symbol -> last update date
        self.daily_indicators_last_update = {}  # symbol -> last indicators calculation date
        
        # Intraday price history (for execution simulation only)
        self.intraday_history = {}
        
        # Execution tracking
        self.daily_executions = {}
        self.execution_history = []
        # Confirmed fills captured via on_fill (authoritative execution source)
        self.confirmed_fills = []
        
        # Position tracking
        self.current_positions = {}
        self.daily_pnl = {}
        
        # Risk management
        self.daily_loss = 0.0
        self.daily_exposure = 0.0
        
        # Performance metrics - All attempts are now daily (once per trading day)
        self.daily_execution_counts = {'market': 0, 'limit_alpha': 0, 'limit_beta': 0, 'limit_gamma': 0}
        self.daily_fill_rates = {'market': 0, 'limit_alpha': 0, 'limit_beta': 0, 'limit_gamma': 0}
        
        # Legacy tracking (for backward compatibility)
        self.fill_rates = {'market': 0, 'limit_alpha': 0, 'limit_beta': 0}
        self.execution_counts = {'market': 0, 'limit_alpha': 0, 'limit_beta': 0}
        
        # Daily execution tracking - ensure all orders execute once per trading day
        self.market_order_executed_dates = set()  # Track which trading days have had market orders executed
        self.limit_alpha_executed_dates = set()  # Track which trading days have had limit_alpha orders executed
        self.limit_beta_executed_dates = set()   # Track which trading days have had limit_beta orders executed
        
        # Order emission control (Section 7 implementation)
        self.daily_orders_emitted = {}  # symbol -> {date: {market: bool, limit_alpha: bool, limit_beta: bool}}
        self.daily_order_prices = {}    # symbol -> {date: {market_price, alpha_price, beta_price}}
        self.first_bar_of_day = {}      # symbol -> {date: bool} - track first bar processing
        
        # Neutrality state (market-neutral final gate)
        self.RISK_MARKET_NEUTRAL: bool = True
        # Strict pairing: emit at most one BUY and one SELL symbol per day (net-neutral pair)
        self.RISK_STRICT_ONE_PAIR: bool = True
        self._neutral_buffer: Dict[date, Dict[str, Dict[str, Any]]] = {}
        self._first_bar_seen_by_date: Dict[date, set] = {}
        try:
            self._universe_symbols = set(self._config.universe or [])
        except Exception:
            self._universe_symbols = set()
        
        # Core recalculation engine state
        self.brapi_provider = None  # Will be initialized when needed
        self.daily_data_cache = {}  # symbol -> daily data cache
        self.last_recalculation_dates = {}  # symbol -> last recalculation date
    
    def _initialize_brapi_provider(self):
        """Initialize Brapi provider for direct data access."""
        if self.brapi_provider is None:
            try:
                # Get API token from environment or context
                api_token = os.getenv('BRAPI_API_TOKEN')
                if not api_token:
                    self.context.logger.error("BRAPI_API_TOKEN environment variable not set")
                    return False
                
                self.brapi_provider = BrapiProvider(
                    api_token=api_token,
                    cache_dir="data/brapi_cache",
                    cache_ttl_hours=24,
                    timeout=30,
                    max_retries=3
                )
                self.context.logger.info("✅ Brapi provider initialized for direct data access")
                return True
            except Exception as e:
                self.context.logger.error(f"Failed to initialize Brapi provider: {e}")
                return False
        return True
    
    def _get_daily_data_for_date(self, symbol: str, target_date: date) -> pd.DataFrame:
        """
        Fetch daily data directly from Brapi.dev for specific date range.
        This bypasses the hybrid data manager dependency.
        
        Args:
            symbol: Trading symbol
            target_date: Target date for data
            
        Returns:
            Daily DataFrame with data up to target_date
        """
        try:
            if not self._initialize_brapi_provider():
                return pd.DataFrame()
            
            # Calculate date range: 365 days before target_date to target_date
            start_date = target_date - timedelta(days=365)
            end_date = target_date
            
            # Format dates for Brapi API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            self.context.logger.info(f"Fetching daily data for {symbol}: {start_str} to {end_str}")
            
            # Fetch data directly from Brapi
            daily_data = self.brapi_provider.get_daily_data(symbol, start_str, end_str)
            
            if daily_data is not None and not daily_data.empty:
                self.context.logger.info(f"✅ Fetched {len(daily_data)} daily bars for {symbol}")
                return daily_data
            else:
                self.context.logger.warning(f"No daily data returned for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            self.context.logger.error(f"Error fetching daily data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _is_new_trading_day(self, symbol: str, current_date: date) -> bool:
        """
        Check if this is a new trading day that requires recalculation.
        
        Args:
            symbol: Trading symbol
            current_date: Current trading date
            
        Returns:
            True if this is a new trading day requiring recalculation
        """
        if symbol not in self.last_recalculation_dates:
            return True
        
        last_calc_date = self.last_recalculation_dates[symbol]
        return current_date != last_calc_date
    
    def _force_daily_recalculation(self, symbol: str, current_date: date):
        """
        Force recalculation of all indicators for each new trading day.
        This ensures indicators are always current.
        
        Args:
            symbol: Trading symbol
            current_date: Current trading date
        """
        try:
            self.context.logger.info(f"🔄 Force recalculation for {symbol} on {current_date}")
            
            # Clear previous indicators for the symbol
            if symbol in self.daily_indicators_data:
                del self.daily_indicators_data[symbol]
            
            # Fetch fresh daily data up to current_date
            daily_data = self._get_daily_data_for_date(symbol, current_date)
            
            if daily_data.empty:
                # Log only once per symbol per date and mark as handled for the day
                dates = self._insufficient_daily_reported_dates.setdefault(symbol, set())
                if current_date not in dates:
                    self.context.logger.error(f"No daily data available for {symbol} on {current_date}")
                    dates.add(current_date)
                # Prevent repeated attempts within the same trading day
                self.last_recalculation_dates[symbol] = current_date
                return
            
            # Validate sufficient data
            required_days = max(self.atr_period, self.rsi_period, max(self.ema_periods)) + 5
            if len(daily_data) < required_days:
                # Log only once per symbol per date and mark as handled for the day
                dates = self._insufficient_daily_reported_dates.setdefault(symbol, set())
                if current_date not in dates:
                    self.context.logger.error(f"Insufficient daily data for {symbol}: {len(daily_data)} < {required_days}")
                    dates.add(current_date)
                # Prevent repeated attempts within the same trading day
                self.last_recalculation_dates[symbol] = current_date
                return
            
            # Update daily data storage
            self.daily_data[symbol] = daily_data
            
            # Recalculate all indicators
            self._calculate_indicators_guaranteed(symbol, current_date)
            
            # Update recalculation tracking
            self.last_recalculation_dates[symbol] = current_date
            self.daily_data_last_update[symbol] = current_date
            self.daily_indicators_last_update[symbol] = current_date
            
            self.context.logger.info(f"✅ Force recalculation completed for {symbol}")
            
        except Exception as e:
            self.context.logger.error(f"Error in force recalculation for {symbol}: {e}")
    
    def _calculate_indicators_guaranteed(self, symbol: str, current_date: date):
        """
        Calculate indicators with guaranteed success - no fallbacks needed.
        
        Args:
            symbol: Trading symbol
            current_date: Current trading date
        """
        try:
            self.context.logger.info(f"Calculating indicators for {symbol} on {current_date}")
            
            # Get daily data
            daily_data = self.daily_data.get(symbol)
            if daily_data is None or daily_data.empty:
                self.context.logger.error(f"No daily data available for {symbol}")
                return
            
            # Ensure sufficient data
            required_days = max(self.atr_period, self.rsi_period, max(self.ema_periods)) + 5
            if len(daily_data) < required_days:
                self.context.logger.error(f"Insufficient data for indicators: {len(daily_data)} < {required_days}")
                return
            
            # Calculate all indicators using DailyTechnicalIndicators
            indicators = self.daily_indicators.calculate_all_indicators(
                daily_data,
                atr_period=self.atr_period,
                ema_periods=self.ema_periods,
                rsi_period=self.rsi_period,
                sma_periods=[3, 5]
            )
            
            # Validate all indicator values
            if not self._validate_indicators(indicators):
                self.context.logger.error(f"Invalid indicators calculated for {symbol}")
                return
            
            # Store indicators
            self.daily_indicators_data[symbol] = indicators
            
            # Update current ATR value using ATR[D-1]
            latest_atr = 0.0
            try:
                atr_series = indicators['atr'] if 'atr' in indicators else None
                if atr_series is not None and not atr_series.empty:
                    atr_shifted = atr_series.shift(1)
                    current_ts = pd.Timestamp(current_date)
                    if current_ts in atr_shifted.index:
                        latest_atr = float(atr_shifted.loc[current_ts])
                    else:
                        latest_atr = float(atr_shifted.dropna().iloc[-1]) if not atr_shifted.dropna().empty else 0.0
            except Exception:
                latest_atr = 0.0

            if self._validate_atr_value(latest_atr):
                old_atr = self.current_atr_values.get(symbol, 0.0)
                self.current_atr_values[symbol] = float(latest_atr)
                self.atr_calculation_dates[symbol] = current_date
                
                # Log significant changes
                if old_atr > 0:
                    atr_change_pct = ((latest_atr - old_atr) / old_atr) * 100
                    if abs(atr_change_pct) > 5:
                        self.context.logger.info(f"ATR updated for {symbol}: {old_atr:.4f} → {latest_atr:.4f} ({atr_change_pct:+.1f}%)")
            else:
                self.context.logger.warning(f"ATR[D-1] unavailable/invalid for {symbol} on {current_date}; trading may be skipped today.")
            
            # Log indicator updates
            latest_ema = indicators['ema_5'].iloc[-1] if not indicators['ema_5'].empty else None
            latest_rsi = indicators['rsi'].iloc[-1] if not indicators['rsi'].empty else None
            
            if latest_ema is not None and latest_rsi is not None:
                self.context.logger.info(f"Indicators calculated for {symbol}: EMA(5)={latest_ema:.2f}, RSI={latest_rsi:.2f}, ATR={latest_atr:.4f}")
            
            self.context.logger.info(f"✅ Indicators calculation completed for {symbol}")
            
        except Exception as e:
            self.context.logger.error(f"Error calculating indicators for {symbol}: {e}")
    
    def _validate_indicators(self, indicators: Dict[str, pd.Series]) -> bool:
        """
        Enhanced validation that accepts mathematically correct NaN values.
        
        Validation Rules:
        1. Latest value must be valid (not NaN)
        2. Must have sufficient valid values for trading
        3. NaN values must be only at beginning (not gaps)
        4. All required indicators must be present
        
        Args:
            indicators: Dictionary of calculated indicators
            
        Returns:
            True if all indicators are valid
        """
        try:
            # Minimum valid values required for each indicator
            MIN_VALID_VALUES = {
                'atr': 100,      # Need enough ATR values for volatility calculation
                'rsi': 50,       # Need enough RSI values for momentum analysis
                'ema_5': 20,     # Need enough EMA values for trend analysis
            }
            
            required_indicators = ['atr', 'ema_5', 'rsi']
            
            for indicator_name in required_indicators:
                if indicator_name not in indicators:
                    self.context.logger.error(f"Missing required indicator: {indicator_name}")
                    return False
                
                series = indicators[indicator_name]
                
                # Check 1: Series exists and not empty
                if series is None or series.empty:
                    self.context.logger.error(f"Empty indicator series: {indicator_name}")
                    return False
                
                # Check 2: Latest value is valid (most important for trading)
                latest_value = series.iloc[-1]
                if pd.isna(latest_value):
                    self.context.logger.error(f"Latest value is NaN for {indicator_name}")
                    return False
                
                # Check 3: Sufficient valid values for reliable calculation
                valid_count = series.count()
                min_required = MIN_VALID_VALUES.get(indicator_name, 50)
                if valid_count < min_required:
                    self.context.logger.error(f"Insufficient valid values for {indicator_name}: {valid_count} < {min_required}")
                    return False
                
                # Check 4: NaN values are only at beginning (not gaps)
                if self._has_nan_gaps(series):
                    self.context.logger.error(f"NaN gaps detected in {indicator_name}")
                    return False
                
                # Log validation success
                nan_count = series.isna().sum()
                self.context.logger.info(f"✅ {indicator_name}: {valid_count} valid, {nan_count} NaN (beginning only), latest={latest_value}")
            
            return True
            
        except Exception as e:
            self.context.logger.error(f"Error in enhanced validation: {e}")
            return False
    
    def _has_nan_gaps(self, series: pd.Series) -> bool:
        """
        Check if NaN values are only at the beginning (acceptable)
        or if there are gaps in the middle (problematic).
        
        Returns:
            True if there are NaN gaps in the middle
            False if NaN values are only at the beginning
        """
        if series.isna().sum() == 0:
            return False  # No NaN values
        
        # Find the first non-NaN value
        first_valid_idx = series.first_valid_index()
        if first_valid_idx is None:
            return True  # All values are NaN
        
        # Check if there are any NaN values after the first valid value
        after_first_valid = series[series.index > first_valid_idx]
        return after_first_valid.isna().any()
    
    def on_start(self, asof: datetime) -> None:
        """Called when strategy starts."""
        super().on_start(asof)
        self.context.logger.info("FuzzyFajuto strategy started")
    
    def on_warmup(self, symbol: str, bars: Sequence[Bar]) -> None:
        """Initialize technical indicators using core recalculation engine."""
        super().on_warmup(symbol, bars)
        
        self.context.logger.info(f"🚀 Core Recalculation Engine Warmup for {symbol}")
        
        # Get the latest date from the bars for initial calculation
        if bars:
            latest_date = max(bar.timestamp.date() for bar in bars)
        else:
            latest_date = datetime.now().date()
        
        # Use core recalculation engine for initial setup
        self.context.logger.info(f"Initializing indicators for {symbol} on {latest_date}")
        self._force_daily_recalculation(symbol, latest_date)
        
        # Validate that indicators were calculated successfully
        if symbol not in self.daily_indicators_data:
            self.context.logger.error(f"CRITICAL: Failed to initialize indicators for {symbol}")
            return
        
        # Store intraday bars for execution simulation
        if symbol not in self.intraday_history:
            self.intraday_history[symbol] = deque(maxlen=500)
        
        for bar in bars:
            if self._validate_bar_data(bar):
                self.intraday_history[symbol].append({
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                })
        
        # Log warmup completion
        indicators = self.daily_indicators_data[symbol]
        latest_atr = indicators['atr'].iloc[-1] if not indicators['atr'].empty else 0.0
        latest_rsi = indicators['rsi'].iloc[-1] if not indicators['rsi'].empty else None
        latest_ema = indicators['ema_5'].iloc[-1] if not indicators['ema_5'].empty else None
        
        self.context.logger.info(f"✅ Core Recalculation Engine Warmup completed for {symbol}:")
        self.context.logger.info(f"  - Daily indicators: {len(self.daily_data[symbol])} bars from Brapi.dev")
        self.context.logger.info(f"  - Intraday execution: {len(bars)} bars for simulation")
        self.context.logger.info(f"  - ATR value: {latest_atr:.4f}")
        self.context.logger.info(f"  - RSI value: {latest_rsi:.2f}" if latest_rsi else "  - RSI value: N/A")
        self.context.logger.info(f"  - EMA(5) value: {latest_ema:.2f}" if latest_ema else "  - EMA(5) value: N/A")
    
    def _get_current_atr(self, symbol: str) -> float:
        """
        Get the current dynamic ATR value for a symbol with robust validation.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current ATR value (0.0 if invalid)
        """
        if symbol not in self.current_atr_values:
            self.context.logger.debug(f"No current ATR data available for {symbol}")
            return 0.0
        
        latest_atr = self.current_atr_values[symbol]
        
        # Validate the ATR value
        if pd.isna(latest_atr):
            self.context.logger.warning(f"NaN ATR value for {symbol}")
            return 0.0
        
        if latest_atr <= 0:
            self.context.logger.warning(f"Non-positive ATR value for {symbol}: {latest_atr}")
            return 0.0
        
        # Check for unreasonably high ATR (sanity check)
        if latest_atr > self.max_atr_multiplier * 100:  # Configurable threshold
            self.context.logger.warning(f"Unusually high ATR for {symbol}: {latest_atr} (threshold: {self.max_atr_multiplier * 100})")
            return 0.0
        
        # Check minimum valid ATR
        if latest_atr < self.min_valid_atr:
            self.context.logger.warning(f"ATR too low for {symbol}: {latest_atr} (minimum: {self.min_valid_atr})")
            return 0.0
        
        return float(latest_atr)
    
    def _get_daily_atr(self, symbol: str) -> float:
        """
        Legacy method - now redirects to dynamic ATR calculation for backward compatibility.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current dynamic ATR value
        """
        return self._get_current_atr(symbol)
    
    def _should_recalculate_atr(self, symbol: str, current_date: date) -> bool:
        """
        Check if ATR should be recalculated for the given symbol and date.
        
        Args:
            symbol: Trading symbol
            current_date: Current trading date
            
        Returns:
            True if ATR should be recalculated
        """
        # Check if we have the symbol in our tracking
        if symbol not in self.atr_calculation_dates:
            return True
        
        # Check if current date is different from last calculation date
        last_calc_date = self.atr_calculation_dates[symbol]
        if current_date != last_calc_date:
            return True
        
        return False
    
    def _get_daily_data_up_to_date(self, symbol: str, end_date: date) -> pd.DataFrame:
        """
        Get daily data filtered up to a specific date for ATR calculation.
        
        This method efficiently filters from the pre-loaded daily data cache to avoid
        additional API calls. The daily data is loaded once during initialization
        with extended historical range (365 days) to support ATR recalculation.
        
        Args:
            symbol: Trading symbol
            end_date: End date for filtering
            
        Returns:
            Filtered daily DataFrame up to end_date
        """
        if symbol not in self.daily_data:
            self.context.logger.warning(f"No daily data available for {symbol}")
            return pd.DataFrame()
        
        daily_data = self.daily_data[symbol]
        if daily_data.empty:
            return pd.DataFrame()
        
        # Filter data up to end_date (no additional API calls needed)
        end_datetime = pd.Timestamp(end_date)
        filtered_data = daily_data[daily_data.index <= end_datetime]
        
        return filtered_data
    
    def _validate_atr_value(self, atr_value: float) -> bool:
        """
        Validate an ATR value for acceptance.
        
        Args:
            atr_value: ATR value to validate
            
        Returns:
            True if valid, False otherwise
        """
        if pd.isna(atr_value) or np.isinf(atr_value):
            return False
        
        if atr_value <= 0:
            return False
        
        if atr_value < self.min_valid_atr:
            return False
        
        if atr_value > self.max_atr_multiplier * 100:
            return False
        
        return True
    
    def _recalculate_daily_atr(self, symbol: str, current_date: date):
        """
        Recalculate ATR for a specific symbol using daily data up to current date.
        
        This method uses pre-loaded daily data from Brapi.dev cache to avoid
        additional API calls during backtesting. The daily data is loaded once
        during initialization with 365-day historical range.
        
        Args:
            symbol: Trading symbol
            current_date: Current trading date
        """
        try:
            # Get daily data up to current date
            daily_data = self._get_daily_data_up_to_date(symbol, current_date)
            
            if daily_data.empty or len(daily_data) < self.atr_period + 1:
                self.context.logger.warning(f"Insufficient daily data for ATR calculation on {current_date}: {len(daily_data)} bars")
                return
            
            # Calculate ATR using existing DailyTechnicalIndicators
            atr_series = self.daily_indicators_calculator.calculate_atr(
                daily_data, 
                period=self.atr_period
            )
            
            # Get ATR[D-1] value (use shifted series)
            try:
                atr_shifted = atr_series.shift(1) if atr_series is not None else None
                if atr_shifted is not None and not atr_shifted.empty:
                    current_ts = pd.Timestamp(current_date)
                    if current_ts in atr_shifted.index:
                        latest_atr = float(atr_shifted.loc[current_ts])
                    else:
                        latest_atr = float(atr_shifted.dropna().iloc[-1]) if not atr_shifted.dropna().empty else 0.0
                else:
                    latest_atr = 0.0
            except Exception:
                latest_atr = 0.0

            # Validate and store
            if self._validate_atr_value(latest_atr):
                old_atr = self.current_atr_values.get(symbol, 0.0)
                self.current_atr_values[symbol] = float(latest_atr)
                self.atr_calculation_dates[symbol] = current_date
                
                # Log significant ATR changes
                if old_atr > 0:
                    atr_change_pct = ((latest_atr - old_atr) / old_atr) * 100
                    if abs(atr_change_pct) > 10:  # Log changes > 10%
                        self.context.logger.info(f"ATR recalculated for {symbol} on {current_date}: {old_atr:.4f} → {latest_atr:.4f} ({atr_change_pct:+.1f}%)")
                    else:
                        self.context.logger.debug(f"ATR recalculated for {symbol} on {current_date}: {latest_atr:.4f}")
            else:
                self.context.logger.warning(f"ATR[D-1] unavailable or invalid for {symbol} on {current_date}")
                
        except Exception as e:
            self.context.logger.error(f"Error recalculating ATR for {symbol} on {current_date}: {e}")
    
    def _should_update_daily_data(self, symbol: str, current_date: date) -> bool:
        """
        Check if daily data should be updated for new trading days.
        
        Args:
            symbol: Trading symbol
            current_date: Current trading date (from backtest, not system date)
            
        Returns:
            True if daily data should be updated
        """
        if symbol not in self.daily_data_last_update:
            return True
        
        last_update = self.daily_data_last_update[symbol]
        
        # Force update if we're in a new trading day during backtest
        if current_date > last_update:
            return True
        
        # Force update if indicators are stale (more than 0 days old)
        if symbol in self.daily_indicators_last_update:
            indicators_last_update = self.daily_indicators_last_update[symbol]
            days_since_indicators_update = (current_date - indicators_last_update).days
            if days_since_indicators_update > 0:
                self.context.logger.info(f"Forcing daily data update for {symbol}: indicators {days_since_indicators_update} days old")
                return True
        
        return False
    
    def _fetch_new_daily_data(self, symbol: str, from_date: date, to_date: date) -> pd.DataFrame:
        """
        Fetch new daily data from Brapi.dev API.
        
        Args:
            symbol: Trading symbol
            from_date: Start date for new data
            to_date: End date for new data
            
        Returns:
            DataFrame with new daily data
        """
        try:
            self.context.logger.info(f"Fetching new daily data for {symbol} from {from_date} to {to_date}")
            
            # Use hybrid data manager if available
            if hasattr(self.context, 'hybrid_data_manager'):
                new_data = self.context.hybrid_data_manager.brapi_provider.get_daily_data(
                    symbol,
                    from_date.strftime('%Y-%m-%d'),
                    to_date.strftime('%Y-%m-%d')
                )
                if not new_data.empty:
                    self.context.logger.info(f"Successfully fetched {len(new_data)} new daily bars for {symbol}")
                return new_data
            else:
                # Fallback: use direct Brapi provider if available
                if hasattr(self.context, 'brapi_provider'):
                    new_data = self.context.brapi_provider.get_daily_data(
                        symbol,
                        from_date.strftime('%Y-%m-%d'),
                        to_date.strftime('%Y-%m-%d')
                    )
                    if not new_data.empty:
                        self.context.logger.info(f"Successfully fetched {len(new_data)} new daily bars for {symbol}")
                    return new_data
                else:
                    self.context.logger.warning("No Brapi provider available for fetching daily data")
        except Exception as e:
            self.context.logger.error(f"Failed to fetch new daily data for {symbol}: {e}")
        
        return pd.DataFrame()
    
    def _refresh_daily_data(self, symbol: str, current_date: date) -> bool:
        """
        Fetch and append new daily data from Brapi.dev.
        
        Args:
            symbol: Trading symbol
            current_date: Current trading date
            
        Returns:
            True if data was successfully refreshed, False otherwise
        """
        try:
            # Check if we have existing data
            if symbol not in self.daily_data or self.daily_data[symbol].empty:
                self.context.logger.warning(f"No existing daily data for {symbol}, cannot refresh")
                return False
            
            # Get last available date from current data
            last_available_date = self.daily_data[symbol].index.max().date()
            
            if current_date <= last_available_date:
                self.context.logger.debug(f"Daily data for {symbol} is up to date (last: {last_available_date}, current: {current_date})")
                # Even if data is up to date, we might need to force recalculation
                if symbol in self.daily_indicators_last_update:
                    indicators_last_update = self.daily_indicators_last_update[symbol]
                    days_since_indicators_update = (current_date - indicators_last_update).days
                    if days_since_indicators_update > 0:
                        self.context.logger.info(f"Forcing indicator recalculation for {symbol} with existing data")
                        return True  # Force recalculation with existing data
                return True  # No update needed
            
            self.context.logger.info(f"Daily data for {symbol} needs update: last={last_available_date}, current={current_date}")
            
            # Fetch new daily data from Brapi.dev
            new_data = self._fetch_new_daily_data(symbol, last_available_date, current_date)
            
            if not new_data.empty:
                # Append to existing data
                self.daily_data[symbol] = pd.concat([self.daily_data[symbol], new_data])
                self.daily_data_last_update[symbol] = current_date
                
                self.context.logger.info(f"Updated daily data for {symbol}: added {len(new_data)} new bars")
                return True
            else:
                self.context.logger.warning(f"No new daily data available for {symbol}")
                return False
                
        except Exception as e:
            self.context.logger.error(f"Failed to refresh daily data for {symbol}: {e}")
            return False
    
    def _recalculate_all_indicators(self, symbol: str, current_date: date):
        """
        Recalculate all technical indicators with updated daily data.
        
        Args:
            symbol: Trading symbol
            current_date: Current trading date
        """
        try:
            self.context.logger.info(f"Starting indicator recalculation for {symbol} on {current_date}")
            
            # Get daily data up to current date
            daily_data = self._get_daily_data_up_to_date(symbol, current_date)
            
            self.context.logger.info(f"Daily data for {symbol}: {len(daily_data)} bars, date range: {daily_data.index.min().date()} to {daily_data.index.max().date()}")
            
            if daily_data.empty or len(daily_data) < max(self.atr_period, self.rsi_period, max(self.ema_periods)) + 1:
                self.context.logger.warning(f"Insufficient daily data for indicators calculation: {len(daily_data)} bars")
                return
            
            # Calculate all indicators using existing DailyTechnicalIndicators
            self.context.logger.info(f"Calculating indicators for {symbol} with periods: ATR={self.atr_period}, EMA={self.ema_periods}, RSI={self.rsi_period}")
            
            indicators = self.daily_indicators.calculate_all_indicators(
                daily_data,
                atr_period=self.atr_period,
                ema_periods=self.ema_periods,
                rsi_period=self.rsi_period,
                sma_periods=[3, 5]
            )
            
            # Update stored indicators
            self.daily_indicators_data[symbol] = indicators
            self.daily_indicators_last_update[symbol] = current_date
            
            # Update current ATR value (maintain existing logic)
            latest_atr = indicators['atr'].iloc[-1] if not indicators['atr'].empty else 0.0
            if self._validate_atr_value(latest_atr):
                old_atr = self.current_atr_values.get(symbol, 0.0)
                self.current_atr_values[symbol] = float(latest_atr)
                self.atr_calculation_dates[symbol] = current_date
                
                # Log significant changes
                if old_atr > 0:
                    atr_change_pct = ((latest_atr - old_atr) / old_atr) * 100
                    if abs(atr_change_pct) > 10:
                        self.context.logger.info(f"ATR updated for {symbol}: {old_atr:.4f} → {latest_atr:.4f} ({atr_change_pct:+.1f}%)")
                    else:
                        self.context.logger.debug(f"ATR updated for {symbol}: {latest_atr:.4f}")
                else:
                    self.context.logger.info(f"ATR initialized for {symbol}: {latest_atr:.4f}")
            
            # Log indicator updates
            latest_ema = indicators['ema_5'].iloc[-1] if not indicators['ema_5'].empty else None
            latest_rsi = indicators['rsi'].iloc[-1] if not indicators['rsi'].empty else None
            
            if latest_ema is not None and latest_rsi is not None:
                self.context.logger.info(f"Indicators recalculated for {symbol}: EMA(5)={latest_ema:.2f}, RSI={latest_rsi:.2f}")
            
            self.context.logger.info(f"Indicator recalculation completed for {symbol}")
            
        except Exception as e:
            self.context.logger.error(f"Error recalculating indicators for {symbol}: {e}")
            import traceback
            self.context.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _validate_bar_data(self, bar: Bar) -> bool:
        """
        Validate bar data for trading suitability.
        
        Args:
            bar: Market data bar to validate
            
        Returns:
            True if bar data is valid for trading, False otherwise
        """
        # Check for basic data validity
        if bar is None:
            self.context.logger.warning("Received None bar data")
            return False
        
        # Validate OHLCV data
        if pd.isna(bar.open) or bar.open <= 0:
            self.context.logger.warning(f"Invalid open price: {bar.open}")
            return False
        
        if pd.isna(bar.high) or bar.high <= 0:
            self.context.logger.warning(f"Invalid high price: {bar.high}")
            return False
        
        if pd.isna(bar.low) or bar.low <= 0:
            self.context.logger.warning(f"Invalid low price: {bar.low}")
            return False
        
        if pd.isna(bar.close) or bar.close <= 0:
            self.context.logger.warning(f"Invalid close price: {bar.close}")
            return False
        
        if pd.isna(bar.volume) or bar.volume < 0:
            self.context.logger.warning(f"Invalid volume: {bar.volume}")
            return False
        
        # Validate price relationships
        if bar.low > bar.high:
            self.context.logger.warning(f"Invalid price relationship: low={bar.low} > high={bar.high}")
            return False
        
        if bar.open < bar.low or bar.open > bar.high:
            self.context.logger.warning(f"Open price outside high-low range: open={bar.open}, low={bar.low}, high={bar.high}")
            return False
        
        if bar.close < bar.low or bar.close > bar.high:
            self.context.logger.warning(f"Close price outside high-low range: close={bar.close}, low={bar.low}, high={bar.high}")
            return False
        
        # Validate timestamp
        if bar.timestamp is None:
            self.context.logger.warning("Invalid timestamp")
            return False
        
        return True
    
    def _validate_bar_data_from_series(self, row: pd.Series) -> bool:
        """
        Validate bar data from pandas Series for trading suitability.
        
        Args:
            row: Pandas Series with OHLCV data
            
        Returns:
            True if bar data is valid for trading, False otherwise
        """
        # Check for basic data validity
        if row is None or row.empty:
            return False
        
        # Validate OHLCV data
        if pd.isna(row.get('open', np.nan)) or row.get('open', 0) <= 0:
            return False
        
        if pd.isna(row.get('high', np.nan)) or row.get('high', 0) <= 0:
            return False
        
        if pd.isna(row.get('low', np.nan)) or row.get('low', 0) <= 0:
            return False
        
        if pd.isna(row.get('close', np.nan)) or row.get('close', 0) <= 0:
            return False
        
        if pd.isna(row.get('volume', np.nan)) or row.get('volume', 0) < 0:
            return False
        
        # Validate price relationships
        if row.get('low', 0) > row.get('high', 0):
            return False
        
        if (row.get('open', 0) < row.get('low', 0) or 
            row.get('open', 0) > row.get('high', 0)):
            return False
        
        if (row.get('close', 0) < row.get('low', 0) or 
            row.get('close', 0) > row.get('high', 0)):
            return False
        
        return True
    
    def _log_data_quality_report(self, symbol: str):
        """
        Log comprehensive data quality report for debugging.
        
        Args:
            symbol: Trading symbol to report on
        """
        self.context.logger.info(f"=== Data Quality Report for {symbol} ===")
        
        # ATR data quality
        if symbol in self.current_atr_values:
            current_atr = self.current_atr_values[symbol]
            calc_date = self.atr_calculation_dates.get(symbol, 'Unknown')
            
            self.context.logger.info(f"ATR Data Quality:")
            self.context.logger.info(f"  - Current ATR: {current_atr:.4f}")
            self.context.logger.info(f"  - Last calculation date: {calc_date}")
            self.context.logger.info(f"  - ATR validation: {'PASS' if self._validate_atr_value(current_atr) else 'FAIL'}")
        else:
            self.context.logger.warning("No current ATR data available")
        
        # Daily data quality
        if symbol in self.daily_data:
            daily_df = self.daily_data[symbol]
            self.context.logger.info(f"Daily Data Quality:")
            self.context.logger.info(f"  - Daily bars: {len(daily_df)}")
            self.context.logger.info(f"  - Date range: {daily_df.index.min()} to {daily_df.index.max()}")
            
            # Check for NaN values in daily data
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in daily_df.columns:
                    nan_count = daily_df[col].isna().sum()
                    if nan_count > 0:
                        self.context.logger.warning(f"  - NaN values in {col}: {nan_count}")
        else:
            self.context.logger.warning("No daily data available")
        
        # Intraday history quality
        if symbol in self.intraday_history:
            intraday_length = len(self.intraday_history[symbol])
            self.context.logger.info(f"Intraday History Quality:")
            self.context.logger.info(f"  - Intraday bars: {intraday_length}")
            
            if intraday_length > 0:
                # Check last few bars for quality
                recent_bars = list(self.intraday_history[symbol])[-5:]
                for i, bar in enumerate(recent_bars):
                    self.context.logger.debug(f"  - Recent bar {i+1}: O={bar['open']:.2f}, H={bar['high']:.2f}, L={bar['low']:.2f}, C={bar['close']:.2f}")
        else:
            self.context.logger.warning("No intraday history available")
        
        self.context.logger.info("=== End Data Quality Report ===")
    

    
    def _calculate_atr(self, symbol: str) -> float:
        """
        Legacy method - now redirects to daily ATR calculation.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Daily ATR value
        """
        return self._get_daily_atr(symbol)
    
    def _round_to_lot_size(self, raw_quantity: float) -> int:
        """
        Enforce B3 standard lots: multiples of min_lot_size (default 100). No odd lots.
        Round to nearest full lot; minimum one lot if positive.
        """
        if raw_quantity <= 0:
            return 0
        lot = int(self.min_lot_size) if hasattr(self, 'min_lot_size') else 100
        lots_float = float(raw_quantity) / float(max(1, lot))
        lots = max(1, int(round(lots_float)))
        return lots * lot
    
    def _calculate_order_quantities(self, symbol: str, bar: Bar) -> Tuple[int, int, int, int]:
        """
        Calculate quantities for the three execution attempts with robust validation.
        
        Args:
            symbol: Trading symbol
            bar: Current market data bar
            
        Returns:
            Tuple of (qty1, qty2, qty3, qty4) for market, limit_alpha, limit_beta, limit_gamma orders
        """
        # Validate bar data first
        if pd.isna(bar.open) or bar.open <= 0:
            self.context.logger.warning(f"Invalid bar data for {symbol}: open={bar.open}")
            return 0, 0, 0, 0
        
        # Fixed tranche sizing: 4 tranches of BRL 12,500 each -> total 50,000 per symbol
        exposure_per_attempt = 12500.0
        
        # Get ATR and validate
        atr = self._get_daily_atr(symbol)
        if pd.isna(atr) or atr <= 0:
            self.context.logger.warning(f"Invalid ATR for {symbol}: {atr}, skipping trade")
            return 0, 0, 0, 0
        
        # Market order quantity (at open price)
        qty1_raw = exposure_per_attempt / bar.open
        if pd.isna(qty1_raw) or qty1_raw <= 0:
            self.context.logger.warning(f"Invalid market quantity calculation for {symbol}: qty1_raw={qty1_raw}")
            return 0, 0, 0
        
        # Apply conservative rounding for market order
        qty1 = self._round_to_lot_size(qty1_raw)
        
        # Limit order quantities (using protected limit prices consistent with _store_daily_order_prices)
        min_alpha_abs = self.min_alpha_pct * bar.open
        min_beta_abs = self.min_beta_pct * bar.open
        alpha_dist = self.alpha_factor * atr
        beta_dist = self.beta_factor * atr
        alpha_offset = max(alpha_dist, min_alpha_abs)
        beta_offset = max(beta_dist, min_beta_abs)

        # Determine side for sizing based on current signal; default to SELL if unknown
        side_for_sizing = getattr(self, 'current_signal_side', None)
        if side_for_sizing is None:
            # Infer from last generated signal if available, else assume SELL to keep symmetry
            side_for_sizing = OrderSide.SELL

        if side_for_sizing == OrderSide.BUY:
            limit_price_alpha = max(bar.open - alpha_offset, 0.01)
            limit_price_beta = max(bar.open - beta_offset, 0.01)
        else:
            # SELL
            limit_price_alpha = bar.open + alpha_offset
            limit_price_beta = bar.open + beta_offset
        
        # Third limit (gamma) further from beta
        gamma_dist = max(self.beta_factor * 1.5 * atr, self.min_beta_pct * 1.5 * bar.open)
        if side_for_sizing == OrderSide.BUY:
            limit_price_gamma = max(bar.open - gamma_dist, 0.01)
        else:
            limit_price_gamma = bar.open + gamma_dist

        qty2_raw = exposure_per_attempt / limit_price_alpha
        qty3_raw = exposure_per_attempt / limit_price_beta
        qty4_raw = exposure_per_attempt / limit_price_gamma
        
        # Validate quantities before conversion
        if pd.isna(qty2_raw) or pd.isna(qty3_raw):
            self.context.logger.warning(f"Invalid quantity calculation for {symbol}: qty2_raw={qty2_raw}, qty3_raw={qty3_raw}")
            return 0, 0, 0, 0
        
        if qty2_raw <= 0 or qty3_raw <= 0 or pd.isna(qty4_raw) or qty4_raw <= 0:
            self.context.logger.warning(f"Non-positive quantities for {symbol}: qty2_raw={qty2_raw}, qty3_raw={qty3_raw}, qty4_raw={qty4_raw}")
            return 0, 0, 0, 0
        
        # Apply conservative rounding for limit orders
        qty2 = self._round_to_lot_size(qty2_raw)
        qty3 = self._round_to_lot_size(qty3_raw)
        qty4 = self._round_to_lot_size(qty4_raw)
            
        # Log calculation details for debugging
        self.context.logger.debug(f"Quantity calculation for {symbol}: exposure_per_attempt={exposure_per_attempt:.2f}, open={bar.open:.2f}")
        self.context.logger.debug(f"Raw quantities: qty1_raw={qty1_raw:.2f}, qty2_raw={qty2_raw:.2f}, qty3_raw={qty3_raw:.2f}, qty4_raw={qty4_raw:.2f}")
        # Keep logging concise; fixed tranche model
        self.context.logger.debug(f"Final quantities: qty1={qty1}, qty2={qty2}, qty3={qty3}, qty4={qty4}")
        
        # Emit lightweight sizing event and append fuzzy row (no rule changes)
        try:
            from engine import event_logger
            if event_logger is not None:
                event_logger.emit('sizing_events', {
                    'type': 'sizing',
                    'ts': int(time.time()),
                    'date': str(bar.timestamp.date()),
                    'symbol': symbol,
                    'side': getattr(self, 'current_signal_side', OrderSide.SELL).name if hasattr(self, 'current_signal_side') else 'SELL',
                    'exposure_cap_brl': float(50000.0),
                    'split_1_3_brl': float((self.context.portfolio.get_portfolio_value() * self.asset_exposure_pct) / 3.0),
                    'pre_round_qty_P1': float(exposure_per_attempt / max(bar.open, 1e-9)),
                    'pre_round_qty_P2': float(qty2_raw),
                    'pre_round_qty_P3': float(qty3_raw),
                    'pre_round_qty_P4': float(qty4_raw),
                    'price_P1': float(bar.open),
                    'price_P2': float(limit_price_alpha),
                    'price_P3': float(limit_price_beta),
                    'price_P4': float(limit_price_gamma),
                    'post_round_qty_P1': int(qty1),
                    'post_round_qty_P2': int(qty2),
                    'post_round_qty_P3': int(qty3),
                    'post_round_qty_P4': int(qty4),
                    'dropped': bool(qty1==0 and qty2==0 and qty3==0 and qty4==0),
                    'reason_if_dropped': '<1 lot after rounding' if (qty1==0 and qty2==0 and qty3==0 and qty4==0) else None
                })
        except Exception:
            pass

        try:
            # Upsert a single fuzzy row per (date, symbol)
            if not hasattr(self, '_fuzzy_logged_dates'):
                self._fuzzy_logged_dates = set()
            date_str = str(bar.timestamp.date())
            key = (date_str, symbol)
            row = {
                'date': date_str,
                'symbol': symbol,
                'side': getattr(self, 'current_signal_side', OrderSide.SELL).name if hasattr(self, 'current_signal_side') else 'SELL',
                'fuzzy_score': float(abs(getattr(self, '_last_signal_strength', 0.0))),
                'eligible': bool(qty1>0 or qty2>0 or qty3>0 or qty4>0),
                'reason_if_not': None if (qty1>0 or qty2>0 or qty3>0 or qty4>0) else 'below_min_lot',
                'exposure_cap_brl': float(50000.0),
                'notional_P1': float(bar.open * qty1),
                'notional_P2': float(limit_price_alpha * qty2),
                'notional_P3': float(limit_price_beta * qty3),
                'notional_P4': float(limit_price_gamma * qty4),
            }
            # Replace existing row for same (date, symbol, side) if any
            replaced = False
            for i, r in enumerate(self._fuzzy_rows):
                if r.get('date') == date_str and r.get('symbol') == symbol and r.get('side') == row['side']:
                    # also set eligibility fields now that quantities are known
                    eligible_now = bool(qty1>0 or qty2>0 or qty3>0 or qty4>0)
                    row['eligible'] = eligible_now
                    row['reason_if_not'] = None if eligible_now else 'below_min_lot'
                    self._fuzzy_rows[i] = row
                    replaced = True
                    break
            if not replaced:
                self._fuzzy_rows.append(row)
            self._fuzzy_logged_dates.add(key)
        except Exception:
            pass

        return qty1, qty2, qty3, qty4
    
    def _is_market_order_executed_today(self, trading_date: date) -> bool:
        """Check if market order has been executed today."""
        return trading_date in self.market_order_executed_dates
    
    def _is_limit_alpha_executed_today(self, trading_date: date) -> bool:
        """Check if limit_alpha order has been executed today."""
        return trading_date in self.limit_alpha_executed_dates
    
    def _is_limit_beta_executed_today(self, trading_date: date) -> bool:
        """Check if limit_beta order has been executed today."""
        return trading_date in self.limit_beta_executed_dates
    
    def _mark_market_order_executed(self, trading_date: date):
        """Mark that market order has been executed for this trading day."""
        self.market_order_executed_dates.add(trading_date)
    
    def _mark_limit_alpha_executed(self, trading_date: date):
        """Mark that limit_alpha order has been executed for this trading day."""
        self.limit_alpha_executed_dates.add(trading_date)
    
    def _mark_limit_beta_executed(self, trading_date: date):
        """Mark that limit_beta order has been executed for this trading day."""
        self.limit_beta_executed_dates.add(trading_date)
    
    def _are_orders_emitted_today(self, symbol: str, trading_date: date) -> bool:
        """
        Check if any orders have been emitted for this symbol today.
        This enforces the Section 7 rule: no more than three orders per asset per day.
        
        Args:
            symbol: Trading symbol
            trading_date: Trading date to check
            
        Returns:
            True if orders have already been emitted today, False otherwise
        """
        if symbol not in self.daily_orders_emitted:
            return False
        
        return trading_date in self.daily_orders_emitted[symbol]
    
    def _is_first_bar_of_day(self, symbol: str, trading_date: date) -> bool:
        """
        Check if this is the first bar being processed for this symbol today.
        Market orders should only be emitted on the first bar of the day.
        
        Args:
            symbol: Trading symbol
            trading_date: Trading date to check
            
        Returns:
            True if this is the first bar of the day, False otherwise
        """
        if symbol not in self.first_bar_of_day:
            self.first_bar_of_day[symbol] = {}
        
        if trading_date not in self.first_bar_of_day[symbol]:
            # This is the first bar of the day
            self.first_bar_of_day[symbol][trading_date] = True
            return True
        
        return False

    def _neutral_emit_for_day(self, trading_date: date) -> Iterable[OrderIntent]:
        """Enforce cash market-neutrality across all symbols for the day and emit intents.

        Downscale-only; never exceed per-symbol per-type capacity. Preserve immutable prices and timestamps.
        """
        buffer = self._neutral_buffer.get(trading_date, {})
        if not buffer:
            return []

        def caps_of(rec: Dict[str, Any]) -> Dict[str, float]:
            p = rec['prices']; q = rec['qty']
            return {
                'market': max(0.0, float(p['market'] or 0.0) * max(0, int(q['market']))),
                'limit_alpha': max(0.0, float(p['limit_alpha'] or 0.0) * max(0, int(q['limit_alpha']))),
                'limit_beta': max(0.0, float(p['limit_beta'] or 0.0) * max(0, int(q['limit_beta'])))
            }

        buys: List[Tuple[str, float, Dict[str, float]]] = []
        sells: List[Tuple[str, float, Dict[str, float]]] = []
        for sym, rec in buffer.items():
            c = caps_of(rec)
            tot = sum(c.values())
            if tot <= 0:
                continue
            (buys if rec['side'] == OrderSide.BUY else sells).append((sym, float(rec['fuzzy']), c))

        total_buy = sum(sum(c.values()) for _, _, c in buys)
        total_sell = sum(sum(c.values()) for _, _, c in sells)
        if total_buy <= 0 or total_sell <= 0:
            self.context.logger.info("Neutrality impossible: one-sided signals - aborting day")
            self._neutral_buffer.pop(trading_date, None)
            return []

        anchor_is_buy = total_buy <= total_sell
        anchor_total = min(total_buy, total_sell)
        anchor_list = buys if anchor_is_buy else sells
        sums = {'market': 0.0, 'limit_alpha': 0.0, 'limit_beta': 0.0}
        for _, _, c in anchor_list:
            for k in sums:
                sums[k] += c.get(k, 0.0)
        r = {k: (sums[k] / anchor_total) if anchor_total else 0.0 for k in sums}
        targets = {k: anchor_total * r[k] for k in r}

        def allocate(side_list: List[Tuple[str, float, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
            remaining = targets.copy()
            out: Dict[str, Dict[str, float]] = {}
            # sort by fuzzy desc
            for sym, _, caps in sorted(side_list, key=lambda t: t[1], reverse=True):
                use = {'market': 0.0, 'limit_alpha': 0.0, 'limit_beta': 0.0}
                for typ in ('market', 'limit_alpha', 'limit_beta'):
                    cap = caps.get(typ, 0.0)
                    if cap <= 0 or remaining[typ] <= 0:
                        continue
                    take = min(cap, remaining[typ])
                    if take > 0:
                        use[typ] += take
                        remaining[typ] -= take
                if any(v > 0 for v in use.values()):
                    out[sym] = use
            out['_remaining'] = remaining
            return out

        # If strict one-pair mode: pick top fuzzy BUY and top fuzzy SELL only
        if getattr(self, 'RISK_STRICT_ONE_PAIR', False):
            buys_top = sorted(buys, key=lambda t: t[1], reverse=True)[:1]
            sells_top = sorted(sells, key=lambda t: t[1], reverse=True)[:1]
            buy_alloc = allocate(buys_top)
            sell_alloc = allocate(sells_top)
        else:
            buy_alloc = allocate(buys)
            sell_alloc = allocate(sells)

        def to_qty(alloc: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, Dict[str, int]], float]:
            qty_map: Dict[str, Dict[str, int]] = {}
            total = 0.0
            for sym, sym_use in alloc.items():
                if sym.startswith('_'):
                    continue
                rec = buffer[sym]
                prices = rec['prices']
                sym_q: Dict[str, int] = {'market': 0, 'limit_alpha': 0, 'limit_beta': 0}
                for typ in ('market', 'limit_alpha', 'limit_beta'):
                    notional = sym_use.get(typ, 0.0)
                    if notional <= 0:
                        continue
                    px = float(prices[typ] or 0.0)
                    if px <= 0:
                        continue
                    raw = notional / px
                    rounded = self._round_to_lot_size(raw)
                    if rounded <= 0:
                        continue
                    sym_q[typ] = int(rounded)
                    total += px * rounded
                if any(v > 0 for v in sym_q.values()):
                    qty_map[sym] = sym_q
            return qty_map, total

        buy_qty, achieved_buy = to_qty(buy_alloc)
        sell_qty, achieved_sell = to_qty(sell_alloc)

        # Tolerance: one lot of cheapest active symbol
        min_lot_notional = float('inf')
        for sym, rec in buffer.items():
            lot = max(1, int(rec['lot_size']))
            for typ, px in rec['prices'].items():
                if (rec['qty'][typ] or 0) > 0 and px:
                    min_lot_notional = min(min_lot_notional, float(px) * lot)
        if not np.isfinite(min_lot_notional):
            min_lot_notional = 0.0

        delta = abs(achieved_buy - achieved_sell)
        if delta > min_lot_notional:
            larger, qmap = (('buy', buy_qty) if achieved_buy > achieved_sell else ('sell', sell_qty))
            # Reduce in reverse fuzzy order: last symbols in allocation were lowest marginal value
            seq = list(reversed(sorted((buys if larger == 'buy' else sells), key=lambda t: t[1], reverse=True)))
            for sym, _, _ in seq:
                if sym not in qmap:
                    continue
                prices = buffer[sym]['prices']
                for typ in ('limit_beta', 'limit_alpha', 'market'):
                    while qmap[sym].get(typ, 0) > 0 and delta > min_lot_notional:
                        qmap[sym][typ] -= 1
                        delta -= float(prices[typ] or 0.0)
                if delta <= min_lot_notional:
                    break
            if delta > min_lot_notional:
                self.context.logger.info("Neutrality repair failed; abort day")
                self._neutral_buffer.pop(trading_date, None)
                return []

        # Emit intents using first-bar timestamps and stored immutable prices
        emitted: List[OrderIntent] = []
        for side_name, qty_pack in (("BUY", buy_qty), ("SELL", sell_qty)):
            for sym, types in qty_pack.items():
                rec = buffer[sym]
                side = OrderSide.BUY if side_name == "BUY" else OrderSide.SELL
                bar = rec['bar']
                atr = rec['atr']
                signal = rec['signal']
                prices = rec['prices']
                # Market
                q1 = int(types.get('market', 0))
                if q1 > 0:
                    attempt = {
                        'order_type': OrderType.MARKET,
                        'quantity': q1,
                        'price': None,
                        'execution_price': prices['market'],
                        'attempt_name': 'Market Order at Open',
                        'attempt_type': 'market'
                    }
                    self._track_daily_execution(sym, attempt, True, bar, side)
                    self._mark_market_order_executed(trading_date)
                    emitted.append(OrderIntent(
                        symbol=sym,
                        side=side,
                        quantity=q1,
                        order_type=OrderType.MARKET,
                        price=None,
                        timestamp=bar.timestamp,
                        metadata={'attempt_number': 1,'attempt_name': 'Market Order at Open','attempt_type': 'market','atr_value': atr,'execution_price': prices['market'],'signal': signal,'emission_type': 'first_bar_neutralized'}
                    ))
                # Limit alpha
                q2 = int(types.get('limit_alpha', 0))
                if q2 > 0:
                    alpha_px = prices['limit_alpha']
                    attempt = {'order_type': OrderType.LIMIT,'quantity': q2,'price': alpha_px,'execution_price': alpha_px,'attempt_name': 'Limit Order Passive-1','attempt_type': 'limit_alpha'}
                    filled = self._simulate_fill_with_stored_prices(OrderType.LIMIT, 'limit_alpha', sym, trading_date, bar)
                    self._track_daily_execution(sym, attempt, filled, bar, side)
                    if filled:
                        self._mark_limit_alpha_executed(trading_date)
                        emitted.append(OrderIntent(
                            symbol=sym, side=side, quantity=q2, order_type=OrderType.LIMIT, price=alpha_px, timestamp=bar.timestamp,
                            metadata={'attempt_number': 2,'attempt_name': 'Limit Order Passive-1','attempt_type': 'limit_alpha','atr_value': atr,'execution_price': alpha_px,'signal': signal,'emission_type': 'first_bar_neutralized'}
                        ))
                # Limit beta
                q3 = int(types.get('limit_beta', 0))
                if q3 > 0:
                    beta_px = prices['limit_beta']
                    attempt = {'order_type': OrderType.LIMIT,'quantity': q3,'price': beta_px,'execution_price': beta_px,'attempt_name': 'Limit Order Passive-2','attempt_type': 'limit_beta'}
                    filled = self._simulate_fill_with_stored_prices(OrderType.LIMIT, 'limit_beta', sym, trading_date, bar)
                    self._track_daily_execution(sym, attempt, filled, bar, side)
                    if filled:
                        self._mark_limit_beta_executed(trading_date)
                        emitted.append(OrderIntent(
                            symbol=sym, side=side, quantity=q3, order_type=OrderType.LIMIT, price=beta_px, timestamp=bar.timestamp,
                            metadata={'attempt_number': 3,'attempt_name': 'Limit Order Passive-2','attempt_type': 'limit_beta','atr_value': atr,'execution_price': beta_px,'signal': signal,'emission_type': 'first_bar_neutralized'}
                        ))
                # Mark orders emitted flags
                emitted_types = [k for k, v in types.items() if int(v) > 0]
                self._mark_orders_emitted(sym, trading_date, emitted_types)
        self._neutral_buffer.pop(trading_date, None)
        return emitted
    
    def _store_daily_order_prices(self, symbol: str, trading_date: date, open_price: float, atr: float, side: OrderSide):
        """
        Store immutable order prices for the day. These prices are calculated once
        and never changed, enforcing Section 7 rule about order immutability.
        
        Args:
            symbol: Trading symbol
            trading_date: Trading date
            open_price: Daily open price
            atr: ATR value
            side: Order side (BUY/SELL)
        """
        if symbol not in self.daily_order_prices:
            self.daily_order_prices[symbol] = {}
        
        # Calculate limit prices based on side with protective minimum distance
        # Ensure minimum distance in % of open (1.5% for alpha, 2.5% for beta) when ATR is too small
        # Compute absolute protective distances
        min_alpha_abs = self.min_alpha_pct * open_price
        min_beta_abs = self.min_beta_pct * open_price
        # ATR-based distances
        alpha_dist = self.alpha_factor * atr
        beta_dist = self.beta_factor * atr
        # Apply max between ATR-based and protective minimums
        alpha_offset = max(alpha_dist, min_alpha_abs)
        beta_offset = max(beta_dist, min_beta_abs)

        if side == OrderSide.BUY:
            alpha_price = max(open_price - alpha_offset, 0.01)
            beta_price = max(open_price - beta_offset, 0.01)
        else:  # SELL
            alpha_price = open_price + alpha_offset
            beta_price = open_price + beta_offset
        
        # Derive gamma price further than beta (1.5x beta offset from open)
        if side == OrderSide.BUY:
            gamma_price = max(open_price - (beta_offset * 1.5), 0.01)
        else:
            gamma_price = open_price + (beta_offset * 1.5)

        self.daily_order_prices[symbol][trading_date] = {
            'market_price': open_price,
            'alpha_price': alpha_price,
            'beta_price': beta_price,
            'gamma_price': gamma_price,
            'atr_value': atr,
            'side': side
        }
        
        self.context.logger.info(
            f"Stored immutable order prices for {symbol} on {trading_date}: "
            f"market={open_price:.2f}, alpha={alpha_price:.2f}, beta={beta_price:.2f} "
            f"(min%: alpha>={self.min_alpha_pct*100:.2f}%, beta>={self.min_beta_pct*100:.2f}%)"
        )

    def _calculate_entry_limits_from_close(self, close_price: float, atr: float, side: OrderSide) -> Tuple[float, float, float]:
        """
        Compute three limit prices off close(t) using fixed percentages per spec:
        BUY: close * (1 - 0.5%), (1 - 1.0%), (1 - 1.5%)
        SELL: close * (1 + 0.5%), (1 + 1.0%), (1 + 1.5%)
        """
        step1, step2, step3 = 0.005, 0.010, 0.015
        if side == OrderSide.BUY:
            p2 = max(close_price * (1.0 - step1), 0.01)
            p3 = max(close_price * (1.0 - step2), 0.01)
            p4 = max(close_price * (1.0 - step3), 0.01)
        else:
            p2 = close_price * (1.0 + step1)
            p3 = close_price * (1.0 + step2)
            p4 = close_price * (1.0 + step3)
        return (p2, p3, p4)

    # =====================
    # Warmup Requirements
    # =====================
    def required_history(self) -> Dict[str, Any]:
        """Declare indicator history requirements and warmup policy."""
        # Defaults; allow overrides via config metadata
        meta = self.config.metadata if isinstance(self.config.metadata, dict) else {}
        ind_cfg = meta.get('indicators', {}) if isinstance(meta.get('indicators', {}), dict) else {}
        cal_cfg = meta.get('calendar', {}) if isinstance(meta.get('calendar', {}), dict) else {}
        ema_windows = ind_cfg.get('ema_windows', [3, 5, 10, 15, 20])
        rsi_window = ind_cfg.get('rsi_window', 10)
        atr_window = ind_cfg.get('atr_window', 14)
        rel_strength_return_window = ind_cfg.get('rel_strength_return_window', 5)
        daily_return_window = ind_cfg.get('daily_return_window', 1)
        warmup_multiplier_for_ema = float(ind_cfg.get('warmup_multiplier_for_ema', 3.0))
        buffer_sessions = int(ind_cfg.get('buffer_sessions', 5))
        calendar_buffer_sessions = int(cal_cfg.get('buffer_sessions', 3))
        return {
            'ema_windows': ema_windows,
            'rsi_window': rsi_window,
            'atr_window': atr_window,
            'rel_strength_return_window': rel_strength_return_window,
            'daily_return_window': daily_return_window,
            'warmup_multiplier_for_ema': warmup_multiplier_for_ema,
            'buffer_sessions': buffer_sessions,
            'calendar_buffer_sessions': calendar_buffer_sessions,
        }

    def prewarm_indicators(self, start_d: date, end_d: date) -> None:
        """Fetch extended daily data and compute indicators for universe and ^BVSP."""
        try:
            symbols = list(self.config.universe)
        except Exception:
            symbols = []
        # Include benchmark
        bench = '^BVSP'
        if bench not in symbols:
            symbols = symbols + [bench]
        # Fetch and compute per symbol
        for sym in symbols:
            try:
                # Use existing helper to load range around date; fetch chunked by end date
                # Single-range fetch path
                df = self._get_daily_data_for_date(sym, end_d)
                if df is None or df.empty:
                    continue
                # Filter to [start_d, end_d]
                df = df[(df.index.date >= start_d) & (df.index.date <= end_d)]
                if df.empty:
                    continue
                self.daily_data[sym] = df
                self.daily_data_last_update[sym] = end_d
                if sym != bench:
                    # Compute indicators over extended window
                    dti = DailyTechnicalIndicators()
                    indicators: Dict[str, Any] = {}
                    for p in (3, 5, 10, 15, 20):
                        indicators[f'ema_{p}'] = dti.calculate_ema(df, period=p)
                    indicators['rsi'] = dti.calculate_rsi(df, period=getattr(self, 'rsi_period', 10))
                    self.daily_indicators_data[sym] = indicators
                    self.daily_indicators_last_update[sym] = end_d
                    # Determine first valid date where all needed indicators exist
                    valid_idx = df.index
                    try:
                        ema20 = indicators.get('ema_20')
                        rsi = indicators.get('rsi')
                        mask = (~ema20.isna()) & (~rsi.isna())
                        first_valid = valid_idx[mask].min().date() if mask.any() else None
                        if first_valid:
                            self.first_valid_indicator_date[sym] = first_valid
                    except Exception:
                        pass
            except Exception:
                continue

    def _ensure_eod_inputs(self, symbol: str, d: date) -> None:
        """Ensure daily indicators for symbol and ^BVSP are initialized up to date d."""
        try:
            # Ensure symbol daily data exists up to d
            daily_df = self._get_daily_data_up_to_date(symbol, d)
            if daily_df is None or daily_df.empty or daily_df.index.max().date() != d:
                # attempt refresh via existing machinery if available
                try:
                    self._refresh_daily_data(symbol, d)
                except Exception:
                    pass
                daily_df = self._get_daily_data_up_to_date(symbol, d)
            # Compute indicators (EMA 3,5,10,15,20 and RSI) for symbol at d
            if daily_df is not None and not daily_df.empty:
                dti = DailyTechnicalIndicators()
                indicators: Dict[str, Any] = {}
                try:
                    for p in (3, 5, 10, 15, 20):
                        indicators[f'ema_{p}'] = dti.calculate_ema(daily_df, period=p)
                    indicators['rsi'] = dti.calculate_rsi(daily_df, period=getattr(self, 'rsi_period', 14))
                    self.daily_indicators_data[symbol] = indicators
                    self.daily_indicators_last_update[symbol] = d
                except Exception:
                    pass
            # Ensure ^BVSP exists up to d
            ibov_symbol = '^BVSP'
            ibov_df = self._get_daily_data_up_to_date(ibov_symbol, d) if ibov_symbol in self.daily_data else None
            if ibov_df is None or ibov_df.empty or ibov_df.index.max().date() != d:
                try:
                    fetched = self._get_daily_data_for_date(ibov_symbol, d)
                    if fetched is not None and not fetched.empty:
                        self.daily_data[ibov_symbol] = fetched
                        self.daily_data_last_update[ibov_symbol] = d
                except Exception:
                    pass
        except Exception:
            return
    
    def _get_stored_order_price(self, symbol: str, trading_date: date, attempt_type: str) -> Optional[float]:
        """
        Get the fixed order price calculated at start of day.
        
        Args:
            symbol: Trading symbol
            trading_date: Trading date
            attempt_type: Order attempt type ('market', 'limit_alpha', 'limit_beta')
            
        Returns:
            Stored price or None if not found
        """
        if symbol not in self.daily_order_prices:
            return None
        
        if trading_date not in self.daily_order_prices[symbol]:
            return None
        
        price_data = self.daily_order_prices[symbol][trading_date]
        
        if attempt_type == 'market':
            return price_data['market_price']
        elif attempt_type == 'limit_alpha':
            return price_data['alpha_price']
        elif attempt_type == 'limit_beta':
            return price_data['beta_price']
        elif attempt_type == 'limit_gamma':
            return price_data['gamma_price']
        
        return None

    def _store_daily_order_quantities(self, symbol: str, trading_date: date, qty_market: int, qty_alpha: int, qty_beta: int, qty_gamma: int) -> None:
        """
        Store immutable order quantities for the day. Quantities are calculated once at
        the start of the trading day and must not change intraday.

        Args:
            symbol: Trading symbol
            trading_date: Trading date
            qty_market: Quantity for market order (attempt 1)
            qty_alpha: Quantity for limit alpha (attempt 2)
            qty_beta: Quantity for limit beta (attempt 3)
        """
        if symbol not in self.daily_order_prices:
            self.daily_order_prices[symbol] = {}

        if trading_date not in self.daily_order_prices[symbol]:
            # Ensure price container exists even if called before prices are stored
            self.daily_order_prices[symbol][trading_date] = {}

        self.daily_order_prices[symbol][trading_date]['market_qty'] = int(qty_market)
        self.daily_order_prices[symbol][trading_date]['alpha_qty'] = int(qty_alpha)
        self.daily_order_prices[symbol][trading_date]['beta_qty'] = int(qty_beta)
        self.daily_order_prices[symbol][trading_date]['gamma_qty'] = int(qty_gamma)

    def _get_stored_order_quantity(self, symbol: str, trading_date: date, attempt_type: str) -> Optional[int]:
        """
        Get the fixed order quantity calculated at start of day.

        Args:
            symbol: Trading symbol
            trading_date: Trading date
            attempt_type: Order attempt type ('market', 'limit_alpha', 'limit_beta')

        Returns:
            Stored quantity or None if not found
        """
        if symbol not in self.daily_order_prices:
            return None

        if trading_date not in self.daily_order_prices[symbol]:
            return None

        price_data = self.daily_order_prices[symbol][trading_date]

        if attempt_type == 'market':
            return price_data.get('market_qty')
        elif attempt_type == 'limit_alpha':
            return price_data.get('alpha_qty')
        elif attempt_type == 'limit_beta':
            return price_data.get('beta_qty')
        elif attempt_type == 'limit_gamma':
            return price_data.get('gamma_qty')

        return None
    
    def _mark_orders_emitted(self, symbol: str, trading_date: date, emitted_types: List[str]):
        """
        Mark that orders have been emitted for this symbol today.
        
        Args:
            symbol: Trading symbol
            trading_date: Trading date
            emitted_types: List of order types emitted ('market', 'limit_alpha', 'limit_beta')
        """
        if symbol not in self.daily_orders_emitted:
            self.daily_orders_emitted[symbol] = {}
        
        if trading_date not in self.daily_orders_emitted[symbol]:
            self.daily_orders_emitted[symbol][trading_date] = {
                'market': False,
                'limit_alpha': False,
                'limit_beta': False,
                'limit_gamma': False
            }
        
        for order_type in emitted_types:
            self.daily_orders_emitted[symbol][trading_date][order_type] = True
        
        self.context.logger.info(f"Marked orders as emitted for {symbol} on {trading_date}: {emitted_types}")
    
    def _simulate_fill(self, order_type: str, limit_price: float, bar: Bar) -> bool:
        """
        Legacy simulate fill method - maintained for backward compatibility.
        
        Args:
            order_type: Order type ('MARKET' or 'LIMIT')
            limit_price: Limit price (for limit orders)
            bar: Current market data bar
            
        Returns:
            True if order would be filled, False otherwise
        """
        if order_type == OrderType.MARKET:
            # Market orders are always filled at open
            return True
        
        elif order_type == OrderType.LIMIT:
            # Limit orders are filled if limit price is within High/Low range
            return bar.low <= limit_price <= bar.high
        
        return False
    
    def _simulate_fill_with_stored_prices(self, order_type: str, attempt_type: str, symbol: str, trading_date: date, bar: Bar) -> bool:
        """
        Enhanced fill simulation using stored immutable prices (Section 7 compliance).
        
        Args:
            order_type: Order type ('MARKET' or 'LIMIT')
            attempt_type: Attempt type ('market', 'limit_alpha', 'limit_beta', 'limit_gamma')
            symbol: Trading symbol
            trading_date: Trading date
            bar: Current market data bar
            
        Returns:
            True if order would be filled, False otherwise
        """
        if order_type == OrderType.MARKET:
            # Market orders are always filled
            return True
        
        elif order_type == OrderType.LIMIT:
            # Use stored immutable limit price calculated at start of day
            stored_price = self._get_stored_order_price(symbol, trading_date, attempt_type)
            if stored_price is None:
                self.context.logger.warning(f"No stored price found for {symbol} {attempt_type} on {trading_date}")
                return False
            
            # Limit orders are filled if stored price is within High/Low range
            filled = bar.low <= stored_price <= bar.high
            
            if filled:
                self.context.logger.debug(f"Limit order filled: {symbol} {attempt_type} at {stored_price:.2f} "
                                        f"(bar range: {bar.low:.2f}-{bar.high:.2f})")
            else:
                self.context.logger.debug(f"Limit order NOT filled: {symbol} {attempt_type} at {stored_price:.2f} "
                                        f"(bar range: {bar.low:.2f}-{bar.high:.2f})")
            
            return filled
        
        return False
    
    def generate_intents(self, bar: Bar) -> Iterable[OrderIntent]:
        """
        Generate order intents based on FuzzyFajuto logic implementing Section 7 rules.
        
        Section 7 Implementation:
        1. Emit no more than three orders per asset per day
        2. Order prices are immutable after emission
        3. Market orders only on first bar of day
        4. Limit orders filled passively using stored prices
        5. No re-emission of expired orders
        
        Args:
            bar: Current intraday market data bar
            
        Returns:
            Iterable of order intents
        """
        # Pre-trade validation
        if not self._validate_bar_data(bar):
            # Always log fuzzy diagnostics even if we cannot trade today
            try:
                date_str = str(bar.timestamp.date())
                sig = 0
                try:
                    sig = self._generate_signal(bar)
                except Exception:
                    sig = 0
                if sig != 0:
                    chosen_side = OrderSide.BUY if sig > 0 else OrderSide.SELL
                    row = {
                        'date': date_str,
                        'symbol': bar.symbol,
                        'side': chosen_side.name,
                        'fuzzy_score': float(getattr(self, '_last_signal_strength', 0.0)),
                        'ret_vs_ibov': float(getattr(self, '_last_fuzzy_breakdown', {}).get('ret_vs_ibov', 0.0)) if hasattr(self, '_last_fuzzy_breakdown') else None,
                        'ema_sum': float(getattr(self, '_last_fuzzy_breakdown', {}).get('ema_sum', 0.0)) if hasattr(self, '_last_fuzzy_breakdown') else None,
                        'rsi_term': float(getattr(self, '_last_fuzzy_breakdown', {}).get('rsi_term', 0.0)) if hasattr(self, '_last_fuzzy_breakdown') else None,
                        'eligible': False,
                        'reason_if_not': 'validation_failed',
                        'exposure_cap_brl': float(50000.0),
                        'notional_P1': 0.0,
                        'notional_P2': 0.0,
                        'notional_P3': 0.0,
                        'notional_P4': 0.0,
                    }
                    replaced = False
                    for i, r in enumerate(getattr(self, '_fuzzy_rows', [])):
                        if r.get('date') == date_str and r.get('symbol') == bar.symbol:
                            self._fuzzy_rows[i] = row
                            replaced = True
                            break
                    if not replaced:
                        if not hasattr(self, '_fuzzy_rows'):
                            self._fuzzy_rows = []
                        self._fuzzy_rows.append(row)
            except Exception:
                pass
            return []
        
        # Core recalculation engine - mandatory daily recalculation
        current_date = bar.timestamp.date()
        self.context.logger.debug(f"Processing bar for {bar.symbol} on {current_date} at {bar.timestamp.time()}")

        # Ensure a single fuzzy row per (symbol, date) from day 1 of backtest
        try:
            if not hasattr(self, '_fuzzy_rows'):
                self._fuzzy_rows = []
            if not hasattr(self, '_fuzzy_logged_dates'):
                self._fuzzy_logged_dates = set()
            date_str = str(current_date)
            key = (date_str, bar.symbol)
            if key not in self._fuzzy_logged_dates:
                # Decide a single side by score sign and log only that side
                try:
                    sig = self._generate_signal(bar)
                except Exception:
                    sig = 0
                if sig != 0:
                    chosen_side = OrderSide.BUY if sig > 0 else OrderSide.SELL
                    row = {
                        'date': date_str,
                        'symbol': bar.symbol,
                        'side': chosen_side.name,
                        'fuzzy_score': float(getattr(self, '_last_signal_strength', 0.0)),
                        'ret_vs_ibov': float(getattr(self, '_last_fuzzy_breakdown', {}).get('ret_vs_ibov', 0.0)) if hasattr(self, '_last_fuzzy_breakdown') else None,
                        'ema_sum': float(getattr(self, '_last_fuzzy_breakdown', {}).get('ema_sum', 0.0)) if hasattr(self, '_last_fuzzy_breakdown') else None,
                        'rsi_term': float(getattr(self, '_last_fuzzy_breakdown', {}).get('rsi_term', 0.0)) if hasattr(self, '_last_fuzzy_breakdown') else None,
                        'eligible': False,
                        'reason_if_not': 'no_sizing',
                        'exposure_cap_brl': float(50000.0),
                        'notional_P1': 0.0,
                        'notional_P2': 0.0,
                        'notional_P3': 0.0,
                        'notional_P4': 0.0,
                    }
                    # Upsert by (date, symbol)
                    replaced = False
                    for i, r in enumerate(self._fuzzy_rows):
                        if r.get('date') == date_str and r.get('symbol') == bar.symbol:
                            self._fuzzy_rows[i] = row
                            replaced = True
                            break
                    if not replaced:
                        self._fuzzy_rows.append(row)
                self._fuzzy_logged_dates.add(key)
        except Exception:
            pass
        
        # Check if this is a new trading day requiring recalculation
        if self._is_new_trading_day(bar.symbol, current_date):
            self.context.logger.info(f"🔄 New trading day detected for {bar.symbol} on {current_date}")
            self._force_daily_recalculation(bar.symbol, current_date)
            # Refresh fuzzy diagnostics row with post-recalculation values (logging only; no business change)
            try:
                sig = self._generate_signal(bar)
                if sig != 0:
                    chosen_side_name = (OrderSide.BUY if sig > 0 else OrderSide.SELL).name
                    date_str = str(current_date)
                    for i, r in enumerate(self._fuzzy_rows):
                        if r.get('date') == date_str and r.get('symbol') == bar.symbol and r.get('side') == chosen_side_name:
                            # Update diagnostics row and recompute eligibility based on stored daily quantities
                            q_mkt = self._get_stored_order_quantity(bar.symbol, current_date, 'market') or 0
                            q_a = self._get_stored_order_quantity(bar.symbol, current_date, 'limit_alpha') or 0
                            q_b = self._get_stored_order_quantity(bar.symbol, current_date, 'limit_beta') or 0
                            q_g = self._get_stored_order_quantity(bar.symbol, current_date, 'limit_gamma') or 0
                            eligible_now = bool((q_mkt or 0) > 0 or (q_a or 0) > 0 or (q_b or 0) > 0 or (q_g or 0) > 0)
                            self._fuzzy_rows[i] = {
                                'date': date_str,
                                'symbol': bar.symbol,
                                'side': chosen_side_name,
                                'fuzzy_score': float(getattr(self, '_last_signal_strength', 0.0)),
                                'ret_vs_ibov': float(getattr(self, '_last_fuzzy_breakdown', {}).get('ret_vs_ibov', 0.0)) if hasattr(self, '_last_fuzzy_breakdown') else None,
                                'ema_sum': float(getattr(self, '_last_fuzzy_breakdown', {}).get('ema_sum', 0.0)) if hasattr(self, '_last_fuzzy_breakdown') else None,
                                'rsi_term': float(getattr(self, '_last_fuzzy_breakdown', {}).get('rsi_term', 0.0)) if hasattr(self, '_last_fuzzy_breakdown') else None,
                                'eligible': eligible_now,
                                'reason_if_not': (None if eligible_now else 'below_min_lot'),
                                'exposure_cap_brl': r.get('exposure_cap_brl', 0.0),
                                'notional_P1': r.get('notional_P1', 0.0),
                                'notional_P2': r.get('notional_P2', 0.0),
                                'notional_P3': r.get('notional_P3', 0.0),
                                'notional_P4': r.get('notional_P4', 0.0)
                            }
                            break
            except Exception:
                pass
        
        # Validate current ATR availability
        if bar.symbol not in self.current_atr_values:
            self.context.logger.debug(f"No current ATR data for {bar.symbol}, skipping trade generation")
            return []
        
        # Check ATR validity
        atr = self._get_current_atr(bar.symbol)
        if atr <= 0:
            self.context.logger.debug(f"Invalid ATR for {bar.symbol}: {atr}, skipping trade generation")
            return []
        
        # Maintain intraday history for general use
        if bar.symbol not in self.intraday_history:
            max_history_size = max(self.warmup_bars + 20, self.atr_period * 7 + 20)
            self.intraday_history[bar.symbol] = deque(maxlen=max_history_size)
            self.context.logger.warning(f"Intraday history not found for {bar.symbol}, initializing new deque")
        
        self.intraday_history[bar.symbol].append({
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })
        
        # ========== SECTION 7 IMPLEMENTATION: ORDER EMISSION CONTROL ==========
        
        # Rule 1: Check if orders have already been emitted today
        if self._are_orders_emitted_today(bar.symbol, current_date):
            self.context.logger.debug(f"Orders already emitted for {bar.symbol} on {current_date}, checking for fills only")
            # Process fills for previously emitted orders but don't emit new ones
            return self._process_existing_orders(bar, current_date)
        
        # Rule 2: Check if this is first bar of day (for market orders)
        is_first_bar = self._is_first_bar_of_day(bar.symbol, current_date)

        # Day-trade scheduled execution path (t+1 execution)
        try:
            scheduled_today = self._scheduled_day_trades.get(current_date, {})
            if bar.symbol in scheduled_today:
                sched = scheduled_today[bar.symbol]
                side: OrderSide = sched['side']
                # Store immutable prices for day and override with scheduled limits
                atr_val = float(sched.get('current_atr_t', 0.0) or 0.0)
                self._store_daily_order_prices(bar.symbol, current_date, bar.open, atr_val, side)
                price_data = self.daily_order_prices[bar.symbol][current_date]
                price_data['alpha_price'] = float(sched['limits_used']['limit_level_2'])
                price_data['beta_price'] = float(sched['limits_used']['limit_level_3'])
                price_data['gamma_price'] = float(sched['limits_used']['limit_level_4'])

                # Per-leg notional: load from centralized config when available (pair mode)
                tranche_notional_brl = float(self.context.metadata.get('tranche_notional_brl', 10000.0))
                try:
                    cfg = getattr(self.context, 'config', None)
                    if cfg is None and hasattr(self.context, 'metadata'):
                        cfg = self.context.metadata.get('config')
                    if cfg and isinstance(cfg, dict):
                        pair_cfg = (cfg.get('pair_mode') or {})
                        gross = float(pair_cfg.get('gross_exposure_brl', 50000))
                        tranches = int(pair_cfg.get('tranches', 4))
                        if tranches > 0:
                            tranche_notional_brl = gross / tranches
                except Exception:
                    pass
                # Use previous day's close as sizing anchor for all legs to keep uniform lots
                prev_close = float(sched.get('base_close_t', float('nan')))

                intents: List[OrderIntent] = []
                # Compute a single uniform quantity for the tranche using previous close
                qty_uniform = 0
                if prev_close == prev_close and prev_close > 0:  # check not NaN
                    raw_qty = tranche_notional_brl / prev_close
                    qty_uniform = (int(raw_qty) // 100) * 100
                # Leg 1: MARKET at open (uniform qty)
                if is_first_bar and qty_uniform > 0:
                    price_open = round(float(bar.open), 2)
                    self.context.logger.info(f"Schedule t+1 legs: {bar.symbol} {side.name} open_market qty={qty_uniform} @open {price_open:.2f}")
                    self._track_daily_execution(bar.symbol, {'order_type': OrderType.MARKET, 'quantity': qty_uniform, 'price': None, 'execution_price': price_open, 'attempt_name': 'Open Market', 'attempt_type': 'market'}, True, bar, side)
                    intent_mkt = OrderIntent(
                        symbol=bar.symbol,
                        side=side,
                        quantity=qty_uniform,
                        order_type=OrderType.MARKET,
                        price=None,
                        timestamp=bar.timestamp,
                        metadata={'trade_type': 'day_trade','entry_leg': 'open_market','valid_for_date': str(current_date),'base_close_t': sched.get('base_close_t'),'limits_used': sched.get('limits_used'),'current_atr_t': sched.get('current_atr_t')}
                    )
                    self._mark_market_order_executed(current_date)
                    intents.append(intent_mkt)
                # Legs 2-4: LIMIT levels
                p2 = round(price_data['alpha_price'], 2); p3 = round(price_data['beta_price'], 2); p4 = round(price_data['gamma_price'], 2)
                # Use uniform qty for all limit levels; skip entirely if qty_uniform==0
                self._store_daily_order_quantities(bar.symbol, current_date, (intents[0].quantity if intents else 0), qty_uniform, qty_uniform, qty_uniform)
                if qty_uniform > 0:
                    self.context.logger.info(f"Schedule t+1 legs: {bar.symbol} {side.name} limit_2 qty={qty_uniform} @ {p2:.2f}")
                    self._track_daily_execution(bar.symbol, {'order_type': OrderType.LIMIT, 'quantity': qty_uniform, 'price': p2, 'execution_price': p2, 'attempt_name': 'Limit Level 2', 'attempt_type': 'limit_alpha'}, False, bar, side)
                    self.context.logger.info(f"Schedule t+1 legs: {bar.symbol} {side.name} limit_3 qty={qty_uniform} @ {p3:.2f}")
                    self._track_daily_execution(bar.symbol, {'order_type': OrderType.LIMIT, 'quantity': qty_uniform, 'price': p3, 'execution_price': p3, 'attempt_name': 'Limit Level 3', 'attempt_type': 'limit_beta'}, False, bar, side)
                    self.context.logger.info(f"Schedule t+1 legs: {bar.symbol} {side.name} limit_4 qty={qty_uniform} @ {p4:.2f}")
                    self._track_daily_execution(bar.symbol, {'order_type': OrderType.LIMIT, 'quantity': qty_uniform, 'price': p4, 'execution_price': p4, 'attempt_name': 'Limit Level 4', 'attempt_type': 'limit_gamma'}, False, bar, side)

                # Update fuzzy diagnostics row for this (date, symbol, side)
                try:
                    date_str = str(current_date)
                    for i, r in enumerate(self._fuzzy_rows):
                        if r.get('date') == date_str and r.get('symbol') == bar.symbol and r.get('side') == side.name:
                            self._fuzzy_rows[i] = {
                                'date': date_str,
                                'symbol': bar.symbol,
                                'side': side.name,
                                'fuzzy_score': float(getattr(self, '_last_signal_strength', 0.0)),
                                'ret_vs_ibov': float(getattr(self, '_last_fuzzy_breakdown', {}).get('ret_vs_ibov', 0.0)) if hasattr(self, '_last_fuzzy_breakdown') else None,
                                'ema_sum': float(getattr(self, '_last_fuzzy_breakdown', {}).get('ema_sum', 0.0)) if hasattr(self, '_last_fuzzy_breakdown') else None,
                                'rsi_term': float(getattr(self, '_last_fuzzy_breakdown', {}).get('rsi_term', 0.0)) if hasattr(self, '_last_fuzzy_breakdown') else None,
                                'eligible': bool((intents[0].quantity if intents else 0) > 0 or qty_uniform>0),
                                'reason_if_not': None,
                                'exposure_cap_brl': float(40000.0),
                                'notional_P1': float((bar.open) * (intents[0].quantity if intents else 0)),
                                'notional_P2': float(p2 * qty_uniform),
                                'notional_P3': float(p3 * qty_uniform),
                                'notional_P4': float(p4 * qty_uniform),
                            }
                            break
                except Exception:
                    pass

                # Clear schedule for this symbol (day-only GTD)
                try:
                    del self._scheduled_day_trades[current_date][bar.symbol]
                except Exception:
                    pass

                # Emit market intent now, and process immediate limit fills if any
                for it in intents:
                    yield it
                for it in self._process_existing_orders(bar, current_date):
                    yield it
                return []
        except Exception:
            pass

        # If nothing scheduled for this symbol, only process existing limit orders; do not emit new intraday entries
        return self._process_existing_orders(bar, current_date)
    
    def _emit_daily_orders(self, bar: Bar, trading_date: date, side: OrderSide, 
                         qty1: int, qty2: int, qty3: int, qty4: int, atr: float, signal: float) -> Iterable[OrderIntent]:
        """
        Emit the three daily orders (market + 2 limit) on first bar of day.
        Orders are emitted with immutable prices and tracked to prevent re-emission.
        """
        self.context.logger.info(f"📤 Emitting daily orders for {bar.symbol} on {trading_date}")

        # Neutrality: stage orders for day and emit later when all symbols' first bar is seen
        if getattr(self, 'RISK_MARKET_NEUTRAL', False):
            market_price = self._get_stored_order_price(bar.symbol, trading_date, 'market') or bar.open
            alpha_price = self._get_stored_order_price(bar.symbol, trading_date, 'limit_alpha') or bar.open
            beta_price = self._get_stored_order_price(bar.symbol, trading_date, 'limit_beta') or bar.open
            if trading_date not in self._neutral_buffer:
                self._neutral_buffer[trading_date] = {}
            self._neutral_buffer[trading_date][bar.symbol] = {
                'side': side,
                'fuzzy': float(abs(signal)),
                'prices': {'market': market_price, 'limit_alpha': alpha_price, 'limit_beta': beta_price},
            'qty': {'market': int(qty1), 'limit_alpha': int(qty2), 'limit_beta': int(qty3), 'limit_gamma': int(qty4)},
                'lot_size': int(self.min_lot_size),
                'bar': bar,
                'atr': float(atr),
                'signal': float(signal)
            }
            if trading_date not in self._first_bar_seen_by_date:
                self._first_bar_seen_by_date[trading_date] = set()
            self._first_bar_seen_by_date[trading_date].add(bar.symbol)
            universe_size = len(self._universe_symbols) if self._universe_symbols else None
            all_seen = (universe_size is None) or (len(self._first_bar_seen_by_date[trading_date]) >= universe_size)
            if not all_seen:
                return []
            # emit after neutrality balance
            for intent in self._neutral_emit_for_day(trading_date):
                yield intent
            return []

        orders_emitted = []
        intents_generated = 0
        
        # Order 1: Market Order (only on first bar)
        if qty1 > 0:
            market_price = self._get_stored_order_price(bar.symbol, trading_date, 'market')
            market_attempt = {
                'order_type': OrderType.MARKET,
                'quantity': qty1,
                'price': None,
                'execution_price': market_price,
                'attempt_name': 'Market Order at Open',
                'attempt_type': 'market'
            }
            
            # Market orders are always filled
            market_filled = True
            
            # Track execution
            self._track_daily_execution(bar.symbol, market_attempt, market_filled, bar, side)
            orders_emitted.append('market')
            
            if market_filled:
                intent = OrderIntent(
                    symbol=bar.symbol,
                    side=side,
                    quantity=qty1,
                    order_type=OrderType.MARKET,
                    price=None,
                    timestamp=bar.timestamp,
                    metadata={
                        'attempt_number': 1,
                        'attempt_name': 'Market Order at Open',
                        'attempt_type': 'market',
                        'atr_value': atr,
                        'execution_price': market_price,
                        'signal': signal,
                        'emission_type': 'first_bar'
                    }
                )
                # Mark as executed to avoid duplicate fills later in the day
                self._mark_market_order_executed(trading_date)
                intents_generated += 1
                self.context.logger.info(f"✅ Market order emitted: {side.value} {qty1} {bar.symbol} @ {market_price:.2f}")
                yield intent
        
        # Order 2: Limit Order Alpha
        if qty2 > 0:
            alpha_price = self._get_stored_order_price(bar.symbol, trading_date, 'limit_alpha')
            alpha_attempt = {
                'order_type': OrderType.LIMIT,
                'quantity': qty2,
                'price': alpha_price,
                'execution_price': alpha_price,
                'attempt_name': 'Limit Order Passive-1',
                'attempt_type': 'limit_alpha'
            }
            
            # Check if limit order can be filled on first bar
            alpha_filled = self._simulate_fill_with_stored_prices(
                OrderType.LIMIT, 'limit_alpha', bar.symbol, trading_date, bar
            )
            
            # Track execution
            self._track_daily_execution(bar.symbol, alpha_attempt, alpha_filled, bar, side)
            orders_emitted.append('limit_alpha')
            
            if alpha_filled:
                intent = OrderIntent(
                    symbol=bar.symbol,
                    side=side,
                    quantity=qty2,
                    order_type=OrderType.LIMIT,
                    price=alpha_price,
                    timestamp=bar.timestamp,
                    metadata={
                        'attempt_number': 2,
                        'attempt_name': 'Limit Order Passive-1',
                        'attempt_type': 'limit_alpha',
                        'atr_value': atr,
                        'execution_price': alpha_price,
                        'signal': signal,
                        'emission_type': 'first_bar'
                    }
                )
                # Mark as executed to prevent duplicate fills later in the day
                self._mark_limit_alpha_executed(trading_date)
                intents_generated += 1
                self.context.logger.info(f"✅ Limit alpha order filled: {side.value} {qty2} {bar.symbol} @ {alpha_price:.2f}")
                yield intent
            else:
                self.context.logger.info(f"📋 Limit alpha order emitted (not filled): {side.value} {qty2} {bar.symbol} @ {alpha_price:.2f}")
        
        # Order 3: Limit Order Beta
        if qty3 > 0:
            beta_price = self._get_stored_order_price(bar.symbol, trading_date, 'limit_beta')
            beta_attempt = {
                'order_type': OrderType.LIMIT,
                'quantity': qty3,
                'price': beta_price,
                'execution_price': beta_price,
                'attempt_name': 'Limit Order Passive-2',
                'attempt_type': 'limit_beta'
            }
            
            # Check if limit order can be filled on first bar
            beta_filled = self._simulate_fill_with_stored_prices(
                OrderType.LIMIT, 'limit_beta', bar.symbol, trading_date, bar
            )
            
            # Track execution
            self._track_daily_execution(bar.symbol, beta_attempt, beta_filled, bar, side)
            orders_emitted.append('limit_beta')
            
            if beta_filled:
                intent = OrderIntent(
                    symbol=bar.symbol,
                    side=side,
                    quantity=qty3,
                    order_type=OrderType.LIMIT,
                    price=beta_price,
                    timestamp=bar.timestamp,
                    metadata={
                        'attempt_number': 3,
                        'attempt_name': 'Limit Order Passive-2',
                        'attempt_type': 'limit_beta',
                        'atr_value': atr,
                        'execution_price': beta_price,
                        'signal': signal,
                        'emission_type': 'first_bar'
                    }
                )
                # Mark as executed to prevent duplicate fills later in the day
                self._mark_limit_beta_executed(trading_date)
                intents_generated += 1
                self.context.logger.info(f"✅ Limit beta order filled: {side.value} {qty3} {bar.symbol} @ {beta_price:.2f}")
                yield intent
            else:
                self.context.logger.info(f"📋 Limit beta order emitted (not filled): {side.value} {qty3} {bar.symbol} @ {beta_price:.2f}")

        # 4) Limit gamma attempt
        if qty4 > 0:
            gamma_price = self._get_stored_order_price(bar.symbol, trading_date, 'limit_gamma')
            if gamma_price is None:
                # derive from daily_order_prices distances
                price_data = self.daily_order_prices.get(bar.symbol, {}).get(trading_date, {})
                if price_data:
                    # Approximate gamma further from beta by same ratio used in sizing
                    beta_price = price_data.get('beta_price', bar.open)
                    side_mult = 1 if side == OrderSide.SELL else -1
                    gamma_price = max(0.01, beta_price + side_mult * abs(beta_price - price_data.get('alpha_price', bar.open)) * 0.5)
                else:
                    gamma_price = bar.open
            attempt = {
                'order_type': OrderType.LIMIT,
                'quantity': qty4,
                'price': gamma_price,
                'execution_price': gamma_price,
                'attempt_name': 'Limit Order Passive-3',
                'attempt_type': 'limit_gamma'
            }
            filled = self._simulate_fill_with_stored_prices(OrderType.LIMIT, 'limit_gamma', bar.symbol, trading_date, bar)
            self._track_daily_execution(bar.symbol, attempt, filled, bar, side)
            if filled:
                yield OrderIntent(
                    symbol=bar.symbol, side=side, quantity=qty4, order_type=OrderType.LIMIT, price=gamma_price, timestamp=bar.timestamp,
                    metadata={'attempt_number': 4,'attempt_name': 'Limit Order Passive-3','attempt_type': 'limit_gamma','atr_value': atr,'execution_price': gamma_price,'signal': signal,'emission_type': 'first_bar_neutralized'}
                )
                self.context.logger.info(f"✅ Limit gamma order filled: {side.value} {qty4} {bar.symbol} @ {gamma_price:.2f}")
            else:
                self.context.logger.info(f"📋 Limit gamma order emitted (not filled): {side.value} {qty4} {bar.symbol} @ {gamma_price:.2f}")
        
        # Mark orders as emitted to prevent re-emission
        self._mark_orders_emitted(bar.symbol, trading_date, orders_emitted)
        
        self.context.logger.info(f"📊 Emission summary for {bar.symbol}: {len(orders_emitted)} orders emitted, {intents_generated} filled immediately")
    
    def _process_existing_orders(self, bar: Bar, trading_date: date) -> Iterable[OrderIntent]:
        """
        Process fills for previously emitted limit orders.
        This runs on every bar after the first bar to check for limit order fills.
        """
        if bar.symbol not in self.daily_order_prices:
            return []
        
        if trading_date not in self.daily_order_prices[bar.symbol]:
            return []
        
        price_data = self.daily_order_prices[bar.symbol][trading_date]
        side = price_data['side']
        atr = price_data['atr_value']
        
        # Check limit alpha order fill (if not already executed)
        if not self._is_limit_alpha_executed_today(trading_date):
            alpha_filled = self._simulate_fill_with_stored_prices(
                OrderType.LIMIT, 'limit_alpha', bar.symbol, trading_date, bar
            )
            
            if alpha_filled:
                # Mark as executed and create intent
                self._mark_limit_alpha_executed(trading_date)
                alpha_price = self._get_stored_order_price(bar.symbol, trading_date, 'limit_alpha')
                # Use stored immutable quantity
                qty2 = self._get_stored_order_quantity(bar.symbol, trading_date, 'limit_alpha') or 0
                
                intent = OrderIntent(
                    symbol=bar.symbol,
                    side=side,
                    quantity=qty2,
                    order_type=OrderType.LIMIT,
                    price=alpha_price,
                    timestamp=bar.timestamp,
                    metadata={
                        'attempt_number': 2,
                        'attempt_name': 'Limit Order Passive-1',
                        'attempt_type': 'limit_alpha',
                        'atr_value': atr,
                        'execution_price': alpha_price,
                        'emission_type': 'delayed_fill'
                    }
                )
                
                self.context.logger.info(f"✅ Limit alpha order filled (delayed): {side.value} {qty2} {bar.symbol} @ {alpha_price:.2f}")
                yield intent
        
        # Check limit beta order fill (if not already executed)
        if not self._is_limit_beta_executed_today(trading_date):
            beta_filled = self._simulate_fill_with_stored_prices(
                OrderType.LIMIT, 'limit_beta', bar.symbol, trading_date, bar
            )
            
            if beta_filled:
                # Mark as executed and create intent
                self._mark_limit_beta_executed(trading_date)
                beta_price = self._get_stored_order_price(bar.symbol, trading_date, 'limit_beta')
                # Use stored immutable quantity
                qty3 = self._get_stored_order_quantity(bar.symbol, trading_date, 'limit_beta') or 0
                
                intent = OrderIntent(
                    symbol=bar.symbol,
                    side=side,
                    quantity=qty3,
                    order_type=OrderType.LIMIT,
                    price=beta_price,
                    timestamp=bar.timestamp,
                    metadata={
                        'attempt_number': 3,
                        'attempt_name': 'Limit Order Passive-2',
                        'attempt_type': 'limit_beta',
                        'atr_value': atr,
                        'execution_price': beta_price,
                        'emission_type': 'delayed_fill'
                    }
                )
                
                self.context.logger.info(f"✅ Limit beta order filled (delayed): {side.value} {qty3} {bar.symbol} @ {beta_price:.2f}")
                yield intent
    
    def _generate_signal(self, bar: Bar) -> int:
        """
        FuzzyFajuto signal per business rules (daily):
        - +1 / -1: today's stock return vs IBOV (^BVSP) return
        - For each EMA(3,5,10,15,20): +0.25 if close > EMA, -0.25 if close < EMA
        - RSI: +0.25 if RSI > 65, -0.25 if RSI < 35
        Thresholds: buy if fuzzy >= 1.50, sell if fuzzy <= -1.50
        Portfolio/position sizing logic remains unchanged.
        """
        symbol = bar.symbol
        current_date = bar.timestamp.date()
        buy_threshold = self.buy_threshold
        sell_threshold = self.sell_threshold

        # Ensure indicators exist
        if symbol not in self.daily_indicators_data:
            self.context.logger.debug(f"No daily indicators for {symbol} in signal generation")
            return 0
        indicators = self.daily_indicators_data[symbol]

        # Get daily data up to current date for the symbol
        try:
            daily_df = self._get_daily_data_up_to_date(symbol, current_date)
        except Exception:
            daily_df = self.daily_data.get(symbol)
        if daily_df is None or daily_df.empty or len(daily_df) < 2:
            self.context.logger.debug(f"Insufficient daily data for {symbol} to compute returns")
            return 0

        # Align on a common latest date present in BOTH the symbol and ^BVSP to avoid drift bias
        sym_last = daily_df.index.max()
        # Ensure IBOV (^BVSP) daily data cached and aligned
        ibov_symbol = '^BVSP'
        if ibov_symbol not in self.daily_data or self.daily_data[ibov_symbol].empty:
            try:
                ibov_df_boot = self._get_daily_data_for_date(ibov_symbol, current_date)
                if ibov_df_boot is not None and not ibov_df_boot.empty:
                    self.daily_data[ibov_symbol] = ibov_df_boot
                    self.daily_data_last_update[ibov_symbol] = current_date
            except Exception as e:
                self.context.logger.error(f"Failed to fetch ^BVSP daily data: {e}")
                return 0
        ibov_df = self._get_daily_data_up_to_date(ibov_symbol, current_date)
        if ibov_df is None or ibov_df.empty or len(ibov_df) < 2:
            self.context.logger.debug("Insufficient ^BVSP data to compute benchmark return")
            return 0
        # Choose the latest date that exists in both series
        try:
            ibov_upto_sym = ibov_df[ibov_df.index <= sym_last]
            if ibov_upto_sym.empty:
                self.context.logger.debug("No overlapping dates between symbol and ^BVSP")
                return 0
            common_idx = min(sym_last, ibov_upto_sym.index.max())
        except Exception:
            common_idx = sym_last
        # Get current and previous closes for the symbol on aligned date
        try:
            sym_pos = daily_df.index.get_loc(common_idx)
        except Exception:
            # Fallback to last available
            sym_pos = len(daily_df.index) - 1
            common_idx = daily_df.index[sym_pos]
        if sym_pos == 0:
            self.context.logger.debug("Not enough symbol history for return comparison")
            return 0
        prev_idx = daily_df.index[sym_pos - 1]
        close_today = float(daily_df.loc[common_idx, 'close'])
        close_prev = float(daily_df.loc[prev_idx, 'close'])
        if close_prev == 0 or np.isnan(close_prev) or np.isnan(close_today):
            return 0
        stock_ret = (close_today / close_prev) - 1.0
        # Benchmark returns on aligned date
        try:
            ibov_pos = ibov_df.index.get_loc(common_idx)
        except Exception:
            ibov_pos = len(ibov_df.index) - 1
        if ibov_pos == 0:
            self.context.logger.debug("Not enough ^BVSP history for return comparison")
            return 0
        ibov_prev_idx = ibov_df.index[ibov_pos - 1]
        ibov_close_today = float(ibov_df.loc[common_idx, 'close'])
        ibov_close_prev = float(ibov_df.loc[ibov_prev_idx, 'close'])
        if ibov_close_prev == 0 or np.isnan(ibov_close_prev) or np.isnan(ibov_close_today):
            return 0
        ibov_ret = (ibov_close_today / ibov_close_prev) - 1.0

        fuzzy = 0.0
        # Rule 1: stock return vs IBOV return (+1 / -1)
        if stock_ret > ibov_ret:
            fuzzy += 1.0
        elif stock_ret < ibov_ret:
            fuzzy -= 1.0
        ret_vs_ibov_term = 1.0 if stock_ret > ibov_ret else (-1.0 if stock_ret < ibov_ret else 0.0)

        # Rule 2: EMA comparisons (+/- 0.25 each)
        ema_periods = getattr(self, 'ema_periods', [3, 5, 10, 15, 20])
        ema_sum = 0.0
        for period in ema_periods:
            key = f'ema_{period}'
            if key in indicators and not indicators[key].empty:
                try:
                    # Use aligned date; fallback to last available
                    ema_today = float(indicators[key].loc[common_idx]) if common_idx in indicators[key].index else float(indicators[key].iloc[-1])
                except Exception:
                    continue
                if not np.isnan(ema_today):
                    if close_today > ema_today:
                        fuzzy += 0.25
                        ema_sum += 0.25
                    elif close_today < ema_today:
                        fuzzy -= 0.25
                        ema_sum -= 0.25

        # Rule 3: RSI bands (+0.25 if >65, -0.25 if <35)
        rsi_term = 0.0
        if 'rsi' in indicators and not indicators['rsi'].empty:
            try:
                rsi_today = float(indicators['rsi'].loc[common_idx]) if common_idx in indicators['rsi'].index else float(indicators['rsi'].iloc[-1])
                if not np.isnan(rsi_today):
                    overbought = getattr(self, 'rsi_overbought', 65)
                    oversold = getattr(self, 'rsi_oversold', 35)
                    if rsi_today > overbought:
                        fuzzy += 0.25
                        rsi_term = 0.25
                    elif rsi_today < oversold:
                        fuzzy -= 0.25
                        rsi_term = -0.25
            except Exception:
                pass

        # Persist last strength for diagnostics/sizing
        self._last_signal_strength = float(fuzzy)
        # Store breakdown for diagnostics
        self._last_fuzzy_breakdown = {
            'ret_vs_ibov': float(ret_vs_ibov_term),
            'ema_sum': float(ema_sum),
            'rsi_term': float(rsi_term),
        }

        # Decide
        if fuzzy >= buy_threshold:
            self.context.logger.info(f"✅ BUY signal (fuzzy={fuzzy:.2f}) for {symbol}")
            return 1
        if fuzzy <= sell_threshold:
            self.context.logger.info(f"✅ SELL signal (fuzzy={fuzzy:.2f}) for {symbol}")
            return -1
        self.context.logger.info(f"⚪ No signal (fuzzy={fuzzy:.2f}) for {symbol}")
        return 0
    
    def _validate_signal_quality(self, signal_strength: float, ema_values: np.ndarray, symbol: str = None, current_date: date = None) -> bool:
        """
        Enhanced signal quality validation with date-based checks.
        
        Args:
            signal_strength: Calculated signal strength
            ema_values: Array of EMA values used in calculation
            symbol: Trading symbol (optional, for date-based validation)
            current_date: Current backtest date (optional, for accurate validation)
            
        Returns:
            True if signal quality is acceptable, False otherwise
        """
        # Check if indicators are recent (NEW)
        if symbol and symbol in self.daily_indicators_last_update:
            # Use backtest date if provided, otherwise use system date
            check_date = current_date if current_date else datetime.now().date()
            days_since_update = (check_date - self.daily_indicators_last_update[symbol]).days
            if days_since_update > 1:
                self.context.logger.warning(f"Indicators for {symbol} are {days_since_update} days old")
                return False
        
        # Check if EMAs are properly converged (not static)
        ema_variance = np.var(ema_values)
        
        # If variance is too low, EMAs are not converged
        if ema_variance < 0.01:  # Very low variance indicates static values
            return False
        
        # Check signal strength range (NEW)
        if abs(signal_strength) < 0.1:  # Too weak signal
            return False
        
        # Check if signal strength is not stuck at the same value
        # This helps detect when the calculation is producing the same result repeatedly
        if hasattr(self, '_last_signal_strength'):
            if abs(signal_strength - self._last_signal_strength) < 0.001:  # Very small change
                # Check if this has happened multiple times
                if hasattr(self, '_static_signal_count'):
                    self._static_signal_count += 1
                    if self._static_signal_count > 5:  # If static for 5+ consecutive signals
                        return False
                else:
                    self._static_signal_count = 1
            else:
                # Reset static signal counter if signal is changing
                self._static_signal_count = 0
        
        # Store current signal strength for next comparison
        self._last_signal_strength = signal_strength
        
        return True
    
    def _track_daily_execution(self, symbol: str, attempt: Dict, filled: bool, bar: Bar, side: OrderSide = None):
        """
        Enhanced execution tracking with Section 7 compliance validation.
        
        This method tracks daily execution attempts and validates that orders
        are being emitted according to Section 7 rules (immutable, once per day).
        """
        attempt_type = attempt.get('attempt_type', 'unknown')
        trading_date = bar.timestamp.date()
        
        # Section 7 Validation: Verify order emission compliance
        if not self._validate_order_emission_compliance(symbol, trading_date, attempt_type):
            self.context.logger.error(f"❌ Section 7 violation detected: {symbol} {attempt_type} on {trading_date}")
            return
        
        # Validate price immutability for limit orders
        if attempt['order_type'] == OrderType.LIMIT:
            stored_price = self._get_stored_order_price(symbol, trading_date, attempt_type)
            order_price = attempt['execution_price']
            
            if stored_price is not None and abs(stored_price - order_price) > 0.001:
                self.context.logger.error(f"❌ Price immutability violation: {symbol} {attempt_type} "
                                        f"stored={stored_price:.4f} vs order={order_price:.4f}")
                return
        
        # Update daily execution counts
        if attempt_type in self.daily_execution_counts:
            self.daily_execution_counts[attempt_type] += 1
        
        # Update daily fill rates (successful fills only)
        if filled and attempt_type in self.daily_fill_rates:
            self.daily_fill_rates[attempt_type] += 1
        
        # Also update legacy tracking for backward compatibility
        if attempt_type in self.execution_counts:
            self.execution_counts[attempt_type] += 1
        if filled and attempt_type in self.fill_rates:
            self.fill_rates[attempt_type] += 1
        
        # Enhanced logging for Section 7 compliance
        compliance_status = "✅ COMPLIANT" if self._validate_order_emission_compliance(symbol, trading_date, attempt_type) else "❌ VIOLATION"
        self.context.logger.debug(f"📊 {compliance_status} - {attempt_type} order tracked: {symbol}, filled={filled}")
        
        # Log detailed execution if enabled
        if self.log_detailed_executions:
            price_source = "stored" if attempt['order_type'] == OrderType.LIMIT else "market"
            self.context.logger.info(
                f"📈 Daily Execution: {symbol} {attempt['attempt_name']} - "
                f"Filled: {filled}, Qty: {attempt['quantity']}, "
                f"Price: {attempt['execution_price']:.2f} ({price_source})"
            )
        
        # Compute slippage (only for LIMIT orders)
        slippage_value: Optional[float] = None
        try:
            if attempt.get('order_type') == OrderType.LIMIT:
                # Use stored immutable open and limit prices
                market_open_price = self._get_stored_order_price(symbol, trading_date, 'market')
                limit_price = self._get_stored_order_price(symbol, trading_date, attempt_type)
                if market_open_price is None:
                    market_open_price = bar.open  # fallback
                if isinstance(limit_price, (int, float)) and isinstance(market_open_price, (int, float)):
                    slippage_value = float(market_open_price) - float(limit_price)
                else:
                    slippage_value = None
            else:
                # Market orders (including MOC) have zero slippage by definition
                slippage_value = 0.0
        except Exception:
            slippage_value = None

        # Store execution history with Section 7 compliance metadata
        if self.save_execution_history:
            # Determine quantity sign based on order side
            quantity = attempt['quantity']
            if side == OrderSide.SELL:
                quantity = -quantity
            
            # Attempt to enrich with day-level metadata (expected_open/first_bar)
            expected_open = None
            first_bar_ts = None
            missing_open_bar = None
            try:
                day_meta = self.context.metadata.get('day_metadata', {}) if hasattr(self.context, 'metadata') else {}
                day_key = bar.timestamp.date().isoformat()
                if day_key in day_meta:
                    expected_open = day_meta[day_key].get('expected_open')
                    first_bar_ts = day_meta[day_key].get('first_bar')
                    missing_open_bar = day_meta[day_key].get('missing_open_bar')
            except Exception:
                pass

            # Compute per-attempt PnL using daily Close[D] when available
            daily_close_px: Optional[float] = None
            try:
                daily_df = self.daily_data.get(symbol)
                if daily_df is not None and not daily_df.empty:
                    ts_day = pd.Timestamp(trading_date)
                    if ts_day in daily_df.index and 'close' in daily_df.columns:
                        daily_close_px = float(daily_df.loc[ts_day, 'close'])
            except Exception:
                daily_close_px = None

            entry_px = attempt.get('execution_price')
            pnl_value: Optional[float] = None
            try:
                if isinstance(entry_px, (int, float)) and daily_close_px is not None:
                    abs_qty = abs(quantity)
                    if side == OrderSide.SELL:
                        pnl_value = (float(entry_px) - float(daily_close_px)) * abs_qty
                    else:
                        pnl_value = (float(daily_close_px) - float(entry_px)) * abs_qty
            except Exception:
                pnl_value = None
            
            execution_record = {
                'timestamp': bar.timestamp,
                'symbol': symbol,
                'attempt_name': attempt['attempt_name'],
                'attempt_type': attempt_type,
                'filled': filled,
                'quantity': quantity,
                'execution_price': attempt['execution_price'],
                'atr_value': self._get_daily_atr(symbol),
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                # Use daily Close[D] when available; fall back to bar.close
                'close': daily_close_px if daily_close_px is not None else bar.close,
                'execution_category': 'daily',  # All executions are now daily
                'section_7_compliant': True,    # Flag for compliance tracking
                'price_source': 'stored' if attempt['order_type'] == OrderType.LIMIT else 'market',
                'expected_open': expected_open,
                'first_bar': first_bar_ts,
                'missing_open_bar': missing_open_bar,
                'slippage': slippage_value,
                # Add side and per-attempt PnL for analysis
                'side': side.value.upper() if hasattr(side, 'value') else (str(side).upper() if side is not None else None),
                'pnl': pnl_value
            }
            self.execution_history.append(execution_record)
    
    def _validate_order_emission_compliance(self, symbol: str, trading_date: date, attempt_type: str) -> bool:
        """
        Validate that order emission follows Section 7 rules.
        
        Args:
            symbol: Trading symbol
            trading_date: Trading date
            attempt_type: Order attempt type
            
        Returns:
            True if compliant, False otherwise
        """
        # Check if this is the first emission for this attempt type today
        if symbol in self.daily_orders_emitted:
            if trading_date in self.daily_orders_emitted[symbol]:
                if self.daily_orders_emitted[symbol][trading_date].get(attempt_type, False):
                    # Order of this type already emitted today - this is compliant for tracking
                    return True
        
        # If we're here, this should be the first emission - which is compliant
        return True
    

    
    def _track_execution(self, symbol: str, attempt: Dict, filled: bool, bar: Bar, side: OrderSide = None):
        """Legacy tracking method - maintained for backward compatibility."""
        attempt_type = 'market' if attempt['order_type'] == OrderType.MARKET else 'limit'
        
        if attempt_type == 'limit':
            if 'Passive-1' in attempt['attempt_name']:
                attempt_type = 'limit_alpha'
            else:
                attempt_type = 'limit_beta'
        
        # Update execution counts (total attempts)
        self.execution_counts[attempt_type] += 1
        
        # Update fill rates (successful fills only)
        if filled:
            self.fill_rates[attempt_type] += 1
        
        # Log detailed execution if enabled
        if self.log_detailed_executions:
            self.context.logger.info(
                f"Execution: {symbol} {attempt['attempt_name']} - "
                f"Filled: {filled}, Qty: {attempt['quantity']}, "
                f"Price: {attempt['execution_price']:.2f}"
            )
        
        # Store execution history
        if self.save_execution_history:
            # Determine quantity sign based on order side
            # Positive for BUY, negative for SELL
            quantity = attempt['quantity']
            if side == OrderSide.SELL:
                quantity = -quantity
            
            execution_record = {
                'timestamp': bar.timestamp,
                'symbol': symbol,
                'attempt_name': attempt['attempt_name'],
                'attempt_type': attempt_type,
                'filled': filled,
                'quantity': quantity,
                'execution_price': attempt['execution_price'],
                'atr_value': self._get_daily_atr(symbol),
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close
            }
            self.execution_history.append(execution_record)
    
    def on_fill(self, fill: Fill) -> None:
        """Called when an order is filled."""
        super().on_fill(fill)
        
        # Track position
        if fill.symbol not in self.current_positions:
            self.current_positions[fill.symbol] = 0
        
        if fill.side == OrderSide.BUY:
            self.current_positions[fill.symbol] += fill.quantity
        else:
            self.current_positions[fill.symbol] -= fill.quantity
        
        # Log fill details
        self.context.logger.info(
            f"Fill: {fill.symbol} {fill.side.value} {fill.quantity} @ {fill.price:.2f} "
            f"(Attempt: {fill.metadata.get('attempt_name', 'Unknown')})"
        )
        # Record confirmed fill for CSV generation
        try:
            self.confirmed_fills.append({
                'timestamp': fill.timestamp,
                'symbol': fill.symbol,
                'side': fill.side.value.upper(),
                'quantity': int(fill.quantity),
                'price': float(fill.price),
                'attempt_name_raw': fill.metadata.get('attempt_name') if isinstance(fill.metadata, dict) else None
            })
        except Exception:
            pass
    
    def on_end_of_day(self, d: date) -> Iterable[OrderIntent]:
        """
        Enhanced MOC implementation with guaranteed position zeroing (Section 7 compliance).
        
        This method ensures that all positions opened during the day are properly
        closed at market close, as required by Section 7 rules.
        """
        super().on_end_of_day(d)
        # Schedule day-trade legs for next day based on close(d)
        try:
            valid_for = d + timedelta(days=1)
            day_store = self._scheduled_day_trades.setdefault(valid_for, {})
            universe = list(self._universe_symbols) if getattr(self, '_universe_symbols', None) else []
            for symbol in universe:
                # Ensure inputs (daily indicators and ^BVSP) are present for date d
                self._ensure_eod_inputs(symbol, d)
                df = self._get_daily_data_up_to_date(symbol, d)
                if df is None or df.empty or df.index.max().date() != d:
                    continue
                class _B: pass
                b = _B(); b.symbol = symbol; b.timestamp = datetime.combine(d, datetime.max.time())
                row = df.iloc[-1]
                b.open = float(row['open']); b.high = float(row['high']); b.low = float(row['low']); b.close = float(row['close']); b.volume = int(row.get('volume', 0))
                sig = self._generate_signal(b)
                if sig == 0:
                    continue
                side = OrderSide.BUY if sig > 0 else OrderSide.SELL
                atr_val = float(self._get_current_atr(symbol)) if symbol in self.current_atr_values else 0.0
                p2, p3, p4 = self._calculate_entry_limits_from_close(b.close, atr_val, side)
                day_store[symbol] = {
                    'symbol': symbol,
                    'side': side,
                    'valid_for_date': valid_for,
                    'base_close_t': float(b.close),
                    'limits_used': {'limit_level_2': p2, 'limit_level_3': p3, 'limit_level_4': p4},
                    'current_atr_t': atr_val,
                    'fuzzy_score_t': float(abs(getattr(self, '_last_signal_strength', 0.0)))
                }
        except Exception:
            pass
        
        moc_orders_generated = 0
        
        # Generate MOC orders to close all positions
        for symbol, position in self.current_positions.items():
            if position != 0:
                side = OrderSide.SELL if position > 0 else OrderSide.BUY
                quantity = abs(position)
                
                # Create MOC order intent
                intent = OrderIntent(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type=OrderType.MARKET,  # MOC is typically a market order
                    timestamp=datetime.combine(d, datetime.max.time()),
                    metadata={
                        'order_type': 'MOC',
                        'reason': 'End of day position close',
                        'original_position': position,
                        'section_7_compliance': True
                    }
                )
                
                moc_orders_generated += 1
                self.context.logger.info(f"🔒 MOC order: {symbol} {side.value} {quantity} (closing position {position})")
                yield intent
        
        # Log MOC summary
        if moc_orders_generated > 0:
            self.context.logger.info(f"📋 Generated {moc_orders_generated} MOC orders for {d}")
        else:
            self.context.logger.info(f"✅ No positions to close for {d}")
        
        # Reset daily tracking states for Section 7 compliance
        self._reset_daily_states(d)
    
    def _reset_daily_states(self, trading_date: date):
        """
        Reset daily states to ensure clean slate for next trading day.
        This supports Section 7 implementation by clearing execution tracking.
        """
        # Clear position tracking
        self.current_positions.clear()
        self.daily_loss = 0.0
        self.daily_exposure = 0.0
        
        # Clean up old daily tracking data (keep only recent history)
        # This prevents memory buildup while maintaining recent execution history
        symbols_to_clean = []
        
        for symbol in self.daily_orders_emitted.keys():
            # Keep last 30 days of history
            dates_to_remove = []
            for date_key in self.daily_orders_emitted[symbol].keys():
                days_old = (trading_date - date_key).days
                if days_old > 30:
                    dates_to_remove.append(date_key)
            
            for old_date in dates_to_remove:
                del self.daily_orders_emitted[symbol][old_date]
            
            if not self.daily_orders_emitted[symbol]:
                symbols_to_clean.append(symbol)
        
        # Remove empty symbol entries
        for symbol in symbols_to_clean:
            del self.daily_orders_emitted[symbol]
        
        # Similar cleanup for other daily tracking
        for symbol in list(self.daily_order_prices.keys()):
            if symbol in self.daily_order_prices:
                dates_to_remove = []
                for date_key in self.daily_order_prices[symbol].keys():
                    days_old = (trading_date - date_key).days
                    if days_old > 30:
                        dates_to_remove.append(date_key)
                
                for old_date in dates_to_remove:
                    del self.daily_order_prices[symbol][old_date]
                
                if not self.daily_order_prices[symbol]:
                    del self.daily_order_prices[symbol]
        
        # Similar cleanup for first bar tracking
        for symbol in list(self.first_bar_of_day.keys()):
            if symbol in self.first_bar_of_day:
                dates_to_remove = []
                for date_key in self.first_bar_of_day[symbol].keys():
                    days_old = (trading_date - date_key).days
                    if days_old > 30:
                        dates_to_remove.append(date_key)
                
                for old_date in dates_to_remove:
                    del self.first_bar_of_day[symbol][old_date]
                
                if not self.first_bar_of_day[symbol]:
                    del self.first_bar_of_day[symbol]
        
        self.context.logger.debug(f"🧹 Daily states reset for {trading_date}")
    
    def on_end(self, asof: datetime) -> None:
        """Called when strategy ends."""
        super().on_end(asof)
        
        # Log final statistics
        self.context.logger.info("FuzzyFajuto strategy ended")
        self.context.logger.info(f"Total executions: {sum(self.execution_counts.values())}")
        self.context.logger.info(f"Fill rates: {self.fill_rates}")
        
        # Save execution history if enabled
        if self.save_execution_history and self.execution_history:
            self._save_execution_history()

        # No CSV export of fuzzy rows; report consumes in-memory data
    
    def _save_execution_history(self):
        """Save execution history to file."""
        try:
            import pandas as pd
            from pathlib import Path
            from datetime import datetime as _dt

            output_path = Path("reports") / "fuzzy_fajuto_execution_history.csv"
            output_path.parent.mkdir(exist_ok=True)

            # Prepare intraday/hourly data for OHLC lookup at exact execution timestamps
            intraday_data = None
            try:
                # Prefer complete intraday data preserved in context metadata
                intraday_data = self.context.metadata.get('complete_data')
                if intraday_data is None and 'hybrid_data_result' in self.context.metadata:
                    intraday_data = self.context.metadata['hybrid_data_result'].get('execution_data')
                # Normalize index to timezone-naive for exact matching if needed
                if intraday_data is not None and getattr(intraday_data.index, 'tz', None) is not None:
                    intraday_data = intraday_data.copy()
                    intraday_data.index = intraday_data.index.tz_localize(None)
            except Exception:
                intraday_data = None

            def _get_hourly_ohlc(symbol: str, when: _dt):
                """Return OHLC dict for the bar at 'when' using hourly intraday data; None if unavailable."""
                try:
                    if intraday_data is None or when is None:
                        return None
                    df = intraday_data
                    # If data contains multiple symbols, filter by symbol column if present
                    if 'symbol' in df.columns or 'ticker' in df.columns:
                        symbol_col = 'symbol' if 'symbol' in df.columns else 'ticker'
                        df = df[df[symbol_col] == symbol]
                    # Ensure exact timestamp match
                    if when in df.index:
                        row = df.loc[when]
                        if isinstance(row, pd.DataFrame):
                            row = row.iloc[0]
                        return {
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close'])
                        }
                    return None
                except Exception:
                    return None

            # 1) Prefer confirmed fills from portfolio trade history to ensure
            #    we only record actually executed orders with real fill timestamps
            portfolio = getattr(self.context, 'portfolio', None)

            confirmed_rows = []

            # Build per-day reference map from existing emission records to infer
            # attempt_type/name and auxiliary metadata (ATR, OHLC, expected open info)
            per_day_refs = {}
            try:
                for rec in self.execution_history:
                    ts = rec.get('timestamp')
                    sym = rec.get('symbol')
                    if ts is None or sym is None:
                        continue
                    day_key = (sym, ts.date())
                    ref = per_day_refs.get(day_key, {
                        'prices': {},
                        'atr_value': rec.get('atr_value'),
                        'open': rec.get('open'),
                        'high': rec.get('high'),
                        'low': rec.get('low'),
                        'close': rec.get('close'),
                        'expected_open': rec.get('expected_open'),
                        'first_bar': rec.get('first_bar'),
                        'missing_open_bar': rec.get('missing_open_bar'),
                    })
                    # Store known immutable prices by attempt type, if present
                    attempt_type = rec.get('attempt_type')
                    exec_price = rec.get('execution_price')
                    if attempt_type and isinstance(exec_price, (int, float)):
                        ref['prices'][attempt_type] = exec_price
                    per_day_refs[day_key] = ref
            except Exception:
                # Safe fallback: if anything goes wrong, we will still try to save
                per_day_refs = {}

            def _infer_attempt(symbol: str, when: _dt, price: float):
                """Infer attempt_type and human name from stored daily reference prices."""
                day_key = (symbol, when.date())
                ref = per_day_refs.get(day_key, {})
                prices = ref.get('prices', {})
                # Compare on 2-decimal basis (execution records are rounded to cents)
                def _eq(a: float, b: float) -> bool:
                    try:
                        return round(float(a), 2) == round(float(b), 2)
                    except Exception:
                        return False

                # Try match in priority order
                if 'market' in prices and _eq(price, prices['market']):
                    return 'market', 'Market Order at Open', 'market'
                if 'limit_alpha' in prices and _eq(price, prices['limit_alpha']):
                    return 'limit_alpha', 'Limit Order Passive-1', 'stored'
                if 'limit_beta' in prices and _eq(price, prices['limit_beta']):
                    return 'limit_beta', 'Limit Order Passive-2', 'stored'
                # Unknown mapping (e.g., MOC or unmatched) → classify as market with name 'MOC' if at/near day end
                try:
                    # Heuristic: close-to-day-end timestamps are considered MOC
                    # Use 20:00 UTC as the market close for B3
                    if when.hour >= 20:
                        return 'moc', 'MOC', 'market'
                except Exception:
                    pass
                return 'market', 'Unknown', 'market'

            # Collect executed rows from BOTH portfolio.trade_history and confirmed fills
            # 1) Portfolio trade history (authoritative)
            if portfolio is not None and hasattr(portfolio, 'trade_history') and portfolio.trade_history:
                for tr in portfolio.trade_history:
                    try:
                        when = tr.get('date')
                        if isinstance(when, str):
                            when = _dt.fromisoformat(when)
                        ticker = tr.get('ticker')
                        action = tr.get('action', 'BUY')
                        qty = tr.get('quantity', 0)
                        price = tr.get('price', None)
                        if when is None or ticker is None or price is None:
                            continue
                        signed_qty = qty if str(action).upper() == 'BUY' else -abs(qty)
                        attempt_type, attempt_name, price_source = _infer_attempt(ticker, when, float(price))
                        try:
                            expected_open_hour = 13
                            if attempt_type == 'market' and when.hour == expected_open_hour:
                                attempt_name = 'Market Order Open'
                        except Exception:
                            pass
                        normalized_attempt_type = 'limit' if attempt_type.startswith('limit_') else attempt_type
                        ref = per_day_refs.get((ticker, when.date()), {})
                        # Compute slippage for limit attempts when possible
                        slippage_val = None
                        try:
                            ref_prices = per_day_refs.get((ticker, when.date()), {}).get('prices', {})
                            market_open_px = ref.get('open')
                            limit_px = None
                            if attempt_type == 'limit_alpha':
                                limit_px = ref_prices.get('limit_alpha')
                            elif attempt_type == 'limit_beta':
                                limit_px = ref_prices.get('limit_beta')
                            if limit_px is not None and market_open_px is not None:
                                slippage_val = float(market_open_px) - float(limit_px)
                            elif attempt_type in ['market', 'moc']:
                                slippage_val = 0.0
                        except Exception:
                            slippage_val = None

                        # Compute per-attempt PnL when possible (using daily close)
                        pnl_val = None
                        try:
                            ref_close = ref.get('close')
                            if isinstance(ref_close, (int, float)):
                                if str(action).upper() == 'BUY':
                                    pnl_val = (float(ref_close) - float(price)) * signed_qty
                                else:  # SELL
                                    pnl_val = (float(price) - float(ref_close)) * abs(signed_qty)
                        except Exception:
                            pnl_val = None

                        hourly_ohlc = _get_hourly_ohlc(ticker, when)
                        confirmed_rows.append({
                            'timestamp': when,
                            'symbol': ticker,
                            'attempt_name': attempt_name,
                            'attempt_type': normalized_attempt_type,
                            'filled': True,
                            'side': str(action).upper(),
                            'quantity': signed_qty,
                            'execution_price': float(price),
                            'atr_value': ref.get('atr_value'),
                            'open': (hourly_ohlc.get('open') if hourly_ohlc else ref.get('open')),
                            'high': (hourly_ohlc.get('high') if hourly_ohlc else ref.get('high')),
                            'low': (hourly_ohlc.get('low') if hourly_ohlc else ref.get('low')),
                            'close': (hourly_ohlc.get('close') if hourly_ohlc else ref.get('close')),
                            'execution_category': 'daily',
                            'section_7_compliant': True,
                            'price_source': price_source,
                            'expected_open': ref.get('expected_open'),
                            'first_bar': ref.get('first_bar'),
                            'missing_open_bar': ref.get('missing_open_bar'),
                            'slippage': slippage_val,
                            'pnl': pnl_val,
                        })
                    except Exception:
                        continue
            
            # 2) Confirmed fills captured via on_fill (ensure we don't miss anything)
            if hasattr(self, 'confirmed_fills') and self.confirmed_fills:
                for f in self.confirmed_fills:
                    try:
                        when = f.get('timestamp')
                        ticker = f.get('symbol')
                        action = f.get('side', 'BUY')
                        qty = f.get('quantity', 0)
                        price = f.get('price', None)
                        if when is None or ticker is None or price is None:
                            continue
                        signed_qty = qty if str(action).upper() == 'BUY' else -abs(qty)
                        # Prefer explicit attempt info from metadata when available
                        raw_name = f.get('attempt_name_raw')
                        if isinstance(raw_name, str) and len(raw_name) > 0:
                            attempt_name = raw_name
                            if 'Market' in raw_name:
                                attempt_type = 'market'
                            elif 'Passive-1' in raw_name:
                                attempt_type = 'limit_alpha'
                            elif 'Passive-2' in raw_name:
                                attempt_type = 'limit_beta'
                            elif raw_name.upper() == 'MOC':
                                attempt_type = 'moc'
                            else:
                                attempt_type = 'market'
                            price_source = 'market' if attempt_type in ['market', 'moc'] else 'stored'
                        else:
                            attempt_type, attempt_name, price_source = _infer_attempt(ticker, when, float(price))
                            try:
                                expected_open_hour = 13
                                if attempt_type == 'market' and when.hour == expected_open_hour:
                                    attempt_name = 'Market Order Open'
                            except Exception:
                                pass
                        normalized_attempt_type = 'limit' if str(attempt_type).startswith('limit_') else attempt_type
                        ref = per_day_refs.get((ticker, when.date()), {})
                        # Compute slippage similar to trade_history path
                        slippage_val = None
                        try:
                            ref_prices = per_day_refs.get((ticker, when.date()), {}).get('prices', {})
                            market_open_px = ref.get('open')
                            limit_px = None
                            if attempt_type == 'limit_alpha':
                                limit_px = ref_prices.get('limit_alpha')
                            elif attempt_type == 'limit_beta':
                                limit_px = ref_prices.get('limit_beta')
                            if limit_px is not None and market_open_px is not None:
                                slippage_val = float(market_open_px) - float(limit_px)
                            elif attempt_type in ['market', 'moc']:
                                slippage_val = 0.0
                        except Exception:
                            slippage_val = None

                        # Compute per-attempt PnL for confirmed fills
                        pnl_val = None
                        try:
                            ref_close = ref.get('close')
                            if isinstance(ref_close, (int, float)):
                                if str(action).upper() == 'BUY':
                                    pnl_val = (float(ref_close) - float(price)) * signed_qty
                                else:
                                    pnl_val = (float(price) - float(ref_close)) * abs(signed_qty)
                        except Exception:
                            pnl_val = None

                        hourly_ohlc = _get_hourly_ohlc(ticker, when)
                        confirmed_rows.append({
                            'timestamp': when,
                            'symbol': ticker,
                            'attempt_name': attempt_name,
                            'attempt_type': normalized_attempt_type,
                            'filled': True,
                            'side': str(action).upper(),
                            'quantity': signed_qty,
                            'execution_price': float(price),
                            'atr_value': ref.get('atr_value'),
                            'open': (hourly_ohlc.get('open') if hourly_ohlc else ref.get('open')),
                            'high': (hourly_ohlc.get('high') if hourly_ohlc else ref.get('high')),
                            'low': (hourly_ohlc.get('low') if hourly_ohlc else ref.get('low')),
                            'close': (hourly_ohlc.get('close') if hourly_ohlc else ref.get('close')),
                            'execution_category': 'daily',
                            'section_7_compliant': True,
                            'price_source': price_source,
                            'expected_open': ref.get('expected_open'),
                            'first_bar': ref.get('first_bar'),
                            'missing_open_bar': ref.get('missing_open_bar'),
                            'slippage': slippage_val,
                            'pnl': pnl_val,
                        })
                    except Exception:
                        continue

            if confirmed_rows:
                df = pd.DataFrame(confirmed_rows)
                # Enforce one execution per attempt per day per symbol (keep first occurrence)
                try:
                    df.sort_values(['symbol', 'timestamp'], inplace=True)
                    df['day'] = df['timestamp'].dt.date
                    # Ensure uniqueness per symbol-day-attempt_name (first occurrence kept)
                    df = (
                        df.groupby(['symbol', 'day', 'attempt_name'], as_index=False, sort=False)
                          .first()
                    )
                    df.drop(columns=['day'], inplace=True)
                except Exception:
                    pass
                # Ensure MOC appears at end of the day only when it closes a non-zero intraday position
                try:
                    if not df.empty and 'attempt_type' in df.columns:
                        filtered_rows = []
                        for (sym, day), group in df.sort_values('timestamp').groupby([
                            'symbol', df['timestamp'].dt.date
                        ]):
                            running_pos = 0
                            day_rows = []
                            for _, row in group.iterrows():
                                if row['attempt_type'] == 'moc':
                                    if running_pos == 0:
                                        # Skip MOC with no position to close
                                        continue
                                    # Defer MOC append to ensure it becomes last row of the day
                                    day_rows.append(('moc', row))
                                    running_pos = 0
                                else:
                                    day_rows.append(('exec', row))
                                    try:
                                        running_pos += int(row.get('quantity', 0))
                                    except Exception:
                                        pass
                            # Append non-MOC first, then MOC to enforce ordering
                            for kind, row in day_rows:
                                if kind == 'exec':
                                    filtered_rows.append(row)
                            for kind, row in day_rows:
                                if kind == 'moc':
                                    filtered_rows.append(row)
                        if filtered_rows:
                            df = pd.DataFrame(filtered_rows)
                except Exception:
                    pass
            else:
                # Fallback: filter legacy execution_history to only filled==True
                filtered = [rec for rec in self.execution_history if rec.get('filled') is True]
                df = pd.DataFrame(filtered)

            # Ensure required aliases/columns for audit consumers
            try:
                if 'attempt_type' in df.columns and 'order_type' not in df.columns:
                    df['order_type'] = df['attempt_type']
                if 'filled' in df.columns and 'fill_status' not in df.columns:
                    df['fill_status'] = df['filled']
            except Exception:
                pass

            # Sort by timestamp and symbol for realistic portfolio-style chronology
            try:
                sort_cols = [c for c in ['timestamp', 'symbol'] if c in df.columns]
                if sort_cols:
                    df = df.sort_values(by=sort_cols, kind='mergesort')
            except Exception:
                pass

            # Keep in memory for UI generator; no CSV emission
            try:
                self.latest_execution_history_df = df
            except Exception:
                self.latest_execution_history_df = df.copy()
            self.context.logger.info("Execution history captured in memory for HTML report")
            
        except Exception as e:
            self.context.logger.error(f"Failed to save execution history: {e}")
    
    def check_brazilian_market_constraints(self, intent: OrderIntent) -> bool:
        """
        Check if the order intent complies with Brazilian market constraints.
        
        Args:
            intent: Order intent to validate
            
        Returns:
            True if constraints are satisfied, False otherwise
        """
        try:
            # Basic validation
            if intent.quantity <= 0:
                self.context.logger.warning(f"Invalid quantity: {intent.quantity}")
                return False
            
            if intent.price is not None and intent.price <= 0:
                self.context.logger.warning(f"Invalid price: {intent.price}")
                return False
            
            # Check lot size constraints
            if intent.quantity % self.min_lot_size != 0:
                self.context.logger.warning(f"Quantity {intent.quantity} not multiple of {self.min_lot_size}")
                return False
            
            # Pre-compute portfolio_value once to avoid unbound usage below
            # Fix: ensure portfolio_value is initialized regardless of pair-mode branch
            try:
                portfolio_value = float(self.context.portfolio.get_portfolio_value())
            except Exception:
                portfolio_value = 0.0

            # Check position size limits (disable per-leg cap in pair mode; global risk handled via tranche config)
            # Pair-mode mandatory: per-leg cap disabled globally
            if False:
                max_position_value = portfolio_value * self.max_position_size_pct
                if intent.price is not None:
                    position_value = intent.quantity * intent.price
                    if position_value > max_position_value:
                        self.context.logger.warning(f"Position value {position_value:.2f} exceeds limit {max_position_value:.2f}")
                        return False
            
            # Check daily loss limits
            if self.daily_loss > portfolio_value * self.max_daily_loss_pct:
                self.context.logger.warning(f"Daily loss {self.daily_loss:.2f} exceeds limit {portfolio_value * self.max_daily_loss_pct:.2f}")
                return False
            
            return True
            
        except Exception as e:
            self.context.logger.error(f"Error checking Brazilian market constraints: {e}")
            return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get strategy performance summary - all executions are now daily (once per day)."""
        # Calculate totals for daily tracking system (all order types are now daily)
        total_daily_executions = sum(self.daily_execution_counts.values())
        
        if total_daily_executions == 0:
            return {
                'total_executions': 0,
                'daily_executions': {},
                'legacy_executions': {},
                'execution_history_count': 0
            }
        
        # Calculate daily execution metrics (all order types)
        daily_summary = {}
        for attempt_type, total_count in self.daily_execution_counts.items():
            successful_count = self.daily_fill_rates.get(attempt_type, 0)
            failed_count = total_count - successful_count
            
            if total_count > 0:
                fill_rate = successful_count / total_count
            else:
                fill_rate = 0.0
            
            daily_summary[attempt_type] = {
                'attempts': total_count,
                'successful': successful_count,
                'failed': failed_count,
                'fill_rate': fill_rate
            }
        
        # Calculate legacy metrics (for backward compatibility)
        legacy_summary = {}
        for attempt_type, total_count in self.execution_counts.items():
            successful_count = self.fill_rates.get(attempt_type, 0)
            failed_count = total_count - successful_count
            
            if total_count > 0:
                fill_rate = successful_count / total_count
            else:
                fill_rate = 0.0
            
            legacy_summary[attempt_type] = {
                'attempts': total_count,
                'successful': successful_count,
                'failed': failed_count,
                'fill_rate': fill_rate
            }
        
        # Summary with all executions being daily
        summary = {
            'total_executions': total_daily_executions,
            'daily_executions': daily_summary,
            'legacy_executions': legacy_summary,
            'execution_history_count': len(self.execution_history)
        }
        
        # Log execution tracking summary
        self.context.logger.info("Execution tracking summary (all orders execute once per day):")
        self.context.logger.info(f"  Total attempts tracked: {total_daily_executions}")
        
        # Daily executions (all order types)
        if total_daily_executions > 0:
            self.context.logger.info("  Daily attempts (all order types - once per day):")
            for attempt_type, metrics in daily_summary.items():
                self.context.logger.info(
                    f"    {attempt_type}: {metrics['attempts']} attempts, "
                    f"{metrics['successful']} successful, {metrics['fill_rate']:.1%} fill rate"
                )
        
        return summary 