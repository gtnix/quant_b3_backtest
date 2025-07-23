"""
Enhanced Portfolio Manager for Brazilian Market Backtesting

Advanced portfolio management with comprehensive Brazilian market compliance:
- Enhanced loss carryforward with temporal management and audit trails
- Sophisticated T+2 settlement tracking with business day handling
- Comprehensive error handling and defensive programming
- Performance optimization and regulatory compliance
- Detailed logging and audit trail generation
- Transaction Cost Analysis (TCA) integration

Author: Your Name
Date: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Literal, Any
from dataclasses import dataclass, field
import logging
import yaml
import pytz
from enum import Enum

from engine.loss_manager import EnhancedLossCarryforwardManager
from engine.settlement_manager import AdvancedSettlementManager
from engine.tca import TransactionCostAnalyzer
from engine.market_utils import BrazilianMarketUtils, OrderType, LotType

# Configure logging
logger = logging.getLogger(__name__)


class AssetType(Enum):
    """
    Brazilian asset types for tax classification and exemption rules.
    
    Brazilian Tax Law (2025):
    - STOCK: Common shares - eligible for R$ 20,000 monthly exemption
    - ETF: Exchange Traded Funds - NOT eligible for exemption
    - FII: Real Estate Investment Trusts - NOT eligible for exemption
    - BDR: Brazilian Depositary Receipts - NOT eligible for exemption
    - OPTION: Options and derivatives - NOT eligible for exemption
    - FUTURE: Futures contracts - NOT eligible for exemption
    - BOND: Fixed income securities - NOT eligible for exemption
    """
    STOCK = "STOCK"      # Ações ordinárias - elegível para isenção R$ 20k
    ETF = "ETF"          # ETFs - NÃO elegível para isenção
    FII = "FII"          # Fundos Imobiliários - NÃO elegível para isenção
    BDR = "BDR"          # Brazilian Depositary Receipts - NÃO elegível para isenção
    OPTION = "OPTION"    # Opções - NÃO elegível para isenção
    FUTURE = "FUTURE"    # Futuros - NÃO elegível para isenção
    BOND = "BOND"        # Títulos de renda fixa - NÃO elegível para isenção
    
    @classmethod
    def is_exempt_eligible(cls, asset_type: str) -> bool:
        """
        Check if asset type is eligible for R$ 20,000 monthly exemption.
        
        Brazilian Tax Law: Only STOCK assets are eligible for the exemption.
        ETFs, FIIs, BDRs, options, futures, and bonds are NOT eligible.
        
        Args:
            asset_type: Asset type string
            
        Returns:
            bool: True if eligible for exemption, False otherwise
        """
        return asset_type == cls.STOCK.value
    
    @classmethod
    def validate(cls, asset_type: str) -> str:
        """
        Validate and normalize asset type string.
        
        Args:
            asset_type: Asset type string to validate
            
        Returns:
            str: Normalized asset type string
            
        Raises:
            ValueError: If asset_type is not a valid enum value
        """
        try:
            return cls(asset_type.upper()).value
        except ValueError:
            valid_types = [e.value for e in cls]
            raise ValueError(f"Invalid asset_type '{asset_type}'. Valid types: {valid_types}")


@dataclass
class Position:
    """Enhanced position tracking with comprehensive metadata."""
    ticker: str
    quantity: int
    avg_price: float
    current_price: float
    last_update: datetime
    trade_type: str = "swing_trade"  # 'day_trade' or 'swing_trade'
    position_id: Optional[str] = None
    description: str = ""
    
    @property
    def market_value(self) -> float:
        """Calculate current market value of position."""
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized profit/loss."""
        return (self.current_price - self.avg_price) * self.quantity
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized profit/loss percentage."""
        if self.avg_price == 0:
            return 0.0
        return ((self.current_price - self.avg_price) / self.avg_price) * 100


@dataclass
class ClassifiedTrade:
    """Classified trade with Brazilian tax compliance."""
    trade_id: str
    ticker: str
    action: str  # 'BUY' or 'SELL'
    quantity: int
    price: float
    value: float
    date: datetime
    broker: str = "default"
    asset_type: str = "STOCK"  # AssetType enum value - for exemption filtering
    
    # Classification results
    day_trade_qty: int = 0
    swing_trade_qty: int = 0
    day_trade_value: float = 0.0
    swing_trade_value: float = 0.0
    
    # Cost and tax information
    costs: Dict[str, float] = field(default_factory=dict)
    taxes: Dict[str, float] = field(default_factory=dict)
    gross_profit: float = 0.0
    net_profit: float = 0.0
    final_profit: float = 0.0
    
    # Audit information
    classification_audit: List[Dict] = field(default_factory=list)
    description: str = ""
    
    def __post_init__(self):
        if self.costs is None:
            self.costs = {}
        if self.taxes is None:
            self.taxes = {}
        if self.classification_audit is None:
            self.classification_audit = []
        
        # Validate asset_type
        self.asset_type = AssetType.validate(self.asset_type)
    
    @property
    def is_exempt_eligible(self) -> bool:
        """Check if this trade's asset type is eligible for R$ 20k exemption."""
        return AssetType.is_exempt_eligible(self.asset_type)


@dataclass
class BuyBlock:
    """Buy block for FIFO tracking."""
    trade_id: str
    quantity: int
    price: float
    date: datetime
    broker: str = "default"
    remaining_qty: int = 0  # Track remaining quantity after consumption
    
    def __post_init__(self):
        self.remaining_qty = self.quantity


@dataclass
class SwingInventory:
    """Swing inventory tracking with aggregated cost basis."""
    ticker: str
    quantity: int
    avg_cost: float
    last_update: datetime
    broker: str = "default"
    
    @property
    def total_value(self) -> float:
        return self.quantity * self.avg_cost


@dataclass
class ModalitySummary:
    """Monthly summary by trade modality (DT vs Swing)."""
    month: str
    modality: str  # 'DAY' or 'SWING'
    total_buys: int = 0
    total_sells: int = 0
    total_buy_value: float = 0.0
    total_sell_value: float = 0.0
    gross_profit: float = 0.0
    net_profit: float = 0.0
    taxable_profit: float = 0.0
    total_costs: float = 0.0
    total_taxes: float = 0.0
    irrf_credit: float = 0.0
    exemption_applied: bool = False
    exemption_amount: float = 0.0


class EnhancedPortfolio:
    """
    Advanced portfolio management with comprehensive Brazilian market compliance.
    
    Features:
    - Enhanced loss carryforward with temporal management
    - Sophisticated T+2 settlement tracking
    - Comprehensive error handling and validation
    - Performance optimization and caching
    - Detailed audit trails and regulatory compliance
    - Transaction Cost Analysis (TCA) integration
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initialize enhanced portfolio with comprehensive configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.timezone = pytz.timezone(self.config['market']['trading_hours']['timezone'])
        
        # Initialize enhanced managers
        self.loss_manager = EnhancedLossCarryforwardManager(
            config_path=config_path,
            timezone=self.config['market']['trading_hours']['timezone']
        )
        
        self.settlement_manager = AdvancedSettlementManager(
            initial_cash=self.config['portfolio']['initial_cash'],
            settlement_days=self.config.get('settlement', {}).get('cycle_days', 2),
            market_timezone=self.config['market']['trading_hours']['timezone']
        )
        
        # Initialize Transaction Cost Analyzer
        self.tca = TransactionCostAnalyzer(config_path)
        
        # Initialize market utilities
        market_config = self.config['market']
        self.market_utils = BrazilianMarketUtils(
            tick_size=market_config.get('tick_size', 0.01),
            round_lot_size=market_config.get('round_lot_size', 100)
        )
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        # self.cash: float = self.config['portfolio']['initial_cash']  # ← REMOVIDO: caixa gerenciado pelo settlement manager
        self.initial_cash: float = self.config['portfolio']['initial_cash']
        # self.total_value: float = self.cash  # ← REMOVIDO: será calculado dinamicamente
        self.total_value: float = 0.0  # Será calculado dinamicamente
        self.trade_history: List[Dict] = []
        self.daily_pnl: Dict[str, float] = {}  # {date: pnl}
        self.day_trade_pnl: float = 0.0  # Consolidated daily P&L for day trades
        
        # Enhanced classification system - Refactored for Brazilian tax law
        self.swing_inventory: Dict[str, SwingInventory] = {}  # {ticker: SwingInventory} - single aggregated record per ticker
        self.intraday_fifo: Dict[Tuple[str, str], List[BuyBlock]] = defaultdict(list)  # {(day_id, ticker): [BuyBlock]}
        self.intraday_bucket: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))  # {ticker: {broker: {date: {action: qty}}}}
        self.classified_trades: List[ClassifiedTrade] = []
        self.monthly_summaries: Dict[str, Dict[str, ModalitySummary]] = defaultdict(lambda: defaultdict(lambda: ModalitySummary(month="", modality="")))
        
        # Performance tracking
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.total_commission: float = 0.0
        self.total_taxes: float = 0.0
        
        logger.info(f"Enhanced Portfolio initialized with R$ {self.settlement_manager.total_cash:,.2f} initial capital")
    
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
    
    def get_tax_rule_for_date(self, trade_date: datetime) -> Dict:
        """
        Get applicable tax rule for a specific date.
        
        Brazilian tax rules are versioned with date ranges.
        This method returns the correct rule based on the trade date.
        
        Args:
            trade_date: Date of the trade
            
        Returns:
            Dict containing the applicable tax rule
            
        Raises:
            ValueError: If no applicable rule is found
        """
        if 'tax_rules' not in self.config:
            # Fallback to legacy configuration
            return {
                'swing_rate': self.config['taxes']['swing_trade'],
                'daytrade_rate': self.config['taxes']['day_trade'],
                'swing_exemption_brl': self.config['taxes']['swing_exemption_limit'],
                'dt_same_broker_required': False,
                'irrf_swing_rate': self.config['taxes']['irrf_swing_rate'],
                'irrf_day_rate': self.config['taxes']['irrf_day_rate'],
                'person_type': self.config['taxes']['person_type'],
                'max_loss_offset_percentage': self.config['taxes']['max_loss_offset_percentage'],
                'loss_carryforward_perpetual': self.config['taxes']['loss_carryforward_perpetual']
            }
        
        trade_date_str = trade_date.date().isoformat()
        
        for rule in self.config['tax_rules']:
            if rule['start'] <= trade_date_str <= rule['end']:
                logger.debug(f"Applied tax rule for {trade_date_str}: {rule['start']} to {rule['end']}")
                return rule
        
        raise ValueError(f"No applicable tax rule found for date {trade_date_str}")
    
    def classify_trades(self, trades: List[Dict]) -> List[ClassifiedTrade]:
        """
        Classify trades chronologically according to Brazilian tax law.
        
        Brazilian Day Trade Classification Rules (Simplified):
        1. Sort by (datetime_sp, ticker, trade_id) - full timestamp in São Paulo TZ
        2. Day Trade (DT): Same ticker, buy and sell on the same trading day
        3. Swing: Anything that is not DT
        4. FIFO rules: DT pairing within same day, swing inventory across days
        5. Isenção R$ 20k: Only for swing stock sales in the month
        
        Args:
            trades: List of trade dictionaries with keys: trade_id, ticker, action, quantity, 
                   price, value, date, broker, description, asset_type
                   
        Returns:
            List of ClassifiedTrade objects with day_trade_qty and swing_trade_qty
        """
        if not trades:
            return []
        
        # Sort trades by (datetime_sp, ticker, trade_id) - full timestamp in São Paulo TZ
        sp_timezone = pytz.timezone('America/Sao_Paulo')
        sorted_trades = []
        
        for trade in trades:
            # Ensure datetime is in São Paulo timezone
            trade_date = trade['date']
            if trade_date.tzinfo is None:
                trade_date = sp_timezone.localize(trade_date)
            else:
                trade_date = trade_date.astimezone(sp_timezone)
            
            sorted_trades.append({
                **trade,
                'date': trade_date,
                'datetime_sp': trade_date
            })
        
        # Sort by (datetime_sp, ticker, trade_id)
        sorted_trades.sort(key=lambda x: (x['datetime_sp'], x['ticker'], x.get('trade_id', '')))
        
        classified_trades = []
        current_day = None  # Track current day being processed
        
        for trade in sorted_trades:
            trade_day = trade['date'].date().isoformat()
            
            # If we're moving to a new day, rollover the previous day
            if current_day is not None and trade_day != current_day:
                # Rollover only the previous day
                self.rollover_day(current_day)
            
            classified_trade = self._classify_single_trade_refactored(trade)
            classified_trades.append(classified_trade)
            current_day = trade_day
        
        # Rollover the last day after processing all trades
        if current_day is not None:
            self.rollover_day(current_day)
        
        return classified_trades
    
    def _classify_single_trade_refactored(self, trade: Dict) -> ClassifiedTrade:
        """
        Refactored classification of a single trade according to Brazilian tax law.
        
        Args:
            trade: Trade dictionary with datetime_sp field
            
        Returns:
            ClassifiedTrade object
        """
        ticker = trade['ticker']
        action = trade['action']
        quantity = trade['quantity']
        price = trade['price']
        value = trade['value']
        date = trade['date']
        broker = trade.get('broker', 'default')
        asset_type = trade.get('asset_type', AssetType.STOCK.value)  # Default to STOCK for exemption filtering
        trade_id = trade.get('trade_id', f"{ticker}_{action}_{date.isoformat()}")
        description = trade.get('description', '')
        
        # Check if this is an after-hours trade
        if self._is_after_hours_trade(date):
            logger.debug(f"After-hours trade detected for {ticker} at {date.strftime('%H:%M:%S')}")
            return self._classify_after_hours_trade(trade)
        
        # Initialize classified trade
        classified_trade = ClassifiedTrade(
            trade_id=trade_id,
            ticker=ticker,
            action=action,
            quantity=quantity,
            price=price,
            value=value,
            date=date,
            broker=broker,
            asset_type=asset_type,
            description=description
        )
        
        # Transfer costs from original trade if available
        if 'costs' in trade:
            classified_trade.costs = trade['costs'].copy()
        if 'taxes' in trade:
            classified_trade.taxes = trade['taxes'].copy()
        if 'gross_profit' in trade:
            classified_trade.gross_profit = trade['gross_profit']
        if 'net_profit' in trade:
            classified_trade.net_profit = trade['net_profit']
        if 'final_profit' in trade:
            classified_trade.final_profit = trade['final_profit']
        
        if action == 'BUY':
            # Add buy block to intraday FIFO ONLY
            day_id = date.date().isoformat()
            buy_block = BuyBlock(
                trade_id=trade_id,
                quantity=quantity,
                price=price,
                date=date,
                broker=broker
            )
            self.intraday_fifo[(day_id, ticker)].append(buy_block)
            
            # Update intraday bucket for day trade detection
            self.intraday_bucket[ticker][broker][day_id]['BUY'] += quantity
            
            # DO NOT add to swing inventory immediately - will be done at day close rollover
            # All buys go to swing inventory initially (will be consumed by sells)
            classified_trade.swing_trade_qty = quantity
            classified_trade.swing_trade_value = value
            
            classified_trade.classification_audit.append({
                'step': 'buy_classification_refactored',
                'dt_qty': 0,
                'swing_qty': quantity,
                'day_id': day_id,
                'reason': f"Buy {quantity} {ticker} added to intraday FIFO (swing inventory at day close)"
            })
        
        elif action == 'SELL':
            # Consume from intraday FIFO first (DT), then swing inventory
            day_id = date.date().isoformat()
            
            # Update intraday bucket for day trade detection
            self.intraday_bucket[ticker][broker][day_id]['SELL'] += quantity
            
            # First, try to consume from intraday FIFO (day trade)
            dt_consumed = self._consume_intraday_fifo(ticker, day_id, quantity, price, trade_id)
            
            # Remaining quantity goes to swing inventory
            remaining_qty = quantity - dt_consumed['total_qty']
            swing_consumed = self._consume_swing_fifo(ticker, remaining_qty, price, trade_id)
            
            # Set classification results
            classified_trade.day_trade_qty = dt_consumed['total_qty']
            classified_trade.swing_trade_qty = swing_consumed['total_qty']
            classified_trade.day_trade_value = dt_consumed['total_value']
            classified_trade.swing_trade_value = swing_consumed['total_value']
            
            # Add audit trail
            classified_trade.classification_audit.extend(dt_consumed['audit'])
            classified_trade.classification_audit.extend(swing_consumed['audit'])
            
            classified_trade.classification_audit.append({
                'step': 'sell_classification_summary',
                'dt_qty': dt_consumed['total_qty'],
                'swing_qty': swing_consumed['total_qty'],
                'unmatched_qty': quantity - dt_consumed['total_qty'] - swing_consumed['total_qty'],
                'day_id': day_id,
                'reason': f"Sell {quantity} {ticker}: {dt_consumed['total_qty']} DT, {swing_consumed['total_qty']} swing"
            })
        
        return classified_trade
    
    def _consume_intraday_fifo(self, ticker: str, day_id: str, sell_qty: int, sell_price: float, sell_trade_id: str) -> Dict:
        """
        Consume from intraday FIFO for day trade classification.
        
        Args:
            ticker: Trading asset identifier
            day_id: Trading day identifier (YYYY-MM-DD)
            sell_qty: Quantity to sell
            sell_price: Sell price
            sell_trade_id: Sell trade ID
            
        Returns:
            Dict with consumption details and audit trail
        """
        total_consumed = 0
        total_value = 0.0
        audit = []
        
        fifo_key = (day_id, ticker)
        if fifo_key not in self.intraday_fifo:
            return {
                'total_qty': 0,
                'total_value': 0.0,
                'audit': audit
            }
        
        buy_blocks = self.intraday_fifo[fifo_key]
        remaining_sell_qty = sell_qty
        
        while remaining_sell_qty > 0 and buy_blocks:
            buy_block = buy_blocks[0]  # FIFO
            
            if buy_block.remaining_qty <= 0:
                buy_blocks.pop(0)  # Remove fully consumed block
                continue
            
            # Consume from this buy block
            consume_qty = min(remaining_sell_qty, buy_block.remaining_qty)
            consume_value = consume_qty * sell_price
            
            # Update buy block
            buy_block.remaining_qty -= consume_qty
            
            # Update totals
            total_consumed += consume_qty
            total_value += consume_value
            remaining_sell_qty -= consume_qty
            
            # Add to audit trail
            audit.append({
                'step': 'intraday_fifo_consumption',
                'buy_id': buy_block.trade_id,
                'sell_id': sell_trade_id,
                'qty': consume_qty,
                'buy_price': buy_block.price,
                'sell_price': sell_price,
                'modality': 'DT',
                'day_id': day_id,
                'reason': f"Day trade: {consume_qty} {ticker} @ R$ {sell_price:.2f} matched with buy @ R$ {buy_block.price:.2f}"
            })
            
            # Remove fully consumed block
            if buy_block.remaining_qty <= 0:
                buy_blocks.pop(0)
        
        return {
            'total_qty': total_consumed,
            'total_value': total_value,
            'audit': audit
        }
    
    def _consume_swing_fifo(self, ticker: str, sell_qty: int, sell_price: float, sell_trade_id: str) -> Dict:
        """
        Consume from swing inventory using FIFO for quantity but current average cost for profit calculation.
        
        Args:
            ticker: Trading asset identifier
            sell_qty: Quantity to sell
            sell_price: Sell price
            sell_trade_id: Sell trade ID
            
        Returns:
            Dict with consumption details and audit trail
        """
        total_consumed = 0
        total_value = 0.0
        audit = []
        
        if ticker not in self.swing_inventory:
            # No swing inventory available - this is an unmatched sell
            # In real trading, this would be a short sale or error
            # For backtesting purposes, we'll mark it as unmatched
            audit.append({
                'step': 'swing_fifo_consumption',
                'buy_id': f"unmatched_{ticker}",
                'sell_id': sell_trade_id,
                'qty': 0,
                'buy_price': 0.0,
                'sell_price': sell_price,
                'modality': 'SWING',
                'remaining_qty': 0,
                'reason': f"Unmatched swing trade: {sell_qty} {ticker} @ R$ {sell_price:.2f} (no swing inventory available)"
            })
            
            return {
                'total_qty': 0,
                'total_value': 0.0,
                'audit': audit
            }
        
        inventory = self.swing_inventory[ticker]
        available_qty = inventory.quantity
        
        if available_qty <= 0:
            # Empty inventory - same as no inventory
            audit.append({
                'step': 'swing_fifo_consumption',
                'buy_id': f"empty_{ticker}",
                'sell_id': sell_trade_id,
                'qty': 0,
                'buy_price': 0.0,
                'sell_price': sell_price,
                'modality': 'SWING',
                'remaining_qty': 0,
                'reason': f"Unmatched swing trade: {sell_qty} {ticker} @ R$ {sell_price:.2f} (empty swing inventory)"
            })
            
            return {
                'total_qty': 0,
                'total_value': 0.0,
                'audit': audit
            }
        
        # Consume using FIFO for quantity, but use current average cost for profit calculation
        consume_qty = min(sell_qty, available_qty)
        consume_value = consume_qty * sell_price
        
        # Update inventory
        inventory.quantity -= consume_qty
        
        # Add to audit trail
        audit.append({
            'step': 'swing_fifo_consumption',
            'buy_id': f"swing_{ticker}_aggregated",
            'sell_id': sell_trade_id,
            'qty': consume_qty,
            'buy_price': inventory.avg_cost,  # Current average cost
            'sell_price': sell_price,
            'modality': 'SWING',
            'remaining_qty': inventory.quantity,
            'reason': f"Swing trade: {consume_qty} {ticker} @ R$ {sell_price:.2f} matched with avg cost @ R$ {inventory.avg_cost:.2f}"
        })
        
        # Remove inventory if empty
        if inventory.quantity <= 0:
            del self.swing_inventory[ticker]
        
        return {
            'total_qty': consume_qty,
            'total_value': consume_value,
            'audit': audit
        }
    
    def rollover_day(self, day_id: str) -> None:
        """
        Rollover remaining intraday quantities to swing inventory at day close.
        
        This method should be called after processing all trades of a specific day
        to move any remaining quantities from intraday FIFO to swing inventory.
        
        Args:
            day_id: Trading day identifier (YYYY-MM-DD)
        """
        logger.info(f"Rolling over day {day_id} - moving remaining intraday quantities to swing inventory")
        
        # Find all tickers with intraday FIFO for this day
        day_tickers = [ticker for (d, ticker) in self.intraday_fifo.keys() if d == day_id]
        
        for ticker in day_tickers:
            fifo_key = (day_id, ticker)
            buy_blocks = self.intraday_fifo[fifo_key]
            
            # Process remaining quantities from each buy block
            for buy_block in buy_blocks:
                if buy_block.remaining_qty > 0:
                    # Add remaining quantity to swing inventory
                    self._add_to_swing_inventory(
                        ticker=ticker,
                        quantity=buy_block.remaining_qty,
                        price=buy_block.price,
                        date=buy_block.date,
                        broker=buy_block.broker
                    )
                    
                    logger.debug(f"Rolled over {buy_block.remaining_qty} {ticker} @ R$ {buy_block.price:.2f} "
                               f"from day {day_id} to swing inventory")
            
            # Clear the intraday FIFO for this day/ticker
            del self.intraday_fifo[fifo_key]
        
        logger.info(f"Day {day_id} rollover completed - {len(day_tickers)} tickers processed")

    def _add_to_swing_inventory(self, ticker: str, quantity: int, price: float, date: datetime, broker: str = "default") -> None:
        """
        Add buy blocks to swing inventory with aggregated cost basis.
        
        Args:
            ticker: Trading asset identifier
            quantity: Quantity to add
            price: Buy price
            date: Buy date
            broker: Broker identifier
        """
        inv = self.swing_inventory.get(ticker)
        if inv is None:
            self.swing_inventory[ticker] = SwingInventory(
                ticker=ticker,
                quantity=quantity,
                avg_cost=price,
                last_update=date,
                broker=broker
            )
            logger.debug(f"Created swing inventory: {quantity} {ticker} @ R$ {price:.2f} (broker: {broker})")
        else:
            # Update existing aggregated record
            total_qty = inv.quantity + quantity
            total_cost = inv.quantity * inv.avg_cost + quantity * price
            inv.avg_cost = total_cost / total_qty
            inv.quantity = total_qty
            inv.last_update = date
            
            logger.debug(f"Updated swing inventory: {ticker} - {quantity} @ R$ {price:.2f} -> "
                        f"Total: {total_qty} @ R$ {inv.avg_cost:.2f}")
    
    def _detect_day_trade_simplified(self, ticker: str, sell_date: datetime) -> bool:
        """
        Simplified day trade detection for single broker scenario.
        
        Args:
            ticker: Trading asset identifier
            sell_date: Date of the sell operation
            
        Returns:
            True if this is a day trade, False otherwise
        """
        try:
            # Use cached intraday bucket for O(1) lookup instead of linear search
            date_str = sell_date.date().isoformat()
            
            # Check if there are any buys on the same day for this ticker
            same_day_buys = 0
            for broker_key in self.intraday_bucket[ticker]:
                same_day_buys += self.intraday_bucket[ticker][broker_key].get(date_str, {}).get('BUY', 0)
            
            # If there are no buys on the same day, it's not a day trade
            if same_day_buys == 0:
                return False
            
            # Check if there are any sells on the same day (indicating day trading activity)
            same_day_sells = 0
            for broker_key in self.intraday_bucket[ticker]:
                same_day_sells += self.intraday_bucket[ticker][broker_key].get(date_str, {}).get('SELL', 0)
            
            # If there are sells on the same day, this is part of day trading
            if same_day_sells > 0:
                return True
            
            # Check if the most recent buy before this sell is on the same day
            # This is a fallback for the first sell of the day
            return same_day_buys > 0
            
        except Exception as e:
            logger.error(f"Error in simplified day trade detection for {ticker}: {str(e)}")
            return False
    
    def aggregate_monthly(self, classified_trades: List[ClassifiedTrade]) -> Dict[str, Dict[str, ModalitySummary]]:
        """
        Aggregate classified trades by month and modality (DT vs Swing) with enhanced profit calculation.
        
        Args:
            classified_trades: List of ClassifiedTrade objects
            
        Returns:
            Dict[month, Dict[modality, ModalitySummary]]
        """
        monthly_summaries = defaultdict(lambda: defaultdict(lambda: ModalitySummary(month="", modality="")))
        
        for trade in classified_trades:
            month = trade.date.strftime('%Y-%m')
            
            # Process day trade portion
            if trade.day_trade_qty > 0:
                if monthly_summaries[month]['DAY'].month == "":
                    monthly_summaries[month]['DAY'] = ModalitySummary(month=month, modality='DAY')
                
                summary = monthly_summaries[month]['DAY']
                
                if trade.action == 'BUY':
                    summary.total_buys += trade.day_trade_qty
                    summary.total_buy_value += trade.day_trade_value
                else:  # SELL
                    summary.total_sells += trade.day_trade_qty
                    summary.total_sell_value += trade.day_trade_value
                    
                    # Calculate profit for sells using audit trail for price matching
                    if trade.classification_audit:
                        # Look for price matching information in audit trail
                        profit = self._calculate_profit_from_audit(trade, 'day_trade')
                        summary.gross_profit += profit
            
            # Process swing trade portion
            if trade.swing_trade_qty > 0:
                if monthly_summaries[month]['SWING'].month == "":
                    monthly_summaries[month]['SWING'] = ModalitySummary(month=month, modality='SWING')
                
                summary = monthly_summaries[month]['SWING']
                
                if trade.action == 'BUY':
                    summary.total_buys += trade.swing_trade_qty
                    summary.total_buy_value += trade.swing_trade_value
                else:  # SELL
                    summary.total_sells += trade.swing_trade_qty
                    summary.total_sell_value += trade.swing_trade_value
                    
                    # Calculate profit for sells using audit trail for price matching
                    if trade.classification_audit:
                        profit = self._calculate_profit_from_audit(trade, 'swing_trade')
                        summary.gross_profit += profit
        
        # Calculate net profits and apply exemptions
        for month in monthly_summaries:
            for modality in monthly_summaries[month]:
                summary = monthly_summaries[month][modality]
                
                # Calculate net profit (gross - costs)
                # Aggregate costs for this month/modality from classified trades
                total_costs = 0.0
                for trade in classified_trades:
                    if (trade.date.strftime('%Y-%m') == month and 
                        trade.costs and 'total_costs' in trade.costs):
                        # Distribute costs proportionally based on trade classification
                        total_qty = trade.day_trade_qty + trade.swing_trade_qty
                        if total_qty > 0:  # Only distribute if there's actual quantity
                            if modality == 'DAY' and trade.day_trade_qty > 0:
                                # Day trade portion of costs
                                day_ratio = trade.day_trade_qty / total_qty
                                total_costs += trade.costs['total_costs'] * day_ratio
                            elif modality == 'SWING' and trade.swing_trade_qty > 0:
                                # Swing trade portion of costs
                                swing_ratio = trade.swing_trade_qty / total_qty
                                total_costs += trade.costs['total_costs'] * swing_ratio
                        else:
                            # If no quantity is classified, distribute costs equally or based on value
                            if modality == 'DAY' and trade.day_trade_value > 0:
                                day_ratio = trade.day_trade_value / (trade.day_trade_value + trade.swing_trade_value) if (trade.day_trade_value + trade.swing_trade_value) > 0 else 0.5
                                total_costs += trade.costs['total_costs'] * day_ratio
                            elif modality == 'SWING' and trade.swing_trade_value > 0:
                                swing_ratio = trade.swing_trade_value / (trade.day_trade_value + trade.swing_trade_value) if (trade.day_trade_value + trade.swing_trade_value) > 0 else 0.5
                                total_costs += trade.costs['total_costs'] * swing_ratio
                
                summary.total_costs = total_costs
                summary.net_profit = summary.gross_profit - total_costs
                
                # Apply swing trade exemption - ONLY for exempt-eligible assets
                if modality == 'SWING':
                    # Calculate total swing sales for exempt-eligible assets only
                    exempt_eligible_sales = 0.0
                    for trade in classified_trades:
                        if (trade.date.strftime('%Y-%m') == month and 
                            trade.action == 'SELL' and 
                            trade.swing_trade_qty > 0 and
                            trade.is_exempt_eligible):
                            exempt_eligible_sales += trade.swing_trade_value
                    
                    # Use loss_manager to calculate exemption (centralized logic)
                    tax_rule = self.get_tax_rule_for_date(trade.date)  # Use last trade date as reference
                    exemption_limit = tax_rule['swing_exemption_brl']
                    
                    if exempt_eligible_sales <= exemption_limit:
                        summary.exemption_applied = True
                        summary.exemption_amount = exempt_eligible_sales
                        summary.taxable_profit = 0.0
                    else:
                        summary.taxable_profit = summary.net_profit
                else:  # DAY
                    summary.taxable_profit = summary.net_profit
        
        return monthly_summaries
    
    def _calculate_profit_from_audit(self, trade: ClassifiedTrade, trade_type: str) -> float:
        """
        Calculate profit from audit trail information for enhanced accuracy.
        
        Args:
            trade: ClassifiedTrade object
            trade_type: 'day_trade' or 'swing_trade'
            
        Returns:
            Calculated profit
        """
        if trade.action != 'SELL':
            return 0.0
        
        total_profit = 0.0
        
        # Look for consumption details in audit trail
        for audit_entry in trade.classification_audit:
            if trade_type == 'day_trade' and audit_entry.get('step') == 'intraday_fifo_consumption':
                # Day trade profit calculation
                buy_price = audit_entry.get('buy_price', 0.0)
                sell_price = audit_entry.get('sell_price', 0.0)
                qty = audit_entry.get('qty', 0)
                
                profit = (sell_price - buy_price) * qty
                total_profit += profit
                
            elif trade_type == 'swing_trade' and audit_entry.get('step') == 'swing_fifo_consumption':
                # Swing trade profit calculation
                buy_price = audit_entry.get('buy_price', 0.0)  # avg_cost from swing inventory
                sell_price = audit_entry.get('sell_price', 0.0)
                qty = audit_entry.get('qty', 0)
                
                profit = (sell_price - buy_price) * qty
                total_profit += profit
        
        return total_profit
    
    def compute_tax(self, monthly_summaries: Dict[str, Dict[str, ModalitySummary]], 
                   trade_date: datetime = None) -> Dict[str, Any]:
        """
        Compute tax based on monthly summaries using the correct loss_manager approach.
        
        This method now delegates to calculate_monthly_tax_liability to ensure
        proper loss carryforward application and prevent double taxation.
        
        Args:
            monthly_summaries: Output from aggregate_monthly()
            trade_date: Reference date for tax rule selection (defaults to current date)
            
        Returns:
            Dict containing tax computation results
        """
        if trade_date is None:
            trade_date = datetime.now()
        
        tax_rule = self.get_tax_rule_for_date(trade_date)
        
        tax_report = {
            'tax_rule_applied': tax_rule,
            'months': {},
            'total_tax_liability': 0.0,
            'total_irrf_credit': 0.0,
            'final_darf_liability': 0.0,
            'calculation_method': 'loss_manager_delegated'
        }
        
        # Process each month using the correct loss_manager approach
        for month_str in monthly_summaries.keys():
            month_date = datetime.strptime(month_str, '%Y-%m').date()
            
            # Use the correct monthly tax calculation method
            monthly_tax_result = self.calculate_monthly_tax_liability(month_date)
            
            # Convert the result to the expected format
            month_tax = {
                'swing_trade': {
                    'taxable_profit': monthly_tax_result['swing_trade']['taxable_profit'],
                    'tax_rate': tax_rule['swing_rate'],
                    'capital_gains_tax': monthly_tax_result['swing_trade']['capital_gains_tax'],
                    'irrf_credit': monthly_tax_result['swing_trade']['irrf_credit'],
                    'exemption_applied': monthly_tax_result['swing_trade']['taxable_profit'] == 0.0 and 
                                       monthly_tax_result['swing_trade']['monthly_sales'] > 0,
                    'exemption_amount': monthly_tax_result['swing_trade']['monthly_sales'] if 
                                      monthly_tax_result['swing_trade']['taxable_profit'] == 0.0 else 0.0
                },
                'day_trade': {
                    'taxable_profit': monthly_tax_result['day_trade']['taxable_profit'],
                    'tax_rate': tax_rule['daytrade_rate'],
                    'capital_gains_tax': monthly_tax_result['day_trade']['capital_gains_tax'],
                    'irrf_credit': monthly_tax_result['day_trade']['irrf_credit']
                },
                'total': {
                    'capital_gains_tax': monthly_tax_result['total']['capital_gains_tax'],
                    'irrf_credit': monthly_tax_result['total']['irrf_credit'],
                    'final_darf_liability': monthly_tax_result['total']['final_darf_liability']
                }
            }
            
            tax_report['months'][month_str] = month_tax
            tax_report['total_tax_liability'] += monthly_tax_result['total']['capital_gains_tax']
            tax_report['total_irrf_credit'] += monthly_tax_result['total']['irrf_credit']
            tax_report['final_darf_liability'] += monthly_tax_result['total']['final_darf_liability']
        
        logger.info(f"Tax computation completed using loss_manager delegation for {len(monthly_summaries)} months")
        return tax_report
    
    def _validate_trade_inputs(self, ticker: str, quantity: int, price: float, 
                              trade_date: datetime, trade_type: str) -> None:
        """
        Validate trade inputs with comprehensive error checking.
        
        Args:
            ticker: Trading asset identifier
            quantity: Number of shares
            price: Price per share
            trade_date: Date of trade
            trade_type: Type of trade ('day_trade', 'swing_trade', or 'auto')
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Ticker must be a non-empty string")
        
        if not isinstance(quantity, int) or quantity <= 0:
            raise ValueError("Quantity must be a positive integer")
        
        if not isinstance(price, (int, float)) or price <= 0:
            raise ValueError("Price must be a positive numeric value")
        
        if not isinstance(trade_date, datetime):
            raise ValueError("Trade date must be a datetime object")
        
        if trade_type not in ['day_trade', 'swing_trade', 'auto']:
            raise ValueError("Trade type must be 'day_trade', 'swing_trade', or 'auto'")
    
    def _handle_corporate_action(self, ticker: str, action_type: str, ratio: float, 
                                effective_date: datetime) -> None:
        """
        Handle corporate actions that affect position classification.
        
        Brazilian tax law requires special handling for:
        - Stock splits (affect quantity and cost basis)
        - Dividends (may affect classification)
        - Mergers/Acquisitions (affect ticker and classification)
        
        Args:
            ticker: Affected ticker
            action_type: Type of corporate action
            ratio: Adjustment ratio
            effective_date: Date when action takes effect
        """
        if ticker not in self.positions:
            return
        
        position = self.positions[ticker]
        
        if action_type == 'stock_split':
            # Adjust quantity and cost basis
            old_quantity = position.quantity
            position.quantity = int(old_quantity * ratio)
            position.avg_price = position.avg_price / ratio
            
            logger.info(f"Stock split applied to {ticker}: {old_quantity} -> {position.quantity} shares, "
                       f"cost basis adjusted from R$ {position.avg_price * ratio:.2f} to R$ {position.avg_price:.2f}")
        
        elif action_type == 'reverse_split':
            # Reverse stock split
            old_quantity = position.quantity
            position.quantity = int(old_quantity / ratio)
            position.avg_price = position.avg_price * ratio
            
            logger.info(f"Reverse split applied to {ticker}: {old_quantity} -> {position.quantity} shares, "
                       f"cost basis adjusted from R$ {position.avg_price / ratio:.2f} to R$ {position.avg_price:.2f}")
        
        # Update swing inventory for corporate actions
        if ticker in self.swing_inventory:
            inventory = self.swing_inventory[ticker]
            if action_type == 'stock_split':
                inventory.quantity = int(inventory.quantity * ratio)
                inventory.avg_cost = inventory.avg_cost / ratio
            elif action_type == 'reverse_split':
                inventory.quantity = int(inventory.quantity / ratio)
                inventory.avg_cost = inventory.avg_cost * ratio
    
    def _is_after_hours_trade(self, trade_date: datetime) -> bool:
        """
        Check if trade occurred during after-hours session.
        
        Brazilian market after-hours session varies by auction type and daylight saving time.
        Configuration allows for flexible time windows (typically 17:00-18:25).
        After-hours trades are typically classified as swing trades regardless of same-day activity.
        
        Args:
            trade_date: Date and time of trade
            
        Returns:
            True if after-hours trade, False otherwise
        """
        # Check if after-hours detection is enabled
        after_hours_config = self.config['market'].get('after_hours', {})
        if not after_hours_config.get('enabled', True):
            return False
        
        # Convert to São Paulo timezone
        sp_timezone = pytz.timezone('America/Sao_Paulo')
        if trade_date.tzinfo is None:
            trade_date = sp_timezone.localize(trade_date)
        else:
            trade_date = trade_date.astimezone(sp_timezone)
        
        # Parse after-hours time window from configuration
        try:
            start_time_str = after_hours_config.get('start', '17:00')
            end_time_str = after_hours_config.get('end', '18:25')
            
            # Parse time strings (HH:MM format)
            start_hour, start_minute = map(int, start_time_str.split(':'))
            end_hour, end_minute = map(int, end_time_str.split(':'))
            
            # Create time boundaries for the same date as trade_date
            after_hours_start = trade_date.replace(
                hour=start_hour, minute=start_minute, second=0, microsecond=0
            )
            after_hours_end = trade_date.replace(
                hour=end_hour, minute=end_minute, second=0, microsecond=0
            )
            
            # Handle case where end time is on the next day (e.g., 23:59)
            if after_hours_end < after_hours_start:
                after_hours_end += timedelta(days=1)
            
            is_after_hours = after_hours_start <= trade_date <= after_hours_end
            
            if is_after_hours:
                logger.debug(f"After-hours trade detected: {trade_date.strftime('%H:%M:%S')} "
                           f"(window: {start_time_str}-{end_time_str})")
            
            return is_after_hours
            
        except (ValueError, KeyError) as e:
            logger.warning(f"Error parsing after-hours configuration: {e}. Using fallback 17:00-18:00")
            # Fallback to hardcoded values if configuration parsing fails
            after_hours_start = trade_date.replace(hour=17, minute=0, second=0, microsecond=0)
            after_hours_end = trade_date.replace(hour=18, minute=0, second=0, microsecond=0)
            return after_hours_start <= trade_date <= after_hours_end
    
    def _classify_after_hours_trade(self, trade: Dict) -> ClassifiedTrade:
        """
        Special classification for after-hours trades.
        
        After-hours trades in Brazil are typically treated as swing trades
        regardless of same-day activity, unless explicitly marked as day trades.
        Classification can be configured via settings.yaml.
        
        Args:
            trade: Trade dictionary
            
        Returns:
            ClassifiedTrade object
        """
        ticker = trade['ticker']
        action = trade['action']
        quantity = trade['quantity']
        price = trade['price']
        value = trade['value']
        date = trade['date']
        broker = trade.get('broker', 'default')
        trade_id = trade.get('trade_id', f"{ticker}_{action}_{date.isoformat()}")
        description = trade.get('description', '')
        
        # Initialize classified trade
        classified_trade = ClassifiedTrade(
            trade_id=trade_id,
            ticker=ticker,
            action=action,
            quantity=quantity,
            price=price,
            value=value,
            date=date,
            broker=broker,
            description=description
        )
        
        # Transfer costs from original trade if available
        if 'costs' in trade:
            classified_trade.costs = trade['costs'].copy()
        if 'taxes' in trade:
            classified_trade.taxes = trade['taxes'].copy()
        if 'gross_profit' in trade:
            classified_trade.gross_profit = trade['gross_profit']
        if 'net_profit' in trade:
            classified_trade.net_profit = trade['net_profit']
        if 'final_profit' in trade:
            classified_trade.final_profit = trade['final_profit']
        
        # Get classification override from configuration
        after_hours_config = self.config['market'].get('after_hours', {})
        classification_override = after_hours_config.get('classification_override', 'swing_trade')
        
        # Apply classification override
        if classification_override == 'swing_trade':
            classified_trade.swing_trade_qty = quantity
            classified_trade.swing_trade_value = value
            trade_type = 'swing_trade'
        elif classification_override == 'day_trade':
            classified_trade.day_trade_qty = quantity
            classified_trade.day_trade_value = value
            trade_type = 'day_trade'
        else:
            # Default to swing trade if invalid configuration
            classified_trade.swing_trade_qty = quantity
            classified_trade.swing_trade_value = value
            trade_type = 'swing_trade'
            logger.warning(f"Invalid after-hours classification_override: {classification_override}. Using swing_trade")
        
        # Get time window for audit trail
        start_time = after_hours_config.get('start', '17:00')
        end_time = after_hours_config.get('end', '18:25')
        
        classified_trade.classification_audit.append({
            'step': 'after_hours_classification',
            'dt_qty': classified_trade.day_trade_qty,
            'swing_qty': classified_trade.swing_trade_qty,
            'classification_override': classification_override,
            'time_window': f"{start_time}-{end_time}",
            'reason': f"After-hours trade ({date.strftime('%H:%M:%S')}) classified as {trade_type}: "
                     f"{quantity} {ticker} @ R$ {price:.2f} (window: {start_time}-{end_time})"
        })
        
        return classified_trade
    
    def _resolve_trade_type(self, ticker: str, trade_date: datetime, 
                           trade_type: str, action: str) -> str:
        """
        Resolve the actual trade type, handling 'auto' detection.
        
        Args:
            ticker: Trading asset identifier
            trade_date: Date of the trade
            trade_type: Requested trade type ('day_trade', 'swing_trade', or 'auto')
            action: Trade action ('BUY' or 'SELL')
            
        Returns:
            Resolved trade type ('day_trade' or 'swing_trade')
        """
        if trade_type != 'auto':
            return trade_type
        
        # For buy operations, we can't determine if it's a day trade yet
        # Default to swing trade and let the sell operation determine the type
        if action == 'BUY':
            return 'swing_trade'
        
        # For sell operations, detect if it's a day trade
        if action == 'SELL':
            is_day_trade = self._detect_day_trade_simplified(ticker, trade_date)
            detected_type = 'day_trade' if is_day_trade else 'swing_trade'
            
            logger.info(f"Auto-detected trade type for {ticker} sell on {trade_date.date()}: {detected_type}")
            return detected_type
        
        return 'swing_trade'  # Default fallback
    
    def _calculate_trade_costs(self, trade_value: float, trade_type: str) -> Dict[str, float]:
        """
        Calculate comprehensive trade costs using Transaction Cost Analyzer.
        
        Args:
            trade_value: Total trade value
            trade_type: Type of trade
            
        Returns:
            Dict containing all cost components
        """
        # Use TCA module for cost calculation
        cost_breakdown = self.tca.calculate_costs(
            order_value=trade_value,
            is_buy=True,  # Will be adjusted in buy/sell methods
            trade_type=trade_type
        )
        
        return {
            'emolument': cost_breakdown.emolument,
            'settlement_fee': cost_breakdown.settlement_fee,
            'brokerage_fee': cost_breakdown.brokerage_fee,
            'iss_fee': cost_breakdown.iss_fee,
            'total_costs': cost_breakdown.total_costs,
            'min_brokerage_applied': cost_breakdown.min_brokerage_applied,
            'cost_percentage': cost_breakdown.cost_percentage
        }
    
    def _calculate_taxes(self, profit: float, trade_type: str, ticker: str = None, 
                        gross_sales: float = 0.0, trade_date: datetime = None,
                        gross_profit: float = None, asset_type: str = AssetType.STOCK.value) -> Dict[str, float]:
        """
        DEPRECATED: Calculate Brazilian taxes for individual taxpayers.
        
        WARNING: This method calculates per-trade taxes which may lead to double application
        of loss carryforward when used in conjunction with monthly tax calculations.
        
        For proper Brazilian tax compliance, use calculate_monthly_tax_liability() instead.
        This method is kept for backward compatibility but should be avoided.
        
        Brazilian PF Tax Rules (July 2025):
        - Swing Trade: 15% on monthly net profit, IRRF 0.005% on each sale (credit only)
        - Day Trade: 20% on monthly net profit, IRRF 1% on daily net profit per asset (credit only)
        - Swing exemption: if total sales ≤ R$ 20,000/month, profit is exempt and doesn't consume losses
        - IRRF is NOT deducted from tax base - it's a credit against final tax liability
        - Capital gains tax is calculated on monthly aggregated profit, not per trade
        
        Args:
            profit: Net trade profit (after costs, before loss carryforward) - used for tax calculation
            trade_type: Type of trade ('swing_trade' or 'day_trade')
            ticker: Trading asset identifier (optional)
            gross_sales: Total sales amount for this trade (for IRRF calculation)
            trade_date: Date of trade (for month reference)
            gross_profit: Gross trade profit (before costs) - used for IRRF calculation on day trades
            asset_type: Asset type for exemption filtering (STOCK, ETF, FII, etc.)
            
        Returns:
            Dict containing tax components for this trade
        """
        import warnings
        warnings.warn(
            "DEPRECATED: _calculate_taxes() may cause double application of loss carryforward. "
            "Use calculate_monthly_tax_liability() for proper Brazilian tax compliance.",
            DeprecationWarning,
            stacklevel=2
        )
        taxes = self.config['taxes']
        
        # Convert trade_type to modality
        modality = "DAY" if trade_type == 'day_trade' else "SWING"
        
        # Get month reference for exemption calculation
        month_ref = trade_date.date() if trade_date else datetime.now().date()
        
        # Get monthly aggregated data for proper tax calculation
        if trade_type == 'swing_trade':
            # For swing trades, use STOCK sales only for exemption calculation
            monthly_sales = self.loss_manager.get_monthly_swing_sales_by_asset_type(month_ref, asset_type)
            monthly_profits = self.loss_manager.get_monthly_swing_profits(month_ref)
        else:  # day_trade
            monthly_sales = 0.0  # Day trades don't use sales for exemption
            monthly_profits = self.loss_manager.get_monthly_day_profits(month_ref)
        
        # Apply loss carryforward with new interface
        taxable_profit, audit_log = self.loss_manager.calculate_taxable_amount(
            gross_profit=profit,
            modality=modality,
            gross_sales=monthly_sales,  # Use monthly aggregated sales filtered by asset_type
            month_ref=month_ref
        )
        
        # Calculate IRRF withholding for this specific trade (this is a credit, not a deduction)
        if trade_type == 'day_trade':
            # Day trade: IRRF 1% on daily net profit per asset (if positive)
            # Brazilian law: IRRF is calculated on consolidated daily profit per asset
            # For individual trades, we return 0 - IRRF will be calculated at day/month end
            # This prevents double-counting and ensures proper consolidation
            irrf_withholding = 0.0  # Will be calculated at day/month end consolidation
        else:
            # Swing trade: IRRF 0.005% on each sale (regardless of profit/loss)
            # Brazilian law: IRRF is withheld on each sale, regardless of exemption
            irrf_withholding = gross_sales * taxes['irrf_swing_rate']  # 0.005%
        
        # For capital gains tax calculation, we need to use monthly aggregated data
        # This is just for this trade's contribution to monthly tax
        if trade_type == 'day_trade':
            capital_gains_rate = taxes['day_trade']  # 20%
        else:
            capital_gains_rate = taxes['swing_trade']  # 15%
        
        # Calculate this trade's contribution to monthly capital gains tax
        # Note: This is simplified - in reality, capital gains tax is calculated monthly
        capital_gains_tax = taxable_profit * capital_gains_rate
        
        # Calculate final tax liability for this trade (capital gains tax - IRRF credit)
        final_tax_liability = max(0.0, capital_gains_tax - irrf_withholding)
        
        # Total taxes for this trade = final tax liability + IRRF withholding
        total_taxes = final_tax_liability + irrf_withholding
        
        return {
            'capital_gains_tax': capital_gains_tax,
            'irrf_withholding': irrf_withholding,
            'total_taxes': total_taxes,
            'taxable_profit': taxable_profit,
            'irrf_credit': irrf_withholding,  # IRRF acts as a credit
            'final_tax_liability': final_tax_liability,  # Amount due to government
            'audit_log': audit_log,
            'monthly_aggregated': {
                'monthly_sales': monthly_sales,
                'monthly_profits': monthly_profits,
                'month_ref': month_ref.isoformat()
            }
        }
    
    def calculate_monthly_tax_liability(self, month_ref: date, trade_type: str = None) -> Dict[str, float]:
        """
        Calculate monthly tax liability according to Brazilian tax rules.
        
        Brazilian PF Tax Rules (July 2025):
        - Swing Trade: 15% on monthly net profit, IRRF 0.005% on each sale (credit only)
        - Day Trade: 20% on monthly net profit, IRRF 1% on daily net profit per asset (credit only)
        - Final DARF = Monthly capital gains tax - Total IRRF credits
        
        Args:
            month_ref: Reference month for tax calculation
            trade_type: Optional filter for specific trade type
            
        Returns:
            Dict containing monthly tax components
        """
        taxes = self.config['taxes']
        
        # Get monthly aggregated data
        if trade_type == 'swing_trade' or trade_type is None:
            # For swing trades, use exempt-eligible sales only for exemption calculation
            # Centralized logic: only STOCK assets are exempt-eligible
            monthly_swing_sales = self.loss_manager.get_monthly_swing_sales_by_asset_type(month_ref, AssetType.STOCK.value)
            monthly_swing_profits = self.loss_manager.get_monthly_swing_profits(month_ref)
        else:
            monthly_swing_sales = 0.0
            monthly_swing_profits = 0.0
            
        if trade_type == 'day_trade' or trade_type is None:
            monthly_day_profits = self.loss_manager.get_monthly_day_profits(month_ref)
        else:
            monthly_day_profits = 0.0
        
        # Calculate taxable profits after loss carryforward
        # Note: R$ 20,000 exemption is automatically applied in loss_manager
        # If monthly sales ≤ R$ 20,000, taxable_profit = 0 (no capital gains tax)
        # But IRRF is still withheld on each sale regardless of exemption
        swing_taxable_profit, swing_audit = self.loss_manager.calculate_taxable_amount(
            gross_profit=monthly_swing_profits,
            modality="SWING",
            gross_sales=monthly_swing_sales,  # STOCK sales only for exemption
            month_ref=month_ref
        )
        
        day_taxable_profit, day_audit = self.loss_manager.calculate_taxable_amount(
            gross_profit=monthly_day_profits,
            modality="DAY",
            gross_sales=0.0,  # Day trades don't use sales for exemption
            month_ref=month_ref
        )
        
        # Calculate capital gains taxes
        swing_capital_gains_tax = swing_taxable_profit * taxes['swing_trade']  # 15%
        day_capital_gains_tax = day_taxable_profit * taxes['day_trade']  # 20%
        total_capital_gains_tax = swing_capital_gains_tax + day_capital_gains_tax
        
        # Calculate total IRRF credits for the month
        # Swing trade IRRF: 0.005% on total monthly sales (withheld on each sale)
        # Note: IRRF is withheld regardless of R$ 20,000 exemption
        swing_irrf_credit = monthly_swing_sales * taxes['irrf_swing_rate']  # 0.005%
        
        # Day trade IRRF: 1% on daily net profit per asset (Brazilian law compliance)
        # Note: R$ 1.00 minimum rule does NOT apply to day trades
        day_irrf_credit = self.loss_manager.calculate_day_trade_irrf(month_ref)
        
        # Apply R$ 1.00 minimum withholding rule ONLY to swing trades
        # Brazilian law: R$ 1.00 minimum rule does NOT apply to day trades
        swing_irrf_waived = False
        if swing_irrf_credit <= 1.00 and swing_irrf_credit > 0:
            logger.info(f"Swing trade IRRF withholding waived: R$ {swing_irrf_credit:.2f} ≤ R$ 1.00")
            swing_irrf_credit = 0.0
            swing_irrf_waived = True
        
        total_irrf_credit = swing_irrf_credit + day_irrf_credit
        
        # Calculate final DARF liability
        final_darf_liability = max(0.0, total_capital_gains_tax - total_irrf_credit)
        
        return {
            'month_ref': month_ref.isoformat(),
            'swing_trade': {
                'monthly_sales': monthly_swing_sales,
                'monthly_profits': monthly_swing_profits,
                'taxable_profit': swing_taxable_profit,
                'capital_gains_tax': swing_capital_gains_tax,
                'irrf_credit': swing_irrf_credit,
                'irrf_withholding_waived': swing_irrf_waived,
                'audit_log': swing_audit
            },
            'day_trade': {
                'monthly_profits': monthly_day_profits,
                'taxable_profit': day_taxable_profit,
                'capital_gains_tax': day_capital_gains_tax,
                'irrf_credit': day_irrf_credit,
                'irrf_withholding_waived': False,  # R$ 1.00 rule does NOT apply to day trades
                'audit_log': day_audit
            },
            'total': {
                'capital_gains_tax': total_capital_gains_tax,
                'irrf_credit': total_irrf_credit,
                'irrf_withholding_waived': swing_irrf_waived,  # Only swing trades can have waiver
                'final_darf_liability': final_darf_liability,
                'total_taxes_paid': final_darf_liability + total_irrf_credit
            }
        }
    
    def buy(self, ticker: str, quantity: int, price: float, 
            trade_date: datetime, trade_type: str = "swing_trade",
            trade_id: Optional[str] = None, description: str = "", 
            asset_type: str = AssetType.STOCK.value) -> bool:
        """
        Execute buy order with comprehensive validation and tracking.
        
        Args:
            ticker: Trading asset identifier
            quantity: Number of shares to buy
            price: Price per share
            trade_date: Date of trade
            trade_type: Type of trade ('day_trade', 'swing_trade', or 'auto')
            trade_id: Optional trade identifier
            description: Trade description
            asset_type: Asset type for exemption filtering (STOCK, ETF, FII, etc.)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self._validate_trade_inputs(ticker, quantity, price, trade_date, trade_type)
            
            # Validate and normalize price/quantity using Brazilian market conventions
            market_config = self.config['market']
            allow_fractional = market_config.get('allow_fractional_lots', True)
            
            validation = self.market_utils.validate_order(
                price=price,
                quantity=quantity,
                order_type=OrderType.MARKET,
                allow_fractional=allow_fractional
            )
            
            if not validation.is_valid:
                for message in validation.validation_messages:
                    logger.warning(f"Buy order validation failed: {message}")
                return False
            
            # Use normalized values
            normalized_price = validation.normalized_price
            normalized_quantity = validation.normalized_quantity
            
            # Log validation results
            if validation.validation_messages:
                for message in validation.validation_messages:
                    logger.debug(f"Buy order validation: {message}")
            
            # Resolve trade type (handle auto-detection)
            resolved_trade_type = self._resolve_trade_type(ticker, trade_date, trade_type, 'BUY')
            
            trade_value = normalized_quantity * normalized_price
            
            # Use TCA for buy order costs
            cost_breakdown = self.tca.calculate_costs(
                order_value=trade_value,
                is_buy=True,
                trade_type=resolved_trade_type
            )
            costs = {
                'emolument': cost_breakdown.emolument,
                'settlement_fee': cost_breakdown.settlement_fee,
                'brokerage_fee': cost_breakdown.brokerage_fee,
                'iss_fee': cost_breakdown.iss_fee,
                'total_costs': cost_breakdown.total_costs,
                'min_brokerage_applied': cost_breakdown.min_brokerage_applied,
                'cost_percentage': cost_breakdown.cost_percentage
            }
            
            total_cost = trade_value + costs['total_costs']
            
            # Check available cash
            available_cash = self.settlement_manager.get_available_cash(trade_date.date())
            if available_cash < total_cost:
                logger.warning(f"Insufficient cash for {ticker} buy: "
                             f"need R$ {total_cost:,.2f}, have R$ {available_cash:,.2f}")
                return False
            
            # Execute trade
            if ticker in self.positions:
                # Update existing position
                pos = self.positions[ticker]
                total_quantity = pos.quantity + normalized_quantity
                total_cost_basis = (pos.quantity * pos.avg_price) + trade_value
                pos.quantity = total_quantity
                pos.avg_price = total_cost_basis / total_quantity
                pos.current_price = normalized_price
                pos.last_update = trade_date
            else:
                # Create new position
                self.positions[ticker] = Position(
                    ticker=ticker,
                    quantity=normalized_quantity,
                    avg_price=normalized_price,
                    current_price=normalized_price,
                    last_update=trade_date,
                    trade_type=resolved_trade_type,
                    position_id=trade_id,
                    description=description
                )
            
            # Update cash and settlement - CORRIGIDO: não debitar caixa imediatamente
            # O dinheiro só sai no T+2 através do settlement_manager
            # self.cash -= total_cost  # ← REMOVIDO: não debitar imediatamente
            
            self.settlement_manager.schedule_trade(
                trade_date=trade_date,
                amount=total_cost,
                trade_type='BUY',
                ticker=ticker,
                trade_id=trade_id,
                description=f"Buy {quantity} {ticker} @ R$ {price:.2f}"
            )
            
            # Update tracking
            self.total_trades += 1
            self.total_commission += costs['total_costs']
            
            # Record trade
            trade_record = {
                'date': trade_date,
                'ticker': ticker,
                'action': 'BUY',
                'quantity': normalized_quantity,
                'price': normalized_price,
                'value': trade_value,
                'costs': costs,
                'trade_type': resolved_trade_type,
                'trade_id': trade_id,
                'description': description,
                'asset_type': asset_type,
                'lot_type': validation.lot_type.value,
                'is_fractional': validation.is_fractional,
                'original_quantity': quantity,
                'original_price': price
            }
            self.trade_history.append(trade_record)
            
            logger.info(f"Buy executed: {normalized_quantity} {ticker} @ R$ {normalized_price:.2f} "
                       f"(costs: R$ {costs['total_costs']:.2f}, lot_type: {validation.lot_type.value})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing buy for {ticker}: {str(e)}")
            return False
    
    def sell(self, ticker: str, quantity: int, price: float, 
             trade_date: datetime, trade_type: str = "swing_trade",
             trade_id: Optional[str] = None, description: str = "", 
             asset_type: str = AssetType.STOCK.value) -> bool:
        """
        Execute sell order with comprehensive validation and tracking.
        
        NEW TAX FLOW (Brazilian Compliance):
        1. Record trade result in loss_manager for carryforward tracking
        2. Calculate only IRRF withholding for this specific trade
        3. Store raw data (gross profit, costs, IRRF) for monthly consolidation
        4. Capital gains tax is calculated monthly via calculate_monthly_tax_liability()
        5. This prevents double application of loss carryforward
        
        Args:
            ticker: Trading asset identifier
            quantity: Number of shares to sell
            price: Price per share
            trade_date: Date of trade
            trade_type: Type of trade ('day_trade', 'swing_trade', or 'auto')
            trade_id: Optional trade identifier
            description: Trade description
            asset_type: Asset type for exemption filtering (STOCK, ETF, FII, etc.)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self._validate_trade_inputs(ticker, quantity, price, trade_date, trade_type)
            
            # Validate and normalize price/quantity using Brazilian market conventions
            market_config = self.config['market']
            allow_fractional = market_config.get('allow_fractional_lots', True)
            
            validation = self.market_utils.validate_order(
                price=price,
                quantity=quantity,
                order_type=OrderType.MARKET,
                allow_fractional=allow_fractional
            )
            
            if not validation.is_valid:
                for message in validation.validation_messages:
                    logger.warning(f"Sell order validation failed: {message}")
                return False
            
            # Use normalized values
            normalized_price = validation.normalized_price
            normalized_quantity = validation.normalized_quantity
            
            # Log validation results
            if validation.validation_messages:
                for message in validation.validation_messages:
                    logger.debug(f"Sell order validation: {message}")
            
            # Resolve trade type (handle auto-detection)
            resolved_trade_type = self._resolve_trade_type(ticker, trade_date, trade_type, 'SELL')
            
            # Check position
            if ticker not in self.positions:
                logger.warning(f"No position in {ticker} to sell")
                return False
            
            position = self.positions[ticker]
            if position.quantity < normalized_quantity:
                logger.warning(f"Insufficient shares in {ticker}: "
                             f"have {position.quantity}, trying to sell {normalized_quantity}")
                return False
            
            # Calculate trade details
            trade_value = normalized_quantity * normalized_price
            
            # Use TCA for sell order costs
            cost_breakdown = self.tca.calculate_costs(
                order_value=trade_value,
                is_buy=False,
                trade_type=resolved_trade_type
            )
            costs = {
                'emolument': cost_breakdown.emolument,
                'settlement_fee': cost_breakdown.settlement_fee,
                'brokerage_fee': cost_breakdown.brokerage_fee,
                'iss_fee': cost_breakdown.iss_fee,
                'total_costs': cost_breakdown.total_costs,
                'min_brokerage_applied': cost_breakdown.min_brokerage_applied,
                'cost_percentage': cost_breakdown.cost_percentage
            }
            
            # Calculate profit/loss using normalized quantities
            cost_basis = normalized_quantity * position.avg_price
            gross_profit = trade_value - cost_basis
            net_profit = gross_profit - costs['total_costs']
            
            # Convert trade_type to modality for loss manager
            modality = "DAY" if resolved_trade_type == 'day_trade' else "SWING"
            
            # Record loss/profit for carryforward
            self.loss_manager.record_trade_result(
                ticker=ticker,
                trade_profit=net_profit,
                trade_date=trade_date,
                modality=modality,
                trade_id=trade_id,
                description=description,
                gross_sales=trade_value,  # Pass gross sales for monthly tracking
                asset_type=asset_type
            )
            
            # Calculate IRRF withholding for this specific trade (this is a credit, not a deduction)
            # Brazilian law: IRRF is withheld on each sale, regardless of exemption
            if resolved_trade_type == 'day_trade':
                # Day trade: IRRF 1% on daily net profit per asset (if positive)
                # Brazilian law: IRRF is calculated on consolidated daily profit per asset
                # For individual trades, we return 0 - IRRF will be calculated at day/month end
                # This prevents double-counting and ensures proper consolidation
                irrf_withholding = 0.0  # Will be calculated at day/month end consolidation
            else:
                # Swing trade: IRRF 0.005% on each sale (regardless of profit/loss)
                # Brazilian law: IRRF is withheld on each sale, regardless of exemption
                irrf_withholding = trade_value * self.config['taxes']['irrf_swing_rate']  # 0.005%
            
            # Apply R$ 1.00 minimum withholding rule ONLY to swing trades
            # Brazilian law: R$ 1.00 minimum rule does NOT apply to day trades
            if resolved_trade_type == 'swing_trade' and irrf_withholding <= 1.00 and irrf_withholding > 0:
                logger.debug(f"Swing trade IRRF withholding waived: R$ {irrf_withholding:.2f} ≤ R$ 1.00")
                irrf_withholding = 0.0
            
            # For now, we don't calculate capital gains tax here - it will be calculated monthly
            # This prevents double application of loss carryforward
            capital_gains_tax = 0.0  # Will be calculated at month end
            
            # Total taxes for this trade = IRRF withholding only (capital gains tax calculated monthly)
            total_taxes = irrf_withholding
            
            # Final profit for this trade (net profit - IRRF withholding only)
            final_profit = net_profit - total_taxes
            
            # Update position
            position.quantity -= normalized_quantity
            position.current_price = normalized_price
            position.last_update = trade_date
            
            # Remove position if empty
            if position.quantity == 0:
                del self.positions[ticker]
            
            # Update cash and settlement
            cash_received = trade_value - costs['total_costs'] - total_taxes
            # self.cash += cash_received  # ← REMOVIDO: não creditar imediatamente
            
            self.settlement_manager.schedule_trade(
                trade_date=trade_date,
                amount=cash_received,
                trade_type='SELL',
                ticker=ticker,
                trade_id=trade_id,
                description=f"Sell {normalized_quantity} {ticker} @ R$ {normalized_price:.2f}"
            )
            
            # Update tracking
            self.total_trades += 1
            self.total_commission += costs['total_costs']
            self.total_taxes += total_taxes  # Only IRRF withholding for now
            
            if final_profit > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Update daily P&L
            date_key = trade_date.date().isoformat()
            if date_key not in self.daily_pnl:
                self.daily_pnl[date_key] = 0.0
            self.daily_pnl[date_key] += final_profit
            
            # Update day trade P&L
            if resolved_trade_type == 'day_trade':
                self.day_trade_pnl += final_profit
            
            # Record trade with raw data for monthly consolidation
            trade_record = {
                'date': trade_date,
                'ticker': ticker,
                'action': 'SELL',
                'quantity': normalized_quantity,
                'price': normalized_price,
                'value': trade_value,
                'costs': costs,
                'gross_profit': gross_profit,
                'net_profit': net_profit,
                'trade_type': resolved_trade_type,
                'trade_id': trade_id,
                'description': description,
                'asset_type': asset_type,
                'lot_type': validation.lot_type.value,
                'is_fractional': validation.is_fractional,
                'original_quantity': quantity,
                'original_price': price,
                # Raw tax data for monthly consolidation
                'irrf_withholding': irrf_withholding,
                'capital_gains_tax': capital_gains_tax,  # Will be 0.0, calculated monthly
                'total_taxes': total_taxes,  # Only IRRF withholding for now
                'final_profit': final_profit  # Net profit - IRRF withholding only
            }
            self.trade_history.append(trade_record)
            
            logger.info(f"Sell executed: {normalized_quantity} {ticker} @ R$ {normalized_price:.2f} "
                       f"(profit: R$ {final_profit:.2f}, IRRF: R$ {irrf_withholding:.2f}, lot_type: {validation.lot_type.value})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing sell for {ticker}: {str(e)}")
            return False
    
    def update_prices(self, price_updates: Dict[str, float], 
                     update_date: datetime) -> None:
        """
        Update position prices and calculate unrealized P&L.
        
        Args:
            price_updates: Dict of {ticker: price} updates
            update_date: Date of price update
        """
        for ticker, price in price_updates.items():
            if ticker in self.positions:
                self.positions[ticker].current_price = price
                self.positions[ticker].last_update = update_date
        
        # Recalculate total value
        self._update_total_value()
    
    def _update_total_value(self) -> None:
        """Update total portfolio value."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        # Use settlement manager's total_cash instead of local self.cash
        # since cash is now managed by the settlement manager
        self.total_value = self.settlement_manager.total_cash + positions_value
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary."""
        self._update_total_value()
        
        # Calculate current month's tax liability
        current_month = datetime.now().date().replace(day=1)
        monthly_tax_liability = self.calculate_monthly_tax_liability(current_month)
        
        return {
            'total_value': self.total_value,
            'cash': self.settlement_manager.total_cash,  # Use settlement manager's total_cash
            'positions_value': self.total_value - self.settlement_manager.total_cash,
            'num_positions': len(self.positions),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
            'total_commission': self.total_commission,
            'total_taxes': self.total_taxes,  # Only IRRF withholding for now
            'day_trade_pnl': self.day_trade_pnl,
            'loss_carryforward_balance': self.loss_manager.get_total_loss_balance(),
            'settlement_summary': self.settlement_manager.get_settlement_summary(),
            'monthly_tax_liability': monthly_tax_liability
        }
    
    def get_position_summary(self) -> List[Dict]:
        """Get detailed position summary."""
        self._update_total_value()
        
        positions = []
        for ticker, pos in self.positions.items():
            positions.append({
                'ticker': ticker,
                'quantity': pos.quantity,
                'avg_price': pos.avg_price,
                'current_price': pos.current_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                'trade_type': pos.trade_type,
                'last_update': pos.last_update.isoformat()
            })
        
        return positions
    
    def process_monthly_close(self, month_ref: date) -> Dict[str, Any]:
        """
        Process monthly close and calculate consolidated tax liability.
        
        This method should be called at the end of each month to:
        1. Calculate final monthly tax liability
        2. Apply loss carryforward correctly
        3. Generate tax reports
        4. Reset monthly tracking for next month
        
        Args:
            month_ref: Reference month for closing (e.g., 2024-01-01 for January 2024)
            
        Returns:
            Dict containing monthly close results and tax liability
        """
        logger.info(f"Processing monthly close for {month_ref.strftime('%Y-%m')}")
        
        # Calculate final monthly tax liability
        monthly_tax_liability = self.calculate_monthly_tax_liability(month_ref)
        
        # Calculate total IRRF withheld during the month
        total_irrf_withheld = 0.0
        for trade in self.trade_history:
            trade_date = trade['date']
            if isinstance(trade_date, str):
                trade_date = datetime.fromisoformat(trade_date.replace('Z', '+00:00'))
            
            if (trade_date.date().year == month_ref.year and 
                trade_date.date().month == month_ref.month and
                trade['action'] == 'SELL'):
                total_irrf_withheld += trade.get('irrf_withholding', 0.0)
        
        # Calculate final tax payment/refund
        final_darf_liability = monthly_tax_liability['total']['final_darf_liability']
        irrf_credit = monthly_tax_liability['total']['irrf_credit']
        
        # If IRRF credit > final DARF liability, there's a refund
        # If final DARF liability > IRRF credit, there's additional payment due
        additional_payment = max(0.0, final_darf_liability - irrf_credit)
        refund_amount = max(0.0, irrf_credit - final_darf_liability)
        
        monthly_close_result = {
            'month_ref': month_ref.isoformat(),
            'monthly_tax_liability': monthly_tax_liability,
            'total_irrf_withheld': total_irrf_withheld,
            'additional_payment_due': additional_payment,
            'refund_amount': refund_amount,
            'final_tax_status': 'payment_due' if additional_payment > 0 else 'refund_due' if refund_amount > 0 else 'balanced'
        }
        
        logger.info(f"Monthly close completed for {month_ref.strftime('%Y-%m')}: "
                   f"DARF: R$ {final_darf_liability:,.2f}, "
                   f"IRRF Credit: R$ {irrf_credit:,.2f}, "
                   f"Additional Payment: R$ {additional_payment:,.2f}")
        
        return monthly_close_result
    
    def get_monthly_tax_summary(self, month_ref: date) -> Dict[str, Any]:
        """
        Get detailed monthly tax summary for reporting purposes.
        
        Args:
            month_ref: Reference month for tax summary
            
        Returns:
            Dict containing detailed monthly tax information
        """
        # Get monthly tax liability
        monthly_tax = self.calculate_monthly_tax_liability(month_ref)
        
        # Get trades for this month
        month_trades = []
        total_irrf_withheld = 0.0
        total_gross_profit = 0.0
        total_net_profit = 0.0
        total_costs = 0.0
        
        for trade in self.trade_history:
            trade_date = trade['date']
            if isinstance(trade_date, str):
                trade_date = datetime.fromisoformat(trade_date.replace('Z', '+00:00'))
            
            if (trade_date.date().year == month_ref.year and 
                trade_date.date().month == month_ref.month):
                month_trades.append(trade)
                
                if trade['action'] == 'SELL':
                    total_irrf_withheld += trade.get('irrf_withholding', 0.0)
                    total_gross_profit += trade.get('gross_profit', 0.0)
                    total_net_profit += trade.get('net_profit', 0.0)
                    total_costs += trade.get('costs', {}).get('total_costs', 0.0)
        
        return {
            'month_ref': month_ref.isoformat(),
            'monthly_tax_liability': monthly_tax,
            'trade_summary': {
                'total_trades': len(month_trades),
                'sell_trades': len([t for t in month_trades if t['action'] == 'SELL']),
                'buy_trades': len([t for t in month_trades if t['action'] == 'BUY']),
                'total_irrf_withheld': total_irrf_withheld,
                'total_gross_profit': total_gross_profit,
                'total_net_profit': total_net_profit,
                'total_costs': total_costs
            },
            'tax_summary': {
                'capital_gains_tax_swing': monthly_tax['swing_trade']['capital_gains_tax'],
                'capital_gains_tax_day': monthly_tax['day_trade']['capital_gains_tax'],
                'total_capital_gains_tax': monthly_tax['total']['capital_gains_tax'],
                'total_irrf_credit': monthly_tax['total']['irrf_credit'],
                'final_darf_liability': monthly_tax['total']['final_darf_liability'],
                'additional_payment_due': max(0.0, monthly_tax['total']['final_darf_liability'] - monthly_tax['total']['irrf_credit']),
                'refund_amount': max(0.0, monthly_tax['total']['irrf_credit'] - monthly_tax['total']['final_darf_liability'])
            }
        }
    
    def reset_monthly_tracking(self, current_month: int) -> None:
        """Reset monthly tracking while preserving loss carryforward."""
        self.daily_pnl.clear()
        self.day_trade_pnl = 0.0
        self.loss_manager.reset_monthly_tracking(current_month)
        
        logger.info(f"Monthly tracking reset for month {current_month}")
    
    def export_audit_trails(self, base_path: str = "audit_trails") -> None:
        """
        Export comprehensive audit trails for regulatory compliance.
        
        Args:
            base_path: Base directory for audit trail files
        """
        import os
        os.makedirs(base_path, exist_ok=True)
        
        # Export loss carryforward audit trail
        self.loss_manager.export_audit_trail(f"{base_path}/loss_carryforward_audit.json")
        
        # Export settlement audit trail
        self.settlement_manager.export_audit_trail(f"{base_path}/settlement_audit.json")
        
        # Export portfolio audit trail
        portfolio_audit = {
            'portfolio_summary': self.get_portfolio_summary(),
            'positions': self.get_position_summary(),
            'trade_history': self.trade_history,
            'daily_pnl': self.daily_pnl,
            'export_date': datetime.now().isoformat()
        }
        
        with open(f"{base_path}/portfolio_audit.json", 'w') as f:
            import json
            json.dump(portfolio_audit, f, indent=2)
        
        logger.info(f"Audit trails exported to {base_path}/")
    
    def print_summary(self):
        """Print comprehensive portfolio summary."""
        summary = self.get_portfolio_summary()
        positions = self.get_position_summary()
        
        print("\n" + "="*80)
        print("ENHANCED PORTFOLIO SUMMARY")
        print("="*80)
        print(f"Total Portfolio Value: R$ {summary['total_value']:,.2f}")
        print(f"Cash: R$ {summary['cash']:,.2f}")
        print(f"Positions Value: R$ {summary['positions_value']:,.2f}")
        print(f"Number of Positions: {summary['num_positions']}")
        print(f"Total Trades: {summary['total_trades']}")
        print(f"Win Rate: {summary['win_rate']:.1f}%")
        print(f"Total Commission: R$ {summary['total_commission']:,.2f}")
        print(f"Total IRRF Withholding: R$ {summary['total_taxes']:,.2f}")
        print(f"Day Trade P&L: R$ {summary['day_trade_pnl']:,.2f}")
        print(f"Loss Carryforward Balance: R$ {summary['loss_carryforward_balance']:,.2f}")
        
        # Print monthly tax liability
        monthly_tax = summary['monthly_tax_liability']
        print(f"\nMonthly Tax Liability (Current Month):")
        print(f"  Swing Trade Capital Gains Tax: R$ {monthly_tax['swing_trade']['capital_gains_tax']:,.2f}")
        print(f"  Day Trade Capital Gains Tax: R$ {monthly_tax['day_trade']['capital_gains_tax']:,.2f}")
        print(f"  Total IRRF Credits: R$ {monthly_tax['total']['irrf_credit']:,.2f}")
        print(f"  Final DARF Liability: R$ {monthly_tax['total']['final_darf_liability']:,.2f}")
        
        if positions:
            print("\nCurrent Positions:")
            for pos in positions:
                pnl_color = "🟢" if pos['unrealized_pnl'] >= 0 else "🔴"
                print(f"  {pos['ticker']}: {pos['quantity']} shares @ R$ {pos['avg_price']:.2f} "
                      f"(current: R$ {pos['current_price']:.2f}) "
                      f"{pnl_color} R$ {pos['unrealized_pnl']:,.2f} ({pos['unrealized_pnl_pct']:+.1f}%)")
        
        print("="*80)
        
        # Print enhanced manager summaries
        self.loss_manager.print_summary()
        self.settlement_manager.print_summary()

    def get_portfolio_value(self) -> float:
        """
        Return the current total portfolio value (cash + positions' market value).
        """
        total_positions_value = sum(
            pos.market_value for pos in self.positions.values()
        )
        # Use settlement manager's total_cash instead of local self.cash
        # since cash is now managed by the settlement manager
        return self.settlement_manager.total_cash + total_positions_value


def main():
    """Example usage demonstrating enhanced portfolio functionality."""
    
    # Initialize enhanced portfolio
    portfolio = EnhancedPortfolio()
    
    print("=== Enhanced Portfolio Test ===")
    
    # Scenario: Multiple trades with comprehensive tracking
    base_date = datetime.now()
    
    print("\n--- Day 1: Buy VALE3 ---")
    portfolio.buy("VALE3", 100, 50.0, base_date, "swing_trade", 
                  trade_id="VALE3_001", description="Initial VALE3 position")
    
    print("\n--- Day 2: Buy PETR4 ---")
    portfolio.buy("PETR4", 50, 30.0, base_date + timedelta(days=1), "day_trade",
                  trade_id="PETR4_001", description="Day trade PETR4")
    
    print("\n--- Day 3: Sell PETR4 at profit ---")
    portfolio.sell("PETR4", 50, 32.0, base_date + timedelta(days=2), "day_trade",
                   trade_id="PETR4_002", description="Close PETR4 day trade")
    
    print("\n--- Day 4: Sell VALE3 at loss ---")
    portfolio.sell("VALE3", 100, 48.0, base_date + timedelta(days=3), "swing_trade",
                   trade_id="VALE3_002", description="Close VALE3 position")
    
    print("\n--- Day 5: Buy ITUB4 with loss carryforward ---")
    portfolio.buy("ITUB4", 200, 25.0, base_date + timedelta(days=4), "swing_trade",
                  trade_id="ITUB4_001", description="New position with loss carryforward")
    
    print("\n--- Day 6: Sell ITUB4 at profit (loss carryforward applied) ---")
    portfolio.sell("ITUB4", 200, 26.0, base_date + timedelta(days=5), "swing_trade",
                   trade_id="ITUB4_002", description="Close ITUB4 with loss offset")
    
    # Print comprehensive summary
    portfolio.print_summary()
    
    # Demonstrate monthly close process
    print("\n=== Monthly Close Process ===")
    current_month = datetime.now().date().replace(day=1)
    monthly_close = portfolio.process_monthly_close(current_month)
    
    print(f"Monthly Close Results for {monthly_close['month_ref']}:")
    print(f"  Total IRRF Withheld: R$ {monthly_close['total_irrf_withheld']:,.2f}")
    print(f"  Additional Payment Due: R$ {monthly_close['additional_payment_due']:,.2f}")
    print(f"  Refund Amount: R$ {monthly_close['refund_amount']:,.2f}")
    print(f"  Final Tax Status: {monthly_close['final_tax_status']}")
    
    # Demonstrate detailed monthly tax summary
    print("\n=== Detailed Monthly Tax Summary ===")
    tax_summary = portfolio.get_monthly_tax_summary(current_month)
    print(f"Trade Summary for {tax_summary['month_ref']}:")
    print(f"  Total Trades: {tax_summary['trade_summary']['total_trades']}")
    print(f"  Sell Trades: {tax_summary['trade_summary']['sell_trades']}")
    print(f"  Buy Trades: {tax_summary['trade_summary']['buy_trades']}")
    print(f"  Total Gross Profit: R$ {tax_summary['trade_summary']['total_gross_profit']:,.2f}")
    print(f"  Total Net Profit: R$ {tax_summary['trade_summary']['total_net_profit']:,.2f}")
    print(f"  Total Costs: R$ {tax_summary['trade_summary']['total_costs']:,.2f}")
    
    print(f"\nTax Summary:")
    print(f"  Swing Trade Capital Gains Tax: R$ {tax_summary['tax_summary']['capital_gains_tax_swing']:,.2f}")
    print(f"  Day Trade Capital Gains Tax: R$ {tax_summary['tax_summary']['capital_gains_tax_day']:,.2f}")
    print(f"  Total Capital Gains Tax: R$ {tax_summary['tax_summary']['total_capital_gains_tax']:,.2f}")
    print(f"  Total IRRF Credit: R$ {tax_summary['tax_summary']['total_irrf_credit']:,.2f}")
    print(f"  Final DARF Liability: R$ {tax_summary['tax_summary']['final_darf_liability']:,.2f}")
    print(f"  Additional Payment Due: R$ {tax_summary['tax_summary']['additional_payment_due']:,.2f}")
    print(f"  Refund Amount: R$ {tax_summary['tax_summary']['refund_amount']:,.2f}")
    
    # Export audit trails
    portfolio.export_audit_trails("enhanced_audit_trails")
    print("\nAudit trails exported to enhanced_audit_trails/")


if __name__ == "__main__":
    main() 