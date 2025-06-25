"""
Enhanced Portfolio Manager for Brazilian Market Backtesting

Advanced portfolio management with comprehensive Brazilian market compliance:
- Enhanced loss carryforward with temporal management and audit trails
- Sophisticated T+2 settlement tracking with business day handling
- Comprehensive error handling and defensive programming
- Performance optimization and regulatory compliance
- Detailed logging and audit trail generation

Compliance: Receita Federal IN RFB 1.585/2015, CVM Resolution 378/2009

Author: Your Name
Date: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
import logging
import yaml
import pytz

from .loss_manager import EnhancedLossCarryforwardManager
from .settlement_manager import AdvancedSettlementManager

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class FeeConfig:
    """
    Configuration for all Brazilian market fees and taxes.
    
    Attributes:
        emolument: B3 negotiation fee (0.005%)
        settlement_day_trade: Settlement fee for day trades (0.018%)
        settlement_swing_trade: Settlement fee for swing trades (0.025%)
        brokerage_fee: Brokerage fee percentage (0% for Modal web/app)
        min_brokerage: Minimum brokerage charge (R$ 0 for Modal)
        iss_rate: ISS tax rate on brokerage (5% statutory max)
        swing_trade_tax: Swing trade tax rate (15%)
        day_trade_tax: Day trade tax rate (20%)
        exemption_limit: Monthly swing trade exemption (R$ 20,000)
        irrf_swing_rate: IRRF withholding rate for swing trades (0.005%)
        irrf_day_rate: IRRF withholding rate for day trades (1%)
    """
    emolument: float = 0.00005
    settlement_day_trade: float = 0.00018
    settlement_swing_trade: float = 0.00025
    brokerage_fee: float = 0.0
    min_brokerage: float = 0.0
    iss_rate: float = 0.05
    swing_trade_tax: float = 0.15
    day_trade_tax: float = 0.20
    exemption_limit: float = 20000.0
    irrf_swing_rate: float = 0.00005  # 0.005% on swing-trade sale value
    irrf_day_rate: float = 0.01       # 1% on day-trade profit


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


class EnhancedPortfolio:
    """
    Advanced portfolio management with comprehensive Brazilian market compliance.
    
    Features:
    - Enhanced loss carryforward with temporal management
    - Sophisticated T+2 settlement tracking
    - Comprehensive error handling and validation
    - Performance optimization and caching
    - Detailed audit trails and regulatory compliance
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
            max_tracking_years=self.config.get('loss_carryforward', {}).get('max_tracking_years', 5),
            timezone=self.config['market']['trading_hours']['timezone']
        )
        
        self.settlement_manager = AdvancedSettlementManager(
            initial_cash=self.config['portfolio']['initial_cash'],
            settlement_days=self.config.get('settlement', {}).get('cycle_days', 2),
            market_timezone=self.config['market']['trading_hours']['timezone']
        )
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.cash: float = self.config['portfolio']['initial_cash']
        self.total_value: float = self.cash
        self.trade_history: List[Dict] = []
        self.daily_pnl: Dict[str, float] = {}  # {date: pnl}
        self.day_trade_pnl: float = 0.0  # Consolidated daily P&L for day trades
        
        # Performance tracking
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.total_commission: float = 0.0
        self.total_taxes: float = 0.0
        
        logger.info(f"Enhanced Portfolio initialized with R$ {self.cash:,.2f} initial capital")
    
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
    
    def _validate_trade_inputs(self, ticker: str, quantity: int, price: float, 
                              trade_date: datetime, trade_type: str) -> None:
        """
        Validate trade inputs with comprehensive error checking.
        
        Args:
            ticker: Trading asset identifier
            quantity: Number of shares
            price: Price per share
            trade_date: Date of trade
            trade_type: Type of trade
            
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
        
        if trade_type not in ['day_trade', 'swing_trade']:
            raise ValueError("Trade type must be 'day_trade' or 'swing_trade'")
    
    def _calculate_trade_costs(self, trade_value: float, trade_type: str) -> Dict[str, float]:
        """
        Calculate comprehensive trade costs with Brazilian market compliance.
        
        Args:
            trade_value: Total trade value
            trade_type: Type of trade
            
        Returns:
            Dict containing all cost components
        """
        costs = self.config['market']['costs']
        
        # B3 fees
        emolument = trade_value * costs['emolument']
        
        # Settlement fees based on trade type
        if trade_type == 'day_trade':
            settlement_fee = trade_value * costs['settlement_day_trade']
        else:
            settlement_fee = trade_value * costs['settlement_swing_trade']
        
        # Modal brokerage (electronic only)
        brokerage_fee = max(costs['min_brokerage'], 
                           trade_value * costs['brokerage_fee'])
        
        # ISS on brokerage
        iss_fee = brokerage_fee * costs['iss_rate']
        
        total_costs = emolument + settlement_fee + brokerage_fee + iss_fee
        
        return {
            'emolument': emolument,
            'settlement_fee': settlement_fee,
            'brokerage_fee': brokerage_fee,
            'iss_fee': iss_fee,
            'total_costs': total_costs
        }
    
    def _calculate_taxes(self, profit: float, trade_type: str) -> Dict[str, float]:
        """
        Calculate Brazilian taxes with comprehensive compliance.
        
        Args:
            profit: Trade profit
            trade_type: Type of trade
            
        Returns:
            Dict containing tax components
        """
        taxes = self.config['taxes']
        
        if profit <= 0:
            return {
                'capital_gains_tax': 0.0,
                'irrf_withholding': 0.0,
                'total_taxes': 0.0
            }
        
        # Capital gains tax
        if trade_type == 'day_trade':
            capital_gains_rate = taxes['day_trade']
            irrf_rate = taxes['irrf_day_rate']
        else:
            capital_gains_rate = taxes['swing_trade']
            irrf_rate = taxes['irrf_swing_rate']
        
        # Apply loss carryforward
        taxable_profit = self.loss_manager.calculate_taxable_amount(
            profit, datetime.now(), None
        )
        
        # Calculate taxes
        capital_gains_tax = taxable_profit * capital_gains_rate
        
        # IRRF withholding
        if trade_type == 'day_trade':
            irrf_withholding = profit * irrf_rate  # On gross profit
        else:
            irrf_withholding = profit * irrf_rate  # On sale value
        
        total_taxes = capital_gains_tax + irrf_withholding
        
        return {
            'capital_gains_tax': capital_gains_tax,
            'irrf_withholding': irrf_withholding,
            'total_taxes': total_taxes,
            'taxable_profit': taxable_profit
        }
    
    def buy(self, ticker: str, quantity: int, price: float, 
            trade_date: datetime, trade_type: str = "swing_trade",
            trade_id: Optional[str] = None, description: str = "") -> bool:
        """
        Execute buy order with comprehensive validation and tracking.
        
        Args:
            ticker: Trading asset identifier
            quantity: Number of shares to buy
            price: Price per share
            trade_date: Date of trade
            trade_type: Type of trade
            trade_id: Optional trade identifier
            description: Trade description
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self._validate_trade_inputs(ticker, quantity, price, trade_date, trade_type)
            
            trade_value = quantity * price
            costs = self._calculate_trade_costs(trade_value, trade_type)
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
                total_quantity = pos.quantity + quantity
                total_cost_basis = (pos.quantity * pos.avg_price) + trade_value
                pos.quantity = total_quantity
                pos.avg_price = total_cost_basis / total_quantity
                pos.current_price = price
                pos.last_update = trade_date
            else:
                # Create new position
                self.positions[ticker] = Position(
                    ticker=ticker,
                    quantity=quantity,
                    avg_price=price,
                    current_price=price,
                    last_update=trade_date,
                    trade_type=trade_type,
                    position_id=trade_id,
                    description=description
                )
            
            # Update cash and settlement
            self.cash -= total_cost
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
                'quantity': quantity,
                'price': price,
                'value': trade_value,
                'costs': costs,
                'trade_type': trade_type,
                'trade_id': trade_id,
                'description': description
            }
            self.trade_history.append(trade_record)
            
            logger.info(f"Buy executed: {quantity} {ticker} @ R$ {price:.2f} "
                       f"(costs: R$ {costs['total_costs']:.2f})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing buy for {ticker}: {str(e)}")
            return False
    
    def sell(self, ticker: str, quantity: int, price: float, 
             trade_date: datetime, trade_type: str = "swing_trade",
             trade_id: Optional[str] = None, description: str = "") -> bool:
        """
        Execute sell order with comprehensive validation and tracking.
        
        Args:
            ticker: Trading asset identifier
            quantity: Number of shares to sell
            price: Price per share
            trade_date: Date of trade
            trade_type: Type of trade
            trade_id: Optional trade identifier
            description: Trade description
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self._validate_trade_inputs(ticker, quantity, price, trade_date, trade_type)
            
            # Check position
            if ticker not in self.positions:
                logger.warning(f"No position in {ticker} to sell")
                return False
            
            position = self.positions[ticker]
            if position.quantity < quantity:
                logger.warning(f"Insufficient shares in {ticker}: "
                             f"have {position.quantity}, trying to sell {quantity}")
                return False
            
            # Calculate trade details
            trade_value = quantity * price
            costs = self._calculate_trade_costs(trade_value, trade_type)
            
            # Calculate profit/loss
            cost_basis = quantity * position.avg_price
            gross_profit = trade_value - cost_basis
            net_profit = gross_profit - costs['total_costs']
            
            # Calculate taxes
            taxes = self._calculate_taxes(net_profit, trade_type)
            final_profit = net_profit - taxes['total_taxes']
            
            # Update position
            position.quantity -= quantity
            position.current_price = price
            position.last_update = trade_date
            
            # Remove position if empty
            if position.quantity == 0:
                del self.positions[ticker]
            
            # Record loss/profit for carryforward
            if net_profit < 0:
                self.loss_manager.record_trade_result(
                    ticker=ticker,
                    trade_profit=net_profit,
                    trade_date=trade_date,
                    trade_type=trade_type,
                    trade_id=trade_id,
                    description=description
                )
            else:
                # Apply loss carryforward and calculate final taxable amount
                taxable_profit = self.loss_manager.record_trade_result(
                    ticker=ticker,
                    trade_profit=net_profit,
                    trade_date=trade_date,
                    trade_type=trade_type,
                    trade_id=trade_id,
                    description=description
                )
                
                # Recalculate taxes on final taxable amount
                if taxable_profit > 0:
                    final_taxes = self._calculate_taxes(taxable_profit, trade_type)
                    final_profit = net_profit - final_taxes['total_taxes']
                    taxes = final_taxes
            
            # Update cash and settlement
            cash_received = trade_value - costs['total_costs'] - taxes['total_taxes']
            self.cash += cash_received
            
            self.settlement_manager.schedule_trade(
                trade_date=trade_date,
                amount=cash_received,
                trade_type='SELL',
                ticker=ticker,
                trade_id=trade_id,
                description=f"Sell {quantity} {ticker} @ R$ {price:.2f}"
            )
            
            # Update tracking
            self.total_trades += 1
            self.total_commission += costs['total_costs']
            self.total_taxes += taxes['total_taxes']
            
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
            if trade_type == 'day_trade':
                self.day_trade_pnl += final_profit
            
            # Record trade
            trade_record = {
                'date': trade_date,
                'ticker': ticker,
                'action': 'SELL',
                'quantity': quantity,
                'price': price,
                'value': trade_value,
                'costs': costs,
                'taxes': taxes,
                'gross_profit': gross_profit,
                'net_profit': net_profit,
                'final_profit': final_profit,
                'trade_type': trade_type,
                'trade_id': trade_id,
                'description': description
            }
            self.trade_history.append(trade_record)
            
            logger.info(f"Sell executed: {quantity} {ticker} @ R$ {price:.2f} "
                       f"(profit: R$ {final_profit:.2f}, taxes: R$ {taxes['total_taxes']:.2f})")
            
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
        self.total_value = self.cash + positions_value
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary."""
        self._update_total_value()
        
        return {
            'total_value': self.total_value,
            'cash': self.cash,
            'positions_value': self.total_value - self.cash,
            'num_positions': len(self.positions),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
            'total_commission': self.total_commission,
            'total_taxes': self.total_taxes,
            'day_trade_pnl': self.day_trade_pnl,
            'loss_carryforward_balance': self.loss_manager.get_total_loss_balance(),
            'settlement_summary': self.settlement_manager.get_settlement_summary()
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
        print(f"Total Taxes: R$ {summary['total_taxes']:,.2f}")
        print(f"Day Trade P&L: R$ {summary['day_trade_pnl']:,.2f}")
        print(f"Loss Carryforward Balance: R$ {summary['loss_carryforward_balance']:,.2f}")
        
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
    
    # Export audit trails
    portfolio.export_audit_trails("enhanced_audit_trails")
    print("\nAudit trails exported to enhanced_audit_trails/")


if __name__ == "__main__":
    main() 