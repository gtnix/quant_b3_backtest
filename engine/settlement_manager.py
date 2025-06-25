"""
Settlement Manager for Brazilian Market T+2 Cash Flow

This module handles T+2 settlement queue management to accurately model 
Brazilian market's two-day cash settlement cycle, preventing the use of 
unsettled funds for new trades.

Author: Your Name
Date: 2024
"""

import logging
from collections import deque
from datetime import datetime, timedelta, date
from dataclasses import dataclass
from typing import List, Dict

logger = logging.getLogger(__name__)


@dataclass
class SettlementItem:
    """Represents a pending settlement item."""
    settlement_date: date
    amount: float
    trade_type: str  # 'buy' or 'sell'
    description: str
    original_trade_date: date


class SettlementManager:
    """
    Manages T+2 settlement queue for accurate cash flow simulation.
    
    Brazilian market operates on T+2 settlement, meaning cash from trades
    is only available 2 business days after the trade date.
    """
    
    def __init__(self, initial_cash: float = 0.0):
        """
        Initialize T+2 settlement queue.
        
        Why this matters: Accurately models Brazilian market's 
        two-day cash settlement cycle
        """
        self.settlement_queue: deque = deque()
        self.settled_cash: float = initial_cash  # Cash available for trading
        self.total_cash: float = initial_cash    # Total cash including unsettled
        self.settlement_history: List[SettlementItem] = []
        
        logger.info(f"Settlement Manager initialized with R$ {initial_cash:,.2f}")
    
    def schedule_trade(self, trade_date: date, amount: float, is_buy: bool, 
                      description: str = "") -> None:
        """
        Schedule cash movement for T+2 settlement.
        
        Args:
            trade_date (date): Date of original trade
            amount (float): Cash amount to settle
            is_buy (bool): Whether trade is buy (debit) or sell (credit)
            description (str): Description of the trade for logging
        """
        # Calculate T+2 settlement date
        settlement_date = self._calculate_t2_settlement_date(trade_date)
        
        # Create settlement item
        settlement_item = SettlementItem(
            settlement_date=settlement_date,
            amount=amount,
            trade_type='buy' if is_buy else 'sell',
            description=description,
            original_trade_date=trade_date
        )
        
        # Add to queue
        self.settlement_queue.append(settlement_item)
        
        # Update total cash immediately (but not settled cash)
        if is_buy:
            self.total_cash -= amount
        else:
            self.total_cash += amount
        
        logger.info(f"Scheduled {settlement_item.trade_type} settlement: "
                   f"R$ {amount:,.2f} for {settlement_date}, "
                   f"Description: {description}")
    
    def process_settlements(self, current_date: date) -> float:
        """
        Process settled trades, return newly available cash.
        
        Args:
            current_date (date): Current simulation date
        
        Returns:
            float: Cash newly available after settlement
        """
        newly_available = 0.0
        pending_settlements = deque()
        
        # Process all settlements due today or earlier
        while self.settlement_queue:
            settlement = self.settlement_queue.popleft()
            
            if settlement.settlement_date <= current_date:
                # Execute settlement
                if settlement.trade_type == 'buy':
                    # Buy settlement: cash was already deducted, just confirm
                    newly_available -= settlement.amount  # Negative because it's a debit
                else:
                    # Sell settlement: add cash to settled balance
                    self.settled_cash += settlement.amount
                    newly_available += settlement.amount
                
                # Add to history
                self.settlement_history.append(settlement)
                
                logger.info(f"T+2 settlement processed: {settlement.trade_type} "
                           f"R$ {settlement.amount:,.2f} "
                           f"(original trade: {settlement.original_trade_date})")
            else:
                # Keep for future processing
                pending_settlements.append(settlement)
        
        # Restore pending settlements
        self.settlement_queue = pending_settlements
        
        return newly_available
    
    def get_available_cash(self, current_date: date) -> float:
        """
        Calculate currently available cash considering settlements.
        
        Args:
            current_date (date): Current simulation date
        
        Returns:
            float: Cash available for trading
        """
        # Process any settlements due
        self.process_settlements(current_date)
        
        return self.settled_cash
    
    def get_unsettled_cash(self) -> float:
        """Get cash that is not yet settled (in transit)."""
        return self.total_cash - self.settled_cash
    
    def get_settlement_summary(self) -> Dict:
        """Get comprehensive settlement summary."""
        pending_buys = sum(item.amount for item in self.settlement_queue 
                          if item.trade_type == 'buy')
        pending_sells = sum(item.amount for item in self.settlement_queue 
                           if item.trade_type == 'sell')
        
        return {
            'settled_cash': self.settled_cash,
            'total_cash': self.total_cash,
            'unsettled_cash': self.get_unsettled_cash(),
            'pending_settlements': len(self.settlement_queue),
            'pending_buys': pending_buys,
            'pending_sells': pending_sells,
            'settlement_history_count': len(self.settlement_history)
        }
    
    def _calculate_t2_settlement_date(self, trade_date: date) -> date:
        """
        Calculate T+2 settlement date (2 business days after trade).
        
        Note: This is a simplified implementation. In production, you'd want
        to handle weekends, holidays, and market closures.
        """
        # Simple implementation: add 2 days
        # In reality, you'd skip weekends and holidays
        return trade_date + timedelta(days=2)
    
    def get_pending_settlements(self) -> List[Dict]:
        """Get list of pending settlements for reporting."""
        return [
            {
                'settlement_date': item.settlement_date,
                'amount': item.amount,
                'trade_type': item.trade_type,
                'description': item.description,
                'original_trade_date': item.original_trade_date
            }
            for item in self.settlement_queue
        ]
    
    def print_summary(self):
        """Print comprehensive settlement summary."""
        summary = self.get_settlement_summary()
        
        print("\n" + "="*50)
        print("SETTLEMENT SUMMARY")
        print("="*50)
        print(f"Settled cash (available): R$ {summary['settled_cash']:,.2f}")
        print(f"Total cash (including unsettled): R$ {summary['total_cash']:,.2f}")
        print(f"Unsettled cash (in transit): R$ {summary['unsettled_cash']:,.2f}")
        print(f"Pending settlements: {summary['pending_settlements']}")
        print(f"Pending buys: R$ {summary['pending_buys']:,.2f}")
        print(f"Pending sells: R$ {summary['pending_sells']:,.2f}")
        
        if self.settlement_queue:
            print("\nPending settlements:")
            for item in self.settlement_queue:
                print(f"  {item.settlement_date}: {item.trade_type.upper()} "
                      f"R$ {item.amount:,.2f} - {item.description}")
        
        print("="*50)


def main():
    """Example usage demonstrating T+2 settlement functionality."""
    
    # Initialize settlement manager
    settlement_mgr = SettlementManager(initial_cash=10000.0)
    
    print("=== Settlement Manager Test ===")
    
    # Scenario: Multiple trades over several days
    today = date.today()
    
    print(f"\n--- Day 1 ({today}) ---")
    settlement_mgr.schedule_trade(today, 5000.0, is_buy=True, 
                                 description="Buy VALE3")
    print(f"Available cash: R$ {settlement_mgr.get_available_cash(today):,.2f}")
    
    print(f"\n--- Day 2 ({today + timedelta(days=1)}) ---")
    settlement_mgr.schedule_trade(today + timedelta(days=1), 3000.0, is_buy=False,
                                 description="Sell PETR4")
    print(f"Available cash: R$ {settlement_mgr.get_available_cash(today + timedelta(days=1)):,.2f}")
    
    print(f"\n--- Day 3 ({today + timedelta(days=2)}) ---")
    # First settlement should be processed
    available = settlement_mgr.get_available_cash(today + timedelta(days=2))
    print(f"Available cash: R$ {available:,.2f}")
    
    print(f"\n--- Day 4 ({today + timedelta(days=3)}) ---")
    # Second settlement should be processed
    available = settlement_mgr.get_available_cash(today + timedelta(days=3))
    print(f"Available cash: R$ {available:,.2f}")
    
    # Print final summary
    settlement_mgr.print_summary()


if __name__ == "__main__":
    main() 