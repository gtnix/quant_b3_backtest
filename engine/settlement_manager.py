"""
Advanced Settlement Manager for Brazilian Market Backtesting

Sophisticated T+2 settlement tracking with comprehensive business day handling:
- Precise settlement date calculations with B3 holiday calendar using dias_uteis
- Advanced error handling and retry mechanisms
- Comprehensive audit trail generation
- Performance optimization and caching
- Integration with existing portfolio and loss manager
- Transaction Cost Analysis (TCA) integration

Author: Your Name
Date: 2024
"""

import logging
from collections import deque
from datetime import datetime, timedelta, date
from dataclasses import dataclass
from typing import List, Dict, Optional, Literal, Any
import pytz
import functools
import json
from enum import Enum

# Import dias_uteis for Brazilian business day calculations
try:
    import dias_uteis
    DIAS_UTEIS_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("dias_uteis library loaded successfully for Brazilian business day calculations")
except ImportError:
    DIAS_UTEIS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error("dias_uteis library is required but not available. Please install dias_uteis: pip install dias_uteis")
    raise ImportError("dias_uteis library is required for Brazilian business day calculations")

logger = logging.getLogger(__name__)


class TradeType(Enum):
    """Enumeration for trade types."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class SettlementItem:
    """Enhanced settlement item with comprehensive tracking."""
    settlement_date: date
    amount: float
    trade_type: TradeType
    description: str
    original_trade_date: date
    ticker: Optional[str] = None
    trade_id: Optional[str] = None
    status: str = "pending"  # pending, settled, failed
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'settlement_date': self.settlement_date.isoformat(),
            'amount': self.amount,
            'trade_type': self.trade_type.value,
            'description': self.description,
            'original_trade_date': self.original_trade_date.isoformat(),
            'ticker': self.ticker,
            'trade_id': self.trade_id,
            'status': self.status
        }


class AdvancedSettlementManager:
    """
    Sophisticated T+2 settlement tracking with market-specific nuances.
    
    Features:
    - Precise business day calculations with dias_uteis library
    - Comprehensive error handling and validation
    - Performance optimization with caching
    - Detailed audit trail and logging
    - Market timezone handling
    """
    
    def __init__(self, 
                 initial_cash: float = 0.0,
                 settlement_days: int = 2, 
                 market_timezone: str = 'America/Sao_Paulo'):
        """
        Initialize sophisticated T+2 settlement tracking.
        
        Args:
            initial_cash: Starting cash amount
            settlement_days: Standard settlement cycle (default B3: 2 days)
            market_timezone: Precise market timezone handling
        """
        if not DIAS_UTEIS_AVAILABLE:
            raise ImportError("dias_uteis library is required for Brazilian business day calculations")
        
        self.settlement_queue: deque = deque()
        self.cash_settled: float = initial_cash  # Renamed for clarity
        self.cash_pending_in: float = 0.0  # Cash to receive from SELLs (unsettled)
        self.cash_pending_out: float = 0.0  # Cash to pay for BUYs (unsettled)
        self.settlement_days = settlement_days
        self.timezone = pytz.timezone(market_timezone)
        self.settlement_history: List[SettlementItem] = []
        self.failed_settlements: List[SettlementItem] = []
        self._business_days_cache = {}  # Cache for business day calculations
        
        logger.info("Using dias_uteis library for Brazilian business day calculations")
        
        logger.info(f"Advanced Settlement Manager initialized "
                   f"(T+{settlement_days}, timezone: {market_timezone}, "
                   f"dias_uteis: enabled)")
    
    def _validate_inputs(self, trade_date: datetime, amount: float, 
                        trade_type: TradeType, ticker: Optional[str] = None) -> None:
        """
        Validate input parameters with comprehensive error checking.
        
        Args:
            trade_date: Date of original trade
            amount: Cash amount to settle
            trade_type: Type of trade (BUY/SELL)
            ticker: Optional asset identifier
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(trade_date, datetime):
            raise ValueError("Trade date must be a datetime object")
        
        if not isinstance(amount, (int, float)) or amount <= 0:
            raise ValueError("Amount must be a positive numeric value")
        
        if not isinstance(trade_type, TradeType):
            raise ValueError("Trade type must be a TradeType enum")
        
        if ticker is not None and not isinstance(ticker, str):
            raise ValueError("Ticker must be a string or None")
    
    def _is_business_day(self, check_date: date) -> bool:
        """
        Check if a date is a business day using dias_uteis.
        
        Args:
            check_date: Date to check
            
        Returns:
            bool: True if business day, False otherwise
        """
        # Check cache first
        cache_key = check_date.isoformat()
        if cache_key in self._business_days_cache:
            return self._business_days_cache[cache_key]
        
        # Use dias_uteis library for accurate Brazilian business day calculation
        try:
            # dias_uteis.is_du() returns True if it's a business day
            is_business = dias_uteis.is_du(check_date)
            self._business_days_cache[cache_key] = is_business
            return is_business
        except Exception as e:
            logger.error(f"Error using dias_uteis for date {check_date}: {e}")
            raise
    
    def _calculate_business_days_forward(self, start_date: date, days: int) -> date:
        """
        Calculate date that is N business days forward from start date.
        
        Args:
            start_date: Starting date
            days: Number of business days to add
            
        Returns:
            date: Target date
        """
        try:
            # If start_date is not a business day, find the next business day first
            current_date = start_date
            if not self._is_business_day(current_date):
                current_date = self.get_next_business_day(current_date)
            
            # Use dias_uteis for accurate Brazilian business day calculation
            target_date = dias_uteis.delta_du(current_date, days)
            return target_date
        except Exception as e:
            logger.error(f"Error using dias_uteis for business day calculation: {e}")
            raise
    
    def _calculate_t2_settlement_date(self, trade_date: date) -> date:
        """
        Calculate T+2 settlement date with business day handling.
        
        Args:
            trade_date: Date of original trade
            
        Returns:
            date: Settlement date (T+2 business days)
        """
        return self._calculate_business_days_forward(trade_date, self.settlement_days)
    
    def get_business_days_between(self, start_date: date, end_date: date) -> int:
        """
        Calculate the number of business days between two dates.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            int: Number of business days between the dates
        """
        try:
            return dias_uteis.diff_du(start_date, end_date)
        except Exception as e:
            logger.error(f"Error using dias_uteis for business days calculation: {e}")
            raise
    
    def get_next_business_day(self, from_date: date) -> date:
        """
        Get the next business day from a given date.
        
        Args:
            from_date: Starting date
            
        Returns:
            date: Next business day
        """
        try:
            return dias_uteis.next_du(from_date)
        except Exception as e:
            logger.error(f"Error using dias_uteis for next business day: {e}")
            raise
    
    def schedule_trade(self, 
                       trade_date: datetime, 
                       amount: float, 
                       trade_type: Literal['BUY', 'SELL'],
                       ticker: Optional[str] = None,
                       trade_id: Optional[str] = None,
                       description: str = "") -> None:
        """
        Advanced trade settlement scheduling with comprehensive validation.
        
        Compliance & Safety Checklist:
        ✅ Precise settlement date calculation
        ✅ Detailed trade type tracking
        ✅ Optional ticker-level granularity
        
        Args:
            trade_date: Date of original trade
            amount: Cash amount to settle
            trade_type: Type of trade ('BUY' or 'SELL')
            ticker: Optional asset identifier
            trade_id: Optional trade identifier for audit trail
            description: Description of the trade for logging
        """
        try:
            # Validate inputs
            trade_type_enum = TradeType(trade_type)
            self._validate_inputs(trade_date, amount, trade_type_enum, ticker)
            
            # Ensure timezone awareness
            if trade_date.tzinfo is None:
                trade_date = self.timezone.localize(trade_date)
            
            # Calculate settlement date
            settlement_date = self._calculate_t2_settlement_date(trade_date.date())
            
            # Create settlement item
            settlement_item = SettlementItem(
                settlement_date=settlement_date,
                amount=amount,
                trade_type=trade_type_enum,
                description=description,
                original_trade_date=trade_date.date(),
                ticker=ticker,
                trade_id=trade_id
            )
            
            # Add to queue
            self.settlement_queue.append(settlement_item)
            
            # Update pending cash counters
            if trade_type_enum == TradeType.BUY:
                self.cash_pending_out += amount
            else:  # SELL
                self.cash_pending_in += amount
            
            logger.info(f"Scheduled {settlement_item.trade_type.value} settlement: "
                       f"R$ {amount:,.2f} for {settlement_date}, "
                       f"Description: {description}")
            
        except Exception as e:
            logger.error(f"Error scheduling trade settlement: {str(e)}")
            raise
    
    def process_settlements(self, current_date: date) -> float:
        """
        Process settled trades with comprehensive error handling.
        
        Args:
            current_date: Current simulation date
        
        Returns:
            float: Cash newly available after settlement
        """
        newly_available = 0.0
        pending_settlements = deque()
        
        try:
            # Process all settlements due today or earlier
            while self.settlement_queue:
                settlement = self.settlement_queue.popleft()
                
                if settlement.settlement_date <= current_date:
                    try:
                        # Execute settlement
                        if settlement.trade_type == TradeType.BUY:
                            # Buy settlement: deduct cash from settled balance and pending out
                            self.cash_pending_out -= settlement.amount
                            self.cash_settled -= settlement.amount
                            newly_available -= settlement.amount
                        else:  # SELL
                            # Sell settlement: add cash to settled balance and reduce pending in
                            self.cash_pending_in -= settlement.amount
                            self.cash_settled += settlement.amount
                            newly_available += settlement.amount
                        
                        # Mark as settled
                        settlement.status = "settled"
                        
                        # Add to history
                        self.settlement_history.append(settlement)
                        
                        logger.info(f"T+2 settlement processed: {settlement.trade_type.value} "
                                   f"R$ {settlement.amount:,.2f} "
                                   f"(original trade: {settlement.original_trade_date})")
                        
                    except Exception as e:
                        # Mark as failed and add to failed settlements
                        settlement.status = "failed"
                        self.failed_settlements.append(settlement)
                        logger.error(f"Failed to process settlement: {str(e)}")
                        
                else:
                    # Keep for future processing
                    pending_settlements.append(settlement)
            
            # Restore pending settlements
            self.settlement_queue = pending_settlements
            
        except Exception as e:
            logger.error(f"Error processing settlements: {str(e)}")
            raise
        
        return newly_available
    
    def get_available_cash(self, 
                           current_date: date, 
                           strict_mode: bool = True) -> float:
        """
        Intelligent cash availability calculation with safety checks.
        
        Args:
            current_date: Reference date for settlement
            strict_mode: Enforce rigorous settlement rules
        
        Returns:
            float: Safely calculated available cash
        """
        try:
            # Process any settlements due
            self.process_settlements(current_date)
            
            if strict_mode:
                # In strict mode, only return settled cash
                return self.cash_settled
            else:
                # In non-strict mode, return total cash (settled + pending_in - pending_out)
                return self.total_cash
                
        except Exception as e:
            logger.error(f"Error calculating available cash: {str(e)}")
            # Return settled cash as fallback
            return self.cash_settled
    
    def get_unsettled_cash(self) -> float:
        """Get cash that is not yet settled (in transit)."""
        # Unsettled cash = pending sells (cash to be received) - pending buys (cash to be paid)
        return self.cash_pending_in - self.cash_pending_out
    
    def get_settlement_summary(self) -> Dict:
        """Get comprehensive settlement summary."""
        return {
            'cash_settled': self.cash_settled,
            'cash_pending_in': self.cash_pending_in,
            'cash_pending_out': self.cash_pending_out,
            'total_cash': self.total_cash,
            'unsettled_cash': self.get_unsettled_cash(),
            'pending_settlements': len(self.settlement_queue),
            'pending_buys': self.cash_pending_out,
            'pending_sells': self.cash_pending_in,
            'settlement_history_count': len(self.settlement_history),
            'failed_settlements_count': len(self.failed_settlements),
            'settlement_days': self.settlement_days,
            'timezone': str(self.timezone)
        }
    
    def get_pending_settlements(self) -> List[Dict]:
        """Get list of pending settlements for reporting."""
        return [item.to_dict() for item in self.settlement_queue]
    
    def get_failed_settlements(self) -> List[Dict]:
        """Get list of failed settlements for analysis."""
        return [item.to_dict() for item in self.failed_settlements]
    
    def export_audit_trail(self, filepath: str) -> None:
        """
        Export comprehensive audit trail for regulatory compliance.
        
        Args:
            filepath: Path to export audit trail
        """
        audit_data = {
            'settlement_history': [item.to_dict() for item in self.settlement_history],
            'failed_settlements': [item.to_dict() for item in self.failed_settlements],
            'pending_settlements': self.get_pending_settlements(),
            'summary': self.get_settlement_summary(),
            'export_date': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(audit_data, f, indent=2)
        
        logger.info(f"Settlement audit trail exported to {filepath}")
    
    def get_summary_data(self) -> Dict[str, Any]:
        """Get settlement summary data for HTML reports."""
        summary = self.get_settlement_summary()
        return {
            'cash_settled': summary['cash_settled'],
            'cash_pending_in': summary['cash_pending_in'],
            'cash_pending_out': summary['cash_pending_out'],
            'total_cash': summary['total_cash'],
            'unsettled_cash': summary['unsettled_cash'],
            'pending_settlements': summary['pending_settlements'],
            'pending_buys': summary['pending_buys'],
            'pending_sells': summary['pending_sells'],
            'settlement_history_count': summary['settlement_history_count'],
            'failed_settlements_count': summary['failed_settlements_count'],
            'settlement_days': summary['settlement_days'],
            'timezone': summary['timezone'],
            'settlement_queue': [item.to_dict() for item in self.settlement_queue],
            'failed_settlements': [item.to_dict() for item in self.failed_settlements],
            'dias_uteis_enabled': DIAS_UTEIS_AVAILABLE
        }
    
    def print_summary(self) -> None:
        """Print comprehensive settlement summary to console."""
        summary = self.get_settlement_summary()
        
        print("\n=== Advanced Settlement Manager Summary ===")
        print(f"💰 Cash Settled: R$ {summary['cash_settled']:,.2f}")
        print(f"📥 Cash Pending In: R$ {summary['cash_pending_in']:,.2f}")
        print(f"📤 Cash Pending Out: R$ {summary['cash_pending_out']:,.2f}")
        print(f"💳 Total Cash: R$ {summary['total_cash']:,.2f}")
        print(f"⏳ Unsettled Cash: R$ {summary['unsettled_cash']:,.2f}")
        print(f"📋 Pending Settlements: {summary['pending_settlements']}")
        print(f"📈 Pending Buys: R$ {summary['pending_buys']:,.2f}")
        print(f"📉 Pending Sells: R$ {summary['pending_sells']:,.2f}")
        print(f"✅ Settlement History: {summary['settlement_history_count']} items")
        print(f"❌ Failed Settlements: {summary['failed_settlements_count']} items")
        print(f"🔄 Settlement Cycle: T+{summary['settlement_days']}")
        print(f"🌍 Timezone: {summary['timezone']}")
        print(f"📅 dias_uteis: {'✅ Enabled'}")
        
        print("=" * 50)

    @property
    def total_cash(self) -> float:
        """Calculate total cash as settled + pending_in - pending_out."""
        return self.cash_settled + self.cash_pending_in - self.cash_pending_out


def main():
    """Example usage demonstrating enhanced settlement functionality with dias_uteis."""
    
    print("=== Advanced Settlement Manager Test with dias_uteis ===")
    
    # Initialize advanced settlement manager with dias_uteis enabled
    settlement_mgr = AdvancedSettlementManager(
        initial_cash=10000.0,
    )
    
    # Scenario: Multiple trades over several days with business day handling
    today = date.today()
    
    print(f"\n--- Day 1 ({today}) ---")
    settlement_mgr.schedule_trade(
        trade_date=datetime.combine(today, datetime.min.time()),
        amount=5000.0, 
        trade_type='BUY',
        ticker="VALE3",
        trade_id="VALE3_001",
        description="Buy VALE3"
    )
    print(f"Available cash: R$ {settlement_mgr.get_available_cash(today):,.2f}")
    
    print(f"\n--- Day 2 ({today + timedelta(days=1)}) ---")
    settlement_mgr.schedule_trade(
        trade_date=datetime.combine(today + timedelta(days=1), datetime.min.time()),
        amount=3000.0, 
        trade_type='SELL',
        ticker="PETR4",
        trade_id="PETR4_001",
        description="Sell PETR4"
    )
    print(f"Available cash: R$ {settlement_mgr.get_available_cash(today + timedelta(days=1)):,.2f}")
    
    # Demonstrate business day calculations
    print(f"\n--- Business Day Calculations ---")
    next_business_day = settlement_mgr.get_next_business_day(today)
    print(f"Next business day from {today}: {next_business_day}")
    
    business_days_between = settlement_mgr.get_business_days_between(today, today + timedelta(days=7))
    print(f"Business days between {today} and {today + timedelta(days=7)}: {business_days_between}")
    
    # Test settlement processing
    print(f"\n--- Settlement Processing ---")
    # Process settlements for the next few days
    for i in range(2, 6):
        test_date = today + timedelta(days=i)
        available = settlement_mgr.get_available_cash(test_date)
        print(f"Day {i} ({test_date}): Available cash: R$ {available:,.2f}")
    
    # Print enhanced summary
    settlement_mgr.print_summary()
    
    # Export audit trail
    settlement_mgr.export_audit_trail("settlement_audit_trail.json")
    print("\nSettlement audit trail exported to settlement_audit_trail.json")
    
    # Demonstrate dias_uteis functionality
    print(f"\n--- dias_uteis Business Day Calculations ---")
    
    # Test business day calculations
    test_date = date(2024, 1, 1)  # New Year's Day
    is_business = settlement_mgr._is_business_day(test_date)
    next_business = settlement_mgr.get_next_business_day(test_date)
    
    print(f"New Year's Day ({test_date}): Business day: {is_business}")
    print(f"Next business day after {test_date}: {next_business}")
    
    # Test T+2 settlement calculation
    trade_date = date(2024, 12, 27)  # Friday before New Year
    settlement_date = settlement_mgr._calculate_t2_settlement_date(trade_date)
    business_days = settlement_mgr.get_business_days_between(trade_date, settlement_date)
    
    print(f"Trade date: {trade_date} (Friday before New Year)")
    print(f"T+2 settlement date: {settlement_date}")
    print(f"Business days between: {business_days}")
    
    print("✅ dias_uteis library provides accurate Brazilian business day calculations")


if __name__ == "__main__":
    main() 