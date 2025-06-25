"""
Loss Carryforward Manager for Brazilian Market Compliance

This module handles loss carryforward tracking in accordance with Brazilian tax law:
- Indefinite loss carryforward across different assets and trading periods
- Per-asset loss tracking for tax-efficient loss application
- Global cumulative loss tracking for comprehensive tax planning

Author: Your Name
Date: 2024
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LossRecord:
    """Record of losses for tax carryforward purposes."""
    amount: float
    date: str  # YYYY-MM-DD format
    asset: str
    trade_type: str  # 'day_trade' or 'swing_trade'


class LossCarryforwardManager:
    """
    Manages loss carryforward tracking per asset and globally.
    
    Brazilian tax law allows indefinite loss carryforward across different 
    assets and trading periods, making this essential for accurate tax calculation.
    """
    
    def __init__(self):
        """
        Initialize loss tracking per asset and globally.
        
        Why this matters: Brazilian tax law allows indefinite loss carryforward 
        across different assets and trading periods
        """
        self.asset_losses: Dict[str, float] = {}  # {ticker: accumulated_loss}
        self.total_cumulative_loss: float = 0.0
        self.loss_history: list[LossRecord] = []  # For audit trail
        
        logger.info("Loss Carryforward Manager initialized")
    
    def record_trade_result(self, ticker: str, trade_profit: float, 
                           trade_date: str, trade_type: str) -> float:
        """
        Record trade profit/loss and update loss carryforward balance.
        
        Args:
            ticker (str): Trading asset identifier
            trade_profit (float): Net profit/loss from trade (negative = loss)
            trade_date (str): Date of trade in YYYY-MM-DD format
            trade_type (str): 'day_trade' or 'swing_trade'
        
        Returns:
            float: Taxable profit after applying loss carryforward
        """
        if trade_profit < 0:
            # Record loss
            loss_amount = abs(trade_profit)
            
            # Update asset-specific loss
            if ticker not in self.asset_losses:
                self.asset_losses[ticker] = 0.0
            self.asset_losses[ticker] += loss_amount
            
            # Update global cumulative loss
            self.total_cumulative_loss += loss_amount
            
            # Record for audit trail
            self.loss_history.append(LossRecord(
                amount=loss_amount,
                date=trade_date,
                asset=ticker,
                trade_type=trade_type
            ))
            
            logger.info(f"Recorded loss for {ticker}: R$ {loss_amount:.2f} "
                       f"({trade_type}), Total asset loss: R$ {self.asset_losses[ticker]:.2f}")
            
            return 0.0  # No taxable profit when there's a loss
        
        else:
            # Apply loss carryforward against profit
            taxable_profit = self.get_taxable_profit(ticker, trade_profit)
            
            logger.info(f"Trade profit for {ticker}: R$ {trade_profit:.2f}, "
                       f"Taxable after loss carryforward: R$ {taxable_profit:.2f}")
            
            return taxable_profit
    
    def get_taxable_profit(self, ticker: str, gross_profit: float) -> float:
        """
        Calculate taxable profit after applying asset and global loss carryforward.
        
        Args:
            ticker (str): Trading asset identifier
            gross_profit (float): Gross trade profit
        
        Returns:
            float: Profit subject to capital gains tax
        """
        if gross_profit <= 0:
            return 0.0
        
        # First, apply asset-specific losses
        asset_loss_available = self.asset_losses.get(ticker, 0.0)
        remaining_profit = gross_profit
        
        if asset_loss_available > 0:
            loss_to_apply = min(asset_loss_available, gross_profit)
            remaining_profit -= loss_to_apply
            self.asset_losses[ticker] -= loss_to_apply
            
            logger.info(f"Applied asset loss for {ticker}: R$ {loss_to_apply:.2f}, "
                       f"Remaining asset loss: R$ {self.asset_losses[ticker]:.2f}")
        
        # If there's still profit, apply global losses
        if remaining_profit > 0 and self.total_cumulative_loss > 0:
            global_loss_to_apply = min(self.total_cumulative_loss, remaining_profit)
            remaining_profit -= global_loss_to_apply
            self.total_cumulative_loss -= global_loss_to_apply
            
            logger.info(f"Applied global loss: R$ {global_loss_to_apply:.2f}, "
                       f"Remaining global loss: R$ {self.total_cumulative_loss:.2f}")
        
        return max(0.0, remaining_profit)
    
    def get_asset_loss_balance(self, ticker: str) -> float:
        """Get current loss balance for a specific asset."""
        return self.asset_losses.get(ticker, 0.0)
    
    def get_total_loss_balance(self) -> float:
        """Get total cumulative loss balance across all assets."""
        return self.total_cumulative_loss
    
    def get_loss_summary(self) -> Dict:
        """Get comprehensive loss summary for reporting."""
        return {
            'asset_losses': self.asset_losses.copy(),
            'total_cumulative_loss': self.total_cumulative_loss,
            'total_losses_recorded': len(self.loss_history),
            'assets_with_losses': len([k for k, v in self.asset_losses.items() if v > 0])
        }
    
    def reset_monthly_tracking(self, current_month: int):
        """
        Reset monthly tracking (loss carryforward persists across months).
        
        Note: In Brazil, loss carryforward is indefinite, so this is mainly
        for reporting purposes.
        """
        logger.info(f"Monthly tracking reset for month {current_month} "
                   f"(loss carryforward preserved)")
    
    def print_summary(self):
        """Print comprehensive loss carryforward summary."""
        print("\n" + "="*50)
        print("LOSS CARRYFORWARD SUMMARY")
        print("="*50)
        print(f"Total cumulative loss: R$ {self.total_cumulative_loss:,.2f}")
        print(f"Assets with losses: {len([k for k, v in self.asset_losses.items() if v > 0])}")
        
        if self.asset_losses:
            print("\nAsset-specific losses:")
            for ticker, loss in self.asset_losses.items():
                if loss > 0:
                    print(f"  {ticker}: R$ {loss:,.2f}")
        
        print("="*50)


def main():
    """Example usage demonstrating loss carryforward functionality."""
    
    # Initialize loss manager
    loss_mgr = LossCarryforwardManager()
    
    print("=== Loss Carryforward Manager Test ===")
    
    # Scenario 1: Record losses
    print("\n--- Recording Losses ---")
    loss_mgr.record_trade_result("VALE3", -1000.0, "2024-01-15", "swing_trade")
    loss_mgr.record_trade_result("PETR4", -500.0, "2024-01-16", "day_trade")
    loss_mgr.record_trade_result("VALE3", -750.0, "2024-01-17", "swing_trade")
    
    # Scenario 2: Apply losses against profits
    print("\n--- Applying Losses Against Profits ---")
    taxable_profit1 = loss_mgr.get_taxable_profit("VALE3", 2000.0)
    print(f"VALE3 profit R$ 2,000.00 -> Taxable: R$ {taxable_profit1:.2f}")
    
    taxable_profit2 = loss_mgr.get_taxable_profit("PETR4", 1000.0)
    print(f"PETR4 profit R$ 1,000.00 -> Taxable: R$ {taxable_profit2:.2f}")
    
    # Print summary
    loss_mgr.print_summary()


if __name__ == "__main__":
    main() 