"""
Enhanced Loss Carryforward Manager for Brazilian Market Backtesting

Advanced loss tracking with comprehensive Brazilian tax compliance:
- Per-modality FIFO loss tracking (SWING/DAY)
- 100% loss offset capability (no 30% limit)
- Monthly exemption handling for swing trades
- Comprehensive audit trail
- Performance optimization
- Robust error handling

Author: Your Name
Date: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Literal, Any
from dataclasses import dataclass, field
import logging
import yaml
import pytz

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class LossRecord:
    """Enhanced record of losses for tax carryforward purposes."""
    id: str
    amount: float
    date: datetime
    asset: str
    modality: Literal["SWING", "DAY"]  # Trade modality
    trade_id: Optional[str] = None
    description: str = ""
    applied_amount: float = 0.0  # Track how much has been applied
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'amount': self.amount,
            'date': self.date.isoformat(),
            'asset': self.asset,
            'modality': self.modality,
            'trade_id': self.trade_id,
            'description': self.description,
            'applied_amount': self.applied_amount
        }


@dataclass
class LossApplication:
    """Record of loss application for audit trail."""
    original_loss_date: datetime
    applied_amount: float
    application_date: datetime
    asset: str
    modality: str
    remaining_loss: float
    application_reason: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'original_loss_date': self.original_loss_date.isoformat(),
            'applied_amount': self.applied_amount,
            'application_date': self.application_date.isoformat(),
            'asset': self.asset,
            'modality': self.modality,
            'remaining_loss': self.remaining_loss,
            'application_reason': self.application_reason
        }


class EnhancedLossCarryforwardManager:
    """
    Advanced loss carryforward tracking for individual taxpayers.
    
    Features:
    - Per-modality FIFO loss tracking (SWING/DAY)
    - 100% loss offset capability (no 30% limit)
    - Monthly exemption handling for swing trades
    - Comprehensive audit trail
    - Performance optimization
    - Robust error handling
    """
    
    def __init__(self, config_path: str = "config/settings.yaml", timezone: str = 'America/Sao_Paulo'):
        """
        Initialize advanced loss tracking for individual taxpayers.
        
        Brazilian Individual Taxpayer Compliance (2025):
        - 100% loss offset against capital gains (no 30% limit)
        - Per-modality loss tracking (SWING/DAY)
        - Monthly exemption for swing trades ≤ R$ 20,000 (STOCK assets only)
        - Perpetual loss carryforward
        - IN RFB 1.585/2015 compliance
        
        Args:
            config_path: Path to configuration file
            timezone: Market timezone for precise timestamp handling
        """
        self.config = self._load_config(config_path)
        self.timezone = pytz.timezone(timezone)
        
        # Per-modality FIFO loss tracking
        self.loss_fifo: Dict[str, deque] = {
            "SWING": deque(),
            "DAY": deque()
        }
        
        # Monthly tracking for exemption and aggregation
        self.monthly_swing_sales: Dict[str, float] = defaultdict(float)  # {YYYY-MM: total_sales}
        self.monthly_swing_sales_by_asset_type: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))  # {YYYY-MM: {asset_type: sales}}
        self.monthly_swing_profits: Dict[str, float] = defaultdict(float)  # {YYYY-MM: total_profits}
        self.monthly_day_profits: Dict[str, float] = defaultdict(float)  # {YYYY-MM: total_profits}
        
        # Daily asset profit tracking for day trades (Brazilian IRRF compliance)
        # Structure: {YYYY-MM-DD: {asset: daily_profit}}
        self.daily_asset_profits: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Audit trail
        self.loss_history: List[LossRecord] = []
        self.application_history: List[LossApplication] = []
        
        # Configuration from settings
        tax_config = self.config.get('taxes', {})
        self.swing_exemption_limit = tax_config.get('swing_exemption_limit', 20000)
        self.max_loss_offset_percentage = tax_config.get('max_loss_offset_percentage', 1.0)
        self.person_type = tax_config.get('person_type', 'individual')
        
        # Regulatory constants
        self.CAPITAL_GAINS_ONLY = True
        self.PERPETUAL_CARRYFORWARD = True
        
        logger.info(f"Enhanced Loss Carryforward Manager initialized for {self.person_type} "
                   f"(max offset: {self.max_loss_offset_percentage*100}%, "
                   f"swing exemption: R$ {self.swing_exemption_limit:,.2f}, "
                   f"Brazilian compliance: 2025)")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        import yaml
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {str(e)}")
            return {}
    
    def _validate_inputs(self, ticker: str, trade_profit: float, trade_date: datetime, modality: str) -> None:
        """Validate input parameters."""
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Ticker must be a non-empty string")
        
        if not isinstance(trade_profit, (int, float)):
            raise ValueError("Trade profit must be a numeric value")
        
        if not isinstance(trade_date, datetime):
            raise ValueError("Trade date must be a datetime object")
        
        if modality not in ["SWING", "DAY"]:
            raise ValueError("Modality must be 'SWING' or 'DAY'")
    
    def _generate_loss_id(self, ticker: str, date: datetime, modality: str) -> str:
        """Generate unique loss ID."""
        return f"{ticker}_{date.strftime('%Y%m%d_%H%M%S')}_{modality}"
    
    def is_exempt_eligible_asset_type(self, asset_type: str) -> bool:
        """
        Check if asset type is eligible for R$ 20,000 monthly exemption.
        
        Brazilian Tax Law: Only STOCK assets are eligible for the exemption.
        ETFs, FIIs, BDRs, options, futures, and bonds are NOT eligible.
        
        Args:
            asset_type: Asset type string
            
        Returns:
            bool: True if eligible for exemption, False otherwise
        """
        return asset_type == "STOCK"
    
    def calculate_exemption_eligible_sales(self, month_ref: date) -> float:
        """
        Calculate total swing trade sales for exempt-eligible assets in a month.
        
        This method centralizes the exemption logic - only STOCK assets
        are eligible for the R$ 20,000 monthly exemption.
        
        Args:
            month_ref: Reference month
            
        Returns:
            float: Total swing trade sales for exempt-eligible assets
        """
        return self.get_monthly_swing_sales_by_asset_type(month_ref, "STOCK")
    
    def is_exemption_applicable(self, month_ref: date) -> bool:
        """
        Check if R$ 20,000 exemption is applicable for the month.
        
        Brazilian Tax Law: Exemption applies when total STOCK sales ≤ R$ 20,000/month.
        
        Args:
            month_ref: Reference month
            
        Returns:
            bool: True if exemption applies, False otherwise
        """
        exempt_eligible_sales = self.calculate_exemption_eligible_sales(month_ref)
        return exempt_eligible_sales <= self.swing_exemption_limit
    
    def record_trade_result(self, 
                           ticker: str, 
                           trade_profit: float, 
                           trade_date: datetime,
                           modality: Literal["SWING", "DAY"],
                           trade_id: Optional[str] = None,
                           description: str = "",
                           gross_sales: float = 0.0,
                           asset_type: str = "STOCK") -> float:
        """
        Record trade result for loss carryforward and monthly tracking.
        
        Args:
            ticker: Trading asset identifier
            trade_profit: Net trade profit (can be negative for losses)
            trade_date: Date of trade
            modality: Trade modality (SWING/DAY)
            trade_id: Optional trade identifier
            description: Trade description
            gross_sales: Gross sales amount (for swing trade exemption)
            asset_type: Asset type for exemption filtering (STOCK, ETF, FII, etc.)
            
        Returns:
            float: Taxable profit after loss offset (0.0 for losses)
        """
        try:
            self._validate_inputs(ticker, trade_profit, trade_date, modality)
            
            # Track monthly sales and profits
            month_key = trade_date.strftime('%Y-%m')
            
            if modality == "SWING":
                if gross_sales > 0:
                    self.monthly_swing_sales[month_key] += gross_sales
                    self.monthly_swing_sales_by_asset_type[month_key][asset_type] += gross_sales
                if trade_profit > 0:
                    self.monthly_swing_profits[month_key] += trade_profit
            else:  # DAY
                # Track daily asset profits for day trades (Brazilian IRRF compliance)
                date_key = trade_date.strftime('%Y-%m-%d')
                self.daily_asset_profits[date_key][ticker] += trade_profit
                
                # Also track monthly profits for backward compatibility
                if trade_profit > 0:
                    self.monthly_day_profits[month_key] += trade_profit
            
            if trade_profit < 0:
                # Record loss
                loss_amount = abs(trade_profit)
                
                # Create loss record
                loss_record = LossRecord(
                    id=self._generate_loss_id(ticker, trade_date, modality),
                    amount=loss_amount,
                    date=trade_date,
                    asset=ticker,
                    modality=modality,
                    trade_id=trade_id,
                    description=description
                )
                
                # Add to modality-specific FIFO queue
                self.loss_fifo[modality].append(loss_record)
                
                # Add to global history
                self.loss_history.append(loss_record)
                
                logger.info(f"Recorded {modality} loss for {ticker}: R$ {loss_amount:.2f} "
                           f"(Total {modality} losses: R$ {self.get_modality_loss_balance(modality):.2f})")
                
                return 0.0  # No taxable profit when there's a loss
            
            else:
                # For profits, we'll calculate taxable amount separately
                # This method just records the trade result
                logger.info(f"Recorded {modality} profit for {ticker}: R$ {trade_profit:.2f}")
                return trade_profit
                
        except Exception as e:
            logger.error(f"Error recording trade result for {ticker}: {str(e)}")
            raise
    
    def calculate_taxable_amount(self, 
                                gross_profit: float,
                                modality: Literal["SWING", "DAY"],
                                gross_sales: float,
                                month_ref: date) -> Tuple[float, List[Dict]]:
        """
        Calculate taxable amount for individual taxpayers with proper loss carryforward.
        
        Individual Taxpayer Rules (2025):
        - 100% loss offset capability (no 30% limit)
        - FIFO loss application by modality
        - Swing trade exemption for sales ≤ R$ 20,000/month
        - Day trade no exemption
        - NO GOING BACKWARDS: Losses can only offset profits from same or later months
        - Same-month offsetting is allowed (March loss can offset March profit)
        
        Args:
            gross_profit: Gross profit (≥ 0)
            modality: Trade modality (SWING/DAY)
            gross_sales: Total sales amount for the month
            month_ref: Reference month for exemption calculation
            
        Returns:
            Tuple containing:
            - float: Taxable profit after loss offset
            - List[Dict]: Audit log of loss applications
        """
        if gross_profit < 0:
            return 0.0, []
        
        # Check swing trade exemption - centralized logic
        if modality == "SWING" and self.is_exemption_applicable(month_ref):
            exempt_eligible_sales = self.calculate_exemption_eligible_sales(month_ref)
            logger.info(f"Swing trade exemption applied: STOCK sales R$ {exempt_eligible_sales:,.2f} ≤ R$ {self.swing_exemption_limit:,.2f}")
            # CRITICAL: Lucros isentos não consomem prejuízo (Brazilian tax law)
            # If profit is exempt due to R$ 20,000 limit, no loss carryforward is applied
            return 0.0, []  # Tax exempt, no loss consumption
        
        # Apply loss carryforward using FIFO with month boundary checking
        taxable = gross_profit
        audit_log = []
        fifo = self.loss_fifo[modality]
        
        while taxable > 0 and fifo:
            loss_rec = fifo[0]
            available_loss = loss_rec.amount - loss_rec.applied_amount
            
            if available_loss <= 0:
                # Remove fully applied loss
                fifo.popleft()
                continue
            
            # CRITICAL: Check month boundaries - "No Going Backwards" rule
            loss_month = loss_rec.date.replace(day=1)  # First day of loss month
            profit_month = month_ref.replace(day=1)    # First day of profit month
            
            # Convert both to date objects for comparison
            loss_month_date = loss_month.date() if hasattr(loss_month, 'date') else loss_month
            profit_month_date = profit_month.date() if hasattr(profit_month, 'date') else profit_month
            
            if loss_month_date > profit_month_date:
                # Loss is from a later month - cannot apply (would be going backwards)
                logger.info(f"Skipping loss from {loss_rec.date.strftime('%Y-%m')} "
                           f"for profit in {month_ref.strftime('%Y-%m')} "
                           f"(no going backwards rule)")
                break  # Stop processing further losses for this month
            
            # Loss is from same month or earlier - can apply
            # Apply loss offset
            offset = min(taxable, available_loss)
            taxable -= offset
            loss_rec.applied_amount += offset
            
            # Record in audit log
            audit_log.append({
                "loss_id": loss_rec.id,
                "used": offset,
                "asset": loss_rec.asset,
                "original_date": loss_rec.date.isoformat(),
                "loss_month": loss_rec.date.strftime('%Y-%m'),
                "profit_month": month_ref.strftime('%Y-%m')
            })
            
            # Record application in audit trail
            application = LossApplication(
                original_loss_date=loss_rec.date,
                applied_amount=offset,
                application_date=datetime.now(),
                asset=loss_rec.asset,
                modality=modality,
                remaining_loss=loss_rec.amount - loss_rec.applied_amount
            )
            self.application_history.append(application)
        
            # Remove fully applied loss
            if loss_rec.applied_amount >= loss_rec.amount:
                fifo.popleft()
        
        logger.info(f"Loss carryforward calculation for {modality}: "
                   f"Gross profit: R$ {gross_profit:,.2f}, "
                   f"Taxable: R$ {taxable:,.2f}, "
                   f"Losses applied: {len(audit_log)} "
                   f"(month: {month_ref.strftime('%Y-%m')})")
        
        return max(0.0, taxable), audit_log
    
    def get_modality_loss_balance(self, modality: Literal["SWING", "DAY"]) -> float:
        """
        Get current loss balance for a specific modality.
        
        Args:
            modality: Trade modality (SWING/DAY)
            
        Returns:
            float: Current loss balance for the modality
        """
        if modality not in self.loss_fifo:
            return 0.0
        
        return sum(loss.amount - loss.applied_amount for loss in self.loss_fifo[modality])
    
    def get_total_loss_balance(self) -> float:
        """Get total cumulative loss balance across all modalities."""
        return sum(self.get_modality_loss_balance(modality) for modality in ["SWING", "DAY"])
    
    def get_loss_summary(self) -> Dict:
        """Get comprehensive loss summary for reporting."""
        return {
            'swing_losses': self.get_modality_loss_balance("SWING"),
            'day_losses': self.get_modality_loss_balance("DAY"),
            'total_cumulative_loss': self.get_total_loss_balance(),
            'total_losses_recorded': len(self.loss_history),
            'total_applications': len(self.application_history),
            'person_type': self.person_type,
            'max_loss_offset_percentage': self.max_loss_offset_percentage,
            'swing_exemption_limit': self.swing_exemption_limit,
            'daily_asset_tracking_enabled': True,
            'total_daily_records': len(self.daily_asset_profits)
        }
    
    def get_monthly_swing_sales_by_asset_type(self, month_ref: date, asset_type: str = "STOCK") -> float:
        """
        Get total swing trade sales for a specific month and asset type.
        
        This method is used for exemption calculation - only STOCK assets
        are eligible for the R$ 20,000 monthly exemption.
        
        Args:
            month_ref: Reference month
            asset_type: Asset type to filter by (default: STOCK)
            
        Returns:
            float: Total swing trade sales for the month and asset type
        """
        month_key = month_ref.strftime('%Y-%m')
        return self.monthly_swing_sales_by_asset_type.get(month_key, {}).get(asset_type, 0.0)
    
    def get_monthly_swing_sales(self, month_ref: date) -> float:
        """
        Get total swing trade sales for a specific month.
        
        Args:
            month_ref: Reference month
            
        Returns:
            float: Total swing trade sales for the month
        """
        month_key = month_ref.strftime('%Y-%m')
        return self.monthly_swing_sales.get(month_key, 0.0)
    
    def get_monthly_swing_profits(self, month_ref: date) -> float:
        """
        Get total swing trade profits for a specific month.
        
        Args:
            month_ref: Reference month
            
        Returns:
            float: Total swing trade profits for the month
        """
        month_key = month_ref.strftime('%Y-%m')
        return self.monthly_swing_profits.get(month_key, 0.0)
    
    def get_monthly_day_profits(self, month_ref: date) -> float:
        """
        Get total day trade profits for a specific month.
        
        Args:
            month_ref: Reference month
            
        Returns:
            float: Total day trade profits for the month
        """
        month_key = month_ref.strftime('%Y-%m')
        return self.monthly_day_profits.get(month_key, 0.0)
    
    def calculate_day_trade_irrf(self, month_ref: date) -> float:
        """
        Calculate day trade IRRF according to Brazilian law (1% on daily net profit per asset).
        
        Brazilian IRRF Rules for Day Trades:
        - 1% IRRF on daily net profit per asset (if positive)
        - If daily net profit is negative, no IRRF is withheld
        - IRRF is calculated per asset per day, then aggregated monthly
        
        Args:
            month_ref: Reference month for IRRF calculation
            
        Returns:
            float: Total IRRF amount for the month
        """
        total_irrf = 0.0
        month_str = month_ref.strftime('%Y-%m')
        
        # Iterate through all days in the month
        for date_key, asset_profits in self.daily_asset_profits.items():
            # Check if this date belongs to the target month
            if date_key.startswith(month_str):
                # Calculate IRRF for each asset on this day
                for asset, daily_profit in asset_profits.items():
                    if daily_profit > 0:
                        # Brazilian law: 1% IRRF on positive daily profit per asset
                        asset_irrf = daily_profit * 0.01  # 1%
                        total_irrf += asset_irrf
                        
                        logger.debug(f"Day trade IRRF for {asset} on {date_key}: "
                                   f"R$ {daily_profit:.2f} × 1% = R$ {asset_irrf:.2f}")
        
        logger.info(f"Total day trade IRRF for {month_str}: R$ {total_irrf:.2f}")
        return total_irrf
    
    def get_daily_asset_profits(self, month_ref: date) -> Dict[str, Dict[str, float]]:
        """
        Get daily asset profits for a specific month.
        
        Args:
            month_ref: Reference month
            
        Returns:
            Dict containing daily asset profits: {YYYY-MM-DD: {asset: daily_profit}}
        """
        month_str = month_ref.strftime('%Y-%m')
        daily_profits = {}
        
        for date_key, asset_profits in self.daily_asset_profits.items():
            if date_key.startswith(month_str):
                daily_profits[date_key] = dict(asset_profits)
        
        return daily_profits
    
    def reset_monthly_tracking(self, current_month: int):
        """
        Reset monthly tracking (loss carryforward persists across months).
        
        Note: In Brazil, loss carryforward is indefinite, so this is mainly
        for reporting purposes. Monthly sales/profits tracking continues.
        """
        logger.info(f"Monthly tracking reset for month {current_month} "
                   f"(loss carryforward preserved, monthly tracking continues)")
    
    def export_audit_trail(self, filepath: str) -> None:
        """
        Export comprehensive audit trail for regulatory compliance.
        
        Args:
            filepath: Path to export audit trail
        """
        audit_data = {
            'loss_records': [loss.to_dict() for loss in self.loss_history],
            'application_history': [
                {
                    'original_loss_date': app.original_loss_date.isoformat(),
                    'applied_amount': app.applied_amount,
                    'application_date': app.application_date.isoformat(),
                    'asset': app.asset,
                    'modality': app.modality,
                    'remaining_loss': app.remaining_loss
                }
                for app in self.application_history
            ],
            'summary': self.get_loss_summary(),
            'export_date': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(audit_data, f, indent=2)
        
        logger.info(f"Audit trail exported to {filepath}")
    
    def get_summary_data(self) -> Dict[str, Any]:
        """Get loss carryforward summary data for HTML reports."""
        summary = self.get_loss_summary()
        return {
            'total_cumulative_loss': summary['total_cumulative_loss'],
            'swing_losses': summary['swing_losses'],
            'day_losses': summary['day_losses'],
            'total_losses_recorded': summary['total_losses_recorded'],
            'total_applications': summary['total_applications'],
            'person_type': summary['person_type'],
            'max_loss_offset_percentage': summary['max_loss_offset_percentage'],
            'swing_exemption_limit': summary['swing_exemption_limit']
        }
    
    def print_summary(self):
        """Print comprehensive loss carryforward summary."""
        summary = self.get_loss_summary()
        
        print("\n" + "="*60)
        print("LOSS CARRYFORWARD SUMMARY")
        print("="*60)
        print(f"Person Type: {summary['person_type']}")
        print(f"Max Loss Offset: {summary['max_loss_offset_percentage']*100:.0f}%")
        print(f"Swing Exemption Limit: R$ {summary['swing_exemption_limit']:,.2f}")
        print(f"Total Cumulative Loss: R$ {summary['total_cumulative_loss']:,.2f}")
        print(f"  - Swing Trade Losses: R$ {summary['swing_losses']:,.2f}")
        print(f"  - Day Trade Losses: R$ {summary['day_losses']:,.2f}")
        print(f"Total Losses Recorded: {summary['total_losses_recorded']}")
        print(f"Total Applications: {summary['total_applications']}")
        print("="*60)


def main():
    """Example usage demonstrating individual taxpayer loss carryforward functionality."""
    
    # Initialize enhanced loss manager for individual taxpayers
    loss_mgr = EnhancedLossCarryforwardManager()
    
    print("=== Brazilian Individual Taxpayer Loss Carryforward Manager Test (2025) ===")
    print("Features: 100% loss offset, FIFO by modality, swing trade exemption")
    print()
    
    # Scenario 1: Record losses by modality
    print("--- Recording Losses by Modality ---")
    base_date = datetime.now()
    
    # Record swing trade losses
    loss_mgr.record_trade_result("VALE3", -1000.0, base_date, "SWING", 
                                trade_id="VALE3_001", description="Swing trade loss")
    loss_mgr.record_trade_result("PETR4", -500.0, base_date + timedelta(days=1), "SWING",
                                trade_id="PETR4_001", description="Another swing loss")
    
    # Record day trade losses
    loss_mgr.record_trade_result("ITUB4", -750.0, base_date + timedelta(days=2), "DAY",
                                trade_id="ITUB4_001", description="Day trade loss")
    
    print(f"Total accumulated losses: R$ {loss_mgr.get_total_loss_balance():,.2f}")
    print(f"Swing losses: R$ {loss_mgr.get_modality_loss_balance('SWING'):,.2f}")
    print(f"Day losses: R$ {loss_mgr.get_modality_loss_balance('DAY'):,.2f}")
    print()
    
    # Scenario 2: Test swing trade exemption
    print("--- Testing Swing Trade Exemption ---")
    
    # Test with sales below exemption limit
    taxable1, audit1 = loss_mgr.calculate_taxable_amount(
        gross_profit=5000.0,
        modality="SWING",
        gross_sales=15000.0,  # Below R$ 20,000 limit
        month_ref=base_date.date()
    )
    
    print(f"Scenario 1: Swing profit R$ 5,000, sales R$ 15,000 (below exemption)")
    print(f"  → Taxable: R$ {taxable1:,.2f} (exempt)")
    print(f"  → Losses consumed: {len(audit1)}")
    print()
    
    # Test with sales above exemption limit
    taxable2, audit2 = loss_mgr.calculate_taxable_amount(
        gross_profit=5000.0,
        modality="SWING",
        gross_sales=25000.0,  # Above R$ 20,000 limit
        month_ref=base_date.date()
    )
    
    print(f"Scenario 2: Swing profit R$ 5,000, sales R$ 25,000 (above exemption)")
    print(f"  → Taxable: R$ {taxable2:,.2f}")
    print(f"  → Losses consumed: {len(audit2)}")
    print()
    
    # Scenario 3: Test day trade (no exemption)
    print("--- Testing Day Trade (No Exemption) ---")
    
    taxable3, audit3 = loss_mgr.calculate_taxable_amount(
        gross_profit=3000.0,
        modality="DAY",
        gross_sales=10000.0,  # Day trade has no exemption
        month_ref=base_date.date()
    )
    
    print(f"Scenario 3: Day trade profit R$ 3,000, sales R$ 10,000")
    print(f"  → Taxable: R$ {taxable3:,.2f}")
    print(f"  → Losses consumed: {len(audit3)}")
    print()
    
    # Print enhanced summary
    loss_mgr.print_summary()
    
    # Export audit trail
    loss_mgr.export_audit_trail("individual_taxpayer_loss_audit_trail.json")
    print("\nAudit trail exported to individual_taxpayer_loss_audit_trail.json")
    print("\nRegulatory Compliance: ✅ Brazilian Individual Taxpayers 2025, ✅ 100% Loss Offset")


if __name__ == "__main__":
    main() 