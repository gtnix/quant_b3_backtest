"""
Enhanced Portfolio tracker with full Brazilian market automation
Handles positions, cash, and comprehensive Brazilian tax/fee implications

This module provides functionality to:
- Track portfolio positions and cash with automatic fee calculation
- Execute buy/sell orders with Modal-specific cost automation
- Handle Brazilian taxes with swing-trade exemption tracking
- Support both day-trade and swing-trade modes
- Maintain comprehensive trade history for analysis
- Full loss carryforward and T+2 settlement compliance

Author: Your Name
Date: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
import logging
import yaml

# Import new managers
from loss_manager import LossCarryforwardManager
from settlement_manager import SettlementManager

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


class Portfolio:
    """
    Enhanced portfolio manager with full Brazilian market automation.
    
    This class handles:
    - Automatic fee calculation (settlement, brokerage, ISS, dealing desk)
    - Brazilian tax calculation with swing-trade exemption tracking
    - Support for both day-trade and swing-trade modes
    - Comprehensive trade history and performance tracking
    """
    
    def __init__(self, initial_cash: float, fee_config: Optional[FeeConfig] = None):
        """
        Initialize the Portfolio with fee configuration.
        
        Args:
            initial_cash (float): Starting cash amount in BRL
            fee_config (Optional[FeeConfig]): Fee configuration, loads from settings.yaml if None
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        
        # Load fee configuration
        if fee_config is None:
            self.fee_config = self._load_fee_config()
        else:
            self.fee_config = fee_config
        
        # Positions structure: {ticker: {'shares': int, 'avg_price': float, 'purchase_date': datetime}}
        self.positions = {}
        
        # Trade history
        self.trades = []
        
        # Daily portfolio values for tracking performance
        self.daily_values = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.total_taxes_paid = 0.0
        self.total_fees_paid = 0.0
        
        # Swing trade exemption tracking (rolling calendar month)
        self.swing_sales_mtd = 0.0  # Monthly swing sales total
        self.current_month = datetime.now().month
        
        # Daily P&L tracking for day trades
        self.daily_pnl_by_asset = defaultdict(float)  # Track daily P&L per asset
        self.last_trade_date = None  # Track last trade date for daily reset
        
        # Initialize new managers
        self.loss_manager = LossCarryforwardManager()
        self.settlement_manager = SettlementManager(initial_cash=initial_cash)
        
        logger.info(f"Portfolio initialized with R$ {initial_cash:,.2f}")
        logger.info(f"Fee config loaded: Modal brokerage (R$ 0), "
                   f"Day trade tax {self.fee_config.day_trade_tax*100}%, "
                   f"Swing trade tax {self.fee_config.swing_trade_tax*100}%")
    
    def _load_fee_config(self) -> FeeConfig:
        """Load fee configuration from settings.yaml."""
        try:
            with open('config/settings.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            costs = config['market']['costs']
            taxes = config['taxes']
            
            return FeeConfig(
                emolument=costs['emolument'],
                settlement_day_trade=costs['settlement_day_trade'],
                settlement_swing_trade=costs['settlement_swing_trade'],
                brokerage_fee=costs['brokerage_fee'],
                min_brokerage=costs['min_brokerage'],
                iss_rate=costs['iss_rate'],
                swing_trade_tax=taxes['swing_trade'],
                day_trade_tax=taxes['day_trade'],
                exemption_limit=taxes['exemption_limit'],
                irrf_swing_rate=taxes.get('irrf_swing_rate', 0.00005),
                irrf_day_rate=taxes.get('irrf_day_rate', 0.01)
            )
        except Exception as e:
            logger.warning(f"Could not load fee config from settings.yaml: {e}")
            logger.info("Using default Modal fee configuration")
            return FeeConfig()
    
    def _calculate_fees(self, order_value: float, is_buy: bool) -> Dict[str, float]:
        """
        Calculate all fees for a trade (electronic orders only).
        
        Args:
            order_value (float): Total order value in BRL
            is_buy (bool): Whether this is a buy order
            
        Returns:
            Dict[str, float]: Breakdown of all fees
        """
        fees = {}
        
        # Emolumentos (B3 negotiation fee)
        fees['emolument'] = order_value * self.fee_config.emolument
        
        # Brokerage fee (0% for Modal electronic orders)
        brokerage_amount = max(
            order_value * self.fee_config.brokerage_fee,
            self.fee_config.min_brokerage
        )
        fees['brokerage'] = brokerage_amount
        
        # Feature 7: ISS 5% on Brokerage
        fees['iss'] = brokerage_amount * self.fee_config.iss_rate  # Why this matters: Models the municipal service tax on brokerage costs
        
        # Settlement fee (calculated in sell method based on day trade vs swing trade)
        fees['settlement'] = 0.0  # Will be calculated in sell method
        
        fees['total'] = sum(fees.values())
        
        return fees
    
    def _reset_monthly_tracking(self, current_date: datetime):
        """Reset monthly tracking if we've moved to a new month."""
        if current_date.month != self.current_month:
            self.swing_sales_mtd = 0.0
            self.current_month = current_date.month
            self.loss_manager.reset_monthly_tracking(current_date.month)
            logger.info(f"Reset monthly tracking for month {current_date.month}")
    
    def _reset_daily_tracking(self, current_date: datetime):
        """Reset daily P&L tracking if we've moved to a new day."""
        if self.last_trade_date is None or current_date.date() != self.last_trade_date:
            self.daily_pnl_by_asset.clear()
            self.last_trade_date = current_date.date()
            logger.info(f"Reset daily P&L tracking for date {current_date.date()}")
    
    def buy(self, ticker: str, shares: int, price: float, date: datetime) -> Tuple[bool, str]:
        """
        Execute a buy order with automatic fee calculation (electronic only).
        
        Args:
            ticker (str): Stock ticker symbol
            shares (int): Number of shares to buy
            price (float): Price per share
            date (datetime): Date of the trade
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        # Process settlement queue first
        self.settlement_manager.process_settlements(date.date())
        
        order_value = shares * price
        fees = self._calculate_fees(order_value, is_buy=True)
        total_cost = order_value + fees['total']
        
        # Check if we have enough settled cash
        available_cash = self.settlement_manager.get_available_cash(date.date())
        if total_cost > available_cash:
            return False, f"Insufficient settled funds. Need R$ {total_cost:,.2f}, have R$ {available_cash:,.2f}"
        
        # Schedule T+2 settlement
        self.settlement_manager.schedule_trade(
            trade_date=date.date(),
            amount=total_cost,
            is_buy=True,
            description=f'Buy {shares} {ticker} @ R$ {price:.2f}'
        )
        
        # Update cash immediately for tracking (but not settled_cash)
        self.cash -= total_cost
        
        # Update or create position
        if ticker in self.positions:
            # Update existing position (average price calculation)
            current = self.positions[ticker]
            total_shares = current['shares'] + shares
            
            # Calculate new average price
            avg_price = ((current['shares'] * current['avg_price']) + (shares * price)) / total_shares
            
            self.positions[ticker] = {
                'shares': total_shares,
                'avg_price': avg_price,
                'purchase_date': current['purchase_date']  # Keep original purchase date
            }
            
            logger.info(f"Updated position {ticker}: {shares} shares at R$ {price:.2f}, "
                       f"new avg: R$ {avg_price:.2f}, total shares: {total_shares}")
        else:
            # Create new position
            self.positions[ticker] = {
                'shares': shares,
                'avg_price': price,
                'purchase_date': date
            }
            
            logger.info(f"New position {ticker}: {shares} shares at R$ {price:.2f}")
        
        # Record the trade
        trade_record = {
            'date': date,
            'ticker': ticker,
            'action': 'BUY',
            'shares': shares,
            'price': price,
            'order_value': order_value,
            'fees': fees,
            'total_cost': total_cost,
            'cash_after': self.cash
        }
        self.trades.append(trade_record)
        
        self.total_trades += 1
        self.total_fees_paid += fees['total']
        
        logger.info(f"Buy order executed: {shares} shares of {ticker} at R$ {price:.2f} "
                   f"Fees: R$ {fees['total']:.2f}")
        
        return True, f"Buy order executed: {shares} shares of {ticker} at R$ {price:.2f}"
    
    def sell(self, ticker: str, shares: int, price: float, date: datetime) -> Tuple[bool, str]:
        """
        Execute a sell order with automatic fee and tax calculation (electronic only).
        
        Args:
            ticker (str): Stock ticker symbol
            shares (int): Number of shares to sell
            price (float): Price per share
            date (datetime): Date of the trade
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        # Process settlement queue first
        self.settlement_manager.process_settlements(date.date())
        
        # Reset daily tracking if needed
        self._reset_daily_tracking(date)
        
        # Reset monthly tracking if needed
        self._reset_monthly_tracking(date)
        
        # Check if we have the position
        if ticker not in self.positions:
            return False, f"No position in {ticker}"
        
        position = self.positions[ticker]
        
        # Check if we have enough shares
        if shares > position['shares']:
            return False, f"Insufficient shares. Have {position['shares']}, trying to sell {shares}"
        
        # Calculate gross profit/loss
        gross_profit = shares * (price - position['avg_price'])
        gross_proceeds = shares * price
        
        # Determine if this is a day trade or swing trade
        days_held = (date - position['purchase_date']).days
        is_daytrade = days_held == 0
        
        # Calculate fees (including settlement fee)
        fees = self._calculate_fees(gross_proceeds, is_buy=False)
        
        # Add settlement fee based on trade type
        settlement_rate = (self.fee_config.settlement_day_trade if is_daytrade 
                          else self.fee_config.settlement_swing_trade)
        fees['settlement'] = gross_proceeds * settlement_rate
        
        # Feature 3: Swing IRRF 0.005%
        if not is_daytrade:
            fees['irrf'] = gross_proceeds * self.fee_config.irrf_swing_rate  # Why this matters: Models mandatory withholding on swing-trade settlements
        else:
            fees['irrf'] = 0.0  # Will be calculated later for day trades
        
        fees['total'] = sum(fees.values())
        
        # Initialize tax variables
        tax_owed = 0.0
        exempt_amount = 0.0
        irrf_credit = 0.0
        
        if gross_profit > 0:  # Only pay tax on profits
            if is_daytrade:
                # Feature 4: Day-Trade CG Tax 20% on consolidated daily P&L
                self.daily_pnl_by_asset[ticker] += gross_profit  # Add to daily P&L for this asset
                consolidated_pnl = self.daily_pnl_by_asset[ticker]
                
                # Apply loss carryforward
                if self.loss_manager.get_asset_loss_balance(ticker) > 0:
                    consolidated_pnl -= self.loss_manager.get_asset_loss_balance(ticker)
                    # Loss application is handled by the manager
                
                if consolidated_pnl > 0:
                    tax_owed = consolidated_pnl * self.fee_config.day_trade_tax  # Why this matters: Consolidates same-day trades before applying day-trade tax
                    exempt_amount = 0.0
                else:
                    tax_owed = 0.0
                    # Add to loss carryforward
                    self.loss_manager.record_trade_result(ticker, consolidated_pnl, 
                                                        date.strftime('%Y-%m-%d'), 'day_trade')  # Why this matters: Allows past losses to offset future gains per tax rules
                
                # Feature 5: Day-Trade IRRF 1%
                if consolidated_pnl > 0:
                    irrf_credit = consolidated_pnl * self.fee_config.irrf_day_rate  # Why this matters: Captures the advance withholding on day-trade profits
                    fees['irrf'] = irrf_credit
                    fees['total'] = sum(fees.values())
                
            else:
                # Feature 2: R$20,000 Exemption
                remaining_exemption = max(0, self.fee_config.exemption_limit - self.swing_sales_mtd)  # Why this matters: Applies the legal monthly exemption before calculating tax
                exempt_amount = min(remaining_exemption, gross_proceeds)
                taxable_proceeds = gross_proceeds - exempt_amount
                taxable_profit = gross_profit * (taxable_proceeds / gross_proceeds) if gross_proceeds > 0 else 0
                
                # Apply loss carryforward
                if self.loss_manager.get_asset_loss_balance(ticker) > 0:
                    taxable_profit -= self.loss_manager.get_asset_loss_balance(ticker)
                    self.loss_manager.record_trade_result(ticker, taxable_profit, 
                                                        date.strftime('%Y-%m-%d'), 'swing_trade')
                
                # Feature 1: Swing CG Tax 15%
                if taxable_profit > 0:
                    tax_owed = taxable_profit * self.fee_config.swing_trade_tax  # Why this matters: Ensures swing-trade gains over the exemption limit are taxed correctly
                else:
                    tax_owed = 0.0
                    # Add to loss carryforward
                    self.loss_manager.record_trade_result(ticker, taxable_profit, 
                                                        date.strftime('%Y-%m-%d'), 'swing_trade')
                
                # Update monthly swing sales tracking
                self.swing_sales_mtd += gross_proceeds
        
        # Calculate net proceeds
        net_proceeds = gross_proceeds - fees['total'] - tax_owed
        
        # Schedule T+2 settlement for sell proceeds
        self.settlement_manager.schedule_trade(
            trade_date=date.date(),
            amount=net_proceeds,
            is_buy=False,
            description=f'Sell {shares} {ticker} @ R$ {price:.2f}'
        )
        
        # Update cash immediately for tracking (but not settled_cash)
        self.cash += net_proceeds
        
        # Update position
        position['shares'] -= shares
        
        # Remove position if no shares left
        if position['shares'] == 0:
            del self.positions[ticker]
            logger.info(f"Closed position {ticker}")
        else:
            logger.info(f"Reduced position {ticker}: sold {shares} shares, "
                       f"remaining: {position['shares']} shares")
        
        # Record the trade
        trade_record = {
            'date': date,
            'ticker': ticker,
            'action': 'SELL',
            'shares': shares,
            'price': price,
            'order_value': gross_proceeds,
            'fees': fees,
            'gross_profit': gross_profit,
            'tax_owed': tax_owed,
            'exempt_amount': exempt_amount,
            'net_proceeds': net_proceeds,
            'days_held': days_held,
            'is_daytrade': is_daytrade,
            'cash_after': self.cash,
            'swing_sales_mtd': self.swing_sales_mtd,
            'loss_balance': self.loss_manager.get_asset_loss_balance(ticker),
            'consolidated_pnl': self.daily_pnl_by_asset.get(ticker, 0) if is_daytrade else None
        }
        self.trades.append(trade_record)
        
        # Update performance metrics
        self.total_trades += 1
        if gross_profit > 0:
            self.winning_trades += 1
        
        self.total_profit += gross_profit
        self.total_taxes_paid += tax_owed
        self.total_fees_paid += fees['total']
        
        trade_type = "DAY TRADE" if is_daytrade else "SWING TRADE"
        logger.info(f"Sell order executed: {shares} shares of {ticker} at R$ {price:.2f} "
                   f"({trade_type}), Profit: R$ {gross_profit:.2f}, "
                   f"Tax: R$ {tax_owed:.2f}, Fees: R$ {fees['total']:.2f}")
        
        return True, f"Sell order executed: {shares} shares of {ticker} at R$ {price:.2f}"
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value using settled cash."""
        positions_value = sum(
            self.positions[ticker]['shares'] * prices.get(ticker, 0)
            for ticker in self.positions
        )
        return self.settlement_manager.settled_cash + positions_value
    
    def get_position_summary(self) -> Dict[str, Dict]:
        """Get summary of all current positions."""
        summary = {}
        for ticker, position in self.positions.items():
            summary[ticker] = {
                'shares': position['shares'],
                'avg_price': position['avg_price'],
                'purchase_date': position['purchase_date'],
                'days_held': (datetime.now() - position['purchase_date']).days
            }
        return summary
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive portfolio performance metrics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_profit': 0.0,
                'total_taxes_paid': 0.0,
                'total_fees_paid': 0.0,
                'net_profit': 0.0,
                'swing_sales_mtd': self.swing_sales_mtd
            }
        
        # Calculate metrics
        win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        net_profit = self.total_profit - self.total_taxes_paid - self.total_fees_paid
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'total_profit': self.total_profit,
            'total_taxes_paid': self.total_taxes_paid,
            'total_fees_paid': self.total_fees_paid,
            'net_profit': net_profit,
            'swing_sales_mtd': self.swing_sales_mtd
        }
    
    def record_daily_value(self, date: datetime, prices: Dict[str, float]):
        """Record daily portfolio value for performance tracking."""
        portfolio_value = self.get_portfolio_value(prices)
        
        daily_record = {
            'date': date,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions_value': portfolio_value - self.cash,
            'num_positions': len(self.positions),
            'swing_sales_mtd': self.swing_sales_mtd
        }
        
        self.daily_values.append(daily_record)
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as a pandas DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)
    
    def get_daily_values(self) -> pd.DataFrame:
        """Get daily portfolio values as a pandas DataFrame."""
        if not self.daily_values:
            return pd.DataFrame()
        
        return pd.DataFrame(self.daily_values)
    
    def print_summary(self):
        """Print a comprehensive summary of the current portfolio state."""
        print("\n" + "="*60)
        print("PORTFOLIO SUMMARY")
        print("="*60)
        print(f"Cash (including unsettled): R$ {self.cash:,.2f}")
        print(f"Settled cash (T+2): R$ {self.settlement_manager.settled_cash:,.2f}")
        print(f"Number of positions: {len(self.positions)}")
        print(f"Swing sales this month: R$ {self.swing_sales_mtd:,.2f}")
        
        if self.positions:
            print("\nCurrent Positions:")
            for ticker, position in self.positions.items():
                days_held = (datetime.now() - position['purchase_date']).days
                loss_carryforward = self.loss_manager.get_asset_loss_balance(ticker)
                print(f"  {ticker}: {position['shares']} shares @ R$ {position['avg_price']:.2f} "
                      f"(held {days_held} days, loss carryforward: R$ {loss_carryforward:.2f})")
        
        # Performance metrics
        metrics = self.get_performance_metrics()
        print(f"\nPerformance:")
        print(f"  Total trades: {metrics['total_trades']}")
        print(f"  Win rate: {metrics['win_rate']:.1f}%")
        print(f"  Total profit: R$ {metrics['total_profit']:,.2f}")
        print(f"  Taxes paid: R$ {metrics['total_taxes_paid']:,.2f}")
        print(f"  Fees paid: R$ {metrics['total_fees_paid']:,.2f}")
        print(f"  Net profit: R$ {metrics['net_profit']:,.2f}")
        
        # Loss carryforward summary
        loss_summary = self.loss_manager.get_loss_summary()
        print(f"\nLoss Carryforward:")
        print(f"  Total cumulative loss: R$ {loss_summary['total_cumulative_loss']:,.2f}")
        print(f"  Assets with losses: {loss_summary['assets_with_losses']}")
        
        # Settlement summary
        settlement_summary = self.settlement_manager.get_settlement_summary()
        print(f"\nSettlement Status:")
        print(f"  Pending settlements: {settlement_summary['pending_settlements']}")
        print(f"  Unsettled cash: R$ {settlement_summary['unsettled_cash']:,.2f}")
        
        print("="*60)


def main():
    """Example usage demonstrating both day-trade and swing-trade flows."""
    
    # Initialize portfolio with R$ 100,000
    portfolio = Portfolio(initial_cash=100000)
    
    print("=== Enhanced Portfolio Simulation (Modal Broker) ===")
    
    # Example 1: Day trade flow
    print("\n--- DAY TRADE EXAMPLE ---")
    
    # Buy VALE3
    success, msg = portfolio.buy(
        ticker="VALE3",
        shares=100,
        price=50.0,
        date=datetime.now()
    )
    print(f"Buy VALE3: {msg}")
    
    # Sell VALE3 same day (day trade)
    success, msg = portfolio.sell(
        ticker="VALE3",
        shares=100,
        price=52.0,
        date=datetime.now()
    )
    print(f"Sell VALE3 (day trade): {msg}")
    
    # Example 2: Swing trade flow
    print("\n--- SWING TRADE EXAMPLE ---")
    
    # Buy PETR4
    success, msg = portfolio.buy(
        ticker="PETR4",
        shares=200,
        price=30.0,
        date=datetime.now()
    )
    print(f"Buy PETR4: {msg}")
    
    # Sell PETR4 next day (swing trade)
    tomorrow = datetime.now() + timedelta(days=1)
    success, msg = portfolio.sell(
        ticker="PETR4",
        shares=200,
        price=31.0,
        date=tomorrow
    )
    print(f"Sell PETR4 (swing trade): {msg}")
    
    # Example 3: Electronic order (Modal web/app)
    print("\n--- ELECTRONIC ORDER EXAMPLE ---")
    
    success, msg = portfolio.buy(
        ticker="ITUB4",
        shares=150,
        price=35.0,
        date=datetime.now()
    )
    print(f"Buy ITUB4 (electronic): {msg}")
    
    # Print comprehensive portfolio summary
    portfolio.print_summary()
    
    # Show trade history
    print("\n--- TRADE HISTORY ---")
    trades_df = portfolio.get_trade_history()
    if not trades_df.empty:
        for _, trade in trades_df.iterrows():
            print(f"{trade['date'].strftime('%Y-%m-%d %H:%M')} | "
                  f"{trade['action']} {trade['ticker']} | "
                  f"{trade['shares']} shares @ R$ {trade['price']:.2f} | "
                  f"Fees: R$ {trade['fees']['total']:.2f}")


if __name__ == "__main__":
    main() 