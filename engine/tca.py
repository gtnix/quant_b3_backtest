"""
Transaction Cost Analysis (TCA) Module for Brazilian Market

Comprehensive transaction cost calculation for Brazilian stock market (B3):
- Brokerage fees with minimum charge enforcement
- B3 emolument fees (negotiation fees)
- Settlement fees (day trade vs swing trade)
- ISS (Service Tax) on brokerage fees
- Detailed cost breakdown and reporting

Compliance: B3 fee structure, Brazilian tax regulations

Author: Your Name
Date: 2024
"""

import yaml
import logging
from typing import Dict, Optional, Literal
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CostBreakdown:
    """
    Detailed breakdown of transaction costs.
    
    Attributes:
        brokerage_fee: Brokerage commission
        min_brokerage_applied: Whether minimum brokerage was applied
        emolument: B3 negotiation fee
        settlement_fee: Settlement fee (day trade or swing trade)
        iss_fee: ISS tax on brokerage fee
        total_costs: Sum of all costs
        cost_percentage: Total costs as percentage of order value
    """
    brokerage_fee: float
    min_brokerage_applied: bool
    emolument: float
    settlement_fee: float
    iss_fee: float
    total_costs: float
    cost_percentage: float


class TransactionCostAnalyzer:
    """
    Transaction Cost Analyzer for Brazilian market compliance.
    
    Features:
    - Loads cost parameters from configuration
    - Calculates comprehensive transaction costs
    - Supports both buy and sell operations
    - Handles day trade vs swing trade differentiation
    - Provides detailed cost breakdown
    - Validates cost parameters
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initialize TCA with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.cost_params = self.config['market']['costs']
        
        # Validate cost parameters
        self._validate_cost_parameters()
        
        logger.info("Transaction Cost Analyzer initialized with Brazilian market parameters")
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration with error handling.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file not found
            yaml.YAMLError: If config file is invalid
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            if 'market' not in config or 'costs' not in config['market']:
                raise ValueError("Configuration must contain 'market.costs' section")
            
            logger.info(f"Configuration loaded from {config_path}")
            return config
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML configuration: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def _validate_cost_parameters(self) -> None:
        """
        Validate cost parameters for logical consistency.
        
        Raises:
            ValueError: If parameters are invalid
        """
        costs = self.cost_params
        
        # Validate brokerage parameters
        if costs['brokerage_fee'] < 0:
            raise ValueError("Brokerage fee must be non-negative")
        
        if costs['min_brokerage'] < 0:
            raise ValueError("Minimum brokerage must be non-negative")
        
        # Validate B3 fees
        if costs['emolument'] < 0:
            raise ValueError("Emolument fee must be non-negative")
        
        if costs['settlement_day_trade'] < 0:
            raise ValueError("Day trade settlement fee must be non-negative")
        
        if costs['settlement_swing_trade'] < 0:
            raise ValueError("Swing trade settlement fee must be non-negative")
        
        # Validate ISS rate
        if costs['iss_rate'] < 0 or costs['iss_rate'] > 0.05:
            raise ValueError("ISS rate must be between 0 and 5% (0.05)")
        
        logger.info("Cost parameters validated successfully")
    
    def calculate_costs(self, order_value: float, is_buy: bool = True, 
                       trade_type: str = "swing_trade") -> CostBreakdown:
        """
        Calculate comprehensive transaction costs for Brazilian market.
        
        Args:
            order_value: Total order value in BRL
            is_buy: True for buy orders, False for sell orders
            trade_type: 'day_trade' or 'swing_trade'
            
        Returns:
            CostBreakdown object with detailed cost components
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if order_value <= 0:
            raise ValueError("Order value must be positive")
        
        if trade_type not in ['day_trade', 'swing_trade']:
            raise ValueError("Trade type must be 'day_trade' or 'swing_trade'")
        
        # Calculate brokerage fee with minimum enforcement
        brokerage_fee = order_value * self.cost_params['brokerage_fee']
        min_brokerage_applied = brokerage_fee < self.cost_params['min_brokerage']
        
        if min_brokerage_applied:
            brokerage_fee = self.cost_params['min_brokerage']
        
        # Calculate B3 emolument fee
        emolument = order_value * self.cost_params['emolument']
        
        # Calculate settlement fee based on trade type
        if trade_type == 'day_trade':
            settlement_fee = order_value * self.cost_params['settlement_day_trade']
        else:
            settlement_fee = order_value * self.cost_params['settlement_swing_trade']
        
        # Calculate ISS on brokerage fee
        iss_fee = brokerage_fee * self.cost_params['iss_rate']
        
        # Calculate total costs
        total_costs = brokerage_fee + emolument + settlement_fee + iss_fee
        
        # Calculate cost percentage
        cost_percentage = (total_costs / order_value) * 100 if order_value > 0 else 0
        
        # Create cost breakdown
        breakdown = CostBreakdown(
            brokerage_fee=brokerage_fee,
            min_brokerage_applied=min_brokerage_applied,
            emolument=emolument,
            settlement_fee=settlement_fee,
            iss_fee=iss_fee,
            total_costs=total_costs,
            cost_percentage=cost_percentage
        )
        
        logger.debug(f"Cost calculation completed: {breakdown}")
        return breakdown
    
    def get_cost_summary(self, order_value: float, is_buy: bool = True,
                        trade_type: str = "swing_trade") -> Dict:
        """
        Get comprehensive cost summary as dictionary.
        
        Args:
            order_value: Total order value in BRL
            is_buy: True for buy orders, False for sell orders
            trade_type: 'day_trade' or 'swing_trade'
            
        Returns:
            Dictionary with cost breakdown and metadata
        """
        breakdown = self.calculate_costs(order_value, is_buy, trade_type)
        
        return {
            'order_value': order_value,
            'is_buy': is_buy,
            'trade_type': trade_type,
            'costs': {
                'brokerage_fee': breakdown.brokerage_fee,
                'min_brokerage_applied': breakdown.min_brokerage_applied,
                'emolument': breakdown.emolument,
                'settlement_fee': breakdown.settlement_fee,
                'iss_fee': breakdown.iss_fee,
                'total_costs': breakdown.total_costs,
                'cost_percentage': breakdown.cost_percentage
            },
            'net_amount': order_value + (breakdown.total_costs if is_buy else -breakdown.total_costs),
            'calculation_timestamp': datetime.now().isoformat()
        }
    
    def compare_costs(self, order_value: float, trade_types: list = None) -> Dict:
        """
        Compare costs across different trade types.
        
        Args:
            order_value: Total order value in BRL
            trade_types: List of trade types to compare
            
        Returns:
            Dictionary with cost comparison
        """
        if trade_types is None:
            trade_types = ['day_trade', 'swing_trade']
        
        comparison = {}
        
        for trade_type in trade_types:
            buy_costs = self.calculate_costs(order_value, is_buy=True, trade_type=trade_type)
            sell_costs = self.calculate_costs(order_value, is_buy=False, trade_type=trade_type)
            
            comparison[trade_type] = {
                'buy': {
                    'total_costs': buy_costs.total_costs,
                    'cost_percentage': buy_costs.cost_percentage
                },
                'sell': {
                    'total_costs': sell_costs.total_costs,
                    'cost_percentage': sell_costs.cost_percentage
                },
                'round_trip': {
                    'total_costs': buy_costs.total_costs + sell_costs.total_costs,
                    'cost_percentage': buy_costs.cost_percentage + sell_costs.cost_percentage
                }
            }
        
        return comparison
    
    def get_cost_parameters(self) -> Dict:
        """
        Get current cost parameters for reference.
        
        Returns:
            Dictionary with current cost parameters
        """
        return self.cost_params.copy()
    
    def update_cost_parameters(self, new_params: Dict) -> None:
        """
        Update cost parameters (for testing or dynamic updates).
        
        Args:
            new_params: Dictionary with new cost parameters
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Store original parameters
        original_params = self.cost_params.copy()
        
        try:
            # Update parameters
            self.cost_params.update(new_params)
            
            # Validate updated parameters
            self._validate_cost_parameters()
            
            logger.info("Cost parameters updated successfully")
            
        except Exception as e:
            # Restore original parameters on error
            self.cost_params = original_params
            logger.error(f"Failed to update cost parameters: {str(e)}")
            raise


def main():
    """Example usage of Transaction Cost Analyzer."""
    try:
        # Initialize TCA
        tca = TransactionCostAnalyzer()
        
        # Example calculations
        order_value = 10000.0  # R$ 10,000
        
        print("=== Transaction Cost Analysis Example ===")
        print(f"Order Value: R$ {order_value:,.2f}")
        print()
        
        # Calculate costs for different scenarios
        scenarios = [
            ("Buy Swing Trade", True, "swing_trade"),
            ("Sell Swing Trade", False, "swing_trade"),
            ("Buy Day Trade", True, "day_trade"),
            ("Sell Day Trade", False, "day_trade")
        ]
        
        for scenario_name, is_buy, trade_type in scenarios:
            breakdown = tca.calculate_costs(order_value, is_buy, trade_type)
            
            print(f"--- {scenario_name} ---")
            print(f"Brokerage Fee: R$ {breakdown.brokerage_fee:.2f}")
            print(f"Emolument: R$ {breakdown.emolument:.2f}")
            print(f"Settlement Fee: R$ {breakdown.settlement_fee:.2f}")
            print(f"ISS Fee: R$ {breakdown.iss_fee:.2f}")
            print(f"Total Costs: R$ {breakdown.total_costs:.2f}")
            print(f"Cost Percentage: {breakdown.cost_percentage:.4f}%")
            print()
        
        # Cost comparison
        print("=== Cost Comparison ===")
        comparison = tca.compare_costs(order_value)
        for trade_type, costs in comparison.items():
            print(f"{trade_type.upper()}:")
            print(f"  Round-trip cost: R$ {costs['round_trip']['total_costs']:.2f} ({costs['round_trip']['cost_percentage']:.4f}%)")
        print()
        
        # Current parameters
        print("=== Current Cost Parameters ===")
        params = tca.get_cost_parameters()
        for param, value in params.items():
            print(f"{param}: {value}")
        
    except Exception as e:
        logger.error(f"Error in main example: {str(e)}")
        raise


if __name__ == "__main__":
    main() 