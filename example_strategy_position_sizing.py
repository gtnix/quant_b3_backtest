#!/usr/bin/env python3
"""
Example: How Strategies Control Position Sizing

This example shows how trading strategies control their own position sizing
through the signal generation process, rather than having it imposed by
the base class.

Author: Your Name
Date: 2024
"""

import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.base_strategy import BaseStrategy, TradingSignal, SignalType, TradeType
from engine.portfolio import EnhancedPortfolio


class ConservativeStrategy(BaseStrategy):
    """
    Conservative strategy that buys small positions.
    """
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[TradingSignal]:
        """Generate conservative signals with small position sizes."""
        signals = []
        current_price = market_data.get('current_price', 0)
        
        if current_price > 0:
            # Conservative: always buy small amounts
            quantity = 50  # Small position
            
            signal = TradingSignal(
                signal_type=SignalType.BUY,
                ticker=self.symbol,
                price=current_price,
                quantity=quantity,  # ← Strategy controls: small position
                confidence=0.6,
                trade_type=TradeType.SWING_TRADE
            )
            signals.append(signal)
        
        return signals
    
    def manage_risk(self, current_positions: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        return {'conservative': True}
    
    def execute_trade(self, signal: TradingSignal) -> bool:
        return True


class AggressiveStrategy(BaseStrategy):
    """
    Aggressive strategy that buys large positions.
    """
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[TradingSignal]:
        """Generate aggressive signals with large position sizes."""
        signals = []
        current_price = market_data.get('current_price', 0)
        
        if current_price > 0:
            # Aggressive: buy large amounts
            quantity = 500  # Large position
            
            signal = TradingSignal(
                signal_type=SignalType.BUY,
                ticker=self.symbol,
                price=current_price,
                quantity=quantity,  # ← Strategy controls: large position
                confidence=0.9,
                trade_type=TradeType.SWING_TRADE
            )
            signals.append(signal)
        
        return signals
    
    def manage_risk(self, current_positions: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        return {'aggressive': True}
    
    def execute_trade(self, signal: TradingSignal) -> bool:
        return True


class SmartStrategy(BaseStrategy):
    """
    Smart strategy that adjusts position size based on market conditions.
    """
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[TradingSignal]:
        """Generate smart signals with dynamic position sizing."""
        signals = []
        current_price = market_data.get('current_price', 0)
        
        if current_price > 0:
            # Smart: adjust position size based on price
            if current_price < 20.0:  # Very cheap
                quantity = 300  # Buy more
            elif current_price < 25.0:  # Cheap
                quantity = 200  # Buy medium
            elif current_price < 30.0:  # Normal
                quantity = 100  # Buy less
            else:  # Expensive
                quantity = 50   # Buy very little
            
            signal = TradingSignal(
                signal_type=SignalType.BUY,
                ticker=self.symbol,
                price=current_price,
                quantity=quantity,  # ← Strategy controls: dynamic sizing
                confidence=0.8,
                trade_type=TradeType.SWING_TRADE
            )
            signals.append(signal)
        
        return signals
    
    def manage_risk(self, current_positions: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        return {'smart': True}
    
    def execute_trade(self, signal: TradingSignal) -> bool:
        return True


def demonstrate_position_sizing():
    """Demonstrate how different strategies control their own position sizing."""
    
    print("🎯 Strategy Position Sizing Demonstration")
    print("=" * 50)
    
    # Create portfolio
    portfolio = EnhancedPortfolio("config/settings.yaml")
    
    # Test different strategies
    strategies = [
        ("Conservative", ConservativeStrategy(portfolio, "PETR4")),
        ("Aggressive", AggressiveStrategy(portfolio, "PETR4")),
        ("Smart", SmartStrategy(portfolio, "PETR4"))
    ]
    
    # Test with different prices
    test_prices = [15.0, 22.0, 28.0, 35.0]
    
    for strategy_name, strategy in strategies:
        print(f"\n📊 {strategy_name} Strategy:")
        print("-" * 30)
        
        for price in test_prices:
            market_data = {'current_price': price}
            signals = strategy.generate_signals(market_data)
            
            if signals:
                signal = signals[0]
                print(f"  Price: R$ {price:.2f} → Quantity: {signal.quantity} shares")
            else:
                print(f"  Price: R$ {price:.2f} → No signal")
    
    print("\n" + "=" * 50)
    print("✅ Key Point: Each strategy controls its own position sizing!")
    print("   - Conservative: Always buys 50 shares")
    print("   - Aggressive: Always buys 500 shares") 
    print("   - Smart: Adjusts quantity based on price")
    print("\n💡 The strategy logic determines position size, not the base class!")


if __name__ == "__main__":
    demonstrate_position_sizing() 