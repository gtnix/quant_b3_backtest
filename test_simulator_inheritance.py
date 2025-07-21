#!/usr/bin/env python3
"""
Test script to verify Simulator-BaseStrategy inheritance fixes.

This script tests that:
1. The simulator can properly validate BaseStrategy instances
2. Optional methods are handled gracefully
3. Required abstract methods are enforced
4. No import conflicts exist

Author: Your Name
Date: 2024
"""

import sys
import os
import logging
from datetime import datetime
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.base_strategy import BaseStrategy, TradingSignal, SignalType, TradeType
from engine.simulator import BacktestSimulator
from engine.portfolio import EnhancedPortfolio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestStrategy(BaseStrategy):
    """
    Simple test strategy that implements all required abstract methods.
    """
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[TradingSignal]:
        """Generate simple test signals with strategy-controlled position sizing."""
        signals = []
        
        # Strategy logic: decide how much to buy based on market conditions
        current_price = market_data.get('current_price', 0)
        if current_price > 0:
            # Strategy decides position size based on its logic
            if current_price < 25.0:  # Cheap price
                quantity = 200  # Strategy wants more shares
            elif current_price < 30.0:  # Medium price
                quantity = 100  # Strategy wants fewer shares
            else:  # Expensive price
                quantity = 50   # Strategy wants even fewer shares
            
            signal = TradingSignal(
                signal_type=SignalType.BUY,
                ticker=self.symbol,
                price=current_price,
                quantity=quantity,  # ← Strategy controls position sizing
                confidence=0.8,
                trade_type=TradeType.SWING_TRADE
            )
            signals.append(signal)
        
        return signals
    
    def manage_risk(self, current_positions: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simple risk management."""
        return {
            'max_position_size': 0.1,
            'stop_loss_pct': 0.05
        }
    
    def execute_trade(self, signal: TradingSignal) -> bool:
        """Simple trade execution."""
        logger.info(f"Executing trade: {signal}")
        return True


class MinimalStrategy(BaseStrategy):
    """
    Minimal strategy that only implements required abstract methods.
    Does not implement optional methods to test graceful handling.
    """
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[TradingSignal]:
        """Generate minimal test signals."""
        return []
    
    def manage_risk(self, current_positions: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Minimal risk management."""
        return {}
    
    def execute_trade(self, signal: TradingSignal) -> bool:
        """Minimal trade execution."""
        return True


def test_simulator_initialization():
    """Test that simulator can initialize with different strategy types."""
    logger.info("Testing simulator initialization...")
    
    # Test 1: Full strategy with all methods
    try:
        portfolio = EnhancedPortfolio("config/settings.yaml")
        full_strategy = TestStrategy(portfolio, "PETR4")
        simulator = BacktestSimulator(full_strategy, initial_capital=100000.0)
        logger.info("✓ Full strategy initialization successful")
    except Exception as e:
        logger.error(f"✗ Full strategy initialization failed: {e}")
        return False
    
    # Test 2: Minimal strategy with only required methods
    try:
        portfolio = EnhancedPortfolio("config/settings.yaml")
        minimal_strategy = MinimalStrategy(portfolio, "VALE3")
        simulator = BacktestSimulator(minimal_strategy, initial_capital=100000.0)
        logger.info("✓ Minimal strategy initialization successful")
    except Exception as e:
        logger.error(f"✗ Minimal strategy initialization failed: {e}")
        return False
    
    return True


def test_abstract_method_enforcement():
    """Test that abstract methods are properly enforced."""
    logger.info("Testing abstract method enforcement...")
    
    class InvalidStrategy(BaseStrategy):
        """Strategy missing required abstract methods."""
        
        def generate_signals(self, market_data: Dict[str, Any]) -> List[TradingSignal]:
            return []
        
        # Missing manage_risk and execute_trade methods
    
    try:
        portfolio = EnhancedPortfolio("config/settings.yaml")
        invalid_strategy = InvalidStrategy(portfolio, "ITUB4")
        simulator = BacktestSimulator(invalid_strategy, initial_capital=100000.0)
        logger.error("✗ Invalid strategy should have failed initialization")
        return False
    except ValueError as e:
        if "missing required abstract methods" in str(e):
            logger.info("✓ Abstract method enforcement working correctly")
            return True
        else:
            logger.error(f"✗ Unexpected error: {e}")
            return False
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        return False


def test_optional_method_handling():
    """Test that optional methods are handled gracefully."""
    logger.info("Testing optional method handling...")
    
    portfolio = EnhancedPortfolio("config/settings.yaml")
    minimal_strategy = MinimalStrategy(portfolio, "BBDC4")
    
    # Test that optional methods are not required
    optional_methods = [
        'validate_market_data',
        'reset_strategy'
    ]
    
    for method in optional_methods:
        if hasattr(minimal_strategy, method):
            logger.info(f"✓ Optional method {method} is available")
        else:
            logger.info(f"✓ Optional method {method} is missing (expected)")
    
    # Test that default methods are available
    default_methods = [
        'check_brazilian_market_constraints',
        'calculate_position_size'
    ]
    
    for method in default_methods:
        if hasattr(minimal_strategy, method):
            logger.info(f"✓ Default method {method} is available")
        else:
            logger.error(f"✗ Default method {method} is missing (unexpected)")
            return False
    
    return True


def test_import_conflicts():
    """Test that there are no import conflicts between PerformanceMetrics classes."""
    logger.info("Testing import conflicts...")
    
    try:
        # Import both PerformanceMetrics classes
        from engine.performance_metrics import PerformanceMetrics as PM1
        from engine.simulator import SimulationMetrics as PM2
        
        logger.info("✓ No import conflicts detected")
        return True
    except Exception as e:
        logger.error(f"✗ Import conflict detected: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting Simulator-BaseStrategy inheritance tests...")
    
    tests = [
        ("Import Conflicts", test_import_conflicts),
        ("Simulator Initialization", test_simulator_initialization),
        ("Abstract Method Enforcement", test_abstract_method_enforcement),
        ("Optional Method Handling", test_optional_method_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n--- Test Results ---")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests passed! Simulator-BaseStrategy inheritance is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 