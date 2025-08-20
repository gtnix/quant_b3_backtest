"""
Test Enhanced Pairing Configuration System

This test validates the new configuration-driven pairing system
and enhanced logging functionality.
"""

import pytest
from datetime import datetime, date
import types
from typing import Dict, Any, List

from strategies.fuzzy_fajuto_strategy import FuzzyFajutoStrategy
from engine.base_strategy import StrategyConfig, StrategyContext, OrderSide, OrderIntent, OrderType


class MockLogger:
    def __init__(self):
        self.messages = []
    
    def info(self, msg): 
        self.messages.append(('INFO', msg))
    
    def warning(self, msg): 
        self.messages.append(('WARNING', msg))
    
    def debug(self, msg): 
        self.messages.append(('DEBUG', msg))
    
    def error(self, msg): 
        self.messages.append(('ERROR', msg))


class MockPortfolio:
    def get_portfolio_value(self):
        return 1_000_000.0


def make_strategy_with_config(config_data: Dict[str, Any]) -> FuzzyFajutoStrategy:
    """Create a strategy instance with specific configuration."""
    cfg = StrategyConfig(
        universe=["BUY1", "BUY2", "SELL1", "SELL2"], 
        warmup_bars=1, 
        risk_tolerance=0.01, 
        max_position_size=0.1
    )
    
    logger = MockLogger()
    ctx = StrategyContext(
        data_portal=None, 
        portfolio=MockPortfolio(), 
        broker=None, 
        market_rules=None, 
        logger=logger,
        metadata={'config': config_data}  # Pass config through metadata
    )
    
    strat = FuzzyFajutoStrategy(cfg, ctx)
    return strat


def make_buffer_record(side: OrderSide, fuzzy: float, symbol: str, timestamp: datetime) -> Dict[str, Any]:
    """Create a buffer record for testing pairing logic."""
    prices = {
        'market': 10.0,
        'limit_alpha': 10.0,
        'limit_beta': 10.0, 
        'limit_gamma': 10.0
    }
    quantities = {
        'market': 300,
        'limit_alpha': 300,
        'limit_beta': 200,
        'limit_gamma': 100
    }
    
    # Create a mock bar
    bar = types.SimpleNamespace(
        timestamp=timestamp, 
        open=10.0, 
        high=10.5, 
        low=9.5, 
        close=10.0
    )
    
    return {
        'side': side,
        'fuzzy': fuzzy,
        'prices': prices,
        'qty': quantities,
        'lot_size': 100,
        'bar': bar,
        'atr': 0.5,
        'signal': fuzzy,
    }


def test_pairing_configuration_loading():
    """Test that pairing configuration is loaded correctly from settings."""
    
    # Test with pairing enabled
    config_enabled = {
        'strategy': {
            'pairing': {
                'enabled': True,
                'mode': 'bidirectional',
                'strict_neutrality': True,
                'allow_partial_pairs': False,
                'min_signal_strength': 1.8,
                'logging': {
                    'enabled': True,
                    'log_pair_formation': True,
                    'log_rejected_signals': True
                }
            }
        }
    }
    
    strategy = make_strategy_with_config(config_enabled)
    
    # Validate configuration loading
    assert strategy.RISK_PAIR_MATCHING == True
    assert strategy.PAIRING_MODE == 'bidirectional'
    assert strategy.PAIRING_STRICT_NEUTRALITY == True
    assert strategy.PAIRING_ALLOW_PARTIAL == False
    assert strategy.PAIRING_MIN_SIGNAL_STRENGTH == 1.8
    assert strategy.PAIRING_LOG_ENABLED == True
    assert strategy.PAIRING_LOG_FORMATION == True
    assert strategy.PAIRING_LOG_REJECTED == True
    
    # Check logger messages
    logger_messages = [msg[1] for msg in strategy.context.logger.messages if msg[0] == 'INFO']
    config_messages = [msg for msg in logger_messages if 'PAIRING: Configuration loaded' in msg]
    assert len(config_messages) > 0, "Configuration loading should be logged"


def test_pairing_configuration_disabled():
    """Test behavior when pairing is disabled via configuration."""
    
    config_disabled = {
        'strategy': {
            'pairing': {
                'enabled': False,
                'logging': {'enabled': True}
            }
        }
    }
    
    strategy = make_strategy_with_config(config_disabled)
    
    # Validate that pairing is disabled
    assert strategy.RISK_PAIR_MATCHING == False


def test_pairing_configuration_defaults():
    """Test that defaults are applied when configuration is missing."""
    
    # Empty configuration - should use defaults
    config_empty = {}
    
    strategy = make_strategy_with_config(config_empty)
    
    # Validate defaults
    assert strategy.RISK_PAIR_MATCHING == True  # Default enabled
    assert strategy.PAIRING_MODE == 'bidirectional'
    assert strategy.PAIRING_STRICT_NEUTRALITY == True
    assert strategy.PAIRING_ALLOW_PARTIAL == False
    assert strategy.PAIRING_LOG_ENABLED == True


def test_enhanced_pairing_logging():
    """Test the enhanced logging system during pairing execution."""
    
    config_with_logging = {
        'strategy': {
            'pairing': {
                'enabled': True,
                'mode': 'bidirectional',
                'logging': {'enabled': True, 'log_pair_formation': True}
            }
        }
    }
    
    strategy = make_strategy_with_config(config_with_logging)
    trading_date = date(2025, 8, 20)
    timestamp = datetime(2025, 8, 20, 13, 0, 0)
    
    # Setup test scenario
    strategy._neutral_buffer[trading_date] = {
        'LONG1': make_buffer_record(OrderSide.BUY, 2.0, 'LONG1', timestamp),
        'SHORT1': make_buffer_record(OrderSide.SELL, -2.0, 'SHORT1', timestamp),
    }
    
    # Execute pairing
    order_intents = list(strategy._neutral_emit_for_day(trading_date))
    
    # Check that enhanced logging occurred
    logger_messages = [msg[1] for msg in strategy.context.logger.messages if msg[0] == 'INFO']
    
    # Should have processing start message
    processing_messages = [msg for msg in logger_messages if 'PAIRING: Processing' in msg]
    assert len(processing_messages) > 0, "Should log pairing processing start"
    
    # Should have signals summary
    signals_messages = [msg for msg in logger_messages if 'BUY signals:' in msg or 'SELL signals:' in msg]
    assert len(signals_messages) > 0, "Should log signal summary"
    
    # Should have results summary
    results_messages = [msg for msg in logger_messages if 'PAIRING: Generated' in msg]
    assert len(results_messages) > 0, "Should log pairing results"
    
    # Should have balance status
    balance_messages = [msg for msg in logger_messages if 'Perfect balance achieved' in msg or 'Unbalanced execution' in msg]
    assert len(balance_messages) > 0, "Should log balance status"


def test_pairing_mode_validation():
    """Test different pairing mode configurations."""
    
    # Test bidirectional mode
    config_bidirectional = {
        'strategy': {
            'pairing': {
                'enabled': True,
                'mode': 'bidirectional',
                'logging': {'enabled': True}
            }
        }
    }
    
    strategy = make_strategy_with_config(config_bidirectional)
    assert strategy.PAIRING_MODE == 'bidirectional'
    
    # Test with different mode
    config_long_only = {
        'strategy': {
            'pairing': {
                'enabled': True,
                'mode': 'long_only',
                'logging': {'enabled': True}
            }
        }
    }
    
    strategy = make_strategy_with_config(config_long_only)
    assert strategy.PAIRING_MODE == 'long_only'


def test_min_signal_strength_configuration():
    """Test minimum signal strength configuration."""
    
    config_custom_strength = {
        'strategy': {
            'pairing': {
                'enabled': True,
                'min_signal_strength': 2.0,
                'logging': {'enabled': True}
            }
        }
    }
    
    strategy = make_strategy_with_config(config_custom_strength)
    assert strategy.PAIRING_MIN_SIGNAL_STRENGTH == 2.0
