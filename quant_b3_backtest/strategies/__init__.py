"""
Trading Strategies Package

This package contains various trading strategies for the Brazilian market backtesting system.
Each strategy inherits from BaseStrategy and implements specific trading logic.

Available Strategies:
- FuzzyFajutoStrategy: Strict day-trading strategy with FuzzyFajuto methodology
- EnhancedFuzzyFajutoStrategy: Enhanced version with three-attempt execution system

Strategy Factory:
- create_fuzzy_fajuto(): Factory function for creating FuzzyFajuto with different profiles

Author: Quantitative Trading Specialist
Date: 2025
"""

from .fuzzy_fajuto_strategy import FuzzyFajutoStrategy
from engine.base_strategy import StrategyConfig, StrategyContext
from typing import Optional

def create_fuzzy_fajuto(cfg: StrategyConfig, ctx: StrategyContext, 
                       profile: str = "default", config_file: str = None) -> FuzzyFajutoStrategy:
    """
    Create FuzzyFajuto strategy with specific profile or custom config file.
    
    Args:
        cfg: Strategy configuration
        ctx: Strategy context
        profile: Configuration profile name (default, conservative, aggressive, etc.)
        config_file: Custom config file path (overrides profile)
        
    Returns:
        FuzzyFajutoStrategy instance
        
    Examples:
        # Use default profile
        strategy = create_fuzzy_fajuto(cfg, ctx)
        
        # Use specific profile
        strategy = create_fuzzy_fajuto(cfg, ctx, profile="conservative")
        
        # Use custom config file
        strategy = create_fuzzy_fajuto(cfg, ctx, config_file="my_custom_config.yaml")
    """
    
    # Determine config file to use
    if config_file is not None:
        # Use custom config file directly
        final_config_file = config_file
    elif profile == "default":
        # Use the default enhanced_strategy_config.yaml
        final_config_file = None  # Will use default in strategy
    else:
        # Use profile-based config file
        final_config_file = f"profiles/fuzzy_fajuto_{profile}.yaml"
    
    # Create and return strategy instance
    return FuzzyFajutoStrategy(cfg, ctx, config_file=final_config_file)

__all__ = ['FuzzyFajutoStrategy', 'create_fuzzy_fajuto'] 