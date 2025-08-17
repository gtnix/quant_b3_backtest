"""
Simulation components for the BacktestSimulator.

This package contains cohesive helpers extracted from the monolithic
`engine/simulator.py` to improve readability and maintainability while
preserving all public APIs and behavior.

All helpers are designed to accept the active simulator instance as the first
argument when they need access to simulator state. This keeps dependencies
explicit and avoids hidden globals.
"""

from .types import SimulationResult, SimulationMetrics
from .io import SimulationDataPortal, SimulationBroker
from .config import load_config, setup_logging
from .warmup import get_warmup_bars, calculate_calendar_days_for_trading_days
from .daycycle import (
    process_end_of_trading_day,
    prepare_market_data,
    load_sgs_data_for_date,
    load_ibov_data_for_date,
    calculate_selic_cdi_spread,
    classify_interest_rate_environment,
    classify_inflation_environment,
)
# Deprecated trade shims removed to avoid duplication; authoritative implementations live in engine.simulator
# Keep import facade for backward compatibility if some modules still reference these names.
def execute_trade(*args, **kwargs):  # pragma: no cover - deprecated facade
    raise ImportError("execute_trade is deprecated; use engine.simulator.BacktestSimulator methods")

def append_unified_fill(*args, **kwargs):  # pragma: no cover - deprecated facade
    raise ImportError("append_unified_fill is deprecated; use engine.simulator.BacktestSimulator methods")

def get_tranche_notional_brl(*args, **kwargs):  # pragma: no cover - deprecated facade
    raise ImportError("get_tranche_notional_brl is deprecated; use engine.simulator.BacktestSimulator methods")
from .results import (
    calculate_performance_metrics,
    create_simulation_result,
    get_unified_fills_dataframe,
    get_performance_summary,
    export_results,
    get_summary_data,
)

__all__ = [
    # Types
    'SimulationResult', 'SimulationMetrics',
    # IO
    'SimulationDataPortal', 'SimulationBroker',
    # Config/logging
    'load_config', 'setup_logging',
    # Warmup
    'get_warmup_bars', 'calculate_calendar_days_for_trading_days',
    # Day cycle
    'process_end_of_trading_day', 'prepare_market_data', 'load_sgs_data_for_date',
    'load_ibov_data_for_date', 'calculate_selic_cdi_spread',
    'classify_interest_rate_environment', 'classify_inflation_environment',
    # Trade
    'execute_trade', 'append_unified_fill', 'get_tranche_notional_brl',
    # Results
    'calculate_performance_metrics', 'create_simulation_result',
    'get_unified_fills_dataframe', 'get_performance_summary',
    'export_results', 'get_summary_data',
]


