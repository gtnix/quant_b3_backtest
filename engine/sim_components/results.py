from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from engine.performance_metrics import ComprehensivePerformanceAnalysis
from .types import SimulationResult


def calculate_performance_metrics(sim) -> None:
    if not sim.daily_portfolio_values:
        return
    all_metrics = sim.performance_metrics.calculate_all_metrics(
        portfolio_values=sim.daily_portfolio_values,
        daily_returns=sim.daily_returns,
        start_date=sim.start_date,
        end_date=sim.end_date,
    )
    sim.performance_metrics.total_return = all_metrics.get('total_return', 0.0)
    sim.performance_metrics.annualized_return = all_metrics.get('annualized_return', 0.0)
    sim.performance_metrics.sharpe_ratio = all_metrics.get('sharpe_ratio', 0.0)
    sim.performance_metrics.max_drawdown = all_metrics.get('max_drawdown', 0.0)
    sim.performance_metrics.total_trades = all_metrics.get('total_trades', 0)
    sim.performance_metrics.winning_trades = all_metrics.get('winning_trades', 0)
    sim.performance_metrics.losing_trades = all_metrics.get('losing_trades', 0)
    sim.performance_metrics.total_commission = all_metrics.get('total_commission', 0.0)
    sim.performance_metrics.total_taxes = sim.portfolio.total_taxes
    sim.performance_metrics.final_portfolio_value = sim.daily_portfolio_values[-1]
    sim.performance_metrics.initial_capital = sim.daily_portfolio_values[0]
    sim.performance_metrics.net_profit = sim.daily_portfolio_values[-1] - sim.daily_portfolio_values[0]
    sim.benchmark_metrics = all_metrics
    try:
        analysis = ComprehensivePerformanceAnalysis(sim.portfolio, strategy=sim.strategy)
        analysis.run_comprehensive_analysis(
            portfolio_values=sim.daily_portfolio_values,
            daily_returns=sim.daily_returns,
            start_date=sim.start_date,
        )
    except Exception:
        pass


def _safe_metrics(sim) -> Dict[str, Any]:
    pm = sim.performance_metrics
    return {
        'total_return': pm.total_return or 0.0,
        'annualized_return': pm.annualized_return or 0.0,
        'sharpe_ratio': pm.sharpe_ratio or 0.0,
        'max_drawdown': pm.max_drawdown or 0.0,
        'win_loss_ratio': pm.win_loss_ratio or 0.0,
        'profit_factor': pm.profit_factor or 0.0,
        'total_trades': pm.total_trades or 0,
        'winning_trades': pm.winning_trades or 0,
        'losing_trades': pm.losing_trades or 0,
        'avg_win': pm.avg_win or 0.0,
        'avg_loss': pm.avg_loss or 0.0,
        'largest_win': pm.largest_win or 0.0,
        'largest_loss': pm.largest_loss or 0.0,
        'total_commission': pm.total_commission or 0.0,
        'total_taxes': pm.total_taxes or 0.0,
        'net_profit': pm.net_profit or 0.0,
        'final_portfolio_value': pm.final_portfolio_value or 0.0,
        'initial_capital': pm.initial_capital or 0.0,
    }


def create_simulation_result(sim) -> SimulationResult:
    duration = (sim.simulation_end_time - sim.simulation_start_time).total_seconds()
    summary = _safe_metrics(sim)
    bm = sim.benchmark_metrics or {}
    return SimulationResult(
        total_return=summary['total_return'],
        sharpe_ratio=summary['sharpe_ratio'],
        max_drawdown=summary['max_drawdown'],
        win_loss_ratio=summary['win_loss_ratio'],
        total_trades=summary['total_trades'],
        winning_trades=summary['winning_trades'],
        losing_trades=summary['losing_trades'],
        final_portfolio_value=summary['final_portfolio_value'],
        initial_capital=summary['initial_capital'],
        total_commission=summary['total_commission'],
        total_taxes=summary['total_taxes'],
        daily_returns=sim.daily_returns.copy(),
        portfolio_values=sim.daily_portfolio_values.copy(),
        trade_log=sim.trade_log.copy(),
        simulation_duration=duration,
        start_date=sim.simulation_start_time,
        end_date=sim.simulation_end_time,
        benchmark_return=bm.get('benchmark_return', 0.0),
        excess_return=bm.get('excess_return', 0.0),
        information_ratio=bm.get('information_ratio', 0.0),
        beta=bm.get('beta', 0.0),
        alpha=bm.get('alpha', 0.0),
        tracking_error=bm.get('tracking_error', 0.0),
        rolling_correlation=bm.get('rolling_correlation', 0.0),
        benchmark_sharpe=bm.get('benchmark_sharpe', 0.0),
        benchmark_max_drawdown=bm.get('benchmark_max_drawdown', 0.0),
        benchmark_win_rate=bm.get('benchmark_win_rate', 0.0),
    )


def get_unified_fills_dataframe(sim) -> pd.DataFrame:
    if sim.unified_fills_df is None:
        try:
            sim.unified_fills_df = pd.DataFrame(sim.unified_fills) if sim.unified_fills else pd.DataFrame(
                columns=['timestamp','symbol','side','quantity','price','lot_type','rounding','tranche_notional_brl','trade_type','order_type','attempt_type','attempt_name']
            )
        except Exception:
            sim.unified_fills_df = pd.DataFrame()
    return sim.unified_fills_df


def get_performance_summary(sim) -> Dict[str, Any]:
    summary = _safe_metrics(sim)
    bm = sim.benchmark_metrics or {}
    summary.update({
        'benchmark_return': bm.get('benchmark_return', 0.0),
        'excess_return': bm.get('excess_return', 0.0),
        'information_ratio': bm.get('information_ratio', 0.0),
        'beta': bm.get('beta', 0.0),
        'alpha': bm.get('alpha', 0.0),
        'tracking_error': bm.get('tracking_error', 0.0),
        'rolling_correlation': bm.get('rolling_correlation', 0.0),
        'benchmark_sharpe': bm.get('benchmark_sharpe', 0.0),
        'benchmark_max_drawdown': bm.get('benchmark_max_drawdown', 0.0),
        'benchmark_win_rate': bm.get('benchmark_win_rate', 0.0),
        'benchmark_symbol': bm.get('benchmark_symbol', 'IBOV'),
    })
    return summary


def export_results(sim, filepath: str) -> None:
    try:
        results = {
            'simulation_info': {
                'strategy_name': sim.strategy.name,
                'initial_capital': sim.initial_capital,
                'start_date': sim.start_date.isoformat() if sim.start_date else None,
                'end_date': sim.end_date.isoformat() if sim.end_date else None,
                'simulation_duration_seconds': (
                    sim.simulation_end_time - sim.simulation_start_time
                ).total_seconds() if sim.simulation_start_time and sim.simulation_end_time else None,
            },
            'performance_metrics': get_performance_summary(sim),
            'daily_data': {
                'dates': [d.isoformat() for d in pd.date_range(
                    start=sim.start_date or pd.Timestamp.min,
                    end=sim.end_date or pd.Timestamp.max,
                    periods=len(sim.daily_portfolio_values),
                )],
                'portfolio_values': sim.daily_portfolio_values,
                'daily_returns': sim.daily_returns,
            },
            'trade_log': sim.trade_log,
        }
        import os as _os, json
        if not (_os.getenv('AUDIT_EXECUTIONS_ONLY', '1').lower() in ('1', 'true', 'yes')):
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
    except Exception:
        pass


def get_summary_data(sim) -> Dict[str, Any]:
    pm = _safe_metrics(sim)
    return {
        'strategy_name': sim.strategy.name,
        'initial_capital': sim.initial_capital,
        'final_portfolio_value': pm['final_portfolio_value'],
        'net_profit': pm['net_profit'],
        'total_return': pm['total_return'],
        'annualized_return': pm['annualized_return'],
        'sharpe_ratio': pm['sharpe_ratio'],
        'max_drawdown': pm['max_drawdown'],
        'win_loss_ratio': pm['win_loss_ratio'],
        'profit_factor': pm['profit_factor'],
        'total_trades': pm['total_trades'],
        'winning_trades': pm['winning_trades'],
        'losing_trades': pm['losing_trades'],
        'avg_win': pm['avg_win'],
        'avg_loss': pm['avg_loss'],
        'largest_win': pm['largest_win'],
        'largest_loss': pm['largest_loss'],
        'total_commission': pm['total_commission'],
        'total_taxes': pm['total_taxes'],
        'simulation_duration': (
            sim.simulation_end_time - sim.simulation_start_time
        ).total_seconds() if sim.simulation_start_time and sim.simulation_end_time else None,
    }


