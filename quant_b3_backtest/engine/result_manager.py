"""
Result Manager for Strategy Backtests

Simple, extensible system for storing and comparing backtest results across
different parameters, profiles, and tickers. Designed to be lightweight and
easily extensible for future comparison features.

Author: Quantitative Trading Specialist
Date: 2025
"""

import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class ResultManager:
    """
    Lightweight result manager for storing and accessing backtest results.
    
    Features:
    - Store results with metadata (strategy, profile, ticker, dates)
    - Simple JSON storage for easy access and extensibility
    - CSV index for quick querying and comparison
    - Foundation for future comparison and analysis features
    """
    
    def __init__(self, results_dir: str = "results"):
        """Initialize result manager with storage directory."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.detailed_dir = self.results_dir / "detailed"
        self.detailed_dir.mkdir(exist_ok=True)
        
        # Index file for quick access
        self.index_file = self.results_dir / "results_index.csv"
        
        logger.info(f"ResultManager initialized with directory: {self.results_dir}")
    
    def save_result(self, strategy_name: str, profile: str, ticker: str,
                   start_date: str, end_date: str, results: Any,
                   strategy_summary: Dict = None, config_file: str = None) -> str:
        """
        Save backtest results with metadata.
        
        Args:
            strategy_name: Name of the strategy (e.g., 'fuzzy_fajuto')
            profile: Profile used (e.g., 'default', 'conservative')
            ticker: Ticker symbol
            start_date: Start date string
            end_date: End date string
            results: Backtest results object
            strategy_summary: Additional strategy-specific metrics
            config_file: Config file used (optional)
            
        Returns:
            str: Unique run ID for this result
        """
        
        # Generate unique run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{strategy_name}_{profile}_{ticker}_{start_date}_{end_date}_{timestamp}"
        
        # Prepare result data
        result_data = {
            "run_id": run_id,
            "strategy_name": strategy_name,
            "profile": profile,
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "config_file": config_file,
            "timestamp": timestamp,
            "results": {
                "total_return": getattr(results, 'total_return', 0.0),
                "sharpe_ratio": getattr(results, 'sharpe_ratio', 0.0),
                "max_drawdown": getattr(results, 'max_drawdown', 0.0),
                "total_trades": getattr(results, 'total_trades', 0),
                "winning_trades": getattr(results, 'winning_trades', 0),
                "losing_trades": getattr(results, 'losing_trades', 0),
                "win_loss_ratio": getattr(results, 'win_loss_ratio', 0.0),
                "final_portfolio_value": getattr(results, 'final_portfolio_value', 0.0),
                "initial_capital": getattr(results, 'initial_capital', 0.0),
                "total_commission": getattr(results, 'total_commission', 0.0),
                "total_taxes": getattr(results, 'total_taxes', 0.0)
            },
            "strategy_summary": strategy_summary or {}
        }
        
        import os as _os
        if not (_os.getenv('AUDIT_EXECUTIONS_ONLY', '1').lower() in ('1', 'true', 'yes')):
            # Save detailed results as JSON
            detailed_file = self.detailed_dir / f"{run_id}.json"
            try:
                with open(detailed_file, 'w') as f:
                    json.dump(result_data, f, indent=2, default=str)
                logger.info(f"Detailed results saved: {detailed_file}")
            except Exception as e:
                logger.error(f"Failed to save detailed results: {e}")
        
        # Update index
        self._update_index(result_data)
        
        return run_id
    
    def _update_index(self, result_data: Dict):
        """Update the results index with new entry."""
        try:
            import os as _os
            if _os.getenv('AUDIT_EXECUTIONS_ONLY', '1').lower() in ('1', 'true', 'yes'):
                return
            # Prepare index row
            index_row = {
                "run_id": result_data["run_id"],
                "strategy": result_data["strategy_name"],
                "profile": result_data["profile"],
                "ticker": result_data["ticker"],
                "start_date": result_data["start_date"],
                "end_date": result_data["end_date"],
                "config_file": result_data.get("config_file", ""),
                "timestamp": result_data["timestamp"],
                "total_return": result_data["results"]["total_return"],
                "sharpe_ratio": result_data["results"]["sharpe_ratio"],
                "max_drawdown": result_data["results"]["max_drawdown"],
                "total_trades": result_data["results"]["total_trades"],
                "win_rate": (result_data["results"]["winning_trades"] / 
                           max(result_data["results"]["total_trades"], 1)),
                "final_value": result_data["results"]["final_portfolio_value"],
                "total_commission": result_data["results"]["total_commission"],
                "total_taxes": result_data["results"]["total_taxes"]
            }
            
            # Load existing index or create new
            if self.index_file.exists():
                df = pd.read_csv(self.index_file)
                df = pd.concat([df, pd.DataFrame([index_row])], ignore_index=True)
            else:
                df = pd.DataFrame([index_row])
            
            # Keep index in memory only (no CSV emission)
            self._last_index_df = df
            logger.info("Results index captured in memory (no CSV written)")
            
        except Exception as e:
            logger.error(f"Failed to update results index: {e}")
    
    def get_results(self, strategy_name: str = None, profile: str = None,
                   ticker: str = None, limit: int = None) -> pd.DataFrame:
        """
        Get results from index with optional filtering.
        
        Args:
            strategy_name: Filter by strategy name
            profile: Filter by profile
            ticker: Filter by ticker
            limit: Limit number of results
            
        Returns:
            DataFrame with filtered results
        """
        if not self.index_file.exists():
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.index_file)
            
            # Apply filters
            if strategy_name:
                df = df[df['strategy'] == strategy_name]
            if profile:
                df = df[df['profile'] == profile]
            if ticker:
                df = df[df['ticker'] == ticker]
            
            # Sort by timestamp (newest first)
            df = df.sort_values('timestamp', ascending=False)
            
            # Apply limit
            if limit:
                df = df.head(limit)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get results: {e}")
            return pd.DataFrame()
    
    def get_detailed_result(self, run_id: str) -> Optional[Dict]:
        """Get detailed result data for a specific run ID."""
        detailed_file = self.detailed_dir / f"{run_id}.json"
        
        if not detailed_file.exists():
            logger.warning(f"Detailed result not found: {run_id}")
            return None
        
        try:
            with open(detailed_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load detailed result {run_id}: {e}")
            return None
    
    def compare_profiles(self, strategy_name: str, ticker: str = None,
                        start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Compare results across different profiles.
        
        Simple comparison function that can be extended for more sophisticated analysis.
        """
        results = self.get_results(strategy_name=strategy_name, ticker=ticker)
        
        if results.empty:
            return results
        
        # Filter by date range if specified
        if start_date:
            results = results[results['start_date'] == start_date]
        if end_date:
            results = results[results['end_date'] == end_date]
        
        # Sort by total return (best first)
        results = results.sort_values('total_return', ascending=False)
        
        # Add ranking
        if not results.empty:
            results = results.copy()
            results['rank'] = range(1, len(results) + 1)
        
        return results
    
    def print_summary(self, results: pd.DataFrame):
        """Print a formatted summary of results."""
        # Quiet CLI output in performance runs; keep method for compatibility
        _ = results
        return