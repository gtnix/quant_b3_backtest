"""
Centralized Output Manager for Backtesting System

Eliminates duplicate file writes, provides consistent output handling,
and centralizes all file I/O operations with proper error handling.

Author: Senior Python Developer
Date: 2025
"""

import pandas as pd
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class OutputManager:
    """
    Centralized manager for all file outputs in the backtesting system.
    
    Features:
    - Single source of truth for all file writes
    - Eliminates duplicate writes
    - Consistent error handling
    - Configurable output paths
    - Audit mode support
    - Atomic write operations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize output manager with configuration."""
        self.config = config or self._load_default_config()
        self.audit_only = self._is_audit_mode()
        
        # Initialize output directories
        if not self.audit_only:
            self._create_directories()
        
        # Track written files to prevent duplicates
        self._written_files = set()
        
        logger.info(f"OutputManager initialized (audit_only={self.audit_only})")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default output configuration."""
        return {
            'results_dir': 'results',
            'reports_dir': 'reports', 
            'logs_dir': 'logs',
            'data_dir': 'data',
            'detailed_subdir': 'detailed',
            'validation_subdir': 'validation'
        }
    
    def _is_audit_mode(self) -> bool:
        """Check if system is in audit-only mode."""
        return os.getenv('AUDIT_EXECUTIONS_ONLY', '0').lower() in ('1', 'true', 'yes')
    
    def _create_directories(self):
        """Create all required output directories."""
        dirs_to_create = [
            self.config['results_dir'],
            self.config['reports_dir'],
            self.config['logs_dir'],
            f"{self.config['results_dir']}/{self.config['detailed_subdir']}",
            f"{self.config['reports_dir']}/{self.config['validation_subdir']}"
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def _atomic_write(self, filepath: Union[str, Path]):
        """Context manager for atomic file writes."""
        filepath = Path(filepath)
        temp_filepath = filepath.with_suffix(filepath.suffix + '.tmp')
        
        try:
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                yield f
            # Atomic move on success
            temp_filepath.rename(filepath)
        except Exception as e:
            # Cleanup temp file on failure
            if temp_filepath.exists():
                temp_filepath.unlink()
            raise e
    
    def write_unified_fills(self, fills_df: pd.DataFrame, 
                          base_name: str = "unified_fills") -> Dict[str, str]:
        """
        Write unified fills to both CSV and JSON formats.
        
        This is the SINGLE source for unified fills output.
        All other duplicate writes should be removed.
        """
        if self.audit_only or fills_df is None or fills_df.empty:
            return {}
        
        artifacts = {}
        results_dir = Path(self.config['results_dir'])
        
        # CSV output
        csv_path = results_dir / f"{base_name}.csv"
        csv_key = str(csv_path)
        
        if csv_key not in self._written_files:
            try:
                with self._atomic_write(csv_path) as f:
                    fills_df.to_csv(f, index=False)
                artifacts['csv'] = str(csv_path)
                self._written_files.add(csv_key)
                logger.info(f"Written unified fills CSV: {csv_path} ({len(fills_df)} rows)")
            except Exception as e:
                logger.error(f"Failed to write unified fills CSV: {e}")
        
        # JSON output
        json_path = results_dir / f"{base_name}.json"
        json_key = str(json_path)
        
        if json_key not in self._written_files:
            try:
                with self._atomic_write(json_path) as f:
                    fills_df.to_json(f, orient='records', indent=2, date_format='iso')
                artifacts['json'] = str(json_path)
                self._written_files.add(json_key)
                logger.info(f"Written unified fills JSON: {json_path}")
            except Exception as e:
                logger.error(f"Failed to write unified fills JSON: {e}")
        
        return artifacts
    
    def write_signal_data(self, signals_df: pd.DataFrame, 
                         filename: str = "portfolio_fuzzy_indicators.csv") -> Optional[str]:
        """Write signal data to reports directory."""
        if self.audit_only or signals_df is None or signals_df.empty:
            return None
        
        reports_dir = Path(self.config['reports_dir'])
        filepath = reports_dir / filename
        filepath_key = str(filepath)
        
        if filepath_key not in self._written_files:
            try:
                with self._atomic_write(filepath) as f:
                    signals_df.to_csv(f, index=False)
                self._written_files.add(filepath_key)
                logger.info(f"Written signal data: {filepath} ({len(signals_df)} rows)")
                return str(filepath)
            except Exception as e:
                logger.error(f"Failed to write signal data: {e}")
        
        return None
    
    def write_performance_report(self, report_data: Dict[str, Any], 
                               filename: str = "performance_report.json") -> Optional[str]:
        """Write performance analysis to reports directory."""
        if self.audit_only or not report_data:
            return None
        
        reports_dir = Path(self.config['reports_dir'])
        filepath = reports_dir / filename
        filepath_key = str(filepath)
        
        if filepath_key not in self._written_files:
            try:
                with self._atomic_write(filepath) as f:
                    json.dump(report_data, f, indent=2, default=str)
                self._written_files.add(filepath_key)
                logger.info(f"Written performance report: {filepath}")
                return str(filepath)
            except Exception as e:
                logger.error(f"Failed to write performance report: {e}")
        
        return None
    
    def write_backtest_results(self, results_data: Dict[str, Any], 
                             run_id: str) -> Optional[str]:
        """Write detailed backtest results to results/detailed directory."""
        if self.audit_only or not results_data:
            return None
        
        detailed_dir = Path(self.config['results_dir']) / self.config['detailed_subdir']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{run_id}_{timestamp}.json"
        filepath = detailed_dir / filename
        
        try:
            with self._atomic_write(filepath) as f:
                json.dump(results_data, f, indent=2, default=str)
            logger.info(f"Written backtest results: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to write backtest results: {e}")
            return None
    
    def write_validation_log(self, log_entries: List[str], 
                           filename: str = "validation_log.txt") -> Optional[str]:
        """Write validation log entries."""
        if self.audit_only or not log_entries:
            return None
        
        results_dir = Path(self.config['results_dir'])
        filepath = results_dir / filename
        filepath_key = str(filepath)
        
        if filepath_key not in self._written_files:
            try:
                with self._atomic_write(filepath) as f:
                    f.write('\n'.join(log_entries) + '\n')
                self._written_files.add(filepath_key)
                logger.info(f"Written validation log: {filepath} ({len(log_entries)} entries)")
                return str(filepath)
            except Exception as e:
                logger.error(f"Failed to write validation log: {e}")
        
        return None
    
    def get_output_path(self, category: str, filename: str) -> Path:
        """Get standardized output path for a given category and filename."""
        category_dirs = {
            'results': self.config['results_dir'],
            'reports': self.config['reports_dir'],
            'logs': self.config['logs_dir'],
            'detailed': f"{self.config['results_dir']}/{self.config['detailed_subdir']}",
            'validation': f"{self.config['reports_dir']}/{self.config['validation_subdir']}"
        }
        
        if category not in category_dirs:
            raise ValueError(f"Unknown output category: {category}")
        
        return Path(category_dirs[category]) / filename
    
    def cleanup_temp_files(self):
        """Clean up any temporary files that may have been left behind."""
        if self.audit_only:
            return
        
        temp_patterns = ['*.tmp', '*.temp']
        dirs_to_clean = [
            self.config['results_dir'],
            self.config['reports_dir']
        ]
        
        cleaned_count = 0
        for dir_path in dirs_to_clean:
            dir_obj = Path(dir_path)
            if dir_obj.exists():
                for pattern in temp_patterns:
                    for temp_file in dir_obj.glob(pattern):
                        try:
                            temp_file.unlink()
                            cleaned_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to clean temp file {temp_file}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} temporary files")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get output manager statistics."""
        return {
            'audit_only': self.audit_only,
            'files_written': len(self._written_files),
            'written_files': list(self._written_files),
            'config': self.config
        }


# Global instance for easy access
_output_manager = None

def get_output_manager(config: Optional[Dict[str, Any]] = None) -> OutputManager:
    """Get or create global output manager instance."""
    global _output_manager
    if _output_manager is None:
        _output_manager = OutputManager(config)
    return _output_manager

def reset_output_manager():
    """Reset global output manager (mainly for testing)."""
    global _output_manager
    _output_manager = None
