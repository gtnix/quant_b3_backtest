"""
Central Configuration Manager for Backtesting System

Provides unified configuration management, eliminating scattered
hardcoded paths and configuration throughout the codebase.

Author: Senior Python Developer
Date: 2025
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import os

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Centralized configuration management for the backtesting system.
    
    Features:
    - Unified configuration loading
    - Environment variable override support
    - Default value management
    - Configuration validation
    - Hot reload capability
    """
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """Initialize configuration manager."""
        self.config_file = Path(config_file) if config_file else Path("config/settings.yaml")
        self._config = {}
        self._defaults = self._get_default_config()
        self._load_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            'paths': {
                'results_dir': 'results',
                'reports_dir': 'reports',
                'logs_dir': 'logs',
                'data_dir': 'data',
                'cache_dir': 'data/brapi_cache',
                'detailed_subdir': 'detailed',
                'validation_subdir': 'validation'
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'console_output': True,
                'file_output': False,
                'structured_logging': True,
                'batch_size': 256,
                'flush_ms': 200
            },
            'output': {
                'audit_only': False,
                'atomic_writes': True,
                'cleanup_temp_files': True,
                'duplicate_prevention': True
            },
            'performance': {
                'initial_capital': 100000.0,
                'commission_rate': 0.00025,
                'min_position_size': 100,
                'max_position_size': 50000
            },
            'strategy': {
                'fuzzy_threshold': 1.5,
                'risk_pair_matching': True,
                'moc_flattening': True,
                'position_sizing_method': 'fixed_notional'
            }
        }
    
    def _load_config(self):
        """Load configuration from file with fallback to defaults."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    if self.config_file.suffix.lower() == '.yaml':
                        file_config = yaml.safe_load(f) or {}
                    else:
                        file_config = json.load(f)
                
                # Deep merge with defaults
                self._config = self._deep_merge(self._defaults, file_config)
                logger.info(f"Configuration loaded from {self.config_file}")
            else:
                logger.warning(f"Config file not found: {self.config_file}, using defaults")
                self._config = self._defaults.copy()
        
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_file}: {e}")
            logger.info("Using default configuration")
            self._config = self._defaults.copy()
        
        # Apply environment variable overrides
        self._apply_env_overrides()
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            'AUDIT_EXECUTIONS_ONLY': ('output', 'audit_only'),
            'LOG_LEVEL': ('logging', 'level'),
            'LOG_BATCH_SIZE': ('logging', 'batch_size'),
            'LOG_FLUSH_MS': ('logging', 'flush_ms'),
            'INITIAL_CAPITAL': ('performance', 'initial_capital'),
            'RESULTS_DIR': ('paths', 'results_dir'),
            'REPORTS_DIR': ('paths', 'reports_dir'),
            'LOGS_DIR': ('paths', 'logs_dir')
        }
        
        for env_var, (section, key) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                converted_value = self._convert_env_value(env_value)
                if section not in self._config:
                    self._config[section] = {}
                self._config[section][key] = converted_value
                logger.debug(f"Applied env override: {env_var}={converted_value}")
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', '1', 'yes', 'on'):
            return True
        elif value.lower() in ('false', '0', 'no', 'off'):
            return False
        
        # Numeric conversion
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            section: Configuration section name
            key: Optional key within section
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        try:
            if key is None:
                return self._config.get(section, default)
            else:
                return self._config.get(section, {}).get(key, default)
        except Exception as e:
            logger.warning(f"Failed to get config {section}.{key}: {e}")
            return default
    
    def get_path(self, path_name: str) -> Path:
        """Get configured path as Path object."""
        path_str = self.get('paths', path_name, path_name)
        return Path(path_str)
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration dictionary."""
        return self.get('logging', default={})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration dictionary."""
        return self.get('output', default={})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration dictionary."""
        return self.get('performance', default={})
    
    def get_strategy_config(self) -> Dict[str, Any]:
        """Get strategy configuration dictionary."""
        return self.get('strategy', default={})
    
    def is_audit_mode(self) -> bool:
        """Check if system is in audit-only mode."""
        return self.get('output', 'audit_only', False)
    
    def reload(self):
        """Reload configuration from file."""
        logger.info("Reloading configuration")
        self._load_config()
    
    def save_current_config(self, output_file: Optional[Union[str, Path]] = None):
        """Save current configuration to file."""
        if output_file is None:
            output_file = self.config_file.with_name(f"{self.config_file.stem}_current.yaml")
        else:
            output_file = Path(output_file)
        
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, indent=2, default_flow_style=False)
            logger.info(f"Current configuration saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def validate_config(self) -> Dict[str, List[str]]:
        """
        Validate current configuration.
        
        Returns:
            Dictionary with validation errors by section
        """
        errors = {}
        
        # Validate paths
        path_errors = []
        required_paths = ['results_dir', 'reports_dir', 'logs_dir', 'data_dir']
        for path_name in required_paths:
            if not self.get('paths', path_name):
                path_errors.append(f"Missing required path: {path_name}")
        
        if path_errors:
            errors['paths'] = path_errors
        
        # Validate logging
        logging_errors = []
        log_level = self.get('logging', 'level')
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if log_level not in valid_levels:
            logging_errors.append(f"Invalid log level: {log_level}")
        
        if logging_errors:
            errors['logging'] = logging_errors
        
        # Validate performance
        perf_errors = []
        initial_capital = self.get('performance', 'initial_capital')
        if not isinstance(initial_capital, (int, float)) or initial_capital <= 0:
            perf_errors.append("initial_capital must be positive number")
        
        if perf_errors:
            errors['performance'] = perf_errors
        
        return errors
    
    def get_stats(self) -> Dict[str, Any]:
        """Get configuration manager statistics."""
        return {
            'config_file': str(self.config_file),
            'config_exists': self.config_file.exists(),
            'sections': list(self._config.keys()),
            'audit_mode': self.is_audit_mode(),
            'validation_errors': self.validate_config()
        }


# Global instance for easy access
_config_manager = None

def get_config_manager(config_file: Optional[Union[str, Path]] = None) -> ConfigManager:
    """Get or create global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    return _config_manager

def reset_config_manager():
    """Reset global configuration manager (mainly for testing)."""
    global _config_manager
    _config_manager = None
