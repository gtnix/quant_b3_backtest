"""
Centralized Logging Configuration for Backtesting System

Eliminates scattered logging.basicConfig() calls and provides
consistent logging setup across all modules.

Author: Senior Python Developer
Date: 2025
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from .config_manager import get_config_manager
from .utils.async_logger import AsyncJsonlLogger


class LoggingManager:
    """
    Centralized logging configuration and management.
    
    Features:
    - Unified logging setup
    - Structured logging support
    - Console and file output
    - Environment-based configuration
    - Business event tracking
    """
    
    def __init__(self, config_manager=None):
        """Initialize logging manager."""
        self.config_manager = config_manager or get_config_manager()
        self.structured_logger = None
        self._configured = False
    
    def setup_logging(self):
        """Setup logging configuration for the entire system."""
        if self._configured:
            return
        
        logging_config = self.config_manager.get_logging_config()
        
        # Create logging configuration dictionary
        config_dict = self._create_logging_config(logging_config)
        
        # Apply configuration
        logging.config.dictConfig(config_dict)
        
        # Setup structured logging if enabled
        if logging_config.get('structured_logging', True):
            self._setup_structured_logging(logging_config)
        
        # Silence noisy third-party loggers
        self._silence_noisy_loggers()
        
        self._configured = True
        logger = logging.getLogger(__name__)
        logger.info("Centralized logging configuration applied")
    
    def _create_logging_config(self, logging_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create logging configuration dictionary."""
        log_level = logging_config.get('level', 'INFO').upper()
        log_format = logging_config.get('format', 
                                      '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': log_format,
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                }
            },
            'handlers': {},
            'root': {
                'level': log_level,
                'handlers': []
            },
            'loggers': {}
        }
        
        # Console handler
        if logging_config.get('console_output', True):
            config['handlers']['console'] = {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'standard',
                'stream': sys.stdout
            }
            config['root']['handlers'].append('console')
        
        # File handler
        if logging_config.get('file_output', False):
            logs_dir = self.config_manager.get_path('logs_dir')
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            config['handlers']['file'] = {
                'class': 'logging.FileHandler',
                'level': log_level,
                'formatter': 'detailed',
                'filename': str(logs_dir / 'backtesting.log'),
                'mode': 'a',
                'encoding': 'utf-8'
            }
            config['root']['handlers'].append('file')
        
        return config
    
    def _setup_structured_logging(self, logging_config: Dict[str, Any]):
        """Setup structured logging with AsyncJsonlLogger."""
        try:
            logs_dir = self.config_manager.get_path('logs_dir')
            batch_size = logging_config.get('batch_size', 256)
            flush_ms = logging_config.get('flush_ms', 200)
            
            self.structured_logger = AsyncJsonlLogger(
                base_dir=logs_dir,
                batch_size=batch_size,
                flush_ms=flush_ms
            )
            
            # Make it globally accessible
            import engine
            engine.event_logger = self.structured_logger
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to setup structured logging: {e}")
    
    def _silence_noisy_loggers(self):
        """Silence or reduce verbosity of noisy third-party loggers."""
        noisy_loggers = [
            'urllib3.connectionpool',
            'requests.packages.urllib3',
            'matplotlib',
            'PIL',
            'asyncio'
        ]
        
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with consistent configuration."""
        if not self._configured:
            self.setup_logging()
        
        return logging.getLogger(name)
    
    def emit_business_event(self, phase: str, action: str, **fields):
        """Emit a structured business event."""
        if self.structured_logger:
            try:
                import time
                payload = {
                    'phase': str(phase),
                    'action': str(action),
                    'ts': int(time.time()),
                    'label': f"[{str(phase)}] {str(action)}",
                }
                payload.update({k: v for k, v in fields.items()})
                self.structured_logger.emit('business', payload)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to emit business event: {e}")
    
    def shutdown(self):
        """Shutdown logging system gracefully."""
        if self.structured_logger:
            self.structured_logger.shutdown()
        
        # Shutdown all handlers
        logging.shutdown()


# Global instance
_logging_manager = None

def get_logging_manager() -> LoggingManager:
    """Get or create global logging manager instance."""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager()
    return _logging_manager

def setup_logging():
    """Setup logging for the entire system."""
    get_logging_manager().setup_logging()

def get_logger(name: str) -> logging.Logger:
    """Get a properly configured logger."""
    return get_logging_manager().get_logger(name)

def emit_business_event(phase: str, action: str, **fields):
    """Emit a structured business event."""
    get_logging_manager().emit_business_event(phase, action, **fields)

def shutdown_logging():
    """Shutdown logging system."""
    global _logging_manager
    if _logging_manager:
        _logging_manager.shutdown()
        _logging_manager = None
