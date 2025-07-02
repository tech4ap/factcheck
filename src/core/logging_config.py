"""
Centralized Logging Configuration

This module provides a unified logging system for the deepfake detection project,
eliminating duplicate logging setup code across modules and providing consistent
log formatting and handling.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[41m',   # Red background
    }
    RESET = '\033[0m'
    
    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)

class DeepfakeLogger:
    """
    Centralized logger configuration for the deepfake detection system.
    
    Features:
    - Consistent formatting across all modules
    - File and console logging with separate levels
    - Automatic log rotation
    - Emoji-enhanced messages for better UX
    - Environment-aware configuration
    """
    
    _loggers: Dict[str, logging.Logger] = {}
    _configured = False
    
    @classmethod
    def setup_logging(cls, 
                     level: str = 'INFO',
                     log_file: Optional[Path] = None,
                     log_to_console: bool = True,
                     log_to_file: bool = True,
                     console_level: str = 'INFO',
                     file_level: str = 'DEBUG',
                     max_log_size_mb: int = 10,
                     backup_count: int = 5,
                     format_string: Optional[str] = None,
                     date_format: Optional[str] = None,
                     colored_console: bool = True,
                     suppress_external: bool = True) -> None:
        """
        Configure logging for the entire application.
        
        Args:
            level: Base logging level
            log_file: Path to log file (auto-generated if None)
            log_to_console: Enable console logging
            log_to_file: Enable file logging
            console_level: Console-specific log level
            file_level: File-specific log level
            max_log_size_mb: Maximum log file size before rotation
            backup_count: Number of backup log files to keep
            format_string: Custom format string
            date_format: Custom date format
            colored_console: Use colored output for console
            suppress_external: Suppress verbose external library logs
        """
        if cls._configured:
            return
        
        # Default format strings
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        if date_format is None:
            date_format = '%Y-%m-%d %H:%M:%S'
        
        # Create formatters
        file_formatter = logging.Formatter(format_string, date_format)
        
        if colored_console:
            console_formatter = ColoredFormatter(format_string, date_format)
        else:
            console_formatter = logging.Formatter(format_string, date_format)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, console_level.upper()))
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # File handler with rotation
        if log_to_file:
            if log_file is None:
                # Auto-generate log file path
                logs_dir = Path('logs')
                logs_dir.mkdir(exist_ok=True)
                log_file = logs_dir / f"deepfake_{datetime.now().strftime('%Y%m%d')}.log"
            
            # Ensure log directory exists
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=max_log_size_mb * 1024 * 1024,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, file_level.upper()))
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        
        # Suppress verbose external library logs
        if suppress_external:
            cls._suppress_external_logs()
        
        cls._configured = True
        
        # Log initial setup message
        logger = cls.get_logger('core.logging')
        logger.info("🚀 Logging system initialized")
        logger.info(f"📍 Console level: {console_level}, File level: {file_level}")
        if log_file:
            logger.info(f"📁 Log file: {log_file}")
    
    @classmethod
    def _suppress_external_logs(cls) -> None:
        """Suppress verbose logs from external libraries."""
        external_loggers = {
            'tensorflow': 'WARNING',
            'boto3': 'WARNING', 
            'botocore': 'WARNING',
            'urllib3': 'WARNING',
            's3transfer': 'WARNING',
            'matplotlib': 'WARNING',
            'PIL': 'WARNING',
            'numba': 'WARNING'
        }
        
        for logger_name, level in external_loggers.items():
            logging.getLogger(logger_name).setLevel(getattr(logging, level))
    
    @classmethod
    def get_logger(cls, name: str, 
                  level: Optional[str] = None,
                  add_emoji: bool = True) -> logging.Logger:
        """
        Get a configured logger instance.
        
        Args:
            name: Logger name (usually __name__)
            level: Optional custom level for this logger
            add_emoji: Add emoji enhancement to log messages
            
        Returns:
            Configured logger instance
        """
        if not cls._configured:
            cls.setup_logging()
        
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        
        if level:
            logger.setLevel(getattr(logging, level.upper()))
        
        if add_emoji:
            logger = EmojiLoggerAdapter(logger)
        
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def set_level(cls, level: str, logger_name: Optional[str] = None) -> None:
        """
        Change logging level at runtime.
        
        Args:
            level: New logging level
            logger_name: Specific logger name (None for root logger)
        """
        if logger_name:
            logging.getLogger(logger_name).setLevel(getattr(logging, level.upper()))
        else:
            logging.getLogger().setLevel(getattr(logging, level.upper()))
    
    @classmethod
    def reset(cls) -> None:
        """Reset logging configuration."""
        cls._loggers.clear()
        cls._configured = False
        
        # Clear all handlers from root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

class EmojiLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds emoji prefixes to log messages for better UX.
    """
    
    EMOJI_MAP = {
        'DEBUG': '🔍',
        'INFO': '📝',
        'WARNING': '⚠️ ',
        'ERROR': '❌',
        'CRITICAL': '🚨'
    }
    
    def __init__(self, logger):
        super().__init__(logger, {})
    
    def process(self, msg, kwargs):
        # Add emoji based on the calling method
        import inspect
        frame = inspect.currentframe()
        try:
            # Go up the call stack to find the logging method
            for _ in range(10):  # Limit search depth
                frame = frame.f_back
                if frame is None:
                    break
                
                func_name = frame.f_code.co_name.upper()
                if func_name in self.EMOJI_MAP:
                    emoji = self.EMOJI_MAP[func_name]
                    msg = f"{emoji} {msg}"
                    break
        finally:
            del frame
        
        return msg, kwargs
    
    def debug(self, msg, *args, **kwargs):
        msg = f"🔍 {msg}"
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        msg = f"📝 {msg}"
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        msg = f"⚠️  {msg}"
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        msg = f"❌ {msg}"
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        msg = f"🚨 {msg}"
        self.logger.critical(msg, *args, **kwargs)

# Convenience functions for easier adoption
def setup_logging(**kwargs) -> None:
    """Setup logging with default configuration."""
    DeepfakeLogger.setup_logging(**kwargs)

def get_logger(name: str = __name__, **kwargs) -> logging.Logger:
    """Get a configured logger instance."""
    return DeepfakeLogger.get_logger(name, **kwargs)

def set_log_level(level: str, logger_name: Optional[str] = None) -> None:
    """Change logging level at runtime."""
    DeepfakeLogger.set_level(level, logger_name)

# Auto-configure logging when module is imported
def _auto_configure():
    """Auto-configure logging based on environment variables."""
    if not DeepfakeLogger._configured:
        # Get configuration from environment
        level = os.getenv('LOG_LEVEL', 'INFO').upper()
        log_file = os.getenv('LOG_FILE')
        console_level = os.getenv('CONSOLE_LOG_LEVEL', 'INFO').upper()
        
        setup_logging(
            level=level,
            log_file=Path(log_file) if log_file else None,
            console_level=console_level,
            colored_console=sys.stdout.isatty(),  # Only use colors for interactive terminals
            suppress_external=True
        )

# Auto-configure on import
_auto_configure() 