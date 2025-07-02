"""
Core Package for Deepfake Detection System

This package provides centralized core functionality including:
- Configuration management
- Logging setup
- Common utilities
- Base classes and patterns
- Error handling
"""

from .config import Config, get_config, reset_config
from .logging_config import setup_logging, get_logger
from .exceptions import DeepfakeDetectionError, ValidationError, ModelError
from .constants import *
from .base import BaseProcessor, BaseModel, get_resource_manager
from .decorators import (
    validate_inputs, handle_errors, log_execution, robust_processing,
    measure_performance, get_performance_stats, validate_file_exists
)

__all__ = [
    # Configuration
    'Config',
    'get_config',
    
    # Logging
    'setup_logging',
    'get_logger',
    
    # Exceptions
    'DeepfakeDetectionError',
    'ValidationError', 
    'ModelError',
    
    # Base classes
    'BaseProcessor',
    'BaseModel',
    
    # Decorators
    'validate_inputs',
    'handle_errors',
    'log_execution',
    
    # Constants (imported via *)
]

__version__ = '2.0.0'
__author__ = 'Deepfake Detection Team'
__description__ = 'Core functionality for deepfake detection system' 