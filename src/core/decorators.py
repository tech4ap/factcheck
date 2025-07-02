"""
Common Decorators

This module provides decorators for cross-cutting concerns like validation,
error handling, logging, and performance monitoring to reduce code duplication.
"""

import time
import functools
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from pathlib import Path

from .logging_config import get_logger
from .exceptions import ValidationError, DeepfakeDetectionError
from .constants import SUPPORTED_EXTENSIONS, MIN_FILE_SIZE, MAX_PROCESSING_TIME_SECONDS

F = TypeVar('F', bound=Callable[..., Any])

def validate_inputs(*validators: Callable[[Any], bool]) -> Callable[[F], F]:
    """
    Decorator to validate function inputs using provided validator functions.
    
    Args:
        *validators: Functions that take an argument and return True if valid
    
    Example:
        @validate_inputs(lambda x: isinstance(x, str), lambda x: len(x) > 0)
        def process_string(text: str):
            return text.upper()
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate positional arguments
            for i, (arg, validator) in enumerate(zip(args, validators)):
                if not validator(arg):
                    raise ValidationError(
                        f"Validation failed for argument {i} in {func.__name__}",
                        context={'arg_index': i, 'value': str(arg)}
                    )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def validate_file_exists(func: F) -> F:
    """
    Decorator to validate that file path arguments exist.
    Automatically validates any argument named 'file_path', 'input_path', etc.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check for file path arguments in kwargs
        file_path_args = ['file_path', 'input_path', 'model_path', 'data_path']
        
        for arg_name in file_path_args:
            if arg_name in kwargs:
                file_path = Path(kwargs[arg_name])
                if not file_path.exists():
                    raise ValidationError(f"File not found: {file_path}")
                if not file_path.is_file():
                    raise ValidationError(f"Path is not a file: {file_path}")
        
        return func(*args, **kwargs)
    return wrapper

def validate_supported_extension(extensions: List[str]) -> Callable[[F], F]:
    """
    Decorator to validate file extensions.
    
    Args:
        extensions: List of supported extensions (e.g., ['.jpg', '.png'])
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            file_path_args = ['file_path', 'input_path']
            
            for arg_name in file_path_args:
                if arg_name in kwargs:
                    file_path = Path(kwargs[arg_name])
                    if file_path.suffix.lower() not in [ext.lower() for ext in extensions]:
                        raise ValidationError(
                            f"Unsupported file extension: {file_path.suffix}. "
                            f"Supported: {', '.join(extensions)}"
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def handle_errors(
    error_types: Optional[List[type]] = None,
    default_return: Any = None,
    raise_on_unexpected: bool = True,
    log_errors: bool = True
) -> Callable[[F], F]:
    """
    Decorator to handle specific error types with optional logging.
    
    Args:
        error_types: List of exception types to handle (defaults to common ones)
        default_return: Value to return on handled errors
        raise_on_unexpected: Whether to re-raise unexpected errors
        log_errors: Whether to log handled errors
    """
    if error_types is None:
        error_types = [ValidationError, FileNotFoundError, PermissionError]
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except tuple(error_types) as e:
                if log_errors:
                    logger.error(f"Handled error in {func.__name__}: {e}")
                return default_return
            except Exception as e:
                if log_errors:
                    logger.error(f"Unexpected error in {func.__name__}: {e}")
                if raise_on_unexpected:
                    raise
                return default_return
        
        return wrapper
    return decorator

def log_execution(
    level: str = 'INFO',
    include_args: bool = False,
    include_result: bool = False,
    include_timing: bool = True
) -> Callable[[F], F]:
    """
    Decorator to log function execution with optional details.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        include_args: Whether to log function arguments
        include_result: Whether to log return value
        include_timing: Whether to log execution time
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            # Start timing
            start_time = time.time() if include_timing else None
            
            # Log function start
            log_msg = f"Executing {func.__name__}"
            if include_args and (args or kwargs):
                log_msg += f" with args={args}, kwargs={kwargs}"
            
            getattr(logger, level.lower())(log_msg)
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful completion
                end_msg = f"Completed {func.__name__}"
                if include_timing and start_time:
                    execution_time = time.time() - start_time
                    end_msg += f" in {execution_time:.3f}s"
                if include_result:
                    end_msg += f" -> {result}"
                
                getattr(logger, level.lower())(end_msg)
                return result
                
            except Exception as e:
                # Log error
                error_msg = f"Failed {func.__name__}: {e}"
                if include_timing and start_time:
                    execution_time = time.time() - start_time
                    error_msg += f" after {execution_time:.3f}s"
                
                logger.error(error_msg)
                raise
        
        return wrapper
    return decorator

def timeout(seconds: int = MAX_PROCESSING_TIME_SECONDS) -> Callable[[F], F]:
    """
    Decorator to enforce function execution timeout.
    
    Args:
        seconds: Maximum execution time in seconds
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            # Set up timeout (Unix only)
            if hasattr(signal, 'alarm'):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(seconds)
                
                try:
                    result = func(*args, **kwargs)
                    signal.alarm(0)  # Cancel timeout
                    return result
                finally:
                    signal.signal(signal.SIGALRM, old_handler)
            else:
                # Fallback for Windows - just log warning and proceed
                logger = get_logger(func.__module__)
                logger.warning(f"Timeout decorator not supported on this platform for {func.__name__}")
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Optional[List[type]] = None
) -> Callable[[F], F]:
    """
    Decorator to retry function execution on failure.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts (seconds)
        backoff: Multiplier for delay on each retry
        exceptions: List of exception types to retry on (defaults to common ones)
    """
    if exceptions is None:
        exceptions = [ConnectionError, TimeoutError, OSError]
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except tuple(exceptions) as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
                except Exception as e:
                    # Don't retry unexpected exceptions
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise
            
            # Re-raise the last exception if all attempts failed
            raise last_exception
        
        return wrapper
    return decorator

def cache_result(max_size: int = 128, ttl_seconds: Optional[int] = None) -> Callable[[F], F]:
    """
    Decorator to cache function results.
    
    Args:
        max_size: Maximum number of cached results
        ttl_seconds: Time-to-live for cached results (None for no expiry)
    """
    def decorator(func: F) -> F:
        cache = {}
        access_times = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            # Check if result is cached and not expired
            if key in cache:
                if ttl_seconds is None or (current_time - access_times[key]) < ttl_seconds:
                    access_times[key] = current_time
                    return cache[key]
                else:
                    # Remove expired entry
                    del cache[key]
                    del access_times[key]
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            
            # Implement simple LRU eviction if cache is full
            if len(cache) >= max_size:
                # Remove oldest entry
                oldest_key = min(access_times.keys(), key=lambda k: access_times[k])
                del cache[oldest_key]
                del access_times[oldest_key]
            
            cache[key] = result
            access_times[key] = current_time
            return result
        
        # Add cache management methods
        wrapper.cache_info = lambda: {
            'size': len(cache),
            'max_size': max_size,
            'ttl_seconds': ttl_seconds
        }
        wrapper.cache_clear = lambda: (cache.clear(), access_times.clear())
        
        return wrapper
    return decorator

def measure_performance(func: F) -> F:
    """
    Decorator to measure and log function performance metrics.
    """
    if not hasattr(measure_performance, 'stats'):
        measure_performance.stats = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__name__}"
        
        if func_name not in measure_performance.stats:
            measure_performance.stats[func_name] = {
                'call_count': 0,
                'total_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0,
                'errors': 0
            }
        
        stats = measure_performance.stats[func_name]
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Update statistics
            stats['call_count'] += 1
            stats['total_time'] += execution_time
            stats['min_time'] = min(stats['min_time'], execution_time)
            stats['max_time'] = max(stats['max_time'], execution_time)
            
            return result
            
        except Exception as e:
            stats['errors'] += 1
            raise
    
    return wrapper

def get_performance_stats() -> Dict[str, Dict[str, Any]]:
    """Get performance statistics for all measured functions."""
    if not hasattr(measure_performance, 'stats'):
        return {}
    
    # Calculate averages
    stats = measure_performance.stats.copy()
    for func_name, func_stats in stats.items():
        if func_stats['call_count'] > 0:
            func_stats['avg_time'] = func_stats['total_time'] / func_stats['call_count']
            func_stats['success_rate'] = (
                (func_stats['call_count'] - func_stats['errors']) / func_stats['call_count']
            )
        else:
            func_stats['avg_time'] = 0.0
            func_stats['success_rate'] = 0.0
    
    return stats

def reset_performance_stats() -> None:
    """Reset all performance statistics."""
    if hasattr(measure_performance, 'stats'):
        measure_performance.stats.clear()

# Convenience combinators
def validate_and_log(
    extensions: Optional[List[str]] = None,
    log_level: str = 'INFO',
    include_timing: bool = True
) -> Callable[[F], F]:
    """
    Combination decorator for validation and logging.
    
    Args:
        extensions: Supported file extensions to validate
        log_level: Logging level
        include_timing: Whether to include timing information
    """
    def decorator(func: F) -> F:
        # Apply multiple decorators
        decorated = func
        
        if extensions:
            decorated = validate_supported_extension(extensions)(decorated)
        
        decorated = validate_file_exists(decorated)
        decorated = log_execution(level=log_level, include_timing=include_timing)(decorated)
        decorated = handle_errors(log_errors=True)(decorated)
        
        return decorated
    
    return decorator

def robust_processing(
    timeout_seconds: int = MAX_PROCESSING_TIME_SECONDS,
    max_retries: int = 3,
    extensions: Optional[List[str]] = None
) -> Callable[[F], F]:
    """
    Combination decorator for robust file processing.
    
    Args:
        timeout_seconds: Maximum execution time
        max_retries: Maximum retry attempts
        extensions: Supported file extensions
    """
    def decorator(func: F) -> F:
        decorated = func
        
        if extensions:
            decorated = validate_supported_extension(extensions)(decorated)
        
        decorated = validate_file_exists(decorated)
        decorated = timeout(timeout_seconds)(decorated)
        decorated = retry(max_retries)(decorated)
        decorated = handle_errors(log_errors=True)(decorated)
        decorated = measure_performance(decorated)
        
        return decorated
    
    return decorator 