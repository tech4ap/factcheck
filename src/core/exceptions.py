"""
Centralized Exception Classes

This module provides standardized exception classes for the deepfake detection system,
ensuring consistent error handling and messaging across all modules.
"""

from typing import Optional, Dict, Any

class DeepfakeDetectionError(Exception):
    """
    Base exception class for all deepfake detection related errors.
    
    This provides a common interface for all project-specific exceptions
    with support for error codes, context data, and structured messaging.
    """
    
    def __init__(self, 
                 message: str,
                 error_code: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None,
                 cause: Optional[Exception] = None):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            context: Optional context data (file paths, parameters, etc.)
            cause: Optional underlying exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.cause = cause
    
    def __str__(self) -> str:
        """Return formatted error message."""
        parts = [self.message]
        
        if self.error_code:
            parts.append(f"(Code: {self.error_code})")
        
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"[Context: {context_str}]")
        
        if self.cause:
            parts.append(f"Caused by: {self.cause}")
        
        return " ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for structured logging/API responses."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None
        }

class ValidationError(DeepfakeDetectionError):
    """
    Exception raised when input validation fails.
    
    Used for file format validation, parameter validation, 
    configuration validation, etc.
    """
    
    def __init__(self, message: str, field: Optional[str] = None, 
                 value: Optional[Any] = None, **kwargs):
        """
        Initialize validation error.
        
        Args:
            message: Error message
            field: Name of the field that failed validation
            value: The invalid value
            **kwargs: Additional context passed to parent
        """
        context = kwargs.get('context', {})
        if field:
            context['field'] = field
        if value is not None:
            context['value'] = str(value)
        
        kwargs['context'] = context
        super().__init__(message, error_code='VALIDATION_ERROR', **kwargs)

class ModelError(DeepfakeDetectionError):
    """
    Exception raised when model operations fail.
    
    Used for model loading, training, inference, and serialization errors.
    """
    
    def __init__(self, message: str, model_type: Optional[str] = None,
                 model_path: Optional[str] = None, **kwargs):
        """
        Initialize model error.
        
        Args:
            message: Error message
            model_type: Type of model (image, video, audio)
            model_path: Path to model file
            **kwargs: Additional context passed to parent
        """
        context = kwargs.get('context', {})
        if model_type:
            context['model_type'] = model_type
        if model_path:
            context['model_path'] = model_path
        
        kwargs['context'] = context
        super().__init__(message, error_code='MODEL_ERROR', **kwargs)

class DataError(DeepfakeDetectionError):
    """
    Exception raised when data operations fail.
    
    Used for file I/O, data loading, preprocessing, and format errors.
    """
    
    def __init__(self, message: str, file_path: Optional[str] = None,
                 operation: Optional[str] = None, **kwargs):
        """
        Initialize data error.
        
        Args:
            message: Error message
            file_path: Path to file that caused the error
            operation: Operation that failed (load, save, preprocess, etc.)
            **kwargs: Additional context passed to parent
        """
        context = kwargs.get('context', {})
        if file_path:
            context['file_path'] = file_path
        if operation:
            context['operation'] = operation
        
        kwargs['context'] = context
        super().__init__(message, error_code='DATA_ERROR', **kwargs)

class AWSError(DeepfakeDetectionError):
    """
    Exception raised when AWS operations fail.
    
    Used for S3, SQS, and other AWS service errors.
    """
    
    def __init__(self, message: str, service: Optional[str] = None,
                 aws_error_code: Optional[str] = None, **kwargs):
        """
        Initialize AWS error.
        
        Args:
            message: Error message
            service: AWS service name (S3, SQS, etc.)
            aws_error_code: AWS-specific error code
            **kwargs: Additional context passed to parent
        """
        context = kwargs.get('context', {})
        if service:
            context['aws_service'] = service
        if aws_error_code:
            context['aws_error_code'] = aws_error_code
        
        kwargs['context'] = context
        super().__init__(message, error_code='AWS_ERROR', **kwargs)

class ConfigurationError(DeepfakeDetectionError):
    """
    Exception raised when configuration is invalid or missing.
    
    Used for missing environment variables, invalid configuration files,
    and runtime configuration errors.
    """
    
    def __init__(self, message: str, config_key: Optional[str] = None,
                 config_file: Optional[str] = None, **kwargs):
        """
        Initialize configuration error.
        
        Args:
            message: Error message
            config_key: Configuration key that caused the error
            config_file: Configuration file path
            **kwargs: Additional context passed to parent
        """
        context = kwargs.get('context', {})
        if config_key:
            context['config_key'] = config_key
        if config_file:
            context['config_file'] = config_file
        
        kwargs['context'] = context
        super().__init__(message, error_code='CONFIG_ERROR', **kwargs)

class InferenceError(DeepfakeDetectionError):
    """
    Exception raised during inference/prediction operations.
    
    Used for prediction failures, preprocessing errors during inference,
    and result processing errors.
    """
    
    def __init__(self, message: str, input_file: Optional[str] = None,
                 model_type: Optional[str] = None, **kwargs):
        """
        Initialize inference error.
        
        Args:
            message: Error message
            input_file: Input file that caused the error
            model_type: Type of model used for inference
            **kwargs: Additional context passed to parent
        """
        context = kwargs.get('context', {})
        if input_file:
            context['input_file'] = input_file
        if model_type:
            context['model_type'] = model_type
        
        kwargs['context'] = context
        super().__init__(message, error_code='INFERENCE_ERROR', **kwargs)

class TrainingError(DeepfakeDetectionError):
    """
    Exception raised during model training operations.
    
    Used for training failures, data loading errors during training,
    and checkpoint/model saving errors.
    """
    
    def __init__(self, message: str, epoch: Optional[int] = None,
                 model_type: Optional[str] = None, **kwargs):
        """
        Initialize training error.
        
        Args:
            message: Error message
            epoch: Training epoch when error occurred
            model_type: Type of model being trained
            **kwargs: Additional context passed to parent
        """
        context = kwargs.get('context', {})
        if epoch is not None:
            context['epoch'] = epoch
        if model_type:
            context['model_type'] = model_type
        
        kwargs['context'] = context
        super().__init__(message, error_code='TRAINING_ERROR', **kwargs)

# Convenience functions for common error patterns
def validation_failed(field: str, value: Any, reason: str) -> ValidationError:
    """Create a validation error for a specific field."""
    message = f"Validation failed for field '{field}': {reason}"
    return ValidationError(message, field=field, value=value)

def model_not_found(model_type: str, model_path: str) -> ModelError:
    """Create a model not found error."""
    message = f"Model file not found: {model_path}"
    return ModelError(message, model_type=model_type, model_path=model_path)

def file_not_found(file_path: str, operation: str = "read") -> DataError:
    """Create a file not found error."""
    message = f"File not found: {file_path}"
    return DataError(message, file_path=file_path, operation=operation)

def aws_access_denied(service: str, resource: str) -> AWSError:
    """Create an AWS access denied error."""
    message = f"Access denied to {service} resource: {resource}"
    return AWSError(message, service=service, context={'resource': resource})

def config_missing(key: str, config_file: Optional[str] = None) -> ConfigurationError:
    """Create a missing configuration error."""
    message = f"Required configuration key missing: {key}"
    return ConfigurationError(message, config_key=key, config_file=config_file) 