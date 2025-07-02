"""
Base Classes and Common Patterns

This module provides base classes and common patterns used throughout
the deepfake detection system to reduce code duplication and ensure
consistent interfaces.
"""

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from contextlib import contextmanager

from .logging_config import get_logger
from .exceptions import DeepfakeDetectionError, ValidationError, DataError
from .constants import SUPPORTED_EXTENSIONS, MIN_FILE_SIZE

logger = get_logger(__name__)

class BaseProcessor(ABC):
    """
    Base class for all data processors (image, video, audio pipelines).
    
    Provides common functionality for:
    - File validation
    - Progress tracking  
    - Error handling
    - Resource management
    """
    
    def __init__(self, name: str, supported_extensions: List[str]):
        """
        Initialize the processor.
        
        Args:
            name: Processor name for logging
            supported_extensions: List of supported file extensions
        """
        self.name = name
        self.supported_extensions = [ext.lower() for ext in supported_extensions]
        self.logger = get_logger(f"processor.{name}")
        self._stats = {
            'files_processed': 0,
            'files_failed': 0,
            'processing_time': 0.0,
            'errors': []
        }
    
    def validate_file(self, file_path: Union[str, Path]) -> Path:
        """
        Validate that a file exists and has a supported extension.
        
        Args:
            file_path: Path to file to validate
            
        Returns:
            Validated Path object
            
        Raises:
            ValidationError: If file doesn't exist or has unsupported extension
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ValidationError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")
        
        if file_path.stat().st_size < MIN_FILE_SIZE:
            raise ValidationError(f"File too small: {file_path} ({file_path.stat().st_size} bytes)")
        
        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValidationError(
                f"Unsupported file extension: {file_path.suffix}. "
                f"Supported: {', '.join(self.supported_extensions)}"
            )
        
        return file_path
    
    def validate_directory(self, dir_path: Union[str, Path], create: bool = False) -> Path:
        """
        Validate that a directory exists or create it.
        
        Args:
            dir_path: Path to directory
            create: Whether to create directory if it doesn't exist
            
        Returns:
            Validated Path object
            
        Raises:
            ValidationError: If directory doesn't exist and create=False
        """
        dir_path = Path(dir_path)
        
        if not dir_path.exists():
            if create:
                dir_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created directory: {dir_path}")
            else:
                raise ValidationError(f"Directory not found: {dir_path}")
        
        if not dir_path.is_dir():
            raise ValidationError(f"Path is not a directory: {dir_path}")
        
        return dir_path
    
    def find_files(self, directory: Union[str, Path], 
                   recursive: bool = True) -> List[Path]:
        """
        Find all supported files in a directory.
        
        Args:
            directory: Directory to search
            recursive: Whether to search recursively
            
        Returns:
            List of found file paths
        """
        directory = self.validate_directory(directory)
        files = []
        
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    self.validate_file(file_path)
                    files.append(file_path)
                except ValidationError as e:
                    self.logger.warning(f"Skipping invalid file {file_path}: {e}")
        
        self.logger.info(f"Found {len(files)} supported files in {directory}")
        return files
    
    @contextmanager
    def track_processing(self, file_path: Path):
        """
        Context manager to track processing time and handle errors.
        
        Args:
            file_path: File being processed
        """
        start_time = time.time()
        try:
            self.logger.debug(f"Processing {file_path}")
            yield
            self._stats['files_processed'] += 1
            self.logger.debug(f"Successfully processed {file_path}")
        except Exception as e:
            self._stats['files_failed'] += 1
            self._stats['errors'].append({
                'file': str(file_path),
                'error': str(e),
                'timestamp': time.time()
            })
            self.logger.error(f"Failed to process {file_path}: {e}")
            raise
        finally:
            processing_time = time.time() - start_time
            self._stats['processing_time'] += processing_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self._stats.copy()
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self._stats = {
            'files_processed': 0,
            'files_failed': 0, 
            'processing_time': 0.0,
            'errors': []
        }
    
    @abstractmethod
    def process_file(self, input_path: Path, output_path: Path, **kwargs) -> bool:
        """
        Process a single file.
        
        Args:
            input_path: Input file path
            output_path: Output file/directory path
            **kwargs: Additional processing parameters
            
        Returns:
            True if processing succeeded, False otherwise
        """
        pass
    
    def process_batch(self, input_files: List[Path], 
                     output_dir: Path, **kwargs) -> Tuple[int, int]:
        """
        Process multiple files.
        
        Args:
            input_files: List of input file paths
            output_dir: Output directory
            **kwargs: Additional processing parameters
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        output_dir = self.validate_directory(output_dir, create=True)
        
        successful = 0
        failed = 0
        
        for file_path in input_files:
            try:
                with self.track_processing(file_path):
                    output_path = output_dir / file_path.name
                    if self.process_file(file_path, output_path, **kwargs):
                        successful += 1
                    else:
                        failed += 1
            except Exception:
                failed += 1
        
        self.logger.info(f"Batch processing complete: {successful} successful, {failed} failed")
        return successful, failed

class BaseModel(ABC):
    """
    Base class for all ML models.
    
    Provides common functionality for:
    - Model loading/saving
    - Configuration management
    - Performance tracking
    - Resource management
    """
    
    def __init__(self, model_type: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model.
        
        Args:
            model_type: Type of model (image, video, audio)
            config: Model configuration
        """
        self.model_type = model_type
        self.config = config or {}
        self.logger = get_logger(f"model.{model_type}")
        self.model = None
        self.is_loaded = False
        self.model_path = None
        self._metrics = {
            'predictions_made': 0,
            'prediction_time': 0.0,
            'accuracy': None,
            'last_updated': None
        }
    
    def validate_model_path(self, model_path: Union[str, Path]) -> Path:
        """
        Validate model file path.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Validated Path object
            
        Raises:
            ValidationError: If model file doesn't exist or has wrong extension
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise ValidationError(f"Model file not found: {model_path}")
        
        if not model_path.is_file():
            raise ValidationError(f"Model path is not a file: {model_path}")
        
        # Check for common model extensions
        valid_extensions = ['.h5', '.pkl', '.pth', '.onnx', '.pb']
        if model_path.suffix.lower() not in valid_extensions:
            self.logger.warning(
                f"Unknown model extension: {model_path.suffix}. "
                f"Expected one of: {', '.join(valid_extensions)}"
            )
        
        return model_path
    
    @contextmanager
    def track_prediction(self):
        """Context manager to track prediction performance."""
        start_time = time.time()
        try:
            yield
            self._metrics['predictions_made'] += 1
        finally:
            prediction_time = time.time() - start_time
            self._metrics['prediction_time'] += prediction_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics."""
        metrics = self._metrics.copy()
        if metrics['predictions_made'] > 0:
            metrics['avg_prediction_time'] = metrics['prediction_time'] / metrics['predictions_made']
        return metrics
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._metrics = {
            'predictions_made': 0,
            'prediction_time': 0.0,
            'accuracy': None,
            'last_updated': None
        }
    
    @abstractmethod
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load model from file.
        
        Args:
            model_path: Path to model file
        """
        pass
    
    @abstractmethod
    def save_model(self, model_path: Union[str, Path]) -> None:
        """
        Save model to file.
        
        Args:
            model_path: Path to save model
        """
        pass
    
    @abstractmethod
    def predict(self, input_data: Any) -> Dict[str, Any]:
        """
        Make prediction on input data.
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            Prediction results
        """
        pass
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self.is_loaded and self.model is not None

class ResourceManager:
    """
    Utility class for managing system resources.
    
    Provides functionality for:
    - Memory monitoring
    - Temporary file cleanup
    - Resource limiting
    """
    
    def __init__(self):
        self.logger = get_logger("resource_manager")
        self._temp_files: List[Path] = []
    
    def track_temp_file(self, file_path: Union[str, Path]) -> Path:
        """
        Track a temporary file for cleanup.
        
        Args:
            file_path: Path to temporary file
            
        Returns:
            Path object
        """
        file_path = Path(file_path)
        self._temp_files.append(file_path)
        return file_path
    
    def cleanup_temp_files(self) -> int:
        """
        Clean up all tracked temporary files.
        
        Returns:
            Number of files cleaned up
        """
        cleaned = 0
        for file_path in self._temp_files[:]:
            try:
                if file_path.exists():
                    file_path.unlink()
                    cleaned += 1
                self._temp_files.remove(file_path)
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp file {file_path}: {e}")
        
        if cleaned > 0:
            self.logger.info(f"Cleaned up {cleaned} temporary files")
        
        return cleaned
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage information.
        
        Returns:
            Dictionary with memory usage stats
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': memory_percent
            }
        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")
            return {}
    
    @contextmanager
    def temporary_file(self, suffix: str = "", prefix: str = "tmp", 
                      directory: Optional[Path] = None):
        """
        Context manager for temporary files with automatic cleanup.
        
        Args:
            suffix: File suffix
            prefix: File prefix
            directory: Directory for temp file
        """
        import tempfile
        
        if directory:
            directory.mkdir(parents=True, exist_ok=True)
        
        with tempfile.NamedTemporaryFile(
            suffix=suffix, prefix=prefix, dir=directory, delete=False
        ) as tmp_file:
            temp_path = Path(tmp_file.name)
        
        try:
            yield temp_path
        finally:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temporary file {temp_path}: {e}")

# Global resource manager instance
_resource_manager: Optional[ResourceManager] = None

def get_resource_manager() -> ResourceManager:
    """Get the global resource manager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager 