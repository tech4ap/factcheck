"""
Centralized Configuration Management

This module provides a unified configuration system for the deepfake detection project,
eliminating scattered constants and providing environment-aware settings.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for ML models."""
    # Image model settings
    image_size: tuple = (256, 256)
    image_batch_size: int = 32
    image_dropout_rate: float = 0.3
    image_learning_rate: float = 1e-4
    
    # Video model settings
    video_fps: float = 1.0
    video_frame_sequence_length: int = 10
    video_batch_size: int = 16
    video_dropout_rate: float = 0.3
    video_learning_rate: float = 1e-4
    
    # Audio model settings
    audio_sample_rate: int = 22050
    audio_duration: float = 3.0
    audio_batch_size: int = 32
    audio_dropout_rate: float = 0.3
    audio_learning_rate: float = 1e-4
    
    # Training settings
    epochs: int = 50
    patience: int = 10
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Model architecture
    base_models: Dict[str, str] = field(default_factory=lambda: {
        'image': 'efficientnet',
        'video': 'efficientnet', 
        'audio': 'custom'
    })

@dataclass
class DataConfig:
    """Configuration for data handling."""
    # Base directories
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / 'data')
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / 'models')
    results_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / 'results')
    logs_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / 'logs')
    
    # External data directory (user configurable)
    external_data_dir: Optional[Path] = None
    
    # Supported file extensions
    supported_extensions: Dict[str, list] = field(default_factory=lambda: {
        'image': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'],
        'video': ['.mp4', '.avi', '.mov', '.mkv', '.wmv'],
        'audio': ['.wav', '.mp3', '.flac', '.m4a', '.ogg'],
        'text': ['.txt', '.csv', '.json']
    })
    
    # Cache settings
    temp_dir: Path = field(default_factory=lambda: Path('/tmp/deepfake_cache'))
    s3_cache_dir: Path = field(default_factory=lambda: Path('/tmp/deepfake_s3_cache'))
    cleanup_temp_files: bool = True
    max_cache_size_mb: int = 1024  # 1GB

@dataclass
class AWSConfig:
    """Configuration for AWS services."""
    # Default region and credentials
    region: str = 'us-east-1'
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    session_token: Optional[str] = None
    
    # S3 settings
    s3_bucket: Optional[str] = None
    s3_prefix: str = ''
    s3_endpoint_url: Optional[str] = None
    s3_use_ssl: bool = True
    
    # SQS settings
    sqs_queue_url: Optional[str] = None
    sqs_max_messages: int = 10
    sqs_wait_time_seconds: int = 20
    sqs_visibility_timeout: int = 30
    sqs_poll_interval: int = 5

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = 'INFO'
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format: str = '%Y-%m-%d %H:%M:%S'
    
    # File logging
    log_to_file: bool = True
    log_file: Optional[Path] = None
    max_log_size_mb: int = 10
    backup_count: int = 5
    
    # Console logging
    log_to_console: bool = True
    console_level: str = 'INFO'
    
    # Specific logger levels
    logger_levels: Dict[str, str] = field(default_factory=lambda: {
        'tensorflow': 'WARNING',
        'boto3': 'WARNING',
        'botocore': 'WARNING',
        'urllib3': 'WARNING'
    })

class Config:
    """
    Main configuration class that manages all settings.
    
    This class provides environment-aware configuration with support for:
    - Environment variables
    - Configuration files
    - Runtime overrides
    - Validation and defaults
    """
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None, 
                 environment: str = 'development'):
        """
        Initialize configuration.
        
        Args:
            config_file: Optional path to configuration file
            environment: Environment name (development, production, test)
        """
        self.environment = environment
        self.config_file = Path(config_file) if config_file else None
        
        # Initialize configuration objects
        self.model = ModelConfig()
        self.data = DataConfig()
        self.aws = AWSConfig()
        self.logging = LoggingConfig()
        
        # Load configuration from various sources
        self._load_from_environment()
        if self.config_file and self.config_file.exists():
            self._load_from_file()
        
        # Validate configuration
        self._validate_config()
        
        # Create directories
        self._ensure_directories()
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        # AWS configuration
        if os.getenv('AWS_ACCESS_KEY_ID'):
            self.aws.access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        if os.getenv('AWS_SECRET_ACCESS_KEY'):
            self.aws.secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        if os.getenv('AWS_SESSION_TOKEN'):
            self.aws.session_token = os.getenv('AWS_SESSION_TOKEN')
        if os.getenv('AWS_REGION'):
            self.aws.region = os.getenv('AWS_REGION')
        
        # SQS configuration
        if os.getenv('QUEUE_URL'):
            self.aws.sqs_queue_url = os.getenv('QUEUE_URL')
        
        # Data directories
        if os.getenv('DATA_DIR'):
            self.data.external_data_dir = Path(os.getenv('DATA_DIR'))
        if os.getenv('MODELS_DIR'):
            self.data.models_dir = Path(os.getenv('MODELS_DIR'))
        
        # Logging
        if os.getenv('LOG_LEVEL'):
            self.logging.level = os.getenv('LOG_LEVEL').upper()
        if os.getenv('LOG_FILE'):
            self.logging.log_file = Path(os.getenv('LOG_FILE'))
    
    def _load_from_file(self) -> None:
        """Load configuration from file."""
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configuration objects with file data
            self._update_from_dict(config_data)
            
        except Exception as e:
            logger.warning(f"Failed to load config file {self.config_file}: {e}")
    
    def _update_from_dict(self, config_data: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for section, data in config_data.items():
            if hasattr(self, section) and isinstance(data, dict):
                config_obj = getattr(self, section)
                for key, value in data.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
    
    def _validate_config(self) -> None:
        """Validate configuration settings."""
        # Validate model settings
        if self.model.image_size[0] <= 0 or self.model.image_size[1] <= 0:
            raise ValueError("Image size must be positive")
        
        if not 0 < self.model.validation_split < 1:
            raise ValueError("Validation split must be between 0 and 1")
        
        # Validate data directories
        if self.data.external_data_dir and not self.data.external_data_dir.exists():
            logger.warning(f"External data directory does not exist: {self.data.external_data_dir}")
        
        # Environment-specific validation
        if self.environment == 'production':
            if not self.aws.access_key_id or not self.aws.secret_access_key:
                logger.warning("AWS credentials not configured for production environment")
    
    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.data.models_dir,
            self.data.results_dir,
            self.data.logs_dir,
            self.data.temp_dir,
            self.data.s3_cache_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_data_paths(self) -> Dict[str, Path]:
        """Get all configured data paths."""
        base_dir = self.data.external_data_dir or self.data.data_dir
        
        return {
            'base': base_dir,
            'images': base_dir / 'images',
            'videos': base_dir / 'videos', 
            'audio': base_dir / 'audio',
            'cleaned': self.data.project_root / 'cleaned_data',
            'models': self.data.models_dir,
            'results': self.data.results_dir,
            'logs': self.data.logs_dir,
            'temp': self.data.temp_dir,
            's3_cache': self.data.s3_cache_dir
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'environment': self.environment,
            'model': self.model.__dict__,
            'data': {k: str(v) for k, v in self.data.__dict__.items()},
            'aws': self.aws.__dict__,
            'logging': self.logging.__dict__
        }
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        config_dict = self.to_dict()
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def override(self, **kwargs) -> 'Config':
        """Create a new config with overridden values."""
        new_config = Config(self.config_file, self.environment)
        
        for key, value in kwargs.items():
            if '.' in key:
                section, attr = key.split('.', 1)
                if hasattr(new_config, section):
                    config_obj = getattr(new_config, section)
                    if hasattr(config_obj, attr):
                        setattr(config_obj, attr, value)
            else:
                if hasattr(new_config, key):
                    setattr(new_config, key, value)
        
        return new_config

# Global configuration instance
_config: Optional[Config] = None

def get_config(config_file: Optional[Union[str, Path]] = None,
               environment: Optional[str] = None,
               force_reload: bool = False) -> Config:
    """
    Get the global configuration instance.
    
    Args:
        config_file: Optional configuration file path
        environment: Environment name (auto-detected if not provided)
        force_reload: Force reloading of configuration
        
    Returns:
        Configuration instance
    """
    global _config
    
    if _config is None or force_reload:
        # Auto-detect environment if not provided
        if environment is None:
            environment = os.getenv('DEEPFAKE_ENV', 'development')
        
        _config = Config(config_file, environment)
    
    return _config

def reset_config() -> None:
    """Reset the global configuration instance."""
    global _config
    _config = None 