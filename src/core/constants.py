"""
Centralized Constants

This module contains all constants used throughout the deepfake detection project,
eliminating magic numbers and providing a single source of truth for system-wide values.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================================
# FILE FORMATS AND EXTENSIONS
# ============================================================================

# Supported media file extensions
IMAGE_EXTENSIONS: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif']
VIDEO_EXTENSIONS: List[str] = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
AUDIO_EXTENSIONS: List[str] = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
TEXT_EXTENSIONS: List[str] = ['.txt', '.csv', '.json', '.xml']

# MIME types for validation
IMAGE_MIME_TYPES: List[str] = [
    'image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp', 'image/gif'
]
VIDEO_MIME_TYPES: List[str] = [
    'video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo', 'video/webm'
]
AUDIO_MIME_TYPES: List[str] = [
    'audio/wav', 'audio/mpeg', 'audio/flac', 'audio/ogg', 'audio/aac'
]

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# Image processing
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (256, 256)
MIN_IMAGE_SIZE: Tuple[int, int] = (64, 64)
MAX_IMAGE_SIZE: Tuple[int, int] = (1024, 1024)
DEFAULT_IMAGE_CHANNELS: int = 3

# Video processing
DEFAULT_VIDEO_FPS: float = 1.0
MIN_VIDEO_FPS: float = 0.1
MAX_VIDEO_FPS: float = 30.0
DEFAULT_FRAME_SEQUENCE_LENGTH: int = 10
MIN_FRAME_SEQUENCE_LENGTH: int = 5
MAX_FRAME_SEQUENCE_LENGTH: int = 50

# Audio processing
DEFAULT_AUDIO_SAMPLE_RATE: int = 22050
SUPPORTED_AUDIO_SAMPLE_RATES: List[int] = [16000, 22050, 44100, 48000]
DEFAULT_AUDIO_DURATION: float = 3.0
MIN_AUDIO_DURATION: float = 1.0
MAX_AUDIO_DURATION: float = 10.0

# ============================================================================
# TRAINING CONFIGURATIONS
# ============================================================================

# Batch sizes
DEFAULT_BATCH_SIZE: int = 32
MIN_BATCH_SIZE: int = 1
MAX_BATCH_SIZE: int = 128

# Training parameters
DEFAULT_EPOCHS: int = 50
MIN_EPOCHS: int = 1
MAX_EPOCHS: int = 1000
DEFAULT_PATIENCE: int = 10
MIN_PATIENCE: int = 3
MAX_PATIENCE: int = 50

# Learning rates
DEFAULT_LEARNING_RATE: float = 1e-4
MIN_LEARNING_RATE: float = 1e-6
MAX_LEARNING_RATE: float = 1e-1
FINE_TUNE_LEARNING_RATE: float = 1e-5

# Regularization
DEFAULT_DROPOUT_RATE: float = 0.3
MIN_DROPOUT_RATE: float = 0.0
MAX_DROPOUT_RATE: float = 0.8
DEFAULT_L2_REGULARIZATION: float = 0.01

# Data splits
DEFAULT_VALIDATION_SPLIT: float = 0.2
DEFAULT_TEST_SPLIT: float = 0.1
MIN_SPLIT_RATIO: float = 0.05
MAX_SPLIT_RATIO: float = 0.4

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

# Supported base models
SUPPORTED_IMAGE_MODELS: List[str] = ['efficientnet', 'resnet', 'vgg', 'inception']
SUPPORTED_VIDEO_MODELS: List[str] = ['efficientnet', 'resnet', 'lstm', 'gru']
SUPPORTED_AUDIO_MODELS: List[str] = ['custom', 'wav2vec', 'mfcc']

# Model file extensions
MODEL_EXTENSIONS: List[str] = ['.h5', '.pkl', '.pth', '.onnx', '.pb']

# ============================================================================
# PREDICTION THRESHOLDS
# ============================================================================

# Confidence thresholds
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5
HIGH_CONFIDENCE_THRESHOLD: float = 0.8
LOW_CONFIDENCE_THRESHOLD: float = 0.2

# Class labels
REAL_LABEL: int = 0
FAKE_LABEL: int = 1
CLASS_NAMES: List[str] = ['real', 'fake']

# ============================================================================
# AWS CONFIGURATIONS
# ============================================================================

# Default AWS settings
DEFAULT_AWS_REGION: str = 'us-east-1'
DEFAULT_S3_BUCKET_PREFIX: str = 'deepfake-detection'

# SQS settings
DEFAULT_SQS_MAX_MESSAGES: int = 10
DEFAULT_SQS_WAIT_TIME: int = 20
DEFAULT_SQS_VISIBILITY_TIMEOUT: int = 30
DEFAULT_SQS_POLL_INTERVAL: int = 5
MAX_SQS_MESSAGE_SIZE: int = 262144  # 256KB

# S3 settings
MAX_S3_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
S3_MULTIPART_THRESHOLD: int = 64 * 1024 * 1024  # 64MB
S3_MAX_CONCURRENCY: int = 10

# ============================================================================
# CACHING AND PERFORMANCE
# ============================================================================

# Cache settings
DEFAULT_CACHE_SIZE_MB: int = 1024  # 1GB
MIN_CACHE_SIZE_MB: int = 100
MAX_CACHE_SIZE_MB: int = 10240  # 10GB
CACHE_CLEANUP_THRESHOLD: float = 0.8  # Clean when 80% full

# Memory management
MAX_MEMORY_USAGE_MB: int = 4096  # 4GB
MEMORY_WARNING_THRESHOLD: float = 0.85
MEMORY_CRITICAL_THRESHOLD: float = 0.95

# Processing limits
MAX_CONCURRENT_DOWNLOADS: int = 5
MAX_CONCURRENT_PREDICTIONS: int = 3
DEFAULT_TIMEOUT_SECONDS: int = 300  # 5 minutes
MAX_TIMEOUT_SECONDS: int = 1800  # 30 minutes

# ============================================================================
# LOGGING AND MONITORING
# ============================================================================

# Log levels
LOG_LEVELS: List[str] = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
DEFAULT_LOG_LEVEL: str = 'INFO'

# Log rotation
DEFAULT_LOG_SIZE_MB: int = 10
DEFAULT_LOG_BACKUP_COUNT: int = 5
LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'

# Monitoring intervals
HEALTH_CHECK_INTERVAL: int = 60  # seconds
METRICS_COLLECTION_INTERVAL: int = 300  # 5 minutes
CLEANUP_INTERVAL: int = 3600  # 1 hour

# ============================================================================
# VALIDATION RULES
# ============================================================================

# File size limits
MIN_FILE_SIZE: int = 1024  # 1KB
MAX_IMAGE_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
MAX_VIDEO_FILE_SIZE: int = 500 * 1024 * 1024  # 500MB
MAX_AUDIO_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB

# Quality thresholds
MIN_IMAGE_QUALITY: float = 0.5
MIN_AUDIO_QUALITY: float = 0.3
MIN_VIDEO_QUALITY: float = 0.4

# Processing limits
MAX_PROCESSING_TIME_SECONDS: int = 600  # 10 minutes
MAX_RETRIES: int = 3
RETRY_DELAY_SECONDS: int = 5

# ============================================================================
# DIRECTORY STRUCTURES
# ============================================================================

# Standard directory names
DATA_DIR_NAME: str = 'data'
MODELS_DIR_NAME: str = 'models'
RESULTS_DIR_NAME: str = 'results'
LOGS_DIR_NAME: str = 'logs'
CACHE_DIR_NAME: str = 'cache'
TEMP_DIR_NAME: str = 'temp'

# Data subdirectories
TRAIN_DIR_NAME: str = 'training'
VALIDATION_DIR_NAME: str = 'validation'
TEST_DIR_NAME: str = 'testing'
REAL_DIR_NAME: str = 'real'
FAKE_DIR_NAME: str = 'fake'

# ============================================================================
# API AND MESSAGING
# ============================================================================

# API response codes
SUCCESS_CODE: int = 200
VALIDATION_ERROR_CODE: int = 400
NOT_FOUND_CODE: int = 404
SERVER_ERROR_CODE: int = 500

# Message queue settings
MAX_MESSAGE_RETRIES: int = 3
MESSAGE_RETENTION_HOURS: int = 24
DEAD_LETTER_QUEUE_ENABLED: bool = True

# Callback settings
DEFAULT_CALLBACK_TIMEOUT: int = 30
MAX_CALLBACK_RETRIES: int = 3

# ============================================================================
# SECURITY AND VALIDATION
# ============================================================================

# Security settings
MAX_REQUEST_SIZE: int = 100 * 1024 * 1024  # 100MB
ALLOWED_HOSTS: List[str] = ['localhost', '127.0.0.1']
SECURE_FILENAME_PATTERN: str = r'^[a-zA-Z0-9._-]+$'

# Validation patterns
S3_URL_PATTERN: str = r'^s3://[a-zA-Z0-9.\-_]{1,255}/[a-zA-Z0-9.\-_/]{1,1024}$'
UUID_PATTERN: str = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'

# Rate limiting
DEFAULT_RATE_LIMIT: int = 100  # requests per minute
BURST_RATE_LIMIT: int = 10  # requests per second

# ============================================================================
# HELPER DICTIONARIES
# ============================================================================

# Consolidated extension mapping
SUPPORTED_EXTENSIONS: Dict[str, List[str]] = {
    'image': IMAGE_EXTENSIONS,
    'video': VIDEO_EXTENSIONS,
    'audio': AUDIO_EXTENSIONS,
    'text': TEXT_EXTENSIONS
}

# MIME type mapping
SUPPORTED_MIME_TYPES: Dict[str, List[str]] = {
    'image': IMAGE_MIME_TYPES,
    'video': VIDEO_MIME_TYPES,
    'audio': AUDIO_MIME_TYPES
}

# Default model configurations
DEFAULT_MODEL_CONFIG: Dict[str, Dict[str, any]] = {
    'image': {
        'size': DEFAULT_IMAGE_SIZE,
        'batch_size': DEFAULT_BATCH_SIZE,
        'dropout_rate': DEFAULT_DROPOUT_RATE,
        'learning_rate': DEFAULT_LEARNING_RATE,
        'base_model': 'efficientnet'
    },
    'video': {
        'fps': DEFAULT_VIDEO_FPS,
        'sequence_length': DEFAULT_FRAME_SEQUENCE_LENGTH,
        'batch_size': DEFAULT_BATCH_SIZE // 2,  # Smaller batch for video
        'dropout_rate': DEFAULT_DROPOUT_RATE,
        'learning_rate': DEFAULT_LEARNING_RATE,
        'base_model': 'efficientnet'
    },
    'audio': {
        'sample_rate': DEFAULT_AUDIO_SAMPLE_RATE,
        'duration': DEFAULT_AUDIO_DURATION,
        'batch_size': DEFAULT_BATCH_SIZE,
        'dropout_rate': DEFAULT_DROPOUT_RATE,
        'learning_rate': DEFAULT_LEARNING_RATE,
        'base_model': 'custom'
    }
}

# Environment variable names
ENV_VAR_NAMES: Dict[str, str] = {
    'log_level': 'LOG_LEVEL',
    'data_dir': 'DATA_DIR',
    'models_dir': 'MODELS_DIR',
    'aws_region': 'AWS_REGION',
    'aws_access_key': 'AWS_ACCESS_KEY_ID',
    'aws_secret_key': 'AWS_SECRET_ACCESS_KEY',
    'queue_url': 'QUEUE_URL',
    'environment': 'DEEPFAKE_ENV'
}

# Error messages
ERROR_MESSAGES: Dict[str, str] = {
    'file_not_found': 'File not found: {file_path}',
    'invalid_format': 'Invalid file format: {file_path}. Supported formats: {formats}',
    'model_not_loaded': 'Model not loaded: {model_type}',
    'aws_credentials_missing': 'AWS credentials not configured',
    'invalid_s3_url': 'Invalid S3 URL format: {url}',
    'processing_timeout': 'Processing timeout after {timeout} seconds',
    'memory_limit_exceeded': 'Memory usage exceeded limit: {usage}MB > {limit}MB'
} 