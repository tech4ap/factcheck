"""
Database Schema for Deepfake Prediction Results

This module defines the database schema and data structures for storing
comprehensive deepfake detection results with LLM explanations and metadata.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum

class MediaType(Enum):
    """Enumeration for media types."""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"

class PredictionStatus(Enum):
    """Enumeration for prediction status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class DeepfakeReasons:
    """Structured reasons for deepfake detection."""
    visual_artifacts: List[str] = field(default_factory=list)
    temporal_inconsistencies: List[str] = field(default_factory=list)
    audio_anomalies: List[str] = field(default_factory=list)
    metadata_issues: List[str] = field(default_factory=list)
    model_confidence_factors: Dict[str, float] = field(default_factory=dict)
    technical_indicators: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeepfakeReasons':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class PredictionResult:
    """
    Comprehensive deepfake prediction result with database schema mapping.
    
    This class represents the complete prediction result that can be stored
    in a database with the following Sequelize-compatible schema:
    
    {
        id: { type: Sequelize.UUID, primaryKey: true },
        is_deepfake: { type: Sequelize.BOOLEAN },
        confidence_score: { type: Sequelize.FLOAT },
        deepfake_detection_summary: { type: Sequelize.TEXT },
        deepfake_reasons: { type: Sequelize.JSONB },
        received_at: { type: Sequelize.DATE },
        processed_at: { type: Sequelize.DATE },
        error_message: { type: Sequelize.TEXT }
    }
    """
    
    # Primary Fields (Database Schema)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_deepfake: bool = False
    confidence_score: float = 0.0
    deepfake_detection_summary: str = ""
    deepfake_reasons: Dict[str, Any] = field(default_factory=dict)
    received_at: datetime = field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Extended Fields (Additional Metadata)
    request_id: Optional[str] = None
    file_path: str = ""
    file_type: Optional[MediaType] = None
    file_size_bytes: Optional[int] = None
    prediction_score: float = 0.0  # Raw model output (0-1)
    threshold_used: float = 0.5
    model_version: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    status: PredictionStatus = PredictionStatus.PENDING
    
    # S3/AWS Metadata
    is_s3_file: bool = False
    s3_bucket: Optional[str] = None
    s3_key: Optional[str] = None
    local_path: Optional[str] = None
    
    # Model-specific Metadata
    model_metadata: Dict[str, Any] = field(default_factory=dict)
    ensemble_scores: Dict[str, float] = field(default_factory=dict)
    
    # Additional Context
    callback_url: Optional[str] = None
    user_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if isinstance(self.file_type, str):
            self.file_type = MediaType(self.file_type)
        if isinstance(self.status, str):
            self.status = PredictionStatus(self.status)
        if isinstance(self.deepfake_reasons, dict) and not isinstance(self.deepfake_reasons, DeepfakeReasons):
            # Convert dict to DeepfakeReasons if needed
            if self.deepfake_reasons:
                self.deepfake_reasons = DeepfakeReasons.from_dict(self.deepfake_reasons)
            else:
                self.deepfake_reasons = DeepfakeReasons()
    
    def mark_as_processing(self) -> None:
        """Mark prediction as currently being processed."""
        self.status = PredictionStatus.PROCESSING
        if not self.processed_at:
            self.processed_at = datetime.utcnow()
    
    def mark_as_completed(self) -> None:
        """Mark prediction as completed."""
        self.status = PredictionStatus.COMPLETED
        self.processed_at = datetime.utcnow()
    
    def mark_as_failed(self, error_message: str) -> None:
        """Mark prediction as failed with error message."""
        self.status = PredictionStatus.FAILED
        self.error_message = error_message
        self.processed_at = datetime.utcnow()
    
    def calculate_confidence(self) -> float:
        """Calculate confidence score from prediction score."""
        # Confidence is distance from decision boundary (0.5)
        self.confidence_score = abs(self.prediction_score - 0.5) * 2
        return self.confidence_score
    
    def set_prediction_result(self, prediction_score: float, threshold: float = 0.5) -> None:
        """Set prediction result and derived fields."""
        self.prediction_score = prediction_score
        self.threshold_used = threshold
        self.is_deepfake = prediction_score > threshold
        self.calculate_confidence()
    
    def to_database_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary suitable for database insertion.
        Only includes fields that match the database schema.
        """
        return {
            'id': self.id,
            'is_deepfake': self.is_deepfake,
            'confidence_score': self.confidence_score,
            'deepfake_detection_summary': self.deepfake_detection_summary,
            'deepfake_reasons': self.deepfake_reasons.to_dict() if isinstance(self.deepfake_reasons, DeepfakeReasons) else self.deepfake_reasons,
            'received_at': self.received_at.isoformat() if self.received_at else None,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'error_message': self.error_message
        }
    
    def to_extended_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary with all fields for API responses.
        """
        result = self.to_database_dict()
        result.update({
            'request_id': self.request_id,
            'file_path': self.file_path,
            'file_type': self.file_type.value if self.file_type else None,
            'file_size_bytes': self.file_size_bytes,
            'prediction_score': self.prediction_score,
            'threshold_used': self.threshold_used,
            'model_version': self.model_version,
            'processing_time_seconds': self.processing_time_seconds,
            'status': self.status.value if self.status else None,
            'is_s3_file': self.is_s3_file,
            's3_bucket': self.s3_bucket,
            's3_key': self.s3_key,
            'local_path': self.local_path,
            'model_metadata': self.model_metadata,
            'ensemble_scores': self.ensemble_scores,
            'callback_url': self.callback_url,
            'user_metadata': self.user_metadata
        })
        return result
    
    def to_sqs_result(self) -> Dict[str, Any]:
        """
        Convert to dictionary suitable for SQS result messages.
        Maintains backward compatibility with existing format.
        """
        return {
            'id': self.id,
            'request_id': self.request_id,
            'file_path': self.file_path,
            'file_type': self.file_type.value if self.file_type else None,
            'prediction_score': self.prediction_score,
            'confidence': self.confidence_score,
            'label': 'FAKE' if self.is_deepfake else 'REAL',
            'threshold_used': self.threshold_used,
            'is_s3_file': self.is_s3_file,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'deepfake_detection_summary': self.deepfake_detection_summary,
            'deepfake_reasons': self.deepfake_reasons.to_dict() if isinstance(self.deepfake_reasons, DeepfakeReasons) else self.deepfake_reasons,
            'model_metadata': self.model_metadata,
            'ensemble_scores': self.ensemble_scores,
            'error_message': self.error_message,
            'status': self.status.value if self.status else None
        } 