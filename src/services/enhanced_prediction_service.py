"""
Enhanced Prediction Service

This module provides an enhanced prediction service that integrates the new
database schema, LLM explanations, and comprehensive result structure with
existing deepfake detection functionality.
"""

import logging
import time
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from ..models.prediction_schema import PredictionResult, MediaType, DeepfakeReasons, PredictionStatus
from ..services.llm_explainer import create_explainer
from ..inference.predict_deepfake import EnhancedDeepfakePredictor

logger = logging.getLogger(__name__)

class EnhancedPredictionService:
    """
    Enhanced prediction service that provides comprehensive deepfake detection
    with LLM explanations and database-ready results.
    
    This service wraps the existing prediction functionality and enhances it with:
    - Comprehensive database schema support
    - LLM-powered explanations
    - Structured reasoning
    - Enhanced metadata collection
    - Performance tracking
    """
    
    def __init__(self, models_dir: str = "models", 
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 aws_session_token: Optional[str] = None,
                 aws_region: str = 'us-east-1'):
        """
        Initialize the enhanced prediction service.
        
        Args:
            models_dir: Directory containing trained models
            aws_access_key_id: AWS access key ID (optional)
            aws_secret_access_key: AWS secret access key (optional)
            aws_session_token: AWS session token (optional)
            aws_region: AWS region
        """
        self.models_dir = models_dir
        
        # Initialize the underlying predictor
        self.predictor = EnhancedDeepfakePredictor(
            models_dir=models_dir,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            aws_region=aws_region
        )
        
        # Load models automatically
        try:
            loaded_models = self.predictor.load_models(use_best=True)
            loaded_count = sum(loaded_models.values())
            logger.info(f"Loaded {loaded_count} models: {[k for k, v in loaded_models.items() if v]}")
        except Exception as e:
            logger.warning(f"Failed to load some models: {e}")
            # Try loading final models as fallback
            try:
                loaded_models = self.predictor.load_models(use_best=False)
                loaded_count = sum(loaded_models.values())
                logger.info(f"Loaded {loaded_count} final models: {[k for k, v in loaded_models.items() if v]}")
            except Exception as e2:
                logger.error(f"Failed to load any models: {e2}")
        
        # Initialize the LLM explainer
        self.explainer = create_explainer()
        
        logger.info("Enhanced prediction service initialized")
    
    def predict_comprehensive(self, file_path: str, 
                            confidence_threshold: float = 0.5,
                            request_id: Optional[str] = None,
                            user_metadata: Optional[Dict[str, Any]] = None,
                            callback_url: Optional[str] = None) -> PredictionResult:
        """
        Perform comprehensive deepfake prediction with enhanced results.
        
        Args:
            file_path: Path to the file (local or S3 URL)
            confidence_threshold: Threshold for classification
            request_id: Optional request ID for tracking
            user_metadata: Optional user-provided metadata
            callback_url: Optional callback URL for notifications
            
        Returns:
            Comprehensive PredictionResult with all database fields populated
        """
        # Create initial prediction result
        result = PredictionResult(
            request_id=request_id,
            file_path=file_path,
            received_at=datetime.utcnow(),
            threshold_used=confidence_threshold,
            callback_url=callback_url,
            user_metadata=user_metadata or {}
        )
        
        # Mark as processing
        result.mark_as_processing()
        
        try:
            # Start timing
            start_time = time.time()
            
            # Get file metadata
            self._collect_file_metadata(result)
            
            # Perform the actual prediction using existing predictor
            prediction_dict = self._perform_prediction(file_path, confidence_threshold)
            
            # Calculate processing time
            result.processing_time_seconds = time.time() - start_time
            
            # Map prediction results to our enhanced schema
            self._map_prediction_results(result, prediction_dict)
            
            # Generate LLM explanation and structured reasons
            self._generate_explanations(result)
            
            # Mark as completed
            result.mark_as_completed()
            
            logger.info(f"Prediction completed for {file_path}: {result.is_deepfake} "
                       f"(confidence: {result.confidence_score:.1%})")
            
        except Exception as e:
            error_message = f"Prediction failed: {str(e)}"
            logger.error(error_message)
            result.mark_as_failed(error_message)
        
        return result
    
    def _collect_file_metadata(self, result: PredictionResult):
        """Collect file metadata and populate result fields."""
        try:
            # Determine if it's an S3 file
            if result.file_path.startswith('s3://'):
                result.is_s3_file = True
                # Parse S3 URL
                parts = result.file_path.replace('s3://', '').split('/', 1)
                if len(parts) == 2:
                    result.s3_bucket = parts[0]
                    result.s3_key = parts[1]
            
            # Determine file type from extension
            file_extension = Path(result.file_path).suffix.lower()
            if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                result.file_type = MediaType.IMAGE
            elif file_extension in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']:
                result.file_type = MediaType.VIDEO
            elif file_extension in ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']:
                result.file_type = MediaType.AUDIO
            
            # Get file size if it's a local file
            if not result.is_s3_file and os.path.exists(result.file_path):
                result.file_size_bytes = os.path.getsize(result.file_path)
                
        except Exception as e:
            logger.warning(f"Failed to collect file metadata: {e}")
    
    def _perform_prediction(self, file_path: str, confidence_threshold: float) -> Dict[str, Any]:
        """Perform the actual prediction using the existing predictor."""
        try:
            # Use the existing predictor's auto-prediction functionality
            return self.predictor.predict_auto(file_path, confidence_threshold)
        except Exception as e:
            logger.error(f"Prediction failed for {file_path}: {e}")
            raise
    
    def _map_prediction_results(self, result: PredictionResult, prediction_dict: Dict[str, Any]):
        """Map results from existing predictor format to enhanced schema."""
        try:
            # Core prediction fields
            result.prediction_score = prediction_dict.get('prediction_score', 0.0)
            result.set_prediction_result(result.prediction_score, result.threshold_used)
            
            # Model metadata
            result.model_metadata = prediction_dict.get('model_metadata', {})
            
            # File metadata from prediction
            if 'local_path' in prediction_dict:
                result.local_path = prediction_dict['local_path']
            
            # Ensemble scores if available
            if 'ensemble_scores' in prediction_dict:
                result.ensemble_scores = prediction_dict['ensemble_scores']
            
            # Additional fields from original prediction
            for field in ['original_shape', 'frames_extracted', 'total_frames', 
                         'duration_seconds', 'fps', 'sample_rate', 's3_file_info']:
                if field in prediction_dict:
                    result.model_metadata[field] = prediction_dict[field]
                    
        except Exception as e:
            logger.error(f"Failed to map prediction results: {e}")
            raise
    
    def _generate_explanations(self, result: PredictionResult):
        """Generate LLM explanations and structured reasons."""
        try:
            # Generate structured reasons first
            result.deepfake_reasons = self.explainer.generate_structured_reasons(result)
            
            # Generate human-readable explanation
            result.deepfake_detection_summary = self.explainer.generate_explanation(result)
            
            logger.debug(f"Generated explanation for {result.file_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate explanations: {e}")
            # Provide fallback explanation
            result.deepfake_detection_summary = (
                f"AI analysis classified this {result.file_type.value if result.file_type else 'content'} "
                f"as {'deepfake' if result.is_deepfake else 'authentic'} with "
                f"{result.confidence_score:.1%} confidence."
            )
            result.deepfake_reasons = DeepfakeReasons()
    
    def predict_batch(self, file_paths: list, 
                     confidence_threshold: float = 0.5,
                     user_metadata: Optional[Dict[str, Any]] = None) -> list[PredictionResult]:
        """
        Perform batch prediction on multiple files.
        
        Args:
            file_paths: List of file paths to process
            confidence_threshold: Threshold for classification
            user_metadata: Optional user-provided metadata
            
        Returns:
            List of PredictionResult objects
        """
        results = []
        
        for i, file_path in enumerate(file_paths):
            try:
                logger.info(f"Processing batch item {i+1}/{len(file_paths)}: {file_path}")
                
                result = self.predict_comprehensive(
                    file_path=file_path,
                    confidence_threshold=confidence_threshold,
                    request_id=f"batch_{int(time.time())}_{i}",
                    user_metadata=user_metadata
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                # Create failed result
                failed_result = PredictionResult(
                    file_path=file_path,
                    request_id=f"batch_{int(time.time())}_{i}",
                    user_metadata=user_metadata or {}
                )
                failed_result.mark_as_failed(str(e))
                results.append(failed_result)
        
        logger.info(f"Batch processing completed: {len(results)} files processed")
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        try:
            # Get loaded models from predictor
            loaded_models = getattr(self.predictor, 'loaded_models', {})
            models_available = [k for k, v in loaded_models.items() if v]
            
            return {
                'models_dir': str(self.models_dir),
                'available_models': models_available,
                'models_loaded': len(models_available)
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {'error': str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the prediction service."""
        try:
            # Get loaded models from predictor
            loaded_models = getattr(self.predictor, 'loaded_models', {})
            models_ready = sum(loaded_models.values())
            
            return {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'models_loaded': models_ready,
                'explainer_ready': self.explainer is not None
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

# Factory function for easy instantiation
def create_prediction_service(**kwargs) -> EnhancedPredictionService:
    """Create and return a configured EnhancedPredictionService instance."""
    return EnhancedPredictionService(**kwargs) 