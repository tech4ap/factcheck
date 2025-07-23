"""
Enhanced Deepfake Detection Inference Script

This script loads trained deepfake detection models and makes predictions
on new images, videos, and audio files with improved error handling and
user experience.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import glob
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from models.deepfake_detector import (
    ImageDeepfakeDetector, VideoDeepfakeDetector, AudioDeepfakeDetector,
    EnsembleDeepfakeDetector
)
from training.model_saver import ModelSaver
from utils.s3_utils import S3FileHandler, is_s3_url, get_file_extension_from_s3_url

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
SUPPORTED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'}

class EnhancedDeepfakePredictor:
    """Enhanced class for making deepfake predictions with better error handling."""
    
    def __init__(self, models_dir: str = "models", 
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 aws_session_token: Optional[str] = None,
                 aws_region: str = 'us-east-1'):
        """Initialize the predictor with models directory and AWS credentials."""
        self.models_dir = Path(models_dir)
        self.model_saver = ModelSaver(models_dir)
        self.image_model = None
        self.video_model = None
        self.audio_model = None
        self.ensemble_model = None
        self.loaded_models = {}
        
        # Initialize S3 handler
        self.s3_handler = S3FileHandler(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=aws_region
        )
        
        # Configure TensorFlow
        self._configure_tensorflow()
        
    def _configure_tensorflow(self):
        """Configure TensorFlow settings for inference."""
        try:
            # Set memory growth to avoid GPU memory issues
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            logger.warning(f"Could not configure GPU settings: {e}")
            
    def load_models(self, load_image: bool = True, load_video: bool = True, 
                   load_audio: bool = True, use_best: bool = True) -> Dict[str, bool]:
        """
        Load trained models with enhanced error handling.
        
        Args:
            load_image: Whether to load image model
            load_video: Whether to load video model
            load_audio: Whether to load audio model
            use_best: Whether to use best models or final models
            
        Returns:
            Dictionary indicating which models were successfully loaded
        """
        loaded_models = {}
        
        # Load image model
        if load_image:
            loaded_models['image'] = self._load_single_model('image', use_best)
        
        # Load video model
        if load_video:
            loaded_models['video'] = self._load_single_model('video', use_best)
        
        # Load audio model
        if load_audio:
            loaded_models['audio'] = self._load_single_model('audio', use_best)
            
        self.loaded_models = loaded_models
        return loaded_models
    
    def _load_single_model(self, model_type: str, use_best: bool = True) -> bool:
        """Load a single model of specified type."""
        try:
            # Get model path
            if use_best:
                model_path = self.model_saver.get_best_model_path(model_type)
            else:
                model_path = self.models_dir / f"{model_type}_model_final.h5"
                model_path = str(model_path) if model_path.exists() else None
            
            if not model_path or not Path(model_path).exists():
                logger.warning(f"{model_type.title()} model not found")
                return False
            
            # Load the model
            if model_type == 'image':
                self.image_model = ImageDeepfakeDetector()
                self.image_model.model = tf.keras.models.load_model(model_path)
            elif model_type == 'video':
                self.video_model = VideoDeepfakeDetector()
                self.video_model.model = tf.keras.models.load_model(model_path)
            elif model_type == 'audio':
                self.audio_model = AudioDeepfakeDetector()
                self.audio_model.model = tf.keras.models.load_model(model_path)
            
            logger.info(f"{model_type.title()} model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading {model_type} model: {e}")
            return False
    
    def _download_s3_file_if_needed(self, file_path: str) -> str:
        """Download S3 file if needed and return local path."""
        if is_s3_url(file_path):
            logger.info(f"🔗 Detected S3 URL: {file_path}")
            try:
                local_path = self.s3_handler.download_file(file_path)
                return local_path
            except Exception as e:
                raise RuntimeError(f"Failed to download S3 file: {e}")
        else:
            return file_path
    
    def get_file_type(self, file_path: str) -> Optional[str]:
        """Determine the type of media file."""
        # Handle S3 URLs
        if is_s3_url(file_path):
            file_ext = get_file_extension_from_s3_url(file_path)
        else:
            file_ext = Path(file_path).suffix.lower()
        
        if file_ext in SUPPORTED_IMAGE_EXTENSIONS:
            return 'image'
        elif file_ext in SUPPORTED_VIDEO_EXTENSIONS:
            return 'video'
        elif file_ext in SUPPORTED_AUDIO_EXTENSIONS:
            return 'audio'
        else:
            return None
    
    def predict_image(self, image_path: str, confidence_threshold: float = 0.5) -> Dict[str, any]:
        """
        Predict if an image is fake or real with enhanced output.
        
        Args:
            image_path: Path to image file or S3 URL
            confidence_threshold: Threshold for classification
            
        Returns:
            Dictionary with prediction results
        """
        if self.image_model is None:
            raise ValueError("Image model not loaded")
        
        try:
            # Handle S3 URLs
            original_path = image_path
            local_path = self._download_s3_file_if_needed(image_path)
            
            # Load and validate image
            img = cv2.imread(local_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            original_shape = img.shape
            
            # Preprocess image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Make prediction
            prediction = self.image_model.model.predict(img, verbose=0)[0][0]
            logger.info(f"📄 AJAY1: Prediction: {prediction}")
            if prediction <= 0.5:
                prediction = prediction + 0.5
                logger.info(f"📄 AJAY2: Prediction: {prediction}")
            # Calculate confidence and label
            confidence = abs(prediction - 0.5) * 2
            label = "FAKE" if prediction > confidence_threshold else "REAL"
            
            # Get model metadata
            metadata = self.model_saver.load_model_metadata('image')
            
            return {
                'file_path': original_path,
                'local_path': local_path if is_s3_url(original_path) else None,
                'file_type': 'image',
                'prediction_score': float(prediction),
                'confidence': float(confidence),
                'label': label,
                'threshold_used': confidence_threshold,
                'original_shape': original_shape,
                'model_metadata': metadata,
                'is_s3_file': is_s3_url(original_path),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error predicting image {image_path}: {e}")
            raise
    
    def predict_video(self, video_path: str, frames_per_video: int = 10, 
                     confidence_threshold: float = 0.5) -> Dict[str, any]:
        """
        Predict if a video is fake or real with enhanced output.
        
        Args:
            video_path: Path to video file or S3 URL
            frames_per_video: Number of frames to extract
            confidence_threshold: Threshold for classification
            
        Returns:
            Dictionary with prediction results
        """
        if self.video_model is None:
            raise ValueError("Video model not loaded")
        
        try:
            # Handle S3 URLs
            original_path = video_path
            local_path = self._download_s3_file_if_needed(video_path)
            
            # Extract frames from video
            frames = []
            cap = cv2.VideoCapture(local_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            frame_count = 0
            frame_interval = max(1, total_frames // frames_per_video) if total_frames > 0 else 1
            
            while cap.isOpened() and len(frames) < frames_per_video:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (256, 256))
                    frame = frame / 255.0
                    frames.append(frame)
                
                frame_count += 1
            
            cap.release()
            
            if len(frames) < frames_per_video:
                # Pad with last frame if not enough frames
                while len(frames) < frames_per_video:
                    frames.append(frames[-1] if frames else np.zeros((256, 256, 3)))
            
            # Prepare input
            video_sequence = np.array(frames[:frames_per_video])
            video_sequence = np.expand_dims(video_sequence, axis=0)
            
            # Make prediction
            prediction = self.video_model.model.predict(video_sequence, verbose=0)[0][0]
            
            # Calculate confidence and label
            confidence = abs(prediction - 0.5) * 2
            label = "FAKE" if prediction > confidence_threshold else "REAL"
            
            # Get model metadata
            metadata = self.model_saver.load_model_metadata('video')
            
            return {
                'file_path': original_path,
                'local_path': local_path if is_s3_url(original_path) else None,
                'file_type': 'video',
                'prediction_score': float(prediction),
                'confidence': float(confidence),
                'label': label,
                'threshold_used': confidence_threshold,
                'frames_extracted': len([f for f in frames if f is not None]),
                'total_frames': total_frames,
                'duration_seconds': duration,
                'fps': fps,
                'model_metadata': metadata,
                'is_s3_file': is_s3_url(original_path),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error predicting video {video_path}: {e}")
            raise
    
    def predict_audio(self, audio_path: str, confidence_threshold: float = 0.5) -> Dict[str, any]:
        """
        Predict if an audio file is fake or real with enhanced output.
        
        Args:
            audio_path: Path to audio file or S3 URL
            confidence_threshold: Threshold for classification
            
        Returns:
            Dictionary with prediction results
        """
        if self.audio_model is None:
            raise ValueError("Audio model not loaded")
        
        try:
            # Handle S3 URLs
            original_path = audio_path
            local_path = self._download_s3_file_if_needed(audio_path)
            
            # Load audio and create spectrogram
            y, sr = librosa.load(local_path, sr=22050)
            duration = len(y) / sr
            
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Convert to image format
            spec_img = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            spec_img = cv2.resize(spec_img, (128, 128))
            spec_img = np.expand_dims(spec_img, axis=-1)
            spec_img = np.expand_dims(spec_img, axis=0)
            
            # Make prediction
            prediction = self.audio_model.model.predict(spec_img, verbose=0)[0][0]
            
            # Calculate confidence and label
            confidence = abs(prediction - 0.5) * 2
            label = "FAKE" if prediction > confidence_threshold else "REAL"
            
            # Get model metadata
            metadata = self.model_saver.load_model_metadata('audio')
            
            return {
                'file_path': original_path,
                'local_path': local_path if is_s3_url(original_path) else None,
                'file_type': 'audio',
                'prediction_score': float(prediction),
                'confidence': float(confidence),
                'label': label,
                'threshold_used': confidence_threshold,
                'duration_seconds': duration,
                'sample_rate': sr,
                'model_metadata': metadata,
                'is_s3_file': is_s3_url(original_path),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error predicting audio {audio_path}: {e}")
            raise
    
    def predict_auto(self, file_path: str, confidence_threshold: float = 0.5) -> Dict[str, any]:
        """
        Automatically detect file type and make appropriate prediction.
        
        Args:
            file_path: Path to the file
            confidence_threshold: Threshold for classification
            
        Returns:
            Dictionary with prediction results
        """
        file_type = self.get_file_type(file_path)
        
        if file_type is None:
            raise ValueError(f"Unsupported file type: {Path(file_path).suffix}")
        
        if file_type == 'image':
            return self.predict_image(file_path, confidence_threshold)
        elif file_type == 'video':
            return self.predict_video(file_path, confidence_threshold=confidence_threshold)
        elif file_type == 'audio':
            return self.predict_audio(file_path, confidence_threshold)
        else:
            raise ValueError(f"Unknown file type: {file_type}")
    
    def batch_predict(self, input_dir: str, output_file: str = None,
                     confidence_threshold: float = 0.5) -> pd.DataFrame:
        """
        Batch predict on all supported files in a directory.
        
        Args:
            input_dir: Directory containing files
            output_file: Optional output CSV file
            confidence_threshold: Threshold for classification
            
        Returns:
            DataFrame with all results
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory not found: {input_dir}")
        
        results = []
        
        # Find all supported files
        all_files = []
        for ext in SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS | SUPPORTED_AUDIO_EXTENSIONS:
            all_files.extend(input_path.glob(f"*{ext}"))
            all_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(all_files)} supported files")
        
        # Process each file
        for i, file_path in enumerate(all_files, 1):
            try:
                logger.info(f"Processing {i}/{len(all_files)}: {file_path.name}")
                result = self.predict_auto(str(file_path), confidence_threshold)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                # Add error result
                results.append({
                    'file_path': str(file_path),
                    'file_type': self.get_file_type(str(file_path)),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save to file if specified
        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        
        return df
    
    def list_available_models(self) -> Dict[str, Dict]:
        """List all available models."""
        return self.model_saver.list_available_models()

# Import the common function - keep this for backward compatibility  
try:
    from .common import print_prediction_result
except ImportError:
    # Handle direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from common import print_prediction_result

def main():
    """Enhanced main inference function."""
    try:
        from .common import (
            add_aws_arguments, add_common_arguments, create_predictor_from_args,
            validate_and_load_models, list_available_models, print_batch_summary,
            save_prediction_result, handle_prediction_error, create_enhanced_examples_text
        )
    except ImportError:
        # Handle direct execution
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent))
        from common import (
            add_aws_arguments, add_common_arguments, create_predictor_from_args,
            validate_and_load_models, list_available_models, print_batch_summary,
            save_prediction_result, handle_prediction_error, create_enhanced_examples_text
        )
    
    parser = argparse.ArgumentParser(
        description='Enhanced Deepfake Detection Inference Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=create_enhanced_examples_text('predict_deepfake.py'))
    
    # Add common arguments
    add_common_arguments(parser)
    
    # Add specific arguments for this script
    parser.add_argument('--input', type=str, 
                       help='Input file or directory')
    parser.add_argument('--batch', action='store_true', 
                       help='Process directory in batch mode')
    parser.add_argument('--format', choices=['csv', 'json'], default='csv',
                       help='Output format (default: csv)')
    parser.add_argument('--use-best', action='store_true', default=True,
                       help='Use best models instead of final models')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models and exit')
    
    # Add AWS arguments
    add_aws_arguments(parser)
    
    args = parser.parse_args()
    
    # Create predictor from arguments
    predictor = create_predictor_from_args(args)
    
    # List models if requested
    if args.list_models:
        list_available_models(predictor)
        return
    
    # Validate input is provided when not listing models
    if not args.input:
        logger.error("❌ Input file/directory is required unless using --list-models")
        parser.print_help()
        return
    
    # Load and validate models
    try:
        loaded = validate_and_load_models(predictor, use_best=args.use_best)
    except RuntimeError as e:
        handle_prediction_error(e, "model loading")
        return
    
    # Process input
    input_path = Path(args.input)
    
    # Check if it's an S3 URL or local file
    if not is_s3_url(args.input) and not input_path.exists():
        logger.error(f"❌ Input not found: {args.input}")
        return
    
    try:
        if args.batch or (not is_s3_url(args.input) and input_path.is_dir()):
            # Batch processing
            logger.info("🔄 Starting batch processing...")
            results_df = predictor.batch_predict(
                args.input, 
                output_file=args.output,
                confidence_threshold=args.threshold
            )
            
            # Print summary
            print_batch_summary(results_df)
            
        else:
            # Single file processing
            logger.info(f"🔍 Analyzing file: {args.input}")
            result = predictor.predict_auto(args.input, args.threshold)
            
            # Print result
            print_prediction_result(result)
            
            # Save result if requested
            if args.output:
                save_prediction_result(result, args.output, args.format)
    
    except Exception as e:
        handle_prediction_error(e, "processing")
        raise

if __name__ == "__main__":
    main() 