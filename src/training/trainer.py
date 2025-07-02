"""
Trainer Module for Deepfake Detection

This module provides modular training capabilities for different media types
with comprehensive logging, error handling, and memory management.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
import gc

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.deepfake_detector import (
    ImageDeepfakeDetector, VideoDeepfakeDetector, AudioDeepfakeDetector,
    create_callbacks
)
from .data_loader import DataLoader
from .visualization import ModelVisualizer
from .evaluation import ModelEvaluator

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Modular trainer class for deepfake detection models.
    
    This class provides separate training methods for each media type
    with comprehensive logging, error handling, and memory management.
    """
    
    def __init__(self, data_dir: str, output_dir: str = "models", 
                 max_samples: Optional[int] = None):
        """
        Initialize the model trainer.
        
        Args:
            data_dir (str): Path to the directory containing processed data
            output_dir (str): Directory to save trained models and results
            max_samples (Optional[int]): Maximum number of samples to use (for testing)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.max_samples = max_samples
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader(str(self.data_dir), max_samples)
        self.visualizer = ModelVisualizer(str(self.output_dir))
        self.evaluator = ModelEvaluator(str(self.output_dir))
        
        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        logger.info(f"Found {len(gpus)} GPU(s): {gpus}")
        
        if gpus:
            # Enable memory growth to prevent GPU memory overflow
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU memory growth enabled")
        
        logger.info(f"Initialized ModelTrainer with output directory: {self.output_dir}")
    
    def _create_data_augmentation(self) -> ImageDataGenerator:
        """
        Create data augmentation generator for image training.
        
        Returns:
            ImageDataGenerator: Configured data augmentation generator
        """
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest',
            validation_split=0.2
        )
    
    def _save_training_history(self, history: tf.keras.callbacks.History, 
                              model_name: str, training_time: float = None) -> None:
        """
        Save training history plots and metrics using the visualizer.
        
        Args:
            history (tf.keras.callbacks.History): Training history object
            model_name (str): Name of the model for file naming
            training_time (float): Total training time in seconds
        """
        try:
            # Use the visualizer to create comprehensive plots
            self.visualizer.plot_training_history(history, model_name)
            
            # Create training summary report if training time is provided
            if training_time is not None:
                self.visualizer.create_training_summary_report(history, model_name, training_time)
            
            logger.info(f"Saved comprehensive training visualizations for {model_name}")
            
        except Exception as e:
            logger.warning(f"Error saving training history for {model_name}: {e}")
    
    def train_image_model(self, epochs: int = 50, batch_size: int = 32, 
                         fine_tune: bool = True, learning_rate: float = 1e-4) -> Optional[ImageDeepfakeDetector]:
        """
        Train the image deepfake detection model.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            fine_tune (bool): Whether to perform fine-tuning
            learning_rate (float): Initial learning rate
            
        Returns:
            Optional[ImageDeepfakeDetector]: Trained model or None if training failed
        """
        logger.info("Starting image model training...")
        start_time = time.time()
        
        try:
            # Load data
            logger.info("Loading image data...")
            train_images, train_labels = self.data_loader.load_image_data('train')
            val_images, val_labels = self.data_loader.load_image_data('val')
            
            if len(train_images) == 0:
                logger.error("No training data found for images")
                return None
            
            logger.info(f"Loaded {len(train_images)} training images, {len(val_images)} validation images")
            
            # Create model
            logger.info("Building image model...")
            image_model = ImageDeepfakeDetector()
            model = image_model.build_model(learning_rate=learning_rate)
            
            # Data augmentation
            datagen = self._create_data_augmentation()
            
            # Create callbacks
            callbacks = create_callbacks("image_model")
            
            # Train model
            logger.info("Starting initial training...")
            history = model.fit(
                datagen.flow(train_images, train_labels, batch_size=batch_size),
                steps_per_epoch=max(1, len(train_images) // batch_size),
                epochs=epochs,
                validation_data=(val_images, val_labels),
                callbacks=callbacks,
                verbose=1
            )
            
            # Fine-tuning
            if fine_tune:
                logger.info("Starting fine-tuning...")
                image_model.fine_tune()
                
                fine_tune_history = model.fit(
                    datagen.flow(train_images, train_labels, batch_size=batch_size),
                    steps_per_epoch=max(1, len(train_images) // batch_size),
                    epochs=epochs // 2,
                    validation_data=(val_images, val_labels),
                    callbacks=callbacks,
                    verbose=1
                )
            
            # Save model and history
            model.save(self.output_dir / "image_model_final.h5")
            self._save_training_history(history, "image_model", time.time() - start_time)
            
            logger.info("Image model training completed")
            
            return image_model
            
        except Exception as e:
            logger.error(f"Error during image model training: {e}")
            return None
    
    def train_video_model(self, epochs: int = 50, batch_size: int = 16, 
                         frames_per_video: int = 10, learning_rate: float = 1e-4) -> Optional[VideoDeepfakeDetector]:
        """
        Train the video deepfake detection model.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            frames_per_video (int): Number of frames per video sequence
            learning_rate (float): Learning rate
            
        Returns:
            Optional[VideoDeepfakeDetector]: Trained model or None if training failed
        """
        logger.info("Starting video model training...")
        start_time = time.time()
        
        try:
            # Load data using directory structure
            logger.info("Loading video data...")
            train_videos, train_labels = self.data_loader.load_video_data_from_directories('train', frames_per_video)
            val_videos, val_labels = self.data_loader.load_video_data_from_directories('validation', frames_per_video)
            
            if len(train_videos) == 0:
                logger.error("No training data found for videos")
                return None
            
            logger.info(f"Loaded {len(train_videos)} training videos, {len(val_videos)} validation videos")
            
            # Create model
            logger.info("Building video model...")
            video_model = VideoDeepfakeDetector(frame_sequence_length=frames_per_video)
            model = video_model.build_model(learning_rate=learning_rate)
            
            # Create callbacks
            callbacks = create_callbacks("video_model")
            
            # Train model
            logger.info("Starting video model training...")
            history = model.fit(
                train_videos, train_labels,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(val_videos, val_labels),
                callbacks=callbacks,
                verbose=1
            )
            
            # Save model and history
            model.save(self.output_dir / "video_model_final.h5")
            self._save_training_history(history, "video_model", time.time() - start_time)
            
            logger.info("Video model training completed")
            
            return video_model
            
        except Exception as e:
            logger.error(f"Error during video model training: {e}")
            return None
    
    def train_audio_model(self, epochs: int = 50, batch_size: int = 32, 
                         learning_rate: float = 1e-4) -> Optional[AudioDeepfakeDetector]:
        """
        Train the audio deepfake detection model.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            learning_rate (float): Learning rate
            
        Returns:
            Optional[AudioDeepfakeDetector]: Trained model or None if training failed
        """
        logger.info("Starting audio model training...")
        start_time = time.time()
        
        try:
            # Load data
            logger.info("Loading audio data...")
            train_audio, train_labels = self.data_loader.load_audio_data('train')
            val_audio, val_labels = self.data_loader.load_audio_data('val')
            
            if len(train_audio) == 0:
                logger.error("No training data found for audio")
                return None
            
            logger.info(f"Loaded {len(train_audio)} training spectrograms, {len(val_audio)} validation spectrograms")
            
            # Create model
            logger.info("Building audio model...")
            audio_model = AudioDeepfakeDetector()
            model = audio_model.build_model(learning_rate=learning_rate)
            
            # Create callbacks
            callbacks = create_callbacks("audio_model")
            
            # Train model
            logger.info("Starting audio model training...")
            history = model.fit(
                train_audio, train_labels,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(val_audio, val_labels),
                callbacks=callbacks,
                verbose=1
            )
            
            # Save model and history
            model.save(self.output_dir / "audio_model_final.h5")
            self._save_training_history(history, "audio_model", time.time() - start_time)
            
            logger.info("Audio model training completed")
            
            return audio_model
            
        except Exception as e:
            logger.error(f"Error during audio model training: {e}")
            return None
    
    def evaluate_models(self, image_model: Optional[ImageDeepfakeDetector] = None,
                       video_model: Optional[VideoDeepfakeDetector] = None,
                       audio_model: Optional[AudioDeepfakeDetector] = None,
                       frames_per_video: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models on test data with comprehensive visualizations.
        
        Args:
            image_model: Trained image model
            video_model: Trained video model
            audio_model: Trained audio model
            frames_per_video: Number of frames per video for evaluation
            
        Returns:
            Dict[str, Dict[str, float]]: Evaluation results for each model
        """
        logger.info("Starting comprehensive model evaluation...")
        results = {}
        detailed_results = {}
        
        # Evaluate image model
        if image_model:
            logger.info("Evaluating image model...")
            test_images, test_labels = self.data_loader.load_image_data('test')
            if len(test_images) > 0:
                detailed_results['image'] = self.evaluator.evaluate_model_comprehensive(
                    image_model.model, test_images, test_labels, "image_model"
                )
                results['image'] = detailed_results['image']['metrics']
                
                # Create comprehensive visualizations
                self.visualizer.plot_comprehensive_evaluation(
                    test_labels, 
                    detailed_results['image']['predictions']['binary'],
                    detailed_results['image']['predictions']['probabilities'],
                    "image_model",
                    detailed_results['image']['curves']['roc']['fpr'],
                    detailed_results['image']['curves']['roc']['tpr'],
                    detailed_results['image']['curves']['pr']['precision'],
                    detailed_results['image']['curves']['pr']['recall'],
                    detailed_results['image']['classification_report']
                )
                logger.info(f"Image model evaluation completed")
        
        # Evaluate video model
        if video_model:
            logger.info("Evaluating video model...")
            test_videos, test_labels = self.data_loader.load_video_data_from_directories('testing', frames_per_video)
            if len(test_videos) > 0:
                detailed_results['video'] = self.evaluator.evaluate_model_comprehensive(
                    video_model.model, test_videos, test_labels, "video_model"
                )
                results['video'] = detailed_results['video']['metrics']
                
                # Create comprehensive visualizations
                self.visualizer.plot_comprehensive_evaluation(
                    test_labels, 
                    detailed_results['video']['predictions']['binary'],
                    detailed_results['video']['predictions']['probabilities'],
                    "video_model",
                    detailed_results['video']['curves']['roc']['fpr'],
                    detailed_results['video']['curves']['roc']['tpr'],
                    detailed_results['video']['curves']['pr']['precision'],
                    detailed_results['video']['curves']['pr']['recall'],
                    detailed_results['video']['classification_report']
                )
                logger.info(f"Video model evaluation completed")
        
        # Evaluate audio model
        if audio_model:
            logger.info("Evaluating audio model...")
            test_audio, test_labels = self.data_loader.load_audio_data('test')
            if len(test_audio) > 0:
                detailed_results['audio'] = self.evaluator.evaluate_model_comprehensive(
                    audio_model.model, test_audio, test_labels, "audio_model"
                )
                results['audio'] = detailed_results['audio']['metrics']
                
                # Create comprehensive visualizations
                self.visualizer.plot_comprehensive_evaluation(
                    test_labels, 
                    detailed_results['audio']['predictions']['binary'],
                    detailed_results['audio']['predictions']['probabilities'],
                    "audio_model",
                    detailed_results['audio']['curves']['roc']['fpr'],
                    detailed_results['audio']['curves']['roc']['tpr'],
                    detailed_results['audio']['curves']['pr']['precision'],
                    detailed_results['audio']['curves']['pr']['recall'],
                    detailed_results['audio']['classification_report']
                )
                logger.info(f"Audio model evaluation completed")
        
        # Save comprehensive results
        if results:
            # Save basic metrics
            results_df = pd.DataFrame(results).T
            results_df.to_csv(self.output_dir / "model_evaluation_results.csv")
            logger.info("Saved evaluation results to CSV")
            
            # Save detailed results
            self.evaluator.save_detailed_evaluation_results(detailed_results)
            
            # Create enhanced model comparison visualization
            self.visualizer.plot_enhanced_model_comparison(results)
            self.visualizer.create_performance_summary_report(results, detailed_results)
            self.evaluator.create_comprehensive_evaluation_report(results, detailed_results)
            logger.info("Created enhanced model comparison visualizations")
        
        return results
    
    def print_training_summary(self, models: Dict[str, Any], evaluation_results: Dict[str, Dict[str, float]]) -> None:
        """
        Print a comprehensive training and evaluation summary.
        
        Args:
            models: Dictionary of trained models
            evaluation_results: Evaluation results dictionary
        """
        logger.info("=" * 50)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 50)
        for model_type, model in models.items():
            if model is not None:
                logger.info(f"✓ {model_type.capitalize()} model trained successfully")
            else:
                logger.warning(f"✗ {model_type.capitalize()} model training failed")
        
        # Print evaluation summary
        if evaluation_results:
            self.evaluator.print_evaluation_summary(evaluation_results) 