#!/usr/bin/env python3
"""
Enhanced Deepfake Detection Training Script

This script implements comprehensive improvements for better model performance:
- Enhanced model architectures with residual connections and attention
- Advanced training strategies with better hyperparameters
- Improved data augmentation and regularization
- Better learning rate scheduling and callbacks
- Comprehensive evaluation and monitoring
"""

import sys
import logging
import argparse
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from training.train_deepfake_detector import ModelTrainer, ModelVisualizer
from config import USER_DATA_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Run enhanced training for all model types."""
    
    parser = argparse.ArgumentParser(description='Enhanced Deepfake Detection Training')
    parser.add_argument('--data-dir', type=str, default=USER_DATA_DIR,
                       help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory for models')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples per class for testing')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--frames-per-video', type=int, default=10,
                       help='Number of frames per video')
    parser.add_argument('--train-images', action='store_true',
                       help='Train image model')
    parser.add_argument('--train-videos', action='store_true',
                       help='Train video model')
    parser.add_argument('--train-audio', action='store_true',
                       help='Train audio model')
    parser.add_argument('--train-all', action='store_true',
                       help='Train all models')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate trained models')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    
    args = parser.parse_args()
    
    logger.info("=== Enhanced Deepfake Detection Training ===")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Max samples: {args.max_samples}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    
    # Initialize trainer
    trainer = ModelTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )
    
    # Initialize visualizer
    visualizer = ModelVisualizer(output_dir=args.output_dir)
    
    trained_models = {}
    
    # Train models based on arguments
    if args.train_all or args.train_images:
        logger.info("\n=== Training Enhanced Image Model ===")
        image_model = trainer.train_image_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        if image_model:
            trained_models['image'] = image_model
            logger.info("✅ Image model training completed successfully")
        else:
            logger.error("❌ Image model training failed")
    
    if args.train_all or args.train_videos:
        logger.info("\n=== Training Enhanced Video Model ===")
        video_model = trainer.train_video_model(
            epochs=args.epochs,
            batch_size=min(args.batch_size // 2, 8),  # Smaller batch size for videos
            frames_per_video=args.frames_per_video,
            learning_rate=args.learning_rate
        )
        if video_model:
            trained_models['video'] = video_model
            logger.info("✅ Video model training completed successfully")
        else:
            logger.error("❌ Video model training failed")
    
    if args.train_all or args.train_audio:
        logger.info("\n=== Training Enhanced Audio Model ===")
        audio_model = trainer.train_audio_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        if audio_model:
            trained_models['audio'] = audio_model
            logger.info("✅ Audio model training completed successfully")
        else:
            logger.error("❌ Audio model training failed")
    
    # Evaluate models
    if args.evaluate and trained_models:
        logger.info("\n=== Evaluating Enhanced Models ===")
        results = trainer.evaluate_models(
            image_model=trained_models.get('image'),
            video_model=trained_models.get('video'),
            audio_model=trained_models.get('audio'),
            frames_per_video=args.frames_per_video
        )
        
        # Print results
        logger.info("\n=== Enhanced Model Performance Summary ===")
        for model_type, metrics in results.items():
            logger.info(f"\n{model_type.upper()} Model:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
    
    # Create visualizations
    if args.visualize:
        logger.info("\n=== Creating Enhanced Visualizations ===")
        try:
            visualizer.create_data_visualizations()
            logger.info("✅ Data visualizations created")
        except Exception as e:
            logger.error(f"❌ Error creating visualizations: {e}")
    
    logger.info("\n=== Enhanced Training Complete ===")
    logger.info(f"Trained models: {list(trained_models.keys())}")
    logger.info(f"Check {args.output_dir} for model files and logs")

if __name__ == "__main__":
    main() 