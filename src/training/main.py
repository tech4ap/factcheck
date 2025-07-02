"""
Main Training Script for Deepfake Detection

This script provides a clean interface for training deepfake detection models
using the modular components. It handles command-line arguments and orchestrates
the training and evaluation process.
"""

import argparse
import logging
from pathlib import Path
from typing import List

from .trainer import ModelTrainer

# Configure logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def validate_arguments(args: argparse.Namespace) -> bool:
    """
    Validate command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        bool: True if arguments are valid, False otherwise
    """
    if not Path(args.data_dir).exists():
        logger.error(f"Data directory does not exist: {args.data_dir}")
        return False
    
    if args.epochs <= 0:
        logger.error("Number of epochs must be positive")
        return False
    
    if args.batch_size <= 0:
        logger.error("Batch size must be positive")
        return False
    
    if args.learning_rate <= 0:
        logger.error("Learning rate must be positive")
        return False
    
    if args.frames_per_video <= 0:
        logger.error("Frames per video must be positive")
        return False
    
    return True


def train_models(args: argparse.Namespace) -> None:
    """
    Train models based on specified media types.
    
    Args:
        args: Parsed command-line arguments
    """
    # Create trainer
    trainer = ModelTrainer(args.data_dir, args.output_dir, args.max_samples)
    
    # Train models based on specified media types
    models = {}
    
    if 'image' in args.media_types:
        logger.info("=" * 50)
        logger.info("TRAINING IMAGE MODEL")
        logger.info("=" * 50)
        models['image'] = trainer.train_image_model(
            epochs=args.epochs, 
            batch_size=args.batch_size,
            fine_tune=not args.no_fine_tune,
            learning_rate=args.learning_rate
        )
    
    if 'video' in args.media_types:
        logger.info("=" * 50)
        logger.info("TRAINING VIDEO MODEL")
        logger.info("=" * 50)
        models['video'] = trainer.train_video_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            frames_per_video=args.frames_per_video,
            learning_rate=args.learning_rate
        )
    
    if 'audio' in args.media_types:
        logger.info("=" * 50)
        logger.info("TRAINING AUDIO MODEL")
        logger.info("=" * 50)
        models['audio'] = trainer.train_audio_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    
    # Evaluate models
    logger.info("=" * 50)
    logger.info("EVALUATING MODELS")
    logger.info("=" * 50)
    evaluation_results = trainer.evaluate_models(
        image_model=models.get('image'),
        video_model=models.get('video'),
        audio_model=models.get('audio'),
        frames_per_video=args.frames_per_video
    )
    
    # Print summary
    trainer.print_training_summary(models, evaluation_results)


def main():
    """
    Main function with command-line argument parsing.
    
    This function orchestrates the training process for all media types
    based on user specifications.
    """
    parser = argparse.ArgumentParser(
        description='Train Deepfake Detection Models with GPU Acceleration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models with default settings
  python -m src.training.main --data-dir src/cleaned_data
  
  # Train only video model with custom parameters
  python -m src.training.main --data-dir src/cleaned_data --media-types video --epochs 20 --batch-size 8
  
  # Test run with limited samples
  python -m src.training.main --data-dir src/cleaned_data --max-samples 100 --epochs 2
        """
    )
    
    # Required arguments
    parser.add_argument('--data-dir', type=str, required=True, 
                       help='Directory containing processed data (CSV files and media folders)')
    
    # Optional arguments
    parser.add_argument('--output-dir', type=str, default='models', 
                       help='Output directory for trained models (default: models)')
    parser.add_argument('--media-types', nargs='+', 
                       choices=['image', 'video', 'audio'], 
                       default=['image', 'video', 'audio'], 
                       help='Media types to train (default: all)')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Training batch size (default: 32)')
    parser.add_argument('--no-fine-tune', action='store_true', 
                       help='Skip fine-tuning for image model')
    parser.add_argument('--max-samples', type=int, 
                       help='Maximum samples to use (useful for testing)')
    parser.add_argument('--learning-rate', type=float, default=1e-4, 
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--frames-per-video', type=int, default=10, 
                       help='Number of frames per video sequence (default: 10)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not validate_arguments(args):
        return
    
    # Train models
    train_models(args)


if __name__ == "__main__":
    main() 