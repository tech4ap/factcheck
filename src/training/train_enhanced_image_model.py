#!/usr/bin/env python3
"""
Enhanced Image Model Training Script

This script trains the enhanced deepfake detection image model with advanced techniques
including focal loss, attention mechanisms, multi-scale feature extraction, and 
advanced data augmentation to achieve higher accuracy and AUC-ROC.

Usage:
    python train_enhanced_image_model.py --data-dir data --epochs 100
    python train_enhanced_image_model.py --data-dir data --base-model efficientnet_b2 --use-multiscale
    python train_enhanced_image_model.py --data-dir data --demo --epochs 50
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.train_deepfake_detector import ModelTrainer

# Configure logging
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
    """Main training function with comprehensive argument parsing."""
    parser = argparse.ArgumentParser(
        description='Train Enhanced Deepfake Detection Image Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic enhanced training
  python train_enhanced_image_model.py --data-dir cleaned_data
  
  # Training with EfficientNet-B2 and multi-scale features
  python train_enhanced_image_model.py --data-dir cleaned_data --base-model efficientnet_b2 --use-multiscale
  
  # Quick demo training
  python train_enhanced_image_model.py --data-dir demo_data --demo --epochs 20
  
  # Advanced training with all techniques
  python train_enhanced_image_model.py --data-dir cleaned_data --epochs 150 \\
    --base-model efficientnet_b2 --use-multiscale --use-attention \\
    --use-mixup --use-cutmix --batch-size 16
        """
    )
    
    # Required arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing training data (with train/validation/test splits)')
    
    # Model architecture options
    parser.add_argument('--base-model', type=str, 
                       choices=['efficientnet', 'efficientnet_b2', 'resnet', 'densenet'],
                       default='efficientnet',
                       help='Base model architecture (default: efficientnet)')
    parser.add_argument('--use-multiscale', action='store_true',
                       help='Use multi-scale feature extraction with multiple base models')
    parser.add_argument('--use-attention', action='store_true', default=True,
                       help='Use self-attention mechanisms (default: True)')
    parser.add_argument('--no-attention', dest='use_attention', action='store_false',
                       help='Disable attention mechanisms')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Initial learning rate (default: 1e-4)')
    parser.add_argument('--no-fine-tune', action='store_true',
                       help='Skip fine-tuning phase')
    
    # Data augmentation options
    parser.add_argument('--use-advanced-augmentation', action='store_true', default=True,
                       help='Use advanced data augmentation (default: True)')
    parser.add_argument('--basic-augmentation', dest='use_advanced_augmentation', action='store_false',
                       help='Use basic data augmentation instead of advanced')
    parser.add_argument('--use-mixup', action='store_true', default=True,
                       help='Use MixUp augmentation (default: True)')
    parser.add_argument('--no-mixup', dest='use_mixup', action='store_false',
                       help='Disable MixUp augmentation')
    parser.add_argument('--use-cutmix', action='store_true', default=True,
                       help='Use CutMix augmentation (default: True)')
    parser.add_argument('--no-cutmix', dest='use_cutmix', action='store_false',
                       help='Disable CutMix augmentation')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save trained models (default: models)')
    parser.add_argument('--demo', action='store_true',
                       help='Run in demo mode with limited samples for testing')
    parser.add_argument('--max-samples', type=int,
                       help='Maximum samples to use (for testing purposes)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not Path(args.data_dir).exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        return 1
    
    # Set demo parameters
    if args.demo:
        if args.max_samples is None:
            args.max_samples = 1000
        if args.epochs > 50:
            args.epochs = 20
        logger.info("Demo mode enabled: limited samples and epochs")
    
    # Log training configuration
    logger.info("Enhanced Image Model Training Configuration:")
    logger.info(f"  Data directory: {args.data_dir}")
    logger.info(f"  Base model: {args.base_model}")
    logger.info(f"  Multi-scale features: {args.use_multiscale}")
    logger.info(f"  Attention mechanisms: {args.use_attention}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Fine-tuning: {not args.no_fine_tune}")
    logger.info(f"  Advanced augmentation: {args.use_advanced_augmentation}")
    logger.info(f"  MixUp augmentation: {args.use_mixup}")
    logger.info(f"  CutMix augmentation: {args.use_cutmix}")
    logger.info(f"  Output directory: {args.output_dir}")
    
    try:
        # Initialize trainer
        logger.info("Initializing enhanced model trainer...")
        trainer = ModelTrainer(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            max_samples=args.max_samples
        )
        
        # Train enhanced image model
        logger.info("Starting enhanced image model training...")
        model = trainer.train_enhanced_image_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            fine_tune=not args.no_fine_tune,
            learning_rate=args.learning_rate,
            use_advanced_augmentation=args.use_advanced_augmentation,
            use_mixup=args.use_mixup,
            use_cutmix=args.use_cutmix,
            base_model=args.base_model,
            use_multiscale=args.use_multiscale,
            use_attention=args.use_attention
        )
        
        if model is None:
            logger.error("Training failed!")
            return 1
        
        # Log success
        logger.info("Enhanced image model training completed successfully!")
        logger.info(f"Model saved to: {args.output_dir}")
        
        # Print model summary
        summary = model.get_model_summary()
        logger.info("Final model summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 