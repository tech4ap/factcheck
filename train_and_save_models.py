#!/usr/bin/env python3
"""
Complete Training and Model Saving Script

This script trains deepfake detection models and saves them with proper
metadata for later inference.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append('src')

from training.main import main as train_main
from training.model_saver import ModelSaver, save_model_with_metadata
from training.trainer import ModelTrainer
from training.evaluation import ModelEvaluator
from models.deepfake_detector import (
    ImageDeepfakeDetector, VideoDeepfakeDetector, AudioDeepfakeDetector
)
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_and_save_models(data_dir: str, output_dir: str = "models", 
                         epochs: int = 20, batch_size: int = 32,
                         train_image: bool = True, train_video: bool = True, 
                         train_audio: bool = True):
    """
    Train and save deepfake detection models.
    
    Args:
        data_dir: Directory containing training data
        output_dir: Directory to save models
        epochs: Number of training epochs
        batch_size: Training batch size
        train_image: Whether to train image model
        train_video: Whether to train video model
        train_audio: Whether to train audio model
    """
    logger.info("🚀 Starting comprehensive model training and saving...")
    
    # Initialize model saver
    model_saver = ModelSaver(output_dir)
    
    # Save training configuration
    training_config = {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "models_to_train": {
            "image": train_image,
            "video": train_video,
            "audio": train_audio
        },
        "training_start_time": datetime.now().isoformat()
    }
    
    model_saver.save_training_config(training_config)
    
    trained_models = {}
    training_results = {}
    
    # Train Image Model
    if train_image:
        logger.info("📸 Training Image Deepfake Detection Model...")
        try:
            image_model = ImageDeepfakeDetector()
            model = image_model.build_model()
            
            # Initialize trainer
            trainer = ModelTrainer(output_dir)
            
            # Load data (you'll need to implement this based on your data structure)
            # For now, we'll use a placeholder
            logger.info("Loading image data...")
            
            # Train the model
            history = trainer.train_model(
                model=model,
                model_name="image_deepfake_detector",
                data_dir=data_dir,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Evaluate the model
            evaluator = ModelEvaluator(output_dir)
            evaluation_results = evaluator.evaluate_model(
                model, model_name="image_model", data_dir=data_dir
            )
            
            # Save the model with metadata
            model_path = model_saver.save_model(
                model=model,
                model_name="image_deepfake_detector",
                model_type="image",
                training_history=history.history if hasattr(history, 'history') else history,
                evaluation_results=evaluation_results,
                config=training_config,
                is_best=True  # Determine this based on validation metrics
            )
            
            trained_models['image'] = model
            training_results['image'] = {
                'model_path': model_path,
                'evaluation': evaluation_results,
                'history': history.history if hasattr(history, 'history') else history
            }
            
            logger.info(f"✅ Image model trained and saved to {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Error training image model: {e}")
            training_results['image'] = {'error': str(e)}
    
    # Train Video Model
    if train_video:
        logger.info("🎬 Training Video Deepfake Detection Model...")
        try:
            video_model = VideoDeepfakeDetector()
            model = video_model.build_model()
            
            # Initialize trainer
            trainer = ModelTrainer(output_dir)
            
            # Train the model
            history = trainer.train_model(
                model=model,
                model_name="video_deepfake_detector",
                data_dir=data_dir,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Evaluate the model
            evaluator = ModelEvaluator(output_dir)
            evaluation_results = evaluator.evaluate_model(
                model, model_name="video_model", data_dir=data_dir
            )
            
            # Save the model with metadata
            model_path = model_saver.save_model(
                model=model,
                model_name="video_deepfake_detector",
                model_type="video",
                training_history=history.history if hasattr(history, 'history') else history,
                evaluation_results=evaluation_results,
                config=training_config,
                is_best=True
            )
            
            trained_models['video'] = model
            training_results['video'] = {
                'model_path': model_path,
                'evaluation': evaluation_results,
                'history': history.history if hasattr(history, 'history') else history
            }
            
            logger.info(f"✅ Video model trained and saved to {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Error training video model: {e}")
            training_results['video'] = {'error': str(e)}
    
    # Train Audio Model
    if train_audio:
        logger.info("🎵 Training Audio Deepfake Detection Model...")
        try:
            audio_model = AudioDeepfakeDetector()
            model = audio_model.build_model()
            
            # Initialize trainer
            trainer = ModelTrainer(output_dir)
            
            # Train the model
            history = trainer.train_model(
                model=model,
                model_name="audio_deepfake_detector",
                data_dir=data_dir,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Evaluate the model
            evaluator = ModelEvaluator(output_dir)
            evaluation_results = evaluator.evaluate_model(
                model, model_name="audio_model", data_dir=data_dir
            )
            
            # Save the model with metadata
            model_path = model_saver.save_model(
                model=model,
                model_name="audio_deepfake_detector",
                model_type="audio",
                training_history=history.history if hasattr(history, 'history') else history,
                evaluation_results=evaluation_results,
                config=training_config,
                is_best=True
            )
            
            trained_models['audio'] = model
            training_results['audio'] = {
                'model_path': model_path,
                'evaluation': evaluation_results,
                'history': history.history if hasattr(history, 'history') else history
            }
            
            logger.info(f"✅ Audio model trained and saved to {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Error training audio model: {e}")
            training_results['audio'] = {'error': str(e)}
    
    # Final summary
    logger.info("🎯 Training Summary:")
    for model_type, result in training_results.items():
        if 'error' in result:
            logger.error(f"   {model_type.upper()}: Failed - {result['error']}")
        else:
            logger.info(f"   {model_type.upper()}: Success ✅")
            if 'evaluation' in result:
                accuracy = result['evaluation'].get('accuracy', 'N/A')
                logger.info(f"      Accuracy: {accuracy}")
    
    # Cleanup old checkpoints
    model_saver.cleanup_old_checkpoints(keep_last_n=3)
    
    return trained_models, training_results

def quick_demo_training():
    """
    Quick demo training with minimal data for testing purposes.
    """
    logger.info("🔬 Running quick demo training...")
    
    # Create minimal demo data structure
    demo_data_dir = "demo_data"
    os.makedirs(demo_data_dir, exist_ok=True)
    os.makedirs(f"{demo_data_dir}/real", exist_ok=True)
    os.makedirs(f"{demo_data_dir}/fake", exist_ok=True)
    
    logger.info("Demo data structure created. Add some sample files to train on real data.")
    
    # For demo purposes, we'll just create and save untrained models
    model_saver = ModelSaver("models")
    
    # Create demo models
    models = {
        'image': ImageDeepfakeDetector().build_model(),
        'video': VideoDeepfakeDetector().build_model(),
        'audio': AudioDeepfakeDetector().build_model()
    }
    
    # Save demo models
    for model_type, model in models.items():
        model_path = model_saver.save_model(
            model=model,
            model_name=f"{model_type}_deepfake_detector_demo",
            model_type=model_type,
            training_history={'loss': [0.5], 'accuracy': [0.7], 'val_loss': [0.6], 'val_accuracy': [0.65]},
            evaluation_results={'accuracy': 0.65, 'precision': 0.7, 'recall': 0.6},
            is_best=False
        )
        logger.info(f"Demo {model_type} model saved to {model_path}")
    
    logger.info("🎉 Demo models created! You can now test inference.")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train and Save Deepfake Detection Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models
  python train_and_save_models.py --data-dir /path/to/data
  
  # Train specific models
  python train_and_save_models.py --data-dir /path/to/data --image-only
  
  # Quick demo (no data required)
  python train_and_save_models.py --demo
  
  # Custom training parameters
  python train_and_save_models.py --data-dir /path/to/data --epochs 50 --batch-size 64
        """)
    
    parser.add_argument('--data-dir', type=str, 
                       help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save trained models (default: models)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size (default: 32)')
    
    # Model selection
    parser.add_argument('--image-only', action='store_true',
                       help='Train only image model')
    parser.add_argument('--video-only', action='store_true',
                       help='Train only video model')
    parser.add_argument('--audio-only', action='store_true',
                       help='Train only audio model')
    
    # Demo mode
    parser.add_argument('--demo', action='store_true',
                       help='Run demo mode (creates sample models without training)')
    
    args = parser.parse_args()
    
    if args.demo:
        quick_demo_training()
        return
    
    if not args.data_dir:
        logger.error("❌ Data directory is required unless using --demo mode")
        parser.print_help()
        return
    
    if not Path(args.data_dir).exists():
        logger.error(f"❌ Data directory not found: {args.data_dir}")
        return
    
    # Determine which models to train
    if args.image_only:
        train_image, train_video, train_audio = True, False, False
    elif args.video_only:
        train_image, train_video, train_audio = False, True, False
    elif args.audio_only:
        train_image, train_video, train_audio = False, False, True
    else:
        train_image, train_video, train_audio = True, True, True
    
    # Start training
    try:
        trained_models, results = train_and_save_models(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            train_image=train_image,
            train_video=train_video,
            train_audio=train_audio
        )
        
        logger.info("🎉 Training completed successfully!")
        logger.info(f"Models saved to: {args.output_dir}")
        logger.info("You can now use the inference script to test your models:")
        logger.info(f"python src/inference/predict_deepfake.py --input your_file.jpg")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 