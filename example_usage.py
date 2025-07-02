#!/usr/bin/env python3
"""
Example Usage Script for Deepfake Detection

This script demonstrates how to use the deepfake detection system
for training models and running inference on various media types.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append('src')

from inference.predict_deepfake import EnhancedDeepfakePredictor
import numpy as np
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_media():
    """Create sample media files for testing."""
    logger.info("📁 Creating sample media files for testing...")
    
    # Create test directory
    test_dir = Path("test_media")
    test_dir.mkdir(exist_ok=True)
    
    # Create a sample image (random noise for demo)
    sample_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    cv2.imwrite(str(test_dir / "sample_image.jpg"), sample_image)
    
    logger.info(f"✅ Sample media created in {test_dir}")
    return test_dir

def demo_model_creation():
    """Demo: Create and save models without training."""
    logger.info("🚀 Demo: Creating sample models...")
    
    try:
        # Run the demo mode of the training script
        os.system("python train_and_save_models.py --demo")
        logger.info("✅ Demo models created successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Error creating demo models: {e}")
        return False

def demo_inference():
    """Demo: Run inference on sample files."""
    logger.info("🔍 Demo: Running inference on sample files...")
    
    # Check if we have any models
    models_dir = Path("models")
    if not models_dir.exists():
        logger.warning("No models directory found. Creating demo models first...")
        if not demo_model_creation():
            return False
    
    # Create sample media
    test_dir = create_sample_media()
    
    # Initialize predictor
    try:
        predictor = EnhancedDeepfakePredictor("models")
        
        # Load available models
        loaded_models = predictor.load_models()
        if not any(loaded_models.values()):
            logger.error("❌ No models could be loaded!")
            return False
        
        logger.info(f"✅ Loaded models: {[k for k, v in loaded_models.items() if v]}")
        
        # Test inference on sample image
        sample_image = test_dir / "sample_image.jpg"
        if sample_image.exists():
            logger.info(f"🖼️  Testing inference on {sample_image}")
            try:
                result = predictor.predict_auto(str(sample_image))
                
                print("\n" + "="*50)
                print("📋 SAMPLE INFERENCE RESULT")
                print("="*50)
                print(f"File: {result['file_path']}")
                print(f"Type: {result['file_type']}")
                print(f"Prediction: {result['label']}")
                print(f"Confidence: {result['confidence']:.1%}")
                print(f"Score: {result['prediction_score']:.4f}")
                print("="*50)
                
                logger.info("✅ Inference test completed successfully!")
                return True
                
            except Exception as e:
                logger.error(f"❌ Error during inference: {e}")
                return False
        else:
            logger.error("❌ Sample image not found!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error initializing predictor: {e}")
        return False

def demo_list_models():
    """Demo: List available models."""
    logger.info("📋 Demo: Listing available models...")
    
    try:
        predictor = EnhancedDeepfakePredictor("models")
        models = predictor.list_available_models()
        
        if not models:
            logger.warning("No models found. Creating demo models first...")
            demo_model_creation()
            models = predictor.list_available_models()
        
        print("\n" + "="*60)
        print("📋 AVAILABLE MODELS")
        print("="*60)
        
        for model_type, metadata in models.items():
            print(f"\n🤖 {model_type.upper()} Model:")
            print(f"   📅 Created: {metadata.get('timestamp', 'N/A')}")
            print(f"   📊 Accuracy: {metadata.get('evaluation_results', {}).get('accuracy', 'N/A')}")
            print(f"   📁 Path: {metadata.get('model_path', 'N/A')}")
            print(f"   🔧 TensorFlow: {metadata.get('tensorflow_version', 'N/A')}")
            if metadata.get('total_params'):
                print(f"   🎛️  Parameters: {metadata['total_params']:,}")
        
        print("="*60)
        logger.info("✅ Model listing completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error listing models: {e}")
        return False

def show_usage_examples():
    """Show usage examples."""
    print("\n" + "="*80)
    print("🚀 DEEPFAKE DETECTION SYSTEM - USAGE EXAMPLES")
    print("="*80)
    
    print("\n📚 TRAINING MODELS:")
    print("   # Create demo models (no data needed)")
    print("   python train_and_save_models.py --demo")
    print()
    print("   # Train with your data")
    print("   python train_and_save_models.py --data-dir /path/to/your/data")
    print()
    print("   # Train specific model types")
    print("   python train_and_save_models.py --data-dir /path/to/data --image-only")
    print("   python train_and_save_models.py --data-dir /path/to/data --video-only")
    
    print("\n🔍 RUNNING INFERENCE:")
    print("   # Detect single file (auto-detects type)")
    print("   python src/inference/predict_deepfake.py --input photo.jpg")
    print("   python src/inference/predict_deepfake.py --input video.mp4")
    print("   python src/inference/predict_deepfake.py --input audio.wav")
    print()
    print("   # Batch process directory")
    print("   python src/inference/predict_deepfake.py --input /media/folder --batch")
    print()
    print("   # Custom threshold and output")
    print("   python src/inference/predict_deepfake.py --input photo.jpg --threshold 0.7 --output results.json")
    print()
    print("   # List available models")
    print("   python src/inference/predict_deepfake.py --list-models")
    
    print("\n📋 SUPPORTED FILE TYPES:")
    print("   📸 Images: .jpg, .jpeg, .png, .bmp, .tiff, .webp")
    print("   🎬 Videos: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm") 
    print("   🎵 Audio: .wav, .mp3, .flac, .m4a, .ogg, .aac")
    
    print("\n🗂️  EXPECTED DATA STRUCTURE FOR TRAINING:")
    print("   your_data/")
    print("   ├── real/")
    print("   │   ├── image1.jpg")
    print("   │   ├── video1.mp4")
    print("   │   └── audio1.wav")
    print("   └── fake/")
    print("       ├── image2.jpg")
    print("       ├── video2.mp4")
    print("       └── audio2.wav")
    
    print("="*80)

def main():
    """Main demo function."""
    print("🎯 Welcome to the Deepfake Detection System Demo!")
    
    # Show usage examples first
    show_usage_examples()
    
    print("\n🚀 Running interactive demo...")
    
    # Run demos
    demos = [
        ("📋 List Available Models", demo_list_models),
        ("🔍 Test Inference", demo_inference),
    ]
    
    for demo_name, demo_func in demos:
        print(f"\n{'='*60}")
        print(f"Running: {demo_name}")
        print('='*60)
        
        success = demo_func()
        if success:
            print(f"✅ {demo_name} completed successfully!")
        else:
            print(f"❌ {demo_name} failed!")
    
    print("\n🎉 Demo completed!")
    print("\n💡 Next steps:")
    print("   1. Add your training data to a folder with 'real' and 'fake' subdirectories")
    print("   2. Run: python train_and_save_models.py --data-dir /path/to/your/data")
    print("   3. Test with: python src/inference/predict_deepfake.py --input your_file.jpg")
    print("\n📚 For more information, see the README.md file.")

if __name__ == "__main__":
    main() 