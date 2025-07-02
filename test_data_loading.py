#!/usr/bin/env python3
"""
Test script to demonstrate data loading from the organized data directory.
This script shows how to use the DataLoader class with your data structure.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from training.data_loader import DataLoader
from config import USER_DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_image_loading():
    """Test loading image data from the organized directory structure."""
    logger.info("=== Testing Image Data Loading ===")
    
    # Initialize data loader with your data directory
    data_loader = DataLoader(USER_DATA_DIR, max_samples=10)  # Limit to 10 samples for testing
    
    # Test loading training data
    logger.info("Loading training images...")
    train_images, train_labels = data_loader.load_image_data_from_directories('train', target_size=(256, 256))
    
    if len(train_images) > 0:
        logger.info(f"Successfully loaded {len(train_images)} training images")
        logger.info(f"Image shape: {train_images.shape}")
        logger.info(f"Labels shape: {train_labels.shape}")
        logger.info(f"Fake samples: {sum(train_labels)}")
        logger.info(f"Real samples: {len(train_labels) - sum(train_labels)}")
    else:
        logger.error("No training images loaded!")
    
    # Test loading validation data
    logger.info("Loading validation images...")
    val_images, val_labels = data_loader.load_image_data_from_directories('validation', target_size=(256, 256))
    
    if len(val_images) > 0:
        logger.info(f"Successfully loaded {len(val_images)} validation images")
        logger.info(f"Fake samples: {sum(val_labels)}")
        logger.info(f"Real samples: {len(val_labels) - sum(val_labels)}")
    else:
        logger.error("No validation images loaded!")

def test_video_loading():
    """Test loading video data from the organized directory structure."""
    logger.info("\n=== Testing Video Data Loading ===")
    
    # Initialize data loader with your data directory
    data_loader = DataLoader(USER_DATA_DIR, max_samples=5)  # Limit to 5 videos for testing
    
    # Test loading training data
    logger.info("Loading training videos...")
    train_videos, train_labels = data_loader.load_video_data_from_directories(
        'train', 
        frames_per_video=10, 
        target_size=(256, 256)
    )
    
    if len(train_videos) > 0:
        logger.info(f"Successfully loaded {len(train_videos)} training videos")
        logger.info(f"Video shape: {train_videos.shape}")
        logger.info(f"Labels shape: {train_labels.shape}")
        logger.info(f"Fake videos: {sum(train_labels)}")
        logger.info(f"Real videos: {len(train_labels) - sum(train_labels)}")
    else:
        logger.error("No training videos loaded!")

def test_audio_loading():
    """Test loading audio data from the organized directory structure."""
    logger.info("\n=== Testing Audio Data Loading ===")
    
    # Initialize data loader with your data directory
    data_loader = DataLoader(USER_DATA_DIR, max_samples=10)  # Limit to 10 samples for testing
    
    # Test loading training data with spectrograms
    logger.info("Loading training audio (spectrograms)...")
    train_audio, train_labels = data_loader.load_audio_data_from_directories(
        'train', 
        target_size=(128, 128),
        feature_type='spectrogram',
        sample_rate=22050,
        duration=3.0
    )
    
    if len(train_audio) > 0:
        logger.info(f"Successfully loaded {len(train_audio)} training audio files")
        logger.info(f"Audio shape: {train_audio.shape}")
        logger.info(f"Labels shape: {train_labels.shape}")
        logger.info(f"Fake audio: {sum(train_labels)}")
        logger.info(f"Real audio: {len(train_labels) - sum(train_labels)}")
    else:
        logger.error("No training audio loaded!")
    
    # Test loading with MFCC features
    logger.info("Loading training audio (MFCC)...")
    train_audio_mfcc, train_labels_mfcc = data_loader.load_audio_data_from_directories(
        'train', 
        target_size=(128, 128),
        feature_type='mfcc',
        sample_rate=22050,
        duration=3.0
    )
    
    if len(train_audio_mfcc) > 0:
        logger.info(f"Successfully loaded {len(train_audio_mfcc)} training audio files (MFCC)")
        logger.info(f"MFCC shape: {train_audio_mfcc.shape}")
        logger.info(f"Fake audio: {sum(train_labels_mfcc)}")
        logger.info(f"Real audio: {len(train_labels_mfcc) - sum(train_labels_mfcc)}")
    else:
        logger.error("No training audio loaded (MFCC)!")

def main():
    """Run all data loading tests."""
    logger.info(f"Testing data loading from: {USER_DATA_DIR}")
    
    # Test each data type
    test_image_loading()
    test_video_loading()
    test_audio_loading()
    
    logger.info("\n=== Data Loading Tests Complete ===")

if __name__ == "__main__":
    main() 