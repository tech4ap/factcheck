#!/usr/bin/env python3
"""
Example script demonstrating audio processing functionality.
This shows how to use the DataLoader with audio-to-spectrogram conversion.
"""

import sys
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from training.data_loader import DataLoader
from config import USER_DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demonstrate_audio_processing():
    """Demonstrate audio processing with different feature types."""
    
    logger.info("=== Audio Processing Demonstration ===")
    
    # Initialize data loader
    data_loader = DataLoader(USER_DATA_DIR, max_samples=4)  # Small sample for demonstration
    
    # 1. Load audio as spectrograms
    logger.info("1. Loading audio as spectrograms...")
    spectrograms, spec_labels = data_loader.load_audio_data_from_directories(
        'train',
        target_size=(128, 128),
        feature_type='spectrogram',
        sample_rate=22050,
        duration=3.0
    )
    
    logger.info(f"Spectrograms shape: {spectrograms.shape}")
    logger.info(f"Labels: {spec_labels}")
    
    # 2. Load audio as MFCC features
    logger.info("2. Loading audio as MFCC features...")
    mfcc_features, mfcc_labels = data_loader.load_audio_data_from_directories(
        'train',
        target_size=(128, 128),
        feature_type='mfcc',
        sample_rate=22050,
        duration=3.0
    )
    
    logger.info(f"MFCC shape: {mfcc_features.shape}")
    logger.info(f"Labels: {mfcc_labels}")
    
    # 3. Visualize the features
    logger.info("3. Creating visualizations...")
    create_audio_visualizations(spectrograms, mfcc_features, spec_labels)
    
    return spectrograms, mfcc_features, spec_labels

def create_audio_visualizations(spectrograms, mfcc_features, labels):
    """Create visualizations of audio features."""
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Audio Feature Visualizations', fontsize=16)
    
    # Plot spectrograms
    for i in range(4):
        if i < len(spectrograms):
            # Remove channel dimension for plotting
            spec = spectrograms[i].squeeze()
            axes[0, i].imshow(spec, cmap='viridis', aspect='auto')
            axes[0, i].set_title(f'Spectrogram ({"Fake" if labels[i] else "Real"})')
            axes[0, i].axis('off')
    
    # Plot MFCC features
    for i in range(4):
        if i < len(mfcc_features):
            # Remove channel dimension for plotting
            mfcc = mfcc_features[i].squeeze()
            axes[1, i].imshow(mfcc, cmap='viridis', aspect='auto')
            axes[1, i].set_title(f'MFCC ({"Fake" if labels[i] else "Real"})')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = 'audio_features_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Visualization saved to: {output_path}")
    
    # Show the plot
    plt.show()

def demonstrate_training_usage():
    """Show how to use audio data for training."""
    
    logger.info("\n=== Training Usage Example ===")
    
    # Initialize data loader for training
    data_loader = DataLoader(USER_DATA_DIR, max_samples=100)  # Larger sample for training
    
    # Load training data
    logger.info("Loading training data...")
    train_spectrograms, train_labels = data_loader.load_audio_data_from_directories(
        'train',
        target_size=(128, 128),
        feature_type='spectrogram'
    )
    
    # Load validation data
    logger.info("Loading validation data...")
    val_spectrograms, val_labels = data_loader.load_audio_data_from_directories(
        'validation',
        target_size=(128, 128),
        feature_type='spectrogram'
    )
    
    logger.info(f"Training data: {train_spectrograms.shape}")
    logger.info(f"Training labels: {train_labels.shape}")
    logger.info(f"Validation data: {val_spectrograms.shape}")
    logger.info(f"Validation labels: {val_labels.shape}")
    
    # Example of how you might use this in a training loop
    logger.info("\nExample training usage:")
    logger.info("1. Data is ready for CNN training (spectrograms as images)")
    logger.info("2. Labels are binary (0=real, 1=fake)")
    logger.info("3. Data is normalized to [0, 1] range")
    logger.info("4. Shape is (batch_size, height, width, channels)")
    
    return train_spectrograms, train_labels, val_spectrograms, val_labels

def main():
    """Run the audio processing demonstration."""
    
    logger.info("Starting audio processing demonstration...")
    
    # Demonstrate audio processing
    spectrograms, mfcc_features, labels = demonstrate_audio_processing()
    
    # Show training usage
    train_data, train_labels, val_data, val_labels = demonstrate_training_usage()
    
    logger.info("\n=== Audio Processing Demo Complete ===")
    logger.info("Key features implemented:")
    logger.info("✓ Audio file loading (.wav, .mp3, .flac)")
    logger.info("✓ Spectrogram generation (mel spectrograms)")
    logger.info("✓ MFCC feature extraction")
    logger.info("✓ Audio normalization and resizing")
    logger.info("✓ Memory-efficient batch processing")
    logger.info("✓ Support for different sample rates and durations")

if __name__ == "__main__":
    main() 