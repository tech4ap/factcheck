"""
Audio Processing Pipeline

This script is dedicated to cleaning, analyzing, and preparing audio datasets for
deepfake detection. It handles audio file validation, spectrogram generation, dataset
splitting, and visualization specific to audio data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import glob
import shutil
from sklearn.model_selection import train_test_split
import logging
import sys
import traceback
from datetime import datetime
import argparse
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    CLEANED_DATA_DIR, RESULTS_DIR, SUPPORTED_EXTENSIONS, DEFAULT_AUDIO_SAMPLE_RATE
)

# --- CONFIG & SETUP ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'audio_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PipelineError(Exception):
    """Base exception class for pipeline errors"""
    pass

class DataValidationError(PipelineError):
    """Raised when data validation fails"""
    pass

def create_directories():
    """Create necessary directories for the audio pipeline"""
    media_type = 'audio'
    for directory in [CLEANED_DATA_DIR, RESULTS_DIR]:
        base_path = Path(directory)
        (base_path / "real" / media_type).mkdir(parents=True, exist_ok=True)
        (base_path / "fake" / media_type).mkdir(parents=True, exist_ok=True)
    logger.info("Created audio directories")

create_directories()

# --- DATA VALIDATION & PROCESSING ---

def is_valid_audio(file_path: str) -> bool:
    """Check if the audio file is valid."""
    try:
        librosa.load(file_path, sr=None)
        return True
    except Exception as e:
        logger.warning(f"Invalid audio file {file_path}: {e}")
        return False

def process_audio(audio_path: str, output_dir: str, target_sr: int = 22050):
    """Create a spectrogram from an audio file and save it as an image."""
    try:
        y, sr = librosa.load(audio_path, sr=target_sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        
        spec_path = Path(output_dir) / (Path(audio_path).stem + '.png')
        plt.savefig(spec_path)
        plt.close()
        return str(spec_path)
    except Exception as e:
        logger.error(f"Error processing audio {audio_path}: {e}")
        return None

def clean_dataset_audio(real_files: list, fake_files: list):
    """Clean the audio dataset by validating and creating spectrograms."""
    cleaned_real = 0
    cleaned_fake = 0
    
    audio_cleaned_dir_real = Path(CLEANED_DATA_DIR) / "real" / "audio"
    audio_cleaned_dir_fake = Path(CLEANED_DATA_DIR) / "fake" / "audio"

    logger.info("Cleaning real audio files...")
    for file in tqdm(real_files, desc="Processing real audio"):
        if is_valid_audio(file):
            if process_audio(file, str(audio_cleaned_dir_real), target_sr=DEFAULT_AUDIO_SAMPLE_RATE):
                cleaned_real += 1
    
    logger.info("Cleaning fake audio files...")
    for file in tqdm(fake_files, desc="Processing fake audio"):
        if is_valid_audio(file):
            if process_audio(file, str(audio_cleaned_dir_fake), target_sr=DEFAULT_AUDIO_SAMPLE_RATE):
                cleaned_fake += 1
    
    logger.info(f"Cleaning complete. Processed {cleaned_real} real and {cleaned_fake} fake audio files.")
    return cleaned_real, cleaned_fake

def create_audio_train_val_test_split(test_size=0.2, val_size=0.25):
    """Split cleaned spectrograms into train, val, and test sets."""
    real_files = glob.glob(os.path.join(CLEANED_DATA_DIR, "real", "audio", "*.png"))
    fake_files = glob.glob(os.path.join(CLEANED_DATA_DIR, "fake", "audio", "*.png"))

    if not real_files and not fake_files:
        logger.warning("No cleaned audio spectrograms found to create splits.")
        return

    df = pd.DataFrame({
        'filepath': real_files + fake_files,
        'label': ['real'] * len(real_files) + ['fake'] * len(fake_files)
    })
    
    train_val_df, test_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, stratify=train_val_df['label'], random_state=42)
    
    logger.info(f"Audio dataset split: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test samples.")
    
    train_df.to_csv(os.path.join(CLEANED_DATA_DIR, "train_audio.csv"), index=False)
    val_df.to_csv(os.path.join(CLEANED_DATA_DIR, "val_audio.csv"), index=False)
    test_df.to_csv(os.path.join(CLEANED_DATA_DIR, "test_audio.csv"), index=False)
    logger.info("Audio split CSVs saved.")

# --- VISUALIZATION ---

def visualize_audio_distribution():
    """Visualize the distribution of real and fake audio spectrograms."""
    real_count = len(glob.glob(os.path.join(CLEANED_DATA_DIR, "real", "audio", "*.png")))
    fake_count = len(glob.glob(os.path.join(CLEANED_DATA_DIR, "fake", "audio", "*.png")))

    if real_count == 0 and fake_count == 0:
        logger.info("No cleaned audio files to visualize.")
        return

    plt.figure(figsize=(8, 6))
    plt.pie([real_count, fake_count], labels=['Real', 'Fake'], autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
    plt.title('Audio Dataset Distribution')
    plt.axis('equal')
    plt.savefig(os.path.join(RESULTS_DIR, 'audio_distribution.png'))
    plt.close()
    logger.info("Audio distribution chart saved.")

# --- MAIN ---

def main():
    """Main function to run the audio processing pipeline."""
    parser = argparse.ArgumentParser(description='Audio Deepfake Detection Pipeline')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing the dataset (with real/ and fake/ subfolders)')
    parser.add_argument('--clean', action='store_true', help='Clean and preprocess the dataset')
    parser.add_argument('--split', action='store_true', help='Split the dataset into train/val/test sets')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    args = parser.parse_args()

    try:
        real_dir = Path(args.data_dir) / 'real'
        fake_dir = Path(args.data_dir) / 'fake'

        if not real_dir.exists() or not fake_dir.exists():
            raise DataValidationError(f"Source data directory must contain 'real' and 'fake' subfolders. Path checked: {args.data_dir}")

        audio_ext = SUPPORTED_EXTENSIONS['audio']
        real_files = [f for ext in audio_ext for f in glob.glob(str(real_dir / f"**/*{ext}"), recursive=True)]
        fake_files = [f for ext in audio_ext for f in glob.glob(str(fake_dir / f"**/*{ext}"), recursive=True)]
        
        logger.info(f"Found {len(real_files)} real audio files and {len(fake_files)} fake audio files.")

        if args.clean:
            clean_dataset_audio(real_files, fake_files)
        
        if args.split:
            create_audio_train_val_test_split()
            
        if args.visualize:
            visualize_audio_distribution()

        logger.info("Audio pipeline completed successfully.")

    except PipelineError as e:
        logger.error(f"Pipeline error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 