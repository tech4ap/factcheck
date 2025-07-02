"""
Video Processing Pipeline

This script is dedicated to cleaning, analyzing, and preparing video datasets for
deepfake detection. It handles video file validation, frame extraction, dataset
splitting, and visualization specific to video data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
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
    CLEANED_DATA_DIR, RESULTS_DIR, SUPPORTED_EXTENSIONS, DEFAULT_VIDEO_FPS
)

# --- CONFIG & SETUP ---

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'video_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
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

class ProcessingError(PipelineError):
    """Raised when processing fails"""
    pass

def create_directories():
    """Create necessary directories for the video pipeline"""
    media_type = 'videos'
    for directory in [CLEANED_DATA_DIR, RESULTS_DIR]:
        base_path = Path(directory)
        (base_path / "real" / media_type).mkdir(parents=True, exist_ok=True)
        (base_path / "fake" / media_type).mkdir(parents=True, exist_ok=True)
    logger.info("Created video directories")

create_directories()

# --- DATA VALIDATION & PROCESSING ---

def is_valid_video(file_path: str) -> bool:
    """Check if the video file is valid and can be opened."""
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            logger.warning(f"Invalid video file (cannot be opened): {file_path}")
            return False
        cap.release()
        return True
    except Exception as e:
        logger.error(f"Error validating video file {file_path}: {e}")
        return False

def process_video(video_path: str, output_dir: str, target_size: tuple = (256, 256), fps: int = 1):
    """Extract frames from a video and save them as images."""
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % int(cap.get(cv2.CAP_PROP_FPS) / fps) == 0:
                frame = cv2.resize(frame, target_size)
                frame_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        return saved_count
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}")
        return 0

def clean_dataset_videos(real_files: list, fake_files: list):
    """Clean the video dataset by validating and extracting frames."""
    cleaned_real = 0
    cleaned_fake = 0
    
    video_cleaned_dir_real = Path(CLEANED_DATA_DIR) / "real" / "videos"
    video_cleaned_dir_fake = Path(CLEANED_DATA_DIR) / "fake" / "videos"

    logger.info("Cleaning real videos...")
    for file in tqdm(real_files, desc="Processing real videos"):
        if is_valid_video(file):
            try:
                output_dir = video_cleaned_dir_real / os.path.splitext(os.path.basename(file))[0]
                output_dir.mkdir(exist_ok=True)
                frames_saved = process_video(file, str(output_dir), fps=DEFAULT_VIDEO_FPS)
                if frames_saved > 0:
                    cleaned_real += 1
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
    
    logger.info("Cleaning fake videos...")
    for file in tqdm(fake_files, desc="Processing fake videos"):
        if is_valid_video(file):
            try:
                output_dir = video_cleaned_dir_fake / os.path.splitext(os.path.basename(file))[0]
                output_dir.mkdir(exist_ok=True)
                frames_saved = process_video(file, str(output_dir), fps=DEFAULT_VIDEO_FPS)
                if frames_saved > 0:
                    cleaned_fake += 1
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
    
    logger.info(f"Cleaning complete. Processed {cleaned_real} real and {cleaned_fake} fake videos.")
    return cleaned_real, cleaned_fake

def create_video_train_val_test_split(test_size=0.2, val_size=0.25):
    """Split cleaned videos into train, val, and test sets and save as CSVs."""
    real_video_dirs = glob.glob(os.path.join(CLEANED_DATA_DIR, "real", "videos", "*"))
    fake_video_dirs = glob.glob(os.path.join(CLEANED_DATA_DIR, "fake", "videos", "*"))
    
    real_videos = [d for d in real_video_dirs if os.path.isdir(d)]
    fake_videos = [d for d in fake_video_dirs if os.path.isdir(d)]

    if not real_videos and not fake_videos:
        logger.warning("No cleaned videos found to create splits.")
        return

    df = pd.DataFrame({
        'video_dir': [os.path.abspath(d) for d in real_videos] + [os.path.abspath(d) for d in fake_videos],
        'label': ['real'] * len(real_videos) + ['fake'] * len(fake_videos)
    })
    
    train_val_df, test_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, stratify=train_val_df['label'], random_state=42)
    
    logger.info(f"Video dataset split: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test samples.")
    
    train_df.to_csv(os.path.join(CLEANED_DATA_DIR, "train_videos.csv"), index=False)
    val_df.to_csv(os.path.join(CLEANED_DATA_DIR, "val_videos.csv"), index=False)
    test_df.to_csv(os.path.join(CLEANED_DATA_DIR, "test_videos.csv"), index=False)
    logger.info("Video split CSVs saved.")

# --- VISUALIZATION ---

def visualize_video_distribution():
    """Visualize the distribution of real and fake videos."""
    real_count = len(glob.glob(os.path.join(CLEANED_DATA_DIR, "real", "videos", "*")))
    fake_count = len(glob.glob(os.path.join(CLEANED_DATA_DIR, "fake", "videos", "*")))

    if real_count == 0 and fake_count == 0:
        logger.info("No cleaned videos to visualize.")
        return

    plt.figure(figsize=(8, 6))
    plt.pie([real_count, fake_count], labels=['Real', 'Fake'], autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
    plt.title('Video Dataset Distribution')
    plt.axis('equal')
    plt.savefig(os.path.join(RESULTS_DIR, 'video_distribution.png'))
    plt.close()
    logger.info("Video distribution chart saved.")

# --- MAIN ---

def main():
    """Main function to run the video processing pipeline."""
    parser = argparse.ArgumentParser(description='Video Deepfake Detection Pipeline')
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

        video_ext = SUPPORTED_EXTENSIONS['videos']
        real_files = [f for ext in video_ext for f in glob.glob(str(real_dir / f"**/*{ext}"), recursive=True)]
        fake_files = [f for ext in video_ext for f in glob.glob(str(fake_dir / f"**/*{ext}"), recursive=True)]
        
        logger.info(f"Found {len(real_files)} real videos and {len(fake_files)} fake videos.")

        if args.clean:
            clean_dataset_videos(real_files, fake_files)
        
        if args.split:
            create_video_train_val_test_split()
            
        if args.visualize:
            visualize_video_distribution()
            # Add more visualizations if needed, e.g., sample frames

        logger.info("Video pipeline completed successfully.")

    except PipelineError as e:
        logger.error(f"Pipeline error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 