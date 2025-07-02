"""
Image Processing Pipeline

This script is dedicated to cleaning, analyzing, and preparing image datasets for
deepfake detection. It handles image validation, face extraction, dataset splitting,
and various visualizations specific to image data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
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
    CLEANED_DATA_DIR, RESULTS_DIR, SUPPORTED_EXTENSIONS, DEFAULT_IMAGE_SIZE
)

# --- CONFIG & SETUP ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'image_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
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
    """Create necessary directories for the image pipeline"""
    media_type = 'images'
    for directory in [CLEANED_DATA_DIR, RESULTS_DIR]:
        base_path = Path(directory)
        (base_path / "real" / media_type).mkdir(parents=True, exist_ok=True)
        (base_path / "fake" / media_type).mkdir(parents=True, exist_ok=True)
    logger.info("Created image directories")

create_directories()

# --- DATA VALIDATION & PROCESSING ---

def is_valid_image(file_path: str) -> bool:
    """Check if the image is valid and can be opened."""
    try:
        img = Image.open(file_path)
        img.verify()
        return True
    except Exception as e:
        logger.warning(f"Invalid image file {file_path}: {e}")
        return False

def clean_dataset_images(real_files: list, fake_files: list):
    """Clean the image dataset by validating and standardizing formats."""
    cleaned_real = 0
    cleaned_fake = 0
    
    img_cleaned_dir_real = Path(CLEANED_DATA_DIR) / "real" / "images"
    img_cleaned_dir_fake = Path(CLEANED_DATA_DIR) / "fake" / "images"

    logger.info("Cleaning real images...")
    for file in tqdm(real_files, desc="Processing real images"):
        if is_valid_image(file):
            try:
                img = Image.open(file).convert('RGB').resize(DEFAULT_IMAGE_SIZE)
                new_path = img_cleaned_dir_real / (Path(file).stem + '.jpg')
                img.save(new_path, 'JPEG')
                cleaned_real += 1
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")

    logger.info("Cleaning fake images...")
    for file in tqdm(fake_files, desc="Processing fake images"):
        if is_valid_image(file):
            try:
                img = Image.open(file).convert('RGB').resize(DEFAULT_IMAGE_SIZE)
                new_path = img_cleaned_dir_fake / (Path(file).stem + '.jpg')
                img.save(new_path, 'JPEG')
                cleaned_fake += 1
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")

    logger.info(f"Cleaning complete. Saved {cleaned_real} real and {cleaned_fake} fake images.")
    return cleaned_real, cleaned_fake

def extract_faces(face_cascade_path: str = None):
    """Extract faces from the cleaned images to focus on relevant regions."""
    if face_cascade_path is None:
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        raise PipelineError(f"Failed to load face cascade model from {face_cascade_path}")

    face_dir_real = Path(CLEANED_DATA_DIR) / "real" / "faces"
    face_dir_fake = Path(CLEANED_DATA_DIR) / "fake" / "faces"
    face_dir_real.mkdir(exist_ok=True)
    face_dir_fake.mkdir(exist_ok=True)

    extracted_count = 0
    for label in ['real', 'fake']:
        logger.info(f"Extracting faces from {label} images...")
        img_dir = Path(CLEANED_DATA_DIR) / label / "images"
        output_dir = Path(CLEANED_DATA_DIR) / label / "faces"
        
        for file in tqdm(list(img_dir.glob("*.jpg")), desc=f"Extracting {label} faces"):
            try:
                img = cv2.imread(str(file))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda r: r[2] * r[3]) # a single largest face
                    face_img = img[y:y+h, x:x+w]
                    cv2.imwrite(str(output_dir / file.name), cv2.resize(face_img, DEFAULT_IMAGE_SIZE))
                    extracted_count += 1
            except Exception as e:
                logger.error(f"Error extracting face from {file}: {e}")

    logger.info(f"Face extraction complete. Extracted {extracted_count} faces.")
    return extracted_count

def create_image_train_val_test_split(use_faces: bool = False, test_size=0.2, val_size=0.25):
    """Split cleaned images (or faces) into train, val, and test sets."""
    data_subdir = "faces" if use_faces else "images"
    real_files = glob.glob(os.path.join(CLEANED_DATA_DIR, "real", data_subdir, "*.jpg"))
    fake_files = glob.glob(os.path.join(CLEANED_DATA_DIR, "fake", data_subdir, "*.jpg"))

    if not real_files and not fake_files:
        logger.warning(f"No cleaned images found in '{data_subdir}' to create splits.")
        return

    df = pd.DataFrame({
        'filepath': real_files + fake_files,
        'label': ['real'] * len(real_files) + ['fake'] * len(fake_files)
    })
    
    train_val_df, test_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, stratify=train_val_df['label'], random_state=42)
    
    logger.info(f"Image dataset split: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test samples.")
    
    suffix = "_faces" if use_faces else ""
    train_df.to_csv(os.path.join(CLEANED_DATA_DIR, f"train{suffix}.csv"), index=False)
    val_df.to_csv(os.path.join(CLEANED_DATA_DIR, f"val{suffix}.csv"), index=False)
    test_df.to_csv(os.path.join(CLEANED_DATA_DIR, f"test{suffix}.csv"), index=False)
    logger.info("Image split CSVs saved.")

# --- VISUALIZATION ---

def visualize_image_distribution(use_faces: bool = False):
    """Visualize the distribution of real and fake images."""
    data_subdir = "faces" if use_faces else "images"
    real_count = len(glob.glob(os.path.join(CLEANED_DATA_DIR, "real", data_subdir, "*.jpg")))
    fake_count = len(glob.glob(os.path.join(CLEANED_DATA_DIR, "fake", data_subdir, "*.jpg")))

    if real_count == 0 and fake_count == 0:
        logger.info(f"No cleaned images in '{data_subdir}' to visualize.")
        return

    plt.figure(figsize=(8, 6))
    plt.pie([real_count, fake_count], labels=['Real', 'Fake'], autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
    plt.title(f'Image Dataset Distribution ({"Faces" if use_faces else "Full"})')
    plt.axis('equal')
    plt.savefig(os.path.join(RESULTS_DIR, f'image_distribution{"_faces" if use_faces else ""}.png'))
    plt.close()
    logger.info("Image distribution chart saved.")

# --- MAIN ---

def main():
    """Main function to run the image processing pipeline."""
    parser = argparse.ArgumentParser(description='Image Deepfake Detection Pipeline')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing the dataset (with real/ and fake/ subfolders)')
    parser.add_argument('--clean', action='store_true', help='Clean and preprocess the dataset')
    parser.add_argument('--extract-faces', action='store_true', help='Extract faces from cleaned images')
    parser.add_argument('--split', action='store_true', help='Split the dataset into train/val/test sets')
    parser.add_argument('--use-faces-for-split', action='store_true', help='Use face-extracted images for splitting')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    args = parser.parse_args()

    try:
        real_dir = Path(args.data_dir) / 'real'
        fake_dir = Path(args.data_dir) / 'fake'

        if not real_dir.exists() or not fake_dir.exists():
            raise DataValidationError(f"Source data directory must contain 'real' and 'fake' subfolders. Path checked: {args.data_dir}")

        img_ext = SUPPORTED_EXTENSIONS['images']
        real_files = [f for ext in img_ext for f in glob.glob(str(real_dir / f"**/*{ext}"), recursive=True)]
        fake_files = [f for ext in img_ext for f in glob.glob(str(fake_dir / f"**/*{ext}"), recursive=True)]
        
        logger.info(f"Found {len(real_files)} real images and {len(fake_files)} fake images.")

        if args.clean:
            clean_dataset_images(real_files, fake_files)
        
        if args.extract_faces:
            extract_faces()

        if args.split:
            create_image_train_val_test_split(use_faces=args.use_faces_for_split)
            
        if args.visualize:
            visualize_image_distribution(use_faces=False)
            if (Path(CLEANED_DATA_DIR) / "real" / "faces").exists():
                 visualize_image_distribution(use_faces=True)

        logger.info("Image pipeline completed successfully.")

    except PipelineError as e:
        logger.error(f"Pipeline error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 