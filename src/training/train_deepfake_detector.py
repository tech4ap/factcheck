"""
Deepfake Detection Model Training Script

This script provides a modular framework for training deepfake detection models
for images, videos, and audio. It includes:

- Memory-efficient data loading with generators
- Modular training functions for each media type
- Comprehensive logging and error handling
- GPU acceleration support
- Model evaluation and visualization

Author: Deepfake Detection Team
Date: 2024
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import glob
import logging
from pathlib import Path
import argparse
from typing import Tuple, Dict, Any, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, cohen_kappa_score, hamming_loss,
    jaccard_score, log_loss, roc_auc_score
)
import gc
import time

# Audio processing imports
try:
    import librosa
    import librosa.display
    from scipy import signal
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logging.warning("Audio processing libraries (librosa, scipy) not available. Audio processing will be disabled.")

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import CLEANED_DATA_DIR, RESULTS_DIR, VIDEO_TRAIN_DIR, VIDEO_VAL_DIR, VIDEO_TEST_DIR, IMAGE_TRAIN_DIR, IMAGE_VAL_DIR, IMAGE_TEST_DIR, AUDIO_TRAIN_DIR, AUDIO_VAL_DIR, AUDIO_TEST_DIR
from models.deepfake_detector import (
    ImageDeepfakeDetector, VideoDeepfakeDetector, AudioDeepfakeDetector,
    EnsembleDeepfakeDetector, create_callbacks, evaluate_model
)

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

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class DataLoader:
    """
    Memory-efficient data loader for different media types.
    
    This class handles loading and preprocessing of image, video, and audio data
    with support for batch processing to manage memory usage.
    """
    
    def __init__(self, data_dir: str, max_samples: Optional[int] = None):
        """
        Initialize the data loader.
        
        Args:
            data_dir (str): Path to the directory containing processed data
            max_samples (Optional[int]): Maximum number of samples to load (for testing)
        """
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples
        logger.info(f"Initialized DataLoader with data directory: {self.data_dir}")
        
    def _validate_csv_file(self, csv_path: Path, split_name: str) -> bool:
        """
        Validate that a CSV file exists and is readable.
        
        Args:
            csv_path (Path): Path to the CSV file
            split_name (str): Name of the data split (train/val/test)
            
        Returns:
            bool: True if file is valid, False otherwise
        """
        if not csv_path.exists():
            logger.warning(f"CSV file not found for {split_name}: {csv_path}")
            return False
        return True
    
    def _load_csv_data(self, split: str, file_suffix: str = "") -> pd.DataFrame:
        """
        Load and validate CSV data for a specific split.
        
        Args:
            split (str): Data split name (train/val/test)
            file_suffix (str): Optional suffix for the CSV filename
            
        Returns:
            pd.DataFrame: Loaded CSV data
        """
        csv_filename = f"{split}{file_suffix}.csv"
        csv_path = self.data_dir / csv_filename
        
        if not self._validate_csv_file(csv_path, split):
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} samples for {split} split")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file {csv_path}: {e}")
            return pd.DataFrame()
    
    def _preprocess_image(self, image_path: str, target_size: Tuple[int, int] = (256, 256)) -> Optional[np.ndarray]:
        """
        Load and preprocess a single image.
        
        Args:
            image_path (str): Path to the image file
            target_size (Tuple[int, int]): Target size for resizing (width, height)
            
        Returns:
            Optional[np.ndarray]: Preprocessed image or None if loading failed
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, target_size)
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            return img
        except Exception as e:
            logger.warning(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def load_image_data(self, split: str = 'train', target_size: Tuple[int, int] = (256, 256)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load image data for training/validation/testing.
        
        Args:
            split (str): Data split name (train/val/test)
            target_size (Tuple[int, int]): Target size for image resizing
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (images, labels) arrays
        """
        logger.info(f"Loading image data for {split} split...")
        
        df = self._load_csv_data(split)
        if df.empty:
            return np.array([]), np.array([])
        
        # Limit samples if specified (useful for testing)
        if self.max_samples:
            df = df.head(self.max_samples)
            logger.info(f"Limited to {self.max_samples} samples for testing")
        
        images = []
        labels = []
        successful_loads = 0
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:  # Progress logging
                logger.info(f"Processing image {idx + 1}/{len(df)}")
            
            img = self._preprocess_image(row['filepath'], target_size)
            if img is not None:
                images.append(img)
                # Convert label to binary (fake=1, real=0)
                labels.append(1 if row['label'] == 'fake' else 0)
                successful_loads += 1
        
        logger.info(f"Successfully loaded {successful_loads}/{len(df)} images for {split} split")
        
        return np.array(images), np.array(labels)
    
    def load_image_data_from_directories(self, split: str = 'train', target_size: Tuple[int, int] = (256, 256)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load image data directly from directory structure.
        
        Args:
            split (str): Data split name (train/val/test)
            target_size (Tuple[int, int]): Target size for image resizing
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (images, labels) arrays
        """
        logger.info(f"Loading image data from directories for {split} split...")
        
        # Map split names to directory paths
        split_dirs = {
            'train': IMAGE_TRAIN_DIR,
            'val': IMAGE_VAL_DIR,
            'validation': IMAGE_VAL_DIR,
            'test': IMAGE_TEST_DIR,
            'testing': IMAGE_TEST_DIR
        }
        
        if split not in split_dirs:
            logger.error(f"Invalid split: {split}. Must be one of {list(split_dirs.keys())}")
            return np.array([]), np.array([])
        
        base_dir = Path(split_dirs[split])
        fake_dir = base_dir / "fake"
        real_dir = base_dir / "real"
        
        if not fake_dir.exists() or not real_dir.exists():
            logger.error(f"Directory structure not found: {fake_dir} or {real_dir}")
            return np.array([]), np.array([])
        
        # Get image files
        fake_images = list(fake_dir.glob("*.jpg")) + list(fake_dir.glob("*.jpeg")) + list(fake_dir.glob("*.png"))
        real_images = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.jpeg")) + list(real_dir.glob("*.png"))
        
        logger.info(f"Found {len(fake_images)} fake images and {len(real_images)} real images")
        
        # Limit samples if specified
        if self.max_samples:
            max_per_class = self.max_samples // 2
            fake_images = fake_images[:max_per_class]
            real_images = real_images[:max_per_class]
            logger.info(f"Limited to {len(fake_images)} fake and {len(real_images)} real images for testing")
        
        images = []
        labels = []
        successful_loads = 0
        
        # Process fake images (label = 1)
        for i, image_path in enumerate(fake_images):
            if i % 100 == 0:
                logger.info(f"Processing fake image {i + 1}/{len(fake_images)}")
            
            img = self._preprocess_image(str(image_path), target_size)
            if img is not None:
                images.append(img)
                labels.append(1)  # fake
                successful_loads += 1
        
        # Process real images (label = 0)
        for i, image_path in enumerate(real_images):
            if i % 100 == 0:
                logger.info(f"Processing real image {i + 1}/{len(real_images)}")
            
            img = self._preprocess_image(str(image_path), target_size)
            if img is not None:
                images.append(img)
                labels.append(0)  # real
                successful_loads += 1
        
        logger.info(f"Successfully loaded {successful_loads} images for {split} split")
        
        if successful_loads == 0:
            return np.array([]), np.array([])
        
        return np.array(images), np.array(labels)
    
    def load_video_data(self, split: str = 'train', frames_per_video: int = 10, 
                       target_size: Tuple[int, int] = (256, 256)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load video frame data for training/validation/testing.
        
        Args:
            split (str): Data split name (train/val/test)
            frames_per_video (int): Number of frames to extract per video
            target_size (Tuple[int, int]): Target size for frame resizing
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (video_sequences, labels) arrays
        """
        logger.info(f"Loading video data for {split} split...")
        
        df = self._load_csv_data(split, "_videos")
        if df.empty:
            return np.array([]), np.array([])
        
        # Limit samples if specified
        if self.max_samples:
            df = df.head(self.max_samples)
            logger.info(f"Limited to {self.max_samples} videos for testing")
        
        video_sequences = []
        labels = []
        successful_loads = 0
        
        for idx, row in df.iterrows():
            if idx % 10 == 0:  # Progress logging (videos are slower to process)
                logger.info(f"Processing video {idx + 1}/{len(df)}")
            
            try:
                video_dir = Path(row['video_dir'])
                frame_files = sorted(glob.glob(str(video_dir / "*.jpg")))[:frames_per_video]
                
                if len(frame_files) == frames_per_video:
                    frames = []
                    frame_load_success = True
                    
                    for frame_file in frame_files:
                        img = self._preprocess_image(frame_file, target_size)
                        if img is not None:
                            frames.append(img)
                        else:
                            frame_load_success = False
                            break
                    
                    if frame_load_success and len(frames) == frames_per_video:
                        video_sequences.append(frames)
                        labels.append(1 if row['label'] == 'fake' else 0)
                        successful_loads += 1
                        
                        # Memory management: clear frames after processing
                        del frames
                        gc.collect()
                        
            except Exception as e:
                logger.warning(f"Error loading video {row['video_dir']}: {e}")
        
        logger.info(f"Successfully loaded {successful_loads}/{len(df)} videos for {split} split")
        
        return np.array(video_sequences), np.array(labels)
    
    def load_video_data_from_directories(self, split: str = 'train', frames_per_video: int = 10, 
                                        target_size: Tuple[int, int] = (256, 256)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load video frame data directly from directory structure.
        
        Args:
            split (str): Data split name (train/val/test)
            frames_per_video (int): Number of frames to extract per video
            target_size (Tuple[int, int]): Target size for frame resizing
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (video_sequences, labels) arrays
        """
        logger.info(f"Loading video data from directories for {split} split...")
        
        # Map split names to directory paths
        split_dirs = {
            'train': VIDEO_TRAIN_DIR,
            'val': VIDEO_VAL_DIR,
            'validation': VIDEO_VAL_DIR,
            'test': VIDEO_TEST_DIR,
            'testing': VIDEO_TEST_DIR
        }
        
        if split not in split_dirs:
            logger.error(f"Invalid split: {split}. Must be one of {list(split_dirs.keys())}")
            return np.array([]), np.array([])
        
        base_dir = Path(split_dirs[split])
        fake_dir = base_dir / "fake"
        real_dir = base_dir / "real"
        
        if not fake_dir.exists() or not real_dir.exists():
            logger.error(f"Directory structure not found: {fake_dir} or {real_dir}")
            return np.array([]), np.array([])
        
        # Get video files
        fake_videos = list(fake_dir.glob("*.mp4"))
        real_videos = list(real_dir.glob("*.mp4"))
        
        logger.info(f"Found {len(fake_videos)} fake videos and {len(real_videos)} real videos")
        
        # Limit samples if specified
        if self.max_samples:
            max_per_class = self.max_samples // 2
            fake_videos = fake_videos[:max_per_class]
            real_videos = real_videos[:max_per_class]
            logger.info(f"Limited to {len(fake_videos)} fake and {len(real_videos)} real videos for testing")
        
        video_sequences = []
        labels = []
        successful_loads = 0
        
        # Process fake videos (label = 1)
        for i, video_path in enumerate(fake_videos):
            if i % 10 == 0:
                logger.info(f"Processing fake video {i + 1}/{len(fake_videos)}")
            
            frames = self._extract_video_frames(str(video_path), frames_per_video, target_size)
            if frames is not None and len(frames) == frames_per_video:
                video_sequences.append(frames)
                labels.append(1)  # fake
                successful_loads += 1
        
        # Process real videos (label = 0)
        for i, video_path in enumerate(real_videos):
            if i % 10 == 0:
                logger.info(f"Processing real video {i + 1}/{len(real_videos)}")
            
            frames = self._extract_video_frames(str(video_path), frames_per_video, target_size)
            if frames is not None and len(frames) == frames_per_video:
                video_sequences.append(frames)
                labels.append(0)  # real
                successful_loads += 1
        
        logger.info(f"Successfully loaded {successful_loads} videos for {split} split")
        
        if successful_loads == 0:
            return np.array([]), np.array([])
        
        return np.array(video_sequences), np.array(labels)
    
    def _extract_video_frames(self, video_path: str, frames_per_video: int, 
                             target_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Extract frames from a video file.
        
        Args:
            video_path (str): Path to the video file
            frames_per_video (int): Number of frames to extract
            target_size (Tuple[int, int]): Target size for frame resizing
            
        Returns:
            Optional[np.ndarray]: Array of frames or None if extraction failed
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning(f"Could not open video: {video_path}")
                return None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                logger.warning(f"Video has no frames: {video_path}")
                cap.release()
                return None
            
            # Calculate frame indices to extract
            frame_indices = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Resize frame
                    frame = cv2.resize(frame, target_size)
                    # Normalize to [0, 1]
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)
                else:
                    logger.warning(f"Could not read frame {frame_idx} from {video_path}")
                    cap.release()
                    return None
            
            cap.release()
            
            if len(frames) == frames_per_video:
                return np.array(frames)
            else:
                logger.warning(f"Extracted {len(frames)} frames, expected {frames_per_video} from {video_path}")
                return None
                
        except Exception as e:
            logger.warning(f"Error extracting frames from {video_path}: {e}")
            return None
    
    def load_audio_data(self, split: str = 'train', target_size: Tuple[int, int] = (128, 128)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load audio spectrogram data for training/validation/testing.
        
        Args:
            split (str): Data split name (train/val/test)
            target_size (Tuple[int, int]): Target size for spectrogram resizing
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (spectrograms, labels) arrays
        """
        logger.info(f"Loading audio data for {split} split...")
        
        df = self._load_csv_data(split, "_audio")
        if df.empty:
            return np.array([]), np.array([])
        
        # Limit samples if specified
        if self.max_samples:
            df = df.head(self.max_samples)
            logger.info(f"Limited to {self.max_samples} audio samples for testing")
        
        spectrograms = []
        labels = []
        successful_loads = 0
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:  # Progress logging
                logger.info(f"Processing audio {idx + 1}/{len(df)}")
            
            try:
                # Load spectrogram as grayscale image
                img = cv2.imread(row['filepath'], cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, target_size)
                    img = img.astype(np.float32) / 255.0
                    img = np.expand_dims(img, axis=-1)  # Add channel dimension
                    spectrograms.append(img)
                    labels.append(1 if row['label'] == 'fake' else 0)
                    successful_loads += 1
            except Exception as e:
                logger.warning(f"Error loading spectrogram {row['filepath']}: {e}")
        
        logger.info(f"Successfully loaded {successful_loads}/{len(df)} spectrograms for {split} split")
        
        return np.array(spectrograms), np.array(labels)
    
    def _audio_to_spectrogram(self, audio_path: str, target_size: Tuple[int, int] = (128, 128), 
                             sample_rate: int = 22050, duration: float = 3.0) -> Optional[np.ndarray]:
        """
        Convert audio file to spectrogram.
        
        Args:
            audio_path (str): Path to the audio file
            target_size (Tuple[int, int]): Target size for spectrogram (height, width)
            sample_rate (int): Target sample rate for audio processing
            duration (float): Duration to extract from audio (seconds)
            
        Returns:
            Optional[np.ndarray]: Spectrogram as grayscale image or None if processing failed
        """
        if not AUDIO_AVAILABLE:
            logger.warning("Audio processing libraries not available")
            return None
            
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=sample_rate, duration=duration)
            
            # If audio is shorter than target duration, pad with zeros
            target_length = int(sample_rate * duration)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
            elif len(y) > target_length:
                y = y[:target_length]
            
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sample_rate,
                n_mels=target_size[0],
                n_fft=2048,
                hop_length=512,
                fmin=20,
                fmax=8000
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to [0, 1]
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
            
            # Resize to target size
            mel_spec_resized = cv2.resize(mel_spec_norm, (target_size[1], target_size[0]))
            
            # Convert to float32 and add channel dimension
            spectrogram = mel_spec_resized.astype(np.float32)
            spectrogram = np.expand_dims(spectrogram, axis=-1)  # Add channel dimension
            
            return spectrogram
            
        except Exception as e:
            logger.warning(f"Error processing audio {audio_path}: {e}")
            return None
    
    def _audio_to_mfcc(self, audio_path: str, target_size: Tuple[int, int] = (128, 128), 
                      sample_rate: int = 22050, duration: float = 3.0) -> Optional[np.ndarray]:
        """
        Convert audio file to MFCC features.
        
        Args:
            audio_path (str): Path to the audio file
            target_size (Tuple[int, int]): Target size for MFCC (height, width)
            sample_rate (int): Target sample rate for audio processing
            duration (float): Duration to extract from audio (seconds)
            
        Returns:
            Optional[np.ndarray]: MFCC features as grayscale image or None if processing failed
        """
        if not AUDIO_AVAILABLE:
            logger.warning("Audio processing libraries not available")
            return None
            
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=sample_rate, duration=duration)
            
            # If audio is shorter than target duration, pad with zeros
            target_length = int(sample_rate * duration)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
            elif len(y) > target_length:
                y = y[:target_length]
            
            # Compute MFCC features
            mfcc = librosa.feature.mfcc(
                y=y, 
                sr=sample_rate,
                n_mfcc=target_size[0],
                n_fft=2048,
                hop_length=512
            )
            
            # Normalize to [0, 1]
            mfcc_norm = (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min() + 1e-8)
            
            # Resize to target size
            mfcc_resized = cv2.resize(mfcc_norm, (target_size[1], target_size[0]))
            
            # Convert to float32 and add channel dimension
            mfcc_features = mfcc_resized.astype(np.float32)
            mfcc_features = np.expand_dims(mfcc_features, axis=-1)  # Add channel dimension
            
            return mfcc_features
            
        except Exception as e:
            logger.warning(f"Error processing audio {audio_path}: {e}")
            return None
    
    def load_audio_data_from_directories(self, split: str = 'train', target_size: Tuple[int, int] = (128, 128),
                                        feature_type: str = 'spectrogram', sample_rate: int = 22050, 
                                        duration: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load audio data directly from directory structure.
        
        Args:
            split (str): Data split name (train/val/test)
            target_size (Tuple[int, int]): Target size for spectrogram resizing
            feature_type (str): Type of audio features to extract ('spectrogram' or 'mfcc')
            sample_rate (int): Target sample rate for audio processing
            duration (float): Duration to extract from audio (seconds)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (spectrograms, labels) arrays
        """
        logger.info(f"Loading audio data from directories for {split} split...")
        
        if not AUDIO_AVAILABLE:
            logger.error("Audio processing libraries not available. Please install librosa and scipy.")
            return np.array([]), np.array([])
        
        # Map split names to directory paths
        split_dirs = {
            'train': AUDIO_TRAIN_DIR,
            'val': AUDIO_VAL_DIR,
            'validation': AUDIO_VAL_DIR,
            'test': AUDIO_TEST_DIR,
            'testing': AUDIO_TEST_DIR
        }
        
        if split not in split_dirs:
            logger.error(f"Invalid split: {split}. Must be one of {list(split_dirs.keys())}")
            return np.array([]), np.array([])
        
        base_dir = Path(split_dirs[split])
        fake_dir = base_dir / "fake"
        real_dir = base_dir / "real"
        
        if not fake_dir.exists() or not real_dir.exists():
            logger.error(f"Directory structure not found: {fake_dir} or {real_dir}")
            return np.array([]), np.array([])
        
        # Get audio files
        fake_audio = list(fake_dir.glob("*.wav")) + list(fake_dir.glob("*.mp3")) + list(fake_dir.glob("*.flac"))
        real_audio = list(real_dir.glob("*.wav")) + list(real_dir.glob("*.mp3")) + list(real_dir.glob("*.flac"))
        
        logger.info(f"Found {len(fake_audio)} fake audio files and {len(real_audio)} real audio files")
        
        # Limit samples if specified
        if self.max_samples:
            max_per_class = self.max_samples // 2
            fake_audio = fake_audio[:max_per_class]
            real_audio = real_audio[:max_per_class]
            logger.info(f"Limited to {len(fake_audio)} fake and {len(real_audio)} real audio files for testing")
        
        spectrograms = []
        labels = []
        successful_loads = 0
        
        # Choose the appropriate audio processing method
        if feature_type.lower() == 'spectrogram':
            process_audio = self._audio_to_spectrogram
        elif feature_type.lower() == 'mfcc':
            process_audio = self._audio_to_mfcc
        else:
            logger.error(f"Invalid feature_type: {feature_type}. Must be 'spectrogram' or 'mfcc'")
            return np.array([]), np.array([])
        
        # Process fake audio (label = 1)
        for i, audio_path in enumerate(fake_audio):
            if i % 100 == 0:
                logger.info(f"Processing fake audio {i + 1}/{len(fake_audio)}")
            
            features = process_audio(str(audio_path), target_size, sample_rate, duration)
            if features is not None:
                spectrograms.append(features)
                labels.append(1)  # fake
                successful_loads += 1
        
        # Process real audio (label = 0)
        for i, audio_path in enumerate(real_audio):
            if i % 100 == 0:
                logger.info(f"Processing real audio {i + 1}/{len(real_audio)}")
            
            features = process_audio(str(audio_path), target_size, sample_rate, duration)
            if features is not None:
                spectrograms.append(features)
                labels.append(0)  # real
                successful_loads += 1
        
        logger.info(f"Successfully loaded {successful_loads} audio files for {split} split")
        
        if successful_loads == 0:
            return np.array([]), np.array([])
        
        return np.array(spectrograms), np.array(labels)


class ModelTrainer:
    """
    Modular trainer class for deepfake detection models.
    
    This class provides separate training methods for each media type
    with comprehensive logging, error handling, and memory management.
    """
    
    def __init__(self, data_dir: str, output_dir: str = "models", 
                 max_samples: Optional[int] = None):
        """
        Initialize the model trainer.
        
        Args:
            data_dir (str): Path to the directory containing processed data
            output_dir (str): Directory to save trained models and results
            max_samples (Optional[int]): Maximum number of samples to use (for testing)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.max_samples = max_samples
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize data loader and visualizer
        self.data_loader = DataLoader(str(self.data_dir), max_samples)
        self.visualizer = ModelVisualizer(str(self.output_dir))
        
        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        logger.info(f"Found {len(gpus)} GPU(s): {gpus}")
        
        if gpus:
            # Enable memory growth to prevent GPU memory overflow
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU memory growth enabled")
        
        logger.info(f"Initialized ModelTrainer with output directory: {self.output_dir}")
    
    def _create_data_augmentation(self) -> ImageDataGenerator:
        """Create basic data augmentation for images."""
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    
    def _create_enhanced_data_augmentation(self) -> ImageDataGenerator:
        """Create enhanced data augmentation for better generalization."""
        return ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            preprocessing_function=self._preprocessing_function
        )
    
    def _preprocessing_function(self, img):
        """Custom preprocessing function for better normalization."""
        # Add noise for robustness
        if np.random.random() < 0.1:  # 10% chance
            noise = np.random.normal(0, 0.01, img.shape)
            img = np.clip(img + noise, 0, 1)
        
        # Random contrast adjustment
        if np.random.random() < 0.2:  # 20% chance
            factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 1)
        
        return img
    
    def _create_advanced_data_augmentation(self) -> ImageDataGenerator:
        """Create advanced data augmentation for maximum generalization."""
        return ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[0.7, 1.3],
            channel_shift_range=0.2,
            fill_mode='reflect',  # Better than 'nearest' for deepfakes
            preprocessing_function=self._advanced_preprocessing_function
        )
    
    def _advanced_preprocessing_function(self, img):
        """Advanced preprocessing function with multiple augmentation techniques."""
        # Convert to numpy if tensor
        if hasattr(img, 'numpy'):
            img = img.numpy()
        
        # Ensure correct data type and range
        if img.max() <= 1.0:
            img = img * 255.0
        img = img.astype(np.uint8)
        
        # Random gaussian noise (helps with generalization)
        if np.random.random() < 0.3:
            noise = np.random.normal(0, np.random.uniform(1, 5), img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        # Random blur (simulates compression artifacts)
        if np.random.random() < 0.2:
            kernel_size = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        # Random JPEG compression simulation
        if np.random.random() < 0.2:
            quality = np.random.randint(70, 95)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, img_encoded = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)
        
        # Random saturation adjustment
        if np.random.random() < 0.3:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            saturation_factor = np.random.uniform(0.7, 1.3)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Convert back to float32 and normalize
        img = img.astype(np.float32) / 255.0
        
        # Random contrast adjustment
        if np.random.random() < 0.3:
            factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 1)
        
        return img
    
    def _mixup_generator(self, generator, alpha=0.2):
        """Create mixup augmentation generator."""
        while True:
            batch_x, batch_y = next(generator)
            batch_size = batch_x.shape[0]
            
            # Generate mixup coefficients
            lambda_vals = np.random.beta(alpha, alpha, batch_size)
            
            # Shuffle indices for mixing
            indices = np.random.permutation(batch_size)
            
            # Apply mixup
            mixed_x = np.zeros_like(batch_x)
            mixed_y = np.zeros_like(batch_y)
            
            for i in range(batch_size):
                lam = lambda_vals[i]
                mixed_x[i] = lam * batch_x[i] + (1 - lam) * batch_x[indices[i]]
                mixed_y[i] = lam * batch_y[i] + (1 - lam) * batch_y[indices[i]]
            
            yield mixed_x, mixed_y
    
    def _cutmix_generator(self, generator, alpha=1.0):
        """Create cutmix augmentation generator."""
        while True:
            batch_x, batch_y = next(generator)
            batch_size = batch_x.shape[0]
            
            for i in range(batch_size):
                if np.random.random() < 0.5:  # Apply cutmix with 50% probability
                    # Generate random box
                    lam = np.random.beta(alpha, alpha)
                    H, W = batch_x.shape[1], batch_x.shape[2]
                    
                    cut_ratio = np.sqrt(1 - lam)
                    cut_w = int(W * cut_ratio)
                    cut_h = int(H * cut_ratio)
                    
                    cx = np.random.randint(W)
                    cy = np.random.randint(H)
                    
                    bbx1 = np.clip(cx - cut_w // 2, 0, W)
                    bby1 = np.clip(cy - cut_h // 2, 0, H)
                    bbx2 = np.clip(cx + cut_w // 2, 0, W)
                    bby2 = np.clip(cy + cut_h // 2, 0, H)
                    
                    # Select random image to mix with
                    rand_index = np.random.randint(batch_size)
                    
                    # Apply cutmix
                    batch_x[i, bby1:bby2, bbx1:bbx2, :] = batch_x[rand_index, bby1:bby2, bbx1:bbx2, :]
                    
                    # Adjust labels
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
                    batch_y[i] = lam * batch_y[i] + (1 - lam) * batch_y[rand_index]
            
            yield batch_x, batch_y
    
    def _create_enhanced_callbacks(self, model_name: str, learning_rate: float, patience: int = 20):
        """Create enhanced callbacks for better training."""
        callbacks_list = [
            # Enhanced early stopping based on validation AUC
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=patience,
                restore_best_weights=True,
                mode='max',
                verbose=1,
                min_delta=0.001
            ),
            
            # Advanced learning rate scheduling
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc',
                factor=0.3,
                patience=patience // 3,
                min_lr=learning_rate / 1000,
                mode='max',
                verbose=1,
                cooldown=5
            ),
            
            # Model checkpointing with multiple criteria
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'{self.output_dir}/{model_name}_best_auc.h5',
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1,
                save_weights_only=False
            ),
            
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'{self.output_dir}/{model_name}_best_loss.h5',
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1,
                save_weights_only=False
            ),
            
            # Cosine annealing with warm restarts
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: self._cosine_annealing_with_warmup(epoch, learning_rate),
                verbose=0
            ),
            
            # Advanced logging
            tf.keras.callbacks.CSVLogger(
                f'{self.output_dir}/{model_name}_training_log.csv',
                append=True
            ),
            
            # Gradient accumulation simulation through batch size adjustment
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: self._log_advanced_metrics(epoch, logs, model_name)
            )
        ]
        
        return callbacks_list
    
    def _cosine_annealing_with_warmup(self, epoch, initial_lr, warmup_epochs=5, total_epochs=100):
        """Cosine annealing learning rate schedule with warmup."""
        if epoch < warmup_epochs:
            # Linear warmup
            return initial_lr * (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
    
    def _log_advanced_metrics(self, epoch, logs, model_name):
        """Log advanced metrics for better monitoring."""
        if logs:
            # Calculate F1 score from precision and recall
            precision = logs.get('val_precision', 0)
            recall = logs.get('val_recall', 0)
            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
                logger.info(f"Epoch {epoch + 1} - F1 Score: {f1_score:.4f}")
            
            # Log AUC and other important metrics
            val_auc = logs.get('val_auc', 0)
            val_pr_auc = logs.get('val_pr_auc', 0)
            logger.info(f"Epoch {epoch + 1} - Val AUC: {val_auc:.4f}, Val PR-AUC: {val_pr_auc:.4f}")
    
    def _save_training_history(self, history: tf.keras.callbacks.History, 
                              model_name: str, training_time: float = None) -> None:
        """
        Save training history plots and metrics using the visualizer.
        
        Args:
            history (tf.keras.callbacks.History): Training history object
            model_name (str): Name of the model for file naming
            training_time (float): Total training time in seconds
        """
        try:
            # Use the visualizer to create comprehensive plots
            self.visualizer.plot_training_history(history, model_name)
            
            # Create training summary report if training time is provided
            if training_time is not None:
                self.visualizer.create_training_summary_report(history, model_name, training_time)
            
            logger.info(f"Saved comprehensive training visualizations for {model_name}")
            
        except Exception as e:
            logger.warning(f"Error saving training history for {model_name}: {e}")
    
    def train_image_model(self, epochs: int = 100, batch_size: int = 32, 
                         fine_tune: bool = True, learning_rate: float = 1e-4) -> Optional[ImageDeepfakeDetector]:
        """
        Train the enhanced image deepfake detection model.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            fine_tune (bool): Whether to perform fine-tuning
            learning_rate (float): Initial learning rate
            
        Returns:
            Optional[ImageDeepfakeDetector]: Trained model or None if training failed
        """
        logger.info("Starting enhanced image model training...")
        start_time = time.time()
        
        try:
            # Load data
            logger.info("Loading image data...")
            train_images, train_labels = self.data_loader.load_image_data_from_directories('train')
            val_images, val_labels = self.data_loader.load_image_data_from_directories('validation')
            
            if len(train_images) == 0:
                logger.error("No training data found for images")
                return None
            
            logger.info(f"Loaded {len(train_images)} training images, {len(val_images)} validation images")
            
            # Create model
            logger.info("Building enhanced image model...")
            image_model = ImageDeepfakeDetector()
            model = image_model.build_model(learning_rate=learning_rate)
            
            # Enhanced data augmentation
            datagen = self._create_enhanced_data_augmentation()
            
            # Enhanced callbacks
            callbacks = self._create_enhanced_callbacks("image_model", learning_rate)
            
            # Train model with better strategy
            logger.info("Starting initial training...")
            history = model.fit(
                datagen.flow(train_images, train_labels, batch_size=batch_size),
                steps_per_epoch=max(1, len(train_images) // batch_size),
                epochs=epochs,
                validation_data=(val_images, val_labels),
                callbacks=callbacks,
                verbose=1,
                class_weight={0: 1.0, 1: 1.0}  # Balanced class weights
            )
            
            # Enhanced fine-tuning
            if fine_tune:
                logger.info("Starting enhanced fine-tuning...")
                image_model.fine_tune()
                
                fine_tune_callbacks = self._create_enhanced_callbacks("image_model_finetune", learning_rate/10)
                
                fine_tune_history = model.fit(
                    datagen.flow(train_images, train_labels, batch_size=batch_size),
                    steps_per_epoch=max(1, len(train_images) // batch_size),
                    epochs=epochs // 2,
                    validation_data=(val_images, val_labels),
                    callbacks=fine_tune_callbacks,
                    verbose=1,
                    class_weight={0: 1.0, 1: 1.0}
                )
            
            # Save model and history
            model.save(self.output_dir / "image_model_final.h5")
            self._save_training_history(history, "image_model", time.time() - start_time)
            
            logger.info("Enhanced image model training completed")
            
            return image_model
            
        except Exception as e:
            logger.error(f"Error during enhanced image model training: {e}")
            return None
    
    def train_enhanced_image_model(self, epochs: int = 100, batch_size: int = 32, 
                                 fine_tune: bool = True, learning_rate: float = 1e-4,
                                 use_advanced_augmentation: bool = True,
                                 use_mixup: bool = True, use_cutmix: bool = True,
                                 base_model: str = 'efficientnet',
                                 use_multiscale: bool = True,
                                 use_attention: bool = True) -> Optional[ImageDeepfakeDetector]:
        """
        Train the enhanced image deepfake detection model with advanced techniques.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            fine_tune (bool): Whether to perform fine-tuning
            learning_rate (float): Initial learning rate
            use_advanced_augmentation (bool): Use advanced data augmentation
            use_mixup (bool): Use mixup augmentation
            use_cutmix (bool): Use cutmix augmentation
            base_model (str): Base model architecture
            use_multiscale (bool): Use multi-scale feature extraction
            use_attention (bool): Use attention mechanisms
            
        Returns:
            Optional[ImageDeepfakeDetector]: Trained model or None if training failed
        """
        logger.info("Starting enhanced image model training with advanced techniques...")
        start_time = time.time()
        
        try:
            # Load data
            logger.info("Loading image data...")
            train_images, train_labels = self.data_loader.load_image_data_from_directories('train')
            val_images, val_labels = self.data_loader.load_image_data_from_directories('validation')
            
            if len(train_images) == 0:
                logger.error("No training data found for images")
                return None
            
            logger.info(f"Loaded {len(train_images)} training images, {len(val_images)} validation images")
            
            # Calculate class weights for imbalanced data
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(train_labels),
                y=train_labels
            )
            class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
            logger.info(f"Calculated class weights: {class_weight_dict}")
            
            # Create enhanced model
            logger.info(f"Building enhanced image model with {base_model}...")
            image_model = ImageDeepfakeDetector(
                base_model=base_model,
                use_attention=use_attention,
                use_multiscale=use_multiscale
            )
            model = image_model.build_model(
                learning_rate=learning_rate,
                use_focal_loss=True,
                focal_alpha=0.25,
                focal_gamma=2.0
            )
            
            # Print model summary
            logger.info("Model architecture summary:")
            model.summary(print_fn=logger.info)
            summary_info = image_model.get_model_summary()
            logger.info(f"Model details: {summary_info}")
            
            # Create data generators with advanced augmentation
            if use_advanced_augmentation:
                train_datagen = self._create_advanced_data_augmentation()
            else:
                train_datagen = self._create_enhanced_data_augmentation()
            
            val_datagen = ImageDataGenerator()  # No augmentation for validation
            
            # Create data generators
            train_generator = train_datagen.flow(
                train_images, train_labels, 
                batch_size=batch_size,
                shuffle=True
            )
            
            val_generator = val_datagen.flow(
                val_images, val_labels,
                batch_size=batch_size,
                shuffle=False
            )
            
            # Apply advanced augmentation techniques
            if use_mixup:
                logger.info("Applying MixUp augmentation...")
                train_generator = self._mixup_generator(train_generator, alpha=0.2)
            
            if use_cutmix:
                logger.info("Applying CutMix augmentation...")
                train_generator = self._cutmix_generator(train_generator, alpha=1.0)
            
            # Enhanced callbacks
            callbacks = self._create_enhanced_callbacks("enhanced_image_model", learning_rate)
            
            # Calculate steps per epoch
            steps_per_epoch = max(1, len(train_images) // batch_size)
            validation_steps = max(1, len(val_images) // batch_size)
            
            # Initial training phase
            logger.info("Starting initial training phase...")
            history = model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=val_generator,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1,
                class_weight=class_weight_dict,
                workers=4,
                use_multiprocessing=True,
                max_queue_size=10
            )
            
            # Enhanced fine-tuning phase
            if fine_tune:
                logger.info("Starting enhanced fine-tuning phase...")
                image_model.fine_tune(learning_rate=learning_rate/10, unfreeze_layers=-50)
                
                # Create new callbacks for fine-tuning
                fine_tune_callbacks = self._create_enhanced_callbacks(
                    "enhanced_image_model_finetune", 
                    learning_rate/10,
                    patience=15
                )
                
                # Fine-tuning with reduced epochs and learning rate
                fine_tune_history = model.fit(
                    train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs // 2,
                    validation_data=val_generator,
                    validation_steps=validation_steps,
                    callbacks=fine_tune_callbacks,
                    verbose=1,
                    class_weight=class_weight_dict,
                    workers=4,
                    use_multiprocessing=True,
                    max_queue_size=10
                )
                
                # Combine training histories
                for key in history.history:
                    if key in fine_tune_history.history:
                        history.history[key].extend(fine_tune_history.history[key])
            
            # Save final model
            final_model_path = self.output_dir / "enhanced_image_model_final.h5"
            model.save(final_model_path)
            logger.info(f"Saved final model to {final_model_path}")
            
            # Save training history and create visualizations
            training_time = time.time() - start_time
            self._save_training_history(history, "enhanced_image_model", training_time)
            
            # Log final performance
            final_metrics = {
                'val_auc': max(history.history.get('val_auc', [0])),
                'val_accuracy': max(history.history.get('val_accuracy', [0])),
                'val_precision': max(history.history.get('val_precision', [0])),
                'val_recall': max(history.history.get('val_recall', [0])),
                'training_time_minutes': training_time / 60
            }
            
            logger.info("Enhanced image model training completed!")
            logger.info(f"Final performance metrics: {final_metrics}")
            
            return image_model
            
        except Exception as e:
            logger.error(f"Error during enhanced image model training: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
        
        finally:
            # Cleanup
            tf.keras.backend.clear_session()
            gc.collect()
    
    def train_video_model(self, epochs: int = 100, batch_size: int = 8, 
                         frames_per_video: int = 10, learning_rate: float = 1e-4) -> Optional[VideoDeepfakeDetector]:
        """
        Train the enhanced video deepfake detection model.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Training batch size (reduced for memory)
            frames_per_video (int): Number of frames per video sequence
            learning_rate (float): Learning rate
            
        Returns:
            Optional[VideoDeepfakeDetector]: Trained model or None if training failed
        """
        logger.info("Starting enhanced video model training...")
        start_time = time.time()
        
        try:
            # Load data using directory structure
            logger.info("Loading video data...")
            train_videos, train_labels = self.data_loader.load_video_data_from_directories('train', frames_per_video)
            val_videos, val_labels = self.data_loader.load_video_data_from_directories('validation', frames_per_video)
            
            if len(train_videos) == 0:
                logger.error("No training data found for videos")
                return None
            
            logger.info(f"Loaded {len(train_videos)} training videos, {len(val_videos)} validation videos")
            
            # Create model
            logger.info("Building enhanced video model...")
            video_model = VideoDeepfakeDetector(frame_sequence_length=frames_per_video)
            model = video_model.build_model(learning_rate=learning_rate)
            
            # Enhanced callbacks
            callbacks = self._create_enhanced_callbacks("video_model", learning_rate)
            
            # Train model with better strategy
            logger.info("Starting enhanced video model training...")
            history = model.fit(
                train_videos, train_labels,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(val_videos, val_labels),
                callbacks=callbacks,
                verbose=1,
                class_weight={0: 1.0, 1: 1.0}
            )
            
            # Save model and history
            model.save(self.output_dir / "video_model_final.h5")
            self._save_training_history(history, "video_model", time.time() - start_time)
            
            logger.info("Enhanced video model training completed")
            
            return video_model
            
        except Exception as e:
            logger.error(f"Error during enhanced video model training: {e}")
            return None
    
    def train_audio_model(self, epochs: int = 100, batch_size: int = 32, 
                         learning_rate: float = 1e-4) -> Optional[AudioDeepfakeDetector]:
        """
        Train the enhanced audio deepfake detection model.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            learning_rate (float): Learning rate
            
        Returns:
            Optional[AudioDeepfakeDetector]: Trained model or None if training failed
        """
        logger.info("Starting enhanced audio model training...")
        start_time = time.time()
        
        try:
            # Load data
            logger.info("Loading audio data...")
            train_audio, train_labels = self.data_loader.load_audio_data_from_directories('train')
            val_audio, val_labels = self.data_loader.load_audio_data_from_directories('validation')
            
            if len(train_audio) == 0:
                logger.error("No training data found for audio")
                return None
            
            logger.info(f"Loaded {len(train_audio)} training spectrograms, {len(val_audio)} validation spectrograms")
            
            # Create model
            logger.info("Building enhanced audio model...")
            audio_model = AudioDeepfakeDetector()
            model = audio_model.build_model(learning_rate=learning_rate)
            
            # Enhanced callbacks
            callbacks = self._create_enhanced_callbacks("audio_model", learning_rate)
            
            # Train model with better strategy
            logger.info("Starting enhanced audio model training...")
            history = model.fit(
                train_audio, train_labels,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(val_audio, val_labels),
                callbacks=callbacks,
                verbose=1,
                class_weight={0: 1.0, 1: 1.0}
            )
            
            # Save model and history
            model.save(self.output_dir / "audio_model_final.h5")
            self._save_training_history(history, "audio_model", time.time() - start_time)
            
            logger.info("Enhanced audio model training completed")
            
            return audio_model
            
        except Exception as e:
            logger.error(f"Error during enhanced audio model training: {e}")
            return None
    
    def _calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics for model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted binary labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dict[str, float]: Comprehensive metrics dictionary
        """
        # Basic metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate comprehensive metrics
        metrics = {
            # Basic classification metrics
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            
            # Additional metrics
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Same as recall
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred),
            
            # Probability-based metrics
            'auc': roc_auc_score(y_true, y_pred_proba),
            'average_precision': average_precision_score(y_true, y_pred_proba),
            'log_loss': log_loss(y_true, y_pred_proba),
            
            # Additional classification metrics
            'hamming_loss': hamming_loss(y_true, y_pred),
            'jaccard_score': jaccard_score(y_true, y_pred),
            
            # Confusion matrix components
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            
            # Rates
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Same as precision
            'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0,
        }
        
        return metrics
    
    def _evaluate_model_comprehensive(self, model, test_data: np.ndarray, 
                                    test_labels: np.ndarray, model_name: str) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of a single model.
        
        Args:
            model: Trained model
            test_data: Test data
            test_labels: True labels
            model_name: Name of the model
            
        Returns:
            Dict[str, Any]: Comprehensive evaluation results
        """
        logger.info(f"Performing comprehensive evaluation for {model_name}...")
        
        # Get predictions
        y_pred_proba = model.predict(test_data)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_pred_proba = y_pred_proba.flatten()
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(test_labels, y_pred, y_pred_proba)
        
        # Generate classification report
        class_report = classification_report(test_labels, y_pred, 
                                           target_names=['Real', 'Fake'], 
                                           output_dict=True)
        
        # Calculate ROC and PR curves
        fpr, tpr, roc_thresholds = roc_curve(test_labels, y_pred_proba)
        precision, recall, pr_thresholds = precision_recall_curve(test_labels, y_pred_proba)
        
        # Create comprehensive visualizations
        self.visualizer.plot_comprehensive_evaluation(
            test_labels, y_pred, y_pred_proba, model_name,
            fpr, tpr, precision, recall, class_report
        )
        
        # Detailed metrics are already created by plot_detailed_metrics method above
        
        return {
            'metrics': metrics,
            'classification_report': class_report,
            'predictions': {
                'probabilities': y_pred_proba,
                'binary': y_pred
            },
            'curves': {
                'roc': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds},
                'pr': {'precision': precision, 'recall': recall, 'thresholds': pr_thresholds}
            }
        }

    def evaluate_models(self, image_model: Optional[ImageDeepfakeDetector] = None,
                       video_model: Optional[VideoDeepfakeDetector] = None,
                       audio_model: Optional[AudioDeepfakeDetector] = None,
                       frames_per_video: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models on test data with comprehensive visualizations.
        
        Args:
            image_model: Trained image model
            video_model: Trained video model
            audio_model: Trained audio model
            frames_per_video: Number of frames per video for evaluation
            
        Returns:
            Dict[str, Dict[str, float]]: Evaluation results for each model
        """
        logger.info("Starting comprehensive model evaluation...")
        results = {}
        detailed_results = {}
        
        # Evaluate image model
        if image_model:
            logger.info("Evaluating image model...")
            test_images, test_labels = self.data_loader.load_image_data_from_directories('testing')
            if len(test_images) > 0:
                detailed_results['image'] = self._evaluate_model_comprehensive(
                    image_model.model, test_images, test_labels, "image_model"
                )
                results['image'] = detailed_results['image']['metrics']
                logger.info(f"Image model evaluation completed")
        
        # Evaluate video model
        if video_model:
            logger.info("Evaluating video model...")
            test_videos, test_labels = self.data_loader.load_video_data_from_directories('testing', frames_per_video)
            if len(test_videos) > 0:
                detailed_results['video'] = self._evaluate_model_comprehensive(
                    video_model.model, test_videos, test_labels, "video_model"
                )
                results['video'] = detailed_results['video']['metrics']
                logger.info(f"Video model evaluation completed")
        
        # Evaluate audio model
        if audio_model:
            logger.info("Evaluating audio model...")
            test_audio, test_labels = self.data_loader.load_audio_data_from_directories('testing')
            if len(test_audio) > 0:
                detailed_results['audio'] = self._evaluate_model_comprehensive(
                    audio_model.model, test_audio, test_labels, "audio_model"
                )
                results['audio'] = detailed_results['audio']['metrics']
                logger.info(f"Audio model evaluation completed")
        
        # Save comprehensive results
        if results:
            # Save basic metrics
            results_df = pd.DataFrame(results).T
            results_df.to_csv(self.output_dir / "model_evaluation_results.csv")
            logger.info("Saved evaluation results to CSV")
            
            # Save detailed results
            self._save_detailed_evaluation_results(detailed_results)
            
            # Create enhanced model comparison visualization
            self.visualizer.plot_enhanced_model_comparison(results)
            self.visualizer.create_performance_summary_report(results, detailed_results)
            self.visualizer.create_comprehensive_evaluation_report(results, detailed_results)
            logger.info("Created enhanced model comparison visualizations")
        
        return results
    
    def _save_detailed_evaluation_results(self, detailed_results: Dict[str, Dict[str, Any]]) -> None:
        """
        Save detailed evaluation results including classification reports and curves.
        
        Args:
            detailed_results: Detailed evaluation results for each model
        """
        import json
        
        # Prepare data for JSON serialization
        serializable_results = {}
        for model_name, results in detailed_results.items():
            serializable_results[model_name] = {
                'metrics': results['metrics'],
                'classification_report': results['classification_report'],
                'predictions': {
                    'probabilities': results['predictions']['probabilities'].tolist(),
                    'binary': results['predictions']['binary'].tolist()
                },
                'curves': {
                    'roc': {
                        'fpr': results['curves']['roc']['fpr'].tolist(),
                        'tpr': results['curves']['roc']['tpr'].tolist(),
                        'thresholds': results['curves']['roc']['thresholds'].tolist()
                    },
                    'pr': {
                        'precision': results['curves']['pr']['precision'].tolist(),
                        'recall': results['curves']['pr']['recall'].tolist(),
                        'thresholds': results['curves']['pr']['thresholds'].tolist()
                    }
                }
            }
        
        # Save to JSON file
        with open(self.output_dir / "detailed_evaluation_results.json", 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info("Saved detailed evaluation results to JSON")


class ModelVisualizer:
    """
    Comprehensive visualization class for deepfake detection models.
    
    This class provides various visualization capabilities including:
    - Training history plots
    - Model performance metrics
    - Confusion matrices
    - ROC curves
    - Data distribution analysis
    """
    
    def __init__(self, output_dir: str = "models"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir (str): Directory to save visualization plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different types of visualizations
        (self.output_dir / "training_plots").mkdir(exist_ok=True)
        (self.output_dir / "performance_plots").mkdir(exist_ok=True)
        (self.output_dir / "data_analysis").mkdir(exist_ok=True)
    
    def plot_training_history(self, history: tf.keras.callbacks.History, model_name: str) -> None:
        """
        Create comprehensive training history plots.
        
        Args:
            history: Training history object
            model_name: Name of the model for file naming
        """
        logger.info(f"Creating training history plots for {model_name}")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name.replace("_", " ").title()} Training History', fontsize=16, fontweight='bold')
        
        # Plot accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history.history:
            axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[0, 1].plot(history.history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history.history:
            axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot precision
        if 'precision' in history.history:
            axes[1, 0].plot(history.history['precision'], label='Training Precision', linewidth=2)
            if 'val_precision' in history.history:
                axes[1, 0].plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot recall
        if 'recall' in history.history:
            axes[1, 1].plot(history.history['recall'], label='Training Recall', linewidth=2)
            if 'val_recall' in history.history:
                axes[1, 1].plot(history.history['val_recall'], label='Validation Recall', linewidth=2)
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_plots" / f"{model_name}_training_history.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history plots saved to {self.output_dir / 'training_plots' / f'{model_name}_training_history.png'}")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            model_name: str, class_names: List[str] = None) -> None:
        """
        Create and save confusion matrix plot.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            class_names: Names of the classes
        """
        if class_names is None:
            class_names = ['Real', 'Fake']
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_name.replace("_", " ").title()} Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Add text annotations
        total = np.sum(cm)
        accuracy = np.trace(cm) / total
        plt.text(0.5, -0.1, f'Accuracy: {accuracy:.3f}', 
                ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_plots" / f"{model_name}_confusion_matrix.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {self.output_dir / 'performance_plots' / f'{model_name}_confusion_matrix.png'}")
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                      model_name: str) -> None:
        """
        Create and save ROC curve plot.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
        """
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name.replace("_", " ").title()} ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_plots" / f"{model_name}_roc_curve.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved to {self.output_dir / 'performance_plots' / f'{model_name}_roc_curve.png'}")
    
    def plot_data_distribution(self, train_data: Dict[str, int], val_data: Dict[str, int], 
                             test_data: Dict[str, int], model_name: str) -> None:
        """
        Create data distribution visualization.
        
        Args:
            train_data: Training data counts
            val_data: Validation data counts
            test_data: Test data counts
            model_name: Name of the model
        """
        # Prepare data for plotting
        splits = ['Train', 'Validation', 'Test']
        real_counts = [train_data.get('real', 0), val_data.get('real', 0), test_data.get('real', 0)]
        fake_counts = [train_data.get('fake', 0), val_data.get('fake', 0), test_data.get('fake', 0)]
        
        # Create stacked bar plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Stacked bar chart
        x = np.arange(len(splits))
        width = 0.35
        
        ax1.bar(x, real_counts, width, label='Real', color='skyblue')
        ax1.bar(x, fake_counts, width, bottom=real_counts, label='Fake', color='lightcoral')
        
        ax1.set_xlabel('Data Split')
        ax1.set_ylabel('Number of Samples')
        ax1.set_title(f'{model_name.replace("_", " ").title()} Data Distribution')
        ax1.set_xticks(x)
        ax1.set_xticklabels(splits)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (real, fake) in enumerate(zip(real_counts, fake_counts)):
            ax1.text(i, real/2, str(real), ha='center', va='center', fontweight='bold')
            ax1.text(i, real + fake/2, str(fake), ha='center', va='center', fontweight='bold')
        
        # Pie chart for total distribution
        total_real = sum(real_counts)
        total_fake = sum(fake_counts)
        
        ax2.pie([total_real, total_fake], labels=['Real', 'Fake'], 
               autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
        ax2.set_title('Overall Data Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "data_analysis" / f"{model_name}_data_distribution.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Data distribution plot saved to {self.output_dir / 'data_analysis' / f'{model_name}_data_distribution.png'}")
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        Create model comparison visualization.
        
        Args:
            results: Dictionary containing model evaluation results
        """
        if not results:
            logger.warning("No results provided for model comparison")
            return
        
        # Prepare data
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            row = i // 3
            col = i % 3
            
            values = [results[model].get(metric, 0) for model in models]
            
            bars = axes[row, col].bar(models, values, color=['skyblue', 'lightcoral', 'lightgreen'])
            axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
            axes[row, col].set_ylabel(metric.replace("_", " ").title())
            axes[row, col].set_ylim(0, 1)
            axes[row, col].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[row, col].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                  f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Remove the last subplot if not needed
        if len(metrics) < 6:
            axes[1, 2].remove()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_plots" / "model_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison plot saved to {self.output_dir / 'performance_plots' / 'model_comparison.png'}")
    
    def create_training_summary_report(self, history: tf.keras.callbacks.History, 
                                     model_name: str, training_time: float) -> None:
        """
        Create a comprehensive training summary report.
        
        Args:
            history: Training history object
            model_name: Name of the model
            training_time: Total training time in seconds
        """
        # Create summary figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name.replace("_", " ").title()} Training Summary', 
                    fontsize=16, fontweight='bold')
        
        # Final metrics
        final_metrics = {}
        for metric in ['accuracy', 'loss', 'precision', 'recall']:
            if metric in history.history:
                final_metrics[metric] = history.history[metric][-1]
                if f'val_{metric}' in history.history:
                    final_metrics[f'val_{metric}'] = history.history[f'val_{metric}'][-1]
        
        # Create metrics table
        metrics_text = "Final Training Metrics:\n\n"
        for metric, value in final_metrics.items():
            metrics_text += f"{metric.replace('_', ' ').title()}: {value:.4f}\n"
        metrics_text += f"\nTraining Time: {training_time:.2f} seconds"
        
        axes[0, 0].text(0.1, 0.9, metrics_text, transform=axes[0, 0].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[0, 0].set_title('Training Summary')
        axes[0, 0].axis('off')
        
        # Training curves
        if 'accuracy' in history.history:
            axes[0, 1].plot(history.history['accuracy'], label='Training', linewidth=2)
            if 'val_accuracy' in history.history:
                axes[0, 1].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
            axes[0, 1].set_title('Accuracy Over Time')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Loss curves
        if 'loss' in history.history:
            axes[1, 0].plot(history.history['loss'], label='Training', linewidth=2)
            if 'val_loss' in history.history:
                axes[1, 0].plot(history.history['val_loss'], label='Validation', linewidth=2)
            axes[1, 0].set_title('Loss Over Time')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate (if available)
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'], linewidth=2)
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Learning Rate Schedule')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_plots" / f"{model_name}_training_summary.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training summary report saved to {self.output_dir / 'training_plots' / f'{model_name}_training_summary.png'}")

    def plot_comprehensive_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    y_pred_proba: np.ndarray, model_name: str,
                                    fpr: np.ndarray, tpr: np.ndarray, 
                                    precision: np.ndarray, recall: np.ndarray,
                                    class_report: Dict[str, Any]) -> None:
        """
        Create comprehensive evaluation visualizations.
        
        Args:
            y_true: True labels
            y_pred: Predicted binary labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            fpr: False positive rate
            tpr: True positive rate
            precision: Precision
            recall: Recall
            class_report: Classification report
        """
        # Create confusion matrix plot
        self.plot_confusion_matrix(y_true, y_pred, model_name)
        
        # Create ROC curve plot
        self.plot_roc_curve(y_true, y_pred_proba, model_name)
        
        # Create precision-recall curve
        self.plot_precision_recall_curve(y_true, y_pred_proba, model_name)
        
        # Create detailed metrics visualization
        self.plot_detailed_metrics(y_true, y_pred, y_pred_proba, model_name)
        
        # Create classification report visualization
        self.plot_classification_report(class_report, model_name)
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                   model_name: str) -> None:
        """
        Create and save precision-recall curve plot.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
        """
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkgreen', lw=2, 
                label=f'PR curve (AP = {avg_precision:.3f})')
        plt.axhline(y=0.5, color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_name.replace("_", " ").title()} Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_plots" / f"{model_name}_precision_recall_curve.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Precision-recall curve saved to {self.output_dir / 'performance_plots' / f'{model_name}_precision_recall_curve.png'}")
    
    def plot_detailed_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_pred_proba: np.ndarray, model_name: str) -> None:
        """
        Create detailed metrics visualization.
        
        Args:
            y_true: True labels
            y_pred: Predicted binary labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
        """
        # Calculate comprehensive metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Create detailed metrics figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{model_name.replace("_", " ").title()} Detailed Metrics Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Confusion matrix heatmap
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'], ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # Metrics comparison
        metrics = {
            'Accuracy': (tp + tn) / (tp + tn + fp + fn),
            'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'F1-Score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'AUC': roc_auc_score(y_true, y_pred_proba)
        }
        
        axes[0, 1].bar(metrics.keys(), metrics.values(), color=['skyblue', 'lightcoral', 'lightgreen', 
                                                               'gold', 'purple', 'orange'])
        axes[0, 1].set_title('Key Metrics')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Prediction distribution
        axes[0, 2].hist(y_pred_proba[y_true == 0], bins=20, alpha=0.7, label='Real', color='blue')
        axes[0, 2].hist(y_pred_proba[y_true == 1], bins=20, alpha=0.7, label='Fake', color='red')
        axes[0, 2].set_title('Prediction Distribution')
        axes[0, 2].set_xlabel('Predicted Probability')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].legend()
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        axes[1, 0].set_xlim([0.0, 1.0])
        axes[1, 0].set_ylim([0.0, 1.05])
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curve')
        axes[1, 0].legend(loc="lower right")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        axes[1, 1].plot(recall, precision, color='darkgreen', lw=2, 
                       label=f'PR curve (AP = {avg_precision:.3f})')
        axes[1, 1].axhline(y=0.5, color='navy', lw=2, linestyle='--', label='Random')
        axes[1, 1].set_xlim([0.0, 1.0])
        axes[1, 1].set_ylim([0.0, 1.05])
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision-Recall Curve')
        axes[1, 1].legend(loc="lower left")
        axes[1, 1].grid(True, alpha=0.3)
        
        # Additional metrics
        additional_metrics = {
            'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
            'Matthews Corr': matthews_corrcoef(y_true, y_pred),
            'Cohen Kappa': cohen_kappa_score(y_true, y_pred),
            'Hamming Loss': hamming_loss(y_true, y_pred),
            'Jaccard Score': jaccard_score(y_true, y_pred),
            'Log Loss': log_loss(y_true, y_pred_proba)
        }
        
        axes[1, 2].bar(additional_metrics.keys(), additional_metrics.values(), 
                      color=['lightblue', 'lightpink', 'lightyellow', 'lightgray', 'lightcyan', 'lightcoral'])
        axes[1, 2].set_title('Additional Metrics')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_plots" / f"{model_name}_detailed_metrics.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Detailed metrics plot saved to {self.output_dir / 'performance_plots' / f'{model_name}_detailed_metrics.png'}")
    
    def plot_classification_report(self, class_report: Dict[str, Any], model_name: str) -> None:
        """
        Create classification report visualization.
        
        Args:
            class_report: Classification report dictionary
            model_name: Name of the model
        """
        # Create classification report figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{model_name.replace("_", " ").title()} Classification Report', 
                    fontsize=16, fontweight='bold')
        
        # Extract metrics for each class - handle different class name formats
        precision_scores = []
        recall_scores = []
        f1_scores = []
        class_names = []
        
        # Find the actual class names in the report (excluding 'accuracy', 'macro avg', 'weighted avg')
        for key in class_report.keys():
            if key not in ['accuracy', 'macro avg', 'weighted avg']:
                if isinstance(class_report[key], dict) and 'precision' in class_report[key]:
                    precision_scores.append(class_report[key]['precision'])
                    recall_scores.append(class_report[key]['recall'])
                    f1_scores.append(class_report[key]['f1-score'])
                    # Map class names to readable labels
                    if key in ['0', 'real', 'Real']:
                        class_names.append('Real')
                    elif key in ['1', 'fake', 'Fake']:
                        class_names.append('Fake')
                    else:
                        class_names.append(key.title())
        
        # If no classes found, use default names
        if not class_names:
            class_names = ['Class 0', 'Class 1']
            logger.warning(f"No valid class metrics found in classification report for {model_name}")
            return
        
        # Plot metrics by class
        x = np.arange(len(class_names))
        width = 0.25
        
        if len(precision_scores) > 0:
            axes[0].bar(x - width, precision_scores, width, label='Precision', color='skyblue')
            axes[0].bar(x, recall_scores, width, label='Recall', color='lightcoral')
            axes[0].bar(x + width, f1_scores, width, label='F1-Score', color='lightgreen')
            
            axes[0].set_xlabel('Classes')
            axes[0].set_ylabel('Score')
            axes[0].set_title('Metrics by Class')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(class_names)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (prec, rec, f1) in enumerate(zip(precision_scores, recall_scores, f1_scores)):
                axes[0].text(i - width, prec + 0.01, f'{prec:.3f}', ha='center', va='bottom')
                axes[0].text(i, rec + 0.01, f'{rec:.3f}', ha='center', va='bottom')
                axes[0].text(i + width, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom')
        
        # Overall metrics
        if 'accuracy' in class_report:
            overall_metrics = {
                'Accuracy': class_report['accuracy'],
                'Macro Avg Precision': class_report['macro avg']['precision'],
                'Macro Avg Recall': class_report['macro avg']['recall'],
                'Macro Avg F1': class_report['macro avg']['f1-score'],
                'Weighted Avg Precision': class_report['weighted avg']['precision'],
                'Weighted Avg Recall': class_report['weighted avg']['recall'],
                'Weighted Avg F1': class_report['weighted avg']['f1-score']
            }
            
            axes[1].bar(overall_metrics.keys(), overall_metrics.values(), 
                       color=['lightblue', 'lightpink', 'lightyellow', 'lightgray', 'lightcyan', 'lightcoral', 'lightgreen'])
            axes[1].set_title('Overall Metrics')
            axes[1].set_ylabel('Score')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3)
            
            # Add value labels
            for i, (metric, value) in enumerate(overall_metrics.items()):
                axes[1].text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_plots" / f"{model_name}_classification_report.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Classification report plot saved to {self.output_dir / 'performance_plots' / f'{model_name}_classification_report.png'}")
    
    def plot_enhanced_model_comparison(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        Create enhanced model comparison visualization with comprehensive metrics.
        
        Args:
            results: Dictionary containing model evaluation results
        """
        if not results:
            logger.warning("No results provided for model comparison")
            return
        
        # Prepare data
        models = list(results.keys())
        
        # Define metric categories
        basic_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        advanced_metrics = ['specificity', 'sensitivity', 'balanced_accuracy', 'matthews_corrcoef']
        probability_metrics = ['auc', 'average_precision', 'log_loss']
        
        # Create comprehensive comparison plot
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Enhanced Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Basic metrics
        for i, metric in enumerate(basic_metrics):
            row = i // 2
            col = i % 2
            
            values = [results[model].get(metric, 0) for model in models]
            colors = ['skyblue', 'lightcoral', 'lightgreen'][:len(models)]
            
            bars = axes[row, col].bar(models, values, color=colors)
            axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
            axes[row, col].set_ylabel(metric.replace("_", " ").title())
            axes[row, col].set_ylim(0, 1)
            axes[row, col].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[row, col].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                  f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Advanced metrics
        for i, metric in enumerate(advanced_metrics):
            row = 1
            col = i % 2
            
            values = [results[model].get(metric, 0) for model in models]
            colors = ['lightblue', 'lightpink', 'lightyellow'][:len(models)]
            
            bars = axes[row, col].bar(models, values, color=colors)
            axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
            axes[row, col].set_ylabel(metric.replace("_", " ").title())
            axes[row, col].set_ylim(0, 1)
            axes[row, col].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[row, col].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                  f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Probability-based metrics
        for i, metric in enumerate(probability_metrics):
            row = 2
            col = i
            
            values = [results[model].get(metric, 0) for model in models]
            colors = ['lightgray', 'lightcyan', 'lightcoral'][:len(models)]
            
            bars = axes[row, col].bar(models, values, color=colors)
            axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
            axes[row, col].set_ylabel(metric.replace("_", " ").title())
            axes[row, col].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[row, col].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                  f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_plots" / "enhanced_model_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Enhanced model comparison plot saved to {self.output_dir / 'performance_plots' / 'enhanced_model_comparison.png'}")
    
    def create_performance_summary_report(self, basic_results: Dict[str, Dict[str, float]], 
                                        detailed_results: Dict[str, Dict[str, Any]]) -> None:
        """
        Create comprehensive performance summary report.
        
        Args:
            basic_results: Basic evaluation results
            detailed_results: Detailed evaluation results
        """
        # Create performance summary figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Model Performance Summary', fontsize=16, fontweight='bold')
        
        # Overall accuracy comparison
        models = list(basic_results.keys())
        accuracies = [basic_results[model]['accuracy'] for model in models]
        
        bars = axes[0, 0].bar(models, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'][:len(models)])
        axes[0, 0].set_title('Overall Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # F1-Score comparison
        f1_scores = [basic_results[model]['f1_score'] for model in models]
        
        bars = axes[0, 1].bar(models, f1_scores, color=['lightblue', 'lightpink', 'lightyellow'][:len(models)])
        axes[0, 1].set_title('F1-Score Comparison')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, f1 in zip(bars, f1_scores):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # AUC comparison
        auc_scores = [basic_results[model]['auc'] for model in models]
        
        bars = axes[1, 0].bar(models, auc_scores, color=['lightgray', 'lightcyan', 'lightcoral'][:len(models)])
        axes[1, 0].set_title('AUC Comparison')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, auc_score in zip(bars, auc_scores):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{auc_score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Performance summary text
        summary_text = "Performance Summary:\n\n"
        for model, metrics in basic_results.items():
            summary_text += f"{model.title()} Model:\n"
            summary_text += f"  Accuracy: {metrics['accuracy']:.3f}\n"
            summary_text += f"  F1-Score: {metrics['f1_score']:.3f}\n"
            summary_text += f"  AUC: {metrics['auc']:.3f}\n"
            summary_text += f"  Precision: {metrics['precision']:.3f}\n"
            summary_text += f"  Recall: {metrics['recall']:.3f}\n\n"
        
        summary_text += "Recommendations:\n"
        summary_text += "- Consider ensemble methods\n"
        summary_text += "- Try data augmentation\n"
        summary_text += "- Experiment with different architectures\n"
        summary_text += "- Optimize hyperparameters\n"
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Summary & Recommendations')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_plots" / "comprehensive_performance_summary.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comprehensive performance summary saved to {self.output_dir / 'performance_plots' / 'comprehensive_performance_summary.png'}")
    
    def create_comprehensive_evaluation_report(self, results: Dict[str, Dict[str, float]], 
                                             detailed_results: Dict[str, Dict[str, Any]]) -> None:
        """
        Create a comprehensive text-based evaluation report.
        
        Args:
            results: Basic evaluation results
            detailed_results: Detailed evaluation results
        """
        report_path = self.output_dir / "comprehensive_evaluation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE DEEPFAKE DETECTION MODEL EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            best_model = None
            best_accuracy = 0
            
            for model_name, metrics in results.items():
                if metrics['accuracy'] > best_accuracy:
                    best_accuracy = metrics['accuracy']
                    best_model = model_name
            
            f.write(f"Best Performing Model: {best_model.title()} (Accuracy: {best_accuracy:.3f})\n")
            f.write(f"Total Models Evaluated: {len(results)}\n\n")
            
            # Detailed Model Analysis
            for model_name, metrics in results.items():
                f.write(f"{model_name.upper()} MODEL ANALYSIS\n")
                f.write("-" * 40 + "\n")
                
                # Basic Metrics
                f.write("Basic Classification Metrics:\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n")
                f.write(f"  Specificity: {metrics['specificity']:.4f}\n")
                f.write(f"  Sensitivity: {metrics['sensitivity']:.4f}\n\n")
                
                # Advanced Metrics
                f.write("Advanced Metrics:\n")
                f.write(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}\n")
                f.write(f"  Matthews Correlation Coefficient: {metrics['matthews_corrcoef']:.4f}\n")
                f.write(f"  Cohen's Kappa: {metrics['cohen_kappa']:.4f}\n")
                f.write(f"  Hamming Loss: {metrics['hamming_loss']:.4f}\n")
                f.write(f"  Jaccard Score: {metrics['jaccard_score']:.4f}\n\n")
                
                # Probability-based Metrics
                f.write("Probability-based Metrics:\n")
                f.write(f"  AUC (ROC): {metrics['auc']:.4f}\n")
                f.write(f"  Average Precision: {metrics['average_precision']:.4f}\n")
                f.write(f"  Log Loss: {metrics['log_loss']:.4f}\n\n")
                
                # Confusion Matrix Analysis
                f.write("Confusion Matrix Analysis:\n")
                f.write(f"  True Positives: {metrics['true_positives']}\n")
                f.write(f"  True Negatives: {metrics['true_negatives']}\n")
                f.write(f"  False Positives: {metrics['false_positives']}\n")
                f.write(f"  False Negatives: {metrics['false_negatives']}\n\n")
                
                # Error Rates
                f.write("Error Rates:\n")
                f.write(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}\n")
                f.write(f"  False Negative Rate: {metrics['false_negative_rate']:.4f}\n")
                f.write(f"  Positive Predictive Value: {metrics['positive_predictive_value']:.4f}\n")
                f.write(f"  Negative Predictive Value: {metrics['negative_predictive_value']:.4f}\n\n")
                
                # Performance Assessment
                f.write("Performance Assessment:\n")
                if metrics['accuracy'] >= 0.9:
                    f.write("  Overall Performance: EXCELLENT\n")
                elif metrics['accuracy'] >= 0.8:
                    f.write("  Overall Performance: GOOD\n")
                elif metrics['accuracy'] >= 0.7:
                    f.write("  Overall Performance: FAIR\n")
                else:
                    f.write("  Overall Performance: POOR\n")
                
                if metrics['auc'] >= 0.9:
                    f.write("  Discriminative Ability: EXCELLENT\n")
                elif metrics['auc'] >= 0.8:
                    f.write("  Discriminative Ability: GOOD\n")
                elif metrics['auc'] >= 0.7:
                    f.write("  Discriminative Ability: FAIR\n")
                else:
                    f.write("  Discriminative Ability: POOR\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
            
            # Model Comparison
            f.write("MODEL COMPARISON\n")
            f.write("-" * 40 + "\n")
            
            # Create comparison table
            metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'balanced_accuracy']
            
            f.write(f"{'Metric':<20}")
            for model_name in results.keys():
                f.write(f"{model_name.title():<15}")
            f.write("\n")
            
            f.write("-" * (20 + 15 * len(results)) + "\n")
            
            for metric in metrics_to_compare:
                f.write(f"{metric.replace('_', ' ').title():<20}")
                for model_name in results.keys():
                    value = results[model_name].get(metric, 0)
                    f.write(f"{value:.3f}".ljust(15))
                f.write("\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            
            f.write("Based on the evaluation results, consider the following recommendations:\n\n")
            
            # Find areas for improvement
            for model_name, metrics in results.items():
                f.write(f"{model_name.title()} Model Recommendations:\n")
                
                if metrics['precision'] < 0.8:
                    f.write("  - Low precision indicates high false positive rate\n")
                    f.write("  - Consider adjusting classification threshold\n")
                    f.write("  - Implement additional preprocessing steps\n")
                
                if metrics['recall'] < 0.8:
                    f.write("  - Low recall indicates high false negative rate\n")
                    f.write("  - Consider data augmentation techniques\n")
                    f.write("  - Review feature engineering approaches\n")
                
                if metrics['auc'] < 0.8:
                    f.write("  - Low AUC indicates poor discriminative ability\n")
                    f.write("  - Consider ensemble methods\n")
                    f.write("  - Try different model architectures\n")
                
                if metrics['balanced_accuracy'] < 0.8:
                    f.write("  - Low balanced accuracy indicates class imbalance issues\n")
                    f.write("  - Consider class balancing techniques\n")
                    f.write("  - Review data collection strategy\n")
                
                f.write("\n")
            
            f.write("General Recommendations:\n")
            f.write("- Implement ensemble methods combining multiple models\n")
            f.write("- Use cross-validation for more robust evaluation\n")
            f.write("- Consider domain-specific feature engineering\n")
            f.write("- Implement real-time monitoring of model performance\n")
            f.write("- Regular model retraining with new data\n")
            f.write("- A/B testing for model deployment decisions\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Comprehensive evaluation report saved to {report_path}")
    
    def create_data_visualizations(self) -> None:
        """
        Create data distribution visualizations for all media types.
        """
        logger.info("Creating data distribution visualizations...")
        
        try:
            # Get video data distribution
            train_fake_videos = len(list(Path(VIDEO_TRAIN_DIR).glob("fake/*.mp4")))
            train_real_videos = len(list(Path(VIDEO_TRAIN_DIR).glob("real/*.mp4")))
            val_fake_videos = len(list(Path(VIDEO_VAL_DIR).glob("fake/*.mp4")))
            val_real_videos = len(list(Path(VIDEO_VAL_DIR).glob("real/*.mp4")))
            test_fake_videos = len(list(Path(VIDEO_TEST_DIR).glob("fake/*.mp4")))
            test_real_videos = len(list(Path(VIDEO_TEST_DIR).glob("real/*.mp4")))
            
            train_data = {'fake': train_fake_videos, 'real': train_real_videos}
            val_data = {'fake': val_fake_videos, 'real': val_real_videos}
            test_data = {'fake': test_fake_videos, 'real': test_real_videos}
            
            # Create data distribution visualization
            self.plot_data_distribution(train_data, val_data, test_data, "video_model")
            logger.info("Created video data distribution visualization")
            
        except Exception as e:
            logger.warning(f"Error creating data visualizations: {e}")


def main():
    """
    Main training function with command-line argument parsing.
    
    This function orchestrates the training process for all media types
    based on user specifications.
    """
    parser = argparse.ArgumentParser(
        description='Train Deepfake Detection Models with GPU Acceleration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models with default settings
  python train_deepfake_detector.py --data-dir src/cleaned_data
  
  # Train only video model with custom parameters
  python train_deepfake_detector.py --data-dir src/cleaned_data --media-types video --epochs 20 --batch-size 8
  
  # Test run with limited samples
  python train_deepfake_detector.py --data-dir src/cleaned_data --max-samples 100 --epochs 2
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
    if not Path(args.data_dir).exists():
        logger.error(f"Data directory does not exist: {args.data_dir}")
        return
    
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
    logger.info("=" * 50)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 50)
    for model_type, model in models.items():
        if model is not None:
            logger.info(f"✓ {model_type.capitalize()} model trained successfully")
        else:
            logger.warning(f"✗ {model_type.capitalize()} model training failed")
    
    # Print evaluation summary
    if evaluation_results:
        logger.info("=" * 50)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 50)
        
        for model_type, metrics in evaluation_results.items():
            logger.info(f"\n{model_type.upper()} MODEL PERFORMANCE:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
            logger.info(f"  AUC: {metrics['auc']:.4f}")
            logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
            logger.info(f"  Specificity: {metrics['specificity']:.4f}")
            logger.info(f"  Matthews Correlation: {metrics['matthews_corrcoef']:.4f}")
        
        # Find best performing model
        best_model = max(evaluation_results.items(), key=lambda x: x[1]['accuracy'])
        logger.info(f"\n🏆 BEST PERFORMING MODEL: {best_model[0].upper()}")
        logger.info(f"   Accuracy: {best_model[1]['accuracy']:.4f}")
        logger.info(f"   F1-Score: {best_model[1]['f1_score']:.4f}")
        logger.info(f"   AUC: {best_model[1]['auc']:.4f}")
    
    logger.info("\n" + "=" * 50)
    logger.info("TRAINING COMPLETED!")
    logger.info("=" * 50)
    logger.info("📊 Check the following files for detailed results:")
    logger.info(f"   - {args.output_dir}/model_evaluation_results.csv")
    logger.info(f"   - {args.output_dir}/detailed_evaluation_results.json")
    logger.info(f"   - {args.output_dir}/comprehensive_evaluation_report.txt")
    logger.info(f"   - {args.output_dir}/performance_plots/ (for visualizations)")
    logger.info("=" * 50)


if __name__ == "__main__":
    main() 