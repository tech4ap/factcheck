"""
Data Loader Module for Deepfake Detection

This module provides memory-efficient data loading and preprocessing
for different media types (images, videos, audio) with support for
batch processing to manage memory usage.
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
import glob
import logging
from pathlib import Path
from typing import Tuple, Optional
import gc

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
from config import (
    VIDEO_TRAIN_DIR, VIDEO_VAL_DIR, VIDEO_TEST_DIR,
    AUDIO_TRAIN_DIR, AUDIO_VAL_DIR, AUDIO_TEST_DIR,
    IMAGE_TRAIN_DIR, IMAGE_VAL_DIR, IMAGE_TEST_DIR
)

logger = logging.getLogger(__name__)


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
        
        # First try to load from CSV (legacy support)
        df = self._load_csv_data(split)
        if not df.empty:
            return self._load_image_data_from_csv(df, target_size)
        
        # If no CSV, try loading from directory structure
        return self.load_image_data_from_directories(split, target_size)
    
    def _load_image_data_from_csv(self, df: pd.DataFrame, target_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load image data from CSV file.
        
        Args:
            df: DataFrame containing image paths and labels
            target_size: Target size for image resizing
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (images, labels) arrays
        """
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
        
        logger.info(f"Successfully loaded {successful_loads}/{len(df)} images from CSV")
        
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
        
        # Use the data_dir passed to constructor instead of hardcoded paths
        # For testing, we expect the data_dir to contain the split structure directly
        base_dir = self.data_dir
        
        # If data_dir doesn't contain the split structure directly, try to find it
        if not (base_dir / 'real').exists() and not (base_dir / 'fake').exists():
            # Try to find the split directory within data_dir
            split_dirs = {
                'train': 'train',
                'val': 'validation',
                'validation': 'validation',
                'test': 'testing',
                'testing': 'testing'
            }
            
            if split in split_dirs:
                base_dir = self.data_dir / split_dirs[split]
                logger.info(f"Looking for split directory: {base_dir}")
        
        fake_dir = base_dir / "fake"
        real_dir = base_dir / "real"
        
        # Check if at least one class directory exists
        if not fake_dir.exists() and not real_dir.exists():
            logger.error(f"No class directories found: {fake_dir} or {real_dir}")
            return np.array([]), np.array([])
        
        # Get image files (handle cases where only one class exists)
        fake_images = []
        if fake_dir.exists():
            fake_images = list(fake_dir.glob("*.jpg")) + list(fake_dir.glob("*.jpeg")) + list(fake_dir.glob("*.png"))
        
        real_images = []
        if real_dir.exists():
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
        
        # Use the data_dir passed to constructor instead of hardcoded paths
        # For testing, we expect the data_dir to contain the split structure directly
        base_dir = self.data_dir
        
        # If data_dir doesn't contain the split structure directly, try to find it
        if not (base_dir / 'real').exists() and not (base_dir / 'fake').exists():
            # Try to find the split directory within data_dir
            split_dirs = {
                'train': 'train',
                'val': 'validation',
                'validation': 'validation',
                'test': 'testing',
                'testing': 'testing'
            }
            
            if split in split_dirs:
                base_dir = self.data_dir / split_dirs[split]
                logger.info(f"Looking for split directory: {base_dir}")
        
        fake_dir = base_dir / "fake"
        real_dir = base_dir / "real"
        
        # Check if at least one class directory exists
        if not fake_dir.exists() and not real_dir.exists():
            logger.error(f"No class directories found: {fake_dir} or {real_dir}")
            return np.array([]), np.array([])
        
        # Get video files (handle cases where only one class exists)
        fake_videos = []
        if fake_dir.exists():
            fake_videos = list(fake_dir.glob("*.mp4")) + list(fake_dir.glob("*.avi")) + list(fake_dir.glob("*.mov"))
        
        real_videos = []
        if real_dir.exists():
            real_videos = list(real_dir.glob("*.mp4")) + list(real_dir.glob("*.avi")) + list(real_dir.glob("*.mov"))
        
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
        
        # First try to load from CSV (legacy support)
        df = self._load_csv_data(split, "_audio")
        if not df.empty:
            return self._load_audio_data_from_csv(df, target_size)
        
        # If no CSV, try loading from directory structure
        return self.load_audio_data_from_directories(split, target_size)
    
    def _load_audio_data_from_csv(self, df: pd.DataFrame, target_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load audio data from CSV file.
        
        Args:
            df: DataFrame containing audio paths and labels
            target_size: Target size for spectrogram resizing
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (spectrograms, labels) arrays
        """
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
        
        logger.info(f"Successfully loaded {successful_loads}/{len(df)} spectrograms from CSV")
        
        return np.array(spectrograms), np.array(labels)
    
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
        
        # Use the data_dir passed to constructor instead of hardcoded paths
        # For testing, we expect the data_dir to contain the split structure directly
        base_dir = self.data_dir
        
        # If data_dir doesn't contain the split structure directly, try to find it
        if not (base_dir / 'real').exists() and not (base_dir / 'fake').exists():
            # Try to find the split directory within data_dir
            split_dirs = {
                'train': 'train',
                'val': 'validation',
                'validation': 'validation',
                'test': 'testing',
                'testing': 'testing'
            }
            
            if split in split_dirs:
                base_dir = self.data_dir / split_dirs[split]
                logger.info(f"Looking for split directory: {base_dir}")
        fake_dir = base_dir / "fake"
        real_dir = base_dir / "real"
        
        # Check if at least one class directory exists
        if not fake_dir.exists() and not real_dir.exists():
            logger.error(f"No class directories found: {fake_dir} or {real_dir}")
            return np.array([]), np.array([])
        
        # Get audio files (handle cases where only one class exists)
        fake_audio = []
        if fake_dir.exists():
            fake_audio = list(fake_dir.glob("*.wav")) + list(fake_dir.glob("*.mp3")) + list(fake_dir.glob("*.flac"))
        
        real_audio = []
        if real_dir.exists():
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