# Configuration file for the deepfake detection pipeline
# Fill in or adjust paths and parameters as needed

from pathlib import Path

# Base data directory
DATA_DIR = str(Path(__file__).parent / 'data')
REAL_DIR = str(Path(DATA_DIR) / 'real')
FAKE_DIR = str(Path(DATA_DIR) / 'fake')
CLEANED_DATA_DIR = str(Path(__file__).parent.parent / 'cleaned_data')
RESULTS_DIR = str(Path(__file__).parent.parent / 'results')

# User's data directory structure
USER_DATA_DIR = "/Users/ajayp/work/gcu/capstone-data/data"

# Video data structure
VIDEO_DATA_DIR = str(Path(USER_DATA_DIR) / "video")
VIDEO_TRAIN_DIR = str(Path(VIDEO_DATA_DIR) / "training")
VIDEO_VAL_DIR = str(Path(VIDEO_DATA_DIR) / "validation")  # Fixed: using "validation"
VIDEO_TEST_DIR = str(Path(VIDEO_DATA_DIR) / "testing")

# Audio data structure
AUDIO_DATA_DIR = str(Path(USER_DATA_DIR) / "audio")
AUDIO_TRAIN_DIR = str(Path(AUDIO_DATA_DIR) / "training")
AUDIO_VAL_DIR = str(Path(AUDIO_DATA_DIR) / "validation")  # Fixed: using "validation"
AUDIO_TEST_DIR = str(Path(AUDIO_DATA_DIR) / "testing")

# Image data structure
IMAGE_DATA_DIR = str(Path(USER_DATA_DIR) / "images")
IMAGE_TRAIN_DIR = str(Path(IMAGE_DATA_DIR) / "training")
IMAGE_VAL_DIR = str(Path(IMAGE_DATA_DIR) / "validation")  # Fixed: using "validation"
IMAGE_TEST_DIR = str(Path(IMAGE_DATA_DIR) / "testing")

# Supported file extensions for each media type
SUPPORTED_EXTENSIONS = {
    'images': ['.jpg', '.jpeg', '.png', '.bmp'],
    'videos': ['.mp4', '.avi', '.mov', '.mkv'],
    'audio': ['.wav', '.mp3', '.flac', '.ogg'],
    'text': ['.txt', '.csv', '.json']
}

# Default parameters
DEFAULT_BATCH_SIZE = 32
DEFAULT_IMAGE_SIZE = (256, 256)
DEFAULT_AUDIO_SAMPLE_RATE = 22050
DEFAULT_AUDIO_DURATION = 3  # seconds
DEFAULT_VIDEO_FPS = 1 