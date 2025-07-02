import os
import numpy as np
import pytest
from pathlib import Path
from src.training.data_loader import DataLoader
from src.training.train_deepfake_detector import ModelVisualizer, ModelTrainer

# Dummy config for test
class DummyConfig:
    IMAGE_TRAIN_DIR = 'cleaned_data/image/train'
    IMAGE_VAL_DIR = 'cleaned_data/image/validation'
    IMAGE_TEST_DIR = 'cleaned_data/image/testing'
    VIDEO_TRAIN_DIR = 'cleaned_data/video/train'
    VIDEO_VAL_DIR = 'cleaned_data/video/validation'
    VIDEO_TEST_DIR = 'cleaned_data/video/testing'
    AUDIO_TRAIN_DIR = 'cleaned_data/audio/train'
    AUDIO_VAL_DIR = 'cleaned_data/audio/validation'
    AUDIO_TEST_DIR = 'cleaned_data/audio/testing'

def test_empty_image_directory(tmp_path):
    # Create empty directory structure
    (tmp_path / 'real').mkdir(parents=True)
    (tmp_path / 'fake').mkdir(parents=True)
    loader = DataLoader(str(tmp_path))
    images, labels = loader.load_image_data_from_directories(split='train')
    assert images.size == 0
    assert labels.size == 0

def test_one_class_image_directory(tmp_path):
    # Only 'real' class present
    real_dir = tmp_path / 'real'
    real_dir.mkdir(parents=True)
    # Add a dummy image
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    import cv2
    cv2.imwrite(str(real_dir / 'img1.jpg'), img)
    loader = DataLoader(str(tmp_path))
    images, labels = loader.load_image_data_from_directories(split='train')
    # Should only load real images, label 0
    assert images.shape[0] == 1
    assert np.all(labels == 0)

def test_visualizer_empty_metrics(tmp_path):
    visualizer = ModelVisualizer(str(tmp_path))
    # Should not fail on empty metrics
    visualizer.plot_enhanced_model_comparison({})
    # Should not fail on empty detailed results
    visualizer.create_performance_summary_report({}, {})

def test_metrics_all_correct():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_pred_proba = np.array([0.01, 0.99, 0.02, 0.98])
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    assert accuracy_score(y_true, y_pred) == 1.0
    assert precision_score(y_true, y_pred) == 1.0
    assert recall_score(y_true, y_pred) == 1.0
    assert f1_score(y_true, y_pred) == 1.0

def test_metrics_all_wrong():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0])
    from sklearn.metrics import accuracy_score
    assert accuracy_score(y_true, y_pred) == 0.0

def test_metrics_random():
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_true, y_pred)
    assert 0.0 <= acc <= 1.0 