"""
Training Package for Deepfake Detection

This package provides modular training capabilities for deepfake detection models
with comprehensive evaluation and visualization features.
"""

from .data_loader import DataLoader
from .trainer import ModelTrainer
from .evaluation import ModelEvaluator
from .visualization import ModelVisualizer

__all__ = [
    'DataLoader',
    'ModelTrainer', 
    'ModelEvaluator',
    'ModelVisualizer'
]

__version__ = '1.0.0'
__author__ = 'Deepfake Detection Team'
__description__ = 'Modular training framework for deepfake detection models' 