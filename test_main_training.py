#!/usr/bin/env python3
"""
Tests for the main training module.

This module tests the main training entry point to improve coverage
of the main.py file which currently has 0% coverage.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Import the main module to test
from src.training.main import main, validate_arguments, train_models


class TestMainTraining:
    """Test cases for the main training module."""
    
    def test_validate_arguments_valid(self):
        """Test argument validation with valid arguments."""
        mock_args = Mock()
        mock_args.data_dir = '/valid/path'
        mock_args.epochs = 50
        mock_args.batch_size = 32
        mock_args.learning_rate = 1e-4
        mock_args.frames_per_video = 10
        
        with patch('pathlib.Path.exists', return_value=True):
            result = validate_arguments(mock_args)
            assert result is True
    
    def test_validate_arguments_invalid_data_dir(self):
        """Test argument validation with invalid data directory."""
        mock_args = Mock()
        mock_args.data_dir = '/invalid/path'
        mock_args.epochs = 50
        mock_args.batch_size = 32
        mock_args.learning_rate = 1e-4
        mock_args.frames_per_video = 10
        
        with patch('pathlib.Path.exists', return_value=False):
            result = validate_arguments(mock_args)
            assert result is False
    
    def test_validate_arguments_invalid_epochs(self):
        """Test argument validation with invalid epochs."""
        mock_args = Mock()
        mock_args.data_dir = '/valid/path'
        mock_args.epochs = 0
        mock_args.batch_size = 32
        mock_args.learning_rate = 1e-4
        mock_args.frames_per_video = 10
        
        with patch('pathlib.Path.exists', return_value=True):
            result = validate_arguments(mock_args)
            assert result is False
    
    def test_validate_arguments_invalid_batch_size(self):
        """Test argument validation with invalid batch size."""
        mock_args = Mock()
        mock_args.data_dir = '/valid/path'
        mock_args.epochs = 50
        mock_args.batch_size = 0
        mock_args.learning_rate = 1e-4
        mock_args.frames_per_video = 10
        
        with patch('pathlib.Path.exists', return_value=True):
            result = validate_arguments(mock_args)
            assert result is False
    
    def test_validate_arguments_invalid_learning_rate(self):
        """Test argument validation with invalid learning rate."""
        mock_args = Mock()
        mock_args.data_dir = '/valid/path'
        mock_args.epochs = 50
        mock_args.batch_size = 32
        mock_args.learning_rate = 0
        mock_args.frames_per_video = 10
        
        with patch('pathlib.Path.exists', return_value=True):
            result = validate_arguments(mock_args)
            assert result is False
    
    def test_validate_arguments_invalid_frames_per_video(self):
        """Test argument validation with invalid frames per video."""
        mock_args = Mock()
        mock_args.data_dir = '/valid/path'
        mock_args.epochs = 50
        mock_args.batch_size = 32
        mock_args.learning_rate = 1e-4
        mock_args.frames_per_video = 0
        
        with patch('pathlib.Path.exists', return_value=True):
            result = validate_arguments(mock_args)
            assert result is False
    
    @patch('src.training.main.ModelTrainer')
    def test_train_models_all(self, mock_trainer_class):
        """Test training all models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock trainer
            mock_trainer = Mock()
            mock_trainer_class.return_value = mock_trainer
            
            # Mock successful training
            mock_trainer.train_image_model.return_value = Mock()
            mock_trainer.train_video_model.return_value = Mock()
            mock_trainer.train_audio_model.return_value = Mock()
            mock_trainer.evaluate_models.return_value = {
                'image_model': {'accuracy': 0.8},
                'video_model': {'accuracy': 0.75},
                'audio_model': {'accuracy': 0.7}
            }
            
            # Mock arguments
            mock_args = Mock()
            mock_args.data_dir = temp_dir
            mock_args.output_dir = temp_dir
            mock_args.epochs = 2
            mock_args.batch_size = 4
            mock_args.learning_rate = 1e-4
            mock_args.no_fine_tune = False
            mock_args.media_types = ['image', 'video', 'audio']
            mock_args.max_samples = 10
            mock_args.frames_per_video = 5
            
            # This should not raise an exception
            train_models(mock_args)
            
            # Verify trainer was called
            mock_trainer_class.assert_called_once()
            mock_trainer.train_image_model.assert_called_once()
            mock_trainer.train_video_model.assert_called_once()
            mock_trainer.train_audio_model.assert_called_once()
            mock_trainer.evaluate_models.assert_called_once()
    
    @patch('src.training.main.ModelTrainer')
    def test_train_models_image_only(self, mock_trainer_class):
        """Test training image model only."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock trainer
            mock_trainer = Mock()
            mock_trainer_class.return_value = mock_trainer
            
            # Mock successful training
            mock_trainer.train_image_model.return_value = Mock()
            mock_trainer.evaluate_models.return_value = {
                'image_model': {'accuracy': 0.8}
            }
            
            # Mock arguments
            mock_args = Mock()
            mock_args.data_dir = temp_dir
            mock_args.output_dir = temp_dir
            mock_args.epochs = 2
            mock_args.batch_size = 4
            mock_args.learning_rate = 1e-4
            mock_args.no_fine_tune = False
            mock_args.media_types = ['image']
            mock_args.max_samples = 10
            mock_args.frames_per_video = 5
            
            # This should not raise an exception
            train_models(mock_args)
            
            # Verify only image model was trained
            mock_trainer.train_image_model.assert_called_once()
            mock_trainer.train_video_model.assert_not_called()
            mock_trainer.train_audio_model.assert_not_called()
    
    @patch('src.training.main.ModelTrainer')
    def test_train_models_video_only(self, mock_trainer_class):
        """Test training video model only."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock trainer
            mock_trainer = Mock()
            mock_trainer_class.return_value = mock_trainer
            
            # Mock successful training
            mock_trainer.train_video_model.return_value = Mock()
            mock_trainer.evaluate_models.return_value = {
                'video_model': {'accuracy': 0.75}
            }
            
            # Mock arguments
            mock_args = Mock()
            mock_args.data_dir = temp_dir
            mock_args.output_dir = temp_dir
            mock_args.epochs = 2
            mock_args.batch_size = 4
            mock_args.learning_rate = 1e-4
            mock_args.no_fine_tune = False
            mock_args.media_types = ['video']
            mock_args.max_samples = 10
            mock_args.frames_per_video = 5
            
            # This should not raise an exception
            train_models(mock_args)
            
            # Verify only video model was trained
            mock_trainer.train_image_model.assert_not_called()
            mock_trainer.train_video_model.assert_called_once()
            mock_trainer.train_audio_model.assert_not_called()
    
    @patch('src.training.main.ModelTrainer')
    def test_train_models_audio_only(self, mock_trainer_class):
        """Test training audio model only."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock trainer
            mock_trainer = Mock()
            mock_trainer_class.return_value = mock_trainer
            
            # Mock successful training
            mock_trainer.train_audio_model.return_value = Mock()
            mock_trainer.evaluate_models.return_value = {
                'audio_model': {'accuracy': 0.7}
            }
            
            # Mock arguments
            mock_args = Mock()
            mock_args.data_dir = temp_dir
            mock_args.output_dir = temp_dir
            mock_args.epochs = 2
            mock_args.batch_size = 4
            mock_args.learning_rate = 1e-4
            mock_args.no_fine_tune = False
            mock_args.media_types = ['audio']
            mock_args.max_samples = 10
            mock_args.frames_per_video = 5
            
            # This should not raise an exception
            train_models(mock_args)
            
            # Verify only audio model was trained
            mock_trainer.train_image_model.assert_not_called()
            mock_trainer.train_video_model.assert_not_called()
            mock_trainer.train_audio_model.assert_called_once()
    
    @patch('src.training.main.ModelTrainer')
    def test_train_models_with_fine_tuning(self, mock_trainer_class):
        """Test training with fine-tuning enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock trainer
            mock_trainer = Mock()
            mock_trainer_class.return_value = mock_trainer
            
            # Mock successful training
            mock_trainer.train_image_model.return_value = Mock()
            mock_trainer.train_video_model.return_value = Mock()
            mock_trainer.train_audio_model.return_value = Mock()
            mock_trainer.evaluate_models.return_value = {
                'image_model': {'accuracy': 0.8},
                'video_model': {'accuracy': 0.75},
                'audio_model': {'accuracy': 0.7}
            }
            
            # Mock arguments with fine-tuning enabled
            mock_args = Mock()
            mock_args.data_dir = temp_dir
            mock_args.output_dir = temp_dir
            mock_args.epochs = 2
            mock_args.batch_size = 4
            mock_args.learning_rate = 1e-4
            mock_args.no_fine_tune = False  # Fine-tuning enabled
            mock_args.media_types = ['image', 'video', 'audio']
            mock_args.max_samples = 10
            mock_args.frames_per_video = 5
            
            # This should not raise an exception
            train_models(mock_args)
            
            # Verify trainer was called with fine_tune=True (not no_fine_tune)
            mock_trainer.train_image_model.assert_called_once_with(
                epochs=2, batch_size=4, fine_tune=True, learning_rate=1e-4
            )
    
    @patch('src.training.main.ModelTrainer')
    def test_train_models_without_fine_tuning(self, mock_trainer_class):
        """Test training without fine-tuning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock trainer
            mock_trainer = Mock()
            mock_trainer_class.return_value = mock_trainer
            
            # Mock successful training
            mock_trainer.train_image_model.return_value = Mock()
            mock_trainer.evaluate_models.return_value = {
                'image_model': {'accuracy': 0.8}
            }
            
            # Mock arguments with fine-tuning disabled
            mock_args = Mock()
            mock_args.data_dir = temp_dir
            mock_args.output_dir = temp_dir
            mock_args.epochs = 2
            mock_args.batch_size = 4
            mock_args.learning_rate = 1e-4
            mock_args.no_fine_tune = True  # Fine-tuning disabled
            mock_args.media_types = ['image']
            mock_args.max_samples = 10
            mock_args.frames_per_video = 5
            
            # This should not raise an exception
            train_models(mock_args)
            
            # Verify trainer was called with fine_tune=False
            mock_trainer.train_image_model.assert_called_once_with(
                epochs=2, batch_size=4, fine_tune=False, learning_rate=1e-4
            )


if __name__ == "__main__":
    pytest.main([__file__]) 