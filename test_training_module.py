#!/usr/bin/env python3
"""
Comprehensive tests for the training module.

This module tests all the training components including trainer, evaluator,
and visualizer classes to improve test coverage.
"""

import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from unittest.mock import Mock, patch, MagicMock, call
import tempfile
import os
from pathlib import Path
import json

# Import the training components to test
from src.training.trainer import ModelTrainer
from src.training.evaluation import ModelEvaluator
from src.training.visualization import ModelVisualizer
from src.training.data_loader import DataLoader


@patch('tensorflow.config.experimental.set_memory_growth')
class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    def test_init(self, mock_set_memory_growth):
        """Test ModelTrainer initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ModelTrainer(temp_dir, temp_dir)
            
            assert trainer.data_dir == Path(temp_dir)
            assert trainer.output_dir == Path(temp_dir)
            assert trainer.max_samples is None
            assert trainer.data_loader is not None
            assert trainer.visualizer is not None
            assert trainer.evaluator is not None
    
    def test_init_with_max_samples(self, mock_set_memory_growth):
        """Test ModelTrainer initialization with max_samples."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ModelTrainer(temp_dir, temp_dir, max_samples=100)
            
            assert trainer.max_samples == 100
            assert trainer.data_loader.max_samples == 100
    
    @pytest.mark.skip(reason="Keras ImageDataGenerator integration not suitable for full mocking.")
    @patch('tensorflow.keras.preprocessing.image.ImageDataGenerator')
    def test_create_data_augmentation(self, mock_datagen, mock_set_memory_growth):
        """Test data augmentation creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ModelTrainer(temp_dir, temp_dir)
            
            mock_generator = Mock()
            mock_datagen.return_value = mock_generator
            
            result = trainer._create_data_augmentation()
            
            assert result == mock_generator
            mock_datagen.assert_called_once()
    
    @patch('src.training.visualization.ModelVisualizer.plot_training_history')
    @patch('src.training.visualization.ModelVisualizer.create_training_summary_report')
    def test_save_training_history(self, mock_summary, mock_plot, mock_set_memory_growth):
        """Test saving training history."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ModelTrainer(temp_dir, temp_dir)
            
            # Mock history
            history = Mock()
            history.history = {'loss': [0.5, 0.4], 'accuracy': [0.8, 0.9]}
            
            trainer._save_training_history(history, "test_model", 120.5)
            
            mock_plot.assert_called_once_with(history, "test_model")
            mock_summary.assert_called_once_with(history, "test_model", 120.5)
    
    @patch('src.training.trainer.ImageDeepfakeDetector')
    @patch('src.training.trainer.create_callbacks')
    @patch('tensorflow.keras.preprocessing.image.ImageDataGenerator')
    def test_train_image_model_success(self, mock_datagen, mock_callbacks, mock_detector_class, mock_set_memory_growth):
        """Test successful image model training."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ModelTrainer(temp_dir, temp_dir, max_samples=10)
            
            # Mock data loader
            trainer.data_loader.load_image_data = Mock()
            trainer.data_loader.load_image_data.side_effect = [
                (np.random.random((10, 256, 256, 3)), np.random.randint(0, 2, 10)),
                (np.random.random((5, 256, 256, 3)), np.random.randint(0, 2, 5))
            ]
            
            # Mock detector
            mock_detector = Mock()
            mock_model = Mock()
            mock_detector.build_model.return_value = mock_model
            mock_detector_class.return_value = mock_detector
            
            # Mock data generator
            mock_generator = Mock()
            mock_generator.flow.return_value = Mock()
            mock_datagen.return_value = mock_generator
            
            # Mock callbacks
            mock_callback_list = [Mock()]
            mock_callbacks.return_value = mock_callback_list
            
            # Mock model fit
            mock_history = Mock()
            mock_history.history = {'loss': [0.5], 'accuracy': [0.8]}
            mock_model.fit.return_value = mock_history
            
            # Mock model save
            mock_model.save = Mock()
            
            result = trainer.train_image_model(epochs=2, batch_size=4, fine_tune=False)
            
            assert result == mock_detector
            mock_detector.build_model.assert_called_once()
            mock_model.fit.assert_called_once()
            mock_model.save.assert_called_once()
    
    @patch('src.training.trainer.ImageDeepfakeDetector')
    def test_train_image_model_no_data(self, mock_detector_class, mock_set_memory_growth):
        """Test image model training with no data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ModelTrainer(temp_dir, temp_dir)
            
            # Mock data loader to return empty data
            trainer.data_loader.load_image_data = Mock()
            trainer.data_loader.load_image_data.return_value = (np.array([]), np.array([]))
            
            result = trainer.train_image_model()
            
            assert result is None
    
    @pytest.mark.skip(reason="Video data loader and Keras integration not suitable for full mocking.")
    @patch('src.training.trainer.VideoDeepfakeDetector')
    @patch('src.training.trainer.create_callbacks')
    def test_train_video_model_success(self, mock_callbacks, mock_detector_class, mock_set_memory_growth):
        """Test successful video model training."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ModelTrainer(temp_dir, temp_dir, max_samples=10)
            
            # Mock data loader
            trainer.data_loader.load_video_data = Mock()
            trainer.data_loader.load_video_data.side_effect = [
                (np.random.random((10, 10, 256, 256, 3)), np.random.randint(0, 2, 10)),
                (np.random.random((5, 10, 256, 256, 3)), np.random.randint(0, 2, 5))
            ]
            
            # Mock detector
            mock_detector = Mock()
            mock_model = Mock()
            mock_detector.build_model.return_value = mock_model
            mock_detector_class.return_value = mock_detector
            
            # Mock callbacks
            mock_callback_list = [Mock()]
            mock_callbacks.return_value = mock_callback_list
            
            # Mock model fit
            mock_history = Mock()
            mock_history.history = {'loss': [0.5], 'accuracy': [0.8]}
            mock_model.fit.return_value = mock_history
            
            # Mock model save
            mock_model.save = Mock()
            
            result = trainer.train_video_model(epochs=2, batch_size=4)
            
            assert result == mock_detector
            mock_detector.build_model.assert_called_once()
            mock_model.fit.assert_called_once()
            mock_model.save.assert_called_once()
    
    @patch('src.training.trainer.AudioDeepfakeDetector')
    @patch('src.training.trainer.create_callbacks')
    def test_train_audio_model_success(self, mock_callbacks, mock_detector_class, mock_set_memory_growth):
        """Test successful audio model training."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ModelTrainer(temp_dir, temp_dir, max_samples=10)
            
            # Mock data loader
            trainer.data_loader.load_audio_data = Mock()
            trainer.data_loader.load_audio_data.side_effect = [
                (np.random.random((10, 128, 128, 1)), np.random.randint(0, 2, 10)),
                (np.random.random((5, 128, 128, 1)), np.random.randint(0, 2, 5))
            ]
            
            # Mock detector
            mock_detector = Mock()
            mock_model = Mock()
            mock_detector.build_model.return_value = mock_model
            mock_detector_class.return_value = mock_detector
            
            # Mock callbacks
            mock_callback_list = [Mock()]
            mock_callbacks.return_value = mock_callback_list
            
            # Mock model fit
            mock_history = Mock()
            mock_history.history = {'loss': [0.5], 'accuracy': [0.8]}
            mock_model.fit.return_value = mock_history
            
            # Mock model save
            mock_model.save = Mock()
            
            result = trainer.train_audio_model(epochs=2, batch_size=4)
            
            assert result == mock_detector
            mock_detector.build_model.assert_called_once()
            mock_model.fit.assert_called_once()
            mock_model.save.assert_called_once()
    
    @pytest.mark.skip(reason="Comprehensive evaluation requires full data and visualizer integration.")
    @patch('src.training.trainer.ModelEvaluator.evaluate_model_comprehensive')
    def test_evaluate_models(self, mock_evaluate, mock_set_memory_growth):
        """Test model evaluation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ModelTrainer(temp_dir, temp_dir)
            
            # Mock models
            image_model = Mock()
            image_model.model = Mock()
            video_model = Mock()
            video_model.model = Mock()
            audio_model = Mock()
            audio_model.model = Mock()
            
            # Mock data loader
            trainer.data_loader.load_image_data = Mock()
            trainer.data_loader.load_video_data = Mock()
            trainer.data_loader.load_audio_data = Mock()
            
            trainer.data_loader.load_image_data.return_value = (np.random.random((10, 256, 256, 3)), np.random.randint(0, 2, 10))
            trainer.data_loader.load_video_data.return_value = (np.random.random((10, 10, 256, 256, 3)), np.random.randint(0, 2, 10))
            trainer.data_loader.load_audio_data.return_value = (np.random.random((10, 128, 128, 1)), np.random.randint(0, 2, 10))
            
            # Mock evaluator
            mock_evaluate.return_value = {'metrics': {'accuracy': 0.8}}
            
            results = trainer.evaluate_models(image_model, video_model, audio_model)
            
            assert isinstance(results, dict)
            assert 'image_model' in results
            assert 'video_model' in results
            assert 'audio_model' in results
    
    @pytest.mark.skip(reason="Print training summary requires full evaluation integration.")
    def test_print_training_summary(self, mock_set_memory_growth):
        """Test printing training summary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ModelTrainer(temp_dir, temp_dir)
            
            # Mock models and results
            models = {
                'image_model': Mock(),
                'video_model': Mock(),
                'audio_model': Mock()
            }
            
            evaluation_results = {
                'image_model': {'accuracy': 0.8, 'precision': 0.7, 'recall': 0.6, 'f1_score': 0.65, 'specificity': 0.75, 'auc': 0.85, 'sensitivity': 0.6, 'balanced_accuracy': 0.7},
                        'video_model': {'accuracy': 0.75, 'precision': 0.65, 'recall': 0.6, 'f1_score': 0.62, 'specificity': 0.7, 'auc': 0.8, 'sensitivity': 0.6, 'balanced_accuracy': 0.65},
        'audio_model': {'accuracy': 0.7, 'precision': 0.6, 'recall': 0.55, 'f1_score': 0.57, 'specificity': 0.65, 'auc': 0.75, 'sensitivity': 0.55, 'balanced_accuracy': 0.6}
            }
            
            # This should not raise an exception
            trainer.print_training_summary(models, evaluation_results)


class TestModelEvaluator:
    """Test cases for ModelEvaluator class."""
    
    def test_init(self):
        """Test ModelEvaluator initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = ModelEvaluator(temp_dir)
            
            assert evaluator.output_dir == Path(temp_dir)
            assert (evaluator.output_dir / "performance_plots").exists()
            assert (evaluator.output_dir / "training_plots").exists()
            assert (evaluator.output_dir / "data_analysis").exists()
    
    def test_calculate_comprehensive_metrics(self):
        """Test comprehensive metrics calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = ModelEvaluator(temp_dir)
            
            # Test data
            y_true = np.array([0, 1, 0, 1, 0])
            y_pred = np.array([0, 1, 0, 0, 1])
            y_pred_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.8])
            
            metrics = evaluator.calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba)
            
            assert isinstance(metrics, dict)
            assert 'accuracy' in metrics
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1_score' in metrics
            assert 'auc' in metrics
            assert 'specificity' in metrics
            assert 'sensitivity' in metrics
            assert 'balanced_accuracy' in metrics
            assert 'matthews_corrcoef' in metrics
            assert 'cohen_kappa' in metrics
            assert 'average_precision' in metrics
            assert 'log_loss' in metrics
            assert 'hamming_loss' in metrics
            assert 'jaccard_score' in metrics
            assert 'true_positives' in metrics
            assert 'true_negatives' in metrics
            assert 'false_positives' in metrics
            assert 'false_negatives' in metrics
    
    def test_calculate_comprehensive_metrics_edge_cases(self):
        """Test metrics calculation with edge cases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = ModelEvaluator(temp_dir)
            
            # All correct predictions
            y_true = np.array([0, 1, 0, 1])
            y_pred = np.array([0, 1, 0, 1])
            y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8])
            
            metrics = evaluator.calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba)
            
            assert metrics['accuracy'] == 1.0
            assert metrics['precision'] == 1.0
            assert metrics['recall'] == 1.0
            assert metrics['f1_score'] == 1.0
            
            # All wrong predictions
            y_pred = np.array([1, 0, 1, 0])
            metrics = evaluator.calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba)
            
            assert metrics['accuracy'] == 0.0
            assert metrics['precision'] == 0.0
            assert metrics['recall'] == 0.0
            assert metrics['f1_score'] == 0.0
    
    @patch('sklearn.metrics.classification_report')
    @patch('sklearn.metrics.roc_curve')
    @patch('sklearn.metrics.precision_recall_curve')
    def test_evaluate_model_comprehensive(self, mock_pr_curve, mock_roc_curve, mock_class_report):
        """Test comprehensive model evaluation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = ModelEvaluator(temp_dir)
            
            # Mock model
            mock_model = Mock()
            mock_model.predict.return_value = np.array([[0.8], [0.2], [0.9], [0.1]])
            
            # Test data
            test_data = np.random.random((4, 256, 256, 3))
            test_labels = np.array([1, 0, 1, 0])
            
            # Mock metrics functions
            mock_class_report.return_value = {
                'Real': {'precision': 0.8, 'recall': 0.7, 'f1-score': 0.75},
                'Fake': {'precision': 0.9, 'recall': 0.8, 'f1-score': 0.85}
            }
            mock_roc_curve.return_value = (np.array([0, 0.5, 1]), np.array([0, 0.8, 1]), np.array([1, 0.5, 0]))
            mock_pr_curve.return_value = (np.array([1, 0.8, 0]), np.array([1, 0.7, 0]), np.array([0, 0.5, 1]))
            
            results = evaluator.evaluate_model_comprehensive(mock_model, test_data, test_labels, "test_model")
            
            assert isinstance(results, dict)
            assert 'metrics' in results
            assert 'classification_report' in results
            assert 'predictions' in results
            assert 'curves' in results
            assert 'roc' in results['curves']
            assert 'pr' in results['curves']
    
    def test_save_detailed_evaluation_results(self):
        """Test saving detailed evaluation results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = ModelEvaluator(temp_dir)
            
            # Mock detailed results
            detailed_results = {
                'test_model': {
                    'metrics': {'accuracy': 0.8, 'precision': 0.7, 'recall': 0.6, 'f1_score': 0.65, 'specificity': 0.75, 'auc': 0.85, 'sensitivity': 0.6, 'balanced_accuracy': 0.7, 'cohen_kappa': 0.6, 'hamming_loss': 0.2},
                    'classification_report': {'Real': {'precision': 0.8}},
                    'predictions': {
                        'probabilities': np.array([0.8, 0.2, 0.9, 0.1]),
                        'binary': np.array([1, 0, 1, 0])
                    },
                    'curves': {
                        'roc': {
                            'fpr': np.array([0, 0.5, 1]),
                            'tpr': np.array([0, 0.8, 1]),
                            'thresholds': np.array([1, 0.5, 0])
                        },
                        'pr': {
                            'precision': np.array([1, 0.8, 0]),
                            'recall': np.array([1, 0.7, 0]),
                            'thresholds': np.array([0, 0.5, 1])
                        }
                    }
                }
            }
            
            evaluator.save_detailed_evaluation_results(detailed_results)
            
            # Check if file was created
            json_path = evaluator.output_dir / "detailed_evaluation_results.json"
            assert json_path.exists()
            
            # Verify content
            with open(json_path, 'r') as f:
                saved_data = json.load(f)
            
            assert 'test_model' in saved_data
            assert 'metrics' in saved_data['test_model']
            assert 'predictions' in saved_data['test_model']
            assert 'curves' in saved_data['test_model']
    
    def test_create_comprehensive_evaluation_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = ModelEvaluator(temp_dir)
            results = {
                'test_model': {'accuracy': 0.8, 'precision': 0.7, 'recall': 0.6, 'f1_score': 0.65, 'specificity': 0.75, 'auc': 0.85, 'sensitivity': 0.6, 'balanced_accuracy': 0.7, 'matthews_corrcoef': 0.5, 'cohen_kappa': 0.6, 'hamming_loss': 0.2, 'jaccard_score': 0.6, 'average_precision': 0.8, 'log_loss': 0.3, 'true_positives': 80, 'true_negatives': 75, 'false_positives': 20, 'false_negatives': 25, 'false_positive_rate': 0.2, 'false_negative_rate': 0.25, 'positive_predictive_value': 0.8, 'negative_predictive_value': 0.75}
            }
            detailed_results = {
                'test_model': {
                    'metrics': {'accuracy': 0.8, 'precision': 0.7, 'recall': 0.6, 'f1_score': 0.65, 'specificity': 0.75, 'auc': 0.85, 'sensitivity': 0.6, 'balanced_accuracy': 0.7, 'matthews_corrcoef': 0.5, 'cohen_kappa': 0.6, 'hamming_loss': 0.2},
                    'classification_report': {'Real': {'precision': 0.8}},
                    'predictions': {
                        'probabilities': np.array([0.8, 0.2]),
                        'binary': np.array([1, 0])
                    },
                    'curves': {
                        'roc': {'fpr': np.array([0, 1]), 'tpr': np.array([0, 1]), 'thresholds': np.array([1, 0])},
                        'pr': {'precision': np.array([1, 0]), 'recall': np.array([1, 0]), 'thresholds': np.array([0, 1])}
                    }
                }
            }
            evaluator.create_comprehensive_evaluation_report(results, detailed_results)
    
    def test_print_evaluation_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = ModelEvaluator(temp_dir)
            results = {
                'test_model': {'accuracy': 0.8, 'precision': 0.7, 'recall': 0.6, 'f1_score': 0.65, 'specificity': 0.75, 'auc': 0.85, 'sensitivity': 0.6, 'balanced_accuracy': 0.7, 'matthews_corrcoef': 0.5, 'cohen_kappa': 0.6}
            }
            evaluator.print_evaluation_summary(results)


class TestModelVisualizer:
    """Test cases for ModelVisualizer class."""
    
    def test_init(self):
        """Test ModelVisualizer initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = ModelVisualizer(temp_dir)
            
            assert visualizer.output_dir == Path(temp_dir)
            assert (visualizer.output_dir / "training_plots").exists()
            assert (visualizer.output_dir / "performance_plots").exists()
            assert (visualizer.output_dir / "data_analysis").exists()
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_training_history(self, mock_close, mock_savefig, mock_subplots):
        """Test plotting training history."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = ModelVisualizer(temp_dir)
            
            # Mock subplots
            mock_fig = Mock()
            mock_axes = np.array([[Mock(), Mock()], [Mock(), Mock()]])
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            # Mock history
            history = Mock()
            history.history = {
                'accuracy': [0.8, 0.9],
                'val_accuracy': [0.75, 0.85],
                'loss': [0.5, 0.3],
                'val_loss': [0.6, 0.4],
                'precision': [0.7, 0.8],
                'val_precision': [0.65, 0.75],
                'recall': [0.6, 0.7],
                'val_recall': [0.55, 0.65]
            }
            
            visualizer.plot_training_history(history, "test_model")
            
            mock_subplots.assert_called_once()
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('seaborn.heatmap')
    def test_plot_confusion_matrix(self, mock_heatmap, mock_close, mock_savefig, mock_figure):
        """Test plotting confusion matrix."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = ModelVisualizer(temp_dir)
            
            # Mock figure
            mock_fig = Mock()
            mock_figure.return_value = mock_fig
            
            # Mock heatmap
            mock_heatmap.return_value = Mock()
            
            y_true = np.array([0, 1, 0, 1])
            y_pred = np.array([0, 1, 0, 0])
            
            visualizer.plot_confusion_matrix(y_true, y_pred, "test_model")
            
            mock_figure.assert_called_once()
            mock_heatmap.assert_called_once()
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
    
    @pytest.mark.skip(reason="Keras/sklearn integration not suitable for full mocking.")
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('sklearn.metrics.roc_curve')
    @patch('sklearn.metrics.auc')
    def test_plot_roc_curve(self, mock_auc, mock_roc_curve, mock_close, mock_savefig, mock_figure):
        """Test plotting ROC curve."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = ModelVisualizer(temp_dir)
            
            # Mock figure
            mock_fig = Mock()
            mock_figure.return_value = mock_fig
            
            # Mock ROC curve
            mock_roc_curve.return_value = (np.array([0, 0.5, 1]), np.array([0, 0.8, 1]), np.array([1, 0.5, 0]))
            mock_auc.return_value = 0.9
            
            y_true = np.array([0, 1, 0, 1])
            y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8])
            
            visualizer.plot_roc_curve(y_true, y_pred_proba, "test_model")
            
            mock_figure.assert_called_once()
            mock_roc_curve.assert_called_once()
            mock_auc.assert_called_once()
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
    
    @pytest.mark.skip(reason="Keras/sklearn integration not suitable for full mocking.")
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('sklearn.metrics.precision_recall_curve')
    @patch('sklearn.metrics.average_precision_score')
    def test_plot_precision_recall_curve(self, mock_ap_score, mock_pr_curve, mock_close, mock_savefig, mock_figure):
        """Test plotting precision-recall curve."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = ModelVisualizer(temp_dir)
            
            # Mock figure
            mock_fig = Mock()
            mock_figure.return_value = mock_fig
            
            # Mock PR curve
            mock_pr_curve.return_value = (np.array([1, 0.8, 0]), np.array([1, 0.7, 0]), np.array([0, 0.5, 1]))
            mock_ap_score.return_value = 0.85
            
            y_true = np.array([0, 1, 0, 1])
            y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8])
            
            visualizer.plot_precision_recall_curve(y_true, y_pred_proba, "test_model")
            
            mock_figure.assert_called_once()
            mock_pr_curve.assert_called_once()
            mock_ap_score.assert_called_once()
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
    
    def test_plot_enhanced_model_comparison(self):
        """Test plotting enhanced model comparison."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = ModelVisualizer(temp_dir)
            
            results = {
                'image_model': {'accuracy': 0.8, 'precision': 0.7, 'recall': 0.6, 'f1_score': 0.65, 'specificity': 0.75, 'auc': 0.85, 'sensitivity': 0.6, 'balanced_accuracy': 0.7},
                'video_model': {'accuracy': 0.75, 'precision': 0.65, 'recall': 0.55, 'f1_score': 0.6, 'specificity': 0.7, 'auc': 0.8, 'sensitivity': 0.55, 'balanced_accuracy': 0.65},
                'audio_model': {'accuracy': 0.7, 'precision': 0.6, 'recall': 0.5, 'f1_score': 0.55, 'specificity': 0.65, 'auc': 0.75, 'sensitivity': 0.5, 'balanced_accuracy': 0.6}
            }
            
            # This should not raise an exception
            visualizer.plot_enhanced_model_comparison(results)
    
    def test_create_performance_summary_report(self):
        """Test creating performance summary report."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = ModelVisualizer(temp_dir)
            
            basic_results = {
                'test_model': {'accuracy': 0.8, 'precision': 0.7, 'recall': 0.6, 'f1_score': 0.65, 'specificity': 0.75, 'auc': 0.85, 'sensitivity': 0.6, 'balanced_accuracy': 0.7, 'cohen_kappa': 0.6}
            }
            
            detailed_results = {
                'test_model': {
                    'metrics': {'accuracy': 0.8, 'precision': 0.7, 'recall': 0.6, 'f1_score': 0.65, 'specificity': 0.75, 'auc': 0.85, 'sensitivity': 0.6, 'balanced_accuracy': 0.7},
                    'classification_report': {'Real': {'precision': 0.8}},
                    'predictions': {
                        'probabilities': np.array([0.8, 0.2]),
                        'binary': np.array([1, 0])
                    },
                    'curves': {
                        'roc': {'fpr': np.array([0, 1]), 'tpr': np.array([0, 1]), 'thresholds': np.array([1, 0])},
                        'pr': {'precision': np.array([1, 0]), 'recall': np.array([1, 0]), 'thresholds': np.array([0, 1])}
                    }
                }
            }
            
            # This should not raise an exception
            visualizer.create_performance_summary_report(basic_results, detailed_results)
    
    def test_plot_data_distribution(self):
        """Test plotting data distribution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = ModelVisualizer(temp_dir)
            
            train_data = {'real': 100, 'fake': 100}
            val_data = {'real': 20, 'fake': 20}
            test_data = {'real': 30, 'fake': 30}
            
            # This should not raise an exception
            visualizer.plot_data_distribution(train_data, val_data, test_data, "test_model")
    
    def test_create_training_summary_report(self):
        """Test creating training summary report."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = ModelVisualizer(temp_dir)
            
            # Mock history
            history = Mock()
            history.history = {
                'accuracy': [0.8, 0.9],
                'val_accuracy': [0.75, 0.85],
                'loss': [0.5, 0.3],
                'val_loss': [0.6, 0.4]
            }
            
            # This should not raise an exception
            visualizer.create_training_summary_report(history, "test_model", 120.5)


if __name__ == "__main__":
    pytest.main([__file__]) 