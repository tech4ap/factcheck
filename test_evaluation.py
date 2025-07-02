#!/usr/bin/env python3
"""
Test script for enhanced model evaluation metrics.

This script demonstrates the comprehensive evaluation capabilities
by creating sample predictions and running the evaluation pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.append('src')

from training.train_deepfake_detector import ModelTrainer, ModelVisualizer
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, cohen_kappa_score, hamming_loss,
    jaccard_score, log_loss, roc_auc_score, precision_score, recall_score, f1_score
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(n_samples=1000, noise_level=0.2):
    """
    Create sample data for testing evaluation metrics.
    
    Args:
        n_samples: Number of samples to create
        noise_level: Level of noise to add to predictions
        
    Returns:
        Tuple of (y_true, y_pred_proba, y_pred)
    """
    # Create true labels (60% fake, 40% real)
    np.random.seed(42)
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    
    # Create realistic predictions with some noise
    y_pred_proba = np.zeros(n_samples)
    
    for i in range(n_samples):
        if y_true[i] == 1:  # Fake
            # Most fake samples should have high probability
            base_prob = np.random.beta(8, 2)  # Skewed towards high values
        else:  # Real
            # Most real samples should have low probability
            base_prob = np.random.beta(2, 8)  # Skewed towards low values
        
        # Add noise
        noise = np.random.normal(0, noise_level)
        y_pred_proba[i] = np.clip(base_prob + noise, 0, 1)
    
    # Create binary predictions
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    return y_true, y_pred_proba, y_pred

def test_comprehensive_metrics():
    """Test the comprehensive metrics calculation."""
    logger.info("Testing comprehensive metrics calculation...")
    
    # Create sample data
    y_true, y_pred_proba, y_pred = create_sample_data(n_samples=1000, noise_level=0.15)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate comprehensive metrics manually
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    matthews_corr = matthews_corrcoef(y_true, y_pred)
    cohen_kappa = cohen_kappa_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    log_loss_score = log_loss(y_true, y_pred_proba)
    hamming_loss_score = hamming_loss(y_true, y_pred)
    jaccard_score_val = jaccard_score(y_true, y_pred)
    
    # Print results
    logger.info("Sample Metrics Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1_score:.4f}")
    logger.info(f"  Specificity: {specificity:.4f}")
    logger.info(f"  Sensitivity: {sensitivity:.4f}")
    logger.info(f"  Balanced Accuracy: {balanced_accuracy:.4f}")
    logger.info(f"  Matthews Correlation: {matthews_corr:.4f}")
    logger.info(f"  Cohen's Kappa: {cohen_kappa:.4f}")
    logger.info(f"  AUC: {auc_score:.4f}")
    logger.info(f"  Average Precision: {avg_precision:.4f}")
    logger.info(f"  Log Loss: {log_loss_score:.4f}")
    logger.info(f"  Hamming Loss: {hamming_loss_score:.4f}")
    logger.info(f"  Jaccard Score: {jaccard_score_val:.4f}")
    
    return y_true, y_pred_proba, y_pred

def test_visualization():
    """Test the visualization capabilities."""
    logger.info("Testing visualization capabilities...")
    
    # Create output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create visualizer
    visualizer = ModelVisualizer(str(output_dir))
    
    # Create sample data
    y_true, y_pred_proba, y_pred = create_sample_data(n_samples=1000, noise_level=0.15)
    
    # Test different visualizations
    logger.info("Creating confusion matrix...")
    visualizer.plot_confusion_matrix(y_true, y_pred, "test_model")
    
    logger.info("Creating ROC curve...")
    visualizer.plot_roc_curve(y_true, y_pred_proba, "test_model")
    
    logger.info("Creating precision-recall curve...")
    visualizer.plot_precision_recall_curve(y_true, y_pred_proba, "test_model")
    
    logger.info("Creating detailed metrics visualization...")
    visualizer.plot_detailed_metrics(y_true, y_pred, y_pred_proba, "test_model")
    
    # Create classification report
    class_report = classification_report(y_true, y_pred, target_names=['Real', 'Fake'], output_dict=True)
    logger.info("Creating classification report visualization...")
    visualizer.plot_classification_report(class_report, "test_model")
    
    logger.info(f"Visualizations saved to {output_dir}")

def test_model_comparison():
    """Test the model comparison capabilities."""
    logger.info("Testing model comparison capabilities...")
    
    # Create output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create visualizer
    visualizer = ModelVisualizer(str(output_dir))
    
    # Create sample results for multiple models
    results = {}
    
    # Model 1: Good performance
    y_true1, y_pred_proba1, y_pred1 = create_sample_data(n_samples=1000, noise_level=0.1)
    tn, fp, fn, tp = confusion_matrix(y_true1, y_pred1).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    true_negative_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
    positive_predictive_value = tp / (tp + fp) if (tp + fp) > 0 else 0
    negative_predictive_value = tn / (tn + fn) if (tn + fn) > 0 else 0
    false_discovery_rate = fp / (tp + fp) if (tp + fp) > 0 else 0
    false_omission_rate = fn / (tn + fn) if (tn + fn) > 0 else 0
    results['image'] = {
        'accuracy': balanced_accuracy_score(y_true1, y_pred1),
        'precision': precision_score(y_true1, y_pred1),
        'recall': recall_score(y_true1, y_pred1),
        'f1_score': f1_score(y_true1, y_pred1),
        'auc': roc_auc_score(y_true1, y_pred_proba1),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'balanced_accuracy': balanced_accuracy_score(y_true1, y_pred1),
        'matthews_corrcoef': matthews_corrcoef(y_true1, y_pred1),
        'cohen_kappa': cohen_kappa_score(y_true1, y_pred1),
        'hamming_loss': hamming_loss(y_true1, y_pred1),
        'jaccard_score': jaccard_score(y_true1, y_pred1),
        'average_precision': average_precision_score(y_true1, y_pred_proba1),
        'log_loss': log_loss(y_true1, y_pred_proba1),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
        'true_positive_rate': true_positive_rate,
        'true_negative_rate': true_negative_rate,
        'positive_predictive_value': positive_predictive_value,
        'negative_predictive_value': negative_predictive_value,
        'false_discovery_rate': false_discovery_rate,
        'false_omission_rate': false_omission_rate
    }
    
    # Model 2: Moderate performance
    y_true2, y_pred_proba2, y_pred2 = create_sample_data(n_samples=1000, noise_level=0.3)
    tn, fp, fn, tp = confusion_matrix(y_true2, y_pred2).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    true_negative_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
    positive_predictive_value = tp / (tp + fp) if (tp + fp) > 0 else 0
    negative_predictive_value = tn / (tn + fn) if (tn + fn) > 0 else 0
    false_discovery_rate = fp / (tp + fp) if (tp + fp) > 0 else 0
    false_omission_rate = fn / (tn + fn) if (tn + fn) > 0 else 0
    results['video'] = {
        'accuracy': balanced_accuracy_score(y_true2, y_pred2),
        'precision': precision_score(y_true2, y_pred2),
        'recall': recall_score(y_true2, y_pred2),
        'f1_score': f1_score(y_true2, y_pred2),
        'auc': roc_auc_score(y_true2, y_pred_proba2),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'balanced_accuracy': balanced_accuracy_score(y_true2, y_pred2),
        'matthews_corrcoef': matthews_corrcoef(y_true2, y_pred2),
        'cohen_kappa': cohen_kappa_score(y_true2, y_pred2),
        'hamming_loss': hamming_loss(y_true2, y_pred2),
        'jaccard_score': jaccard_score(y_true2, y_pred2),
        'average_precision': average_precision_score(y_true2, y_pred_proba2),
        'log_loss': log_loss(y_true2, y_pred_proba2),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
        'true_positive_rate': true_positive_rate,
        'true_negative_rate': true_negative_rate,
        'positive_predictive_value': positive_predictive_value,
        'negative_predictive_value': negative_predictive_value,
        'false_discovery_rate': false_discovery_rate,
        'false_omission_rate': false_omission_rate
    }
    
    # Model 3: Poor performance
    y_true3, y_pred_proba3, y_pred3 = create_sample_data(n_samples=1000, noise_level=0.5)
    tn, fp, fn, tp = confusion_matrix(y_true3, y_pred3).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    true_negative_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
    positive_predictive_value = tp / (tp + fp) if (tp + fp) > 0 else 0
    negative_predictive_value = tn / (tn + fn) if (tn + fn) > 0 else 0
    false_discovery_rate = fp / (tp + fp) if (tp + fp) > 0 else 0
    false_omission_rate = fn / (tn + fn) if (tn + fn) > 0 else 0
    results['audio'] = {
        'accuracy': balanced_accuracy_score(y_true3, y_pred3),
        'precision': precision_score(y_true3, y_pred3),
        'recall': recall_score(y_true3, y_pred3),
        'f1_score': f1_score(y_true3, y_pred3),
        'auc': roc_auc_score(y_true3, y_pred_proba3),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'balanced_accuracy': balanced_accuracy_score(y_true3, y_pred3),
        'matthews_corrcoef': matthews_corrcoef(y_true3, y_pred3),
        'cohen_kappa': cohen_kappa_score(y_true3, y_pred3),
        'hamming_loss': hamming_loss(y_true3, y_pred3),
        'jaccard_score': jaccard_score(y_true3, y_pred3),
        'average_precision': average_precision_score(y_true3, y_pred_proba3),
        'log_loss': log_loss(y_true3, y_pred_proba3),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
        'true_positive_rate': true_positive_rate,
        'true_negative_rate': true_negative_rate,
        'positive_predictive_value': positive_predictive_value,
        'negative_predictive_value': negative_predictive_value,
        'false_discovery_rate': false_discovery_rate,
        'false_omission_rate': false_omission_rate
    }
    
    # Create comparison visualizations
    logger.info("Creating enhanced model comparison...")
    visualizer.plot_enhanced_model_comparison(results)
    
    logger.info("Creating performance summary report...")
    visualizer.create_performance_summary_report(results, {})
    
    logger.info("Creating comprehensive evaluation report...")
    visualizer.create_comprehensive_evaluation_report(results, {})
    
    logger.info(f"Model comparison results saved to {output_dir}")

def main():
    """Main test function."""
    logger.info("Starting comprehensive evaluation metrics test...")
    
    try:
        # Test comprehensive metrics
        test_comprehensive_metrics()
        
        # Test visualizations
        test_visualization()
        
        # Test model comparison
        test_model_comparison()
        
        logger.info("✅ All tests completed successfully!")
        logger.info("📊 Check the 'test_output' directory for generated visualizations and reports.")
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        raise

if __name__ == "__main__":
    main() 