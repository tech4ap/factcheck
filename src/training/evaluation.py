"""
Evaluation Module for Deepfake Detection

This module provides comprehensive model evaluation capabilities including
multiple metrics calculation, visualization generation, and performance analysis.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import time

from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, cohen_kappa_score, hamming_loss,
    jaccard_score, log_loss, roc_auc_score
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluator for deepfake detection models.
    
    This class provides evaluation capabilities for different media types
    with extensive metrics calculation and analysis.
    """
    
    def __init__(self, output_dir: str = "models"):
        """
        Initialize the model evaluator.
        
        Args:
            output_dir (str): Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "performance_plots").mkdir(exist_ok=True)
        (self.output_dir / "training_plots").mkdir(exist_ok=True)
        (self.output_dir / "data_analysis").mkdir(exist_ok=True)
        
        logger.info(f"Initialized ModelEvaluator with output directory: {self.output_dir}")
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
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
    
    def evaluate_model_comprehensive(self, model, test_data: np.ndarray, 
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
        metrics = self.calculate_comprehensive_metrics(test_labels, y_pred, y_pred_proba)
        
        # Generate classification report
        class_report = classification_report(test_labels, y_pred, 
                                           target_names=['Real', 'Fake'], 
                                           output_dict=True)
        
        # Calculate ROC and PR curves
        fpr, tpr, roc_thresholds = roc_curve(test_labels, y_pred_proba)
        precision, recall, pr_thresholds = precision_recall_curve(test_labels, y_pred_proba)
        
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
    
    def save_detailed_evaluation_results(self, detailed_results: Dict[str, Dict[str, Any]]) -> None:
        """
        Save detailed evaluation results including classification reports and curves.
        
        Args:
            detailed_results: Detailed evaluation results for each model
        """
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
    
    def print_evaluation_summary(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        Print a formatted evaluation summary to console.
        
        Args:
            results: Evaluation results dictionary
        """
        if not results:
            logger.warning("No evaluation results to display")
            return
        
        logger.info("=" * 50)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 50)
        
        for model_type, metrics in results.items():
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
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        logger.info(f"\n🏆 BEST PERFORMING MODEL: {best_model[0].upper()}")
        logger.info(f"   Accuracy: {best_model[1]['accuracy']:.4f}")
        logger.info(f"   F1-Score: {best_model[1]['f1_score']:.4f}")
        logger.info(f"   AUC: {best_model[1]['auc']:.4f}")
        
        logger.info("\n" + "=" * 50)
        logger.info("EVALUATION COMPLETED!")
        logger.info("=" * 50)
        logger.info("📊 Check the following files for detailed results:")
        logger.info(f"   - {self.output_dir}/model_evaluation_results.csv")
        logger.info(f"   - {self.output_dir}/detailed_evaluation_results.json")
        logger.info(f"   - {self.output_dir}/comprehensive_evaluation_report.txt")
        logger.info(f"   - {self.output_dir}/performance_plots/ (for visualizations)")
        logger.info("=" * 50) 