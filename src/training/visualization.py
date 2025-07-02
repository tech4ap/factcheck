"""
Visualization Module for Deepfake Detection

This module provides comprehensive visualization capabilities for model training,
evaluation, and performance analysis with publication-ready plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve, 
    average_precision_score, roc_auc_score, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score, hamming_loss, jaccard_score, log_loss
)

logger = logging.getLogger(__name__)


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
        
        # Configure matplotlib for better plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
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
        
        # Extract metrics for each class
        classes = ['Real', 'Fake']
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for cls in classes:
            if cls.lower() in class_report:
                precision_scores.append(class_report[cls.lower()]['precision'])
                recall_scores.append(class_report[cls.lower()]['recall'])
                f1_scores.append(class_report[cls.lower()]['f1-score'])
        
        # Plot metrics by class
        x = np.arange(len(classes))
        width = 0.25
        
        axes[0].bar(x - width, precision_scores, width, label='Precision', color='skyblue')
        axes[0].bar(x, recall_scores, width, label='Recall', color='lightcoral')
        axes[0].bar(x + width, f1_scores, width, label='F1-Score', color='lightgreen')
        
        axes[0].set_xlabel('Classes')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Metrics by Class')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(classes)
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