"""
Model Saver Module

This module provides functionality to save trained deepfake detection models
with proper metadata, versioning, and organization.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)

class ModelSaver:
    """Class for saving and managing trained models."""
    
    def __init__(self, output_dir: str = "models"):
        """
        Initialize ModelSaver.
        
        Args:
            output_dir: Directory to save models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "final_models").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
    def save_model(self, model, model_name: str, model_type: str, 
                  training_history: Optional[Dict] = None,
                  evaluation_results: Optional[Dict] = None,
                  config: Optional[Dict] = None,
                  is_best: bool = False) -> str:
        """
        Save a model with metadata.
        
        Args:
            model: Trained Keras model
            model_name: Name of the model
            model_type: Type of model ('image', 'video', 'audio', 'ensemble')
            training_history: Training history dictionary
            evaluation_results: Evaluation results
            config: Model configuration
            is_best: Whether this is the best performing model
            
        Returns:
            Path to saved model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine save path
        if is_best:
            model_filename = f"{model_type}_model_best.h5"
        else:
            model_filename = f"{model_type}_model_final.h5"
        
        model_path = self.output_dir / model_filename
        
        # Save the model
        try:
            model.save(model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
        
        # Save metadata
        metadata = {
            "model_name": model_name,
            "model_type": model_type,
            "timestamp": timestamp,
            "model_path": str(model_path),
            "is_best": is_best,
            "tensorflow_version": tf.__version__,
            "model_architecture": self._get_model_summary(model),
            "total_params": model.count_params() if hasattr(model, 'count_params') else None
        }
        
        if training_history:
            metadata["training_history"] = training_history
            metadata["final_epoch"] = len(training_history.get('loss', []))
            metadata["best_val_accuracy"] = max(training_history.get('val_accuracy', [0]))
        
        if evaluation_results:
            metadata["evaluation_results"] = evaluation_results
        
        if config:
            metadata["config"] = config
        
        # Save metadata
        metadata_path = self.output_dir / "metadata" / f"{model_type}_model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model metadata saved to {metadata_path}")
        
        # Also save a versioned copy
        versioned_path = self.output_dir / "checkpoints" / f"{model_type}_model_{timestamp}.h5"
        model.save(versioned_path)
        
        return str(model_path)
    
    def save_ensemble_model(self, ensemble_model, individual_models: Dict[str, Any],
                          model_name: str = "ensemble_deepfake_detector",
                          evaluation_results: Optional[Dict] = None) -> str:
        """
        Save ensemble model with individual component information.
        
        Args:
            ensemble_model: Trained ensemble model
            individual_models: Dictionary of individual models
            model_name: Name of the ensemble model
            evaluation_results: Ensemble evaluation results
            
        Returns:
            Path to saved ensemble model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main ensemble model
        ensemble_path = self.output_dir / "ensemble_model_final.h5"
        ensemble_model.save(ensemble_path)
        
        # Save individual models if provided
        for model_type, model in individual_models.items():
            if model is not None:
                individual_path = self.output_dir / f"{model_type}_model_final.h5"
                model.model.save(individual_path)
        
        # Save ensemble metadata
        metadata = {
            "model_name": model_name,
            "model_type": "ensemble",
            "timestamp": timestamp,
            "ensemble_path": str(ensemble_path),
            "individual_models": {k: v is not None for k, v in individual_models.items()},
            "tensorflow_version": tf.__version__
        }
        
        if evaluation_results:
            metadata["evaluation_results"] = evaluation_results
        
        metadata_path = self.output_dir / "metadata" / "ensemble_model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Ensemble model and metadata saved")
        return str(ensemble_path)
    
    def load_model_metadata(self, model_type: str) -> Optional[Dict]:
        """Load model metadata."""
        metadata_path = self.output_dir / "metadata" / f"{model_type}_model_metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
    
    def list_available_models(self) -> Dict[str, Dict]:
        """List all available models with their metadata."""
        models = {}
        metadata_dir = self.output_dir / "metadata"
        
        if metadata_dir.exists():
            for metadata_file in metadata_dir.glob("*_metadata.json"):
                model_type = metadata_file.stem.replace("_model_metadata", "")
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    models[model_type] = metadata
                except Exception as e:
                    logger.warning(f"Could not load metadata for {model_type}: {e}")
        
        return models
    
    def get_best_model_path(self, model_type: str) -> Optional[str]:
        """Get path to the best performing model of a given type."""
        best_path = self.output_dir / f"{model_type}_model_best.h5"
        final_path = self.output_dir / f"{model_type}_model_final.h5"
        
        if best_path.exists():
            return str(best_path)
        elif final_path.exists():
            return str(final_path)
        return None
    
    def _get_model_summary(self, model) -> str:
        """Get a string representation of the model architecture."""
        try:
            import io
            import contextlib
            
            summary_string = io.StringIO()
            with contextlib.redirect_stdout(summary_string):
                model.summary()
            return summary_string.getvalue()
        except Exception:
            return "Model summary not available"
    
    def save_training_config(self, config: Dict[str, Any], config_name: str = "training_config") -> str:
        """Save training configuration."""
        config_path = self.output_dir / "metadata" / f"{config_name}.json"
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        logger.info(f"Training config saved to {config_path}")
        return str(config_path)
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """Clean up old checkpoint files, keeping only the last N."""
        checkpoints_dir = self.output_dir / "checkpoints"
        
        if not checkpoints_dir.exists():
            return
        
        # Group checkpoints by model type
        checkpoints = {}
        for checkpoint in checkpoints_dir.glob("*.h5"):
            model_type = checkpoint.stem.split('_')[0]
            if model_type not in checkpoints:
                checkpoints[model_type] = []
            checkpoints[model_type].append(checkpoint)
        
        # Keep only the most recent N checkpoints for each model type
        for model_type, checkpoint_list in checkpoints.items():
            # Sort by modification time (newest first)
            checkpoint_list.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove old checkpoints
            for checkpoint in checkpoint_list[keep_last_n:]:
                try:
                    checkpoint.unlink()
                    logger.info(f"Removed old checkpoint: {checkpoint}")
                except Exception as e:
                    logger.warning(f"Could not remove checkpoint {checkpoint}: {e}")

def save_model_with_metadata(model, model_type: str, output_dir: str = "models",
                           training_history: Optional[Dict] = None,
                           evaluation_results: Optional[Dict] = None,
                           is_best: bool = False) -> str:
    """
    Convenience function to save a model with metadata.
    
    Args:
        model: Trained Keras model
        model_type: Type of model ('image', 'video', 'audio')
        output_dir: Output directory
        training_history: Training history
        evaluation_results: Evaluation results
        is_best: Whether this is the best model
        
    Returns:
        Path to saved model
    """
    saver = ModelSaver(output_dir)
    return saver.save_model(
        model=model,
        model_name=f"{model_type}_deepfake_detector",
        model_type=model_type,
        training_history=training_history,
        evaluation_results=evaluation_results,
        is_best=is_best
    ) 