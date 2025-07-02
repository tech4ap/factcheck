"""
Deepfake Detection Model

This module contains the neural network architectures for detecting deepfakes
in images, videos, and audio. It includes individual models for each media type
and an ensemble model that combines predictions from all three.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

class ImageDeepfakeDetector:
    """Enhanced CNN-based model for detecting deepfakes in images."""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (256, 256, 3), 
                 base_model: str = 'efficientnet'):
        self.input_shape = input_shape
        self.base_model = base_model
        self.model = None
        
    def build_model(self, dropout_rate: float = 0.3, learning_rate: float = 1e-4):
        """Build the enhanced image deepfake detection model."""
        
        if self.base_model == 'efficientnet':
            base = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.base_model == 'resnet':
            base = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported base model: {self.base_model}")
        
        # Freeze base model layers initially
        base.trainable = False
        
        # Enhanced model architecture
        model = models.Sequential([
            base,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            # Enhanced dense layers with better regularization
            layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Enhanced optimizer with better learning rate scheduling
        optimizer = optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        self.model = model
        logger.info(f"Built enhanced image model with {self.base_model} base")
        return model
    
    def fine_tune(self, learning_rate: float = 1e-5):
        """Enhanced fine-tuning with gradual unfreezing."""
        if self.model is None:
            raise ValueError("Model must be built before fine-tuning")
        
        # Unfreeze base model layers gradually
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Set different learning rates for different layers
        for layer in base_model.layers:
            if isinstance(layer, layers.Conv2D):
                layer.kernel_regularizer = tf.keras.regularizers.l2(0.01)
        
        # Recompile with lower learning rate
        optimizer = optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        logger.info("Enhanced fine-tuning enabled for image model")

class VideoDeepfakeDetector:
    """Enhanced CNN-LSTM model for detecting deepfakes in video frames."""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (256, 256, 3),
                 frame_sequence_length: int = 10):
        self.input_shape = input_shape
        self.frame_sequence_length = frame_sequence_length
        self.model = None
        
    def build_model(self, dropout_rate: float = 0.3, learning_rate: float = 1e-4):
        """Build the enhanced video deepfake detection model with attention."""
        
        # Input for frame sequence
        input_layer = layers.Input(shape=(self.frame_sequence_length,) + self.input_shape)
        
        # Enhanced CNN feature extractor for each frame
        frame_features = []
        for i in range(self.frame_sequence_length):
            frame_input = layers.Lambda(lambda x: x[:, i, :, :, :])(input_layer)
            
            # Enhanced CNN for frame processing
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(frame_input)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.2)(x)
            
            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.2)(x)
            
            x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.2)(x)
            
            x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.GlobalAveragePooling2D()(x)
            
            frame_features.append(x)
        
        # Concatenate frame features
        concatenated = layers.Concatenate()(frame_features)
        
        # Reshape for LSTM processing
        reshaped = layers.Reshape((self.frame_sequence_length, -1))(concatenated)
        
        # Bidirectional LSTM for temporal modeling
        lstm_out = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
        )(reshaped)
        
        # Attention mechanism
        attention = layers.Attention()([lstm_out, lstm_out])
        
        # Global average pooling over time
        temporal_features = layers.GlobalAveragePooling1D()(attention)
        
        # Enhanced dense layers with regularization
        x = layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(temporal_features)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        output = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs=input_layer, outputs=output)
        
        # Enhanced optimizer
        optimizer = optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        self.model = model
        logger.info(f"Built enhanced video model with {self.frame_sequence_length} frame sequence and attention")
        return model

class AudioDeepfakeDetector:
    """Enhanced CNN-based model for detecting deepfakes in audio spectrograms."""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (128, 128, 1)):
        self.input_shape = input_shape
        self.model = None
        
    def build_model(self, dropout_rate: float = 0.3, learning_rate: float = 1e-4):
        """Build the enhanced audio deepfake detection model."""
        
        input_layer = layers.Input(shape=self.input_shape)
        
        # Enhanced CNN for spectrogram processing with residual connections
        # Initial convolution
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
        # Residual block 1
        residual = x
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
        # Residual block 2
        residual = layers.Conv2D(128, (1, 1), padding='same')(x)  # 1x1 conv for dimension matching
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
        # Residual block 3
        residual = layers.Conv2D(256, (1, 1), padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Enhanced dense layers with regularization
        x = layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        output = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs=input_layer, outputs=output)
        
        # Enhanced optimizer
        optimizer = optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        self.model = model
        logger.info("Built enhanced audio model with residual connections")
        return model

class EnsembleDeepfakeDetector:
    """Ensemble model that combines predictions from all three media type models."""
    
    def __init__(self, image_model: ImageDeepfakeDetector = None,
                 video_model: VideoDeepfakeDetector = None,
                 audio_model: AudioDeepfakeDetector = None):
        self.image_model = image_model
        self.video_model = video_model
        self.audio_model = audio_model
        self.ensemble_model = None
        
    def build_ensemble(self, weights: Dict[str, float] = None):
        """Build ensemble model that combines predictions."""
        
        if weights is None:
            weights = {'image': 0.4, 'video': 0.4, 'audio': 0.2}
        
        # Create ensemble model
        ensemble_inputs = []
        ensemble_outputs = []
        
        if self.image_model and self.image_model.model:
            ensemble_inputs.append(self.image_model.model.input)
            ensemble_outputs.append(self.image_model.model.output * weights['image'])
        
        if self.video_model and self.video_model.model:
            ensemble_inputs.append(self.video_model.model.input)
            ensemble_outputs.append(self.video_model.model.output * weights['video'])
        
        if self.audio_model and self.audio_model.model:
            ensemble_inputs.append(self.audio_model.model.input)
            ensemble_outputs.append(self.audio_model.model.output * weights['audio'])
        
        if not ensemble_outputs:
            raise ValueError("At least one model must be provided")
        
        # Combine predictions
        if len(ensemble_outputs) == 1:
            ensemble_output = ensemble_outputs[0]
        else:
            ensemble_output = layers.Average()(ensemble_outputs)
        
        self.ensemble_model = models.Model(
            inputs=ensemble_inputs,
            outputs=ensemble_output
        )
        
        logger.info("Built ensemble model")
        return self.ensemble_model
    
    def predict(self, image_data: np.ndarray = None, 
                video_data: np.ndarray = None, 
                audio_data: np.ndarray = None) -> float:
        """Make ensemble prediction."""
        
        if self.ensemble_model is None:
            raise ValueError("Ensemble model must be built before prediction")
        
        inputs = []
        
        if image_data is not None and self.image_model:
            inputs.append(image_data)
        
        if video_data is not None and self.video_model:
            inputs.append(video_data)
        
        if audio_data is not None and self.audio_model:
            inputs.append(audio_data)
        
        if not inputs:
            raise ValueError("At least one type of data must be provided")
        
        prediction = self.ensemble_model.predict(inputs)
        return prediction[0][0]

def create_callbacks(model_name: str, patience: int = 10) -> list:
    """Create training callbacks."""
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=f'models/{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        callbacks.TensorBoard(
            log_dir=f'logs/{model_name}',
            histogram_freq=1
        )
    ]
    return callbacks_list

def evaluate_model(model, test_data, test_labels) -> Dict[str, float]:
    """Evaluate model performance."""
    predictions = model.predict(test_data)
    predictions_binary = (predictions > 0.5).astype(int)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        'accuracy': accuracy_score(test_labels, predictions_binary),
        'precision': precision_score(test_labels, predictions_binary),
        'recall': recall_score(test_labels, predictions_binary),
        'f1_score': f1_score(test_labels, predictions_binary),
        'auc': roc_auc_score(test_labels, predictions)
    }
    
    return metrics 