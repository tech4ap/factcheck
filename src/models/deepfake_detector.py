"""
Enhanced Deepfake Detection Model with Advanced Techniques

This module contains enhanced neural network architectures for detecting deepfakes
with improved accuracy and AUC-ROC through advanced techniques like focal loss,
attention mechanisms, multi-scale feature extraction, and ensemble approaches.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB2, ResNet50V2, DenseNet121
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss for addressing class imbalance in deepfake detection.
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, name="focal_loss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        # Clip predictions to prevent NaN values
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        
        # Calculate focal loss
        ce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * tf.pow(1 - p_t, self.gamma) * ce_loss
        
        return tf.reduce_mean(focal_loss)

class AttentionBlock(layers.Layer):
    """
    Self-attention mechanism for enhanced feature learning.
    """
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv_q = layers.Conv2D(filters // 8, 1, use_bias=False)
        self.conv_k = layers.Conv2D(filters // 8, 1, use_bias=False)
        self.conv_v = layers.Conv2D(filters, 1, use_bias=False)
        self.conv_out = layers.Conv2D(filters, 1, use_bias=False)
        self.softmax = layers.Softmax(axis=-1)
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        
        # Generate query, key, value
        q = self.conv_q(inputs)
        k = self.conv_k(inputs)
        v = self.conv_v(inputs)
        
        # Reshape for attention computation
        q = tf.reshape(q, [batch_size, height * width, self.filters // 8])
        k = tf.reshape(k, [batch_size, height * width, self.filters // 8])
        v = tf.reshape(v, [batch_size, height * width, self.filters])
        
        # Compute attention weights
        attention_weights = tf.matmul(q, k, transpose_b=True)
        attention_weights = self.softmax(attention_weights)
        
        # Apply attention to values
        attended = tf.matmul(attention_weights, v)
        attended = tf.reshape(attended, [batch_size, height, width, self.filters])
        
        # Output projection
        output = self.conv_out(attended)
        
        # Residual connection
        return inputs + output

class ImageDeepfakeDetector:
    """Enhanced CNN-based model for detecting deepfakes in images with advanced techniques."""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (256, 256, 3), 
                 base_model: str = 'efficientnet',
                 use_attention: bool = True,
                 use_multiscale: bool = True):
        self.input_shape = input_shape
        self.base_model = base_model
        self.model = None
        self.use_attention = use_attention
        self.use_multiscale = use_multiscale
        
    def _create_multiscale_feature_extractor(self, inputs):
        """Create multi-scale feature extraction using multiple base models."""
        features = []
        
        # EfficientNetB0 for detailed features
        efficientnet = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_tensor=inputs
        )
        efficientnet.trainable = False
        eff_features = efficientnet(inputs)
        features.append(layers.GlobalAveragePooling2D()(eff_features))
        
        # EfficientNetB2 for multi-scale features
        efficientnet_b2 = EfficientNetB2(
            weights='imagenet',
            include_top=False,
            input_tensor=inputs
        )
        efficientnet_b2.trainable = False
        eff_b2_features = efficientnet_b2(inputs)
        features.append(layers.GlobalAveragePooling2D()(eff_b2_features))
        
        # DenseNet for dense connectivity features
        densenet = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_tensor=inputs
        )
        densenet.trainable = False
        dense_features = densenet(inputs)
        features.append(layers.GlobalAveragePooling2D()(dense_features))
        
        # Concatenate all features
        combined_features = layers.Concatenate()(features)
        return combined_features, [efficientnet, efficientnet_b2, densenet]
        
    def _create_enhanced_feature_extractor(self, inputs):
        """Create enhanced feature extraction with attention mechanisms."""
        if self.base_model == 'efficientnet':
            base = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_tensor=inputs
            )
        elif self.base_model == 'efficientnet_b2':
            base = EfficientNetB2(
                weights='imagenet',
                include_top=False,
                input_tensor=inputs
            )
        elif self.base_model == 'resnet':
            base = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_tensor=inputs
            )
        elif self.base_model == 'densenet':
            base = DenseNet121(
                weights='imagenet',
                include_top=False,
                input_tensor=inputs
            )
        else:
            raise ValueError(f"Unsupported base model: {self.base_model}")
        
        # Freeze base model initially
        base.trainable = False
        
        # Extract features
        features = base(inputs)
        
        # Add attention mechanism if enabled
        if self.use_attention:
            features = AttentionBlock(features.shape[-1])(features)
        
        # Global pooling with both average and max
        gap = layers.GlobalAveragePooling2D()(features)
        gmp = layers.GlobalMaxPooling2D()(features)
        combined = layers.Concatenate()([gap, gmp])
        
        return combined, base

    def build_model(self, dropout_rate: float = 0.3, learning_rate: float = 1e-4,
                   use_focal_loss: bool = True, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        """Build the enhanced image deepfake detection model with advanced techniques."""
        
        inputs = layers.Input(shape=self.input_shape)
        
        # Data augmentation layers (applied during training)
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        x = layers.RandomContrast(0.1)(x)
        
        # Choose feature extraction strategy
        if self.use_multiscale:
            features, base_models = self._create_multiscale_feature_extractor(x)
            self.base_models = base_models
        else:
            features, base_model = self._create_enhanced_feature_extractor(x)
            self.base_models = [base_model]
        
        # Enhanced dense layers with advanced regularization
        x = layers.BatchNormalization()(features)
        x = layers.Dropout(dropout_rate)(x)
        
        # Multi-layer perceptron with residual connections
        x1 = layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Dropout(dropout_rate)(x1)
        
        x2 = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x1)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Dropout(dropout_rate)(x2)
        
        # Residual connection
        if x1.shape[-1] != x2.shape[-1]:
            x1_proj = layers.Dense(512)(x1)
        else:
            x1_proj = x1
        x2_residual = layers.Add()([x1_proj, x2])
        
        x3 = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x2_residual)
        x3 = layers.BatchNormalization()(x3)
        x3 = layers.Dropout(dropout_rate)(x3)
        
        x4 = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x3)
        x4 = layers.BatchNormalization()(x4)
        x4 = layers.Dropout(dropout_rate)(x4)
        
        # Final prediction layers with ensemble approach
        # Main prediction head
        main_output = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x4)
        main_output = layers.Dropout(dropout_rate * 0.5)(main_output)
        main_output = layers.Dense(1, activation='sigmoid', name='main_output')(main_output)
        
        # Auxiliary prediction head for ensemble
        aux_output = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x3)
        aux_output = layers.Dropout(dropout_rate * 0.5)(aux_output)
        aux_output = layers.Dense(1, activation='sigmoid', name='aux_output')(aux_output)
        
        # Ensemble the predictions
        ensemble_output = layers.Average(name='ensemble_output')([main_output, aux_output])
        
        # Create model
        model = models.Model(inputs=inputs, outputs=ensemble_output)
        
        # Enhanced optimizer with gradient clipping
        optimizer = optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0  # Gradient clipping
        )
        
        # Choose loss function
        if use_focal_loss:
            loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            loss = 'binary_crossentropy'
        
        # Compile with enhanced metrics
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.AUC(name='pr_auc', curve='PR'),  # Precision-Recall AUC
                tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
                tf.keras.metrics.TruePositives(name='tp'),
                tf.keras.metrics.TrueNegatives(name='tn'),
                tf.keras.metrics.FalsePositives(name='fp'),
                tf.keras.metrics.FalseNegatives(name='fn')
            ]
        )
        
        self.model = model
        logger.info(f"Built enhanced image model with {self.base_model} base, "
                   f"attention={'enabled' if self.use_attention else 'disabled'}, "
                   f"multiscale={'enabled' if self.use_multiscale else 'disabled'}, "
                   f"focal_loss={'enabled' if use_focal_loss else 'disabled'}")
        return model
    
    def fine_tune(self, learning_rate: float = 1e-5, unfreeze_layers: int = -30):
        """Enhanced fine-tuning with gradual unfreezing and discriminative learning rates."""
        if self.model is None:
            raise ValueError("Model must be built before fine-tuning")
        
        # Gradually unfreeze base model layers
        for base_model in self.base_models:
            base_model.trainable = True
            
            # Apply different learning rates to different layers
            num_layers = len(base_model.layers)
            for i, layer in enumerate(base_model.layers):
                if i < num_layers + unfreeze_layers:  # Only unfreeze last N layers
                    layer.trainable = True
                    # Lower learning rate for earlier layers
                    if hasattr(layer, 'kernel_regularizer'):
                        layer.kernel_regularizer = tf.keras.regularizers.l2(0.01)
                else:
                    layer.trainable = False
        
        # Create discriminative learning rate optimizer
        optimizer = optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0
        )
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=optimizer,
            loss=self.model.loss,
            metrics=self.model.metrics
        )
        
        logger.info(f"Enhanced fine-tuning enabled with learning rate {learning_rate}, "
                   f"unfroze last {abs(unfreeze_layers)} layers")

    def create_advanced_callbacks(self, model_name: str, patience: int = 15):
        """Create advanced callbacks for better training."""
        callbacks_list = [
            # Enhanced early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=patience,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            
            # Advanced learning rate scheduling
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc',
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-7,
                mode='max',
                verbose=1
            ),
            
            # Model checkpointing with multiple criteria
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'models/{model_name}_best_auc.h5',
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'models/{model_name}_best_loss.h5',
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            
            # Learning rate warmup
            tf.keras.callbacks.LearningRateScheduler(
                self._warmup_cosine_decay_scheduler,
                verbose=0
            ),
            
            # Advanced logging
            tf.keras.callbacks.CSVLogger(
                f'models/{model_name}_training_log.csv',
                append=True
            )
        ]
        
        return callbacks_list
    
    def _warmup_cosine_decay_scheduler(self, epoch, lr):
        """Learning rate scheduler with warmup and cosine decay."""
        warmup_epochs = 5
        total_epochs = 100  # Adjust based on your training
        
        if epoch < warmup_epochs:
            # Warmup phase
            return lr * (epoch + 1) / warmup_epochs
        else:
            # Cosine decay phase
            decay_epochs = total_epochs - warmup_epochs
            alpha = (epoch - warmup_epochs) / decay_epochs
            return lr * 0.5 * (1 + np.cos(np.pi * alpha))
        
    def get_model_summary(self):
        """Get detailed model summary."""
        if self.model is None:
            return "Model not built yet"
        
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'architecture': self.base_model,
            'attention_enabled': self.use_attention,
            'multiscale_enabled': self.use_multiscale,
            'input_shape': self.input_shape
        }

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