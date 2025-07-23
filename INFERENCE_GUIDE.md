# Deepfake Detection Inference Guide

This guide explains how to use the enhanced deepfake detection system for both training models and running inference on various media types.

## Quick Start

### 1. Create Demo Models (No Training Data Required)
```bash
# Create sample models for testing
python train_and_save_models.py --demo
```

### 2. Run Inference on Your Files
```bash
# Single file detection (auto-detects type)
python src/inference/predict_deepfake.py --input your_photo.jpg
python src/inference/predict_deepfake.py --input your_video.mp4
python src/inference/predict_deepfake.py --input your_audio.wav

# Batch process a directory
python src/inference/predict_deepfake.py --input /path/to/media/folder --batch

# List available models
python src/inference/predict_deepfake.py --list-models
```

### 3. Interactive Demo
```bash
# Run the interactive demo
python example_usage.py
```

## Training Your Own Models

### Data Structure
Organize your training data as follows:
```
your_data/
├── real/
│   ├── real_image1.jpg
│   ├── real_video1.mp4
│   ├── real_audio1.wav
│   └── ...
└── fake/
    ├── fake_image1.jpg
    ├── fake_video1.mp4
    ├── fake_audio1.wav
    └── ...
```

### Training Commands
```bash
# Train all models (image, video, audio)
python train_and_save_models.py --data-dir /path/to/your/data

# Train specific model types
python train_and_save_models.py --data-dir /path/to/data --image-only
python train_and_save_models.py --data-dir /path/to/data --video-only
python train_and_save_models.py --data-dir /path/to/data --audio-only

# Custom training parameters
python train_and_save_models.py --data-dir /path/to/data --epochs 50 --batch-size 64
```

## Inference Options

### Single File Detection
```bash
# Basic detection
python src/inference/predict_deepfake.py --input photo.jpg

# Custom confidence threshold
python src/inference/predict_deepfake.py --input photo.jpg --threshold 0.7

# Save results to file
python src/inference/predict_deepfake.py --input photo.jpg --output results.json --format json
```

### Batch Processing
```bash
# Process entire directory
python src/inference/predict_deepfake.py --input /media/folder --batch

# Save batch results
python src/inference/predict_deepfake.py --input /media/folder --batch --output batch_results.csv
```

### Advanced Options
```bash
# Use specific model directory
python src/inference/predict_deepfake.py --input photo.jpg --models-dir /path/to/models

# Use final models instead of best models
python src/inference/predict_deepfake.py --input photo.jpg --no-use-best
```

## Supported File Types

### Images
- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

### Videos
- `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.flv`, `.webm`

### Audio
- `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`

## Understanding Results

### Single File Results
```python
{
    'file_path': 'photo.jpg',
    'file_type': 'image',
    'prediction_score': 0.7834,     # Raw model output (0-1)
    'confidence': 0.5668,           # Distance from decision boundary
    'label': 'FAKE',                # Final classification
    'threshold_used': 0.5,          # Classification threshold
    'timestamp': '2024-01-15T10:30:00',
    'model_metadata': {...}         # Model information
}
```

### Interpreting Scores
- **Prediction Score**: Raw model output (0 = Real, 1 = Fake)
- **Confidence**: How confident the model is (0-100%)
- **Label**: Final classification based on threshold
- **Threshold**: Decision boundary (default: 0.5)

## Model Management

### List Available Models
```bash
python src/inference/predict_deepfake.py --list-models
```

### Model Storage Structure
```
models/
├── image_model_best.h5          # Best performing image model
├── video_model_best.h5          # Best performing video model
├── audio_model_best.h5          # Best performing audio model
├── metadata/                    # Model metadata and configs
│   ├── image_model_metadata.json
│   ├── video_model_metadata.json
│   └── audio_model_metadata.json
└── checkpoints/                 # Training checkpoints
    ├── image_model_20240115_103000.h5
    └── ...
```

## Advanced Usage

### Python API Usage
```python
from src.inference.predict_deepfake import EnhancedDeepfakePredictor

# Initialize predictor
predictor = EnhancedDeepfakePredictor("models")

# Load models
loaded = predictor.load_models()

# Predict single file (auto-detects type)
result = predictor.predict_auto("photo.jpg")
print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.1%}")

# Batch predict directory
results_df = predictor.batch_predict("/path/to/media")
print(results_df[['file_path', 'label', 'confidence']])
```

### Custom Model Training
```python
from src.training.model_saver import ModelSaver
from src.models.deepfake_detector import ImageDeepfakeDetector

# Create and train model
model = ImageDeepfakeDetector().build_model()
# ... training code ...

# Save with metadata
saver = ModelSaver("models")
model_path = saver.save_model(
    model=model,
    model_name="my_custom_model",
    model_type="image",
    training_history=history,
    evaluation_results=metrics,
    is_best=True
)
```

## ⚡ Performance Tips

1. **GPU Usage**: Models automatically use GPU if available
2. **Batch Processing**: More efficient for multiple files
3. **Model Selection**: Use `--use-best` for highest accuracy
4. **Memory Management**: Large videos are processed frame-by-frame

## 🐛 Troubleshooting

### Common Issues

**"No models loaded successfully"**
- Run `python train_and_save_models.py --demo` to create sample models
- Check if models directory exists and contains .h5 files

**"Could not load image/video/audio"**
- Verify file exists and is not corrupted
- Check if file extension is supported
- Ensure proper file permissions

**"TensorFlow GPU errors"**
- Models work with CPU if GPU is unavailable
- Update GPU drivers and CUDA if needed

### Getting Help
- Check logs in `training.log` for training issues
- Use `--help` flag with any script for options
- Run `python example_usage.py` for interactive demo

## Model Performance

The models provided are trained for demonstration purposes. For production use:

1. **Collect Quality Data**: Use diverse, high-quality real and fake samples
2. **Increase Training Time**: Use more epochs for better accuracy
3. **Fine-tune Parameters**: Adjust learning rate, batch size, etc.
4. **Evaluate Thoroughly**: Test on held-out data before deployment

## Updates and Maintenance

- Models save metadata automatically for tracking
- Old checkpoints are cleaned up periodically
- Training history is preserved for analysis
- Model performance metrics are logged

---
