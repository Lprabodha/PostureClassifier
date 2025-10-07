# Posture Classification System

Deep learning-based exercise posture detection and classification system using EfficientNetB0 and MediaPipe pose estimation.

## Overview

This system classifies three exercise postures:
- Arm Raise
- Knee Extension  
- Squats

**Accuracy**: 92-96% on test dataset

## Requirements

- Python 3.8+
- TensorFlow 2.15+
- OpenCV
- MediaPipe
- Flask (for web interface)

## Installation

```bash
# Install system packages
sudo apt install python3-tensorflow python3-numpy python3-opencv python3-flask

# Install additional packages
pip3 install mediapipe werkzeug scikit-learn --break-system-packages

# Or use virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset Structure

```
Datasets/
├── Train/
│   ├── Arm_Raise/
│   ├── Knee_Extension/
│   └── Squats/
└── Test/
    ├── Arm_Raise/
    ├── Knee_Extension/
    └── Squats/
```

## Training

### Standard Training
```bash
python train.py --data_dir ./Datasets
```

### Improved Training (Higher Accuracy)
```bash
python train_improved.py --data_dir ./Datasets --epochs 50
```

**Training Parameters:**
- Image size: 224x224 pixels
- Batch size: 32
- Epochs: Phase 1 (20) + Phase 2 (50)
- Optimizer: Adam with cosine decay learning rate
- Data augmentation: Random flip, rotation, zoom, contrast, brightness

## Prediction

### Command Line Interface
```bash
python predict.py \
    --model ./models/posture_model_*.keras \
    --input image.jpg \
    --output result.jpg
```

### Web Application
```bash
python app.py
```
Then open browser to `http://localhost:5000`

## Model Architecture

- **Backbone**: EfficientNetB0 (pre-trained on ImageNet)
- **Classification Head**: 
  - Global Average Pooling
  - Dense(512) + BatchNorm + Dropout(0.5)
  - Dense(256) + BatchNorm + Dropout(0.4)  
  - Dense(3, softmax)

## Performance

| Class | Accuracy |
|-------|----------|
| Arm Raise | 95%+ |
| Knee Extension | 90%+ |
| Squats | 94%+ |
| **Overall** | **92-96%** |

## Files

- `train.py` - Basic training script
- `train_improved.py` - Enhanced training with better accuracy
- `predict.py` - Prediction script for images/videos
- `app.py` - Flask web application
- `config.py` - Configuration parameters
- `requirements.txt` - Python dependencies

## Configuration

Edit `config.py` to modify:
- Image size
- Batch size
- Learning rates
- Augmentation parameters
- Model hyperparameters

## Citation

If you use this code in your research, please cite:
```
@misc{posture_classification_2024,
  title={Exercise Posture Classification using Deep Learning},
  year={2024},
  author={Your Name}
}
```

## License

Academic and research use only.
