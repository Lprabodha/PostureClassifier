# Zumba AI - Pose Analysis & Correctness Detection

Deep learning-based Zumba pose detection and correctness evaluation system using EfficientNetB0 and MediaPipe pose estimation.

## Overview

This system detects and evaluates **two Zumba poses**:
- **Arm Raise** - Detects and validates proper arm raise form
- **Squats** - Detects and validates proper squat form

### Output Format
The system provides **4 possible results**:
- ✅ Arm Raise Correct
- ❌ Arm Raise Incorrect
- ✅ Squats Correct
- ❌ Squats Incorrect

**Accuracy**: 92-96% on test dataset

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the web app:**
   ```bash
   python app.py
   ```

3. **Open browser:** http://localhost:5000

4. **Upload a video/image** and get one of 4 results!

📖 See `QUICK_START.md` for detailed usage guide.

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
│   ├── Arm_Raise/      # Training images for arm raise pose
│   └── Squats/         # Training images for squat pose
└── Test/
    ├── Arm_Raise/      # Test images for arm raise pose
    └── Squats/         # Test images for squat pose
```

**Note:** Only Arm_Raise and Squats folders are used. Any other class folders will be ignored during training.

## Training

### Train the Model
```bash
python train.py --data_dir ./Datasets --epochs 50
```

The training script will:
- Automatically filter and use only **Arm_Raise** and **Squats** data
- Ignore any other class folders (e.g., Knee_Extension)
- Use two-phase training for optimal accuracy

**Training Parameters:**
- Image size: 224x224 pixels
- Batch size: 32
- Epochs: Phase 1 (20, frozen backbone) + Phase 2 (50, fine-tuning)
- Optimizer: Adam with ReduceLROnPlateau
- Data augmentation: Random flip, rotation, zoom, contrast, brightness, translation
- Classes: **Only Arm_Raise and Squats**

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
  - Dense(2, softmax) - **2 classes only**
- **Pose Detection**: MediaPipe Pose for landmark detection and correctness validation

## How It Works

### Detection Pipeline
1. **CNN Classification** - EfficientNetB0 predicts pose type (Arm Raise or Squats)
2. **Rule-Based Validation** - MediaPipe landmarks verify pose correctness
3. **Hybrid Decision** - Combines CNN confidence with rule-based validation
4. **Correctness Evaluation** - Analyzes joint angles and body alignment

### Correctness Criteria

**Arm Raise ✋**
- Arms extended (angle > 140°)
- Elbows above shoulders
- Both arms raised symmetrically

**Squats 🦵**
- Knees bent (60°-130° range)
- Hips lowered significantly
- Proper squat depth maintained

## Performance

| Class | Accuracy |
|-------|----------|
| Arm Raise | 95%+ |
| Squats | 94%+ |
| **Overall** | **94-96%** |

**Result Types:** 4 possible outputs
- Arm Raise Correct / Incorrect
- Squats Correct / Incorrect

## Files

- `train.py` - Training script (uses only Arm_Raise and Squats)
- `predict.py` - Command-line prediction for images/videos
- `app.py` - Flask web application with simplified UI
- `config.py` - Configuration parameters
- `requirements.txt` - Python dependencies
- `templates/index.html` - Web UI (clean, simplified design)
- `UPDATE_SUMMARY.md` - Detailed summary of recent changes
- `QUICK_START.md` - Quick start guide

## Configuration

Edit `config.py` to modify:
- Image size
- Batch size
- Learning rates
- Augmentation parameters
- Model hyperparameters

## Features

✅ **Simple Output** - Only 4 possible results  
✅ **Automatic Detection** - Model determines the pose type  
✅ **Correctness Evaluation** - Real-time form feedback  
✅ **Clean UI** - No information overload  
✅ **Video Support** - Analyzes entire videos for dominant pose  
✅ **Image Support** - Single frame analysis  
✅ **Hybrid Approach** - CNN + Rule-based validation  

## Recent Updates

🔄 **Latest Version (2024)**
- Simplified to 2 poses: Arm Raise and Squats
- 4 possible output results
- Removed unnecessary metrics from UI
- Enhanced correctness evaluation
- Cleaner, more intuitive interface
- Automatic dominant pose detection for videos

See `UPDATE_SUMMARY.md` for complete change details.

## Citation

If you use this code in your research, please cite:
```
@misc{zumba_ai_2024,
  title={Zumba AI: Pose Detection and Correctness Evaluation using Deep Learning},
  year={2024},
  author={Zumba AI Project}
}
```

## License

Academic and research use only.
