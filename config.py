"""
Configuration file for Posture Classification Project
Modify these settings according to your needs
"""

import os

# ==========================================
# PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
DATA_DIR = os.path.join(BASE_DIR, "Datasets")
TRAIN_DIR = os.path.join(DATA_DIR, "Train")
TEST_DIR = os.path.join(DATA_DIR, "Test")

# Model save directory
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")

# ==========================================
# MODEL PARAMETERS
# ==========================================
# Image size (height, width)
IMG_SIZE = (224, 224)

# Batch size for training
BATCH_SIZE = 32

# Number of epochs for each training phase
EPOCHS_PHASE1 = 15  # Training with frozen backbone
EPOCHS_PHASE2 = 40  # Fine-tuning

# Random seed for reproducibility
RANDOM_SEED = 123

# Number of layers to unfreeze in phase 2
UNFREEZE_LAYERS = 30

# ==========================================
# TRAINING PARAMETERS
# ==========================================
# Learning rates
LEARNING_RATE_PHASE1 = 1e-3  # Initial training
LEARNING_RATE_PHASE2 = 1e-5  # Fine-tuning

# Dropout rates
DROPOUT_RATE_1 = 0.5
DROPOUT_RATE_2 = 0.4

# Dense layer size
DENSE_UNITS = 256

# L2 regularization
L2_REG = 1e-4

# ==========================================
# DATA AUGMENTATION
# ==========================================
AUGMENTATION_CONFIG = {
    'random_flip': 'horizontal',
    'random_rotation': 0.2,
    'random_zoom': 0.2,
    'random_contrast': 0.2,
}

# ==========================================
# CALLBACKS
# ==========================================
# Early stopping patience
EARLY_STOPPING_PATIENCE = 10

# ReduceLROnPlateau settings
REDUCE_LR_FACTOR = 0.2
REDUCE_LR_PATIENCE = 5
REDUCE_LR_MIN_LR = 1e-6

# ==========================================
# PREDICTION PARAMETERS
# ==========================================
# Confidence threshold for CNN predictions
CNN_CONFIDENCE_THRESHOLD = 0.65

# Frame skip for video processing (process every Nth frame)
VIDEO_FRAME_SKIP = 10

# MediaPipe Pose settings
MEDIAPIPE_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_TRACKING_CONFIDENCE = 0.5

# ==========================================
# POSTURE DETECTION THRESHOLDS
# ==========================================
# Arm Raise thresholds
ARM_RAISE_MIN_ANGLE = 140  # Minimum arm angle (degrees)

# Squat thresholds
SQUAT_MIN_KNEE_ANGLE = 60   # Minimum knee angle for squat
SQUAT_MAX_KNEE_ANGLE = 130  # Maximum knee angle for squat
SQUAT_HIP_OFFSET = 0.05     # Hip position offset threshold

# ==========================================
# CLASS NAMES
# ==========================================
# Default class names (will be overridden by training data)
DEFAULT_CLASS_NAMES = ['Arm_Raise', 'Squats']

# ==========================================
# LOGGING
# ==========================================
# Enable verbose output
VERBOSE = True

# Log file path (optional)
LOG_FILE = None  # Set to a path like "training.log" to enable file logging

# ==========================================
# SYSTEM
# ==========================================
# Enable mixed precision training (faster on modern GPUs)
ENABLE_MIXED_PRECISION = False  # Set to True if you have a compatible GPU

# Number of parallel workers for data loading
NUM_WORKERS = 4  # Adjust based on your CPU cores

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def create_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
def get_config():
    """Return configuration as dictionary"""
    return {
        'data_dir': DATA_DIR,
        'model_save_dir': MODEL_SAVE_DIR,
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'epochs_phase1': EPOCHS_PHASE1,
        'epochs_phase2': EPOCHS_PHASE2,
        'seed': RANDOM_SEED,
        'lr_phase1': LEARNING_RATE_PHASE1,
        'lr_phase2': LEARNING_RATE_PHASE2,
        'dropout_1': DROPOUT_RATE_1,
        'dropout_2': DROPOUT_RATE_2,
        'dense_units': DENSE_UNITS,
        'l2_reg': L2_REG,
        'augmentation': AUGMENTATION_CONFIG,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'reduce_lr_factor': REDUCE_LR_FACTOR,
        'reduce_lr_patience': REDUCE_LR_PATIENCE,
        'cnn_threshold': CNN_CONFIDENCE_THRESHOLD,
        'frame_skip': VIDEO_FRAME_SKIP,
    }

def print_config():
    """Print current configuration"""
    config = get_config()
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    for key, value in config.items():
        print(f"{key:30s}: {value}")
    print("=" * 60)

if __name__ == "__main__":
    # Test configuration
    print_config()
    create_directories()
    print("\nConfiguration loaded successfully")

