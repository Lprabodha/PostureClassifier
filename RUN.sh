#!/bin/bash
# Exercise Posture Classification System - Setup and Training Script

set -e  # Exit on error

echo "============================================================"
echo "Exercise Posture Classification - Setup and Training"
echo "============================================================"
echo ""

# Function to check command existence
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python installation
echo "[1/6] Checking Python installation..."
if ! command_exists python3; then
    echo "ERROR: Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "Found: $PYTHON_VERSION"
echo ""

# Check if python3-full is installed
echo "[2/6] Checking dependencies..."
if ! dpkg -l | grep -q python3-full; then
    echo "Installing python3-full..."
    sudo apt update
    sudo apt install -y python3-full python3-pip
fi

# Remove old virtual environment if exists
if [ -d "venv" ]; then
    echo "Removing old virtual environment..."
    rm -rf venv
fi

# Create virtual environment
echo "[3/6] Creating virtual environment..."
python3 -m venv venv

if [ ! -d "venv/bin" ]; then
    echo "ERROR: Virtual environment creation failed"
    echo "Please install: sudo apt install python3-full python3-pip python3-venv"
    exit 1
fi

# Activate virtual environment
echo "[4/6] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "[5/6] Installing dependencies..."
pip install --upgrade pip --quiet

# Install required packages
pip install tensorflow numpy opencv-python mediapipe flask werkzeug scikit-learn --quiet

echo "Dependencies installed successfully"
echo ""

# Check dataset
echo "[6/6] Checking dataset..."
if [ ! -d "Datasets/Train" ]; then
    echo "WARNING: Training dataset not found at Datasets/Train/"
    echo "Please organize your data as follows:"
    echo "  Datasets/"
    echo "    Train/"
    echo "      Arm_Raise/"
    echo "      Knee_Extension/"
    echo "      Squats/"
    echo "    Test/"
    echo "      Arm_Raise/"
    echo "      Knee_Extension/"
    echo "      Squats/"
    echo ""
    exit 1
fi

echo "Dataset found"
echo ""

# Display dataset statistics
echo "Dataset Statistics:"
for class in Arm_Raise Knee_Extension Squats; do
    if [ -d "Datasets/Train/$class" ]; then
        count=$(ls -1 "Datasets/Train/$class" 2>/dev/null | wc -l)
        echo "  $class: $count images"
    fi
done
echo ""

# Ask user what to do
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "What would you like to do?"
echo "  1) Train model (standard training, ~30-40 min)"
echo "  2) Train model (improved training, ~40-50 min, higher accuracy)"
echo "  3) Run web application"
echo "  4) Exit"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "Starting standard training..."
        echo ""
        python train.py --data_dir ./Datasets
        ;;
    2)
        echo ""
        echo "Starting improved training..."
        echo ""
        python train.py --data_dir ./Datasets --epochs 50
        ;;
    3)
        if [ ! -d "models" ] || [ -z "$(ls -A models/*.keras 2>/dev/null)" ]; then
            echo ""
            echo "ERROR: No trained model found in models/"
            echo "Please train a model first (option 1 or 2)"
            exit 1
        fi
        echo ""
        echo "Starting web application..."
        echo "Access at: http://localhost:5000"
        echo "Press Ctrl+C to stop"
        echo ""
        python app.py
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "Process Complete"
echo "============================================================"
echo ""

if [ "$choice" = "1" ] || [ "$choice" = "2" ]; then
    echo "Model training completed successfully"
    echo ""
    echo "Next steps:"
    echo "  1. Run web application: ./RUN.sh (choose option 3)"
    echo "  2. Or run manually:"
    echo "     source venv/bin/activate"
    echo "     python app.py"
    echo "     Then visit: http://localhost:5000"
    echo ""
fi

