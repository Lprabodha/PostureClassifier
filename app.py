"""
Flask Web Application for Posture Classification
Production-ready interface for uploading images/videos
"""

from flask import Flask, request, render_template, send_file, jsonify, url_for
import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
MODEL_PATH = None  # Will be set on startup

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Global variables for model and MediaPipe
model = None
class_names = []
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model_and_classes():
    """Load the trained model and class names"""
    global model, class_names, pose
    
    # Find the latest model
    models_dir = 'models'
    if not os.path.exists(models_dir):
        raise FileNotFoundError("Models directory not found. Please train a model first.")
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
    if not model_files:
        raise FileNotFoundError("No trained model found. Please train a model first.")
    
    # Use the latest model
    model_files.sort(reverse=True)
    model_path = os.path.join(models_dir, model_files[0])
    
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Load class names
    class_names_path = os.path.join(models_dir, 'class_names.txt')
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        # Auto-detect from dataset
        train_dir = 'Datasets/Train'
        if os.path.exists(train_dir):
            class_names = sorted([d for d in os.listdir(train_dir) 
                                if os.path.isdir(os.path.join(train_dir, d))])
        else:
            class_names = ['Arm_Raise', 'Knee_Extension', 'Squats']
    
    print(f"Classes: {class_names}")
    
    # Initialize MediaPipe Pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    print("Model and MediaPipe loaded successfully!")


def preprocess_frame(frame, img_size=224):
    """Preprocess frame for model prediction"""
    img = cv2.resize(frame, (img_size, img_size))
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return np.expand_dims(img, axis=0)


def get_angle(a, b, c):
    """Calculate angle between three landmarks"""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def detect_posture_rule_based(landmarks):
    """Detect posture using rule-based approach"""
    # Arm Raise
    left_arm = get_angle(
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    )
    right_arm = get_angle(
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    )
    if left_arm > 140 and right_arm > 140:
        return "Arm_Raise"
    
    # Squats or Knee_Extension
    left_knee = get_angle(
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    )
    right_knee = get_angle(
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    )
    
    if 60 < left_knee < 130 and 60 < right_knee < 130:
        return "Squats"
    elif left_knee > 160 or right_knee > 160:
        return "Knee_Extension"
    
    return None


def check_posture_correctness(posture, landmarks):
    """Check if posture is performed correctly"""
    if posture == "Arm_Raise":
        left_arm = get_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        )
        right_arm = get_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        )
        return left_arm > 140 and right_arm > 140
        
    elif posture == "Squats":
        left_knee = get_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        )
        right_knee = get_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        )
        return 60 < left_knee < 130 and 60 < right_knee < 130
        
    elif posture == "Knee_Extension":
        left_knee = get_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        )
        right_knee = get_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        )
        return left_knee > 160 or right_knee > 160
    
    return False


def process_image(image_path):
    """Process a single image"""
    frame = cv2.imread(image_path)
    
    if frame is None:
        return None, "Error: Could not read image"
    
    # CNN prediction
    cnn_input = preprocess_frame(frame)
    prediction = model.predict(cnn_input, verbose=0)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    
    # MediaPipe detection
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        # Rule-based detection
        rb_posture = detect_posture_rule_based(results.pose_landmarks.landmark)
        
        # Hybrid decision
        posture = predicted_class if confidence > 0.65 else rb_posture
        if posture is None:
            posture = predicted_class
        
        # Check correctness
        correct = check_posture_correctness(posture, results.pose_landmarks.landmark)
        
        # Draw landmarks
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Add text
        status = "✅ Correct" if correct else "❌ Incorrect"
        color = (0, 255, 0) if correct else (0, 0, 255)
        text = f"{posture} {status} ({confidence:.2f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, color, 2)
        
        # Save result
        result_filename = f"result_{uuid.uuid4().hex[:8]}.jpg"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        cv2.imwrite(result_path, frame)
        
        return result_filename, {
            'posture': posture,
            'confidence': confidence,
            'correct': correct,
            'method': 'CNN' if confidence > 0.65 else 'Rule-based'
        }
    else:
        return None, "Error: No human detected in image"


def process_video(video_path):
    """Process a video file"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None, "Error: Could not open video"
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video
    result_filename = f"result_{uuid.uuid4().hex[:8]}.mp4"
    result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
    
    frame_count = 0
    detections = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process every 5th frame for efficiency
        if frame_count % 5 == 0:
            # CNN prediction
            cnn_input = preprocess_frame(frame)
            prediction = model.predict(cnn_input, verbose=0)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = float(np.max(prediction))
            
            # MediaPipe detection
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                rb_posture = detect_posture_rule_based(results.pose_landmarks.landmark)
                posture = predicted_class if confidence > 0.65 else rb_posture
                if posture is None:
                    posture = predicted_class
                
                correct = check_posture_correctness(posture, results.pose_landmarks.landmark)
                
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Add text
                status = "✅" if correct else "❌"
                color = (0, 255, 0) if correct else (0, 0, 255)
                text = f"{posture} {status} ({confidence:.2f})"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, color, 2)
                
                detections.append({
                    'frame': frame_count,
                    'posture': posture,
                    'confidence': confidence,
                    'correct': correct
                })
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    # Summary statistics
    summary = {
        'total_frames': frame_count,
        'detections': len(detections),
        'posture_counts': {}
    }
    
    for det in detections:
        posture = det['posture']
        summary['posture_counts'][posture] = summary['posture_counts'].get(posture, 0) + 1
    
    return result_filename, summary


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', classes=class_names)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, mp4, avi, mov'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Process based on file type
        if file_ext in ['jpg', 'jpeg', 'png']:
            result_file, result_data = process_image(filepath)
            file_type = 'image'
        else:
            result_file, result_data = process_video(filepath)
            file_type = 'video'
        
        # Clean up uploaded file
        os.remove(filepath)
        
        if result_file:
            return jsonify({
                'success': True,
                'file_type': file_type,
                'result_url': url_for('get_result', filename=result_file),
                'data': result_data
            })
        else:
            return jsonify({
                'success': False,
                'error': result_data
            }), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/results/<filename>')
def get_result(filename):
    """Serve result file"""
    return send_file(os.path.join(app.config['RESULTS_FOLDER'], filename))


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': class_names
    })


if __name__ == '__main__':
    print("=" * 60)
    print("ZUMBA AI - Posture Classification Web App")
    print("=" * 60)
    
    # Load model
    try:
        load_model_and_classes()
    except Exception as e:
        print(f"ERROR: {e}")
        print("\nPlease train a model first:")
        print("  python train.py --data_dir ./Datasets")
        exit(1)
    
    print("\n" + "=" * 60)
    print("Server starting...")
    print("Access the web app at: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

