"""
Posture Classification Prediction Script (Local Version)
Supports video analysis with MediaPipe pose detection
"""

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import argparse
import os


class PosturePredictor:
    def __init__(self, model_path, class_names_path=None):
        """
        Initialize posture predictor for Zumba poses
        
        Args:
            model_path: Path to trained .keras model
            class_names_path: Path to class_names.txt file (optional)
        """
        print("Loading model...")
        self.model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded from: {model_path}")
        
        # Load class names - only Arm_Raise and Squats
        if class_names_path and os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                all_classes = [line.strip() for line in f.readlines()]
                # Filter to only include Arm_Raise and Squats
                self.class_names = [c for c in all_classes if c in ['Arm_Raise', 'Squats']]
        else:
            # Default class names
            self.class_names = ['Arm_Raise', 'Squats']
        
        print(f"Classes: {self.class_names}")
        
        # Setup MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # For visualization
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def preprocess_frame(self, frame, img_size=224):
        """Preprocess frame for model prediction"""
        img = cv2.resize(frame, (img_size, img_size))
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return np.expand_dims(img, axis=0)
    
    def check_if_human(self, frame):
        """Check if a human is detected in the frame"""
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return results.pose_landmarks is not None
    
    def get_angle(self, a, b, c):
        """Calculate angle between three landmarks"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle
    
    def detect_posture_rule_based(self, landmarks):
        """Detect Zumba poses: Arm Raise and Squats only"""
        # Arm Raise: both arms mostly straight and raised
        left_arm = self.get_angle(
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        )
        right_arm = self.get_angle(
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        )
        
        left_elbow_y = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y
        right_elbow_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
        left_shoulder_y = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        right_shoulder_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        
        if (left_arm > 140 and right_arm > 140 and 
            left_elbow_y < left_shoulder_y and right_elbow_y < right_shoulder_y):
            return "Arm_Raise"
        
        # Squats: knees bent
        left_knee = self.get_angle(
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        )
        right_knee = self.get_angle(
            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        )
        hip_y = (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y +
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
        shoulder_y = (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y +
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2
        
        if ((50 < left_knee < 140) or (50 < right_knee < 140)) and hip_y > shoulder_y - 0.2:
            return "Squats"
        
        return None
    
    def check_posture_correctness(self, posture, landmarks):
        """Check if the Zumba pose is performed correctly"""
        if posture == "Arm_Raise":
            left_arm = self.get_angle(
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            )
            right_arm = self.get_angle(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            )
            
            left_elbow_y = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y
            right_elbow_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
            left_shoulder_y = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            right_shoulder_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            
            arms_extended = left_arm > 140 and right_arm > 140
            arms_raised = left_elbow_y < left_shoulder_y and right_elbow_y < right_shoulder_y
            
            return arms_extended and arms_raised
            
        elif posture == "Squats":
            left_knee = self.get_angle(
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            )
            right_knee = self.get_angle(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            )
            hip_y = (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y +
                     landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
            shoulder_y = (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y +
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2
            
            # More lenient thresholds for Zumba squats
            knees_bent = (50 < left_knee < 140) or (50 < right_knee < 140)
            hip_lowered = hip_y > shoulder_y - 0.2
            
            return knees_bent and hip_lowered
            
        return False
    
    def analyze_video(self, video_path, show_video=False, save_output=None):
        """
        Analyze video for posture detection
        
        Args:
            video_path: Path to input video
            show_video: Whether to display the video (requires display)
            save_output: Path to save output video (optional)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_count = 0
        detected_any = False
        last_posture = None
        
        # For saving output video
        if save_output:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
        
        print(f"\nAnalyzing video: {video_path}")
        print("="*60)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process every 10th frame for efficiency
            if frame_count % 10 != 0:
                if save_output:
                    out_writer.write(frame)
                continue
            
            if not self.check_if_human(frame):
                print(f"Frame {frame_count}: No human detected ❌")
                if save_output:
                    out_writer.write(frame)
                continue
            
            # CNN prediction
            cnn_input = self.preprocess_frame(frame)
            prediction = self.model.predict(cnn_input, verbose=0)
            predicted_class = self.class_names[np.argmax(prediction)]
            confidence = np.max(prediction)
            
            # MediaPipe landmarks
            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                detected_any = True
                
                # Rule-based detection
                rb_posture = self.detect_posture_rule_based(results.pose_landmarks.landmark)
                
                # Use CNN if confidence is high, otherwise rule-based
                posture = predicted_class if confidence > 0.65 else rb_posture
                
                # Handle transitional frames
                if posture is None and last_posture is not None:
                    posture = last_posture
                
                # Check correctness
                correct = self.check_posture_correctness(posture, results.pose_landmarks.landmark) if posture else False
                last_posture = posture
                
                # Display results with simplified format
                status = "Correct" if correct else "Incorrect"
                display_name = "Arm Raise" if posture == "Arm_Raise" else "Squats"
                
                print(f"Frame {frame_count}: {display_name} - {status}")
                
                # Draw pose landmarks on frame
                if show_video or save_output:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    
                    # Add simplified text overlay
                    text = f"{display_name} - {status}"
                    color = (0, 255, 0) if correct else (0, 0, 255)
                    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                               1.2, color, 3)
            
            # Show or save frame
            if show_video:
                cv2.imshow('Posture Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            if save_output:
                out_writer.write(frame)
        
        cap.release()
        if save_output:
            out_writer.release()
            print(f"\n✅ Output video saved to: {save_output}")
            
        if show_video:
            cv2.destroyAllWindows()
        
        if not detected_any:
            print("\n❌ No valid postures detected in the video")
        else:
            print("\n✅ Analysis complete!")
        
    def analyze_image(self, image_path, show_result=True, save_output=None):
        """
        Analyze a single image for posture detection
        
        Args:
            image_path: Path to input image
            show_result: Whether to display the result
            save_output: Path to save output image (optional)
        """
        frame = cv2.imread(image_path)
        
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        print(f"\nAnalyzing image: {image_path}")
        print("="*60)
        
        if not self.check_if_human(frame):
            print("❌ No human detected in image")
            return
        
        # CNN prediction
        cnn_input = self.preprocess_frame(frame)
        prediction = self.model.predict(cnn_input, verbose=0)
        predicted_class = self.class_names[np.argmax(prediction)]
        confidence = np.max(prediction)
        
        # MediaPipe landmarks
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            # Rule-based detection
            rb_posture = self.detect_posture_rule_based(results.pose_landmarks.landmark)
            
            # Use CNN if confidence is high, otherwise rule-based
            posture = predicted_class if confidence > 0.65 else rb_posture
            
            # Check correctness
            correct = self.check_posture_correctness(posture, results.pose_landmarks.landmark) if posture else False
            
            # Display results with simplified format
            status = "Correct" if correct else "Incorrect"
            display_name = "Arm Raise" if posture == "Arm_Raise" else "Squats"
            
            print(f"Detected Pose: {display_name} — Status: {status}")
            
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Add simplified text overlay
            text = f"{display_name} - {status}"
            color = (0, 255, 0) if correct else (0, 0, 255)
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.3, color, 3)
            
            if show_result:
                cv2.imshow('Posture Detection', frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            if save_output:
                cv2.imwrite(save_output, frame)
                print(f"✅ Output saved to: {save_output}")


def main():
    parser = argparse.ArgumentParser(description='Posture classification prediction')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained .keras model')
    parser.add_argument('--class_names', type=str, default=None,
                       help='Path to class_names.txt file')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input video or image')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output video/image')
    parser.add_argument('--show', action='store_true',
                       help='Display video/image (requires display)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = PosturePredictor(args.model, args.class_names)
    
    # Check if input is video or image
    input_ext = os.path.splitext(args.input)[1].lower()
    
    if input_ext in ['.mp4', '.avi', '.mov', '.mkv']:
        predictor.analyze_video(args.input, show_video=args.show, save_output=args.output)
    elif input_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        predictor.analyze_image(args.input, show_result=args.show, save_output=args.output)
    else:
        print(f"❌ Unsupported file format: {input_ext}")


if __name__ == "__main__":
    main()

