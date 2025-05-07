import cv2
import numpy as np
from scipy.spatial import distance
from ultralytics import YOLO
import mediapipe as mp
import time

# Initialize models
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
model = YOLO('yolov8n-face.pt')

# Constants
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
EAR_THRESHOLD = 0.21  # Typical threshold for eye closure
YAWN_THRESHOLD = 0.5  # Mouth aspect ratio threshold
CALIBRATION_FRAMES = 30  # About 1 second at 30 FPS
ALARM_DURATION = 1.5  # Seconds before alarm triggers

class DrowsinessDetector:
    def __init__(self):
        self.ear_history = []
        self.calibrated = False
        self.state = "CALIBRATING"
        self.frame_count = 0
        self.alarm_start_time = 0
        self.alarm_active = False

    def update(self, ear, yawn_ratio):
        self.frame_count += 1
        
        # Calibration phase
        if not self.calibrated:
            self.ear_history.append(ear)
            
            if len(self.ear_history) >= CALIBRATION_FRAMES:
                self.calibrated = True
                self.state = "NORMAL"
                print(f"Calibration complete! Avg EAR: {np.mean(self.ear_history):.2f}")
                return self._get_status(ear, yawn_ratio)
        
        # Detection phase
        if ear < EAR_THRESHOLD or yawn_ratio > YAWN_THRESHOLD:
            if self.state != "DROWSY":
                if not self.alarm_active:
                    self.alarm_start_time = time.time()
                    self.alarm_active = True
                elif time.time() - self.alarm_start_time > ALARM_DURATION:
                    self.state = "DROWSY"
            else:
                self.alarm_active = True
        else:
            self.state = "NORMAL"
            self.alarm_active = False
        
        return self._get_status(ear, yawn_ratio)

    def _get_status(self, ear, yawn_ratio):
        return {
            'state': self.state,
            'ear': ear,
            'yawn': yawn_ratio,
            'calibrated': self.calibrated,
            'frames': self.frame_count,
            'alarm': self.alarm_active and self.state == "DROWSY"
        }

def get_ear(landmarks, eye_indices, frame_width, frame_height):
    points = [(landmarks[i].x * frame_width, landmarks[i].y * frame_height) 
              for i in eye_indices]
    
    if len(points) != 6:
        return 0.0
    
    # Calculate EAR
    A = distance.euclidean(points[1], points[5])
    B = distance.euclidean(points[2], points[4])
    C = distance.euclidean(points[0], points[3])
    ear = (A + B) / (2.0 * C) if C != 0 else 0.0
    return ear

def get_yawn_ratio(landmarks, frame_width, frame_height):
    # Simple mouth openness detection
    upper_lip = landmarks[13].y * frame_height
    lower_lip = landmarks[14].y * frame_height
    mouth_height = lower_lip - upper_lip
    return mouth_height / frame_height  # Normalized ratio

def draw_info(frame, status, face_bbox):
    # Status color coding
    color_map = {
        "CALIBRATING": (255, 165, 0),  # Orange
        "NORMAL": (0, 255, 0),         # Green
        "DROWSY": (0, 0, 255)          # Red
    }
    color = color_map.get(status['state'], (255, 255, 255))
    
    # Draw face bounding box
    if face_bbox:
        x1, y1, x2, y2 = face_bbox
        thickness = 4 if status['alarm'] else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Status text
    cv2.putText(frame, f"Status: {status['state']}", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Metrics
    cv2.putText(frame, f"EAR: {status['ear']:.2f} (Threshold: {EAR_THRESHOLD:.2f})", 
               (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Yawn: {status['yawn']:.2f}", (20, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Alarm indicator
    if status['alarm']:
        cv2.putText(frame, "ALARM!", (frame.shape[1] - 150, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Calibration progress
    if not status['calibrated']:
        progress = min(1.0, status['frames'] / CALIBRATION_FRAMES)
        cv2.rectangle(frame, (20, frame.shape[0] - 30), 
                     (int(20 + 200 * progress), frame.shape[0] - 20),
                     (255, 165, 0), -1)
        cv2.putText(frame, "Calibrating...", (20, frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

def sound_alarm():
    # Simple console beep (replace with actual alarm sound)
    print("\aALARM! Drowsiness detected!")

def main():
    detector = DrowsinessDetector()
    cap = cv2.VideoCapture(0)
    last_alarm_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Default status when no face detected
        status = {
            'state': "No face",
            'ear': 0,
            'yawn': 0,
            'calibrated': detector.calibrated,
            'frames': detector.frame_count,
            'alarm': False
        }
        face_bbox = None
        
        # Face detection
        faces = model(frame, verbose=False)[0]
        if len(faces.boxes) > 0:
            # Get largest face by confidence
            face = max(faces.boxes.data.tolist(), key=lambda x: x[4])
            if face[4] >= 0.5:  # Confidence threshold
                x1, y1, x2, y2 = map(int, face[:4])
                face_bbox = (x1, y1, x2, y2)
                
                # Face landmarks
                results = face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    
                    # Calculate metrics
                    left_ear = get_ear(landmarks, LEFT_EYE, frame_width, frame_height)
                    right_ear = get_ear(landmarks, RIGHT_EYE, frame_width, frame_height)
                    avg_ear = (left_ear + right_ear) / 2
                    yawn_ratio = get_yawn_ratio(landmarks, frame_width, frame_height)
                    
                    # Update detector
                    status = detector.update(avg_ear, yawn_ratio)
                    
                    # Draw eye landmarks
                    for i in LEFT_EYE + RIGHT_EYE:
                        point = (int(landmarks[i].x * frame_width), 
                                int(landmarks[i].y * frame_height))
                        cv2.circle(frame, point, 2, (0, 255, 0), -1)
        
        # Visual feedback
        draw_info(frame, status, face_bbox)
        
        # Trigger alarm if needed
        if status['alarm'] and time.time() - last_alarm_time > 1:
            sound_alarm()
            last_alarm_time = time.time()
        
        cv2.imshow('Drowsiness Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()