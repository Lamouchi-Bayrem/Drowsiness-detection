import cv2
import numpy as np
from scipy.spatial import distance
from ultralytics import YOLO
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load YOLOv8 face detection model
model = YOLO('yolov8n-face.pt')

# Landmark indices for eyes (MediaPipe)
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Landmarks for head pose estimation (nose, chin, left eye, right eye, left mouth, right mouth)
HEAD_POSE_LANDMARKS = [1, 199, 33, 263, 61, 291]

# Eye Aspect Ratio (EAR) calculation
def eye_aspect_ratio(eye_landmarks):
    A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Head pose calculation (simple pitch estimation)
def head_pitch_angle(face_landmarks, image_shape):
    # Get relevant landmarks
    nose = face_landmarks[HEAD_POSE_LANDMARKS[0]]
    chin = face_landmarks[HEAD_POSE_LANDMARKS[1]]
    
    # Convert to image coordinates
    nose_x, nose_y = int(nose.x * image_shape[1]), int(nose.y * image_shape[0])
    chin_x, chin_y = int(chin.x * image_shape[1]), int(chin.y * image_shape[0])
    
    # Calculate vertical distance (simple pitch estimation)
    vertical_dist = chin_y - nose_y
    return vertical_dist

# Drowsiness thresholds
EAR_THRESHOLD = 0.25      # Eye closure threshold
HEAD_POSE_THRESHOLD = 200  # Head nod threshold (pixels)
CONSEC_FRAMES = 15        # Frames before alarm triggers

# Initialize webcam
cap = cv2.VideoCapture(0)
counter = 0
alarm_on = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_h, frame_w = frame.shape[:2]
    
    # Detect faces with YOLOv8
    results = model(frame, verbose=False)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_roi = frame[y1:y2, x1:x2]
            
            # Process face ROI with MediaPipe
            face_results = face_mesh.process(rgb_frame[y1:y2, x1:x2])
            
            if face_results.multi_face_landmarks:
                landmarks = face_results.multi_face_landmarks[0].landmark
                
                # Extract eye landmarks
                left_eye = []
                right_eye = []
                
                for idx in LEFT_EYE_INDICES:
                    landmark = landmarks[idx]
                    px, py = int(landmark.x * (x2-x1)) + x1, int(landmark.y * (y2-y1)) + y1
                    left_eye.append((px, py))
                    cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)
                
                for idx in RIGHT_EYE_INDICES:
                    landmark = landmarks[idx]
                    px, py = int(landmark.x * (x2-x1)) + x1, int(landmark.y * (y2-y1)) + y1
                    right_eye.append((px, py))
                    cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)
                
                # Calculate EAR for both eyes
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Calculate head pose (pitch)
                pitch = head_pitch_angle(landmarks, frame.shape)
                
                # Visual feedback
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (x1, y1 - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Head Pitch: {pitch:.1f}", (x1, y1 - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Drowsiness detection (eyes closed OR head nodding)
                if avg_ear < EAR_THRESHOLD or pitch > HEAD_POSE_THRESHOLD:
                    counter += 1
                    if counter >= CONSEC_FRAMES:
                        if not alarm_on:
                            alarm_on = True
                            print("ALERT! Drowsiness Detected!")
                        cv2.putText(frame, "DROWSY!", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        # Draw red bounding box when drowsy
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                else:
                    counter = 0
                    alarm_on = False
                    # Draw green bounding box when awake
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Display status
    status = "Awake" if not alarm_on else "Drowsy!"
    color = (0, 255, 0) if not alarm_on else (0, 0, 255)
    cv2.putText(frame, f"Status: {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Show frame
    cv2.imshow("Drowsiness Detection (Head Pose + Eyes)", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()