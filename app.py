import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mediapipe_pose_landmarker_path = "models/pose_landmarker.task"

# Load and Cache MediaPipe Detector
@st.cache_resource
def load_pose_detector(model_path):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return vision.PoseLandmarker.create_from_options(options)

detector = load_pose_detector(mediapipe_pose_landmarker_path)

# Standard MediaPipe 33-landmark body connections for drawing the skeleton
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Upper body / Arms
    (11, 23), (12, 24), (23, 24),                     # Torso
    (23, 25), (25, 27), (24, 26), (26, 28)            # Legs
]

def draw_landmarks(frame, mp_points):
    """Draws the 33 MediaPipe joints and skeletal connections on the BGR frame."""
    # Draw bones
    for idx1, idx2 in POSE_CONNECTIONS:
        pt1 = tuple(mp_points[idx1].astype(int))
        pt2 = tuple(mp_points[idx2].astype(int))
        cv2.line(frame, pt1, pt2, (255, 200, 0), 2)
        
    # Draw joints
    for x, y in mp_points:
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)


st.set_page_config(page_title="Rehab Pose Correction", layout="centered")
st.title("Physiotherapy Exercise Tracker")

# Exercise Selection Dropdown
EXERCISES = {
    "Ex1": "Ex1 - Arm Abduction",
    "Ex2": "Ex2 - Arms V/W Shape",
    "Ex3": "Ex3 - Push-up",
    "Ex4": "Ex4 - Standing Side Leg Raise",
    "Ex5": "Ex5 - Lunge",
    "Ex6": "Ex6 - Squat",
}

selected_ex = st.selectbox(
    "Select Exercise:",
    options=list(EXERCISES.keys()),
    format_func=lambda x: EXERCISES[x]
)

run_camera = st.checkbox("Start Webcam")
frame_placeholder = st.empty()

if run_camera:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Error: Could not access the webcam.")
    else:
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame.")
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = detector.detect(mp_image)
            
            if detection_result.pose_landmarks:
                landmarks = detection_result.pose_landmarks[0]
                mp_points = np.array([[lm.x * w, lm.y * h] for lm in landmarks])
                draw_landmarks(frame_rgb, mp_points)
            
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
        cap.release()
else:
    st.info("Check 'Start Webcam' above to begin tracking your pose.")