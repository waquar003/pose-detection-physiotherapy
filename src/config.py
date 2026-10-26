from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"

MP_MODEL_PATH = MODELS_DIR / "pose_landmarker.task"

# Copy these from Kaggle /kaggle/working into models/
MLP_WEIGHTS = {
    "C17": MODELS_DIR / "mapper_model_C17.pth",
    "C18": MODELS_DIR / "mapper_model_C18.pth",
    "combined": MODELS_DIR / "mapper_model_combined.pth",
}
DEFAULT_MODEL = "combined"

EXERCISES = {
    "Ex1": "Ex1 - Arm Abduction",
    "Ex2": "Ex2 - Arms V/W Shape",
    "Ex3": "Ex3 - Push-up",
    "Ex4": "Ex4 - Standing Side Leg Raise",
    "Ex5": "Ex5 - Lunge",
    "Ex6": "Ex6 - Squat",
}

# MediaPipe
MP_MIN_DETECTION_CONFIDENCE = 0.5

# A landmark group counts as hidden if its mean visibility is below this (check and tune it later)
VIS_THR = 0.3

# Mirroring (same behaviour as the current app: flip BEFORE detection, flip again for display)
MIRROR_INPUT = True      # flip the raw frame before detection and drawing
MIRROR_DISPLAY = True    # flip the finished frame before showing it
# Raw frames to the model, mirrored selfie view for the user:  MIRROR_INPUT = False, MIRROR_DISPLAY = True

# Drawing
JOINT26_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (6, 7), (7, 8), (8, 9), (11, 12), (12, 13), (13, 14),
    (0, 6), (0, 11), (1, 6), (1, 11),
    (0, 16), (0, 21), (16, 17), (17, 18), (18, 19), (19, 20), (21, 22), (22, 23), (23, 24), (24, 25),
    (4, 5), (9, 10), (14, 15),
]