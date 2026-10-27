from dataclasses import dataclass
from typing import Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from . import config


@dataclass
class Landmarks:
    pts: np.ndarray   # (33, 2) pixels
    vis: np.ndarray   # (33,) visibility 0-1
    w: int            # frame width
    h: int            # frame height


class PoseExtractor:
    """MediaPipe PoseLandmarker wrapper. One instance per video stream."""

    def __init__(self, model_path):
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            output_segmentation_masks=False,
            min_pose_detection_confidence=config.MP_MIN_DETECTION_CONFIDENCE,
        )
        self._detector = vision.PoseLandmarker.create_from_options(options)

    def extract(self, frame_rgb: np.ndarray) -> Optional[Landmarks]:
        """frame_rgb: uint8 (H, W, 3) RGB. Returns None if no person is detected."""
        h, w, _ = frame_rgb.shape
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        result = self._detector.detect(image)
        if not result.pose_landmarks:
            return None
        lms = result.pose_landmarks[0]
        pts = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)
        vis = np.array([lm.visibility if lm.visibility is not None else 0.0 for lm in lms], dtype=np.float32)
        return Landmarks(pts=pts, vis=vis, w=w, h=h)

    def close(self) -> None:
        self._detector.close()