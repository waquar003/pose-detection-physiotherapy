from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import config
from .pose_extractor import Landmarks, PoseExtractor
from .pose_mapper import MapResult, get_mapper


@dataclass
class FrameResult:
    landmarks: Optional[Landmarks] = None
    mapped: Optional[MapResult] = None
    message: str = ""          # why nothing was mapped

    @property
    def ok(self) -> bool:
        return self.mapped is not None


class PosePipeline:
    """frame -> MediaPipe landmarks -> mapped 26 joints. Owns the extractor, selects the mapper model."""

    def __init__(self, model_name: str = config.DEFAULT_MODEL):
        self.model_name = model_name
        self.extractor = PoseExtractor(config.MP_MODEL_PATH)

    def process(self, frame_rgb: np.ndarray) -> FrameResult:
        lm = self.extractor.extract(frame_rgb)
        if lm is None:
            return FrameResult(message="no person detected")

        mapped = get_mapper(self.model_name).map(lm.pts, lm.vis, lm.w, lm.h)
        if mapped is None:
            return FrameResult(landmarks=lm, message="shoulders and hips not visible, cannot map")

        # SMOOTHING HOOK: temporal filtering of `lm` (before mapping) or `mapped.joints` (after) goes here later.
        return FrameResult(landmarks=lm, mapped=mapped)

    def close(self) -> None:
        self.extractor.close()