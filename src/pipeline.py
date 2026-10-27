from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import config
from .pose_extractor import Landmarks, PoseExtractor
from .pose_mapper import MapResult, get_mapper
from .smoothing import JointSmoother


@dataclass
class FrameResult:
    landmarks: Optional[Landmarks] = None
    mapped: Optional[MapResult] = None
    message: str = ""          # why nothing was mapped

    @property
    def ok(self) -> bool:
        return self.mapped is not None


class PosePipeline:
    """frame -> MediaPipe landmarks -> mapped 26 joints -> smoothed joints. Owns the extractor and the smoother."""

    def __init__(self, model_name: str = config.DEFAULT_MODEL):
        self._model_name = model_name
        self.extractor = PoseExtractor(config.MP_MODEL_PATH)
        self.smoother = JointSmoother(config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA, config.ONE_EURO_D_CUTOFF)

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, name: str) -> None:
        if name != self._model_name:
            self._model_name = name
            self.smoother.reset()          # a different mapper has a different output, do not blend across it

    def reset(self) -> None:
        self.smoother.reset()

    def process(self, frame_rgb: np.ndarray, t: float) -> FrameResult:
        """t: frame timestamp in seconds (monotonic)."""
        lm = self.extractor.extract(frame_rgb)
        if lm is None:
            return FrameResult(message="no person detected")

        mapped = get_mapper(self._model_name).map(lm.pts, lm.vis, lm.w, lm.h)
        if mapped is None:
            return FrameResult(landmarks=lm, message="shoulders and hips not visible, cannot map")

        if config.SMOOTH_JOINTS:
            raw = mapped.joints
            torso = float(np.linalg.norm(raw[1] - raw[0]))        # Hips -> Spine, same scale the coach uses
            mapped.joints = self.smoother(t, raw, mapped.joint_ok, torso)

        return FrameResult(landmarks=lm, mapped=mapped)

    def close(self) -> None:
        self.extractor.close()