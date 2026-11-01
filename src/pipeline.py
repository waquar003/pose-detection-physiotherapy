from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import config
from .pose_extractor import Landmarks, PoseExtractor
from .pose_mapper import MapResult, get_mapper
from .smoothing import JointSmoother, LandmarkStabilizer


@dataclass
class FrameResult:
    landmarks: Optional[Landmarks] = None
    mapped: Optional[MapResult] = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.mapped is not None


class PosePipeline:
    """frame -> MediaPipe -> landmark stabiliser -> MLP mapper (26 joints) -> optional joint smoothing."""

    def __init__(self, model_name: str = config.DEFAULT_MODEL):
        self._model_name = model_name
        self.extractor = PoseExtractor(config.MP_MODEL_PATH)
        self.stab = LandmarkStabilizer(config.VIS_THR, config.LM_ON_FRAMES, config.LM_OFF_FRAMES, config.LM_DECAY,
                                       config.LM_MAX_JUMP, config.LM_MIN_CUTOFF, config.LM_BETA)
        self.smoother = JointSmoother(config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA, config.ONE_EURO_D_CUTOFF)

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, name: str) -> None:
        if name != self._model_name:
            self._model_name = name
            self.reset()

    def reset(self) -> None:
        self.stab.reset()
        self.smoother.reset()

    def process(self, frame_rgb: np.ndarray, t: float) -> FrameResult:
        lm = self.extractor.extract(frame_rgb)
        mapper = get_mapper(self._model_name)

        if config.STABILIZE_LANDMARKS:
            if lm is None:
                st = self.stab.update(t, None, None, None, None, mapper.groups)
            else:
                st = self.stab.update(t, lm.pts, lm.vis, lm.w, lm.h, mapper.groups)
            if st is None:
                return FrameResult(message="no person detected")
            lm = Landmarks(pts=st[0], vis=st[1], w=st[2], h=st[3])
        elif lm is None:
            return FrameResult(message="no person detected")

        mapped = mapper.map(lm.pts, lm.vis, lm.w, lm.h)
        if mapped is None:
            return FrameResult(landmarks=lm, message="shoulders and hips not visible, cannot map")

        if config.SMOOTH_JOINTS:
            raw = mapped.joints
            torso = float(np.linalg.norm(raw[1] - raw[0]))
            mapped.joints = self.smoother(t, raw, mapped.joint_ok, torso)
        return FrameResult(landmarks=lm, mapped=mapped)

    def close(self) -> None:
        self.extractor.close()