from typing import List

import cv2
import numpy as np

from . import config
from .pipeline import FrameResult


class PoseDrawer:
    """All overlay drawing. Works in place on an RGB frame."""

    BONE_COLOR = (255, 200, 0)
    LM_USED_COLOR = (0, 120, 255)
    LM_HIDDEN_COLOR = (150, 150, 150)
    JOINT_COLOR = (0, 255, 0)

    def __init__(self, bones=config.JOINT26_CONNECTIONS):
        self.bones = bones

    def draw_pose(self, img: np.ndarray, result: FrameResult) -> None:
        if not result.ok:
            return
        m, lm = result.mapped, result.landmarks
        for a, b in self.bones:                              # bones only between trusted joints
            if m.joint_ok[a] and m.joint_ok[b]:
                cv2.line(img, tuple(m.joints[a].astype(int)), tuple(m.joints[b].astype(int)), self.BONE_COLOR, 2)
        for (x, y), used in zip(lm.pts, m.lm_vis):           # MediaPipe: filled = used, ring = hidden
            if used:
                cv2.circle(img, (int(x), int(y)), 4, self.LM_USED_COLOR, -1)
            else:
                cv2.circle(img, (int(x), int(y)), 5, self.LM_HIDDEN_COLOR, 1)
        for (x, y), ok in zip(m.joints, m.joint_ok):         # mapped joints, only if trusted
            if ok:
                cv2.circle(img, (int(x), int(y)), 5, self.JOINT_COLOR, -1)

    def draw_status(self, img: np.ndarray, lines: List[str]) -> None:
        for i, text in enumerate(lines):
            y = 25 + 25 * i
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    @staticmethod
    def status_lines(result: FrameResult) -> List[str]:
        if not result.ok:
            return [result.message]
        return [f"hidden: {sorted(result.mapped.hidden) or 'nothing'}",
                f"min visibility: {result.landmarks.vis.min():.2f}"]