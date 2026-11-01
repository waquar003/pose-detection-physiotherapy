from typing import Iterable, List, Optional

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
    GHOST_COLOR = (255, 0, 255)

    def __init__(self, bones=config.JOINT26_CONNECTIONS):
        self.bones = bones

    def draw_pose(self, img: np.ndarray, result: FrameResult) -> None:
        if not result.ok:
            return
        m, lm = result.mapped, result.landmarks
        for a, b in self.bones:
            if m.joint_ok[a] and m.joint_ok[b]:
                cv2.line(img, tuple(m.joints[a].astype(int)), tuple(m.joints[b].astype(int)), self.BONE_COLOR, 2)
        for (x, y), used in zip(lm.pts, m.lm_vis):         # MediaPipe: filled = used, ring = hidden
            if used:
                cv2.circle(img, (int(x), int(y)), 4, self.LM_USED_COLOR, -1)
            else:
                cv2.circle(img, (int(x), int(y)), 5, self.LM_HIDDEN_COLOR, 1)
        for (x, y), ok in zip(m.joints, m.joint_ok):       # mapped joints, only if trusted
            if ok:
                cv2.circle(img, (int(x), int(y)), 5, self.JOINT_COLOR, -1)

    def draw_ghost(self, img: np.ndarray, ghost_px: np.ndarray, joints: Optional[Iterable[int]] = None,
                   alpha: float = 0.55) -> None:
        """Semi-transparent correct pose. `joints` limits the drawing to the joints the exercise uses."""
        if ghost_px is None or not np.isfinite(ghost_px).all():
            return
        keep = set(range(26)) if joints is None else {int(j) for j in joints}
        pt = lambda j: (int(ghost_px[j, 0]), int(ghost_px[j, 1]))
        layer = img.copy()
        for a, b in self.bones:
            if a in keep and b in keep:
                cv2.line(layer, pt(a), pt(b), self.GHOST_COLOR, 3, cv2.LINE_AA)
        for j in keep:
            cv2.circle(layer, pt(j), 5, self.GHOST_COLOR, -1, cv2.LINE_AA)
        cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, img)

    def draw_status(self, img: np.ndarray, lines: List[str]) -> None:
        for i, text in enumerate(lines):
            y = 25 + 25 * i
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    @staticmethod
    def status_lines(result: FrameResult) -> List[str]:
        if not result.ok:
            return [result.message]
        return [f"hidden: {sorted(result.mapped.hidden) or 'nothing'}"]