import time
from typing import Optional

import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase

from . import config
from .coach import build_coach, output_lines
from .drawing import PoseDrawer
from .pipeline import PosePipeline


class PoseProcessor(VideoProcessorBase):
    """raw frame -> MediaPipe -> MLP mapper (26 joints) -> ghost model -> draw skeleton + ghost -> mirror -> text."""

    def __init__(self):
        self.pipeline = PosePipeline()
        self.drawer = PoseDrawer()
        self.error: Optional[str] = None
        self.coach = None
        self.coach_info = ""
        self._exercise = None
        self._restart = False                  # set by the UI thread, executed inside recv
        self._t_prev, self.fps = None, 0.0

    # ---- settings written by the Streamlit thread
    @property
    def model_name(self) -> str:
        return self.pipeline.model_name

    @model_name.setter
    def model_name(self, name: str) -> None:
        self.pipeline.model_name = name

    @property
    def exercise(self):
        return self._exercise

    @exercise.setter
    def exercise(self, ex: str) -> None:
        if ex == self._exercise:
            return
        self._exercise = ex
        self.coach = None
        self.coach, self.coach_info = build_coach(ex)

    def request_restart(self) -> None:
        self._restart = True

    # ---- video thread
    def _update_fps(self, t):
        if self._t_prev is not None and t > self._t_prev:
            inst = 1.0 / (t - self._t_prev)
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
        self._t_prev = t

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        t = frame.time if frame.time is not None else time.monotonic()
        self._update_fps(t)
        if config.MIRROR_INPUT:
            img = np.ascontiguousarray(cv2.flip(img, 1))

        coach = self.coach
        lines = [f"{self.fps:.0f} fps"]
        try:
            if self._restart:
                self._restart = False
                if coach is not None:
                    coach.reset()

            result = self.pipeline.process(img)
            self.drawer.draw_pose(img, result)
            lines += self.drawer.status_lines(result)

            if coach is not None:
                out = coach.push(result.mapped.joints, result.mapped.joint_ok, t) if result.ok else coach.skip()
                if out.ghost_px is not None:
                    self.drawer.draw_ghost(img, out.ghost_px, coach.draw_joints)
                lines += output_lines(out)
            self.error = None
        except Exception as e:                      # keep the stream alive, show the error in the app
            self.error = f"{type(e).__name__}: {e}"
            lines.append("processing error (see app)")

        if config.MIRROR_DISPLAY:
            img = np.ascontiguousarray(cv2.flip(img, 1))
        self.drawer.draw_status(img, lines)         # text after the flip so it is not mirrored
        return av.VideoFrame.from_ndarray(img, format="rgb24")

    def __del__(self):
        try:
            self.pipeline.close()
        except Exception:
            pass