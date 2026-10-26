from typing import Optional

import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase

from . import config
from .drawing import PoseDrawer
from .pipeline import PosePipeline


class PoseProcessor(VideoProcessorBase):
    """Frame flow: raw frame -> (optional mirror) -> pipeline -> draw -> (optional mirror) -> status text."""

    def __init__(self):
        self.pipeline = PosePipeline()
        self.drawer = PoseDrawer()
        self.error: Optional[str] = None

    @property
    def model_name(self) -> str:
        return self.pipeline.model_name

    @model_name.setter
    def model_name(self, name: str) -> None:
        self.pipeline.model_name = name

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        if config.MIRROR_INPUT:
            img = np.ascontiguousarray(cv2.flip(img, 1))

        lines = []
        try:
            result = self.pipeline.process(img)
            self.drawer.draw_pose(img, result)
            lines = self.drawer.status_lines(result)
            self.error = None
        except Exception as e:                               # keep the stream alive, show the error in the app
            self.error = f"{type(e).__name__}: {e}"
            lines = ["processing error (see app)"]

        if config.MIRROR_DISPLAY:
            img = np.ascontiguousarray(cv2.flip(img, 1))
        self.drawer.draw_status(img, lines)                  # text after the flips so it is not mirrored
        return av.VideoFrame.from_ndarray(img, format="rgb24")

    def __del__(self):
        try:
            self.pipeline.close()
        except Exception:
            pass