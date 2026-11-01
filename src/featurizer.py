import numpy as np
from . import config
from .utils import normalize_frame, frame_angles

class CausalFeaturizer:
    def __init__(self, exercise):
        self.pos_idx = [config.JN[j] for j in config.EX_JOINTS[exercise]]
        self.ang_idx = config.EX_ANG[exercise]
        self.reset()

    def reset(self):
        self.s, self.hist = None, []

    def push(self, pts):
        p = np.nan_to_num(np.asarray(pts, dtype=np.float64)[:, :2])
        self.s = p.copy() if self.s is None else config.EMA_ALPHA * p + (1 - config.EMA_ALPHA) * self.s
        n = normalize_frame(self.s)
        ang = frame_angles(n)[self.ang_idx]
        self.hist.append(ang)
        if len(self.hist) > config.VEL_LAG + 1:
            self.hist.pop(0)
        return np.concatenate([n[self.pos_idx].ravel(), ang, ang - self.hist[0]]), n