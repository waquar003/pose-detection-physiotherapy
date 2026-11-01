from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from . import config

ACTS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
IN_DIM = 66 + 33


class PoseMapper(nn.Module):
    """Same class as in training: linear(x) + MLP(x). Input = 66 normalized coords + 33 visibility mask."""
    def __init__(self, hidden, activation, dropout):
        super().__init__()
        self.linear = nn.Linear(IN_DIM, 52)
        layers, prev = [], IN_DIM
        for h in hidden:
            layers += [nn.Linear(prev, h), ACTS[activation](), nn.Dropout(dropout)]
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 52)

    def forward(self, x):
        return self.linear(x) + self.head(self.mlp(x))


@dataclass
class MapResult:
    joints: np.ndarray      # (26, 2) pixels
    joint_ok: np.ndarray    # (26,) bool, False where a required landmark group is hidden
    lm_vis: np.ndarray      # (33,) bool, landmarks used as input
    hidden: set             # names of hidden landmark groups


class Mapper:
    def __init__(self, ckpt):
        self.model = PoseMapper(**ckpt["cfg"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.K = ckpt["K"]                       # fallback-anchor ratios fitted on the training set
        self.groups = ckpt["groups"]             # group name -> landmark indices
        self.needs = ckpt["joint_needs"]         # joint index -> groups that must be visible

    def hidden_groups(self, pts, vis, w, h) -> set:
        """A group is hidden if its mean visibility is low or most of its landmarks are outside the frame."""
        inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        return {name for name, idx in self.groups.items()
                if vis[idx].mean() < config.VIS_THR or inside[idx].mean() < 0.5}

    def map(self, pts, vis, w, h) -> Optional[MapResult]:
        """pts (33,2) pixels, vis (33,) visibility. None if neither shoulders nor hips are visible (no anchor)."""
        pts = pts.astype(np.float32)
        hidden = self.hidden_groups(pts, vis, w, h)

        lm_vis = np.ones(33, dtype=bool)
        for name in hidden:
            lm_vis[self.groups[name]] = False
        joint_ok = np.array([not any(n in hidden for n in self.needs[j]) for j in range(26)])

        sh_ok = lm_vis[11] and lm_vis[12]
        hp_ok = lm_vis[23] and lm_vis[24]
        if not (sh_ok or hp_ok):
            return None
        sh, hp = pts[[11, 12]], pts[[23, 24]]
        if sh_ok and hp_ok:
            center = np.concatenate([sh, hp]).mean(0)
            scale = np.linalg.norm(sh - hp, axis=-1).mean()
        elif sh_ok:
            center = sh.mean(0)
            scale = np.linalg.norm(sh[0] - sh[1]) * self.K["sh"]
        else:
            center = hp.mean(0)
            scale = np.linalg.norm(hp[0] - hp[1]) * self.K["hp"]
        scale = max(float(scale), 1.0)

        xn = (pts - center) / scale * lm_vis[:, None]
        x = np.concatenate([xn.flatten(), lm_vis.astype(np.float32)]).astype(np.float32)
        with torch.no_grad():
            out = self.model(torch.from_numpy(x)[None]).numpy().reshape(26, 2)
        return MapResult(joints=out * scale + center, joint_ok=joint_ok, lm_vis=lm_vis, hidden=hidden)


@lru_cache(maxsize=None)
def get_mapper(name: str) -> Mapper:
    ckpt = torch.load(str(config.MLP_WEIGHTS[name]), map_location="cpu", weights_only=True)
    return Mapper(ckpt)