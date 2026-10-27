from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from . import config
from . import utils
from . import ghost_runtime as gr

@dataclass
class CoachOutput:
    status: str = "idle"          # idle | no_pose | joints_hidden | ok
    phase: float = float("nan")   # 0..1 position inside the rep
    conf: float = float("nan")    # phase confidence
    dist: float = float("nan")    # RMS z-distance between the user and the ghost pose (torso-normalised features)
    missing: List[str] = field(default_factory=list)
    ghost_norm: Optional[np.ndarray] = None   # (26,2) hips-centred, torso units
    ghost_px: Optional[np.ndarray] = None     # (26,2) pixels, ready to draw


class FixedRate:
    """Resamples joints to a fixed rate by linear interpolation (models were trained on 30 fps reps)."""

    def __init__(self, fps):
        self.dt = 1.0 / fps
        self.reset()

    def reset(self):
        self.t0 = self.p0 = self.next = None

    def feed(self, t, pts):
        if self.p0 is None or t <= self.t0 or t - self.t0 > 0.5:
            self.t0, self.p0, self.next = t, pts, t + self.dt
            return [pts]
        out = []
        while self.next <= t:
            a = (self.next - self.t0) / (t - self.t0)
            out.append(self.p0 + a * (pts - self.p0))
            self.next += self.dt
        self.t0, self.p0 = t, pts
        return out


class ExerciseCoach:
    """One exercise = one ghost pipeline (tracker -> enforcer -> ghost). Tracking starts at the first frame."""
    MISS_RESET = 45     # frames without usable joints before tracking restarts from the beginning

    def __init__(self, exercise):
        self.exercise = exercise
        self.arch, pipe = gr.load_pipeline(config.GHOST_MODELS[exercise])
        if pipe.ctx.ex != exercise:
            raise ValueError(f"{config.GHOST_MODELS[exercise].name} was trained for {pipe.ctx.ex}, not {exercise}")
        self.live = gr.LiveGhost(pipe, init="start")
        self.required = utils.required_joints(exercise)
        self.draw_joints = self.required
        self.feeder = FixedRate(config.COACH_FPS)
        self.missed, self.last = 0, None

    def reset(self):
        self.live.reset()
        self.feeder.reset()
        self.missed, self.last = 0, None

    def skip(self):
        self._miss()
        return CoachOutput(status="no_pose")

    def _miss(self):
        self.missed += 1
        if self.missed == self.MISS_RESET:
            self.reset()
            self.missed = self.MISS_RESET     # keep counting so reset is not repeated every frame

    def push(self, joints, joint_ok, t):
        missing = [i for i in self.required if not joint_ok[i]]
        if missing:
            self._miss()
            return CoachOutput(status="joints_hidden", missing=[config.JN_NAMES[i] for i in missing])
        
        self.missed = 0
        pts = np.asarray(joints, dtype=np.float32)
        for p in self.feeder.feed(t, pts):
            self.last = self.live.push_frame(p, k=1)
        
        o = self.last
        nb = o["neighbours"][0]
        return CoachOutput(status="ok", phase=o["phase"], conf=o["phase_conf"], dist=nb["z_dist"],
                           ghost_norm=nb["pose"], ghost_px=self._to_pixels(nb["pose"], pts))

    @staticmethod
    def _to_pixels(ghost_norm, pts):
        hips = pts[config.JN["Hips"]]
        tl = max(float(np.linalg.norm(pts[config.JN["Spine"]] - hips)), 1e-6)
        return ghost_norm * tl + hips          # follows the user's current position and size


def build_coach(exercise):
    """returns (coach or None, info text)"""
    path = config.GHOST_MODELS.get(exercise)
    if path is None or not path.exists():
        return None, f"Model file not found: {path.name if path else exercise}"
    try:
        coach = ExerciseCoach(exercise)
        return coach, f"Loaded {path.name} ({coach.arch}). Tracking starts from the first frame, so begin in the starting pose."
    except Exception as e:
        return None, f"Could not load model: {type(e).__name__}: {e}"


def output_lines(o: CoachOutput) -> List[str]:
    if o.status == "joints_hidden":
        return ["coach paused, joints not visible: " + ", ".join(o.missing[:4])]
    if o.status == "ok":
        return [f"phase {o.phase * 100:.0f}%  deviation {o.dist:.2f}"]
    return []