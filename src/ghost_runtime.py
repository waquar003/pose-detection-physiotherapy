"""Runtime for the ghost_lab pipelines: Tracker -> Enforcer -> Ghost, wrapped by LiveGhost."""

import joblib
import numpy as np
from joblib.numpy_pickle import NumpyUnpickler

from .featurizer import CausalFeaturizer
from .trackers import Tracker, NullTracker, RegressorTracker, HMMTracker, StreamingDTWTracker, EmbeddingTracker, EnsembleTracker
from .enforcers import Enforcer, NoEnforcer, ForwardClamp, AlphaBeta
from .ghosts import Ghost, ConstrainedNN, UnconstrainedNN, TemplateGhost, EmbedNN
from .utils import circ_signed

try:
    from .trackers import TemporalEncoder
except ImportError:
    pass


class Ctx:
    pass

class Pipeline:
    pass


class PhaseLead:
    """Estimates phase speed from the enforced phase and returns phase + speed * lead_s (latency compensation)."""
    def __init__(self, lead_s=0.15, smooth=0.15, max_lead=0.08, fps=30.0):
        self.lead_s, self.smooth, self.max_lead, self.dt = lead_s, smooth, max_lead, 1.0 / fps
        self.reset()

    def reset(self):
        self.phi, self.v = None, 0.0

    def __call__(self, phi):
        if self.phi is not None:
            v_inst = circ_signed(phi, self.phi) / self.dt           # phase units per second
            self.v += self.smooth * (np.clip(v_inst, 0.0, 1.0) - self.v)   # forward motion only
        self.phi = phi
        return (phi + float(np.clip(self.v * self.lead_s, 0.0, self.max_lead))) % 1.0


class LiveGhost:
    """push_frame(26x2 pixel joints) -> dict(phase, phase_raw, phase_conf, locked, user_pose, neighbours)"""

    def __init__(self, pipe, init="start"):
        self.p, self.init = pipe, init
        self.f = CausalFeaturizer(pipe.ctx.ex)
        self.lead = PhaseLead()
        self.reset()

    def reset(self):
        self.f.reset()
        self.lead.reset()
        self.p.tracker.reset(self.init)
        self.p.enf.reset(self.p.ctx, self.init)

    def push_frame(self, pts, k=1):
        c = self.p.ctx
        feat, nrm = self.f.push(pts)
        z = np.clip((feat - c.mu) / c.sd, -8, 8)
        m, conf = self.p.tracker.step(z)
        phi, ok = self.p.enf.step(m, conf)
        phi_g = self.lead(phi)                       # ghost is looked up slightly ahead of the tracked phase
        return dict(phase=phi, phase_raw=m, phase_conf=conf, locked=ok, user_pose=nrm,
                    neighbours=self.p.ghost.query(z, phi_g, self.p.tracker.last_extras, k))

# The notebook pickled these classes as __main__.<Name>. Instead of patching sys.modules["__main__"]
# (which Streamlit replaces on every rerun), redirect those lookups inside joblib's unpickler.
# Because models were pickled under __main__, we route Joblib to find them properly.
_MINE = {
    "Ctx": Ctx, "Pipeline": Pipeline,
    "Tracker": Tracker, "NullTracker": NullTracker, "RegressorTracker": RegressorTracker, 
    "HMMTracker": HMMTracker, "StreamingDTWTracker": StreamingDTWTracker, 
    "EmbeddingTracker": EmbeddingTracker, "EnsembleTracker": EnsembleTracker,
    "Enforcer": Enforcer, "NoEnforcer": NoEnforcer, "ForwardClamp": ForwardClamp, "AlphaBeta": AlphaBeta,
    "Ghost": Ghost, "ConstrainedNN": ConstrainedNN, "UnconstrainedNN": UnconstrainedNN, 
    "TemplateGhost": TemplateGhost, "EmbedNN": EmbedNN
}

if "TemporalEncoder" in globals():
    _MINE["TemporalEncoder"] = TemporalEncoder


if not getattr(NumpyUnpickler, "_ghost_patched", False):
    _orig_find_class = NumpyUnpickler.find_class

    def _find_class(self, module, name):
        if module == "__main__" and name in _MINE:
            return _MINE[name]
        return _orig_find_class(self, module, name)

    NumpyUnpickler.find_class = _find_class
    NumpyUnpickler._ghost_patched = True


def _single_thread(t):
    m = getattr(t, "model", None)
    if hasattr(m, "n_jobs"):
        m.n_jobs = 1
    for s in getattr(t, "ms", []):
        _single_thread(s)


def load_pipeline(path):
    """returns (arch_name, pipeline). Each call returns a fresh object: the trackers are stateful, never share them."""
    obj = joblib.load(str(path))
    pipe = obj["pipeline"]
    _single_thread(pipe.tracker)
    return obj["arch"], pipe