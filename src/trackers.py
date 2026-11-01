import numpy as np
from .utils import rms, decode

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

class Tracker:
    last_extras = {}
    def reset(self, init="start"):
        pass

class NullTracker(Tracker):
    def step(self, z):
        self.last_extras = {}
        return 0.0, 1.0

class RegressorTracker(Tracker):
    def _feats(self, Z):
        idx = np.arange(len(Z))
        return np.concatenate([Z] + [Z - Z[np.maximum(idx - l, 0)] for l in self.lags], 1)

    def reset(self, init="start"):
        self.buf = []

    def step(self, z):
        self.buf = (self.buf + [z])[-(max(self.lags, default=0) + 1):]
        ph, c = decode(self.model.predict(self._feats(np.array(self.buf))[-1:]))
        self.last_extras = {}
        return float(ph[0]), float(c[0])

class HMMTracker(Tracker):
    def reset(self, init="start"):
        self.b = np.full(self.L, 1.0 / self.L)
        if init == "start":
            self.b = np.zeros(self.L)
            self.b[:3] = [0.6, 0.3, 0.1]

    def step(self, z):
        pred = sum(pk * np.roll(self.b, k) for k, pk in enumerate(self.p))
        pred = (1 - self.leak) * pred + self.leak / self.L
        d2 = (rms(z[None], self.Zt) / self.sig) ** 2
        ll = -self.temp * 0.5 * (self.nu + 1) * np.log1p(d2 / self.nu)
        b = pred * np.exp(ll - ll.max()) + 1e-12
        self.b = b / b.sum()
        j = int(self.b.argmax())
        offs = np.arange(-2, 3)
        w = self.b[(j + offs) % self.L]
        self.last_extras = {}
        return (float(((j + (w * offs).sum() / w.sum()) / (self.L - 1)) % 1.0),
                float(self.b[(j + np.arange(-3, 4)) % self.L].sum()))

class StreamingDTWTracker(Tracker):
    def reset(self, init="start"):
        j = np.arange(self.L)
        self.A = self.start_pen * self.sg * np.minimum(j, self.L - j) if init == "start" else np.zeros(self.L)

    def step(self, z):
        c = np.minimum(rms(z[None], self.Zt), self.cap_mult * self.sg)
        Ap = self.A * self.decay
        c1, c2 = np.roll(Ap, 1), np.roll(Ap, 2) + self.skip_pen * self.sg
        if not self.wrap:
            c1[0] = np.inf
            c2[:2] = np.inf
        A = c + np.minimum(np.minimum(Ap + self.stay_pen * self.sg, c1), c2)
        A -= A.min()
        self.A = A
        j = int(A.argmin())
        dj = np.abs((np.arange(self.L) - j + self.L // 2) % self.L - self.L // 2)
        alt = A[dj > 3].min() if (dj > 3).any() else 0.0
        self.last_extras = {}
        return float((j / (self.L - 1)) % 1.0), float(1 - np.exp(-alt / (self.tau * self.sg)))

if HAS_TORCH:
    class TemporalEncoder(nn.Module):
        def __init__(self, D, emb):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(D, 64, 5, padding=2), nn.ReLU(), nn.Conv1d(64, 64, 5, padding=2), nn.ReLU())
            self.gru = nn.GRU(64, 64, batch_first=True)
            self.emb = nn.Linear(64, emb)
            self.head = nn.Linear(emb, 2)

        def forward(self, x):
            h = self.conv(x.transpose(1, 2)).transpose(1, 2)
            _, hn = self.gru(h)
            e = self.emb(hn[-1])
            return e, self.head(e)

class EmbeddingTracker(Tracker):
    def _embed(self, Z):
        idx = np.maximum(np.arange(len(Z))[:, None] + self.offs[None], 0)
        X = torch.tensor(Z[idx].astype(np.float32))
        with torch.no_grad():
            e, p = self.net(X)
        return e.numpy(), p.numpy()

    def reset(self, init="start"):
        self.buf = []

    def step(self, z):
        self.buf = (self.buf + [z])[-self.W:]
        w = np.array(self.buf)
        if len(w) < self.W:
            w = np.concatenate([np.repeat(w[:1], self.W - len(w), 0), w])
        e, p = self._embed(w)
        ph, c = decode(p[-1:])
        self.last_extras = {"emb": e[-1]}
        return float(ph[0]), float(c[0])

class EnsembleTracker(Tracker):
    def _comb(self, ms, cs):
        w = np.array(cs) * self.w
        S, C = (w * np.sin(2 * np.pi * np.array(ms))).sum(), (w * np.cos(2 * np.pi * np.array(ms))).sum()
        R = np.hypot(S, C) / max(w.sum(), 1e-9)
        return float((np.arctan2(S, C) / (2 * np.pi)) % 1.0), float(np.mean(cs) * R)

    def reset(self, init="start"):
        for m in self.ms:
            m.reset(init)

    def step(self, z):
        out = [m.step(z) for m in self.ms]
        self.last_extras = {k: v for m in self.ms for k, v in m.last_extras.items()}
        return self._comb([o[0] for o in out], [o[1] for o in out])