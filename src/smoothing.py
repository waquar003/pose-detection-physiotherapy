import numpy as np


class JointSmoother:
    """One Euro filter for an (N, 2) array of joints with per-joint validity.

    cutoff = min_cutoff + beta * speed, where speed is measured in body-scale units per second.
    Low speed -> low cutoff (strong smoothing, no jitter). High speed -> high cutoff (little lag).
    """

    def __init__(self, min_cutoff=1.5, beta=4.0, d_cutoff=1.0, max_gap=0.5):
        self.min_cutoff, self.beta, self.d_cutoff, self.max_gap = min_cutoff, beta, d_cutoff, max_gap
        self.reset()

    def reset(self):
        self.t = None
        self.x = None          # previous filtered positions (N, 2)
        self.dx = None         # previous filtered velocity (N, 2), pixels/s
        self.valid = None      # (N,) joint has filter state
        self.scale = None      # smoothed body scale in pixels

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t, joints, ok, scale):
        """t: seconds. joints: (N,2) pixels. ok: (N,) bool. scale: body size in pixels (e.g. torso length)."""
        x = np.asarray(joints, dtype=np.float64)
        ok = np.asarray(ok, dtype=bool)
        scale = max(float(scale), 1.0)

        if self.t is None or t <= self.t or t - self.t > self.max_gap:
            self.t, self.x, self.dx = t, x.copy(), np.zeros_like(x)
            self.valid, self.scale = ok.copy(), scale
            return x.astype(np.float32)

        dt = t - self.t
        self.scale = 0.9 * self.scale + 0.1 * scale          # body size changes slowly; do not let noise in

        # velocity: derivative of the raw signal against the previous filtered value, then low-passed
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * (x - self.x) / dt + (1.0 - a_d) * self.dx

        # one cutoff per joint (shared by x and y, so the filter is not anisotropic)
        speed = np.linalg.norm(dx_hat, axis=1) / self.scale   # body-scales per second
        cutoff = self.min_cutoff + self.beta * speed
        a = self._alpha(cutoff, dt)[:, None]
        x_hat = a * x + (1.0 - a) * self.x

        fresh = ok & ~self.valid                              # joint just became visible: start from its raw value
        x_hat[fresh] = x[fresh]
        dx_hat[fresh] = 0.0
        x_hat[~ok] = x[~ok]                                   # hidden joints pass through, the coach ignores them
        dx_hat[~ok] = 0.0

        self.t, self.x, self.dx, self.valid = t, x_hat, dx_hat, ok.copy()
        return x_hat.astype(np.float32)

class LandmarkStabilizer:
    """Debounce + momentum hold + jump rejection + One Euro filtering on the 33 MediaPipe landmarks.
    Returns (pts, vis, w, h) to feed the mapper. Groups come from the mapper checkpoint (name -> landmark indices)."""

    def __init__(self, vis_thr, on_frames=3, off_frames=5, decay=0.7, max_jump=0.6,
                 min_cutoff=2.0, beta=3.0, d_cutoff=1.0, max_gap=0.5, hard_reset=3.0):
        self.vis_thr, self.on_frames, self.off_frames = vis_thr, on_frames, off_frames
        self.decay, self.max_jump, self.max_gap, self.hard_reset = decay, max_jump, max_gap, hard_reset
        self.smoother = JointSmoother(min_cutoff, beta, d_cutoff, max_gap=hard_reset)   # no snap on slow frames
        self.reset()

    def reset(self):
        self.smoother.reset()
        self.t = None
        self.shown, self.on, self.off = {}, {}, {}
        self.last = self.vel = self.jumps = None
        self.scale, self.wh = 100.0, None

    def _init(self, t, pts, groups, w, h):
        self.t, self.wh = t, (w, h)
        self.shown = {n: False for n in groups}
        self.on = {n: 0 for n in groups}
        self.off = {n: 0 for n in groups}
        self.last = pts.astype(np.float64).copy()
        self.vel = np.zeros_like(self.last)
        self.jumps = np.zeros(len(pts), dtype=int)
        sw = float(np.linalg.norm(pts[11] - pts[12]))
        self.scale = sw if sw > 1.0 else 100.0

    def update(self, t, pts, vis, w, h, groups):
        """pts/vis None = no person detected this frame. Returns None if there is nothing to hold yet."""
        stale = False
        if self.t is not None:
            if t - self.t > self.hard_reset or t < self.t - 1.0:       # really lost, or the clock jumped back
                self.reset()
            elif t <= self.t:                                          # duplicate / out-of-order timestamp
                t = self.t + 1e-3
            elif t - self.t > self.max_gap:                            # slow frame: keep the state, drop stale momentum
                stale = True
        if pts is None:
            if self.last is None:
                return None
            w, h = self.wh
            pts, vis = self.last.copy(), np.zeros(33, dtype=np.float32)
        pts = np.asarray(pts, dtype=np.float64)
        vis = np.asarray(vis, dtype=np.float32)
        if self.last is None:
            self._init(t, pts, groups, w, h)
        if stale:
            self.vel[:] = 0.0
        dt = float(np.clip(t - self.t, 1e-3, 0.25))
        self.wh = (w, h)

        inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        vis_out = np.zeros(33, dtype=np.float32)
        shown_lm = np.zeros(33, dtype=bool)

        for n, idx in groups.items():
            idx = np.asarray(idx)
            raw_ok = vis[idx].mean() >= self.vis_thr and inside[idx].mean() >= 0.5
            if raw_ok:
                self.on[n] += 1; self.off[n] = 0
            else:
                self.off[n] += 1; self.on[n] = 0

            if not self.shown[n] and self.on[n] >= self.on_frames:      # confirmed: start tracking from the raw value
                self.shown[n] = True
                self.last[idx], self.vel[idx], self.jumps[idx] = pts[idx], 0.0, 0
            elif self.shown[n] and self.off[n] >= self.off_frames:      # gone for long enough: drop
                self.shown[n] = False
            if not self.shown[n]:
                continue

            if raw_ok:
                pred = self.last[idx] + self.vel[idx] * dt
                far = np.linalg.norm(pts[idx] - pred, axis=1) / self.scale
                bad = (far > self.max_jump) & (self.jumps[idx] < 2)     # one-frame glitch: keep the prediction
                new = np.where(bad[:, None], pred, pts[idx])
                self.jumps[idx] = np.where(bad, self.jumps[idx] + 1, 0)
                self.vel[idx] = 0.5 * self.vel[idx] + 0.5 * (new - self.last[idx]) / dt
                self.last[idx] = new
                vis_out[idx] = np.maximum(vis[idx], 0.5)
            else:                                                       # grace period: decaying momentum
                self.vel[idx] *= self.decay
                new = self.last[idx] + self.vel[idx] * dt
                new[:, 0] = np.clip(new[:, 0], 0, w - 1)
                new[:, 1] = np.clip(new[:, 1], 0, h - 1)
                self.last[idx] = new
                vis_out[idx] = 0.5
            shown_lm[idx] = True

        if shown_lm[[11, 12]].all():
            sw = float(np.linalg.norm(self.last[11] - self.last[12]))
            if sw > 1.0:
                self.scale = 0.9 * self.scale + 0.1 * sw

        out = pts.copy()
        out[shown_lm] = self.last[shown_lm]
        out = self.smoother(t, out, shown_lm, self.scale)               # scale = shoulder width in pixels
        self.t = t
        return out.astype(np.float32), vis_out, w, h
