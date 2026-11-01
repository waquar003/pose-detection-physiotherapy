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