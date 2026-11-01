import numpy as np
from .utils import circ_signed

class Enforcer:
    def reset(self, ctx, init="start"):
        self.v0, self.phi = 1.0 / ctx.rep_len, (0.0 if init == "start" else None)

class NoEnforcer(Enforcer):
    def step(self, m, conf):
        self.phi = m
        return m, True

class ForwardClamp(Enforcer):
    def step(self, m, conf):
        if self.phi is None:
            self.phi = m
            return m, True
        if conf < self.cm:
            self.phi = (self.phi + self.v0) % 1.0
            return self.phi, False
        d = float(np.clip(circ_signed(m, self.phi), -self.back, self.vm * self.v0))
        self.phi = (self.phi + d) % 1.0
        return self.phi, True

class AlphaBeta(Enforcer):
    def reset(self, ctx, init="start"):
        super().reset(ctx, init)
        self.v, self.skipped = self.v0, 0

    def step(self, m, conf):
        if self.phi is None:
            self.phi = m
            return m, True
        pred = (self.phi + self.v) % 1.0
        nu = circ_signed(m, pred)
        ok = conf >= self.pmin and abs(nu) <= self.gate
        if not ok and conf >= 0.7 and self.skipped >= self.relock:
            self.phi, self.v, self.skipped = m, self.v0, 0
            return self.phi, True
        if ok:
            self.phi = (pred + self.k0 * conf * nu) % 1.0
            self.v = float(np.clip(self.v + self.kv * nu, 0, 3 * self.v0))
            self.skipped = 0
        else:
            self.phi = pred
            self.skipped += 1
        return self.phi, ok