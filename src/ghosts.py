import numpy as np
from .utils import circ_signed, rms

class Ghost:
    def _topk(self, idx, D, k):
        out, seen = [], set()
        B = self.ctx.bank
        for o in np.argsort(D):
            j = idx[o]
            if B["rid"][j] in seen:
                continue
            seen.add(B["rid"][j])
            out.append(j)
            if len(out) >= k:
                break
        return out

    def _pack(self, js, D_by_j, z):
        B = self.ctx.bank
        return [dict(j=int(j), pose=B["pose"][j], phase=float(B["ph"][j]), person=str(B["person"][j]), rep=int(B["rep"][j]),
                     frame=int(B["frame"][j]), dist=float(D_by_j[j]), z_dist=float(rms(z, B["Z"][j]))) for j in js]

class ConstrainedNN(Ghost):
    def query(self, z, phi, extras, k=1):
        B = self.ctx.bank
        s = circ_signed(B["ph"], (phi + self.la) % 1.0)
        idx = np.where((s >= -self.back) & (s <= self.fwd))[0]
        if len(idx) < 20:
            idx = np.arange(len(B["ph"]))
        D = np.sqrt(((B["Z"][idx] - z) ** 2).mean(1))
        return self._pack(self._topk(idx, D, k), dict(zip(idx, D)), z)

class UnconstrainedNN(Ghost):
    def query(self, z, phi, extras, k=1):
        B = self.ctx.bank
        idx = np.arange(len(B["ph"]))
        D = np.sqrt(((B["Z"] - z) ** 2).mean(1))
        return self._pack(self._topk(idx, D, k), dict(zip(idx, D)), z)

class TemplateGhost(Ghost):
    def query(self, z, phi, extras, k=1):
        c = self.ctx
        b = int(np.rint(((phi + self.la) % 1.0) * (c.L - 1)))
        return [dict(j=-1, pose=c.tmplP[b], phase=float(b / (c.L - 1)), person="template", rep=0, frame=b,
                     dist=float(rms(z, c.tmplZ[b])), z_dist=float(rms(z, c.tmplZ[b])))]

class EmbedNN(Ghost):
    def query(self, z, phi, extras, k=1):
        B = self.ctx.bank
        s = circ_signed(B["ph"], phi)
        idx = np.where((s >= -self.back) & (s <= self.fwd))[0]
        if len(idx) < 20:
            idx = np.arange(len(B["ph"]))
        D = np.linalg.norm(self.bank_emb[idx] - extras["emb"], axis=1)
        return self._pack(self._topk(idx, D, k), dict(zip(idx, D)), z)