import numpy as np
from . import config

circ_signed = lambda a, b: (a - b + 0.5) % 1.0 - 0.5
rms = lambda a, b: np.sqrt(((a - b) ** 2).mean(-1))

def decode(pred):
    return (np.arctan2(pred[:, 0], pred[:, 1]) / (2 * np.pi)) % 1.0, np.clip(np.hypot(pred[:, 0], pred[:, 1]), 0, 1)

def angle_abc(a, b, c):
    ab, bc = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc) + 1e-9), -1, 1))))

def normalize_frame(p):
    hips = p[config.JN["Hips"]]
    return (p - hips) / max(float(np.linalg.norm(p[config.JN["Spine"]] - hips)), 1e-6)

def frame_angles(n):
    g = lambda k: n[config.JN[k]]
    out = [angle_abc(g(a), g(b), g(c)) for a, b, c in config.ANGLE_DEFS]
    sh, hp, tr, hd = g("RightShoulder") - g("LeftShoulder"), g("RightUpLeg") - g("LeftUpLeg"), g("Neck") - g("Hips"), g("Head") - g("Neck")
    out += [np.degrees(np.arctan2(sh[1], abs(sh[0]) + 1e-9)), np.degrees(np.arctan2(hp[1], abs(hp[0]) + 1e-9)),
            np.degrees(np.arctan2(tr[0], abs(tr[1]) + 1e-9)), np.degrees(np.arctan2(hd[0], abs(hd[1]) + 1e-9))]
    return np.array(out, dtype=np.float64)

def required_joints(ex):
    names = set(config.EX_JOINTS[ex]) | {"Hips", "Spine"}
    for i in config.EX_ANG[ex]:
        names |= config._ANGLE_JOINTS[i]
    return sorted(config.JN[n] for n in names)