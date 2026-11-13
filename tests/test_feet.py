"""Foot orientation acceptance test: outward / inward / asymmetric feet, heel + toe raises.
MediaPipe-like landmarks are taken from the posed SMPL-X *mesh* (heel vertex, toe-tip vertices), i.e.
where MediaPipe's heel (29/30) and foot-index (31/32) actually are, not from model joints.
Run: python tests/test_feet.py"""
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.camera.smplx_fitter import SMPLXFittingEngine  # noqa: E402
from src.camera.body_model import BODY_JOINTS  # noqa: E402

eng = SMPLXFittingEngine(model_dir=os.path.join(ROOT, "models"))
J = {n: i for i, n in enumerate(BODY_JOINTS)}
HEEL = {"l": 8846, "r": 8635}
TOES = {"l": (5770, 5780), "r": (8463, 8474)}
MP_JOINTS = {11: 16, 12: 17, 13: 18, 14: 19, 15: 20, 16: 21, 23: 1, 24: 2, 25: 4, 26: 5, 27: 7, 28: 8,
             2: 23, 5: 24, 19: 25, 20: 40, 17: 31, 18: 46}


def sl(name):
    return slice(J[name] * 3, J[name] * 3 + 3)


def posed(bp):
    with torch.no_grad():
        o = eng.layer(betas=eng.betas, body_pose=torch.tensor(bp)[None])
    return o.joints[0].numpy(), o.vertices[0].numpy()


def mediapipe_from_mesh(joints, verts):
    mp = np.zeros((33, 4), np.float32)
    mp[:, 3] = 0.95
    for m, j in MP_JOINTS.items():
        mp[m, :3] = joints[j]
    mp[7, :3] = mp[8, :3] = joints[15]
    for s, (h, t) in {"l": (29, 31), "r": (30, 32)}.items():
        mp[h, :3] = verts[HEEL[s]]
        mp[t, :3] = verts[list(TOES[s])].mean(0)
    mp[:, :3] *= [1, -1, -1]  # SMPL frame -> MediaPipe world (x, -y, -z)
    return mp.tolist()


def foot_angles(verts, s):
    """yaw (deg, + = toes outward) and pitch (deg, + = toes up) of the heel->toe direction."""
    d = verts[list(TOES[s])].mean(0) - verts[HEEL[s]]
    sign = 1 if s == "l" else -1  # outward = +x for the left foot, -x for the right
    yaw = np.degrees(np.arctan2(sign * d[0], d[2]))
    pitch = np.degrees(np.arctan2(d[1], np.hypot(d[0], d[2])))
    return yaw, pitch


cases = {
    "both forward": {},
    "both outward": {"left_hip": [0, 0.45, 0], "right_hip": [0, -0.45, 0],
                     "left_ankle": [0, 0.3, 0], "right_ankle": [0, -0.3, 0]},
    "left outward only": {"left_hip": [0, 0.5, 0], "left_ankle": [0, 0.3, 0]},
    "right outward only": {"right_hip": [0, -0.5, 0], "right_ankle": [0, -0.3, 0]},
    "both inward": {"left_hip": [0, -0.3, 0], "right_hip": [0, 0.3, 0],
                    "left_ankle": [0, -0.25, 0], "right_ankle": [0, 0.25, 0]},
    "heels raised (toes down)": {"left_ankle": [0.6, 0, 0], "right_ankle": [0.6, 0, 0]},
    "toes raised": {"left_ankle": [-0.35, 0, 0], "right_ankle": [-0.35, 0, 0]},
    "seated, feet out (screenshot)": {"left_hip": [-1.4, 0.35, 0.35], "right_hip": [-1.4, -0.35, -0.35],
                                      "left_knee": [1.5, 0, 0], "right_knee": [1.5, 0, 0],
                                      "left_ankle": [0, 0.35, 0], "right_ankle": [0, -0.35, 0]},
}
worst = 0.0
for name, rots in cases.items():
    bp = np.zeros(63, np.float32)
    bp[sl("left_shoulder")], bp[sl("right_shoulder")] = [0, 0, -1.2], [0, 0, 1.2]
    for j, r in rots.items():
        bp[sl(j)] = r
    gj, gv = posed(bp)
    eng.reset()
    for _ in range(4):
        eng.update_from_world_landmarks(mediapipe_from_mesh(gj, gv))
    _, fv = posed(eng.body_pose[0].numpy())
    line = []
    for s in ("l", "r"):
        gy, gp = foot_angles(gv, s)
        fy, fp = foot_angles(fv, s)
        worst = max(worst, abs(gy - fy), abs(gp - fp))
        line.append(f"{s}: yaw {gy:6.1f} -> {fy:6.1f}, pitch {gp:6.1f} -> {fp:6.1f}")
    print(f"[feet] {name:30s} " + " | ".join(line))
print(f"worst foot yaw/pitch error: {worst:.1f} deg")
assert worst < 6.0, "foot orientation not reproduced"
print("FEET OK")
