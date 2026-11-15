"""Joint limits + self-collision on the live solver, using synthetic landmarks that ask for
impossible poses. Run: python tests/test_body_anatomy.py"""
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.camera.smplx_fitter import SMPLXFittingEngine, KEYPOINTS, KP  # noqa: E402
from src.camera.body_model import BODY_JOINTS, mat_to_euler, rotvec_to_mat, clamp_body_pose  # noqa: E402

eng = SMPLXFittingEngine(model_dir=os.path.join(ROOT, "models"))
J = {n: i for i, n in enumerate(BODY_JOINTS)}
TOL = 0.005  # m of overlap beyond slack still counted as "touching", not penetrating


def sl(name):
    i = J[name]
    return slice(i * 3, i * 3 + 3)


def landmarks_for(bp, go=(0, 0, 0)):
    """Exact MediaPipe-style world landmarks of an SMPL-X pose (may be physically impossible)."""
    with torch.no_grad():
        kp = eng._fk_keypoints(torch.tensor(go, dtype=torch.float32)[None],
                               torch.tensor(bp, dtype=torch.float32)[None]).numpy()
    mp = np.zeros((33, 4), np.float32)
    mp[:, 3] = 1.0
    for k, (_, _, mids) in enumerate(KEYPOINTS):
        if len(mids) == 1:
            mp[mids[0], :3] = kp[k] * [1, -1, -1]
    h = kp[KP["head"]] * [1, -1, -1]
    mp[7, :3] = mp[8, :3] = h
    return mp.tolist()


def fit(bp, frames=4):
    eng.reset()
    for _ in range(frames):
        out = eng.update_from_world_landmarks(landmarks_for(bp))
    return out


def euler_of(name):
    lim = eng.limits[name]
    r = eng.body_pose[0].numpy()[sl(name)]
    return np.degrees(mat_to_euler(rotvec_to_mat(r), lim["order"])), lim


def within_limits():
    bad = []
    for name in BODY_JOINTS:
        e, lim = euler_of(name)
        if np.any(e < np.degrees(lim["lo"]) - 0.5) or np.any(e > np.degrees(lim["hi"]) + 0.5):
            bad.append((name, e.round(1)))
    return bad


results = []

# 1. Hyperextended knee (bent the wrong way) and hyperextended elbow
bp = np.zeros(63, np.float32)
bp[sl("left_knee")] = [-0.6, 0, 0]       # knee bending forward = impossible
bp[sl("right_elbow")] = [0, -0.7, 0]     # right elbow bends about +y; -y = hyperextension
bp[sl("left_shoulder")] = [0, 0, -1.2]
bp[sl("right_shoulder")] = [0, 0, 1.2]
fit(bp)
bad = within_limits()
ek, _ = euler_of("left_knee")
ee, _ = euler_of("right_elbow")
print(f"[limits] hyperextension request -> knee x {ek[0]:.1f} deg (>=0), elbow y {ee[1]:.1f} deg (>=-5); violations {bad}")
assert not bad and ek[0] >= -0.5 and ee[1] >= -5.5
results.append("joint limits: hyperextension blocked")

# 2. Random limb-into-body requests (legal joint angles, but the limbs go through the torso)
rng = np.random.default_rng(3)
worst_in, worst_out, n = 0.0, 0.0, 0
while n < 12:
    bp = np.zeros(63, np.float32)
    for s, sg in (("left", 1), ("right", -1)):
        bp[sl(f"{s}_shoulder")] = [rng.uniform(-0.5, 0.5), sg * rng.uniform(-1.6, -0.6), sg * rng.uniform(-1.3, -0.3)]
        bp[sl(f"{s}_elbow")] = [0, sg * -rng.uniform(1.2, 2.4), 0]
    bp = clamp_body_pose(bp, eng.limits)[0]
    with torch.no_grad():
        _, R, P = eng._fk_keypoints(torch.zeros(1, 3), torch.tensor(bp)[None], return_rot=True)
    req = eng.collision.penetrations(P, R, tol=0.02)
    if not req:
        continue  # want requests that clearly penetrate
    n += 1
    out = fit(bp)
    got = max(out["penetrations"].values(), default=0.0)
    worst_in = max(worst_in, max(req.values()))
    worst_out = max(worst_out, got)
print(f"[collision] 12 penetrating requests: worst requested {worst_in * 100:.1f} cm -> worst after fit {worst_out * 100:.2f} cm")
assert worst_out <= TOL + 1e-4
results.append(f"collision: requested up to {worst_in * 100:.1f} cm penetration, fitted <= {worst_out * 100:.2f} cm")

# 3. Crossed arms in front of the chest
bp = np.zeros(63, np.float32)
bp[sl("left_shoulder")] = [0, -1.3, -0.5]
bp[sl("right_shoulder")] = [0, 1.3, 0.5]
bp[sl("left_elbow")] = [0, -2.1, 0]
bp[sl("right_elbow")] = [0, 2.1, 0]
out = fit(bp, frames=6)
print(f"[collision] crossed arms: penetrations after fit {out['penetrations']}, limit violations {within_limits()}")
assert not out["penetrations"] and not within_limits()
results.append("crossed arms: no torso penetration")

# 4. Hand pushed into the abdomen over a sequence: must stay outside, no jumps/oscillation
eng.reset()
deltas, pens = [], []
prev = None
for t in np.linspace(0, 1, 30):
    bp = np.zeros(63, np.float32)
    bp[sl("left_shoulder")] = [0, -0.3 - 0.9 * t, -1.0]
    bp[sl("left_elbow")] = [0, -0.3 - 1.9 * t, 0]           # forearm sweeps into/through the belly
    bp[sl("right_shoulder")] = [0, 0, 1.2]
    out = eng.update_from_world_landmarks(landmarks_for(bp))
    cur = eng.body_pose[0].numpy().copy()
    if prev is not None:
        deltas.append(np.degrees(np.abs(cur - prev).max()))
    prev = cur
    pens.append(max(out["penetrations"].values(), default=0.0))
print(f"[stability] hand-into-abdomen sweep: max penetration {max(pens) * 100:.2f} cm, "
      f"max per-frame joint change {max(deltas):.1f} deg (request moves <= {np.degrees(1.9 / 29):.1f} deg/frame)")
assert max(pens) <= TOL + 1e-4 and max(deltas) < 15.0
results.append("hand-into-abdomen sweep: stays outside, smooth")

# 5. Head orientation from the face landmarker (nod + tilt + turn); eyes/ears alone barely see nods
bp = np.zeros(63, np.float32)
bp[sl("left_shoulder")], bp[sl("right_shoulder")] = [0, 0, -1.2], [0, 0, 1.2]
bp[sl("neck")] = [0.30, 0.25, 0.15]
bp[sl("head")] = [0.25, 0.20, 0.10]
with torch.no_grad():
    _, Rg, _ = eng._fk_keypoints(torch.zeros(1, 3), torch.tensor(bp)[None], return_rot=True)
R_true = Rg[eng._pos_of[15]].numpy()
face_M = np.eye(4)
face_M[:3, :3] = R_true
errs = {}
for src, fm in (("pose only", None), ("with face matrix", face_M.tolist())):
    eng.reset()
    for _ in range(4):
        eng.update_from_world_landmarks(landmarks_for(bp), face_matrix=fm)
    with torch.no_grad():
        _, Rf, _ = eng._fk_keypoints(eng.global_orient, eng.body_pose, return_rot=True)
    Rrel = Rf[eng._pos_of[15]].numpy() @ R_true.T
    errs[src] = float(np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))))
print(f"[head] nod+tilt+turn head error: pose only {errs['pose only']:.1f} deg, with face matrix {errs['with face matrix']:.1f} deg (source {eng.head_source})")
assert errs["with face matrix"] < 5.0 and eng.head_source == "face"
results.append(f"head orientation: {errs['pose only']:.1f} deg -> {errs['with face matrix']:.1f} deg with face matrix")

print("\n".join(["ANATOMY OK:"] + ["  - " + r for r in results]))
