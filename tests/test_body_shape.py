"""Body shape (betas) estimation: landmarks from a randomly shaped SMPL-X body (+1 cm noise) over a
varied pose sequence -> the live engine must recover that body's segment lengths and stature.
Run: python tests/test_body_shape.py"""
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.camera.smplx_fitter import SMPLXFittingEngine  # noqa: E402
from src.camera.body_shape import SEGMENTS  # noqa: E402

eng = SMPLXFittingEngine(model_dir=os.path.join(ROOT, "models"))
MP_OF = {16: 11, 17: 12, 18: 13, 19: 14, 20: 15, 21: 16, 1: 23, 2: 24, 4: 25, 5: 26, 7: 27, 8: 28,
         23: 2, 24: 5, 15: 7, 25: 19, 40: 20, 31: 17, 46: 18, 10: 31, 11: 32}


def rest_joints(b):
    with torch.no_grad():
        return eng.layer(betas=torch.tensor(b, dtype=torch.float32)[None]).joints[0].numpy()


def seg_lengths(J):
    return {n: np.linalg.norm(J[list(a)].mean(0) - J[list(b)].mean(0)) for n, (a, b, _, _) in SEGMENTS.items()}


def stature(J):
    return J[15, 1] - 0.5 * (J[7, 1] + J[8, 1])  # head joint to ankles, rest pose


rng = np.random.default_rng(5)
worst_seg, worst_h = 0.0, 0.0
for trial in range(3):
    gt = rng.normal(0, 1.0, 10).astype(np.float32)
    eng.reset(keep_shape=False)
    for k in range(90):
        bp = np.zeros(63, np.float32)
        bp[15 * 3 + 2], bp[16 * 3 + 2] = -1.2 + 0.8 * np.sin(k / 7), 1.2 - 0.8 * np.cos(k / 9)
        bp[17 * 3 + 1], bp[18 * 3 + 1] = -0.6 - 0.5 * np.sin(k / 5), 0.6 + 0.5 * np.sin(k / 6)
        bp[0], bp[3] = -0.3 * (1 + np.sin(k / 8)), -0.3 * (1 + np.cos(k / 8))
        with torch.no_grad():
            J = eng.layer(betas=torch.tensor(gt)[None], body_pose=torch.tensor(bp)[None]).joints[0].numpy()
        mp = np.zeros((33, 4), np.float32)
        mp[:, 3] = 0.95
        for j, m in MP_OF.items():
            mp[m, :3] = (J[j] + rng.normal(0, 0.01, 3)) * [1, -1, -1]
        mp[8, :3] = mp[7, :3]
        mp[29, :3], mp[30, :3] = mp[27, :3], mp[28, :3]
        eng.update_from_world_landmarks(mp.tolist())
    est = eng.betas[0].numpy()
    Lg, Le = seg_lengths(rest_joints(gt)), seg_lengths(rest_joints(est))
    errs = {n: 100 * abs(Le[n] - Lg[n]) for n in Lg}
    dh = 100 * abs(stature(rest_joints(est)) - stature(rest_joints(gt)))
    d0 = 100 * abs(stature(rest_joints(np.zeros(10))) - stature(rest_joints(gt)))
    worst_seg, worst_h = max(worst_seg, max(errs.values())), max(worst_h, dh)
    print(f"[shape] body {trial}: segment error mean {np.mean(list(errs.values())):.2f} cm, max {max(errs.values()):.2f} cm "
          f"({max(errs, key=errs.get)}); stature error {dh:.1f} cm (template would be {d0:.1f} cm)")
assert worst_seg < 1.5 and worst_h < 3.0
print("SHAPE OK")
