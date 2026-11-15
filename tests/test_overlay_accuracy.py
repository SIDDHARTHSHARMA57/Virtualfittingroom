"""Overlay accuracy with MediaPipe-like errors: precise 2D image landmarks (+-1.5 px) but noisy 3D
world landmarks (depth +-6 cm, xy +-1.5 cm per landmark per frame). Measures how far the fitted
model's joints land from the true joints in the image (px) and in 3D bone direction (deg), with and
without the 2D reprojection term. Run: python tests/test_overlay_accuracy.py"""
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.camera.smplx_fitter import SMPLXFittingEngine, KEYPOINTS, BONES, KP  # noqa: E402
from src.camera.body_model import clamp_body_pose  # noqa: E402

W, H = 640, 480
f = (W / 2) / np.tan(np.radians(30))
K = {"w": W, "h": H, "fx": f, "fy": f, "cx": W / 2, "cy": H / 2}
eng = SMPLXFittingEngine(model_dir=os.path.join(ROOT, "models"), intrinsics=K)
FLIP = np.array([1, -1, -1])


def project(kp, t):
    Pc = kp * FLIP + t
    return np.stack([K["fx"] * Pc[:, 0] / Pc[:, 2] + K["cx"], K["fy"] * Pc[:, 1] / Pc[:, 2] + K["cy"]], 1)


def bone_err(a, b):
    def dirs(k):
        d = np.array([k[KP[c]] - k[KP[p]] for p, c, _ in BONES])
        return d / np.linalg.norm(d, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((dirs(a) * dirs(b)).sum(1), -1, 1))).mean()


def run(reproj_w, seed=0):
    rng = np.random.default_rng(seed)
    eng.reproj_w = reproj_w
    eng.reset()
    keys = []
    for _ in range(4):
        bp = np.zeros(63, np.float32)
        bp[15 * 3:16 * 3] = [0, rng.uniform(-0.8, 0.2), rng.uniform(-1.3, 0.6)]     # left shoulder
        bp[16 * 3:17 * 3] = [0, -rng.uniform(-0.8, 0.2), -rng.uniform(-1.3, 0.6)]   # right shoulder
        bp[17 * 3 + 1], bp[18 * 3 + 1] = -rng.uniform(0.1, 1.5), rng.uniform(0.1, 1.5)  # elbows
        bp[0], bp[3] = -rng.uniform(0, 0.6), -rng.uniform(0, 0.6)                   # hips flex
        bp[9], bp[12] = rng.uniform(0, 0.8), rng.uniform(0, 0.8)                    # knees
        keys.append(clamp_body_pose(bp, eng.limits)[0])
    px, deg = [], []
    t_gt = np.array([0.05, -0.1, 2.2])
    for s in range(3):
        for a in np.linspace(0, 1, 6, endpoint=False):
            bp = (1 - a) * keys[s] + a * keys[s + 1]
            go = np.array([0, 0.3 * np.sin(s + a), 0], np.float32)
            with torch.no_grad():
                kp = eng._fk_keypoints(torch.tensor(go)[None], torch.tensor(bp, dtype=torch.float32)[None]).numpy()
            uv = project(kp, t_gt)
            world = np.zeros((33, 4), np.float32)
            img = np.zeros((33, 3), np.float32)
            world[:, 3] = img[:, 2] = 0.95
            for k, (_, _, mids) in enumerate(KEYPOINTS):
                for m in mids:
                    noisy = kp[k] + rng.normal(0, [0.015, 0.015, 0.06])  # MediaPipe-like: depth is the weak axis
                    world[m, :3] = noisy * FLIP
                    img[m, :2] = (uv[k] + rng.normal(0, 1.5, 2)) / [W, H]
            eng.update_from_world_landmarks(world.tolist(), body_points=img.tolist())
            with torch.no_grad():
                kp_fit = eng._fk_keypoints(eng.global_orient, eng.body_pose).numpy()
            px.append(np.linalg.norm(project(kp_fit, eng.transl) - uv, axis=1).mean())
            deg.append(bone_err(kp_fit, kp))
    return np.mean(px[2:]), np.mean(deg[2:])


cfg_w = eng.reproj_w
rows = {}
for wgt in (0.0, cfg_w):
    rows[wgt] = run(wgt)
    print(f"[overlay] reprojection_weight {wgt:6.1f} (depth weight {eng.depth_w if wgt else 1.0}): "
          f"image error {rows[wgt][0]:5.1f} px, 3D bone error {rows[wgt][1]:5.1f} deg")
base, cur = rows[0.0], rows[cfg_w]
assert cur[0] < 0.5 * base[0] and cur[1] < base[1], "2D term must cut image error and 3D error"
print(f"OVERLAY OK: image error {base[0]:.1f} -> {cur[0]:.1f} px, 3D {base[1]:.1f} -> {cur[1]:.1f} deg")
