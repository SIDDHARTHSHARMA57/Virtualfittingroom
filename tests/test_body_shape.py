"""Body-shape calibration end to end, on rendered synthetic people.

For random SMPL-X bodies: render the real silhouette of the posed mesh, project its landmarks with a
pinhole camera, add MediaPipe-like noise, and run the live engine through guided calibration.
Checks: height / segment / front-width accuracy (with vs without silhouette widths), shape stays
fixed during arbitrary movement afterwards (pose independence), and switching person forgets the
previous person's shape. Run: python tests/test_body_shape.py"""
import os
import sys
import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.camera.smplx_fitter import SMPLXFittingEngine  # noqa: E402
from src.camera.body_shape import SEGMENTS, WIDTH_BANDS  # noqa: E402

W, H = 640, 480
f = (W / 2) / np.tan(np.radians(30))
K = {"w": W, "h": H, "fx": f, "fy": f, "cx": W / 2, "cy": H / 2}
FLIP = np.array([1.0, -1.0, -1.0])
T_CAM = np.array([0.0, 0.25, 3.0])  # person ~3 m away so the whole body is in frame
MP_OF = {16: 11, 17: 12, 18: 13, 19: 14, 20: 15, 21: 16, 1: 23, 2: 24, 4: 25, 5: 26, 7: 27, 8: 28,
         23: 2, 24: 5, 15: 7, 25: 19, 40: 20, 31: 17, 46: 18}
HEEL, TOES = {29: 8846, 30: 8635}, {31: (5770, 5780), 32: (8463, 8474)}
eng = SMPLXFittingEngine(model_dir=os.path.join(ROOT, "models"), intrinsics=K)
eng.calib.clothing = 0.0   # synthetic people wear nothing
FACES = np.asarray(eng.faces, np.int64)
rng = np.random.default_rng(11)


def project(P):
    Pc = P * FLIP + T_CAM
    return np.stack([K["fx"] * Pc[:, 0] / Pc[:, 2] + K["cx"], K["fy"] * Pc[:, 1] / Pc[:, 2] + K["cy"]], 1)


def frame_of(betas, bp):
    with torch.no_grad():
        o = eng.layer(betas=torch.tensor(betas)[None], body_pose=torch.tensor(bp)[None])
    J, V = o.joints[0].numpy(), o.vertices[0].numpy()
    pts = {m: J[j] for j, m in MP_OF.items()}
    pts.update({m: V[v] for m, v in HEEL.items()})
    pts.update({m: V[list(v)].mean(0) for m, v in TOES.items()})
    pts[8] = pts[7]
    world, img = np.zeros((33, 4)), np.zeros((33, 3))
    world[:, 3] = img[:, 2] = 0.95
    for m, p in pts.items():
        world[m, :3] = (p + rng.normal(0, 0.01, 3)) * FLIP
        img[m, :2] = (project(p[None])[0] + rng.normal(0, 1.0, 2)) / [W, H]
    uv = project(V).astype(np.int32)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, list(uv[FACES]), 1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sil = max(cnts, key=cv2.contourArea)[:, 0, :].astype(float)
    return world.tolist(), img.tolist(), sil


def a_pose():
    bp = np.zeros(63, np.float32)
    bp[15 * 3 + 2], bp[16 * 3 + 2] = -0.85 + rng.normal(0, 0.03), 0.85 + rng.normal(0, 0.03)
    return bp


def moving_pose(k):
    bp = np.zeros(63, np.float32)
    bp[15 * 3 + 2], bp[16 * 3 + 2] = -1.2 + 2.0 * (k % 7) / 6, 1.2 - 1.8 * (k % 5) / 4   # arms up/down
    bp[17 * 3 + 1], bp[18 * 3 + 1] = -1.2 * (k % 3) / 2, 1.0 * (k % 4) / 3            # elbows
    bp[2 * 3] = 0.35 * np.sin(k / 3)                                                  # bend forward
    bp[0], bp[3 * 3] = -0.8 * (k % 2), 1.2 * (k % 2)                                  # knee raise
    return bp


def calibrate(betas, width_weight=1.0):
    eng.shape_est.width_weight = width_weight
    eng.reset(keep_shape=False)
    eng._transl_valid = False
    for _ in range(40):
        w, i, s = frame_of(betas, a_pose())
        eng.update_from_world_landmarks(w, body_points=i, silhouette=s)
    for _ in range(25):  # let the rate-limited shape transition finish
        w, i, s = frame_of(betas, a_pose())
        eng.update_from_world_landmarks(w, body_points=i, silhouette=s)
    return eng.betas[0].numpy().copy()


def metrics(b_est, b_gt):
    est = eng.shape_est
    v_e, J_e = est.shaped(b_est)
    v_g, J_g = est.shaped(b_gt)
    seg = [n for n in SEGMENTS]
    widths = ["w:" + b for b in WIDTH_BANDS]
    le, _ = est.model_terms(b_est.astype(np.float64), seg + widths)
    lg, _ = est.model_terms(b_gt.astype(np.float64), seg + widths)
    d = 100 * np.abs(le - lg)
    return {"height_cm": 100 * abs(np.ptp(v_e[:, 1]) - np.ptp(v_g[:, 1])),
            "segments_max_cm": d[:len(seg)].max(), "widths_cm": d[len(seg):]}


ok = True
for trial in range(3):
    gt = rng.normal(0, 1.0, 10).astype(np.float32)
    b_full = calibrate(gt, 1.0)
    state = eng.calib.state
    m_full = metrics(b_full, gt)
    b_len = calibrate(gt, 0.0)
    m_len = metrics(b_len, gt)
    m_tmp = metrics(np.zeros(10, np.float32), gt)
    print(f"[shape] body {trial}: state {state}; height err {m_full['height_cm']:.1f} cm (template {m_tmp['height_cm']:.1f}); "
          f"segments max {m_full['segments_max_cm']:.1f} cm; height lengths-only {m_len['height_cm']:.1f}; front widths waist/hips "
          f"{np.round(m_full['widths_cm'], 1).tolist()} cm with silhouette vs {np.round(m_len['widths_cm'], 1).tolist()} lengths-only "
          f"vs {np.round(m_tmp['widths_cm'], 1).tolist()} template")
    ok &= state == "CALIBRATED" and m_full["height_cm"] < 3.0 and m_full["widths_cm"].max() < 2.5
    ok &= m_full["widths_cm"].mean() < m_len["widths_cm"].mean()
eng.shape_est.width_weight = 1.0
assert ok, "calibration accuracy"

# Pose independence: after calibration, big movements must not change the body shape
gt = rng.normal(0, 1.0, 10).astype(np.float32)
b0 = calibrate(gt)
for k in range(70):
    w, i, s = frame_of(gt, moving_pose(k))
    eng.update_from_world_landmarks(w, body_points=i, silhouette=s)
drift_cm = eng.shape_est.max_change_cm(b0, eng.betas[0].numpy())
print(f"[pose independence] 70 frames of arms up/down, elbows, bending, knee raises: largest change of any "
      f"body measurement {drift_cm:.2f} cm")
assert drift_cm < 1.5

# Person switching: A then B -> the result is B's shape, not a mix
A, B = rng.normal(0, 1.0, 10).astype(np.float32), rng.normal(0, 1.0, 10).astype(np.float32)
bA = calibrate(A)
bB = calibrate(B)
hA, hB, hE = (np.ptp(eng.shape_est.shaped(x)[0][:, 1]) * 100 for x in (A, B, bB))
print(f"[switch] person A height {hA:.1f} cm, person B {hB:.1f} cm -> after switching to B: {hE:.1f} cm; "
      f"|bB-B| {np.linalg.norm(metrics(bB, B)['widths_cm']):.1f} cm widths vs |bB-A| {np.linalg.norm(metrics(bB, A)['widths_cm']):.1f}")
assert abs(hE - hB) < 3.0 and abs(hE - hB) < abs(hE - hA) or abs(hA - hB) < 3.0

# Calibration pose gate: non-neutral poses are refused with the right instruction
def gate(bp, go=(0, 0, 0)):
    with torch.no_grad():
        o = eng.layer(body_pose=torch.tensor(bp)[None], global_orient=torch.tensor(go, dtype=torch.float32)[None])
    J = o.joints[0].numpy()
    P = np.zeros((33, 3))
    img = np.zeros((33, 2))
    for j, m in MP_OF.items():
        P[m] = J[j] * FLIP
        img[m] = project(J[j][None])[0]
    ok, msg, _ = eng.calib.check_pose(P, np.full(33, 0.95), img)
    return ok, msg


cases = {"neutral A-pose": (a_pose(), (0, 0, 0)),
         "arms at sides": (np.r_[np.zeros(45), [0, 0, -1.45, 0, 0, 1.45], np.zeros(12)].astype(np.float32), (0, 0, 0)),
         "turned 50 deg": (a_pose(), (0, 0.87, 0)),
         "leaning 20 deg": (a_pose(), (0, 0, 0.35)),
         "arms raised": (np.r_[np.zeros(45), [0, 0, 0.6, 0, 0, -0.6], np.zeros(12)].astype(np.float32), (0, 0, 0))}
res = {k: gate(*v) for k, v in cases.items()}
for k, (ok_, msg) in res.items():
    print(f"[gate] {k:15s}: {'accepted' if ok_ else 'rejected - ' + msg}")
assert res["neutral A-pose"][0] and not any(res[k][0] for k in cases if k != "neutral A-pose")
print("SHAPE OK")
