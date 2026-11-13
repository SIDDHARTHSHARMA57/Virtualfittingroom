"""Synthetic round-trip: known SMPL-X pose -> fake MediaPipe landmarks -> fit -> compare bone directions.
Covers front / side / rear (body yaw 0, 90, 180 deg). Run: python tests/test_smplx_fitter.py"""
import os
import sys
import time
import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.camera.smplx_fitter import SMPLXFittingEngine, KEYPOINTS, BONES, KP  # noqa: E402

eng = SMPLXFittingEngine(model_dir=os.path.join(ROOT, "models"))
assert eng.enabled, "SMPL-X model missing"


from src.camera.body_model import clamp_body_pose  # noqa: E402


def is_valid(bp):
    """Ground truth must itself be anatomically legal and collision-free, or the fitter is
    *right* to deviate from it."""
    if clamp_body_pose(bp, eng.limits)[1]:
        return False
    with torch.no_grad():
        _, R, P = eng._fk_keypoints(torch.zeros(1, 3), torch.tensor(bp)[None], return_rot=True)
    return not eng.collision.penetrations(P, R)


def gt_pose(rng):
    while True:
        bp = _raw_pose(rng)
        if is_valid(bp):
            return bp


def _raw_pose(rng):
    bp = np.zeros(63, dtype=np.float32)
    def j(i): return slice((i - 1) * 3, i * 3)  # smplx joint id -> body_pose slice
    bp[j(16)] = rng.uniform(-0.8, 0.8, 3)          # L shoulder
    bp[j(17)] = rng.uniform(-0.8, 0.8, 3)          # R shoulder
    bp[j(18)][1] = -rng.uniform(0.2, 1.8)          # L elbow bend (-y)
    bp[j(19)][1] = rng.uniform(0.2, 1.8)           # R elbow bend (+y)
    bp[j(1)] = rng.uniform(-0.5, 0.5, 3) * [1, 0.3, 0.3]  # hips
    bp[j(2)] = rng.uniform(-0.5, 0.5, 3) * [1, 0.3, 0.3]
    bp[j(4)][0] = rng.uniform(0.0, 1.2)            # knees (+x)
    bp[j(5)][0] = rng.uniform(0.0, 1.2)
    bp[j(3)] = rng.uniform(-0.15, 0.15, 3)         # spine
    return bp


def fake_mediapipe(kp_smplx):
    """Inverse of mp_to_smplx, writing each keypoint onto its MediaPipe ids (33 x [x,y,z,vis])."""
    mp = np.zeros((33, 4), dtype=np.float32)
    mp[:, 3] = 1.0
    for k, (_, _, mids) in enumerate(KEYPOINTS):
        if len(mids) == 1:
            x, y, z = kp_smplx[k]
            mp[mids[0], :3] = [x, -y, -z]
    # composite keypoints (pelvis, chest, head) are means of their ids; ears need explicit values
    h = kp_smplx[KP["head"]]
    mp[7, :3] = mp[8, :3] = [h[0], -h[1], -h[2]]
    return mp.tolist()


def bone_dirs(kp):
    d = np.array([kp[KP[c]] - kp[KP[p]] for p, c, _ in BONES])
    return d / np.linalg.norm(d, axis=1, keepdims=True)


rng = np.random.default_rng(0)
worst = 0.0
for yaw_deg in (0, 90, 180):
    for trial in range(4):
        bp = gt_pose(rng)
        go = np.array([0, np.deg2rad(yaw_deg), 0], dtype=np.float32)
        with torch.no_grad():
            kp_gt = eng._fk_keypoints(torch.tensor(go)[None], torch.tensor(bp)[None]).numpy()

        eng.reset()
        t0 = time.perf_counter()
        out = eng.update_from_world_landmarks(fake_mediapipe(kp_gt))
        cold_ms = (time.perf_counter() - t0) * 1000
        assert out["fitted"]
        # warm frame (steady-state live latency)
        t0 = time.perf_counter()
        eng.update_from_world_landmarks(fake_mediapipe(kp_gt))
        warm_ms = (time.perf_counter() - t0) * 1000

        with torch.no_grad():
            kp_fit = eng._fk_keypoints(eng.global_orient, eng.body_pose).numpy()
        err = np.degrees(np.arccos(np.clip((bone_dirs(kp_fit) * bone_dirs(kp_gt)).sum(1), -1, 1)))
        # head->eye bones: head keypoint differs by definition (mid-ears vs joint) only in real data
        worst = max(worst, err.max())
        print(f"yaw {yaw_deg:3d} #{trial}: mean {err.mean():5.2f} deg, max {err.max():5.2f} deg "
              f"({BONES[int(err.argmax())][1]}), cold {cold_ms:5.0f} ms, warm {warm_ms:4.0f} ms")

assert worst < 10.0, f"worst bone error {worst:.1f} deg"

# Continuous motion: random pose keyframes 0.5 s apart at ~10 fps inference, while turning a full 360 deg.
eng.reset()
keys = [gt_pose(rng) for _ in range(7)]
errs, times = [], []
steps = 5
for s in range(len(keys) - 1):
    for t in range(steps):
        a = t / steps
        bp = (1 - a) * keys[s] + a * keys[s + 1]
        yaw = np.deg2rad(360.0 * (s * steps + t) / ((len(keys) - 1) * steps))
        go = np.array([0, yaw, 0], dtype=np.float32)
        with torch.no_grad():
            kp_gt = eng._fk_keypoints(torch.tensor(go)[None], torch.tensor(bp)[None]).numpy()
        t0 = time.perf_counter()
        eng.update_from_world_landmarks(fake_mediapipe(kp_gt))
        times.append((time.perf_counter() - t0) * 1000)
        with torch.no_grad():
            kp_fit = eng._fk_keypoints(eng.global_orient, eng.body_pose).numpy()
        errs.append(np.degrees(np.arccos(np.clip((bone_dirs(kp_fit) * bone_dirs(kp_gt)).sum(1), -1, 1))).mean())
errs = np.array(errs[1:])  # skip cold frame
print(f"motion: mean {errs.mean():.2f} deg, p95 {np.percentile(errs, 95):.2f} deg, "
      f"latency p50 {np.median(times[1:]):.0f} ms p95 {np.percentile(times[1:], 95):.0f} ms")
assert np.percentile(errs, 95) < 8.0

# Hands: random wrist + finger curl, rendered through the real SMPL-X layer (incl. posed fingertip
# vertices) -> fake MediaPipe hand world landmarks -> fit -> compare finger segment directions.
from src.camera.smplx_fitter import HAND_MP_CHAINS, TIP_VERTS, HAND_BASE, WRIST  # noqa: E402


def mp_hand(joints, verts, side):
    b, w = HAND_BASE[side], WRIST[side]
    H = np.zeros((21, 3), np.float32)
    H[0] = joints[w]
    for f, chain in enumerate(HAND_MP_CHAINS):
        for k in range(3):
            H[chain[k]] = joints[b + f * 3 + k]
        H[chain[3]] = verts[TIP_VERTS[side][f]]
    return H


def seg_dirs(H):
    d = np.array([H[c[k + 1]] - H[c[k]] for c in HAND_MP_CHAINS for k in range(3)]
                 + [H[9] - H[0], H[5] - H[17]])
    return d / np.linalg.norm(d, axis=1, keepdims=True)


hand_worst = 0.0
for yaw_deg in (0, 90, 180):
    for trial in range(3):
        while True:  # random wrists, kept only if anatomically legal
            bp = gt_pose(rng)
            bp[19 * 3:19 * 3 + 3] = rng.uniform(-0.7, 0.7, 3) * [1, 0.5, 1]   # left wrist
            bp[20 * 3:20 * 3 + 3] = rng.uniform(-0.7, 0.7, 3) * [1, 0.5, 1]   # right wrist
            if is_valid(bp):
                break
        hp = {s: np.zeros(45, np.float32) for s in ("left", "right")}
        for s in hp:
            for f in range(5):
                curl = rng.uniform(0.0, 1.3)
                for k in range(3):
                    hp[s][(f * 3 + k) * 3 + 2] = curl * (1 if s == "left" else -1) * (0.6 if f == 4 else 1)
        go = np.array([0, np.deg2rad(yaw_deg), 0], dtype=np.float32)
        with torch.no_grad():
            out = eng.layer(betas=eng.betas, global_orient=torch.tensor(go)[None],
                            body_pose=torch.tensor(bp)[None],
                            left_hand_pose=torch.tensor(hp["left"])[None],
                            right_hand_pose=torch.tensor(hp["right"])[None])
        J, V = out.joints[0].numpy(), out.vertices[0].numpy()
        with torch.no_grad():
            kp_gt = eng._fk_keypoints(torch.tensor(go)[None], torch.tensor(bp)[None]).numpy()
        Hs = {s: mp_hand(J, V, s) for s in ("left", "right")}
        to_mp = lambda P: np.stack([P[:, 0], -P[:, 1], -P[:, 2]], 1)
        # image coords only steer left/right assignment: put each hand next to its pose wrist
        hands_img = {"A": [[0.2, 0.5, 0]], "B": [[0.8, 0.5, 0]]}
        body_img = [[0.5, 0.5, 1]] * 33  # in-frame (off-frame landmarks are now ignored)
        body_img[15], body_img[16] = [0.8, 0.5, 1], [0.2, 0.5, 1]   # left wrist <-> "B"
        hands_world = {"B": to_mp(Hs["left"]).tolist(), "A": to_mp(Hs["right"]).tolist()}

        eng.reset()
        for _ in range(2):
            res = eng.update_from_world_landmarks(fake_mediapipe(kp_gt), body_points=body_img,
                                                  hands=hands_img, hands_world=hands_world)
        assert res["hands_tracked"] == 2
        with torch.no_grad():
            fo = eng.layer(betas=eng.betas, global_orient=eng.global_orient, body_pose=eng.body_pose,
                           left_hand_pose=torch.tensor(eng.hand_pose["left"])[None],
                           right_hand_pose=torch.tensor(eng.hand_pose["right"])[None])
        Jf, Vf = fo.joints[0].numpy(), fo.vertices[0].numpy()
        for s in ("left", "right"):
            e = np.degrees(np.arccos(np.clip((seg_dirs(mp_hand(Jf, Vf, s)) * seg_dirs(Hs[s])).sum(1), -1, 1)))
            hand_worst = max(hand_worst, e.max())
            print(f"hand {s:5s} yaw {yaw_deg:3d} #{trial}: mean {e.mean():5.2f} deg, max {e.max():5.2f} deg")
assert hand_worst < 10.0, f"worst finger error {hand_worst:.1f} deg"

# Close to the mirror: only shoulders/arms/head visible, body turned 45 deg -> must stay upright and turn.
bp = gt_pose(rng)
bp[:15 * 3] = 0  # legs/spine neutral
go = np.array([0, np.deg2rad(45), 0], dtype=np.float32)
with torch.no_grad():
    kp_gt = eng._fk_keypoints(torch.tensor(go)[None], torch.tensor(bp)[None]).numpy()
mp = np.array(fake_mediapipe(kp_gt))
mp[23:33, 3] = 0.05  # hips + legs out of frame
eng.reset()
for _ in range(3):
    out = eng.update_from_world_landmarks(mp.tolist())
assert out["fitted"]
R = cv2.Rodrigues(eng.global_orient[0].numpy().astype(np.float64))[0]
tilt = np.degrees(np.arccos(R[1, 1]))
yaw = np.degrees(np.arctan2(R[0, 2], R[2, 2]))
print(f"upper-body only: tilt {tilt:.1f} deg (want ~0), yaw {yaw:.1f} deg (want ~45)")
assert tilt < 10 and abs(yaw - 45) < 10

# The "falling body" case: MediaPipe guesses legs *below the image* with HIGH visibility.
# Those legs are garbage (here: swung up sideways); off-image gating must ignore them.
mp2 = np.array(fake_mediapipe(kp_gt))
mp2[23:33, 3] = 0.9
mp2[25:33, 0] += 0.5   # nonsense leg positions
mp2[25:33, 1] -= 0.6
body_img = [[0.5, 0.4, 1.0]] * 33
body_img = [list(p) for p in body_img]
for i in range(23, 33):
    body_img[i] = [0.5, 1.3, 0.9]  # below the frame
eng.reset()
for _ in range(3):
    out = eng.update_from_world_landmarks(mp2.tolist(), body_points=body_img)
R = cv2.Rodrigues(eng.global_orient[0].numpy().astype(np.float64))[0]
tilt = np.degrees(np.arccos(R[1, 1]))
print(f"off-frame hallucinated legs: tilt {tilt:.1f} deg, tracked limbs {out['valid_joints']}/12")
assert tilt < 10 and out["valid_joints"] == 6
print("OK")
