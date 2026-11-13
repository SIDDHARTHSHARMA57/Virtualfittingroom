"""
Anatomical joint limits for the SMPL-X body, shared by the solver and the Blender twin.

Limits are Euler ranges in each joint's local frame, using Blender's Euler conventions
(order "XYZ" = X applied first, matrix = Rz @ Ry @ Rx), so Blender's native Limit Rotation
constraint with the same order and LOCAL space enforces exactly the same thing.

numpy only at import time (Blender has no torch); torch is imported inside the solver helpers.
"""
import os
import json
from typing import Dict, List, Tuple

import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "config", "body_model.json")

# SMPL-X joints 1..21 (body_pose order)
BODY_JOINTS = [
    "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
    "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar",
    "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]
AXIS = {"X": 0, "Y": 1, "Z": 2}


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def joint_limits(cfg: dict) -> Dict[str, dict]:
    """{joint: {"order": str, "lo": rad[3] (x,y,z), "hi": rad[3]}} incl. mirrored right side.
    Mirroring across the body's sagittal plane keeps X rotations and negates Y and Z."""
    out = {}
    for name, lim in cfg["joint_limits_deg"].items():
        lo = np.radians([lim["x"][0], lim["y"][0], lim["z"][0]])
        hi = np.radians([lim["x"][1], lim["y"][1], lim["z"][1]])
        out[name] = {"order": lim["order"], "lo": lo, "hi": hi}
        if name.startswith("left_"):
            out["right_" + name[5:]] = {
                "order": lim["order"],
                "lo": np.array([lo[0], -hi[1], -hi[2]]),
                "hi": np.array([hi[0], -lo[1], -lo[2]]),
            }
    return out


# ---------------- Euler <-> matrix (Blender convention) ----------------
def _parity(order: str) -> float:
    i, j, k = (AXIS[c] for c in order)
    return 1.0 if (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1.0


def _axis_mat(axis: int, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    m = np.eye(3)
    p, q = [(1, 2), (2, 0), (0, 1)][axis]
    m[p, p], m[p, q], m[q, p], m[q, q] = c, -s, s, c
    return m


def euler_to_mat(xyz: np.ndarray, order: str) -> np.ndarray:
    """xyz = angles about X, Y, Z (radians); applied in `order`."""
    R = np.eye(3)
    for c in order:
        R = _axis_mat(AXIS[c], xyz[AXIS[c]]) @ R
    return R


def mat_to_euler(R: np.ndarray, order: str) -> np.ndarray:
    """Inverse of euler_to_mat; returns angles indexed by axis (x, y, z)."""
    i, j, k = (AXIS[c] for c in order)
    s = _parity(order)
    out = np.zeros(3)
    out[j] = np.arcsin(np.clip(-s * R[k, i], -1.0, 1.0))
    out[i] = np.arctan2(s * R[k, j], R[k, k])
    out[k] = np.arctan2(s * R[j, i], R[i, i])
    return out


def rotvec_to_mat(r: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(r))
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def mat_to_rotvec(R: np.ndarray) -> np.ndarray:
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-8:
        return np.zeros(3)
    if th > np.pi - 1e-5:  # near 180 deg: axis from the symmetric part
        k = np.sqrt(np.clip((np.diag(R) + 1) / 2, 0, None))
        k *= np.sign([R[2, 1] - R[1, 2] or 1, R[0, 2] - R[2, 0] or 1, R[1, 0] - R[0, 1] or 1])
        return k / (np.linalg.norm(k) + 1e-12) * th
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th)) * th


def clamp_body_pose(body_pose: np.ndarray, limits: Dict[str, dict]) -> Tuple[np.ndarray, List[str]]:
    """Hard projection of body_pose (63,) into the limits. Returns (clamped, names of joints changed)."""
    bp = np.array(body_pose, dtype=np.float64).copy()
    changed = []
    for idx, name in enumerate(BODY_JOINTS):
        lim = limits.get(name)
        if lim is None:
            continue
        r = bp[idx * 3: idx * 3 + 3]
        e = mat_to_euler(rotvec_to_mat(r), lim["order"])
        ec = np.clip(e, lim["lo"], lim["hi"])
        if np.any(np.abs(ec - e) > 1e-6):
            bp[idx * 3: idx * 3 + 3] = mat_to_rotvec(euler_to_mat(ec, lim["order"]))
            changed.append(name)
    return bp.astype(np.float32), changed


# ---------------- torch (solver) ----------------
class TorchLimits:
    """Differentiable soft limit penalty for the LBFGS solver."""

    def __init__(self, limits: Dict[str, dict], device="cpu"):
        import torch
        self.torch = torch
        self.groups = []  # (order, joint idx tensor, lo (n,3), hi (n,3))
        by_order: Dict[str, list] = {}
        for idx, name in enumerate(BODY_JOINTS):
            if name in limits:
                by_order.setdefault(limits[name]["order"], []).append((idx, limits[name]))
        for order, items in by_order.items():
            self.groups.append((
                order,
                torch.tensor([i for i, _ in items], device=device),
                torch.tensor(np.stack([l["lo"] for _, l in items]), dtype=torch.float32, device=device),
                torch.tensor(np.stack([l["hi"] for _, l in items]), dtype=torch.float32, device=device),
            ))

    def euler(self, R, order: str):
        """Batched mat_to_euler. R: (n,3,3) -> (n,3) angles indexed by axis."""
        torch = self.torch
        i, j, k = (AXIS[c] for c in order)
        s = _parity(order)
        ang = [None, None, None]
        ang[j] = torch.asin((-s * R[:, k, i]).clamp(-1 + 1e-6, 1 - 1e-6))
        ang[i] = torch.atan2(s * R[:, k, j], R[:, k, k])
        ang[k] = torch.atan2(s * R[:, j, i], R[:, i, i])
        return torch.stack(ang, dim=1)

    def penalty(self, R_body):
        """R_body: (21,3,3) local rotation matrices -> scalar sum of squared violations (rad^2)."""
        torch = self.torch
        total = R_body.new_zeros(())
        for order, idx, lo, hi in self.groups:
            e = self.euler(R_body[idx], order)
            total = total + (torch.relu(lo - e) ** 2).sum() + (torch.relu(e - hi) ** 2).sum()
        return total


if __name__ == "__main__":
    # self-check: euler round trip for every order, and clamping is a no-op inside the limits
    rng = np.random.default_rng(0)
    for order in ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"):
        for _ in range(200):
            e = rng.uniform(-1.4, 1.4, 3)
            assert np.allclose(mat_to_euler(euler_to_mat(e, order), order), e, atol=1e-9), order
    for _ in range(200):
        r = rng.normal(size=3) * rng.uniform(0, 3)
        assert np.allclose(rotvec_to_mat(mat_to_rotvec(rotvec_to_mat(r))), rotvec_to_mat(r), atol=1e-6)
    lim = joint_limits(load_config())
    assert np.allclose(clamp_body_pose(np.zeros(63), lim)[0], 0)
    print("body_model OK")
