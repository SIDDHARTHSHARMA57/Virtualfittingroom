"""
Body shape (SMPL-X betas) from MediaPipe segment lengths.

SMPL-X joints are linear in the shape coefficients: J(b) = J0 + Jd @ b. So we collect robust
(median) lengths of segments whose MediaPipe and SMPL-X definitions agree (limbs, shoulder width,
torso) over many frames, then solve a small regularised least-squares problem for b.

Hip width is deliberately not used: MediaPipe's hip landmarks sit at the outer hip, SMPL-X's hip
joints are inside the pelvis. Monocular scale is MediaPipe's metric estimate; proportions are the
reliable part, absolute height is approximate (a height entered by the operator would fix it).
"""
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# name -> (SMPL-X joints at end A (averaged), at end B, MediaPipe ids at A, at B)
SEGMENTS: Dict[str, Tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int]]] = {
    "l_upper_arm": ((16,), (18,), (11,), (13,)),
    "r_upper_arm": ((17,), (19,), (12,), (14,)),
    "l_forearm":   ((18,), (20,), (13,), (15,)),
    "r_forearm":   ((19,), (21,), (14,), (16,)),
    "l_thigh":     ((1,), (4,), (23,), (25,)),
    "r_thigh":     ((2,), (5,), (24,), (26,)),
    "l_shin":      ((4,), (7,), (25,), (27,)),
    "r_shin":      ((5,), (8,), (26,), (28,)),
    "shoulders":   ((16,), (17,), (11,), (12,)),
    "torso":       ((1, 2), (16, 17), (23, 24), (11, 12)),
}


class ShapeEstimator:
    def __init__(self, J_regressor: np.ndarray, v_template: np.ndarray, shapedirs: np.ndarray,
                 n_betas: int = 10, window: int = 90, min_samples: int = 15, reg: float = 0.001):
        """shapedirs: (V, 3, >=n_betas). window: samples kept per segment (median over them)."""
        self.J0 = J_regressor @ v_template                                        # (55,3)
        self.Jd = np.einsum("jv,vcb->jcb", J_regressor, shapedirs[:, :, :n_betas])  # (55,3,B)
        self.n = n_betas
        self.min_samples = min_samples
        self.reg = reg
        self.samples = {k: deque(maxlen=window) for k in SEGMENTS}
        self.last_residual_cm: Dict[str, float] = {}

    def add(self, world_points: List[List[float]], conf: np.ndarray, min_conf: float = 0.5):
        """One frame of MediaPipe world landmarks (metres) + per-landmark confidence."""
        P = np.array([p[:3] for p in world_points], dtype=np.float64)
        for name, (_, _, ma, mb) in SEGMENTS.items():
            if min(conf[i] for i in (*ma, *mb)) >= min_conf:
                self.samples[name].append(float(np.linalg.norm(P[list(ma)].mean(0) - P[list(mb)].mean(0))))

    def targets(self) -> Dict[str, float]:
        return {k: float(np.median(v)) for k, v in self.samples.items() if len(v) >= self.min_samples}

    def _lengths(self, b: np.ndarray, names) -> Tuple[np.ndarray, np.ndarray]:
        """Segment lengths for betas b and their Jacobian (len, B)."""
        J = self.J0 + self.Jd @ b
        L, G = [], []
        for n in names:
            ja, jb = SEGMENTS[n][0], SEGMENTS[n][1]
            d = J[list(ja)].mean(0) - J[list(jb)].mean(0)
            dd = self.Jd[list(ja)].mean(0) - self.Jd[list(jb)].mean(0)            # (3,B)
            l = np.linalg.norm(d)
            L.append(l)
            G.append(d @ dd / max(l, 1e-9))
        return np.array(L), np.array(G)

    def estimate(self, b0: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Betas matching the observed segment lengths (relative error), or None if too few segments."""
        tg = self.targets()
        if len(tg) < 4:
            return None
        names = list(tg)
        t = np.array([tg[n] for n in names])
        b = np.zeros(self.n) if b0 is None else np.array(b0, dtype=np.float64)
        for _ in range(15):  # Gauss-Newton on relative length error + ridge prior
            L, G = self._lengths(b, names)
            r = (L - t) / t
            Jr = G / t[:, None]
            A = Jr.T @ Jr + self.reg * np.eye(self.n)
            step = np.linalg.solve(A, -(Jr.T @ r + self.reg * b))
            b = np.clip(b + step, -3.0, 3.0)
            if np.abs(step).max() < 1e-4:
                break
        L, _ = self._lengths(b, names)
        self.last_residual_cm = {n: round(100 * (l - x), 1) for n, l, x in zip(names, L, t)}
        return b.astype(np.float32)
