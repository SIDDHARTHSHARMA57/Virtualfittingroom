"""
Self-collision proxies for the SMPL-X body (separate from joint limits: a legal elbow angle
can still put the forearm inside the chest).

Proxies are capsules (segment + radius) attached to SMPL-X joints. Radii are measured from the
template mesh: the median distance of each bone's skinned vertices to its segment. The torso's
elliptic cross-section is approximated by two side-by-side capsules.

The solver uses `penalty()` (differentiable, batched: one call checks every pair) so corrections
are smooth and stay as close as possible to the tracked motion; `penetrations()` verifies a pose.
"""
from typing import Dict, List, Tuple

import numpy as np
import torch

# name -> (joint a, joint b, bone whose vertices define the radius, (start, end) fraction of a->b)
LIMB_PROXIES = {
    "l_elbow":   (16, 18, 16, (0.6, 1.0)),   # distal upper arm (the shoulder end touches the torso legally)
    "r_elbow":   (17, 19, 17, (0.6, 1.0)),
    "l_forearm": (18, 20, 18, (0.0, 1.0)),
    "r_forearm": (19, 21, 19, (0.0, 1.0)),
    "l_thigh":   (1, 4, 1, (0.25, 1.0)),     # thighs legally touch at the crotch
    "r_thigh":   (2, 5, 2, (0.25, 1.0)),
    "l_shin":    (4, 7, 4, (0.0, 1.0)),
    "r_shin":    (5, 8, 5, (0.0, 1.0)),
}
HANDS = {"l_hand": (20, 25, 31), "r_hand": (21, 40, 46)}   # wrist, index1, pinky1 -> sphere
TORSO = (0, 9)            # pelvis -> spine3
TORSO_BONES = (0, 3, 6, 9)


def needed_joints() -> List[int]:
    js = set(TORSO) | {j for a, b, _, _ in LIMB_PROXIES.values() for j in (a, b)}
    return sorted(js | {j for t in HANDS.values() for j in t})


class BodyCollision:
    def __init__(self, rest_joints: np.ndarray, rest_verts: np.ndarray, lbs_weights: np.ndarray,
                 cfg: dict, index_of: Dict[int, int], device="cpu"):
        """index_of maps SMPL-X joint id -> row in the posed-joint tensor the solver passes in."""
        self.enabled = bool(cfg.get("enabled", True))
        self.weight = float(cfg.get("weight", 300.0))
        self.slack = float(cfg.get("slack_m", 0.01))
        s = float(cfg.get("radius_scale", 0.9))
        owner = lbs_weights.argmax(axis=1)
        J, V = rest_joints, rest_verts

        def seg_dist(P, a, b):
            d = b - a
            t = np.clip(((P - a) @ d) / (d @ d + 1e-12), 0, 1)
            return np.linalg.norm(P - (a + t[:, None] * d), axis=1)

        names, radii = [], []
        for name, (a, b, bone, _) in LIMB_PROXIES.items():
            names.append(name)
            radii.append(s * float(np.median(seg_dist(V[owner == bone], J[a], J[b]))))
        for name, (w, i1, p1) in HANDS.items():
            c = 0.5 * (J[w] + 0.5 * (J[i1] + J[p1]))
            names.append(name)
            radii.append(s * float(np.median(np.linalg.norm(V[owner == w] - c, axis=1))))
        T = V[np.isin(owner, TORSO_BONES)]
        half_depth = 0.5 * (np.percentile(T[:, 2], 90) - np.percentile(T[:, 2], 10))
        half_width = 0.5 * (np.percentile(T[:, 0], 90) - np.percentile(T[:, 0], 10))
        names += ["torso", "torso"]
        radii += [s * float(half_depth)] * 2
        self.torso_offset = float(max(0.0, half_width - half_depth))
        self.radius = {n: r for n, r in zip(names, radii)}

        ix = lambda js: torch.tensor([index_of[j] for j in js], device=device)
        self._la = ix([v[0] for v in LIMB_PROXIES.values()])
        self._lb = ix([v[1] for v in LIMB_PROXIES.values()])
        self._f0 = torch.tensor([[v[3][0]] for v in LIMB_PROXIES.values()], device=device)
        self._f1 = torch.tensor([[v[3][1]] for v in LIMB_PROXIES.values()], device=device)
        self._hw, self._hi, self._hp = (ix([v[k] for v in HANDS.values()]) for k in range(3))
        self._tlo, self._thi = index_of[TORSO[0]], index_of[TORSO[1]]
        self._rad = torch.tensor(radii, device=device)

        caps_of: Dict[str, List[int]] = {}
        for i, n in enumerate(names):
            caps_of.setdefault(n, []).append(i)
        self.pairs: List[Tuple[str, str]] = [tuple(p) for p in cfg.get("pairs", [])]
        ci, cj, cp = [], [], []
        for p, (na, nb) in enumerate(self.pairs):
            for a in caps_of[na]:
                for b in caps_of[nb]:
                    ci.append(a), cj.append(b), cp.append(p)
        self._ci, self._cj = torch.tensor(ci, device=device), torch.tensor(cj, device=device)
        self._cp = cp

    def capsules(self, P: torch.Tensor, R: torch.Tensor):
        """P (n,3) posed joints, R (n,3,3) global rotations -> segment starts A, ends B (caps,3)."""
        Pa, Pb = P[self._la], P[self._lb]
        d = Pb - Pa
        c = 0.5 * (P[self._hw] + 0.5 * (P[self._hi] + P[self._hp]))
        o_lo = R[self._tlo][:, 0] * self.torso_offset   # body-left axis follows the torso
        o_hi = R[self._thi][:, 0] * self.torso_offset
        plo, phi = P[self._tlo], P[self._thi]
        A = torch.cat([Pa + self._f0 * d, c, torch.stack([plo + o_lo, plo - o_lo])])
        B = torch.cat([Pa + self._f1 * d, c, torch.stack([phi + o_hi, phi - o_hi])])
        return A, B

    @staticmethod
    def segment_distance(p1, q1, p2, q2):
        """Batched closest distance between segments p1q1 and p2q2 (rows), differentiable."""
        eps = 1e-9
        d1, d2, r = q1 - p1, q2 - p2, p1 - p2
        a = (d1 * d1).sum(1) + eps
        e = (d2 * d2).sum(1) + eps
        b, c, f = (d1 * d2).sum(1), (d1 * r).sum(1), (d2 * r).sum(1)
        s = ((b * f - c * e) / (a * e - b * b).clamp_min(eps)).clamp(0, 1)
        t = (b * s + f) / e
        s = torch.where(t < 0, (-c / a).clamp(0, 1), torch.where(t > 1, ((b - c) / a).clamp(0, 1), s))
        t = t.clamp(0, 1)
        return ((p1 + d1 * s[:, None]) - (p2 + d2 * t[:, None])).norm(dim=1)

    def depths(self, A, B) -> torch.Tensor:
        """Penetration depth per capsule combination (>0 = overlapping beyond slack)."""
        dist = self.segment_distance(A[self._ci], B[self._ci], A[self._cj], B[self._cj])
        return self._rad[self._ci] + self._rad[self._cj] - self.slack - dist

    def penalty(self, P, R) -> torch.Tensor:
        if not self.enabled or not self.pairs:
            return P.new_zeros(())
        return self.weight * (torch.relu(self.depths(*self.capsules(P, R))) ** 2).sum()

    def penetrations(self, P, R, tol: float = 0.0) -> Dict[str, float]:
        """{'a/b': depth_m} for configured pairs overlapping beyond slack + tol."""
        with torch.no_grad():
            dep = self.depths(*self.capsules(P, R)).tolist()
        out: Dict[str, float] = {}
        for d, p in zip(dep, self._cp):
            if d > tol:
                k = "/".join(self.pairs[p])
                out[k] = max(out.get(k, 0.0), d)
        return out
