"""
Real-time garment cloth solver (XPBD, numpy, no Blender dependency).

Why not Blender's built-in cloth: measured on this machine (i7-10th gen, no GPU) a 3k-vertex cloth
colliding with the 10k-vertex body takes 68 ms/step free-falling and > 2 s/step in contact; it is
frame/cache based (resets when settings change) and its colliders can only push, never grab. The
garment still uses Blender for fitting and attachment (Surface Deform to the personalised, posed
body); this solver moves the cloth around that attached "target" shape.

Model per step (substeps x Jacobi iterations):
  gravity, damping and air drag  ->  predict positions
  stretch + bending distance constraints (compliance = fabric softness)
  attachment: pinned regions are pulled to the target (garment-specific pin weights 0..1)
  body collision: (a) half-space under each vertex's own skin (target minus ease), (b) capsules for
                  other body parts (a swinging sleeve must not enter the torso)
  hands: spheres push the fabric; a grab attaches the particles near the grab point to the hand
         (local pull; the rest of the garment follows through the constraints) until release
Forces from a real hand cannot be sensed by a camera: contact is inferred from tracked hand
position/velocity only.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

DEFAULT_FABRIC = {
    "mass": 0.15,               # kg/m^2 areal density (drag acceleration scales with 1/mass)
    "stretch_compliance": 1e-6,  # lower = less stretchy
    "bend_compliance": 2e-3,     # lower = stiffer folds
    "damping": 1.0,              # 1/s velocity damping
    "air_drag": 0.02,            # drag coefficient
    "friction": 0.3,             # tangential velocity removed on body contact (0..1)
    "thickness": 0.004,          # m kept between cloth and body / hands
    "attach_strength": 0.6,      # fraction of the distance to the target removed per iteration for pin weight 1
    "gravity": -9.81,            # m/s^2 along the world up axis
    "substeps": 3,
    "iterations": 6,
    "max_speed": 8.0,            # m/s clamp: tracking glitches must not explode the cloth
    "tether_slack": 0.04,        # loose regions may stretch 4% beyond rest distance to their anchor
    "relaxation": 1.6,           # Jacobi over-relaxation (1 = plain averaging)
    "teleport_m": 0.25,          # mean target jump per frame treated as a teleport (snap, don't simulate)
}


def mesh_edges(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Unique edges and bending pairs (the two vertices opposite each interior edge)."""
    tri = np.asarray(faces, np.int64)
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    opp = np.concatenate([tri[:, 2], tri[:, 0], tri[:, 1]])
    key = np.sort(e, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    key, opp = key[order], opp[order]
    same = np.all(key[1:] == key[:-1], axis=1)
    edges = np.unique(key, axis=0)
    bend = np.stack([opp[:-1][same], opp[1:][same]], 1)
    bend = bend[bend[:, 0] != bend[:, 1]]
    return edges, bend


def vertex_normals(x: np.ndarray, faces: np.ndarray) -> np.ndarray:
    fn = np.cross(x[faces[:, 1]] - x[faces[:, 0]], x[faces[:, 2]] - x[faces[:, 0]])
    n = np.zeros_like(x)
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


@dataclass
class Grab:
    indices: np.ndarray
    offsets: np.ndarray       # particle position - hand position at grab time


class ClothSim:
    def __init__(self, rest: np.ndarray, faces: np.ndarray, pin_weights: np.ndarray,
                 fabric: Optional[dict] = None, up_axis: int = 2):
        self.f = dict(DEFAULT_FABRIC, **(fabric or {}))
        self.faces = np.asarray(faces, np.int64)
        self.edges, self.bend = mesh_edges(self.faces)
        self.pin = np.clip(np.asarray(pin_weights, np.float64), 0, 1)
        self.x = np.array(rest, np.float64)
        self.v = np.zeros_like(self.x)
        self.up = up_axis
        self.set_rest(rest)
        n = len(self.x)
        self._cnt_e = np.bincount(self.edges.ravel(), minlength=n).astype(np.float64)[:, None] + 1e-9
        self._cnt_b = np.bincount(self.bend.ravel(), minlength=n).astype(np.float64)[:, None] + 1e-9
        self.grabs: Dict[str, Grab] = {}
        self.last_contacts = 0
        self._hand_prev: Dict[str, np.ndarray] = {}
        self._prev_target: Optional[np.ndarray] = None

    def set_rest(self, rest: np.ndarray):
        """Rest lengths from the fitted garment (call again when the body shape changes).
        Tethers: each loosely-attached vertex may not get farther from its nearest firmly-pinned
        vertex than its rest distance (+ tether_slack). Jacobi constraint solving spreads corrections
        only one edge per iteration, so without tethers long free regions sag and over-stretch."""
        rest = np.asarray(rest, np.float64)
        self.L0 = np.linalg.norm(rest[self.edges[:, 1]] - rest[self.edges[:, 0]], axis=1)
        self.Lb = np.linalg.norm(rest[self.bend[:, 1]] - rest[self.bend[:, 0]], axis=1)
        anchors = np.nonzero(self.pin >= 0.5)[0]
        free = np.nonzero(self.pin < 0.5)[0]
        self.t_free = self.t_anchor = self.t_len = np.zeros(0)
        if len(anchors) and len(free):
            # nearest anchors by rest distance (chunked to bound memory)
            best = np.empty(len(free), np.int64)
            dist = np.empty(len(free))
            for s in range(0, len(free), 2048):
                D = np.linalg.norm(rest[free[s:s + 2048], None] - rest[None, anchors], axis=2)
                j = D.argmin(1)
                best[s:s + 2048], dist[s:s + 2048] = anchors[j], D[np.arange(len(j)), j]
            self.t_free, self.t_anchor = free, best
            self.t_len = dist * (1.0 + self.f.get("tether_slack", 0.1))

    def reset(self, positions: np.ndarray):
        self.x = np.array(positions, np.float64)
        self.v[:] = 0
        self.grabs.clear()

    # ---- hands ----
    def grab(self, hand: str, center: np.ndarray, radius: float) -> int:
        """Attach the particles within `radius` of `center` to the hand (pull). Returns count."""
        idx = np.nonzero(np.linalg.norm(self.x - center, axis=1) < radius)[0]
        if len(idx):
            self.grabs[hand] = Grab(idx, self.x[idx] - center)
        return len(idx)

    def release(self, hand: str):
        self.grabs.pop(hand, None)

    # ---- solver ----
    def _distance(self, p, pairs, L0, cnt, alpha):
        d = p[pairs[:, 1]] - p[pairs[:, 0]]
        l = np.linalg.norm(d, axis=1)
        C = l - L0
        lam = -C / (2.0 + alpha)                      # unit inverse mass on both ends
        corr = (lam / np.maximum(l, 1e-9))[:, None] * d
        dp = np.zeros_like(p)
        for k in range(3):
            dp[:, k] += np.bincount(pairs[:, 1], corr[:, k], len(p)) - np.bincount(pairs[:, 0], corr[:, k], len(p))
        return p + self.f["relaxation"] * dp / cnt   # over-relaxed Jacobi (averaging is over-conservative)

    @staticmethod
    def _push_out_segments(p, A, B, R):
        """Push points out of capsules (A,B,R arrays; spheres have A == B), all capsules at once:
        each point is pushed out of the capsule it penetrates deepest. Returns p, contact mask."""
        AB = B - A                                                     # (C,3)
        t = np.clip(np.einsum("nck,ck->nc", p[:, None, :] - A[None], AB) / np.maximum((AB * AB).sum(1), 1e-12), 0, 1)
        C = A[None] + t[..., None] * AB[None]                          # (N,C,3) closest points
        D = p[:, None, :] - C
        dist = np.linalg.norm(D, axis=2)
        pen = R[None] - dist                                           # > 0 inside
        j = pen.argmax(1)
        rows = np.arange(len(p))
        hit = pen[rows, j] > 0
        if hit.any():
            h = rows[hit]
            n = D[h, j[hit]] / np.maximum(dist[h, j[hit]], 1e-9)[:, None]
            p[h] = C[h, j[hit]] + n * R[j[hit]][:, None]
        return p, hit

    @staticmethod
    def _push_out_hand(p, x_prev, c, c_prev, r):
        """Hand spheres with a swept side test: a particle is pushed out on the side of the hand it
        was on BEFORE the hand moved (otherwise a fast hand tunnels through and the fabric pops out
        in front of it, instead of being carried along).
        ponytail: particle-vs-sphere only; a push deeper than ~2x the hand radius into UNSUPPORTED
        cloth lets the fabric slide round the sphere's rim. On a worn garment the body stops the push
        long before that; triangle-vs-sphere contact would remove the limit."""
        dist = np.linalg.norm(p[:, None, :] - c[None], axis=2)        # (N,K)
        pen = r[None] - dist
        j = pen.argmax(1)
        rows = np.arange(len(p))
        hit = pen[rows, j] > 0
        if hit.any():
            h, jh = rows[hit], j[hit]
            side = x_prev[h] - c_prev[jh]
            n = side / np.maximum(np.linalg.norm(side, axis=1), 1e-9)[:, None]
            p[h] = c[jh] + n * r[jh][:, None]
        return p, hit

    def step(self, dt: float, target: np.ndarray, capsules: Optional[Tuple] = None,
             hands: Optional[Dict[str, Tuple[np.ndarray, float]]] = None, ease: float = 0.012) -> np.ndarray:
        """Advance by dt seconds. target: garment glued to the posed body (N,3).
        capsules: (A, B, R) arrays of body capsules. hands: {name: (centers (k,3), radii (k,))}.
        ease: how far the fitted garment sits outside the skin (half-space margin)."""
        f = self.f
        dt = float(np.clip(dt, 1e-3, 0.1))     # a stalled frame must not become a huge step
        target = np.asarray(target, np.float64)
        # Teleport (tracking glitch, person re-acquired elsewhere, first frame): a body cannot move
        # this far in one frame, so snap the cloth to the fitted shape instead of dragging it.
        if self._prev_target is None or np.abs(target - self._prev_target).mean() > f["teleport_m"]:
            self.reset(target)
            self._prev_target = target.copy()
            return self.x
        self._prev_target = target.copy()
        s = int(f["substeps"])
        h = dt / s
        g = np.zeros(3)
        g[self.up] = f["gravity"]
        target = np.asarray(target, np.float64)
        n_t = vertex_normals(target, self.faces)
        inward = max(0.0, ease - f["thickness"])  # cloth may sink this far towards its own skin
        a_s, a_b = f["stretch_compliance"] / h ** 2, f["bend_compliance"] / h ** 2
        attach = f["attach_strength"] * self.pin[:, None]
        if capsules is not None:
            cA, cB, cR = (np.asarray(c, np.float64) for c in capsules)
            cR = cR + f["thickness"]
        contact = np.zeros(len(self.x), bool)
        hands = hands or {}
        prev = {k: (self._hand_prev[k] if k in self._hand_prev and len(self._hand_prev[k]) == len(v[0])
                    else np.asarray(v[0], np.float64)) for k, v in hands.items()}
        x_start = self.x.copy()
        for si in range(s):
            self.v += h * g
            self.v *= max(0.0, 1.0 - (f["damping"] + f["air_drag"] / f["mass"]) * h)
            p = self.x + h * self.v
            for _ in range(int(f["iterations"])):
                p = self._distance(p, self.edges, self.L0, self._cnt_e, a_s)
                p = self._distance(p, self.bend, self.Lb, self._cnt_b, a_b)
                p += attach * (target - p)
                if len(self.t_free):  # tethers (unilateral: only pull back when too far)
                    d = p[self.t_free] - p[self.t_anchor]
                    l = np.linalg.norm(d, axis=1)
                    over = l > self.t_len
                    if over.any():
                        fi = self.t_free[over]
                        p[fi] = p[self.t_anchor[over]] + d[over] * (self.t_len[over] / l[over])[:, None]
                # own-skin half-space: never deeper than `inward` behind the fitted surface
                depth = np.einsum("ij,ij->i", p - target, n_t) + inward
                under = depth < 0
                p[under] -= depth[under, None] * n_t[under]
                contact |= under
                for hand, gr in self.grabs.items():
                    if hand in hands:
                        p[gr.indices] = hands[hand][0][0] + gr.offsets
            # colliders once per substep, after the constraints (they are the costliest part)
            if capsules is not None:
                p, c = self._push_out_segments(p, cA, cB, cR)
                contact |= c
            for name, (hc, hr) in hands.items():
                a = (si + 1) / s   # hand moves smoothly across the substeps
                hc_now = prev[name] + a * (np.asarray(hc, np.float64) - prev[name])
                p, c = self._push_out_hand(p, x_start, hc_now, prev[name], np.asarray(hr) + f["thickness"])
                contact |= c
            for hand, gr in self.grabs.items():
                if hand in hands:
                    p[gr.indices] = hands[hand][0][0] + gr.offsets
            v = (p - self.x) / h
            if contact.any():  # friction on body/hand contact
                v[contact] *= 1.0 - f["friction"]
            sp = np.linalg.norm(v, axis=1, keepdims=True)
            v *= np.minimum(1.0, f["max_speed"] / np.maximum(sp, 1e-9))
            self.v, self.x = v, p
        self.last_contacts = int(contact.sum())
        self._hand_prev = {k: np.asarray(v[0], np.float64).copy() for k, v in hands.items()}
        return self.x
