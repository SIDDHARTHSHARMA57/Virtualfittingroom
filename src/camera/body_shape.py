"""
Body shape (SMPL-X betas) estimation for the selected person: guided calibration, gated refinement,
and a measurement report with confidence.

Evidence used
  * Segment lengths (limbs, shoulder width, torso) from MediaPipe world landmarks (metric estimate).
    SMPL-X joints are linear in the betas, so lengths are a small least-squares problem.
  * Front-view silhouette WIDTHS at chest / waist / hips from the person's segmentation outline,
    taken as a ratio to the shoulder width in the same image (scale-free, so camera distance and
    focal length cancel). This is the only evidence of girth; it includes clothing
    (`clothing_allowance_m` is subtracted). Model widths come from fixed extreme vertices of the
    template, which are also linear in the betas.
  Hip width from landmarks is NOT used (MediaPipe's hip points are at the outer hip, SMPL-X's hip
  joints are inside the pelvis).

What cannot be observed from one front RGB camera: body depth and therefore true circumferences.
They are reported as model-estimated (from the fitted body mesh) with low confidence.
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
    "neck_head":   ((16, 17), (15,), (11, 12), (7, 8)),   # shoulders -> head (ears); height above shoulders
}
LEG_SEGMENTS = ("l_thigh", "r_thigh", "l_shin", "r_shin")

# front-view width bands: anchor joints (rest height = their mean y) + vertical offset (m),
# and which bones' vertices form the silhouette there (torso only: excludes the arms).
# No chest band: in any comfortable pose the upper arm touches the torso just below the armpit, so
# the silhouette row at chest height includes the arm (measured +3 cm bias). Chest is model-inferred.
WIDTH_BANDS = {
    "waist": ((3,), 0.04, (0, 3, 6)),
    "hips":  ((1, 2), -0.04, (0, 1, 2)),
}
CHEST_BAND = ((9,), 0.0, (3, 6, 9))   # used only for the model-inferred chest circumference
TORSO_BONES = (0, 3, 6, 9)


class ShapeEstimator:
    """Robust measurement buffers + regularised Gauss-Newton fit of betas."""

    def __init__(self, J_regressor: np.ndarray, v_template: np.ndarray, shapedirs: np.ndarray,
                 lbs_weights: Optional[np.ndarray] = None, n_betas: int = 10, window: int = 150,
                 min_samples: int = 10, reg: float = 0.001, width_weight: float = 1.0):
        self.v0 = v_template.astype(np.float64)
        self.S = shapedirs[:, :, :n_betas].astype(np.float64)                       # (V,3,B)
        self.Jr = J_regressor.astype(np.float64)
        self.J0 = self.Jr @ self.v0
        self.Jd = np.einsum("jv,vcb->jcb", self.Jr, self.S)
        self.n, self.min_samples, self.reg, self.width_weight = n_betas, min_samples, reg, width_weight
        self.samples = {k: deque(maxlen=window) for k in SEGMENTS}
        self.width_ratio = {k: deque(maxlen=window) for k in WIDTH_BANDS}
        self.owner = lbs_weights.argmax(axis=1) if lbs_weights is not None else None
        self.band_verts: Dict[str, Tuple[int, int]] = {}
        if self.owner is not None:
            for name, (anchor, dy, bones) in WIDTH_BANDS.items():
                y = self.J0[list(anchor), 1].mean() + dy
                idx = np.nonzero(np.isin(self.owner, bones) & (np.abs(self.v0[:, 1] - y) < 0.015))[0]
                self.band_verts[name] = (int(idx[self.v0[idx, 0].argmin()]), int(idx[self.v0[idx, 0].argmax()]))
        self.last_residual_cm: Dict[str, float] = {}

    def clear(self):
        for d in list(self.samples.values()) + list(self.width_ratio.values()):
            d.clear()

    # ---- evidence ----
    def add_lengths(self, P: np.ndarray, conf: np.ndarray, min_conf: float = 0.5, legs: bool = True,
                    max_foreshorten: float = 0.5):
        """P: MediaPipe world landmarks (33,3) m. Segments pointing at the camera (|dz|/len >
        max_foreshorten) are skipped: MediaPipe's depth is its least reliable axis."""
        for name, (_, _, ma, mb) in SEGMENTS.items():
            if not legs and name in LEG_SEGMENTS:
                continue
            if min(conf[i] for i in (*ma, *mb)) < min_conf:
                continue
            d = P[list(ma)].mean(0) - P[list(mb)].mean(0)
            L = float(np.linalg.norm(d))
            if L > 1e-3 and abs(d[2]) / L <= max_foreshorten:
                self.samples[name].append(L)

    def add_width_ratio(self, band: str, ratio: float):
        self.width_ratio[band].append(float(ratio))

    # ---- model ----
    def model_terms(self, b: np.ndarray, names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Values and Jacobians (rows) for segment lengths and 'w:<band>' widths."""
        J = self.J0 + self.Jd @ b
        vals, grads = [], []
        for n in names:
            if n.startswith("w:"):
                vl, vr = self.band_verts[n[2:]]
                xl, xr = self.v0[vl, 0] + self.S[vl, 0] @ b, self.v0[vr, 0] + self.S[vr, 0] @ b
                vals.append(xr - xl)
                grads.append(self.S[vr, 0] - self.S[vl, 0])
            else:
                ja, jb = SEGMENTS[n][0], SEGMENTS[n][1]
                d = J[list(ja)].mean(0) - J[list(jb)].mean(0)
                dd = self.Jd[list(ja)].mean(0) - self.Jd[list(jb)].mean(0)
                l = np.linalg.norm(d)
                vals.append(l)
                grads.append(d @ dd / max(l, 1e-9))
        return np.array(vals), np.array(grads)

    def targets(self, clothing_allowance_m: float = 0.0) -> Dict[str, Tuple[float, float]]:
        """{name: (target metres, relative spread)} for measurements with enough samples."""
        out = {}
        for k, v in self.samples.items():
            if len(v) >= self.min_samples:
                a = np.array(v)
                med = float(np.median(a))
                out[k] = (med, float(np.median(np.abs(a - med)) / med))
        if "shoulders" in out:  # widths: silhouette/shoulder ratio x metric shoulder length
            for k, v in self.width_ratio.items():
                if len(v) >= self.min_samples:
                    a = np.array(v)
                    med = float(np.median(a))
                    out["w:" + k] = (med * out["shoulders"][0] - clothing_allowance_m,
                                     float(np.median(np.abs(a - med)) / med))
        return out

    def estimate(self, b0=None, clothing_allowance_m: float = 0.0) -> Optional[np.ndarray]:
        tg = self.targets(clothing_allowance_m)
        if sum(1 for k in tg if not k.startswith("w:")) < 4:
            return None
        names = list(tg)
        t = np.array([tg[n][0] for n in names])
        wts = np.array([self.width_weight if n.startswith("w:") else 1.0 for n in names])
        b = np.zeros(self.n) if b0 is None else np.array(b0, dtype=np.float64)
        for _ in range(20):  # Gauss-Newton on relative errors + ridge prior towards the mean body
            L, G = self.model_terms(b, names)
            r = wts * (L - t) / t
            Jr = wts[:, None] * G / t[:, None]
            step = np.linalg.solve(Jr.T @ Jr + self.reg * np.eye(self.n), -(Jr.T @ r + self.reg * b))
            b = np.clip(b + step, -3.0, 3.0)
            if np.abs(step).max() < 1e-4:
                break
        L, _ = self.model_terms(b, names)
        self.last_residual_cm = {n: round(100 * (l - x), 1) for n, l, x in zip(names, L, t)}
        return b.astype(np.float32)

    def max_change_cm(self, b_old, b_new) -> float:
        """Largest change in any measurable quantity (segments, front widths, height) between two
        shapes. Some beta directions barely change anything visible, so betas distance is a poor
        criterion for 'the body actually looks different'."""
        names = list(SEGMENTS) + ["w:" + k for k in WIDTH_BANDS]
        a, _ = self.model_terms(np.asarray(b_old, np.float64), names)
        c, _ = self.model_terms(np.asarray(b_new, np.float64), names)
        h = abs(np.ptp(self.shaped(b_old)[0][:, 1]) - np.ptp(self.shaped(b_new)[0][:, 1]))
        return float(100 * max(np.abs(a - c).max(), h))

    # ---- report ----
    def shaped(self, b) -> Tuple[np.ndarray, np.ndarray]:
        v = self.v0 + self.S @ np.asarray(b, dtype=np.float64)
        return v, self.Jr @ v

    def _slice_perimeter(self, v, bones, y, half=0.012) -> float:
        import cv2
        idx = np.nonzero(np.isin(self.owner, bones) & (np.abs(v[:, 1] - y) < half))[0]
        if len(idx) < 3:
            return float("nan")
        hull = cv2.convexHull((v[idx][:, [0, 2]] * 1e4).astype(np.float32))
        return float(cv2.arcLength(hull, True) / 1e4)

    def report(self, b, clothing_allowance_m: float = 0.0) -> List[dict]:
        """Measurements of the fitted body with source + confidence (0..1).
        source: 'observed' = measured on the person; 'model' = read off the fitted body model
        (inferred, not measured); circumferences also need a side view / depth to be measured."""
        v, J = self.shaped(b)
        tg = self.targets(clothing_allowance_m)
        out = []

        def conf_of(key):
            if key not in tg:
                return 0.0
            n = len(self.width_ratio[key[2:]]) if key.startswith("w:") else len(self.samples[key])
            return round(float(min(1.0, n / 30.0) * max(0.0, 1.0 - 5.0 * tg[key][1])), 2)

        def add(name, value_m, source, conf, note=""):
            out.append({"name": name, "cm": round(100 * value_m, 1), "source": source,
                        "confidence": conf, "note": note})

        # height: top of head to soles, rest pose (inferred from the observed segments)
        seg_conf = np.mean([conf_of(k) for k in SEGMENTS]) if tg else 0.0
        add("height", v[:, 1].max() - v[:, 1].min(), "model", round(0.8 * float(seg_conf), 2),
            "scale from MediaPipe's metric estimate; calibrate camera / enter height for exact")
        names = {"shoulders": "shoulder width (joint to joint)", "torso": "torso length",
                 "l_upper_arm": "upper arm L", "r_upper_arm": "upper arm R", "l_forearm": "forearm L",
                 "r_forearm": "forearm R", "l_thigh": "thigh L", "r_thigh": "thigh R",
                 "l_shin": "lower leg L", "r_shin": "lower leg R"}
        for k, label in names.items():
            if k in tg:
                add(label, tg[k][0], "observed", conf_of(k))
        for band in WIDTH_BANDS:
            key = "w:" + band
            if key in tg:
                add(f"{band} width (front)", tg[key][0], "observed", conf_of(key),
                    f"silhouette minus {clothing_allowance_m * 100:.0f} cm clothing allowance")
        if self.owner is not None:
            for band, (anchor, dy, bones) in dict(chest=CHEST_BAND, **WIDTH_BANDS).items():
                y = J[list(anchor), 1].mean() + dy
                add(f"{band} circumference", self._slice_perimeter(v, bones, y), "model", 0.3,
                    "inferred: one front camera cannot see body depth (needs side view / depth camera)")
            # limb slices: one leg only (x side of the left leg), wider band (sparser vertices)
            for label, bones, j0, j1, frac in (("thigh circumference", (1, 4), 1, 4, 0.35),
                                               ("calf circumference", (4, 7), 4, 7, 0.3),
                                               ("neck circumference", (12, 15), 12, 15, 0.5)):
                y = J[j0, 1] + frac * (J[j1, 1] - J[j0, 1])
                add(label, self._slice_perimeter(v, bones, y, half=0.02), "model", 0.25, "inferred")
        return out


# ---------------- calibration controller ----------------
IDLE, WAITING, CAPTURING, CALIBRATED = "IDLE", "WAITING_FOR_POSE", "CAPTURING", "CALIBRATED"


class BodyCalibration:
    """Guided initial calibration, then gated slow refinement.

    WAITING_FOR_POSE: frames are checked for a neutral pose (facing the camera, upright, arms a
    little away from the body, legs visible); the reason a frame is rejected is the instruction
    shown to the person. CAPTURING: good frames are collected until `frames_needed`, then the shape
    is solved once. CALIBRATED: lengths keep accumulating from reliable frames (not foreshortened),
    widths only from neutral-pose frames; every `refine_every` good frames the shape is re-solved and
    applied only if some measurement changed by more than `refine_min_change_cm`."""

    def __init__(self, cfg: Optional[dict] = None):
        c = cfg or {}
        self.frames_needed = int(c.get("frames_needed", 25))
        self.refine_every = int(c.get("refine_every", 60))
        self.refine_min_change_cm = float(c.get("refine_min_change_cm", 1.5))
        self.require_legs = bool(c.get("require_legs", False))
        self.clothing = float(c.get("clothing_allowance_m", 0.02))
        self.max_facing_deg = float(c.get("max_facing_deg", 25.0))
        self.max_tilt_deg = float(c.get("max_tilt_deg", 12.0))
        self.min_arm_angle_deg = float(c.get("min_arm_angle_deg", 20.0))
        self.state = IDLE
        self.good = 0
        self.since_refine = 0
        self.message = "Select a person to start body calibration"
        self.legs_seen = False

    def restart(self):
        self.state, self.good, self.since_refine, self.legs_seen = WAITING, 0, 0, False
        self.message = "Stand facing the camera, arms slightly away from your body"

    def check_pose(self, P: np.ndarray, conf: np.ndarray, img: Optional[np.ndarray], min_conf=0.5) -> Tuple[bool, str, bool]:
        """(neutral?, instruction if not, legs visible?). P = MediaPipe world (33,3), y down."""
        core = (11, 12, 23, 24)
        if min(conf[i] for i in core) < min_conf:
            return False, "Step back so your shoulders and hips are visible", False
        sh = P[11] - P[12]
        facing = np.degrees(np.arctan2(abs(sh[2]), np.hypot(sh[0], sh[1])))
        if facing > self.max_facing_deg:
            return False, "Turn to face the camera", False
        up = 0.5 * (P[11] + P[12]) - 0.5 * (P[23] + P[24])
        tilt = np.degrees(np.arccos(np.clip(-up[1] / (np.linalg.norm(up) + 1e-9), -1, 1)))
        if tilt > self.max_tilt_deg:
            return False, "Stand upright", False
        if img is not None and min(conf[i] for i in (13, 14, 15, 16)) >= min_conf:
            for s, w, hp in ((11, 15, 23), (12, 16, 24)):
                if img[w, 1] < img[s, 1]:
                    return False, "Lower your arms (hands below shoulders)", False
                # arm vs torso side in the image: arms hanging against the body hide the waist/hip outline
                a, b = img[w] - img[s], img[hp] - img[s]
                ang = np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9), -1, 1)))
                if ang < self.min_arm_angle_deg:
                    return False, "Move your arms slightly away from your body", False
        legs = min(conf[i] for i in (25, 26, 27, 28)) >= min_conf
        if legs:
            for h, k, a in ((23, 25, 27), (24, 26, 28)):
                u, w = P[h] - P[k], P[a] - P[k]
                if np.degrees(np.arccos(np.clip(u @ w / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-9), -1, 1))) < 160:
                    return False, "Stand with straight legs", legs
        elif self.require_legs:
            return False, "Step back so your whole body is visible", False
        return True, "", legs
