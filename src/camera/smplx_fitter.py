"""
SMPL-X live motion solver.

Fits global_orient + body_pose (22 joints) to MediaPipe 3D world landmarks by
optimising joint-only forward kinematics (no mesh in the loop), then runs one
full SMPL-X forward pass for the mesh. Betas stay fixed.

Why this works from any view: the loss compares *bone directions* in a shared
camera-aligned frame, so body yaw (side / rear) is solved by global_orient and
the kinematic chain, not by per-bone hacks. Directions make it scale-free, so
MediaPipe's limb lengths needn't match the SMPL-X template.
"""
import os
import time
import logging
import threading
from typing import Optional, Dict, Any, List
import cv2
import numpy as np
import torch
import smplx
from smplx.lbs import batch_rodrigues

from src.camera.body_model import (load_config as load_body_config, joint_limits, clamp_body_pose,
                                   TorchLimits)
from src.camera.body_collision import BodyCollision, needed_joints as collision_joints
from src.camera.body_shape import ShapeEstimator, BodyCalibration, WIDTH_BANDS, IDLE as CAL_IDLE, \
    WAITING as CAL_WAITING, CAPTURING as CAL_CAPTURING, CALIBRATED as CAL_DONE

logger = logging.getLogger(__name__)

# --- Keypoints: each is the mean of SMPL-X joint ids and the mean of MediaPipe pose ids ---
# (name, smplx_joint_ids, mediapipe_ids)
KEYPOINTS = [
    ("pelvis",  (1, 2),   (23, 24)),
    ("l_hip",   (1,),     (23,)),
    ("r_hip",   (2,),     (24,)),
    ("l_knee",  (4,),     (25,)),
    ("r_knee",  (5,),     (26,)),
    ("l_ankle", (7,),     (27,)),
    ("r_ankle", (8,),     (28,)),
    ("l_heel",  ("l_heel",), (29,)),  # surface points, see FOOT_POINTS: MediaPipe's heel and
    ("r_heel",  ("r_heel",), (30,)),  # foot-index are ON the skin (heel back, toe tips), not at
    ("l_toe",   ("l_toe",),  (31,)),  # joints; matching them to joints biased foot yaw/pitch
    ("r_toe",   ("r_toe",),  (32,)),
    ("chest",   (16, 17), (11, 12)),
    ("l_sho",   (16,),    (11,)),
    ("r_sho",   (17,),    (12,)),
    ("l_elb",   (18,),    (13,)),
    ("r_elb",   (19,),    (14,)),
    ("l_wri",   (20,),    (15,)),
    ("r_wri",   (21,),    (16,)),
    ("l_index", (25,),    (19,)),   # hand orientation -> forearm/wrist twist
    ("r_index", (40,),    (20,)),
    ("l_pinky", (31,),    (17,)),
    ("r_pinky", (46,),    (18,)),
    ("head",    (15,),    (7, 8)),  # mid-ears
    ("l_eye",   (23,),    (2,)),
    ("r_eye",   (24,),    (5,)),
]
KP = {name: i for i, (name, _, _) in enumerate(KEYPOINTS)}

# Skin points rigidly attached to a joint: name -> (SMPL-X vertex ids averaged, parent joint).
# Toes ride the foot joint (toe bend), the heel rides the ankle.
FOOT_POINTS = {
    "l_heel": ((8846,), 7), "r_heel": ((8635,), 8),
    "l_toe": ((5770, 5780), 10), "r_toe": ((8463, 8474), 11),   # big + small toe tip
}

# (parent_kp, child_kp, weight)
BONES = [
    ("pelvis", "l_hip", 1.0), ("pelvis", "r_hip", 1.0),
    ("l_hip", "l_knee", 1.0), ("r_hip", "r_knee", 1.0),
    ("l_knee", "l_ankle", 1.0), ("r_knee", "r_ankle", 1.0),
    ("l_heel", "l_toe", 1.5), ("r_heel", "r_toe", 1.5),       # foot yaw + pitch
    ("l_ankle", "l_heel", 0.5), ("r_ankle", "r_heel", 0.5),   # heel lift
    ("pelvis", "chest", 2.0),
    ("chest", "l_sho", 1.5), ("chest", "r_sho", 1.5),
    ("l_sho", "l_elb", 1.0), ("r_sho", "r_elb", 1.0),
    ("l_elb", "l_wri", 1.0), ("r_elb", "r_wri", 1.0),
    ("l_wri", "l_index", 0.4), ("r_wri", "r_index", 0.4),
    ("l_wri", "l_pinky", 0.4), ("r_wri", "r_pinky", 0.4),
    ("chest", "head", 1.0),
    ("head", "l_eye", 0.5), ("head", "r_eye", 0.5),
]

# Joints counted for the dashboard "tracked joints" stat (MediaPipe ids)
KEY_LIMB_IDS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)

# Per-joint prior weight on body_pose (21 joints). Spine/neck/collars resist; limbs are free.
_PRIOR_W = np.full(21, 0.02, dtype=np.float32)
for j in (2, 5, 8):          # spine1..3 (body_pose index = smplx joint id - 1)
    _PRIOR_W[j] = 0.3
for j in (11, 12, 13):       # neck, collars
    _PRIOR_W[j] = 0.3
for j in (6, 7, 9, 10):      # ankles, feet
    _PRIOR_W[j] = 0.1


# Prior mean = relaxed standing (arms down ~70 deg). Hidden limbs drift here, not to a T-pose.
_PRIOR_MEAN = np.zeros(63, dtype=np.float32)
_PRIOR_MEAN[15 * 3 + 2] = -1.2   # left_shoulder (smplx 16) about z: arm down
_PRIOR_MEAN[16 * 3 + 2] = 1.2    # right_shoulder (smplx 17)

# --- Hands ---
# SMPL-X hand_pose order (15 joints): index1-3, middle1-3, pinky1-3, ring1-3, thumb1-3
# Matching MediaPipe hand chains [base, mid, distal, tip]:
HAND_MP_CHAINS = [(5, 6, 7, 8), (9, 10, 11, 12), (17, 18, 19, 20), (13, 14, 15, 16), (1, 2, 3, 4)]
TIP_VERTS = {  # smplx.vertex_ids['smplx'] fingertip vertices, same finger order as above
    "left": (4933, 5058, 5286, 5169, 5361),
    "right": (7669, 7794, 8022, 7905, 8079),
}
HAND_BASE = {"left": 25, "right": 40}    # first finger joint id
WRIST = {"left": 20, "right": 21}
ELBOW = {"left": 18, "right": 19}


def _rotvec_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation (axis-angle) taking direction a onto direction b."""
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    ang = np.arctan2(s, float(np.dot(a, b)))
    if s < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (axis / s * ang).astype(np.float32)


def _rotvec_to_mat(r: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(r, dtype=np.float64))[0]


def _mat_to_rotvec(R: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(R, dtype=np.float64))[0].ravel().astype(np.float32)


def _frame(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Orthonormal frame (columns) from a primary axis and a roughly-perpendicular secondary."""
    u = primary / (np.linalg.norm(primary) + 1e-9)
    n = np.cross(u, secondary)
    n /= np.linalg.norm(n) + 1e-9
    return np.stack([u, np.cross(n, u), n], axis=1)


def mp_to_smplx(p) -> np.ndarray:
    """MediaPipe world (x right, y down, z away from camera) -> SMPL-X (x left-of-body, y up, z forward).
    (x, -y, -z) is a proper rotation; (x, -y, z) would be a mirror and invert depth motion."""
    return np.array([p[0], -p[1], -p[2]], dtype=np.float32)


class SMPLXFittingEngine:
    MAX_TILT_DEG = 35.0   # body lean limit from vertical (calibration knob)
    MAX_JUMP_DEG = 60.0   # max body rotation between two fits before the frame is distrusted

    def __init__(
        self,
        model_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models"),
        device: str = "cpu",
        iters: int = 8,
        temporal_w: float = 0.1,
        body_cfg: Optional[dict] = None,
        intrinsics: Optional[dict] = None,
    ):
        """body_cfg: config/body_model.json contents (joint limits + collision); loaded if None.
        intrinsics: {"w","h","fx","fy","cx","cy"} of the (mirrored) image MediaPipe sees; enables
        camera-space placement of the body (transl). None -> 60 deg horizontal FOV guess."""
        self.device = torch.device(device)
        self.iters = iters
        self.temporal_w = temporal_w
        self.body_cfg = body_cfg or load_body_config()
        self.limits = joint_limits(self.body_cfg)
        self.limit_w = float(self.body_cfg.get("limit_weight", 50.0))
        self._finger_max = float(np.radians(self.body_cfg.get("finger_max_bend_deg", 110.0)))
        self.reproj_w = float(self.body_cfg.get("reprojection_weight", 150.0))
        self.depth_w = float(self.body_cfg.get("world_depth_weight", 0.5))
        self.intrinsics = intrinsics
        self.transl = np.array([0.0, 0.0, 2.5], np.float32)   # camera-space pelvis-root offset (m)
        self._transl_valid = False
        self.last_penetrations: Dict[str, float] = {}
        self.last_face_agreement_deg = float("nan")
        self.auto_shape = bool(self.body_cfg.get("estimate_body_shape", True))
        self.betas_target: Optional[np.ndarray] = None
        self._report_cache: Optional[List[dict]] = None
        self.last_reproj_px = float("nan")
        self.head_source = "pose"
        self.last_clamped: List[str] = []
        self.enabled = False
        self.layer = None
        self.faces = None
        self.num_vertices = 0

        self.betas = torch.zeros((1, 10), dtype=torch.float32, device=self.device)
        self.body_pose = torch.zeros((1, 63), dtype=torch.float32, device=self.device)
        self.global_orient = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        self._prior_w = torch.tensor(np.repeat(_PRIOR_W, 3), device=self.device)
        self._prior_mean = torch.tensor(_PRIOR_MEAN, device=self.device)
        self.hand_pose = {"left": np.zeros(45, np.float32), "right": np.zeros(45, np.float32)}
        self.hands_tracked = 0
        self._rejects = 0
        self.rejected_frames = 0
        self._cos_max_tilt = float(np.cos(np.radians(self.MAX_TILT_DEG)))

        self.last_vertices: Optional[np.ndarray] = None
        self.last_joints: Optional[np.ndarray] = None
        self.last_latency_ms = 0.0
        self.last_update_time = 0.0
        self.last_loss = 0.0
        self.valid_joints_count = 0
        self.is_live_fitted = False
        self._has_prev = False

        self._lock = threading.Lock()
        self._load_model(model_dir)

    def _load_model(self, model_dir: str):
        try:
            logger.info(f"Loading SMPL-X neutral body layer on {self.device}...")
            self.layer = smplx.create(
                model_path=model_dir, model_type="smplx", gender="neutral",
                use_face_contour=False, num_betas=10, ext="npz", batch_size=1,
                use_pca=False, flat_hand_mean=True,  # full 45-D finger rotations; zero = flat hand
            ).to(self.device)
            self.faces = self.layer.faces
            self.num_vertices = self.layer.v_template.shape[0]
            self.shape_est = ShapeEstimator(self.layer.J_regressor.cpu().numpy(),
                                            self.layer.v_template.cpu().numpy(),
                                            self.layer.shapedirs.cpu().numpy(),
                                            self.layer.lbs_weights.cpu().numpy())
            self.calib = BodyCalibration(self.body_cfg.get("body_shape", {}))
            self._init_kinematics()
            with torch.no_grad():
                out = self.layer(betas=self.betas)
            self.last_vertices = out.vertices[0].cpu().numpy()
            self.last_joints = out.joints[0].cpu().numpy()
            self.enabled = True
            logger.info(f"SMPL-X loaded: {self.num_vertices} vertices.")
        except Exception as e:
            logger.warning(f"SMPL-X model could not be loaded: {e}")
            self.enabled = False

    def _init_kinematics(self):
        """Rest joints for current betas + keypoint/bone index tables."""
        with torch.no_grad():
            v_shaped = self.layer.v_template + torch.einsum(
                "bl,mkl->bmk", self.betas, self.layer.shapedirs[:, :, :10])
            self._rest_J = torch.einsum("bik,ji->bjk", v_shaped, self.layer.J_regressor)  # (1,55,3)
        parents = self.layer.parents.tolist()
        rest = self._rest_J[0]
        self._rest_np = rest.cpu().numpy()
        vs = v_shaped[0].cpu().numpy()
        self._tip_rest = {s: vs[list(v)] for s, v in TIP_VERTS.items()}

        # Joint subset = keypoint + collision-proxy joints and their ancestors, grouped by tree
        # depth so FK runs ~9 batched steps instead of smplx's 55-step 4x4 loop (~7x faster).
        need = set()
        kp_joints = [FOOT_POINTS[j][1] if isinstance(j, str) else j for _, sids, _ in KEYPOINTS for j in sids]
        for j in kp_joints + collision_joints():
            while j >= 0 and j not in need:
                need.add(j)
                j = parents[j]
        sub = sorted(need)
        pos_of = {j: i for i, j in enumerate(sub)}
        self._pos_of = pos_of
        depth = {0: 0}
        for j in sub[1:]:
            depth[j] = depth[parents[j]] + 1
        self._sub = sub
        self._levels = []
        for d in range(1, max(depth.values()) + 1):
            js = [j for j in sub if depth[j] == d]
            self._levels.append((
                torch.tensor([pos_of[j] for j in js], device=self.device),
                torch.tensor([pos_of[parents[j]] for j in js], device=self.device),
                rest[js] - rest[[parents[j] for j in js]],
            ))
        self._root = rest[0]
        # only body joints (0..21) carry rotation; eyes/finger roots inherit their parent's
        self._rot_sub = torch.tensor([pos_of[j] for j in sub if j <= 21], device=self.device)
        self._rot_ids = [j for j in sub if j <= 21]

        # surface points: rest offset from their parent joint, carried rigidly by its rotation
        names = list(FOOT_POINTS)
        self._pt_parent = torch.tensor([pos_of[FOOT_POINTS[n][1]] for n in names], device=self.device)
        self._pt_offset = torch.tensor(np.stack([
            vs[list(FOOT_POINTS[n][0])].mean(0) - self._rest_np[FOOT_POINTS[n][1]] for n in names]),
            dtype=torch.float32, device=self.device)
        col = dict(pos_of, **{n: len(sub) + i for i, n in enumerate(names)})
        A = torch.zeros((len(KEYPOINTS), len(sub) + len(names)), device=self.device)
        for k, (_, sids, _) in enumerate(KEYPOINTS):
            for j in sids:
                A[k, col[j]] = 1.0 / len(sids)
        self._kp_from_joints = A
        self._bone_p = torch.tensor([KP[p] for p, _, _ in BONES], device=self.device)
        self._bone_c = torch.tensor([KP[c] for _, c, _ in BONES], device=self.device)
        self._bone_w = torch.tensor([w for _, _, w in BONES], device=self.device)

        self._tlimits = TorchLimits(self.limits, self.device)
        self.collision = BodyCollision(self._rest_np, vs, self.layer.lbs_weights.cpu().numpy(),
                                       self.body_cfg.get("collision", {}), pos_of, self.device)

    def _fk_keypoints(self, global_orient: torch.Tensor, body_pose: torch.Tensor,
                      return_rot: bool = False):
        """Joint-only FK on the keypoint joint subset -> keypoints (K,3)
        [, global rotations (n,3,3), posed joint positions (n,3)] (rows indexed by self._pos_of)."""
        full = torch.cat([global_orient, body_pose], dim=1).view(22, 3)[self._rot_ids]
        R_loc = batch_rodrigues(full)                       # (n_rot,3,3)
        n = len(self._sub)
        eye = torch.eye(3, device=self.device).expand(n, 3, 3)
        R_loc = eye.index_copy(0, self._rot_sub, R_loc)     # non-rotating joints = identity
        # sub[0] is the pelvis: its global rotation is its local one. index_copy keeps autograd happy.
        R_all = R_loc
        P_all = self._root.expand(n, 3)
        for c, p, off in self._levels:
            Rp = R_all[p]
            P_all = P_all.index_copy(0, c, P_all[p] + (Rp @ off[..., None])[..., 0])
            R_all = R_all.index_copy(0, c, Rp @ R_loc[c])
        Rp = R_all[self._pt_parent]
        P_pts = P_all[self._pt_parent] + (Rp @ self._pt_offset[..., None])[..., 0]
        kp = self._kp_from_joints @ torch.cat([P_all, P_pts])  # (K,3)
        return (kp, R_all, P_all) if return_rot else kp

    def _torso_orient(self, kp: np.ndarray, kc: np.ndarray) -> np.ndarray:
        """Analytic global_orient from hips+shoulders (cold-start seed; avoids 180-deg local minima).
        Without visible hips, assume the body is upright and take yaw from the shoulders only."""
        hips_ok = kc[KP["l_hip"]] > 0 and kc[KP["r_hip"]] > 0

        def frame(k, use_hips):
            x = k[KP["l_sho"]] - k[KP["r_sho"]]
            if use_hips:
                x = x + (k[KP["l_hip"]] - k[KP["r_hip"]])
                y = k[KP["chest"]] - k[KP["pelvis"]]
            else:
                y = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            z = np.cross(x, y)
            x = np.cross(y, z)
            return np.stack([v / (np.linalg.norm(v) + 1e-9) for v in (x, y, z)], axis=1)
        with torch.no_grad():
            kp_rest = self._fk_keypoints(torch.zeros((1, 3), device=self.device),
                                         torch.zeros((1, 63), device=self.device)).cpu().numpy()
        R = frame(kp, hips_ok) @ frame(kp_rest, hips_ok).T
        rvec, _ = cv2.Rodrigues(R.astype(np.float64))
        return rvec.ravel().astype(np.float32)

    @staticmethod
    def landmark_conf(world_points, body_points=None) -> np.ndarray:
        """Per-landmark confidence. MediaPipe reports high visibility for limbs it *guesses*
        outside the frame (legs when you stand close), so anything off-image counts as unseen."""
        conf = np.array([p[3] if len(p) > 3 else 1.0 for p in world_points], dtype=np.float32)
        if body_points and len(body_points) == len(world_points):
            xy = np.array([p[:2] for p in body_points], dtype=np.float32)
            inside = np.all((xy > 0.01) & (xy < 0.99), axis=1)
            conf[~inside] = 0.0
        return conf

    def _observed(self, world_points, min_conf, conf):
        """MediaPipe world landmarks -> keypoints (K,3) + confidences (K,)."""
        pts = np.array([mp_to_smplx(p) for p in world_points], dtype=np.float32)
        kp = np.zeros((len(KEYPOINTS), 3), dtype=np.float32)
        kc = np.zeros(len(KEYPOINTS), dtype=np.float32)
        for k, (_, _, mids) in enumerate(KEYPOINTS):
            kp[k] = pts[list(mids)].mean(axis=0)
            c = float(conf[list(mids)].min())
            kc[k] = c if c >= min_conf else 0.0
        return kp, kc

    HEAD_W = 1.0   # weight of the face-landmarker head rotation vs. the eye/ear bone directions

    def head_rotation_from_face(self, face_matrix, kp: np.ndarray, kc: np.ndarray) -> Optional[np.ndarray]:
        """MediaPipe facial transformation matrix -> head global rotation in the SMPL frame.
        Its camera frame (x right, y up, z towards the viewer) is the frame used here, and the
        canonical face looks down +z like the SMPL-X head at rest, so the rotation applies directly.
        Accepted only if it is a proper rotation and agrees (< 45 deg) with the head direction from
        the pose landmarks (ears -> eyes); otherwise None (bad frame or unexpected convention)."""
        if face_matrix is None:
            return None
        M = np.asarray(face_matrix, dtype=np.float64)[:3, :3]
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            return None
        eyes, head = KP["l_eye"], KP["head"]
        if min(kc[eyes], kc[KP["r_eye"]], kc[head]) <= 0:
            return R  # nothing to cross-check against; the face model is the better source anyway
        fwd = 0.5 * (kp[eyes] + kp[KP["r_eye"]]) - kp[head]
        fwd[1] = 0.0  # ears->eyes has a large rest pitch; compare heading only
        f_face = R[:, 2].copy()
        f_face[1] = 0.0
        c = fwd @ f_face / (np.linalg.norm(fwd) * np.linalg.norm(f_face) + 1e-9)
        self.last_face_agreement_deg = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
        return R if self.last_face_agreement_deg < 45.0 else None

    TRANSL_SMOOTH = 2.0   # temporal weight on the camera-space position (m^-2)

    def solve(self, kp_obs: np.ndarray, kp_conf: np.ndarray, iters: Optional[int] = None,
              R_head: Optional[np.ndarray] = None, uv_obs: Optional[np.ndarray] = None,
              uv_w: Optional[np.ndarray] = None) -> float:
        """Fits global_orient/body_pose (and, with uv_obs + intrinsics, the camera-space position)
        to observed keypoints: 3D bone directions + 2D reprojection. Returns final loss."""
        iters = iters or (self.iters if self._has_prev else 60)
        obs = torch.tensor(kp_obs, device=self.device)
        cf = torch.tensor(kp_conf, device=self.device)
        d_obs = obs[self._bone_c] - obs[self._bone_p]
        d_obs = d_obs / d_obs.norm(dim=1, keepdim=True).clamp_min(1e-6)
        w = self._bone_w * torch.minimum(cf[self._bone_c], cf[self._bone_p])

        if not self._has_prev:
            self.global_orient = torch.tensor(self._torso_orient(kp_obs, kp_conf), device=self.device)[None]
        # Hips out of frame (close to the mirror): nothing constrains torso tilt, so hold it upright.
        hips_ok = kp_conf[KP["l_hip"]] > 0 and kp_conf[KP["r_hip"]] > 0
        up_w = 0.05 if hips_ok else 3.0
        up = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        R_head_t = None if R_head is None else torch.tensor(R_head, dtype=torch.float32, device=self.device)
        prev_go, prev_bp = self.global_orient.clone(), self.body_pose.clone()
        go = prev_go.clone().requires_grad_(True)
        bp = prev_bp.clone().requires_grad_(True)
        t_w = self.temporal_w if self._has_prev else 0.0
        params = [go, bp]
        # 2D reprojection: MediaPipe's image landmarks are far more precise than its 3D depth, and they
        # are what the overlay must match. The camera-space position t is optimised jointly.
        use_2d = uv_obs is not None and self.intrinsics is not None and self.reproj_w > 0
        if use_2d:
            K = self.intrinsics
            f_t = torch.tensor([K["fx"], K["fy"]], device=self.device)
            c_t = torch.tensor([K["cx"], K["cy"]], device=self.device)
            uv_t = torch.tensor(uv_obs, dtype=torch.float32, device=self.device)
            uvw_t = torch.tensor(uv_w, dtype=torch.float32, device=self.device)
            prev_t = torch.tensor(self.transl, device=self.device)
            tr = prev_t.clone().requires_grad_(True)
            params.append(tr)
            flip = torch.tensor([1.0, -1.0, -1.0], device=self.device)
        axis_w = torch.tensor([1.0, 1.0, self.depth_w if use_2d else 1.0], device=self.device)
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            kp, R_glob, P_glob = self._fk_keypoints(go, bp, return_rot=True)
            d = kp[self._bone_c] - kp[self._bone_p]
            d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-6)
            # MediaPipe's world depth (z) is its weakest axis; with 2D landmarks constraining x/y,
            # trust the 3D directions' depth component less (axis_w = [1, 1, depth_w]).
            loss = (w * (axis_w * (d - d_obs) ** 2).sum(dim=1)).sum()
            loss = loss + (self._prior_w * (bp[0] - self._prior_mean) ** 2).sum()
            # anatomy: joint ranges (config/body_model.json) and limb-vs-body penetration
            loss = loss + self.limit_w * self._tlimits.penalty(batch_rodrigues(bp.view(21, 3)))
            loss = loss + self.collision.penalty(P_glob, R_glob)
            if R_head_t is not None:  # full head orientation (nod / tilt / turn) from the face model
                loss = loss + self.HEAD_W * ((R_glob[self._pos_of[15]] - R_head_t) ** 2).sum()
            body_up = batch_rodrigues(go)[0][:, 1]
            loss = loss + up_w * ((body_up - up) ** 2).sum()
            # mirror customers stand: hard-ish wall beyond MAX_TILT from vertical
            loss = loss + 50.0 * torch.relu(self._cos_max_tilt - body_up[1]) ** 2
            if t_w:
                loss = loss + t_w * (((bp - prev_bp) ** 2).sum() + ((go - prev_go) ** 2).sum())
            if use_2d:  # squared angular reprojection error (pixels / focal length)
                Pc = kp * flip + tr
                uv = f_t * Pc[:, :2] / Pc[:, 2:3].clamp_min(0.1) + c_t
                loss = loss + self.reproj_w * (uvw_t * (((uv - uv_t) / f_t) ** 2).sum(1)).sum()
                loss = loss + self.TRANSL_SMOOTH * ((tr - prev_t) ** 2).sum()
            loss.backward()
            return loss

        loss = opt.step(closure)

        # Reject implausible frames (hand over the lens, mis-detection): a body can't spin
        # > MAX_JUMP in one inference step. Accept anyway after a few in a row (it really moved).
        if self._has_prev:
            R_rel = _rotvec_to_mat(go.detach()[0].numpy()) @ _rotvec_to_mat(prev_go[0].numpy()).T
            jump = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))
            if jump > self.MAX_JUMP_DEG and self._rejects < 4:
                self._rejects += 1
                self.rejected_frames += 1
                return float(loss)
        self._rejects = 0
        self.global_orient = go.detach()
        self.body_pose = bp.detach()
        if use_2d and 0.3 < float(tr[2]) < 8.0:
            self.transl = tr.detach().cpu().numpy().astype(np.float32)
            self._transl_valid = True
        self._has_prev = True
        return float(loss)

    def _finalize_pose(self):
        """Hard guarantees after all solving (incl. analytic wrists): clamp every joint into its
        anatomical range (the soft penalty can leave tiny overshoots), then record any remaining
        self-penetration for telemetry."""
        bp, self.last_clamped = clamp_body_pose(self.body_pose[0].cpu().numpy(), self.limits)
        self.body_pose = torch.tensor(bp, device=self.device)[None]
        with torch.no_grad():
            _, R, P = self._fk_keypoints(self.global_orient, self.body_pose, return_rot=True)
        self.last_penetrations = self.collision.penetrations(P, R, tol=0.005)
        return P

    def _solve_translation(self, body_points, conf_img, min_conf: float):
        """Camera-space position of the body from 2D landmarks with known intrinsics.
        Rotation is already solved, so projecting the posed model joints is linear in the
        translation t:  (u - cx) * (Z + tz) = fx * (X + tx)  (same for v/y).
        Body size is the SMPL-X template's (real height needs betas -> Phase 1)."""
        K = self.intrinsics
        if K is None or not body_points:
            return
        with torch.no_grad():
            kp = self._fk_keypoints(self.global_orient, self.body_pose).cpu().numpy()
        Xc = kp * np.array([1.0, -1.0, -1.0], np.float32)   # SMPL frame -> camera frame (x right, y down, z fwd)
        rows, rhs, wts = [], [], []
        for k, (name, _, mids) in enumerate(KEYPOINTS):
            if name.endswith(("_index", "_pinky")):
                continue
            c = min(conf_img[m] for m in mids)
            if c < min_conf:
                continue
            u = np.mean([body_points[m][0] for m in mids]) * K["w"] - K["cx"]
            v = np.mean([body_points[m][1] for m in mids]) * K["h"] - K["cy"]
            X, Y, Z = Xc[k]
            rows += [[K["fx"], 0.0, -u], [0.0, K["fy"], -v]]
            rhs += [u * Z - K["fx"] * X, v * Z - K["fy"] * Y]
            wts += [c, c]
        if len(rows) < 6:  # need >= 3 points for a stable depth
            return
        A, b, W = np.array(rows), np.array(rhs), np.sqrt(np.array(wts))[:, None]
        t, *_ = np.linalg.lstsq(A * W, b * W[:, 0], rcond=None)
        if not (0.3 < t[2] < 8.0):  # behind the camera / absurd depth: keep the last estimate
            return
        a = 0.5 if self._transl_valid else 1.0   # light smoothing; depth from 2D is noisy
        self.transl = ((1 - a) * self.transl + a * t).astype(np.float32)
        self._transl_valid = True

    # ---------------- hands ----------------
    @staticmethod
    def assign_hands(hands_img: Dict[str, list], body_points: Optional[list]) -> Dict[str, str]:
        """Map hand-detector entries to body side by nearest pose wrist (handedness labels are
        unreliable on a mirrored feed). Returns {category: 'left'|'right'}."""
        out = {}
        cats = list(hands_img)
        if body_points and len(body_points) >= 17:
            wr = {"left": np.array(body_points[15][:2]), "right": np.array(body_points[16][:2])}
            cost = [(np.linalg.norm(np.array(hands_img[c][0][:2]) - wr[s]), c, s)
                    for c in cats for s in wr]
            for _, c, s in sorted(cost, key=lambda t: t[0]):
                if c not in out and s not in out.values():
                    out[c] = s
        else:
            # facing the camera, MediaPipe's "left" side is on the image right
            for c in cats:
                out[c] = "left" if hands_img[c][0][0] > 0.5 else "right"
        return out

    def _fit_hand(self, side: str, H: np.ndarray, R_elbow: np.ndarray):
        """Analytic wrist + finger rotations from 21 metric hand landmarks (SMPL-X frame).
        Wrist: palm frame alignment. Fingers: per-segment minimal rotation in the parent frame."""
        rest, w, b = self._rest_np, WRIST[side], HAND_BASE[side]
        # palm frame: wrist->middle MCP, and index MCP - pinky MCP across the knuckles
        F_obs = _frame(H[9] - H[0], H[5] - H[17])
        F_rest = _frame(rest[b + 3] - rest[w], rest[b] - rest[b + 6])
        R_wrist = F_obs @ F_rest.T
        self.body_pose[0, (w - 1) * 3:(w - 1) * 3 + 3] = torch.tensor(
            _mat_to_rotvec(R_elbow.T @ R_wrist), device=self.device)

        hp = np.zeros(45, np.float32)
        tips = self._tip_rest[side]
        for f, chain in enumerate(HAND_MP_CHAINS):
            R_p = R_wrist
            for k in range(3):
                j = b + f * 3 + k
                child = rest[j + 1] if k < 2 else tips[f]
                target_local = R_p.T @ (H[chain[k + 1]] - H[chain[k]])
                r = _rotvec_between(child - rest[j], target_local)
                ang = float(np.linalg.norm(r))
                if ang > self._finger_max:  # anatomical cap per finger segment (config: finger_max_bend_deg)
                    r = r * (self._finger_max / ang)
                hp[(f * 3 + k) * 3:(f * 3 + k) * 3 + 3] = r
                R_p = R_p @ _rotvec_to_mat(r)
        self.hand_pose[side] = hp

    # ---------------- main entry ----------------
    def update_from_world_landmarks(
        self,
        world_points: List[List[float]],
        body_points: Optional[List[List[float]]] = None,
        min_conf: float = 0.35,
        hands: Optional[Dict[str, list]] = None,
        hands_world: Optional[Dict[str, list]] = None,
        face_matrix: Optional[list] = None,
        silhouette: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """world_points: 33 MediaPipe pose world landmarks [x,y,z,vis].
        hands / hands_world: hand detector output (image coords / metric) keyed by category;
        body_points (image coords) assign hands to sides, gate off-image landmarks, and place the body.
        face_matrix: face landmarker's 4x4 facial transformation (head orientation).
        silhouette: the selected person's outline polygon (px) for body-width calibration."""
        stale = {"fitted": False, "vertices": self.last_vertices, "latency_ms": 0.0,
                 "valid_joints": 0, "is_live_fitted": self.is_live_fitted}
        if not self.enabled or not world_points or len(world_points) < 33:
            return stale

        t0 = time.perf_counter()
        conf = self.landmark_conf(world_points, body_points)
        valid = int(sum(conf[i] >= min_conf for i in KEY_LIMB_IDS))
        # Both shoulders are enough for torso yaw + head; limbs join in as they become visible.
        if not (conf[11] >= min_conf and conf[12] >= min_conf) and valid < 4:
            stale["valid_joints"] = valid
            return stale

        kp, kc = self._observed(world_points, min_conf, conf)
        self._step_shape()  # layer A (shape) changes only here, rate-limited; the pose solve below is layer B

        sides = {}
        if hands and hands_world:
            sides = {c: s for c, s in self.assign_hands(hands, body_points).items() if c in hands_world}
        for s in sides.values():  # hand detector owns wrist orientation; drop coarse pose hand points
            kc[KP[f"{s[0]}_index"]] = kc[KP[f"{s[0]}_pinky"]] = 0.0

        R_head = self.head_rotation_from_face(face_matrix, kp, kc)
        self.head_source = "face" if R_head is not None else "pose"
        uv_obs = uv_w = None
        if self.intrinsics is not None and body_points and len(body_points) == len(world_points):
            K = self.intrinsics
            uv_obs = np.array([np.mean([body_points[m][:2] for m in mids], axis=0) * [K["w"], K["h"]]
                               for _, _, mids in KEYPOINTS], dtype=np.float32)
            uv_w = kc.copy()  # visibility + in-frame; hand points already zeroed when the hand model owns them
            if not self._transl_valid:
                self._solve_translation(body_points, conf, min_conf)  # linear initial position
        loss = self.solve(kp, kc, R_head=R_head, uv_obs=uv_obs, uv_w=uv_w)

        if sides:
            with torch.no_grad():
                _, R_all, _ = self._fk_keypoints(self.global_orient, self.body_pose, return_rot=True)
            for c, s in sides.items():
                H = np.array([mp_to_smplx(p) for p in hands_world[c]], dtype=np.float32)
                self._fit_hand(s, H, R_all[self._pos_of[ELBOW[s]]].cpu().numpy())
        for s in ("left", "right"):
            if s not in sides.values():
                self.hand_pose[s] *= 0.9  # hand lost: relax toward open hand
        self.hands_tracked = len(sides)
        self._finalize_pose()
        if uv_obs is not None and self._transl_valid and uv_w.sum() > 0:  # overlay error telemetry
            K = self.intrinsics
            with torch.no_grad():
                kpf = self._fk_keypoints(self.global_orient, self.body_pose).cpu().numpy()
            Pc = kpf * [1.0, -1.0, -1.0] + self.transl
            uv = np.stack([K["fx"] * Pc[:, 0] / Pc[:, 2] + K["cx"], K["fy"] * Pc[:, 1] / Pc[:, 2] + K["cy"]], 1)
            m = uv_w > 0
            self.last_reproj_px = float(np.linalg.norm(uv[m] - uv_obs[m], axis=1).mean())
        self._calibration_step(world_points, conf, body_points, silhouette)  # needs the fitted pose + position

        with torch.no_grad():
            out = self.layer(betas=self.betas, global_orient=self.global_orient, body_pose=self.body_pose,
                             left_hand_pose=torch.tensor(self.hand_pose["left"], device=self.device)[None],
                             right_hand_pose=torch.tensor(self.hand_pose["right"], device=self.device)[None])
        verts = out.vertices[0].cpu().numpy()
        joints = out.joints[0].cpu().numpy()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        with self._lock:
            self.last_vertices = verts
            self.last_joints = joints
            self.last_latency_ms = latency_ms
            self.last_update_time = time.time()
            self.last_loss = loss
            self.valid_joints_count = valid
            self.is_live_fitted = True

        return {
            "fitted": True, "vertices": verts, "joints": joints, "latency_ms": latency_ms,
            "num_vertices": self.num_vertices, "valid_joints": valid, "is_live_fitted": True,
            "global_orient": self.global_orient[0].cpu().numpy(),
            "body_pose": self.body_pose[0].cpu().numpy(),
            "left_hand_pose": self.hand_pose["left"].copy(),
            "right_hand_pose": self.hand_pose["right"].copy(),
            "hands_tracked": self.hands_tracked,
            "transl": self.transl.copy() if self._transl_valid else None,
            "penetrations": dict(self.last_penetrations),
            "clamped_joints": list(self.last_clamped),
            "head_source": self.head_source,
            "reproj_px": self.last_reproj_px,
            "calibration": {"state": self.calib.state, "message": self.calib.message,
                            "good": self.calib.good, "needed": self.calib.frames_needed},
            "betas": self.betas[0].cpu().numpy().copy(),
            "loss": loss,
        }

    # ---------------- body shape (layer A; pose is layer B) ----------------
    SHAPE_MAX_STEP = 0.15   # max change per beta per frame: shape morphs over ~1 s, no visible jump

    def _calibration_step(self, world_points, conf, body_points, silhouette):
        """Feed one tracked frame of the SELECTED person to the shape calibration (the service only
        calls the fitter for the locked person, so other people never contribute)."""
        if not self.auto_shape:
            return
        cal, est = self.calib, self.shape_est
        if cal.state == CAL_IDLE:
            cal.restart()
        P = np.array([p[:3] for p in world_points], dtype=np.float64)
        img = None
        if body_points and len(body_points) == len(world_points) and self.intrinsics:
            img = np.array([p[:2] for p in body_points], dtype=np.float64) * [self.intrinsics["w"], self.intrinsics["h"]]
        neutral, why, legs = cal.check_pose(P, conf, img)

        if cal.state in (CAL_WAITING, CAL_CAPTURING):
            if not neutral:
                cal.message = why + (f"  ({cal.good}/{cal.frames_needed})" if cal.good else "")
                return
            cal.state = CAL_CAPTURING
            est.add_lengths(P, conf, legs=legs)
            self._add_widths(silhouette, img)
            cal.good += 1
            cal.legs_seen |= legs
            cal.message = f"Hold still - measuring {cal.good}/{cal.frames_needed}"
            if cal.good >= cal.frames_needed:
                b = est.estimate(self.betas[0].cpu().numpy(), cal.clothing)
                if b is not None:
                    self.betas_target = b
                    self._report_cache = est.report(b, cal.clothing)
                    cal.state = CAL_DONE
                    cal.message = "Body calibrated" + ("" if cal.legs_seen else " (upper body only - legs not visible)")
            return

        # CALIBRATED: slow, gated refinement (segments pointing at the camera are skipped in add_lengths)
        est.add_lengths(P, conf, legs=True)
        if neutral:
            self._add_widths(silhouette, img)
        cal.since_refine += 1
        if cal.since_refine >= cal.refine_every:
            cal.since_refine = 0
            b = est.estimate(self.betas_target if self.betas_target is not None else self.betas[0].cpu().numpy(),
                             cal.clothing)
            cur = self.betas_target if self.betas_target is not None else self.betas[0].cpu().numpy()
            if b is not None and est.max_change_cm(cur, b) > cal.refine_min_change_cm:
                self.betas_target = b
                self._report_cache = est.report(b, cal.clothing)
                cal.message = "Body shape refined"

    def _add_widths(self, silhouette, img):
        """Front-view silhouette width at chest / waist / hips, as a ratio to the image shoulder width
        (scale-free). Band rows come from the fitted model projected into the image."""
        if silhouette is None or img is None or self.intrinsics is None or not self._transl_valid:
            return
        import cv2
        K = self.intrinsics
        w, h = int(K["w"]), int(K["h"])
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [np.asarray(silhouette, np.int32)], 1)
        sh_px = float(np.linalg.norm(img[11] - img[12]))
        if sh_px < 10:
            return
        with torch.no_grad():
            _, R, Pj = self._fk_keypoints(self.global_orient, self.body_pose, return_rot=True)
        R, Pj = R.cpu().numpy(), Pj.cpu().numpy()
        up = R[self._pos_of[0]][:, 1]
        for band, (anchor, dy, _) in WIDTH_BANDS.items():
            p = Pj[[self._pos_of[j] for j in anchor]].mean(0) + dy * up
            Pc = p * [1.0, -1.0, -1.0] + self.transl
            u = int(round(K["fx"] * Pc[0] / Pc[2] + K["cx"]))
            v = int(round(K["fy"] * Pc[1] / Pc[2] + K["cy"]))
            if not (0 <= v < h and 0 <= u < w) or not mask[v, u]:
                continue
            row = mask[v]
            l = u
            while l > 0 and row[l - 1]:
                l -= 1
            r = u
            while r < w - 1 and row[r + 1]:
                r += 1
            if l == 0 or r == w - 1:
                continue  # cut by the image border: not a full width
            ratio = (r - l + 1) / sh_px
            if 0.4 < ratio < 2.5:
                self.shape_est.add_width_ratio(band, ratio)

    def _step_shape(self):
        """Move the live body shape towards the calibrated target, rate-limited."""
        if self.betas_target is None:
            return
        cur = self.betas[0].cpu().numpy()
        step = np.clip(self.betas_target - cur, -self.SHAPE_MAX_STEP, self.SHAPE_MAX_STEP)
        if np.abs(step).max() > 1e-3:
            self.set_betas(cur + step)

    def recalibrate(self):
        """Operator: measure the current person again from scratch (keeps the current shape until done)."""
        self.shape_est.clear()
        self.calib.restart()

    def measurements(self) -> List[dict]:
        """Measurement report of the calibrated body. Computed on the fit thread whenever the shape
        target changes; safe to read from the UI thread."""
        return self._report_cache or []

    def set_betas(self, betas):
        """Reshape the body (rest joints, skin points, collision radii follow)."""
        self.betas = torch.tensor(np.asarray(betas, np.float32), device=self.device).view(1, -1)
        self._init_kinematics()

    def reset(self, keep_shape: bool = True):
        """Call when the person leaves/re-enters so the next fit starts cold. keep_shape=False for
        a different person: forget their measurements and shape, start a new calibration."""
        if not keep_shape:
            self.shape_est.clear()
            self.calib.restart()
            self.betas_target = None
            self._report_cache = None
            if float(self.betas.abs().sum()) > 0:
                self.set_betas(np.zeros(self.betas.shape[1], np.float32))
        self.body_pose.zero_()
        self.global_orient.zero_()
        for s in self.hand_pose:
            self.hand_pose[s][:] = 0
        self._has_prev = False
        self._transl_valid = False
