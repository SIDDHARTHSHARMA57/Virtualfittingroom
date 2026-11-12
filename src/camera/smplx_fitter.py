"""
SMPL-X Parametric 3D Body Surface Engine & Live Motion Retargeter.
Integrates the local SMPL-X model ('models/smplx/SMPLX_NEUTRAL.pkl').
Solves live articulation parameters (body_pose) from MediaPipe 3D world landmarks,
keeping body shape (betas) stable over time.
"""
import os
import time
import logging
import threading
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import smplx

logger = logging.getLogger(__name__)

def vector_to_axis_angle(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """
    Computes the Rodrigues axis-angle rotation vector (3,) that rotates unit vector v_from to v_to.
    """
    norm_from = np.linalg.norm(v_from)
    norm_to = np.linalg.norm(v_to)
    if norm_from < 1e-6 or norm_to < 1e-6:
        return np.zeros(3, dtype=np.float32)

    u1 = v_from / norm_from
    u2 = v_to / norm_to

    dot = np.clip(np.dot(u1, u2), -1.0, 1.0)
    angle = np.arccos(dot)

    if angle < 1e-5:
        return np.zeros(3, dtype=np.float32)
    if angle > np.pi - 1e-5:
        # 180-degree flip: choose an orthogonal axis
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(u1[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axis = np.cross(u1, axis)
        axis = axis / np.linalg.norm(axis)
        return axis * angle

    axis = np.cross(u1, u2)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        return np.zeros(3, dtype=np.float32)

    axis = axis / axis_norm
    return (axis * angle).astype(np.float32)


class SMPLXFittingEngine:
    def __init__(
        self,
        model_dir: str = "/media/siddharth-sharam/Workspace/VirtualFittingRoom/models",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        self.enabled = False
        self.layer = None
        self.faces = None
        self.num_vertices = 0

        # Persistent body shape betas (10 parameters, kept stable over time)
        self.betas = torch.zeros((1, 10), dtype=torch.float32, device=self.device)

        # Active body pose: 21 kinematic joints * 3 axis-angles = 63 values
        self.body_pose = torch.zeros((1, 63), dtype=torch.float32, device=self.device)
        self.global_orient = torch.zeros((1, 3), dtype=torch.float32, device=self.device)

        # Baseline rest bone directions in SMPL-X model space
        self.rest_bones = {
            "L_upper_arm": np.array([0.254, -0.072, -0.042], dtype=np.float32),
            "R_upper_arm": np.array([-0.271, -0.036, -0.026], dtype=np.float32),
            "L_forearm": np.array([0.245, -0.025, 0.015], dtype=np.float32),
            "R_forearm": np.array([-0.245, -0.025, 0.015], dtype=np.float32),
            "L_thigh": np.array([0.055, -0.379, -0.009], dtype=np.float32),
            "R_thigh": np.array([-0.044, -0.362, -0.017], dtype=np.float32),
            "L_calf": np.array([0.015, -0.403, -0.032], dtype=np.float32),
            "R_calf": np.array([-0.015, -0.403, -0.032], dtype=np.float32),
        }

        # Cache & telemetry
        self.last_vertices: Optional[np.ndarray] = None
        self.last_joints: Optional[np.ndarray] = None
        self.last_latency_ms: float = 0.0
        self.last_update_time: float = 0.0
        self.valid_joints_count: int = 0
        self.is_live_fitted: bool = False

        self._lock = threading.Lock()
        self._load_model(model_dir)

    def _load_model(self, model_dir: str):
        try:
            logger.info(f"Loading SMPL-X neutral body layer on {self.device}...")
            self.layer = smplx.create(
                model_path=model_dir,
                model_type="smplx",
                gender="neutral",
                use_face_contour=False,
                num_betas=10,
                ext="pkl",
                batch_size=1
            ).to(self.device)
            self.faces = self.layer.faces
            self.num_vertices = self.layer.v_template.shape[0]

            # Generate initial T-pose vertices
            with torch.no_grad():
                init_out = self.layer(betas=self.betas)
                self.last_vertices = init_out.vertices[0].detach().cpu().numpy()
                self.last_joints = init_out.joints[0].detach().cpu().numpy()

            self.enabled = True
            logger.info(f"SMPL-X loaded: {self.num_vertices} vertices, {len(self.faces)} faces.")
        except Exception as e:
            logger.warning(f"SMPL-X model could not be loaded: {e}")
            self.enabled = False

    def update_from_world_landmarks(
        self,
        world_points: List[List[float]],
        body_points: Optional[List[List[float]]] = None,
        min_conf: float = 0.35,
        smoothing: float = 0.35
    ) -> Dict[str, Any]:
        """
        Solves real-time SMPL-X articulation from MediaPipe 3D metric world landmarks.
        Uses anatomical limb vectors to compute joint rotations.
        """
        if not self.enabled:
            return {"fitted": False, "vertices": self.last_vertices, "latency_ms": 0.0}

        t0 = time.perf_counter()
        now = time.time()

        # Check if we have valid 3D points
        pts_3d = None
        if world_points and len(world_points) == 33:
            pts_3d = world_points
        elif body_points and len(body_points) == 33:
            # Fallback to normalized 2D points with synthetic depth if world landmarks are absent
            pts_3d = [[p[0], p[1], 0.0, p[2]] for p in body_points]

        if pts_3d is None:
            return {
                "fitted": False,
                "vertices": self.last_vertices,
                "latency_ms": 0.0,
                "valid_joints": 0,
                "is_live_fitted": self.is_live_fitted
            }

        def get_joint_vec(idx: int) -> Optional[np.ndarray]:
            p = pts_3d[idx]
            conf = p[3] if len(p) > 3 else (p[2] if len(p) > 2 else 1.0)
            if conf >= min_conf:
                # MediaPipe world coordinates: X right, Y down, Z forward away from camera.
                # Invert Y so up is +Y, matching SMPL-X native space (+X right, +Y up, +Z forward).
                return np.array([p[0], -p[1], p[2]], dtype=np.float32)
            return None

        # Extract MediaPipe 3D joint positions
        l_sho = get_joint_vec(11)
        r_sho = get_joint_vec(12)
        l_elb = get_joint_vec(13)
        r_elb = get_joint_vec(14)
        l_wri = get_joint_vec(15)
        r_wri = get_joint_vec(16)
        l_hip = get_joint_vec(23)
        r_hip = get_joint_vec(24)
        l_kne = get_joint_vec(25)
        r_kne = get_joint_vec(26)
        l_ank = get_joint_vec(27)
        r_ank = get_joint_vec(28)

        valid_count = sum(1 for pt in [l_sho, r_sho, l_elb, r_elb, l_wri, r_wri, l_hip, r_hip, l_kne, r_kne, l_ank, r_ank] if pt is not None)

        if valid_count < 4:
            # Not enough joints to safely update pose; maintain previous pose
            return {
                "fitted": False,
                "vertices": self.last_vertices,
                "latency_ms": 0.0,
                "valid_joints": valid_count,
                "is_live_fitted": self.is_live_fitted
            }

        # Build target body_pose (21 joints, 63 values)
        # SMPL-X body_pose indexing:
        # 0: left_hip (idx 0..2)
        # 1: right_hip (idx 3..5)
        # 3: left_knee (idx 9..11)
        # 4: right_knee (idx 12..14)
        # 15: left_shoulder (idx 45..47)
        # 16: right_shoulder (idx 48..50)
        # 17: left_elbow (idx 51..53)
        # 18: right_elbow (idx 54..56)

        current_pose = self.body_pose.clone().detach().cpu().numpy()[0]
        target_pose = current_pose.copy()

        # 1. Left Upper Arm (Shoulder)
        if l_sho is not None and l_elb is not None:
            v_obs = l_elb - l_sho
            rot_vec = vector_to_axis_angle(self.rest_bones["L_upper_arm"], v_obs)
            target_pose[45:48] = rot_vec

        # 2. Right Upper Arm (Shoulder)
        if r_sho is not None and r_elb is not None:
            v_obs = r_elb - r_sho
            rot_vec = vector_to_axis_angle(self.rest_bones["R_upper_arm"], v_obs)
            target_pose[48:51] = rot_vec

        # 3. Left Forearm (Elbow)
        if l_elb is not None and l_wri is not None:
            v_obs = l_wri - l_elb
            rot_vec = vector_to_axis_angle(self.rest_bones["L_forearm"], v_obs)
            target_pose[51:54] = rot_vec * 0.8

        # 4. Right Forearm (Elbow)
        if r_elb is not None and r_wri is not None:
            v_obs = r_wri - r_elb
            rot_vec = vector_to_axis_angle(self.rest_bones["R_forearm"], v_obs)
            target_pose[54:57] = rot_vec * 0.8

        # 5. Left Thigh (Hip)
        if l_hip is not None and l_kne is not None:
            v_obs = l_kne - l_hip
            rot_vec = vector_to_axis_angle(self.rest_bones["L_thigh"], v_obs)
            target_pose[0:3] = rot_vec

        # 6. Right Thigh (Hip)
        if r_hip is not None and r_kne is not None:
            v_obs = r_kne - r_hip
            rot_vec = vector_to_axis_angle(self.rest_bones["R_thigh"], v_obs)
            target_pose[3:6] = rot_vec

        # 7. Left Calf (Knee)
        if l_kne is not None and l_ank is not None:
            v_obs = l_ank - l_kne
            rot_vec = vector_to_axis_angle(self.rest_bones["L_calf"], v_obs)
            target_pose[9:12] = rot_vec

        # 8. Right Calf (Knee)
        if r_kne is not None and r_ank is not None:
            v_obs = r_ank - r_kne
            rot_vec = vector_to_axis_angle(self.rest_bones["R_calf"], v_obs)
            target_pose[12:15] = rot_vec

        # Temporal smoothing: smooth transition between frames
        smoothed_pose = current_pose * (1.0 - smoothing) + target_pose * smoothing
        self.body_pose = torch.tensor(smoothed_pose, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Forward pass to calculate deformed mesh vertices
        with torch.no_grad():
            out = self.layer(
                betas=self.betas,
                global_orient=self.global_orient,
                body_pose=self.body_pose
            )
            verts = out.vertices[0].detach().cpu().numpy()
            joints = out.joints[0].detach().cpu().numpy()

        latency_ms = (time.perf_counter() - t0) * 1000.0

        with self._lock:
            self.last_vertices = verts
            self.last_joints = joints
            self.last_latency_ms = latency_ms
            self.last_update_time = now
            self.valid_joints_count = valid_count
            self.is_live_fitted = True

        return {
            "fitted": True,
            "vertices": verts,
            "joints": joints,
            "latency_ms": latency_ms,
            "num_vertices": self.num_vertices,
            "valid_joints": valid_count,
            "is_live_fitted": True
        }
