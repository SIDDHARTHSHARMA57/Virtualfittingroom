"""
SMPL-X Parametric 3D Body Surface Engine.
Integrates the local SMPL-X model ('models/smplx/SMPLX_NEUTRAL.pkl')
to reconstruct a 10,475-vertex anatomical 3D body surface mesh from detected body landmarks.
Separates shape (betas, updated slowly/averaged) from articulation (frame-by-frame pose).
"""
import os
import time
import logging
import threading
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import smplx

logger = logging.getLogger(__name__)

# MediaPipe Pose 33 to SMPL-X Joint Index Mapping
# MediaPipe indices:
# 11: L Shoulder, 12: R Shoulder, 13: L Elbow, 14: R Elbow, 15: L Wrist, 16: R Wrist
# 23: L Hip, 24: R Hip, 25: L Knee, 26: R Knee, 27: L Ankle, 28: R Ankle
# SMPL-X indices (from smplx.joint_names):
# 0: pelvis, 1: left_hip, 2: right_hip, 4: left_knee, 5: right_knee, 7: left_ankle, 8: right_ankle
# 16: left_shoulder, 17: right_shoulder, 18: left_elbow, 19: right_elbow, 20: left_wrist, 21: right_wrist
MP_TO_SMPLX = {
    11: 16,  # L Shoulder
    12: 17,  # R Shoulder
    13: 18,  # L Elbow
    14: 19,  # R Elbow
    15: 20,  # L Wrist
    16: 21,  # R Wrist
    23: 1,   # L Hip
    24: 2,   # R Hip
    25: 4,   # L Knee
    26: 5,   # R Knee
    27: 7,   # L Ankle
    28: 8,   # R Ankle
}

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

        # Optimization variables
        self.last_vertices: Optional[np.ndarray] = None
        self.last_joints: Optional[np.ndarray] = None
        self.last_latency_ms: float = 0.0

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
            self.enabled = True
            logger.info(f"SMPL-X loaded: {self.num_vertices} vertices, {len(self.faces)} faces.")
        except Exception as e:
            logger.warning(f"SMPL-X model could not be loaded: {e}")
            self.enabled = False

    def fit_step(
        self,
        body_points: list,
        img_w: int = 640,
        img_h: int = 480,
        iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Runs a fast real-time fitting step on observed landmarks.
        Uses cached body shape to preserve user proportions while updating pose.
        """
        if not self.enabled or not body_points or len(body_points) != 33:
            return {"fitted": False, "vertices": None, "latency_ms": 0.0}

        t0 = time.perf_counter()

        # Extract mapped 2D coordinates and confidences
        mp_indices = list(MP_TO_SMPLX.keys())
        smplx_indices = [MP_TO_SMPLX[k] for k in mp_indices]

        target_coords = []
        confidences = []
        for idx in mp_indices:
            p = body_points[idx]
            target_coords.append([p[0] * img_w, p[1] * img_h])
            confidences.append([p[2]])

        kps_2d = torch.tensor(target_coords, dtype=torch.float32, device=self.device)
        weights = torch.tensor(confidences, dtype=torch.float32, device=self.device)

        # Quick analytic/gradient adjustment of global pose & orientation
        transl = nn.Parameter(torch.tensor([[0.0, 0.2, 2.2]], device=self.device, requires_grad=True))
        global_orient = nn.Parameter(torch.zeros(1, 3, device=self.device, requires_grad=True))
        body_pose = nn.Parameter(torch.zeros(1, 63, device=self.device, requires_grad=True))

        optimizer = torch.optim.Adam([
            {'params': [transl, global_orient], 'lr': 0.08},
            {'params': [body_pose], 'lr': 0.05}
        ])

        focal_length = max(img_w, img_h)
        cx, cy = img_w / 2.0, img_h / 2.0

        for _ in range(iterations):
            optimizer.zero_grad()
            out = self.layer(
                betas=self.betas,
                global_orient=global_orient,
                body_pose=body_pose,
                transl=transl
            )
            # SMPL-X joints (127, 3)
            joints3d = out.joints[0][smplx_indices]
            X = joints3d[:, 0]
            Y = joints3d[:, 1]
            Z = torch.clamp(joints3d[:, 2], min=0.5)

            u = focal_length * (X / Z) + cx
            v = focal_length * (Y / Z) + cy
            proj = torch.stack([u, v], dim=-1)

            loss = torch.mean(weights * torch.norm(proj - kps_2d, dim=-1, keepdim=True))
            loss += 0.005 * torch.mean(body_pose ** 2)
            loss.backward()
            optimizer.step()

        # Forward pass for final vertices
        with torch.no_grad():
            final_out = self.layer(
                betas=self.betas,
                global_orient=global_orient,
                body_pose=body_pose,
                transl=transl
            )
            verts = final_out.vertices[0].detach().cpu().numpy()
            joints = final_out.joints[0].detach().cpu().numpy()

        latency_ms = (time.perf_counter() - t0) * 1000.0

        with self._lock:
            self.last_vertices = verts
            self.last_joints = joints
            self.last_latency_ms = latency_ms

        return {
            "fitted": True,
            "vertices": verts,
            "joints": joints,
            "latency_ms": latency_ms,
            "num_vertices": self.num_vertices,
            "faces": self.faces
        }
