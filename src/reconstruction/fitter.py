"""
SMPL Body Shape and Pose Fitter
Optimizes SMPL shape (betas) and pose parameters against 2D/3D keypoints from YOLOv8 pose detector.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import smplx

# Standard COCO 17 keypoint indices for reference:
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

# Mapping between SMPL joints (first 24/45 joints) and COCO 17
# SMPL joint 0: Pelvis
# SMPL 1: L_Hip, 2: R_Hip, 4: L_Knee, 5: R_Knee, 7: L_Ankle, 8: R_Ankle
# SMPL 16: L_Shoulder, 17: R_Shoulder, 18: L_Elbow, 19: R_Elbow, 20: L_Wrist, 21: R_Wrist
# SMPL joint 24 / head joints for nose/ears/eyes

COCO_TO_SMPL_JOINTS = {
    5: 16,   # L Shoulder
    6: 17,   # R Shoulder
    7: 18,   # L Elbow
    8: 19,   # R Elbow
    9: 20,   # L Wrist
    10: 21,  # R Wrist
    11: 1,   # L Hip
    12: 2,   # R Hip
    13: 4,   # L Knee
    14: 5,   # R Knee
    15: 7,   # L Ankle
    16: 8,   # R Ankle
}


class SMPLFitter:
    def __init__(self, model_dir="/media/siddharth-sharam/Workspace/VirtualFittingRoom/models/smpl", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.model = smplx.create(
            model_path=model_dir,
            model_type="smpl",
            gender="neutral",
            ext="pkl",
            batch_size=1
        ).to(self.device)
        self.faces = self.model.faces
        print(f"[SMPLFitter] Initialized on {self.device} with SMPL model (faces: {len(self.faces)})")

    def fit_from_coco_keypoints(self, keypoints_2d, confidences=None, img_w=640, img_h=480, num_iters=120, lr=0.05):
        """
        Fit SMPL betas, global orient, body pose, and camera translation
        to 2D keypoints (x, y) with perspective camera projection.
        """
        kps_2d = torch.tensor(keypoints_2d, dtype=torch.float32, device=self.device) # (17, 2)
        if confidences is not None:
            weights = torch.tensor(confidences, dtype=torch.float32, device=self.device).unsqueeze(-1)
        else:
            weights = torch.ones((17, 1), dtype=torch.float32, device=self.device)

        # Optimization variables
        betas = nn.Parameter(torch.zeros(1, 10, device=self.device, requires_grad=True))
        global_orient = nn.Parameter(torch.zeros(1, 3, device=self.device, requires_grad=True))
        body_pose = nn.Parameter(torch.zeros(1, 69, device=self.device, requires_grad=True))
        
        # Translation estimate: [tx, ty, tz]
        # tz default ~2.2m from camera
        transl = nn.Parameter(torch.tensor([[0.0, 0.2, 2.2]], device=self.device, requires_grad=True))
        
        # Camera intrinsics approximation
        focal_length = max(img_w, img_h)
        cx, cy = img_w / 2.0, img_h / 2.0

        optimizer = torch.optim.Adam([
            {'params': [transl, global_orient], 'lr': lr * 1.5},
            {'params': [body_pose], 'lr': lr},
            {'params': [betas], 'lr': lr * 0.5}
        ])

        coco_idxs = list(COCO_TO_SMPL_JOINTS.keys())
        smpl_idxs = [COCO_TO_SMPL_JOINTS[k] for k in coco_idxs]

        for step in range(num_iters):
            optimizer.zero_grad()
            output = self.model(
                betas=betas,
                global_orient=global_orient,
                body_pose=body_pose,
                transl=transl
            )
            joints3d = output.joints[0] # (45, 3)
            selected_smpl_joints = joints3d[smpl_idxs] # (N, 3)

            # Perspective projection
            X = selected_smpl_joints[:, 0]
            Y = selected_smpl_joints[:, 1]
            Z = torch.clamp(selected_smpl_joints[:, 2], min=0.5)

            u = focal_length * (X / Z) + cx
            v = focal_length * (Y / Z) + cy
            proj_2d = torch.stack([u, v], dim=-1)

            target_2d = kps_2d[coco_idxs]
            w = weights[coco_idxs]

            # 2D Keypoint Loss
            loss_kps = torch.mean(w * torch.norm(proj_2d - target_2d, dim=-1, keepdim=True))

            # Regularization terms: keep shape realistic, avoid extreme poses
            loss_shape = torch.mean(betas ** 2) * 0.05
            loss_pose = torch.mean(body_pose ** 2) * 0.01

            total_loss = loss_kps + loss_shape + loss_pose
            total_loss.backward()
            optimizer.step()

        # Final forward pass
        with torch.no_grad():
            final_output = self.model(
                betas=betas,
                global_orient=global_orient,
                body_pose=body_pose,
                transl=transl
            )
            verts = final_output.vertices[0].cpu().numpy()
            joints = final_output.joints[0].cpu().numpy()

        results = {
            "vertices": verts,
            "faces": self.faces,
            "joints": joints,
            "betas": betas.detach().cpu().numpy()[0],
            "global_orient": global_orient.detach().cpu().numpy()[0],
            "body_pose": body_pose.detach().cpu().numpy()[0],
            "transl": transl.detach().cpu().numpy()[0],
            "loss": float(total_loss.item())
        }
        return results

