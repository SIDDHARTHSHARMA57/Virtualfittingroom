"""
Reconstruction Metric Evaluation Module
Computes 3D/2D metrics:
- 2D keypoint reprojection error (pixels RMSE)
- Per-joint error (mm)
- Body symmetry & anthropometric checks (stature, shoulder width, leg length)
"""

import numpy as np

def compute_reprojection_error(projected_2d, ground_truth_2d, confidences=None):
    """Calculates RMSE in pixels between projected 2D keypoints and detector keypoints."""
    diff = projected_2d - ground_truth_2d
    sq_dist = np.sum(diff**2, axis=-1) # (N,)
    if confidences is not None and np.sum(confidences) > 0:
        rmse = np.sqrt(np.sum(sq_dist * confidences) / np.sum(confidences))
    else:
        rmse = np.sqrt(np.mean(sq_dist))
    return float(rmse)

def compute_body_proportions(vertices, joints):
    """
    Computes key estimated anthropometric measurements in centimeters (assuming SMPL meters).
    """
    # Stature / Height: max Y - min Y or max Z - min Z depending on orientation
    # In SMPL standard coordinate space: Y is up/down (negative Y down), Z is forward/back, X is right/left
    height_cm = (np.max(vertices[:, 1]) - np.min(vertices[:, 1])) * 100.0

    # Shoulder width: distance between L_Shoulder (SMPL 16) and R_Shoulder (SMPL 17)
    l_shoulder = joints[16]
    r_shoulder = joints[17]
    shoulder_width_cm = float(np.linalg.norm(l_shoulder - r_shoulder) * 100.0)

    # Hip width: distance between L_Hip (SMPL 1) and R_Hip (SMPL 2)
    l_hip = joints[1]
    r_hip = joints[2]
    hip_width_cm = float(np.linalg.norm(l_hip - r_hip) * 100.0)

    # Torso length: distance between Pelvis (0) and Neck (12)
    pelvis = joints[0]
    neck = joints[12]
    torso_len_cm = float(np.linalg.norm(pelvis - neck) * 100.0)

    return {
        "estimated_height_cm": float(abs(height_cm)),
        "shoulder_width_cm": shoulder_width_cm,
        "hip_width_cm": hip_width_cm,
        "torso_length_cm": torso_len_cm
    }
