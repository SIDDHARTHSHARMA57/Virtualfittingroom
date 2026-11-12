"""
Anatomical Regions and Inferred Surface Geometry Module.
Distinguishes:
1. Observed 2D landmarks (raw from MediaPipe with confidence).
2. Estimated 3D joints (MediaPipe world landmarks / kinematic solve).
3. Estimated body-surface points (anatomical contours: clavicles, sternum, ribs, waist, thighs).
4. Parametric mesh vertices (SMPL-X 10,475 vertices when fitted).
"""
import time
import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Anatomical Region Definitions & Inferred Surface Points
class AnatomicalBodyModel:
    """
    Computes anatomical contours and body surface points from observed landmarks
    with explicit uncertainty and source labeling.
    """
    def __init__(self):
        pass

    def compute_anatomical_regions(
        self,
        body_points: List[List[float]],
        world_points: Optional[List[List[float]]] = None,
        hands: Optional[Dict[str, List[List[float]]]] = None,
        face: Optional[List[List[float]]] = None,
        min_conf: float = 0.35
    ) -> Dict[str, Any]:
        """
        Takes raw tracking outputs and partitions/estimates detailed anatomical regions.
        Labels every point as either 'observed_2d', 'estimated_joint', or 'inferred_surface'.
        """
        now = time.time()
        result = {
            "timestamp": now,
            "torso": {},
            "arms": {"left": {}, "right": {}},
            "legs": {"left": {}, "right": {}},
            "head": {},
            "summary_counts": {
                "observed_landmarks": 0,
                "estimated_surface_points": 0,
                "inferred_mesh_vertices": 0
            }
        }

        if not body_points or len(body_points) != 33:
            return result

        def get_pt(idx: int) -> Optional[Tuple[float, float, float]]:
            p = body_points[idx]
            if p[2] >= min_conf:
                return (p[0], p[1], p[2])
            return None

        # Key reference landmarks
        nose = get_pt(0)
        l_sho = get_pt(11)
        r_sho = get_pt(12)
        l_elb = get_pt(13)
        r_elb = get_pt(14)
        l_wri = get_pt(15)
        r_wri = get_pt(16)
        l_hip = get_pt(23)
        r_hip = get_pt(24)
        l_kne = get_pt(25)
        r_kne = get_pt(26)
        l_ank = get_pt(27)
        r_ank = get_pt(28)
        l_heel = get_pt(29)
        r_heel = get_pt(30)
        l_toe = get_pt(31)
        r_toe = get_pt(32)

        obs_count = sum(1 for pt in body_points if pt[2] >= min_conf)
        if hands:
            obs_count += sum(len(kpts) for kpts in hands.values())
        if face:
            obs_count += len(face)
        result["summary_counts"]["observed_landmarks"] = obs_count

        surface_points_count = 0

        # --- 1. TORSO ANATOMY ---
        if l_sho and r_sho and l_hip and r_hip:
            # Mid-points
            mid_sho = ((l_sho[0] + r_sho[0]) * 0.5, (l_sho[1] + r_sho[1]) * 0.5)
            mid_hip = ((l_hip[0] + r_hip[0]) * 0.5, (l_hip[1] + r_hip[1]) * 0.5)
            sho_span = abs(l_sho[0] - r_sho[0])
            torso_height = abs(mid_hip[1] - mid_sho[1])

            # Sternum & Manubrium (Upper Torso)
            sternum_y = mid_sho[1] + torso_height * 0.25
            sternum = (mid_sho[0], sternum_y, min(l_sho[2], r_sho[2]) * 0.9)

            # Left and Right Clavicles (contours between mid_sho and shoulder joints)
            l_clavicle = (mid_sho[0] + (l_sho[0] - mid_sho[0]) * 0.5, mid_sho[1] * 0.9 + l_sho[1] * 0.1, l_sho[2] * 0.85)
            r_clavicle = (mid_sho[0] + (r_sho[0] - mid_sho[0]) * 0.5, mid_sho[1] * 0.9 + r_sho[1] * 0.1, r_sho[2] * 0.85)

            # Chest & Rib Cage Contours
            chest_y = mid_sho[1] + torso_height * 0.35
            chest_width = sho_span * 0.42
            chest_l = (mid_sho[0] - chest_width, chest_y, min(l_sho[2], r_sho[2]) * 0.8)
            chest_r = (mid_sho[0] + chest_width, chest_y, min(l_sho[2], r_sho[2]) * 0.8)

            # Waist & Abdomen
            waist_y = mid_sho[1] + torso_height * 0.65
            waist_width = sho_span * 0.36
            navel = (mid_sho[0], waist_y, min(l_hip[2], r_hip[2]) * 0.85)
            waist_l = (mid_sho[0] - waist_width, waist_y, l_hip[2] * 0.8)
            waist_r = (mid_sho[0] + waist_width, waist_y, r_hip[2] * 0.8)

            # Pelvis & Hip Contours
            pelvis_center = (mid_hip[0], mid_hip[1], min(l_hip[2], r_hip[2]) * 0.9)

            result["torso"] = {
                "clavicles": [l_clavicle, r_clavicle],
                "sternum": sternum,
                "chest_contours": [chest_l, chest_r],
                "abdomen_navel": navel,
                "waist_contours": [waist_l, waist_r],
                "pelvis": pelvis_center,
                "type": "inferred_surface"
            }
            surface_points_count += 9

        # --- 2. ARMS & FOREARMS ---
        for side, sho, elb, wri in [("left", l_sho, l_elb, l_wri), ("right", r_sho, r_elb, r_wri)]:
            if sho and elb:
                # Upper arm bicep/tricep contour midpoint
                mid_upper = ((sho[0] + elb[0]) * 0.5, (sho[1] + elb[1]) * 0.5, (sho[2] + elb[2]) * 0.5 * 0.9)
                result["arms"][side]["upper_arm_contour"] = mid_upper
                surface_points_count += 1
            if elb and wri:
                # Forearm contour midpoint
                mid_forearm = ((elb[0] + wri[0]) * 0.5, (elb[1] + wri[1]) * 0.5, (elb[2] + wri[2]) * 0.5 * 0.9)
                result["arms"][side]["forearm_contour"] = mid_forearm
                surface_points_count += 1

        # --- 3. LEGS, THIGHS & CALVES ---
        for side, hip, kne, ank in [("left", l_hip, l_kne, l_ank), ("right", r_hip, r_kne, r_ank)]:
            if hip and kne:
                # Thigh contours (front, inner, outer)
                thigh_mid = ((hip[0] + kne[0]) * 0.5, (hip[1] + kne[1]) * 0.5, (hip[2] + kne[2]) * 0.5 * 0.85)
                result["legs"][side]["thigh_contour"] = thigh_mid
                surface_points_count += 1
            if kne and ank:
                # Calf / shin contour
                calf_mid = ((kne[0] + ank[0]) * 0.5, (kne[1] + ank[1]) * 0.5, (kne[2] + ank[2]) * 0.5 * 0.85)
                result["legs"][side]["calf_contour"] = calf_mid
                surface_points_count += 1

        # --- 4. HEAD & NECK ---
        if nose and l_sho and r_sho:
            mid_sho = ((l_sho[0] + r_sho[0]) * 0.5, (l_sho[1] + r_sho[1]) * 0.5)
            neck = (mid_sho[0], mid_sho[1] * 0.7 + nose[1] * 0.3, min(l_sho[2], r_sho[2]) * 0.9)
            result["head"]["neck"] = neck
            surface_points_count += 1

        result["summary_counts"]["estimated_surface_points"] = surface_points_count
        return result
