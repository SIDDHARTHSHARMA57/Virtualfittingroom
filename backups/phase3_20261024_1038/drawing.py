"""
Drawing utilities for rendering skeleton, anatomical contours, hand, and face landmark overlays
onto video frames for the VirtualFittingRoom Dashboard.
"""
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

# MediaPipe Pose 33 connections
POSE_CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left Arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right Arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Left Leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right Leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    # Face outline
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10)
]

# MediaPipe Hand 21 connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

def render_tracking_overlays(
    frame: np.ndarray,
    tracking_data: Dict[str, Any],
    draw_skeleton: bool = True,
    draw_body_points: bool = True,
    draw_hands: bool = True,
    draw_face: bool = True,
    draw_anatomy: bool = True,
    anatomical_data: Optional[Dict[str, Any]] = None,
    min_conf: float = 0.35
) -> np.ndarray:
    """
    Renders overlays directly onto a copy of the input frame.
    Distinct styling for:
    - Observed landmarks (Solid green/yellow circles)
    - Inferred anatomical contours/surface points (Dashed lines / hollow cyan diamonds)
    - Hand landmarks (Cyan / Magenta)
    - Face mesh (Subtle lavender points)
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # 1. Pose Skeleton Lines
    body_points = tracking_data.get("body_points", [])
    if body_points and len(body_points) == 33:
        if draw_skeleton:
            for p1_idx, p2_idx in POSE_CONNECTIONS:
                p1 = body_points[p1_idx]
                p2 = body_points[p2_idx]
                if p1[2] >= min_conf and p2[2] >= min_conf:
                    x1, y1 = int(p1[0] * w), int(p1[1] * h)
                    x2, y2 = int(p2[0] * w), int(p2[1] * h)
                    if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)

        # 2. Observed Body Joint Landmarks
        if draw_body_points:
            for idx, pt in enumerate(body_points):
                x, y, conf = pt
                if conf >= min_conf:
                    px, py = int(x * w), int(y * h)
                    if 0 <= px < w and 0 <= py < h:
                        color = (0, 165, 255) if idx in (11, 12, 15, 16, 23, 24) else (0, 255, 0)
                        cv2.circle(out, (px, py), 4, color, -1, cv2.LINE_AA)
                        cv2.circle(out, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)

    # 3. Dense Anatomical Surface Contours (Torso, Clavicle, Sternum, Ribs, Waist, Limbs)
    if draw_anatomy and anatomical_data:
        torso = anatomical_data.get("torso", {})
        if torso:
            # Clavicles (Cyan lines)
            for cl in torso.get("clavicles", []):
                cx, cy = int(cl[0] * w), int(cl[1] * h)
                cv2.circle(out, (cx, cy), 3, (255, 255, 0), -1)

            # Sternum & Chest Contours (Hollow diamond marker)
            st = torso.get("sternum")
            if st:
                sx, sy = int(st[0] * w), int(st[1] * h)
                cv2.drawMarker(out, (sx, sy), (255, 200, 0), markerType=cv2.MARKER_DIAMOND, markerSize=7, thickness=1)

            chest = torso.get("chest_contours", [])
            if len(chest) == 2:
                c1x, c1y = int(chest[0][0] * w), int(chest[0][1] * h)
                c2x, c2y = int(chest[1][0] * w), int(chest[1][1] * h)
                cv2.line(out, (c1x, c1y), (c2x, c2y), (255, 180, 0), 1, cv2.LINE_4)

            # Waist & Navel
            navel = torso.get("abdomen_navel")
            if navel:
                nx, ny = int(navel[0] * w), int(navel[1] * h)
                cv2.drawMarker(out, (nx, ny), (0, 200, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=6, thickness=1)

            waist = torso.get("waist_contours", [])
            if len(waist) == 2:
                w1x, w1y = int(waist[0][0] * w), int(waist[0][1] * h)
                w2x, w2y = int(waist[1][0] * w), int(waist[1][1] * h)
                cv2.line(out, (w1x, w1y), (w2x, w2y), (0, 200, 255), 1, cv2.LINE_4)

        # Limb contours
        arms = anatomical_data.get("arms", {})
        for side in ("left", "right"):
            s_data = arms.get(side, {})
            u_pt = s_data.get("upper_arm_contour")
            if u_pt:
                cv2.circle(out, (int(u_pt[0] * w), int(u_pt[1] * h)), 3, (0, 255, 200), -1)
            f_pt = s_data.get("forearm_contour")
            if f_pt:
                cv2.circle(out, (int(f_pt[0] * w), int(f_pt[1] * h)), 3, (0, 255, 200), -1)

        legs = anatomical_data.get("legs", {})
        for side in ("left", "right"):
            l_data = legs.get(side, {})
            t_pt = l_data.get("thigh_contour")
            if t_pt:
                cv2.circle(out, (int(t_pt[0] * w), int(t_pt[1] * h)), 3, (200, 200, 0), -1)
            c_pt = l_data.get("calf_contour")
            if c_pt:
                cv2.circle(out, (int(c_pt[0] * w), int(c_pt[1] * h)), 3, (200, 200, 0), -1)

    # 4. Hand Overlays
    hands = tracking_data.get("hands", {})
    if draw_hands and hands:
        for handedness, kpts in hands.items():
            bone_color = (255, 255, 0) if handedness == "Left" else (255, 0, 255)
            for s_idx, e_idx in HAND_CONNECTIONS:
                if s_idx < len(kpts) and e_idx < len(kpts):
                    p1 = kpts[s_idx]
                    p2 = kpts[e_idx]
                    x1, y1 = int(p1[0] * w), int(p1[1] * h)
                    x2, y2 = int(p2[0] * w), int(p2[1] * h)
                    if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                        cv2.line(out, (x1, y1), (x2, y2), bone_color, 1, cv2.LINE_AA)

            for idx, pt in enumerate(kpts):
                px, py = int(pt[0] * w), int(pt[1] * h)
                if 0 <= px < w and 0 <= py < h:
                    tip_color = (0, 0, 255) if idx in (4, 8, 12, 16, 20) else (255, 255, 255)
                    cv2.circle(out, (px, py), 3, tip_color, -1, cv2.LINE_AA)

    # 5. Face Mesh Overlays
    face_points = tracking_data.get("face", [])
    if draw_face and face_points:
        for idx, pt in enumerate(face_points):
            if idx % 2 == 0 or idx in (1, 33, 61, 199, 263, 291):
                px, py = int(pt[0] * w), int(pt[1] * h)
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(out, (px, py), 1, (200, 200, 255), -1, cv2.LINE_AA)

    return out

def render_3d_orthographic_view(
    vertices: np.ndarray,
    view_angle: str = "front",
    width: int = 320,
    height: int = 360
) -> np.ndarray:
    """
    Renders an orthographic wireframe / point cloud view of the 10,475 SMPL-X body vertices
    from Front, Side, or Rear perspectives.
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if vertices is None or len(vertices) == 0:
        cv2.putText(canvas, "SMPL-X Inactive", (width // 4, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 90), 1)
        return canvas

    # SMPL-X coordinates: X=left/right, Y=up/down (inverted in image), Z=depth
    pts = vertices.copy()

    if view_angle == "side":
        # Project Z to X, keep Y
        proj_x = pts[:, 2]
        proj_y = pts[:, 1]
    elif view_angle == "rear":
        # Invert X, keep Y
        proj_x = -pts[:, 0]
        proj_y = pts[:, 1]
    else:  # front
        proj_x = pts[:, 0]
        proj_y = pts[:, 1]

    # Normalize to canvas dimensions
    min_x, max_x = np.min(proj_x), np.max(proj_x)
    min_y, max_y = np.min(proj_y), np.max(proj_y)

    span_x = max(max_x - min_x, 1e-3)
    span_y = max(max_y - min_y, 1e-3)

    scale = min((width * 0.8) / span_x, (height * 0.8) / span_y)

    mid_x = (min_x + max_x) * 0.5
    mid_y = (min_y + max_y) * 0.5

    cx, cy = width // 2, height // 2

    # Subsample vertices for fast rendering (every 4th vertex -> ~2,600 points)
    step = 4
    for i in range(0, len(pts), step):
        px = int(cx + (proj_x[i] - mid_x) * scale)
        py = int(cy + (proj_y[i] - mid_y) * scale)
        if 0 <= px < width and 0 <= py < height:
            cv2.circle(canvas, (px, py), 1, (180, 220, 255), -1)

    cv2.putText(canvas, f"SMPL-X 3D View ({view_angle.upper()})", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return canvas
