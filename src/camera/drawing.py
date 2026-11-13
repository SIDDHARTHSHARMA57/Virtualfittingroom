"""
Drawing utilities for rendering skeleton, anatomical contours, hand, face landmark overlays,
and correctly-oriented 3D orthographic mesh viewports.
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
    - Inferred anatomical contours/surface points (Cyan lines / hollow diamonds)
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
            # Clavicles (Cyan dots)
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

def render_person_selection(frame: np.ndarray, selection: Dict[str, Any]) -> np.ndarray:
    """Operator/debug overlay: a box + temporary ID per detected person, the selected one
    highlighted, and the selector state. (The mirror/Blender view never shows this.)"""
    if not selection or not selection.get("enabled"):
        return frame
    out = frame
    target = selection.get("target_id")
    for tid, (x0, y0, x1, y1), conf in selection.get("people", []):
        sel = tid == target
        color = (80, 220, 80) if sel else (170, 170, 170)
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), color, 3 if sel else 1)
        label = f"ID {tid}" + ("  SELECTED" if sel else "  tap to select")
        cv2.rectangle(out, (int(x0), int(y0) - 20), (int(x0) + 9 * len(label), int(y0)), color, -1)
        cv2.putText(out, label, (int(x0) + 3, int(y0) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
    state = selection.get("state", "")
    msg = selection.get("message", "")
    banner = f"{state}" + (f" - {msg}" if msg else "")
    good = state == "LOCKED"
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (30, 90, 30) if good else (20, 90, 160), -1)
    cv2.putText(out, banner, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def render_3d_orthographic_view(
    vertices: np.ndarray,
    view_angle: str = "front",
    width: int = 320,
    height: int = 360,
    is_live_fitted: bool = False,
    latency_ms: float = 0.0,
    valid_joints: int = 0
) -> np.ndarray:
    """
    Renders an orthographic wireframe / point cloud view of the 10,475 SMPL-X body vertices
    from Front, Side, or Rear perspectives.
    
    COORDINATE FRAME CONVENTION:
    - SMPL-X 3D space: +X = Right, +Y = Up, +Z = Forward.
    - Screen raster space: +X = Right, +Y = Down.
    - Therefore:
        Screen_X = proj_x
        Screen_Y = -proj_y (Up in 3D maps to Top in Screen)
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if vertices is None or len(vertices) == 0:
        cv2.putText(canvas, "SMPL-X Inactive", (width // 4, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 90), 1)
        return canvas

    pts = vertices.copy()

    # Determine horizontal and vertical 3D projection axes based on viewpoint
    if view_angle == "side":
        # Looking from person's right side towards left:
        # Depth (+Z) maps to horizontal axis (facing right). Height (+Y) is vertical.
        proj_h = pts[:, 2]
        proj_v = pts[:, 1]
    elif view_angle == "rear":
        # Looking from behind:
        # Left-right (-X) maps to horizontal axis. Height (+Y) is vertical.
        proj_h = -pts[:, 0]
        proj_v = pts[:, 1]
    else:  # front
        # Looking from front:
        # Left-right (+X) maps to horizontal axis. Height (+Y) is vertical.
        proj_h = pts[:, 0]
        proj_v = pts[:, 1]

    # Convert 3D vertical (+Y up) to 2D image vertical (+Y down):
    # SCREEN_Y = -proj_v so head is near top (small y) and feet are near bottom (large y)
    screen_y_pts = -proj_v
    screen_x_pts = proj_h

    min_x, max_x = np.min(screen_x_pts), np.max(screen_x_pts)
    min_y, max_y = np.min(screen_y_pts), np.max(screen_y_pts)

    span_x = max(max_x - min_x, 1e-3)
    span_y = max(max_y - min_y, 1e-3)

    # Uniform scale with 15% margin
    scale = min((width * 0.75) / span_x, (height * 0.75) / span_y)

    mid_x = (min_x + max_x) * 0.5
    mid_y = (min_y + max_y) * 0.5

    cx, cy = width // 2, height // 2

    # Point cloud rendering (subsample step = 4 -> ~2,600 points)
    # Highlight torso/head in warm gold, limbs in soft cyan
    step = 4
    for i in range(0, len(pts), step):
        px = int(cx + (screen_x_pts[i] - mid_x) * scale)
        py = int(cy + (screen_y_pts[i] - mid_y) * scale)
        if 0 <= px < width and 0 <= py < height:
            color = (180, 220, 255) if pts[i, 1] > -0.4 else (120, 190, 230)
            cv2.circle(canvas, (px, py), 1, color, -1)

    # Diagnostic HUD Overlay in the 3D viewport
    view_label = f"SMPL-X: {view_angle.upper()}"
    status_text = "LIVE RETARGETING" if is_live_fitted else "REST POSE"
    status_color = (0, 255, 0) if is_live_fitted else (180, 180, 180)

    cv2.putText(canvas, view_label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Mode: {status_text}", (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.40, status_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Joints: {valid_joints}/12", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    if latency_ms > 0:
        cv2.putText(canvas, f"Fit: {latency_ms:.1f}ms", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 220, 160), 1, cv2.LINE_AA)

    return canvas
