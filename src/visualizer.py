import cv2
import numpy as np
from typing import Dict, Any, List, Tuple

class PoseVisualizer:
    """
    Renders clean UI overlays, keypoints, skeleton links, bounding boxes,
    and accurate FPS/latency diagnostics.
    """
    def __init__(self, config: Dict[str, Any]):
        self.show_fps = config.get("show_fps", True)
        self.show_latency = config.get("show_latency", True)
        self.show_bbox = config.get("show_bbox", True)
        self.show_landmarks = config.get("show_landmarks", True)
        self.show_skeleton = config.get("show_skeleton", True)

    def draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray, connections: List[Tuple[int, int]], min_conf: float = 0.3):
        for p1, p2 in connections:
            if p1 < len(keypoints) and p2 < len(keypoints):
                pt1 = keypoints[p1]
                pt2 = keypoints[p2]
                if pt1[2] >= min_conf and pt2[2] >= min_conf:
                    x1, y1 = int(pt1[0]), int(pt1[1])
                    x2, y2 = int(pt2[0]), int(pt2[1])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 220, 255), 2, cv2.LINE_AA)

    def draw_keypoints(self, frame: np.ndarray, keypoints: np.ndarray, min_conf: float = 0.3):
        for idx, pt in enumerate(keypoints):
            x, y, conf = pt[0], pt[1], pt[2]
            if conf >= min_conf:
                ix, iy = int(x), int(y)
                # Outer circle
                cv2.circle(frame, (ix, iy), 5, (0, 0, 255), -1, cv2.LINE_AA)
                # Inner dot
                cv2.circle(frame, (ix, iy), 2, (255, 255, 255), -1, cv2.LINE_AA)

    def draw_bbox(self, frame: np.ndarray, bbox: np.ndarray, label: str = ""):
        if bbox is not None and len(bbox) >= 4:
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
            if label:
                cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1, cv2.LINE_AA)

    def draw_hud(self, frame: np.ndarray, fps: float, tracking_data: Dict[str, Any], device_name: str = ""):
        """
        Draws semi-transparent HUD telemetry bar at top of frame.
        """
        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Status text
        detected = tracking_data.get("detected", False)
        status_color = (0, 255, 0) if detected else (0, 140, 255)
        status_text = "TRACKING: LOCKED" if detected else "TRACKING: SEARCHING"

        backend = tracking_data.get("backend", "UNKNOWN").upper()
        latency_ms = tracking_data.get("latency_ms", 0.0)

        line1 = f"FPS: {fps:.1f}  |  Model: {backend}  |  Latency: {latency_ms:.1f}ms"
        if device_name:
            line1 += f" ({device_name})"
        
        cv2.putText(frame, line1, (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, status_text, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)
        
        shortcut_text = "[Q] Quit | [B] Backend | [M] Mirror"
        cv2.putText(frame, shortcut_text, (w - 310, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    def render(self, frame: np.ndarray, fps: float, tracking_data: Dict[str, Any], device_name: str = "") -> np.ndarray:
        out_frame = frame.copy()
        connections = tracking_data.get("connections", [])
        
        for person in tracking_data.get("persons", []):
            kpts = person.get("keypoints")
            bbox = person.get("bbox")
            
            if self.show_bbox and bbox is not None:
                conf = person.get("bbox_conf", 1.0)
                self.draw_bbox(out_frame, bbox, f"Person {person['id']} ({conf:.2f})")

            if self.show_skeleton and kpts is not None and connections:
                self.draw_skeleton(out_frame, kpts, connections)

            if self.show_landmarks and kpts is not None:
                self.draw_keypoints(out_frame, kpts)

        self.draw_hud(out_frame, fps, tracking_data, device_name)
        return out_frame
