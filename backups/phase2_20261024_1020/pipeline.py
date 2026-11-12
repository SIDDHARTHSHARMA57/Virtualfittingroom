"""
Pipeline Coordinator Module.
Coordinates threaded capture, unified tracking inference, network streaming,
and optional GUI preview.
"""
import time
import socket
import logging
from typing import Optional, Dict, Any
import cv2
import numpy as np

from src.camera.capture import ThreadedCamera
from src.camera.tracking import UnifiedTracker
from src.camera.protocol import serialize_packet
from src.camera.monitor import PerformanceMonitor

logger = logging.getLogger(__name__)

class TrackingPipeline:
    def __init__(
        self,
        camera: ThreadedCamera,
        tracker: UnifiedTracker,
        host: str = "127.0.0.1",
        port: int = 8766,
        show_preview: bool = False,
        preview_title: str = "VirtualFittingRoom Tracking Preview",
        connect_retry_s: float = 1.0,
        network_timeout_s: float = 0.5
    ):
        self.camera = camera
        self.tracker = tracker
        self.host = host
        self.port = port
        self.show_preview = show_preview
        self.preview_title = preview_title
        self.connect_retry_s = connect_retry_s
        self.network_timeout_s = network_timeout_s

        self.monitor = PerformanceMonitor()
        self.sock: Optional[socket.socket] = None
        self._running = False
        self._last_connect_attempt = 0.0

    def _ensure_socket(self) -> bool:
        if self.sock is not None:
            return True

        now = time.time()
        if now - self._last_connect_attempt < self.connect_retry_s:
            return False

        self._last_connect_attempt = now
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.network_timeout_s)
            s.connect((self.host, self.port))
            self.sock = s
            self.monitor.update_network(is_connected=True, success=True)
            logger.info(f"Connected to Blender receiver at {self.host}:{self.port}")
            return True
        except (OSError, socket.timeout):
            self.monitor.update_network(is_connected=False, success=False)
            return False

    def _send_packet(self, payload: bytes) -> bool:
        if not self._ensure_socket():
            return False

        try:
            self.sock.sendall(payload)
            self.monitor.update_network(is_connected=True, success=True)
            return True
        except (OSError, socket.timeout) as exc:
            logger.warning(f"Socket send failed ({exc}), resetting connection.")
            if self.sock:
                try:
                    self.sock.close()
                except Exception:
                    pass
            self.sock = None
            self.monitor.update_network(is_connected=False, success=False)
            return False

    def start(self):
        """Starts camera and begins the main pipeline loop."""
        if not self.camera.start():
            logger.error("Failed to start camera hardware.")
            return

        self._running = True
        logger.info("Pipeline started. Press 'q' or 'ESC' in preview to quit, or Ctrl+C in terminal.")

        try:
            while self._running:
                # 1. Fetch latest frame without blocking capture thread
                frame, frame_time, is_stale = self.camera.get_latest_frame()
                if frame is None:
                    time.sleep(0.005)
                    continue

                now = time.time()
                frame_age = now - frame_time

                # 2. Run multi-model inference
                # Optional flip for mirror display / natural tracking
                display_frame = cv2.flip(frame, 1)
                tracking_result = self.tracker.process(display_frame, timestamp=frame_time)

                self.monitor.update_inference(
                    frame_age_s=frame_age,
                    is_stale=is_stale,
                    latencies=tracking_result["latencies"]
                )

                # 3. Serialize and send to Blender
                payload = serialize_packet(
                    body_points=tracking_result["body_points"],
                    world_points=tracking_result["world_points"],
                    hands=tracking_result["hands"],
                    face=tracking_result["face"],
                    face_matrix=tracking_result["face_matrix"],
                    blendshapes=tracking_result["blendshapes"],
                    timestamp=tracking_result["timestamp"]
                )
                self._send_packet(payload)

                # 4. Optional GUI Preview
                if self.show_preview:
                    h, w = display_frame.shape[:2]
                    # Draw body landmarks
                    for pt in tracking_result["body_points"]:
                        x, y, conf = pt
                        if conf >= 0.35:
                            px, py = int(x * w), int(y * h)
                            if 0 <= px < w and 0 <= py < h:
                                cv2.circle(display_frame, (px, py), 4, (0, 255, 0), -1)

                    # Overlay status HUD
                    hud_text = (
                        f"Body: {len(tracking_result['body_points'])} | "
                        f"Face: {len(tracking_result['face'])} | "
                        f"Hands: {len(tracking_result['hands'])} | "
                        f"FPS: {self.monitor.inference_fps:.1f}"
                    )
                    cv2.putText(
                        display_frame, hud_text, (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
                    )

                    cv2.imshow(self.preview_title, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        logger.info("Quit key pressed in preview window.")
                        break

                # 5. Rate-limited Telemetry Log
                if self.monitor.should_log():
                    self.monitor.log_status(self.camera.get_stats())

        except KeyboardInterrupt:
            logger.info("Pipeline received interrupt signal.")
        finally:
            self.stop()

    def stop(self):
        """Clean shutdown of all resources."""
        self._running = False
        logger.info("Shutting down pipeline...")
        self.camera.stop()
        self.tracker.close()

        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

        if self.show_preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        logger.info("Pipeline shutdown complete.")
