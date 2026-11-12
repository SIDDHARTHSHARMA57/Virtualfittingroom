"""
Tracking Service Worker with Anatomical Regions and Live SMPL-X Motion Retargeting.
Runs camera capture, MediaPipe inference, anatomical contour extraction,
and live SMPL-X parametric 3D body motion retargeting asynchronously.
"""
import time
import socket
import logging
import threading
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np

from src.camera.capture import ThreadedCamera
from src.camera.tracking import UnifiedTracker
from src.camera.anatomy import AnatomicalBodyModel
from src.camera.smplx_fitter import SMPLXFittingEngine
from src.camera.protocol import serialize_packet
from src.camera.monitor import PerformanceMonitor

logger = logging.getLogger(__name__)

class TrackingService:
    def __init__(
        self,
        pose_model_path: str,
        hand_model_path: str,
        face_model_path: str,
        smplx_model_dir: str = "/media/siddharth-sharam/Workspace/VirtualFittingRoom/models",
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        fourcc: str = "MJPG",
        host: str = "127.0.0.1",
        port: int = 8766,
        stream_to_blender: bool = True
    ):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.host = host
        self.port = port
        self.stream_to_blender = stream_to_blender

        # Model paths
        self.pose_model_path = pose_model_path
        self.hand_model_path = hand_model_path
        self.face_model_path = face_model_path
        self.smplx_model_dir = smplx_model_dir

        # Core subsystems
        self.camera: Optional[ThreadedCamera] = None
        self.tracker: Optional[UnifiedTracker] = None
        self.anatomy_model = AnatomicalBodyModel()
        self.smplx_engine: Optional[SMPLXFittingEngine] = None
        self.monitor = PerformanceMonitor()

        # Threading state
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Latest UI snapshots
        self._latest_raw_frame: Optional[np.ndarray] = None
        self._latest_tracking: Dict[str, Any] = {}
        self._latest_anatomy: Dict[str, Any] = {}
        self._latest_smplx_verts: Optional[np.ndarray] = None
        self._smplx_latency_ms: float = 0.0
        self._smplx_is_live: bool = False
        self._smplx_valid_joints: int = 0

        # Feature flags
        self.enable_hands = True
        self.enable_face = True
        self.enable_smoothing = True
        self.enable_smplx = True

        # Network
        self.sock: Optional[socket.socket] = None
        self._last_sock_attempt = 0.0
        self.status_message = "Idle"
        self.camera_busy = False

    def start_service(self) -> bool:
        if self._running:
            return True

        self.status_message = "Starting camera..."
        self.camera_busy = False

        self.camera = ThreadedCamera(
            device_id=self.device_id,
            width=self.width,
            height=self.height,
            fps=self.fps,
            fourcc=self.fourcc,
            stale_threshold_s=0.5
        )

        if not self.camera.start():
            self.status_message = f"Error: Cannot open /dev/video{self.device_id} (Busy or Disconnected)"
            self.camera_busy = True
            logger.error(self.status_message)
            return False

        self.status_message = "Loading tracking models..."
        self.tracker = UnifiedTracker(
            pose_model_path=self.pose_model_path,
            hand_model_path=self.hand_model_path,
            face_model_path=self.face_model_path,
            enable_pose=True,
            enable_hands=self.enable_hands,
            enable_face=self.enable_face,
            enable_smoothing=self.enable_smoothing,
        )

        # Initialize SMPL-X parametric motion engine
        self.smplx_engine = SMPLXFittingEngine(model_dir=self.smplx_model_dir)

        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, name="VFR-ServiceWorker", daemon=True)
        self._worker_thread.start()
        self.status_message = "Running"
        logger.info("Tracking service with live SMPL-X motion retargeting started.")
        return True

    def _worker_loop(self):
        while self._running:
            if self.camera is None or self.tracker is None:
                time.sleep(0.01)
                continue

            self.tracker.enable_hands = self.enable_hands
            self.tracker.enable_face = self.enable_face

            frame, frame_time, is_stale = self.camera.get_latest_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            now = time.time()
            frame_age = now - frame_time
            mirrored = cv2.flip(frame, 1)

            # 1. MediaPipe Inference
            tracking_res = self.tracker.process(mirrored, timestamp=frame_time)

            # 2. Anatomical Regions & Inferred Surface Points
            anatomy_res = self.anatomy_model.compute_anatomical_regions(
                body_points=tracking_res["body_points"],
                world_points=tracking_res["world_points"],
                hands=tracking_res["hands"],
                face=tracking_res["face"]
            )

            # 3. Live SMPL-X Motion Retargeting
            smplx_verts = None
            smplx_lat = 0.0
            is_live = False
            valid_j = 0
            if self.enable_smplx and self.smplx_engine and self.smplx_engine.enabled:
                fit_out = self.smplx_engine.update_from_world_landmarks(
                    world_points=tracking_res.get("world_points", []),
                    body_points=tracking_res.get("body_points", []),
                    min_conf=0.35,
                    smoothing=0.40
                )
                smplx_verts = fit_out.get("vertices")
                smplx_lat = fit_out.get("latency_ms", 0.0)
                is_live = fit_out.get("is_live_fitted", False)
                valid_j = fit_out.get("valid_joints", 0)

            # 4. Telemetry Update
            latencies = dict(tracking_res["latencies"])
            latencies["smplx_ms"] = smplx_lat
            self.monitor.update_inference(
                frame_age_s=frame_age,
                is_stale=is_stale,
                latencies=latencies
            )

            # 5. Blender Streaming
            if self.stream_to_blender:
                payload = serialize_packet(
                    body_points=tracking_res["body_points"],
                    world_points=tracking_res["world_points"],
                    hands=tracking_res["hands"],
                    face=tracking_res["face"],
                    face_matrix=tracking_res["face_matrix"],
                    blendshapes=tracking_res["blendshapes"],
                    timestamp=tracking_res["timestamp"]
                )
                self._send_to_blender(payload)

            # 6. Store for UI
            with self._lock:
                self._latest_raw_frame = mirrored
                self._latest_tracking = tracking_res
                self._latest_anatomy = anatomy_res
                self._latest_smplx_verts = smplx_verts
                self._smplx_latency_ms = smplx_lat
                self._smplx_is_live = is_live
                self._smplx_valid_joints = valid_j

            if self.monitor.should_log():
                self.monitor.log_status(self.camera.get_stats())

    def _send_to_blender(self, payload: bytes):
        if self.sock is None:
            now = time.time()
            if now - self._last_sock_attempt < 1.0:
                return
            self._last_sock_attempt = now
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.3)
                s.connect((self.host, self.port))
                self.sock = s
                self.monitor.update_network(is_connected=True, success=True)
            except (OSError, socket.timeout):
                self.monitor.update_network(is_connected=False, success=False)
                return

        try:
            self.sock.sendall(payload)
            self.monitor.update_network(is_connected=True, success=True)
        except (OSError, socket.timeout):
            if self.sock:
                try:
                    self.sock.close()
                except Exception:
                    pass
            self.sock = None
            self.monitor.update_network(is_connected=False, success=False)

    def get_ui_snapshot(self) -> Tuple[Optional[np.ndarray], Dict[str, Any], Dict[str, Any], Optional[np.ndarray], Dict[str, Any], str]:
        """
        Returns:
            (mirrored_frame, tracking_dict, anatomy_dict, smplx_vertices, monitor_summary, status_msg)
        """
        with self._lock:
            frame = self._latest_raw_frame.copy() if self._latest_raw_frame is not None else None
            tracking = dict(self._latest_tracking)
            anatomy = dict(self._latest_anatomy)
            verts = self._latest_smplx_verts.copy() if self._latest_smplx_verts is not None else None
            smplx_lat = self._smplx_latency_ms
            is_live = self._smplx_is_live
            valid_j = self._smplx_valid_joints
            cam_stats = self.camera.get_stats() if self.camera else {}

        summary = self.monitor.get_summary()
        summary["capture_fps"] = cam_stats.get("capture_fps", 0.0)
        summary["read_failures"] = cam_stats.get("read_failures", 0)
        summary["reconnect_count"] = cam_stats.get("reconnect_count", 0)
        summary["smplx_latency_ms"] = round(smplx_lat, 1)
        summary["smplx_is_live"] = is_live
        summary["smplx_valid_joints"] = valid_j
        return frame, tracking, anatomy, verts, summary, self.status_message

    def stop_service(self):
        self._running = False
        self.status_message = "Stopped"
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        if self.camera:
            self.camera.stop()
            self.camera = None
        if self.tracker:
            self.tracker.close()
            self.tracker = None
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        logger.info("Tracking service stopped cleanly.")
