"""
Tracking Service Worker with Anatomical Regions and Live SMPL-X Motion Retargeting.
Runs camera capture, MediaPipe inference, anatomical contour extraction,
and live SMPL-X parametric 3D body motion retargeting asynchronously.
"""
import os
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
from src.camera.frame_share import FrameWriter

logger = logging.getLogger(__name__)

INTRINSICS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                               "config", "camera_intrinsics.json")
DEFAULT_HFOV_DEG = 60.0  # typical webcam; replace by running scripts/calibrate_camera.py


def load_intrinsics(width: int, height: int, mirrored: bool = True, path: str = INTRINSICS_PATH) -> Dict[str, float]:
    """Pinhole intrinsics for the image MediaPipe sees, scaled to the capture resolution.
    Uses config/camera_intrinsics.json (from scripts/calibrate_camera.py) if present, else a
    DEFAULT_HFOV_DEG guess. mirrored=True flips cx because tracking runs on the flipped frame."""
    import json
    K = None
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                c = json.load(f)
            sx, sy = width / c["width"], height / c["height"]
            K = {"fx": c["fx"] * sx, "fy": c["fy"] * sy, "cx": c["cx"] * sx, "cy": c["cy"] * sy, "calibrated": True}
        except (OSError, KeyError, ValueError) as exc:
            logger.warning(f"Ignoring bad {path}: {exc}")
    if K is None:
        f = (width / 2) / np.tan(np.radians(DEFAULT_HFOV_DEG) / 2)
        K = {"fx": f, "fy": f, "cx": width / 2, "cy": height / 2, "calibrated": False}
    if mirrored:
        K["cx"] = width - K["cx"]
    K.update({"w": width, "h": height})
    return {k: (round(float(v), 3) if not isinstance(v, bool) else v) for k, v in K.items()}

class TrackingService:
    def __init__(
        self,
        pose_model_path: str,
        hand_model_path: str,
        face_model_path: str,
        smplx_model_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models"),
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
        self._smplx_hands: int = 0
        self._smplx_params: Optional[Dict[str, list]] = None  # latest solved pose, streamed to Blender
        self.frame_writer: Optional[FrameWriter] = None
        self.intrinsics: Optional[Dict[str, float]] = None
        self._fit_cv = threading.Condition()
        self._fit_input: Optional[Dict[str, Any]] = None
        self._fit_thread: Optional[threading.Thread] = None
        self._feed_thread: Optional[threading.Thread] = None

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
        self.intrinsics = load_intrinsics(self.width, self.height, mirrored=True)
        self.smplx_engine = SMPLXFittingEngine(model_dir=self.smplx_model_dir, intrinsics=self.intrinsics)

        # Live camera frames for Blender's background (same capture; no second camera stream)
        try:
            self.frame_writer = FrameWriter()
        except Exception as exc:
            self.frame_writer = None
            logger.warning(f"Camera frame sharing disabled: {exc}")

        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, name="VFR-ServiceWorker", daemon=True)
        self._worker_thread.start()
        self._fit_thread = threading.Thread(target=self._fit_loop, name="VFR-SMPLXFit", daemon=True)
        self._fit_thread.start()
        self._feed_thread = threading.Thread(target=self._feed_loop, name="VFR-CameraFeed", daemon=True)
        self._feed_thread.start()
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

            # 3. Hand the frame to the SMPL-X fit thread (latest-wins; never blocks inference)
            if self.enable_smplx and self.smplx_engine and self.smplx_engine.enabled:
                with self._fit_cv:
                    self._fit_input = tracking_res
                    self._fit_cv.notify()

            # 4. Telemetry Update
            latencies = dict(tracking_res["latencies"])
            latencies["smplx_ms"] = self._smplx_latency_ms
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
                    timestamp=tracking_res["timestamp"],
                    smplx=self._smplx_params,
                )
                self._send_to_blender(payload)

            # 6. Store for UI
            with self._lock:
                self._latest_raw_frame = mirrored
                self._latest_tracking = tracking_res
                self._latest_anatomy = anatomy_res

            if self.monitor.should_log():
                self.monitor.log_status(self.camera.get_stats())

    def _feed_loop(self):
        """Publishes every new camera frame (mirrored, as the customer sees it) to shared memory
        for Blender's live background, at camera rate - independent of the slower inference loop.
        ponytail: background runs ahead of the pose by the inference latency; publish the frame
        tracking used instead if exact sync matters more than smoothness."""
        last_ts = 0.0
        while self._running:
            if not (self.stream_to_blender and self.camera and self.frame_writer):
                time.sleep(0.05)
                continue
            frame, ts, _ = self.camera.get_latest_frame()
            if frame is None or ts == last_ts:
                time.sleep(0.004)
                continue
            last_ts = ts
            try:
                self.frame_writer.write(cv2.flip(frame, 1), ts)
            except ValueError as exc:  # unsupported resolution: stop trying, keep tracking
                logger.warning(f"Camera feed to Blender disabled: {exc}")
                self.frame_writer = None

    def _fit_loop(self):
        """SMPL-X fitting on its own thread so MediaPipe and the fit run in parallel."""
        while self._running:
            with self._fit_cv:
                while self._fit_input is None and self._running:
                    self._fit_cv.wait(timeout=0.2)
                res, self._fit_input = self._fit_input, None
            if res is None or self.smplx_engine is None:
                continue
            if not res.get("world_points"):
                self.smplx_engine.reset()  # person left: next detection starts cold
                continue
            out = self.smplx_engine.update_from_world_landmarks(
                world_points=res["world_points"],
                body_points=res.get("body_points"),
                hands=res.get("hands"),
                hands_world=res.get("hands_world"),
                min_conf=0.35,
            )
            with self._lock:
                self._latest_smplx_verts = out.get("vertices")
                if out.get("fitted"):
                    self._smplx_latency_ms = out.get("latency_ms", 0.0)
                self._smplx_is_live = out.get("is_live_fitted", False)
                self._smplx_valid_joints = out.get("valid_joints", 0)
                self._smplx_hands = out.get("hands_tracked", 0)
                if out.get("fitted"):
                    self._smplx_params = {
                        k: [round(float(v), 5) for v in out[k]]
                        for k in ("global_orient", "body_pose", "left_hand_pose", "right_hand_pose")
                    }
                    self._smplx_params["betas"] = self.smplx_engine.betas[0].tolist()
                    if out.get("transl") is not None:  # camera frame: x right, y down, z forward (m)
                        self._smplx_params["transl"] = [round(float(v), 4) for v in out["transl"]]
                    self._smplx_params["camera"] = self.intrinsics
                    self._smplx_params["penetrations"] = out.get("penetrations", {})

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
        summary["smplx_hands"] = self._smplx_hands
        return frame, tracking, anatomy, verts, summary, self.status_message

    def stop_service(self):
        self._running = False
        self.status_message = "Stopped"
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        with self._fit_cv:
            self._fit_cv.notify()
        if self._fit_thread and self._fit_thread.is_alive():
            self._fit_thread.join(timeout=1.0)
        if self._feed_thread and self._feed_thread.is_alive():
            self._feed_thread.join(timeout=1.0)
        if self.camera:
            self.camera.stop()
            self.camera = None
        if self.tracker:
            self.tracker.close()
            self.tracker = None
        if self.frame_writer is not None:
            self.frame_writer.close()
            self.frame_writer = None
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        logger.info("Tracking service stopped cleanly.")
