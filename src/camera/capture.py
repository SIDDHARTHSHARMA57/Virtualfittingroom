"""
Threaded Camera Capture Module.
Runs a dedicated capture thread that keeps only the latest frame in a thread-safe buffer,
preventing slow inference or network pauses from stalling camera frame acquisition.
"""
import time
import logging
import threading
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class ThreadedCamera:
    """
    Owns the physical camera on a background thread.
    Maintains the latest frame, tracks capture FPS, drops stale frames, and handles auto-reconnect.
    """
    def __init__(
        self,
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        fourcc: str = "MJPG",
        max_reconnect_attempts: int = 5,
        reconnect_delay_s: float = 1.0,
        stale_threshold_s: float = 0.5
    ):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay_s = reconnect_delay_s
        self.stale_threshold_s = stale_threshold_s

        self.cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Frame storage & telemetry
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_time: float = 0.0
        self._frame_count: int = 0
        self._read_failures: int = 0
        self._reconnect_count: int = 0
        self._capture_fps: float = 0.0
        self._last_fps_calc_time = time.time()
        self._last_fps_frame_count = 0

    def _open_device(self) -> bool:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        logger.info(f"Opening camera /dev/video{self.device_id} (V4L2, {self.fourcc}, {self.width}x{self.height}@{self.fps})...")
        cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
        if not cap.isOpened():
            logger.warning(f"V4L2 backend open failed on device {self.device_id}, falling back to default backend...")
            cap.release()
            cap = cv2.VideoCapture(self.device_id)

        if cap.isOpened():
            if len(self.fourcc) == 4:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(0.15)
            self.cap = cap
            return True
        else:
            self.cap = None
            return False

    def start(self) -> bool:
        """Starts the capture thread. Returns True if device opened successfully."""
        if self._running:
            return True
        if not self._open_device():
            logger.error(f"Failed to open camera device {self.device_id}")
            return False

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name="VFR-CameraCapture", daemon=True)
        self._thread.start()
        logger.info("Camera capture thread started successfully.")
        return True

    def _capture_loop(self):
        while self._running:
            if self.cap is None or not self.cap.isOpened():
                if not self._reconnect():
                    time.sleep(self.reconnect_delay_s)
                    continue

            ok, frame = self.cap.read()
            now = time.time()

            if not ok or frame is None:
                self._read_failures += 1
                logger.warning(f"Camera frame read failed (failures: {self._read_failures})")
                time.sleep(0.01)
                if self._read_failures % 30 == 0:
                    logger.warning("Consecutive read failures; initiating reconnect...")
                    self._reconnect()
                continue

            # Update latest frame
            with self._lock:
                self._latest_frame = frame
                self._latest_frame_time = now
                self._frame_count += 1

            # Update capture FPS every 1 second
            if now - self._last_fps_calc_time >= 1.0:
                elapsed = now - self._last_fps_calc_time
                frames = self._frame_count - self._last_fps_frame_count
                self._capture_fps = frames / elapsed
                self._last_fps_calc_time = now
                self._last_fps_frame_count = self._frame_count

    def _reconnect(self) -> bool:
        self._reconnect_count += 1
        logger.info(f"Attempting camera reconnect #{self._reconnect_count}...")
        if self._open_device():
            logger.info("Camera reconnected successfully.")
            return True
        return False

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], float, bool]:
        """
        Returns:
            (frame, frame_timestamp, is_stale)
        """
        now = time.time()
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0, True
            frame_copy = self._latest_frame.copy()
            frame_time = self._latest_frame_time

        age = now - frame_time
        is_stale = age > self.stale_threshold_s
        return frame_copy, frame_time, is_stale

    def get_stats(self) -> dict:
        now = time.time()
        with self._lock:
            age = (now - self._latest_frame_time) if self._latest_frame_time > 0 else -1.0
            total_frames = self._frame_count
        return {
            "device_id": self.device_id,
            "capture_fps": round(self._capture_fps, 1),
            "total_frames": total_frames,
            "frame_age_s": round(age, 3),
            "read_failures": self._read_failures,
            "reconnect_count": self._reconnect_count,
            "is_alive": self._running and (self.cap is not None and self.cap.isOpened()),
        }

    def stop(self):
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
            self.cap = None
        logger.info("Camera capture stopped and hardware released.")
