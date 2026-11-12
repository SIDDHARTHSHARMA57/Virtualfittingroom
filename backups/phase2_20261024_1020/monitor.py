"""
Diagnostic & Telemetry Monitor.
Maintains structured statistics on capture FPS, inference FPS, latency, socket status,
dropped/stale frames, and landmark detection rates.
"""
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self, log_interval_s: float = 2.0):
        self.log_interval_s = log_interval_s
        self.last_log_time = time.time()

        # Telemetry metrics
        self.capture_fps: float = 0.0
        self.inference_fps: float = 0.0
        self.dropped_frames: int = 0
        self.stale_frames: int = 0
        self.total_inferred_frames: int = 0
        self.last_frame_age_s: float = 0.0

        # Latency metrics
        self.last_pose_latency_ms: float = 0.0
        self.last_hand_latency_ms: float = 0.0
        self.last_face_latency_ms: float = 0.0
        self.last_total_latency_ms: float = 0.0

        # Network metrics
        self.is_connected: bool = False
        self.send_failures: int = 0
        self.packets_sent: int = 0

        # Internal inference FPS tracking
        self._fps_window_start = time.time()
        self._fps_window_count = 0

    def update_inference(
        self,
        frame_age_s: float,
        is_stale: bool,
        latencies: Dict[str, float]
    ):
        now = time.time()
        self.total_inferred_frames += 1
        self._fps_window_count += 1
        self.last_frame_age_s = frame_age_s
        if is_stale:
            self.stale_frames += 1

        self.last_pose_latency_ms = latencies.get("pose_ms", 0.0)
        self.last_hand_latency_ms = latencies.get("hand_ms", 0.0)
        self.last_face_latency_ms = latencies.get("face_ms", 0.0)
        self.last_total_latency_ms = latencies.get("total_ms", 0.0)

        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            self.inference_fps = self._fps_window_count / elapsed
            self._fps_window_start = now
            self._fps_window_count = 0

    def update_network(self, is_connected: bool, success: bool):
        self.is_connected = is_connected
        if success:
            self.packets_sent += 1
        else:
            self.send_failures += 1

    def should_log(self) -> bool:
        return (time.time() - self.last_log_time) >= self.log_interval_s

    def log_status(self, capture_stats: Optional[Dict[str, Any]] = None):
        now = time.time()
        self.last_log_time = now

        cap_fps = capture_stats.get("capture_fps", 0.0) if capture_stats else 0.0
        read_fails = capture_stats.get("read_failures", 0) if capture_stats else 0
        reconnects = capture_stats.get("reconnect_count", 0) if capture_stats else 0

        logger.info(
            f"[MONITOR] Cap: {cap_fps:.1f} FPS | Inf: {self.inference_fps:.1f} FPS | "
            f"Latency (P/H/F/Tot): {self.last_pose_latency_ms:.1f}/{self.last_hand_latency_ms:.1f}/"
            f"{self.last_face_latency_ms:.1f}/{self.last_total_latency_ms:.1f} ms | "
            f"Age: {self.last_frame_age_s*1000:.1f}ms | Drops/Stale: {self.stale_frames} | "
            f"Net: {'CONN' if self.is_connected else 'DISCONN'} (pkts: {self.packets_sent}, err: {self.send_failures}) | "
            f"CamFails: {read_fails} (reconn: {reconnects})"
        )

    def get_summary(self) -> Dict[str, Any]:
        return {
            "inference_fps": round(self.inference_fps, 1),
            "total_inferred_frames": self.total_inferred_frames,
            "stale_frames": self.stale_frames,
            "last_frame_age_s": round(self.last_frame_age_s, 3),
            "latency_ms": {
                "pose": round(self.last_pose_latency_ms, 1),
                "hands": round(self.last_hand_latency_ms, 1),
                "face": round(self.last_face_latency_ms, 1),
                "total": round(self.last_total_latency_ms, 1),
            },
            "network": {
                "connected": self.is_connected,
                "packets_sent": self.packets_sent,
                "send_failures": self.send_failures,
            }
        }
